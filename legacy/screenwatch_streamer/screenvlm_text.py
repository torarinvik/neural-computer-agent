#!/usr/bin/env python3
"""screenvlm_text — the representation-ladder sibling of screenvlm. Same violin, same ring, same
Qwen model — but the frames are handed to the LANGUAGE pathway as TEXT (a per-frame bounding-box
log) instead of to the vision encoder as pixels.

WHY (V1.10): V1.9 measured the raster violin at 0/15 on the metamorphic flip battery — a constant
"No." prior, and a 4x frame-density control (32f) did not rescue it, so the failure is the vision
encoder reading our downscaled synthetic rasters, not frame starvation. This member tests the other
side of the token-budget trade: spend the context on MANY frames at LOW detail-per-frame (numbers
the LLM can actually reason over) instead of FEW frames at high pixel detail.

PROVENANCE — the load-bearing rule. The bounding boxes come from an INDEPENDENT dumb per-frame
threshold+connected-components pass (extract_blobs below): no temporal model, no track ids, no
direction/event computation. The model must itself notice that cx rises then falls (a reversal) or
that a box disappears then returns (a vanish gap). That makes it a genuine independent temporal
witness that can CHECK viola — not a paraphrase of viola's tracks (which would inherit viola's
authority and be worthless as cross-examination). Every answer still carries the exact archive seqs
it saw, exactly like screenvlm.

    screenvlm_text <batch_dir> --span t0,t1 [--frames 64] [--repr table|svg] [--region x,y,w,h]
                   [--q "question"] [--arch-tool ./arch_tool] [--model ...] [--json]

Frame recovery (load_index / select_seqs / recover_frames) is imported verbatim from screenvlm.py
so R2/R3 see the SAME seqs R0/R1 would — the only variable across rungs is the representation.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

# Reuse the exact ring-reading path from the raster violin so the frame provenance is identical.
import screenvlm as sv

# Text reps can afford far more frames than the vision encoder's 32-frame ceiling — that IS the point
# of the ladder (high n, low detail-per-frame). Cap generously.
MAX_TEXT_FRAMES = 256
MIN_BLOB_AREA = 40             # px; drops single-pixel AA/noise specks, keeps real objects
BG_THRESH = 60                 # L1 color distance from the corner (background) to count as foreground


def extract_blobs(ppm_path):
    """INDEPENDENT per-frame extractor: threshold vs the background color, label connected
    components, return [(cx, cy, w, h, bright)] sorted left-to-right. No temporal state whatsoever —
    this function cannot see any other frame."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    im = np.asarray(Image.open(ppm_path).convert("RGB")).astype(int)
    H, W = im.shape[:2]
    bg = im[0, 0]                                   # corners are always background in these scenes
    fg = np.abs(im - bg).sum(axis=2) > BG_THRESH
    lbl, n = ndimage.label(fg)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if xs.size < MIN_BLOB_AREA:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bright = int(im[ys, xs].mean())
        blobs.append(((x0 + x1) // 2, (y0 + y1) // 2, x1 - x0 + 1, y1 - y0 + 1, bright))
    blobs.sort(key=lambda b: (b[0], b[1]))
    return W, H, blobs


def label_for(bright):
    """A per-frame attribute, NOT an identity assignment — 'bright'/'dim' is observed fresh each
    frame from that frame's pixels, so it leaks no cross-frame tracking."""
    return "bright" if bright >= 136 else "dim"


def build_table(dims, samples):
    """R-table: one line per frame, boxes as compact ints. Lowest detail-per-frame -> most frames."""
    W, H = dims
    out = [f"# Object bounding-box log from a screen recording, {W}x{H} px. "
           f"x increases rightward, y increases downward. Each line is one frame in time order. "
           f"Each box is a rectangle: cx,cy = center; w,h = width,height. No object = empty."]
    for t_ms, _seq, (_w, _h, blobs) in samples:
        if blobs:
            parts = " ".join(f"{label_for(b[4])}[cx={b[0]},cy={b[1]},w={b[2]},h={b[3]}]" for b in blobs)
        else:
            parts = "(no objects visible)"
        out.append(f"t={t_ms}ms: {parts}")
    return "\n".join(out)


def build_svg(dims, samples):
    """R-svg: one tiny SVG per frame with colored rects. Higher detail-per-frame -> fewer frames."""
    W, H = dims
    out = [f"# A screen recording as a sequence of SVG frames ({W}x{H}), in time order. "
           f"Each <rect> is one visible object; brighter fill = brighter object."]
    for t_ms, _seq, (_w, _h, blobs) in samples:
        rects = "".join(
            f'<rect x="{b[0]-b[2]//2}" y="{b[1]-b[3]//2}" width="{b[2]}" height="{b[3]}" '
            f'fill="rgb({b[4]},{b[4]},{b[4]})"/>'
            for b in blobs)
        out.append(f'<!-- t={t_ms}ms --><svg width="{W}" height="{H}">{rects}</svg>')
    return "\n".join(out)


def run_text_model(text, model_id, max_new_tokens):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16).to(device).eval()
    load_s = time.time() - t_load

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t_inf = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    infer_s = time.time() - t_inf
    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = processor.decode(gen, skip_special_tokens=True).strip()
    return " ".join(ans.split()), load_s, infer_s, device


def main():
    ap = argparse.ArgumentParser(prog="screenvlm_text", add_help=True)
    ap.add_argument("batch_dir")
    ap.add_argument("--span", required=True, help="t0,t1 in ms")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--repr", choices=["table", "svg"], default=os.environ.get("LADDER_REPR", "table"))
    ap.add_argument("--region", default=None, help="x,y,w,h in source pixels")
    ap.add_argument("--q", default=sv.DEFAULT_Q)
    ap.add_argument("--arch-tool", default=None)
    ap.add_argument("--model", default=sv.DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.frames < 1:
        sv.die(2, "screenvlm_text: --frames must be >= 1")
    n_frames = min(args.frames, MAX_TEXT_FRAMES)
    t0, t1 = sv.parse_span(args.span)
    region = sv.parse_region(args.region)

    arch_tool = args.arch_tool or os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_tool")
    if not (os.path.exists(arch_tool) and os.access(arch_tool, os.X_OK)):
        sv.die(2, f"screenvlm_text: arch_tool not found/executable at {arch_tool}")
    if not os.path.isdir(args.batch_dir):
        sv.die(2, f"screenvlm_text: no such dir {args.batch_dir}")

    entries = sv.load_index(args.batch_dir)
    if not entries:
        sv.die(1, f"screenvlm_text: evidence-exhausted — no archive index in {args.batch_dir}")
    earliest, latest = entries[0][1], entries[-1][1]
    picks = sv.select_seqs(entries, t0, t1, n_frames)
    if not picks:
        sv.die(1, f"screenvlm_text: no frames in span [{t0},{t1}]ms "
                  f"(ring t={earliest}..{latest})")
    span_partial = (t0 < earliest) or (t1 > latest)

    tmp = tempfile.mkdtemp(prefix="screenvlm_text_")
    try:
        paths, used, prune_partial = sv.recover_frames(arch_tool, args.batch_dir, picks, entries, region, tmp)
        if len(paths) < sv.MIN_PARTIAL_FRAMES:
            sv.die(1, f"screenvlm_text: evidence-exhausted — only {len(paths)} frame(s) recoverable")
        # extract_blobs is INDEPENDENT per frame; samples pairs each recovered frame with its time
        samples, dims = [], None
        for k, p in enumerate(paths):
            W, H, blobs = extract_blobs(p)
            dims = (W, H)
            samples.append((picks[k][1], used[k], (W, H, blobs)))
        rep = build_table(dims, samples) if args.repr == "table" else build_svg(dims, samples)
        full = f"{rep}\n\n{args.q}"
        partial = span_partial or prune_partial
        ans, load_s, infer_s, device = run_text_model(full, args.model, args.max_new_tokens)
    finally:
        for p in glob.glob(os.path.join(tmp, "*")):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    s0, s1 = used[0], used[-1]
    ts0, ts1 = picks[0][1], picks[-1][1]
    reg = f"{region[0]},{region[1]},{region[2]},{region[3]}" if region else "full"
    short_model = args.model.split("/")[-1].replace("-Instruct", "")

    if args.json:
        print(json.dumps({
            "answer": ans,
            "evidence": {"seq0": s0, "seq1": s1, "t0": ts0, "t1": ts1,
                         "frames": len(paths), "region": reg, "partial": partial,
                         "repr": args.repr},
            "cost": {"load_s": round(load_s, 1), "infer_s": round(infer_s, 1),
                     "model": short_model, "device": device},
        }))
    else:
        print(f"ANSWER {ans}")
        ev = f"EVIDENCE arch seq {s0}..{s1} t={ts0}..{ts1} frames={len(paths)} repr={args.repr} region={reg}"
        if partial:
            ev += " partial=true"
        print(ev)
        print(f"COST load={load_s:.1f} infer={infer_s:.1f} model={short_model} device={device}")

    sys.exit(1 if partial else 0)


if __name__ == "__main__":
    main()
