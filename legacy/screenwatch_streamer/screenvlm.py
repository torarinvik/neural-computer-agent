#!/usr/bin/env python3
"""screenvlm — the orchestra's "violin": a local video-VLM member that answers WHAT HAPPENS during a
recorded time span. Stateless CLI, like screenocr: frames in -> one text claim out. No daemon, no
memory (I3). It is a NEURAL claimant, not an oracle — its answer is Inferred (I1/I8), always carries
the exact archive seqs it saw, and never overrides a symbolic measurement.

Frames come from the Tier-A exact-frame ring via `arch_tool show`, NOT from a video file: no ffmpeg,
no PyAV, no torchvision.read_video (the API that broke in the VJEPA2 session). Because the input
frames ARE archive seqs, every answer is automatically evidence-pointed and re-groundable.

Model: Qwen2.5-VL-3B-Instruct, fp16, MPS — the caption_qwen.py recipe verbatim (greedy decode for
reproducible eval answers).

    screenvlm <batch_dir> --span t0,t1 [--frames 16] [--region x,y,w,h]
              [--q "question"] [--arch-tool ./arch_tool] [--model ...] [--json]

stdout (text mode):
    ANSWER <one-paragraph answer>
    EVIDENCE arch seq <s0>..<s1> t=<t0>..<t1> frames=<n> region=<x,y,w,h|full>[ partial=true]
    COST load=<s> infer=<s> model=Qwen2.5-VL-3B

exit 0 success; 2 bad args; 1 span not (fully) in the live ring (partial answer if >=4 frames were
recoverable, marked partial=true; else no answer + an evidence-exhausted reason on stderr).
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time

# Quiet, deterministic stdout for the sub-watcher parser: no HF progress bars / rate-limit warnings /
# tokenizer-fork noise. Set before torch/transformers import. (Cache is pre-provisioned by setup_vlm.)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MIN_PARTIAL_FRAMES = 4          # below this we refuse to answer rather than guess from a sliver
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_Q = "Describe what happens in this screen recording clip. Report only what is visible."
RESIZE_W = 448                  # proven recipe: 448-wide into the vision encoder


def die(code, msg):
    print(msg, file=sys.stderr)
    sys.exit(code)


def parse_span(s):
    try:
        a, b = s.split(",")
        t0, t1 = int(a), int(b)
    except Exception:
        die(2, f"screenvlm: --span wants t0,t1 in ms (got {s!r})")
    if t1 < t0:
        t0, t1 = t1, t0
    return t0, t1


def parse_region(s):
    if not s:
        return None
    try:
        x, y, w, h = (int(v) for v in s.split(","))
    except Exception:
        die(2, f"screenvlm: --region wants x,y,w,h (got {s!r})")
    if w <= 0 or h <= 0:
        die(2, f"screenvlm: --region w,h must be positive (got {s!r})")
    return (x, y, w, h)


def load_index(batch_dir):
    """Parse every arch_<seg>.idx present -> sorted [(seq, t_ms, w, h)]. Lines are
    '<seq> <t_ms> <kind> <len> <w> <h> <hash>' (hash is 'a:b'; we ignore it here)."""
    entries = []
    for idx in sorted(glob.glob(os.path.join(batch_dir, "arch_*.idx"))):
        try:
            with open(idx, "r") as f:
                for line in f:
                    p = line.split()
                    if len(p) < 6:
                        continue
                    try:
                        seq = int(p[0]); t_ms = int(p[1]); w = int(p[4]); h = int(p[5])
                    except ValueError:
                        continue
                    if w < 1 or h < 1:          # trailing-newline / partial line guard
                        continue
                    entries.append((seq, t_ms, w, h))
        except OSError:
            continue
    entries.sort(key=lambda e: e[0])
    return entries


def select_seqs(entries, t0, t1, n_frames):
    """Uniform sample of <= n_frames entries whose t_ms falls in [t0,t1]."""
    span = [e for e in entries if t0 <= e[1] <= t1]
    if not span:
        return []
    if len(span) <= n_frames:
        return span
    m = len(span)
    picks, seen = [], set()
    for i in range(n_frames):
        j = round(i * (m - 1) / (n_frames - 1))
        if j not in seen:
            seen.add(j)
            picks.append(span[j])
    return picks


def show_frame(arch_tool, batch_dir, seq, out_ppm, region):
    cmd = [arch_tool, "show", batch_dir, str(seq), out_ppm]
    if region:
        cmd += [str(region[0]), str(region[1]), str(region[2]), str(region[3])]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(out_ppm) and os.path.getsize(out_ppm) > 0


def recover_frames(arch_tool, batch_dir, picks, entries, region, tmp):
    """Decode each picked seq; if one fails (pruned mid-run), substitute the nearest surviving
    neighbor in `entries`. Returns (paths, used_seqs, partial)."""
    all_seqs = [e[0] for e in entries]
    paths, used, partial = [], [], False
    for k, e in enumerate(picks):
        seq = e[0]
        out = os.path.join(tmp, f"f_{k:03d}.ppm")
        if show_frame(arch_tool, batch_dir, seq, out, region):
            paths.append(out); used.append(seq); continue
        # pruned under us — walk outward for a live neighbor
        partial = True
        got = False
        for d in range(1, len(all_seqs)):
            for cand in (seq + d, seq - d):
                if cand in all_seqs and cand not in used and show_frame(arch_tool, batch_dir, cand, out, region):
                    paths.append(out); used.append(cand); got = True
                    break
            if got:
                break
    return paths, used, partial


def stack_frames(paths):
    import numpy as np
    from PIL import Image
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        h2 = max(2, round(h * RESIZE_W / w))
        h2 -= h2 % 2                       # even height (mirrors scale=448:-2)
        im = im.resize((RESIZE_W, h2), Image.LANCZOS)
        frames.append(np.asarray(im))
    return np.stack(frames)


def run_model(frames, question, model_id, max_new_tokens):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    t_load = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16).to(device).eval()
    load_s = time.time() - t_load

    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], videos=[frames], return_tensors="pt")
    inputs = {k: (v.to(device, dtype=torch.float16) if v.is_floating_point() else v.to(device))
              for k, v in inputs.items()}

    t_inf = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    infer_s = time.time() - t_inf
    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = processor.decode(gen, skip_special_tokens=True).strip()
    return " ".join(ans.split()), load_s, infer_s, device


def main():
    ap = argparse.ArgumentParser(prog="screenvlm", add_help=True)
    ap.add_argument("batch_dir")
    ap.add_argument("--span", required=True, help="t0,t1 in ms")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--region", default=None, help="x,y,w,h in source pixels")
    ap.add_argument("--q", default=DEFAULT_Q)
    ap.add_argument("--arch-tool", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.frames < 1:
        die(2, "screenvlm: --frames must be >= 1")
    n_frames = min(args.frames, 32)
    t0, t1 = parse_span(args.span)
    region = parse_region(args.region)

    arch_tool = args.arch_tool or os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_tool")
    if not (os.path.exists(arch_tool) and os.access(arch_tool, os.X_OK)):
        die(2, f"screenvlm: arch_tool not found/executable at {arch_tool}")
    if not os.path.isdir(args.batch_dir):
        die(2, f"screenvlm: no such dir {args.batch_dir}")

    entries = load_index(args.batch_dir)
    if not entries:
        die(1, f"screenvlm: evidence-exhausted — no archive index in {args.batch_dir}")
    earliest, latest = entries[0][1], entries[-1][1]

    picks = select_seqs(entries, t0, t1, n_frames)
    if not picks:
        if t1 < earliest:
            die(1, f"screenvlm: evidence-exhausted — span [{t0},{t1}]ms predates ring (earliest t={earliest})")
        if t0 > latest:
            die(1, f"screenvlm: evidence-exhausted — span [{t0},{t1}]ms beyond recorded range (latest t={latest})")
        die(1, f"screenvlm: no frames in span [{t0},{t1}]ms (ring has {len(entries)} frames t={earliest}..{latest})")

    span_partial = (t0 < earliest) or (t1 > latest)

    tmp = tempfile.mkdtemp(prefix="screenvlm_")
    try:
        paths, used, prune_partial = recover_frames(arch_tool, args.batch_dir, picks, entries, region, tmp)
        if len(paths) < MIN_PARTIAL_FRAMES:
            die(1, f"screenvlm: evidence-exhausted — only {len(paths)} frame(s) recoverable in span "
                   f"[{t0},{t1}]ms (need >= {MIN_PARTIAL_FRAMES})")
        partial = span_partial or prune_partial
        frames = stack_frames(paths)
        ans, load_s, infer_s, device = run_model(frames, args.q, args.model, args.max_new_tokens)
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
        import json
        print(json.dumps({
            "answer": ans,
            "evidence": {"seq0": s0, "seq1": s1, "t0": ts0, "t1": ts1,
                         "frames": len(paths), "region": reg, "partial": partial},
            "cost": {"load_s": round(load_s, 1), "infer_s": round(infer_s, 1),
                     "model": short_model, "device": device},
        }))
    else:
        print(f"ANSWER {ans}")
        ev = f"EVIDENCE arch seq {s0}..{s1} t={ts0}..{ts1} frames={len(paths)} region={reg}"
        if partial:
            ev += " partial=true"
        print(ev)
        print(f"COST load={load_s:.1f} infer={infer_s:.1f} model={short_model} device={device}")

    sys.exit(1 if partial else 0)


if __name__ == "__main__":
    main()
