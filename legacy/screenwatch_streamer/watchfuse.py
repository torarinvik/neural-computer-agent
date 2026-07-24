#!/usr/bin/env python3
"""Fused real-time cascade watcher: glue the audited members into one triage loop.

Tiers (budget-driven, cheap gates expensive):
  T0 pixel-delta triage        every frame      ~1 ms
  T1 YOLO11n detect            every frame      ~20-45 ms
  T1b YOLO11n-pose             when persons>0, every 3rd frame
  T2 MViT action window        on activity spike (delta z>2) or every 4 s
  T3 SmolVLM caption           on scene boundary (huge delta) or T2 event

Simulates a live 12 fps stream over pre-extracted frames; each frame carries its
stream timestamp. Expensive tiers run inline but are TIMED so we can report the
async-latency they'd add. Output: fused event log + latency accounting.
"""
import os, sys, time, statistics
import numpy as np
from PIL import Image

FRAMES = sys.argv[1]
FPS = 12.0
T0_STRIDE = 1
POSE_STRIDE = 3

files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".png"))
imgs = [os.path.join(FRAMES, f) for f in files]
print(f"{len(imgs)} frames @ {FPS} fps ({len(imgs)/FPS:.1f} s of stream)")

# ---- load members ----
t = time.time()
from ultralytics import YOLO
det = YOLO("yolo11n.pt"); pose = YOLO("yolo11n-pose.pt")
import torch
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
w = MViT_V2_S_Weights.KINETICS400_V1
act = mvit_v2_s(weights=w).eval()
act_tf = w.transforms(); K_LABELS = w.meta["categories"]
from transformers import AutoProcessor, AutoModelForImageTextToText
MP = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
proc = AutoProcessor.from_pretrained(MP)
vlm = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("mps").eval()
print(f"members loaded in {time.time()-t:.1f} s")

def gray64(path):
    return np.asarray(Image.open(path).convert("L").resize((64, 36)), dtype=np.float32)

def run_action(paths):
    frames = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2,0,1) for p in paths]
    clip = torch.stack(frames)  # T,C,H,W
    batch = act_tf(clip).unsqueeze(0)
    with torch.no_grad():
        logits = act(batch)[0]
    p = logits.softmax(-1)
    i = int(p.argmax())
    return K_LABELS[i], float(p[i])

def run_caption(path):
    im = Image.open(path).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":im},
            {"type":"text","text":"Describe only the visible scene-level facts in one short sentence. Do not infer motion or unseen events. Say UNKNOWN if unclear."}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to("mps", dtype=torch.bfloat16)
    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=24, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

# ---- stream loop ----
events = []          # (t_stream, tier, latency_ms, text)
lat = {"T0":[], "T1":[], "T1b":[], "T2":[], "T3":[]}
prev = None
deltas = []
count_hist = []; count_state = None; count_cand = None; count_dwell = 0
prone_frames = 0
last_t2 = -10.0; last_t3 = -10.0
wall0 = time.time()

for i, path in enumerate(imgs):
    ts = i / FPS
    # T0 delta triage
    t0 = time.time()
    g = gray64(path)
    d = float(np.abs(g - prev).mean()) if prev is not None else 0.0
    prev = g
    lat["T0"].append((time.time()-t0)*1000)
    deltas.append(d)
    mu = statistics.mean(deltas[-48:]) if len(deltas) > 4 else d
    sd = statistics.pstdev(deltas[-48:]) if len(deltas) > 4 else 1.0
    z = (d - mu) / sd if sd > 0.5 else 0.0
    scene_cut = d > 40

    # T1 detect
    t0 = time.time()
    r = det(path, verbose=False)[0]
    n_person = sum(1 for c in r.boxes.cls if int(c) == 0)
    lat["T1"].append((time.time()-t0)*1000)
    # belief layer: median-of-last-1s count, emitted only when the SMOOTHED value
    # holds a new level for a full second (dwell) -- screentext's rule, applied to counts
    count_hist.append(n_person)
    smooth = int(statistics.median(count_hist[-12:]))
    if count_state is None:
        count_state = smooth
    elif smooth != count_state:
        if smooth == count_cand:
            count_dwell += 1
        else:
            count_cand, count_dwell = smooth, 1
        if count_dwell >= 12:
            events.append((ts, "T1", lat["T1"][-1], f"person count {count_state} -> {smooth} (dwell-confirmed)"))
            count_state = smooth; count_dwell = 0
    else:
        count_dwell = 0

    # T1b pose (prone geometry)
    if n_person and i % POSE_STRIDE == 0:
        t0 = time.time()
        rp = pose(path, verbose=False)[0]
        prone = False
        for b, c in zip(rp.boxes.xywh, rp.boxes.cls):
            if int(c) == 0 and float(b[2]) / max(float(b[3]), 1) > 1.3:
                prone = True
        lat["T1b"].append((time.time()-t0)*1000)
        if prone:
            prone_frames += 1
            if prone_frames == 2:
                events.append((ts, "T1b", lat["T1b"][-1], "PRONE-GEOMETRY person (aspect>1.3, 2 consecutive)"))
        else:
            prone_frames = 0

    # T2 action on spike or heartbeat
    if (z > 2.0 or ts - last_t2 > 4.0) and ts - last_t2 > 1.5 and i >= 16:
        t0 = time.time()
        label, p = run_action(imgs[i-16:i])
        ms = (time.time()-t0)*1000
        lat["T2"].append(ms)
        last_t2 = ts
        events.append((ts, "T2", ms, f"action[{'spike' if z>2 else 'heartbeat'}]: {label} ({p:.2f})"))

    # T3 caption on scene cut or first frame
    if (scene_cut or i == 0) and ts - last_t3 > 2.0:
        t0 = time.time()
        cap = run_caption(path)
        ms = (time.time()-t0)*1000
        lat["T3"].append(ms)
        last_t3 = ts
        events.append((ts, "T3", ms, f"caption[cut d={d:.0f}]: {cap}"))

wall = time.time() - wall0
print(f"\n=== fused event log (stream time) ===")
for ts, tier, ms, txt in events:
    print(f"t={ts:5.1f}s [{tier} {ms:6.0f}ms] {txt}")

print(f"\n=== latency accounting ===")
for k, v in lat.items():
    if v:
        print(f"{k}: n={len(v):3d} median={statistics.median(v):7.1f}ms p95={sorted(v)[int(0.95*len(v))-1]:7.1f}ms")
per_frame_core = statistics.median(lat['T0']) + statistics.median(lat['T1']) + \
                 (statistics.median(lat['T1b'])/POSE_STRIDE if lat['T1b'] else 0)
print(f"\ncore per-frame budget (T0+T1+T1b/{POSE_STRIDE}): {per_frame_core:.1f} ms -> max {1000/per_frame_core:.1f} fps")
print(f"wall {wall:.1f}s for {len(imgs)/FPS:.1f}s of stream (everything inline) = {len(imgs)/wall:.2f} fps")
