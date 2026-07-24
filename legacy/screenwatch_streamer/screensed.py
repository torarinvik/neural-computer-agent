#!/usr/bin/env python3
"""screensed.py — the closed-vocabulary SOUND-EVENT-DETECTION member (orchestra_v3_plan.md V4): an
audio middle tier between the cymbal (exact DSP: transient/silence/tone timing) and the sax (generative
description). It cannot invent timestamps or narratives — it emits (class, score, window) tuples from a
fixed AudioSet-style ontology — but "can't confabulate" != "calibrated", so it takes the SAME per-class
admission exam as every member: a class is admissible evidence ONLY where its measured precision clears
the bar on real audio.

  screensed.py tag   <in.wav> [--model panns|efficientat] [--topk 5] [--json]
       -> stdout: one `SED t=<ms> dur=<ms> class=<name> score=<0..1>` line per top-k class per window
  screensed.py audit <predictions.jsonl> <gold.jsonl> [--min-prec 0.8] [--min-n 10]
       -> per-class precision/recall table + the ADMITTED set (this scoring path needs NO model)

Output contract mirrors `screenaud.py`: stateless, greedy/deterministic, a NEURAL claimant not an
oracle. Admitted classes later emit into the ledger as INFERRED with `license: sed:<class>@<precision>`;
the cymbal remains the ONLY timing authority (a cymbal transient inside an SED "impact" window is the
AV-sync join — temporal coincidence is OBSERVED, causation is INFERRED, per I10).

MODEL STATUS: the tagger backends (PANNs CNN14 / EfficientAT) require a torch+weights install that is
resource-gated; `tag` reports the exact install steps if the backend is absent. The `audit` path (the
admission rule + P/R math, the actual V4.2 deliverable logic) runs WITHOUT any model and is unit-tested
via --selftest, so the scoring is verified independent of the download.
"""
import argparse
import json
import os
import sys
import wave

WINDOW_MS = 1000          # SED windows are coarse (~1 s); the cymbal refines timing within them
HOP_MS = 500

# The subset of the AudioSet ontology we probe on desktop/broadcast audio (the exam's candidate classes;
# the ADMITTED subset is the deliverable, decided by measured precision — never asserted here).
CANDIDATE_CLASSES = ["speech", "music", "crowd", "impact", "bell", "whistle",
                     "keyboard", "silence", "applause", "footsteps"]


def die(code, msg):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


def wav_windows(path):
    """Yield (t_ms, dur_ms, mono_frames_bytes) over WINDOW_MS/HOP_MS. Pure stdlib — no model needed."""
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        win = int(sr * WINDOW_MS / 1000)
        hop = int(sr * HOP_MS / 1000)
        i = 0
        while i < n:
            w.setpos(i)
            frames = w.readframes(min(win, n - i))
            yield int(i * 1000 / sr), WINDOW_MS, frames, sr, ch
            i += hop


# Map the raw AudioSet-527 label strings PANNs emits to our compact CANDIDATE_CLASSES ontology.
# The tagger proposes over 527 fine labels; we fold the relevant ones down to the exam's vocabulary
# (an admitted class still earns authority per-class by measured precision, never by this mapping).
AUDIOSET_TO_ONTOLOGY = {
    "speech": "speech", "male speech, man speaking": "speech",
    "female speech, woman speaking": "speech", "conversation": "speech", "narration, monologue": "speech",
    "music": "music", "musical instrument": "music",
    "cheering": "crowd", "crowd": "crowd", "applause": "applause", "clapping": "applause",
    "hubbub, speech noise, speech babble": "crowd",
    "bell": "bell", "bicycle bell": "bell", "church bell": "bell", "ding": "bell", "ding-dong": "bell",
    "whistle": "whistle", "whistling": "whistle",
    "computer keyboard": "keyboard", "typing": "keyboard", "typewriter": "keyboard",
    "silence": "silence",
    "walk, footsteps": "footsteps", "footsteps": "footsteps",
    # impact-like AudioSet labels (boxing/desktop): fold to a single 'impact' class
    "thump, thud": "impact", "smash, crash": "impact", "slap, smack": "impact", "bang": "impact",
    "knock": "impact", "whack, thwack": "impact", "punch": "impact", "hands": "impact",
}


def load_backend(model, python_hint=".venv-aud/bin/python"):
    """Return a callable(wav_path)->[(t_ms, dur_ms, [(class,score)...])...] or raise with install help.
    PANNs needs torch; if this interpreter lacks it, point at the audio venv rather than failing blind."""
    if model != "panns":
        die(3, f"screensed: backend '{model}' not installed (resource-gated; see module docstring).")
    try:
        import numpy as np
        import torch  # noqa: F401
        from panns_inference import AudioTagging
        from panns_inference.config import labels as AS_LABELS
    except ImportError:
        die(3, "screensed: PANNs backend needs torch+panns_inference. This interpreter lacks them.\n"
               f"  Run with the audio venv:  {python_hint} screensed.py tag <wav>\n"
               "  (provision once: uv pip install --python .venv-aud/bin/python panns_inference;\n"
               "   Cnn14 weights ~300MB download on first run.)\n"
               "The `audit` scoring path needs no model and runs today (see --selftest).")
    at = AudioTagging(checkpoint_path=None, device="cpu")   # loads/downloads Cnn14 (32kHz)

    def tag_file(wav_path, topk=5):
        import wave as _w
        with _w.open(wav_path, "rb") as w:
            sr, n = w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        import numpy as _np
        wav = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
        if sr != 32000:                                     # PANNs is trained at 32 kHz
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=32000)
            sr = 32000
        win, hop = int(sr * WINDOW_MS / 1000), int(sr * HOP_MS / 1000)
        out = []
        for i in range(0, len(wav), hop):
            seg = wav[i:i + win]
            if len(seg) < win // 2:
                break
            clipwise, _ = at.inference(seg[None, :])        # (1, 527)
            scores = clipwise[0]
            folded = {}                                     # ontology class -> max score over its labels
            for idx, lab in enumerate(AS_LABELS):
                oc = AUDIOSET_TO_ONTOLOGY.get(lab.lower())
                if oc:
                    folded[oc] = max(folded.get(oc, 0.0), float(scores[idx]))
            ranked = sorted(folded.items(), key=lambda kv: -kv[1])[:topk]
            out.append((int(i * 1000 / sr), WINDOW_MS, ranked))
        return out

    return tag_file


def cmd_tag(a):
    if not os.path.exists(a.wav):
        die(2, f"screensed: no such wav {a.wav}")
    tag_file = load_backend(a.model)   # raises install help if absent
    for (t_ms, dur_ms, ranked) in tag_file(a.wav, a.topk):
        for cls, score in ranked:
            print(f"SED t={t_ms} dur={dur_ms} class={cls} score={score:.3f}")


def audit(preds, gold, min_prec=0.8, min_n=10, tol_ms=1500):
    """Per-class precision/recall. A prediction is a true positive if a gold event of the same class
    lies within tol_ms. The admission rule: precision >= min_prec on >= min_n instances. NO model."""
    by_cls = {}
    for p in preds:
        by_cls.setdefault(p["class"], {"tp": 0, "fp": 0, "n": 0, "gold": 0})
    for g in gold:
        by_cls.setdefault(g["class"], {"tp": 0, "fp": 0, "n": 0, "gold": 0})["gold"] += 1
    for p in preds:
        c = by_cls[p["class"]]
        c["n"] += 1
        hit = any(g["class"] == p["class"] and abs(g["t"] - p["t"]) <= tol_ms for g in gold)
        c["tp" if hit else "fp"] += 1
    rows, admitted = [], []
    for cls, c in sorted(by_cls.items()):
        prec = c["tp"] / c["n"] if c["n"] else 0.0
        rec = c["tp"] / c["gold"] if c["gold"] else 0.0
        ok = c["n"] >= min_n and prec >= min_prec
        rows.append((cls, prec, rec, c["n"], c["gold"], ok))
        if ok:
            admitted.append(cls)
    return rows, admitted


def cmd_audit(a):
    preds = [json.loads(l) for l in open(a.predictions) if l.strip()]
    gold = [json.loads(l) for l in open(a.gold) if l.strip()]
    rows, admitted = audit(preds, gold, a.min_prec, a.min_n)
    print(f"{'class':12} {'prec':>6} {'rec':>6} {'n':>4} {'gold':>5}  admit")
    for cls, prec, rec, n, g, ok in rows:
        print(f"{cls:12} {prec:6.2f} {rec:6.2f} {n:4d} {g:5d}  {'YES' if ok else 'no'}")
    print(f"\nADMITTED (prec>={a.min_prec} on >={a.min_n}): {admitted or '(none)'}")


def selftest():
    # 5 predictions of "impact" (4 within tol of a gold impact, 1 far) + 3 "music" (all matched).
    preds = ([{"class": "impact", "t": t} for t in (1000, 2000, 3000, 4000, 90000)]
             + [{"class": "music", "t": t} for t in (500, 1500, 2500)])
    gold = ([{"class": "impact", "t": t} for t in (1000, 2000, 3000, 4000)]
            + [{"class": "music", "t": 1000}])
    rows, admitted = audit(preds, gold, min_prec=0.8, min_n=3)
    d = {c: (round(p, 2), n) for (c, p, r, n, g, ok) in rows}
    assert d["impact"] == (0.8, 5), d          # 4/5 within tol
    assert d["music"] == (1.0, 3), d           # 3/3
    assert set(admitted) == {"impact", "music"}, admitted
    print("screensed selftest: PASS (audit P/R + admission rule verified without a model)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="screensed")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("tag"); t.add_argument("wav"); t.add_argument("--model", default="panns")
    t.add_argument("--topk", type=int, default=5); t.add_argument("--json", action="store_true")
    au = sub.add_parser("audit"); au.add_argument("predictions"); au.add_argument("gold")
    au.add_argument("--min-prec", type=float, default=0.8); au.add_argument("--min-n", type=int, default=10)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.cmd == "tag":
        cmd_tag(a)
    elif a.cmd == "audit":
        cmd_audit(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
