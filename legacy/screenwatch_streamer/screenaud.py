#!/usr/bin/env python3
"""screenaud.py — the SAX: a local audio-language member (MiDashengLM-0.6B) that interprets *what a
sound is* over a recorded window — impacts, tones, music-state, off-screen activity — the semantic
layer above the cymbal's symbolic transient/silence/tone measurements (orchestra_v2 plan M6).

Stateless, greedy-decode (reproducible), a NEURAL claimant not an oracle (I1/I8-analogue): its answer
is Inferred and never overrides the cymbal's exact timing/count. This M6 audition drives it from an
audiogen WAV directly (the deterministic path); the live path reads the aud_* PCM ring.

  screenaud.py <in.wav> [--q "question"] [--model ...] [--json]
stdout: ANSWER <text> / COST load=<s> infer=<s> model=<..> device=<..>   (--json for one JSON line)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_MODEL = "mispeech/midashenglm-0.6b-fp32"
DEFAULT_Q = "Write a detailed caption about this audio within 1-2 sentences. Report only what is audible."


def die(code, msg):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


def run_model(wav_path, question, model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_s = time.time() - t_load

    messages = [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "audio", "path": wav_path},
    ]}]
    model_inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, add_special_tokens=True,
        return_dict=True).to(device=model.device, dtype=model.dtype)

    t_inf = time.time()
    with torch.no_grad():
        gen = model.generate(**model_inputs, do_sample=False, max_new_tokens=96)
    infer_s = time.time() - t_inf
    # MiDashengLM.generate returns only the new tokens (per the model card); decode the full output
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return " ".join(out.split()), load_s, infer_s, device


def main():
    ap = argparse.ArgumentParser(prog="screenaud")
    ap.add_argument("wav")
    ap.add_argument("--q", default=DEFAULT_Q)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.wav):
        die(2, f"screenaud: no such wav {a.wav}")

    answer, load_s, infer_s, device = run_model(a.wav, a.q, a.model)
    short = a.model.split("/")[-1]
    if a.json:
        print(json.dumps({"answer": answer,
                          "cost": {"load_s": round(load_s, 1), "infer_s": round(infer_s, 1),
                                   "model": short, "device": device}}))
    else:
        print(f"ANSWER {answer}")
        print(f"COST load={load_s:.1f} infer={infer_s:.1f} model={short} device={device}")


if __name__ == "__main__":
    main()
