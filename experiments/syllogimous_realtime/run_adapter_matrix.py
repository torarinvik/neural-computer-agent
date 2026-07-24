#!/usr/bin/env python3
"""Evaluate dense/fixed/random/learned sensory gates with one frozen VLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .adapter_rl import StreamerGate, packet_features
from .checkpointing import load_adapter
from .event_gate import EventGate
from .run_vlm_episode import run
from .vlm_policy import SmolVLMPolicy


def learned_gate(checkpoint: Path):
    gate = StreamerGate()
    load_adapter(checkpoint, gate)
    gate.eval()
    previous = {"frame": None}

    def accept(packet):
        features = packet_features(packet, previous["frame"]).unsqueeze(0)
        previous["frame"] = packet.frame.astype("float32") / 255.0
        with torch.no_grad():
            return bool((torch.sigmoid(gate.logits(features))[0] > 0.5).any().item())
    return accept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--premises", type=int, default=2)
    parser.add_argument("--deadline-ms", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("adapter_matrix.json"))
    args = parser.parse_args()
    model = SmolVLMPolicy.from_pretrained(args.model, device=args.device,
                                           local_files_only=args.local_files_only)
    gates = {
        "dense": lambda packet: True,
        "fixed-gate": EventGate(mode="fixed-gate", frame_threshold=1.0,
                                 audio_silence_ms=80).accept,
        "random-control": EventGate(mode="random-control").accept,
    }
    gates["learned-gate"] = (learned_gate(args.adapter_checkpoint)
                              if args.adapter_checkpoint is not None
                              else lambda packet: True)
    results = []
    for name, gate in gates.items():
        rows, traces = run(model, episodes=args.episodes, premises=args.premises,
                           deadline_ms=args.deadline_ms, seed=args.seed,
                           packet_gate=gate, difficulty="standard")
        from .evaluation import summarize
        results.append({"variant": name,
                        "trained": name == "learned-gate" and args.adapter_checkpoint is not None,
                        "metrics": summarize(rows), "action_traces": traces})
    payload = {"schema": "syllogimous.adapter-matrix.v1", "frozen_listener": True,
               "model": args.model, "results": results}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "variants": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
