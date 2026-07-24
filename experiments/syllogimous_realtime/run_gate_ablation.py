#!/usr/bin/env python3
"""Run the causal stream-gate ablation matrix against the native host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .event_gate import EventGate
from .host_client import run_host
from .vlm_policy import AudioOnlyPolicy

def one(host: str, mode: str, frame_threshold: float, audio_silence_ms: int,
        packets: int) -> dict:
    gate = EventGate(mode=mode, frame_threshold=frame_threshold,
                     audio_rms_threshold=0.0, audio_silence_ms=audio_silence_ms)
    events = run_host([host], AudioOnlyPolicy(), max_packets=packets,
                      packet_gate=gate.accept)
    received = sum(e.kind == "stream_received" for e in events)
    suppressed = sum(e.kind == "stream_suppressed" for e in events)
    actions = sum(e.kind == "action_sent" for e in events)
    inference_ms = sum((b.wall_ns - a.wall_ns) / 1_000_000
                       for a, b in zip(events, events[1:])
                       if a.kind == "stream_received" and b.kind == "inference_complete")
    return {"mode": mode, "frame_threshold": frame_threshold,
            "audio_silence_ms": audio_silence_ms, "packets_requested": packets,
            "packets_received": received, "packets_suppressed": suppressed,
            "inference_calls": actions, "suppression_rate": suppressed / received if received else 0.0,
            "inference_wall_ms": inference_ms}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="./experiments/syllogimous_realtime/syllogimous_host")
    parser.add_argument("--packets", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("gate_ablation.json"))
    args = parser.parse_args()
    rows = []
    for mode in ("dense", "fixed-gate", "random-control", "learned-gate"):
        for frame_threshold in (0.0, 1.0, 4.0):
            for silence_ms in (0, 80, 180):
                rows.append(one(args.host, mode, frame_threshold, silence_ms, args.packets))
    result = {"schema": "syllogimous.gate-ablation.v1", "rows": rows,
              "note": "AudioOnlyPolicy is a transport control; rows measure sparsity and call reduction, not gameplay accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
