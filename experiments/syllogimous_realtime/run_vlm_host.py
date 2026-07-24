#!/usr/bin/env python3
"""Connect a real VLM checkpoint to the native audiovisual host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .host_client import run_host
from .event_gate import EventGate
from .vlm_policy import AudioOnlyPolicy, SmolVLMPolicy

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Hugging Face model id or local directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--gate", choices=("dense", "fixed-gate", "random-control", "learned-gate"), default="dense")
    parser.add_argument("--frame-threshold", type=float, default=0.0)
    parser.add_argument("--audio-rms-threshold", type=float, default=0.0)
    parser.add_argument("--audio-silence-ms", type=int, default=0)
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--host", default="./experiments/syllogimous_realtime/syllogimous_host")
    parser.add_argument("--packets", type=int, default=0, help="0 means until host EOF")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    if args.audio_only:
        policy = AudioOnlyPolicy()
    else:
        if not args.model:
            parser.error("--model is required unless --audio-only is selected")
        policy = SmolVLMPolicy.from_pretrained(args.model, device=args.device,
                                               local_files_only=args.local_files_only,
                                               image_size=args.image_size,
                                               max_new_tokens=args.max_new_tokens,
                                               dtype=args.dtype)
    gate = EventGate(mode=args.gate, frame_threshold=args.frame_threshold,
                     audio_rms_threshold=args.audio_rms_threshold,
                     audio_silence_ms=args.audio_silence_ms)
    events = run_host([args.host], policy, max_packets=args.packets or None,
                      packet_gate=gate.accept)
    payload = []
    for event in events:
        item = {"kind": event.kind, "wall_ns": event.wall_ns,
                "stream_timestamp_ms": event.stream_timestamp_ms,
                "action": event.action}
        if event.kind == "stream_received":
            # The native envelope synchronizes the frame and PCM read.  They
            # therefore share a wall-clock boundary, but are logged separately
            # for frame/audio latency analysis.
            item["frame_received_ns"] = event.wall_ns
            item["audio_received_ns"] = event.wall_ns
            item["modalities"] = list(event.modalities)
        payload.append(item)
    if args.log:
        args.log.write_text("\n".join(json.dumps(item) for item in payload) + "\n")
    inference_latencies_ms = []
    for previous, current in zip(events, events[1:]):
        if previous.kind == "stream_received" and current.kind == "inference_complete":
            inference_latencies_ms.append((current.wall_ns - previous.wall_ns) / 1_000_000)
    print(json.dumps({"events": len(events), "actions": sum(item["action"] is not None for item in payload),
                      "inference_latency_ms": inference_latencies_ms,
                      "suppressed_packets": sum(item["kind"] == "stream_suppressed" for item in payload),
                      "log": str(args.log) if args.log else None}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
