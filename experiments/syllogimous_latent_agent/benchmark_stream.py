from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import collate_episodes, generate_public_episode
from .model import LatentAgent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--premises", default="16,24,32,64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    model = LatentAgent(metadata["core"], metadata["hidden"], metadata["recursive_steps"],
                        entity_count=metadata.get("entity_count", 64),
                        use_positions=metadata.get("use_positions", True)).to(device).eval()
    model.load_state_dict(saved["model"], strict=False)
    rows = []
    for premises in map(int, args.premises.split(",")):
        batch = collate_episodes([generate_public_episode(100_321 + premises, premises,
                                                          heldout=True, final=True,
                                                          entity_count=metadata["entity_count"])])
        frames, pcm = batch["frames"].to(device), batch["pcm"].to(device)

        def run_once() -> None:
            state = model.init_stream_state()
            for index in range(frames.shape[1]):
                _, state = model.stream_step(frames[:, index], pcm[:, index], state)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            for _ in range(args.warmup):
                run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.repetitions):
                run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
        sequence_ms = (time.perf_counter() - started) * 1000 / args.repetitions
        rows.append({"premises": premises, "events": premises + 1,
                     "stream_sequence_ms": sequence_ms,
                     "mean_stream_event_ms": sequence_ms / (premises + 1)})
    payload = {"schema": "syllogimous-latent-incremental-benchmark-v1",
               "checkpoint": args.checkpoint.name, "device": str(device), "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
