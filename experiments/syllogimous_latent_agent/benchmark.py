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
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--premises", default="2,8,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []
    for checkpoint_path in args.checkpoints:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = saved["metadata"]
        model = LatentAgent(metadata["core"], metadata["hidden"],
                            metadata["recursive_steps"],
                            entity_count=metadata.get("entity_count", 64),
                            use_positions=metadata.get("use_positions", True)).to(device)
        model.load_state_dict(saved["model"], strict=False)
        model.eval()
        for premises in map(int, args.premises.split(",")):
            batch = collate_episodes([generate_public_episode(100_123 + premises, premises,
                                                              heldout=True, final=True)])
            frames = batch["frames"].to(device)
            pcm = batch["pcm"].to(device)
            mask = batch["mask"].to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                for _ in range(args.warmup):
                    model(frames, pcm, mask)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.repetitions):
                    model(frames, pcm, mask)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000 / args.repetitions
            events = premises + 1
            rows.append({"core": metadata["core"], "parameters": metadata["parameters"],
                         "premises": premises, "events": events,
                         "batch_one_sequence_ms": elapsed_ms,
                         "batch_one_ms_per_event": elapsed_ms / events})
    payload = {"schema": "syllogimous-latent-batch-one-benchmark-v1",
               "device": str(device), "warmup": args.warmup,
               "repetitions": args.repetitions, "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
