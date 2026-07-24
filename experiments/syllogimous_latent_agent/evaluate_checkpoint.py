from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import EpisodeDataset, collate_episodes
from .model import LatentAgent
from .train import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--premises", default="24,32,64")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    model = LatentAgent(metadata["core"], metadata["hidden"], metadata["recursive_steps"],
                        entity_count=metadata.get("entity_count", 64),
                        use_positions=metadata.get("use_positions", True)).to(args.device)
    model.load_state_dict(saved["model"], strict=False)
    dataset = EpisodeDataset(args.samples, start_seed=100_000,
                             premise_choices=tuple(map(int, args.premises.split(","))),
                             heldout=True, final=True,
                             entity_count=metadata.get("entity_count", 64))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_episodes,
                        pin_memory=str(args.device).startswith("cuda"))
    result = {"schema": "syllogimous-latent-evaluation-v1",
              "checkpoint": args.checkpoint.name, "metrics": evaluate(model, loader, torch.device(args.device))}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
