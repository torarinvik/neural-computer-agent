"""Build a reusable, provenance-stamped cache of frozen-agent representations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .probe_factorized_event_action import _collect
from .probe_temporal_rule_memory import _load
from .train import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--seed-starts", required=True,
                        help="comma-separated deterministic lifetime seed starts")
    parser.add_argument("--lifetimes-per-block", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--seed", type=int, default=461)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    starts = [int(value) for value in args.seed_starts.split(",") if value]
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    model, consolidator = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    blocks = []
    pair_offset = 0
    for start in starts:
        block = _collect(
            model, consolidator, start=start,
            lifetimes=args.lifetimes_per_block, batch_size=args.batch_size,
            shots=2, heldout=args.heldout, device=device)
        block["pair"] = block["pair"] + pair_offset
        pair_offset += args.lifetimes_per_block
        blocks.append(block)
    tensors = {
        key: torch.cat([block[key] for block in blocks])
        for key in blocks[0]}
    provenance = {
        "schema": "factorized-snapshot-cache-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "controller_checkpoint": str(args.controller_checkpoint),
        "consolidator_checkpoint": str(args.consolidator_checkpoint),
        "pairwise_transfer_checkpoint": str(
            args.pairwise_transfer_checkpoint),
        "projection_transfer_checkpoint": str(
            args.projection_transfer_checkpoint),
        "transfer_strength": args.transfer_strength,
        "seed_starts": starts,
        "lifetimes_per_block": args.lifetimes_per_block,
        "logical_lifetimes": pair_offset,
        "examples_with_counterfactuals": int(tensors["action"].numel()),
        "heldout": args.heldout,
        "sensory_only": True,
        "weights_frozen": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": tensors, "provenance": provenance}, args.output)
    manifest = args.manifest or args.output.with_suffix(".json")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
