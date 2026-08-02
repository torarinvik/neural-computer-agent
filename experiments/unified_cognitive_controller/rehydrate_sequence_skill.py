"""Materialize one cold skill artifact on top of a frozen parent checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .audit_sequence_skill_memory import _load, _rehydrate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--skill-memory", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    skill = torch.load(
        args.skill_memory, map_location="cpu", weights_only=False)
    model = _rehydrate(parent, skill, device=device)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": skill["child_model_configuration"],
        "state_dict": model.state_dict(),
        "source_parent": str(args.parent),
        "source_skill_memory": str(args.skill_memory),
        "admission_status": "cold_skill_rehydrated",
    }, args.checkpoint_out)
    print({
        "checkpoint": str(args.checkpoint_out),
        "skill_memory": str(args.skill_memory),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, flush=True)


if __name__ == "__main__":
    main()
