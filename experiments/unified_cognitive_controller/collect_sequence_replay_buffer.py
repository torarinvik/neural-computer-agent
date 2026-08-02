"""Collect controller-visible sequence experience for cold replay storage."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train_sequence_reward_buffer import _collect_buffer, _save_buffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--buffer-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--span", type=int, default=9)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--position-augmentation", action="store_true")
    parser.add_argument("--include-workspace", action="store_true")
    parser.add_argument("--include-workspace-usage", action="store_true")
    parser.add_argument("--include-event-age", action="store_true")
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 1 or args.span < 1 or args.distractors < 0:
        raise ValueError("invalid collection dimensions")
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **dict(payload["model_configuration"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    tensors = _collect_buffer(
        model, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed, device=device,
        position_augmentation=args.position_augmentation,
        include_workspace=args.include_workspace,
        include_workspace_usage=args.include_workspace_usage,
        include_event_age=args.include_event_age)
    _save_buffer(
        args.buffer_out, tensors, parent=args.checkpoint,
        stream_specs=[{"kind": "target", "span": args.span,
                       "lifetimes": args.count}])
    print({
        "buffer": str(args.buffer_out),
        "transitions": int(tensors[0].shape[0]),
        "feature_width": int(tensors[0].shape[1]),
    }, flush=True)


if __name__ == "__main__":
    main()
