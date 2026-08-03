"""Disposable probe: does the frozen visual encoder preserve a context cue?

This is a representation diagnostic only.  Context identifiers are generated
by the verifier and never reach the controller during ordinary training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import generate_lifetimes
from .legacy_model import UnifiedCognitiveController


@torch.no_grad()
def _features(
        model: UnifiedCognitiveController, *, count: int, trials: int,
        seed: int, heldout: bool, device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    batch = generate_lifetimes(
        count, trials, seed=seed, heldout=heldout,
        task="contextual_mapping", support_trials=2, device=device)
    assert batch.context_ids is not None
    frames = batch.frames.reshape(-1, *batch.frames.shape[2:])
    labels = batch.context_ids.reshape(-1)
    return model.vision(frames).detach(), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-lifetimes", type=int, default=1024)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    train_x, train_y = _features(
        model, count=args.train_lifetimes, trials=6, seed=args.seed,
        heldout=False, device=device)
    test_x, test_y = _features(
        model, count=args.test_lifetimes, trials=6, seed=args.seed + 1,
        heldout=True, device=device)
    probe = nn.Linear(model.width, 2).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2)
    for _ in range(args.steps):
        logits = probe(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_x).argmax(-1) == train_y).float().mean())
        test_accuracy = float((probe(test_x).argmax(-1) == test_y).float().mean())
    report = {
        "schema": "contextual-encoder-probe-v1",
        "claim_boundary": (
            "Supervised diagnostic of frozen visual context decodability; "
            "probe weights are disposable and never enter controller training."),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "steps": args.steps,
        "chance_accuracy": 0.5,
        "train_accuracy": train_accuracy,
        "heldout_accuracy": test_accuracy,
        "representation_preserves_context": test_accuracy >= 0.90,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
