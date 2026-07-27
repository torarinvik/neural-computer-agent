"""Diagnostic for the generic relation needed to turn contextual state into action.

This compares additive and multiplicative state/event readouts using frozen
controller features.  Labels are verifier-only and all probe weights are
discarded after reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, generate_lifetimes
from .model import UnifiedCognitiveController


@torch.no_grad()
def _examples(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        heldout: bool, device: torch.device,
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=heldout,
        task="contextual_mapping", support_trials=2, device=device)
    state = model.initial_state(count, device=device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    reward = torch.zeros(count, device=device)
    feedback = torch.zeros(count, device=device)
    for trial in range(3):
        output, state = model.step(
            batch.frames[:, trial], state, action, reward, feedback)
        action = output.logits.argmax(-1)
        reward = (action == batch.correct_actions[:, trial]).float()
        feedback = torch.ones_like(reward)
    event = model.vision(batch.frames[:, 3])
    hidden = state.hidden
    return {
        "additive": torch.cat([hidden, event], dim=-1).detach(),
        "product": (hidden * event).detach(),
        "combined": torch.cat([hidden, event, hidden * event], dim=-1).detach(),
    }, batch.correct_actions[:, 3].detach()


def _fit(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *,
        steps: int, seed: int, hidden: bool) -> tuple[float, float]:
    torch.manual_seed(seed)
    probe: nn.Module
    if hidden:
        probe = nn.Sequential(
            nn.Linear(train_x.shape[-1], 64), nn.GELU(), nn.Linear(64, 2))
    else:
        probe = nn.Linear(train_x.shape[-1], 2)
    probe = probe.to(train_x.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(probe(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return (
            float((probe(train_x).argmax(-1) == train_y).float().mean()),
            float((probe(test_x).argmax(-1) == test_y).float().mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    train, train_y = _examples(
        model, count=args.train_lifetimes, seed=args.seed,
        heldout=False, device=device)
    test, test_y = _examples(
        model, count=args.test_lifetimes, seed=args.seed + 1,
        heldout=True, device=device)
    results: dict[str, dict[str, float]] = {}
    for index, (name, train_x) in enumerate(train.items()):
        for hidden in (False, True):
            train_accuracy, heldout_accuracy = _fit(
                train_x, train_y, test[name], test_y, steps=args.steps,
                seed=args.seed + 10 + index * 2 + int(hidden), hidden=hidden)
            results[f"{name}_{'mlp' if hidden else 'linear'}"] = {
                "train_accuracy": train_accuracy,
                "heldout_accuracy": heldout_accuracy,
            }
    product_linear = results["product_linear"]["heldout_accuracy"]
    report = {
        "schema": "contextual-action-readout-probe-v1",
        "claim_boundary": (
            "Supervised frozen-feature diagnostic; no probe weight or label is "
            "made available to the controller."),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "steps": args.steps,
        "chance_accuracy": 0.5,
        "readouts": results,
        "multiplicative_linear_readout_viable": product_linear >= 0.85,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
