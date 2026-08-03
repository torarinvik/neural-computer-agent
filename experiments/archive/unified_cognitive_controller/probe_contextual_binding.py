"""Locate whether two context-specific outcomes coexist in controller state.

The probe uses verifier rule IDs only after state extraction.  It is a
throwaway measurement, not a training signal for the controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, generate_lifetimes
from .legacy_model import UnifiedCognitiveController


@torch.no_grad()
def _binding_states(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        heldout: bool, device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=heldout,
        task="contextual_mapping", support_trials=2, device=device)
    state = model.initial_state(count, device=device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    reward = torch.zeros(count, device=device)
    feedback = torch.zeros(count, device=device)
    # Frames 0 and 1 are the two support attempts. Frame 2 carries the second
    # verified outcome, so its recurrent state is the first moment both facts
    # could be jointly available.
    for trial in range(3):
        output, state = model.step(
            batch.frames[:, trial], state, action, reward, feedback)
        action = output.logits.argmax(-1)
        reward = (action == batch.correct_actions[:, trial]).float()
        feedback = torch.ones_like(reward)
    return state.hidden.detach(), batch.rule_bits.detach()


def _fit_accuracy(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *,
        seed: int, steps: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    probe = nn.Sequential(
        nn.Linear(train_x.shape[-1], 64), nn.GELU(), nn.Linear(64, 4)).to(train_x.device)
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
    if args.train_lifetimes % 4 or args.test_lifetimes % 4:
        raise ValueError("lifetime counts must be divisible by four")
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    train_x, train_y = _binding_states(
        model, count=args.train_lifetimes, seed=args.seed,
        heldout=False, device=device)
    test_x, test_y = _binding_states(
        model, count=args.test_lifetimes, seed=args.seed + 1,
        heldout=True, device=device)
    train_accuracy, heldout_accuracy = _fit_accuracy(
        train_x, train_y, test_x, test_y, seed=args.seed + 2,
        steps=args.steps)
    shuffled_train_accuracy, shuffled_heldout_accuracy = _fit_accuracy(
        train_x, train_y.roll(1), test_x, test_y, seed=args.seed + 3,
        steps=args.steps)
    report = {
        "schema": "contextual-binding-probe-v1",
        "claim_boundary": (
            "Supervised state-decoding diagnostic after both support outcomes; "
            "probe weights never enter controller training."),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "steps": args.steps,
        "chance_accuracy": 0.25,
        "train_accuracy": train_accuracy,
        "heldout_accuracy": heldout_accuracy,
        "shuffled_train_accuracy": shuffled_train_accuracy,
        "shuffled_heldout_accuracy": shuffled_heldout_accuracy,
        "joint_rule_decodable": (
            heldout_accuracy >= 0.80 and shuffled_heldout_accuracy <= 0.35),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
