"""Test whether separately decoded ingredients compose outside the raw latent.

All three dataset partitions are disjoint by lifetime. Supervision trains only
disposable probes; no probe weights are admitted to the behavioral agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _append, seed_everything
from .train_consolidator import _initial_memory


@torch.no_grad()
def _extract(model, *, start: int, lifetimes: int, batch_size: int,
             heldout: bool, device: torch.device):
    first_features, feedback_features = [], []
    first_labels, rewarded_labels, rule_labels = [], [], []
    captured: dict[str, torch.Tensor] = {}
    handle = model.observation_head.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("observations", inputs[0].detach()))
    try:
        for offset in range(0, lifetimes, batch_size):
            count = min(batch_size, lifetimes - offset)
            items = [generate_temporal_attention_lifetime(
                start + offset + index, heldout=heldout) for index in range(count)]
            memory = _initial_memory(model, items, device)
            _append(model, [item.supports[0] for item in items], memory, device)
            first_features.append(captured["observations"][:, 0].cpu())
            feedback_features.append(captured["observations"][:, 2].cpu())
            first_labels.append(torch.tensor(
                [item.support_features[0][0] for item in items], dtype=torch.long))
            rewarded_labels.append(torch.tensor(
                [item.support_features[0][item.rule] for item in items], dtype=torch.long))
            rule_labels.append(torch.tensor([item.rule for item in items], dtype=torch.long))
    finally:
        handle.remove()
    return (torch.cat(first_features), torch.cat(feedback_features),
            torch.cat(first_labels),
            torch.cat(rewarded_labels), torch.cat(rule_labels))


class Probe(nn.Module):
    def __init__(self, inputs: int, width: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _fit(model, train_x, train_y, *, steps: int, learning_rate: float,
         device: torch.device):
    model = model.to(device)
    x, y = train_x.to(device), train_y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _accuracy(model, x, y, device):
    return float((model(x.to(device)).argmax(-1).cpu() == y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ingredient-lifetimes", type=int, default=2048)
    parser.add_argument("--combiner-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--combiner-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, _ = _load(args.controller_checkpoint, args.consolidator_checkpoint, device)
    ingredient = _extract(
        model, start=23_000_000, lifetimes=args.ingredient_lifetimes,
        batch_size=args.batch_size, heldout=False, device=device)
    combiner = _extract(
        model, start=25_000_000, lifetimes=args.combiner_lifetimes,
        batch_size=args.batch_size, heldout=False, device=device)
    test = _extract(
        model, start=27_000_000, lifetimes=args.test_lifetimes,
        batch_size=args.batch_size, heldout=True, device=device)

    first_mean = ingredient[0].mean(0, keepdim=True)
    first_scale = ingredient[0].std(0, keepdim=True).clamp_min(1e-5)
    rewarded_mean = ingredient[1].mean(0, keepdim=True)
    rewarded_scale = ingredient[1].std(0, keepdim=True).clamp_min(1e-5)
    ingredient_first_x = (ingredient[0] - first_mean) / first_scale
    combiner_first_x = (combiner[0] - first_mean) / first_scale
    test_first_x = (test[0] - first_mean) / first_scale
    ingredient_rewarded_x = (ingredient[1] - rewarded_mean) / rewarded_scale
    combiner_rewarded_x = (combiner[1] - rewarded_mean) / rewarded_scale
    test_rewarded_x = (test[1] - rewarded_mean) / rewarded_scale
    seed_everything(args.seed)
    first_probe = _fit(
        Probe(ingredient_first_x.shape[1]), ingredient_first_x, ingredient[2],
        steps=args.probe_steps, learning_rate=3e-3, device=device)
    seed_everything(args.seed + 1)
    rewarded_probe = _fit(
        Probe(ingredient_rewarded_x.shape[1]), ingredient_rewarded_x, ingredient[3],
        steps=args.probe_steps, learning_rate=3e-3, device=device)

    @torch.no_grad()
    def decoded(first_x, rewarded_x):
        return torch.cat((first_probe(first_x.to(device)).softmax(-1),
                          rewarded_probe(rewarded_x.to(device)).softmax(-1)), dim=-1).cpu()

    stack_x = decoded(combiner_first_x, combiner_rewarded_x)
    test_stack_x = decoded(test_first_x, test_rewarded_x)
    seed_everything(args.seed + 2)
    combiner_probe = _fit(
        Probe(4, width=16), stack_x, combiner[4], steps=args.combiner_steps,
        learning_rate=1e-2, device=device)
    shuffled_accuracies = []
    for calibration_index in range(16):
        calibration_seed = args.seed + 3 + calibration_index
        shuffled = combiner[4][torch.randperm(
            combiner[4].numel(), generator=torch.Generator().manual_seed(
                calibration_seed))]
        seed_everything(calibration_seed)
        shuffled_probe = _fit(
            Probe(4, width=16), stack_x, shuffled, steps=args.combiner_steps,
            learning_rate=1e-2, device=device)
        shuffled_accuracies.append(_accuracy(
            shuffled_probe, test_stack_x, test[4], device))

    with torch.no_grad():
        first_predictions = first_probe(test_first_x.to(device)).argmax(-1).cpu()
        rewarded_predictions = rewarded_probe(
            test_rewarded_x.to(device)).argmax(-1).cpu()
        deterministic_composition = (first_predictions != rewarded_predictions).long()
        truth_composition = (test[2] != test[3]).long()
    report = {
        "schema": "temporal-two-stage-binding-probe-v1",
        "controller_frozen": True,
        "disposable_supervised_probe": True,
        "lifetime_splits_disjoint": True,
        "ingredient_train_examples": args.ingredient_lifetimes,
        "combiner_train_examples": args.combiner_lifetimes,
        "heldout_examples": args.test_lifetimes,
        "test_balance": {
            "first_identity_1_rate": float(test[2].float().mean()),
            "rewarded_identity_1_rate": float(test[3].float().mean()),
            "rewarded_was_first_1_rate": float(test[4].float().mean()),
        },
        "ingredient_sources": {
            "first_identity": "object_1_recurrent_state",
            "rewarded_identity": "post_feedback_recurrent_state",
        },
        "ingredient_accuracy": {
            "first_identity_train": _accuracy(
                first_probe, ingredient_first_x, ingredient[2], device),
            "first_identity_heldout": _accuracy(
                first_probe, test_first_x, test[2], device),
            "rewarded_identity_train": _accuracy(
                rewarded_probe, ingredient_rewarded_x, ingredient[3], device),
            "rewarded_identity_heldout": _accuracy(
                rewarded_probe, test_rewarded_x, test[3], device),
        },
        "composition_accuracy": {
            "learned_combiner_train": _accuracy(
                combiner_probe, stack_x, combiner[4], device),
            "learned_combiner_heldout": _accuracy(
                combiner_probe, test_stack_x, test[4], device),
            "deterministic_from_predicted_ingredients_heldout": float(
                (deterministic_composition == test[4]).float().mean()),
            "deterministic_from_true_ingredients_heldout": float(
                (truth_composition == test[4]).float().mean()),
            "shuffled_label_calibration_heldout": {
                "runs": shuffled_accuracies,
                "mean": float(torch.tensor(shuffled_accuracies).mean()),
                "standard_deviation": float(
                    torch.tensor(shuffled_accuracies).std(unbiased=False)),
                "minimum": min(shuffled_accuracies),
                "maximum": max(shuffled_accuracies),
            },
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
