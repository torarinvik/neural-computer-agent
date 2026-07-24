"""Tiny experienced-vs-fresh audit of palette-invariant temporal learning.

This is a disposable supervised diagnostic. It measures whether an already
learned temporal binder reduces the experience needed to learn the same
relation over new visual-identity combinations. No verifier metadata enters the
frozen controller or the deployed agent.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .environment import (
    _independent_choice, generate_temporal_attention_lifetime)
from .probe_temporal_event_snapshot_binder import (
    PairwiseSnapshotBinder, _accuracy, _reverse_episode_events)
from .probe_temporal_rule_memory import _load
from .train import _append, seed_everything
from .train_consolidator import _initial_memory


TRAIN_START = 41_000_000
VALIDATION_START = 43_000_000
TEST_START = 45_000_000


class PaletteInvariantBinder(PairwiseSnapshotBinder):
    """Expose the generic relation latent for cross-render consistency."""

    def relation_latent(self, snapshots: torch.Tensor) -> torch.Tensor:
        events = self.project(snapshots) + self.positions
        relations = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            relations.extend((events[:, left] * events[:, right],
                              (events[:, left] - events[:, right]).abs()))
        features = torch.cat((*events.unbind(dim=1), *relations), dim=-1)
        return self.head[:-1](features)

    def forward(self, snapshots: torch.Tensor) -> torch.Tensor:
        return self.head[-1](self.relation_latent(snapshots))


def _parse_pairs(value: str) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        tuple(int(identity) for identity in pair.split(","))
        for pair in value.split(";")
    )
    if not pairs or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError("palettes must be semicolon-separated distinct pairs")
    return pairs


def _palette_for(seed: int, palettes: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    digest = hashlib.blake2b(
        f"palette-split-v1:{seed}".encode(), digest_size=8).digest()
    return palettes[int.from_bytes(digest, "little") % len(palettes)]


def _balanced_specs(start: int, lifetimes: int, palettes, heldout: bool):
    """Select an exactly balanced verifier-side palette × rule evaluation set."""
    cells = len(palettes) * 2
    if lifetimes % cells:
        raise ValueError(
            f"lifetimes must be divisible by {cells} for exact balance")
    quota = lifetimes // cells
    counts = {(palette, rule): 0 for palette in palettes for rule in (0, 1)}
    selected = []
    seed = start
    while len(selected) < lifetimes:
        palette = _palette_for(seed, palettes)
        rule = _independent_choice(
            seed, heldout, "temporal-atom-rule", 2)
        cell = (palette, rule)
        if counts[cell] < quota:
            selected.append((seed, palette))
            counts[cell] += 1
        seed += 1
    return selected


def _balanced_logical_seeds(start: int, lifetimes: int, heldout: bool):
    if lifetimes % 2:
        raise ValueError("logical lifetimes must be even for exact rule balance")
    quota = lifetimes // 2
    counts = [0, 0]
    selected = []
    seed = start
    while len(selected) < lifetimes:
        rule = _independent_choice(
            seed, heldout, "temporal-atom-rule", 2)
        if counts[rule] < quota:
            selected.append(seed)
            counts[rule] += 1
        seed += 1
    return selected


@torch.no_grad()
def _extract(model, *, start: int, lifetimes: int, palettes, batch_size: int,
             heldout: bool, device: torch.device,
             all_palettes_per_lifetime: bool = False,
             reverse_events: bool = False):
    snapshots, labels, palette_indices, identities = [], [], [], []
    if all_palettes_per_lifetime:
        specs = [
            (seed, palette)
            for seed in _balanced_logical_seeds(start, lifetimes, heldout)
            for palette in palettes
        ]
    else:
        specs = _balanced_specs(start, lifetimes, palettes, heldout)
    examples = len(specs)
    captured: dict[str, torch.Tensor] = {}
    handle = model.observation_head.register_forward_pre_hook(
        lambda _module, inputs:
        captured.__setitem__("observations", inputs[0].detach()))
    try:
        for offset in range(0, examples, batch_size):
            count = min(batch_size, examples - offset)
            batch_specs = specs[offset:offset + count]
            seeds = [seed for seed, _palette in batch_specs]
            selected = [palette for _seed, palette in batch_specs]
            items = [
                generate_temporal_attention_lifetime(
                    seed, heldout=heldout, feedback_mode="color-button",
                    color_ids=palette)
                for seed, palette in zip(seeds, selected)
            ]
            memory = _initial_memory(model, items, device)
            episodes = [item.supports[0] for item in items]
            if reverse_events:
                episodes = [_reverse_episode_events(episode)
                            for episode in episodes]
            _append(model, episodes, memory, device)
            snapshots.append(captured["observations"].cpu())
            labels.append(torch.tensor(
                [1 - item.rule if reverse_events else item.rule
                 for item in items], dtype=torch.long))
            identities.append(torch.tensor([
                (
                    palette[item.support_features[0][
                        1 if reverse_events else 0]],
                    palette[item.support_features[0][
                        0 if reverse_events else 1]],
                    palette[item.support_features[0][item.rule]],
                )
                for item, palette in zip(items, selected)
            ], dtype=torch.long))
            palette_indices.append(torch.tensor(
                [palettes.index(palette) for palette in selected],
                dtype=torch.long))
    finally:
        handle.remove()
    return (torch.cat(snapshots), torch.cat(labels),
            torch.cat(palette_indices), torch.cat(identities))


def _first_sustained_threshold(history, threshold: float):
    """Require two consecutive evaluations to avoid rewarding a lucky spike."""
    for previous, current in zip(history, history[1:]):
        if (previous["validation_accuracy"] >= threshold and
                current["validation_accuracy"] >= threshold):
            return {
                "step": previous["step"],
                "examples_seen": previous["examples_seen"],
            }
    return None


def _learning_score(history, chance: float) -> float:
    """Area above chance: high early verified accuracy earns the most credit."""
    if not history:
        return 0.0
    return float(sum(
        max(0.0, row["validation_accuracy"] - chance)
        for row in history
    ) / len(history))


def _train_arm(name, initial_state, train_x, train_y, validation_x,
               validation_y, test_x, test_y, *, width, batch_size, steps,
               eval_every, learning_rate, seed, device,
               variants_per_lifetime=1, consistency_weight=0.0):
    seed_everything(seed)
    model = PaletteInvariantBinder(train_x.shape[-1], width=width).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    index_generator = torch.Generator().manual_seed(seed + 1009)
    history = []
    if train_y.numel() % variants_per_lifetime:
        raise ValueError("training examples must form complete variant groups")
    logical_lifetimes = train_y.numel() // variants_per_lifetime
    for step in range(1, steps + 1):
        logical_batch = min(
            logical_lifetimes,
            max(1, batch_size // variants_per_lifetime))
        logical_indices = torch.randint(
            logical_lifetimes, (logical_batch,), generator=index_generator)
        offsets = torch.arange(variants_per_lifetime)
        indices = (
            logical_indices[:, None] * variants_per_lifetime + offsets[None]
        ).flatten()
        batch_x = train_x[indices].to(device)
        latent = model.relation_latent(batch_x)
        logits = model.head[-1](latent)
        loss = nn.functional.cross_entropy(
            logits, train_y[indices].to(device))
        consistency_loss = torch.zeros((), device=device)
        if variants_per_lifetime > 1 and consistency_weight:
            grouped = latent.reshape(
                logical_batch, variants_per_lifetime, -1)
            consistency_loss = (
                grouped - grouped.mean(dim=1, keepdim=True)
            ).square().mean()
            loss = loss + consistency_weight * consistency_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            row = {
                "step": step,
                "examples_seen": step * indices.numel(),
                "loss": float(loss.detach()),
                "consistency_loss": float(consistency_loss.detach()),
                "train_accuracy": _accuracy(
                    model, train_x, train_y, batch_size, device),
                "validation_accuracy": _accuracy(
                    model, validation_x, validation_y, batch_size, device),
            }
            history.append(row)
            print(json.dumps({"arm": name, **row}, sort_keys=True), flush=True)
    chance = max(
        float((validation_y == 0).float().mean()),
        float((validation_y == 1).float().mean()))
    return {
        "history": history,
        "test_accuracy_final": _accuracy(
            model, test_x, test_y, batch_size, device),
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "learning_score_above_chance": _learning_score(history, chance),
        "examples_to_threshold": {
            str(threshold): _first_sustained_threshold(history, threshold)
            for threshold in (0.60, 0.70, 0.80)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--experienced-binder-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-palettes", default="0,1;1,2;2,3")
    parser.add_argument("--test-palettes", default="0,2;0,3;1,3")
    parser.add_argument("--train-lifetimes", type=int, default=120)
    parser.add_argument("--validation-lifetimes", type=int, default=120)
    parser.add_argument("--test-lifetimes", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--all-train-palettes-per-lifetime",
                        action="store_true")
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_palettes = _parse_pairs(args.train_palettes)
    test_palettes = _parse_pairs(args.test_palettes)
    if set(train_palettes) & set(test_palettes):
        raise ValueError("train and held-out palette pairs must be disjoint")

    controller, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device)
    train = _extract(
        controller, start=TRAIN_START, lifetimes=args.train_lifetimes,
        palettes=train_palettes, batch_size=args.batch_size, heldout=False,
        device=device,
        all_palettes_per_lifetime=args.all_train_palettes_per_lifetime)
    validation = _extract(
        controller, start=VALIDATION_START,
        lifetimes=args.validation_lifetimes, palettes=test_palettes,
        batch_size=args.batch_size, heldout=True, device=device)
    test = _extract(
        controller, start=TEST_START, lifetimes=args.test_lifetimes,
        palettes=test_palettes, batch_size=args.batch_size, heldout=True,
        device=device)
    train_x, train_y, train_palette, _train_identities = train
    validation_x, validation_y, validation_palette, _validation_identities = validation
    test_x, test_y, test_palette, _test_identities = test

    payload = torch.load(
        args.experienced_binder_checkpoint, map_location="cpu",
        weights_only=False)
    experienced_state = payload.get("model", payload)
    mean = payload["mean"].cpu()
    scale = payload["scale"].cpu().clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    test_x = (test_x - mean) / scale

    common = dict(
        train_x=train_x, validation_x=validation_x,
        validation_y=validation_y, test_x=test_x, test_y=test_y,
        width=args.width, batch_size=args.batch_size, steps=args.steps,
        eval_every=args.eval_every, learning_rate=args.learning_rate,
        seed=args.seed, device=device,
        variants_per_lifetime=(
            len(train_palettes)
            if args.all_train_palettes_per_lifetime else 1),
        consistency_weight=args.consistency_weight)
    experienced = _train_arm(
        "experienced", experienced_state, train_y=train_y, **common)
    fresh = _train_arm(
        "fresh", None, train_y=train_y, **common)
    shuffled_generator = torch.Generator().manual_seed(args.seed + 77)
    if args.all_train_palettes_per_lifetime:
        grouped_y = train_y.reshape(
            args.train_lifetimes, len(train_palettes))
        order = torch.randperm(
            args.train_lifetimes, generator=shuffled_generator)
        shuffled_y = grouped_y[order].flatten()
    else:
        shuffled_y = train_y[torch.randperm(
            train_y.numel(), generator=shuffled_generator)]
    shuffled = _train_arm(
        "shuffled-label", None, train_y=shuffled_y, **common)

    validation_chance = max(
        float((validation_y == 0).float().mean()),
        float((validation_y == 1).float().mean()))
    test_chance = max(
        float((test_y == 0).float().mean()),
        float((test_y == 1).float().mean()))
    report = {
        "schema": "palette-sample-efficiency-probe-v1",
        "disposable_supervised_diagnostic": True,
        "controller_frozen": True,
        "game_state_inputs": False,
        "primary_objective":
            "verified held-out accuracy per example presentation",
        "train_palettes": train_palettes,
        "heldout_palettes": test_palettes,
        "palette_pair_overlap": False,
        "train_unique_lifetimes": args.train_lifetimes,
        "train_example_count": int(train_y.numel()),
        "all_train_palettes_per_lifetime":
            args.all_train_palettes_per_lifetime,
        "consistency_weight": args.consistency_weight,
        "validation_unique_lifetimes": args.validation_lifetimes,
        "test_unique_lifetimes": args.test_lifetimes,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "validation_majority": validation_chance,
        "test_majority": test_chance,
        "palette_counts": {
            "train": torch.bincount(
                train_palette, minlength=len(train_palettes)).tolist(),
            "validation": torch.bincount(
                validation_palette, minlength=len(test_palettes)).tolist(),
            "test": torch.bincount(
                test_palette, minlength=len(test_palettes)).tolist(),
        },
        "arms": {
            "experienced": experienced,
            "fresh": fresh,
            "shuffled_label": shuffled,
        },
    }
    report["transfer"] = {
        "learning_score_ratio_experienced_over_fresh": (
            experienced["learning_score_above_chance"] /
            max(fresh["learning_score_above_chance"], 1e-9)),
        "final_test_accuracy_delta_experienced_minus_fresh": (
            experienced["test_accuracy_final"] -
            fresh["test_accuracy_final"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
