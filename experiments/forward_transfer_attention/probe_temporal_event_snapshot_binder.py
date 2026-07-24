"""Disposable supervised capacity probe for cross-event recurrent snapshots."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .environment import generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _append, seed_everything
from .train_consolidator import _initial_memory


TRAIN_LOGICAL_START = 29_000_000
TEST_LOGICAL_START = 31_000_000


def _logical_lifetime_ids(start: int, lifetimes: int) -> set[int]:
    """Return task identities before any nuisance-render variants are made."""
    return set(range(start, start + lifetimes))


def _assert_disjoint_logical_splits(train_lifetimes: int,
                                    test_lifetimes: int) -> None:
    overlap = (_logical_lifetime_ids(TRAIN_LOGICAL_START, train_lifetimes)
               & _logical_lifetime_ids(TEST_LOGICAL_START, test_lifetimes))
    if overlap:
        raise ValueError(
            "train/test overlap in logical lifetime IDs before render augmentation: "
            f"{sorted(overlap)[:8]}")


def _subset_rendered_lifetimes(data, *, cached_lifetimes: int,
                               cached_variants: int,
                               subset_lifetimes: int | None,
                               subset_variants: int | None):
    use_lifetimes = subset_lifetimes or cached_lifetimes
    use_variants = subset_variants or cached_variants
    if not 1 <= use_lifetimes <= cached_lifetimes:
        raise ValueError("subset lifetimes must be within the cached lifetime count")
    if not 1 <= use_variants <= cached_variants:
        raise ValueError("subset variants must be within the cached variant count")
    selected = []
    for tensor in data:
        grouped = tensor.reshape(cached_lifetimes, cached_variants, *tensor.shape[1:])
        selected.append(grouped[:use_lifetimes, :use_variants].flatten(0, 1))
    return tuple(selected), use_lifetimes, use_variants


def _reverse_episode_events(episode):
    """Reverse object events while leaving the feedback event byte-identical."""
    order = np.asarray((1, 0, 2))
    return replace(
        episode,
        frames=episode.frames[order].copy(),
        pcm=episode.pcm[order].copy(),
        actions=episode.actions[order].copy(),
        subjects=episode.subjects[order].copy(),
        relations=episode.relations[order].copy(),
        objects=episode.objects[order].copy(),
    )


@torch.no_grad()
def _extract(model, *, start: int, lifetimes: int, batch_size: int,
             heldout: bool, feedback_mode: str, render_variants: int,
             reverse_events: bool = False,
             device: torch.device):
    if render_variants < 1:
        raise ValueError("render_variants must be positive")
    examples = lifetimes * render_variants
    snapshots, labels, auxiliary_labels = [], [], []
    captured: dict[str, torch.Tensor] = {}
    handle = model.observation_head.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("observations", inputs[0].detach()))
    try:
        for offset in range(0, examples, batch_size):
            count = min(batch_size, examples - offset)
            items = []
            for index in range(count):
                flat_index = offset + index
                logical_index, variant = divmod(flat_index, render_variants)
                logical_seed = start + logical_index
                render_seed = (None if render_variants == 1 else
                               start + 100_000_000 + logical_index * render_variants + variant)
                items.append(generate_temporal_attention_lifetime(
                    logical_seed, heldout=heldout, feedback_mode=feedback_mode,
                    render_seed=render_seed))
            episodes = [item.supports[0] for item in items]
            if reverse_events:
                episodes = [_reverse_episode_events(episode) for episode in episodes]
            memory = _initial_memory(model, items, device)
            _append(model, episodes, memory, device)
            snapshots.append(captured["observations"].cpu())
            labels.append(torch.tensor([
                1 - item.rule if reverse_events else item.rule for item in items
            ], dtype=torch.long))
            auxiliary_labels.append(torch.tensor([
                (item.support_features[0][1 if reverse_events else 0],
                 item.support_features[0][item.rule]) for item in items
            ], dtype=torch.long))
    finally:
        handle.remove()
    return torch.cat(snapshots), torch.cat(labels), torch.cat(auxiliary_labels)


class EventSnapshotBinder(nn.Module):
    """Generic position-aware relation module; contains no task semantics."""
    def __init__(self, input_width: int, width: int = 128, heads: int = 4,
                 layers: int = 2) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_width), nn.Linear(input_width, width), nn.GELU())
        self.positions = nn.Parameter(torch.randn(1, 3, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 4, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.relation = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, snapshots: torch.Tensor) -> torch.Tensor:
        events = self.project(snapshots) + self.positions
        cls = self.cls.expand(events.shape[0], -1, -1)
        return self.head(self.relation(torch.cat((cls, events), dim=1))[:, 0])


class PairwiseSnapshotBinder(nn.Module):
    """Generic multiplicative/difference relations across event positions."""
    def __init__(self, input_width: int, width: int = 96) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_width), nn.Linear(input_width, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width))
        self.positions = nn.Parameter(torch.randn(1, 3, width) * 0.02)
        # Three event vectors plus product and absolute difference for each pair.
        self.head = nn.Sequential(
            nn.Linear(width * 9, width * 3), nn.GELU(), nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, snapshots: torch.Tensor) -> torch.Tensor:
        events = self.project(snapshots) + self.positions
        relations = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            relations.extend((events[:, left] * events[:, right],
                              (events[:, left] - events[:, right]).abs()))
        return self.head(torch.cat((*events.unbind(dim=1), *relations), dim=-1))


@torch.no_grad()
def _predictions(model, x, batch_size, device):
    model.eval()
    predictions = []
    for offset in range(0, x.shape[0], batch_size):
        predictions.append(
            model(x[offset:offset + batch_size].to(device)).argmax(-1).cpu())
    return torch.cat(predictions)


def _accuracy(model, x, y, batch_size, device):
    return float((_predictions(model, x, batch_size, device) == y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=32)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--architecture", choices=("transformer", "pairwise"),
                        default="transformer")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--feedback-mode", choices=("white-button", "color-button"),
                        default="white-button")
    parser.add_argument("--train-render-variants", type=int, default=1)
    parser.add_argument("--test-render-variants", type=int, default=1)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--train-subset-lifetimes", type=int)
    parser.add_argument("--train-subset-render-variants", type=int)
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--causal-reversal-audit", action="store_true")
    parser.add_argument("--counterfactual-test-cache", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    _assert_disjoint_logical_splits(args.train_lifetimes, args.test_lifetimes)
    cache_expectations = {
        "feedback_mode": args.feedback_mode,
        "controller_checkpoint": str(args.controller_checkpoint),
    }

    def read_cache(path, lifetimes, variants, heldout, *, reverse_events=False):
        if path is None or not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {**cache_expectations, "lifetimes": lifetimes,
                    "render_variants": variants, "heldout": heldout}
        if reverse_events:
            expected["reverse_events"] = True
        if payload.get("metadata") != expected:
            raise ValueError(f"cache metadata mismatch for {path}")
        return (payload["snapshots"], payload["labels"],
                payload["auxiliary_labels"])

    train = read_cache(args.train_cache, args.train_lifetimes,
                       args.train_render_variants, False)
    test = read_cache(args.test_cache, args.test_lifetimes,
                      args.test_render_variants, True)
    if train is None or test is None:
        controller, _ = _load(
            args.controller_checkpoint, args.consolidator_checkpoint, device)
        if train is None:
            train = _extract(
                controller, start=TRAIN_LOGICAL_START, lifetimes=args.train_lifetimes,
                batch_size=args.batch_size, heldout=False,
                feedback_mode=args.feedback_mode,
                render_variants=args.train_render_variants, device=device)
            if args.train_cache is not None:
                args.train_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"snapshots": train[0], "labels": train[1],
                            "auxiliary_labels": train[2],
                            "metadata": {**cache_expectations,
                                         "lifetimes": args.train_lifetimes,
                                         "render_variants": args.train_render_variants,
                                         "heldout": False}}, args.train_cache)
        if test is None:
            test = _extract(
                controller, start=TEST_LOGICAL_START, lifetimes=args.test_lifetimes,
                batch_size=args.batch_size, heldout=True,
                feedback_mode=args.feedback_mode,
                render_variants=args.test_render_variants, device=device)
            if args.test_cache is not None:
                args.test_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"snapshots": test[0], "labels": test[1],
                            "auxiliary_labels": test[2],
                            "metadata": {**cache_expectations,
                                         "lifetimes": args.test_lifetimes,
                                         "render_variants": args.test_render_variants,
                                         "heldout": True}}, args.test_cache)
    train, effective_train_lifetimes, effective_train_variants = (
        _subset_rendered_lifetimes(
            train, cached_lifetimes=args.train_lifetimes,
            cached_variants=args.train_render_variants,
            subset_lifetimes=args.train_subset_lifetimes,
            subset_variants=args.train_subset_render_variants))
    train_x, train_y, train_auxiliary_y = train
    test_x, test_y, test_auxiliary_y = test
    if args.shuffle_labels:
        train_y = train_y[torch.randperm(
            train_y.numel(), generator=torch.Generator().manual_seed(args.seed + 1))]
    mean = train_x.mean((0, 1), keepdim=True)
    scale = train_x.std((0, 1), keepdim=True).clamp_min(1e-5)
    train_x, test_x = (train_x - mean) / scale, (test_x - mean) / scale
    # Make probe initialization independent of whether sensory snapshots were
    # freshly extracted or loaded from cache.
    seed_everything(args.seed)
    model = (PairwiseSnapshotBinder(train_x.shape[-1], width=args.width)
             if args.architecture == "pairwise"
             else EventSnapshotBinder(train_x.shape[-1], width=args.width)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for step in range(1, args.steps + 1):
        if args.batch_size >= train_y.numel():
            indices = torch.arange(train_y.numel())
        else:
            indices = torch.randint(
            train_y.numel(), (args.batch_size,), generator=generator)
        x, y = train_x[indices].to(device), train_y[indices].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "train_accuracy": _accuracy(
                    model, train_x, train_y, args.batch_size, device),
                "test_accuracy": _accuracy(
                    model, test_x, test_y, args.batch_size, device),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "schema": "temporal-event-snapshot-binder-probe-v1",
        "controller_frozen": True,
        "disposable_supervised_probe": True,
        "game_state_inputs": False,
        "input": "three-recurrent-event-snapshots",
        "architecture": args.architecture,
        "width": args.width,
        "feedback_mode": args.feedback_mode,
        "shuffle_labels": args.shuffle_labels,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "train_render_variants": args.train_render_variants,
        "test_render_variants": args.test_render_variants,
        "effective_train_lifetimes": effective_train_lifetimes,
        "effective_train_render_variants": effective_train_variants,
        "train_examples": train_y.numel(),
        "test_examples": test_y.numel(),
        "optimizer_steps": args.steps,
        "train_label_rate": float(train_y.float().mean()),
        "test_label_rate": float(test_y.float().mean()),
        "train_first_identity_rate": float(train_auxiliary_y[:, 0].float().mean()),
        "train_rewarded_identity_rate": float(train_auxiliary_y[:, 1].float().mean()),
        "test_first_identity_rate": float(test_auxiliary_y[:, 0].float().mean()),
        "test_rewarded_identity_rate": float(test_auxiliary_y[:, 1].float().mean()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "best_train_accuracy": max(row["train_accuracy"] for row in history),
        "best_test_accuracy": max(row["test_accuracy"] for row in history),
    }
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "architecture": args.architecture,
                    "width": args.width, "input_width": train_x.shape[-1],
                    "mean": mean.cpu(), "scale": scale.cpu()}, args.model_output)
    if args.causal_reversal_audit:
        normal_predictions = _predictions(
            model, test_x, args.batch_size, device)
        counterfactual = read_cache(
            args.counterfactual_test_cache, args.test_lifetimes,
            args.test_render_variants, True, reverse_events=True)
        if counterfactual is None:
            controller, _ = _load(
                args.controller_checkpoint, args.consolidator_checkpoint, device)
            counterfactual = _extract(
                controller, start=TEST_LOGICAL_START,
                lifetimes=args.test_lifetimes, batch_size=args.batch_size,
                heldout=True, feedback_mode=args.feedback_mode,
                render_variants=args.test_render_variants,
                reverse_events=True, device=device)
            if args.counterfactual_test_cache is not None:
                args.counterfactual_test_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "snapshots": counterfactual[0], "labels": counterfactual[1],
                    "auxiliary_labels": counterfactual[2],
                    "metadata": {**cache_expectations,
                                 "lifetimes": args.test_lifetimes,
                                 "render_variants": args.test_render_variants,
                                 "heldout": True,
                                 "reverse_events": True}},
                           args.counterfactual_test_cache)
        reversed_x, reversed_y, _ = counterfactual
        reversed_x = (reversed_x - mean) / scale
        reversed_predictions = _predictions(
            model, reversed_x, args.batch_size, device)
        report["causal_reversal_audit"] = {
            "operation": "reverse-object-events-before-frozen-controller-replay",
            "normal_accuracy": float((normal_predictions == test_y).float().mean()),
            "reversed_relabeled_accuracy": float(
                (reversed_predictions == reversed_y).float().mean()),
            "reversed_stale_label_accuracy": float(
                (reversed_predictions == test_y).float().mean()),
            "prediction_flip_rate": float(
                (reversed_predictions != normal_predictions).float().mean()),
            "per_example": {
                "normal_labels": test_y.tolist(),
                "normal_predictions": normal_predictions.tolist(),
                "reversed_labels": reversed_y.tolist(),
                "reversed_predictions": reversed_predictions.tolist(),
            },
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
