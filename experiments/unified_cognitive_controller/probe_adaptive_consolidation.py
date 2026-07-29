"""Probe whether learner-visible state can safely stop consolidation.

This is deliberately a passive diagnostic.  It learns from complete prefix
trajectories after the verifier has measured them, but it never changes the
controller that produced those trajectories.  Only if the cross-stream policy
beats the fixed budget while preserving every gate should a policy head be
integrated into the controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable

import torch
from torch import nn


FEATURES = (
    "log_optimizer_update",
    "new_batch_accuracy",
    "skill_loss",
    "retention_loss",
    "locality",
    "total_loss",
    "skill_loss_reduction",
    "retention_loss_reduction",
    "total_loss_reduction",
)


def feature_names(width: int) -> tuple[str, ...]:
    """Return the auditable schema for one observation width."""
    if width < len(FEATURES):
        raise ValueError("feature width is smaller than the base schema")
    return (
        *FEATURES,
        *(f"sensory_summary_{index}"
          for index in range(width - len(FEATURES))),
    )


@dataclass(frozen=True)
class PrefixObservation:
    """One causally completed consolidation prefix."""

    stream: str
    budget: int
    features: tuple[float, ...]
    passed: bool


class ConsolidationStopProbe(nn.Module):
    """Tiny task-agnostic pass-probability probe."""

    def __init__(self, feature_count: int, hidden: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _latest_history(payload: dict[str, object]) -> dict[str, float]:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("report has no learner-visible optimization history")
    row = history[-1]
    if not isinstance(row, dict):
        raise ValueError("history row must be a mapping")
    return row


def _stream_key(payload: dict[str, object]) -> str:
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("report has no configuration")
    # Seed identifies the complete rendered experience packet.  Blend is
    # intentionally excluded: it is private generator metadata and must never
    # become a learner feature.
    return str(int(configuration["seed"]))


def load_observations(paths: Iterable[Path]) -> list[PrefixObservation]:
    """Load reports and construct learner-visible prefix features."""
    raw: dict[str, list[tuple[int, dict[str, float], bool]]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "pair-magnitude-appearance-bridge-v1":
            continue
        configuration = payload["configuration"]
        budget = int(configuration["epochs_per_batch"])
        stream = _stream_key(payload)
        raw.setdefault(stream, []).append((
            budget, _latest_history(payload),
            bool(payload["all_gates_passed"])))
    observations: list[PrefixObservation] = []
    for stream, rows in raw.items():
        rows.sort(key=lambda value: value[0])
        previous: dict[str, float] | None = None
        seen: set[int] = set()
        for budget, row, passed in rows:
            if budget in seen:
                raise ValueError(
                    f"duplicate budget {budget} in stream {stream}")
            seen.add(budget)
            reductions = (
                0.0, 0.0, 0.0) if previous is None else (
                    float(previous["skill_loss"]) - float(row["skill_loss"]),
                    float(previous["retention_loss"])
                    - float(row["retention_loss"]),
                    float(previous["total_loss"]) - float(row["total_loss"]),
                )
            features = (
                math.log1p(budget),
                float(row["new_batch_accuracy"]),
                float(row["skill_loss"]),
                float(row["retention_loss"]),
                float(row["locality"]),
                float(row["total_loss"]),
                *reductions,
                *(float(value) for value in row.get(
                    "sensory_summary", ())),
            )
            if not all(math.isfinite(value) for value in features):
                raise ValueError("non-finite consolidation feature")
            observations.append(PrefixObservation(
                stream=stream, budget=budget, features=features,
                passed=passed))
            previous = row
    if not observations:
        raise ValueError("no magnitude consolidation reports found")
    widths = {len(row.features) for row in observations}
    if len(widths) != 1:
        raise ValueError("mixed consolidation feature schemas")
    return observations


def _tensors(
        observations: list[PrefixObservation],
        *, mean: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [row.features for row in observations], dtype=torch.float32)
    y = (
        torch.tensor(
            [float(row.passed) for row in observations],
            dtype=torch.float32)
        if labels is None else labels.float())
    if mean is None:
        mean = x.mean(dim=0)
    if scale is None:
        scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (x - mean) / scale, y, mean, scale


def train_probe(
        observations: list[PrefixObservation], *, seed: int,
        labels: torch.Tensor | None = None, hidden: int = 8,
        updates: int = 1000,
        ) -> tuple[ConsolidationStopProbe, torch.Tensor, torch.Tensor]:
    """Fit one tiny probe deterministically."""
    torch.manual_seed(seed)
    x, y, mean, scale = _tensors(observations, labels=labels)
    model = ConsolidationStopProbe(
        len(observations[0].features), hidden=hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=1e-3)
    positives = float(y.sum())
    negatives = float(y.numel() - y.sum())
    positive_weight = torch.tensor(
        negatives / max(positives, 1.0), dtype=torch.float32)
    for _ in range(updates):
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), mean, scale


@torch.no_grad()
def predict(
        model: ConsolidationStopProbe,
        observations: list[PrefixObservation],
        mean: torch.Tensor, scale: torch.Tensor,
        ) -> list[float]:
    x, _, _, _ = _tensors(observations, mean=mean, scale=scale)
    return torch.sigmoid(model(x)).tolist()


def select_budgets(
        observations: list[PrefixObservation], probabilities: list[float],
        *, threshold: float,
        ) -> dict[str, PrefixObservation]:
    """Choose the first predicted-safe prefix, else the largest prefix."""
    grouped: dict[str, list[tuple[PrefixObservation, float]]] = {}
    for row, probability in zip(
            observations, probabilities, strict=True):
        grouped.setdefault(row.stream, []).append((row, probability))
    selected: dict[str, PrefixObservation] = {}
    for stream, rows in grouped.items():
        rows.sort(key=lambda item: item[0].budget)
        passing = [row for row, probability in rows
                   if probability >= threshold]
        selected[stream] = passing[0] if passing else rows[-1][0]
    return selected


def _policy_metrics(
        selected: dict[str, PrefixObservation],
        ) -> dict[str, float | int | bool]:
    rows = list(selected.values())
    return {
        "streams": len(rows),
        "all_streams_passed": all(row.passed for row in rows),
        "pass_rate": sum(row.passed for row in rows) / len(rows),
        "mean_budget": sum(row.budget for row in rows) / len(rows),
        "maximum_budget": max(row.budget for row in rows),
    }


def calibrate_threshold(
        observations: list[PrefixObservation], probabilities: list[float],
        ) -> float:
    """Pick the cheapest training threshold that never selects a failure."""
    candidates = [value / 100 for value in range(50, 100)]
    safe: list[tuple[float, float]] = []
    for threshold in candidates:
        metrics = _policy_metrics(select_budgets(
            observations, probabilities, threshold=threshold))
        if metrics["all_streams_passed"]:
            safe.append((float(metrics["mean_budget"]), threshold))
    # A threshold of one always falls back to the maximum measured budget.
    return min(safe)[1] if safe else 1.0


def _fixed_metrics(
        observations: list[PrefixObservation], budget: int,
        ) -> dict[str, float | int | bool | None]:
    rows = [row for row in observations if row.budget == budget]
    streams = {row.stream for row in observations}
    if len(rows) != len(streams):
        return {
            "streams": len(rows), "all_streams_passed": False,
            "pass_rate": None, "mean_budget": budget,
            "maximum_budget": budget,
        }
    return _policy_metrics({row.stream: row for row in rows})


def _folds(streams: list[str], count: int) -> list[list[str]]:
    return [streams[index::count] for index in range(count)]


def cross_validate(
        observations: list[PrefixObservation], *, seed: int,
        fold_count: int = 3, hidden: int = 8, updates: int = 1000,
        shuffle_labels: bool = False,
        ) -> dict[str, object]:
    streams = sorted({row.stream for row in observations})
    if len(streams) < fold_count:
        raise ValueError("fewer streams than requested folds")
    random.Random(seed).shuffle(streams)
    heldout_selections: dict[str, PrefixObservation] = {}
    fold_reports = []
    generator = torch.Generator().manual_seed(seed + 991)
    for fold_index, heldout_streams in enumerate(_folds(
            streams, fold_count)):
        heldout_set = set(heldout_streams)
        train = [row for row in observations
                 if row.stream not in heldout_set]
        heldout = [row for row in observations
                   if row.stream in heldout_set]
        labels = None
        if shuffle_labels:
            original = torch.tensor(
                [float(row.passed) for row in train])
            labels = original[torch.randperm(
                original.numel(), generator=generator)]
        model, mean, scale = train_probe(
            train, seed=seed + fold_index, labels=labels,
            hidden=hidden, updates=updates)
        train_probabilities = predict(model, train, mean, scale)
        threshold = calibrate_threshold(train, train_probabilities)
        heldout_probabilities = predict(model, heldout, mean, scale)
        selected = select_budgets(
            heldout, heldout_probabilities, threshold=threshold)
        heldout_selections.update(selected)
        fold_reports.append({
            "fold": fold_index,
            "heldout_streams": heldout_streams,
            "threshold": threshold,
            "selected": {
                stream: {
                    "budget": row.budget, "passed": row.passed}
                for stream, row in selected.items()
            },
            "metrics": _policy_metrics(selected),
        })
    return {
        "folds": fold_reports,
        "aggregate": _policy_metrics(heldout_selections),
        "fixed_8": _fixed_metrics(observations, 8),
        "fixed_16": _fixed_metrics(observations, 16),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports", type=Path, nargs="+", required=True,
        help="directories or individual bridge reports")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21850)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--updates", type=int, default=1000)
    args = parser.parse_args()
    paths: list[Path] = []
    for root in args.reports:
        paths.extend(sorted(root.glob("*.json")) if root.is_dir() else [root])
    observations = load_observations(paths)
    normal = cross_validate(
        observations, seed=args.seed, fold_count=args.folds,
        hidden=args.hidden, updates=args.updates)
    shuffled = cross_validate(
        observations, seed=args.seed, fold_count=args.folds,
        hidden=args.hidden, updates=args.updates, shuffle_labels=True)
    normal_metrics = normal["aggregate"]
    shuffled_metrics = shuffled["aggregate"]
    accepted = (
        bool(normal_metrics["all_streams_passed"])
        and float(normal_metrics["mean_budget"]) < 8.0
        and (
            not bool(shuffled_metrics["all_streams_passed"])
            or float(shuffled_metrics["mean_budget"])
            > float(normal_metrics["mean_budget"])))
    report = {
        "schema": "adaptive-consolidation-passive-probe-v1",
        "claim_boundary": (
            "Features are learner-visible optimization statistics. Labels "
            "come only from completed verifier and retention audits. The "
            "probe is discarded and changes no controller parameter."),
        "feature_schema": feature_names(len(observations[0].features)),
        "streams": len({row.stream for row in observations}),
        "observations": len(observations),
        "normal": normal,
        "shuffled_label_control": shuffled,
        "gate": {
            "all_heldout_streams_passed":
                normal_metrics["all_streams_passed"],
            "mean_budget_below_fixed_8":
                float(normal_metrics["mean_budget"]) < 8.0,
            "shuffled_control_worse": (
                not bool(shuffled_metrics["all_streams_passed"])
                or float(shuffled_metrics["mean_budget"])
                > float(normal_metrics["mean_budget"])),
            "accepted_for_controller_integration": accepted,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "streams": report["streams"],
        "mean_budget": normal_metrics["mean_budget"],
        "all_streams_passed": normal_metrics["all_streams_passed"],
        "shuffled_mean_budget": shuffled_metrics["mean_budget"],
        "accepted_for_controller_integration": accepted,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
