"""Pressure-test replay-free online adaptation of the external regime trigger.

The detector is first trained on stable versus fully shifted opaque banks.  It
then receives a non-periodic stream containing stable intervals, partially
overlapping shifts, and disjoint shifts.  After each one-shot decision it gets
only the scalar verifier utility for that fresh pair and updates in place.
Earlier pairs are never replayed.  Held-out stable, partial, and disjoint
families measure whether the new boundary is acquired without forgetting the
old ones.

This isolates an external learning boundary.  The canonical controller and
event encoder stay frozen, and no regime ID, task label, or semantic feature
enters the detector ABI.  It is not a claim of general continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    GatedResidualRegimeChangePolicy,
    OpaqueRegimeChangePolicy,
)

from .external_temporal_query_address_growth import EVENT_WIDTH, _build
from .external_temporal_shared_basis_learned_regime_trigger import (
    DETECTOR_HIDDEN,
    DETECTOR_LEARNING_RATE,
    _evaluate_detector,
    _train_detector,
)
from .external_temporal_shared_basis_policy_growth import _digest

ONLINE_ADAPTATION_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-regime-policy-"
    "online-adaptation.v1"
)
ONLINE_RECORDS = 8
ONLINE_NOISE_SCALE = 0.002
ONLINE_TEMPERATURE = 0.65
ONLINE_STAGE_PLAN = (
    ("stable", 7),
    ("partial", 13),
    ("stable", 4),
    ("disjoint", 19),
    ("stable", 6),
    ("partial", 11),
    ("disjoint", 9),
    ("stable", 3),
)


def _partial_pair(
    *,
    seed: int,
    family: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if family not in {"stable", "partial", "disjoint"}:
        raise ValueError("online regime family is invalid")
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(
        torch.randn(EVENT_WIDTH, 6, generator=generator)
    ).Q[:, :6]
    current = (
        torch.randn(ONLINE_RECORDS, 2, generator=generator)
        @ basis[:, :2].transpose(0, 1)
        + ONLINE_NOISE_SCALE
        * torch.randn(ONLINE_RECORDS, EVENT_WIDTH, generator=generator)
    )
    columns = {
        "stable": (0, 2),
        "partial": (1, 3),
        "disjoint": (2, 4),
    }[family]
    incoming = (
        torch.randn(ONLINE_RECORDS, 2, generator=generator)
        @ basis[:, columns[0] : columns[1]].transpose(0, 1)
        + ONLINE_NOISE_SCALE
        * torch.randn(ONLINE_RECORDS, EVENT_WIDTH, generator=generator)
    )
    return F.normalize(current, dim=-1), F.normalize(incoming, dim=-1), family != "stable"


@torch.no_grad()
def _evaluate_families(
    policy: OpaqueRegimeChangePolicy,
    *,
    seed: int,
    episodes: int = 128,
) -> dict[str, float]:
    occupied = torch.ones(1, ONLINE_RECORDS, dtype=torch.bool)
    scores: dict[str, list[float]] = {"stable_keep": [], "partial_replace": [], "disjoint_replace": []}
    for family, name in (
        ("stable", "stable_keep"),
        ("partial", "partial_replace"),
        ("disjoint", "disjoint_replace"),
    ):
        target_replace = family != "stable"
        for episode in range(episodes):
            current, incoming, _target = _partial_pair(
                seed=seed + episode + (10_000 if family == "partial" else 20_000 if family == "disjoint" else 0),
                family=family,
            )
            plan = policy.propose(
                current.unsqueeze(0),
                occupied,
                incoming.unsqueeze(0),
                occupied,
            )
            scores[name].append(float(plan.replace == target_replace))
    return {name: sum(values) / len(values) for name, values in scores.items()}


def _adapt_online(
    policy: GatedResidualRegimeChangePolicy,
    *,
    seed: int,
    update_scale: int,
) -> dict[str, object]:
    if update_scale < 1:
        raise ValueError("online adaptation scale must be positive")
    optimizer = torch.optim.Adam(
        policy.trainable_parameters(), lr=DETECTOR_LEARNING_RATE
    )
    explorer = torch.Generator().manual_seed(seed + 97_001)
    occupied = torch.ones(1, ONLINE_RECORDS, dtype=torch.bool)
    history: list[dict[str, float | int | str | bool]] = []
    stage_plan = tuple(
        (family, length * update_scale)
        for family, length in ONLINE_STAGE_PLAN
    )
    cursor = 0
    for stage_index, (family, length) in enumerate(stage_plan):
        for step in range(length):
            current, incoming, target_replace = _partial_pair(
                seed=seed + 100_000 + cursor,
                family=family,
            )
            plan = policy.propose(
                current.unsqueeze(0),
                occupied,
                incoming.unsqueeze(0),
                occupied,
                explore=True,
                temperature=ONLINE_TEMPERATURE,
                generator=explorer,
            )
            utility = float(plan.replace == target_replace)
            policy.adaptation_step(
                current.unsqueeze(0),
                occupied,
                incoming.unsqueeze(0),
                occupied,
                plan,
                utility,
                optimizer=optimizer,
            )
            history.append(
                {
                    "stage": stage_index,
                    "family": family,
                    "step": step,
                    "replace": plan.replace,
                    "target_replace": target_replace,
                    "utility": utility,
                }
            )
            cursor += 1
    policy.eval()
    stage_accuracy = {
        str(stage): sum(item["utility"] for item in history if item["stage"] == stage)
        / sum(1 for item in history if item["stage"] == stage)
        for stage in range(len(stage_plan))
    }
    stream_utility = sum(item["utility"] for item in history) / len(history)
    return {
        "unique_scalar_utilities": len(history),
        "optimizer_updates": len(history),
        "replayed_examples": 0,
        "stage_accuracy": stage_accuracy,
        "stream_utility": stream_utility,
        "stage_plan": stage_plan,
        "history": history,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.pretrain_updates < 1 or args.online_updates_scale < 1:
        raise ValueError("online adaptation update settings must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    base_policy, pretraining = _train_detector(
        seed=args.seed,
        updates=args.pretrain_updates,
    )
    policy = GatedResidualRegimeChangePolicy(
        base_policy,
        override_margin=0.0,
    )
    before_scores = _evaluate_families(policy, seed=args.seed + 820_000)
    adaptation = _adapt_online(
        policy,
        seed=args.seed + 1_000,
        update_scale=args.online_updates_scale,
    )
    after_scores = _evaluate_families(policy, seed=args.seed + 821_000)
    exact_scores_after = _evaluate_detector(policy, seed=args.seed + 822_000)
    torch.manual_seed(args.seed + 923_000)
    fresh = OpaqueRegimeChangePolicy(
        value_width=EVENT_WIDTH,
        hidden=DETECTOR_HIDDEN,
        max_spectral_bins=8,
        learning_rate=DETECTOR_LEARNING_RATE,
    ).eval()
    fresh_scores = _evaluate_families(fresh, seed=args.seed + 821_000)
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "pretrained_stable_keep": before_scores["stable_keep"] >= 0.80,
        "pretrained_disjoint_replace": before_scores["disjoint_replace"] >= 0.80,
        "online_partial_replace": after_scores["partial_replace"] >= 0.80,
        "online_stable_retained": after_scores["stable_keep"] >= before_scores["stable_keep"] - 0.10,
        "online_disjoint_retained": after_scores["disjoint_replace"] >= before_scores["disjoint_replace"] - 0.10,
        "online_partial_beats_pretrained": after_scores["partial_replace"] >= before_scores["partial_replace"] + 0.15,
        "online_beats_fresh": sum(after_scores.values()) >= sum(fresh_scores.values()) + 0.40,
        "online_stream_utility": adaptation["stream_utility"] >= 0.65,
        "exact_stable_after": exact_scores_after["stable_keep"] >= 0.80,
        "exact_shift_after": exact_scores_after["shift_replace"] >= 0.80,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": adaptation["replayed_examples"] == 0,
    }
    report = {
        "schema": ONLINE_ADAPTATION_SCHEMA,
        "claim_boundary": (
            "A scalar-trained external regime trigger adapts online without replay "
            "to partially overlapping shifts while retaining stable and disjoint "
            "boundaries; not general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_regime_change_policy_v1",
            "residual_policy": "gated_residual_regime_change_policy_v1",
            "feature_contract": "opaque_spectral_cross_bank_structure_v1",
            "forbidden_features": "task_labels_regime_ids_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "pretraining": pretraining,
        "before_scores": before_scores,
        "online_adaptation": {
            key: value for key, value in adaptation.items() if key != "history"
        },
        "after_scores": after_scores,
        "exact_scores_after": exact_scores_after,
        "fresh_scores": fresh_scores,
        "gates": gates,
        "accounting": {
            "pretraining_scalar_updates": args.pretrain_updates,
            "online_scalar_updates": adaptation["optimizer_updates"],
            "total_optimizer_updates": args.pretrain_updates + adaptation["optimizer_updates"],
            "replayed_examples": 0,
            "controller_updates": 0,
            "unique_verifier_bits": args.pretrain_updates + adaptation["unique_scalar_utilities"],
            "unique_logical_lifetimes": args.pretrain_updates + adaptation["unique_scalar_utilities"],
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_online_partial_overlap"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrain-updates", type=int, default=1_000)
    parser.add_argument("--online-updates-scale", type=int, default=2)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
