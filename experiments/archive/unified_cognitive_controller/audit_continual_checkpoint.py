"""Verifier-side horse race for continual-learning checkpoints.

Every candidate is evaluated on identical lifetime-disjoint episodes.  The
controller sees only its ordinary rendered stream and scalar outcomes; task
labels and retention measurements remain private to this audit.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .continual_objective import score_continual_acquisition
from .legacy_model import UnifiedCognitiveController
from .train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
    rollout_sequence_memory,
)


def _load(path: Path, device: torch.device) -> UnifiedCognitiveController:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _zero_last_slot(model: UnifiedCognitiveController) -> None:
    """Causally remove the newest stored skill, preserving older slots."""
    if not len(model.skill_adapters):
        raise ValueError("candidate contains no skill slot")
    index = len(model.skill_adapters) - 1
    collections = (
        model.skill_adapters,
        model.skill_adapter_gates,
        model.skill_adapter_gate_refiners,
        model.skill_adapter_gate_extensions,
        model.skill_adapter_read_projections,
        model.skill_adapter_critics,
        model.skill_adapter_critic_scales,
    )
    with torch.no_grad():
        for collection in collections:
            if index >= len(collection):
                continue
            item = collection[index]
            if isinstance(item, torch.nn.Parameter):
                item.zero_()
            else:
                for parameter in item.parameters():
                    parameter.zero_()


def _audit(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, operation: str,
        device: torch.device) -> dict[str, Any]:
    return evaluate_sequence_memory(
        model, count=count, span=span, distractors=distractors, seed=seed,
        operation=operation, device=device)


def _parse_spans(value: str) -> tuple[int, ...]:
    try:
        spans = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "old spans must be comma-separated integers") from exc
    if not spans or any(span < 2 for span in spans):
        raise argparse.ArgumentTypeError("old spans must all be at least two")
    return spans


def _chance_control_passed(accuracy: float, tolerance: float) -> bool:
    """Require a missing-evidence control to remain near binary chance."""
    if not 0.0 <= tolerance <= 0.5:
        raise ValueError("chance-control tolerance must be in [0, 0.5]")
    return 0.5 - tolerance <= accuracy <= 0.5 + tolerance


@torch.no_grad()
def _paired_gain_interval(
        parent: UnifiedCognitiveController,
        child: UnifiedCognitiveController,
        zeroed: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, operation: str, device: torch.device,
        bootstrap_samples: int, confidence: float,
        ) -> dict[str, Any]:
    """Bootstrap paired per-lifetime gains on identical normal episodes."""
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, heldout=True, device=device)

    def lifetime_accuracy(model: UnifiedCognitiveController) -> torch.Tensor:
        result = rollout_sequence_memory(model, batch, sample_actions=False)
        return result["rewards"].float().mean(dim=1).cpu()

    parent_accuracy = lifetime_accuracy(parent)
    child_accuracy = lifetime_accuracy(child)
    zeroed_accuracy = lifetime_accuracy(zeroed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 707_071)

    def interval(delta: torch.Tensor) -> dict[str, float]:
        # Sampling whole lifetimes, rather than individual query positions,
        # preserves within-episode dependence in the uncertainty estimate.
        indices = torch.randint(
            count, (bootstrap_samples, count), generator=generator)
        means = delta[indices].mean(dim=1)
        alpha = (1.0 - confidence) / 2.0
        return {
            "mean": float(delta.mean()),
            "lower": float(torch.quantile(means, alpha)),
            "upper": float(torch.quantile(means, 1.0 - alpha)),
            "positive_lifetime_fraction": float((delta > 0).float().mean()),
        }

    return {
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "sampling_unit": "logical_lifetime",
        "child_minus_parent": interval(child_accuracy - parent_accuracy),
        "child_minus_zero_newest_slot": interval(
            child_accuracy - zeroed_accuracy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--new-span", type=int, required=True)
    parser.add_argument("--old-spans", type=_parse_spans, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=810_001)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--operation", default="mixed")
    parser.add_argument("--replay-lifetimes", type=int, required=True)
    parser.add_argument("--reference-replay-lifetimes", type=int, required=True)
    parser.add_argument("--retention-tolerance", type=float, default=0.02)
    parser.add_argument("--new-weight", type=float, default=1.0)
    parser.add_argument("--forgetting-weight", type=float, default=2.0)
    parser.add_argument("--replay-weight", type=float, default=0.25)
    parser.add_argument("--minimum-new-gain", type=float, default=0.0)
    parser.add_argument(
        "--chance-control-tolerance", type=float, default=0.05,
        help=(
            "required half-width around binary chance for blank-sequence "
            "and full-memory-reset controls"))
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--device", default=(
            "mps" if torch.backends.mps.is_available() else "cpu"),
        choices=("cpu", "mps"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and even")
    if args.new_span < 2 or args.distractors < 0:
        raise ValueError("new span must be >=2 and distractors non-negative")
    if not 0.0 <= args.chance_control_tolerance <= 0.5:
        raise ValueError("chance-control tolerance must be in [0, 0.5]")
    if args.bootstrap_samples < 100 or not 0.5 < args.confidence < 1.0:
        raise ValueError("bootstrap samples >=100 and confidence in (0.5,1) required")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)

    parent = _load(args.parent, device)
    child = _load(args.child, device)
    zeroed = _load(args.child, device)
    _zero_last_slot(zeroed)

    new_seed = args.seed + args.new_span * 10_003
    new_parent = _audit(
        parent, count=args.count, span=args.new_span,
        distractors=args.distractors, seed=new_seed,
        operation=args.operation, device=device)
    new_child = _audit(
        child, count=args.count, span=args.new_span,
        distractors=args.distractors, seed=new_seed,
        operation=args.operation, device=device)
    new_zeroed = _audit(
        zeroed, count=args.count, span=args.new_span,
        distractors=args.distractors, seed=new_seed,
        operation=args.operation, device=device)
    paired_interval = _paired_gain_interval(
        parent, child, zeroed, count=args.count, span=args.new_span,
        distractors=args.distractors, seed=new_seed,
        operation=args.operation, device=device,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence)

    old: dict[str, dict[str, Any]] = {}
    for index, span in enumerate(args.old_spans):
        seed = args.seed + 100_000 + index * 20_011 + span
        before = _audit(
            parent, count=args.count, span=span,
            distractors=args.distractors, seed=seed,
            operation=args.operation, device=device)
        after = _audit(
            child, count=args.count, span=span,
            distractors=args.distractors, seed=seed,
            operation=args.operation, device=device)
        old[str(span)] = {
            "parent": before,
            "child": after,
            "accuracy_delta": float(after["accuracy"] - before["accuracy"]),
        }

    objective = score_continual_acquisition(
        new_parent=float(new_parent["accuracy"]),
        new_child=float(new_child["accuracy"]),
        new_causal_baseline=float(new_zeroed["accuracy"]),
        old_parent=[float(old[str(span)]["parent"]["accuracy"])
                    for span in args.old_spans],
        old_child=[float(old[str(span)]["child"]["accuracy"])
                   for span in args.old_spans],
        replay_lifetimes=args.replay_lifetimes,
        reference_replay_lifetimes=args.reference_replay_lifetimes,
        retention_tolerance=args.retention_tolerance,
        new_weight=args.new_weight,
        forgetting_weight=args.forgetting_weight,
        replay_weight=args.replay_weight,
        minimum_new_gain=args.minimum_new_gain)
    causal_passed = (
        objective.causal_gain is not None
        and paired_interval["child_minus_zero_newest_slot"]["lower"] > 0)
    acquisition_passed = (
        objective.acquisition_gate_passed
        and paired_interval["child_minus_parent"]["lower"] > 0)
    blank_control_passed = _chance_control_passed(
        float(new_child["blank_sequence_accuracy"]),
        args.chance_control_tolerance)
    reset_control_passed = _chance_control_passed(
        float(new_child["all_memory_reset_accuracy"]),
        args.chance_control_tolerance)
    report = {
        "schema": "continual-checkpoint-audit-v1",
        "claim_boundary": (
            "All labels, span identities, retention scores, causal ablations, "
            "and the selection score exist only in this verifier-side audit."),
        "parent": str(args.parent),
        "child": str(args.child),
        "configuration": {
            "count": args.count, "seed": args.seed,
            "new_span": args.new_span, "old_spans": args.old_spans,
            "distractors": args.distractors, "operation": args.operation,
            "replay_lifetimes": args.replay_lifetimes,
            "reference_replay_lifetimes": args.reference_replay_lifetimes,
            "retention_tolerance": args.retention_tolerance,
            "minimum_new_gain": args.minimum_new_gain,
            "chance_control_tolerance": args.chance_control_tolerance,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": args.confidence,
        },
        "new_skill": {
            "parent": new_parent,
            "child": new_child,
            "zero_newest_slot": new_zeroed,
            "paired_gain_interval": paired_interval,
        },
        "old_skills": old,
        "objective": asdict(objective),
        "gates": {
            "new_acquisition_passed": acquisition_passed,
            "causal_slot_contribution_passed": causal_passed,
            "retention_passed": objective.retention_gate_passed,
            "blank_control_passed": blank_control_passed,
            "reset_control_passed": reset_control_passed,
            "accepted": (
                acquisition_passed and causal_passed
                and objective.retention_gate_passed
                and blank_control_passed and reset_control_passed),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "accepted": report["gates"]["accepted"],
        "score": objective.score,
        "rewarded_new_gain": objective.new_gain,
        "parent_gain": objective.parent_gain,
        "causal_gain": objective.causal_gain,
        "old_deltas": objective.old_retention_deltas,
        "replay_savings": objective.replay_savings,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
