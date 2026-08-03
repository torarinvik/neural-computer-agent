"""Task-agnostic replay stopping from old-behavior drift.

The policy compares an adapting controller with its frozen parent on stored
old experience.  It sees only controller logits (and the ordinary scalar
outcomes needed to roll the recurrent state forward), never correct-action
labels, task scores, span identities inside the model, or verifier retention.
Span arguments exist only to materialize this synthetic diagnostic stream;
in deployment the same calculation operates directly on replay-buffer rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .legacy_model import UnifiedCognitiveController
from .train_sequence_working_memory import (
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


def replay_stop_decision(mean_policy_kl: float, threshold: float) -> bool:
    """Return true when old-policy drift is small enough to stop replay."""
    if mean_policy_kl < 0 or threshold < 0:
        raise ValueError("KL and threshold must be non-negative")
    return mean_policy_kl <= threshold


def relative_replay_stop_decision(
        mean_policy_kl: float, baseline_policy_kl: float,
        minimum_relative_reduction: float) -> bool:
    """Stop after drift falls enough relative to this lineage's baseline."""
    if mean_policy_kl < 0 or baseline_policy_kl <= 0:
        raise ValueError("current KL must be non-negative and baseline positive")
    if not 0 <= minimum_relative_reduction < 1:
        raise ValueError("relative reduction must be in [0, 1)")
    return mean_policy_kl <= baseline_policy_kl * (
        1.0 - minimum_relative_reduction)


@torch.no_grad()
def measure_drift(
        parent: UnifiedCognitiveController,
        candidate: UnifiedCognitiveController, *, spans: tuple[int, ...],
        count: int, seed: int, distractors: int, operation: str,
        device: torch.device) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    for index, span in enumerate(spans):
        batch = generate_sequence_memory_batch(
            count, span=span, distractors=distractors,
            seed=seed + index * 20_011, operation=operation,
            heldout=False, device=device)
        parent_output = rollout_sequence_memory(
            parent, batch, sample_actions=False)
        child_output = rollout_sequence_memory(
            candidate, batch, sample_actions=False)
        parent_probability = parent_output["logits"].softmax(dim=-1)
        child_probability = child_output["logits"].softmax(dim=-1)
        policy_kl = (
            parent_probability * (
                parent_probability.clamp_min(1e-8).log()
                - child_probability.clamp_min(1e-8).log())
        ).sum(dim=-1)
        rows[str(span)] = {
            "policy_kl": float(policy_kl.mean()),
            "action_disagreement": float((
                parent_output["actions"]
                != child_output["actions"]).float().mean()),
            "logit_mse": float((
                parent_output["logits"]
                - child_output["logits"]).square().mean()),
        }
    return {
        "by_stream": rows,
        "mean_policy_kl": sum(
            row["policy_kl"] for row in rows.values()) / len(rows),
        "mean_action_disagreement": sum(
            row["action_disagreement"] for row in rows.values()) / len(rows),
    }


def _parse_spans(value: str) -> tuple[int, ...]:
    try:
        spans = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "spans must be comma-separated integers") from exc
    if not spans or any(span < 2 for span in spans):
        raise argparse.ArgumentTypeError("all spans must be at least two")
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--spans", type=_parse_spans, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=840_001)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--operation", default="mixed")
    parser.add_argument("--stop-threshold", type=float)
    parser.add_argument("--baseline-policy-kl", type=float)
    parser.add_argument("--minimum-relative-reduction", type=float, default=0.10)
    parser.add_argument(
        "--device", default=(
            "mps" if torch.backends.mps.is_available() else "cpu"),
        choices=("cpu", "mps"))
    args = parser.parse_args()
    if args.count < 1 or args.distractors < 0:
        raise ValueError("count must be positive and distractors non-negative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if (args.stop_threshold is None) == (args.baseline_policy_kl is None):
        raise ValueError(
            "provide exactly one of --stop-threshold or --baseline-policy-kl")
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    candidate = _load(args.candidate, device)
    drift = measure_drift(
        parent, candidate, spans=args.spans, count=args.count,
        seed=args.seed, distractors=args.distractors,
        operation=args.operation, device=device)
    if args.baseline_policy_kl is not None:
        stop = relative_replay_stop_decision(
            drift["mean_policy_kl"], args.baseline_policy_kl,
            args.minimum_relative_reduction)
        policy = "relative_reduction"
        relative_reduction = (
            1.0 - drift["mean_policy_kl"] / args.baseline_policy_kl)
    else:
        assert args.stop_threshold is not None
        stop = replay_stop_decision(
            drift["mean_policy_kl"], args.stop_threshold)
        policy = "absolute_threshold"
        relative_reduction = None
    report = {
        "schema": "sequence-replay-drift-policy-v1",
        "claim_boundary": (
            "The stop decision uses only parent/candidate policy drift on "
            "ordinary old experience. Correct actions and verifier retention "
            "scores are not inputs."),
        "parent": str(args.parent),
        "candidate": str(args.candidate),
        "configuration": {
            "spans": args.spans, "count": args.count, "seed": args.seed,
            "distractors": args.distractors, "operation": args.operation,
            "stop_threshold": args.stop_threshold,
            "baseline_policy_kl": args.baseline_policy_kl,
            "minimum_relative_reduction": args.minimum_relative_reduction,
        },
        "drift": drift,
        "policy": policy,
        "relative_reduction": relative_reduction,
        "decision": "stop" if stop else "continue",
        "stop_replay": stop,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "mean_policy_kl": drift["mean_policy_kl"],
        "threshold": args.stop_threshold,
        "baseline_policy_kl": args.baseline_policy_kl,
        "relative_reduction": relative_reduction,
        "decision": report["decision"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
