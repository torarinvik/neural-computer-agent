"""Retrospectively calibrate outcome-only replay stopping on a lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .legacy_model import UnifiedCognitiveController
from .outcome_replay_controller import OutcomeCalibratedReplayController
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


@torch.no_grad()
def _scalar_lifetime_outcomes(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, operation: str,
        device: torch.device) -> torch.Tensor:
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, heldout=True, device=device)
    result = rollout_sequence_memory(model, batch, sample_actions=False)
    # The controller receives only attempted-action feedback.  The policy
    # reduces each logical lifetime to its scalar success rate and never sees
    # the verifier's correct-action tensor.
    return result["rewards"].float().mean(dim=1).cpu()


def _parse_spans(value: str) -> tuple[int, ...]:
    spans = tuple(int(item) for item in value.split(",") if item)
    if not spans or any(span < 2 for span in spans):
        raise argparse.ArgumentTypeError("protected spans must be >= 2")
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--new-span", type=int, required=True)
    parser.add_argument("--protected-spans", type=_parse_spans, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=821_001)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--operation", default="mixed")
    parser.add_argument("--replay-lifetimes", type=int, default=0)
    parser.add_argument("--maximum-lifetimes", type=int, required=True)
    parser.add_argument("--retention-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-acquisition-gain", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-diagnostic-lifetimes", type=int, default=64)
    parser.add_argument(
        "--device", default=(
            "mps" if torch.backends.mps.is_available() else "cpu"),
        choices=("cpu", "mps"))
    args = parser.parse_args()
    if args.count < 1 or args.distractors < 0 or args.new_span < 2:
        raise ValueError("count and span must be positive")
    if args.replay_lifetimes < 0:
        raise ValueError("replay lifetimes must be non-negative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    child = _load(args.child, device)
    controller = OutcomeCalibratedReplayController(
        maximum_lifetimes=args.maximum_lifetimes,
        retention_tolerance=args.retention_tolerance,
        minimum_acquisition_gain=args.minimum_acquisition_gain,
        confidence=args.confidence,
        minimum_diagnostic_lifetimes=args.minimum_diagnostic_lifetimes)
    controller.consume_replay(args.replay_lifetimes)

    streams: dict[str, dict[str, Any]] = {}

    def observe(name: str, span: int, seed: int, protected: bool) -> None:
        parent_outcomes = _scalar_lifetime_outcomes(
            parent, count=args.count, span=span, distractors=args.distractors,
            seed=seed, operation=args.operation, device=device)
        child_outcomes = _scalar_lifetime_outcomes(
            child, count=args.count, span=span, distractors=args.distractors,
            seed=seed, operation=args.operation, device=device)
        if protected:
            controller.observe_protected(
                name, parent_outcomes.tolist(), child_outcomes.tolist())
        else:
            controller.observe_acquisition(
                parent_outcomes.tolist(), child_outcomes.tolist())
        streams[name] = {
            "span": span,
            "count": args.count,
            "seed": seed,
            "parent_scalar_outcome_mean": float(parent_outcomes.mean()),
            "child_scalar_outcome_mean": float(child_outcomes.mean()),
            "paired_delta_mean": float((child_outcomes - parent_outcomes).mean()),
        }

    observe("acquisition", args.new_span, args.seed + args.new_span * 10_003,
            False)
    for index, span in enumerate(args.protected_spans):
        observe(
            f"protected-{index}", span,
            args.seed + 100_000 + index * 20_011 + span, True)

    report = controller.report()
    report.update({
        "parent": str(args.parent),
        "child": str(args.child),
        "streams": streams,
        "configuration": {
            **report["configuration"],
            "count_per_stream": args.count,
            "new_span": args.new_span,
            "protected_spans": args.protected_spans,
            "distractors": args.distractors,
            "operation": args.operation,
            "replay_lifetimes": args.replay_lifetimes,
        },
        "accounting": {
            "unique_logical_lifetimes": report["budget"]["diagnostic_lifetimes"],
            "unique_verifier_bits": sum(
                value["count"] * value["span"]
                for value in streams.values()),
            "optimizer_updates": 0,
            "replayed_examples": args.replay_lifetimes,
            "diagnostic_lifetimes_charged_to_budget": (
                report["budget"]["diagnostic_lifetimes"]),
        },
    })
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
