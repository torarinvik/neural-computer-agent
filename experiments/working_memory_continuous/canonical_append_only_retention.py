"""Append-only continual-learning audit without replay.

This is the first candidate retention mechanism after the shared-parameter
baseline: acquire span two in the parent controller, then acquire each harder
span in a fresh generic growth slot while the parent and previously acquired
slots remain frozen.  The later stages receive only new current-span
experiences.  Older spans are evaluation-only and never enter a gradient.

The mechanism is deliberately narrow.  It tests whether isolated, append-only
controller capacity can retain a prefix of working-memory skills without
replaying old examples; it does not claim arbitrary lifelong learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from neural_computer import AmodalControllerRuntime

from .canonical_growth_pressure_test import (
    _accuracy,
    _copy_parent_weights,
    _digest_core,
    _freeze_except,
    _rollout,
    _runtime,
)


def _digest_prefix(runtime: AmodalControllerRuntime, prefix: str) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        if name.startswith(prefix):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _train_stage(
    runtime: AmodalControllerRuntime,
    *,
    span: int,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    trainable_prefix: str | None,
) -> list[dict[str, float | int]]:
    if trainable_prefix is None:
        trainable = [
            parameter for parameter in runtime.parameters() if parameter.requires_grad
        ]
    else:
        trainable = [
            parameter
            for name, parameter in runtime.named_parameters()
            if name.startswith(trainable_prefix) and parameter.requires_grad
        ]
    if not trainable:
        raise RuntimeError(f"no trainable parameters for {trainable_prefix!r}")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    runtime.train()
    for update in range(1, updates + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=1,
            seed=seed + update * 10_007,
            operation="forward",
        )
        result = _rollout(runtime, batch, train=True)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
    runtime.eval()
    return history


def _audit(
    runtime: AmodalControllerRuntime,
    *,
    spans: tuple[int, ...],
    count: int,
    seed: int,
) -> dict[str, float]:
    return {
        str(span): _accuracy(
            runtime,
            operation="forward",
            count=count,
            span=span,
            seed=seed + span,
        )
        for span in spans
    }


def _retention(
    stage_audits: list[dict[str, object]],
    stages: tuple[int, ...],
    *,
    threshold: float,
    tolerance: float,
) -> dict[str, object]:
    retention: dict[str, dict[str, float | bool]] = {}
    for index, span in enumerate(stages[:-1]):
        baseline = float(stage_audits[index]["heldout_accuracy"][str(span)])
        later_values = [
            float(audit["heldout_accuracy"][str(span)])
            for audit in stage_audits[index + 1 :]
        ]
        minimum = min(later_values) if later_values else baseline
        retention[str(span)] = {
            "mastery_before_later_learning": baseline,
            "minimum_after_mastery": minimum,
            "retention_delta": minimum - baseline,
            "within_tolerance": (
                baseline >= threshold and minimum >= baseline - tolerance
            ),
        }
    target = str(stages[-1])
    target_final = float(stage_audits[-1]["heldout_accuracy"][target])
    return {
        "retention": retention,
        "all_prior_retained": all(
            values["within_tolerance"] for values in retention.values()
        ),
        "target_final_accuracy": target_final,
        "target_mastered": target_final >= threshold,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    stages = tuple(args.stages)
    if stages != (2, 3, 4):
        raise ValueError("the append-only audit currently requires stages 2 3 4")
    if args.updates_per_stage < 1 or args.batch_size < 1:
        raise ValueError("updates-per-stage and batch-size must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")

    parent = _runtime(seed=args.seed, growth=False)
    parent_history = _train_stage(
        parent,
        span=stages[0],
        updates=args.updates_per_stage,
        batch_size=args.batch_size,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        trainable_prefix=None,
    )

    expanded = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, expanded)
    expanded.eval()
    stage_audits: list[dict[str, object]] = [
        {
            "stage": 0,
            "train_span": stages[0],
            "heldout_accuracy": _audit(
                expanded,
                spans=stages[:1],
                count=args.audit_count,
                seed=args.seed + 10_000,
            ),
            "history": parent_history,
        }
    ]

    # The parent and each earlier growth slot are frozen permanently.  Each
    # new stage gets only its own generic slot, with no old-span batch or
    # replay buffer involved in the optimizer.
    for stage_index, span in enumerate(stages[1:], start=1):
        prefix = f"controller.growth_slots.{stage_index - 1}."
        _freeze_except(expanded, (f"growth_slots.{stage_index - 1}.",))
        stage_history = _train_stage(
            expanded,
            span=span,
            updates=args.updates_per_stage,
            batch_size=args.batch_size,
            seed=args.seed + 100 * (stage_index + 1),
            learning_rate=args.learning_rate,
            trainable_prefix=prefix,
        )
        stage_audits.append(
            {
                "stage": stage_index,
                "train_span": span,
                "heldout_accuracy": _audit(
                    expanded,
                    spans=stages[: stage_index + 1],
                    count=args.audit_count,
                    seed=args.seed + 20_000 + stage_index * 1_000,
                ),
                "history": stage_history,
            }
        )

    expanded.eval()
    retention = _retention(
        stage_audits,
        stages,
        threshold=args.mastery_threshold,
        tolerance=args.retention_tolerance,
    )
    controls = {
        "blank_sequence": _accuracy(
            expanded,
            operation="forward",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 40_001,
            blank_sequence=True,
        ),
        "workspace_disabled": _accuracy(
            expanded,
            operation="forward",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 40_002,
            disable_workspace=True,
        ),
    }
    core_unchanged = _digest_core(parent, ("growth_slots.0.", "growth_slots.1.")) == _digest_core(
        expanded, ("growth_slots.0.", "growth_slots.1.")
    )
    growth_prefixes = ("controller.growth_slots.0.", "controller.growth_slots.1.")
    growth_stage_digests = {
        prefix: _digest_prefix(expanded, prefix.removeprefix("controller."))
        for prefix in growth_prefixes
    }
    total_updates = args.updates_per_stage * len(stages)
    report = {
        "schema": "canonical-append-only-retention-v1",
        "claim_boundary": (
            "A frozen parent and append-only generic growth slots acquire a "
            "span-2 -> span-3 -> span-4 curriculum without replaying old "
            "examples. This is a working-memory retention result, not a claim "
            "of arbitrary lifelong learning."
        ),
        "seed": args.seed,
        "stages": list(stages),
        "updates_per_stage": args.updates_per_stage,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "mastery_threshold": args.mastery_threshold,
        "retention_tolerance": args.retention_tolerance,
        "stage_audits": stage_audits,
        "retention_summary": retention,
        "controls": controls,
        "growth_stage_digests": growth_stage_digests,
        "accounting": {
            "unique_logical_lifetimes": total_updates * args.batch_size,
            "unique_verifier_bits": sum(
                args.updates_per_stage * args.batch_size * span for span in stages
            ),
            "optimizer_updates": total_updates,
            "replayed_examples": 0,
            "diagnostic_lifetimes_charged_to_budget": (
                args.audit_count * sum(range(1, len(stages) + 1))
            ),
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "no_replayed_examples": True,
            "all_prior_spans_retained": retention["all_prior_retained"],
            "target_span_mastered": retention["target_mastered"],
            "blank_sequence_near_chance": controls["blank_sequence"] <= 0.65,
            "workspace_ablation_is_informative": controls["workspace_disabled"] < retention["target_final_accuracy"] - 0.05,
            "frozen_parent_core": core_unchanged,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69303)
    parser.add_argument("--stages", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--updates-per-stage", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--retention-tolerance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "accepted_diagnostic": report["accepted_diagnostic"],
                "retention_summary": report["retention_summary"],
                "controls": report["controls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
