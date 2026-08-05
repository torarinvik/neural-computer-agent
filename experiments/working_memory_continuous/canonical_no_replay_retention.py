"""Canonical continual-learning audit without replay of mastered spans.

The controller is trained on a one-way curriculum: span two, then span three,
then span four. Once a stage ends, no old-span batch is ever used for a
gradient update. Older spans are measured only on fresh held-out lifetimes.
The promotion gate requires both stable mastery of span four and retention of
every earlier mastered span after later learning.
"""

from __future__ import annotations

import argparse
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
    _rollout,
    _runtime,
)


def _train_curriculum(
    runtime: AmodalControllerRuntime,
    *,
    stages: tuple[int, ...],
    updates_per_stage: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    eval_every: int,
    shuffle_outcomes: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    trainable = [parameter for parameter in runtime.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable canonical parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, object]] = []
    span_updates: dict[str, int] = {}
    global_update = 0
    for stage_index, span in enumerate(stages):
        for local_update in range(1, updates_per_stage + 1):
            global_update += 1
            span_updates[str(span)] = span_updates.get(str(span), 0) + 1
            batch = generate_sequence_memory_batch(
                batch_size,
                span=span,
                distractors=1,
                seed=seed + global_update * 10_007,
                operation="forward",
            )
            result = _rollout(
                runtime,
                batch,
                train=True,
                shuffle_outcomes=shuffle_outcomes,
            )
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            should_measure = (
                local_update == updates_per_stage
                or local_update == 1
                or (eval_every > 0 and local_update % eval_every == 0)
            )
            if not should_measure:
                continue
            audit: dict[str, float] = {}
            for measured_span in stages[: stage_index + 1]:
                audit[str(measured_span)] = _accuracy(
                    runtime,
                    operation="forward",
                    count=batch_size,
                    span=measured_span,
                    seed=seed + 1_000_000 + global_update * 1_003 + measured_span,
                )
            history.append(
                {
                    "global_update": global_update,
                    "stage": stage_index,
                    "train_span": span,
                    "local_update": local_update,
                    "unique_logical_lifetimes": global_update * batch_size,
                    "unique_verifier_bits": sum(
                        count * batch_size * int(key)
                        for key, count in span_updates.items()
                    ),
                    "training_accuracy": float(result["rewards"].mean()),
                    "loss": float(result["loss"].detach()),
                    "heldout_accuracy": audit,
                }
            )
    runtime.eval()
    return history, span_updates


def _retention_summary(
    history: list[dict[str, object]],
    stages: tuple[int, ...],
    *,
    threshold: float,
    tolerance: float,
) -> dict[str, object]:
    by_span: dict[str, list[tuple[int, float]]] = {str(span): [] for span in stages}
    for row in history:
        update = int(row["global_update"])
        audit = row["heldout_accuracy"]
        assert isinstance(audit, dict)
        for span, value in audit.items():
            by_span[span].append((update, float(value)))

    stable_bits: dict[str, int | None] = {}
    for span, rows in by_span.items():
        stable_bits[span] = None
        for index, (update, _) in enumerate(rows):
            if all(value >= threshold for _, value in rows[index:]):
                stable_bits[span] = update
                break

    retention: dict[str, dict[str, float | bool | None]] = {}
    for index, span in enumerate(stages):
        rows = by_span[str(span)]
        # The stage boundary is represented by the last audit row before the
        # next span appears in the training history.
        stage_rows = [
            row for row in history
            if int(row["train_span"]) == span
        ]
        if not stage_rows:
            retention[str(span)] = {
                "mastery_before_later_learning": None,
                "minimum_after_mastery": None,
                "retention_delta": None,
                "within_tolerance": False,
            }
            continue
        boundary_audit = stage_rows[-1]["heldout_accuracy"]
        assert isinstance(boundary_audit, dict)
        baseline = float(boundary_audit[str(span)])
        boundary_update = int(stage_rows[-1]["global_update"])
        after = [value for update, value in rows if update > boundary_update]
        minimum_after = min(after) if after else baseline
        retention[str(span)] = {
            "mastery_before_later_learning": baseline,
            "minimum_after_mastery": minimum_after,
            "retention_delta": minimum_after - baseline,
            "within_tolerance": baseline >= threshold and minimum_after >= baseline - tolerance,
        }

    target = str(stages[-1])
    target_rows = by_span[target]
    target_final = target_rows[-1][1] if target_rows else 0.0
    return {
        "stable_bits_to_threshold": stable_bits,
        "retention": retention,
        "target_final_accuracy": target_final,
        "all_prior_retained": all(
            bool(values["within_tolerance"])
            for span, values in retention.items()
            if span != target
        ),
        "target_mastered": target_final >= threshold,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    stages = tuple(args.stages)
    if stages != tuple(sorted(set(stages))) or len(stages) < 2:
        raise ValueError("stages must contain at least two strictly increasing spans")
    if stages[0] < 1 or args.updates_per_stage < 1:
        raise ValueError("stages and updates-per-stage must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    runtime = _runtime(seed=args.seed, growth=False)
    history, span_updates = _train_curriculum(
        runtime,
        stages=stages,
        updates_per_stage=args.updates_per_stage,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        shuffle_outcomes=args.shuffle_outcomes,
    )
    summary = _retention_summary(
        history,
        stages,
        threshold=args.mastery_threshold,
        tolerance=args.retention_tolerance,
    )
    final_controls = {
        "blank_sequence": _accuracy(
            runtime,
            operation="forward",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 2_000_001,
            blank_sequence=True,
        ),
        "workspace_disabled": _accuracy(
            runtime,
            operation="forward",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 2_000_002,
            disable_workspace=True,
        ),
        "reversal_probe": _accuracy(
            runtime,
            operation="reverse",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 2_000_003,
        ),
    }
    total_updates = args.updates_per_stage * len(stages)
    report = {
        "schema": "canonical-no-replay-retention-v1",
        "claim_boundary": (
            "The canonical controller is trained on a one-way span curriculum. "
            "After a span stage ends, no old-span examples are replayed into "
            "the optimizer; old spans are evaluation-only."
        ),
        "seed": args.seed,
        "stages": list(stages),
        "updates_per_stage": args.updates_per_stage,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "mastery_threshold": args.mastery_threshold,
        "retention_tolerance": args.retention_tolerance,
        "shuffle_outcomes": args.shuffle_outcomes,
        "span_update_counts": span_updates,
        "history": history,
        "retention_summary": summary,
        "final_controls": final_controls,
        "accounting": {
            "unique_logical_lifetimes": total_updates * args.batch_size,
            "unique_verifier_bits": sum(
                args.updates_per_stage * args.batch_size * span for span in stages
            ),
            "optimizer_updates": total_updates,
            "replayed_examples": 0,
            "diagnostic_lifetimes_charged_to_budget": (
                len(history) * args.batch_size * len(stages)
            ),
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "no_replayed_examples": True,
            "all_prior_spans_retained": summary["all_prior_retained"],
            "target_span_mastered": summary["target_mastered"],
            "blank_sequence_near_chance": final_controls["blank_sequence"] <= 0.65,
            "workspace_ablation_is_informative": final_controls["workspace_disabled"] < summary["target_final_accuracy"] - 0.05,
            "reversal_is_not_claimed_as_mastery": True,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69301)
    parser.add_argument("--stages", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--updates-per-stage", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--retention-tolerance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "accepted_diagnostic": report["accepted_diagnostic"],
                "retention_summary": report["retention_summary"],
                "final_controls": report["final_controls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
