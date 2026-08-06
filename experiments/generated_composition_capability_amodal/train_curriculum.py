"""Acquire a generated composition grammar one fresh program at a time.

The frozen parent is trained once.  The external composition stack is then
updated in phases: each phase exposes one previously untrained composition,
using fresh verifier outcomes only for that composition, and evaluates every
composition admitted so far.  This is the smallest useful test of continual
capability expansion without replaying old generated-composition examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _capability_accuracy,
    _new_capability,
    _stable_bits,
    _train_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from .train_pipeline import _new_stack


COMPOSITION_COUNT = 6
SPAN = 4
THRESHOLD = 0.75


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.phase_updates,
        args.batch_size,
        args.audit_count,
    ) < 1:
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    composition_ids = tuple(args.composition_ids)
    if not composition_ids:
        raise ValueError("at least one composition phase is required")
    if len(set(composition_ids)) != len(composition_ids):
        raise ValueError("composition phases must be unique")
    if any(
        composition_id < 0 or composition_id >= COMPOSITION_COUNT
        for composition_id in composition_ids
    ):
        raise ValueError("composition phase is out of range")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    _parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode="sampled",
    )
    parent.eval()
    parent_digest_before = _digest_core(parent, ())
    stack = _new_stack(
        args.seed + 1,
        program_count=args.program_count,
        stack="routed",
    )
    _unused_program, decoder = _new_capability(
        args.seed + 10_000 + args.program_count
    )

    phases: list[dict[str, object]] = []
    total_updates = 0
    for phase_index, composition_id in enumerate(composition_ids):
        history, progress = _train_capability(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            updates=args.phase_updates,
            batch_size=args.batch_size,
            seed=args.seed + 20_000 + phase_index * 1_000_003,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            generated_composition_ids=(composition_id,),
        )
        total_updates += args.phase_updates
        admitted = composition_ids[: phase_index + 1]
        retention = {
            str(admitted_id): _capability_accuracy(
                parent,
                stack,
                decoder,
                operation="generated_composition",
                span=SPAN,
                count=args.audit_count,
                seed=args.seed
                + 40_000
                + phase_index * 1_000_003
                + admitted_id * 10_007,
                generated_composition_ids=(admitted_id,),
            )
            for admitted_id in admitted
        }
        phases.append(
            {
                "phase": phase_index + 1,
                "new_composition_id": composition_id,
                "admitted_composition_ids": list(admitted),
                "new_behavior": retention[str(composition_id)],
                "retention": retention,
                "stable_bits_to_threshold": _stable_bits(
                    progress,
                    threshold=THRESHOLD,
                    bits_per_update=args.batch_size * SPAN,
                ),
                "history": history,
                "progress": progress,
                "replayed_generated_examples": 0,
            }
        )

    parent_digest_after = _digest_core(parent, ())
    final_retention = phases[-1]["retention"]
    assert isinstance(final_retention, dict)
    final_retention_values = [float(value) for value in final_retention.values()]
    phase_stable = all(
        phase["stable_bits_to_threshold"] is not None for phase in phases
    )
    report = {
        "schema": "neural-computer.generated-composition-curriculum-report.v1",
        "claim_boundary": (
            "A frozen controller with an external routed capability stack was "
            "tested on sequential admission of generated two-primitive "
            "compositions. Each phase trained only on fresh examples for its "
            "new composition and then measured retention of earlier ones. "
            "This is a bounded no-replay curriculum test, not yet general "
            "continual learning or unrestricted program induction."
        ),
        "seed": args.seed,
        "composition_ids": list(composition_ids),
        "program_count": args.program_count,
        "stack_configuration": stack.configuration(),
        "parent_updates": args.parent_updates,
        "phase_updates": args.phase_updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "parent_stable_bits_to_threshold": _stable_bits(
            parent_progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * 2,
        ),
        "phases": phases,
        "final_retention": final_retention,
        "parent_core_digest_before": parent_digest_before,
        "parent_core_digest_after": parent_digest_after,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "accounting": {
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + total_updates * args.batch_size * 2
            ),
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + total_updates * args.batch_size * (SPAN + 6)
            ),
            "optimizer_updates": args.parent_updates + total_updates,
            "replayed_examples": 0,
            "fresh_generated_examples": total_updates * args.batch_size,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_stable": _stable_bits(
                parent_progress,
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * 2,
            )
            is not None,
            "every_phase_stable": phase_stable,
            "all_final_compositions_mastered": bool(final_retention_values)
            and min(final_retention_values) >= THRESHOLD,
            "core_unchanged": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--phase-updates", type=int, default=64)
    parser.add_argument("--program-count", type=int, default=2)
    parser.add_argument(
        "--composition-ids",
        type=int,
        nargs="+",
        default=tuple(range(COMPOSITION_COUNT)),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "final_retention": report["final_retention"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
