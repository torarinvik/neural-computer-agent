"""Acquire generated compositions with an explicitly serial external stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

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
from neural_computer import (
    ExternalCapabilityComposition,
    ExternalCapabilityPipeline,
)


def _new_stack(
    seed: int,
    *,
    program_count: int,
    stack: str,
) -> ExternalCapabilityPipeline | ExternalCapabilityComposition:
    if program_count < 2:
        raise ValueError("compositional pipeline needs at least two programs")
    programs = tuple(
        _new_capability(seed + index + 1)[0]
        for index in range(program_count)
    )
    if stack == "serial":
        return ExternalCapabilityPipeline(programs)
    if stack == "routed":
        return ExternalCapabilityComposition(programs)
    raise ValueError("stack must be serial or routed")


def expand_routed_stack(
    stack: ExternalCapabilityComposition,
    *,
    seed: int,
) -> ExternalCapabilityComposition:
    """Add one external routed slot while preserving existing slot state."""

    if not isinstance(stack, ExternalCapabilityComposition):
        raise TypeError("only routed compositions can be expanded")
    old_count = len(stack.programs)
    expanded = _new_stack(seed, program_count=old_count + 1, stack="routed")
    if expanded.composition_steps != stack.composition_steps:
        raise ValueError("expanded stack changed composition step count")
    with torch.no_grad():
        for old_program, new_program in zip(
            stack.programs,
            expanded.programs[:old_count],
            strict=True,
        ):
            new_program.load_state_dict(old_program.state_dict(), strict=True)
        expanded.router[0].load_state_dict(stack.router[0].state_dict(), strict=True)
        old_router = stack.router[2]
        new_router = expanded.router[2]
        for step in range(stack.composition_steps):
            old_slice = slice(step * old_count, (step + 1) * old_count)
            new_slice = slice(
                step * (old_count + 1), step * (old_count + 1) + old_count
            )
            new_router.weight[new_slice].copy_(old_router.weight[old_slice])
            new_router.bias[new_slice].copy_(old_router.bias[old_slice])
            new_router.weight[step * (old_count + 1) + old_count].zero_()
            new_router.bias[step * (old_count + 1) + old_count].fill_(-8.0)
    expanded.eval()
    return expanded


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.updates,
        args.batch_size,
        args.audit_count,
    ) < 1:
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

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
    pipeline = _new_stack(
        args.seed + 1,
        program_count=args.program_count,
        stack=args.stack,
    )
    _unused_program, decoder = _new_capability(
        args.seed + 10_000 + args.program_count
    )
    history, progress = _train_capability(
        parent,
        pipeline,
        decoder,
        operation="generated_composition",
        span=4,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 20_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
    )
    behavior = _capability_accuracy(
        parent,
        pipeline,
        decoder,
        operation="generated_composition",
        span=4,
        count=args.audit_count,
        seed=args.seed + 30_000,
    )
    parent_digest_after = _digest_core(parent, ())
    parent_stable_bits = _stable_bits(
        parent_progress,
        threshold=0.75,
        bits_per_update=args.batch_size * 2,
    )
    stable_bits = _stable_bits(
        progress,
        threshold=0.75,
        bits_per_update=args.batch_size * 4,
    )
    report = {
        "schema": "neural-computer.generated-composition-pipeline-report.v1",
        "claim_boundary": (
            "A serial external pipeline of independently stateful programs "
            "was tested on sampled two-primitive compositions with a frozen "
            "controller. This is not yet a continual-learning or open-ended "
            "program-induction claim."
        ),
        "seed": args.seed,
        "program_count": args.program_count,
        "stack": args.stack,
        "pipeline_configuration": pipeline.configuration(),
        "parent_updates": args.parent_updates,
        "capability_updates": args.updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "parent_stable_bits_to_threshold": parent_stable_bits,
        "generated_stable_bits_to_threshold": stable_bits,
        "generated_behavior": behavior,
        "parent_core_digest_before": parent_digest_before,
        "parent_core_digest_after": parent_digest_after,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "history": history,
        "progress": progress,
        "accounting": {
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.updates * args.batch_size * 2
            ),
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.updates * args.batch_size * 6
            ),
            "optimizer_updates": args.parent_updates + args.updates,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_stable": parent_stable_bits is not None,
            "generated_capability_stable": stable_bits is not None,
            "generated_capability_mastered": behavior >= 0.75,
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
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--program-count", type=int, default=2)
    parser.add_argument("--stack", choices=("serial", "routed"), default="serial")
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
                "generated_behavior": report["generated_behavior"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
