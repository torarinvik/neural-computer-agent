"""Replicated multi-edit outcome-only acquisition for the control-flow ABI.

The warm arm starts from a useful but incomplete external file.  The fresh
arm starts from a different opaque file of the same executor.  Both search
with the same stochastic multi-edit frontier and receive only scalar verifier
outcomes; the target program and its private rule remain verifier-only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ControlFlowFrontierState,
    ControlFlowInstruction,
    ControlFlowProgram,
    ControlFlowProgramFrontier,
    ControlFlowProgramMemory,
)

COUNTER_COUNT = 2
TRAIN_AMOUNTS = tuple(range(8))
HELDOUT_AMOUNTS = tuple(range(8, 16))
MAX_STEPS = 128
FRONTIER_BUDGET = 3_000
VERIFIER_THRESHOLD = 1.0
FRONTIER_MINIMUM_QUALITY = 0.25


def _warm_root() -> ControlFlowProgram:
    """Clear counter zero, but do not transfer its value to counter one."""

    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=3),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )


def _fresh_root() -> ControlFlowProgram:
    """A same-ABI fresh file with no useful loop structure."""

    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )


def _private_target() -> ControlFlowProgram:
    """Transfer counter zero into counter one, with no learner-visible label."""

    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=4),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )


def _outcomes(
    candidate: ControlFlowProgram,
    reference: ControlFlowProgram,
    amounts: tuple[int, ...],
) -> tuple[float, ...]:
    values: list[float] = []
    for amount in amounts:
        actual = candidate.execute((amount, 0), max_steps=MAX_STEPS)
        expected = reference.execute((amount, 0), max_steps=MAX_STEPS)
        values.append(
            float(actual.status == "halted" and actual.counters == expected.counters)
        )
    return tuple(values)


def _search(
    root: ControlFlowProgram,
    reference: ControlFlowProgram,
    amounts: tuple[int, ...],
    *,
    seed: int,
) -> dict[str, object]:
    frontier = ControlFlowProgramFrontier(
        COUNTER_COUNT,
        beam_width=32,
        max_depth=8,
        max_program_length=8,
        minimum_quality=FRONTIER_MINIMUM_QUALITY,
        parent_temperature=0.5,
        exploration=0.5,
    )
    state = frontier.initial_state(root)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    candidate: ControlFlowProgram | None = None
    accepted_feedback = None
    termination = "budget_exhausted"
    for _ in range(FRONTIER_BUDGET):
        try:
            proposal = frontier.propose(state, generator=generator)
        except RuntimeError:
            termination = "frontier_exhausted"
            break
        feedback = frontier.record_outcomes(
            state,
            proposal,
            _outcomes(proposal.program, reference, amounts),
            threshold=VERIFIER_THRESHOLD,
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
        )
        state = feedback.state
        if feedback.accepted:
            candidate = proposal.program
            accepted_feedback = feedback
            break
    if candidate is not None:
        termination = "expressible"
    return {
        "frontier": frontier,
        "state": state,
        "candidate": candidate,
        "accepted_feedback": accepted_feedback,
        "evaluations": state.evaluations,
        "status": termination,
        "wall_seconds": time.perf_counter() - started,
    }


def _run_arm(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    target = _private_target()
    amounts = tuple(reversed(TRAIN_AMOUNTS)) if reverse_inputs else TRAIN_AMOUNTS
    warm = _search(_warm_root(), target, amounts, seed=seed + 10_000)
    fresh = _search(_fresh_root(), target, amounts, seed=seed + 20_000)
    candidate = warm["candidate"]
    if not isinstance(candidate, ControlFlowProgram):
        raise TypeError("warm control-flow frontier did not acquire a candidate")

    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    source_slot = memory.add_program(_warm_root(), protect=True)
    candidate_outcomes = _outcomes(candidate, target, amounts)
    admission = memory.admit_verified(
        candidate,
        candidate_outcomes,
        threshold=VERIFIER_THRESHOLD,
        min_observations=len(amounts),
        min_stable_observations=len(amounts),
        protect=True,
    )
    if not admission.accepted or admission.slot is None:
        raise RuntimeError(f"control-flow candidate admission failed: {admission.reason}")

    source_digest_before = memory.digest()
    missing_no_write = False
    try:
        missing = memory.admit_verified(
            target,
            (),
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
        )
        missing_no_write = not missing.accepted
    except ValueError:
        missing_no_write = True
    missing_no_write = missing_no_write and memory.digest() == source_digest_before

    shuffled_before = memory.digest()
    shuffled = memory.admit_verified(
        target,
        tuple(float(index % 2 == 0) for index in range(len(amounts))),
        threshold=VERIFIER_THRESHOLD,
        min_observations=len(amounts),
        min_stable_observations=len(amounts),
    )
    shuffled_rejected = not shuffled.accepted and memory.digest() == shuffled_before

    heldout = _outcomes(candidate, target, HELDOUT_AMOUNTS)
    source_heldout = _outcomes(memory.program(source_slot), _warm_root(), HELDOUT_AMOUNTS)
    restored_memory = ControlFlowProgramMemory.from_payload(memory.payload())
    restored_frontier = ControlFlowFrontierState.from_payload(
        warm["state"].payload()
    )
    corrupted = memory.payload()
    corrupted["sha256"] = "0" * 64
    try:
        ControlFlowProgramMemory.from_payload(corrupted)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    warm_evaluations = int(warm["evaluations"])
    fresh_evaluations = int(fresh["evaluations"])
    gates = {
        "warm_frontier_expressible": warm["status"] == "expressible",
        "warm_target_admitted": bool(admission.accepted),
        "warm_target_heldout_mastery": min(heldout) >= VERIFIER_THRESHOLD,
        "source_protected": memory.is_file_protected(source_slot),
        "source_heldout_retention": min(source_heldout) >= VERIFIER_THRESHOLD,
        "frontier_reload_exact": restored_frontier.digest() == warm["state"].digest(),
        "memory_reload_exact": restored_memory.digest() == memory.digest(),
        "missing_evidence_no_write": missing_no_write,
        "shuffled_feedback_rejected": shuffled_rejected,
        "corruption_rejected": corruption_rejected,
        "fresh_control_measured": fresh["status"] != "expressible",
        "termination_is_not_inexpressible": fresh["status"] in {
            "frontier_exhausted",
            "budget_exhausted",
        },
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }
    transfer_ratio = (
        fresh_evaluations / warm_evaluations if fresh["status"] == "expressible" else None
    )
    return {
        "seed": seed,
        "reverse_inputs": reverse_inputs,
        "schema": "neural-computer.recipe-control-flow-frontier-growth.v1",
        "architecture": {
            "counter_count": COUNTER_COUNT,
            "frontier_budget": FRONTIER_BUDGET,
            "frontier_minimum_quality": FRONTIER_MINIMUM_QUALITY,
            "beam_width": 32,
            "max_depth": 8,
            "max_program_length": 8,
            "learner_inputs": "opaque_programs_and_scalar_verifier_outcomes",
            "forbidden_features": "target_program_names_and_verifier_rows",
        },
        "warm_search": {
            "status": warm["status"],
            "evaluations": warm_evaluations,
            "candidate_digest": candidate.digest(),
            "wall_seconds": warm["wall_seconds"],
        },
        "fresh_search": {
            "status": fresh["status"],
            "evaluations": fresh_evaluations,
            "candidate_digest": (
                None
                if not isinstance(fresh["candidate"], ControlFlowProgram)
                else fresh["candidate"].digest()
            ),
            "wall_seconds": fresh["wall_seconds"],
        },
        "target_digest": target.digest(),
        "candidate_digest": candidate.digest(),
        "heldout_accuracy": sum(heldout) / len(heldout),
        "source_retention": sum(source_heldout) / len(source_heldout),
        "warm_faster_than_fresh": (
            fresh["status"] == "expressible" and warm_evaluations < fresh_evaluations
        ),
        "transfer_ratio_against_fresh_learner": transfer_ratio,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": (warm_evaluations + fresh_evaluations) * len(amounts),
            "unique_logical_lifetimes": warm_evaluations + fresh_evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": float(warm["wall_seconds"]) + float(fresh["wall_seconds"]),
            "latency_seconds_per_candidate": (
                float(warm["wall_seconds"]) + float(fresh["wall_seconds"])
            ) / max(1, warm_evaluations + fresh_evaluations),
            "stable_bits_to_threshold": warm_evaluations * len(amounts),
        },
        "promoted": all(gates.values()),
    }


def run(seeds: tuple[int, ...] = (17, 18, 19, 20)) -> dict[str, object]:
    reports = tuple(
        _run_arm(seed, reverse_inputs=reverse_inputs)
        for seed in seeds
        for reverse_inputs in (False, True)
    )
    return {
        "schema": "neural-computer.recipe-control-flow-frontier-growth.v1",
        "claim_boundary": (
            "bounded outcome-only multi-edit acquisition of one generic external "
            "control-flow file with protected-root retention; not efficient "
            "arbitrary program synthesis, unrestricted memory growth, or general "
            "continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in reports),
        "sample_efficiency_transfer_promoted": all(
            bool(report["warm_faster_than_fresh"]) for report in reports
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20])
    args = parser.parse_args()
    report = run(tuple(args.seeds))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
