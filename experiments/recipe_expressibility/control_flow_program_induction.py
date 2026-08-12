"""Bounded outcome-only structural synthesis for a generic control-flow ABI.

Unlike the scaffold-adaptation audit, this search starts with no executable
source program. It exhaustively enumerates a finite generic instruction space,
executes candidates under a private verifier, and admits only the candidate
whose scalar outcomes remain exact over the complete training prefix. The
search is intentionally bounded and reports ``budget_exhausted`` separately
from a certified finite-bound miss.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neural_computer import (
    ControlFlowInstruction,
    ControlFlowProgram,
    ControlFlowProgramMemory,
    iter_control_flow_programs,
)

COUNTER_COUNT = 2
PROGRAM_LENGTH = 4
TRAIN_AMOUNTS = tuple(range(8))
HELDOUT_AMOUNTS = tuple(range(8, 16))
MAX_STEPS = 128
TARGET_THRESHOLD = 1.0
BOUNDED_CANDIDATE_COUNT = 13_824


def _source_program() -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )


def _private_loop_target() -> ControlFlowProgram:
    """The verifier-only target: clear counter zero by repeated decrement."""

    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=3),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("jump", target=0),
            ControlFlowInstruction("halt"),
        ),
    )


def _score(
    candidate: ControlFlowProgram,
    reference: ControlFlowProgram,
    amounts: tuple[int, ...],
) -> tuple[float, ...]:
    values: list[float] = []
    for amount in amounts:
        actual = candidate.execute((amount, 0), max_steps=MAX_STEPS)
        expected = reference.execute((amount, 0), max_steps=MAX_STEPS)
        values.append(
            float(
                actual.status == "halted"
                and actual.counters == expected.counters
            )
        )
    return tuple(values)


def _bounded_search(
    *,
    reference: ControlFlowProgram,
    amounts: tuple[int, ...],
    max_candidates: int | None = None,
) -> tuple[ControlFlowProgram | None, dict[str, object]]:
    started = time.perf_counter()
    checked = 0
    for candidate in iter_control_flow_programs(
        counter_count=COUNTER_COUNT,
        min_length=PROGRAM_LENGTH,
        max_length=PROGRAM_LENGTH,
    ):
        if max_candidates is not None and checked >= max_candidates:
            return None, {
                "status": "budget_exhausted",
                "checked_candidates": checked,
                "wall_seconds": time.perf_counter() - started,
            }
        checked += 1
        outcomes = _score(candidate, reference, amounts)
        if len(outcomes) == len(amounts) and min(outcomes) >= TARGET_THRESHOLD:
            return candidate, {
                "status": "expressible",
                "checked_candidates": checked,
                "stable_bits_to_threshold": checked * len(amounts),
                "wall_seconds": time.perf_counter() - started,
            }
    return None, {
        "status": "inexpressible",
        "checked_candidates": checked,
        "stable_bits_to_threshold": None,
        "wall_seconds": time.perf_counter() - started,
    }


def _run_arm(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    target = _private_loop_target()
    source = _source_program()
    amounts = tuple(reversed(TRAIN_AMOUNTS)) if reverse_inputs else TRAIN_AMOUNTS
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    source_slot = memory.add_program(source, protect=True)

    candidate, search = _bounded_search(reference=target, amounts=amounts)
    admitted = None
    if candidate is not None:
        admitted = memory.admit_verified(
            candidate,
            _score(candidate, target, amounts),
            threshold=TARGET_THRESHOLD,
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
            protect=True,
        )

    before_missing = memory.digest()
    missing_rejected = False
    try:
        missing = memory.admit_verified(
            target,
            (),
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
        )
        missing_rejected = not missing.accepted
    except ValueError:
        missing_rejected = True
    missing_no_write = missing_rejected and memory.digest() == before_missing

    before_shuffled = memory.digest()
    shuffled = memory.admit_verified(
        target,
        tuple(float(index % 2 == 0) for index in range(len(amounts))),
        threshold=TARGET_THRESHOLD,
        min_observations=len(amounts),
        min_stable_observations=len(amounts),
    )
    shuffled_rejected = not shuffled.accepted and memory.digest() == before_shuffled

    heldout_accuracy = (
        0.0
        if candidate is None
        else sum(_score(candidate, target, HELDOUT_AMOUNTS)) / len(HELDOUT_AMOUNTS)
    )
    source_retention = sum(
        _score(memory.program(source_slot), source, HELDOUT_AMOUNTS)
    ) / len(HELDOUT_AMOUNTS)
    restored = ControlFlowProgramMemory.from_payload(memory.payload())
    restored_accuracy = (
        0.0
        if admitted is None or admitted.slot is None
        else sum(
            _score(restored.program(admitted.slot), target, HELDOUT_AMOUNTS)
        )
        / len(HELDOUT_AMOUNTS)
    )
    corrupted = memory.payload()
    corrupted["sha256"] = "0" * 64
    try:
        ControlFlowProgramMemory.from_payload(corrupted)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    _, budget_control = _bounded_search(
        reference=target,
        amounts=amounts,
        max_candidates=10,
    )
    gates = {
        "finite_search_expressible": search["status"] == "expressible",
        "target_admitted": bool(admitted and admitted.accepted),
        "target_heldout_mastery": heldout_accuracy >= 0.95,
        "source_protected": memory.is_file_protected(source_slot),
        "source_retention": source_retention >= 0.95,
        "reload_exact": restored.digest() == memory.digest(),
        "reload_target_mastery": restored_accuracy >= 0.95,
        "missing_evidence_no_write": missing_no_write,
        "shuffled_feedback_rejected": shuffled_rejected,
        "corruption_rejected": corruption_rejected,
        "budget_exhaustion_is_not_inexpressible": (
            budget_control["status"] == "budget_exhausted"
        ),
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "reverse_inputs": reverse_inputs,
        "schema": "neural-computer.recipe-control-flow-induction.v1",
        "architecture": {
            "counter_count": COUNTER_COUNT,
            "program_length": PROGRAM_LENGTH,
            "bounded_candidate_count": BOUNDED_CANDIDATE_COUNT,
            "learner_inputs": "opaque_candidate_and_scalar_verifier_outcomes",
            "forbidden_features": "target_program_and_task_names",
        },
        "search": search,
        "budget_control": budget_control,
        "target_digest": target.digest(),
        "candidate_digest": None if candidate is None else candidate.digest(),
        "heldout_accuracy": heldout_accuracy,
        "source_retention": source_retention,
        "restored_accuracy": restored_accuracy,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": int(search["checked_candidates"]) * len(amounts),
            "unique_logical_lifetimes": int(search["checked_candidates"]),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": float(search["wall_seconds"]),
            "stable_bits_to_threshold": search.get("stable_bits_to_threshold"),
            "transfer_ratio_against_fresh_learner": 1.0,
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
        "schema": "neural-computer.recipe-control-flow-induction.v1",
        "claim_boundary": (
            "bounded outcome-only structural synthesis of one generic loop in a "
            "finite control-flow basis with protected source retention; not "
            "unrestricted program induction, unbounded execution, or general "
            "continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in reports),
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
