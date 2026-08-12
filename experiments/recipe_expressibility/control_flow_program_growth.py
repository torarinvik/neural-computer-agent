"""Pressure-test outcome-only acquisition of a generic loop program.

The controller is absent from this CPU/files experiment by design: it tests
whether the replaceable external program substrate can add data-dependent
control flow while retaining an earlier straight-line-shaped capability.  The
verifier keeps the target program private and returns only scalar exact-match
outcomes over opaque counter inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ControlFlowInstruction,
    ControlFlowOutcomeSearch,
    ControlFlowProgram,
    ControlFlowProgramMemory,
    ControlFlowSearchState,
)

COUNTER_COUNT = 3
TRAIN_AMOUNTS = tuple(range(8))
HELDOUT_AMOUNTS = tuple(range(8, 16))
MAX_STEPS = 128
TARGET_THRESHOLD = 1.0
MAX_PROPOSALS = 256


def _transfer_program(destination: int) -> ControlFlowProgram:
    """Return a generic counter-transfer program; destination is not learner input."""

    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            # while counter 0 != 0: decrement it and increment destination
            # The verifier, not the search policy, chooses which destination
            # is correct for a given opaque context.
            ControlFlowInstruction(
                "jump_if_zero", counter=0, target=4
            ),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction(
                "inc", counter=destination
            ),
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
        initial = (amount, 0, 0)
        actual = candidate.execute(initial, max_steps=MAX_STEPS)
        expected = reference.execute(initial, max_steps=MAX_STEPS)
        values.append(
            float(
                actual.status == "halted"
                and expected.status == "halted"
                and actual.counters == expected.counters
            )
        )
    return tuple(values)


def _search_until(
    *,
    search: ControlFlowOutcomeSearch,
    state: ControlFlowSearchState,
    parent: ControlFlowProgram,
    reference: ControlFlowProgram,
    seed: int,
    scope: str,
    amounts: tuple[int, ...],
    feedback_mode: str = "verifier",
) -> tuple[ControlFlowSearchState, dict[str, object]]:
    if feedback_mode not in ("verifier", "reward_shuffled"):
        raise ValueError("unsupported control-flow feedback mode")
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    last_quality = 0.0
    for _ in range(MAX_PROPOSALS):
        try:
            proposal = search.propose(
                state,
                parent,
                generator=generator,
                scope=scope,
            )
        except RuntimeError:
            return state, {
                "accepted": False,
                "proposals": state.proposals,
                "program_digest": None,
                "quality": last_quality,
                "unique_verifier_bits": state.proposals * len(amounts),
                "wall_seconds": time.perf_counter() - started,
                "exhausted": True,
            }
        outcomes = _outcomes(proposal.program, reference, amounts)
        if feedback_mode == "reward_shuffled":
            outcomes = tuple(0.0 if index % 2 else 1.0 for index in range(len(outcomes)))
        feedback = search.record_outcomes(
            state,
            proposal,
            outcomes,
            threshold=TARGET_THRESHOLD,
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
        )
        state = feedback.state
        last_quality = feedback.quality
        if feedback.accepted:
            return state, {
                "accepted": True,
                "proposals": state.proposals,
                "program_digest": proposal.program.digest(),
                "program": proposal.program,
                "quality": last_quality,
                "stable_bits_to_threshold": (
                    None
                    if feedback.stable_bits_to_threshold is None
                    else feedback.stable_bits_to_threshold * len(amounts)
                ),
                "unique_verifier_bits": state.proposals * len(amounts),
                "wall_seconds": time.perf_counter() - started,
                "exhausted": False,
            }
    return state, {
        "accepted": False,
        "proposals": state.proposals,
        "program_digest": None,
        "quality": last_quality,
        "unique_verifier_bits": state.proposals * len(amounts),
        "wall_seconds": time.perf_counter() - started,
        "exhausted": False,
    }


def _digest_memory(memory: ControlFlowProgramMemory) -> str:
    return memory.digest()


def _run_seed(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    source = _transfer_program(1)
    target = _transfer_program(2)
    parent = source
    amounts = tuple(reversed(TRAIN_AMOUNTS)) if reverse_inputs else TRAIN_AMOUNTS
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    source_slot = memory.add_program(source, protect=True)

    search = ControlFlowOutcomeSearch()
    state, warm = _search_until(
        search=search,
        state=search.initial_state(),
        parent=parent,
        reference=target,
        seed=seed + 1_000,
        scope=f"warm-{seed}",
        amounts=amounts,
    )
    warm_program = warm.get("program")
    warm_receipt = None
    if isinstance(warm_program, ControlFlowProgram):
        warm_receipt = memory.admit_verified(
            warm_program,
            _outcomes(warm_program, target, amounts),
            threshold=TARGET_THRESHOLD,
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
            protect=True,
        )

    fresh_search = ControlFlowOutcomeSearch()
    _, fresh = _search_until(
        search=fresh_search,
        state=fresh_search.initial_state(),
        parent=_transfer_program(0),
        reference=target,
        seed=seed + 2_000,
        scope=f"fresh-{seed}",
        amounts=amounts,
    )

    shuffled_search = ControlFlowOutcomeSearch()
    _, shuffled = _search_until(
        search=shuffled_search,
        state=shuffled_search.initial_state(),
        parent=parent,
        reference=target,
        seed=seed + 3_000,
        scope=f"shuffled-{seed}",
        amounts=amounts,
        feedback_mode="reward_shuffled",
    )

    missing_program = target
    before_missing = _digest_memory(memory)
    missing_rejected = False
    try:
        missing = memory.admit_verified(
            missing_program,
            (),
            min_observations=len(amounts),
            min_stable_observations=len(amounts),
        )
        missing_rejected = not missing.accepted
    except ValueError:
        missing_rejected = True
    missing_no_write = missing_rejected and _digest_memory(memory) == before_missing

    heldout_accuracy = (
        0.0
        if not isinstance(warm_program, ControlFlowProgram)
        else sum(_outcomes(warm_program, target, HELDOUT_AMOUNTS)) / len(HELDOUT_AMOUNTS)
    )
    source_retention = sum(
        _outcomes(memory.program(source_slot), source, HELDOUT_AMOUNTS)
    ) / len(HELDOUT_AMOUNTS)
    reload = ControlFlowProgramMemory.from_payload(memory.payload())
    reload_accuracy = (
        0.0
        if warm_receipt is None or warm_receipt.slot is None
        else sum(
            _outcomes(reload.program(warm_receipt.slot), target, HELDOUT_AMOUNTS)
        )
        / len(HELDOUT_AMOUNTS)
    )

    corrupted_payload = memory.payload()
    corrupted_payload["sha256"] = "0" * 64
    try:
        ControlFlowProgramMemory.from_payload(corrupted_payload)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    gates = {
        "source_protected": memory.is_file_protected(source_slot),
        "target_admitted": bool(warm_receipt and warm_receipt.accepted),
        "target_heldout_mastery": heldout_accuracy >= 0.95,
        "source_retention": source_retention >= 0.95,
        "reload_exact": reload.digest() == memory.digest(),
        "reload_target_mastery": reload_accuracy >= 0.95,
        "reward_shuffled_rejected": not bool(shuffled["accepted"]),
        "missing_evidence_no_write": missing_no_write,
        "corruption_rejected": corruption_rejected,
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }
    accounting = {
        "unique_verifier_bits": int(warm["unique_verifier_bits"])
        + int(fresh["unique_verifier_bits"])
        + int(shuffled["unique_verifier_bits"]),
        "unique_logical_lifetimes": state.proposals
        + int(fresh["proposals"])
        + int(shuffled["proposals"]),
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": float(warm["wall_seconds"])
        + float(fresh["wall_seconds"])
        + float(shuffled["wall_seconds"]),
        "stable_bits_to_threshold": (
            int(warm["stable_bits_to_threshold"])
            if warm.get("stable_bits_to_threshold") is not None
            else None
        ),
    }
    return {
        "seed": seed,
        "reverse_inputs": reverse_inputs,
        "schema": "neural-computer.recipe-control-flow-growth.v1",
        "architecture": {
            "counter_count": COUNTER_COUNT,
            "executor": "bounded_counter_machine_v1",
            "program_search": "opaque_scalar_outcome_edit_credit_v1",
            "learner_inputs": "opaque_program_and_scalar_verifier_outcomes",
            "forbidden_features": "target_program_destination_names_and_task_ids",
        },
        "warm": {key: value for key, value in warm.items() if key != "program"},
        "fresh": {key: value for key, value in fresh.items() if key != "program"},
        "transfer": {
            "warm_to_fresh_proposal_ratio": (
                float(warm["proposals"]) / float(fresh["proposals"])
            ),
            "warm_not_slower_than_fresh": (
                int(warm["proposals"]) <= int(fresh["proposals"])
            ),
        },
        "reward_shuffled": {
            key: value for key, value in shuffled.items() if key != "program"
        },
        "target_program_digest": target.digest(),
        "warm_program_digest": (
            None if not isinstance(warm_program, ControlFlowProgram) else warm_program.digest()
        ),
        "heldout_accuracy": heldout_accuracy,
        "source_retention": source_retention,
        "reload_accuracy": reload_accuracy,
        "gates": gates,
        "accounting": accounting,
        "promoted": all(gates.values()),
    }


def run(seeds: tuple[int, ...] = (17, 18, 19, 20)) -> dict[str, object]:
    reports = tuple(
        report
        for seed in seeds
        for reverse_inputs in (False, True)
        for report in (_run_seed(seed, reverse_inputs=reverse_inputs),)
    )
    return {
        "schema": "neural-computer.recipe-control-flow-growth.v1",
        "claim_boundary": (
            "bounded outcome-only acquisition of a generic data-dependent loop "
            "program in external memory with source retention; not arbitrary "
            "program induction, unbounded execution, or general continual learning"
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
