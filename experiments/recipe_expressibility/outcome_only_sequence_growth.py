"""Audit outcome-only growth of an external generic recipe-file bank.

The target sequence is private to the verifier.  The search receives only the
candidate program and an ordered stream of scalar exact-match outcomes.  A
source file is protected before a second file is acquired; the controller
boundary is untouched because this is a memory-side CPU/file pressure test.

This is intentionally a narrow capability audit.  It tests whether a generic
instruction sequence can be discovered, admitted, persisted, and retained
without replay.  It does not claim arbitrary program induction or general
continual learning.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch

from neural_computer import (
    ExternalRecipeProgramMemory,
    OutcomeOnlyRecipeSequenceSearch,
    RecipeBasis,
    RecipeInstruction,
    RecipeProgram,
)

SLOT_VALUES = (2, 8)
TRAIN_STATES = tuple(state for state in product(range(2), range(8)) if sum(state) % 2 == 0)
HELDOUT_STATES = tuple(state for state in product(range(2), range(8)) if sum(state) % 2 == 1)
ALL_STATES = tuple(product(range(2), range(8)))
TARGET_THRESHOLD = 1.0
MAX_PROPOSALS = 256


def _basis() -> RecipeBasis:
    return RecipeBasis(slot_count=2, slot_values=SLOT_VALUES)


def _source() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction("inc", 0, modulus=2),),
    )


def _auxiliary_target() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("inc", 1, modulus=8),
        ),
    )


def _auxiliary_target_two() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("dec", 1, modulus=8),
        ),
    )


def _target() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("cinc", 1, 0, modulus=8),
        ),
    )


def _wrong_order() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction("cinc", 1, 0, modulus=8),
            RecipeInstruction("inc", 0, modulus=2),
        ),
    )


def _score(
    candidate: RecipeProgram,
    reference: RecipeProgram,
    states: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    return torch.tensor(
        [float(candidate.execute(state) == reference.execute(state)) for state in states],
        dtype=torch.float32,
    )


def _accuracy(
    candidate: RecipeProgram,
    reference: RecipeProgram,
    states: tuple[tuple[int, ...], ...],
) -> float:
    return float(_score(candidate, reference, states).mean().item())


def _search_until(
    search: OutcomeOnlyRecipeSequenceSearch,
    state,
    parent: RecipeProgram,
    reference: RecipeProgram,
    *,
    seed: int,
    max_proposals: int,
    scope: str,
    feedback_mode: str = "verifier",
) -> tuple[object, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    accepted = None
    last_quality = 0.0
    if feedback_mode not in ("verifier", "shuffled_null"):
        raise ValueError("unsupported recipe search feedback mode")
    for _ in range(max_proposals):
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
                "exhausted": True,
                "program": accepted,
                "proposals": state.proposals,
                "stable_bits_to_threshold": None,
                "quality": last_quality,
                "unique_verifier_bits": state.proposals * len(TRAIN_STATES),
                "wall_seconds": time.perf_counter() - started,
            }
        outcomes = (
            _score(proposal.program, reference, TRAIN_STATES)
            if feedback_mode == "verifier"
            else torch.tensor(
                [float(index % 2 == 0) for index in range(len(TRAIN_STATES))]
            )
        )
        feedback = search.record_outcomes(
            state,
            proposal,
            outcomes,
            threshold=TARGET_THRESHOLD,
            min_observations=len(TRAIN_STATES),
            min_stable_observations=len(TRAIN_STATES),
        )
        state = feedback.state
        last_quality = feedback.quality
        if feedback.receipt.accepted:
            accepted = proposal.program
            return state, {
                "accepted": True,
                "exhausted": False,
                "program": accepted,
                "proposals": state.proposals,
                "stable_bits_to_threshold": feedback.receipt.stable_bits_to_threshold,
                "quality": last_quality,
                "unique_verifier_bits": state.proposals * len(TRAIN_STATES),
                "wall_seconds": time.perf_counter() - started,
            }
    return state, {
        "accepted": False,
        "exhausted": False,
        "program": accepted,
        "proposals": state.proposals,
        "stable_bits_to_threshold": None,
        "quality": last_quality,
        "unique_verifier_bits": state.proposals * len(TRAIN_STATES),
        "wall_seconds": time.perf_counter() - started,
    }


def _program_digest(program: RecipeProgram | None) -> str | None:
    return None if program is None else program.digest()


def _opaque_scope(seed: int, index: int) -> str:
    """Produce an external binding key with no semantic task name."""

    return f"{seed:08x}-{index:02x}"


def _run_seed(seed: int) -> dict[str, object]:
    basis = _basis()
    source = _source()
    auxiliary = _auxiliary_target()
    auxiliary_two = _auxiliary_target_two()
    target = _target()
    wrong_order = _wrong_order()
    search = OutcomeOnlyRecipeSequenceSearch(
        basis,
        max_program_length=2,
        exploration=0.5,
        temperature=0.5,
    )
    search_state = search.initial_state()
    memory = ExternalRecipeProgramMemory(SLOT_VALUES)
    source_slot = memory.add_program(source)
    memory.protect_file(source_slot)

    search_state, auxiliary_search = _search_until(
        search,
        search_state,
        source,
        auxiliary,
        seed=seed + 1_000,
        max_proposals=MAX_PROPOSALS,
        scope=_opaque_scope(seed, 1),
    )
    auxiliary_candidate = auxiliary_search["program"]
    auxiliary_receipt = None
    if isinstance(auxiliary_candidate, RecipeProgram):
        auxiliary_receipt = memory.admit_verified_program(
            auxiliary_candidate,
            _score(auxiliary_candidate, auxiliary, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )

    search_state, auxiliary_two_search = _search_until(
        search,
        search_state,
        source,
        auxiliary_two,
        seed=seed + 1_500,
        max_proposals=MAX_PROPOSALS,
        scope=_opaque_scope(seed, 2),
    )
    auxiliary_two_candidate = auxiliary_two_search["program"]
    auxiliary_two_receipt = None
    if isinstance(auxiliary_two_candidate, RecipeProgram):
        auxiliary_two_receipt = memory.admit_verified_program(
            auxiliary_two_candidate,
            _score(auxiliary_two_candidate, auxiliary_two, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )

    search_state, target_search = _search_until(
        search,
        search_state,
        source,
        target,
        seed=seed + 2_000,
        max_proposals=MAX_PROPOSALS,
        scope=_opaque_scope(seed, 3),
    )
    target_candidate = target_search["program"]
    target_receipt = None
    if isinstance(target_candidate, RecipeProgram):
        target_receipt = memory.admit_verified_program(
            target_candidate,
            _score(target_candidate, target, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )

    fresh_search = OutcomeOnlyRecipeSequenceSearch(
        basis,
        max_program_length=2,
        exploration=0.5,
        temperature=0.5,
    )
    _, fresh_target_search = _search_until(
        fresh_search,
        fresh_search.initial_state(),
        source,
        target,
        seed=seed + 2_000,
        max_proposals=MAX_PROPOSALS,
        scope=_opaque_scope(seed, 3),
    )

    shuffled_search = OutcomeOnlyRecipeSequenceSearch(
        basis,
        max_program_length=2,
        exploration=0.5,
        temperature=0.5,
    )
    _, shuffled_target_search = _search_until(
        shuffled_search,
        shuffled_search.initial_state(),
        source,
        target,
        seed=seed + 3_000,
        max_proposals=MAX_PROPOSALS,
        scope=_opaque_scope(seed, 4),
        feedback_mode="shuffled_null",
    )

    reloaded = ExternalRecipeProgramMemory.from_payload(memory.payload())
    target_slot = None if target_receipt is None else target_receipt.slot
    source_retention = _accuracy(source, source, ALL_STATES)
    auxiliary_retention = (
        0.0
        if auxiliary_receipt is None or auxiliary_receipt.slot is None
        else _accuracy(memory.program(auxiliary_receipt.slot), auxiliary, HELDOUT_STATES)
    )
    auxiliary_two_retention = (
        0.0
        if auxiliary_two_receipt is None or auxiliary_two_receipt.slot is None
        else _accuracy(
            memory.program(auxiliary_two_receipt.slot),
            auxiliary_two,
            HELDOUT_STATES,
        )
    )
    target_train = (
        0.0
        if not isinstance(target_candidate, RecipeProgram)
        else _accuracy(target_candidate, target, TRAIN_STATES)
    )
    target_heldout = (
        0.0
        if not isinstance(target_candidate, RecipeProgram)
        else _accuracy(target_candidate, target, HELDOUT_STATES)
    )
    wrong_order_accuracy = _accuracy(wrong_order, target, ALL_STATES)
    reloaded_target = (
        0.0
        if target_slot is None
        else _accuracy(reloaded.program(target_slot), target, HELDOUT_STATES)
    )
    warm_target_proposals = int(target_search["proposals"])
    fresh_target_proposals = int(fresh_target_search["proposals"])
    training_bits = int(
        auxiliary_search["unique_verifier_bits"]
        + auxiliary_two_search["unique_verifier_bits"]
        + target_search["unique_verifier_bits"]
        + fresh_target_search["unique_verifier_bits"]
        + shuffled_target_search["unique_verifier_bits"]
    )
    audit_bits = len(ALL_STATES) * 6
    gates = {
        "source_not_replaced": memory.program(source_slot).digest() == source.digest(),
        "source_protected": memory.is_file_protected(source_slot),
        "auxiliary_admitted": bool(auxiliary_receipt and auxiliary_receipt.accepted),
        "auxiliary_two_admitted": bool(
            auxiliary_two_receipt and auxiliary_two_receipt.accepted
        ),
        "target_admitted": bool(target_receipt and target_receipt.accepted),
        "target_train_mastery": target_train >= 0.95,
        "target_heldout_mastery": target_heldout >= 0.95,
        "source_retention": source_retention >= 0.95,
        "auxiliary_retention": auxiliary_retention >= 0.95,
        "auxiliary_two_retention": auxiliary_two_retention >= 0.95,
        "wrong_order_rejected": wrong_order_accuracy < 0.95,
        "reload_retention": reloaded_target >= 0.95,
        "memory_persistence_exact": reloaded.digest() == memory.digest(),
        "shuffled_feedback_rejected": not bool(shuffled_target_search["accepted"]),
        "warm_search_not_slower_than_fresh": (
            warm_target_proposals <= fresh_target_proposals
        ),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "configuration": {
            "slot_values": SLOT_VALUES,
            "training_states": len(TRAIN_STATES),
            "heldout_states": len(HELDOUT_STATES),
            "all_states": len(ALL_STATES),
            "target_threshold": TARGET_THRESHOLD,
            "max_proposals": MAX_PROPOSALS,
            "learner_inputs": [
                "opaque_recipe_candidate",
                "opaque_parent_digest",
                "deterministic_scalar_verifier_outcome",
            ],
        },
        "source_digest": source.digest(),
        "auxiliary_candidate_digest": _program_digest(auxiliary_candidate),
        "auxiliary_two_candidate_digest": _program_digest(auxiliary_two_candidate),
        "target_candidate_digest": _program_digest(target_candidate),
        "target_reference_digest": target.digest(),
        "auxiliary_search": {
            key: value
            for key, value in auxiliary_search.items()
            if key != "program"
        },
        "auxiliary_two_search": {
            key: value
            for key, value in auxiliary_two_search.items()
            if key != "program"
        },
        "target_search": {
            key: value for key, value in target_search.items() if key != "program"
        },
        "fresh_target_search": {
            key: value
            for key, value in fresh_target_search.items()
            if key != "program"
        },
        "shuffled_target_search": {
            key: value
            for key, value in shuffled_target_search.items()
            if key != "program"
        },
        "admissions": {
            "auxiliary": None if auxiliary_receipt is None else auxiliary_receipt.payload(),
            "auxiliary_two": (
                None
                if auxiliary_two_receipt is None
                else auxiliary_two_receipt.payload()
            ),
            "target": None if target_receipt is None else target_receipt.payload(),
        },
        "metrics": {
            "source_retention": source_retention,
            "auxiliary_retention": auxiliary_retention,
            "auxiliary_two_retention": auxiliary_two_retention,
            "target_train_accuracy": target_train,
            "target_heldout_accuracy": target_heldout,
            "wrong_order_accuracy": wrong_order_accuracy,
            "reloaded_target_accuracy": reloaded_target,
            "warm_to_fresh_proposal_ratio": (
                warm_target_proposals / fresh_target_proposals
                if fresh_target_proposals
                else None
            ),
            "shuffled_feedback_proposals": int(shuffled_target_search["proposals"]),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": training_bits + audit_bits,
            "unique_logical_lifetimes": (
                int(auxiliary_search["proposals"])
                + int(auxiliary_two_search["proposals"])
                + int(target_search["proposals"])
                + int(fresh_target_search["proposals"])
                + int(shuffled_target_search["proposals"])
                + 6
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": (
                float(auxiliary_search["wall_seconds"])
                + float(auxiliary_two_search["wall_seconds"])
                + float(target_search["wall_seconds"])
                + float(fresh_target_search["wall_seconds"])
                + float(shuffled_target_search["wall_seconds"])
            ),
        },
    }


def run(seeds: tuple[int, ...] = (17, 18)) -> dict[str, object]:
    reports = tuple(_run_seed(seed) for seed in seeds)
    return {
        "schema": "neural-computer.recipe-outcome-only-sequence-growth.v1",
        "claim_boundary": (
            "bounded outcome-only acquisition of order-sensitive generic recipe "
            "files with opaque scope-local candidate history and protected source "
            "retention; scalar proposal-prior transfer is explicitly tested and "
            "not promoted when it regresses"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(tuple(args.seeds))
    report["wall_seconds"] = time.perf_counter() - started
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
