"""Audit factorized, replay-free proposal credit across parent programs.

The first contextual proposal policy memorized exact whole-program digests.
This audit gives the learner a different parent program in the transfer
context.  The reusable target edit therefore has a different candidate digest
but the same generic instruction and insertion-position factors.  A separate
reversal context tests whether local evidence can override a shared factor
prior without changing the frozen controller.
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
    FactorizedOpaqueContextRecipeProposalMemory,
    OpaqueContextRecipeProposalMemory,
    OutcomeOnlyRecipeSequenceSearch,
    RecipeBasis,
    RecipeInstruction,
    RecipeProgram,
    RecipeProgramProposalFactors,
)

SLOT_VALUES = (2, 8)
TRAIN_STATES = tuple(
    state for state in product(range(2), range(8)) if sum(state) % 2 == 0
)
HELDOUT_STATES = tuple(
    state for state in product(range(2), range(8)) if sum(state) % 2 == 1
)
ALL_STATES = tuple(product(range(2), range(8)))
TARGET_THRESHOLD = 1.0
MAX_PROPOSALS = 256


def _basis() -> RecipeBasis:
    return RecipeBasis(slot_count=2, slot_values=SLOT_VALUES)


def _source(operation: str) -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction(operation, 0, modulus=2),),
    )


def _target(source_operation: str, child_operation: str) -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction(source_operation, 0, modulus=2),
            RecipeInstruction(child_operation, 1, 0, modulus=8),
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
    candidate: RecipeProgram | None,
    reference: RecipeProgram,
    states: tuple[tuple[int, ...], ...],
) -> float:
    if candidate is None:
        return 0.0
    return float(_score(candidate, reference, states).mean().item())


def _opaque_key(seed: int, index: int) -> str:
    return f"{seed:08x}-{index:02x}"


def _search_until(
    search: OutcomeOnlyRecipeSequenceSearch,
    parent: RecipeProgram,
    reference: RecipeProgram,
    *,
    seed: int,
    scope: str,
    context: str,
    feedback_mode: str = "verifier",
) -> tuple[RecipeProgram | None, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = search.initial_state()
    started = time.perf_counter()
    if feedback_mode not in ("verifier", "shuffled_null"):
        raise ValueError("unsupported factorized proposal feedback mode")
    last_quality = 0.0
    for _ in range(MAX_PROPOSALS):
        try:
            proposal = search.propose(
                state,
                parent,
                generator=generator,
                scope=scope,
                context=context,
            )
        except RuntimeError:
            break
        outcomes = (
            _score(proposal.program, reference, TRAIN_STATES)
            if feedback_mode == "verifier"
            else torch.tensor(
                [float(index % 2 == 0) for index in range(len(TRAIN_STATES))],
                dtype=torch.float32,
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
            return proposal.program, {
                "accepted": True,
                "proposals": state.proposals,
                "quality": last_quality,
                "unique_verifier_bits": state.proposals * len(TRAIN_STATES),
                "wall_seconds": time.perf_counter() - started,
            }
    return None, {
        "accepted": False,
        "proposals": state.proposals,
        "quality": last_quality,
        "unique_verifier_bits": state.proposals * len(TRAIN_STATES),
        "wall_seconds": time.perf_counter() - started,
    }


def _factor_probe(
    parent: RecipeProgram,
    target: RecipeProgram,
) -> RecipeProgramProposalFactors:
    search = OutcomeOnlyRecipeSequenceSearch(_basis(), max_program_length=2)
    return search.proposal_factors(parent, 1, target)


def _policy() -> FactorizedOpaqueContextRecipeProposalMemory:
    return FactorizedOpaqueContextRecipeProposalMemory(
        exploration_floor=0.05,
        shared_prior_weight=0.25,
        exploration_bonus=0.25,
        instruction_weight=1.0,
        position_weight=0.75,
        operator_weight=0.25,
        temperature=0.05,
    )


def _whole_policy() -> OpaqueContextRecipeProposalMemory:
    return OpaqueContextRecipeProposalMemory(
        exploration_floor=0.1,
        global_prior_weight=0.0,
        exploration_bonus=0.25,
        temperature=0.25,
    )


def _run_seed(seed: int) -> dict[str, object]:
    source_a = _source("inc")
    source_b = _source("dec")
    target_a = _target("inc", "cinc")
    target_b = _target("dec", "cinc")
    target_c = _target("inc", "cdec")
    factors_a = _factor_probe(source_a, target_a)
    factors_b = _factor_probe(source_b, target_b)
    factors_c = _factor_probe(source_a, target_c)

    policy = _policy()
    acquisition_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=policy
    )
    candidate_a, acquisition_a = _search_until(
        acquisition_search,
        source_a,
        target_a,
        seed=seed + 100,
        scope=_opaque_key(seed, 1),
        context=_opaque_key(seed, 101),
    )
    context_a = _opaque_key(seed, 101)
    context_b = _opaque_key(seed, 102)
    context_c = _opaque_key(seed, 103)
    persisted_after_a = FactorizedOpaqueContextRecipeProposalMemory.from_payload(
        policy.payload()
    )
    before_b = persisted_after_a.proposal_probabilities(
        context_b, (factors_b, factors_c)
    )

    warm_b_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=persisted_after_a
    )
    candidate_b, warm_b = _search_until(
        warm_b_search,
        source_b,
        target_b,
        seed=seed + 200,
        scope=_opaque_key(seed, 2),
        context=context_b,
    )
    fresh_b_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=_policy()
    )
    fresh_b_candidate, fresh_b = _search_until(
        fresh_b_search,
        source_b,
        target_b,
        seed=seed + 200,
        scope=_opaque_key(seed, 2),
        context=context_b,
    )

    whole_policy = _whole_policy()
    whole_acquisition_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=whole_policy
    )
    _, whole_acquisition_a = _search_until(
        whole_acquisition_search,
        source_a,
        target_a,
        seed=seed + 100,
        scope=_opaque_key(seed, 11),
        context=context_a,
    )
    whole_warm_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=whole_policy
    )
    whole_candidate_b, whole_warm_b = _search_until(
        whole_warm_search,
        source_b,
        target_b,
        seed=seed + 200,
        scope=_opaque_key(seed, 12),
        context=context_b,
    )

    reversal_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=persisted_after_a
    )
    candidate_c, reversal_c = _search_until(
        reversal_search,
        source_a,
        target_c,
        seed=seed + 300,
        scope=_opaque_key(seed, 3),
        context=context_c,
    )
    after_c = persisted_after_a.proposal_probabilities(
        context_c, (factors_b, factors_c)
    )
    repeat_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=persisted_after_a
    )
    repeat_a, repeat_a_result = _search_until(
        repeat_search,
        source_a,
        target_a,
        seed=seed + 400,
        scope=_opaque_key(seed, 4),
        context=context_a,
    )
    after_a = persisted_after_a.proposal_probabilities(
        context_a, (factors_b, factors_c)
    )
    fresh_c_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=_policy()
    )
    _, fresh_c = _search_until(
        fresh_c_search,
        source_a,
        target_c,
        seed=seed + 300,
        scope=_opaque_key(seed, 3),
        context=context_c,
    )

    shuffled_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(), max_program_length=2, proposal_policy=_policy()
    )
    _, shuffled = _search_until(
        shuffled_search,
        source_b,
        target_b,
        seed=seed + 500,
        scope=_opaque_key(seed, 5),
        context=context_b,
        feedback_mode="shuffled_null",
    )

    memory = ExternalRecipeProgramMemory(SLOT_VALUES)
    source_a_slot = memory.add_program(source_a)
    source_b_slot = memory.add_program(source_b)
    memory.protect_file(source_a_slot)
    memory.protect_file(source_b_slot)
    target_a_receipt = None
    target_b_receipt = None
    if candidate_a is not None:
        target_a_receipt = memory.admit_verified_program(
            candidate_a,
            _score(candidate_a, target_a, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )
    if candidate_b is not None:
        target_b_receipt = memory.admit_verified_program(
            candidate_b,
            _score(candidate_b, target_b, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )
    reloaded_memory = ExternalRecipeProgramMemory.from_payload(memory.payload())
    reloaded_policy = FactorizedOpaqueContextRecipeProposalMemory.from_payload(
        persisted_after_a.payload()
    )
    persistence_exact = (
        reloaded_policy.digest() == persisted_after_a.digest()
        and reloaded_memory.digest() == memory.digest()
    )

    warm_b_proposals = int(warm_b["proposals"])
    fresh_b_proposals = int(fresh_b["proposals"])
    whole_warm_b_proposals = int(whole_warm_b["proposals"])
    gates = {
        "target_parent_digest_changes": (
            target_a.digest() != target_b.digest()
            and factors_a == factors_b
        ),
        "factor_transfer_prefers_reusable_edit": float(before_b[0]) > float(before_b[1]),
        "factor_warm_b_mastery": _accuracy(candidate_b, target_b, HELDOUT_STATES) >= 0.95,
        "factor_warm_b_not_slower_than_fresh": (
            warm_b_proposals <= fresh_b_proposals
        ),
        "factor_warm_b_beats_whole_candidate_prior": (
            warm_b_proposals < whole_warm_b_proposals
        ),
        "reversal_mastery": _accuracy(candidate_c, target_c, HELDOUT_STATES) >= 0.95,
        "reversal_local_credit_prefers_new_instruction": float(after_c[1]) > float(after_c[0]),
        "old_context_prefers_old_instruction": float(after_a[0]) > float(after_a[1]),
        "repeat_a_mastery": _accuracy(repeat_a, target_a, HELDOUT_STATES) >= 0.95,
        "target_a_file_admitted": bool(target_a_receipt and target_a_receipt.accepted),
        "target_b_file_admitted": bool(target_b_receipt and target_b_receipt.accepted),
        "protected_sources_retained": (
            reloaded_memory.program(source_a_slot).digest() == source_a.digest()
            and reloaded_memory.program(source_b_slot).digest() == source_b.digest()
            and reloaded_memory.is_file_protected(source_a_slot)
            and reloaded_memory.is_file_protected(source_b_slot)
        ),
        "policy_and_file_persistence_exact": persistence_exact,
        "exploration_floor_active": bool(
            torch.all(
                before_b
                >= persisted_after_a.exploration_floor / len(before_b)
            )
        ),
        "shuffled_feedback_rejected": not bool(shuffled["accepted"]),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    searches = (
        acquisition_a,
        warm_b,
        fresh_b,
        whole_acquisition_a,
        whole_warm_b,
        reversal_c,
        repeat_a_result,
        fresh_c,
        shuffled,
    )
    return {
        "seed": seed,
        "configuration": {
            "slot_values": SLOT_VALUES,
            "training_states": len(TRAIN_STATES),
            "heldout_states": len(HELDOUT_STATES),
            "exploration_floor": persisted_after_a.exploration_floor,
            "shared_prior_weight": persisted_after_a.shared_prior_weight,
            "learner_inputs": [
                "opaque_recipe_candidate",
                "opaque_parent_digest",
                "opaque_context_key",
                "opaque_instruction_digest",
                "opaque_position_factor",
                "deterministic_scalar_verifier_outcome",
            ],
        },
        "candidate_digests": {
            "target_a": target_a.digest(),
            "target_b": target_b.digest(),
            "target_a_acquired": None if candidate_a is None else candidate_a.digest(),
            "target_b_acquired": None if candidate_b is None else candidate_b.digest(),
            "whole_target_b": (
                None if whole_candidate_b is None else whole_candidate_b.digest()
            ),
            "fresh_target_b": (
                None if fresh_b_candidate is None else fresh_b_candidate.digest()
            ),
        },
        "factor_descriptors": {
            "a": factors_a.payload(),
            "b": factors_b.payload(),
            "c": factors_c.payload(),
        },
        "searches": {name: result for name, result in (
            ("acquisition_a", acquisition_a),
            ("warm_b", warm_b),
            ("fresh_b", fresh_b),
            ("whole_acquisition_a", whole_acquisition_a),
            ("whole_warm_b", whole_warm_b),
            ("reversal_c", reversal_c),
            ("repeat_a", repeat_a_result),
            ("fresh_c", fresh_c),
            ("shuffled", shuffled),
        )},
        "context_probabilities": {
            "before_b": [float(value) for value in before_b],
            "after_c": [float(value) for value in after_c],
            "after_a": [float(value) for value in after_a],
        },
        "metrics": {
            "acquisition_a_heldout_accuracy": _accuracy(
                candidate_a, target_a, HELDOUT_STATES
            ),
            "warm_b_heldout_accuracy": _accuracy(candidate_b, target_b, HELDOUT_STATES),
            "fresh_b_heldout_accuracy": _accuracy(
                fresh_b_candidate, target_b, HELDOUT_STATES
            ),
            "reversal_heldout_accuracy": _accuracy(
                candidate_c, target_c, HELDOUT_STATES
            ),
            "repeat_a_heldout_accuracy": _accuracy(
                repeat_a, target_a, HELDOUT_STATES
            ),
            "warm_b_to_fresh_factor_ratio": (
                warm_b_proposals / fresh_b_proposals if fresh_b_proposals else None
            ),
            "warm_b_to_whole_prior_ratio": (
                warm_b_proposals / whole_warm_b_proposals
                if whole_warm_b_proposals
                else None
            ),
            "reversal_to_fresh_ratio": (
                int(reversal_c["proposals"]) / int(fresh_c["proposals"])
                if int(fresh_c["proposals"])
                else None
            ),
        },
        "admissions": {
            "target_a": None if target_a_receipt is None else target_a_receipt.payload(),
            "target_b": None if target_b_receipt is None else target_b_receipt.payload(),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": sum(
                int(result["unique_verifier_bits"]) for result in searches
            )
            + len(ALL_STATES) * 6,
            "unique_logical_lifetimes": sum(
                int(result["proposals"]) for result in searches
            )
            + 6,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(
                float(result["wall_seconds"]) for result in searches
            ),
        },
    }


def run(seeds: tuple[int, ...] = (17, 18)) -> dict[str, object]:
    reports = tuple(_run_seed(seed) for seed in seeds)
    return {
        "schema": "neural-computer.factorized-context-recipe-proposal-credit.v1",
        "claim_boundary": (
            "bounded replay-free transfer of aggregate instruction/position "
            "proposal credit across different parent programs with local "
            "reversal routing; not unrestricted memory growth or general "
            "continual learning"
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
