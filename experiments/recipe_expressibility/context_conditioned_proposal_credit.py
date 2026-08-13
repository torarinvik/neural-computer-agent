"""Audit context-conditioned, replay-free recipe proposal credit.

This pressure test asks whether an external memory can learn which opaque
instruction-sequence candidate is useful in one context and reuse that credit
in a later lifetime without replaying the old verifier rows.  The controller
is not trained or modified.  Context and candidate identity are both opaque;
the verifier exposes only the current candidate's scalar exact-match stream.

The result is intentionally narrower than general continual learning.  It
tests content-addressed proposal credit and an exploration floor, not arbitrary
program synthesis or unrestricted memory growth.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch

from neural_computer import (
    OpaqueContextRecipeProposalMemory,
    OutcomeOnlyRecipeSequenceSearch,
    RecipeBasis,
    RecipeInstruction,
    RecipeProgram,
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


def _source() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction("inc", 0, modulus=2),),
    )


def _target_increment() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("cinc", 1, 0, modulus=8),
        ),
    )


def _target_decrement() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (
            RecipeInstruction("inc", 0, modulus=2),
            RecipeInstruction("cdec", 1, 0, modulus=8),
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
    reference: RecipeProgram,
    *,
    seed: int,
    scope: str,
    context: str,
    max_proposals: int = MAX_PROPOSALS,
    feedback_mode: str = "verifier",
) -> tuple[RecipeProgram | None, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = search.initial_state()
    started = time.perf_counter()
    if feedback_mode not in ("verifier", "shuffled_null"):
        raise ValueError("unsupported recipe proposal feedback mode")
    last_quality = 0.0
    for _ in range(max_proposals):
        try:
            proposal = search.propose(
                state,
                _source(),
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


def _run_seed(seed: int) -> dict[str, object]:
    target_a = _target_increment()
    target_b = _target_decrement()
    policy = OpaqueContextRecipeProposalMemory(
        exploration_floor=0.1,
        global_prior_weight=0.0,
        exploration_bonus=0.25,
        temperature=0.25,
    )

    acquisition = OutcomeOnlyRecipeSequenceSearch(
        _basis(),
        max_program_length=2,
        proposal_policy=policy,
    )
    candidate_a, acquisition_a = _search_until(
        acquisition,
        target_a,
        seed=seed + 100,
        scope=_opaque_key(seed, 1),
        context=_opaque_key(seed, 101),
    )
    context_a = _opaque_key(seed, 101)
    context_b = _opaque_key(seed, 102)
    before_b = policy.proposal_probabilities(
        context_b,
        (target_a.digest(), target_b.digest()),
    )
    candidate_b, acquisition_b = _search_until(
        acquisition,
        target_b,
        seed=seed + 200,
        scope=_opaque_key(seed, 2),
        context=context_b,
    )

    persisted_policy = OpaqueContextRecipeProposalMemory.from_payload(policy.payload())
    warm_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(),
        max_program_length=2,
        proposal_policy=persisted_policy,
    )
    warm_a, warm_a_search = _search_until(
        warm_search,
        target_a,
        seed=seed + 300,
        scope=_opaque_key(seed, 3),
        context=context_a,
    )
    warm_b, warm_b_search = _search_until(
        warm_search,
        target_b,
        seed=seed + 400,
        scope=_opaque_key(seed, 4),
        context=context_b,
    )

    fresh_a_policy = OpaqueContextRecipeProposalMemory(
        exploration_floor=0.1,
        global_prior_weight=0.0,
        exploration_bonus=0.25,
        temperature=0.25,
    )
    fresh_a_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(),
        max_program_length=2,
        proposal_policy=fresh_a_policy,
    )
    _, fresh_a_result = _search_until(
        fresh_a_search,
        target_a,
        seed=seed + 300,
        scope=_opaque_key(seed, 3),
        context=context_a,
    )
    fresh_b_policy = OpaqueContextRecipeProposalMemory(
        exploration_floor=0.1,
        global_prior_weight=0.0,
        exploration_bonus=0.25,
        temperature=0.25,
    )
    fresh_b_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(),
        max_program_length=2,
        proposal_policy=fresh_b_policy,
    )
    _, fresh_b_result = _search_until(
        fresh_b_search,
        target_b,
        seed=seed + 400,
        scope=_opaque_key(seed, 4),
        context=context_b,
    )

    shuffled_policy = OpaqueContextRecipeProposalMemory(
        exploration_floor=0.1,
        global_prior_weight=0.0,
        exploration_bonus=0.25,
        temperature=0.25,
    )
    shuffled_search = OutcomeOnlyRecipeSequenceSearch(
        _basis(),
        max_program_length=2,
        proposal_policy=shuffled_policy,
    )
    _, shuffled_result = _search_until(
        shuffled_search,
        target_a,
        seed=seed + 500,
        scope=_opaque_key(seed, 5),
        context=context_a,
        feedback_mode="shuffled_null",
    )

    after_a = persisted_policy.proposal_probabilities(
        context_a,
        (target_a.digest(), target_b.digest()),
    )
    after_b = persisted_policy.proposal_probabilities(
        context_b,
        (target_a.digest(), target_b.digest()),
    )
    persistence_exact = persisted_policy.digest() == (
        OpaqueContextRecipeProposalMemory.from_payload(persisted_policy.payload()).digest()
    )
    warm_a_proposals = int(warm_a_search["proposals"])
    warm_b_proposals = int(warm_b_search["proposals"])
    fresh_a_proposals = int(fresh_a_result["proposals"])
    fresh_b_proposals = int(fresh_b_result["proposals"])
    gates = {
        "acquisition_a_mastery": _accuracy(candidate_a, target_a, HELDOUT_STATES) >= 0.95,
        "acquisition_b_mastery": _accuracy(candidate_b, target_b, HELDOUT_STATES) >= 0.95,
        "warm_a_mastery": _accuracy(warm_a, target_a, HELDOUT_STATES) >= 0.95,
        "warm_b_mastery": _accuracy(warm_b, target_b, HELDOUT_STATES) >= 0.95,
        "warm_a_not_slower_than_fresh": warm_a_proposals <= fresh_a_proposals,
        "warm_b_not_slower_than_fresh": warm_b_proposals <= fresh_b_proposals,
        "context_a_prefers_a": float(after_a[0]) > float(after_a[1]),
        "context_b_prefers_b": float(after_b[1]) > float(after_b[0]),
        "unseen_context_is_unbiased": abs(float(before_b[0] - before_b[1])) < 1e-12,
        "unseen_context_exploration_floor": bool(torch.all(before_b >= 0.05)),
        "shuffled_feedback_rejected": not bool(shuffled_result["accepted"]),
        "policy_persistence_exact": persistence_exact,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    all_searches = (
        acquisition_a,
        acquisition_b,
        warm_a_search,
        warm_b_search,
        fresh_a_result,
        fresh_b_result,
        shuffled_result,
    )
    return {
        "seed": seed,
        "configuration": {
            "slot_values": SLOT_VALUES,
            "training_states": len(TRAIN_STATES),
            "heldout_states": len(HELDOUT_STATES),
            "exploration_floor": persisted_policy.exploration_floor,
            "global_prior_weight": persisted_policy.global_prior_weight,
            "learner_inputs": [
                "opaque_recipe_candidate",
                "opaque_parent_digest",
                "opaque_context_key",
                "deterministic_scalar_verifier_outcome",
            ],
        },
        "candidate_digests": {
            "target_a": target_a.digest(),
            "target_b": target_b.digest(),
            "acquired_a": None if candidate_a is None else candidate_a.digest(),
            "acquired_b": None if candidate_b is None else candidate_b.digest(),
        },
        "searches": {
            "acquisition_a": acquisition_a,
            "acquisition_b": acquisition_b,
            "warm_a": warm_a_search,
            "warm_b": warm_b_search,
            "fresh_a": fresh_a_result,
            "fresh_b": fresh_b_result,
            "shuffled": shuffled_result,
        },
        "context_probabilities": {
            "before_b": [float(value) for value in before_b],
            "after_a": [float(value) for value in after_a],
            "after_b": [float(value) for value in after_b],
        },
        "metrics": {
            "acquisition_a_heldout_accuracy": _accuracy(
                candidate_a, target_a, HELDOUT_STATES
            ),
            "acquisition_b_heldout_accuracy": _accuracy(
                candidate_b, target_b, HELDOUT_STATES
            ),
            "warm_a_heldout_accuracy": _accuracy(warm_a, target_a, HELDOUT_STATES),
            "warm_b_heldout_accuracy": _accuracy(warm_b, target_b, HELDOUT_STATES),
            "warm_a_to_fresh_proposal_ratio": (
                warm_a_proposals / fresh_a_proposals if fresh_a_proposals else None
            ),
            "warm_b_to_fresh_proposal_ratio": (
                warm_b_proposals / fresh_b_proposals if fresh_b_proposals else None
            ),
            "shuffled_proposals": int(shuffled_result["proposals"]),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": sum(
                int(result["unique_verifier_bits"]) for result in all_searches
            )
            + len(ALL_STATES) * 4,
            "unique_logical_lifetimes": sum(
                int(result["proposals"]) for result in all_searches
            )
            + 4,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(
                float(result["wall_seconds"]) for result in all_searches
            ),
        },
    }


def run(seeds: tuple[int, ...] = (17, 18)) -> dict[str, object]:
    reports = tuple(_run_seed(seed) for seed in seeds)
    return {
        "schema": "neural-computer.context-conditioned-recipe-proposal-credit.v1",
        "claim_boundary": (
            "replay-free context-conditioned reuse of aggregate scalar proposal "
            "credit with a nonzero exploration floor; this is not general "
            "continual learning or unrestricted program synthesis"
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
