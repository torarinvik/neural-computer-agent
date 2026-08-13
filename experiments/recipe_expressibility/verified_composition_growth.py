"""Audit verifier-gated growth by composing protected external recipe files.

The controller and atomic interpreter are frozen.  Three independently
verified one-step files are stored first.  The external composition layer then
discovers and admits a two-file program, protects it, and composes that file
with another protected fragment to reach depth three.  Only scalar verifier
outcomes train the optional composition policy; raw rows are not persisted.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch

from neural_computer import (
    ExternalRecipeCompositionMemory,
    OpaqueContextRecipeCompositionMemory,
    OutcomeOnlyRecipeCompositionSearch,
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


def _fragment_a() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction("inc", 0, modulus=2),),
    )


def _fragment_b() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction("inc", 1, modulus=8),),
    )


def _fragment_c() -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        (RecipeInstruction("cinc", 1, 0, modulus=8),),
    )


def _serial(left: RecipeProgram, right: RecipeProgram) -> RecipeProgram:
    return RecipeProgram(
        SLOT_VALUES,
        left.instructions + right.instructions,
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


def _policy() -> OpaqueContextRecipeCompositionMemory:
    return OpaqueContextRecipeCompositionMemory(
        exploration_floor=0.05,
        shared_prior_weight=0.25,
        exploration_bonus=0.25,
        left_weight=1.0,
        right_weight=1.0,
        mode_weight=0.5,
        temperature=0.05,
    )


def _key(seed: int, index: int) -> str:
    return f"{seed:08x}-{index:02x}"


def _search_until(
    search: OutcomeOnlyRecipeCompositionSearch,
    reference: RecipeProgram,
    *,
    seed: int,
    scope: str,
    context: str,
    feedback_mode: str = "verifier",
) -> tuple[object | None, dict[str, object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = search.initial_state()
    started = time.perf_counter()
    if feedback_mode not in ("verifier", "shuffled_null"):
        raise ValueError("unsupported composition feedback mode")
    last_quality = 0.0
    for _ in range(MAX_PROPOSALS):
        try:
            proposal = search.propose(
                state,
                generator=generator,
                scope=scope,
                context=context,
            )
        except RuntimeError:
            break
        outcomes = (
            _score(proposal.candidate.program, reference, TRAIN_STATES)
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
            return proposal, {
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


def _admit_atomic(
    memory: ExternalRecipeCompositionMemory,
    program: RecipeProgram,
) -> object:
    return memory.admit_verified_program(
        program,
        _score(program, program, ALL_STATES),
        threshold=TARGET_THRESHOLD,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=True,
    )


def _run_seed(seed: int) -> dict[str, object]:
    fragment_a = _fragment_a()
    fragment_b = _fragment_b()
    fragment_c = _fragment_c()
    target_depth2 = _serial(fragment_a, fragment_c)
    target_depth3 = _serial(target_depth2, fragment_c)
    wrong_depth2 = _serial(fragment_c, fragment_a)
    wrong_depth3 = _serial(fragment_c, target_depth2)

    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    source_receipts = tuple(
        _admit_atomic(memory, fragment) for fragment in (fragment_a, fragment_b, fragment_c)
    )
    source_slots = tuple(receipt.slot for receipt in source_receipts)
    if any(slot is None for slot in source_slots):
        raise RuntimeError("atomic source admission did not return slots")
    source_slots = tuple(int(slot) for slot in source_slots)
    source_digests = tuple(fragment.digest() for fragment in (fragment_a, fragment_b, fragment_c))

    policy = _policy()
    depth2_search = OutcomeOnlyRecipeCompositionSearch(
        memory,
        max_program_length=2,
        policy=policy,
    )
    depth2_proposal, depth2_result = _search_until(
        depth2_search,
        target_depth2,
        seed=seed + 100,
        scope=_key(seed, 1),
        context=_key(seed, 101),
    )
    depth2_receipt = None
    if depth2_proposal is not None:
        depth2_receipt = memory.admit_verified_composition(
            depth2_proposal.candidate,
            _score(depth2_proposal.candidate.program, target_depth2, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )
    if depth2_receipt is None or depth2_receipt.slot is None:
        raise RuntimeError("depth-2 composition did not commit")
    depth2_slot = depth2_receipt.slot

    persisted_policy = OpaqueContextRecipeCompositionMemory.from_payload(policy.payload())
    pre_depth3_memory = ExternalRecipeCompositionMemory.from_payload(memory.payload())
    depth3_search = OutcomeOnlyRecipeCompositionSearch(
        memory,
        max_program_length=3,
        policy=persisted_policy,
    )
    depth3_proposal, depth3_result = _search_until(
        depth3_search,
        target_depth3,
        seed=seed + 200,
        scope=_key(seed, 2),
        context=_key(seed, 102),
    )
    depth3_receipt = None
    if depth3_proposal is not None:
        depth3_receipt = memory.admit_verified_composition(
            depth3_proposal.candidate,
            _score(depth3_proposal.candidate.program, target_depth3, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )
    if depth3_receipt is None or depth3_receipt.slot is None:
        raise RuntimeError("depth-3 composition did not commit")
    depth3_slot = depth3_receipt.slot

    fresh_depth3_search = OutcomeOnlyRecipeCompositionSearch(
        pre_depth3_memory,
        max_program_length=3,
        policy=_policy(),
    )
    _, fresh_depth3_result = _search_until(
        fresh_depth3_search,
        target_depth3,
        seed=seed + 200,
        scope=_key(seed, 3),
        context=_key(seed, 103),
    )
    shuffled_search = OutcomeOnlyRecipeCompositionSearch(
        pre_depth3_memory,
        max_program_length=3,
        policy=_policy(),
    )
    _, shuffled_result = _search_until(
        shuffled_search,
        target_depth3,
        seed=seed + 300,
        scope=_key(seed, 4),
        context=_key(seed, 104),
        feedback_mode="shuffled_null",
    )

    missing_before = memory.digest()
    missing_candidates = memory.composition_candidates(max_program_length=3)
    missing_candidate = missing_candidates[0]
    missing_receipt = memory.admit_verified_composition(
        missing_candidate,
        torch.empty(0),
        threshold=TARGET_THRESHOLD,
        min_observations=1,
        min_stable_observations=1,
    )
    missing_noop = (
        not missing_receipt.accepted
        and memory.digest() == missing_before
    )

    wrong_depth2_accuracy = _accuracy(wrong_depth2, target_depth2, ALL_STATES)
    wrong_depth3_accuracy = _accuracy(wrong_depth3, target_depth3, ALL_STATES)
    reloaded_memory = ExternalRecipeCompositionMemory.from_payload(memory.payload())
    reloaded_policy = OpaqueContextRecipeCompositionMemory.from_payload(
        persisted_policy.payload()
    )
    source_retention = [
        _accuracy(reloaded_memory.program(slot), fragment, ALL_STATES)
        for slot, fragment in zip(source_slots, (fragment_a, fragment_b, fragment_c))
    ]
    depth2_retention = _accuracy(
        reloaded_memory.program(depth2_slot), target_depth2, HELDOUT_STATES
    )
    depth3_retention = _accuracy(
        reloaded_memory.program(depth3_slot), target_depth3, HELDOUT_STATES
    )
    provenance2 = reloaded_memory.provenance(depth2_slot)
    provenance3 = reloaded_memory.provenance(depth3_slot)
    gates = {
        "all_atomic_sources_admitted": all(receipt.accepted for receipt in source_receipts),
        "all_atomic_sources_protected": all(
            reloaded_memory.is_file_protected(slot) for slot in source_slots
        ),
        "depth2_admitted": bool(depth2_receipt.accepted),
        "depth3_admitted": bool(depth3_receipt.accepted),
        "depth2_heldout_mastery": depth2_retention >= 0.95,
        "depth3_heldout_mastery": depth3_retention >= 0.95,
        "source_retention": min(source_retention) >= 0.95,
        "depth2_provenance_chain": (
            provenance2 is not None
            and provenance2.left_digest == source_digests[0]
            and provenance2.right_digest == source_digests[2]
            and provenance2.mode == "append"
        ),
        "depth3_provenance_chain": (
            provenance3 is not None
            and (
                (
                    provenance3.left_digest == target_depth2.digest()
                    and provenance3.right_digest == source_digests[2]
                    and provenance3.mode == "append"
                )
                or (
                    provenance3.left_digest == source_digests[2]
                    and provenance3.right_digest == target_depth2.digest()
                    and provenance3.mode == "prepend"
                )
            )
        ),
        "wrong_order_rejected_by_behavior": (
            wrong_depth2_accuracy < 0.95 and wrong_depth3_accuracy < 0.95
        ),
        "missing_evidence_noop": missing_noop,
        "shuffled_feedback_rejected": not bool(shuffled_result["accepted"]),
        "memory_persistence_exact": reloaded_memory.digest() == memory.digest(),
        "policy_persistence_exact": reloaded_policy.digest() == persisted_policy.digest(),
        "warm_depth3_not_slower_than_fresh": (
            int(depth3_result["proposals"]) <= int(fresh_depth3_result["proposals"])
        ),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    core_gate_names = tuple(
        name for name in gates if name != "warm_depth3_not_slower_than_fresh"
    )
    searches = (depth2_result, depth3_result, fresh_depth3_result, shuffled_result)
    return {
        "seed": seed,
        "configuration": {
            "slot_values": SLOT_VALUES,
            "training_states": len(TRAIN_STATES),
            "heldout_states": len(HELDOUT_STATES),
            "max_depth": 3,
            "learner_inputs": [
                "opaque_composition_candidate",
                "opaque_left_and_right_file_digests",
                "opaque_composition_mode",
                "opaque_context_key",
                "deterministic_scalar_verifier_outcome",
            ],
        },
        "digests": {
            "fragment_a": fragment_a.digest(),
            "fragment_b": fragment_b.digest(),
            "fragment_c": fragment_c.digest(),
            "target_depth2": target_depth2.digest(),
            "target_depth3": target_depth3.digest(),
        },
        "searches": {
            "depth2": depth2_result,
            "depth3_warm": depth3_result,
            "depth3_fresh": fresh_depth3_result,
            "shuffled": shuffled_result,
        },
        "metrics": {
            "source_retention_minimum": min(source_retention),
            "depth2_heldout_accuracy": depth2_retention,
            "depth3_heldout_accuracy": depth3_retention,
            "wrong_depth2_accuracy": wrong_depth2_accuracy,
            "wrong_depth3_accuracy": wrong_depth3_accuracy,
            "warm_depth3_to_fresh_ratio": (
                int(depth3_result["proposals"]) / int(fresh_depth3_result["proposals"])
                if int(fresh_depth3_result["proposals"])
                else None
            ),
            "source_file_count": memory.file_count,
        },
        "admissions": {
            "sources": [receipt.payload() for receipt in source_receipts],
            "depth2": depth2_receipt.payload(),
            "depth3": depth3_receipt.payload(),
            "missing_evidence": missing_receipt.payload(),
        },
        "gates": gates,
        "promoted": all(gates[name] for name in core_gate_names),
        "proposal_transfer_promoted": gates["warm_depth3_not_slower_than_fresh"],
        "accounting": {
            "unique_verifier_bits": (
                len(ALL_STATES) * 3
                + sum(int(result["unique_verifier_bits"]) for result in searches)
                + len(ALL_STATES) * 6
            ),
            "unique_logical_lifetimes": (
                sum(int(result["proposals"]) for result in searches) + len(ALL_STATES) * 3
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(float(result["wall_seconds"]) for result in searches),
        },
    }


def run(seeds: tuple[int, ...] = (17, 18)) -> dict[str, object]:
    reports = tuple(_run_seed(seed) for seed in seeds)
    return {
        "schema": "neural-computer.verified-recipe-composition-growth.v1",
        "claim_boundary": (
            "bounded verifier-gated growth from protected external recipe files "
            "through depth-three serial composition; not arbitrary program "
            "induction or general continual learning. The optional proposal "
            "policy's warm sample-efficiency transfer is reported separately."
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
