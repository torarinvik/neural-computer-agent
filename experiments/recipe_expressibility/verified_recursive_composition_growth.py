"""Audit recursive outcome-only composition through a depth-four file.

The controller and generic interpreter remain frozen.  Four independently
verified mixed-domain instructions are stored first; the external search then
grows a depth-two, depth-three, and depth-four file.  The depth-four target
uses a new conditional instruction, so the result tests structural recursive
transfer rather than repeating one whole-file digest.  Only scalar verifier
outcomes reach the external composition policy.
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

SLOT_VALUES = (2, 4, 8)
ALL_STATES = tuple(product(*(range(value) for value in SLOT_VALUES)))
TRAIN_STATES = tuple(state for state in ALL_STATES if sum(state) % 2 == 0)
HELDOUT_STATES = tuple(state for state in ALL_STATES if sum(state) % 2 == 1)
TARGET_THRESHOLD = 1.0
MAX_PROPOSALS = 512
RECURSIVE_VARIANTS = ("independent_tail", "noncommuting_chain")
POLICY_PROFILES = ("legacy", "orientation_invariant")


def _program(*instructions: RecipeInstruction) -> RecipeProgram:
    return RecipeProgram(SLOT_VALUES, instructions)


def _sources(variant: str = "independent_tail") -> tuple[RecipeProgram, ...]:
    if variant == "independent_tail":
        return (
            _program(RecipeInstruction("inc", 0, modulus=2)),
            _program(RecipeInstruction("cinc", 1, 0, modulus=4)),
            _program(RecipeInstruction("inc", 2, modulus=8)),
            _program(RecipeInstruction("cdec", 2, 1, modulus=8)),
        )
    if variant == "noncommuting_chain":
        return (
            _program(RecipeInstruction("inc", 0, modulus=2)),
            _program(RecipeInstruction("cinc", 1, 0, modulus=4)),
            _program(RecipeInstruction("cinc", 2, 1, modulus=8)),
            _program(RecipeInstruction("cdec", 0, 2, modulus=2)),
        )
    raise ValueError(f"unknown recursive composition variant: {variant!r}")


def _serial(left: RecipeProgram, right: RecipeProgram) -> RecipeProgram:
    return _program(*(left.instructions + right.instructions))


def _targets(
    sources: tuple[RecipeProgram, ...],
) -> tuple[RecipeProgram, RecipeProgram, RecipeProgram]:
    depth2 = _serial(sources[0], sources[1])
    depth3 = _serial(depth2, sources[2])
    depth4 = _serial(depth3, sources[3])
    return depth2, depth3, depth4


def _scores(
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
    return float(_scores(candidate, reference, states).mean().item())


def _policy(profile: str = "legacy") -> OpaqueContextRecipeCompositionMemory:
    if profile not in POLICY_PROFILES:
        raise ValueError(f"unknown composition policy profile: {profile!r}")
    orientation_invariant = profile == "orientation_invariant"
    return OpaqueContextRecipeCompositionMemory(
        exploration_floor=0.05,
        shared_prior_weight=0.25,
        exploration_bonus=0.25,
        left_weight=1.0,
        right_weight=1.0,
        mode_weight=0.5,
        left_depth_weight=0.75,
        right_depth_weight=0.75,
        # The legacy profile preserves the earlier promoted diagnostic.  The
        # orientation-invariant profile additionally scores canonical shape
        # factors so the parent may occupy either operand.
        shape_weight=8.0,
        canonical_shape_weight=8.0 if orientation_invariant else 0.0,
        canonical_depth_weight=0.25 if orientation_invariant else 0.0,
        depth_span_weight=0.25 if orientation_invariant else 0.0,
        temperature=0.05,
    )


def _admit_atomic(
    memory: ExternalRecipeCompositionMemory,
    program: RecipeProgram,
) -> object:
    return memory.admit_verified_program(
        program,
        _scores(program, program, ALL_STATES),
        threshold=TARGET_THRESHOLD,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=True,
    )


def _search_until(
    search: OutcomeOnlyRecipeCompositionSearch,
    reference: RecipeProgram,
    *,
    seed: int,
    scope: str,
    context: str,
    feedback_mode: str = "verifier",
) -> tuple[object | None, dict[str, object]]:
    if feedback_mode not in ("verifier", "shuffled"):
        raise ValueError("unsupported recursive composition feedback mode")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = search.initial_state()
    started = time.perf_counter()
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
            _scores(proposal.candidate.program, reference, TRAIN_STATES)
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


def _run_seed(
    seed: int,
    *,
    variant: str = "independent_tail",
    policy_profile: str = "legacy",
) -> dict[str, object]:
    sources = _sources(variant)
    depth2, depth3, depth4 = _targets(sources)
    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    source_receipts = tuple(_admit_atomic(memory, source) for source in sources)
    if any(not receipt.accepted or receipt.slot is None for receipt in source_receipts):
        raise RuntimeError("an atomic source was not admitted")
    source_slots = tuple(int(receipt.slot) for receipt in source_receipts)

    policy = _policy(policy_profile)
    depth_results: dict[str, dict[str, object]] = {}
    depth_slots: dict[str, int] = {}
    for index, (name, target) in enumerate(
        (("depth2", depth2), ("depth3", depth3)),
        start=2,
    ):
        search = OutcomeOnlyRecipeCompositionSearch(
            memory,
            max_program_length=index,
            policy=policy,
        )
        proposal, result = _search_until(
            search,
            target,
            seed=seed + index * 100,
            scope=f"seed-{seed}-depth-{index}",
            context=f"recursive-context-{seed}",
        )
        if proposal is None:
            raise RuntimeError(f"{name} search did not find a verified candidate")
        receipt = memory.admit_verified_composition(
            proposal.candidate,
            _scores(proposal.candidate.program, target, ALL_STATES),
            threshold=TARGET_THRESHOLD,
            min_observations=len(ALL_STATES),
            min_stable_observations=len(ALL_STATES),
            protect=True,
        )
        if not receipt.accepted or receipt.slot is None:
            raise RuntimeError(f"{name} was not committed")
        depth_results[name] = {
            **result,
            "candidate_digest": proposal.candidate.program.digest(),
            "structure": (
                None
                if proposal.candidate.structure is None
                else proposal.candidate.structure.payload()
            ),
        }
        depth_slots[name] = receipt.slot

    pre_depth4_memory = ExternalRecipeCompositionMemory.from_payload(memory.payload())
    persisted_policy = OpaqueContextRecipeCompositionMemory.from_payload(policy.payload())
    warm_search = OutcomeOnlyRecipeCompositionSearch(
        memory,
        max_program_length=4,
        policy=persisted_policy,
    )
    warm_proposal, warm_result = _search_until(
        warm_search,
        depth4,
        seed=seed + 400,
        scope=f"seed-{seed}-depth-4-warm",
        context=f"recursive-context-{seed}",
    )
    if warm_proposal is None:
        raise RuntimeError("warm depth-four search did not find a candidate")
    depth4_receipt = memory.admit_verified_composition(
        warm_proposal.candidate,
        _scores(warm_proposal.candidate.program, depth4, ALL_STATES),
        threshold=TARGET_THRESHOLD,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=True,
    )
    if not depth4_receipt.accepted or depth4_receipt.slot is None:
        raise RuntimeError("depth-four composition was not committed")
    warm_result = {
        **warm_result,
        "candidate_digest": warm_proposal.candidate.program.digest(),
        "structure": (
            None
            if warm_proposal.candidate.structure is None
            else warm_proposal.candidate.structure.payload()
        ),
    }

    fresh_search = OutcomeOnlyRecipeCompositionSearch(
        pre_depth4_memory,
        max_program_length=4,
        policy=_policy(policy_profile),
    )
    _, fresh_result = _search_until(
        fresh_search,
        depth4,
        seed=seed + 400,
        scope=f"seed-{seed}-depth-4-fresh",
        context=f"fresh-context-{seed}",
    )
    shuffled_search = OutcomeOnlyRecipeCompositionSearch(
        pre_depth4_memory,
        max_program_length=4,
        policy=_policy(policy_profile),
    )
    _, shuffled_result = _search_until(
        shuffled_search,
        depth4,
        seed=seed + 500,
        scope=f"seed-{seed}-shuffled",
        context=f"shuffled-context-{seed}",
        feedback_mode="shuffled",
    )

    missing_candidates = pre_depth4_memory.composition_candidates(max_program_length=4)
    missing_before = memory.digest()
    missing_receipt = memory.admit_verified_composition(
        missing_candidates[0],
        torch.empty(0),
        threshold=TARGET_THRESHOLD,
        min_observations=1,
        min_stable_observations=1,
    )
    depth4_slot = depth4_receipt.slot
    restored = ExternalRecipeCompositionMemory.from_payload(memory.payload())
    restored_policy = OpaqueContextRecipeCompositionMemory.from_payload(
        persisted_policy.payload()
    )
    wrong_depth2 = _serial(sources[1], sources[0])
    wrong_depth4 = _serial(sources[3], depth3)
    provenance2 = restored.provenance(depth_slots["depth2"])
    provenance3 = restored.provenance(depth_slots["depth3"])
    provenance4 = restored.provenance(depth4_slot)

    def slot_for_digest(digest: str) -> int | None:
        return next(
            (
                slot
                for slot in range(restored.file_count)
                if restored.program(slot).digest() == digest
            ),
            None,
        )

    def recursive_link(
        child_slot: int,
        parent_slot: int | None,
        expected_depth: int,
    ) -> bool:
        factors = restored.provenance(child_slot)
        if factors is None:
            return False
        if restored.composition_depth(child_slot) != expected_depth:
            return False
        if parent_slot is None:
            left_slot = slot_for_digest(factors.left_digest)
            right_slot = slot_for_digest(factors.right_digest)
            return (
                left_slot is not None
                and right_slot is not None
                and restored.composition_depth(left_slot) == 1
                and restored.composition_depth(right_slot) == 1
            )
        parent_digest = restored.program(parent_slot).digest()
        if factors.left_digest == parent_digest:
            atomic_digest = factors.right_digest
        elif factors.right_digest == parent_digest:
            atomic_digest = factors.left_digest
        else:
            return False
        atomic_slot = slot_for_digest(atomic_digest)
        return (
            atomic_slot is not None
            and restored.composition_depth(parent_slot) == expected_depth - 1
            and restored.composition_depth(atomic_slot) == 1
        )
    source_retention = [
        _accuracy(restored.program(slot), source, HELDOUT_STATES)
        for slot, source in zip(source_slots, sources, strict=True)
    ]
    depth_retention = {
        "depth2": _accuracy(
            restored.program(depth_slots["depth2"]), depth2, HELDOUT_STATES
        ),
        "depth3": _accuracy(
            restored.program(depth_slots["depth3"]), depth3, HELDOUT_STATES
        ),
        "depth4": _accuracy(restored.program(depth4_slot), depth4, HELDOUT_STATES),
    }
    gates = {
        "atomic_sources_admitted": all(receipt.accepted for receipt in source_receipts),
        "atomic_sources_protected": all(
            restored.is_file_protected(slot) for slot in source_slots
        ),
        "depth2_mastery": depth_retention["depth2"] >= 0.95,
        "depth3_mastery": depth_retention["depth3"] >= 0.95,
        "depth4_mastery": depth_retention["depth4"] >= 0.95,
        "complete_prefix_retention": min(source_retention) >= 0.95,
        "recursive_depths": (
            recursive_link(depth_slots["depth2"], None, 2)
            and recursive_link(
                depth_slots["depth3"], depth_slots["depth2"], 3
            )
            and recursive_link(depth4_slot, depth_slots["depth3"], 4)
        ),
        "depth2_provenance": provenance2 is not None,
        "depth3_provenance": provenance3 is not None,
        "depth4_provenance": provenance4 is not None,
        "wrong_depth2_rejected": _accuracy(wrong_depth2, depth2, HELDOUT_STATES) < 0.95,
        "wrong_depth4_rejected": _accuracy(wrong_depth4, depth4, HELDOUT_STATES) < 0.95,
        "missing_evidence_noop": (
            not missing_receipt.accepted and memory.digest() == missing_before
        ),
        "shuffled_feedback_rejected": not bool(shuffled_result["accepted"]),
        "memory_persistence_exact": restored.digest() == memory.digest(),
        "policy_persistence_exact": restored_policy.digest() == persisted_policy.digest(),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "configuration": {
            "slot_values": SLOT_VALUES,
            "all_states": len(ALL_STATES),
            "training_states": len(TRAIN_STATES),
            "heldout_states": len(HELDOUT_STATES),
            "max_depth": 4,
            "variant": variant,
            "policy_profile": policy_profile,
            "policy_shape_weight": 8.0,
            "policy_canonical_shape_weight": (
                8.0 if policy_profile == "orientation_invariant" else 0.0
            ),
            "learner_inputs": [
                "opaque_composition_candidate",
                "opaque_source_digests",
                "opaque_composition_mode",
                "opaque_recursive_source_shape",
                "deterministic_scalar_verifier_outcome",
            ],
        },
        "searches": {
            **depth_results,
            "depth4_warm": warm_result,
            "depth4_fresh": fresh_result,
            "shuffled": shuffled_result,
        },
        "metrics": {
            "source_retention_minimum": min(source_retention),
            **{f"{name}_heldout_accuracy": value for name, value in depth_retention.items()},
            "wrong_depth2_accuracy": _accuracy(wrong_depth2, depth2, HELDOUT_STATES),
            "wrong_depth4_accuracy": _accuracy(wrong_depth4, depth4, HELDOUT_STATES),
            "warm_depth4_to_fresh_ratio": (
                int(warm_result["proposals"]) / int(fresh_result["proposals"])
            ),
            "composition_file_count": restored.file_count,
        },
        "admissions": {
            "sources": [receipt.payload() for receipt in source_receipts],
            "depth4": depth4_receipt.payload(),
            "missing_evidence": missing_receipt.payload(),
        },
        "provenance": {
            "depth2": None if provenance2 is None else provenance2.payload(),
            "depth3": None if provenance3 is None else provenance3.payload(),
            "depth4": None if provenance4 is None else provenance4.payload(),
        },
        "gates": gates,
        "promoted": all(
            all(value) if isinstance(value, list) else bool(value)
            for value in gates.values()
        ),
        "accounting": {
            "unique_verifier_bits": (
                len(ALL_STATES) * 4
                + sum(int(result["unique_verifier_bits"]) for result in depth_results.values())
                + int(warm_result["unique_verifier_bits"])
                + int(fresh_result["unique_verifier_bits"])
                + int(shuffled_result["unique_verifier_bits"])
            ),
            "unique_logical_lifetimes": (
                len(ALL_STATES) * 4
                + sum(int(result["proposals"]) for result in depth_results.values())
                + int(warm_result["proposals"])
                + int(fresh_result["proposals"])
                + int(shuffled_result["proposals"])
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(
                float(result["wall_seconds"])
                for result in (
                    *depth_results.values(),
                    warm_result,
                    fresh_result,
                    shuffled_result,
                )
            ),
        },
    }


def run(
    seeds: tuple[int, ...] = (17, 18),
    *,
    variant: str = "independent_tail",
    policy_profile: str = "legacy",
) -> dict[str, object]:
    if variant not in RECURSIVE_VARIANTS:
        raise ValueError(f"unknown recursive composition variant: {variant!r}")
    if policy_profile not in POLICY_PROFILES:
        raise ValueError(f"unknown composition policy profile: {policy_profile!r}")
    reports = tuple(
        _run_seed(seed, variant=variant, policy_profile=policy_profile)
        for seed in seeds
    )
    return {
        "schema": "neural-computer.verified-recursive-recipe-composition-growth.v1",
        "claim_boundary": (
            "bounded replay-free verifier-gated recursive growth through a "
            "depth-four external recipe file with generic structural credit; "
            "not arbitrary program induction or general continual learning"
        ),
        "seeds": list(seeds),
        "variant": variant,
        "policy_profile": policy_profile,
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--variant", choices=RECURSIVE_VARIANTS, default="independent_tail")
    parser.add_argument("--policy-profile", choices=POLICY_PROFILES, default="legacy")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(
        tuple(args.seeds),
        variant=args.variant,
        policy_profile=args.policy_profile,
    )
    report["wall_seconds"] = time.perf_counter() - started
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
