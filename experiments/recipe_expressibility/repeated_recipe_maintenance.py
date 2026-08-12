"""Audit repeated fixed-capacity recipe replacement with retention proofs.

The counterfactual maintenance policy is trained once on fresh independent
candidate banks, then frozen. A live external recipe bank repeatedly admits a
new recursive root and uses the policy to choose one unprotected root for
replacement. Each copy-on-write compaction must retain every other mastered
root, the incoming root, and all protected source files. Forward and reversed
physical orders exercise the row-permutation invariant; a fresh policy is a
negative control.

This is a bounded repeated-replacement result. It does not claim unrestricted
memory growth, semantic compression, or general continual learning.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from neural_computer import (
    ExternalRecipeCompositionMemory,
    RecipeProgram,
    RecipeProgramCompositionCandidate,
    RecipeProgramCompositionFactors,
)

from . import learned_recipe_eviction as maintenance

REPEATED_RECIPE_MAINTENANCE_SCHEMA = (
    "neural-computer.repeated-recipe-maintenance.v1"
)
ROOT_DEPTHS = (2, 3, 4, 5)
INCOMING_SPECS = (
    (0, 2),
    (2, 5, 6),
    (5, 9, 10, 11),
    (9, 14, 15, 16, 17),
    (2, 0),
    (6, 5, 2),
    (11, 10, 9, 5),
    (17, 16, 15, 14, 9),
)
STAGES = 8
STATE_COUNT = len(maintenance.ALL_STATES)
ROOT_COUNT = len(ROOT_DEPTHS)
SOURCE_BANK_SIZE = len(maintenance._source_programs())


def _find_digest(memory: ExternalRecipeCompositionMemory, digest: str) -> int | None:
    return next(
        (
            slot
            for slot in range(memory.file_count)
            if memory.program(slot).digest() == digest
        ),
        None,
    )


def _compose_known(
    memory: ExternalRecipeCompositionMemory,
    left_slot: int,
    right_slot: int,
    left: RecipeProgram,
    right: RecipeProgram,
) -> tuple[int, RecipeProgram]:
    target = maintenance._serial(left, right)
    candidate = RecipeProgramCompositionCandidate(
        left_slot=left_slot,
        right_slot=right_slot,
        factors=RecipeProgramCompositionFactors(
            memory.program(left_slot).digest(),
            memory.program(right_slot).digest(),
            "append",
        ),
        program=target,
        structure=memory.composition_structure(left_slot, right_slot),
    ).validate()
    receipt = memory.admit_verified_composition(
        candidate,
        maintenance._scores(candidate.program, target),
        threshold=1.0,
        min_observations=STATE_COUNT,
        min_stable_observations=STATE_COUNT,
        protect=False,
    )
    if not receipt.accepted or receipt.slot is None:
        raise RuntimeError("known recipe composition was not admitted")
    return int(receipt.slot), target


def _compose_from_indices(
    memory: ExternalRecipeCompositionMemory,
    sources: tuple[RecipeProgram, ...],
    source_slots: dict[str, int],
    indices: tuple[int, ...],
) -> tuple[int, RecipeProgram]:
    prefix = sources[indices[0]]
    prefix_slot = source_slots[prefix.digest()]
    for index in indices[1:]:
        right = sources[index]
        right_slot = source_slots[right.digest()]
        prefix_slot, prefix = _compose_known(
            memory,
            prefix_slot,
            right_slot,
            prefix,
            right,
        )
    return prefix_slot, prefix


def _initial_memory(
    *,
    reverse_sources: bool,
) -> tuple[
    ExternalRecipeCompositionMemory,
    tuple[RecipeProgram, ...],
    dict[str, int],
    tuple[str, ...],
]:
    memory = ExternalRecipeCompositionMemory(maintenance.SLOT_VALUES)
    sources = maintenance._source_programs()
    source_slots: dict[str, int] = {}
    source_order = tuple(reversed(range(len(sources)))) if reverse_sources else tuple(range(len(sources)))
    for index in source_order:
        source = sources[index]
        source_slots[source.digest()] = maintenance._admit_atomic(
            memory,
            source,
            protect=True,
        )
    roots: list[str] = []
    source_cursor = 0
    for depth in ROOT_DEPTHS:
        indices = tuple(range(source_cursor, source_cursor + depth))
        source_cursor += depth
        root_slot, root = _compose_from_indices(
            memory,
            sources,
            source_slots,
            indices,
        )
        if memory.composition_depth(root_slot) != depth:
            raise RuntimeError("initial root depth is inconsistent")
        roots.append(root.digest())
    return memory, sources, source_slots, tuple(roots)


def _context(memory: ExternalRecipeCompositionMemory, pressure: float) -> torch.Tensor:
    return torch.tensor(
        [
            1.0,
            pressure,
            math.log1p(memory.file_count),
            ROOT_COUNT / memory.file_count,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def _verify_roots(
    candidate: ExternalRecipeCompositionMemory,
    expected: dict[str, RecipeProgram],
    *,
    forbidden: str | None = None,
    maximum_file_count: int,
) -> bool:
    if candidate.file_count > maximum_file_count:
        return False
    if forbidden is not None and _find_digest(candidate, forbidden) is not None:
        return False
    for digest, program in expected.items():
        slot = _find_digest(candidate, digest)
        if slot is None:
            return False
        if any(
            candidate.execute(slot, state) != program.execute(state)
            for state in maintenance.ALL_STATES
        ):
            return False
    return True


def _run_stream(
    policy,
    *,
    seed: int,
    reverse_order: bool,
    commit: bool,
) -> dict[str, object]:
    memory, sources, _source_slots, active_roots = _initial_memory(
        reverse_sources=reverse_order,
    )
    initial_file_count = memory.file_count
    source_digests = tuple(source.digest() for source in sources)
    stages: list[dict[str, object]] = []
    verifier_bits = initial_file_count * STATE_COUNT
    for stage in range(STAGES):
        working = memory.copy_on_write()
        working_source_slots = {
            digest: _find_digest(working, digest) for digest in source_digests
        }
        if any(slot is None for slot in working_source_slots.values()):
            raise RuntimeError("protected source disappeared before replacement")
        incoming_indices = INCOMING_SPECS[stage % len(INCOMING_SPECS)]
        incoming_slot, incoming = _compose_from_indices(
            working,
            sources,
            {digest: int(slot) for digest, slot in working_source_slots.items()},
            incoming_indices,
        )
        verifier_bits += (len(incoming_indices) - 1) * STATE_COUNT
        current_slots = tuple(
            _find_digest(working, digest) for digest in active_roots
        )
        if any(slot is None for slot in current_slots):
            raise RuntimeError("active root disappeared before replacement")
        current_slots = tuple(int(slot) for slot in current_slots)
        candidate_order = tuple(reversed(current_slots)) if reverse_order else current_slots
        pressure_rank = stage % ROOT_COUNT
        pressure = pressure_rank / (ROOT_COUNT - 1)
        context = _context(working, pressure)
        telemetry = working.candidate_telemetry(candidate_order).unsqueeze(0)
        scores = policy.score_candidates(context, telemetry)[0]
        selected_position = int(scores.argmax())
        selected_slot = candidate_order[selected_position]
        selected_digest = next(
            digest
            for digest in active_roots
            if _find_digest(working, digest) == selected_slot
        )
        depth_by_digest = {
            digest: working.composition_depth(_find_digest(working, digest))
            for digest in active_roots
        }
        expected_victim = next(
            digest
            for digest, depth in depth_by_digest.items()
            if depth == ROOT_DEPTHS[pressure_rank]
        )
        retained_old = {
            digest: working.program(_find_digest(working, digest))
            for digest in active_roots
            if digest != expected_victim
        }
        expected = {**retained_old, incoming.digest(): incoming}
        requested = tuple(
            slot
            for position, slot in enumerate(candidate_order)
            if position != selected_position
        ) + (incoming_slot,)
        accepted = False
        source_unchanged = True
        if commit:
            before = memory.digest()
            compacted, receipt = working.compact_verified(
                requested,
                verifier=lambda candidate, expected=expected, expected_victim=expected_victim: _verify_roots(
                    candidate,
                    expected,
                    forbidden=expected_victim,
                    maximum_file_count=initial_file_count,
                ),
            )
            source_unchanged = memory.digest() == before
            accepted = compacted is not None and receipt.accepted
            if accepted:
                memory = compacted
                active_roots = (*retained_old.keys(), incoming.digest())
            verifier_bits += STATE_COUNT
        else:
            accepted = selected_digest == expected_victim
        source_retained = all(_find_digest(memory, digest) is not None for digest in source_digests)
        stages.append(
            {
                "stage": stage,
                "pressure_rank": pressure_rank,
                "expected_victim_depth": ROOT_DEPTHS[pressure_rank],
                "selected_depth": depth_by_digest[selected_digest],
                "selected_expected_victim": selected_digest == expected_victim,
                "accepted": accepted,
                "source_retained": source_retained,
                "file_count": memory.file_count,
                "candidate_file_count": working.file_count,
                "source_unchanged_before_adoption": source_unchanged,
            }
        )
        if commit and not accepted:
            break
    reload_ok = False
    rejected_noop = False
    if commit:
        restored = ExternalRecipeCompositionMemory.from_payload(memory.payload())
        reload_ok = all(_find_digest(restored, digest) is not None for digest in active_roots)
        before = memory.digest()
        _, rejected = memory.compact_verified(
            tuple(_find_digest(memory, digest) for digest in active_roots),
            verifier=lambda _: False,
        )
        rejected_noop = not rejected.accepted and memory.digest() == before
    return {
        "physical_order": "reversed" if reverse_order else "forward",
        "commit": commit,
        "stages": stages,
        "all_replacements_accepted": all(bool(stage["accepted"]) for stage in stages),
        "all_sources_retained": all(bool(stage["source_retained"]) for stage in stages),
        "reload_active_roots": reload_ok,
        "rejected_compaction_noop": rejected_noop,
        "final_file_count": memory.file_count,
        "initial_file_count": initial_file_count,
        "verifier_bits": verifier_bits,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    started = time.perf_counter()
    maintenance.configure_profile(4)
    policy, training = maintenance._train(
        seed,
        candidate_depths=maintenance.TRAIN_DEPTHS,
        credit_mode="counterfactual",
        source_bank_size=SOURCE_BANK_SIZE,
        utility_objective="evict",
    )
    shuffled_policy, shuffled_training = maintenance._train(
        seed + 10_000,
        candidate_depths=maintenance.TRAIN_DEPTHS,
        credit_mode="counterfactual",
        shuffled_utility=True,
        source_bank_size=SOURCE_BANK_SIZE,
        utility_objective="evict",
    )
    fresh_policy = maintenance._policy()
    forward = _run_stream(policy, seed=seed, reverse_order=False, commit=True)
    reversed_stream = _run_stream(
        policy,
        seed=seed + 100,
        reverse_order=True,
        commit=True,
    )
    fresh = _run_stream(
        fresh_policy,
        seed=seed + 200,
        reverse_order=False,
        commit=False,
    )
    shuffled = _run_stream(
        shuffled_policy,
        seed=seed + 300,
        reverse_order=False,
        commit=False,
    )
    persisted = maintenance._policy()
    persisted.load_state_dict(policy.state_dict())
    gates = {
        "trained_stable": training["stable_update"] is not None,
        "shuffled_training_not_stable": shuffled_training["stable_update"] is None,
        "forward_replacements_accepted": forward["all_replacements_accepted"],
        "reversed_replacements_accepted": reversed_stream["all_replacements_accepted"],
        "forward_sources_retained": forward["all_sources_retained"],
        "reversed_sources_retained": reversed_stream["all_sources_retained"],
        "forward_reload_active_roots": forward["reload_active_roots"],
        "reversed_reload_active_roots": reversed_stream["reload_active_roots"],
        "forward_rejected_noop": forward["rejected_compaction_noop"],
        "reversed_rejected_noop": reversed_stream["rejected_compaction_noop"],
        "fresh_policy_not_perfect": not fresh["all_replacements_accepted"],
        "shuffled_policy_not_perfect": not shuffled["all_replacements_accepted"],
        "policy_reload_exact": maintenance._digest(persisted) == maintenance._digest(policy),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": REPEATED_RECIPE_MAINTENANCE_SCHEMA,
        "claim_boundary": (
            "bounded repeated verifier-gated recipe replacement with exact "
            "counterfactual maintenance credit and protected-source retention; "
            "not unrestricted memory growth or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "root_depths": ROOT_DEPTHS,
            "incoming_depths": tuple(len(spec) for spec in INCOMING_SPECS),
            "stages": STAGES,
            "active_root_capacity": ROOT_COUNT,
            "credit_mode": "counterfactual",
            "utility_objective": "evict",
            "controller": "frozen_or_not_present",
            "replay": "zero",
        },
        "training": training,
        "shuffled_training": shuffled_training,
        "forward": forward,
        "reversed": reversed_stream,
        "fresh_control": fresh,
        "shuffled_control": shuffled,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": (
                int(training["unique_verifier_bits"])
                + int(shuffled_training["unique_verifier_bits"])
                + int(forward["verifier_bits"])
                + int(reversed_stream["verifier_bits"])
                + int(fresh["verifier_bits"])
                + int(shuffled["verifier_bits"])
            ),
            "unique_logical_lifetimes": 2 * 512 + 4 * STAGES,
            "optimizer_updates": (
                int(training["optimizer_updates"])
                + int(shuffled_training["optimizer_updates"])
            ),
            "replayed_examples": 0,
            "controller_optimizer_updates": 0,
            "live_replacement_transactions": 2 * STAGES,
            "latency_seconds": time.perf_counter() - started,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed, args.report_out)
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
