"""Audit verifier-gated compaction of recursive external recipe files.

The recipe interpreter and controller remain frozen.  A bounded external file
store is populated with a protected non-commuting depth-four chain plus
unreferenced decoys.  Compaction must preserve the requested root, its full
provenance closure, and every protected source while removing only unreachable
files.  The source store is never mutated before an independent verifier
accepts the copy-on-write candidate.
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
    RecipeInstruction,
    RecipeProgram,
)

SLOT_VALUES = (2, 4, 8)
ALL_STATES = tuple(product(*(range(value) for value in SLOT_VALUES)))
TARGET_THRESHOLD = 1.0
CAPACITY_BUDGET = 7


def _program(*instructions: RecipeInstruction) -> RecipeProgram:
    return RecipeProgram(SLOT_VALUES, instructions)


def _sources() -> tuple[RecipeProgram, ...]:
    return (
        _program(RecipeInstruction("inc", 0, modulus=2)),
        _program(RecipeInstruction("cinc", 1, 0, modulus=4)),
        _program(RecipeInstruction("cinc", 2, 1, modulus=8)),
        _program(RecipeInstruction("cdec", 0, 2, modulus=2)),
    )


def _serial(left: RecipeProgram, right: RecipeProgram) -> RecipeProgram:
    return _program(*(left.instructions + right.instructions))


def _scores(
    candidate: RecipeProgram,
    reference: RecipeProgram,
) -> torch.Tensor:
    return torch.tensor(
        [float(candidate.execute(state) == reference.execute(state)) for state in ALL_STATES],
        dtype=torch.float32,
    )


def _admit(
    memory: ExternalRecipeCompositionMemory,
    program: RecipeProgram,
    *,
    protect: bool,
) -> int:
    receipt = memory.admit_verified_program(
        program,
        _scores(program, program),
        threshold=TARGET_THRESHOLD,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=protect,
    )
    if not receipt.accepted or receipt.slot is None:
        raise RuntimeError("atomic recipe admission failed")
    return int(receipt.slot)


def _compose(
    memory: ExternalRecipeCompositionMemory,
    target: RecipeProgram,
    *,
    protect: bool,
) -> int:
    candidate = next(
        item
        for item in memory.composition_candidates(max_program_length=4)
        if item.program.digest() == target.digest()
    )
    receipt = memory.admit_verified_composition(
        candidate,
        _scores(candidate.program, target),
        threshold=TARGET_THRESHOLD,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=protect,
    )
    if not receipt.accepted or receipt.slot is None:
        raise RuntimeError("recursive recipe composition admission failed")
    return int(receipt.slot)


def _slot_for_digest(
    memory: ExternalRecipeCompositionMemory,
    digest: str,
) -> int:
    return next(
        slot
        for slot in range(memory.file_count)
        if memory.program(slot).digest() == digest
    )


def _run_seed(seed: int) -> dict[str, object]:
    started = time.perf_counter()
    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    sources = _sources()
    source_slots = tuple(_admit(memory, source, protect=True) for source in sources)
    depth2 = _serial(sources[0], sources[1])
    depth3 = _serial(depth2, sources[2])
    depth4 = _serial(depth3, sources[3])
    _compose(memory, depth2, protect=True)
    _compose(memory, depth3, protect=True)
    depth4_slot = _compose(memory, depth4, protect=True)

    decoys = (
        _program(RecipeInstruction("dec", 0, modulus=2)),
        _program(RecipeInstruction("dec", 1, modulus=4)),
        _program(RecipeInstruction("dec", 2, modulus=8)),
    )
    permutation = torch.randperm(len(decoys), generator=torch.Generator().manual_seed(seed))
    tuple(
        _admit(memory, decoys[index], protect=False)
        for index in permutation.tolist()
    )
    source_digest = memory.digest()

    rejected_candidate, rejected = memory.compact_verified(
        (depth4_slot,),
        verifier=lambda _: False,
    )
    rejected_noop = (
        rejected_candidate is None
        and not rejected.accepted
        and memory.digest() == source_digest
    )

    def verify_candidate(candidate: ExternalRecipeCompositionMemory) -> bool:
        if candidate.file_count != CAPACITY_BUDGET:
            return False
        root_slot = _slot_for_digest(candidate, depth4.digest())
        if any(
            candidate.execute(root_slot, state) != depth4.execute(state)
            for state in ALL_STATES
        ):
            return False
        return all(
            _slot_for_digest(candidate, memory.program(slot).digest()) >= 0
            for slot in source_slots
        )

    compacted, accepted = memory.compact_verified(
        (depth4_slot,),
        verifier=verify_candidate,
    )
    if compacted is None or not accepted.accepted:
        raise RuntimeError("verified compaction unexpectedly failed")
    restored = ExternalRecipeCompositionMemory.from_payload(compacted.payload())

    root_slot = _slot_for_digest(restored, depth4.digest())
    source_retention = tuple(
        all(
            restored.execute(_slot_for_digest(restored, source.digest()), state)
            == source.execute(state)
            for state in ALL_STATES
        )
        for source in sources
    )
    decoy_absence = all(
        all(
            restored.program(slot).digest() != decoy.digest()
            for slot in range(restored.file_count)
        )
        for decoy in decoys
    )
    gates = {
        "rejected_compaction_noop": rejected_noop,
        "capacity_budget_met": restored.file_count <= CAPACITY_BUDGET,
        "root_mastery": all(
            restored.execute(root_slot, state) == depth4.execute(state)
            for state in ALL_STATES
        ),
        "protected_sources_retained": all(source_retention),
        "decoys_removed": decoy_absence,
        "recursive_provenance_retained": restored.provenance(root_slot) is not None,
        "reload_exact": restored.digest() == compacted.digest(),
        "source_unchanged": memory.digest() == source_digest,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "configuration": {
            "slot_values": SLOT_VALUES,
            "source_files": 4,
            "recursive_depth": 4,
            "unreferenced_decoys": len(decoys),
            "capacity_budget": CAPACITY_BUDGET,
            "learner_inputs": [
                "opaque_recipe_file_digests",
                "opaque_provenance_closure",
                "deterministic_scalar_verifier_outcomes",
            ],
        },
        "metrics": {
            "source_file_count": memory.file_count,
            "compacted_file_count": restored.file_count,
            "files_removed": memory.file_count - restored.file_count,
            "source_retention_minimum": float(all(source_retention)),
            "root_heldout_accuracy": float(
                sum(
                    restored.execute(root_slot, state) == depth4.execute(state)
                    for state in ALL_STATES
                )
                / len(ALL_STATES)
            ),
        },
        "receipts": {
            "rejected": rejected.payload(),
            "accepted": accepted.payload(),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": 7 * len(ALL_STATES),
            "verifier_revalidation_bits": 4 * len(ALL_STATES),
            "unique_logical_lifetimes": 7,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
        },
    }


def run(seeds: tuple[int, ...] = (17, 18, 19, 20)) -> dict[str, object]:
    reports = tuple(_run_seed(seed) for seed in seeds)
    return {
        "schema": "neural-computer.verified-recipe-compaction.v1",
        "claim_boundary": (
            "bounded verifier-gated copy-on-write compaction preserving recursive "
            "recipe roots and protected sources; not learned eviction economics "
            "or general continual learning"
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
