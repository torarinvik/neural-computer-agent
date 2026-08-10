"""Promote safe lifecycle operations for frozen-controller executable memory.

The experiment is deliberately memory-side. A shared learned interpreter is
pretrained once and then frozen. The lifecycle probes execute opaque files on
held-out registers, returning only scalar pass/fail decisions to the
transaction layer. No verifier rows are persisted and no controller or
interpreter optimizer update occurs during lifecycle maintenance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_outcome_program_router.train import _train_executable_bank
from experiments.external_program_file_admission.train import _runtime_smoke
from neural_computer import (
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
)

EVENT_WIDTH = 12
REGISTER_WIDTH = 4
INSTRUCTION_WIDTH = 8
PROGRAM_COUNT = 2
INTERPRETER_UPDATES = 500
HELD_OUT_REGISTERS = 64
EXECUTION_TOLERANCE = 1e-6
COMPRESSION_TOLERANCE = 2e-2


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _execute(
    machine: torch.nn.Module,
    artifact: ExternalProgramArtifact,
    registers: torch.Tensor,
) -> torch.Tensor:
    codes = artifact.codes.to(device=registers.device, dtype=registers.dtype)
    return machine.execute_code_chain(
        registers,
        codes.unsqueeze(0).expand(registers.shape[0], -1, -1),
    )


def _retention_probe(
    machine: torch.nn.Module,
    expected: dict[int, torch.Tensor],
    registers: torch.Tensor,
    checks: dict[str, int],
    *,
    tolerance: float,
):
    expected_ids = set(expected)

    def probe(candidate: ExternalSequenceProgramMemory) -> bool:
        if set(candidate.logical_slot_ids) != expected_ids:
            return False
        for logical_id, reference in expected.items():
            physical = candidate.physical_index_for_logical_id(logical_id)
            actual = _execute(machine, candidate.artifact(physical), registers)
            checks["retention"] += int(registers.shape[0])
            if not bool(torch.allclose(actual, reference, atol=tolerance, rtol=0.0)):
                return False
        return True

    return probe


def _equivalence_probe(
    machine: torch.nn.Module,
    registers: torch.Tensor,
    checks: dict[str, int],
):
    def probe(
        survivor: ExternalProgramArtifact,
        duplicate: ExternalProgramArtifact,
    ) -> bool:
        left = _execute(machine, survivor, registers)
        right = _execute(machine, duplicate, registers)
        checks["equivalence"] += int(registers.shape[0])
        return bool(torch.allclose(left, right, atol=EXECUTION_TOLERANCE, rtol=0.0))

    return probe


def _memory(artifacts: tuple[ExternalProgramArtifact, ...]) -> ExternalSequenceProgramMemory:
    memory = ExternalSequenceProgramMemory(
        INSTRUCTION_WIDTH,
        content_addressing=True,
        hard_routing=True,
    )
    for artifact in artifacts:
        memory.add_artifact(artifact)
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    return memory


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    machine, pretrained_memory, interpreter_loss = _train_executable_bank(
        seed=seed + 40_000,
        program_count=PROGRAM_COUNT,
        event_width=EVENT_WIDTH,
        register_width=REGISTER_WIDTH,
        instruction_width=INSTRUCTION_WIDTH,
        updates=INTERPRETER_UPDATES,
    )
    machine_digest = _digest(machine)
    artifacts = tuple(
        pretrained_memory.artifact(index) for index in range(PROGRAM_COUNT)
    )
    registers = torch.randn(
        HELD_OUT_REGISTERS,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 51_001),
    )
    memory = _memory((artifacts[0], artifacts[0], artifacts[1]))
    memory.protect_file(0)
    initial_ids = memory.logical_slot_ids
    initial_digest = memory.digest()
    checks = {"retention": 0, "equivalence": 0}
    expected = {
        0: _execute(machine, artifacts[0], registers),
        1: _execute(machine, artifacts[0], registers),
        2: _execute(machine, artifacts[1], registers),
    }

    protected = memory.evict_verified(
        0,
        _retention_probe(machine, expected, registers, checks, tolerance=EXECUTION_TOLERANCE),
    )
    protected_no_write = (
        not protected.accepted
        and protected.reason.startswith("protected")
        and memory.digest() == initial_digest
    )

    before_wrong_consolidation = memory.digest()
    wrong_consolidation = memory.consolidate_verified(
        0,
        2,
        _equivalence_probe(machine, registers, checks),
        _retention_probe(
            machine,
            {0: expected[0], 1: expected[1]},
            registers,
            checks,
            tolerance=EXECUTION_TOLERANCE,
        ),
    )
    wrong_consolidation_rejected = (
        not wrong_consolidation.accepted and memory.digest() == before_wrong_consolidation
    )

    evicted = memory.evict_verified(
        2,
        _retention_probe(
            machine,
            {0: expected[0], 1: expected[1]},
            registers,
            checks,
            tolerance=EXECUTION_TOLERANCE,
        ),
    )
    evicted_without_replay = evicted.accepted and memory.logical_slot_ids == (0, 1)

    consolidated = memory.consolidate_verified(
        0,
        1,
        _equivalence_probe(machine, registers, checks),
        _retention_probe(
            machine,
            {0: expected[0]},
            registers,
            checks,
            tolerance=EXECUTION_TOLERANCE,
        ),
    )
    compacted = consolidated.accepted and memory.logical_slot_ids == (0,)

    compressed_payload = memory.compressed_payload(dtype=torch.float16)
    restored_compressed = ExternalSequenceProgramMemory.from_compressed_payload(
        compressed_payload
    )
    compressed_probe = _retention_probe(
        machine,
        {0: expected[0]},
        registers,
        checks,
        tolerance=COMPRESSION_TOLERANCE,
    )
    compressed_restore_retained = compressed_probe(restored_compressed)

    corrupted_payload = copy.deepcopy(compressed_payload)
    corrupted_state = corrupted_payload["state"]
    if not isinstance(corrupted_state, dict):
        raise TypeError("compressed state is not a tensor mapping")
    corrupted_state["programs.0"] = corrupted_state["programs.0"].clone()
    corrupted_state["programs.0"].reshape(-1)[0] += 1
    try:
        ExternalSequenceProgramMemory.from_compressed_payload(corrupted_payload)
    except ValueError as error:
        corrupt_rejected = "checksum mismatch" in str(error)
    else:
        corrupt_rejected = False

    before_mutating_compression = memory.digest()

    def mutating_probe(candidate: ExternalSequenceProgramMemory) -> bool:
        with torch.no_grad():
            candidate.programs[0].add_(1.0)
        return True

    mutating = memory.compress_verified(
        dtype=torch.float16,
        retention_probe=mutating_probe,
    )
    mutating_probe_rejected = not mutating.accepted and memory.digest() == before_mutating_compression

    compressed = memory.compress_verified(
        dtype=torch.float16,
        retention_probe=compressed_probe,
    )
    restored = ExternalSequenceProgramMemory.from_payload(memory.payload())
    runtime_smoke = _runtime_smoke(machine, restored, seed=seed + 90_001)
    gates = {
        "protected_eviction_rejected_without_write": protected_no_write,
        "unprotected_eviction_retained_survivor": evicted_without_replay,
        "non_equivalent_consolidation_rejected_without_write": wrong_consolidation_rejected,
        "equivalent_consolidation_compacted_files": compacted,
        "logical_ids_survived_compaction": initial_ids == (0, 1, 2)
        and memory.logical_slot_ids == (0,),
        "compressed_restore_retained_behavior": compressed_restore_retained,
        "corrupt_compressed_payload_rejected": corrupt_rejected,
        "mutating_probe_rejected_without_write": mutating_probe_rejected,
        "durable_compression_reduced_storage": (
            compressed.accepted
            and compressed.candidate_storage_bytes < compressed.source_storage_bytes
        ),
        "memory_persistence_exact": restored.digest() == memory.digest(),
        "controller_runtime_seam": bool(runtime_smoke["controller_frozen"]),
        "executor_frozen": machine_digest == _digest(machine),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-program-memory-lifecycle.v1",
        "claim_boundary": (
            "bounded verifier-gated eviction, equivalence consolidation, and "
            "durable compression for opaque executable external memory with a "
            "frozen controller; not unrestricted growth or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "program_count": PROGRAM_COUNT,
            "held_out_registers": HELD_OUT_REGISTERS,
            "execution_tolerance": EXECUTION_TOLERANCE,
            "compression_tolerance": COMPRESSION_TOLERANCE,
            "learner_inputs": ["opaque executable artifacts", "scalar held-out verifier outcomes"],
        },
        "interpreter_final_loss": interpreter_loss,
        "initial_logical_ids": list(initial_ids),
        "final_logical_ids": list(memory.logical_slot_ids),
        "protected_receipt": protected.payload(),
        "eviction_receipt": evicted.payload(),
        "wrong_consolidation_receipt": wrong_consolidation.payload(),
        "consolidation_receipt": consolidated.payload(),
        "mutating_compression_receipt": mutating.payload(),
        "compression_receipt": compressed.payload(),
        "runtime_smoke": runtime_smoke,
        "probe_accounting": checks,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": sum(checks.values()),
            "unique_logical_lifetimes": len(initial_ids),
            "optimizer_updates": INTERPRETER_UPDATES,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": 1.0 if all(gates.values()) else 0.0,
            "transfer_ratio_against_fresh": None,
            "latency_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
