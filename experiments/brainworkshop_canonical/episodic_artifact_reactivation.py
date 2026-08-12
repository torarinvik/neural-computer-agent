"""Pressure-test episodic reactivation of opaque executable capability files.

One shared external-register interpreter is trained once and then frozen.  A
growable episodic index maps learned context/signature keys to opaque artifact
digests, while a bounded hot executable bank is replaced by copy-on-write
retention probes.  Online reactivation performs no controller or interpreter
optimizer update and replays no old examples.

This isolates a useful continual-learning primitive: old capability state can
be found, verified, and made executable again after it leaves the hot cache.
It does not claim unrestricted memory growth or general continual learning.
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
from neural_computer import (
    EpisodicBindingArtifactIndex,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
)

EVENT_WIDTH = 12
REGISTER_WIDTH = 4
INSTRUCTION_WIDTH = 8
ARTIFACT_COUNT = 4
INTERPRETER_UPDATES = 500
HELD_OUT_REGISTERS = 64
EXECUTION_TOLERANCE = 1e-6


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
        program_count=ARTIFACT_COUNT,
        event_width=EVENT_WIDTH,
        register_width=REGISTER_WIDTH,
        instruction_width=INSTRUCTION_WIDTH,
        updates=INTERPRETER_UPDATES,
    )
    machine_digest = _digest(machine)
    artifacts = tuple(
        pretrained_memory.artifact(index) for index in range(ARTIFACT_COUNT)
    )
    by_handle = {artifact.digest(): artifact for artifact in artifacts}
    registers = torch.randn(
        HELD_OUT_REGISTERS,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 51_001),
    )
    expected = {
        artifact.digest(): _execute(machine, artifact, registers)
        for artifact in artifacts
    }

    index = EpisodicBindingArtifactIndex.create(
        context_width=ARTIFACT_COUNT + 1,
        signature_width=ARTIFACT_COUNT + 1,
        active_slots=2,
        matching_threshold=0.99,
        min_mastery_observations=2,
    )
    key_rows = torch.eye(ARTIFACT_COUNT + 1)
    binding_ids = [
        index.register(key_rows[row], key_rows[row], artifact.digest())
        for row, artifact in enumerate(artifacts)
    ]
    missing_binding = index.register(
        key_rows[-1],
        key_rows[-1],
        "missing-external-artifact",
    )
    index.activate(binding_ids[0], 0)
    index.activate(binding_ids[1], 1)
    index.archive.observe(binding_ids[0], 1.0, step=0)
    index.archive.observe(binding_ids[0], 1.0, step=1)

    checks = {"reactivation": 0, "runtime": 0}

    def candidate_memory(candidate: EpisodicBindingArtifactIndex) -> tuple[ExternalProgramArtifact, ...] | None:
        resolved: list[ExternalProgramArtifact] = []
        for binding_id in candidate.active_binding_ids:
            if binding_id is None:
                return None
            handle = candidate.artifact_handle(binding_id)
            if handle not in by_handle:
                return None
            resolved.append(by_handle[handle])
        return tuple(resolved)

    def retention_probe(candidate: EpisodicBindingArtifactIndex) -> bool:
        resolved = candidate_memory(candidate)
        if resolved is None:
            return False
        for binding_id, artifact in zip(candidate.active_binding_ids, resolved, strict=True):
            assert binding_id is not None
            if artifact.digest() != candidate.artifact_handle(binding_id):
                return False
            actual = _execute(machine, artifact, registers)
            checks["reactivation"] += int(registers.shape[0])
            if not bool(torch.allclose(actual, expected[artifact.digest()], atol=EXECUTION_TOLERANCE, rtol=0.0)):
                return False
        return True

    hot_memory = _memory((artifacts[0], artifacts[1]))

    protected_before = index.payload()
    protected = index.reactivate_verified(binding_ids[2], 0, retention_probe)
    protected_rejected_without_write = (
        not protected.accepted and "protected" in protected.reason and index.payload() == protected_before
    )

    failed_before = index.payload()
    failed = index.reactivate_verified(binding_ids[2], 1, lambda _: False)
    failed_without_write = (
        not failed.accepted
        and failed.reason == "held-out retention probe failed"
        and index.payload() == failed_before
    )

    mutating_before = index.payload()

    def mutating_probe(candidate: EpisodicBindingArtifactIndex) -> bool:
        candidate.archive.observe(binding_ids[2], 1.0, step=2)
        return True

    mutating = index.reactivate_verified(binding_ids[2], 1, mutating_probe)
    mutating_rejected_without_write = (
        not mutating.accepted
        and "mutated" in mutating.reason
        and index.payload() == mutating_before
    )

    def reactivate(binding_id: int) -> bool:
        nonlocal hot_memory
        receipt = index.reactivate_verified(binding_id, 1, retention_probe)
        if not receipt.accepted:
            return False
        resolved = candidate_memory(index)
        if resolved is None:
            return False
        hot_memory = _memory(resolved)
        return True

    c_reactivated = reactivate(binding_ids[2])
    d_reactivated = reactivate(binding_ids[3])
    b_revisited = reactivate(binding_ids[1])

    missing_before = index.payload()
    missing = index.reactivate_verified(missing_binding, 1, retention_probe)
    missing_rejected_without_write = (
        not missing.accepted
        and missing.reason == "held-out retention probe failed"
        and index.payload() == missing_before
    )

    active_runtime = True
    for slot, binding_id in enumerate(index.active_binding_ids):
        if binding_id is None:
            active_runtime = False
            break
        artifact = hot_memory.artifact(slot)
        handle = index.artifact_handle(binding_id)
        actual = _execute(machine, artifact, registers)
        checks["runtime"] += int(registers.shape[0])
        active_runtime = active_runtime and handle == artifact.digest()
        active_runtime = active_runtime and bool(
            torch.allclose(actual, expected[handle], atol=EXECUTION_TOLERANCE, rtol=0.0)
        )

    restored_index = EpisodicBindingArtifactIndex.from_payload(index.payload())
    restored_memory = ExternalSequenceProgramMemory.from_payload(hot_memory.payload())
    persistence_exact = (
        restored_index.payload() == index.payload()
        and restored_memory.digest() == hot_memory.digest()
        and restored_index.active_binding_ids == index.active_binding_ids
    )
    corrupted = copy.deepcopy(index.payload())
    corrupted_handles = corrupted["artifact_handles"]
    assert isinstance(corrupted_handles, list)
    corrupted_handles[0] = "tampered-artifact-handle"
    try:
        EpisodicBindingArtifactIndex.from_payload(corrupted)
    except ValueError as error:
        corruption_rejected = "checksum" in str(error)
    else:
        corruption_rejected = False

    gates = {
        "protected_resident_rejected_without_write": protected_rejected_without_write,
        "failed_probe_rejected_without_write": failed_without_write,
        "mutating_probe_rejected_without_write": mutating_rejected_without_write,
        "cold_artifact_reactivated": c_reactivated and d_reactivated,
        "old_artifact_revisited_without_replay": b_revisited,
        "protected_artifact_survived_all_swaps": index.active_binding_ids[0] == binding_ids[0],
        "missing_artifact_rejected_without_write": missing_rejected_without_write,
        "active_hot_bank_executes_expected_files": active_runtime,
        "index_and_executable_memory_round_trip": persistence_exact,
        "corrupted_index_rejected": corruption_rejected,
        "shared_interpreter_frozen": machine_digest == _digest(machine),
        "zero_online_optimizer_updates": True,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.brainworkshop-episodic-artifact-reactivation.v1",
        "claim_boundary": (
            "bounded replay-free reactivation of opaque executable capability files "
            "through a verifier-gated episodic index with a frozen shared interpreter; "
            "not unrestricted memory growth or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "artifact_count": ARTIFACT_COUNT,
            "hot_active_slots": 2,
            "held_out_registers": HELD_OUT_REGISTERS,
            "execution_tolerance": EXECUTION_TOLERANCE,
            "online_inputs": ["learned context/signature keys", "opaque artifact handles", "scalar verifier outcomes"],
        },
        "interpreter_final_loss": interpreter_loss,
        "initial_handles": [artifact.digest() for artifact in artifacts],
        "final_active_binding_ids": list(index.active_binding_ids),
        "final_active_handles": [
            None if binding_id is None else index.artifact_handle(binding_id)
            for binding_id in index.active_binding_ids
        ],
        "receipts": {
            "protected": protected.__dict__,
            "failed": failed.__dict__,
            "mutating": mutating.__dict__,
            "missing": missing.__dict__,
        },
        "probe_accounting": checks,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": sum(checks.values()),
            "unique_logical_lifetimes": ARTIFACT_COUNT + 1,
            "optimizer_updates": INTERPRETER_UPDATES,
            "online_optimizer_updates": 0,
            "replayed_examples": 0,
            "latency_seconds": time.perf_counter() - begun,
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": 1.0 if all(gates.values()) else 0.0,
            "transfer_ratio_against_fresh": None,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=24101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
