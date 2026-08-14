import pytest
import torch

from neural_computer import (
    EXTERNAL_PROGRAM_ADMISSION_SCHEMA,
    EXTERNAL_PROGRAM_ARTIFACT_SCHEMA,
    ExternalMemoryMaintenancePolicy,
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
    evaluate_program_digest_admission,
)


def _artifact() -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.arange(15, dtype=torch.float32).reshape(3, 5),
        interpreter_schema="neural-computer.external-register.v4",
        execution_schema="neural-computer.external-register-read-execute.v1",
        output_schema="neural-computer.external-register-canonical-readout.v1",
    )


def test_admission_distinguishes_aggregate_observations_from_verifier_bits() -> None:
    receipt = evaluate_program_digest_admission(
        "7" * 64,
        (1.0, 1.0, 1.0),
        min_observations=3,
        min_stable_observations=2,
        verifier_bit_counts=(8, 8, 8),
    )

    assert receipt.observations == 3
    assert receipt.stable_observations_to_threshold == 1
    assert receipt.unique_verifier_bits == 24
    assert receipt.stable_bits_to_threshold == 8


def test_external_program_artifact_round_trips_with_stable_digest() -> None:
    artifact = _artifact()

    payload = artifact.payload()
    restored = ExternalProgramArtifact.from_payload(payload)

    assert payload["schema"] == EXTERNAL_PROGRAM_ARTIFACT_SCHEMA
    assert restored.configuration() == artifact.configuration()
    assert torch.equal(restored.codes, artifact.codes)
    assert restored.digest() == artifact.digest()
    assert payload["codes"].device.type == "cpu"


def test_external_program_artifact_rejects_tampered_tensor() -> None:
    payload = _artifact().payload()
    payload["codes"] = payload["codes"].clone()
    payload["codes"].reshape(-1)[0] += 1.0

    with pytest.raises(ValueError, match="checksum mismatch"):
        ExternalProgramArtifact.from_payload(payload)


def test_sequence_program_memory_validates_and_snapshots_portable_artifacts() -> None:
    memory = ExternalSequenceProgramMemory(5)
    slot = memory.add_artifact(_artifact())
    restored = memory.artifact(slot, output_schema=_artifact().output_schema)

    assert slot == 0
    assert memory.configuration()["artifact_schema"] == EXTERNAL_PROGRAM_ARTIFACT_SCHEMA
    assert restored.configuration() == _artifact().configuration()
    assert restored.digest() == _artifact().digest()


def test_sequence_program_memory_rejects_incompatible_program_abi() -> None:
    artifact = ExternalProgramArtifact(
        codes=torch.zeros(2, 5),
        interpreter_schema="neural-computer.other-interpreter.v1",
        execution_schema="neural-computer.external-register-read-execute.v1",
    )

    with pytest.raises(ValueError, match="interpreter schema"):
        ExternalSequenceProgramMemory(5).add_artifact(artifact)


def test_sequence_program_memory_admission_is_copy_on_write_and_protects_old_files() -> None:
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    source = _artifact()
    source_slot = memory.add_artifact(source)
    memory.protect_file(source_slot)
    before_digest = memory.digest()
    rejected = memory.admit_verified_artifact(
        ExternalProgramArtifact(
            codes=source.codes + 1.0,
            interpreter_schema=source.interpreter_schema,
            execution_schema=source.execution_schema,
        ),
        [0.2, 0.9, 0.1],
        threshold=0.8,
        min_observations=3,
    )

    assert rejected.schema == EXTERNAL_PROGRAM_ADMISSION_SCHEMA
    assert not rejected.accepted
    assert rejected.slot is None
    assert memory.file_count == 1
    assert memory.digest() == before_digest
    assert memory.is_file_protected(source_slot)

    candidate = ExternalProgramArtifact(
        codes=source.codes + 2.0,
        interpreter_schema=source.interpreter_schema,
        execution_schema=source.execution_schema,
    )
    accepted = memory.admit_verified_artifact(
        candidate,
        [0.1, 0.85, 0.9, 0.95],
        threshold=0.8,
        min_observations=3,
        protect=True,
    )

    assert isinstance(accepted, ExternalProgramAdmissionReceipt)
    assert accepted.accepted
    assert accepted.slot == 1
    assert accepted.stable_bits_to_threshold == 2
    assert memory.file_count == 2
    assert memory.is_file_protected(0)
    assert memory.is_file_protected(1)
    assert memory.artifact(0).digest() == source.digest()
    assert memory.artifact(1).digest() == candidate.digest()


def test_sequence_program_memory_payload_round_trip_preserves_files_and_router_state() -> None:
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    first = _artifact()
    second = ExternalProgramArtifact(
        codes=first.codes.flip(0),
        interpreter_schema=first.interpreter_schema,
        execution_schema=first.execution_schema,
    )
    memory.add_artifact(first)
    memory.admit_verified_artifact(second, [1.0, 1.0], protect=True)
    payload = memory.payload()
    restored = ExternalSequenceProgramMemory.from_payload(payload)

    assert restored.digest() == memory.digest()
    assert restored.configuration() == memory.configuration()
    assert restored.protection_mask().tolist() == [False, True]
    assert [restored.artifact(i).digest() for i in range(2)] == [
        first.digest(),
        second.digest(),
    ]


def test_program_admission_requires_a_real_stable_run() -> None:
    artifact = _artifact()
    memory = ExternalSequenceProgramMemory(5)
    receipt = memory.admit_verified_artifact(
        artifact,
        [0.1, 1.0, 1.0],
        threshold=0.8,
        min_observations=3,
        min_stable_observations=3,
    )

    assert not receipt.accepted
    assert receipt.slot is None
    assert memory.file_count == 0


def test_program_memory_lifecycle_preserves_logical_ids_and_protected_files() -> None:
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    first = _artifact()
    second = ExternalProgramArtifact(
        codes=first.codes + 10.0,
        interpreter_schema=first.interpreter_schema,
        execution_schema=first.execution_schema,
    )
    duplicate = ExternalProgramArtifact(
        codes=first.codes.clone(),
        interpreter_schema=first.interpreter_schema,
        execution_schema=first.execution_schema,
        output_schema=first.output_schema,
    )
    memory.add_artifact(first)
    memory.add_artifact(second)
    memory.add_artifact(duplicate)
    memory.protect_file(0)
    source_digest = memory.digest()

    protected = memory.evict_verified(
        memory.logical_slot_id(0),
        lambda candidate: True,
    )
    assert not protected.accepted
    assert memory.digest() == source_digest

    evicted = memory.evict_verified(
        memory.logical_slot_id(1),
        lambda candidate: candidate.logical_slot_ids == (0, 2),
    )
    assert evicted.accepted
    assert memory.logical_slot_ids == (0, 2)
    assert memory.is_file_protected(0)

    consolidated = memory.consolidate_verified(
        0,
        2,
        lambda survivor, candidate: survivor.digest() == candidate.digest(),
        lambda candidate: candidate.logical_slot_ids == (0,),
    )
    assert consolidated.accepted
    assert memory.file_count == 1
    assert memory.logical_slot_ids == (0,)
    assert memory.is_file_protected(0)


def test_program_memory_rejects_non_equivalent_consolidation_without_writing() -> None:
    memory = ExternalSequenceProgramMemory(5)
    first = _artifact()
    second = ExternalProgramArtifact(
        codes=first.codes + 10.0,
        interpreter_schema=first.interpreter_schema,
        execution_schema=first.execution_schema,
    )
    memory.add_artifact(first)
    memory.add_artifact(second)
    before = memory.digest()

    receipt = memory.consolidate_verified(
        0,
        1,
        lambda survivor, candidate: False,
        lambda candidate: True,
    )

    assert not receipt.accepted
    assert memory.digest() == before
    assert memory.logical_slot_ids == (0, 1)


def test_program_memory_compression_is_checksumed_and_retention_gated() -> None:
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    first = _artifact()
    second = ExternalProgramArtifact(
        codes=first.codes + 0.25,
        interpreter_schema=first.interpreter_schema,
        execution_schema=first.execution_schema,
    )
    memory.add_artifact(first)
    memory.add_artifact(second)
    memory.protect_file(0)
    compressed = memory.compressed_payload(dtype=torch.float16)
    restored = ExternalSequenceProgramMemory.from_compressed_payload(compressed)

    assert restored.file_count == memory.file_count
    assert restored.logical_slot_ids == memory.logical_slot_ids
    assert restored.protection_mask().tolist() == [True, False]
    assert torch.allclose(restored.artifact(0).codes, first.codes, atol=1e-3)
    assert compressed["state"]["programs.0"].dtype == torch.float16

    receipt = memory.compress_verified(
        dtype=torch.float16,
        retention_probe=lambda candidate: candidate.file_count == 2
        and candidate.is_file_protected(0),
    )
    assert receipt.accepted
    assert receipt.candidate_storage_bytes < receipt.source_storage_bytes
    assert memory.file_count == 2


def test_program_memory_migrates_legacy_payload_with_default_logical_ids() -> None:
    memory = ExternalSequenceProgramMemory(5)
    memory.add_artifact(_artifact())
    payload = memory.payload()
    legacy_configuration = dict(payload["configuration"])
    legacy_configuration["schema"] = "neural-computer.external-sequence-program-memory.v1"
    legacy_configuration.pop("logical_slot_ids")
    legacy_configuration.pop("next_logical_slot_id")
    legacy = dict(payload)
    legacy["schema"] = "neural-computer.external-sequence-program-memory.v1"
    legacy["configuration"] = legacy_configuration
    legacy["sha256"] = memory._digest_components(
        "neural-computer.external-sequence-program-memory.v1",
        legacy_configuration,
        payload["state"],
    )

    restored = ExternalSequenceProgramMemory.from_payload(legacy)

    assert restored.logical_slot_ids == (0,)
    assert restored.artifact(0).digest() == memory.artifact(0).digest()


def test_program_memory_maintenance_policy_executes_real_transactions() -> None:
    memory = ExternalSequenceProgramMemory(5, content_addressing=True, hard_routing=True)
    first = _artifact()
    distinct = ExternalProgramArtifact(
        codes=first.codes + 10.0,
        interpreter_schema=first.interpreter_schema,
        execution_schema=first.execution_schema,
        output_schema=first.output_schema,
    )
    memory.add_artifact(first)
    memory.add_artifact(first)
    memory.add_artifact(distinct)
    memory.protect_file(0)
    policy = ExternalMemoryMaintenancePolicy(hidden_width=8)

    with torch.no_grad():
        policy.network[-1].bias.fill_(-10.0)
        policy.network[-1].bias[1] = 10.0
    share = memory.propose_maintenance(
        policy,
        capacity_limit=3,
        share_available=True,
        compression_available=True,
        evict_available=True,
        redundancy_pressure=1.0,
        compression_opportunity=0.5,
    )
    assert share.action == "share"
    shared = memory.apply_maintenance_proposal(
        share,
        share_pair=(0, 1),
        equivalence_probe=lambda survivor, duplicate: survivor.digest()
        == duplicate.digest(),
        retention_probe=lambda candidate: candidate.logical_slot_ids == (0, 2),
    )
    assert shared.accepted
    assert memory.logical_slot_ids == (0, 2)

    with torch.no_grad():
        policy.network[-1].bias.fill_(-10.0)
        policy.network[-1].bias[3] = 10.0
    evict = memory.propose_maintenance(
        policy,
        capacity_limit=3,
        evict_available=True,
    )
    assert evict.action == "evict"
    evicted = memory.apply_maintenance_proposal(
        evict,
        evict_slot_id=2,
        retention_probe=lambda candidate: candidate.logical_slot_ids == (0,),
    )
    assert evicted.accepted
    assert memory.logical_slot_ids == (0,)

    with torch.no_grad():
        policy.network[-1].bias.fill_(-10.0)
        policy.network[-1].bias[2] = 10.0
    compress = memory.propose_maintenance(
        policy,
        compression_available=True,
        compression_opportunity=0.5,
    )
    assert compress.action == "compress"
    compressed = memory.apply_maintenance_proposal(
        compress,
        retention_probe=lambda candidate: candidate.file_count == 1,
    )
    assert compressed.accepted
    assert compressed.candidate_storage_bytes < compressed.source_storage_bytes

    growth_memory = ExternalSequenceProgramMemory(5)
    growth_memory.add_artifact(first)
    with torch.no_grad():
        policy.network[-1].bias.fill_(-10.0)
        policy.network[-1].bias[0] = 10.0
    grow = growth_memory.propose_maintenance(policy, growth_available=True)
    assert grow.action == "grow"
    admitted = growth_memory.apply_maintenance_proposal(
        grow,
        growth_artifact=distinct,
        growth_outcomes=[1.0, 1.0],
        protect_growth=True,
    )
    assert admitted.accepted
    assert growth_memory.file_count == 2

    with torch.no_grad():
        policy.network[-1].bias.fill_(-10.0)
        policy.network[-1].bias[4] = 10.0
    before = growth_memory.digest()
    deferred = growth_memory.propose_maintenance(policy)
    assert deferred.action == "defer"
    assert growth_memory.apply_maintenance_proposal(deferred) is None
    assert growth_memory.digest() == before
