import pytest
import torch

from neural_computer import (
    EXTERNAL_PROGRAM_ADMISSION_SCHEMA,
    EXTERNAL_PROGRAM_ARTIFACT_SCHEMA,
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
)


def _artifact() -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.arange(15, dtype=torch.float32).reshape(3, 5),
        interpreter_schema="neural-computer.external-register.v4",
        execution_schema="neural-computer.external-register-read-execute.v1",
        output_schema="neural-computer.external-register-canonical-readout.v1",
    )


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
