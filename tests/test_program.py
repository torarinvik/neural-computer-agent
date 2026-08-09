import pytest
import torch

from neural_computer import (
    EXTERNAL_PROGRAM_ARTIFACT_SCHEMA,
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
