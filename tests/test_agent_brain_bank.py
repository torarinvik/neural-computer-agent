from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_computer import (
    AGENT_BRAIN_EXECUTIVE_KIND,
    AGENT_BRAIN_TEMPORAL_KIND,
    EXTERNAL_AGENT_BRAIN_BANK_SCHEMA,
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    ExternalAgentBrainBank,
    ExternalExecutiveProgramBank,
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
    build_temporal_equality_executive_artifact,
)


def _digest(seed: int = 7) -> str:
    return f"{seed:064x}"


def _temporal_artifact(values: tuple[float, ...]) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.tensor([values], dtype=torch.float32),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def test_mixed_agent_brain_round_trip_preserves_both_skill_families(
    tmp_path: Path,
) -> None:
    controller_digest = _digest()
    bank = ExternalAgentBrainBank(controller_digest=controller_digest, capacity=4)
    executive = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    executive_receipt = bank.admit_executive(
        executive,
        [1.0, 1.0],
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
    )

    legacy = ExternalTemporalProgramBank(
        4,
        3,
        controller_digest=controller_digest,
        min_mastery_observations=3,
    )
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    temporal_receipt = legacy.admit(
        _temporal_artifact((5.0, -3.0, -3.0)),
        context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    bank.import_temporal_bank(legacy)

    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    payload = json.loads(path.read_text())
    restored = ExternalAgentBrainBank.load_bank(path)

    assert payload["schema"] == EXTERNAL_AGENT_BRAIN_BANK_SCHEMA
    assert payload["temporal_bank"]["artifacts"][0]["codes"]["data_b64"]
    assert executive_receipt.accepted and executive_receipt.slot == 0
    assert temporal_receipt.accepted and temporal_receipt.slot == 0
    assert bank.program_count == 2
    assert bank.executive_program_count == 1
    assert bank.temporal_program_count == 1
    assert restored.digest() == bank.digest()
    assert restored.manifest() == bank.manifest()
    assert restored.artifact(AGENT_BRAIN_EXECUTIVE_KIND, 0).digest() == executive.digest()
    assert restored.artifact(AGENT_BRAIN_TEMPORAL_KIND, 0).digest() == legacy.artifact(0).digest()
    assert restored.temporal_bank is not None
    assert restored.temporal_bank.select(context).slot == 0
    restored.executable(0, controller_digest=controller_digest)


def test_legacy_temporal_bank_requires_explicit_migration(tmp_path: Path) -> None:
    controller_digest = _digest(9)
    legacy = ExternalTemporalProgramBank(4, 3, controller_digest=controller_digest)
    legacy.admit(
        _temporal_artifact((5.0, -3.0, -3.0)),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        [1.0] * 8,
    )
    source = tmp_path / "legacy.bank"
    destination = tmp_path / "AgentBrain.bank"
    legacy.save_bank(source)

    with pytest.raises(ValueError, match="not loaded implicitly"):
        ExternalAgentBrainBank.load_bank(source)

    migrated = ExternalAgentBrainBank.migrate_legacy_temporal_bank(source, destination)
    restored = ExternalAgentBrainBank.load_bank(destination)
    assert migrated.digest() == restored.digest()
    assert restored.temporal_program_count == 1
    assert restored.executive_program_count == 0


def test_old_executive_json_bank_is_wrapped_without_losing_artifacts(tmp_path: Path) -> None:
    old = ExternalExecutiveProgramBank(controller_digest=_digest(), capacity=2)
    artifact = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    old.admit(artifact, [1.0])
    path = tmp_path / "old-executive.bank"
    old.save_bank(path)

    wrapped = ExternalAgentBrainBank.load_bank(path)

    assert wrapped.controller_digest == old.controller_digest
    assert wrapped.executive_program_count == 1
    assert wrapped.artifact(AGENT_BRAIN_EXECUTIVE_KIND, 0).digest() == artifact.digest()


def test_mixed_bank_capacity_and_controller_binding_are_fail_closed() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(), capacity=1)
    artifact = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    accepted = bank.admit_executive(artifact, [1.0])
    rejected = bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=2),
        [1.0],
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert bank.program_count == 1
    with pytest.raises(ValueError, match="controller digest is incompatible"):
        bank.executable(0, controller_digest=_digest(99))


def test_unified_bank_checksum_rejects_tampering(tmp_path: Path) -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest())
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=1), [1.0]
    )
    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    path.write_text(path.read_text() + "tamper")

    with pytest.raises(ValueError, match="file checksum mismatch"):
        ExternalAgentBrainBank.load_bank(path)
