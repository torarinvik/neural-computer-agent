from pathlib import Path

import pytest
import torch

from neural_computer.executive import ExecutiveInstruction, ExternalExecutiveProgram
from neural_computer.executive_bank import (
    ExternalExecutiveOperatorSpec,
    ExternalExecutiveProgramArtifact,
    ExternalExecutiveProgramBank,
    executive_artifact_can_handoff,
)
from neural_computer.interface import AmodalEvent, AmodalEventCollection


def _digest(seed: int = 7) -> str:
    return f"{seed:064x}"


def _artifact(delay: int) -> ExternalExecutiveProgramArtifact:
    return ExternalExecutiveProgramArtifact(
        program=ExternalExecutiveProgram(
            5,
            (
                ExecutiveInstruction("receive", destination=0),
                ExecutiveInstruction("call", destination=1, operator_handle=1, arguments=(0,)),
                ExecutiveInstruction("call", destination=2, operator_handle=2, arguments=(1,)),
                ExecutiveInstruction("call", destination=3, operator_handle=3, arguments=(1, 2)),
                ExecutiveInstruction("branch", source=3, true_target=5, false_target=5, unknown_target=7),
                ExecutiveInstruction("call", destination=4, operator_handle=4, arguments=(3,)),
                ExecutiveInstruction("emit", source=4, next_target=0),
                ExecutiveInstruction("wait", next_target=0),
                ExecutiveInstruction("halt"),
            ),
        ),
        operator_specs=(
            ExternalExecutiveOperatorSpec(1, "singleton_event_value", width=3),
            ExternalExecutiveOperatorSpec(2, "value_delay", width=3, delay=delay),
            ExternalExecutiveOperatorSpec(3, "value_equality_evidence"),
            ExternalExecutiveOperatorSpec(4, "evidence_binary_intention"),
        ),
        intention_width=2,
    ).validate()


def _events(value: tuple[float, float, float]) -> AmodalEventCollection:
    return AmodalEventCollection.from_events(
        (AmodalEvent(payload=torch.tensor([value]), confidence=torch.ones(1)),)
    )


def _rollout(bank: ExternalExecutiveProgramBank, slot: int) -> list[torch.Tensor | None]:
    executive = bank.executable(slot, controller_digest=_digest())
    state = executive.initial_state(1, device="cpu")
    outputs: list[torch.Tensor | None] = []
    for value in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        output, state = executive.tick(_events(value), state)
        outputs.append(None if output.intention is None else output.intention.payload)
    return outputs


def test_immutable_executive_metadata_is_cached() -> None:
    artifact = _artifact(99)
    registry = artifact.registry()
    before = executive_artifact_can_handoff.cache_info()

    program_digest = artifact.program.digest()
    artifact_digest = artifact.digest()
    registry_digest = registry.digest()
    assert not executive_artifact_can_handoff(artifact)
    after_miss = executive_artifact_can_handoff.cache_info()
    assert not executive_artifact_can_handoff(artifact)
    after_hit = executive_artifact_can_handoff.cache_info()

    assert artifact.program.digest() is program_digest
    assert artifact.digest() is artifact_digest
    assert registry.digest() is registry_digest
    assert after_miss.misses == before.misses + 1
    assert after_hit.hits == after_miss.hits + 1


def test_verified_executive_skills_round_trip_and_execute_identically(
    tmp_path: Path,
) -> None:
    bank = ExternalExecutiveProgramBank(controller_digest=_digest(), capacity=4)
    source = _artifact(1)
    target = _artifact(2)
    source_receipt = bank.admit(
        source, [1.0, 1.0], threshold=0.9, min_observations=2, min_stable_observations=2
    )
    source_digest = bank.artifact(0).digest()
    target_receipt = bank.admit(
        target, [1.0, 1.0], threshold=0.9, min_observations=2, min_stable_observations=2
    )
    before = _rollout(bank, target_receipt.slot or 0)
    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)

    restored = ExternalExecutiveProgramBank.load_bank(path)
    after = _rollout(restored, target_receipt.slot or 0)

    assert source_receipt.accepted and source_receipt.slot == 0
    assert target_receipt.accepted and target_receipt.slot == 1
    assert restored.program_count == 2
    assert restored.artifact(0).digest() == source_digest
    assert restored.digest() == bank.digest()
    assert before[0] is None and before[1] is None
    assert torch.equal(before[2], torch.tensor([[-1.0, 1.0]]))
    assert torch.equal(before[3], torch.tensor([[1.0, -1.0]]))
    for original, reloaded in zip(before, after, strict=True):
        if original is None:
            assert reloaded is None
        else:
            assert torch.equal(original, reloaded)

    duplicate = restored.admit(
        target, [1.0, 1.0], threshold=0.9, min_observations=2, min_stable_observations=2
    )
    assert duplicate.accepted and duplicate.slot == 1
    assert restored.program_count == 2
    assert restored.artifact(0).digest() == source_digest


def test_rejection_and_corruption_leave_no_executable_change(tmp_path: Path) -> None:
    bank = ExternalExecutiveProgramBank(controller_digest=_digest(), capacity=2)
    bank.admit(
        _artifact(1), [1.0, 1.0], threshold=0.9, min_observations=2, min_stable_observations=2
    )
    before = bank.digest()
    rejected = bank.admit(
        _artifact(2), [0.0, 0.0], threshold=0.9, min_observations=2, min_stable_observations=2
    )
    assert not rejected.accepted
    assert bank.digest() == before

    with pytest.raises(ValueError, match="controller digest is incompatible"):
        bank.executable(0, controller_digest=_digest(8))

    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    with path.open("a") as stream:
        stream.write("tamper")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        ExternalExecutiveProgramBank.load_bank(path)


def test_operator_manifest_is_allow_listed_and_multistream_ambiguity_is_absent() -> None:
    payload = _artifact(1).payload()
    specs = payload["operator_specs"]
    assert isinstance(specs, list)
    specs[0]["kind"] = "arbitrary.module:operator"
    with pytest.raises(ValueError, match="not allow-listed"):
        ExternalExecutiveProgramArtifact.from_payload(payload)

    bank = ExternalExecutiveProgramBank(controller_digest=_digest())
    receipt = bank.admit(_artifact(1), [1.0])
    executive = bank.executable(receipt.slot or 0, controller_digest=_digest())
    state = executive.initial_state(1, device="cpu")
    ambiguous = AmodalEventCollection.from_events(
        (
            AmodalEvent(payload=torch.tensor([[1.0, 0.0, 0.0]])),
            AmodalEvent(payload=torch.tensor([[0.0, 1.0, 0.0]])),
        )
    )
    output, state = executive.tick(ambiguous, state)
    assert output.status == "waiting" and output.intention is None
    assert not bool(state.workspace[1].present.item())
