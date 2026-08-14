from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical import (
    FrozenControllerProgramMachine,
    PhysicalBrainWorkshopConfig,
    PhysicalBrainWorkshopReport,
    PhysicalTrainingSession,
    SourcePreservingTemporalMachine,
    load_physical_training_checkpoint,
    run_physical_training_campaign,
    save_physical_training_checkpoint,
)


def _machine(seed: int = 17) -> SourcePreservingTemporalMachine:
    torch.manual_seed(seed)
    return SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=2,
        max_sources=1,
        action_count=2,
        intention_width=6,
        hidden=8,
        sample=True,
    )


def _program_machine(seed: int = 17) -> FrozenControllerProgramMachine:
    torch.manual_seed(seed)
    return FrozenControllerProgramMachine(
        8,
        source_key_width=3,
        max_history=2,
        max_sources=1,
        action_count=2,
        intention_width=6,
        hidden=8,
        sample=True,
    )


def _report(rewards: tuple[float, ...], *, updates: int) -> PhysicalBrainWorkshopReport:
    return PhysicalBrainWorkshopReport(
        ticks=12,
        input_events=len(rewards) + 1,
        unique_public_outcomes=len(rewards),
        optimizer_updates=updates,
        program_file_updates=0,
        emitted_actions=len(rewards) + 1,
        deadline_misses=0,
        elapsed_seconds=1.0,
        tick_hz=10.0,
        action_delay_seconds=0.0,
        capture_backend="test",
        rewards=rewards,
        actions=tuple(0 for _ in range(len(rewards) + 1)),
        propensities=tuple(0.5 for _ in range(len(rewards) + 1)),
        evidence_digests=tuple((f"{index:064x}",) for index in range(len(rewards))),
        event_payloads=(),
        evidence_archive=None,
    )


def test_physical_campaign_persists_weights_but_resets_session_history(
    tmp_path: Path,
) -> None:
    machine = _machine()
    machine._histories[(1.0,)] = [torch.ones(1, 8)]
    calls = 0

    def run_session(current, _config):
        nonlocal calls
        assert current._histories == {}
        calls += 1
        current._histories[(1.0,)] = [torch.ones(1, 8)]
        current.optimizer_updates += 2
        current.unique_outcome_bits += 2
        rewards = (0.0, 1.0) if calls == 1 else (1.0, 1.0)
        return _report(rewards, updates=current.optimizer_updates)

    report = run_physical_training_campaign(
        machine,
        PhysicalBrainWorkshopConfig(event_width=8, source_key_width=3),
        sessions=2,
        seconds_per_session=1.0,
        seed=17,
        output_directory=tmp_path,
        rolling_window=3,
        session_runner=run_session,
    )

    assert report.completed_sessions == 2
    assert report.unique_public_outcomes == 4
    assert report.optimizer_updates == 4
    assert report.replayed_examples == 0
    assert report.sessions[0].accuracy == 0.5
    assert report.sessions[1].rolling_accuracy == 1.0
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "campaign.json").is_file()
    assert (tmp_path / "session-002.json").is_file()


def test_physical_checkpoint_restores_model_optimizer_counters_and_curve(
    tmp_path: Path,
) -> None:
    source = _machine()
    source.model_version = 4
    source.optimizer_updates = 3
    source.unique_outcome_bits = 3
    expected = next(source.parameters()).detach().clone()
    session = PhysicalTrainingSession(
        session=1,
        unique_public_outcomes=3,
        optimizer_updates=3,
        program_file_updates=0,
        emitted_actions=4,
        deadline_misses=0,
        accuracy=2 / 3,
        rolling_accuracy=2 / 3,
        cumulative_accuracy=2 / 3,
        cumulative_public_outcomes=3,
        cumulative_optimizer_updates=3,
        elapsed_seconds=1.0,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_physical_training_checkpoint(
        source,
        checkpoint,
        completed_sessions=1,
        rewards=(1.0, 0.0, 1.0),
        sessions=(session,),
    )

    restored = _machine(seed=99)
    completed, rewards, sessions = load_physical_training_checkpoint(
        restored, checkpoint
    )

    assert completed == 1
    assert rewards == (1.0, 0.0, 1.0)
    assert sessions == (session,)
    assert restored.model_version == 4
    assert restored.optimizer_updates == 3
    assert restored.unique_outcome_bits == 3
    assert torch.equal(next(restored.parameters()), expected)
    assert restored._histories == {}


def test_physical_checkpoint_restores_external_file_without_controller_update(
    tmp_path: Path,
) -> None:
    source = _program_machine()
    controller_digest = source.controller_digest()
    with torch.no_grad():
        source.relative_address_logits[0] = 0.75
    program_digest = source.program_digest()
    source.program_file_updates = 3
    source.model_version = 3
    source.unique_outcome_bits = 3
    session = PhysicalTrainingSession(
        session=1,
        unique_public_outcomes=3,
        optimizer_updates=0,
        program_file_updates=3,
        emitted_actions=4,
        deadline_misses=0,
        accuracy=2 / 3,
        rolling_accuracy=2 / 3,
        cumulative_accuracy=2 / 3,
        cumulative_public_outcomes=3,
        cumulative_optimizer_updates=0,
        elapsed_seconds=1.0,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_physical_training_checkpoint(
        source,
        checkpoint,
        completed_sessions=1,
        rewards=(1.0, 0.0, 1.0),
        sessions=(session,),
    )

    restored = _program_machine()
    completed, rewards, sessions = load_physical_training_checkpoint(
        restored, checkpoint
    )

    assert completed == 1
    assert rewards == (1.0, 0.0, 1.0)
    assert sessions == (session,)
    assert restored.optimizer_updates == 0
    assert restored.program_file_updates == 3
    assert restored.relative_address_logits[0] == 0.75
    assert restored.program_digest() == program_digest
    assert restored.controller_digest() == controller_digest


def test_physical_campaign_records_frozen_controller_and_changed_program(
    tmp_path: Path,
) -> None:
    machine = _program_machine()
    controller_before = machine.controller_digest()
    program_before = machine.program_digest()

    def run_session(current, _config):
        with torch.no_grad():
            current.relative_address_logits[0] += 0.25
        current.program_file_updates += 2
        current.unique_outcome_bits += 2
        return replace(_report((0.0, 1.0), updates=0), program_file_updates=2)

    report = run_physical_training_campaign(
        machine,
        PhysicalBrainWorkshopConfig(event_width=8, source_key_width=3),
        sessions=1,
        seconds_per_session=1.0,
        seed=17,
        output_directory=tmp_path,
        session_runner=run_session,
    )

    assert report.learning_target == "external_program_file"
    assert report.controller_frozen
    assert report.controller_digest_before == controller_before
    assert report.controller_digest_after == controller_before
    assert report.program_digest_before == program_before
    assert report.program_digest_after == machine.program_digest()
    assert report.program_digest_after != program_before
    assert report.optimizer_updates == 0
    assert report.program_file_updates == 2

    resumed = run_physical_training_campaign(
        machine,
        PhysicalBrainWorkshopConfig(event_width=8, source_key_width=3),
        sessions=2,
        seconds_per_session=1.0,
        seed=17,
        output_directory=tmp_path,
        resume=True,
        session_runner=run_session,
    )

    assert resumed.completed_sessions == 2
    assert resumed.program_digest_before == program_before
    assert resumed.controller_digest_before == controller_before


def test_physical_campaign_rejects_reward_accounting_without_public_outcome(
    tmp_path: Path,
) -> None:
    machine = _machine()

    def invalid_session(current, _config):
        current.optimizer_updates += 2
        return replace(
            _report((1.0,), updates=current.optimizer_updates), unique_public_outcomes=2
        )

    with pytest.raises(RuntimeError, match="reward accounting"):
        run_physical_training_campaign(
            machine,
            PhysicalBrainWorkshopConfig(event_width=8, source_key_width=3),
            sessions=1,
            seconds_per_session=1.0,
            seed=17,
            output_directory=tmp_path,
            session_runner=invalid_session,
        )
