"""Persistent human-parity physical Brain Workshop training campaigns."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from .physical_live import (
    PhysicalBrainWorkshopConfig,
    PhysicalBrainWorkshopReport,
    run_physical_brainworkshop_lifetime,
    save_physical_report,
)
from .rendered_live import SourcePreservingTemporalMachine

PHYSICAL_TRAINING_SCHEMA = "neural-computer.brainworkshop-physical-training.v1"
PHYSICAL_TRAINING_CHECKPOINT_SCHEMA = (
    "neural-computer.brainworkshop-physical-training-checkpoint.v1"
)

PhysicalSessionRunner = Callable[
    [SourcePreservingTemporalMachine, PhysicalBrainWorkshopConfig],
    PhysicalBrainWorkshopReport,
]


@dataclass(frozen=True)
class PhysicalTrainingSession:
    session: int
    unique_public_outcomes: int
    optimizer_updates: int
    program_file_updates: int
    emitted_actions: int
    deadline_misses: int
    accuracy: float
    rolling_accuracy: float
    cumulative_accuracy: float
    cumulative_public_outcomes: int
    cumulative_optimizer_updates: int
    elapsed_seconds: float
    total_seconds_p50: float | None = None
    total_seconds_p99: float | None = None


@dataclass(frozen=True)
class PhysicalTrainingCampaign:
    sessions: tuple[PhysicalTrainingSession, ...]
    rewards: tuple[float, ...]
    requested_sessions: int
    completed_sessions: int
    unique_public_outcomes: int
    optimizer_updates: int
    program_file_updates: int
    replayed_examples: int
    wall_seconds: float
    rolling_window: int
    learning_target: str
    controller_digest_before: str | None
    controller_digest_after: str | None
    controller_frozen: bool
    program_digest_before: str | None
    program_digest_after: str | None
    schema: str = PHYSICAL_TRAINING_SCHEMA

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _machine_configuration(
    machine: SourcePreservingTemporalMachine,
) -> dict[str, object]:
    return {
        "machine_type": type(machine).__name__,
        "learning_target": getattr(machine, "learning_target", "neural_parameters"),
        "event_width": machine.event_width,
        "source_key_width": machine.source_key_width,
        "max_history": machine.max_history,
        "max_sources": machine.max_sources,
        "action_count": machine.action_count,
        "intention_width": machine.intention_width,
    }


def _optional_digest(
    machine: SourcePreservingTemporalMachine, method_name: str
) -> str | None:
    method = getattr(machine, method_name, None)
    return None if method is None else str(method())


def _digest_unchanged(before: str | None, after: str | None) -> bool:
    return before is not None and before == after


def save_physical_training_checkpoint(
    machine: SourcePreservingTemporalMachine,
    path: Path,
    *,
    completed_sessions: int,
    rewards: tuple[float, ...],
    sessions: tuple[PhysicalTrainingSession, ...],
) -> None:
    """Persist weights, optimizer, counters, and public training accounting."""

    if completed_sessions < 0 or completed_sessions != len(sessions):
        raise ValueError("checkpoint session count is inconsistent")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": PHYSICAL_TRAINING_CHECKPOINT_SCHEMA,
            "machine_configuration": _machine_configuration(machine),
            "model_state": (
                {}
                if hasattr(machine, "external_program_payload")
                else machine.state_dict()
            ),
            "optimizer_state": (
                None
                if hasattr(machine, "external_program_payload")
                else machine.optimizer.state_dict()
            ),
            "external_program": (
                machine.external_program_payload()
                if hasattr(machine, "external_program_payload")
                else None
            ),
            "model_version": machine.model_version,
            "optimizer_updates": machine.optimizer_updates,
            "unique_outcome_bits": machine.unique_outcome_bits,
            "completed_sessions": completed_sessions,
            "rewards": rewards,
            "sessions": [asdict(session) for session in sessions],
        },
        path,
    )


def load_physical_training_checkpoint(
    machine: SourcePreservingTemporalMachine,
    path: Path,
) -> tuple[int, tuple[float, ...], tuple[PhysicalTrainingSession, ...]]:
    """Resume a campaign only into an exactly compatible machine."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != PHYSICAL_TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported physical training checkpoint")
    if payload.get("machine_configuration") != _machine_configuration(machine):
        raise ValueError("physical checkpoint machine configuration does not match")
    optimizer_state = payload["optimizer_state"]
    external_program = payload["external_program"]
    if external_program is None:
        machine.load_state_dict(payload["model_state"])
        if optimizer_state is None:
            raise ValueError("neural checkpoint is missing optimizer state")
        machine.optimizer.load_state_dict(optimizer_state)
    else:
        if payload["model_state"] or optimizer_state is not None or not hasattr(
            machine, "load_external_program_payload"
        ):
            raise ValueError("external program checkpoint has incompatible machine")
        machine.load_external_program_payload(external_program)
    machine.model_version = int(payload["model_version"])
    machine.optimizer_updates = int(payload["optimizer_updates"])
    machine.unique_outcome_bits = int(payload["unique_outcome_bits"])
    machine.reset_history()
    sessions = tuple(PhysicalTrainingSession(**item) for item in payload["sessions"])
    completed = int(payload["completed_sessions"])
    if completed != len(sessions):
        raise ValueError("physical checkpoint session history is inconsistent")
    return completed, tuple(float(value) for value in payload["rewards"]), sessions


def _accuracy(values: tuple[float, ...]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def run_physical_training_campaign(
    machine: SourcePreservingTemporalMachine,
    config: PhysicalBrainWorkshopConfig,
    *,
    sessions: int,
    seconds_per_session: float,
    seed: int,
    output_directory: Path,
    rolling_window: int = 44,
    resume: bool = False,
    session_runner: PhysicalSessionRunner | None = None,
) -> PhysicalTrainingCampaign:
    """Train one persistent machine across ordinary visible GUI sessions."""

    if sessions < 1 or seconds_per_session <= 0.0 or rolling_window < 1:
        raise ValueError("physical training campaign settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "checkpoint.pt"
    campaign_path = output_directory / "campaign.json"
    completed = 0
    rewards: tuple[float, ...] = ()
    summaries: tuple[PhysicalTrainingSession, ...] = ()
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        completed, rewards, summaries = load_physical_training_checkpoint(
            machine, checkpoint_path
        )
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing campaign checkpoint: {checkpoint_path}"
        )
    if completed > sessions:
        raise ValueError("checkpoint has more sessions than the requested campaign")

    learning_target = str(
        getattr(machine, "learning_target", "neural_parameters")
    )
    controller_digest_before = _optional_digest(machine, "controller_digest")
    program_digest_before = _optional_digest(machine, "program_digest")
    if resume and campaign_path.is_file():
        prior_campaign = json.loads(campaign_path.read_text())
        prior_controller = prior_campaign.get("controller_digest_before")
        prior_program = prior_campaign.get("program_digest_before")
        prior_target = prior_campaign.get("learning_target")
        if (
            prior_controller != controller_digest_before
            or prior_target != learning_target
            or (prior_program is not None and not isinstance(prior_program, str))
        ):
            raise ValueError("resumed campaign ownership metadata does not match")
        program_digest_before = prior_program

    initial_updates = machine.optimizer_updates - sum(
        item.optimizer_updates for item in summaries
    )
    initial_program_updates = getattr(machine, "program_file_updates", 0) - sum(
        item.program_file_updates for item in summaries
    )
    runner = session_runner
    for session_index in range(completed + 1, sessions + 1):
        machine.reset_history()
        before_updates = machine.optimizer_updates
        before_program_updates = getattr(machine, "program_file_updates", 0)
        session_evidence = (
            None
            if config.evidence_directory is None
            else config.evidence_directory / f"session-{session_index:03d}"
        )
        session_config = replace(config, evidence_directory=session_evidence)
        if runner is None:
            report = run_physical_brainworkshop_lifetime(
                machine,
                session_config,
                seconds=seconds_per_session,
                seed=seed,
                start_session=True,
            )
        else:
            report = runner(machine, session_config)
        update_delta = machine.optimizer_updates - before_updates
        program_update_delta = (
            getattr(machine, "program_file_updates", 0) - before_program_updates
        )
        if update_delta + program_update_delta != report.unique_public_outcomes:
            raise RuntimeError("one-update-per-public-outcome invariant failed")
        assert_frozen = getattr(machine, "assert_controller_frozen", None)
        if assert_frozen is not None:
            assert_frozen()
        if len(report.rewards) != report.unique_public_outcomes:
            raise RuntimeError("physical report reward accounting is inconsistent")
        session_rewards = tuple(report.rewards)
        rewards += session_rewards
        rolling = rewards[-rolling_window:]
        summary = PhysicalTrainingSession(
            session=session_index,
            unique_public_outcomes=report.unique_public_outcomes,
            optimizer_updates=update_delta,
            program_file_updates=program_update_delta,
            emitted_actions=report.emitted_actions,
            deadline_misses=report.deadline_misses,
            accuracy=_accuracy(session_rewards),
            rolling_accuracy=_accuracy(rolling),
            cumulative_accuracy=_accuracy(rewards),
            cumulative_public_outcomes=len(rewards),
            cumulative_optimizer_updates=machine.optimizer_updates - initial_updates,
            elapsed_seconds=report.elapsed_seconds,
            total_seconds_p50=report.total_seconds_p50,
            total_seconds_p99=report.total_seconds_p99,
        )
        summaries += (summary,)
        save_physical_report(
            report, output_directory / f"session-{session_index:03d}.json"
        )
        save_physical_training_checkpoint(
            machine,
            checkpoint_path,
            completed_sessions=session_index,
            rewards=rewards,
            sessions=summaries,
        )
        partial = PhysicalTrainingCampaign(
            sessions=summaries,
            rewards=rewards,
            requested_sessions=sessions,
            completed_sessions=session_index,
            unique_public_outcomes=len(rewards),
            optimizer_updates=machine.optimizer_updates - initial_updates,
            program_file_updates=(
                getattr(machine, "program_file_updates", 0)
                - initial_program_updates
            ),
            replayed_examples=0,
            wall_seconds=sum(item.elapsed_seconds for item in summaries),
            rolling_window=rolling_window,
            learning_target=learning_target,
            controller_digest_before=controller_digest_before,
            controller_digest_after=_optional_digest(machine, "controller_digest"),
            controller_frozen=_digest_unchanged(
                controller_digest_before,
                _optional_digest(machine, "controller_digest"),
            ),
            program_digest_before=program_digest_before,
            program_digest_after=_optional_digest(machine, "program_digest"),
        )
        save_physical_training_report(partial, campaign_path)
        print(
            f"session={session_index} outcomes={len(session_rewards)} "
            f"accuracy={summary.accuracy:.3f} "
            f"rolling_{len(rolling)}={summary.rolling_accuracy:.3f} "
            f"optimizer_updates={summary.cumulative_optimizer_updates} "
            f"program_updates={sum(item.program_file_updates for item in summaries)}",
            flush=True,
        )

    return PhysicalTrainingCampaign(
        sessions=summaries,
        rewards=rewards,
        requested_sessions=sessions,
        completed_sessions=len(summaries),
        unique_public_outcomes=len(rewards),
        optimizer_updates=machine.optimizer_updates - initial_updates,
        program_file_updates=(
            getattr(machine, "program_file_updates", 0) - initial_program_updates
        ),
        replayed_examples=0,
        wall_seconds=sum(item.elapsed_seconds for item in summaries),
        rolling_window=rolling_window,
        learning_target=learning_target,
        controller_digest_before=controller_digest_before,
        controller_digest_after=_optional_digest(machine, "controller_digest"),
        controller_frozen=_digest_unchanged(
            controller_digest_before,
            _optional_digest(machine, "controller_digest"),
        ),
        program_digest_before=program_digest_before,
        program_digest_after=_optional_digest(machine, "program_digest"),
    )


def save_physical_training_report(
    report: PhysicalTrainingCampaign,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
