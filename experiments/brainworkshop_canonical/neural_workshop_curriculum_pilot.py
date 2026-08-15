"""Train a frozen-controller temporal program on a live cell curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from neural_computer import DEFAULT_AGENT_BANK_FILENAME, ExternalTemporalProgramBank

from .controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_live import (
    NeuralWorkshopLiveConfig,
    NeuralWorkshopLiveReport,
    build_neural_workshop_environment,
    run_neural_workshop_live_lifetime,
)
from .physical_program_bank import learned_event_context
from .rendered_live import PretrainedControllerProgramMachine

NEURAL_WORKSHOP_CURRICULUM_SCHEMA = (
    "neural-computer.neural-workshop-cell-curriculum.v1"
)
NEURAL_WORKSHOP_CURRICULUM_CHECKPOINT_SCHEMA = (
    "neural-computer.neural-workshop-cell-curriculum-checkpoint.v1"
)

SessionRunner = Callable[
    [PretrainedControllerProgramMachine, NeuralWorkshopLiveConfig, int, bool, bool],
    NeuralWorkshopLiveReport,
]


@dataclass(frozen=True)
class CurriculumPolicy:
    active_cell_ladder: tuple[int, ...] = (2, 3, 4, 6, 8)
    mastery_threshold: float = 0.8
    stable_sessions: int = 3
    minimum_verifier_bits_per_session: int = 8
    maximum_training_sessions_per_rung: int = 12
    retention_sessions: int = 1

    def validate(self, *, grid_size: int) -> CurriculumPolicy:
        if (
            not self.active_cell_ladder
            or tuple(sorted(set(self.active_cell_ladder))) != self.active_cell_ladder
            or self.active_cell_ladder[0] < 2
            or self.active_cell_ladder[-1] > grid_size * grid_size
        ):
            raise ValueError("active-cell ladder must be unique, increasing, and fit")
        if not 0.0 <= self.mastery_threshold <= 1.0:
            raise ValueError("mastery threshold must lie within [0, 1]")
        if min(
            self.stable_sessions,
            self.minimum_verifier_bits_per_session,
            self.maximum_training_sessions_per_rung,
            self.retention_sessions,
        ) < 1:
            raise ValueError("curriculum counts must be positive")
        if self.stable_sessions > self.maximum_training_sessions_per_rung:
            raise ValueError("stable sessions exceed the per-rung training budget")
        return self


@dataclass(frozen=True)
class CurriculumSession:
    rung: int
    active_cells: int
    kind: str
    seed: int
    unique_verifier_bits: int
    positive_verifier_bits: int
    accuracy: float | None
    optimizer_updates: int
    program_file_updates: int
    replayed_examples: int
    logical_trials: int
    emitted_actions: int
    wall_seconds: float
    controller_frozen: bool
    report_path: str


def _session_mastered(session: CurriculumSession, policy: CurriculumPolicy) -> bool:
    return bool(
        session.kind == "train"
        and session.accuracy is not None
        and session.accuracy >= policy.mastery_threshold
        and session.unique_verifier_bits >= policy.minimum_verifier_bits_per_session
        and session.controller_frozen
    )


def rung_mastered(
    sessions: list[CurriculumSession], policy: CurriculumPolicy
) -> bool:
    training = [item for item in sessions if item.kind == "train"]
    return len(training) >= policy.stable_sessions and all(
        _session_mastered(item, policy)
        for item in training[-policy.stable_sessions :]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_checkpoint(
    machine: PretrainedControllerProgramMachine,
    path: Path,
    *,
    next_rung: int,
    sessions: list[CurriculumSession],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": NEURAL_WORKSHOP_CURRICULUM_CHECKPOINT_SCHEMA,
            "next_rung": next_rung,
            "sessions": [asdict(item) for item in sessions],
            "external_program": machine.external_program_payload(),
            "model_version": machine.model_version,
            "unique_outcome_bits": machine.unique_outcome_bits,
        },
        path,
    )


def _load_checkpoint(
    machine: PretrainedControllerProgramMachine, path: Path
) -> tuple[int, list[CurriculumSession]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != NEURAL_WORKSHOP_CURRICULUM_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Neural Workshop curriculum checkpoint")
    machine.load_external_program_payload(payload["external_program"])
    machine.model_version = int(payload["model_version"])
    machine.unique_outcome_bits = int(payload["unique_outcome_bits"])
    sessions = [CurriculumSession(**item) for item in payload["sessions"]]
    return int(payload["next_rung"]), sessions


def _admit_rung(
    machine: PretrainedControllerProgramMachine,
    bank_path: Path,
    reports: list[NeuralWorkshopLiveReport],
    policy: CurriculumPolicy,
) -> dict[str, Any]:
    if bank_path.exists():
        bank = ExternalTemporalProgramBank.load_bank(bank_path)
        if not machine.accepts_controller_digest(bank.controller_digest):
            raise ValueError("program bank targets another frozen controller")
    else:
        bank = ExternalTemporalProgramBank(
            machine.event_width,
            machine.max_history,
            controller_digest=machine.controller_digest(),
            generalization_tolerance=0.25,
            mastery_threshold=policy.mastery_threshold,
            min_mastery_observations=policy.stable_sessions,
        )
    rows = [row for report in reports for row in report.event_payloads]
    context = learned_event_context(rows, width=machine.event_width)
    scores = [
        float(report.verifier_accuracy)
        for report in reports
        if report.verifier_accuracy is not None
    ]
    before = bank.digest()
    receipt = bank.admit(
        machine.admitted_program_artifact(),
        context,
        scores,
        threshold=policy.mastery_threshold,
        min_observations=policy.stable_sessions,
        min_stable_observations=policy.stable_sessions,
    )
    if not receipt.accepted:
        if bank.digest() != before:
            raise RuntimeError("rejected rung changed the program bank")
        raise RuntimeError(f"mastered rung failed bank admission: {receipt.reason}")
    bank.save_bank(bank_path)
    return {
        **receipt.payload(),
        "bank_sha256": _sha256(bank_path),
        "bank_program_count": bank.program_count,
    }


def run_cell_curriculum(
    machine: PretrainedControllerProgramMachine,
    controller_payload: dict[str, object],
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    policy: CurriculumPolicy | None = None,
    grid_size: int = 3,
    n_back: int = 1,
    trials: int = 60,
    seed: int = 17,
    resume: bool = False,
    session_runner: SessionRunner | None = None,
) -> dict[str, Any]:
    """Train, retain, admit, and increment one difficulty axis at a time."""

    policy = CurriculumPolicy() if policy is None else policy
    policy.validate(grid_size=grid_size)
    if min(grid_size, n_back, trials) < 1:
        raise ValueError("curriculum dimensions must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "checkpoint.pt"
    campaign_path = output_directory / "campaign.json"
    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    if not resume and (checkpoint_path.exists() or campaign_path.exists()):
        raise FileExistsError("curriculum output exists; pass --resume to continue")
    next_rung = 0
    sessions: list[CurriculumSession] = []
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        next_rung, sessions = _load_checkpoint(machine, checkpoint_path)
    controller_digest = machine.controller_digest()
    started = time.perf_counter()
    prior_campaign = (
        json.loads(campaign_path.read_text())
        if resume and campaign_path.is_file()
        else {}
    )
    admissions: list[dict[str, Any]] = list(prior_campaign.get("admissions", []))
    transfer_controls: list[dict[str, Any]] = list(
        prior_campaign.get("transfer_controls", [])
    )

    def execute(
        current_machine: PretrainedControllerProgramMachine,
        config: NeuralWorkshopLiveConfig,
        session_seed: int,
        learn: bool,
        sample: bool,
    ) -> NeuralWorkshopLiveReport:
        torch.manual_seed(session_seed)
        if session_runner is not None:
            return session_runner(current_machine, config, session_seed, learn, sample)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        return run_neural_workshop_live_lifetime(
            current_machine,
            config,
            seed=session_seed,
            environment=environment,
            verifier=verifier,
            learn=learn,
            sample=sample,
        )

    for rung in range(next_rung, len(policy.active_cell_ladder)):
        active_cells = policy.active_cell_ladder[rung]
        config = NeuralWorkshopLiveConfig(
            grid_size=grid_size,
            active_cells=active_cells,
            n_back=n_back,
            trials=trials,
            event_width=machine.event_width,
            source_key_width=machine.source_key_width,
        )
        rung_sessions = [item for item in sessions if item.rung == rung]
        rung_reports: list[NeuralWorkshopLiveReport] = []
        completed_training = sum(item.kind == "train" for item in rung_sessions)
        if rung > 0 and not any(item.kind == "fresh_control" for item in rung_sessions):
            control_seed = seed + rung * 10_000
            fresh = build_pretrained_controller_program_machine(
                controller_payload,
                learning_rate=float(machine.optimizer.param_groups[0]["lr"]),
                sample=True,
                inherit_program_prior=False,
            )
            control = execute(fresh, config, control_seed, True, True)
            control_path = output_directory / (
                f"rung-{rung + 1:02d}-cells-{active_cells:02d}-fresh-control.json"
            )
            control_path.write_text(
                json.dumps(control.as_dict(), indent=2, sort_keys=True) + "\n"
            )
            control_summary = CurriculumSession(
                rung=rung,
                active_cells=active_cells,
                kind="fresh_control",
                seed=control_seed,
                unique_verifier_bits=control.unique_verifier_bits,
                positive_verifier_bits=control.positive_verifier_bits,
                accuracy=control.verifier_accuracy,
                optimizer_updates=control.optimizer_updates,
                program_file_updates=control.program_file_updates,
                replayed_examples=control.replayed_examples,
                logical_trials=control.logical_trials,
                emitted_actions=control.emitted_actions,
                wall_seconds=control.wall_seconds,
                controller_frozen=control.controller_frozen,
                report_path=str(control_path),
            )
            sessions.append(control_summary)
            rung_sessions.append(control_summary)
            _save_checkpoint(machine, checkpoint_path, next_rung=rung, sessions=sessions)
        while (
            not rung_mastered(rung_sessions, policy)
            and completed_training < policy.maximum_training_sessions_per_rung
        ):
            session_seed = seed + rung * 10_000 + completed_training
            report = execute(machine, config, session_seed, True, True)
            report_path = output_directory / (
                f"rung-{rung + 1:02d}-cells-{active_cells:02d}-"
                f"train-{completed_training + 1:03d}.json"
            )
            report_path.write_text(
                json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
            )
            summary = CurriculumSession(
                rung=rung,
                active_cells=active_cells,
                kind="train",
                seed=session_seed,
                unique_verifier_bits=report.unique_verifier_bits,
                positive_verifier_bits=report.positive_verifier_bits,
                accuracy=report.verifier_accuracy,
                optimizer_updates=report.optimizer_updates,
                program_file_updates=report.program_file_updates,
                replayed_examples=report.replayed_examples,
                logical_trials=report.logical_trials,
                emitted_actions=report.emitted_actions,
                wall_seconds=report.wall_seconds,
                controller_frozen=report.controller_frozen,
                report_path=str(report_path),
            )
            sessions.append(summary)
            rung_sessions.append(summary)
            rung_reports.append(report)
            completed_training += 1
            if rung > 0 and completed_training == 1:
                control_summary = next(
                    item for item in rung_sessions if item.kind == "fresh_control"
                )
                inherited_accuracy = report.verifier_accuracy
                fresh_accuracy = control_summary.accuracy
                transfer_controls.append(
                    {
                        "rung": rung,
                        "active_cells": active_cells,
                        "seed": session_seed,
                        "inherited_accuracy": inherited_accuracy,
                        "fresh_accuracy": fresh_accuracy,
                        "accuracy_ratio": (
                            None
                            if inherited_accuracy is None
                            or fresh_accuracy is None
                            or fresh_accuracy == 0.0
                            else inherited_accuracy / fresh_accuracy
                        ),
                        "inherited_unique_verifier_bits": report.unique_verifier_bits,
                        "fresh_unique_verifier_bits": control_summary.unique_verifier_bits,
                    }
                )
            _save_checkpoint(machine, checkpoint_path, next_rung=rung, sessions=sessions)
            print(
                {
                    "rung": rung + 1,
                    "active_cells": active_cells,
                    "session": completed_training,
                    "verifier_accuracy": report.verifier_accuracy,
                    "unique_verifier_bits": report.unique_verifier_bits,
                    "mastered": rung_mastered(rung_sessions, policy),
                },
                flush=True,
            )
        if not rung_mastered(rung_sessions, policy):
            next_rung = rung
            break

        # Frozen evaluation is required before promotion. It cannot mutate the
        # program merely to make the gate pass.
        retention_passed = True
        for retention_index in range(policy.retention_sessions):
            session_seed = seed + rung * 10_000 + 5_000 + retention_index
            before = machine.program_digest()
            report = execute(machine, config, session_seed, False, False)
            if machine.program_digest() != before:
                raise RuntimeError("frozen retention evaluation changed the program")
            report_path = output_directory / (
                f"rung-{rung + 1:02d}-cells-{active_cells:02d}-"
                f"retention-{retention_index + 1:03d}.json"
            )
            report_path.write_text(
                json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
            )
            retained = bool(
                report.verifier_accuracy is not None
                and report.verifier_accuracy >= policy.mastery_threshold
                and report.unique_verifier_bits
                >= policy.minimum_verifier_bits_per_session
                and report.controller_frozen
            )
            retention_passed &= retained
            sessions.append(
                CurriculumSession(
                    rung=rung,
                    active_cells=active_cells,
                    kind="retention",
                    seed=session_seed,
                    unique_verifier_bits=report.unique_verifier_bits,
                    positive_verifier_bits=report.positive_verifier_bits,
                    accuracy=report.verifier_accuracy,
                    optimizer_updates=report.optimizer_updates,
                    program_file_updates=report.program_file_updates,
                    replayed_examples=report.replayed_examples,
                    logical_trials=report.logical_trials,
                    emitted_actions=report.emitted_actions,
                    wall_seconds=report.wall_seconds,
                    controller_frozen=report.controller_frozen,
                    report_path=str(report_path),
                )
            )
        if not retention_passed:
            next_rung = rung
            _save_checkpoint(machine, checkpoint_path, next_rung=rung, sessions=sessions)
            break

        # Admission uses only the stable training suffix and learned visual
        # event context. No rung/task label is placed in memory.
        stable_reports = rung_reports[-policy.stable_sessions :]
        if len(stable_reports) != policy.stable_sessions:
            # Resumed runs can reconstruct the report evidence from JSON.
            stable_paths = [
                Path(item.report_path)
                for item in rung_sessions
                if item.kind == "train"
            ][-policy.stable_sessions :]
            stable_reports = []
            for path in stable_paths:
                raw = json.loads(path.read_text())
                raw.pop("verifier_accuracy", None)
                stable_reports.append(
                    NeuralWorkshopLiveReport(
                        **{
                            key: tuple(value) if isinstance(value, list) else value
                            for key, value in raw.items()
                        }
                    )
                )
        admission = _admit_rung(machine, bank_path, stable_reports, policy)
        admission.update({"rung": rung, "active_cells": active_cells})
        admissions.append(admission)
        next_rung = rung + 1
        _save_checkpoint(machine, checkpoint_path, next_rung=next_rung, sessions=sessions)
    else:
        next_rung = len(policy.active_cell_ladder)

    controller_after = machine.controller_digest()
    campaign = {
        "schema": NEURAL_WORKSHOP_CURRICULUM_SCHEMA,
        "grid_size": grid_size,
        "n_back": n_back,
        "trials_per_session": trials,
        "policy": asdict(policy),
        "sessions": [asdict(item) for item in sessions],
        "completed_rungs": next_rung,
        "next_active_cells": (
            None
            if next_rung >= len(policy.active_cell_ladder)
            else policy.active_cell_ladder[next_rung]
        ),
        "unique_logical_lifetimes": len(sessions),
        "unique_verifier_bits": sum(item.unique_verifier_bits for item in sessions),
        "primary_unique_verifier_bits": sum(
            item.unique_verifier_bits
            for item in sessions
            if item.kind != "fresh_control"
        ),
        "fresh_control_unique_verifier_bits": sum(
            item.unique_verifier_bits
            for item in sessions
            if item.kind == "fresh_control"
        ),
        "optimizer_updates": sum(item.optimizer_updates for item in sessions),
        "program_file_updates": sum(
            item.program_file_updates
            for item in sessions
            if item.kind != "fresh_control"
        ),
        "fresh_control_program_file_updates": sum(
            item.program_file_updates
            for item in sessions
            if item.kind == "fresh_control"
        ),
        "replayed_examples": sum(item.replayed_examples for item in sessions),
        "wall_seconds": sum(item.wall_seconds for item in sessions),
        "orchestration_wall_seconds": time.perf_counter() - started,
        "controller_digest_before": controller_digest,
        "controller_digest_after": controller_after,
        "controller_frozen": controller_digest == controller_after,
        "program_digest": machine.program_digest(),
        "admissions": admissions,
        "transfer_controls": transfer_controls,
        "bank": str(bank_path) if bank_path.exists() else None,
        "bank_sha256": _sha256(bank_path) if bank_path.exists() else None,
        "transfer_ratio_against_fresh": (
            None
            if not transfer_controls
            else transfer_controls[-1]["accuracy_ratio"]
        ),
    }
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--neural-workshop", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--n-back", type=int, default=1)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--cells", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--stable-sessions", type=int, default=3)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--maximum-sessions-per-rung", type=int, default=12)
    parser.add_argument("--retention-sessions", type=int, default=1)
    parser.add_argument("--program-learning-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    arguments = parser.parse_args()
    torch.manual_seed(arguments.seed)
    payload = load_temporal_controller_artifact(arguments.controller_artifact)
    machine = build_pretrained_controller_program_machine(
        payload,
        learning_rate=arguments.program_learning_rate,
        sample=True,
        inherit_program_prior=False,
    )
    policy = CurriculumPolicy(
        active_cell_ladder=tuple(arguments.cells),
        mastery_threshold=arguments.threshold,
        stable_sessions=arguments.stable_sessions,
        minimum_verifier_bits_per_session=arguments.minimum_bits,
        maximum_training_sessions_per_rung=arguments.maximum_sessions_per_rung,
        retention_sessions=arguments.retention_sessions,
    )
    campaign = run_cell_curriculum(
        machine,
        payload,
        arguments.neural_workshop,
        arguments.output_dir,
        policy=policy,
        grid_size=arguments.grid_size,
        n_back=arguments.n_back,
        trials=arguments.trials,
        seed=arguments.seed,
        resume=arguments.resume,
    )
    print(campaign)


if __name__ == "__main__":
    main()
