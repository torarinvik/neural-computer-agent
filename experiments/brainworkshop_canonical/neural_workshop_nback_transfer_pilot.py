"""Held-out live 2-back transfer and causal controls for Neural Workshop."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer import DEFAULT_AGENT_BANK_FILENAME, ExternalTemporalProgramBank

from .controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_live import (
    NeuralWorkshopIntervention,
    NeuralWorkshopLiveConfig,
    NeuralWorkshopLiveReport,
    build_neural_workshop_environment,
    run_neural_workshop_live_lifetime,
)
from .physical_program_bank import learned_event_context
from .rendered_live import PretrainedControllerProgramMachine

NEURAL_WORKSHOP_TRANSFER_SCHEMA = "neural-computer.neural-workshop-nback-transfer.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_session(
    report: NeuralWorkshopLiveReport, *, threshold: float, minimum_bits: int
) -> bool:
    return bool(
        report.verifier_accuracy is not None
        and report.verifier_accuracy >= threshold
        and report.unique_verifier_bits >= minimum_bits
        and report.controller_frozen
    )


def _mastered(
    reports: list[NeuralWorkshopLiveReport],
    *,
    threshold: float,
    minimum_bits: int,
    stable_sessions: int,
) -> bool:
    return len(reports) >= stable_sessions and all(
        _stable_session(item, threshold=threshold, minimum_bits=minimum_bits)
        for item in reports[-stable_sessions:]
    )


def _bits_to_mastery(
    reports: list[NeuralWorkshopLiveReport],
    *,
    threshold: float,
    minimum_bits: int,
    stable_sessions: int,
) -> int | None:
    bits = 0
    for index, report in enumerate(reports):
        bits += report.unique_verifier_bits
        if _mastered(
            reports[: index + 1],
            threshold=threshold,
            minimum_bits=minimum_bits,
            stable_sessions=stable_sessions,
        ):
            return bits
    return None


def _load_inherited_program(
    machine: PretrainedControllerProgramMachine, checkpoint: Path
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    external = payload.get("external_program") if isinstance(payload, dict) else None
    if not isinstance(external, dict):
        raise TypeError("source checkpoint lacks an external temporal program")
    machine.load_external_program_payload(external)
    machine.reset_history()


def _save_report(report: NeuralWorkshopLiveReport, path: Path) -> None:
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")


def run_nback_transfer(
    controller_payload: dict[str, object],
    source_checkpoint: Path,
    source_bank_path: Path,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    active_cells: int = 2,
    trials: int = 60,
    sessions: int = 6,
    control_sessions: int = 3,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    stable_sessions: int = 3,
    learning_rate: float = 0.3,
    seed: int = 51_017,
) -> dict[str, Any]:
    """Compare inherited and fresh 2-back acquisition, then run controls."""

    if (
        min(active_cells, trials, sessions, control_sessions, minimum_bits, stable_sessions)
        < 1
        or stable_sessions > sessions
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("2-back transfer settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    config = NeuralWorkshopLiveConfig(
        grid_size=3,
        active_cells=active_cells,
        n_back=2,
        trials=trials,
        event_width=int(controller_payload["configuration"]["event_width"]),
        source_key_width=int(
            controller_payload["configuration"]["source_key_width"]
        ),
    )

    def new_machine(*, inherited: bool) -> PretrainedControllerProgramMachine:
        machine = build_pretrained_controller_program_machine(
            controller_payload,
            learning_rate=learning_rate,
            sample=True,
            inherit_program_prior=False,
        )
        if inherited:
            _load_inherited_program(machine, source_checkpoint)
        return machine

    def execute(
        machine: PretrainedControllerProgramMachine,
        session_seed: int,
        *,
        learn: bool,
        sample: bool,
        intervention: NeuralWorkshopIntervention | None = None,
    ) -> NeuralWorkshopLiveReport:
        torch.manual_seed(session_seed)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        return run_neural_workshop_live_lifetime(
            machine,
            config,
            seed=session_seed,
            environment=environment,
            verifier=verifier,
            learn=learn,
            sample=sample,
            intervention=intervention,
        )

    started = time.perf_counter()
    inherited = new_machine(inherited=True)
    fresh = new_machine(inherited=False)
    inherited_start = inherited.program_digest()
    fresh_start = fresh.program_digest()
    inherited_reports: list[NeuralWorkshopLiveReport] = []
    fresh_reports: list[NeuralWorkshopLiveReport] = []
    for index in range(sessions):
        session_seed = seed + index
        inherited_report = execute(
            inherited, session_seed, learn=True, sample=True
        )
        fresh_report = execute(fresh, session_seed, learn=True, sample=True)
        inherited_reports.append(inherited_report)
        fresh_reports.append(fresh_report)
        _save_report(
            inherited_report,
            output_directory / f"inherited-{index + 1:03d}.json",
        )
        _save_report(
            fresh_report, output_directory / f"fresh-{index + 1:03d}.json"
        )
        torch.save(
            inherited.external_program_payload(),
            output_directory / "inherited-program.pt",
        )
        print(
            {
                "session": index + 1,
                "inherited": inherited_report.verifier_accuracy,
                "fresh": fresh_report.verifier_accuracy,
                "inherited_mastered": _mastered(
                    inherited_reports,
                    threshold=threshold,
                    minimum_bits=minimum_bits,
                    stable_sessions=stable_sessions,
                ),
                "fresh_mastered": _mastered(
                    fresh_reports,
                    threshold=threshold,
                    minimum_bits=minimum_bits,
                    stable_sessions=stable_sessions,
                ),
            },
            flush=True,
        )
        if _mastered(
            inherited_reports,
            threshold=threshold,
            minimum_bits=minimum_bits,
            stable_sessions=stable_sessions,
        ) and _mastered(
            fresh_reports,
            threshold=threshold,
            minimum_bits=minimum_bits,
            stable_sessions=stable_sessions,
        ):
            break

    inherited_mastered = _mastered(
        inherited_reports,
        threshold=threshold,
        minimum_bits=minimum_bits,
        stable_sessions=stable_sessions,
    )
    inherited_retention: NeuralWorkshopLiveReport | None = None
    if inherited_mastered:
        before = inherited.program_digest()
        inherited_retention = execute(
            inherited, seed + 5_000, learn=False, sample=False
        )
        if inherited.program_digest() != before:
            raise RuntimeError("2-back retention mutated the inherited program")
        _save_report(inherited_retention, output_directory / "retention.json")

    learned_payload = inherited.external_program_payload()

    def learned_control(
        name: str, intervention: NeuralWorkshopIntervention
    ) -> NeuralWorkshopLiveReport:
        machine = new_machine(inherited=False)
        machine.load_external_program_payload(learned_payload)
        report = execute(
            machine,
            seed + 10_000 + len(control_reports),
            learn=False,
            sample=False,
            intervention=intervention,
        )
        _save_report(report, output_directory / f"control-{name}.json")
        return report

    control_reports: dict[str, list[NeuralWorkshopLiveReport]] = {}
    for name, intervention in (
        ("passive", NeuralWorkshopIntervention(action="passive", seed=seed)),
        ("random_action", NeuralWorkshopIntervention(action="random", seed=seed)),
        ("reversal", NeuralWorkshopIntervention(action="reversed", seed=seed)),
        (
            "memory_corruption",
            NeuralWorkshopIntervention(reset_history_each_tick=True, seed=seed),
        ),
    ):
        control_reports[name] = [learned_control(name, intervention)]

    for control_index, (name, intervention) in enumerate(
        (
            (
                "reward_shuffled",
                NeuralWorkshopIntervention(reward="shuffled", seed=seed + 1),
            ),
            ("missing_evidence", NeuralWorkshopIntervention(reward="missing", seed=seed + 2)),
            ("action_shuffled", NeuralWorkshopIntervention(action="random", seed=seed + 3)),
        )
    ):
        machine = new_machine(inherited=False)
        reports = []
        for index in range(control_sessions):
            report = execute(
                machine,
                seed + 20_000 + control_index * 1_000 + index,
                learn=True,
                sample=True,
                intervention=intervention,
            )
            reports.append(report)
            _save_report(
                report,
                output_directory / f"control-{name}-{index + 1:03d}.json",
            )
        control_reports[name] = reports

    inherited_bits = _bits_to_mastery(
        inherited_reports,
        threshold=threshold,
        minimum_bits=minimum_bits,
        stable_sessions=stable_sessions,
    )
    fresh_bits = _bits_to_mastery(
        fresh_reports,
        threshold=threshold,
        minimum_bits=minimum_bits,
        stable_sessions=stable_sessions,
    )
    first_inherited = inherited_reports[0].verifier_accuracy
    first_fresh = fresh_reports[0].verifier_accuracy
    transfer = {
        "first_session_accuracy_ratio": (
            None
            if first_inherited is None or first_fresh in (None, 0.0)
            else first_inherited / first_fresh
        ),
        "inherited_bits_to_mastery": inherited_bits,
        "fresh_bits_to_mastery": fresh_bits,
        "bits_efficiency_ratio": (
            None
            if inherited_bits is None or fresh_bits is None
            else fresh_bits / inherited_bits
        ),
    }

    bank_admission = None
    output_bank = output_directory / DEFAULT_AGENT_BANK_FILENAME
    if inherited_mastered and inherited_retention is not None and _stable_session(
        inherited_retention, threshold=threshold, minimum_bits=minimum_bits
    ):
        bank = ExternalTemporalProgramBank.load_bank(source_bank_path)
        if bank.controller_digest != inherited.controller_digest():
            raise ValueError("source bank targets another frozen controller")
        stable = inherited_reports[-stable_sessions:]
        context = learned_event_context(
            [row for report in stable for row in report.event_payloads],
            width=inherited.event_width,
        )
        receipt = bank.admit(
            inherited.admitted_program_artifact(),
            context,
            [float(report.verifier_accuracy) for report in stable],
            threshold=threshold,
            min_observations=stable_sessions,
            min_stable_observations=stable_sessions,
        )
        if not receipt.accepted:
            raise RuntimeError(f"mastered 2-back program was rejected: {receipt.reason}")
        bank.save_bank(output_bank)
        bank_admission = {
            **receipt.payload(),
            "program_count": bank.program_count,
            "sha256": _sha256(output_bank),
        }

    def summaries(reports: list[NeuralWorkshopLiveReport]) -> list[dict[str, Any]]:
        return [
            {
                "accuracy": item.verifier_accuracy,
                "unique_verifier_bits": item.unique_verifier_bits,
                "learner_outcome_bits": item.learner_outcome_bits,
                "program_file_updates": item.program_file_updates,
                "controller_frozen": item.controller_frozen,
            }
            for item in reports
        ]

    result = {
        "schema": NEURAL_WORKSHOP_TRANSFER_SCHEMA,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "source_bank": str(source_bank_path),
        "source_bank_sha256": _sha256(source_bank_path),
        "configuration": {
            "n_back": 2,
            "grid_size": 3,
            "active_cells": active_cells,
            "trials": trials,
            "threshold": threshold,
            "minimum_bits": minimum_bits,
            "stable_sessions": stable_sessions,
        },
        "inherited": summaries(inherited_reports),
        "fresh": summaries(fresh_reports),
        "inherited_mastered": inherited_mastered,
        "fresh_mastered": _mastered(
            fresh_reports,
            threshold=threshold,
            minimum_bits=minimum_bits,
            stable_sessions=stable_sessions,
        ),
        "retention": (
            None
            if inherited_retention is None
            else summaries([inherited_retention])[0]
        ),
        "transfer": transfer,
        "controls": {
            name: {
                "sessions": summaries(reports),
                "mastered": _mastered(
                    reports,
                    threshold=threshold,
                    minimum_bits=minimum_bits,
                    stable_sessions=stable_sessions,
                ),
            }
            for name, reports in control_reports.items()
        },
        "controller_digest": inherited.controller_digest(),
        "controller_optimizer_updates": 0,
        "inherited_program_digest_before": inherited_start,
        "inherited_program_digest_after": inherited.program_digest(),
        "fresh_program_digest_before": fresh_start,
        "fresh_program_digest_after": fresh.program_digest(),
        "bank_admission": bank_admission,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_directory / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--neural-workshop", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-cells", type=int, default=2)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--control-sessions", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--stable-sessions", type=int, default=3)
    parser.add_argument("--program-learning-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=51_017)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    arguments = parser.parse_args()
    payload = load_temporal_controller_artifact(arguments.controller_artifact)
    report = run_nback_transfer(
        payload,
        arguments.source_checkpoint,
        arguments.source_bank,
        arguments.neural_workshop,
        arguments.output_dir,
        active_cells=arguments.active_cells,
        trials=arguments.trials,
        sessions=arguments.sessions,
        control_sessions=arguments.control_sessions,
        threshold=arguments.threshold,
        minimum_bits=arguments.minimum_bits,
        stable_sessions=arguments.stable_sessions,
        learning_rate=arguments.program_learning_rate,
        seed=arguments.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
