"""Verify PREVIOUS composition on live Neural Workshop without task training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer import (
    DEFAULT_AGENT_BANK_FILENAME,
    RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
    ExternalTemporalProgramBank,
    compose_recursive_temporal_program,
    recursive_temporal_primitive,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
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

NEURAL_WORKSHOP_RECURSIVE_TRANSFER_SCHEMA = (
    "neural-computer.neural-workshop-recursive-transfer.v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_report(report: NeuralWorkshopLiveReport, path: Path) -> None:
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")


def run_recursive_transfer(
    controller_payload: dict[str, object],
    source_bank_path: Path,
    source_slot: int,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    active_cells: int = 2,
    trials: int = 60,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    stable_sessions: int = 3,
    seed: int = 71_017,
) -> dict[str, Any]:
    """Search recursive depths using only attempted public scalar outcomes."""

    if (
        min(active_cells, trials, minimum_bits, stable_sessions) < 1
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("recursive transfer settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    source_bank = ExternalTemporalProgramBank.load_bank(source_bank_path)
    source_artifact = source_bank.artifact(source_slot)
    primitive = recursive_temporal_primitive(source_artifact)
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    if not machine.accepts_controller_digest(source_bank.controller_digest):
        raise ValueError("source primitive targets another legacy controller")
    machine.load_legacy_primitive_artifact(
        source_artifact, controller_digest=source_bank.controller_digest
    )
    recursive_controller_digest = machine.controller_digest()
    controller_before = machine.controller_digest()
    primitive_digest = primitive.digest()
    started = time.perf_counter()

    def execute(
        artifact,
        *,
        n_back: int,
        session_seed: int,
        intervention: NeuralWorkshopIntervention | None = None,
    ) -> NeuralWorkshopLiveReport:
        machine.load_recursive_program_artifact(
            artifact, controller_digest=recursive_controller_digest
        )
        config = NeuralWorkshopLiveConfig(
            grid_size=3,
            active_cells=active_cells,
            n_back=n_back,
            trials=trials,
            event_width=machine.event_width,
            source_key_width=machine.source_key_width,
        )
        torch.manual_seed(session_seed)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        report = run_neural_workshop_live_lifetime(
            machine,
            config,
            seed=session_seed,
            environment=environment,
            verifier=verifier,
            learn=False,
            sample=False,
            intervention=intervention,
        )
        if report.program_file_updates != 0:
            raise RuntimeError("recursive evaluation changed its external program")
        return report

    # Re-verify that the migrated depth-one primitive retains its source skill.
    primitive_retention = [
        execute(primitive, n_back=1, session_seed=seed + index)
        for index in range(stable_sessions)
    ]
    for index, report in enumerate(primitive_retention, start=1):
        _save_report(report, output_directory / f"primitive-retention-{index:03d}.json")

    candidate_reports: list[tuple[int, Any, NeuralWorkshopLiveReport]] = []
    selected = None
    search_bits = 0
    # This frontier is structural and receives no task/rule identity. It tries
    # the admitted primitive, then progressively deeper self-compositions.
    for depth in range(1, machine.max_history + 1):
        artifact = compose_recursive_temporal_program(primitive, depth)
        report = execute(
            artifact,
            n_back=2,
            session_seed=seed + 1_000 + depth - 1,
        )
        candidate_reports.append((depth, artifact, report))
        search_bits += report.unique_verifier_bits
        _save_report(report, output_directory / f"candidate-depth-{depth:02d}.json")
        if (
            report.verifier_accuracy is not None
            and report.verifier_accuracy >= threshold
            and report.unique_verifier_bits >= minimum_bits
        ):
            selected = (depth, artifact, report)
            break
    if selected is None:
        raise RuntimeError("recursive candidate frontier found no supported program")
    selected_depth, selected_artifact, first_selected_report = selected

    selected_reports = [first_selected_report]
    for index in range(1, stable_sessions):
        report = execute(
            selected_artifact,
            n_back=2,
            session_seed=seed + 2_000 + index - 1,
        )
        selected_reports.append(report)
        _save_report(report, output_directory / f"selected-retention-{index:03d}.json")
    stable = all(
        report.verifier_accuracy is not None
        and report.verifier_accuracy >= threshold
        and report.unique_verifier_bits >= minimum_bits
        for report in selected_reports
    )

    controls = {
        "wrong_depth": execute(
            compose_recursive_temporal_program(primitive, 1),
            n_back=2,
            session_seed=seed + 3_000,
        ),
        "over_composed": execute(
            compose_recursive_temporal_program(primitive, 3),
            n_back=2,
            session_seed=seed + 3_001,
        ),
        "memory_corruption": execute(
            selected_artifact,
            n_back=2,
            session_seed=seed + 3_002,
            intervention=NeuralWorkshopIntervention(
                reset_history_each_tick=True, seed=seed + 3_002
            ),
        ),
        "reversal": execute(
            selected_artifact,
            n_back=2,
            session_seed=seed + 3_003,
            intervention=NeuralWorkshopIntervention(
                action="reversed", seed=seed + 3_003
            ),
        ),
    }
    for name, report in controls.items():
        _save_report(report, output_directory / f"control-{name}.json")

    recursive_bank = ExternalTemporalProgramBank(
        machine.event_width,
        machine.max_history,
        controller_digest=recursive_controller_digest,
        generalization_tolerance=0.0,
        mastery_threshold=threshold,
        min_mastery_observations=stable_sessions,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )

    def admit(artifact, reports: list[NeuralWorkshopLiveReport]):
        context = learned_event_context(
            [row for report in reports for row in report.event_payloads],
            width=machine.event_width,
        )
        receipt = recursive_bank.admit(
            artifact,
            context,
            [float(report.verifier_accuracy) for report in reports],
            threshold=threshold,
            min_observations=stable_sessions,
            min_stable_observations=stable_sessions,
        )
        if not receipt.accepted:
            raise RuntimeError(f"verified recursive artifact was rejected: {receipt.reason}")
        return receipt

    primitive_receipt = admit(primitive, primitive_retention)
    selected_receipt = admit(selected_artifact, selected_reports) if stable else None
    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    recursive_bank.save_bank(bank_path)

    def summary(report: NeuralWorkshopLiveReport) -> dict[str, Any]:
        return {
            "accuracy": report.verifier_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
            "program_file_updates": report.program_file_updates,
            "controller_frozen": report.controller_frozen,
        }

    result = {
        "schema": NEURAL_WORKSHOP_RECURSIVE_TRANSFER_SCHEMA,
        "source_bank": str(source_bank_path),
        "source_bank_sha256": _sha256(source_bank_path),
        "source_slot": source_slot,
        "legacy_controller_digest": machine.legacy_controller_digest(),
        "recursive_controller_digest": recursive_controller_digest,
        "controller_frozen": machine.controller_digest() == controller_before,
        "primitive_digest": primitive_digest,
        "primitive_retention": [summary(report) for report in primitive_retention],
        "candidate_search": [
            {"depth": depth, **summary(report)}
            for depth, _artifact, report in candidate_reports
        ],
        "selected_depth": selected_depth,
        "selected_program_digest": selected_artifact.digest(),
        "selected_retention": [summary(report) for report in selected_reports],
        "stable": stable,
        "search_unique_verifier_bits": search_bits,
        "selected_unique_verifier_bits_to_stability": sum(
            report.unique_verifier_bits for report in selected_reports
        ),
        "optimizer_updates": 0,
        "program_file_updates": 0,
        "replayed_examples": 0,
        "controls": {name: summary(report) for name, report in controls.items()},
        "primitive_admission": primitive_receipt.payload(),
        "selected_admission": (
            None if selected_receipt is None else selected_receipt.payload()
        ),
        "bank": str(bank_path),
        "bank_program_count": recursive_bank.program_count,
        "bank_sha256": _sha256(bank_path),
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
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--source-slot", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-cells", type=int, default=2)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--stable-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=71_017)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    arguments = parser.parse_args()
    controller = load_temporal_controller_artifact(arguments.controller_artifact)
    report = run_recursive_transfer(
        controller,
        arguments.source_bank,
        arguments.source_slot,
        arguments.neural_workshop,
        arguments.output_dir,
        active_cells=arguments.active_cells,
        trials=arguments.trials,
        threshold=arguments.threshold,
        minimum_bits=arguments.minimum_bits,
        stable_sessions=arguments.stable_sessions,
        seed=arguments.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
