"""Optional read-only transfer against Neural Workshop's public window."""

from __future__ import annotations

import argparse
from pathlib import Path

from .controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from .physical_live import (
    PhysicalBrainWorkshopConfig,
    compile_macos_capture_helper,
    compile_macos_keypress_helper,
    run_physical_brainworkshop_lifetime,
    save_physical_report,
)
from .physical_train import load_physical_training_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--seconds", type=float, default=64.0)
    parser.add_argument("--tick-hz", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--program-checkpoint",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/brainworkshop_position1back_2cell_program_seed17.pt"
        ),
    )
    parser.add_argument(
        "--capture-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-capture"),
    )
    parser.add_argument(
        "--keypress-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-keypress"),
    )
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    arguments = parser.parse_args()

    controller = load_temporal_controller_artifact(arguments.controller_artifact)
    machine = build_pretrained_controller_program_machine(
        controller,
        learning_rate=0.3,
        sample=False,
    )
    load_physical_training_checkpoint(machine, arguments.program_checkpoint)
    machine.learning_enabled = False
    controller_before = machine.controller_digest()
    program_before = machine.program_digest()
    updates_before = machine.program_file_updates
    config = PhysicalBrainWorkshopConfig(
        tick_hz=arguments.tick_hz,
        capture_backend="native",
        capture_helper=compile_macos_capture_helper(arguments.capture_helper),
        keypress_helper=compile_macos_keypress_helper(arguments.keypress_helper),
        evidence_directory=arguments.evidence_dir,
    )
    report = run_physical_brainworkshop_lifetime(
        machine,
        config,
        seconds=arguments.seconds,
        seed=arguments.seed,
        start_session=True,
    )
    if (
        machine.controller_digest() != controller_before
        or machine.program_digest() != program_before
        or machine.program_file_updates != updates_before
    ):
        raise RuntimeError("read-only physical transfer evaluation changed weights")
    if arguments.report_out is not None:
        save_physical_report(report, arguments.report_out)
    print(
        {
            "input_events": report.input_events,
            "emitted_actions": report.emitted_actions,
            "unique_public_outcomes": report.unique_public_outcomes,
            "accuracy": (
                sum(report.rewards) / len(report.rewards)
                if report.rewards
                else None
            ),
            "controller_updates": 0,
            "program_updates": 0,
            "controller_digest": controller_before,
            "program_digest": program_before,
            "deadline_misses": report.deadline_misses,
            "tick_seconds_p50": report.total_seconds_p50,
            "tick_seconds_p99": report.total_seconds_p99,
        }
    )


if __name__ == "__main__":
    main()
