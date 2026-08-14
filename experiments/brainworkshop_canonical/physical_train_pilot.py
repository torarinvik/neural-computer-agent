"""Run or resume a bounded physical Position N-Back training campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_computer import DEFAULT_AGENT_BANK_FILENAME

from .controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from .physical_live import (
    PhysicalBrainWorkshopConfig,
    compile_macos_capture_helper,
    compile_macos_keypress_helper,
)
from .physical_program_bank import admit_physical_training_program
from .physical_train import run_physical_training_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10)
    # The public Brain Workshop launcher currently uses 60 one-second trials.
    # Four seconds covers start/finish settling without adding another session.
    parser.add_argument("--seconds-per-session", type=float, default=64.0)
    parser.add_argument("--tick-hz", type=float, default=6.0)
    parser.add_argument(
        "--capture-backend",
        choices=("native", "screencapture", "ffmpeg"),
        default="native",
    )
    parser.add_argument("--action-delay-seconds", type=float, default=0.0)
    parser.add_argument("--program-learning-rate", type=float, default=0.3)
    parser.add_argument(
        "--capture-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-capture"),
    )
    parser.add_argument("--rolling-window", type=int, default=44)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/brainworkshop-physical-training"),
    )
    parser.add_argument(
        "--program-bank",
        type=Path,
        help=f"defaults to <output-dir>/{DEFAULT_AGENT_BANK_FILENAME}",
    )
    parser.add_argument("--admission-min-lifetimes", type=int, default=8)
    parser.add_argument(
        "--keypress-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-keypress"),
    )
    parser.add_argument("--archive-evidence", action="store_true")
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            Path(__file__).parents[2]
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--inherit-program-prior",
        action="store_true",
        help="explicit transfer control; default task program is uniform",
    )
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    helper = compile_macos_keypress_helper(arguments.keypress_helper)
    capture_helper = (
        compile_macos_capture_helper(arguments.capture_helper)
        if arguments.capture_backend == "native"
        else None
    )
    config = PhysicalBrainWorkshopConfig(
        tick_hz=arguments.tick_hz,
        capture_backend=arguments.capture_backend,
        action_delay_seconds=arguments.action_delay_seconds,
        capture_helper=capture_helper,
        keypress_helper=helper,
        evidence_directory=(
            arguments.output_dir / "evidence" if arguments.archive_evidence else None
        ),
    )
    controller_payload = load_temporal_controller_artifact(
        arguments.controller_artifact
    )
    machine = build_pretrained_controller_program_machine(
        controller_payload,
        learning_rate=arguments.program_learning_rate,
        sample=True,
        inherit_program_prior=arguments.inherit_program_prior,
    )
    if (
        machine.event_width != config.event_width
        or machine.source_key_width != config.source_key_width
        or machine.action_count != 2
    ):
        raise ValueError("controller artifact is incompatible with physical frontend")
    report = run_physical_training_campaign(
        machine,
        config,
        sessions=arguments.sessions,
        seconds_per_session=arguments.seconds_per_session,
        seed=arguments.seed,
        output_directory=arguments.output_dir,
        rolling_window=arguments.rolling_window,
        resume=arguments.resume,
    )
    bank_path = (
        arguments.program_bank
        or arguments.output_dir / DEFAULT_AGENT_BANK_FILENAME
    )
    admission = admit_physical_training_program(
        machine,
        report,
        arguments.output_dir,
        bank_path,
        min_lifetimes=arguments.admission_min_lifetimes,
    )
    final = report.sessions[-1]
    print(
        {
            "sessions": report.completed_sessions,
            "unique_public_outcomes": report.unique_public_outcomes,
            "optimizer_updates": report.optimizer_updates,
            "program_file_updates": report.program_file_updates,
            "learning_target": report.learning_target,
            "controller_frozen": report.controller_frozen,
            "controller_digest_before": report.controller_digest_before,
            "controller_digest_after": report.controller_digest_after,
            "program_digest_before": report.program_digest_before,
            "program_digest_after": report.program_digest_after,
            "replayed_examples": report.replayed_examples,
            "cumulative_accuracy": final.cumulative_accuracy,
            "rolling_accuracy": final.rolling_accuracy,
            "wall_seconds": report.wall_seconds,
            "capture_backend": arguments.capture_backend,
            "output_directory": str(arguments.output_dir),
            "program_admitted": admission.accepted,
            "program_bank_slot": admission.slot,
            "program_bank": str(bank_path) if admission.accepted else None,
            "admission_reason": admission.reason,
        }
    )


if __name__ == "__main__":
    main()
