"""Optional human-parity Position N-Back I/O against Neural Workshop's window."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .physical_live import (
    PhysicalBrainWorkshopConfig,
    compile_macos_keypress_helper,
    run_physical_brainworkshop_lifetime,
    save_physical_report,
)
from .rendered_live import FrozenControllerProgramMachine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--tick-hz", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--start-session", action="store_true")
    parser.add_argument("--applescript-output", action="store_true")
    parser.add_argument(
        "--keypress-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-keypress"),
    )
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    keypress_helper = (
        None
        if arguments.applescript_output
        else compile_macos_keypress_helper(arguments.keypress_helper)
    )
    config = PhysicalBrainWorkshopConfig(
        tick_hz=arguments.tick_hz,
        keypress_helper=keypress_helper,
        evidence_directory=arguments.evidence_dir,
    )
    machine = FrozenControllerProgramMachine(
        config.event_width,
        source_key_width=config.source_key_width,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=16,
        hidden=24,
        learning_rate=3e-3,
        sample=True,
    )
    report = run_physical_brainworkshop_lifetime(
        machine,
        config,
        seconds=arguments.seconds,
        seed=arguments.seed,
        start_session=arguments.start_session,
    )
    if arguments.report_out is not None:
        save_physical_report(report, arguments.report_out)
    summary = report.as_dict()
    summary["evidence_digests"] = f"{len(report.evidence_digests)} windows"
    print(summary)


if __name__ == "__main__":
    main()
