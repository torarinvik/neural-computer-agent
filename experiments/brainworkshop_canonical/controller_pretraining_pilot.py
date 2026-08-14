"""Build a reusable temporal controller before physical task acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path

from .controller_pretraining import (
    pretrain_previous_event_controller,
    save_temporal_controller_artifact,
    save_temporal_controller_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1_001)
    parser.add_argument("--frontend-families", type=int, default=160)
    parser.add_argument("--lifetimes-per-family", type=int, default=3)
    parser.add_argument("--steps-per-lifetime", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/neural-computer-temporal-controller.pt"),
    )
    parser.add_argument("--report-out", type=Path)
    arguments = parser.parse_args()

    payload, report = pretrain_previous_event_controller(
        seed=arguments.seed,
        frontend_families=arguments.frontend_families,
        lifetimes_per_family=arguments.lifetimes_per_family,
        steps_per_lifetime=arguments.steps_per_lifetime,
    )
    save_temporal_controller_artifact(payload, arguments.output)
    if arguments.report_out is not None:
        save_temporal_controller_report(report, arguments.report_out)
    print({**report.as_dict(), "artifact": str(arguments.output)})


if __name__ == "__main__":
    main()
