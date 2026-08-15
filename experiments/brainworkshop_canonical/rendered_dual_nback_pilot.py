"""Dual N-Back with the frozen two-way decoder packed per source.

Each stream keeps its own PREVIOUS history. The same binary decoder decides
match/no-match for that stream. Bits are packed in bind order, which follows
the public stream list. No four-way head is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from neural_computer import (
    compose_recursive_temporal_program,
    one_hot_temporal_address_artifact,
    recursive_temporal_primitive,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from .rendered_live import run_rendered_live_lifetime

RENDERED_DUAL_NBACK_SCHEMA = "neural-computer.rendered-dual-nback.v1"


def run_rendered_dual_nback(
    controller_payload: dict[str, object],
    *,
    steps: int = 48,
    seed: int = 96_017,
) -> dict[str, object]:
    started = perf_counter()
    machine = build_recursive_temporal_program_machine(
        controller_payload,
        sample=False,
        max_sources=2,
        pack_source_actions=True,
    )
    primitive = recursive_temporal_primitive(
        one_hot_temporal_address_artifact(0, machine.max_history)
    )
    encoders = RenderedBrainWorkshopEncoders(
        machine.event_width, source_key_width=machine.source_key_width
    )
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)

    def evaluate(n_back: int, depth: int) -> dict[str, object]:
        machine.load_recursive_program_artifact(
            compose_recursive_temporal_program(primitive, depth),
            controller_digest=machine.controller_digest(),
        )
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=n_back,
                steps=steps,
                streams=("vision", "audio"),
            ),
            seed=seed + n_back,
            learn=False,
            sample=False,
        )
        return {
            "n_back": n_back,
            "composition_depth": depth,
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
            "optimizer_updates": report.optimizer_updates,
            "program_file_updates": report.program_file_updates,
        }

    one = evaluate(1, 1)
    two = evaluate(2, 2)
    wrong = evaluate(2, 1)
    return {
        "schema": RENDERED_DUAL_NBACK_SCHEMA,
        "action_count": machine.action_count,
        "decoder_key_count": machine.decoder.key_count,
        "max_sources": machine.max_sources,
        "controller_digest": machine.controller_digest(),
        "dual_1back": one,
        "dual_2back": two,
        "wrong_depth_control": wrong,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=96_017)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    arguments = parser.parse_args()
    report = run_rendered_dual_nback(
        load_temporal_controller_artifact(arguments.controller_artifact),
        steps=arguments.steps,
        seed=arguments.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)


if __name__ == "__main__":
    main()
