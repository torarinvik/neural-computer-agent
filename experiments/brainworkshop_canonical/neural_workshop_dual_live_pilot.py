"""Live Neural Workshop Dual N-Back on public pixels plus public PCM.

Position is the play-field crop. Audio is the waveform Neural Workshop
queued for that stimulus, not a letter ID. Each stream uses the frozen
two-way decoder; bits are packed onto the two public ports.
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
from .neural_workshop_live import (
    NeuralWorkshopLiveConfig,
    build_neural_workshop_environment,
    run_neural_workshop_live_lifetime,
)

NEURAL_WORKSHOP_DUAL_LIVE_SCHEMA = "neural-computer.neural-workshop-dual-live.v1"


def _dual_config(machine, *, n_back: int, trials: int) -> NeuralWorkshopLiveConfig:
    return NeuralWorkshopLiveConfig(
        grid_size=3,
        active_cells=8,
        n_back=n_back,
        trials=trials,
        event_width=machine.event_width,
        source_key_width=machine.source_key_width,
        game_mode=2,
        action_ports=2,
    )


def run_neural_workshop_dual(
    controller_payload: dict[str, object],
    neural_workshop_directory: Path,
    *,
    trials: int = 60,
    seed: int = 98_017,
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

    def evaluate(n_back: int, depth: int, suffix: str) -> dict[str, object]:
        artifact = compose_recursive_temporal_program(primitive, depth)
        machine.load_recursive_program_artifact(
            artifact, controller_digest=machine.controller_digest()
        )
        config = _dual_config(machine, n_back=n_back, trials=trials)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=seed + n_back + depth
        )
        report = run_neural_workshop_live_lifetime(
            machine,
            config,
            seed=seed + n_back + depth,
            environment=environment,
            verifier=verifier,
            learn=False,
            sample=False,
        )
        if (
            not report.audio_payloads
            or len(report.audio_payloads) != len(report.event_payloads)
        ):
            raise RuntimeError("Dual live lost the public audio stream")
        if not report.controller_frozen:
            raise RuntimeError("Dual live mutated the frozen controller")
        return {
            "label": suffix,
            "n_back": n_back,
            "composition_depth": depth,
            "accuracy": report.verifier_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
            "audio_events": len(report.audio_payloads),
            "vision_events": len(report.event_payloads),
            "optimizer_updates": report.optimizer_updates,
            "program_file_updates": report.program_file_updates,
            "controller_frozen": report.controller_frozen,
        }

    one = evaluate(1, 1, "dual-1back")
    two = evaluate(2, 2, "dual-2back")
    wrong = evaluate(2, 1, "wrong-depth")
    return {
        "schema": NEURAL_WORKSHOP_DUAL_LIVE_SCHEMA,
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
    parser.add_argument("--neural-workshop", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=98_017)
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
    report = run_neural_workshop_dual(
        load_temporal_controller_artifact(arguments.controller_artifact),
        arguments.neural_workshop,
        trials=arguments.trials,
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
