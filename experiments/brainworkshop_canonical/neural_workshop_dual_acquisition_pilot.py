"""Acquire Neural Workshop Dual 1-Back from a blank file.

Public pixels and public PCM are the only stimuli. The program starts
uniform and may update only its relative address. After Dual 1-Back the
learned primitive is composed and evaluated on Dual 2-Back. A matched
fresh learner trains Dual 2-Back from scratch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from neural_computer import compose_recursive_temporal_program

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_live import (
    NeuralWorkshopIntervention,
    NeuralWorkshopLiveConfig,
    build_neural_workshop_environment,
    run_neural_workshop_live_lifetime,
)

NEURAL_WORKSHOP_DUAL_ACQUISITION_SCHEMA = (
    "neural-computer.neural-workshop-dual-acquisition.v1"
)


def _bits_to_threshold(
    sessions: list[dict[str, object]],
    *,
    threshold: float,
    minimum_bits: int,
) -> int | None:
    total = 0
    for index, session in enumerate(sessions):
        bits = int(session["unique_verifier_bits"])
        total += bits
        if bits < minimum_bits:
            continue
        if all(
            float(item["packed_exact_accuracy"] or 0.0) >= threshold
            and int(item["unique_verifier_bits"]) >= minimum_bits
            for item in sessions[index:]
        ):
            return total
    return None


def _dual_config(
    machine, *, n_back: int, trials: int, visible: bool = False
) -> NeuralWorkshopLiveConfig:
    return NeuralWorkshopLiveConfig(
        grid_size=3,
        active_cells=8,
        n_back=n_back,
        trials=trials,
        event_width=machine.event_width,
        source_key_width=machine.source_key_width,
        game_mode=2,
        action_ports=2,
        visible=visible,
    )


def _new_machine(controller_payload: dict[str, object], *, learning_rate: float):
    return build_recursive_temporal_program_machine(
        controller_payload,
        learning_rate=learning_rate,
        sample=True,
        max_sources=2,
        pack_source_actions=True,
    )


def _run_session(
    machine,
    neural_workshop_directory: Path,
    *,
    n_back: int,
    trials: int,
    seed: int,
    learn: bool,
    sample: bool,
    intervention: NeuralWorkshopIntervention | None = None,
    visible: bool = False,
):
    config = _dual_config(
        machine, n_back=n_back, trials=trials, visible=visible
    )
    environment, verifier = build_neural_workshop_environment(
        neural_workshop_directory, config, seed=seed
    )
    return run_neural_workshop_live_lifetime(
        machine,
        config,
        seed=seed,
        environment=environment,
        verifier=verifier,
        learn=learn,
        sample=sample,
        intervention=intervention,
    )


def _summary(report) -> dict[str, object]:
    if (
        not report.audio_payloads
        or len(report.audio_payloads) != len(report.event_payloads)
    ):
        raise RuntimeError("Dual acquisition lost the public audio stream")
    packed = None if not report.rewards else sum(report.rewards) / len(report.rewards)
    return {
        "accuracy": report.verifier_accuracy,
        "packed_exact_accuracy": packed,
        "unique_verifier_bits": report.unique_verifier_bits,
        "audio_events": len(report.audio_payloads),
        "vision_events": len(report.event_payloads),
        "program_file_updates": report.program_file_updates,
        "optimizer_updates": report.optimizer_updates,
        "controller_frozen": report.controller_frozen,
    }


def run_neural_workshop_dual_acquisition(
    controller_payload: dict[str, object],
    neural_workshop_directory: Path,
    *,
    trials: int = 60,
    sessions: int = 6,
    seed: int = 99_117,
    learning_rate: float = 0.3,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    visible: bool = False,
) -> dict[str, object]:
    started = perf_counter()
    warm = _new_machine(controller_payload, learning_rate=learning_rate)
    dual_1back: list[dict[str, object]] = []
    for index in range(sessions):
        report = _run_session(
            warm,
            neural_workshop_directory,
            n_back=1,
            trials=trials,
            seed=seed + index,
            learn=True,
            sample=True,
            visible=visible,
        )
        row = {"session": index, **_summary(report)}
        dual_1back.append(row)
        if _bits_to_threshold(
            dual_1back, threshold=threshold, minimum_bits=minimum_bits
        ) is not None:
            break
    retention = _summary(
        _run_session(
            warm,
            neural_workshop_directory,
            n_back=1,
            trials=trials,
            seed=seed + 100,
            learn=False,
            sample=False,
            visible=visible,
        )
    )
    primitive = warm.admitted_program_artifact()
    wrong_depth = _summary(
        _run_session(
            warm,
            neural_workshop_directory,
            n_back=2,
            trials=trials,
            seed=seed + 150,
            learn=False,
            sample=False,
            visible=visible,
        )
    )
    warm.load_recursive_program_artifact(
        compose_recursive_temporal_program(primitive, 2),
        controller_digest=warm.controller_digest(),
    )
    warm_2back = _summary(
        _run_session(
            warm,
            neural_workshop_directory,
            n_back=2,
            trials=trials,
            seed=seed + 200,
            learn=False,
            sample=False,
            visible=visible,
        )
    )
    fresh = _new_machine(controller_payload, learning_rate=learning_rate)
    fresh.composition_depth = 2
    fresh_2back: list[dict[str, object]] = []
    for index in range(sessions):
        report = _run_session(
            fresh,
            neural_workshop_directory,
            n_back=2,
            trials=trials,
            seed=seed + 300 + index,
            learn=True,
            sample=True,
            visible=visible,
        )
        row = {"session": index, **_summary(report)}
        fresh_2back.append(row)
        if _bits_to_threshold(
            fresh_2back, threshold=threshold, minimum_bits=minimum_bits
        ) is not None:
            break
    shuffled = _summary(
        _run_session(
            _new_machine(controller_payload, learning_rate=learning_rate),
            neural_workshop_directory,
            n_back=1,
            trials=trials,
            seed=seed + 400,
            learn=True,
            sample=True,
            intervention=NeuralWorkshopIntervention(reward="shuffled", seed=seed + 400),
            visible=visible,
        )
    )
    reversed_actions = _summary(
        _run_session(
            _new_machine(controller_payload, learning_rate=learning_rate),
            neural_workshop_directory,
            n_back=1,
            trials=trials,
            seed=seed + 401,
            learn=True,
            sample=True,
            intervention=NeuralWorkshopIntervention(action="reversed", seed=seed + 401),
            visible=visible,
        )
    )
    missing_history = _summary(
        _run_session(
            _new_machine(controller_payload, learning_rate=learning_rate),
            neural_workshop_directory,
            n_back=1,
            trials=trials,
            seed=seed + 402,
            learn=True,
            sample=True,
            intervention=NeuralWorkshopIntervention(
                reset_history_each_tick=True, seed=seed + 402
            ),
            visible=visible,
        )
    )
    warm_bits = _bits_to_threshold(
        dual_1back, threshold=threshold, minimum_bits=minimum_bits
    )
    fresh_bits = _bits_to_threshold(
        fresh_2back, threshold=threshold, minimum_bits=minimum_bits
    )
    return {
        "schema": NEURAL_WORKSHOP_DUAL_ACQUISITION_SCHEMA,
        "action_count": warm.action_count,
        "decoder_key_count": warm.decoder.key_count,
        "controller_digest": warm.controller_digest(),
        "dual_1back_train": dual_1back,
        "dual_1back_bits_to_threshold": warm_bits,
        "dual_1back_retention": retention,
        "warm_dual_2back": warm_2back,
        "fresh_dual_2back_train": fresh_2back,
        "fresh_dual_2back_bits_to_threshold": fresh_bits,
        "warm_target_learning_bits": 0,
        "transfer_ratio_against_fresh_learner": None,
        "transfer_ratio_note": (
            "warm Dual 2-back is composed execution with zero target updates; "
            "fresh Dual 2-back is a same-task climb"
        ),
        "controls": {
            "wrong_depth": wrong_depth,
            "reward_shuffled": shuffled,
            "action_reversed": reversed_actions,
            "missing_history": missing_history,
        },
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--neural-workshop", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=99_117)
    parser.add_argument("--program-learning-rate", type=float, default=0.3)
    parser.add_argument(
        "--visible",
        action="store_true",
        help="show the Neural Workshop gym window (set NW_HEADLESS=0)",
    )
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
    report = run_neural_workshop_dual_acquisition(
        load_temporal_controller_artifact(arguments.controller_artifact),
        arguments.neural_workshop,
        trials=arguments.trials,
        sessions=arguments.sessions,
        seed=arguments.seed,
        learning_rate=arguments.program_learning_rate,
        visible=arguments.visible,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)


if __name__ == "__main__":
    main()
