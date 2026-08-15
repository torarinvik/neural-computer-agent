"""Acquire Dual N-Back from a blank file and measure retention/transfer.

A fresh recursive program starts uniform. It may update only its relative
address logits. After Dual 1-Back mastery the same primitive is composed
once and evaluated on Dual 2-Back. A matched fresh learner trains Dual
2-Back from scratch. History, action, and reward controls check that the
curve is causal.
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
from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from .rendered_live import run_rendered_live_lifetime

RENDERED_DUAL_TRANSFER_SCHEMA = "neural-computer.rendered-dual-transfer.v1"


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
            float(item["accuracy"]) >= threshold
            and int(item["unique_verifier_bits"]) >= minimum_bits
            for item in sessions[index:]
        ):
            return total
    return None


def _new_machine(controller_payload: dict[str, object], *, learning_rate: float):
    return build_recursive_temporal_program_machine(
        controller_payload,
        learning_rate=learning_rate,
        sample=True,
        max_sources=2,
        pack_source_actions=True,
    )


def _encoders(machine):
    encoders = RenderedBrainWorkshopEncoders(
        machine.event_width, source_key_width=machine.source_key_width
    )
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)
    return encoders


def _train_sessions(
    machine,
    encoders,
    *,
    n_back: int,
    steps: int,
    seed: int,
    sessions: int,
    threshold: float,
    minimum_bits: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(sessions):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=n_back,
                steps=steps,
                streams=("vision", "audio"),
            ),
            seed=seed + index,
            learn=True,
            sample=True,
        )
        rows.append(
            {
                "session": index,
                "accuracy": report.eligible_accuracy,
                "unique_verifier_bits": report.unique_verifier_bits,
                "program_file_updates": report.program_file_updates,
                "optimizer_updates": report.optimizer_updates,
            }
        )
        if _bits_to_threshold(
            rows, threshold=threshold, minimum_bits=minimum_bits
        ) is not None:
            break
    return rows


def run_rendered_dual_transfer(
    controller_payload: dict[str, object],
    *,
    steps: int = 48,
    sessions: int = 6,
    seed: int = 99_017,
    learning_rate: float = 0.3,
    threshold: float = 0.8,
    minimum_bits: int = 8,
) -> dict[str, object]:
    started = perf_counter()
    encoders = None
    warm = _new_machine(controller_payload, learning_rate=learning_rate)
    encoders = _encoders(warm)
    dual_1back = _train_sessions(
        warm,
        encoders,
        n_back=1,
        steps=steps,
        seed=seed,
        sessions=sessions,
        threshold=threshold,
        minimum_bits=minimum_bits,
    )
    retention = run_rendered_live_lifetime(
        warm,
        encoders,
        RenderedBrainWorkshopConfig(
            n_back=1, steps=steps, streams=("vision", "audio")
        ),
        seed=seed + 100,
        learn=False,
        sample=False,
    )
    primitive = warm.admitted_program_artifact()
    warm.load_recursive_program_artifact(
        compose_recursive_temporal_program(primitive, 2),
        controller_digest=warm.controller_digest(),
    )
    warm_2back = run_rendered_live_lifetime(
        warm,
        encoders,
        RenderedBrainWorkshopConfig(
            n_back=2, steps=steps, streams=("vision", "audio")
        ),
        seed=seed + 200,
        learn=False,
        sample=False,
    )
    fresh = _new_machine(controller_payload, learning_rate=learning_rate)
    fresh.composition_depth = 2
    fresh_2back = _train_sessions(
        fresh,
        _encoders(fresh),
        n_back=2,
        steps=steps,
        seed=seed + 300,
        sessions=sessions,
        threshold=threshold,
        minimum_bits=minimum_bits,
    )
    control_machine = _new_machine(controller_payload, learning_rate=learning_rate)
    control_encoders = _encoders(control_machine)
    shuffled = run_rendered_live_lifetime(
        control_machine,
        control_encoders,
        RenderedBrainWorkshopConfig(
            n_back=1, steps=steps, streams=("vision", "audio")
        ),
        seed=seed + 400,
        learn=True,
        sample=True,
        randomized_outcome_seed=seed + 400,
    )
    reversed_actions = run_rendered_live_lifetime(
        _new_machine(controller_payload, learning_rate=learning_rate),
        control_encoders,
        RenderedBrainWorkshopConfig(
            n_back=1, steps=steps, streams=("vision", "audio")
        ),
        seed=seed + 401,
        learn=True,
        sample=True,
        action_permutation=(3, 2, 1, 0),
    )
    missing_history = run_rendered_live_lifetime(
        _new_machine(controller_payload, learning_rate=learning_rate),
        control_encoders,
        RenderedBrainWorkshopConfig(
            n_back=1, steps=steps, streams=("vision", "audio")
        ),
        seed=seed + 402,
        learn=True,
        sample=True,
        reset_history_each_tick=True,
    )
    warm_bits = _bits_to_threshold(
        dual_1back, threshold=threshold, minimum_bits=minimum_bits
    )
    fresh_bits = _bits_to_threshold(
        fresh_2back, threshold=threshold, minimum_bits=minimum_bits
    )
    return {
        "schema": RENDERED_DUAL_TRANSFER_SCHEMA,
        "action_count": warm.action_count,
        "decoder_key_count": warm.decoder.key_count,
        "controller_digest": warm.controller_digest(),
        "dual_1back_train": dual_1back,
        "dual_1back_bits_to_threshold": warm_bits,
        "dual_1back_retention": {
            "accuracy": retention.eligible_accuracy,
            "unique_verifier_bits": retention.unique_verifier_bits,
            "program_file_updates": retention.program_file_updates,
        },
        "warm_dual_2back": {
            "accuracy": warm_2back.eligible_accuracy,
            "unique_verifier_bits": warm_2back.unique_verifier_bits,
            "program_file_updates": warm_2back.program_file_updates,
        },
        "fresh_dual_2back_train": fresh_2back,
        "fresh_dual_2back_bits_to_threshold": fresh_bits,
        "warm_target_learning_bits": 0,
        "transfer_ratio_against_fresh_learner": None,
        "transfer_ratio_note": (
            "warm Dual 2-back is composed execution with zero target updates; "
            "fresh Dual 2-back is a same-task climb"
        ),
        "controls": {
            "reward_shuffled": {
                "accuracy": shuffled.eligible_accuracy,
                "unique_verifier_bits": shuffled.unique_verifier_bits,
            },
            "action_reversed": {
                "accuracy": reversed_actions.eligible_accuracy,
                "unique_verifier_bits": reversed_actions.unique_verifier_bits,
            },
            "missing_history": {
                "accuracy": missing_history.eligible_accuracy,
                "unique_verifier_bits": missing_history.unique_verifier_bits,
            },
        },
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": perf_counter() - started,
    }


def run_rendered_dual_learn_transfer(
    controller_payload: dict[str, object],
    *,
    steps: int = 24,
    sessions: int = 4,
    seed: int = 99_017,
    learning_rate: float = 0.3,
    threshold: float = 0.8,
    minimum_bits: int = 8,
) -> dict[str, object]:
    """Same-task Dual 2-back climb after 1-back, without composing.

    This is not composed execution. A one-row file that mastered 1-back
    keeps learning on 2-back. A fresh one-row file climbs 2-back alone.
    """

    started = perf_counter()
    warm = _new_machine(controller_payload, learning_rate=learning_rate)
    encoders = _encoders(warm)
    dual_1back = _train_sessions(
        warm,
        encoders,
        n_back=1,
        steps=steps,
        seed=seed,
        sessions=sessions,
        threshold=threshold,
        minimum_bits=minimum_bits,
    )
    warm_2back = _train_sessions(
        warm,
        encoders,
        n_back=2,
        steps=steps,
        seed=seed + 200,
        sessions=sessions,
        threshold=threshold,
        minimum_bits=minimum_bits,
    )
    fresh = _new_machine(controller_payload, learning_rate=learning_rate)
    fresh_2back = _train_sessions(
        fresh,
        _encoders(fresh),
        n_back=2,
        steps=steps,
        seed=seed + 300,
        sessions=sessions,
        threshold=threshold,
        minimum_bits=minimum_bits,
    )
    warm_bits = _bits_to_threshold(
        dual_1back, threshold=threshold, minimum_bits=minimum_bits
    )
    warm_2back_bits = _bits_to_threshold(
        warm_2back, threshold=threshold, minimum_bits=minimum_bits
    )
    fresh_bits = _bits_to_threshold(
        fresh_2back, threshold=threshold, minimum_bits=minimum_bits
    )
    ratio = None
    if warm_2back_bits and fresh_bits:
        ratio = float(fresh_bits) / float(warm_2back_bits)
    return {
        "schema": "neural-computer.rendered-dual-learn-transfer.v1",
        "controller_digest": warm.controller_digest(),
        "dual_1back_bits_to_threshold": warm_bits,
        "warm_dual_2back_train": warm_2back,
        "warm_dual_2back_bits_to_threshold": warm_2back_bits,
        "fresh_dual_2back_train": fresh_2back,
        "fresh_dual_2back_bits_to_threshold": fresh_bits,
        "transfer_ratio_against_fresh_learner": ratio,
        "transfer_ratio_note": (
            "same-task Dual 2-back climb on a one-row file; not composed execution"
        ),
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=99_017)
    parser.add_argument("--program-learning-rate", type=float, default=0.3)
    parser.add_argument(
        "--learn-2back",
        action="store_true",
        help="same-task Dual 2-back climb after 1-back instead of compose execute",
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
    runner = (
        run_rendered_dual_learn_transfer
        if arguments.learn_2back
        else run_rendered_dual_transfer
    )
    report = runner(
        load_temporal_controller_artifact(arguments.controller_artifact),
        steps=arguments.steps,
        sessions=arguments.sessions,
        seed=arguments.seed,
        learning_rate=arguments.program_learning_rate,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)


if __name__ == "__main__":
    main()
