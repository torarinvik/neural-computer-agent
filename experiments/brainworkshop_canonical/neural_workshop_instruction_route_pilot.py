"""Retrieve recursive temporal depth from the visible Neural Workshop header.

The play-field encoder still drives the frozen comparator. A second, separately
keyed instruction encoder reads only the public mode line. Candidate depth is
selected from that learned event and authenticated outcomes; ``n_back`` never
enters the controller or the bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from neural_computer import (
    DEFAULT_AGENT_BANK_FILENAME,
    RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
    ExternalTemporalProgramBank,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_live import (
    NeuralWorkshopInstructionEncoder,
    NeuralWorkshopIntervention,
    NeuralWorkshopLiveConfig,
    NeuralWorkshopLiveReport,
    NeuralWorkshopRGBAEncoder,
    build_neural_workshop_environment,
    encode_instruction_context,
    run_neural_workshop_live_lifetime,
)
from .physical_program_bank import (
    learned_event_context,
    retrieve_instruction_program,
)

NEURAL_WORKSHOP_INSTRUCTION_ROUTE_SCHEMA = (
    "neural-computer.neural-workshop-instruction-route.v1"
)
_MIN_HEADER_SEPARATION = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_report(report: NeuralWorkshopLiveReport, path: Path) -> None:
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")


def _summary(report: NeuralWorkshopLiveReport) -> dict[str, Any]:
    return {
        "accuracy": report.verifier_accuracy,
        "unique_verifier_bits": report.unique_verifier_bits,
        "program_file_updates": report.program_file_updates,
        "controller_frozen": report.controller_frozen,
    }


def _shuffle_instruction_pixels(
    observation: dict[str, Any],
    crop: tuple[float, float, float, float],
    *,
    seed: int,
) -> dict[str, Any]:
    width = int(observation["width"])
    height = int(observation["height"])
    pixels = bytearray(observation["rgba"])
    left, top, right, bottom = crop
    x0 = int(left * width)
    y0 = int(top * height)
    x1 = max(x0 + 1, math.ceil(right * width))
    y1 = max(y0 + 1, math.ceil(bottom * height))
    coordinates = [
        (column, row) for row in range(y0, y1) for column in range(x0, x1)
    ]
    values = [
        pixels[(row * width + column) * 4 : (row * width + column) * 4 + 4]
        for column, row in coordinates
    ]
    random.Random(seed).shuffle(values)
    for (column, row), value in zip(coordinates, values, strict=True):
        start = (row * width + column) * 4
        pixels[start : start + 4] = value
    shuffled = dict(observation)
    shuffled["rgba"] = bytes(pixels)
    return shuffled


def _live_config(
    machine,
    *,
    n_back: int,
    active_cells: int,
    trials: int,
) -> NeuralWorkshopLiveConfig:
    return NeuralWorkshopLiveConfig(
        grid_size=3,
        active_cells=active_cells,
        n_back=n_back,
        trials=trials,
        event_width=machine.event_width,
        source_key_width=machine.source_key_width,
    )


def run_instruction_route(
    controller_payload: dict[str, object],
    source_bank_path: Path,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    primitive_slot: int = 0,
    composed_slot: int = 1,
    active_cells: int = 2,
    trials: int = 60,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    stable_sessions: int = 3,
    seed: int = 81_017,
) -> dict[str, Any]:
    """Bind verified depths to public headers, then retrieve without search."""

    if (
        min(active_cells, trials, minimum_bits, stable_sessions) < 1
        or not 0.0 <= threshold <= 1.0
        or primitive_slot == composed_slot
        or min(primitive_slot, composed_slot) < 0
    ):
        raise ValueError("instruction route settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    source_bank = ExternalTemporalProgramBank.load_bank(source_bank_path)
    if source_bank.configuration()["execution_schema"] != (
        RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
    ):
        raise ValueError("instruction route requires a recursive temporal bank")
    primitive = source_bank.artifact(primitive_slot)
    composed = source_bank.artifact(composed_slot)
    if primitive.program_length != 1 or composed.program_length < 2:
        raise ValueError("source bank does not contain a depth-one and composed pair")
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    if source_bank.controller_digest != machine.controller_digest():
        raise ValueError("source recursive programs target another controller")
    controller_before = machine.controller_digest()
    started = time.perf_counter()
    instruction_encoder = NeuralWorkshopInstructionEncoder(
        _live_config(machine, n_back=1, active_cells=active_cells, trials=trials)
    )

    def execute(
        artifact,
        *,
        n_back: int,
        session_seed: int,
        intervention: NeuralWorkshopIntervention | None = None,
        retrieve_context: torch.Tensor | None = None,
        observe_first: bool = False,
    ) -> tuple[NeuralWorkshopLiveReport, torch.Tensor | None, int | None, float | None]:
        config = _live_config(
            machine,
            n_back=n_back,
            active_cells=active_cells,
            trials=trials,
        )
        torch.manual_seed(session_seed)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        selection_slot = None
        selection_propensity = None
        context = retrieve_context
        try:
            if observe_first or retrieve_context is not None:
                observation = environment.observe()
                if retrieve_context is None:
                    context = encode_instruction_context(
                        observation, instruction_encoder
                    )
                selection = retrieve_instruction_program(machine, bank, context)
                selection_slot = selection.slot
                selection_propensity = selection.propensity
            else:
                machine.load_recursive_program_artifact(
                    artifact, controller_digest=controller_before
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
        except Exception:
            environment.close()
            raise
        if report.program_file_updates != 0:
            raise RuntimeError("instruction-route evaluation changed its program")
        if context is None and report.instruction_payloads:
            context = learned_event_context(
                report.instruction_payloads, width=machine.event_width
            )
        return report, context, selection_slot, selection_propensity

    bank: ExternalTemporalProgramBank | None = None
    header_contexts: dict[int, torch.Tensor] = {}
    for n_back, session_seed in ((1, seed), (2, seed + 1)):
        config = _live_config(
            machine, n_back=n_back, active_cells=active_cells, trials=trials
        )
        environment, _verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        try:
            header_contexts[n_back] = encode_instruction_context(
                environment.observe(), instruction_encoder
            )
        finally:
            environment.close()
    header_distance = float(
        torch.linalg.vector_norm(header_contexts[1] - header_contexts[2]).item()
    )
    if header_distance < _MIN_HEADER_SEPARATION:
        raise RuntimeError(
            "instruction encoder does not distinguish the visible 1-back and "
            f"2-back headers: distance={header_distance:.4f}"
        )

    verification: dict[int, list[NeuralWorkshopLiveReport]] = {1: [], 2: []}
    verification_contexts: dict[int, torch.Tensor] = {}
    artifacts = {1: primitive, 2: composed}
    for n_back, artifact in artifacts.items():
        reports = []
        for index in range(stable_sessions):
            report, context, _slot, _propensity = execute(
                artifact,
                n_back=n_back,
                session_seed=seed + n_back * 100 + index,
            )
            reports.append(report)
            _save_report(
                report,
                output_directory / f"verify-n{n_back}-session-{index + 1:03d}.json",
            )
            if context is None:
                raise RuntimeError("verification session emitted no instruction event")
            if n_back not in verification_contexts:
                verification_contexts[n_back] = context
        if not all(
            report.verifier_accuracy is not None
            and report.verifier_accuracy >= threshold
            and report.unique_verifier_bits >= minimum_bits
            for report in reports
        ):
            raise RuntimeError(f"source program failed {n_back}-back verification")
        verification[n_back] = reports

    bank = ExternalTemporalProgramBank(
        machine.event_width,
        machine.max_history,
        controller_digest=controller_before,
        generalization_tolerance=0.0,
        mastery_threshold=threshold,
        min_mastery_observations=stable_sessions,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )

    def admit(n_back: int):
        receipt = bank.admit(
            artifacts[n_back],
            verification_contexts[n_back],
            [float(report.verifier_accuracy) for report in verification[n_back]],
            threshold=threshold,
            min_observations=stable_sessions,
            min_stable_observations=stable_sessions,
        )
        if not receipt.accepted:
            raise RuntimeError(
                f"verified {n_back}-back instruction route was rejected: {receipt.reason}"
            )
        return receipt

    primitive_receipt = admit(1)
    composed_receipt = admit(2)
    if primitive_receipt.slot == composed_receipt.slot:
        raise RuntimeError("instruction contexts collapsed onto one bank slot")

    heldout: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for n_back, expected_slot in (
        (1, primitive_receipt.slot),
        (2, composed_receipt.slot),
    ):
        for index in range(stable_sessions):
            report, context, slot, propensity = execute(
                artifacts[n_back],
                n_back=n_back,
                session_seed=seed + 1_000 + n_back * 100 + index,
                retrieve_context=None,
                observe_first=True,
            )
            _save_report(
                report,
                output_directory / f"retrieve-n{n_back}-session-{index + 1:03d}.json",
            )
            heldout[n_back].append(
                {
                    **_summary(report),
                    "selected_slot": slot,
                    "selected_propensity": propensity,
                    "expected_slot": expected_slot,
                    "route_known": bool(
                        context is not None and bank.router.has_context(context)
                    ),
                }
            )
            if slot != expected_slot:
                raise RuntimeError(
                    f"instruction route selected slot {slot} for {n_back}-back"
                )

    play_config = _live_config(
        machine, n_back=2, active_cells=active_cells, trials=trials
    )
    play_environment, _play_verifier = build_neural_workshop_environment(
        neural_workshop_directory, play_config, seed=seed + 3_000
    )
    try:
        play_observation = play_environment.observe()
        play_encoder = NeuralWorkshopRGBAEncoder(
            play_config, seed=play_config.instruction_encoder_seed
        )
        play_events = play_encoder.encode(play_observation)
        play_context = F.normalize(play_events.payload[0, 0].detach().cpu(), dim=0)
        play_selection = retrieve_instruction_program(machine, bank, play_context)
        play_report = run_neural_workshop_live_lifetime(
            machine,
            play_config,
            seed=seed + 3_000,
            environment=play_environment,
            verifier=_play_verifier,
            learn=False,
            sample=False,
        )
    except Exception:
        play_environment.close()
        raise
    _save_report(play_report, output_directory / "control-play-field-context.json")

    shuffle_environment, shuffle_verifier = build_neural_workshop_environment(
        neural_workshop_directory,
        play_config,
        seed=seed + 3_001,
    )
    try:
        shuffled_observation = _shuffle_instruction_pixels(
            shuffle_environment.observe(),
            play_config.instruction_crop,
            seed=seed + 3_001,
        )
        shuffled_context = encode_instruction_context(
            shuffled_observation, instruction_encoder
        )
        shuffled_selection = retrieve_instruction_program(
            machine, bank, shuffled_context
        )
        shuffled_report = run_neural_workshop_live_lifetime(
            machine,
            play_config,
            seed=seed + 3_001,
            environment=shuffle_environment,
            verifier=shuffle_verifier,
            learn=False,
            sample=False,
        )
    except Exception:
        shuffle_environment.close()
        raise
    _save_report(shuffled_report, output_directory / "control-shuffled-header.json")

    wrong_report, _context, _slot, _propensity = execute(
        primitive,
        n_back=2,
        session_seed=seed + 3_002,
    )
    _save_report(wrong_report, output_directory / "control-wrong-program.json")

    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    bank.save_bank(bank_path)
    heldout_bits = sum(
        int(row["unique_verifier_bits"])
        for rows in heldout.values()
        for row in rows
    )
    result = {
        "schema": NEURAL_WORKSHOP_INSTRUCTION_ROUTE_SCHEMA,
        "source_bank": str(source_bank_path),
        "source_bank_sha256": _sha256(source_bank_path),
        "primitive_slot": primitive_slot,
        "composed_slot": composed_slot,
        "controller_digest": controller_before,
        "controller_frozen": machine.controller_digest() == controller_before,
        "header_context_distance": header_distance,
        "verification": {
            str(n_back): [_summary(report) for report in reports]
            for n_back, reports in verification.items()
        },
        "verification_unique_verifier_bits": sum(
            report.unique_verifier_bits
            for reports in verification.values()
            for report in reports
        ),
        "admissions": {
            "n_back_1": primitive_receipt.payload(),
            "n_back_2": composed_receipt.payload(),
        },
        "heldout_retrieval": {
            str(n_back): rows for n_back, rows in heldout.items()
        },
        "heldout_unique_verifier_bits": heldout_bits,
        "retrieval_search_unique_verifier_bits": 0,
        "optimizer_updates": 0,
        "program_file_updates": 0,
        "replayed_examples": 0,
        "controls": {
            "play_field_context": {
                **_summary(play_report),
                "selected_slot": play_selection.slot,
                "selected_propensity": play_selection.propensity,
                "route_known": bank.router.has_context(play_context),
            },
            "shuffled_header": {
                **_summary(shuffled_report),
                "selected_slot": shuffled_selection.slot,
                "selected_propensity": shuffled_selection.propensity,
                "route_known": bank.router.has_context(shuffled_context),
            },
            "wrong_program": _summary(wrong_report),
        },
        "bank": str(bank_path),
        "bank_program_count": bank.program_count,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primitive-slot", type=int, default=0)
    parser.add_argument("--composed-slot", type=int, default=1)
    parser.add_argument("--active-cells", type=int, default=2)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--stable-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=81_017)
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
    report = run_instruction_route(
        controller,
        arguments.source_bank,
        arguments.neural_workshop,
        arguments.output_dir,
        primitive_slot=arguments.primitive_slot,
        composed_slot=arguments.composed_slot,
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
