"""Close the remaining sealed-frontier holes on one frozen relation.

The campaign does not start from a gifted ``PREVIOUS`` file. It searches
one-hot relative addresses on live 1-back, lifts the winner, composes for
2-back, records a 5-back capacity miss, grows history as a versioned
interpreter, then verifies 5-back. The same primitive is tested on rendered
audio, a second substrate. Dual N-Back still needs a two-port decoder and is
not claimed.
"""

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
    one_hot_temporal_address_artifact,
    pad_recursive_temporal_program,
    recursive_temporal_primitive,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .instruction_autonomy import decide_header_program
from .neural_workshop_live import (
    NeuralWorkshopInstructionEncoder,
    NeuralWorkshopLiveConfig,
    NeuralWorkshopLiveReport,
    build_neural_workshop_environment,
    encode_instruction_context,
    run_neural_workshop_live_lifetime,
)
from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from .rendered_live import run_rendered_live_lifetime

NEURAL_WORKSHOP_SEALED_FRONTIER_SCHEMA = (
    "neural-computer.neural-workshop-sealed-frontier.v1"
)


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


def _mastered(
    report: NeuralWorkshopLiveReport, *, threshold: float, minimum_bits: int
) -> bool:
    return bool(
        report.verifier_accuracy is not None
        and report.verifier_accuracy >= threshold
        and report.unique_verifier_bits >= minimum_bits
        and report.controller_frozen
    )


def _live_config(machine, *, n_back: int, trials: int, active_cells: int = 2):
    return NeuralWorkshopLiveConfig(
        grid_size=3,
        active_cells=active_cells,
        n_back=n_back,
        trials=trials,
        event_width=machine.event_width,
        source_key_width=machine.source_key_width,
    )


def _run_workshop(
    machine,
    neural_workshop_directory: Path,
    *,
    n_back: int,
    trials: int,
    seed: int,
    artifact,
    controller_digest: str,
) -> NeuralWorkshopLiveReport:
    machine.load_recursive_program_artifact(
        artifact, controller_digest=controller_digest
    )
    config = _live_config(machine, n_back=n_back, trials=trials)
    environment, verifier = build_neural_workshop_environment(
        neural_workshop_directory, config, seed=seed
    )
    return run_neural_workshop_live_lifetime(
        machine,
        config,
        seed=seed,
        environment=environment,
        verifier=verifier,
        learn=False,
        sample=False,
    )


def run_sealed_frontier(
    controller_payload: dict[str, object],
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    trials: int = 60,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    grown_history: int = 8,
    seed: int = 95_017,
) -> dict[str, Any]:
    """Discover, compose, grow capacity, and transfer to rendered audio."""

    if min(trials, minimum_bits, grown_history) < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("sealed frontier settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    controller_before = machine.controller_digest()
    instruction_encoder = NeuralWorkshopInstructionEncoder(
        _live_config(machine, n_back=1, trials=trials)
    )
    bank = ExternalTemporalProgramBank(
        machine.event_width,
        machine.max_history,
        controller_digest=controller_before,
        generalization_tolerance=0.0,
        mastery_threshold=threshold,
        min_mastery_observations=1,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )

    discovered = None
    discovery = []
    for offset in range(machine.max_history):
        candidate = recursive_temporal_primitive(
            one_hot_temporal_address_artifact(offset, machine.max_history)
        )
        report = _run_workshop(
            machine,
            neural_workshop_directory,
            n_back=1,
            trials=trials,
            seed=seed + offset,
            artifact=candidate,
            controller_digest=controller_before,
        )
        _save_report(report, output_directory / f"discover-offset-{offset}.json")
        discovery.append({"offset": offset, **_summary(report)})
        if _mastered(report, threshold=threshold, minimum_bits=minimum_bits):
            config = _live_config(machine, n_back=1, trials=trials)
            environment, _verifier = build_neural_workshop_environment(
                neural_workshop_directory, config, seed=seed + offset
            )
            try:
                context = encode_instruction_context(
                    environment.observe(), instruction_encoder
                )
            finally:
                environment.close()
            receipt = bank.admit(
                candidate,
                context,
                list(report.verifier_rewards),
                threshold=threshold,
                min_observations=minimum_bits,
                min_stable_observations=min(minimum_bits, report.unique_verifier_bits),
            )
            if not receipt.accepted:
                raise RuntimeError("discovered primitive was not admitted")
            discovered = {
                "offset": offset,
                "artifact": candidate,
                "receipt": receipt.payload(),
                **_summary(report),
            }
            break
    if discovered is None:
        raise RuntimeError("no one-hot relative address mastered 1-back")
    primitive = discovered["artifact"]

    composed = None
    depth_attempts = []
    failed_slots: set[int] = set()
    failed_depths: set[int] = set()
    config = _live_config(machine, n_back=2, trials=trials)
    environment, _verifier = build_neural_workshop_environment(
        neural_workshop_directory, config, seed=seed + 20
    )
    try:
        context = encode_instruction_context(
            environment.observe(), instruction_encoder
        )
    finally:
        environment.close()
    for index in range(machine.max_history):
        decision = decide_header_program(
            bank,
            context,
            primitive,
            failed_slots=frozenset(failed_slots),
            failed_depths=frozenset(failed_depths),
            max_history=machine.max_history,
        )
        if decision.kind == "capacity" or decision.artifact is None:
            raise RuntimeError("2-back hit capacity before a composed child existed")
        report = _run_workshop(
            machine,
            neural_workshop_directory,
            n_back=2,
            trials=trials,
            seed=seed + 21 + index,
            artifact=decision.artifact,
            controller_digest=controller_before,
        )
        _save_report(
            report, output_directory / f"compose-2back-{decision.kind}-{index}.json"
        )
        depth_attempts.append(
            {
                **_summary(report),
                "kind": decision.kind,
                "proposed_depth": decision.proposed_depth,
            }
        )
        if _mastered(report, threshold=threshold, minimum_bits=minimum_bits):
            receipt = bank.admit(
                decision.artifact,
                context,
                list(report.verifier_rewards),
                threshold=threshold,
                min_observations=minimum_bits,
                min_stable_observations=min(minimum_bits, report.unique_verifier_bits),
            )
            composed = {
                "receipt": receipt.payload(),
                "kind": decision.kind,
                "depth": decision.proposed_depth,
                **_summary(report),
            }
            break
        if decision.slot is not None:
            failed_slots.add(decision.slot)
        if decision.proposed_depth is not None:
            failed_depths.add(decision.proposed_depth)
    if composed is None:
        raise RuntimeError("autonomous 2-back composition failed")

    capacity_miss = None
    try:
        compose_recursive_temporal_program(primitive, 5)
    except ValueError as error:
        capacity_miss = str(error)
    if capacity_miss is None:
        raise RuntimeError("5-back composition should miss at history 4")

    grown = build_recursive_temporal_program_machine(
        controller_payload, sample=False, max_history=grown_history
    )
    grown_primitive = pad_recursive_temporal_program(primitive, grown_history)
    five = compose_recursive_temporal_program(grown_primitive, 5)
    five_report = _run_workshop(
        grown,
        neural_workshop_directory,
        n_back=5,
        trials=trials,
        seed=seed + 40,
        artifact=five,
        controller_digest=grown.controller_digest(),
    )
    _save_report(five_report, output_directory / "grown-5back.json")

    torch.manual_seed(seed + 50)
    audio_encoders = RenderedBrainWorkshopEncoders(
        machine.event_width, source_key_width=machine.source_key_width
    )
    for parameter in audio_encoders.parameters():
        parameter.requires_grad_(False)
    audio_one = RenderedBrainWorkshopConfig(
        n_back=1, steps=max(trials, 24), streams=("audio",), symbol_count=8
    )
    audio_two = RenderedBrainWorkshopConfig(
        n_back=2, steps=max(trials, 24), streams=("audio",), symbol_count=8
    )
    machine.load_recursive_program_artifact(
        primitive, controller_digest=controller_before
    )
    warm_audio_one = run_rendered_live_lifetime(
        machine,
        audio_encoders,
        audio_one,
        seed=seed + 51,
        learn=False,
        sample=False,
    )
    machine.load_recursive_program_artifact(
        compose_recursive_temporal_program(primitive, 2),
        controller_digest=controller_before,
    )
    warm_audio_two = run_rendered_live_lifetime(
        machine,
        audio_encoders,
        audio_two,
        seed=seed + 52,
        learn=False,
        sample=False,
    )
    fresh_audio = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    fresh_audio_bits = 0
    fresh_audio_winner = None
    for offset in range(fresh_audio.max_history):
        candidate = recursive_temporal_primitive(
            one_hot_temporal_address_artifact(offset, fresh_audio.max_history)
        )
        fresh_audio.load_recursive_program_artifact(
            candidate, controller_digest=fresh_audio.controller_digest()
        )
        evaluated = run_rendered_live_lifetime(
            fresh_audio,
            audio_encoders,
            audio_one,
            seed=seed + 53 + offset,
            learn=False,
            sample=False,
        )
        fresh_audio_bits += evaluated.unique_verifier_bits
        if evaluated.eligible_accuracy >= threshold:
            fresh_audio_winner = {
                "offset": offset,
                "accuracy": evaluated.eligible_accuracy,
                "unique_verifier_bits": evaluated.unique_verifier_bits,
            }
            break

    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    bank.save_bank(bank_path)
    result = {
        "schema": NEURAL_WORKSHOP_SEALED_FRONTIER_SCHEMA,
        "controller_digest": controller_before,
        "grown_controller_digest": grown.controller_digest(),
        "controller_frozen": machine.controller_digest() == controller_before,
        "max_history": machine.max_history,
        "grown_history": grown.max_history,
        "discovery": discovery,
        "discovered_primitive": {
            key: value for key, value in discovered.items() if key != "artifact"
        },
        "autonomous_2back": composed,
        "capacity_miss_5back": capacity_miss,
        "grown_5back": _summary(five_report),
        "audio_transfer": {
            "warm_1back_accuracy": warm_audio_one.eligible_accuracy,
            "warm_1back_bits": warm_audio_one.unique_verifier_bits,
            "warm_2back_accuracy": warm_audio_two.eligible_accuracy,
            "warm_2back_bits": warm_audio_two.unique_verifier_bits,
            "fresh_1back": fresh_audio_winner,
            "fresh_1back_search_bits": fresh_audio_bits,
            "optimizer_updates": warm_audio_one.optimizer_updates
            + warm_audio_two.optimizer_updates,
        },
        "optimizer_updates": 0,
        "program_file_updates": 0,
        "replayed_examples": 0,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--grown-history", type=int, default=8)
    parser.add_argument("--seed", type=int, default=95_017)
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
    print(
        run_sealed_frontier(
            controller,
            arguments.neural_workshop,
            arguments.output_dir,
            trials=arguments.trials,
            threshold=arguments.threshold,
            minimum_bits=arguments.minimum_bits,
            grown_history=arguments.grown_history,
            seed=arguments.seed,
        )
    )


if __name__ == "__main__":
    main()
