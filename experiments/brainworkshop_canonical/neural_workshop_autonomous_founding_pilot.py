"""Autonomous header routing and a warm-versus-fresh founding comparison.

A warm bank already holds verified 1-back and 2-back files. Held-out public
lines are resolved without ``n_back``: exact or same-slot invariant retrieve,
else try existing files, else compose ``PREVIOUS`` one step deeper. A matched
fresh learner starts from the same primitive and the same policy. Capacity
overflow fails closed. Instruction events travel on the amodal bus; only the
play-field source is bound to the frozen temporal program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from neural_computer import (
    DEFAULT_AGENT_BANK_FILENAME,
    RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    ExternalTemporalProgramBank,
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

NEURAL_WORKSHOP_AUTONOMOUS_FOUNDING_SCHEMA = (
    "neural-computer.neural-workshop-autonomous-founding.v1"
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


def _mastered(
    report: NeuralWorkshopLiveReport, *, threshold: float, minimum_bits: int
) -> bool:
    return bool(
        report.verifier_accuracy is not None
        and report.verifier_accuracy >= threshold
        and report.unique_verifier_bits >= minimum_bits
        and report.controller_frozen
    )


def resolve_public_line(
    machine,
    bank: ExternalTemporalProgramBank,
    primitive,
    neural_workshop_directory: Path,
    instruction_encoder: NeuralWorkshopInstructionEncoder,
    output_directory: Path,
    *,
    label: str,
    n_back: int,
    active_cells: int,
    trials: int,
    threshold: float,
    minimum_bits: int,
    seed: int,
) -> dict[str, Any]:
    """Resolve one public line by retrieve, rebind, or one-step composition."""

    config = _live_config(
        machine, n_back=n_back, active_cells=active_cells, trials=trials
    )
    environment, _verifier = build_neural_workshop_environment(
        neural_workshop_directory, config, seed=seed
    )
    try:
        context = encode_instruction_context(
            environment.observe(), instruction_encoder
        )
    finally:
        environment.close()

    failed_slots: set[int] = set()
    failed_depths: set[int] = set()
    attempts: list[dict[str, Any]] = []
    search_bits = 0
    accepted = None
    for index in range(machine.max_history + bank.program_count + 1):
        decision = decide_header_program(
            bank,
            context,
            primitive,
            failed_slots=frozenset(failed_slots),
            failed_depths=frozenset(failed_depths),
            max_history=machine.max_history,
        )
        if decision.kind == "capacity" or decision.artifact is None:
            raise RuntimeError(f"{label} hit frozen history capacity")
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=seed + 10 + index
        )
        try:
            machine.load_recursive_program_artifact(
                decision.artifact, controller_digest=bank.controller_digest
            )
            report = run_neural_workshop_live_lifetime(
                machine,
                config,
                seed=seed + 10 + index,
                environment=environment,
                verifier=verifier,
                learn=False,
                sample=False,
            )
        except Exception:
            environment.close()
            raise
        search_bits += report.unique_verifier_bits
        _save_report(
            report,
            output_directory / f"{label}-attempt-{index + 1:02d}-{decision.kind}.json",
        )
        attempts.append(
            {
                **_summary(report),
                "kind": decision.kind,
                "slot": decision.slot,
                "proposed_depth": decision.proposed_depth,
                "known": decision.known,
            }
        )
        if _mastered(report, threshold=threshold, minimum_bits=minimum_bits):
            outcomes = list(report.verifier_rewards)
            receipt = bank.admit(
                decision.artifact,
                context,
                outcomes,
                threshold=threshold,
                min_observations=minimum_bits,
                min_stable_observations=min(minimum_bits, len(outcomes)),
            )
            if not receipt.accepted:
                raise RuntimeError(f"{label} mastered but was not admitted")
            accepted = receipt.payload()
            break
        if decision.slot is not None:
            failed_slots.add(decision.slot)
        if decision.proposed_depth is not None:
            failed_depths.add(decision.proposed_depth)
    if accepted is None:
        raise RuntimeError(f"{label} was not resolved from public outcomes")
    return {
        "n_back": n_back,
        "active_cells": active_cells,
        "attempts": attempts,
        "search_unique_verifier_bits": search_bits,
        "admission": accepted,
        "resolved_kind": attempts[-1]["kind"],
        "resolved_depth": attempts[-1]["proposed_depth"],
    }


def run_autonomous_founding(
    controller_payload: dict[str, object],
    source_bank_path: Path,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    primitive_slot: int = 0,
    trials: int = 60,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    seed: int = 94_017,
) -> dict[str, Any]:
    """Compare warm autonomous resolution with a matched fresh climb."""

    if min(trials, minimum_bits) < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("autonomous founding settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    source_bank = ExternalTemporalProgramBank.load_bank(source_bank_path)
    if source_bank.configuration()["execution_schema"] != (
        RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
    ):
        raise ValueError("autonomous founding requires a recursive temporal bank")
    primitive = source_bank.artifact(primitive_slot)
    if primitive.program_length != 1:
        raise ValueError("autonomous founding needs a one-row recursive primitive")
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    if not machine.accepts_controller_digest(source_bank.controller_digest):
        raise ValueError("source instruction bank targets another controller")
    controller_before = machine.controller_digest()
    warm_bank = ExternalTemporalProgramBank.from_payload(source_bank.payload())
    fresh_bank = ExternalTemporalProgramBank(
        machine.event_width,
        machine.max_history,
        controller_digest=controller_before,
        generalization_tolerance=0.0,
        mastery_threshold=threshold,
        min_mastery_observations=1,
        interpreter_schema=warm_bank.interpreter_schema,
        execution_schema=warm_bank.execution_schema,
        output_schema=warm_bank.output_schema,
    )
    instruction_encoder = NeuralWorkshopInstructionEncoder(
        _live_config(machine, n_back=1, active_cells=2, trials=trials)
    )
    started = time.perf_counter()

    warm_cell = resolve_public_line(
        machine,
        warm_bank,
        primitive,
        neural_workshop_directory,
        instruction_encoder,
        output_directory,
        label="warm-3cell-2back",
        n_back=2,
        active_cells=3,
        trials=trials,
        threshold=threshold,
        minimum_bits=minimum_bits,
        seed=seed,
    )
    warm_depth = resolve_public_line(
        machine,
        warm_bank,
        primitive,
        neural_workshop_directory,
        instruction_encoder,
        output_directory,
        label="warm-2cell-3back",
        n_back=3,
        active_cells=2,
        trials=trials,
        threshold=threshold,
        minimum_bits=minimum_bits,
        seed=seed + 1_000,
    )
    warm_transfer = resolve_public_line(
        machine,
        warm_bank,
        primitive,
        neural_workshop_directory,
        instruction_encoder,
        output_directory,
        label="warm-3cell-3back",
        n_back=3,
        active_cells=3,
        trials=trials,
        threshold=threshold,
        minimum_bits=minimum_bits,
        seed=seed + 2_000,
    )
    fresh_depth = resolve_public_line(
        machine,
        fresh_bank,
        primitive,
        neural_workshop_directory,
        instruction_encoder,
        output_directory,
        label="fresh-2cell-3back",
        n_back=3,
        active_cells=2,
        trials=trials,
        threshold=threshold,
        minimum_bits=minimum_bits,
        seed=seed + 1_000,
    )
    fresh_transfer_bank = ExternalTemporalProgramBank(
        machine.event_width,
        machine.max_history,
        controller_digest=controller_before,
        generalization_tolerance=0.0,
        mastery_threshold=threshold,
        min_mastery_observations=1,
        interpreter_schema=warm_bank.interpreter_schema,
        execution_schema=warm_bank.execution_schema,
        output_schema=warm_bank.output_schema,
    )
    fresh_transfer = resolve_public_line(
        machine,
        fresh_transfer_bank,
        primitive,
        neural_workshop_directory,
        instruction_encoder,
        output_directory,
        label="fresh-3cell-3back",
        n_back=3,
        active_cells=3,
        trials=trials,
        threshold=threshold,
        minimum_bits=minimum_bits,
        seed=seed + 2_000,
    )

    heldout = []
    for n_back, cells, suffix, offset in (
        (1, 2, "retain-1back", 3_000),
        (2, 2, "retain-2back", 3_100),
        (3, 2, "retain-3back", 3_200),
    ):
        config = _live_config(
            machine, n_back=n_back, active_cells=cells, trials=trials
        )
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=seed + offset
        )
        try:
            context = encode_instruction_context(
                environment.observe(), instruction_encoder
            )
            decision = decide_header_program(
                warm_bank,
                context,
                primitive,
                max_history=machine.max_history,
            )
            if not decision.known or decision.artifact is None:
                raise RuntimeError(f"{suffix} lost a claimed header route")
            machine.load_recursive_program_artifact(
                decision.artifact, controller_digest=controller_before
            )
            report = run_neural_workshop_live_lifetime(
                machine,
                config,
                seed=seed + offset,
                environment=environment,
                verifier=verifier,
                learn=False,
                sample=False,
            )
        except Exception:
            environment.close()
            raise
        _save_report(report, output_directory / f"{suffix}.json")
        heldout.append(
            {
                "n_back": n_back,
                "active_cells": cells,
                **_summary(report),
                "kind": decision.kind,
                "known": decision.known,
            }
        )

    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    warm_bank.save_bank(bank_path)
    result = {
        "schema": NEURAL_WORKSHOP_AUTONOMOUS_FOUNDING_SCHEMA,
        "source_bank": str(source_bank_path),
        "source_bank_sha256": _sha256(source_bank_path),
        "controller_digest": controller_before,
        "controller_frozen": machine.controller_digest() == controller_before,
        "max_history": machine.max_history,
        "warm_3cell_2back": warm_cell,
        "warm_2cell_3back": warm_depth,
        "warm_3cell_3back": warm_transfer,
        "fresh_2cell_3back": fresh_depth,
        "fresh_3cell_3back": fresh_transfer,
        "source_retention": heldout,
        "founding": {
            "depth_target": "2-cell 3-back",
            "depth_warm_bits": warm_depth["search_unique_verifier_bits"],
            "depth_fresh_bits": fresh_depth["search_unique_verifier_bits"],
            "depth_fresh_over_warm": (
                None
                if warm_depth["search_unique_verifier_bits"] < 1
                else (
                    fresh_depth["search_unique_verifier_bits"]
                    / warm_depth["search_unique_verifier_bits"]
                )
            ),
            "header_target": "3-cell 3-back",
            "header_warm_bits": warm_transfer["search_unique_verifier_bits"],
            "header_fresh_bits": fresh_transfer["search_unique_verifier_bits"],
            "header_fresh_over_warm": (
                None
                if warm_transfer["search_unique_verifier_bits"] < 1
                else (
                    fresh_transfer["search_unique_verifier_bits"]
                    / warm_transfer["search_unique_verifier_bits"]
                )
            ),
        },
        "optimizer_updates": 0,
        "program_file_updates": 0,
        "replayed_examples": 0,
        "bank": str(bank_path),
        "bank_program_count": warm_bank.program_count,
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
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=94_017)
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
    report = run_autonomous_founding(
        controller,
        arguments.source_bank,
        arguments.neural_workshop,
        arguments.output_dir,
        primitive_slot=arguments.primitive_slot,
        trials=arguments.trials,
        threshold=arguments.threshold,
        minimum_bits=arguments.minimum_bits,
        seed=arguments.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
