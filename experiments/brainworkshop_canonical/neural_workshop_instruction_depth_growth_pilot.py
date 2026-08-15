"""Admit the next recursive depth from the public n-back header.

A verified instruction bank already binds depth-one and depth-two files to
their 2-cell headers. This runner composes the same ``PREVIOUS`` primitive
one more time, verifies the child on live 3-back, and attaches that exact
header. ``n_back`` never enters the controller. Four-back stays unknown.
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
    compose_recursive_temporal_program,
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
    build_neural_workshop_environment,
    encode_instruction_context,
    run_neural_workshop_live_lifetime,
)
from .physical_program_bank import retrieve_instruction_program

NEURAL_WORKSHOP_INSTRUCTION_DEPTH_GROWTH_SCHEMA = (
    "neural-computer.neural-workshop-instruction-depth-growth.v1"
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


def run_instruction_depth_growth(
    controller_payload: dict[str, object],
    source_bank_path: Path,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    target_depth: int = 3,
    primitive_slot: int = 0,
    active_cells: int = 2,
    trials: int = 60,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    stable_sessions: int = 3,
    seed: int = 93_017,
) -> dict[str, Any]:
    """Compose, verify, and header-bind one additional recursive depth."""

    if (
        min(target_depth, active_cells, trials, minimum_bits, stable_sessions) < 1
        or target_depth < 2
        or not 0.0 <= threshold <= 1.0
        or primitive_slot < 0
    ):
        raise ValueError("instruction depth-growth settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    source_bank = ExternalTemporalProgramBank.load_bank(source_bank_path)
    if source_bank.configuration()["execution_schema"] != (
        RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
    ):
        raise ValueError("depth growth requires a recursive temporal bank")
    if source_bank.program_count <= primitive_slot:
        raise ValueError("source bank is missing the recursive primitive")
    primitive = source_bank.artifact(primitive_slot)
    if primitive.program_length != 1:
        raise ValueError("depth growth needs a one-row recursive primitive")
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    if not machine.accepts_controller_digest(source_bank.controller_digest):
        raise ValueError("source instruction bank targets another controller")
    if target_depth > machine.max_history:
        raise ValueError("target depth exceeds frozen history capacity")
    child = compose_recursive_temporal_program(primitive, target_depth)
    existing = [
        index
        for index in range(source_bank.program_count)
        if source_bank.artifact(index).digest() == child.digest()
    ]
    if existing:
        raise ValueError("target composed program is already in the source bank")
    controller_before = machine.controller_digest()
    bank = ExternalTemporalProgramBank.from_payload(source_bank.payload())
    started = time.perf_counter()
    instruction_encoder = NeuralWorkshopInstructionEncoder(
        _live_config(
            machine, n_back=1, active_cells=active_cells, trials=trials
        )
    )

    def execute(
        artifact,
        *,
        n_back: int,
        session_seed: int,
        retrieve: bool = False,
        intervention: NeuralWorkshopIntervention | None = None,
    ) -> dict[str, Any]:
        config = _live_config(
            machine,
            n_back=n_back,
            active_cells=active_cells,
            trials=trials,
        )
        environment, verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        try:
            context = encode_instruction_context(
                environment.observe(), instruction_encoder
            )
            selected_slot = None
            selected_propensity = None
            if retrieve:
                selection = retrieve_instruction_program(machine, bank, context)
                selected_slot = selection.slot
                selected_propensity = selection.propensity
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
            raise RuntimeError("depth-growth evaluation changed its program")
        return {
            **_summary(report),
            "selected_slot": selected_slot,
            "selected_propensity": selected_propensity,
            "route_known": bank.router.has_context(context),
            "context": context,
            "report": report,
        }

    for depth in range(1, target_depth):
        config = _live_config(
            machine, n_back=depth, active_cells=active_cells, trials=trials
        )
        environment, _verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=seed + depth
        )
        try:
            source_context = encode_instruction_context(
                environment.observe(), instruction_encoder
            )
        finally:
            environment.close()
        if not bank.router.has_context(source_context):
            raise RuntimeError(
                f"source depth-{depth} header is unknown under this public line; "
                "keep cells and trials on the trained header or rebind first"
            )

    zeroshot = execute(
        child,
        n_back=target_depth,
        session_seed=seed + 10,
        retrieve=True,
    )
    _save_report(zeroshot["report"], output_directory / "zeroshot-target-header.json")
    if zeroshot["route_known"]:
        raise RuntimeError("unseen target header matched a trained depth")

    shallow = compose_recursive_temporal_program(primitive, target_depth - 1)
    over = compose_recursive_temporal_program(primitive, target_depth + 1)
    controls = {
        "wrong_depth": execute(
            shallow,
            n_back=target_depth,
            session_seed=seed + 1,
        ),
        "over_composed": execute(
            over,
            n_back=target_depth,
            session_seed=seed + 2,
        ),
        "memory_corruption": execute(
            child,
            n_back=target_depth,
            session_seed=seed + 3,
            intervention=NeuralWorkshopIntervention(
                reset_history_each_tick=True, seed=seed + 3
            ),
        ),
    }
    for name, row in controls.items():
        _save_report(row["report"], output_directory / f"control-{name}.json")

    verification: list[NeuralWorkshopLiveReport] = []
    target_context = None
    for index in range(stable_sessions):
        row = execute(
            child,
            n_back=target_depth,
            session_seed=seed + 100 + index,
        )
        verification.append(row["report"])
        _save_report(
            row["report"],
            output_directory / f"verify-depth-{target_depth}-session-{index + 1:03d}.json",
        )
        if target_context is None:
            target_context = row["context"]
    if not all(
        report.verifier_accuracy is not None
        and report.verifier_accuracy >= threshold
        and report.unique_verifier_bits >= minimum_bits
        for report in verification
    ):
        raise RuntimeError("composed target depth failed live verification")
    if target_context is None:
        raise RuntimeError("verification emitted no instruction context")

    receipt = bank.admit(
        child,
        target_context,
        [float(report.verifier_accuracy) for report in verification],
        threshold=threshold,
        min_observations=stable_sessions,
        min_stable_observations=stable_sessions,
    )
    if not receipt.accepted or receipt.slot is None:
        raise RuntimeError(f"verified target depth was rejected: {receipt.reason}")

    heldout: list[dict[str, Any]] = []
    for index in range(stable_sessions):
        row = execute(
            child,
            n_back=target_depth,
            session_seed=seed + 200 + index,
            retrieve=True,
        )
        _save_report(
            row["report"],
            output_directory / f"retrieve-depth-{target_depth}-session-{index + 1:03d}.json",
        )
        heldout.append(
            {
                **_summary(row["report"]),
                "selected_slot": row["selected_slot"],
                "selected_propensity": row["selected_propensity"],
                "expected_slot": receipt.slot,
                "route_known": row["route_known"],
            }
        )
        if row["selected_slot"] != receipt.slot or not row["route_known"]:
            raise RuntimeError("target header did not retrieve the composed child")

    retained: dict[int, list[dict[str, Any]]] = {}
    for depth in range(1, target_depth):
        rows = []
        for index in range(stable_sessions):
            row = execute(
                compose_recursive_temporal_program(primitive, depth),
                n_back=depth,
                session_seed=seed + 300 + depth * 10 + index,
                retrieve=True,
            )
            _save_report(
                row["report"],
                output_directory / f"retain-depth-{depth}-session-{index + 1:03d}.json",
            )
            rows.append(
                {
                    **_summary(row["report"]),
                    "selected_slot": row["selected_slot"],
                    "selected_propensity": row["selected_propensity"],
                    "route_known": row["route_known"],
                }
            )
            if not row["route_known"]:
                raise RuntimeError(f"depth-{depth} header lost its trained route")
        retained[depth] = rows

    unseen_depth = target_depth + 1
    unseen = execute(
        child,
        n_back=unseen_depth,
        session_seed=seed + 400,
        retrieve=True,
    )
    _save_report(
        unseen["report"],
        output_directory / f"control-unseen-depth-{unseen_depth}.json",
    )
    if unseen["route_known"]:
        raise RuntimeError("unseen deeper header claimed a learned route")

    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    bank.save_bank(bank_path)
    result = {
        "schema": NEURAL_WORKSHOP_INSTRUCTION_DEPTH_GROWTH_SCHEMA,
        "source_bank": str(source_bank_path),
        "source_bank_sha256": _sha256(source_bank_path),
        "target_depth": target_depth,
        "primitive_slot": primitive_slot,
        "child_program_length": child.program_length,
        "child_digest": child.digest(),
        "controller_digest": controller_before,
        "controller_frozen": machine.controller_digest() == controller_before,
        "zeroshot_target": {
            **_summary(zeroshot["report"]),
            "selected_slot": zeroshot["selected_slot"],
            "selected_propensity": zeroshot["selected_propensity"],
            "route_known": zeroshot["route_known"],
        },
        "verification": [_summary(report) for report in verification],
        "verification_unique_verifier_bits": sum(
            report.unique_verifier_bits for report in verification
        ),
        "retrieval_search_unique_verifier_bits": 0,
        "admission": receipt.payload(),
        "heldout_retrieval": heldout,
        "source_retention": {
            str(depth): rows for depth, rows in retained.items()
        },
        "controls": {
            name: _summary(row["report"]) for name, row in controls.items()
        },
        "unseen_deeper_header": {
            **_summary(unseen["report"]),
            "n_back": unseen_depth,
            "selected_slot": unseen["selected_slot"],
            "selected_propensity": unseen["selected_propensity"],
            "route_known": unseen["route_known"],
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
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-depth", type=int, default=3)
    parser.add_argument("--primitive-slot", type=int, default=0)
    parser.add_argument("--active-cells", type=int, default=2)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--stable-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=93_017)
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
    report = run_instruction_depth_growth(
        controller,
        arguments.source_bank,
        arguments.neural_workshop,
        arguments.output_dir,
        target_depth=arguments.target_depth,
        primitive_slot=arguments.primitive_slot,
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
