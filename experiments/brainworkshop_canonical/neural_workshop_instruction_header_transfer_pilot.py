"""Transfer verified header routes across one held-out visible variant.

A 2-cell instruction bank already binds depth-one and composed programs to
their public headers. Changing only active cells rewrites the mode line.
Nearest-neighbor over this encoder cannot tell that change from an unseen
depth: 3-cell 2-back and 2-cell 3-back sit the same distance from the trained
2-back header. This runner therefore keeps exact match fail-closed, rebinds
the same artifacts to the new header, and never exposes ``n_back`` or a cell
count to the controller.
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
    ExternalTemporalProgramBank,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_live import (
    NeuralWorkshopInstructionEncoder,
    NeuralWorkshopLiveConfig,
    NeuralWorkshopLiveReport,
    build_neural_workshop_environment,
    encode_instruction_context,
    run_neural_workshop_live_lifetime,
)
from .physical_program_bank import retrieve_instruction_program

NEURAL_WORKSHOP_INSTRUCTION_HEADER_TRANSFER_SCHEMA = (
    "neural-computer.neural-workshop-instruction-header-transfer.v1"
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


def _nearest(
    context: torch.Tensor, trained: dict[str, torch.Tensor]
) -> dict[str, Any]:
    distances = {
        name: float(torch.linalg.vector_norm(context - value).item())
        for name, value in trained.items()
    }
    nearest = min(distances, key=distances.get)
    return {"nearest": nearest, "distances": distances}


def run_instruction_header_transfer(
    controller_payload: dict[str, object],
    source_bank_path: Path,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    source_cells: int = 2,
    target_cells: int = 3,
    primitive_slot: int = 0,
    composed_slot: int = 1,
    trials: int = 60,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    stable_sessions: int = 3,
    seed: int = 91_017,
) -> dict[str, Any]:
    """Rebind existing header routes to one new public cell-count line."""

    if (
        min(source_cells, target_cells, trials, minimum_bits, stable_sessions) < 1
        or source_cells == target_cells
        or not 0.0 <= threshold <= 1.0
        or primitive_slot == composed_slot
        or min(primitive_slot, composed_slot) < 0
    ):
        raise ValueError("instruction header transfer settings are invalid")
    output_directory.mkdir(parents=True, exist_ok=False)
    source_bank = ExternalTemporalProgramBank.load_bank(source_bank_path)
    if source_bank.configuration()["execution_schema"] != (
        RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
    ):
        raise ValueError("header transfer requires a recursive temporal bank")
    if source_bank.program_count <= max(primitive_slot, composed_slot):
        raise ValueError("source bank is missing the required program slots")
    primitive = source_bank.artifact(primitive_slot)
    composed = source_bank.artifact(composed_slot)
    if primitive.program_length != 1 or composed.program_length < 2:
        raise ValueError("source bank does not contain a depth-one and composed pair")
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    if not machine.accepts_controller_digest(source_bank.controller_digest):
        raise ValueError("source instruction bank targets another controller")
    controller_before = machine.controller_digest()
    bank = ExternalTemporalProgramBank.from_payload(source_bank.payload())
    started = time.perf_counter()
    instruction_encoder = NeuralWorkshopInstructionEncoder(
        _live_config(
            machine, n_back=1, active_cells=source_cells, trials=trials
        )
    )
    artifacts = {1: primitive, 2: composed}

    def encode_header(n_back: int, active_cells: int, session_seed: int) -> torch.Tensor:
        config = _live_config(
            machine,
            n_back=n_back,
            active_cells=active_cells,
            trials=trials,
        )
        environment, _verifier = build_neural_workshop_environment(
            neural_workshop_directory, config, seed=session_seed
        )
        try:
            return encode_instruction_context(
                environment.observe(), instruction_encoder
            )
        finally:
            environment.close()

    def execute(
        *,
        n_back: int,
        active_cells: int,
        session_seed: int,
        retrieve: bool,
        artifact=None,
    ) -> dict[str, Any]:
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
                if artifact is None:
                    raise ValueError("forced execution needs an external artifact")
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
            )
        except Exception:
            environment.close()
            raise
        if report.program_file_updates != 0:
            raise RuntimeError("header-transfer evaluation changed its program")
        return {
            **_summary(report),
            "selected_slot": selected_slot,
            "selected_propensity": selected_propensity,
            "route_known": bank.router.has_context(context),
            "context": context,
            "report": report,
        }

    trained = {
        "n1_source_cells": encode_header(1, source_cells, seed),
        "n2_source_cells": encode_header(2, source_cells, seed + 1),
    }
    probes = {
        "n1_target_cells": encode_header(1, target_cells, seed + 2),
        "n2_target_cells": encode_header(2, target_cells, seed + 3),
        "n3_source_cells": encode_header(3, source_cells, seed + 4),
        "n3_target_cells": encode_header(3, target_cells, seed + 5),
    }
    geometry = {
        name: {
            "exact_known": bank.router.has_context(context),
            **_nearest(context, trained),
        }
        for name, context in {**trained, **probes}.items()
    }
    source_depth_distance = float(
        torch.linalg.vector_norm(
            trained["n1_source_cells"] - trained["n2_source_cells"]
        ).item()
    )
    cell_shift = float(
        torch.linalg.vector_norm(
            trained["n2_source_cells"] - probes["n2_target_cells"]
        ).item()
    )
    unseen_depth_shift = float(
        torch.linalg.vector_norm(
            trained["n2_source_cells"] - probes["n3_source_cells"]
        ).item()
    )
    if source_depth_distance <= 0.0:
        raise RuntimeError("source 1-back and 2-back headers are not distinct")

    zeroshot: dict[str, dict[str, Any]] = {}
    zeroshot_bits = 0
    for name, n_back, cells, index in (
        ("n1_target_cells", 1, target_cells, 0),
        ("n2_target_cells", 2, target_cells, 1),
        ("n3_source_cells", 3, source_cells, 2),
    ):
        row = execute(
            n_back=n_back,
            active_cells=cells,
            session_seed=seed + 1_000 + index,
            retrieve=True,
        )
        _save_report(
            row["report"], output_directory / f"zeroshot-{name}.json"
        )
        zeroshot_bits += int(row["unique_verifier_bits"])
        zeroshot[name] = {
            **{key: row[key] for key in _summary(row["report"])},
            "selected_slot": row["selected_slot"],
            "selected_propensity": row["selected_propensity"],
            "route_known": row["route_known"],
            "geometry": geometry[name],
        }
        if row["route_known"]:
            raise RuntimeError(f"unseen header {name} matched a trained context")

    verification: dict[int, list[NeuralWorkshopLiveReport]] = {1: [], 2: []}
    verification_contexts: dict[int, torch.Tensor] = {}
    for n_back, artifact in artifacts.items():
        reports = []
        for index in range(stable_sessions):
            row = execute(
                n_back=n_back,
                active_cells=target_cells,
                session_seed=seed + 2_000 + n_back * 100 + index,
                retrieve=False,
                artifact=artifact,
            )
            reports.append(row["report"])
            _save_report(
                row["report"],
                output_directory
                / f"rebind-n{n_back}-cells-{target_cells}-session-{index + 1:03d}.json",
            )
            if n_back not in verification_contexts:
                verification_contexts[n_back] = row["context"]
        if not all(
            report.verifier_accuracy is not None
            and report.verifier_accuracy >= threshold
            and report.unique_verifier_bits >= minimum_bits
            for report in reports
        ):
            raise RuntimeError(
                f"source program failed {n_back}-back {target_cells}-cell verification"
            )
        verification[n_back] = reports

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
                f"verified {n_back}-back header rebind was rejected: {receipt.reason}"
            )
        if receipt.slot != (primitive_slot if n_back == 1 else composed_slot):
            raise RuntimeError("header rebind created a new program file")
        return receipt

    primitive_receipt = admit(1)
    composed_receipt = admit(2)

    heldout: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for n_back, expected_slot in (
        (1, primitive_receipt.slot),
        (2, composed_receipt.slot),
    ):
        for index in range(stable_sessions):
            row = execute(
                n_back=n_back,
                active_cells=target_cells,
                session_seed=seed + 3_000 + n_back * 100 + index,
                retrieve=True,
            )
            _save_report(
                row["report"],
                output_directory
                / f"retrieve-n{n_back}-cells-{target_cells}-session-{index + 1:03d}.json",
            )
            heldout[n_back].append(
                {
                    **_summary(row["report"]),
                    "selected_slot": row["selected_slot"],
                    "selected_propensity": row["selected_propensity"],
                    "expected_slot": expected_slot,
                    "route_known": row["route_known"],
                }
            )
            if row["selected_slot"] != expected_slot or not row["route_known"]:
                raise RuntimeError(
                    f"rebound header selected slot {row['selected_slot']} "
                    f"for {n_back}-back {target_cells}-cell"
                )

    retention = []
    for index in range(stable_sessions):
        row = execute(
            n_back=2,
            active_cells=source_cells,
            session_seed=seed + 4_000 + index,
            retrieve=True,
        )
        _save_report(
            row["report"],
            output_directory / f"retain-n2-cells-{source_cells}-session-{index + 1:03d}.json",
        )
        retention.append(
            {
                **_summary(row["report"]),
                "selected_slot": row["selected_slot"],
                "selected_propensity": row["selected_propensity"],
                "route_known": row["route_known"],
            }
        )
        if row["selected_slot"] != composed_slot or not row["route_known"]:
            raise RuntimeError("rebind overwrote the source-cell 2-back route")

    unseen = execute(
        n_back=3,
        active_cells=source_cells,
        session_seed=seed + 5_000,
        retrieve=True,
    )
    _save_report(unseen["report"], output_directory / "control-unseen-3back.json")
    if unseen["route_known"]:
        raise RuntimeError("unseen 3-back header claimed a learned route")

    bank_path = output_directory / DEFAULT_AGENT_BANK_FILENAME
    bank.save_bank(bank_path)
    result = {
        "schema": NEURAL_WORKSHOP_INSTRUCTION_HEADER_TRANSFER_SCHEMA,
        "source_bank": str(source_bank_path),
        "source_bank_sha256": _sha256(source_bank_path),
        "source_cells": source_cells,
        "target_cells": target_cells,
        "controller_digest": controller_before,
        "controller_frozen": machine.controller_digest() == controller_before,
        "geometry": {
            "source_depth_distance": source_depth_distance,
            "target_cell_shift_from_n2": cell_shift,
            "unseen_3back_shift_from_n2": unseen_depth_shift,
            "probes": geometry,
        },
        "zeroshot_exact_match": zeroshot,
        "zeroshot_unique_verifier_bits": zeroshot_bits,
        "verification": {
            str(n_back): [_summary(report) for report in reports]
            for n_back, reports in verification.items()
        },
        "rebind_unique_verifier_bits": sum(
            report.unique_verifier_bits
            for reports in verification.values()
            for report in reports
        ),
        "retrieval_search_unique_verifier_bits": 0,
        "admissions": {
            "n_back_1": primitive_receipt.payload(),
            "n_back_2": composed_receipt.payload(),
        },
        "heldout_target_retrieval": {
            str(n_back): rows for n_back, rows in heldout.items()
        },
        "source_retention": retention,
        "controls": {
            "unseen_3back": {
                **_summary(unseen["report"]),
                "selected_slot": unseen["selected_slot"],
                "selected_propensity": unseen["selected_propensity"],
                "route_known": unseen["route_known"],
                "geometry": geometry["n3_source_cells"],
            }
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
    parser.add_argument("--source-cells", type=int, default=2)
    parser.add_argument("--target-cells", type=int, default=3)
    parser.add_argument("--primitive-slot", type=int, default=0)
    parser.add_argument("--composed-slot", type=int, default=1)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-bits", type=int, default=8)
    parser.add_argument("--stable-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=91_017)
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
    report = run_instruction_header_transfer(
        controller,
        arguments.source_bank,
        arguments.neural_workshop,
        arguments.output_dir,
        source_cells=arguments.source_cells,
        target_cells=arguments.target_cells,
        primitive_slot=arguments.primitive_slot,
        composed_slot=arguments.composed_slot,
        trials=arguments.trials,
        threshold=arguments.threshold,
        minimum_bits=arguments.minimum_bits,
        stable_sessions=arguments.stable_sessions,
        seed=arguments.seed,
    )
    print(report)


if __name__ == "__main__":
    main()
