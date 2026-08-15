"""Execute one admitted temporal slot on Dual I/O without learning.

Rendered and Neural Workshop Dual share the bank controller digest.
Packed actions are an adapter. Slot 0 is not rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from neural_computer import ExternalTemporalProgramBank

from .bank_program import install_temporal_artifact
from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_dual_live_pilot import _dual_config
from .neural_workshop_live import (
    build_neural_workshop_environment,
    run_neural_workshop_live_lifetime,
)
from .program_search import search_temporal_programs
from .rendered_dual_transfer_pilot import _encoders
from .rendered_environment import RenderedBrainWorkshopConfig
from .rendered_live import run_rendered_live_lifetime

EXECUTE_BANK_SLOT_SCHEMA = "neural-computer.execute-bank-slot.v1"


def _dual_machine(controller_payload: dict[str, object]):
    return build_recursive_temporal_program_machine(
        controller_payload,
        sample=False,
        max_sources=2,
        pack_source_actions=True,
    )


def load_bank_slot(machine, bank: ExternalTemporalProgramBank, slot: int) -> None:
    if not machine.accepts_controller_digest(bank.controller_digest):
        raise ValueError("bank targets another frozen controller")
    install_temporal_artifact(machine, bank, bank.artifact(slot))
    machine.learning_enabled = False
    machine.sample = False


def search_and_install(
    machine,
    bank: ExternalTemporalProgramBank,
    *,
    n_back: int,
    steps: int,
    seed: int,
) -> dict[str, object]:
    """Select a bank file on rendered Dual without a hardcoded slot."""

    encoders = _encoders(machine)

    def evaluate(proposal):
        del proposal
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=n_back, steps=steps, streams=("vision", "audio")
            ),
            seed=seed,
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    search = search_temporal_programs(
        bank, machine, evaluate, threshold=0.8, minimum_bits=4
    )
    if search["winner"] is None:
        raise RuntimeError("program search found no executable Dual file")
    machine.learning_enabled = False
    machine.sample = False
    return search


def execute_rendered_dual(
    machine,
    *,
    n_back: int,
    steps: int,
    seed: int,
) -> dict[str, object]:
    report = run_rendered_live_lifetime(
        machine,
        _encoders(machine),
        RenderedBrainWorkshopConfig(
            n_back=n_back, steps=steps, streams=("vision", "audio")
        ),
        seed=seed,
        learn=False,
        sample=False,
    )
    return {
        "substrate": "rendered",
        "n_back": n_back,
        "accuracy": report.eligible_accuracy,
        "unique_verifier_bits": report.unique_verifier_bits,
        "program_file_updates": report.program_file_updates,
        "optimizer_updates": report.optimizer_updates,
        "input_events": report.input_events,
        "controller_digest": machine.controller_digest(),
    }


def execute_neural_workshop_dual(
    machine,
    neural_workshop_directory: Path,
    *,
    n_back: int,
    trials: int,
    seed: int,
) -> dict[str, object]:
    config = _dual_config(machine, n_back=n_back, trials=trials)
    environment, verifier = build_neural_workshop_environment(
        neural_workshop_directory, config, seed=seed
    )
    report = run_neural_workshop_live_lifetime(
        machine,
        config,
        seed=seed,
        environment=environment,
        verifier=verifier,
        learn=False,
        sample=False,
    )
    packed = None if not report.rewards else sum(report.rewards) / len(report.rewards)
    if (
        not report.audio_payloads
        or len(report.audio_payloads) != len(report.event_payloads)
    ):
        raise RuntimeError("Neural Workshop Dual lost the public audio stream")
    return {
        "substrate": "neural_workshop",
        "n_back": n_back,
        "accuracy": report.verifier_accuracy,
        "packed_exact_accuracy": packed,
        "unique_verifier_bits": report.unique_verifier_bits,
        "program_file_updates": report.program_file_updates,
        "optimizer_updates": report.optimizer_updates,
        "audio_events": len(report.audio_payloads),
        "vision_events": len(report.event_payloads),
        "audio_aligned": len(report.audio_payloads) == len(report.event_payloads),
        "controller_digest": machine.controller_digest(),
        "controller_frozen": report.controller_frozen,
    }


def run_slot_execute(
    controller_payload: dict[str, object],
    bank_path: Path,
    *,
    slot: int,
    rendered_steps: int = 48,
    seed: int = 113_117,
    neural_workshop: Path | None = None,
    trials: int = 60,
    search: bool = False,
    search_n_back: int = 1,
) -> dict[str, object]:
    started = perf_counter()
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    slot0 = bank.artifact(0).digest()
    machine = _dual_machine(controller_payload)
    search_report = None
    if search:
        search_report = search_and_install(
            machine,
            bank,
            n_back=search_n_back,
            steps=rendered_steps,
            seed=seed,
        )
        winner = search_report["winner"]
        slot = int(winner["slots"][0]) if winner["kind"] == "retrieve" else slot
    else:
        load_bank_slot(machine, bank, slot)
    rendered = execute_rendered_dual(
        machine, n_back=1, steps=rendered_steps, seed=seed
    )
    neural = None
    if neural_workshop is not None:
        load_bank_slot(machine, bank, slot)
        neural = execute_neural_workshop_dual(
            machine,
            neural_workshop,
            n_back=1,
            trials=trials,
            seed=seed + 1,
        )
    after = ExternalTemporalProgramBank.load_bank(bank_path)
    return {
        "schema": EXECUTE_BANK_SLOT_SCHEMA,
        "search": search_report,
        "slot": slot,
        "slot_digest": bank.artifact(slot).digest(),
        "controller_digest": machine.controller_digest(),
        "bank_controller_digest": bank.controller_digest,
        "same_digest": machine.controller_digest() == bank.controller_digest,
        "slot0_unchanged": after.artifact(0).digest() == slot0,
        "program_count": after.program_count,
        "rendered_dual_1back": rendered,
        "neural_workshop_dual_1back": neural,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument(
        "--search",
        action="store_true",
        help="select the Dual file from the bank instead of --slot",
    )
    parser.add_argument("--search-n-back", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=113_117)
    parser.add_argument("--neural-workshop", type=Path, default=None)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument("--report-out", type=Path, default=None)
    arguments = parser.parse_args()
    report = run_slot_execute(
        load_temporal_controller_artifact(arguments.controller_artifact),
        arguments.bank,
        slot=arguments.slot,
        rendered_steps=arguments.steps,
        seed=arguments.seed,
        neural_workshop=arguments.neural_workshop,
        trials=arguments.trials,
        search=arguments.search,
        search_n_back=arguments.search_n_back,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)


if __name__ == "__main__":
    main()
