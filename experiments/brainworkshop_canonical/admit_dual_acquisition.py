"""Admit a gym Dual 1-back file into the canonical temporal bank.

This does not mention n-back inside the bank. Dual is public pixels plus
public PCM; the admitted object is a one-row temporal-address program.
Position slot 0 is never rewritten. An identical file attaches Dual
experience to the existing slot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .bank_program import admit_temporal_program, temporal_address_artifact
from .controller_pretraining import load_temporal_controller_artifact
from .rendered_dual_transfer_pilot import (
    _encoders,
    _new_machine,
    _train_sessions,
)
from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopVerifier,
)


def _dual_event_context(encoders, *, seed: int, steps: int = 24) -> list[list[float]]:
    verifier = RenderedBrainWorkshopVerifier(
        RenderedBrainWorkshopConfig(
            n_back=1, steps=steps, streams=("vision", "audio")
        ),
        seed=seed,
    )
    rows: list[list[float]] = []
    now = 0.0
    while not verifier.done:
        events = encoders.encode(verifier.observation(), now=now)
        present = events.present[0]
        for index in range(int(events.payload.shape[1])):
            if bool(present[index]):
                rows.append(
                    [float(value) for value in events.payload[0, index].tolist()]
                )
        verifier.score(torch.tensor([0], dtype=torch.long))
        now += 0.05
    return rows


def acquire_and_admit_dual(
    controller_payload: dict[str, object],
    bank_path: Path,
    *,
    steps: int = 48,
    sessions: int = 6,
    seed: int = 99_017,
    learning_rate: float = 0.3,
    threshold: float = 0.8,
    minimum_bits: int = 8,
) -> dict[str, object]:
    from neural_computer import ExternalTemporalProgramBank

    before = ExternalTemporalProgramBank.load_bank(bank_path)
    slot0 = before.artifact(0).digest()
    machine = _new_machine(controller_payload, learning_rate=learning_rate)
    encoders = _encoders(machine)
    train = _train_sessions(
        machine,
        encoders,
        n_back=1,
        steps=steps,
        seed=seed,
        sessions=sessions,
        threshold=threshold,
        minimum_bits=minimum_bits,
    )
    outcomes = [float(row["accuracy"]) for row in train]
    artifact = temporal_address_artifact(machine)
    context = _dual_event_context(encoders, seed=seed + 50, steps=min(24, steps))
    receipt = admit_temporal_program(
        bank_path,
        artifact,
        context,
        outcomes,
        machine=machine,
        threshold=threshold,
        min_observations=min(2, max(1, len(outcomes))),
        min_stable_observations=1,
    )
    after = ExternalTemporalProgramBank.load_bank(bank_path)
    return {
        "accepted": receipt.accepted,
        "reason": receipt.reason,
        "slot": receipt.slot,
        "program_count_before": before.program_count,
        "program_count_after": after.program_count,
        "slot0_digest_before": slot0,
        "slot0_digest_after": after.artifact(0).digest(),
        "slot0_unchanged": after.artifact(0).digest() == slot0,
        "candidate_digest": artifact.digest(),
        "controller_digest": machine.controller_digest(),
        "duplicate_of_slot0": artifact.digest() == slot0,
        "dual_1back_train": train,
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=99_017)
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
    report = acquire_and_admit_dual(
        load_temporal_controller_artifact(arguments.controller_artifact),
        arguments.bank,
        steps=arguments.steps,
        sessions=arguments.sessions,
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
