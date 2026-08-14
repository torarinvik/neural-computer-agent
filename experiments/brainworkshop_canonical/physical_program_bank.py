"""Bridge public physical experience to the generic temporal-program bank."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional as F

from neural_computer import (
    ExternalProgramAdmissionReceipt,
    ExternalTemporalProgramBank,
    TemporalProgramSelection,
)

from .physical_train import PhysicalTrainingCampaign
from .rendered_live import PretrainedControllerProgramMachine


def learned_event_context(
    event_payloads: Sequence[Sequence[float]] | torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    """Pool standardized learned events into one opaque memory address.

    This is deliberately not a task label.  Every input row must be a frontend
    event that was available on the deployed amodal bus during experience.
    """

    if isinstance(event_payloads, torch.Tensor):
        values = event_payloads.detach().to(device="cpu", dtype=torch.float32)
    else:
        rows = tuple(event_payloads)
        if not rows:
            raise ValueError("learned event payloads cannot be empty")
        values = torch.stack(
            [
                torch.as_tensor(row, dtype=torch.float32).detach().cpu()
                for row in rows
            ]
        )
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != width:
        raise ValueError(f"learned event payloads must have shape [events, {width}]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("learned event payloads must contain finite values")
    context = values.mean(dim=0)
    if float(torch.linalg.vector_norm(context)) <= 1e-8:
        # A zero mean can occur for a symmetric learned representation.  Its
        # second moment is still ordinary event evidence, not hidden state.
        context = values.square().mean(dim=0)
    if float(torch.linalg.vector_norm(context)) <= 1e-8:
        raise ValueError("learned event experience contains no address evidence")
    return F.normalize(context, dim=0)


def _campaign_event_payloads(directory: Path) -> tuple[tuple[float, ...], ...]:
    payloads: list[tuple[float, ...]] = []
    for path in sorted(Path(directory).glob("session-*.json")):
        report = json.loads(path.read_text())
        rows = report.get("event_payloads")
        if not isinstance(rows, list):
            raise TypeError(f"physical session lacks learned event payloads: {path}")
        for row in rows:
            if not isinstance(row, list):
                raise TypeError("physical learned event payload must be a list")
            payloads.append(tuple(float(value) for value in row))
    if not payloads:
        raise ValueError("physical campaign contains no learned event payloads")
    return tuple(payloads)


def admit_physical_training_program(
    machine: PretrainedControllerProgramMachine,
    campaign: PhysicalTrainingCampaign,
    campaign_directory: Path,
    bank_path: Path,
    *,
    threshold: float = 0.8,
    min_lifetimes: int = 8,
) -> ExternalProgramAdmissionReceipt:
    """Admit a learned program using public per-lifetime scores only."""

    machine.assert_controller_frozen()
    if bank_path.exists():
        bank = ExternalTemporalProgramBank.load_bank(bank_path)
        if bank.controller_digest != machine.controller_digest():
            raise ValueError("physical temporal program bank targets another controller")
        if bank.context_width != machine.event_width:
            raise ValueError("physical temporal program bank context width changed")
    else:
        bank = ExternalTemporalProgramBank(
            machine.event_width,
            machine.max_history,
            controller_digest=machine.controller_digest(),
            generalization_tolerance=0.25,
            mastery_threshold=threshold,
            min_mastery_observations=min_lifetimes,
        )
    context = learned_event_context(
        _campaign_event_payloads(campaign_directory), width=machine.event_width
    )
    outcomes = [session.accuracy for session in campaign.sessions]
    before = bank.digest()
    receipt = bank.admit(
        machine.admitted_program_artifact(),
        context,
        outcomes,
        threshold=threshold,
        min_observations=min_lifetimes,
        min_stable_observations=min_lifetimes,
    )
    if receipt.accepted:
        bank.save_bank(bank_path)
    elif bank.digest() != before:
        raise RuntimeError("rejected physical program changed the live bank")
    receipt_path = bank_path.with_suffix(bank_path.suffix + ".admission.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt.payload(), indent=2, sort_keys=True) + "\n")
    return receipt


def retrieve_physical_program(
    machine: PretrainedControllerProgramMachine,
    bank: ExternalTemporalProgramBank,
    event_payloads: Sequence[Sequence[float]] | torch.Tensor,
    *,
    exploration: float = 0.0,
    sample: bool = False,
) -> TemporalProgramSelection:
    """Select by learned events and activate immutable frozen execution."""

    context = learned_event_context(event_payloads, width=machine.event_width)
    selection = bank.select(
        context,
        exploration=exploration,
        sample=sample,
    )
    machine.load_admitted_program_artifact(
        selection.artifact,
        controller_digest=bank.controller_digest,
    )
    return selection


__all__ = [
    "admit_physical_training_program",
    "learned_event_context",
    "retrieve_physical_program",
]
