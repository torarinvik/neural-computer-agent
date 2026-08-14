"""Outcome-only header routing: retrieve, rebind, or compose one step.

The controller never sees ``n_back``. An unknown public header first tries
an exact or same-slot invariant match, then existing files ordered by
header distance and program length, then ``PREVIOUS`` composed one step
deeper than the deepest verified file. Capacity overflow fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_computer import (
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
    compose_recursive_temporal_program,
)

HEADER_INVARIANT_RESIDUAL_TOLERANCE = 0.02


@dataclass(frozen=True)
class HeaderProgramDecision:
    kind: str
    artifact: ExternalProgramArtifact | None
    slot: int | None
    proposed_depth: int | None
    known: bool

    def validate(self) -> HeaderProgramDecision:
        if self.kind not in {
            "exact",
            "invariant",
            "try_existing",
            "compose",
            "capacity",
        }:
            raise ValueError(f"unsupported header decision: {self.kind}")
        if self.kind == "capacity":
            if self.artifact is not None or self.slot is not None:
                raise ValueError("capacity decisions cannot carry a program")
            return self
        if self.artifact is None:
            raise ValueError("header decision is missing a program")
        if self.kind in {"exact", "invariant", "try_existing"} and self.slot is None:
            raise ValueError("existing-program decisions need a slot")
        if self.kind == "compose" and self.slot is not None:
            raise ValueError("compose decisions cannot reuse a live slot")
        return self


def existing_program_try_order(
    bank: ExternalTemporalProgramBank,
    context: torch.Tensor,
) -> tuple[int, ...]:
    """Order existing files by header distance, then shorter programs."""

    if bank.program_count < 1:
        return ()
    key = torch.nn.functional.normalize(
        context.detach().to(device="cpu", dtype=torch.float32), dim=0
    )
    scored: list[tuple[float, int, int]] = []
    for slot in range(bank.program_count):
        keys = bank.router.preferred_keys_for_slot(slot)
        if keys:
            distance = min(
                float(torch.linalg.vector_norm(key - candidate)) for candidate in keys
            )
        else:
            distance = float("inf")
        scored.append((distance, bank.artifact(slot).program_length, slot))
    scored.sort()
    return tuple(slot for _distance, _length, slot in scored)


def decide_header_program(
    bank: ExternalTemporalProgramBank,
    context: torch.Tensor,
    primitive: ExternalProgramArtifact,
    *,
    failed_slots: frozenset[int] = frozenset(),
    failed_depths: frozenset[int] = frozenset(),
    max_history: int,
    residual_tolerance: float = HEADER_INVARIANT_RESIDUAL_TOLERANCE,
) -> HeaderProgramDecision:
    """Choose the next header-conditioned program without a task label."""

    if max_history < 1:
        raise ValueError("header decisions need a positive history capacity")
    if bank.program_count:
        if bank.router.has_context(context):
            selection = bank.select(context)
            if selection.slot not in failed_slots:
                return HeaderProgramDecision(
                    "exact",
                    selection.artifact,
                    selection.slot,
                    selection.artifact.program_length,
                    True,
                ).validate()
        invariant = bank.router.invariant_preferred_slot(
            context, residual_tolerance=residual_tolerance
        )
        if invariant is not None and invariant not in failed_slots:
            artifact = bank.artifact(invariant)
            return HeaderProgramDecision(
                "invariant",
                artifact,
                invariant,
                artifact.program_length,
                True,
            ).validate()
        for slot in existing_program_try_order(bank, context):
            if slot in failed_slots:
                continue
            artifact = bank.artifact(slot)
            return HeaderProgramDecision(
                "try_existing",
                artifact,
                slot,
                artifact.program_length,
                False,
            ).validate()
    deepest = max(
        (bank.artifact(index).program_length for index in range(bank.program_count)),
        default=0,
    )
    depth = 1 if deepest < 1 else deepest + 1
    while depth in failed_depths:
        depth += 1
    if depth > max_history:
        return HeaderProgramDecision("capacity", None, None, depth, False).validate()
    child = compose_recursive_temporal_program(primitive, depth)
    if any(
        bank.artifact(index).digest() == child.digest()
        for index in range(bank.program_count)
    ):
        return decide_header_program(
            bank,
            context,
            primitive,
            failed_slots=failed_slots,
            failed_depths=failed_depths | {depth},
            max_history=max_history,
            residual_tolerance=residual_tolerance,
        )
    return HeaderProgramDecision(
        "compose", child, None, depth, False
    ).validate()
