"""Outcome-only search over compositions of opaque control-flow files.

Composition is a memory-side operation.  The search can discover an ordered
sequence of existing files from scalar verifier outcomes, but it does not
write to durable program memory and it never persists individual verifier
rows.  A caller must perform the separate admission transaction after a
candidate passes a stable verifier prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import torch

from .control_flow import (
    ControlFlowAdmissionReceipt,
    ControlFlowProgram,
    ControlFlowProgramMemory,
    evaluate_control_flow_admission,
)

CONTROL_FLOW_COMPOSITION_SEARCH_SCHEMA = (
    "neural-computer.external-control-flow-composition-search.v1"
)
CONTROL_FLOW_COMPOSITION_PROPOSAL_SCHEMA = (
    "neural-computer.external-control-flow-composition-proposal.v1"
)


def _candidate_digest(slots: tuple[int, ...], program: ControlFlowProgram) -> str:
    payload = {
        "slots": list(slots),
        "program": program.digest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scoped_candidate_key(scope: str, slots: tuple[int, ...], program: ControlFlowProgram) -> str:
    if not isinstance(scope, str) or not scope or "\0" in scope:
        raise ValueError("control-flow composition scope must be a non-empty opaque key")
    return f"{scope}\0{_candidate_digest(slots, program)}"


def _validate_digest(value: str, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} digest is malformed")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} digest is malformed") from error


@dataclass(frozen=True)
class ControlFlowCompositionSearchState:
    """Persistent aggregate state bound to one immutable file-memory view."""

    memory_digest: str
    slot_count: int
    min_program_length: int
    max_program_length: int
    seen_candidate_keys: tuple[str, ...] = ()
    proposals: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = CONTROL_FLOW_COMPOSITION_SEARCH_SCHEMA

    def validate(self) -> ControlFlowCompositionSearchState:
        if self.schema != CONTROL_FLOW_COMPOSITION_SEARCH_SCHEMA:
            raise ValueError("unsupported control-flow composition search schema")
        _validate_digest(self.memory_digest, label="control-flow composition memory")
        if self.slot_count < 1:
            raise ValueError("control-flow composition slot count must be positive")
        if self.min_program_length < 2 or self.max_program_length < self.min_program_length:
            raise ValueError("control-flow composition lengths are invalid")
        if self.proposals < 0 or self.accepted < 0 or self.accepted > self.proposals:
            raise ValueError("control-flow composition counters are invalid")
        if len(self.seen_candidate_keys) != self.proposals or len(
            set(self.seen_candidate_keys)
        ) != len(self.seen_candidate_keys):
            raise ValueError("control-flow composition candidate history is inconsistent")
        for key in self.seen_candidate_keys:
            if not isinstance(key, str) or "\0" not in key:
                raise ValueError("control-flow composition candidate key is malformed")
            scope, digest = key.rsplit("\0", 1)
            if not scope or "\0" in scope:
                raise ValueError("control-flow composition candidate scope is malformed")
            _validate_digest(digest, label="control-flow composition candidate")
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("control-flow composition best quality is invalid")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "memory_digest": self.memory_digest,
            "slot_count": self.slot_count,
            "min_program_length": self.min_program_length,
            "max_program_length": self.max_program_length,
            "seen_candidate_keys": list(self.seen_candidate_keys),
            "proposals": self.proposals,
            "accepted": self.accepted,
            "best_quality": self.best_quality,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> ControlFlowCompositionSearchState:
        if not isinstance(payload, dict):
            raise TypeError("control-flow composition search payload must be a mapping")
        return cls(
            memory_digest=str(payload.get("memory_digest", "")),
            slot_count=int(payload.get("slot_count", -1)),
            min_program_length=int(payload.get("min_program_length", -1)),
            max_program_length=int(payload.get("max_program_length", -1)),
            seen_candidate_keys=tuple(payload.get("seen_candidate_keys", ())),
            proposals=int(payload.get("proposals", -1)),
            accepted=int(payload.get("accepted", -1)),
            best_quality=float(payload.get("best_quality", float("nan"))),
            schema=str(payload.get("schema", "")),
        ).validate()


@dataclass(frozen=True)
class ControlFlowCompositionProposal:
    """One opaque ordered file sequence and its materialized program."""

    slots: tuple[int, ...]
    program: ControlFlowProgram
    candidate_key: str
    attempt_id: int
    selection_probability: float
    scope: str = "default"
    schema: str = CONTROL_FLOW_COMPOSITION_PROPOSAL_SCHEMA

    def validate(self) -> ControlFlowCompositionProposal:
        if self.schema != CONTROL_FLOW_COMPOSITION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported control-flow composition proposal schema")
        if len(self.slots) < 2 or any(
            not isinstance(slot, int) or isinstance(slot, bool) or slot < 0
            for slot in self.slots
        ):
            raise ValueError("control-flow composition proposal slots are invalid")
        self.program.validate()
        expected = _scoped_candidate_key(self.scope, self.slots, self.program)
        if self.candidate_key != expected:
            raise ValueError("control-flow composition proposal identity is inconsistent")
        if self.attempt_id < 0:
            raise ValueError("control-flow composition proposal attempt is invalid")
        if not math.isfinite(self.selection_probability) or not (
            0.0 < self.selection_probability <= 1.0
        ):
            raise ValueError("control-flow composition proposal probability is invalid")
        return self


@dataclass(frozen=True)
class ControlFlowCompositionFeedback:
    proposal: ControlFlowCompositionProposal
    receipt: ControlFlowAdmissionReceipt
    quality: float
    state: ControlFlowCompositionSearchState


class ControlFlowCompositionSearch:
    """Discover ordered external-file compositions from scalar outcomes only."""

    schema = CONTROL_FLOW_COMPOSITION_SEARCH_SCHEMA

    def __init__(
        self,
        memory: ControlFlowProgramMemory,
        *,
        min_program_length: int = 2,
        max_program_length: int = 2,
    ) -> None:
        if not isinstance(memory, ControlFlowProgramMemory):
            raise TypeError("control-flow composition search memory has the wrong type")
        if memory.file_count < 1:
            raise ValueError("control-flow composition search needs at least one file")
        if min_program_length < 2 or max_program_length < min_program_length:
            raise ValueError("control-flow composition lengths are invalid")
        self.memory = memory
        self.min_program_length = int(min_program_length)
        self.max_program_length = int(max_program_length)

    def initial_state(self) -> ControlFlowCompositionSearchState:
        return ControlFlowCompositionSearchState(
            memory_digest=self.memory.digest(),
            slot_count=self.memory.file_count,
            min_program_length=self.min_program_length,
            max_program_length=self.max_program_length,
        ).validate()

    def _validate_state(self, state: ControlFlowCompositionSearchState) -> None:
        state.validate()
        if state.memory_digest != self.memory.digest():
            raise ValueError("control-flow composition search memory changed")
        if state.slot_count != self.memory.file_count:
            raise ValueError("control-flow composition search slot count changed")
        if (
            state.min_program_length != self.min_program_length
            or state.max_program_length != self.max_program_length
        ):
            raise ValueError("control-flow composition search configuration changed")

    def _candidates(
        self,
        state: ControlFlowCompositionSearchState,
        scope: str,
    ) -> tuple[tuple[tuple[int, ...], ControlFlowProgram, str], ...]:
        self._validate_state(state)
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("control-flow composition scope must be a non-empty opaque key")
        candidates: list[tuple[tuple[int, ...], ControlFlowProgram, str]] = []
        for length in range(self.min_program_length, self.max_program_length + 1):
            for slots in product(range(self.memory.file_count), repeat=length):
                normalized = tuple(int(slot) for slot in slots)
                try:
                    program = self.memory.compose(normalized)
                except (IndexError, TypeError, ValueError):
                    continue
                key = _scoped_candidate_key(scope, normalized, program)
                if key not in state.seen_candidate_keys:
                    candidates.append((normalized, program, key))
        return tuple(candidates)

    def propose(
        self,
        state: ControlFlowCompositionSearchState,
        *,
        generator: torch.Generator,
        scope: str = "default",
    ) -> ControlFlowCompositionProposal:
        candidates = self._candidates(state, scope)
        if not candidates:
            raise RuntimeError("control-flow composition neighborhood is exhausted")
        selected = int(torch.randint(len(candidates), (), generator=generator).item())
        slots, program, key = candidates[selected]
        return ControlFlowCompositionProposal(
            slots,
            program,
            key,
            state.proposals,
            1.0 / len(candidates),
            scope,
        ).validate()

    def propose_exhaustive(
        self,
        state: ControlFlowCompositionSearchState,
        *,
        scope: str = "default",
    ) -> ControlFlowCompositionProposal:
        candidates = self._candidates(state, scope)
        if not candidates:
            raise RuntimeError("control-flow composition neighborhood is exhausted")
        slots, program, key = candidates[0]
        return ControlFlowCompositionProposal(
            slots,
            program,
            key,
            state.proposals,
            1.0 / len(candidates),
            scope,
        ).validate()

    def record_outcomes(
        self,
        state: ControlFlowCompositionSearchState,
        proposal: ControlFlowCompositionProposal,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ControlFlowCompositionFeedback:
        self._validate_state(state)
        proposal.validate()
        if proposal.attempt_id != state.proposals:
            raise ValueError("control-flow composition proposal is out of sequence")
        if proposal.candidate_key in state.seen_candidate_keys:
            raise ValueError("control-flow composition candidate was already evaluated")
        try:
            materialized = self.memory.compose(proposal.slots)
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("control-flow composition proposal is not in memory") from error
        if materialized.digest() != proposal.program.digest():
            raise ValueError("control-flow composition proposal does not match memory")
        values = tuple(float(value) for value in outcomes)
        receipt = evaluate_control_flow_admission(
            proposal.program,
            values,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        quality = sum(values) / len(values)
        next_state = ControlFlowCompositionSearchState(
            state.memory_digest,
            state.slot_count,
            state.min_program_length,
            state.max_program_length,
            (*state.seen_candidate_keys, proposal.candidate_key),
            state.proposals + 1,
            state.accepted + int(receipt.accepted),
            max(state.best_quality, quality),
        ).validate()
        return ControlFlowCompositionFeedback(proposal, receipt, quality, next_state)


__all__ = [
    "CONTROL_FLOW_COMPOSITION_PROPOSAL_SCHEMA",
    "CONTROL_FLOW_COMPOSITION_SEARCH_SCHEMA",
    "ControlFlowCompositionFeedback",
    "ControlFlowCompositionProposal",
    "ControlFlowCompositionSearch",
    "ControlFlowCompositionSearchState",
]
