"""Outcome-only search over insertion of reusable control-flow fragments.

Splicing is a memory-side operation.  The search selects an opaque parent file,
an opaque fragment file, and an insertion boundary from scalar verifier
outcomes.  It never stores verifier rows or task labels in its persistent
state; a caller performs the separate admission transaction after a candidate
passes a stable prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .control_flow import (
    ControlFlowAdmissionReceipt,
    ControlFlowProgram,
    ControlFlowProgramMemory,
    evaluate_control_flow_admission,
)

CONTROL_FLOW_SPLICE_SEARCH_SCHEMA = (
    "neural-computer.external-control-flow-splice-search.v1"
)
CONTROL_FLOW_SPLICE_PROPOSAL_SCHEMA = (
    "neural-computer.external-control-flow-splice-proposal.v1"
)


def _candidate_digest(
    parent_slot: int,
    position: int,
    fragment_slot: int,
    program: ControlFlowProgram,
) -> str:
    payload = {
        "fragment_slot": fragment_slot,
        "parent_slot": parent_slot,
        "position": position,
        "program": program.digest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scoped_candidate_key(
    scope: str,
    parent_slot: int,
    position: int,
    fragment_slot: int,
    program: ControlFlowProgram,
) -> str:
    if not isinstance(scope, str) or not scope or "\0" in scope:
        raise ValueError("control-flow splice scope must be a non-empty opaque key")
    return f"{scope}\0{_candidate_digest(parent_slot, position, fragment_slot, program)}"


def _validate_digest(value: str, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} digest is malformed")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} digest is malformed") from error


@dataclass(frozen=True)
class ControlFlowSpliceSearchState:
    """Persistent aggregate state bound to one immutable file-memory view."""

    memory_digest: str
    slot_count: int
    min_program_length: int
    max_program_length: int
    seen_candidate_keys: tuple[str, ...] = ()
    proposals: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = CONTROL_FLOW_SPLICE_SEARCH_SCHEMA

    def validate(self) -> ControlFlowSpliceSearchState:
        if self.schema != CONTROL_FLOW_SPLICE_SEARCH_SCHEMA:
            raise ValueError("unsupported control-flow splice search schema")
        _validate_digest(self.memory_digest, label="control-flow splice memory")
        if self.slot_count < 1:
            raise ValueError("control-flow splice slot count must be positive")
        if self.min_program_length < 2 or self.max_program_length < self.min_program_length:
            raise ValueError("control-flow splice lengths are invalid")
        if self.proposals < 0 or self.accepted < 0 or self.accepted > self.proposals:
            raise ValueError("control-flow splice counters are invalid")
        if len(self.seen_candidate_keys) != self.proposals or len(
            set(self.seen_candidate_keys)
        ) != len(self.seen_candidate_keys):
            raise ValueError("control-flow splice candidate history is inconsistent")
        for key in self.seen_candidate_keys:
            if not isinstance(key, str) or "\0" not in key:
                raise ValueError("control-flow splice candidate key is malformed")
            scope, digest = key.rsplit("\0", 1)
            if not scope or "\0" in scope:
                raise ValueError("control-flow splice candidate scope is malformed")
            _validate_digest(digest, label="control-flow splice candidate")
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("control-flow splice best quality is invalid")
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

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowSpliceSearchState:
        if not isinstance(payload, dict):
            raise TypeError("control-flow splice search payload must be a mapping")
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
class ControlFlowSpliceProposal:
    """One opaque parent/position/fragment splice and its materialization."""

    parent_slot: int
    position: int
    fragment_slot: int
    program: ControlFlowProgram
    candidate_key: str
    attempt_id: int
    selection_probability: float
    scope: str = "default"
    schema: str = CONTROL_FLOW_SPLICE_PROPOSAL_SCHEMA

    def validate(self) -> ControlFlowSpliceProposal:
        if self.schema != CONTROL_FLOW_SPLICE_PROPOSAL_SCHEMA:
            raise ValueError("unsupported control-flow splice proposal schema")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.parent_slot, self.position, self.fragment_slot)
        ):
            raise ValueError("control-flow splice proposal coordinates are invalid")
        self.program.validate()
        expected = _scoped_candidate_key(
            self.scope,
            self.parent_slot,
            self.position,
            self.fragment_slot,
            self.program,
        )
        if self.candidate_key != expected:
            raise ValueError("control-flow splice proposal identity is inconsistent")
        if self.attempt_id < 0:
            raise ValueError("control-flow splice proposal attempt is invalid")
        if not math.isfinite(self.selection_probability) or not (
            0.0 < self.selection_probability <= 1.0
        ):
            raise ValueError("control-flow splice proposal probability is invalid")
        return self


@dataclass(frozen=True)
class ControlFlowSpliceFeedback:
    proposal: ControlFlowSpliceProposal
    receipt: ControlFlowAdmissionReceipt
    quality: float
    state: ControlFlowSpliceSearchState


class ControlFlowSpliceSearch:
    """Discover reusable fragment insertions from scalar outcomes only."""

    schema = CONTROL_FLOW_SPLICE_SEARCH_SCHEMA

    def __init__(
        self,
        memory: ControlFlowProgramMemory,
        *,
        min_program_length: int = 2,
        max_program_length: int = 16,
    ) -> None:
        if not isinstance(memory, ControlFlowProgramMemory):
            raise TypeError("control-flow splice search memory has the wrong type")
        if memory.file_count < 1:
            raise ValueError("control-flow splice search needs at least one file")
        if min_program_length < 2 or max_program_length < min_program_length:
            raise ValueError("control-flow splice lengths are invalid")
        self.memory = memory
        self.min_program_length = int(min_program_length)
        self.max_program_length = int(max_program_length)

    def initial_state(self) -> ControlFlowSpliceSearchState:
        return ControlFlowSpliceSearchState(
            memory_digest=self.memory.digest(),
            slot_count=self.memory.file_count,
            min_program_length=self.min_program_length,
            max_program_length=self.max_program_length,
        ).validate()

    def _validate_state(self, state: ControlFlowSpliceSearchState) -> None:
        state.validate()
        if state.memory_digest != self.memory.digest():
            raise ValueError("control-flow splice search memory changed")
        if state.slot_count != self.memory.file_count:
            raise ValueError("control-flow splice search slot count changed")
        if (
            state.min_program_length != self.min_program_length
            or state.max_program_length != self.max_program_length
        ):
            raise ValueError("control-flow splice search configuration changed")

    def _candidates(
        self,
        state: ControlFlowSpliceSearchState,
        scope: str,
    ) -> tuple[tuple[int, int, int, ControlFlowProgram, str], ...]:
        self._validate_state(state)
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("control-flow splice scope must be a non-empty opaque key")
        candidates: list[tuple[int, int, int, ControlFlowProgram, str]] = []
        for parent_slot in range(self.memory.file_count):
            parent = self.memory.program(parent_slot)
            for fragment_slot in range(self.memory.file_count):
                fragment = self.memory.program(fragment_slot)
                for position in range(len(parent.instructions)):
                    try:
                        program = self.memory.splice(
                            parent_slot,
                            position,
                            fragment_slot,
                        )
                    except (IndexError, TypeError, ValueError):
                        continue
                    if not self.min_program_length <= len(program.instructions) <= self.max_program_length:
                        continue
                    key = _scoped_candidate_key(
                        scope,
                        parent_slot,
                        position,
                        fragment_slot,
                        program,
                    )
                    if key not in state.seen_candidate_keys:
                        candidates.append(
                            (parent_slot, position, fragment_slot, program, key)
                        )
        return tuple(candidates)

    def _proposal(
        self,
        state: ControlFlowSpliceSearchState,
        candidate: tuple[int, int, int, ControlFlowProgram, str],
        probability: float,
        scope: str,
    ) -> ControlFlowSpliceProposal:
        parent_slot, position, fragment_slot, program, key = candidate
        return ControlFlowSpliceProposal(
            parent_slot,
            position,
            fragment_slot,
            program,
            key,
            state.proposals,
            probability,
            scope,
        ).validate()

    def propose(
        self,
        state: ControlFlowSpliceSearchState,
        *,
        generator: torch.Generator,
        scope: str = "default",
    ) -> ControlFlowSpliceProposal:
        candidates = self._candidates(state, scope)
        if not candidates:
            raise RuntimeError("control-flow splice neighborhood is exhausted")
        selected = int(torch.randint(len(candidates), (), generator=generator).item())
        return self._proposal(state, candidates[selected], 1.0 / len(candidates), scope)

    def propose_exhaustive(
        self,
        state: ControlFlowSpliceSearchState,
        *,
        scope: str = "default",
    ) -> ControlFlowSpliceProposal:
        candidates = self._candidates(state, scope)
        if not candidates:
            raise RuntimeError("control-flow splice neighborhood is exhausted")
        return self._proposal(state, candidates[0], 1.0 / len(candidates), scope)

    def record_outcomes(
        self,
        state: ControlFlowSpliceSearchState,
        proposal: ControlFlowSpliceProposal,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ControlFlowSpliceFeedback:
        self._validate_state(state)
        proposal.validate()
        if proposal.attempt_id != state.proposals:
            raise ValueError("control-flow splice proposal is out of sequence")
        if proposal.candidate_key in state.seen_candidate_keys:
            raise ValueError("control-flow splice candidate was already evaluated")
        try:
            materialized = self.memory.splice(
                proposal.parent_slot,
                proposal.position,
                proposal.fragment_slot,
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("control-flow splice proposal is not in memory") from error
        if materialized.digest() != proposal.program.digest():
            raise ValueError("control-flow splice proposal does not match memory")
        values = tuple(float(value) for value in outcomes)
        receipt = evaluate_control_flow_admission(
            proposal.program,
            values,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        quality = sum(values) / len(values)
        next_state = ControlFlowSpliceSearchState(
            state.memory_digest,
            state.slot_count,
            state.min_program_length,
            state.max_program_length,
            (*state.seen_candidate_keys, proposal.candidate_key),
            state.proposals + 1,
            state.accepted + int(receipt.accepted),
            max(state.best_quality, quality),
        ).validate()
        return ControlFlowSpliceFeedback(proposal, receipt, quality, next_state)


__all__ = [
    "CONTROL_FLOW_SPLICE_PROPOSAL_SCHEMA",
    "CONTROL_FLOW_SPLICE_SEARCH_SCHEMA",
    "ControlFlowSpliceFeedback",
    "ControlFlowSpliceProposal",
    "ControlFlowSpliceSearch",
    "ControlFlowSpliceSearchState",
]
