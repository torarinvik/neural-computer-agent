"""Outcome-only proposal search for generic external control-flow programs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import torch

from .control_flow import ControlFlowInstruction, ControlFlowProgram

CONTROL_FLOW_SEARCH_SCHEMA = "neural-computer.external-control-flow-search.v1"
CONTROL_FLOW_PROPOSAL_SCHEMA = "neural-computer.external-control-flow-proposal.v1"
CONTROL_FLOW_MUTATION_OPERATORS = ("replace_instruction", "swap_instructions")


def control_flow_instruction_bank(
    *, counter_count: int, program_length: int
) -> tuple[ControlFlowInstruction, ...]:
    """Return the generic target-agnostic instruction basis for one length."""

    if counter_count < 2 or program_length < 1:
        raise ValueError("control-flow instruction-bank bounds are invalid")
    instructions: list[ControlFlowInstruction] = []
    for counter in range(counter_count):
        instructions.extend(
            (
                ControlFlowInstruction("inc", counter=counter),
                ControlFlowInstruction("dec", counter=counter),
            )
        )
    for target in range(program_length):
        instructions.append(ControlFlowInstruction("jump", target=target))
        for counter in range(counter_count):
            instructions.extend(
                (
                    ControlFlowInstruction(
                        "jump_if_zero", counter=counter, target=target
                    ),
                    ControlFlowInstruction(
                        "jump_if_nonzero", counter=counter, target=target
                    ),
                )
            )
    return tuple(instructions)


def iter_control_flow_programs(
    *,
    counter_count: int,
    min_length: int = 1,
    max_length: int,
):
    """Enumerate a finite generic program space without target knowledge.

    The final instruction is always ``HALT``.  All earlier positions range
    over the generic counter/jump basis, including every in-range target.
    Callers must treat an interrupted iterator as budget exhaustion rather than
    as proof that a target is inexpressible.
    """

    if counter_count < 2 or min_length < 1 or max_length < min_length:
        raise ValueError("control-flow enumeration bounds are invalid")
    for length in range(min_length, max_length + 1):
        atoms = control_flow_instruction_bank(
            counter_count=counter_count,
            program_length=length,
        )
        halt = ControlFlowInstruction("halt")
        for prefix in product(atoms, repeat=length - 1):
            yield ControlFlowProgram(counter_count, (*prefix, halt)).validate()


def _scoped_digest(scope: str, digest: str) -> str:
    return hashlib.sha256(f"{scope}:{digest}".encode()).hexdigest()


@dataclass(frozen=True)
class ControlFlowSearchState:
    """Aggregate proposal state; raw verifier outcomes are never retained."""

    reward_totals: torch.Tensor
    reward_counts: torch.Tensor
    accepted_counts: torch.Tensor
    failed_counts: torch.Tensor
    seen_candidate_digests: tuple[str, ...] = ()
    proposals: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = CONTROL_FLOW_SEARCH_SCHEMA

    def validate(self) -> ControlFlowSearchState:
        if self.schema != CONTROL_FLOW_SEARCH_SCHEMA:
            raise ValueError("unsupported control-flow search schema")
        expected = (len(CONTROL_FLOW_MUTATION_OPERATORS),)
        for value in (
            self.reward_totals,
            self.reward_counts,
            self.accepted_counts,
            self.failed_counts,
        ):
            if value.shape != expected or not value.is_floating_point():
                raise ValueError("control-flow search statistics have the wrong shape")
            if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0.0)):
                raise ValueError("control-flow search statistics are invalid")
        if self.proposals < 0 or self.accepted < 0 or self.accepted > self.proposals:
            raise ValueError("control-flow search counters are invalid")
        if int(self.reward_counts.sum().item()) != self.proposals:
            raise ValueError("control-flow search proposal count is inconsistent")
        if int(self.accepted_counts.sum().item()) != self.accepted:
            raise ValueError("control-flow search acceptance count is inconsistent")
        if int(self.failed_counts.sum().item()) != self.proposals - self.accepted:
            raise ValueError("control-flow search failure count is inconsistent")
        if len(self.seen_candidate_digests) != self.proposals:
            raise ValueError("control-flow search digest count is inconsistent")
        if len(set(self.seen_candidate_digests)) != len(self.seen_candidate_digests):
            raise ValueError("control-flow search candidate digests are not unique")
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("control-flow search best quality is invalid")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "reward_totals": self.reward_totals.detach().cpu().clone(),
            "reward_counts": self.reward_counts.detach().cpu().clone(),
            "accepted_counts": self.accepted_counts.detach().cpu().clone(),
            "failed_counts": self.failed_counts.detach().cpu().clone(),
            "seen_candidate_digests": list(self.seen_candidate_digests),
            "proposals": self.proposals,
            "accepted": self.accepted,
            "best_quality": self.best_quality,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ControlFlowSearchState:
        required = (
            "reward_totals",
            "reward_counts",
            "accepted_counts",
            "failed_counts",
        )
        if not isinstance(payload, dict) or any(
            not isinstance(payload.get(name), torch.Tensor) for name in required
        ):
            raise TypeError("control-flow search payload is incomplete")
        return cls(
            reward_totals=payload["reward_totals"],
            reward_counts=payload["reward_counts"],
            accepted_counts=payload["accepted_counts"],
            failed_counts=payload["failed_counts"],
            seen_candidate_digests=tuple(payload.get("seen_candidate_digests", ())),
            proposals=int(payload.get("proposals", -1)),
            accepted=int(payload.get("accepted", -1)),
            best_quality=float(payload.get("best_quality", float("nan"))),
            schema=payload.get("schema"),
        ).validate()


@dataclass(frozen=True)
class ControlFlowProposal:
    program: ControlFlowProgram
    parent_digest: str
    operator: str
    operator_index: int
    attempt_id: int
    selection_probability: float
    scope: str
    schema: str = CONTROL_FLOW_PROPOSAL_SCHEMA

    def validate(self) -> ControlFlowProposal:
        self.program.validate()
        if self.schema != CONTROL_FLOW_PROPOSAL_SCHEMA:
            raise ValueError("unsupported control-flow proposal schema")
        if len(self.parent_digest) != 64 or len(self.scope) < 1:
            raise ValueError("control-flow proposal identity is malformed")
        try:
            int(self.parent_digest, 16)
        except ValueError as error:
            raise ValueError("control-flow proposal parent digest is malformed") from error
        if self.operator not in CONTROL_FLOW_MUTATION_OPERATORS:
            raise ValueError("control-flow proposal operator is unknown")
        if not 0 <= self.operator_index < len(CONTROL_FLOW_MUTATION_OPERATORS):
            raise ValueError("control-flow proposal operator index is invalid")
        if CONTROL_FLOW_MUTATION_OPERATORS[self.operator_index] != self.operator:
            raise ValueError("control-flow proposal operator/index disagree")
        if self.attempt_id < 0 or not 0.0 < self.selection_probability <= 1.0:
            raise ValueError("control-flow proposal metadata is invalid")
        if self.program.digest() == self.parent_digest:
            raise ValueError("control-flow proposal must change its parent")
        return self


@dataclass(frozen=True)
class ControlFlowFeedback:
    proposal: ControlFlowProposal
    quality: float
    accepted: bool
    stable_bits_to_threshold: int | None
    state: ControlFlowSearchState


class ControlFlowOutcomeSearch:
    """Learn generic edit credit from scalar outcomes only.

    The neighborhood is deliberately finite and structural: one instruction
    replacement or one swap in a caller-supplied parent.  If the sampled
    neighborhood is exhausted, proposals are selected from its remaining
    members without converting exhaustion into an inexpressibility claim.
    """

    schema = CONTROL_FLOW_SEARCH_SCHEMA

    def __init__(
        self,
        *,
        exploration: float = 0.5,
        temperature: float = 0.5,
    ) -> None:
        if not math.isfinite(exploration) or exploration < 0.0:
            raise ValueError("control-flow search exploration is invalid")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("control-flow search temperature is invalid")
        self.exploration = float(exploration)
        self.temperature = float(temperature)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "operators": CONTROL_FLOW_MUTATION_OPERATORS,
            "updates": "scalar_verifier_aggregate_only_v1",
            "scope": "caller_owned_opaque_binding_v1",
        }

    def initial_state(self) -> ControlFlowSearchState:
        zeros = torch.zeros(len(CONTROL_FLOW_MUTATION_OPERATORS), dtype=torch.float64)
        return ControlFlowSearchState(
            zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()
        ).validate()

    @staticmethod
    def _generic_instructions(
        *, counter_count: int, program_length: int
    ) -> tuple[ControlFlowInstruction, ...]:
        return control_flow_instruction_bank(
            counter_count=counter_count,
            program_length=program_length,
        )

    def neighbors(self, parent: ControlFlowProgram) -> tuple[tuple[str, ControlFlowProgram], ...]:
        parent.validate()
        candidates: list[tuple[str, ControlFlowProgram]] = []
        seen: set[str] = set()
        atoms = self._generic_instructions(
            counter_count=parent.counter_count,
            program_length=len(parent.instructions),
        )

        def add(operator: str, instructions: tuple[object, ...]) -> None:
            program = ControlFlowProgram(parent.counter_count, instructions)
            try:
                program.validate()
            except ValueError:
                return
            digest = program.digest()
            if digest == parent.digest() or digest in seen:
                return
            seen.add(digest)
            candidates.append((operator, program))

        for index in range(len(parent.instructions) - 1):
            for instruction in atoms:
                replacement = (*parent.instructions[:index], instruction, *parent.instructions[index + 1 :])
                add("replace_instruction", replacement)
        for first in range(len(parent.instructions) - 1):
            for second in range(first + 1, len(parent.instructions) - 1):
                instructions = list(parent.instructions)
                instructions[first], instructions[second] = instructions[second], instructions[first]
                add("swap_instructions", tuple(instructions))
        return tuple(candidates)

    def _operator_probabilities(self, state: ControlFlowSearchState) -> torch.Tensor:
        state.validate()
        total = torch.tensor(float(state.proposals + 1), dtype=state.reward_totals.dtype)
        means = torch.where(
            state.reward_counts > 0.0,
            state.reward_totals / state.reward_counts.clamp_min(1.0),
            torch.zeros_like(state.reward_totals),
        )
        bonus = self.exploration * torch.sqrt(
            torch.log(total) / (state.reward_counts + 1.0)
        )
        return torch.softmax((means + bonus) / self.temperature, dim=0)

    def propose(
        self,
        state: ControlFlowSearchState,
        parent: ControlFlowProgram,
        *,
        generator: torch.Generator,
        scope: str,
    ) -> ControlFlowProposal:
        state.validate()
        parent.validate()
        if not scope:
            raise ValueError("control-flow proposal scope cannot be empty")
        neighbors = self.neighbors(parent)
        if not neighbors:
            raise RuntimeError("control-flow program neighborhood is exhausted")
        unseen = tuple(
            (index, operator, program)
            for index, (operator, program) in enumerate(neighbors)
            if _scoped_digest(scope, program.digest()) not in state.seen_candidate_digests
        )
        if not unseen:
            raise RuntimeError("control-flow program neighborhood is exhausted")
        probabilities = self._operator_probabilities(state)
        selected_operator = int(torch.multinomial(probabilities, 1, generator=generator).item())
        matching = tuple(
            item for item in unseen if item[1] == CONTROL_FLOW_MUTATION_OPERATORS[selected_operator]
        )
        if not matching:
            matching = unseen
            selected_operator = CONTROL_FLOW_MUTATION_OPERATORS.index(matching[0][1])
        selected = matching[int(torch.randint(len(matching), (), generator=generator).item())]
        program = selected[2]
        proposal = ControlFlowProposal(
            program,
            parent.digest(),
            selected[1],
            selected_operator,
            state.proposals,
            float(probabilities[selected_operator].item()),
            scope,
        )
        return proposal.validate()

    def record_outcomes(
        self,
        state: ControlFlowSearchState,
        proposal: ControlFlowProposal,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ControlFlowFeedback:
        state.validate()
        proposal.validate()
        if proposal.attempt_id != state.proposals:
            raise ValueError("control-flow proposal is out of sequence")
        key = _scoped_digest(proposal.scope, proposal.program.digest())
        if key in state.seen_candidate_digests:
            raise ValueError("control-flow candidate was already evaluated")
        values = tuple(float(value) for value in outcomes)
        if not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("control-flow outcomes must be finite probabilities")
        if not 0.0 < threshold <= 1.0 or min_observations < 1 or min_stable_observations < 1:
            raise ValueError("control-flow outcome thresholds are invalid")
        stable = None
        for index in range(len(values)):
            if len(values) - index >= min_stable_observations and min(values[index:]) >= threshold:
                stable = index + 1
                break
        accepted = len(values) >= min_observations and stable is not None
        quality = sum(values) / len(values)
        totals = state.reward_totals.clone()
        counts = state.reward_counts.clone()
        accepted_counts = state.accepted_counts.clone()
        failed_counts = state.failed_counts.clone()
        index = proposal.operator_index
        totals[index] += quality
        counts[index] += 1.0
        if accepted:
            accepted_counts[index] += 1.0
        else:
            failed_counts[index] += 1.0
        next_state = ControlFlowSearchState(
            totals,
            counts,
            accepted_counts,
            failed_counts,
            (*state.seen_candidate_digests, key),
            state.proposals + 1,
            state.accepted + int(accepted),
            max(state.best_quality, quality),
        ).validate()
        return ControlFlowFeedback(proposal, quality, accepted, stable, next_state)


__all__ = [
    "CONTROL_FLOW_MUTATION_OPERATORS",
    "CONTROL_FLOW_PROPOSAL_SCHEMA",
    "CONTROL_FLOW_SEARCH_SCHEMA",
    "ControlFlowFeedback",
    "ControlFlowOutcomeSearch",
    "ControlFlowProposal",
    "ControlFlowSearchState",
    "control_flow_instruction_bank",
    "iter_control_flow_programs",
]
