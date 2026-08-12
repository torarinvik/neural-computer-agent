"""Outcome-only structural frontier for reusable external control-flow files.

The frontier is a memory-side search component.  It retains provisional
counter-machine programs, not raw verifier rows, and never commits a program
to durable capability memory by itself.  A caller supplies scalar outcomes,
then performs the independent held-out admission transaction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .control_flow import ControlFlowProgram
from .control_flow_search import control_flow_instruction_bank

CONTROL_FLOW_FRONTIER_SCHEMA = "neural-computer.external-control-flow-frontier.v1"
CONTROL_FLOW_FRONTIER_HYPOTHESIS_SCHEMA = (
    "neural-computer.external-control-flow-hypothesis.v1"
)
CONTROL_FLOW_FRONTIER_PROPOSAL_SCHEMA = (
    "neural-computer.external-control-flow-frontier-proposal.v1"
)
CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS = (
    "replace_instruction",
    "insert_instruction",
    "delete_instruction",
    "swap_instructions",
)


def _digest_is_valid(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _digest_payload(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            detached = item.detach().cpu().contiguous()
            digest.update(str(detached.dtype).encode())
            digest.update(repr(tuple(detached.shape)).encode())
            digest.update(detached.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlFlowFrontierHypothesis:
    """One provisional executable file retained outside durable memory."""

    program: ControlFlowProgram
    parent_digest: str | None
    depth: int
    quality: float
    schema: str = CONTROL_FLOW_FRONTIER_HYPOTHESIS_SCHEMA

    def validate(self) -> ControlFlowFrontierHypothesis:
        if self.schema != CONTROL_FLOW_FRONTIER_HYPOTHESIS_SCHEMA:
            raise ValueError("unsupported control-flow frontier hypothesis schema")
        self.program.validate()
        if self.parent_digest is not None and not _digest_is_valid(self.parent_digest):
            raise ValueError("control-flow frontier parent digest is malformed")
        if self.parent_digest == self.program.digest():
            raise ValueError("control-flow frontier hypothesis cannot parent itself")
        if self.depth < 0:
            raise ValueError("control-flow frontier hypothesis depth is invalid")
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("control-flow frontier hypothesis quality is invalid")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "program": self.program.payload(),
            "parent_digest": self.parent_digest,
            "depth": self.depth,
            "quality": self.quality,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowFrontierHypothesis:
        if not isinstance(payload, dict):
            raise TypeError("control-flow frontier hypothesis must be a mapping")
        raw_program = payload.get("program")
        if not isinstance(raw_program, dict):
            raise TypeError("control-flow frontier hypothesis program is missing")
        return cls(
            program=ControlFlowProgram.from_payload(raw_program),
            parent_digest=payload.get("parent_digest"),
            depth=int(payload.get("depth", -1)),
            quality=float(payload.get("quality", float("nan"))),
            schema=payload.get("schema"),
        ).validate()


@dataclass(frozen=True)
class ControlFlowFrontierState:
    """Persistent frontier and aggregate scalar-credit statistics."""

    hypotheses: tuple[ControlFlowFrontierHypothesis, ...]
    reward_totals: torch.Tensor
    reward_counts: torch.Tensor
    accepted_counts: torch.Tensor
    failed_counts: torch.Tensor
    seen_candidate_digests: tuple[str, ...]
    root_digest: str
    evaluations: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = CONTROL_FLOW_FRONTIER_SCHEMA

    def validate(self) -> ControlFlowFrontierState:
        if self.schema != CONTROL_FLOW_FRONTIER_SCHEMA:
            raise ValueError("unsupported control-flow frontier schema")
        expected = (len(CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS),)
        for name, value in (
            ("reward_totals", self.reward_totals),
            ("reward_counts", self.reward_counts),
            ("accepted_counts", self.accepted_counts),
            ("failed_counts", self.failed_counts),
        ):
            if value.shape != expected or not value.is_floating_point():
                raise ValueError(f"control-flow frontier {name} has the wrong shape")
            if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0.0)):
                raise ValueError(f"control-flow frontier {name} is invalid")
        if not self.hypotheses:
            raise ValueError("control-flow frontier cannot be empty")
        counter_counts = {hypothesis.program.counter_count for hypothesis in self.hypotheses}
        if len(counter_counts) != 1:
            raise ValueError("control-flow frontier hypotheses use different counter widths")
        digests = tuple(hypothesis.program.digest() for hypothesis in self.hypotheses)
        if len(set(digests)) != len(digests):
            raise ValueError("control-flow frontier contains duplicate hypotheses")
        if not _digest_is_valid(self.root_digest) or self.root_digest not in digests:
            raise ValueError("control-flow frontier root is not retained")
        if self.evaluations < 0 or self.accepted < 0 or self.accepted > self.evaluations:
            raise ValueError("control-flow frontier counters are invalid")
        if int(self.reward_counts.sum().item()) != self.evaluations:
            raise ValueError("control-flow frontier proposal count is inconsistent")
        if int(self.accepted_counts.sum().item()) != self.accepted:
            raise ValueError("control-flow frontier acceptance count is inconsistent")
        if int(self.failed_counts.sum().item()) != self.evaluations - self.accepted:
            raise ValueError("control-flow frontier failure count is inconsistent")
        if len(self.seen_candidate_digests) != self.evaluations:
            raise ValueError("control-flow frontier digest count is inconsistent")
        if len(set(self.seen_candidate_digests)) != len(self.seen_candidate_digests):
            raise ValueError("control-flow frontier candidate digests are not unique")
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("control-flow frontier best quality is invalid")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "hypotheses": [hypothesis.payload() for hypothesis in self.hypotheses],
            "reward_totals": self.reward_totals.detach().cpu().clone(),
            "reward_counts": self.reward_counts.detach().cpu().clone(),
            "accepted_counts": self.accepted_counts.detach().cpu().clone(),
            "failed_counts": self.failed_counts.detach().cpu().clone(),
            "seen_candidate_digests": list(self.seen_candidate_digests),
            "root_digest": self.root_digest,
            "evaluations": self.evaluations,
            "accepted": self.accepted,
            "best_quality": self.best_quality,
        }

    def digest(self) -> str:
        """Return a stable checksum over hypotheses and aggregate state."""

        return _digest_payload(self.payload())

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowFrontierState:
        if not isinstance(payload, dict):
            raise TypeError("control-flow frontier state must be a mapping")
        raw_hypotheses = payload.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            raise TypeError("control-flow frontier hypotheses are missing")
        tensors = tuple(payload.get(name) for name in (
            "reward_totals",
            "reward_counts",
            "accepted_counts",
            "failed_counts",
        ))
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("control-flow frontier statistics are incomplete")
        return cls(
            hypotheses=tuple(
                ControlFlowFrontierHypothesis.from_payload(item)
                for item in raw_hypotheses
            ),
            reward_totals=tensors[0],
            reward_counts=tensors[1],
            accepted_counts=tensors[2],
            failed_counts=tensors[3],
            seen_candidate_digests=tuple(payload.get("seen_candidate_digests", ())),
            root_digest=payload.get("root_digest"),
            evaluations=int(payload.get("evaluations", -1)),
            accepted=int(payload.get("accepted", -1)),
            best_quality=float(payload.get("best_quality", float("nan"))),
            schema=payload.get("schema"),
        ).validate()


@dataclass(frozen=True)
class ControlFlowFrontierProposal:
    program: ControlFlowProgram
    parent_digest: str
    operator: str
    operator_index: int
    attempt_id: int
    selection_probability: float
    schema: str = CONTROL_FLOW_FRONTIER_PROPOSAL_SCHEMA

    def validate(self) -> ControlFlowFrontierProposal:
        self.program.validate()
        if self.schema != CONTROL_FLOW_FRONTIER_PROPOSAL_SCHEMA:
            raise ValueError("unsupported control-flow frontier proposal schema")
        if not _digest_is_valid(self.parent_digest):
            raise ValueError("control-flow frontier proposal parent is malformed")
        if self.program.digest() == self.parent_digest:
            raise ValueError("control-flow frontier proposal must change its parent")
        if self.operator not in CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS:
            raise ValueError("control-flow frontier proposal operator is unknown")
        if not 0 <= self.operator_index < len(CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS):
            raise ValueError("control-flow frontier proposal operator index is invalid")
        if CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS[self.operator_index] != self.operator:
            raise ValueError("control-flow frontier proposal operator/index disagree")
        if self.attempt_id < 0 or not 0.0 < self.selection_probability <= 1.0:
            raise ValueError("control-flow frontier proposal metadata is invalid")
        return self


@dataclass(frozen=True)
class ControlFlowFrontierFeedback:
    proposal: ControlFlowFrontierProposal
    quality: float
    accepted: bool
    stable_bits_to_threshold: int | None
    state: ControlFlowFrontierState


class ControlFlowProgramFrontier:
    """Stochastic multi-edit search over an opaque control-flow file space."""

    schema = CONTROL_FLOW_FRONTIER_SCHEMA

    def __init__(
        self,
        counter_count: int,
        *,
        beam_width: int = 16,
        max_depth: int = 8,
        min_program_length: int = 1,
        max_program_length: int = 8,
        minimum_quality: float = 0.0,
        parent_temperature: float = 0.5,
        exploration: float = 0.5,
        proposal_retry_limit: int = 128,
    ) -> None:
        if counter_count < 2:
            raise ValueError("control-flow frontier needs at least two counters")
        if beam_width < 2 or max_depth < 1:
            raise ValueError("control-flow frontier bounds are invalid")
        if min_program_length < 1 or max_program_length < min_program_length:
            raise ValueError("control-flow frontier program bounds are invalid")
        if not 0.0 <= minimum_quality <= 1.0:
            raise ValueError("control-flow frontier minimum quality is invalid")
        if not math.isfinite(parent_temperature) or parent_temperature <= 0.0:
            raise ValueError("control-flow frontier parent temperature is invalid")
        if not math.isfinite(exploration) or exploration < 0.0:
            raise ValueError("control-flow frontier exploration is invalid")
        if proposal_retry_limit < 1:
            raise ValueError("control-flow frontier proposal retry limit is invalid")
        self.counter_count = int(counter_count)
        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.min_program_length = int(min_program_length)
        self.max_program_length = int(max_program_length)
        self.minimum_quality = float(minimum_quality)
        self.parent_temperature = float(parent_temperature)
        self.exploration = float(exploration)
        self.proposal_retry_limit = int(proposal_retry_limit)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "counter_count": self.counter_count,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "min_program_length": self.min_program_length,
            "max_program_length": self.max_program_length,
            "minimum_quality": self.minimum_quality,
            "parent_temperature": self.parent_temperature,
            "exploration": self.exploration,
            "proposal_retry_limit": self.proposal_retry_limit,
            "operators": CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS,
            "updates": "scalar_verifier_aggregate_only_v1",
            "commit": "caller_owned_heldout_copy_on_write_v1",
        }

    def initial_state(
        self,
        root: ControlFlowProgram,
        *,
        root_quality: float = 0.0,
    ) -> ControlFlowFrontierState:
        root.validate()
        if root.counter_count != self.counter_count:
            raise ValueError("control-flow frontier root counter width is incompatible")
        if not self.min_program_length <= len(root.instructions) <= self.max_program_length:
            raise ValueError("control-flow frontier root length is outside bounds")
        if not 0.0 <= root_quality <= 1.0:
            raise ValueError("control-flow frontier root quality is invalid")
        hypothesis = ControlFlowFrontierHypothesis(
            program=root,
            parent_digest=None,
            depth=0,
            quality=float(root_quality),
        ).validate()
        zeros = torch.zeros(
            len(CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS), dtype=torch.float64
        )
        return ControlFlowFrontierState(
            hypotheses=(hypothesis,),
            reward_totals=zeros.clone(),
            reward_counts=zeros.clone(),
            accepted_counts=zeros.clone(),
            failed_counts=zeros.clone(),
            seen_candidate_digests=(),
            root_digest=root.digest(),
            best_quality=float(root_quality),
        ).validate()

    def _eligible(self, state: ControlFlowFrontierState) -> tuple[int, ...]:
        return tuple(
            index
            for index, hypothesis in enumerate(state.hypotheses)
            if hypothesis.depth < self.max_depth
            and (
                hypothesis.program.instructions
                and (
                    len(hypothesis.program.instructions) < self.max_program_length
                    or len(hypothesis.program.instructions) > self.min_program_length
                )
            )
        )

    def _parent(
        self,
        state: ControlFlowFrontierState,
        *,
        generator: torch.Generator,
    ) -> ControlFlowFrontierHypothesis:
        eligible = self._eligible(state)
        if not eligible:
            raise RuntimeError("control-flow frontier has no expandable hypothesis")
        qualities = torch.tensor(
            [state.hypotheses[index].quality for index in eligible], dtype=torch.float64
        )
        probabilities = torch.softmax(qualities / self.parent_temperature, dim=0)
        index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        return state.hypotheses[eligible[index]]

    @staticmethod
    def _random_index(limit: int, generator: torch.Generator) -> int:
        return int(torch.randint(limit, (), generator=generator).item())

    def _operator_probabilities(
        self,
        state: ControlFlowFrontierState,
        parent: ControlFlowFrontierHypothesis,
    ) -> torch.Tensor:
        available = torch.tensor(
            [
                True,
                len(parent.program.instructions) < self.max_program_length,
                len(parent.program.instructions) - 1 > self.min_program_length,
                len(parent.program.instructions) - 1 > 1,
            ],
            dtype=torch.bool,
        )
        counts = state.reward_counts
        means = torch.where(
            counts > 0.0,
            state.reward_totals / counts.clamp_min(1.0),
            torch.zeros_like(counts),
        )
        total = torch.tensor(float(state.evaluations + 1), dtype=means.dtype)
        bonus = self.exploration * torch.sqrt(
            torch.log(total) / (counts + 1.0)
        )
        return torch.softmax(
            (means + bonus).masked_fill(~available, float("-inf")), dim=0
        )

    def _mutate(
        self,
        parent: ControlFlowProgram,
        operator_index: int,
        generator: torch.Generator,
    ) -> ControlFlowProgram:
        instructions = parent.instructions
        non_halt = len(instructions) - 1
        if operator_index == 0:
            position = self._random_index(non_halt, generator)
            atoms = control_flow_instruction_bank(
                counter_count=self.counter_count,
                program_length=len(instructions),
            )
            replacement = atoms[self._random_index(len(atoms), generator)]
            candidate = (*instructions[:position], replacement, *instructions[position + 1 :])
        elif operator_index == 1:
            position = self._random_index(non_halt + 1, generator)
            atoms = control_flow_instruction_bank(
                counter_count=self.counter_count,
                program_length=len(instructions) + 1,
            )
            insertion = atoms[self._random_index(len(atoms), generator)]
            candidate = (*instructions[:position], insertion, *instructions[position:])
        elif operator_index == 2:
            position = self._random_index(non_halt, generator)
            candidate = (*instructions[:position], *instructions[position + 1 :])
        elif operator_index == 3:
            first = self._random_index(non_halt, generator)
            second = self._random_index(non_halt - 1, generator)
            if second >= first:
                second += 1
            values = list(instructions)
            values[first], values[second] = values[second], values[first]
            candidate = tuple(values)
        else:
            raise ValueError("control-flow frontier operator index is invalid")
        return ControlFlowProgram(self.counter_count, tuple(candidate)).validate()

    def propose(
        self,
        state: ControlFlowFrontierState,
        *,
        generator: torch.Generator,
    ) -> ControlFlowFrontierProposal:
        state.validate()
        for _ in range(self.proposal_retry_limit):
            parent = self._parent(state, generator=generator)
            excluded = torch.zeros(
                len(CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS), dtype=torch.bool
            )
            for _ in range(self.proposal_retry_limit):
                probabilities = self._operator_probabilities(state, parent).masked_fill(
                    excluded, 0.0
                )
                normalizer = probabilities.sum()
                if not bool(torch.isfinite(normalizer)) or normalizer <= 0.0:
                    break
                probabilities = probabilities / normalizer
                operator_index = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
                try:
                    candidate = self._mutate(
                        parent.program,
                        operator_index,
                        generator,
                    )
                except ValueError:
                    excluded[operator_index] = True
                    continue
                candidate_digest = candidate.digest()
                if candidate_digest == parent.program.digest():
                    continue
                if candidate_digest in state.seen_candidate_digests:
                    excluded[operator_index] = True
                    continue
                return ControlFlowFrontierProposal(
                    program=candidate,
                    parent_digest=parent.program.digest(),
                    operator=CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS[operator_index],
                    operator_index=operator_index,
                    attempt_id=state.evaluations,
                    selection_probability=float(probabilities[operator_index].item()),
                ).validate()
        raise RuntimeError("control-flow frontier proposal neighborhood is exhausted")

    def _prune(
        self,
        hypotheses: Sequence[ControlFlowFrontierHypothesis],
        *,
        root_digest: str,
    ) -> tuple[ControlFlowFrontierHypothesis, ...]:
        root = next(item for item in hypotheses if item.program.digest() == root_digest)
        others = [item for item in hypotheses if item.program.digest() != root_digest]
        others.sort(key=lambda item: (-item.quality, item.depth, item.program.digest()))
        return (root, *others[: self.beam_width - 1])

    def record_outcomes(
        self,
        state: ControlFlowFrontierState,
        proposal: ControlFlowFrontierProposal,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ControlFlowFrontierFeedback:
        state.validate()
        proposal.validate()
        if proposal.attempt_id != state.evaluations:
            raise ValueError("control-flow frontier proposal is out of sequence")
        parent = next(
            (
                hypothesis
                for hypothesis in state.hypotheses
                if hypothesis.program.digest() == proposal.parent_digest
            ),
            None,
        )
        if parent is None:
            raise ValueError("control-flow frontier proposal parent is not retained")
        if proposal.program.counter_count != parent.program.counter_count:
            raise ValueError("control-flow frontier proposal counter width changed")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("control-flow frontier threshold is invalid")
        if min_observations < 1 or min_stable_observations < 1:
            raise ValueError("control-flow frontier observation bounds are invalid")
        values = tuple(float(value) for value in outcomes)
        if not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("control-flow frontier outcomes must be finite [0, 1]")
        stable: int | None = None
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
        totals[proposal.operator_index] += quality
        counts[proposal.operator_index] += 1.0
        if accepted:
            accepted_counts[proposal.operator_index] += 1.0
        else:
            failed_counts[proposal.operator_index] += 1.0
        hypotheses = list(state.hypotheses)
        if quality >= self.minimum_quality:
            hypotheses.append(
                ControlFlowFrontierHypothesis(
                    program=proposal.program,
                    parent_digest=proposal.parent_digest,
                    depth=parent.depth + 1,
                    quality=quality,
                ).validate()
            )
        next_state = ControlFlowFrontierState(
            hypotheses=self._prune(hypotheses, root_digest=state.root_digest),
            reward_totals=totals,
            reward_counts=counts,
            accepted_counts=accepted_counts,
            failed_counts=failed_counts,
            seen_candidate_digests=(
                *state.seen_candidate_digests,
                proposal.program.digest(),
            ),
            root_digest=state.root_digest,
            evaluations=state.evaluations + 1,
            accepted=state.accepted + int(accepted),
            best_quality=max(state.best_quality, quality),
        ).validate()
        return ControlFlowFrontierFeedback(
            proposal=proposal,
            quality=quality,
            accepted=accepted,
            stable_bits_to_threshold=stable,
            state=next_state,
        )


__all__ = [
    "CONTROL_FLOW_FRONTIER_HYPOTHESIS_SCHEMA",
    "CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS",
    "CONTROL_FLOW_FRONTIER_PROPOSAL_SCHEMA",
    "CONTROL_FLOW_FRONTIER_SCHEMA",
    "ControlFlowFrontierFeedback",
    "ControlFlowFrontierHypothesis",
    "ControlFlowFrontierProposal",
    "ControlFlowFrontierState",
    "ControlFlowProgramFrontier",
]
