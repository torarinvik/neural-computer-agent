"""Outcome-only structural frontier for reusable external control-flow files.

The frontier is a memory-side search component.  It retains provisional
counter-machine programs, not raw verifier rows, and never commits a program
to durable capability memory by itself.  A caller supplies scalar outcomes,
then performs the independent held-out admission transaction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from .control_flow import (
    ControlFlowInstruction,
    ControlFlowProgram,
    delete_control_flow_instruction,
    insert_control_flow_instruction,
)
from .control_flow_search import control_flow_instruction_bank

CONTROL_FLOW_FRONTIER_SCHEMA = "neural-computer.external-control-flow-frontier.v1"
CONTROL_FLOW_FRONTIER_HYPOTHESIS_SCHEMA = (
    "neural-computer.external-control-flow-hypothesis.v1"
)
CONTROL_FLOW_FRONTIER_PROPOSAL_SCHEMA = (
    "neural-computer.external-control-flow-frontier-proposal.v1"
)
CONTROL_FLOW_FRONTIER_PROPOSAL_FACTORS_SCHEMA = (
    "neural-computer.external-control-flow-frontier-proposal-factors.v1"
)
CONTROL_FLOW_FRONTIER_PROPOSAL_MEMORY_SCHEMA = (
    "neural-computer.external-control-flow-frontier-proposal-memory.v1"
)
CONTROL_FLOW_FRONTIER_GROWTH_SCHEMA = (
    "neural-computer.external-control-flow-frontier-growth.v1"
)
CONTROL_FLOW_FRONTIER_GROWTH_PROPOSAL_SCHEMA = (
    "neural-computer.external-control-flow-frontier-growth-proposal.v1"
)
CONTROL_FLOW_FRONTIER_GROWTH_RECEIPT_SCHEMA = (
    "neural-computer.external-control-flow-frontier-growth-receipt.v1"
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


def _instruction_digest(instruction: ControlFlowInstruction) -> str:
    return _digest_payload(instruction.payload())


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
class ControlFlowFrontierProposalFactors:
    """Generic factors describing one structural edit.

    Positions are represented relative to the parent's non-terminal boundary,
    so an edit immediately before ``HALT`` has the same factor at every
    program length.  Instruction identity is opaque and content-addressed;
    no counter-machine operation is interpreted by the policy.
    """

    operator_index: int
    primary_position: int
    secondary_position: int | None = None
    instruction_digests: tuple[str, ...] = ()
    schema: str = CONTROL_FLOW_FRONTIER_PROPOSAL_FACTORS_SCHEMA

    def validate(self) -> ControlFlowFrontierProposalFactors:
        if self.schema != CONTROL_FLOW_FRONTIER_PROPOSAL_FACTORS_SCHEMA:
            raise ValueError("unsupported control-flow frontier factors schema")
        if not 0 <= self.operator_index < len(CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS):
            raise ValueError("control-flow frontier factors operator is invalid")
        if self.operator_index == 3:
            if self.secondary_position is None or not (
                self.primary_position < self.secondary_position
            ):
                raise ValueError("control-flow swap factors need ordered positions")
            expected_instructions = 2
        else:
            if self.secondary_position is not None:
                raise ValueError(
                    "non-swap control-flow factors cannot have a second position"
                )
            expected_instructions = 1
        if len(self.instruction_digests) != expected_instructions:
            raise ValueError("control-flow frontier factor instruction count is invalid")
        for digest in self.instruction_digests:
            if not _digest_is_valid(digest):
                raise ValueError("control-flow frontier factor instruction digest is invalid")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "operator_index": self.operator_index,
            "primary_position": self.primary_position,
            "secondary_position": self.secondary_position,
            "instruction_digests": list(self.instruction_digests),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> ControlFlowFrontierProposalFactors:
        if not isinstance(payload, Mapping):
            raise TypeError("control-flow frontier factors payload must be a mapping")
        raw_instructions = payload.get("instruction_digests", ())
        if not isinstance(raw_instructions, Sequence) or isinstance(
            raw_instructions, (str, bytes)
        ):
            raise TypeError("control-flow frontier factor instructions are malformed")
        secondary = payload.get("secondary_position")
        return cls(
            operator_index=int(payload.get("operator_index", -1)),
            primary_position=int(payload.get("primary_position", -1)),
            secondary_position=None if secondary is None else int(secondary),
            instruction_digests=tuple(str(value) for value in raw_instructions),
            schema=str(payload.get("schema", "")),
        ).validate()


class ControlFlowFrontierProposalMemory:
    """External scalar credit over reusable control-flow edit factors.

    The memory never stores candidate program digests or verifier rows.  It
    stores only aggregate quality for opaque instruction, relative-position,
    and operator factors, with a shared prior plus context-local overrides.
    """

    schema = CONTROL_FLOW_FRONTIER_PROPOSAL_MEMORY_SCHEMA
    _FACTOR_TYPES = ("instruction", "position", "operator")

    def __init__(
        self,
        *,
        exploration_floor: float = 0.1,
        shared_prior_weight: float = 0.25,
        exploration_bonus: float = 0.25,
        instruction_weight: float = 1.0,
        position_weight: float = 0.75,
        operator_weight: float = 0.25,
        temperature: float = 0.25,
    ) -> None:
        for name, value in (
            ("exploration_floor", exploration_floor),
            ("shared_prior_weight", shared_prior_weight),
            ("exploration_bonus", exploration_bonus),
            ("instruction_weight", instruction_weight),
            ("position_weight", position_weight),
            ("operator_weight", operator_weight),
            ("temperature", temperature),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"control-flow proposal {name} is invalid")
        if exploration_floor >= 1.0:
            raise ValueError("control-flow proposal exploration floor must be < 1")
        if instruction_weight + position_weight + operator_weight <= 0.0:
            raise ValueError("control-flow proposal weights cannot all be zero")
        if temperature <= 0.0:
            raise ValueError("control-flow proposal temperature must be positive")
        self.exploration_floor = float(exploration_floor)
        self.shared_prior_weight = float(shared_prior_weight)
        self.exploration_bonus = float(exploration_bonus)
        self.instruction_weight = float(instruction_weight)
        self.position_weight = float(position_weight)
        self.operator_weight = float(operator_weight)
        self.temperature = float(temperature)
        self._context_stats: dict[str, dict[str, dict[str, list[float]]]] = {}
        self._shared_stats = self._empty_stats()

    @classmethod
    def _empty_stats(cls) -> dict[str, dict[str, list[float]]]:
        return {factor_type: {} for factor_type in cls._FACTOR_TYPES}

    @staticmethod
    def _validate_context(context: str) -> None:
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("control-flow proposal context must be a non-empty opaque key")

    @staticmethod
    def _aggregate(entry: Sequence[float] | None) -> tuple[float, float]:
        if entry is None:
            return 0.0, 0.0
        if len(entry) != 2:
            raise ValueError("control-flow proposal aggregate is malformed")
        total, count = float(entry[0]), float(entry[1])
        if not math.isfinite(total) or not math.isfinite(count):
            raise ValueError("control-flow proposal aggregate is non-finite")
        if total < 0.0 or count < 0.0 or total > count:
            raise ValueError("control-flow proposal aggregate is outside [0, 1]")
        return total, count

    @classmethod
    def _factor_entries(
        cls,
        factors: ControlFlowFrontierProposalFactors,
    ) -> tuple[tuple[str, str], ...]:
        factors.validate()
        positions = [factors.primary_position]
        if factors.secondary_position is not None:
            positions.append(factors.secondary_position)
        return (
            ("operator", str(factors.operator_index)),
            *[
                ("position", f"{factors.operator_index}:{position}")
                for position in positions
            ],
            *[("instruction", digest) for digest in factors.instruction_digests],
        )

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "exploration_floor": self.exploration_floor,
            "shared_prior_weight": self.shared_prior_weight,
            "exploration_bonus": self.exploration_bonus,
            "instruction_weight": self.instruction_weight,
            "position_weight": self.position_weight,
            "operator_weight": self.operator_weight,
            "temperature": self.temperature,
            "credit": "scalar_factor_aggregate_without_candidate_rows_v1",
            "context": "opaque_external_key_v1",
            "position": "relative_to_nonterminal_boundary_v1",
            "instruction": "content_addressed_generic_instruction_digest_v1",
        }

    def record(
        self,
        context: str,
        quality: float,
        *,
        factors: ControlFlowFrontierProposalFactors,
    ) -> None:
        self._validate_context(context)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("control-flow proposal quality must lie in [0, 1]")
        context_stats = self._context_stats.setdefault(context, self._empty_stats())
        for factor_type, key in self._factor_entries(factors):
            local = context_stats[factor_type].setdefault(key, [0.0, 0.0])
            local[0] += float(quality)
            local[1] += 1.0
            shared = self._shared_stats[factor_type].setdefault(key, [0.0, 0.0])
            shared[0] += float(quality)
            shared[1] += 1.0

    def _factor_score(
        self,
        context_stats: Mapping[str, Mapping[str, Sequence[float]]],
        factor_type: str,
        key: str,
        *,
        shared_weight: float,
    ) -> float:
        local_total, local_count = self._aggregate(
            context_stats.get(factor_type, {}).get(key)
        )
        shared_total, shared_count = self._aggregate(
            self._shared_stats[factor_type].get(key)
        )
        effective_count = local_count + shared_weight * shared_count
        effective_total = local_total + shared_weight * shared_total
        mean = effective_total / effective_count if effective_count else 0.0
        bonus = self.exploration_bonus * math.sqrt(
            math.log1p(local_count + shared_weight * shared_count + 1.0)
            / (effective_count + 1.0)
        )
        return mean + bonus

    def proposal_probabilities(
        self,
        context: str,
        factors: Sequence[ControlFlowFrontierProposalFactors],
    ) -> torch.Tensor:
        self._validate_context(context)
        factor_list = tuple(factors)
        if not factor_list:
            raise ValueError("control-flow proposal candidate set cannot be empty")
        for item in factor_list:
            item.validate()
        context_stats = self._context_stats.get(context, self._empty_stats())
        scores: list[float] = []
        for item in factor_list:
            entries = self._factor_entries(item)
            scores_by_type: dict[str, list[float]] = {
                factor_type: [] for factor_type in self._FACTOR_TYPES
            }
            for factor_type, key in entries:
                shared_weight = (
                    0.0
                    if any(context_stats[factor_type].values())
                    else self.shared_prior_weight
                )
                scores_by_type[factor_type].append(
                    self._factor_score(
                        context_stats,
                        factor_type,
                        key,
                        shared_weight=shared_weight,
                    )
                )
            instruction = sum(scores_by_type["instruction"]) / max(
                1, len(scores_by_type["instruction"])
            )
            position = sum(scores_by_type["position"]) / max(
                1, len(scores_by_type["position"])
            )
            operator = self._factor_score(
                context_stats,
                "operator",
                str(item.operator_index),
                shared_weight=(
                    0.0
                    if any(context_stats["operator"].values())
                    else self.shared_prior_weight
                ),
            )
            scores.append(
                self.instruction_weight * instruction
                + self.position_weight * position
                + self.operator_weight * operator
            )
        logits = torch.tensor(scores, dtype=torch.float64) / self.temperature
        probabilities = torch.softmax(logits, dim=0)
        return (1.0 - self.exploration_floor) * probabilities + (
            self.exploration_floor / len(factor_list)
        )

    def select(
        self,
        context: str,
        candidate_digests: Sequence[str],
        *,
        factors: Sequence[ControlFlowFrontierProposalFactors],
        generator: torch.Generator,
    ) -> tuple[int, float]:
        digests = tuple(candidate_digests)
        if len(digests) != len(tuple(factors)):
            raise ValueError("control-flow candidates and factors disagree")
        if any(not _digest_is_valid(digest) for digest in digests):
            raise ValueError("control-flow candidate digest is malformed")
        probabilities = self.proposal_probabilities(context, factors)
        index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        return index, float(probabilities[index].item())

    @classmethod
    def _serialize_stats(
        cls,
        stats: Mapping[str, Mapping[str, Sequence[float]]],
    ) -> dict[str, dict[str, list[float]]]:
        serialized: dict[str, dict[str, list[float]]] = {}
        for factor_type in cls._FACTOR_TYPES:
            values = stats.get(factor_type)
            if not isinstance(values, Mapping):
                raise TypeError("control-flow proposal factor table is malformed")
            serialized[factor_type] = {}
            for key, entry in sorted(values.items()):
                total, count = cls._aggregate(entry)
                serialized[factor_type][str(key)] = [total, count]
        return serialized

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "shared_stats": self._serialize_stats(self._shared_stats),
            "context_stats": {
                context: self._serialize_stats(stats)
                for context, stats in sorted(self._context_stats.items())
            },
        }

    def digest(self) -> str:
        return _digest_payload(self._content_payload())

    def payload(self) -> dict[str, object]:
        body = self._content_payload()
        return {**body, "sha256": _digest_payload(body)}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> ControlFlowFrontierProposalMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported control-flow proposal memory payload")
        configuration = payload.get("configuration")
        shared_stats = payload.get("shared_stats")
        context_stats = payload.get("context_stats")
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(shared_stats, Mapping)
            or not isinstance(context_stats, Mapping)
        ):
            raise TypeError("control-flow proposal memory payload is malformed")
        memory = cls(
            exploration_floor=float(configuration.get("exploration_floor", -1.0)),
            shared_prior_weight=float(configuration.get("shared_prior_weight", -1.0)),
            exploration_bonus=float(configuration.get("exploration_bonus", -1.0)),
            instruction_weight=float(configuration.get("instruction_weight", -1.0)),
            position_weight=float(configuration.get("position_weight", -1.0)),
            operator_weight=float(configuration.get("operator_weight", -1.0)),
            temperature=float(configuration.get("temperature", -1.0)),
        )

        def load_stats(raw: Mapping[str, object]) -> dict[str, dict[str, list[float]]]:
            loaded = memory._empty_stats()
            for factor_type in memory._FACTOR_TYPES:
                values = raw.get(factor_type)
                if not isinstance(values, Mapping):
                    raise TypeError("control-flow proposal factor table is malformed")
                for key, entry in values.items():
                    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
                        raise TypeError("control-flow proposal aggregate is malformed")
                    total, count = memory._aggregate(entry)
                    loaded[factor_type][str(key)] = [total, count]
            return loaded

        memory._shared_stats = load_stats(shared_stats)
        for context, stats in context_stats.items():
            memory._validate_context(str(context))
            if not isinstance(stats, Mapping):
                raise TypeError("control-flow proposal context table is malformed")
            memory._context_stats[str(context)] = load_stats(stats)
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != memory.digest():
            raise ValueError("control-flow proposal memory checksum mismatch")
        return memory


@dataclass(frozen=True)
class ControlFlowFrontierProposal:
    program: ControlFlowProgram
    parent_digest: str
    operator: str
    operator_index: int
    attempt_id: int
    selection_probability: float
    context: str = "default"
    factors: ControlFlowFrontierProposalFactors | None = None
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
        if not isinstance(self.context, str) or not self.context or "\0" in self.context:
            raise ValueError("control-flow frontier proposal context is malformed")
        if self.factors is not None:
            self.factors.validate()
            if self.factors.operator_index != self.operator_index:
                raise ValueError("control-flow frontier proposal factors/operator disagree")
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
        proposal_policy: ControlFlowFrontierProposalMemory | None = None,
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
        if proposal_policy is not None and not isinstance(
            proposal_policy, ControlFlowFrontierProposalMemory
        ):
            raise TypeError("control-flow frontier proposal policy has the wrong type")
        self.proposal_policy = proposal_policy

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
            "proposal_credit": (
                None
                if self.proposal_policy is None
                else self.proposal_policy.configuration()
            ),
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

    @staticmethod
    def _relative_position(position: int, program_length: int) -> int:
        """Encode a position relative to the non-terminal boundary."""

        return position - (program_length - 1)

    @classmethod
    def proposal_factors(
        cls,
        parent: ControlFlowProgram,
        operator_index: int,
        program: ControlFlowProgram,
    ) -> ControlFlowFrontierProposalFactors:
        """Infer generic factors for a valid one-edit control-flow neighbor."""

        parent.validate()
        program.validate()
        if parent.counter_count != program.counter_count:
            raise ValueError("control-flow proposal factor counter widths differ")
        parent_boundary = len(parent.instructions) - 1
        if operator_index == 0:
            if len(parent.instructions) != len(program.instructions):
                raise ValueError("replacement candidate has the wrong length")
            differences = [
                index
                for index, (before, after) in enumerate(
                    zip(parent.instructions, program.instructions, strict=True)
                )
                if before != after
            ]
            if len(differences) != 1 or differences[0] >= parent_boundary:
                raise ValueError("replacement candidate is not one non-terminal edit")
            position = differences[0]
            return ControlFlowFrontierProposalFactors(
                operator_index,
                cls._relative_position(position, len(parent.instructions)),
                instruction_digests=(_instruction_digest(program.instructions[position]),),
            ).validate()
        if operator_index == 1:
            if len(program.instructions) != len(parent.instructions) + 1:
                raise ValueError("insertion candidate has the wrong length")
            for position in range(parent_boundary + 1):
                try:
                    reduced = delete_control_flow_instruction(program, position)
                except ValueError:
                    continue
                if reduced.digest() == parent.digest():
                    return ControlFlowFrontierProposalFactors(
                        operator_index,
                        cls._relative_position(position, len(parent.instructions)),
                        instruction_digests=(
                            _instruction_digest(program.instructions[position]),
                        ),
                    ).validate()
            raise ValueError("insertion candidate does not contain its parent")
        if operator_index == 2:
            if len(program.instructions) != len(parent.instructions) - 1:
                raise ValueError("deletion candidate has the wrong length")
            for position in range(parent_boundary):
                try:
                    reduced = delete_control_flow_instruction(parent, position)
                except ValueError:
                    continue
                if reduced.digest() == program.digest():
                    return ControlFlowFrontierProposalFactors(
                        operator_index,
                        cls._relative_position(position, len(parent.instructions)),
                        instruction_digests=(_instruction_digest(parent.instructions[position]),),
                    ).validate()
            raise ValueError("deletion candidate is not a one-edit neighbor")
        if operator_index == 3:
            if len(parent.instructions) != len(program.instructions):
                raise ValueError("swap candidate has the wrong length")
            for first in range(parent_boundary):
                for second in range(first + 1, parent_boundary):
                    swapped = list(parent.instructions)
                    swapped[first], swapped[second] = swapped[second], swapped[first]
                    if tuple(swapped) == program.instructions:
                        return ControlFlowFrontierProposalFactors(
                            operator_index,
                            cls._relative_position(first, len(parent.instructions)),
                            cls._relative_position(second, len(parent.instructions)),
                            (
                                _instruction_digest(parent.instructions[first]),
                                _instruction_digest(parent.instructions[second]),
                            ),
                        ).validate()
            raise ValueError("swap candidate is not a two-position edit")
        raise ValueError("control-flow proposal operator is invalid")

    def exhaustive_candidates(
        self,
        parent: ControlFlowProgram,
    ) -> tuple[tuple[int, ControlFlowProgram], ...]:
        """Enumerate the finite one-edit neighborhood without target labels."""

        parent.validate()
        candidates: list[tuple[int, ControlFlowProgram]] = []
        seen: set[str] = set()

        def add(operator_index: int, candidate: ControlFlowProgram) -> None:
            digest = candidate.digest()
            if digest == parent.digest() or digest in seen:
                return
            seen.add(digest)
            candidates.append((operator_index, candidate))

        non_halt = len(parent.instructions) - 1
        atoms = control_flow_instruction_bank(
            counter_count=self.counter_count,
            program_length=len(parent.instructions),
        )
        for position in range(non_halt):
            for atom in atoms:
                instructions = list(parent.instructions)
                instructions[position] = atom
                try:
                    add(0, ControlFlowProgram(self.counter_count, tuple(instructions)).validate())
                except ValueError:
                    continue
        if len(parent.instructions) < self.max_program_length:
            atoms = control_flow_instruction_bank(
                counter_count=self.counter_count,
                program_length=len(parent.instructions) + 1,
            )
            for position in range(non_halt + 1):
                for atom in atoms:
                    try:
                        add(1, insert_control_flow_instruction(parent, position, atom))
                    except ValueError:
                        continue
        if len(parent.instructions) > self.min_program_length:
            for position in range(non_halt):
                try:
                    add(2, delete_control_flow_instruction(parent, position))
                except ValueError:
                    continue
        if non_halt > 1:
            for first in range(non_halt):
                for second in range(first + 1, non_halt):
                    instructions = list(parent.instructions)
                    instructions[first], instructions[second] = (
                        instructions[second],
                        instructions[first],
                    )
                    try:
                        add(3, ControlFlowProgram(self.counter_count, tuple(instructions)).validate())
                    except ValueError:
                        continue
        return tuple(candidates)

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
            return insert_control_flow_instruction(parent, position, insertion)
        elif operator_index == 2:
            position = self._random_index(non_halt, generator)
            return delete_control_flow_instruction(parent, position)
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
        context: str = "default",
    ) -> ControlFlowFrontierProposal:
        state.validate()
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("control-flow frontier proposal context is malformed")
        if self.proposal_policy is not None:
            parent = self._parent(state, generator=generator)
            available = tuple(
                (
                    operator_index,
                    candidate,
                    self.proposal_factors(parent.program, operator_index, candidate),
                )
                for operator_index, candidate in self.exhaustive_candidates(parent.program)
                if candidate.digest() not in state.seen_candidate_digests
            )
            if not available:
                raise RuntimeError("control-flow frontier proposal neighborhood is exhausted")
            selected, probability = self.proposal_policy.select(
                context,
                tuple(candidate.digest() for _, candidate, _ in available),
                factors=tuple(factors for _, _, factors in available),
                generator=generator,
            )
            operator_index, candidate, factors = available[selected]
            return ControlFlowFrontierProposal(
                program=candidate,
                parent_digest=parent.program.digest(),
                operator=CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS[operator_index],
                operator_index=operator_index,
                attempt_id=state.evaluations,
                selection_probability=probability,
                context=context,
                factors=factors,
            ).validate()
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
                    context=context,
                    factors=self.proposal_factors(
                        parent.program,
                        operator_index,
                        candidate,
                    ),
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
        if self.proposal_policy is not None:
            if proposal.factors is None:
                raise ValueError("control-flow frontier proposal policy requires factors")
            self.proposal_policy.record(
                proposal.context,
                quality,
                factors=proposal.factors,
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


@dataclass(frozen=True)
class ControlFlowFrontierGrowthState:
    """Replay-free adaptive-horizon state for external program induction.

    The frontier statistics and candidate digests survive curriculum growth;
    only the allowed program horizon and current search root change.  No raw
    verifier rows are retained.  Executable files remain owned by the separate
    ``ControlFlowProgramMemory`` boundary.
    """

    frontier: ControlFlowFrontierState
    horizon: int
    qualified_programs: tuple[tuple[str, float], ...]
    rung: int = 0
    schema: str = CONTROL_FLOW_FRONTIER_GROWTH_SCHEMA

    def validate(self) -> ControlFlowFrontierGrowthState:
        if self.schema != CONTROL_FLOW_FRONTIER_GROWTH_SCHEMA:
            raise ValueError("unsupported control-flow frontier-growth schema")
        self.frontier.validate()
        if self.horizon < 1:
            raise ValueError("control-flow frontier growth horizon is invalid")
        if self.rung < 0:
            raise ValueError("control-flow frontier growth rung is invalid")
        if len(self.qualified_programs) < 1:
            raise ValueError("control-flow frontier growth has no qualified programs")
        digests = tuple(item[0] for item in self.qualified_programs)
        if len(set(digests)) != len(digests):
            raise ValueError("control-flow qualified program digests are not unique")
        for digest, quality in self.qualified_programs:
            if not _digest_is_valid(digest):
                raise ValueError("control-flow qualified program digest is malformed")
            if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
                raise ValueError("control-flow qualified program quality is invalid")
        if self.frontier.root_digest not in digests:
            raise ValueError("control-flow frontier root is not qualified")
        for hypothesis in self.frontier.hypotheses:
            if len(hypothesis.program.instructions) > self.horizon:
                raise ValueError("frontier hypothesis exceeds the active horizon")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        body = {
            "schema": self.schema,
            "frontier": self.frontier.payload(),
            "horizon": self.horizon,
            "qualified_programs": [
                {"digest": digest, "quality": quality}
                for digest, quality in self.qualified_programs
            ],
            "rung": self.rung,
        }
        return {**body, "sha256": _digest_payload(body)}

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowFrontierGrowthState:
        if not isinstance(payload, dict):
            raise TypeError("control-flow frontier-growth payload must be a mapping")
        expected = payload.get("sha256")
        body = {key: value for key, value in payload.items() if key != "sha256"}
        if not isinstance(expected, str) or expected != _digest_payload(body):
            raise ValueError("control-flow frontier-growth checksum mismatch")
        raw_frontier = body.get("frontier")
        raw_qualified = body.get("qualified_programs")
        if not isinstance(raw_frontier, dict) or not isinstance(raw_qualified, list):
            raise TypeError("control-flow frontier-growth payload is incomplete")
        qualified: list[tuple[str, float]] = []
        for record in raw_qualified:
            if not isinstance(record, dict):
                raise TypeError("control-flow qualified program record is malformed")
            qualified.append((record.get("digest"), float(record.get("quality", float("nan")))))
        return cls(
            frontier=ControlFlowFrontierState.from_payload(raw_frontier),
            horizon=int(body.get("horizon", -1)),
            qualified_programs=tuple(qualified),
            rung=int(body.get("rung", -1)),
            schema=body.get("schema"),
        ).validate()

    def digest(self) -> str:
        return str(self.payload()["sha256"])


@dataclass(frozen=True)
class ControlFlowFrontierGrowthProposal:
    """A frontier proposal bound to the horizon that generated it."""

    proposal: ControlFlowFrontierProposal
    horizon: int
    rung: int
    schema: str = CONTROL_FLOW_FRONTIER_GROWTH_PROPOSAL_SCHEMA

    def validate(self) -> ControlFlowFrontierGrowthProposal:
        if self.schema != CONTROL_FLOW_FRONTIER_GROWTH_PROPOSAL_SCHEMA:
            raise ValueError("unsupported control-flow frontier-growth proposal schema")
        self.proposal.validate()
        if self.horizon < 1:
            raise ValueError("control-flow frontier-growth proposal horizon is invalid")
        if self.rung < 0:
            raise ValueError("control-flow frontier-growth proposal rung is invalid")
        return self


@dataclass(frozen=True)
class ControlFlowFrontierGrowthReceipt:
    """Copy-on-write receipt for a horizon or search-root transition."""

    accepted: bool
    operation: str
    source_horizon: int
    destination_horizon: int
    source_digest: str
    destination_digest: str
    source_rung: int
    destination_rung: int
    qualified_count: int
    reason: str
    schema: str = CONTROL_FLOW_FRONTIER_GROWTH_RECEIPT_SCHEMA

    def validate(self) -> ControlFlowFrontierGrowthReceipt:
        if self.schema != CONTROL_FLOW_FRONTIER_GROWTH_RECEIPT_SCHEMA:
            raise ValueError("unsupported control-flow frontier-growth receipt schema")
        if self.operation not in {"expand_horizon", "promote_root"}:
            raise ValueError("control-flow frontier-growth operation is invalid")
        if self.source_horizon < 1 or self.destination_horizon < 1:
            raise ValueError("control-flow frontier-growth receipt horizons are invalid")
        if self.source_rung < 0 or self.destination_rung < 0:
            raise ValueError("control-flow frontier-growth receipt rungs are invalid")
        if self.qualified_count < 1:
            raise ValueError("control-flow frontier-growth qualified count is invalid")
        for digest in (self.source_digest, self.destination_digest):
            if not _digest_is_valid(digest):
                raise ValueError("control-flow frontier-growth receipt digest is malformed")
        if not self.reason:
            raise ValueError("control-flow frontier-growth receipt reason is empty")
        if self.accepted:
            if self.destination_horizon < self.source_horizon:
                raise ValueError("accepted frontier growth cannot shrink its horizon")
            if self.destination_rung < self.source_rung:
                raise ValueError("accepted frontier growth cannot rewind its rung")
        elif (
            self.destination_horizon != self.source_horizon
            or self.destination_digest != self.source_digest
            or self.destination_rung != self.source_rung
        ):
            raise ValueError("rejected frontier growth must preserve its source state")
        return self


@dataclass(frozen=True)
class ControlFlowFrontierGrowthFeedback:
    """Outcome feedback with the adaptive-growth state boundary attached."""

    proposal: ControlFlowFrontierGrowthProposal
    quality: float
    accepted: bool
    stable_bits_to_threshold: int | None
    state: ControlFlowFrontierGrowthState


class ControlFlowProgramFrontierGrowth:
    """Adaptive curriculum over the generic external control-flow frontier.

    Horizon expansion and root promotion are independent, verifier-gated
    copy-on-write transactions.  Operator statistics and seen candidate
    digests are carried forward, so the next rung learns from prior proposal
    experience without replaying prior verifier rows.
    """

    schema = CONTROL_FLOW_FRONTIER_GROWTH_SCHEMA

    def __init__(
        self,
        counter_count: int,
        *,
        initial_horizon: int,
        maximum_horizon: int,
        beam_width: int = 16,
        max_depth: int = 8,
        min_program_length: int = 1,
        minimum_quality: float = 0.0,
        parent_temperature: float = 0.5,
        exploration: float = 0.5,
        proposal_retry_limit: int = 128,
        proposal_policy: ControlFlowFrontierProposalMemory | None = None,
    ) -> None:
        if initial_horizon < 1 or maximum_horizon < initial_horizon:
            raise ValueError("control-flow frontier growth horizons are invalid")
        self.counter_count = int(counter_count)
        self.initial_horizon = int(initial_horizon)
        self.maximum_horizon = int(maximum_horizon)
        self._frontier_kwargs = {
            "beam_width": beam_width,
            "max_depth": max_depth,
            "min_program_length": min_program_length,
            "minimum_quality": minimum_quality,
            "parent_temperature": parent_temperature,
            "exploration": exploration,
            "proposal_retry_limit": proposal_retry_limit,
            "proposal_policy": proposal_policy,
        }
        self._frontier_at(initial_horizon)

    def _frontier_at(self, horizon: int) -> ControlFlowProgramFrontier:
        if not 1 <= horizon <= self.maximum_horizon:
            raise ValueError("control-flow frontier growth horizon is outside bounds")
        return ControlFlowProgramFrontier(
            self.counter_count,
            max_program_length=horizon,
            **self._frontier_kwargs,
        )

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "counter_count": self.counter_count,
            "initial_horizon": self.initial_horizon,
            "maximum_horizon": self.maximum_horizon,
            "frontier": self._frontier_at(self.initial_horizon).configuration(),
            "growth": "one_horizon_step_then_retention_v1",
            "state": "opaque_qualified_digests_and_aggregate_credit_v1",
        }

    def initial_state(
        self,
        root: ControlFlowProgram,
        *,
        root_quality: float = 0.0,
    ) -> ControlFlowFrontierGrowthState:
        root.validate()
        if root.counter_count != self.counter_count:
            raise ValueError("control-flow frontier-growth root counter width is invalid")
        if len(root.instructions) > self.initial_horizon:
            raise ValueError("control-flow frontier-growth root exceeds initial horizon")
        frontier = self._frontier_at(self.initial_horizon)
        state = ControlFlowFrontierGrowthState(
            frontier=frontier.initial_state(root, root_quality=root_quality),
            horizon=self.initial_horizon,
            qualified_programs=((root.digest(), float(root_quality)),),
        )
        return state.validate()

    def propose(
        self,
        state: ControlFlowFrontierGrowthState,
        *,
        generator: torch.Generator,
        context: str = "default",
    ) -> ControlFlowFrontierGrowthProposal:
        state.validate()
        proposal = self._frontier_at(state.horizon).propose(
            state.frontier,
            generator=generator,
            context=context,
        )
        return ControlFlowFrontierGrowthProposal(
            proposal,
            state.horizon,
            state.rung,
        ).validate()

    def record_outcomes(
        self,
        state: ControlFlowFrontierGrowthState,
        proposal: ControlFlowFrontierGrowthProposal,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ControlFlowFrontierGrowthFeedback:
        state.validate()
        proposal.validate()
        if proposal.horizon != state.horizon:
            raise ValueError("control-flow frontier-growth proposal horizon is stale")
        if proposal.rung != state.rung:
            raise ValueError("control-flow frontier-growth proposal rung is stale")
        feedback = self._frontier_at(state.horizon).record_outcomes(
            state.frontier,
            proposal.proposal,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        qualified = state.qualified_programs
        if feedback.accepted:
            qualified = (*qualified, (feedback.proposal.program.digest(), feedback.quality))
        next_state = ControlFlowFrontierGrowthState(
            frontier=feedback.state,
            horizon=state.horizon,
            qualified_programs=qualified,
            rung=state.rung,
        ).validate()
        return ControlFlowFrontierGrowthFeedback(
            proposal=proposal,
            quality=feedback.quality,
            accepted=feedback.accepted,
            stable_bits_to_threshold=feedback.stable_bits_to_threshold,
            state=next_state,
        )

    @staticmethod
    def _receipt(
        *,
        accepted: bool,
        operation: str,
        source: ControlFlowFrontierGrowthState,
        destination: ControlFlowFrontierGrowthState,
        reason: str,
    ) -> ControlFlowFrontierGrowthReceipt:
        return ControlFlowFrontierGrowthReceipt(
            accepted=accepted,
            operation=operation,
            source_horizon=source.horizon,
            destination_horizon=destination.horizon,
            source_digest=source.digest(),
            destination_digest=destination.digest(),
            source_rung=source.rung,
            destination_rung=destination.rung,
            qualified_count=len(destination.qualified_programs),
            reason=reason,
        ).validate()

    def expand_horizon_verified(
        self,
        state: ControlFlowFrontierGrowthState,
        retention_probe: Callable[[ControlFlowFrontierGrowthState], bool],
    ) -> tuple[ControlFlowFrontierGrowthReceipt, ControlFlowFrontierGrowthState]:
        state.validate()
        if not callable(retention_probe):
            raise TypeError("control-flow frontier-growth retention probe must be callable")
        destination_horizon = state.horizon + 1
        if destination_horizon > self.maximum_horizon:
            return (
                self._receipt(
                    accepted=False,
                    operation="expand_horizon",
                    source=state,
                    destination=state,
                    reason="maximum frontier horizon reached",
                ),
                state,
            )
        candidate = ControlFlowFrontierGrowthState(
            frontier=state.frontier,
            horizon=destination_horizon,
            qualified_programs=state.qualified_programs,
            rung=state.rung,
        ).validate()
        if not bool(retention_probe(candidate)):
            return (
                self._receipt(
                    accepted=False,
                    operation="expand_horizon",
                    source=state,
                    destination=state,
                    reason="retention_probe_rejected",
                ),
                state,
            )
        return (
            self._receipt(
                accepted=True,
                operation="expand_horizon",
                source=state,
                destination=candidate,
                reason="horizon expanded after retention probe",
            ),
            candidate,
        )

    def promote_root_verified(
        self,
        state: ControlFlowFrontierGrowthState,
        candidate: ControlFlowProgram,
        retention_probe: Callable[[ControlFlowFrontierGrowthState], bool],
    ) -> tuple[ControlFlowFrontierGrowthReceipt, ControlFlowFrontierGrowthState]:
        state.validate()
        candidate.validate()
        if candidate.counter_count != self.counter_count:
            raise ValueError("control-flow frontier-growth candidate counter width is invalid")
        if not callable(retention_probe):
            raise TypeError("control-flow frontier-growth retention probe must be callable")
        candidate_digest = candidate.digest()
        qualified = dict(state.qualified_programs)
        if candidate_digest not in qualified:
            return (
                self._receipt(
                    accepted=False,
                    operation="promote_root",
                    source=state,
                    destination=state,
                    reason="candidate was not verifier-qualified",
                ),
                state,
            )
        current_root = next(
            hypothesis.program
            for hypothesis in state.frontier.hypotheses
            if hypothesis.program.digest() == state.frontier.root_digest
        )
        if len(candidate.instructions) <= len(current_root.instructions):
            return (
                self._receipt(
                    accepted=False,
                    operation="promote_root",
                    source=state,
                    destination=state,
                    reason="root promotion must increase program length",
                ),
                state,
            )
        if len(candidate.instructions) > state.horizon:
            return (
                self._receipt(
                    accepted=False,
                    operation="promote_root",
                    source=state,
                    destination=state,
                    reason="candidate exceeds active frontier horizon",
                ),
                state,
            )
        old = state.frontier
        root_hypothesis = ControlFlowFrontierHypothesis(
            program=candidate,
            parent_digest=None,
            depth=0,
            quality=qualified[candidate_digest],
        ).validate()
        rebased_frontier = ControlFlowFrontierState(
            hypotheses=(root_hypothesis,),
            reward_totals=old.reward_totals.clone(),
            reward_counts=old.reward_counts.clone(),
            accepted_counts=old.accepted_counts.clone(),
            failed_counts=old.failed_counts.clone(),
            seen_candidate_digests=old.seen_candidate_digests,
            root_digest=candidate_digest,
            evaluations=old.evaluations,
            accepted=old.accepted,
            best_quality=old.best_quality,
        ).validate()
        candidate_state = ControlFlowFrontierGrowthState(
            frontier=rebased_frontier,
            horizon=state.horizon,
            qualified_programs=state.qualified_programs,
            rung=state.rung + 1,
        ).validate()
        if not bool(retention_probe(candidate_state)):
            return (
                self._receipt(
                    accepted=False,
                    operation="promote_root",
                    source=state,
                    destination=state,
                    reason="retention_probe_rejected",
                ),
                state,
            )
        return (
            self._receipt(
                accepted=True,
                operation="promote_root",
                source=state,
                destination=candidate_state,
                reason="qualified longer program promoted as next search root",
            ),
            candidate_state,
        )


__all__ = [
    "CONTROL_FLOW_FRONTIER_GROWTH_PROPOSAL_SCHEMA",
    "CONTROL_FLOW_FRONTIER_GROWTH_RECEIPT_SCHEMA",
    "CONTROL_FLOW_FRONTIER_GROWTH_SCHEMA",
    "CONTROL_FLOW_FRONTIER_HYPOTHESIS_SCHEMA",
    "CONTROL_FLOW_FRONTIER_MUTATION_OPERATORS",
    "CONTROL_FLOW_FRONTIER_PROPOSAL_FACTORS_SCHEMA",
    "CONTROL_FLOW_FRONTIER_PROPOSAL_MEMORY_SCHEMA",
    "CONTROL_FLOW_FRONTIER_PROPOSAL_SCHEMA",
    "CONTROL_FLOW_FRONTIER_SCHEMA",
    "ControlFlowFrontierFeedback",
    "ControlFlowFrontierGrowthFeedback",
    "ControlFlowFrontierGrowthProposal",
    "ControlFlowFrontierGrowthReceipt",
    "ControlFlowFrontierGrowthState",
    "ControlFlowFrontierHypothesis",
    "ControlFlowFrontierProposal",
    "ControlFlowFrontierProposalFactors",
    "ControlFlowFrontierProposalMemory",
    "ControlFlowFrontierState",
    "ControlFlowProgramFrontier",
    "ControlFlowProgramFrontierGrowth",
]
