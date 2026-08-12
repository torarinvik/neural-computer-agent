"""Verifier-gated composition of independently stored recipe files.

This module is the external CPU/files composition seam for the generic recipe
interpreter.  It composes already admitted files, records immutable provenance,
and exposes only aggregate scalar outcomes to the optional composition policy.
The controller and the atomic recipe interpreter remain outside this module
and are never updated by composition growth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .recipe_program import (
    ExternalRecipeProgramMemory,
    RecipeProgram,
    RecipeProgramAdmissionReceipt,
    evaluate_recipe_program_admission,
)

RECIPE_COMPOSITION_FACTORS_SCHEMA = "neural-computer.recipe-composition-factors.v1"
RECIPE_COMPOSITION_MEMORY_SCHEMA = "neural-computer.recipe-composition-memory.v1"
RECIPE_COMPOSITION_POLICY_SCHEMA = "neural-computer.recipe-composition-policy.v1"
RECIPE_COMPOSITION_CANDIDATE_SCHEMA = "neural-computer.recipe-composition-candidate.v1"
RECIPE_COMPOSITION_PROPOSAL_SCHEMA = "neural-computer.recipe-composition-proposal.v1"
RECIPE_COMPOSITION_SEARCH_SCHEMA = "neural-computer.recipe-composition-search.v1"
RECIPE_COMPOSITION_MODES = ("append", "prepend")


def _payload_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_digest(value: str, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} digest is malformed")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} digest is malformed") from error


def _scoped_key(scope: str, digest: str) -> str:
    if not isinstance(scope, str) or not scope or "\0" in scope:
        raise ValueError("composition scope must be a non-empty opaque key")
    _validate_digest(digest, label="composition candidate")
    return f"{scope}\0{digest}"


@dataclass(frozen=True)
class RecipeProgramCompositionFactors:
    """Opaque provenance factors for one ordered file composition."""

    left_digest: str
    right_digest: str
    mode: str = "append"
    schema: str = RECIPE_COMPOSITION_FACTORS_SCHEMA

    def validate(self) -> RecipeProgramCompositionFactors:
        if self.schema != RECIPE_COMPOSITION_FACTORS_SCHEMA:
            raise ValueError("unsupported recipe composition factors schema")
        _validate_digest(self.left_digest, label="composition left source")
        _validate_digest(self.right_digest, label="composition right source")
        if self.left_digest == self.right_digest:
            raise ValueError("composition sources must be distinct files")
        if self.mode not in RECIPE_COMPOSITION_MODES:
            raise ValueError("composition mode is unknown")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "left_digest": self.left_digest,
            "right_digest": self.right_digest,
            "mode": self.mode,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> RecipeProgramCompositionFactors:
        if not isinstance(payload, Mapping):
            raise TypeError("composition factors payload must be a mapping")
        return cls(
            left_digest=str(payload.get("left_digest", "")),
            right_digest=str(payload.get("right_digest", "")),
            mode=str(payload.get("mode", "")),
            schema=str(payload.get("schema", "")),
        ).validate()


@dataclass(frozen=True)
class RecipeProgramCompositionCandidate:
    """One composition candidate plus its source-file provenance."""

    left_slot: int
    right_slot: int
    factors: RecipeProgramCompositionFactors
    program: RecipeProgram
    schema: str = RECIPE_COMPOSITION_CANDIDATE_SCHEMA

    def validate(self) -> RecipeProgramCompositionCandidate:
        if self.schema != RECIPE_COMPOSITION_CANDIDATE_SCHEMA:
            raise ValueError("unsupported recipe composition candidate schema")
        if self.left_slot < 0 or self.right_slot < 0:
            raise ValueError("composition source slots cannot be negative")
        if self.left_slot == self.right_slot:
            raise ValueError("composition source slots must be distinct")
        self.factors.validate()
        if not isinstance(self.program, RecipeProgram):
            raise TypeError("composition candidate program has the wrong type")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "left_slot": self.left_slot,
            "right_slot": self.right_slot,
            "factors": self.factors.payload(),
            "program": self.program.payload(),
        }


class ExternalRecipeCompositionMemory:
    """Persistent external files plus verifier-gated composition provenance."""

    schema = RECIPE_COMPOSITION_MEMORY_SCHEMA

    def __init__(
        self,
        slot_values: Sequence[int],
        *,
        allow_parallel: bool = False,
    ) -> None:
        self.files = ExternalRecipeProgramMemory(
            slot_values,
            allow_parallel=allow_parallel,
        )
        self._provenance: list[RecipeProgramCompositionFactors | None] = []

    @property
    def basis(self):
        return self.files.basis

    @property
    def file_count(self) -> int:
        return self.files.file_count

    def program(self, slot: int) -> RecipeProgram:
        return self.files.program(slot)

    def execute(self, slot: int, state: tuple[int, ...]) -> tuple[int, ...]:
        return self.files.execute(slot, state)

    def add_program(self, program: RecipeProgram) -> int:
        slot = self.files.add_program(program)
        self._provenance.append(None)
        return slot

    def admit_verified_program(
        self,
        program: RecipeProgram,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        protect: bool = False,
    ) -> RecipeProgramAdmissionReceipt:
        """Admit an atomic external file while preserving null provenance."""

        receipt = self.files.admit_verified_program(
            program,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
            protect=protect,
        )
        if receipt.accepted:
            if receipt.slot is None or receipt.slot != len(self._provenance):
                raise RuntimeError("atomic provenance slot did not append atomically")
            self._provenance.append(None)
        return receipt

    def protect_file(self, slot: int) -> None:
        self.files.protect_file(slot)

    def is_file_protected(self, slot: int) -> bool:
        return self.files.is_file_protected(slot)

    def provenance(self, slot: int) -> RecipeProgramCompositionFactors | None:
        if not 0 <= slot < self.file_count:
            raise ValueError("composition memory slot is out of range")
        return self._provenance[slot]

    def _validate_source_candidate(
        self,
        candidate: RecipeProgramCompositionCandidate,
    ) -> None:
        candidate.validate()
        if candidate.left_slot >= self.file_count or candidate.right_slot >= self.file_count:
            raise ValueError("composition source slot is outside memory")
        left = self.program(candidate.left_slot)
        right = self.program(candidate.right_slot)
        if left.digest() != candidate.factors.left_digest:
            raise ValueError("composition left provenance does not match its slot")
        if right.digest() != candidate.factors.right_digest:
            raise ValueError("composition right provenance does not match its slot")
        instructions = (
            left.instructions + right.instructions
            if candidate.factors.mode == "append"
            else right.instructions + left.instructions
        )
        expected = RecipeProgram(
            self.basis.slot_values,
            instructions,
            allow_parallel=self.basis.allow_parallel,
        )
        if expected.digest() != candidate.program.digest():
            raise ValueError("composition candidate is not the stated file composition")

    def composition_candidates(
        self,
        *,
        max_program_length: int,
    ) -> tuple[RecipeProgramCompositionCandidate, ...]:
        if max_program_length < 1:
            raise ValueError("composition maximum length must be positive")
        candidates: list[RecipeProgramCompositionCandidate] = []
        seen: set[str] = set()
        existing = {self.program(slot).digest() for slot in range(self.file_count)}
        # Prefer append provenance when the same ordered program has both
        # equivalent decompositions.  This keeps the policy's learned right
        # factor and mode aligned with the natural growth direction instead of
        # making its identity depend on source-slot enumeration order.
        for mode in RECIPE_COMPOSITION_MODES:
            for left_slot in range(self.file_count):
                for right_slot in range(self.file_count):
                    if left_slot == right_slot:
                        continue
                    left = self.program(left_slot)
                    right = self.program(right_slot)
                    instructions = (
                        left.instructions + right.instructions
                        if mode == "append"
                        else right.instructions + left.instructions
                    )
                    if len(instructions) > max_program_length:
                        continue
                    program = RecipeProgram(
                        self.basis.slot_values,
                        instructions,
                        allow_parallel=self.basis.allow_parallel,
                    )
                    digest = program.digest()
                    if digest in seen or digest in existing:
                        continue
                    candidate = RecipeProgramCompositionCandidate(
                        left_slot,
                        right_slot,
                        RecipeProgramCompositionFactors(
                            left.digest(), right.digest(), mode
                        ),
                        program,
                    ).validate()
                    seen.add(digest)
                    candidates.append(candidate)
        return tuple(candidates)

    def admit_verified_composition(
        self,
        candidate: RecipeProgramCompositionCandidate,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        protect: bool = False,
    ) -> RecipeProgramAdmissionReceipt:
        self._validate_source_candidate(candidate)
        receipt = self.files.admit_verified_program(
            candidate.program,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
            protect=protect,
        )
        if receipt.accepted:
            if receipt.slot is None or receipt.slot != len(self._provenance):
                raise RuntimeError("composition provenance slot did not append atomically")
            self._provenance.append(candidate.factors)
        return receipt

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "files": self.files.payload(),
            "provenance": [
                None if factors is None else factors.payload()
                for factors in self._provenance
            ],
        }

    def digest(self) -> str:
        return _payload_digest(self._content_payload())

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> ExternalRecipeCompositionMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported recipe composition memory payload")
        files = payload.get("files")
        provenance = payload.get("provenance")
        if not isinstance(files, Mapping) or not isinstance(provenance, Sequence):
            raise TypeError("recipe composition memory payload is malformed")
        base = ExternalRecipeProgramMemory.from_payload(files)
        memory = cls(base.basis.slot_values, allow_parallel=base.basis.allow_parallel)
        memory.files = base
        if len(provenance) != memory.file_count:
            raise ValueError("recipe composition provenance count is inconsistent")
        digests = {memory.program(slot).digest() for slot in range(memory.file_count)}
        memory._provenance = []
        for raw in provenance:
            if raw is None:
                memory._provenance.append(None)
                continue
            if not isinstance(raw, Mapping):
                raise TypeError("recipe composition provenance row is malformed")
            factors = RecipeProgramCompositionFactors.from_payload(raw)
            if factors.left_digest not in digests or factors.right_digest not in digests:
                raise ValueError("recipe composition provenance references a missing file")
            memory._provenance.append(factors)
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != memory.digest():
            raise ValueError("recipe composition memory checksum mismatch")
        return memory


class OpaqueContextRecipeCompositionMemory:
    """Aggregate scalar composition credit over opaque source-file factors."""

    schema = RECIPE_COMPOSITION_POLICY_SCHEMA
    _FACTOR_TYPES = ("left", "right", "mode")

    def __init__(
        self,
        *,
        exploration_floor: float = 0.05,
        shared_prior_weight: float = 0.25,
        exploration_bonus: float = 0.25,
        left_weight: float = 1.0,
        right_weight: float = 1.0,
        mode_weight: float = 0.5,
        temperature: float = 0.05,
    ) -> None:
        for name, value in (
            ("exploration_floor", exploration_floor),
            ("shared_prior_weight", shared_prior_weight),
            ("exploration_bonus", exploration_bonus),
            ("left_weight", left_weight),
            ("right_weight", right_weight),
            ("mode_weight", mode_weight),
            ("temperature", temperature),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"composition policy {name} is invalid")
        if exploration_floor >= 1.0:
            raise ValueError("composition exploration floor must be < 1")
        if left_weight + right_weight + mode_weight <= 0.0:
            raise ValueError("composition policy weights cannot all be zero")
        if temperature <= 0.0:
            raise ValueError("composition policy temperature must be positive")
        self.exploration_floor = float(exploration_floor)
        self.shared_prior_weight = float(shared_prior_weight)
        self.exploration_bonus = float(exploration_bonus)
        self.left_weight = float(left_weight)
        self.right_weight = float(right_weight)
        self.mode_weight = float(mode_weight)
        self.temperature = float(temperature)
        self._shared: dict[str, dict[str, list[float]]] = self._empty_stats()
        self._contexts: dict[str, dict[str, dict[str, list[float]]]] = {}

    @classmethod
    def _empty_stats(cls) -> dict[str, dict[str, list[float]]]:
        return {factor_type: {} for factor_type in cls._FACTOR_TYPES}

    @staticmethod
    def _validate_context(context: str) -> None:
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("composition context must be a non-empty opaque key")

    @staticmethod
    def _aggregate(entry: Sequence[float] | None) -> tuple[float, float]:
        if entry is None:
            return 0.0, 0.0
        if len(entry) != 2:
            raise ValueError("composition aggregate is malformed")
        total, count = float(entry[0]), float(entry[1])
        if not math.isfinite(total) or not math.isfinite(count):
            raise ValueError("composition aggregate is non-finite")
        if total < 0.0 or count < 0.0 or total > count:
            raise ValueError("composition aggregate is outside [0, 1]")
        return total, count

    @staticmethod
    def _entries(
        factors: RecipeProgramCompositionFactors,
    ) -> tuple[tuple[str, str], ...]:
        factors.validate()
        return (
            ("left", factors.left_digest),
            ("right", factors.right_digest),
            ("mode", factors.mode),
        )

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "exploration_floor": self.exploration_floor,
            "shared_prior_weight": self.shared_prior_weight,
            "exploration_bonus": self.exploration_bonus,
            "left_weight": self.left_weight,
            "right_weight": self.right_weight,
            "mode_weight": self.mode_weight,
            "temperature": self.temperature,
            "credit": "scalar_composition_factor_aggregate_v1",
            "context": "opaque_external_key_v1",
        }

    def record(
        self,
        context: str,
        factors: RecipeProgramCompositionFactors,
        quality: float,
    ) -> None:
        self._validate_context(context)
        factors.validate()
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("composition quality must lie in [0, 1]")
        local = self._contexts.setdefault(context, self._empty_stats())
        for factor_type, key in self._entries(factors):
            local_entry = local[factor_type].setdefault(key, [0.0, 0.0])
            local_entry[0] += float(quality)
            local_entry[1] += 1.0
            shared_entry = self._shared[factor_type].setdefault(key, [0.0, 0.0])
            shared_entry[0] += float(quality)
            shared_entry[1] += 1.0

    def _score(
        self,
        local: Mapping[str, Mapping[str, Sequence[float]]],
        factor_type: str,
        key: str,
    ) -> float:
        local_total, local_count = self._aggregate(local[factor_type].get(key))
        shared_total, shared_count = self._aggregate(self._shared[factor_type].get(key))
        shared_weight = 0.0 if local[factor_type] else self.shared_prior_weight
        effective_count = local_count + shared_weight * shared_count
        effective_total = local_total + shared_weight * shared_total
        mean = effective_total / effective_count if effective_count else 0.0
        bonus = self.exploration_bonus * math.sqrt(
            math.log1p(effective_count + 1.0) / (effective_count + 1.0)
        )
        return mean + bonus

    def proposal_probabilities(
        self,
        context: str,
        factors: Sequence[RecipeProgramCompositionFactors],
    ) -> torch.Tensor:
        self._validate_context(context)
        candidates = tuple(factors)
        if not candidates:
            raise ValueError("composition candidate set cannot be empty")
        for candidate in candidates:
            candidate.validate()
        local = self._contexts.get(context, self._empty_stats())
        scores = []
        for candidate in candidates:
            scores.append(
                self.left_weight * self._score(local, "left", candidate.left_digest)
                + self.right_weight * self._score(local, "right", candidate.right_digest)
                + self.mode_weight * self._score(local, "mode", candidate.mode)
            )
        logits = torch.tensor(scores, dtype=torch.float64) / self.temperature
        probabilities = torch.softmax(logits, dim=0)
        return (1.0 - self.exploration_floor) * probabilities + (
            self.exploration_floor / len(candidates)
        )

    def select(
        self,
        context: str,
        candidates: Sequence[RecipeProgramCompositionCandidate],
        *,
        generator: torch.Generator,
    ) -> tuple[int, float]:
        values = tuple(candidates)
        probabilities = self.proposal_probabilities(
            context, tuple(candidate.factors for candidate in values)
        )
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        return index, float(probabilities[index].item())

    @classmethod
    def _serialize_stats(
        cls,
        stats: Mapping[str, Mapping[str, Sequence[float]]],
    ) -> dict[str, dict[str, list[float]]]:
        result: dict[str, dict[str, list[float]]] = {}
        for factor_type in cls._FACTOR_TYPES:
            values = stats.get(factor_type)
            if not isinstance(values, Mapping):
                raise TypeError("composition factor table is malformed")
            result[factor_type] = {
                str(key): list(cls._aggregate(entry))
                for key, entry in sorted(values.items())
            }
        return result

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "shared": self._serialize_stats(self._shared),
            "contexts": {
                context: self._serialize_stats(stats)
                for context, stats in sorted(self._contexts.items())
            },
        }

    def digest(self) -> str:
        return _payload_digest(self._content_payload())

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> OpaqueContextRecipeCompositionMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported composition policy payload")
        configuration = payload.get("configuration")
        shared = payload.get("shared")
        contexts = payload.get("contexts")
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(shared, Mapping)
            or not isinstance(contexts, Mapping)
        ):
            raise TypeError("composition policy payload is malformed")
        policy = cls(
            exploration_floor=float(configuration.get("exploration_floor", -1.0)),
            shared_prior_weight=float(configuration.get("shared_prior_weight", -1.0)),
            exploration_bonus=float(configuration.get("exploration_bonus", -1.0)),
            left_weight=float(configuration.get("left_weight", -1.0)),
            right_weight=float(configuration.get("right_weight", -1.0)),
            mode_weight=float(configuration.get("mode_weight", -1.0)),
            temperature=float(configuration.get("temperature", -1.0)),
        )

        def load(raw: Mapping[str, object]) -> dict[str, dict[str, list[float]]]:
            loaded = policy._empty_stats()
            for factor_type in policy._FACTOR_TYPES:
                values = raw.get(factor_type)
                if not isinstance(values, Mapping):
                    raise TypeError("composition factor table is malformed")
                for key, entry in values.items():
                    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
                        raise TypeError("composition aggregate is malformed")
                    total, count = policy._aggregate(entry)
                    loaded[factor_type][str(key)] = [total, count]
            return loaded

        policy._shared = load(shared)
        for context, raw in contexts.items():
            policy._validate_context(str(context))
            if not isinstance(raw, Mapping):
                raise TypeError("composition context table is malformed")
            policy._contexts[str(context)] = load(raw)
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != policy.digest():
            raise ValueError("composition policy checksum mismatch")
        return policy


@dataclass(frozen=True)
class RecipeProgramCompositionSearchState:
    seen_candidate_digests: tuple[str, ...] = ()
    proposals: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = RECIPE_COMPOSITION_SEARCH_SCHEMA

    def validate(self) -> RecipeProgramCompositionSearchState:
        if self.schema != RECIPE_COMPOSITION_SEARCH_SCHEMA:
            raise ValueError("unsupported composition search schema")
        if self.proposals < 0 or self.accepted < 0 or self.accepted > self.proposals:
            raise ValueError("composition search counters are invalid")
        if len(self.seen_candidate_digests) != self.proposals or len(
            set(self.seen_candidate_digests)
        ) != len(self.seen_candidate_digests):
            raise ValueError("composition candidate history is inconsistent")
        for value in self.seen_candidate_digests:
            if "\0" not in value:
                raise ValueError("composition candidate history key is malformed")
            scope, digest = value.rsplit("\0", 1)
            _scoped_key(scope, digest)
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("composition best quality is invalid")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "seen_candidate_digests": list(self.seen_candidate_digests),
            "proposals": self.proposals,
            "accepted": self.accepted,
            "best_quality": self.best_quality,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> RecipeProgramCompositionSearchState:
        if not isinstance(payload, Mapping):
            raise TypeError("composition search payload must be a mapping")
        return cls(
            seen_candidate_digests=tuple(payload.get("seen_candidate_digests", ())),
            proposals=int(payload.get("proposals", -1)),
            accepted=int(payload.get("accepted", -1)),
            best_quality=float(payload.get("best_quality", float("nan"))),
            schema=str(payload.get("schema", "")),
        ).validate()


@dataclass(frozen=True)
class RecipeProgramCompositionProposal:
    candidate: RecipeProgramCompositionCandidate
    attempt_id: int
    selection_probability: float
    scope: str = "default"
    context: str = "default"
    schema: str = RECIPE_COMPOSITION_PROPOSAL_SCHEMA

    def validate(self) -> RecipeProgramCompositionProposal:
        if self.schema != RECIPE_COMPOSITION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported composition proposal schema")
        self.candidate.validate()
        if self.attempt_id < 0:
            raise ValueError("composition proposal attempt cannot be negative")
        if not math.isfinite(self.selection_probability) or not (
            0.0 < self.selection_probability <= 1.0
        ):
            raise ValueError("composition proposal probability is invalid")
        for name, value in (("scope", self.scope), ("context", self.context)):
            if not isinstance(value, str) or not value or "\0" in value:
                raise ValueError(f"composition {name} must be a non-empty opaque key")
        return self


@dataclass(frozen=True)
class RecipeProgramCompositionFeedback:
    proposal: RecipeProgramCompositionProposal
    receipt: RecipeProgramAdmissionReceipt
    quality: float
    state: RecipeProgramCompositionSearchState


class OutcomeOnlyRecipeCompositionSearch:
    """Select and evaluate external-file compositions from scalar outcomes."""

    schema = RECIPE_COMPOSITION_SEARCH_SCHEMA

    def __init__(
        self,
        memory: ExternalRecipeCompositionMemory,
        *,
        max_program_length: int,
        policy: OpaqueContextRecipeCompositionMemory | None = None,
    ) -> None:
        if not isinstance(memory, ExternalRecipeCompositionMemory):
            raise TypeError("composition search memory has the wrong type")
        if max_program_length < 2:
            raise ValueError("composition search maximum length must be at least two")
        if policy is not None and not isinstance(
            policy, OpaqueContextRecipeCompositionMemory
        ):
            raise TypeError("composition search policy has the wrong type")
        self.memory = memory
        self.max_program_length = int(max_program_length)
        self.policy = policy

    def initial_state(self) -> RecipeProgramCompositionSearchState:
        return RecipeProgramCompositionSearchState().validate()

    def propose(
        self,
        state: RecipeProgramCompositionSearchState,
        *,
        generator: torch.Generator,
        scope: str = "default",
        context: str = "default",
    ) -> RecipeProgramCompositionProposal:
        state.validate()
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("composition scope must be a non-empty opaque key")
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("composition context must be a non-empty opaque key")
        candidates = tuple(
            candidate
            for candidate in self.memory.composition_candidates(
                max_program_length=self.max_program_length
            )
            if _scoped_key(scope, candidate.program.digest())
            not in state.seen_candidate_digests
        )
        if not candidates:
            raise RuntimeError("composition neighborhood is exhausted")
        if self.policy is None:
            selected = 0
            probability = 1.0 / len(candidates)
        else:
            selected, probability = self.policy.select(
                context,
                candidates,
                generator=generator,
            )
        return RecipeProgramCompositionProposal(
            candidates[selected],
            state.proposals,
            probability,
            scope,
            context,
        ).validate()

    def propose_exhaustive(
        self,
        state: RecipeProgramCompositionSearchState,
        *,
        scope: str = "default",
        context: str = "default",
    ) -> RecipeProgramCompositionProposal:
        state.validate()
        candidates = tuple(
            candidate
            for candidate in self.memory.composition_candidates(
                max_program_length=self.max_program_length
            )
            if _scoped_key(scope, candidate.program.digest())
            not in state.seen_candidate_digests
        )
        if not candidates:
            raise RuntimeError("composition neighborhood is exhausted")
        return RecipeProgramCompositionProposal(
            candidates[0],
            state.proposals,
            1.0 / len(candidates),
            scope,
            context,
        ).validate()

    def record_outcomes(
        self,
        state: RecipeProgramCompositionSearchState,
        proposal: RecipeProgramCompositionProposal,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> RecipeProgramCompositionFeedback:
        state.validate()
        proposal.validate()
        if proposal.attempt_id != state.proposals:
            raise ValueError("composition proposal is out of sequence")
        digest = proposal.candidate.program.digest()
        scoped = _scoped_key(proposal.scope, digest)
        if scoped in state.seen_candidate_digests:
            raise ValueError("composition candidate was already evaluated")
        values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
        receipt = evaluate_recipe_program_admission(
            proposal.candidate.program,
            values,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        quality = float(values.mean().item()) if values.numel() else 0.0
        if self.policy is not None:
            self.policy.record(proposal.context, proposal.candidate.factors, quality)
        next_state = RecipeProgramCompositionSearchState(
            (*state.seen_candidate_digests, scoped),
            state.proposals + 1,
            state.accepted + int(receipt.accepted),
            max(state.best_quality, quality),
        ).validate()
        return RecipeProgramCompositionFeedback(proposal, receipt, quality, next_state)


__all__ = [
    "RECIPE_COMPOSITION_CANDIDATE_SCHEMA",
    "RECIPE_COMPOSITION_FACTORS_SCHEMA",
    "RECIPE_COMPOSITION_MEMORY_SCHEMA",
    "RECIPE_COMPOSITION_MODES",
    "RECIPE_COMPOSITION_POLICY_SCHEMA",
    "RECIPE_COMPOSITION_PROPOSAL_SCHEMA",
    "RECIPE_COMPOSITION_SEARCH_SCHEMA",
    "ExternalRecipeCompositionMemory",
    "OpaqueContextRecipeCompositionMemory",
    "OutcomeOnlyRecipeCompositionSearch",
    "RecipeProgramCompositionCandidate",
    "RecipeProgramCompositionFactors",
    "RecipeProgramCompositionFeedback",
    "RecipeProgramCompositionProposal",
    "RecipeProgramCompositionSearchState",
]
