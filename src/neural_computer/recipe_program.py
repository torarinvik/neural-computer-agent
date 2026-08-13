"""Outcome-only external files for the generic recipe basis.

This module is the storage-side bridge between :mod:`recipe_basis` and the
external-computation boundary.  A recipe program is an opaque sequence of
generic instructions.  Search sees only scalar verifier outcomes; the memory
backend owns copy-on-write admission, protection, and persistence.  No task
name, correct action, verifier row, or controller parameter is stored here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .recipe_basis import RecipeBasis, RecipeInstruction, apply_sequence

RECIPE_PROGRAM_SCHEMA = "neural-computer.external-recipe-program.v1"
RECIPE_PROGRAM_MEMORY_SCHEMA = "neural-computer.external-recipe-program-memory.v1"
RECIPE_PROGRAM_SEARCH_SCHEMA = "neural-computer.external-recipe-program-search.v2"
RECIPE_PROGRAM_PROPOSAL_SCHEMA = "neural-computer.external-recipe-program-proposal.v2"
RECIPE_CONTEXT_PROPOSAL_MEMORY_SCHEMA = (
    "neural-computer.external-context-conditioned-recipe-proposal-memory.v1"
)
RECIPE_PROPOSAL_FACTORS_SCHEMA = "neural-computer.recipe-proposal-factors.v1"
RECIPE_FACTORIZED_CONTEXT_PROPOSAL_MEMORY_SCHEMA = (
    "neural-computer.external-factorized-context-proposal-memory.v1"
)
RECIPE_PROGRAM_ADMISSION_SCHEMA = "neural-computer.external-recipe-program-admission.v1"
RECIPE_PROGRAM_MUTATION_OPERATORS = ("replace", "insert", "delete", "swap")


def _instruction_payload(instruction: RecipeInstruction) -> dict[str, object]:
    return {
        "op": instruction.op,
        "first": instruction.first,
        "second": instruction.second,
        "modulus": instruction.modulus,
        "children": (
            [_instruction_payload(child) for child in instruction.children]
            if instruction.children is not None
            else None
        ),
    }


def _instruction_from_payload(payload: Mapping[str, object]) -> RecipeInstruction:
    if not isinstance(payload, Mapping):
        raise TypeError("recipe instruction payload must be a mapping")
    children_payload = payload.get("children")
    children = None
    if children_payload is not None:
        if not isinstance(children_payload, Sequence) or len(children_payload) != 2:
            raise ValueError("parallel recipe instruction needs two children")
        children = tuple(
            _instruction_from_payload(child)
            for child in children_payload
        )
    return RecipeInstruction(
        op=payload.get("op"),
        first=payload.get("first"),
        second=payload.get("second"),
        modulus=payload.get("modulus"),
        children=children,  # type: ignore[arg-type]
    )


def _canonical_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scoped_candidate_key(scope: str, digest: str) -> str:
    if not isinstance(scope, str) or not scope or "\0" in scope:
        raise ValueError("recipe candidate scope must be a non-empty opaque key")
    if len(digest) != 64:
        raise ValueError("recipe candidate digest is malformed")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("recipe candidate digest is malformed") from error
    return f"{scope}\0{digest}"


def _validate_scoped_candidate_key(value: str) -> None:
    if not isinstance(value, str) or "\0" not in value:
        raise ValueError("recipe candidate history key is malformed")
    scope, digest = value.rsplit("\0", 1)
    _scoped_candidate_key(scope, digest)


@dataclass(frozen=True)
class RecipeProgram:
    """One portable opaque program file over a fixed generic basis."""

    slot_values: tuple[int, ...]
    instructions: tuple[RecipeInstruction, ...]
    allow_parallel: bool = False
    schema: str = RECIPE_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECIPE_PROGRAM_SCHEMA:
            raise ValueError("unsupported recipe program schema")
        values = tuple(int(value) for value in self.slot_values)
        if not values or any(value < 2 for value in values):
            raise ValueError("recipe program needs valid slot value domains")
        instructions = tuple(self.instructions)
        if not instructions:
            raise ValueError("recipe program needs at least one instruction")
        basis = RecipeBasis(
            slot_count=len(values),
            slot_values=values,
            allow_parallel=self.allow_parallel,
        )
        for instruction in instructions:
            if not isinstance(instruction, RecipeInstruction):
                raise TypeError("recipe program instructions must be typed instructions")
            instruction.validate(
                slot_count=basis.slot_count,
                allow_parallel=basis.allow_parallel,
            )
        object.__setattr__(self, "slot_values", values)
        object.__setattr__(self, "instructions", instructions)

    @property
    def program_length(self) -> int:
        return len(self.instructions)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "slot_values": self.slot_values,
            "allow_parallel": self.allow_parallel,
            "program_length": self.program_length,
            "execution": "recipe_basis_sequence_v1",
            "storage": "opaque_typed_instruction_file_v1",
        }

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "instructions": [
                _instruction_payload(instruction)
                for instruction in self.instructions
            ],
        }

    def digest(self) -> str:
        return _canonical_digest(self._content_payload())

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RecipeProgram:
        if not isinstance(payload, Mapping):
            raise TypeError("recipe program payload must be a mapping")
        if payload.get("schema") != RECIPE_PROGRAM_SCHEMA:
            raise ValueError("unsupported recipe program payload schema")
        configuration = payload.get("configuration")
        raw_instructions = payload.get("instructions")
        if not isinstance(configuration, Mapping):
            raise TypeError("recipe program configuration is missing")
        if not isinstance(raw_instructions, Sequence):
            raise TypeError("recipe program instructions are missing")
        program = cls(
            slot_values=tuple(configuration.get("slot_values", ())),
            instructions=tuple(
                _instruction_from_payload(item) for item in raw_instructions
            ),
            allow_parallel=bool(configuration.get("allow_parallel", False)),
        )
        if configuration.get("program_length") != program.program_length:
            raise ValueError("recipe program length metadata is inconsistent")
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != program.digest():
            raise ValueError("recipe program checksum mismatch")
        return program

    def execute(self, state: tuple[int, ...]) -> tuple[int, ...]:
        return apply_sequence(self.instructions, state, values=self.slot_values)


@dataclass(frozen=True)
class RecipeProgramAdmissionReceipt:
    """Scalar-only admission evidence for one staged recipe file."""

    accepted: bool
    candidate_digest: str
    slot: int | None
    observations: int
    stable_bits_to_threshold: int | None
    stable_prefix_minimum: float | None
    reason: str
    schema: str = RECIPE_PROGRAM_ADMISSION_SCHEMA

    def validate(self) -> RecipeProgramAdmissionReceipt:
        if self.schema != RECIPE_PROGRAM_ADMISSION_SCHEMA:
            raise ValueError("unsupported recipe program admission schema")
        if len(self.candidate_digest) != 64:
            raise ValueError("recipe program candidate digest is malformed")
        try:
            int(self.candidate_digest, 16)
        except ValueError as error:
            raise ValueError("recipe program candidate digest is malformed") from error
        if self.observations < 0:
            raise ValueError("recipe program observations cannot be negative")
        if self.slot is not None and self.slot < 0:
            raise ValueError("recipe program slot cannot be negative")
        if self.stable_bits_to_threshold is not None and not (
            1 <= self.stable_bits_to_threshold <= self.observations
        ):
            raise ValueError("recipe program stable prefix is invalid")
        if self.stable_prefix_minimum is not None and not (
            0.0 <= self.stable_prefix_minimum <= 1.0
        ):
            raise ValueError("recipe program stable minimum must lie in [0, 1]")
        if not self.reason:
            raise ValueError("recipe program admission reason is required")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "candidate_digest": self.candidate_digest,
            "slot": self.slot,
            "observations": self.observations,
            "stable_bits_to_threshold": self.stable_bits_to_threshold,
            "stable_prefix_minimum": self.stable_prefix_minimum,
            "reason": self.reason,
        }


def evaluate_recipe_program_admission(
    program: RecipeProgram,
    outcomes: torch.Tensor | Sequence[float],
    *,
    threshold: float = 0.8,
    min_observations: int = 1,
    min_stable_observations: int = 1,
) -> RecipeProgramAdmissionReceipt:
    """Evaluate a candidate from an ordered scalar outcome stream only."""

    if not isinstance(program, RecipeProgram):
        raise TypeError("recipe admission requires a recipe program")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("recipe admission threshold must lie in [0, 1]")
    if min_observations < 1 or min_stable_observations < 1:
        raise ValueError("recipe admission observation bounds must be positive")
    values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return RecipeProgramAdmissionReceipt(
            False, program.digest(), None, 0, None, None, "no verifier outcomes"
        ).validate()
    if not bool(torch.isfinite(values).all()) or bool(
        torch.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("recipe admission outcomes must lie in [0, 1]")
    if values.numel() < min_observations:
        return RecipeProgramAdmissionReceipt(
            False,
            program.digest(),
            None,
            int(values.numel()),
            None,
            float(values.min().item()),
            "candidate has not reached minimum verifier observations",
        ).validate()
    stable_prefix = None
    for index in range(values.numel()):
        if values.numel() - index >= min_stable_observations and bool(
            torch.all(values[index:] >= threshold)
        ):
            stable_prefix = index + 1
            break
    if stable_prefix is None:
        return RecipeProgramAdmissionReceipt(
            False,
            program.digest(),
            None,
            int(values.numel()),
            None,
            float(values.min().item()),
            "candidate did not clear a stable verifier prefix",
        ).validate()
    return RecipeProgramAdmissionReceipt(
        True,
        program.digest(),
        None,
        int(values.numel()),
        stable_prefix,
        float(values[stable_prefix - 1 :].min().item()),
        "candidate cleared the stable verifier prefix",
    ).validate()


class ExternalRecipeProgramMemory:
    """A small persistent, verifier-gated bank of generic recipe files."""

    schema = RECIPE_PROGRAM_MEMORY_SCHEMA

    def __init__(
        self,
        slot_values: Sequence[int],
        *,
        allow_parallel: bool = False,
    ) -> None:
        values = tuple(int(value) for value in slot_values)
        self.basis = RecipeBasis(
            slot_count=len(values),
            slot_values=values,
            allow_parallel=allow_parallel,
        )
        self._programs: list[RecipeProgram] = []
        self._protected: list[bool] = []

    @property
    def file_count(self) -> int:
        return len(self._programs)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "slot_values": self.basis.slot_values,
            "allow_parallel": self.basis.allow_parallel,
            "file_count": self.file_count,
            "protected": list(self._protected),
            "storage": "append_only_verifier_gated_recipe_files_v1",
        }

    def _validate_program(self, program: RecipeProgram) -> None:
        if program.slot_values != self.basis.slot_values:
            raise ValueError("recipe program value domains are incompatible")
        if program.allow_parallel != self.basis.allow_parallel:
            raise ValueError("recipe program atomicity is incompatible")

    def add_program(self, program: RecipeProgram) -> int:
        if not isinstance(program, RecipeProgram):
            raise TypeError("recipe memory requires a RecipeProgram")
        self._validate_program(program)
        self._programs.append(program)
        self._protected.append(False)
        return len(self._programs) - 1

    def program(self, slot: int) -> RecipeProgram:
        if not 0 <= slot < self.file_count:
            raise ValueError("recipe memory slot is out of range")
        return self._programs[slot]

    def protect_file(self, slot: int) -> None:
        if not 0 <= slot < self.file_count:
            raise ValueError("recipe memory slot is out of range")
        self._protected[slot] = True

    def is_file_protected(self, slot: int) -> bool:
        if not 0 <= slot < self.file_count:
            raise ValueError("recipe memory slot is out of range")
        return self._protected[slot]

    def execute(self, slot: int, state: tuple[int, ...]) -> tuple[int, ...]:
        return self.program(slot).execute(state)

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
        self._validate_program(program)
        receipt = evaluate_recipe_program_admission(
            program,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if not receipt.accepted:
            return receipt
        slot = self.add_program(program)
        if protect:
            self.protect_file(slot)
        return RecipeProgramAdmissionReceipt(
            True,
            receipt.candidate_digest,
            slot,
            receipt.observations,
            receipt.stable_bits_to_threshold,
            receipt.stable_prefix_minimum,
            "candidate verified and committed as an external recipe file",
        ).validate()

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "programs": [program.payload() for program in self._programs],
        }

    def digest(self) -> str:
        return _canonical_digest(self._content_payload())

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExternalRecipeProgramMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported recipe program memory payload")
        configuration = payload.get("configuration")
        programs = payload.get("programs")
        if not isinstance(configuration, Mapping) or not isinstance(programs, Sequence):
            raise TypeError("recipe program memory payload is malformed")
        memory = cls(
            tuple(configuration.get("slot_values", ())),
            allow_parallel=bool(configuration.get("allow_parallel", False)),
        )
        for raw_program in programs:
            memory.add_program(RecipeProgram.from_payload(raw_program))
        protected = configuration.get("protected")
        if not isinstance(protected, Sequence) or len(protected) != memory.file_count:
            raise ValueError("recipe memory protection metadata is inconsistent")
        memory._protected = [bool(value) for value in protected]
        if configuration.get("file_count") != memory.file_count:
            raise ValueError("recipe memory file count metadata is inconsistent")
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != memory.digest():
            raise ValueError("recipe memory checksum mismatch")
        return memory


@dataclass(frozen=True)
class RecipeProgramSearchState:
    reward_totals: torch.Tensor
    reward_counts: torch.Tensor
    accepted_counts: torch.Tensor
    failed_counts: torch.Tensor
    seen_candidate_digests: tuple[str, ...] = ()
    proposals: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = RECIPE_PROGRAM_SEARCH_SCHEMA

    def validate(self) -> RecipeProgramSearchState:
        if self.schema != RECIPE_PROGRAM_SEARCH_SCHEMA:
            raise ValueError("unsupported recipe search schema")
        expected = (len(RECIPE_PROGRAM_MUTATION_OPERATORS),)
        for name, value in (
            ("reward_totals", self.reward_totals),
            ("reward_counts", self.reward_counts),
            ("accepted_counts", self.accepted_counts),
            ("failed_counts", self.failed_counts),
        ):
            if value.shape != expected or not value.is_floating_point():
                raise ValueError(f"recipe search {name} has the wrong shape")
            if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0.0)):
                raise ValueError(f"recipe search {name} is invalid")
        if self.proposals < 0 or self.accepted < 0 or self.accepted > self.proposals:
            raise ValueError("recipe search counters are invalid")
        if int(self.reward_counts.sum().item()) != self.proposals:
            raise ValueError("recipe search proposal counts are inconsistent")
        if int(self.accepted_counts.sum().item()) != self.accepted:
            raise ValueError("recipe search acceptance counts are inconsistent")
        if int(self.failed_counts.sum().item()) != self.proposals - self.accepted:
            raise ValueError("recipe search failure counts are inconsistent")
        if len(self.seen_candidate_digests) != self.proposals or len(
            set(self.seen_candidate_digests)
        ) != len(self.seen_candidate_digests):
            raise ValueError("recipe search candidate history is inconsistent")
        for candidate_key in self.seen_candidate_digests:
            _validate_scoped_candidate_key(candidate_key)
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("recipe search best quality is invalid")
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
    def from_payload(cls, payload: Mapping[str, object]) -> RecipeProgramSearchState:
        if not isinstance(payload, Mapping):
            raise TypeError("recipe search payload must be a mapping")
        return cls(
            reward_totals=payload.get("reward_totals"),  # type: ignore[arg-type]
            reward_counts=payload.get("reward_counts"),  # type: ignore[arg-type]
            accepted_counts=payload.get("accepted_counts"),  # type: ignore[arg-type]
            failed_counts=payload.get("failed_counts"),  # type: ignore[arg-type]
            seen_candidate_digests=tuple(payload.get("seen_candidate_digests", ())),
            proposals=int(payload.get("proposals", -1)),
            accepted=int(payload.get("accepted", -1)),
            best_quality=float(payload.get("best_quality", float("nan"))),
            schema=str(payload.get("schema", "")),
        ).validate()


def _instruction_digest(instruction: RecipeInstruction) -> str:
    return _canonical_digest(_instruction_payload(instruction))


@dataclass(frozen=True)
class RecipeProgramProposalFactors:
    """Opaque, generic factors for one sequence edit.

    The factors describe *how* a candidate changes its parent, not what the
    candidate means.  This lets external memory reuse instruction and
    position credit across different parent programs while keeping the
    controller unaware of the recipe ABI.
    """

    operator_index: int
    primary_position: int
    secondary_position: int | None = None
    instruction_digests: tuple[str, ...] = ()
    schema: str = RECIPE_PROPOSAL_FACTORS_SCHEMA

    def validate(self) -> RecipeProgramProposalFactors:
        if self.schema != RECIPE_PROPOSAL_FACTORS_SCHEMA:
            raise ValueError("unsupported recipe proposal factors schema")
        if not 0 <= self.operator_index < len(RECIPE_PROGRAM_MUTATION_OPERATORS):
            raise ValueError("recipe proposal factors operator is invalid")
        if self.primary_position < 0:
            raise ValueError("recipe proposal factors primary position is invalid")
        if self.secondary_position is not None and (
            self.secondary_position < 0
            or self.secondary_position == self.primary_position
        ):
            raise ValueError("recipe proposal factors secondary position is invalid")
        if self.operator_index == 3:
            if self.secondary_position is None or not (
                self.primary_position < self.secondary_position
            ):
                raise ValueError("swap factors need ordered positions")
            expected_instructions = 2
        else:
            if self.secondary_position is not None:
                raise ValueError("non-swap factors cannot have a second position")
            expected_instructions = 1
        if len(self.instruction_digests) != expected_instructions:
            raise ValueError("recipe proposal factors instruction count is invalid")
        for digest in self.instruction_digests:
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("recipe proposal factors instruction digest is invalid")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(
                    "recipe proposal factors instruction digest is invalid"
                ) from error
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
    ) -> RecipeProgramProposalFactors:
        if not isinstance(payload, Mapping):
            raise TypeError("recipe proposal factors payload must be a mapping")
        instructions = payload.get("instruction_digests", ())
        if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
            raise TypeError("recipe proposal factors instructions are malformed")
        secondary = payload.get("secondary_position")
        return cls(
            operator_index=int(payload.get("operator_index", -1)),
            primary_position=int(payload.get("primary_position", -1)),
            secondary_position=None if secondary is None else int(secondary),
            instruction_digests=tuple(str(value) for value in instructions),
            schema=str(payload.get("schema", "")),
        ).validate()


class OpaqueContextRecipeProposalMemory:
    """External scalar credit for content-addressed proposals.

    The memory is deliberately outside the controller.  It stores only
    aggregate verifier quality for an opaque context and an opaque candidate
    digest; it never stores verifier rows or a semantic task label.  A small
    global prior can seed an unseen context, while the context-local estimate
    takes over as soon as that context has evidence.  The exploration floor
    keeps inherited evidence from making any candidate unreachable.
    """

    schema = RECIPE_CONTEXT_PROPOSAL_MEMORY_SCHEMA

    def __init__(
        self,
        *,
        exploration_floor: float = 0.2,
        global_prior_weight: float = 0.1,
        exploration_bonus: float = 0.5,
        temperature: float = 0.5,
    ) -> None:
        for name, value in (
            ("exploration_floor", exploration_floor),
            ("global_prior_weight", global_prior_weight),
            ("exploration_bonus", exploration_bonus),
            ("temperature", temperature),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"recipe proposal {name} is invalid")
        if exploration_floor >= 1.0:
            raise ValueError("recipe proposal exploration floor must be < 1")
        if temperature <= 0.0:
            raise ValueError("recipe proposal temperature must be positive")
        self.exploration_floor = float(exploration_floor)
        self.global_prior_weight = float(global_prior_weight)
        self.exploration_bonus = float(exploration_bonus)
        self.temperature = float(temperature)
        self._context_stats: dict[str, dict[str, list[float]]] = {}
        self._global_stats: dict[str, list[float]] = {}

    @staticmethod
    def _validate_context(context: str) -> None:
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("recipe proposal context must be a non-empty opaque key")

    @staticmethod
    def _validate_candidate_digest(candidate_digest: str) -> None:
        if not isinstance(candidate_digest, str) or len(candidate_digest) != 64:
            raise ValueError("recipe proposal candidate digest is malformed")
        try:
            int(candidate_digest, 16)
        except ValueError as error:
            raise ValueError("recipe proposal candidate digest is malformed") from error

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "exploration_floor": self.exploration_floor,
            "global_prior_weight": self.global_prior_weight,
            "exploration_bonus": self.exploration_bonus,
            "temperature": self.temperature,
            "credit": "scalar_candidate_quality_aggregate_v1",
            "context": "opaque_external_key_v1",
            "candidate": "content_addressed_recipe_digest_v1",
        }

    @staticmethod
    def _validate_candidates(candidate_digests: Sequence[str]) -> tuple[str, ...]:
        candidates = tuple(candidate_digests)
        if not candidates:
            raise ValueError("recipe proposal candidate set cannot be empty")
        for digest in candidates:
            OpaqueContextRecipeProposalMemory._validate_candidate_digest(digest)
        if len(set(candidates)) != len(candidates):
            raise ValueError("recipe proposal candidate set contains duplicates")
        return candidates

    @staticmethod
    def _mean_and_count(entry: Sequence[float] | None) -> tuple[float, float]:
        if entry is None:
            return 0.0, 0.0
        if len(entry) != 2:
            raise ValueError("recipe proposal aggregate is malformed")
        total, count = float(entry[0]), float(entry[1])
        if not math.isfinite(total) or not math.isfinite(count):
            raise ValueError("recipe proposal aggregate is non-finite")
        if total < 0.0 or count < 0.0 or total > count:
            raise ValueError("recipe proposal aggregate is outside [0, 1]")
        return total, count

    def record(
        self,
        context: str,
        candidate_digest: str,
        quality: float,
        *,
        factors: RecipeProgramProposalFactors | None = None,
    ) -> None:
        """Record one scalar result without retaining the underlying outcomes."""

        del factors
        self._validate_context(context)
        self._validate_candidate_digest(candidate_digest)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("recipe proposal quality must lie in [0, 1]")
        context_stats = self._context_stats.setdefault(context, {})
        local = context_stats.setdefault(candidate_digest, [0.0, 0.0])
        local[0] += float(quality)
        local[1] += 1.0
        global_entry = self._global_stats.setdefault(candidate_digest, [0.0, 0.0])
        global_entry[0] += float(quality)
        global_entry[1] += 1.0

    def proposal_probabilities(
        self,
        context: str,
        candidate_digests: Sequence[str],
    ) -> torch.Tensor:
        """Return the context-conditioned distribution over opaque proposals."""

        self._validate_context(context)
        candidates = self._validate_candidates(candidate_digests)
        local_stats = self._context_stats.get(context, {})
        local_total = sum(self._mean_and_count(local_stats.get(digest))[1] for digest in candidates)
        prior_total = sum(
            self._mean_and_count(self._global_stats.get(digest))[1]
            * self.global_prior_weight
            for digest in candidates
        )
        scale = math.log1p(local_total + prior_total + 1.0)
        scores: list[float] = []
        for digest in candidates:
            local_quality, local_count = self._mean_and_count(local_stats.get(digest))
            global_quality, global_count = self._mean_and_count(
                self._global_stats.get(digest)
            )
            effective_count = local_count + self.global_prior_weight * global_count
            effective_total = (
                local_quality * local_count
                + self.global_prior_weight * global_quality * global_count
            )
            mean = effective_total / effective_count if effective_count else 0.0
            bonus = self.exploration_bonus * math.sqrt(
                scale / (effective_count + 1.0)
            )
            scores.append(mean + bonus)
        logits = torch.tensor(scores, dtype=torch.float64) / self.temperature
        probabilities = torch.softmax(logits, dim=0)
        floor = self.exploration_floor
        return (1.0 - floor) * probabilities + floor / len(candidates)

    def select(
        self,
        context: str,
        candidate_digests: Sequence[str],
        *,
        factors: Sequence[RecipeProgramProposalFactors] | None = None,
        generator: torch.Generator,
    ) -> tuple[int, float]:
        del factors
        probabilities = self.proposal_probabilities(context, candidate_digests)
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        return index, float(probabilities[index].item())

    @staticmethod
    def _serialize_stats(
        stats: Mapping[str, Mapping[str, Sequence[float]]],
    ) -> dict[str, dict[str, list[float]]]:
        serialized: dict[str, dict[str, list[float]]] = {}
        for context, candidates in sorted(stats.items()):
            OpaqueContextRecipeProposalMemory._validate_context(context)
            serialized[context] = {}
            for digest, entry in sorted(candidates.items()):
                OpaqueContextRecipeProposalMemory._validate_candidate_digest(digest)
                total, count = OpaqueContextRecipeProposalMemory._mean_and_count(entry)
                serialized[context][digest] = [total, count]
        return serialized

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "context_stats": self._serialize_stats(self._context_stats),
            "global_stats": self._serialize_stats({"global": self._global_stats})[
                "global"
            ],
        }

    def digest(self) -> str:
        return _canonical_digest(self._content_payload())

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> OpaqueContextRecipeProposalMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported context proposal memory payload")
        configuration = payload.get("configuration")
        context_stats = payload.get("context_stats")
        global_stats = payload.get("global_stats")
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(context_stats, Mapping)
            or not isinstance(global_stats, Mapping)
        ):
            raise TypeError("context proposal memory payload is malformed")
        memory = cls(
            exploration_floor=float(configuration.get("exploration_floor", -1.0)),
            global_prior_weight=float(configuration.get("global_prior_weight", -1.0)),
            exploration_bonus=float(configuration.get("exploration_bonus", -1.0)),
            temperature=float(configuration.get("temperature", -1.0)),
        )

        def load_stats(raw: Mapping[str, object]) -> dict[str, dict[str, list[float]]]:
            loaded: dict[str, dict[str, list[float]]] = {}
            for context, candidates in raw.items():
                cls._validate_context(str(context))
                if not isinstance(candidates, Mapping):
                    raise TypeError("context proposal candidate stats are malformed")
                loaded[str(context)] = {}
                for digest, entry in candidates.items():
                    cls._validate_candidate_digest(str(digest))
                    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
                        raise TypeError("context proposal aggregate is malformed")
                    total, count = cls._mean_and_count(entry)
                    loaded[str(context)][str(digest)] = [total, count]
            return loaded

        memory._context_stats = load_stats(context_stats)
        loaded_global = load_stats({"global": global_stats})
        memory._global_stats = loaded_global["global"]
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != memory.digest():
            raise ValueError("context proposal memory checksum mismatch")
        return memory


class FactorizedOpaqueContextRecipeProposalMemory:
    """External scalar credit over reusable recipe-edit factors.

    Unlike :class:`OpaqueContextRecipeProposalMemory`, this memory does not
    retain or score whole candidate digests.  It aggregates quality over
    opaque instruction digests, operator/position factors, and operator
    identity.  A shared factor prior supports related-context transfer, while
    context-local factors can override it after evidence arrives.
    """

    schema = RECIPE_FACTORIZED_CONTEXT_PROPOSAL_MEMORY_SCHEMA
    _FACTOR_TYPES = ("instruction", "position", "operator")

    def __init__(
        self,
        *,
        exploration_floor: float = 0.2,
        shared_prior_weight: float = 0.1,
        exploration_bonus: float = 0.25,
        instruction_weight: float = 1.0,
        position_weight: float = 0.75,
        operator_weight: float = 0.25,
        temperature: float = 0.5,
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
                raise ValueError(f"factorized recipe proposal {name} is invalid")
        if exploration_floor >= 1.0:
            raise ValueError("factorized recipe exploration floor must be < 1")
        if instruction_weight + position_weight + operator_weight <= 0.0:
            raise ValueError("factorized recipe proposal weights cannot all be zero")
        if temperature <= 0.0:
            raise ValueError("factorized recipe proposal temperature must be positive")
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
            raise ValueError("factorized recipe context must be a non-empty opaque key")

    @staticmethod
    def _validate_digest(digest: str, *, label: str) -> None:
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{label} digest is malformed")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"{label} digest is malformed") from error

    @classmethod
    def _factor_entries(
        cls,
        factors: RecipeProgramProposalFactors,
    ) -> tuple[tuple[str, str], ...]:
        factors.validate()
        positions = [factors.primary_position]
        if factors.secondary_position is not None:
            positions.append(factors.secondary_position)
        entries: list[tuple[str, str]] = [
            ("operator", str(factors.operator_index)),
            *[("position", f"{factors.operator_index}:{position}") for position in positions],
            *[("instruction", digest) for digest in factors.instruction_digests],
        ]
        return tuple(entries)

    @staticmethod
    def _aggregate(entry: Sequence[float] | None) -> tuple[float, float]:
        if entry is None:
            return 0.0, 0.0
        if len(entry) != 2:
            raise ValueError("factorized recipe aggregate is malformed")
        total, count = float(entry[0]), float(entry[1])
        if not math.isfinite(total) or not math.isfinite(count):
            raise ValueError("factorized recipe aggregate is non-finite")
        if total < 0.0 or count < 0.0 or total > count:
            raise ValueError("factorized recipe aggregate is outside [0, 1]")
        return total, count

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
            "instruction": "content_addressed_generic_instruction_digest_v1",
            "position": "operator_relative_integer_position_v1",
        }

    def record(
        self,
        context: str,
        candidate_digest: str,
        quality: float,
        *,
        factors: RecipeProgramProposalFactors,
    ) -> None:
        """Record scalar quality against factors, discarding candidate identity."""

        self._validate_context(context)
        self._validate_digest(candidate_digest, label="recipe candidate")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("factorized recipe quality must lie in [0, 1]")
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
        factors: Sequence[RecipeProgramProposalFactors],
    ) -> torch.Tensor:
        self._validate_context(context)
        factor_list = tuple(factors)
        if not factor_list:
            raise ValueError("factorized recipe candidate set cannot be empty")
        for item in factor_list:
            item.validate()
        context_stats = self._context_stats.get(context, self._empty_stats())
        scores: list[float] = []
        for item in factor_list:
            entries = self._factor_entries(item)
            instruction_scores = [
                self._factor_score(
                    context_stats,
                    factor_type,
                    key,
                    shared_weight=(
                        0.0
                        if any(context_stats[factor_type].values())
                        else self.shared_prior_weight
                    ),
                )
                for factor_type, key in entries
                if factor_type == "instruction"
            ]
            position_scores = [
                self._factor_score(
                    context_stats,
                    factor_type,
                    key,
                    shared_weight=(
                        0.0
                        if any(context_stats[factor_type].values())
                        else self.shared_prior_weight
                    ),
                )
                for factor_type, key in entries
                if factor_type == "position"
            ]
            operator_score = self._factor_score(
                context_stats,
                "operator",
                str(item.operator_index),
                shared_weight=(
                    0.0
                    if any(context_stats["operator"].values())
                    else self.shared_prior_weight
                ),
            )
            instruction = sum(instruction_scores) / max(1, len(instruction_scores))
            position = sum(position_scores) / max(1, len(position_scores))
            scores.append(
                self.instruction_weight * instruction
                + self.position_weight * position
                + self.operator_weight * operator_score
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
        factors: Sequence[RecipeProgramProposalFactors],
        generator: torch.Generator,
    ) -> tuple[int, float]:
        digests = tuple(candidate_digests)
        if len(digests) != len(tuple(factors)):
            raise ValueError("factorized recipe candidates and factors disagree")
        for digest in digests:
            self._validate_digest(digest, label="recipe candidate")
        probabilities = self.proposal_probabilities(context, factors)
        index = int(torch.multinomial(probabilities, 1, generator=generator))
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
                raise TypeError("factorized recipe factor table is malformed")
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
        return _canonical_digest(self._content_payload())

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> FactorizedOpaqueContextRecipeProposalMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported factorized recipe proposal memory payload")
        configuration = payload.get("configuration")
        shared_stats = payload.get("shared_stats")
        context_stats = payload.get("context_stats")
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(shared_stats, Mapping)
            or not isinstance(context_stats, Mapping)
        ):
            raise TypeError("factorized recipe proposal memory payload is malformed")
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
                    raise TypeError("factorized recipe factor table is malformed")
                for key, entry in values.items():
                    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
                        raise TypeError("factorized recipe aggregate is malformed")
                    total, count = memory._aggregate(entry)
                    loaded[factor_type][str(key)] = [total, count]
            return loaded

        memory._shared_stats = load_stats(shared_stats)
        for context, stats in context_stats.items():
            cls._validate_context(str(context))
            if not isinstance(stats, Mapping):
                raise TypeError("factorized recipe context table is malformed")
            memory._context_stats[str(context)] = load_stats(stats)
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != memory.digest():
            raise ValueError("factorized recipe proposal memory checksum mismatch")
        return memory


@dataclass(frozen=True)
class RecipeProgramCandidateProposal:
    program: RecipeProgram
    parent_digest: str
    operator: str
    operator_index: int
    attempt_id: int
    selection_probability: float
    scope: str = "default"
    context: str = "default"
    factors: RecipeProgramProposalFactors | None = None
    schema: str = RECIPE_PROGRAM_PROPOSAL_SCHEMA

    def validate(self) -> RecipeProgramCandidateProposal:
        if self.schema != RECIPE_PROGRAM_PROPOSAL_SCHEMA:
            raise ValueError("unsupported recipe proposal schema")
        if not isinstance(self.scope, str) or not self.scope or "\0" in self.scope:
            raise ValueError("recipe proposal scope must be a non-empty opaque key")
        if not isinstance(self.context, str) or not self.context or "\0" in self.context:
            raise ValueError("recipe proposal context must be a non-empty opaque key")
        if self.factors is not None:
            self.factors.validate()
            if self.factors.operator_index != self.operator_index:
                raise ValueError("recipe proposal factors/operator disagree")
        if len(self.parent_digest) != 64:
            raise ValueError("recipe proposal parent digest is malformed")
        try:
            int(self.parent_digest, 16)
        except ValueError as error:
            raise ValueError("recipe proposal parent digest is malformed") from error
        if self.operator not in RECIPE_PROGRAM_MUTATION_OPERATORS:
            raise ValueError("recipe proposal operator is unknown")
        if not 0 <= self.operator_index < len(RECIPE_PROGRAM_MUTATION_OPERATORS):
            raise ValueError("recipe proposal operator index is invalid")
        if RECIPE_PROGRAM_MUTATION_OPERATORS[self.operator_index] != self.operator:
            raise ValueError("recipe proposal operator/index disagree")
        if self.attempt_id < 0:
            raise ValueError("recipe proposal attempt cannot be negative")
        if not math.isfinite(self.selection_probability) or not (
            0.0 < self.selection_probability <= 1.0
        ):
            raise ValueError("recipe proposal selection probability is invalid")
        if self.program.digest() == self.parent_digest:
            raise ValueError("recipe proposal must change its parent")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "program": self.program.payload(),
            "parent_digest": self.parent_digest,
            "operator": self.operator,
            "operator_index": self.operator_index,
            "attempt_id": self.attempt_id,
            "selection_probability": self.selection_probability,
            "scope": self.scope,
            "context": self.context,
            "factors": None if self.factors is None else self.factors.payload(),
        }


@dataclass(frozen=True)
class RecipeProgramCandidateFeedback:
    proposal: RecipeProgramCandidateProposal
    receipt: RecipeProgramAdmissionReceipt
    quality: float
    state: RecipeProgramSearchState


class OutcomeOnlyRecipeSequenceSearch:
    """Learn generic sequence-edit preferences from scalar outcomes only."""

    schema = RECIPE_PROGRAM_SEARCH_SCHEMA

    def __init__(
        self,
        basis: RecipeBasis,
        *,
        min_program_length: int = 1,
        max_program_length: int = 8,
        exploration: float = 0.5,
        temperature: float = 0.5,
        proposal_policy: (
            OpaqueContextRecipeProposalMemory
            | FactorizedOpaqueContextRecipeProposalMemory
            | None
        ) = None,
    ) -> None:
        if not isinstance(basis, RecipeBasis):
            raise TypeError("recipe search requires a RecipeBasis")
        if min_program_length < 1 or max_program_length < min_program_length:
            raise ValueError("recipe search length bounds are invalid")
        if not math.isfinite(exploration) or exploration < 0.0:
            raise ValueError("recipe search exploration is invalid")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("recipe search temperature is invalid")
        self.basis = basis
        self.min_program_length = int(min_program_length)
        self.max_program_length = int(max_program_length)
        self.exploration = float(exploration)
        self.temperature = float(temperature)
        if proposal_policy is not None and not isinstance(
            proposal_policy,
            (
                OpaqueContextRecipeProposalMemory,
                FactorizedOpaqueContextRecipeProposalMemory,
            ),
        ):
            raise TypeError("recipe proposal policy has the wrong type")
        self.proposal_policy = proposal_policy

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "basis": self.basis.configuration(),
            "min_program_length": self.min_program_length,
            "max_program_length": self.max_program_length,
            "exploration": self.exploration,
            "temperature": self.temperature,
            "operators": RECIPE_PROGRAM_MUTATION_OPERATORS,
            "updates": "scalar_verifier_aggregate_only_v1",
            "commit": "verifier_gated_external_recipe_file_v1",
            "candidate_history": "opaque_scope_local_v2",
            "proposal_credit": (
                None
                if self.proposal_policy is None
                else self.proposal_policy.configuration()
            ),
        }

    def initial_state(self) -> RecipeProgramSearchState:
        zeros = torch.zeros(len(RECIPE_PROGRAM_MUTATION_OPERATORS), dtype=torch.float64)
        return RecipeProgramSearchState(
            zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()
        ).validate()

    def _validate_parent(self, parent: RecipeProgram) -> None:
        if not isinstance(parent, RecipeProgram):
            raise TypeError("recipe search parent must be a RecipeProgram")
        if parent.slot_values != self.basis.slot_values or (
            parent.allow_parallel != self.basis.allow_parallel
        ):
            raise ValueError("recipe search parent basis is incompatible")
        if not self.min_program_length <= parent.program_length <= self.max_program_length:
            raise ValueError("recipe search parent length is outside bounds")

    def _available_mask(self, parent: RecipeProgram) -> torch.Tensor:
        length = parent.program_length
        return torch.tensor(
            [
                True,
                length < self.max_program_length,
                length > self.min_program_length,
                length > 1,
            ],
            dtype=torch.bool,
        )

    def _operator_probabilities(
        self,
        state: RecipeProgramSearchState,
        parent: RecipeProgram,
    ) -> torch.Tensor:
        state.validate()
        available = self._available_mask(parent)
        means = torch.where(
            state.reward_counts > 0.0,
            state.reward_totals / state.reward_counts.clamp_min(1.0),
            torch.zeros_like(state.reward_totals),
        )
        total = float(state.proposals + 1)
        bonus = self.exploration * torch.sqrt(
            torch.log(torch.tensor(total, dtype=means.dtype))
            / (state.reward_counts + 1.0)
        )
        return torch.softmax(
            (means + bonus).masked_fill(~available, float("-inf"))
            / self.temperature,
            dim=0,
        )

    def proposal_probabilities(
        self,
        state: RecipeProgramSearchState,
        parent: RecipeProgram,
    ) -> torch.Tensor:
        self._validate_parent(parent)
        return self._operator_probabilities(state, parent)

    def _candidate(
        self,
        parent: RecipeProgram,
        instructions: Sequence[RecipeInstruction],
    ) -> RecipeProgram:
        return RecipeProgram(
            parent.slot_values,
            tuple(instructions),
            allow_parallel=parent.allow_parallel,
        )

    @staticmethod
    def _proposal_factors(
        parent: RecipeProgram,
        operator_index: int,
        program: RecipeProgram,
    ) -> RecipeProgramProposalFactors:
        """Infer the generic edit factors for a valid one-step neighbor."""

        if operator_index == 0:
            differences = [
                index
                for index, (before, after) in enumerate(
                    zip(parent.instructions, program.instructions)
                )
                if before != after
            ]
            if len(parent.instructions) != len(program.instructions) or len(differences) != 1:
                raise ValueError("replacement candidate is not a one-position edit")
            position = differences[0]
            instructions = (program.instructions[position],)
            return RecipeProgramProposalFactors(
                operator_index,
                position,
                instruction_digests=tuple(_instruction_digest(value) for value in instructions),
            ).validate()
        if operator_index == 1:
            if len(program.instructions) != len(parent.instructions) + 1:
                raise ValueError("insertion candidate has the wrong length")
            for position in range(len(program.instructions)):
                reduced = program.instructions[:position] + program.instructions[position + 1 :]
                if reduced == parent.instructions:
                    return RecipeProgramProposalFactors(
                        operator_index,
                        position,
                        instruction_digests=(_instruction_digest(program.instructions[position]),),
                    ).validate()
            raise ValueError("insertion candidate does not contain its parent")
        if operator_index == 2:
            if len(program.instructions) != len(parent.instructions) - 1:
                raise ValueError("deletion candidate has the wrong length")
            for position in range(len(parent.instructions)):
                reduced = parent.instructions[:position] + parent.instructions[position + 1 :]
                if reduced == program.instructions:
                    return RecipeProgramProposalFactors(
                        operator_index,
                        position,
                        instruction_digests=(_instruction_digest(parent.instructions[position]),),
                    ).validate()
            raise ValueError("deletion candidate is not a one-position edit")
        if operator_index == 3:
            if len(parent.instructions) != len(program.instructions):
                raise ValueError("swap candidate has the wrong length")
            for first in range(len(parent.instructions)):
                for second in range(first + 1, len(parent.instructions)):
                    swapped = list(parent.instructions)
                    swapped[first], swapped[second] = swapped[second], swapped[first]
                    if tuple(swapped) == program.instructions:
                        return RecipeProgramProposalFactors(
                            operator_index,
                            first,
                            second,
                            (
                                _instruction_digest(parent.instructions[first]),
                                _instruction_digest(parent.instructions[second]),
                            ),
                        ).validate()
            raise ValueError("swap candidate is not a two-position edit")
        raise ValueError("recipe proposal operator is invalid")

    @classmethod
    def proposal_factors(
        cls,
        parent: RecipeProgram,
        operator_index: int,
        program: RecipeProgram,
    ) -> RecipeProgramProposalFactors:
        """Return the public generic descriptor for a one-step candidate."""

        return cls._proposal_factors(parent, operator_index, program)

    def exhaustive_candidates(
        self,
        parent: RecipeProgram,
    ) -> tuple[tuple[int, RecipeProgram], ...]:
        self._validate_parent(parent)
        atoms = self.basis.atomic_candidates()
        candidates: list[tuple[int, RecipeProgram]] = []
        seen: set[str] = set()

        def add(operator_index: int, instructions: Sequence[RecipeInstruction]) -> None:
            program = self._candidate(parent, instructions)
            digest = program.digest()
            if digest == parent.digest() or digest in seen:
                return
            seen.add(digest)
            candidates.append((operator_index, program))

        for position in range(parent.program_length):
            for atom in atoms:
                replaced = list(parent.instructions)
                replaced[position] = atom
                add(0, replaced)
        if parent.program_length < self.max_program_length:
            for position in range(parent.program_length + 1):
                for atom in atoms:
                    inserted = list(parent.instructions)
                    inserted.insert(position, atom)
                    add(1, inserted)
        if parent.program_length > self.min_program_length:
            for position in range(parent.program_length):
                add(
                    2,
                    parent.instructions[:position] + parent.instructions[position + 1 :],
                )
        if parent.program_length > 1:
            for first in range(parent.program_length):
                for second in range(first + 1, parent.program_length):
                    swapped = list(parent.instructions)
                    swapped[first], swapped[second] = swapped[second], swapped[first]
                    add(3, swapped)
        return tuple(candidates)

    def propose_exhaustive(
        self,
        state: RecipeProgramSearchState,
        parent: RecipeProgram,
        *,
        scope: str = "default",
        context: str = "default",
    ) -> RecipeProgramCandidateProposal:
        state.validate()
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("recipe proposal scope must be a non-empty opaque key")
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("recipe proposal context must be a non-empty opaque key")
        candidates = self.exhaustive_candidates(parent)
        probability = 1.0 / max(1, len(candidates))
        for operator_index, program in candidates:
            if _scoped_candidate_key(scope, program.digest()) in state.seen_candidate_digests:
                continue
            return RecipeProgramCandidateProposal(
                program,
                parent.digest(),
                RECIPE_PROGRAM_MUTATION_OPERATORS[operator_index],
                operator_index,
                state.proposals,
                probability,
                scope,
                context,
                self._proposal_factors(parent, operator_index, program),
            ).validate()
        raise RuntimeError("recipe search parent neighborhood is exhausted")

    def propose(
        self,
        state: RecipeProgramSearchState,
        parent: RecipeProgram,
        *,
        generator: torch.Generator,
        scope: str = "default",
        context: str = "default",
    ) -> RecipeProgramCandidateProposal:
        state.validate()
        self._validate_parent(parent)
        if not isinstance(scope, str) or not scope or "\0" in scope:
            raise ValueError("recipe proposal scope must be a non-empty opaque key")
        if not isinstance(context, str) or not context or "\0" in context:
            raise ValueError("recipe proposal context must be a non-empty opaque key")
        if self.proposal_policy is not None:
            available = tuple(
                (operator_index, program, self._proposal_factors(parent, operator_index, program))
                for operator_index, program in self.exhaustive_candidates(parent)
                if _scoped_candidate_key(scope, program.digest())
                not in state.seen_candidate_digests
            )
            if not available:
                raise RuntimeError("recipe search parent neighborhood is exhausted")
            selected, probability = self.proposal_policy.select(
                context,
                tuple(program.digest() for _, program, _ in available),
                factors=tuple(factors for _, _, factors in available),
                generator=generator,
            )
            operator_index, program, factors = available[selected]
            return RecipeProgramCandidateProposal(
                program,
                parent.digest(),
                RECIPE_PROGRAM_MUTATION_OPERATORS[operator_index],
                operator_index,
                state.proposals,
                probability,
                scope,
                context,
                factors,
            ).validate()
        excluded = torch.zeros(len(RECIPE_PROGRAM_MUTATION_OPERATORS), dtype=torch.bool)
        for _ in range(64):
            probabilities = self._operator_probabilities(state, parent).masked_fill(
                excluded, 0.0
            )
            if float(probabilities.sum()) <= 0.0:
                break
            probabilities = probabilities / probabilities.sum()
            operator_index = int(torch.multinomial(probabilities, 1, generator=generator))
            atoms = self.basis.atomic_candidates()
            instructions = list(parent.instructions)
            if operator_index == 0:
                position = int(torch.randint(parent.program_length, (), generator=generator))
                instructions[position] = atoms[
                    int(torch.randint(len(atoms), (), generator=generator))
                ]
            elif operator_index == 1:
                position = int(torch.randint(parent.program_length + 1, (), generator=generator))
                instructions.insert(
                    position,
                    atoms[int(torch.randint(len(atoms), (), generator=generator))],
                )
            elif operator_index == 2:
                position = int(torch.randint(parent.program_length, (), generator=generator))
                del instructions[position]
            else:
                first = int(torch.randint(parent.program_length, (), generator=generator))
                second = int(torch.randint(parent.program_length - 1, (), generator=generator))
                if second >= first:
                    second += 1
                instructions[first], instructions[second] = (
                    instructions[second],
                    instructions[first],
                )
            if not self.min_program_length <= len(instructions) <= self.max_program_length:
                continue
            program = self._candidate(parent, instructions)
            if program.digest() == parent.digest():
                continue
            if _scoped_candidate_key(scope, program.digest()) in state.seen_candidate_digests:
                continue
            factors = self._proposal_factors(parent, operator_index, program)
            return RecipeProgramCandidateProposal(
                program,
                parent.digest(),
                RECIPE_PROGRAM_MUTATION_OPERATORS[operator_index],
                operator_index,
                state.proposals,
                float(probabilities[operator_index]),
                scope,
                context,
                factors,
            ).validate()
        # Learned proposal priors are allowed to miss a useful edit, but a
        # finite generic neighborhood must not be confused with an
        # inexpressible target.  Exhaustive fallback preserves coverage while
        # leaving scalar outcomes as the only admission signal.
        return self.propose_exhaustive(
            state,
            parent,
            scope=scope,
            context=context,
        )

    def record_outcomes(
        self,
        state: RecipeProgramSearchState,
        proposal: RecipeProgramCandidateProposal,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> RecipeProgramCandidateFeedback:
        state.validate()
        proposal.validate()
        if proposal.attempt_id != state.proposals:
            raise ValueError("recipe proposal is out of sequence")
        digest = proposal.program.digest()
        scoped_digest = _scoped_candidate_key(
            proposal.scope,
            digest,
        )
        if scoped_digest in state.seen_candidate_digests:
            raise ValueError("recipe candidate was already evaluated")
        values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
        receipt = evaluate_recipe_program_admission(
            proposal.program,
            values,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        quality = float(values.mean().item()) if values.numel() else 0.0
        if self.proposal_policy is not None:
            if proposal.factors is None:
                raise ValueError("recipe proposal policy requires proposal factors")
            self.proposal_policy.record(
                proposal.context,
                digest,
                quality,
                factors=proposal.factors,
            )
        totals = state.reward_totals.clone()
        counts = state.reward_counts.clone()
        accepted_counts = state.accepted_counts.clone()
        failed_counts = state.failed_counts.clone()
        index = proposal.operator_index
        totals[index] += quality
        counts[index] += 1.0
        if receipt.accepted:
            accepted_counts[index] += 1.0
        else:
            failed_counts[index] += 1.0
        next_state = RecipeProgramSearchState(
            totals,
            counts,
            accepted_counts,
            failed_counts,
            (*state.seen_candidate_digests, scoped_digest),
            state.proposals + 1,
            state.accepted + int(receipt.accepted),
            max(state.best_quality, quality),
        ).validate()
        return RecipeProgramCandidateFeedback(
            proposal, receipt, quality, next_state
        )


__all__ = [
    "RECIPE_CONTEXT_PROPOSAL_MEMORY_SCHEMA",
    "RECIPE_FACTORIZED_CONTEXT_PROPOSAL_MEMORY_SCHEMA",
    "RECIPE_PROGRAM_ADMISSION_SCHEMA",
    "RECIPE_PROGRAM_MEMORY_SCHEMA",
    "RECIPE_PROGRAM_MUTATION_OPERATORS",
    "RECIPE_PROGRAM_PROPOSAL_SCHEMA",
    "RECIPE_PROGRAM_SCHEMA",
    "RECIPE_PROGRAM_SEARCH_SCHEMA",
    "RECIPE_PROPOSAL_FACTORS_SCHEMA",
    "ExternalRecipeProgramMemory",
    "FactorizedOpaqueContextRecipeProposalMemory",
    "OpaqueContextRecipeProposalMemory",
    "OutcomeOnlyRecipeSequenceSearch",
    "RecipeProgram",
    "RecipeProgramAdmissionReceipt",
    "RecipeProgramCandidateFeedback",
    "RecipeProgramCandidateProposal",
    "RecipeProgramProposalFactors",
    "RecipeProgramSearchState",
    "evaluate_recipe_program_admission",
]
