"""Outcome-driven proposal search for opaque external programs.

This module owns candidate generation outside the controller.  It can edit an
opaque instruction sequence with generic structural operations, but it never
interprets an instruction vector, observes a verifier target, or stores raw
verifier rows.  The caller remains responsible for executing a proposal and
committing it through an independent retention transaction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .program import (
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
    evaluate_external_program_admission,
)

EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA = (
    "neural-computer.external-program-candidate-search.v1"
)
EXTERNAL_PROGRAM_CANDIDATE_PROPOSAL_SCHEMA = (
    "neural-computer.external-program-candidate-proposal.v1"
)
PROGRAM_MUTATION_OPERATORS = (
    "replace",
    "insert",
    "delete",
    "swap",
    "jitter",
)


def _digest_tensor(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    detached = value.detach().cpu().contiguous()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(repr(tuple(detached.shape)).encode("utf-8"))
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalProgramCandidateSearchState:
    """Replay-free aggregate state for one sequential proposal search."""

    reward_totals: torch.Tensor
    reward_counts: torch.Tensor
    accepted_counts: torch.Tensor
    failed_counts: torch.Tensor
    seen_candidate_digests: tuple[str, ...] = ()
    proposals: int = 0
    accepted: int = 0
    best_quality: float = 0.0
    schema: str = EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA

    def validate(
        self,
        *,
        operator_count: int = len(PROGRAM_MUTATION_OPERATORS),
    ) -> ExternalProgramCandidateSearchState:
        if self.schema != EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA:
            raise ValueError("unsupported external program search schema")
        if operator_count < 1:
            raise ValueError("external program search needs one operator")
        expected = (operator_count,)
        for name, value in (
            ("reward_totals", self.reward_totals),
            ("reward_counts", self.reward_counts),
            ("accepted_counts", self.accepted_counts),
            ("failed_counts", self.failed_counts),
        ):
            if value.shape != expected:
                raise ValueError(f"external program search {name} has the wrong shape")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"external program search {name} is invalid")
            if bool(torch.any(value < 0.0)):
                raise ValueError(f"external program search {name} cannot be negative")
        if self.proposals < 0 or self.accepted < 0 or self.accepted > self.proposals:
            raise ValueError("external program search counters are invalid")
        if int(self.reward_counts.sum().item()) != self.proposals:
            raise ValueError("external program search counts do not match proposals")
        if int(self.accepted_counts.sum().item()) != self.accepted:
            raise ValueError("external program search accepts do not match counter")
        if int(self.failed_counts.sum().item()) != self.proposals - self.accepted:
            raise ValueError("external program search failures do not match counter")
        if len(self.seen_candidate_digests) != self.proposals:
            raise ValueError("external program search seen candidates do not match counter")
        if len(set(self.seen_candidate_digests)) != len(self.seen_candidate_digests):
            raise ValueError("external program search seen candidates are not unique")
        for digest in self.seen_candidate_digests:
            if len(digest) != 64:
                raise ValueError("external program search candidate digest is malformed")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(
                    "external program search candidate digest is malformed"
                ) from error
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("external program search best quality is invalid")
        return self

    def payload(self) -> dict[str, Any]:
        """Return tensor-only state; raw candidates and outcomes are excluded."""

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
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        operator_count: int = len(PROGRAM_MUTATION_OPERATORS),
    ) -> ExternalProgramCandidateSearchState:
        if not isinstance(payload, dict):
            raise TypeError("external program search payload must be a dictionary")
        required = (
            "reward_totals",
            "reward_counts",
            "accepted_counts",
            "failed_counts",
        )
        if any(not isinstance(payload.get(name), torch.Tensor) for name in required):
            raise TypeError("external program search payload is missing tensors")
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
        ).validate(operator_count=operator_count)


@dataclass(frozen=True)
class ExternalProgramCandidateProposal:
    """One copy-on-write opaque candidate produced from a parent artifact."""

    artifact: ExternalProgramArtifact
    parent_digest: str
    operator: str
    operator_index: int
    attempt_id: int
    selection_probability: float
    schema: str = EXTERNAL_PROGRAM_CANDIDATE_PROPOSAL_SCHEMA

    def validate(self) -> ExternalProgramCandidateProposal:
        if self.schema != EXTERNAL_PROGRAM_CANDIDATE_PROPOSAL_SCHEMA:
            raise ValueError("unsupported external program candidate schema")
        if len(self.parent_digest) != 64:
            raise ValueError("external program candidate parent digest is malformed")
        try:
            int(self.parent_digest, 16)
        except ValueError as error:
            raise ValueError(
                "external program candidate parent digest is malformed"
            ) from error
        if self.operator not in PROGRAM_MUTATION_OPERATORS:
            raise ValueError("external program candidate operator is unknown")
        if not 0 <= self.operator_index < len(PROGRAM_MUTATION_OPERATORS):
            raise ValueError("external program candidate operator index is invalid")
        if PROGRAM_MUTATION_OPERATORS[self.operator_index] != self.operator:
            raise ValueError("external program candidate operator/index disagree")
        if self.attempt_id < 0:
            raise ValueError("external program candidate attempt cannot be negative")
        if not math.isfinite(self.selection_probability) or not (
            0.0 < self.selection_probability <= 1.0
        ):
            raise ValueError("external program candidate selection probability is invalid")
        if self.artifact.digest() == self.parent_digest:
            raise ValueError("external program candidate must change its parent")
        return self

    def payload(self) -> dict[str, Any]:
        """Return an auditable proposal payload with opaque candidate data."""

        self.validate()
        return {
            "schema": self.schema,
            "artifact": self.artifact.payload(),
            "parent_digest": self.parent_digest,
            "operator": self.operator,
            "operator_index": self.operator_index,
            "attempt_id": self.attempt_id,
            "selection_probability": self.selection_probability,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalProgramCandidateProposal:
        if not isinstance(payload, dict):
            raise TypeError("external program candidate payload must be a dictionary")
        artifact = payload.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("external program candidate artifact is missing")
        return cls(
            artifact=ExternalProgramArtifact.from_payload(artifact),
            parent_digest=payload.get("parent_digest"),
            operator=payload.get("operator"),
            operator_index=int(payload.get("operator_index", -1)),
            attempt_id=int(payload.get("attempt_id", -1)),
            selection_probability=float(payload.get("selection_probability", float("nan"))),
            schema=payload.get("schema"),
        ).validate()


@dataclass(frozen=True)
class ExternalProgramCandidateFeedback:
    """Outcome summary returned after one proposal is externally verified."""

    proposal: ExternalProgramCandidateProposal
    receipt: ExternalProgramAdmissionReceipt
    quality: float
    state: ExternalProgramCandidateSearchState

    def validate(self) -> ExternalProgramCandidateFeedback:
        self.proposal.validate()
        self.receipt.validate()
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("external program candidate quality is invalid")
        self.state.validate()
        return self


class ExternalProgramCandidateSearch:
    """Learn which generic program edits produce verified behavior.

    The search uses only scalar verifier outcomes to update operator
    statistics.  It is deliberately sequential and copy-on-write: an
    accepted proposal becomes the caller's next parent only after the caller
    commits the same artifact through external program memory.  Instruction
    vectors are never decoded into operation names or semantic fields.
    """

    schema = EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA

    def __init__(
        self,
        instruction_width: int,
        *,
        instruction_bank: torch.Tensor | None = None,
        min_program_length: int = 1,
        max_program_length: int = 8,
        mutation_scale: float = 0.05,
        exploration: float = 0.5,
        temperature: float = 0.5,
        proposal_retry_limit: int = 64,
    ) -> None:
        if instruction_width < 1:
            raise ValueError("external program search instruction width must be positive")
        if min_program_length < 1 or max_program_length < min_program_length:
            raise ValueError("external program search length bounds are invalid")
        if not math.isfinite(mutation_scale) or mutation_scale <= 0.0:
            raise ValueError("external program search mutation scale is invalid")
        if not math.isfinite(exploration) or exploration < 0.0:
            raise ValueError("external program search exploration is invalid")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("external program search temperature is invalid")
        if proposal_retry_limit < 1:
            raise ValueError("external program search proposal retry limit is invalid")
        if instruction_bank is not None:
            if (
                instruction_bank.ndim != 2
                or instruction_bank.shape[1] != instruction_width
                or instruction_bank.shape[0] < 1
            ):
                raise ValueError("external program instruction bank has the wrong shape")
            if not instruction_bank.is_floating_point() or not bool(
                torch.isfinite(instruction_bank).all()
            ):
                raise ValueError("external program instruction bank is invalid")
            instruction_bank = instruction_bank.detach().cpu().clone()
        self.instruction_width = int(instruction_width)
        self.instruction_bank = instruction_bank
        self.min_program_length = int(min_program_length)
        self.max_program_length = int(max_program_length)
        self.mutation_scale = float(mutation_scale)
        self.exploration = float(exploration)
        self.temperature = float(temperature)
        self.proposal_retry_limit = int(proposal_retry_limit)

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instruction_width": self.instruction_width,
            "instruction_bank_count": (
                0 if self.instruction_bank is None else int(self.instruction_bank.shape[0])
            ),
            "min_program_length": self.min_program_length,
            "max_program_length": self.max_program_length,
            "mutation_scale": self.mutation_scale,
            "exploration": self.exploration,
            "temperature": self.temperature,
            "proposal_retry_limit": self.proposal_retry_limit,
            "operators": PROGRAM_MUTATION_OPERATORS,
            "updates": "scalar_verifier_aggregate_only_v1",
            "commit": "caller_owned_copy_on_write_admission_v1",
        }

    def initial_state(self) -> ExternalProgramCandidateSearchState:
        zeros = torch.zeros(len(PROGRAM_MUTATION_OPERATORS), dtype=torch.float64)
        return ExternalProgramCandidateSearchState(
            reward_totals=zeros.clone(),
            reward_counts=zeros.clone(),
            accepted_counts=zeros.clone(),
            failed_counts=zeros.clone(),
        ).validate()

    def _available_mask(self, parent: ExternalProgramArtifact) -> torch.Tensor:
        length = parent.program_length
        return torch.tensor(
            [
                True,
                length < self.max_program_length,
                length > self.min_program_length,
                length > 1,
                True,
            ],
            dtype=torch.bool,
        )

    def _operator_probabilities(
        self,
        state: ExternalProgramCandidateSearchState,
        parent: ExternalProgramArtifact,
    ) -> torch.Tensor:
        state.validate()
        available = self._available_mask(parent)
        counts = state.reward_counts
        means = torch.where(
            counts > 0.0,
            state.reward_totals / counts.clamp_min(1.0),
            torch.zeros_like(counts),
        )
        total = float(state.proposals + 1)
        bonus = self.exploration * torch.sqrt(
            torch.log(torch.tensor(total, dtype=means.dtype)) / (counts + 1.0)
        )
        logits = (means + bonus).masked_fill(~available, float("-inf"))
        return torch.softmax(logits / self.temperature, dim=0)

    def proposal_probabilities(
        self,
        state: ExternalProgramCandidateSearchState,
        parent: ExternalProgramArtifact,
    ) -> torch.Tensor:
        """Return the opaque structural-edit distribution without mutation."""

        self._validate_parent(parent)
        return self._operator_probabilities(state, parent)

    @staticmethod
    def _random_index(limit: int, generator: torch.Generator) -> int:
        return int(torch.randint(limit, (), generator=generator).item())

    def _sample_atom(
        self,
        parent: ExternalProgramArtifact,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if self.instruction_bank is not None:
            index = self._random_index(int(self.instruction_bank.shape[0]), generator)
            return self.instruction_bank[index].to(
                device=parent.codes.device,
                dtype=parent.codes.dtype,
            )
        source = parent.codes[self._random_index(parent.program_length, generator)]
        noise = torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
        ).to(device=source.device)
        return source + self.mutation_scale * noise

    def _mutate(
        self,
        parent: ExternalProgramArtifact,
        operator_index: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        codes = parent.codes.detach().clone()
        length = parent.program_length
        if operator_index == 0:  # replace
            codes[self._random_index(length, generator)] = self._sample_atom(
                parent, generator
            )
        elif operator_index == 1:  # insert
            position = self._random_index(length + 1, generator)
            atom = self._sample_atom(parent, generator).unsqueeze(0)
            codes = torch.cat((codes[:position], atom, codes[position:]), dim=0)
        elif operator_index == 2:  # delete
            position = self._random_index(length, generator)
            codes = torch.cat((codes[:position], codes[position + 1 :]), dim=0)
        elif operator_index == 3:  # swap
            first = self._random_index(length, generator)
            second = self._random_index(length - 1, generator)
            if second >= first:
                second += 1
            codes[[first, second]] = codes[[second, first]]
        elif operator_index == 4:  # jitter
            position = self._random_index(length, generator)
            noise = torch.randn(
                codes[position].shape,
                generator=generator,
                dtype=codes.dtype,
            ).to(device=codes.device)
            codes[position] = codes[position] + self.mutation_scale * noise
        else:
            raise ValueError("external program search operator index is invalid")
        return codes

    def _validate_parent(self, parent: ExternalProgramArtifact) -> None:
        if not isinstance(parent, ExternalProgramArtifact):
            raise TypeError("external program search parent must be an artifact")
        if parent.instruction_width != self.instruction_width:
            raise ValueError("external program search parent width is incompatible")
        if not parent.codes.is_floating_point():
            raise ValueError("external program search requires floating code tensors")
        if not self.min_program_length <= parent.program_length <= self.max_program_length:
            raise ValueError("external program search parent length is outside bounds")

    def exhaustive_candidates(
        self,
        parent: ExternalProgramArtifact,
    ) -> tuple[tuple[int, torch.Tensor], ...]:
        """Enumerate finite opaque edits from an external instruction bank.

        This is a structural enumeration only.  It never assigns meaning to
        an instruction row; the bank is simply a replaceable collection of
        learned vectors.  Continuous jitter is intentionally excluded because
        it has no finite exhaustive neighborhood.
        """

        self._validate_parent(parent)
        if self.instruction_bank is None:
            raise ValueError("exhaustive program search requires an instruction bank")
        candidates: list[tuple[int, torch.Tensor]] = []
        seen: set[str] = set()

        def add(operator_index: int, codes: torch.Tensor) -> None:
            artifact = ExternalProgramArtifact(
                codes=codes,
                interpreter_schema=parent.interpreter_schema,
                execution_schema=parent.execution_schema,
                output_schema=parent.output_schema,
            )
            digest = artifact.digest()
            if digest == parent.digest() or digest in seen:
                return
            seen.add(digest)
            candidates.append((operator_index, codes.detach().clone()))

        bank = self.instruction_bank.to(
            device=parent.codes.device,
            dtype=parent.codes.dtype,
        )
        if len(bank):
            for position in range(parent.program_length):
                for atom in bank:
                    codes = parent.codes.detach().clone()
                    codes[position] = atom
                    add(0, codes)
        if parent.program_length < self.max_program_length:
            for position in range(parent.program_length + 1):
                for atom in bank:
                    add(
                        1,
                        torch.cat(
                            (
                                parent.codes[:position],
                                atom.unsqueeze(0),
                                parent.codes[position:],
                            ),
                            dim=0,
                        ),
                    )
        if parent.program_length > self.min_program_length:
            for position in range(parent.program_length):
                add(
                    2,
                    torch.cat(
                        (parent.codes[:position], parent.codes[position + 1 :]),
                        dim=0,
                    ),
                )
        if parent.program_length > 1:
            for first in range(parent.program_length):
                for second in range(first + 1, parent.program_length):
                    codes = parent.codes.detach().clone()
                    codes[[first, second]] = codes[[second, first]]
                    add(3, codes)
        return tuple(candidates)

    def propose_exhaustive(
        self,
        state: ExternalProgramCandidateSearchState,
        parent: ExternalProgramArtifact,
    ) -> ExternalProgramCandidateProposal:
        """Return the first unseen finite edit for a parent hypothesis."""

        state.validate()
        self._validate_parent(parent)
        parent_digest = parent.digest()
        candidates = self.exhaustive_candidates(parent)
        selection_probability = 1.0 / max(1, len(candidates))
        for operator_index, codes in candidates:
            artifact = ExternalProgramArtifact(
                codes=codes,
                interpreter_schema=parent.interpreter_schema,
                execution_schema=parent.execution_schema,
                output_schema=parent.output_schema,
            )
            if artifact.digest() in state.seen_candidate_digests:
                continue
            return ExternalProgramCandidateProposal(
                artifact=artifact,
                parent_digest=parent_digest,
                operator=PROGRAM_MUTATION_OPERATORS[operator_index],
                operator_index=operator_index,
                attempt_id=state.proposals,
                selection_probability=selection_probability,
            ).validate()
        raise RuntimeError("external program search parent neighborhood is exhausted")

    def propose(
        self,
        state: ExternalProgramCandidateSearchState,
        parent: ExternalProgramArtifact,
        *,
        generator: torch.Generator,
    ) -> ExternalProgramCandidateProposal:
        """Create one copy-on-write proposal from scalar-informed operator priors."""

        self._validate_parent(parent)
        parent_digest = parent.digest()
        excluded = torch.zeros(len(PROGRAM_MUTATION_OPERATORS), dtype=torch.bool)
        duplicate_counts = torch.zeros(len(PROGRAM_MUTATION_OPERATORS), dtype=torch.int64)
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
            codes = self._mutate(parent, operator_index, generator)
            if codes.shape[0] > self.max_program_length:
                raise RuntimeError(
                    "external program search produced an oversized candidate"
                )
            if codes.shape[0] < self.min_program_length:
                raise RuntimeError("external program search produced an empty candidate")
            if torch.equal(codes, parent.codes):
                position = self._random_index(parent.program_length, generator)
                noise = torch.randn(
                    codes[position].shape,
                    generator=generator,
                    dtype=codes.dtype,
                ).to(device=codes.device)
                codes[position] = codes[position] + self.mutation_scale * noise
            artifact = ExternalProgramArtifact(
                codes=codes,
                interpreter_schema=parent.interpreter_schema,
                execution_schema=parent.execution_schema,
                output_schema=parent.output_schema,
            )
            if artifact.digest() in state.seen_candidate_digests:
                duplicate_counts[operator_index] += 1
                if duplicate_counts[operator_index] >= 8:
                    excluded[operator_index] = True
                continue
            proposal = ExternalProgramCandidateProposal(
                artifact=artifact,
                parent_digest=parent_digest,
                operator=PROGRAM_MUTATION_OPERATORS[operator_index],
                operator_index=operator_index,
                attempt_id=state.proposals,
                selection_probability=float(probabilities[operator_index].item()),
            )
            return proposal.validate()
        raise RuntimeError("external program search proposal neighborhood is exhausted")

    def record_outcomes(
        self,
        state: ExternalProgramCandidateSearchState,
        proposal: ExternalProgramCandidateProposal,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ExternalProgramCandidateFeedback:
        """Update only aggregate search statistics from one scalar outcome stream."""

        state.validate()
        proposal.validate()
        if proposal.attempt_id != state.proposals:
            raise ValueError("external program search proposal is out of sequence")
        candidate_digest = proposal.artifact.digest()
        if candidate_digest in state.seen_candidate_digests:
            raise ValueError("external program search candidate was already evaluated")
        values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
        receipt = evaluate_external_program_admission(
            proposal.artifact,
            values,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        quality = float(values.mean().item()) if values.numel() else 0.0
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
        updated = ExternalProgramCandidateSearchState(
            reward_totals=totals,
            reward_counts=counts,
            accepted_counts=accepted_counts,
            failed_counts=failed_counts,
            seen_candidate_digests=(*state.seen_candidate_digests, candidate_digest),
            proposals=state.proposals + 1,
            accepted=state.accepted + int(receipt.accepted),
            best_quality=max(state.best_quality, quality),
        ).validate()
        return ExternalProgramCandidateFeedback(
            proposal=proposal,
            receipt=receipt,
            quality=quality,
            state=updated,
        ).validate()


__all__ = [
    "EXTERNAL_PROGRAM_CANDIDATE_PROPOSAL_SCHEMA",
    "EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA",
    "PROGRAM_MUTATION_OPERATORS",
    "ExternalProgramCandidateFeedback",
    "ExternalProgramCandidateProposal",
    "ExternalProgramCandidateSearch",
    "ExternalProgramCandidateSearchState",
]
