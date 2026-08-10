"""Persistent, copy-on-write hypothesis frontiers for external programs.

The frontier is a memory-side search structure layered over
``ExternalProgramCandidateSearch``.  It retains only opaque candidate files,
their parent relationships, depths, and scalar quality summaries.  Protected
files remain outside the frontier and can only be changed by an independent
verifier-gated admission transaction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .program import ExternalProgramArtifact
from .program_search import (
    EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA,
    ExternalProgramCandidateFeedback,
    ExternalProgramCandidateProposal,
    ExternalProgramCandidateSearch,
    ExternalProgramCandidateSearchState,
)

EXTERNAL_PROGRAM_HYPOTHESIS_FRONTIER_SCHEMA = (
    "neural-computer.external-program-hypothesis-frontier.v1"
)
EXTERNAL_PROGRAM_HYPOTHESIS_SCHEMA = (
    "neural-computer.external-program-hypothesis.v1"
)


def _digest_payload(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            detached = item.detach().cpu().contiguous()
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalProgramHypothesis:
    """One provisional opaque program retained by the external frontier."""

    artifact: ExternalProgramArtifact
    parent_digest: str | None
    depth: int
    quality: float
    schema: str = EXTERNAL_PROGRAM_HYPOTHESIS_SCHEMA

    def validate(self) -> ExternalProgramHypothesis:
        if self.schema != EXTERNAL_PROGRAM_HYPOTHESIS_SCHEMA:
            raise ValueError("unsupported external program hypothesis schema")
        if self.parent_digest is not None:
            if len(self.parent_digest) != 64:
                raise ValueError("external program hypothesis parent digest is malformed")
            try:
                int(self.parent_digest, 16)
            except ValueError as error:
                raise ValueError(
                    "external program hypothesis parent digest is malformed"
                ) from error
        if self.depth < 0:
            raise ValueError("external program hypothesis depth cannot be negative")
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("external program hypothesis quality is invalid")
        if self.parent_digest == self.artifact.digest():
            raise ValueError("external program hypothesis cannot parent itself")
        return self

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "artifact": self.artifact.payload(),
            "parent_digest": self.parent_digest,
            "depth": self.depth,
            "quality": self.quality,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalProgramHypothesis:
        if not isinstance(payload, dict):
            raise TypeError("external program hypothesis payload must be a dictionary")
        artifact = payload.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("external program hypothesis artifact is missing")
        return cls(
            artifact=ExternalProgramArtifact.from_payload(artifact),
            parent_digest=payload.get("parent_digest"),
            depth=int(payload.get("depth", -1)),
            quality=float(payload.get("quality", float("nan"))),
            schema=payload.get("schema"),
        ).validate()


@dataclass(frozen=True)
class ExternalProgramHypothesisFrontierState:
    """Persistent frontier contents plus its replay-free search statistics."""

    hypotheses: tuple[ExternalProgramHypothesis, ...]
    search_state: ExternalProgramCandidateSearchState
    root_digest: str
    evaluations: int = 0
    best_quality: float = 0.0
    schema: str = EXTERNAL_PROGRAM_HYPOTHESIS_FRONTIER_SCHEMA

    def validate(self) -> ExternalProgramHypothesisFrontierState:
        if self.schema != EXTERNAL_PROGRAM_HYPOTHESIS_FRONTIER_SCHEMA:
            raise ValueError("unsupported external program frontier schema")
        if not self.hypotheses:
            raise ValueError("external program frontier cannot be empty")
        if len(self.root_digest) != 64:
            raise ValueError("external program frontier root digest is malformed")
        try:
            int(self.root_digest, 16)
        except ValueError as error:
            raise ValueError(
                "external program frontier root digest is malformed"
            ) from error
        digests = []
        for hypothesis in self.hypotheses:
            hypothesis.validate()
            digests.append(hypothesis.artifact.digest())
        if len(set(digests)) != len(digests):
            raise ValueError("external program frontier contains duplicate hypotheses")
        if self.root_digest not in digests:
            raise ValueError("external program frontier lost its protected root")
        if self.evaluations < 0:
            raise ValueError("external program frontier evaluations cannot be negative")
        if self.search_state.proposals != self.evaluations:
            raise ValueError("external program frontier evaluations do not match search")
        if not math.isfinite(self.best_quality) or not 0.0 <= self.best_quality <= 1.0:
            raise ValueError("external program frontier best quality is invalid")
        return self

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "hypotheses": [hypothesis.payload() for hypothesis in self.hypotheses],
            "search_state": self.search_state.payload(),
            "root_digest": self.root_digest,
            "evaluations": self.evaluations,
            "best_quality": self.best_quality,
        }

    def digest(self) -> str:
        """Return a stable checksum over the frontier, including tensor files."""

        return _digest_payload(self.payload())

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> ExternalProgramHypothesisFrontierState:
        if not isinstance(payload, dict):
            raise TypeError("external program frontier payload must be a dictionary")
        hypotheses = payload.get("hypotheses")
        search_payload = payload.get("search_state")
        if not isinstance(hypotheses, list) or not isinstance(search_payload, dict):
            raise TypeError("external program frontier payload is incomplete")
        return cls(
            hypotheses=tuple(
                ExternalProgramHypothesis.from_payload(item) for item in hypotheses
            ),
            search_state=ExternalProgramCandidateSearchState.from_payload(
                search_payload
            ),
            root_digest=payload.get("root_digest"),
            evaluations=int(payload.get("evaluations", -1)),
            best_quality=float(payload.get("best_quality", float("nan"))),
            schema=payload.get("schema"),
        ).validate()


class ExternalProgramHypothesisFrontier:
    """Keep a bounded set of provisional multi-step program hypotheses.

    The frontier never commits a candidate.  A caller evaluates proposals
    using an independent verifier, calls :meth:`record_outcomes`, and then
    admits only the verified winner through external program memory.  The
    protected root is retained in every state so a failed multi-step search
    cannot erase the source capability.
    """

    schema = EXTERNAL_PROGRAM_HYPOTHESIS_FRONTIER_SCHEMA

    def __init__(
        self,
        search: ExternalProgramCandidateSearch,
        *,
        beam_width: int = 4,
        max_depth: int = 8,
        minimum_quality: float = 0.0,
        parent_temperature: float = 0.25,
        proposal_mode: str = "exhaustive",
    ) -> None:
        if not isinstance(search, ExternalProgramCandidateSearch):
            raise TypeError("external program frontier requires a candidate search")
        if beam_width < 2:
            raise ValueError("external program frontier beam width must be at least two")
        if max_depth < 1:
            raise ValueError("external program frontier max depth must be positive")
        if not math.isfinite(minimum_quality) or not 0.0 <= minimum_quality <= 1.0:
            raise ValueError("external program frontier minimum quality is invalid")
        if not math.isfinite(parent_temperature) or parent_temperature <= 0.0:
            raise ValueError("external program frontier parent temperature is invalid")
        if proposal_mode not in {"exhaustive", "stochastic"}:
            raise ValueError("external program frontier proposal mode is invalid")
        if proposal_mode == "exhaustive" and search.instruction_bank is None:
            raise ValueError("exhaustive external program frontier needs an instruction bank")
        self.search = search
        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.minimum_quality = float(minimum_quality)
        self.parent_temperature = float(parent_temperature)
        self.proposal_mode = proposal_mode

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "search_schema": EXTERNAL_PROGRAM_CANDIDATE_SEARCH_SCHEMA,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "minimum_quality": self.minimum_quality,
            "parent_temperature": self.parent_temperature,
            "proposal_mode": self.proposal_mode,
            "retention": "protected_root_plus_top_quality_copy_on_write_v1",
            "storage": "opaque_candidate_tensors_and_scalar_summaries_v1",
        }

    def initial_state(
        self,
        root: ExternalProgramArtifact,
        *,
        root_quality: float = 0.0,
    ) -> ExternalProgramHypothesisFrontierState:
        if not isinstance(root, ExternalProgramArtifact):
            raise TypeError("external program frontier root must be an artifact")
        if not math.isfinite(root_quality) or not 0.0 <= root_quality <= 1.0:
            raise ValueError("external program frontier root quality is invalid")
        hypothesis = ExternalProgramHypothesis(
            artifact=root,
            parent_digest=None,
            depth=0,
            quality=float(root_quality),
        ).validate()
        return ExternalProgramHypothesisFrontierState(
            hypotheses=(hypothesis,),
            search_state=self.search.initial_state(),
            root_digest=root.digest(),
            best_quality=float(root_quality),
        ).validate()

    def _eligible_indices(
        self,
        state: ExternalProgramHypothesisFrontierState,
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, hypothesis in enumerate(state.hypotheses)
            if hypothesis.depth < self.max_depth
            and hypothesis.artifact.program_length < self.search.max_program_length
        ) or tuple(
            index
            for index, hypothesis in enumerate(state.hypotheses)
            if hypothesis.depth < self.max_depth
        )

    def _select_parent(
        self,
        state: ExternalProgramHypothesisFrontierState,
        *,
        generator: torch.Generator,
    ) -> ExternalProgramHypothesis:
        eligible = self._eligible_indices(state)
        if not eligible:
            raise RuntimeError("external program hypothesis frontier is exhausted")
        qualities = torch.tensor(
            [state.hypotheses[index].quality for index in eligible],
            dtype=torch.float64,
        )
        probabilities = torch.softmax(
            qualities / self.parent_temperature,
            dim=0,
        )
        selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
        return state.hypotheses[eligible[selected]]

    def propose(
        self,
        state: ExternalProgramHypothesisFrontierState,
        *,
        generator: torch.Generator,
    ) -> ExternalProgramCandidateProposal:
        """Select one retained hypothesis and produce its next opaque edit."""

        state.validate()
        if self.proposal_mode == "exhaustive":
            eligible = sorted(
                (
                    hypothesis
                    for hypothesis in state.hypotheses
                    if hypothesis.depth < self.max_depth
                ),
                key=lambda hypothesis: (
                    hypothesis.depth,
                    -hypothesis.quality,
                    hypothesis.artifact.digest(),
                ),
            )
            for parent in eligible:
                try:
                    return self.search.propose_exhaustive(
                        state.search_state,
                        parent.artifact,
                    )
                except RuntimeError:
                    continue
            raise RuntimeError("external program hypothesis frontier is exhausted")
        parent = self._select_parent(state, generator=generator)
        return self.search.propose(state.search_state, parent.artifact, generator=generator)

    def _prune(
        self,
        hypotheses: Sequence[ExternalProgramHypothesis],
        *,
        root_digest: str,
    ) -> tuple[ExternalProgramHypothesis, ...]:
        root = next(item for item in hypotheses if item.artifact.digest() == root_digest)
        others = [item for item in hypotheses if item.artifact.digest() != root_digest]
        others.sort(key=lambda item: (-item.quality, item.depth, item.artifact.digest()))
        return (root, *others[: self.beam_width - 1])

    def record_outcomes(
        self,
        state: ExternalProgramHypothesisFrontierState,
        proposal: ExternalProgramCandidateProposal,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> tuple[
        ExternalProgramHypothesisFrontierState,
        ExternalProgramCandidateFeedback,
    ]:
        """Score one proposal and retain it only as provisional frontier state."""

        state.validate()
        if not any(
            hypothesis.artifact.digest() == proposal.parent_digest
            for hypothesis in state.hypotheses
        ):
            raise ValueError("external program frontier proposal parent is not retained")
        feedback = self.search.record_outcomes(
            state.search_state,
            proposal,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        parent = next(
            hypothesis
            for hypothesis in state.hypotheses
            if hypothesis.artifact.digest() == proposal.parent_digest
        )
        hypotheses = list(state.hypotheses)
        if feedback.quality >= self.minimum_quality:
            hypotheses.append(
                ExternalProgramHypothesis(
                    artifact=proposal.artifact,
                    parent_digest=proposal.parent_digest,
                    depth=parent.depth + 1,
                    quality=feedback.quality,
                ).validate()
            )
        next_state = ExternalProgramHypothesisFrontierState(
            hypotheses=self._prune(hypotheses, root_digest=state.root_digest),
            search_state=feedback.state,
            root_digest=state.root_digest,
            evaluations=state.evaluations + 1,
            best_quality=max(state.best_quality, feedback.quality),
        ).validate()
        return next_state, feedback


__all__ = [
    "EXTERNAL_PROGRAM_HYPOTHESIS_FRONTIER_SCHEMA",
    "EXTERNAL_PROGRAM_HYPOTHESIS_SCHEMA",
    "ExternalProgramHypothesis",
    "ExternalProgramHypothesisFrontier",
    "ExternalProgramHypothesisFrontierState",
]
