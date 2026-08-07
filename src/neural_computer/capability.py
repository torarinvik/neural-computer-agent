"""Replaceable memory-side capability programs.

The shared controller remains frozen while a capability owns its recurrent
external state and its learned intention residual.  Output decoding stays
outside this module so a capability can be connected to any compatible
decoder on the intention bus.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .addressing import (
    FactorizedOpaqueAddressRouter,
    PersistentOpaqueContextRouteEvidence,
    PersistentOpaqueRouteEvidence,
    failure_gated_candidate_scores,
)
from .episodic import EpisodicContextEncoder, EpisodicIntentAdapter
from .interface import IntentEvent

EXTERNAL_CAPABILITY_SCHEMA = "neural-computer.external-capability.v1"
EXTERNAL_CAPABILITY_PIPELINE_SCHEMA = "neural-computer.external-capability-pipeline.v1"
EXTERNAL_CAPABILITY_COMPOSITION_SCHEMA = (
    "neural-computer.external-capability-composition.v1"
)
EXTERNAL_CAPABILITY_SHARED_RESIDUAL_SCHEMA = (
    "neural-computer.external-capability-shared-residual.v1"
)
EXTERNAL_CAPABILITY_RESIDUAL_COMPUTE_SCHEMA = (
    "neural-computer.external-capability-residual-compute.v1"
)
EXTERNAL_CAPABILITY_REUSABLE_COMPUTE_SCHEMA = (
    "neural-computer.external-capability-reusable-compute.v1"
)
EXTERNAL_CAPABILITY_COMPUTE_ADMISSION_SCHEMA = (
    "neural-computer.external-capability-compute-admission.v1"
)
EXTERNAL_CAPABILITY_COMPUTE_SCREEN_SCHEMA = (
    "neural-computer.external-capability-compute-screen.v1"
)
EXTERNAL_CAPABILITY_LEARNED_COMPUTE_SCREEN_SCHEMA = (
    "neural-computer.external-capability-learned-compute-screen.v1"
)
EXTERNAL_CAPABILITY_APPEND_ONLY_LEARNED_COMPUTE_SCREEN_SCHEMA = (
    "neural-computer.external-capability-append-only-learned-compute-screen.v1"
)
EXTERNAL_CAPABILITY_SLOT_BINDING_SCHEMA = (
    "neural-computer.external-capability-slot-binding.v1"
)


@dataclass(frozen=True)
class ComputeReuseDecision:
    """Fresh-outcome decision to reuse a compute slot or append one."""

    action: Literal["reuse", "grow"]
    compute_slot_index: int | None
    candidate_scores: tuple[tuple[int, float], ...]
    reason: str


def select_reusable_compute_slot(
    candidate_outcomes: Mapping[int, Sequence[float]],
    *,
    threshold: float,
) -> ComputeReuseDecision:
    """Select the strongest candidate only when every fresh probe passes.

    Candidate identities are opaque physical-slot indices. The policy never
    infers semantic compatibility; it uses only fresh verifier outcomes and
    grows capacity when no candidate reaches the mastery floor.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("compute admission threshold must lie in [0, 1]")
    scores: list[tuple[int, float]] = []
    for slot_index, outcomes in sorted(candidate_outcomes.items()):
        if slot_index < 0:
            raise ValueError("compute candidate indices must be nonnegative")
        values = tuple(float(value) for value in outcomes)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("compute candidates need finite fresh outcomes")
        scores.append((slot_index, min(values)))
    eligible = [item for item in scores if item[1] >= threshold]
    if eligible:
        selected_index, selected_score = max(eligible, key=lambda item: (item[1], -item[0]))
        return ComputeReuseDecision(
            action="reuse",
            compute_slot_index=selected_index,
            candidate_scores=tuple(scores),
            reason=f"fresh_probe_floor_passed:{selected_score:.6f}",
        )
    return ComputeReuseDecision(
        action="grow",
        compute_slot_index=None,
        candidate_scores=tuple(scores),
        reason=(
            "no_compute_candidate_passed_fresh_probe_floor"
            if scores
            else "no_compute_candidates"
        ),
    )


class ExternalComputeCandidateScreen:
    """Order opaque compute candidates using learned-event evidence only.

    The screen is external mutable memory, not a controller branch.  It
    receives a learned event/context query, an opaque candidate index, and a
    deterministic scalar verifier outcome.  It may change trial order, but it
    never authorizes reuse; callers must still apply a fresh admission gate to
    the candidate that was actually tried.
    """

    schema = EXTERNAL_CAPABILITY_COMPUTE_SCREEN_SCHEMA

    def __init__(
        self,
        width: int,
        *,
        matching_tolerance: float = 1e-4,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.75,
        min_mastery_observations: int = 1,
        reversal_threshold: float = 0.5,
        reversal_patience: int = 4,
    ) -> None:
        if width < 1:
            raise ValueError("compute screen width must be positive")
        self.width = int(width)
        self._evidence_parameters = {
            "matching_tolerance": matching_tolerance,
            "prior_strength": prior_strength,
            "mastery_threshold": mastery_threshold,
            "min_mastery_observations": min_mastery_observations,
            "reversal_threshold": reversal_threshold,
            "reversal_patience": reversal_patience,
        }
        self._evidence = PersistentOpaqueContextRouteEvidence(
            self.width,
            **self._evidence_parameters,
        )
        self._global_evidence = PersistentOpaqueRouteEvidence(
            prior_strength=prior_strength,
            mastery_threshold=mastery_threshold,
            min_mastery_observations=min_mastery_observations,
            reversal_threshold=reversal_threshold,
            reversal_patience=reversal_patience,
        )

    @property
    def candidate_count(self) -> int:
        """Return the number of opaque physical candidates in the screen."""

        return self._global_evidence.slot_count

    @property
    def context_count(self) -> int:
        """Return the number of learned event-query rows retained externally."""

        return self._evidence.context_count

    def configuration(self) -> dict[str, int | float | str]:
        """Return the versioned screen contract and its safety boundary."""

        return {
            "schema": self.schema,
            "width": self.width,
            "candidate_count": self.candidate_count,
            "context_count": self.context_count,
            "query": "learned_event_context_v1",
            "evidence": "opaque_scalar_verifier_outcomes_v1",
            "fallback": "global_opaque_candidate_prior_v1",
            "role": "order_only_fresh_admission_required",
        }

    def add_candidate(self) -> int:
        """Append one opaque candidate address without changing old evidence."""

        context_slot = self._evidence.append_slot()
        global_slot = self._global_evidence.append_slot()
        if context_slot != global_slot:
            raise RuntimeError("compute screen evidence slot counts diverged")
        return global_slot

    def order(self, query: torch.Tensor) -> tuple[int, ...]:
        """Return the learned-first trial order for one event/context query."""

        if not isinstance(query, torch.Tensor):
            raise TypeError("compute screen query must be a tensor")
        if query.ndim != 1 or query.shape[0] != self.width:
            raise ValueError(f"compute screen query must have shape [{self.width}]")
        if self.candidate_count < 1:
            raise ValueError("compute screen has no candidates")
        if self._evidence.has_context(query):
            return self._evidence.preferred_order(query)
        return self._global_evidence.preferred_order(slot_count=self.candidate_count)

    def observe(
        self,
        query: torch.Tensor,
        candidate_index: int,
        outcome: float | torch.Tensor,
    ) -> None:
        """Record one attempted opaque candidate and its scalar outcome."""

        if not isinstance(query, torch.Tensor):
            raise TypeError("compute screen query must be a tensor")
        if query.ndim != 1 or query.shape[0] != self.width:
            raise ValueError(f"compute screen query must have shape [{self.width}]")
        self._evidence.observe(query, candidate_index, outcome)
        self._global_evidence.observe(candidate_index, outcome)

    def payload(self) -> dict[str, object]:
        """Serialize only versioned opaque screen state for external memory."""

        return {
            "schema": self.schema,
            "width": self.width,
            "evidence": self._evidence.payload(),
            "global_evidence": self._global_evidence.payload(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ExternalComputeCandidateScreen:
        """Restore a validated candidate screen without semantic fields."""

        if payload.get("schema") != cls.schema:
            raise ValueError("compute screen schema is incompatible")
        evidence_payload = payload.get("evidence")
        global_evidence_payload = payload.get("global_evidence")
        if not isinstance(evidence_payload, dict):
            raise TypeError("compute screen evidence must be a dictionary")
        if not isinstance(global_evidence_payload, dict):
            raise TypeError("compute screen global evidence must be a dictionary")
        evidence = PersistentOpaqueContextRouteEvidence.from_payload(evidence_payload)
        global_evidence = PersistentOpaqueRouteEvidence.from_payload(
            global_evidence_payload
        )
        screen = cls(int(payload["width"]))
        if evidence.width != screen.width:
            raise ValueError("compute screen evidence width is incompatible")
        if evidence.slot_count != global_evidence.slot_count:
            raise ValueError("compute screen evidence slot counts are incompatible")
        screen._evidence = evidence
        screen._global_evidence = global_evidence
        return screen


class LearnedComputeCandidateScreen(nn.Module):
    """Generalize opaque compute-candidate ordering across novel queries.

    This is a replaceable memory-side scorer.  It receives learned event
    queries and opaque candidate keys, and learns only from scalar outcomes
    of attempted candidates.  The score is an ordering aid; fresh verifier
    admission remains a separate authority.  The factorized query/key scorer
    is disabled at construction, so a cold screen returns exact append-order
    ties until an external caller enables it after observing evidence.
    """

    schema = EXTERNAL_CAPABILITY_LEARNED_COMPUTE_SCREEN_SCHEMA

    def __init__(
        self,
        query_width: int,
        key_width: int,
        *,
        latent_width: int = 32,
        hidden: int = 64,
    ) -> None:
        if min(query_width, key_width, latent_width, hidden) < 1:
            raise ValueError("learned compute screen dimensions must be positive")
        super().__init__()
        self.query_width = int(query_width)
        self.key_width = int(key_width)
        self.latent_width = int(latent_width)
        self.hidden = int(hidden)
        self.query_projection = nn.Sequential(
            nn.Linear(self.query_width, self.latent_width),
            nn.GELU(),
        )
        self.key_projection = nn.Sequential(
            nn.Linear(self.key_width, self.latent_width),
            nn.GELU(),
        )
        self.router = FactorizedOpaqueAddressRouter(
            self.latent_width,
            hidden=self.hidden,
        )
        self.register_buffer("enabled", torch.tensor(False, dtype=torch.bool))

    def configuration(self) -> dict[str, bool | int | str]:
        """Return the versioned generalizing screen contract."""

        return {
            "schema": self.schema,
            "query_width": self.query_width,
            "key_width": self.key_width,
            "latent_width": self.latent_width,
            "hidden": self.hidden,
            "query": "learned_event_tensor_v1",
            "candidate_key": "opaque_external_compute_signature_v1",
            "outcome": "attempted_scalar_verifier_v1",
            "cold_start": "zero_until_explicit_evidence_enable_v1",
            "enabled": bool(self.enabled.item()),
            "role": "order_only_fresh_admission_required",
        }

    def enable(self) -> None:
        """Enable learned scores after fresh evidence has been observed."""

        self.enabled.fill_(True)

    def _validate_inputs(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[1] != self.query_width:
            raise ValueError(
                f"learned compute screen query must have shape [batch, {self.query_width}]"
            )
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        if (
            keys.ndim != 3
            or keys.shape[0] != query.shape[0]
            or keys.shape[1] < 1
            or keys.shape[2] != self.key_width
        ):
            raise ValueError(
                "learned compute screen keys must have shape "
                f"[batch, candidates, {self.key_width}] or "
                f"[candidates, {self.key_width}]"
            )
        if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(keys).all()):
            raise ValueError("learned compute screen inputs must be finite")
        return keys

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Return one learned score per opaque candidate row."""

        keys = self._validate_inputs(query, keys)
        if not bool(self.enabled.item()):
            return torch.zeros(
                query.shape[0],
                keys.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
        query_latent = self.query_projection(query)
        key_latent = self.key_projection(keys)
        return self.router(query_latent, key_latent)

    @torch.no_grad()
    def order(self, query: torch.Tensor, keys: torch.Tensor) -> tuple[int, ...]:
        """Return candidate indices in descending learned-screen order."""

        if query.ndim != 1 or query.shape[0] != self.query_width:
            raise ValueError(
                f"learned compute screen query must have shape [{self.query_width}]"
            )
        scores = self(query.unsqueeze(0), keys)[0]
        return tuple(torch.argsort(scores, descending=True, stable=True).tolist())

    def outcome_ranking_loss(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Train compatibility from attempted scalar outcomes only.

        Every informative pair compares two candidates attempted for the same
        learned query.  The outcome difference is detached from the model and
        no correct or unattempted-action label is required.
        """

        scores = self(query, keys)
        if outcomes.ndim == 1:
            outcomes = outcomes.unsqueeze(0)
        if outcomes.shape != scores.shape:
            raise ValueError("candidate outcomes must align with screen scores")
        if not bool(torch.isfinite(outcomes).all()) or not bool(
            ((outcomes >= 0.0) & (outcomes <= 1.0)).all()
        ):
            raise ValueError("candidate outcomes must lie in [0, 1]")
        outcome_delta = outcomes.unsqueeze(2) - outcomes.unsqueeze(1)
        score_delta = scores.unsqueeze(2) - scores.unsqueeze(1)
        informative = outcome_delta > 0.0
        informative_count = int(informative.sum().detach().cpu().item())
        if informative_count == 0:
            return scores.sum() * 0.0, 0
        loss = F.softplus(
            -outcome_delta[informative].detach() * score_delta[informative]
        ).mean()
        return loss, informative_count

    def outcome_calibration_loss(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        attempted_indices: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Calibrate one attempted candidate from its scalar outcome.

        Unlike pairwise ranking, this objective remains informative when an
        extension contains one candidate.  The attempted candidate index and
        verifier outcome are the only supervision; candidates that were not
        attempted are not included in the loss.
        """

        scores = self(query, keys)
        if attempted_indices.ndim != 1 or attempted_indices.shape[0] != scores.shape[0]:
            raise ValueError("attempted candidate indices must align with queries")
        if attempted_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("attempted candidate indices must be integer tensors")
        if outcomes.ndim != 1 or outcomes.shape[0] != scores.shape[0]:
            raise ValueError("attempted candidate outcomes must align with queries")
        if not bool(torch.isfinite(outcomes).all()) or not bool(
            ((outcomes >= 0.0) & (outcomes <= 1.0)).all()
        ):
            raise ValueError("attempted candidate outcomes must lie in [0, 1]")
        indices = attempted_indices.to(device=scores.device, dtype=torch.long)
        if bool((indices < 0).any()) or bool((indices >= scores.shape[1]).any()):
            raise ValueError("attempted candidate indices are out of range")
        logits = scores.gather(1, indices.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            outcomes.to(device=logits.device, dtype=logits.dtype),
        )
        return loss, int(outcomes.shape[0])


class AppendOnlyLearnedComputeCandidateScreen(nn.Module):
    """Grow candidate-screen state without mutating the mastered base.

    The base screen and each appended extension are independent memory-side
    modules.  An extension can compete only after the caller supplies scalar
    verifier failure for it and every earlier stage; before then, the base
    scores and argmax are preserved exactly.  New evidence is therefore
    isolated to an append-only state boundary rather than fine-tuning the
    route that already works.
    """

    schema = EXTERNAL_CAPABILITY_APPEND_ONLY_LEARNED_COMPUTE_SCREEN_SCHEMA

    def __init__(
        self,
        query_width: int,
        key_width: int,
        *,
        latent_width: int = 32,
        hidden: int = 64,
        extension_sizes: Iterable[int] = (),
    ) -> None:
        super().__init__()
        self.query_width = int(query_width)
        self.key_width = int(key_width)
        self.latent_width = int(latent_width)
        self.hidden = int(hidden)
        if min(self.query_width, self.key_width, self.latent_width, self.hidden) < 1:
            raise ValueError("append-only screen dimensions must be positive")
        self.base_screen = LearnedComputeCandidateScreen(
            self.query_width,
            self.key_width,
            latent_width=self.latent_width,
            hidden=self.hidden,
        )
        self.extensions = nn.ModuleList()
        self.extension_sizes: list[int] = []
        for candidate_count in extension_sizes:
            self.append_extension(candidate_count)

    def append_extension(self, candidate_count: int) -> int:
        """Append an isolated candidate group and return its stage index."""

        if candidate_count < 1:
            raise ValueError("extension candidate count must be positive")
        extension = LearnedComputeCandidateScreen(
            self.query_width,
            self.key_width,
            latent_width=self.latent_width,
            hidden=self.hidden,
        )
        self.extensions.append(extension)
        self.extension_sizes.append(int(candidate_count))
        return len(self.extensions) - 1

    def enable_base(self) -> None:
        """Enable the learned base screen after its evidence gate passes."""

        self.base_screen.enable()

    def enable_extension(self, index: int) -> None:
        """Enable one extension after fresh evidence for that group."""

        self.extensions[index].enable()

    def freeze_base(self) -> None:
        """Freeze the mastered screen while allowing extension training."""

        for parameter in self.base_screen.parameters():
            parameter.requires_grad_(False)

    def configuration(self) -> dict[str, object]:
        """Return the versioned append-only screen contract."""

        return {
            "schema": self.schema,
            "query_width": self.query_width,
            "key_width": self.key_width,
            "latent_width": self.latent_width,
            "hidden": self.hidden,
            "extension_sizes": tuple(self.extension_sizes),
            "base_enabled": bool(self.base_screen.enabled.item()),
            "extension_enabled": tuple(
                bool(extension.enabled.item()) for extension in self.extensions
            ),
            "failure_signal": "cumulative_stage_scalar_verifier_failure_v1",
            "growth": "isolated_append_only_memory_state_v1",
            "role": "order_only_fresh_admission_required",
        }

    def _validate_extension_inputs(
        self,
        query: torch.Tensor,
        extension_keys: torch.Tensor,
        failed_extensions: torch.Tensor | bool | None,
    ) -> torch.Tensor:
        if extension_keys.ndim == 2:
            extension_keys = extension_keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        expected = sum(self.extension_sizes)
        if (
            extension_keys.ndim != 3
            or extension_keys.shape[0] != query.shape[0]
            or extension_keys.shape[1] != expected
            or extension_keys.shape[2] != self.key_width
        ):
            raise ValueError(
                "append-only screen extension keys must have shape "
                f"[batch, {expected}, {self.key_width}] or "
                f"[{expected}, {self.key_width}]"
            )
        if (
            failed_extensions is not None
            and not isinstance(failed_extensions, bool)
            and failed_extensions.shape != (query.shape[0], len(self.extensions))
        ):
            raise ValueError("failed_extensions must have shape [batch, extension_count]")
        return extension_keys

    def forward(
        self,
        query: torch.Tensor,
        base_keys: torch.Tensor,
        extension_keys: torch.Tensor | None = None,
        failed_extensions: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        """Return base rows plus failure-gated appended candidate rows."""

        base_scores = self.base_screen(query, base_keys)
        if not self.extensions:
            if extension_keys is not None:
                raise ValueError("extension keys require at least one extension")
            if failed_extensions is not None:
                raise ValueError("failed extensions require at least one extension")
            return base_scores
        if extension_keys is None:
            raise ValueError("extension keys are required when extensions exist")
        extension_keys = self._validate_extension_inputs(
            query, extension_keys, failed_extensions
        )
        if failed_extensions is None:
            failures = torch.zeros(
                query.shape[0],
                len(self.extensions),
                dtype=torch.bool,
                device=query.device,
            )
        elif isinstance(failed_extensions, bool):
            failures = torch.full(
                (query.shape[0], len(self.extensions)),
                failed_extensions,
                dtype=torch.bool,
                device=query.device,
            )
        else:
            failures = failed_extensions.to(device=query.device, dtype=torch.bool)
        scores = base_scores
        offset = 0
        for index, (extension, size) in enumerate(
            zip(self.extensions, self.extension_sizes, strict=True)
        ):
            keys = extension_keys[:, offset : offset + size]
            stage_failed = failures[:, : index + 1].all(dim=-1)
            scores = failure_gated_candidate_scores(
                scores,
                extension(query, keys),
                stage_failed,
            )
            offset += size
        return scores


@dataclass(frozen=True)
class ExternalCapabilityState:
    """External recurrent state owned by one capability instance."""

    context: torch.Tensor

    def validate(self, *, batch_size: int, hidden: int) -> ExternalCapabilityState:
        if self.context.ndim != 2 or self.context.shape != (batch_size, hidden):
            raise ValueError("capability context state has the wrong shape")
        if not bool(torch.isfinite(self.context).all()):
            raise ValueError("capability context state must be finite")
        return self


@dataclass(frozen=True)
class ExternalCapabilityPipelineState:
    """Independent recurrent states for a memory-side capability pipeline."""

    programs: tuple[ExternalCapabilityState, ...]

    def validate(
        self,
        *,
        batch_size: int,
        hidden_sizes: tuple[int, ...],
    ) -> ExternalCapabilityPipelineState:
        if len(self.programs) != len(hidden_sizes):
            raise ValueError("pipeline state does not match program count")
        for state, hidden in zip(self.programs, hidden_sizes, strict=True):
            state.validate(batch_size=batch_size, hidden=hidden)
        return self


class ExternalCapabilityProgram(nn.Module):
    """A generic recurrent memory-side program for one frozen controller.

    The program consumes standardized learned events, opaque action vectors,
    scalar outcomes, and the controller's opaque intention.  It returns an
    adapted intention and keeps its recurrent state outside the controller.
    It never receives raw modality data, task identifiers, correct actions, or
    protocol-specific fields.  A caller may attach any compatible decoder to
    the returned intention through the ordinary output bus.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        *,
        context_hidden: int = 64,
        context_width: int = 32,
        adapter_hidden: int = 64,
    ) -> None:
        super().__init__()
        if (
            min(
                event_width,
                action_width,
                intention_width,
                context_hidden,
                context_width,
                adapter_hidden,
            )
            < 1
        ):
            raise ValueError("external capability dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.context_hidden = int(context_hidden)
        self.context_width = int(context_width)
        self.adapter_hidden = int(adapter_hidden)
        self.context_encoder = EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.context_hidden,
            context_width=self.context_width,
        )
        self.intent_adapter = EpisodicIntentAdapter(
            self.context_width,
            self.intention_width,
            hidden=self.adapter_hidden,
        )

    def configuration(self) -> dict[str, int | str]:
        """Return the versioned capability interface contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "context_hidden": self.context_hidden,
            "context_width": self.context_width,
            "adapter_hidden": self.adapter_hidden,
            "state": "external_recurrent_context_v1",
            "output": "opaque_intention_residual_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityState:
        if batch_size < 1:
            raise ValueError("capability batch size must be positive")
        return ExternalCapabilityState(
            context=torch.zeros(
                batch_size,
                self.context_hidden,
                device=device,
                dtype=dtype,
            )
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityState]:
        """Advance external state and adapt one opaque controller intention."""

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for capability")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for capability")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for capability")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match capability event")
        state.validate(batch_size=event.shape[0], hidden=self.context_hidden)
        context, next_context = self.context_encoder.step(
            event,
            action,
            outcome,
            state.context,
            present,
        )
        adapted = self.intent_adapter(intention, context.context)
        return adapted, ExternalCapabilityState(next_context)


class ExternalCapabilitySharedResidualBank(nn.Module):
    """Share a frozen context basis while growing isolated residual slots.

    The context encoder is one replaceable memory-side base.  Each residual
    adapter has its own externally owned recurrent state and can be trained or
    replaced independently, so adding a capability never updates an earlier
    residual.  The bank is deliberately unaware of task names, raw protocols,
    and correct actions; an external opaque binding chooses ``slot_index``.

    This is a compression candidate, not an unconditional consolidation
    operation.  Callers must freeze ``shared_context_encoder`` before adding
    new slots and retain each alias only after fresh behavior verification.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        *,
        slot_count: int = 1,
        context_hidden: int = 64,
        context_width: int = 32,
        adapter_hidden: int = 64,
    ) -> None:
        super().__init__()
        if slot_count < 1:
            raise ValueError("shared residual bank needs at least one slot")
        if min(
            event_width,
            action_width,
            intention_width,
            context_hidden,
            context_width,
            adapter_hidden,
        ) < 1:
            raise ValueError("shared residual dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.context_hidden = int(context_hidden)
        self.context_width = int(context_width)
        self.adapter_hidden = int(adapter_hidden)
        self.shared_context_encoder = EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.context_hidden,
            context_width=self.context_width,
        )
        self.residual_slots = nn.ModuleList(
            self._new_residual() for _ in range(slot_count)
        )

    def _new_residual(self) -> EpisodicIntentAdapter:
        return EpisodicIntentAdapter(
            self.context_width,
            self.intention_width,
            hidden=self.adapter_hidden,
        )

    @property
    def slot_count(self) -> int:
        return len(self.residual_slots)

    def configuration(self) -> dict[str, int | str]:
        """Return the versioned shared-base/residual contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_SHARED_RESIDUAL_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "context_hidden": self.context_hidden,
            "context_width": self.context_width,
            "adapter_hidden": self.adapter_hidden,
            "slot_count": self.slot_count,
            "state": "independent_external_recurrent_contexts_v1",
            "base": "one_shared_context_encoder_v1",
            "residual": "independent_intention_adapters_v1",
        }

    def freeze_shared_base(self) -> None:
        """Make the shared representation immutable for later slot growth."""

        for parameter in self.shared_context_encoder.parameters():
            parameter.requires_grad_(False)

    def add_slot(self) -> int:
        """Append a zero-initialized residual without changing old weights."""

        residual = self._new_residual()
        reference = next(self.shared_context_encoder.parameters())
        residual.to(device=reference.device, dtype=reference.dtype)
        self.residual_slots.append(residual)
        return self.slot_count - 1

    def freeze_slot(self, slot_index: int) -> None:
        """Protect one residual from later capability-specific updates."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("shared residual slot is out of range")
        for parameter in self.residual_slots[slot_index].parameters():
            parameter.requires_grad_(False)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        if batch_size < 1:
            raise ValueError("shared residual batch size must be positive")
        return ExternalCapabilityPipelineState(
            tuple(
                ExternalCapabilityState(
                    self.shared_context_encoder.initial_state(
                        batch_size,
                        device=device,
                        dtype=dtype,
                    )
                )
                for _ in self.residual_slots
            )
        )

    def step(
        self,
        *,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Execute one opaque residual binding and advance only its state."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("shared residual slot is out of range")
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for shared residual bank")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for shared residual bank")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for shared residual bank")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match shared residual event")
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=(self.context_hidden,) * self.slot_count,
        )
        adapted, next_state = self.step_slot(
            slot_index=slot_index,
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state.programs[slot_index],
            present=present,
        )
        next_states = list(state.programs)
        next_states[slot_index] = next_state
        return adapted, ExternalCapabilityPipelineState(tuple(next_states))

    def step_slot(
        self,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        *,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityState]:
        """Execute one slot using only that slot's externally owned state."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("shared residual slot is out of range")
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for shared residual bank")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for shared residual bank")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for shared residual bank")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match shared residual event")
        state.validate(batch_size=event.shape[0], hidden=self.context_hidden)
        context, next_context = self.shared_context_encoder.step(
            event,
            action,
            outcome,
            state.context,
            present,
        )
        adapted = self.residual_slots[slot_index](intention, context.context)
        return adapted, ExternalCapabilityState(next_context)


class ExternalCapabilityResidualComputeBank(nn.Module):
    """Grow small local recurrent programs behind a frozen shared basis.

    The shared context encoder is trained once and can then be frozen.  Each
    appended slot receives a compact recurrent context encoder and an intention
    adapter that can learn new sequential computation without changing the
    shared basis or any protected slot.  The per-slot state is still external
    and opaque to the controller.  This is an append-only compute candidate;
    it does not claim compression when a procedure needs unique computation.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        *,
        slot_count: int = 1,
        shared_context_hidden: int = 64,
        shared_context_width: int = 32,
        residual_context_hidden: int = 16,
        residual_context_width: int = 8,
        adapter_hidden: int = 64,
    ) -> None:
        super().__init__()
        dimensions = (
            event_width,
            action_width,
            intention_width,
            shared_context_hidden,
            shared_context_width,
            residual_context_hidden,
            residual_context_width,
            adapter_hidden,
        )
        if slot_count < 1:
            raise ValueError("residual compute bank needs at least one slot")
        if min(dimensions) < 1:
            raise ValueError("residual compute dimensions must be positive")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.shared_context_hidden = int(shared_context_hidden)
        self.shared_context_width = int(shared_context_width)
        self.residual_context_hidden = int(residual_context_hidden)
        self.residual_context_width = int(residual_context_width)
        self.adapter_hidden = int(adapter_hidden)
        self.context_hidden = self.shared_context_hidden + self.residual_context_hidden
        self.context_width = self.shared_context_width + self.residual_context_width
        self.shared_context_encoder = EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.shared_context_hidden,
            context_width=self.shared_context_width,
        )
        self.residual_slots = nn.ModuleList(
            self._new_slot() for _ in range(slot_count)
        )

    def _new_slot(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "context_encoder": EpisodicContextEncoder(
                    self.event_width,
                    self.action_width,
                    hidden=self.residual_context_hidden,
                    context_width=self.residual_context_width,
                ),
                "intent_adapter": EpisodicIntentAdapter(
                    self.context_width,
                    self.intention_width,
                    hidden=self.adapter_hidden,
                ),
            }
        )

    @property
    def slot_count(self) -> int:
        return len(self.residual_slots)

    def configuration(self) -> dict[str, int | str]:
        """Return the versioned append-only compute contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_RESIDUAL_COMPUTE_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "shared_context_hidden": self.shared_context_hidden,
            "shared_context_width": self.shared_context_width,
            "residual_context_hidden": self.residual_context_hidden,
            "residual_context_width": self.residual_context_width,
            "adapter_hidden": self.adapter_hidden,
            "slot_count": self.slot_count,
            "state": "independent_external_shared_and_residual_contexts_v1",
            "base": "one_shared_context_encoder_v1",
            "residual": "one_compact_recurrent_compute_encoder_per_slot_v1",
        }

    def freeze_shared_base(self) -> None:
        """Make the shared representation immutable for later slot growth."""

        for parameter in self.shared_context_encoder.parameters():
            parameter.requires_grad_(False)

    def freeze_slot(self, slot_index: int) -> None:
        """Protect one local compute slot from later updates."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("residual compute slot is out of range")
        for parameter in self.residual_slots[slot_index].parameters():
            parameter.requires_grad_(False)

    def add_slot(self) -> int:
        """Append compact local compute without changing old parameters."""

        slot = self._new_slot()
        reference = next(self.shared_context_encoder.parameters())
        slot.to(device=reference.device, dtype=reference.dtype)
        self.residual_slots.append(slot)
        return self.slot_count - 1

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        if batch_size < 1:
            raise ValueError("residual compute batch size must be positive")
        return ExternalCapabilityPipelineState(
            tuple(
                ExternalCapabilityState(
                    torch.cat(
                        (
                            self.shared_context_encoder.initial_state(
                                batch_size,
                                device=device,
                                dtype=dtype,
                            ),
                            self.residual_slots[slot_index][
                                "context_encoder"
                            ].initial_state(
                                batch_size,
                                device=device,
                                dtype=dtype,
                            ),
                        ),
                        dim=-1,
                    )
                )
                for slot_index in range(self.slot_count)
            )
        )

    def step(
        self,
        *,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Execute one slot while preserving every other external state."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("residual compute slot is out of range")
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=(self.context_hidden,) * self.slot_count,
        )
        adapted, next_state = self.step_slot(
            slot_index=slot_index,
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state.programs[slot_index],
            present=present,
        )
        next_states = list(state.programs)
        next_states[slot_index] = next_state
        return adapted, ExternalCapabilityPipelineState(tuple(next_states))

    def step_slot(
        self,
        *,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityState]:
        """Execute one slot using only its shared/local external state."""

        if slot_index < 0 or slot_index >= self.slot_count:
            raise IndexError("residual compute slot is out of range")
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for residual compute bank")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for residual compute bank")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for residual compute bank")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match residual compute event")
        state.validate(batch_size=event.shape[0], hidden=self.context_hidden)
        shared_state, residual_state = torch.split(
            state.context,
            (self.shared_context_hidden, self.residual_context_hidden),
            dim=-1,
        )
        shared_context, next_shared = self.shared_context_encoder.step(
            event,
            action,
            outcome,
            shared_state,
            present,
        )
        slot = self.residual_slots[slot_index]
        residual_context, next_residual = slot["context_encoder"].step(
            event,
            action,
            outcome,
            residual_state,
            present,
        )
        combined = torch.cat((shared_context.context, residual_context.context), dim=-1)
        adapted = slot["intent_adapter"](intention, combined)
        return adapted, ExternalCapabilityState(
            torch.cat((next_shared, next_residual), dim=-1)
        )


class ExternalCapabilityReusableComputeLibrary(nn.Module):
    """Bind capabilities to reusable compute modules without weight copying.

    A physical compute slot contains only a compact recurrent context encoder.
    A logical capability binding owns its own intention adapter and external
    recurrent state, and points to one physical compute slot by an opaque
    integer chosen by memory-side policy. Adding a binding therefore adds no
    recurrent compute parameters; adding a compute slot is explicit and can be
    gated by fresh behavior verification.

    The shared context encoder and physical compute modules are independently
    replaceable from logical bindings. No controller or raw protocol data is
    exposed, and aliases that share a compute slot still receive independent
    recurrent state so simultaneous episodes cannot leak into one another.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        *,
        compute_slot_count: int = 1,
        binding_compute_slots: Iterable[int] = (0,),
        shared_context_hidden: int = 64,
        shared_context_width: int = 32,
        residual_context_hidden: int = 32,
        residual_context_width: int = 16,
        adapter_hidden: int = 64,
    ) -> None:
        super().__init__()
        dimensions = (
            event_width,
            action_width,
            intention_width,
            shared_context_hidden,
            shared_context_width,
            residual_context_hidden,
            residual_context_width,
            adapter_hidden,
        )
        if compute_slot_count < 1:
            raise ValueError("reusable compute library needs one compute slot")
        bindings = tuple(int(index) for index in binding_compute_slots)
        if not bindings:
            raise ValueError("reusable compute library needs one binding")
        if min(dimensions) < 1:
            raise ValueError("reusable compute dimensions must be positive")
        if any(index < 0 or index >= compute_slot_count for index in bindings):
            raise ValueError("binding points to an invalid compute slot")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.shared_context_hidden = int(shared_context_hidden)
        self.shared_context_width = int(shared_context_width)
        self.residual_context_hidden = int(residual_context_hidden)
        self.residual_context_width = int(residual_context_width)
        self.adapter_hidden = int(adapter_hidden)
        self.context_hidden = self.shared_context_hidden + self.residual_context_hidden
        self.context_width = self.shared_context_width + self.residual_context_width
        self.shared_context_encoder = EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.shared_context_hidden,
            context_width=self.shared_context_width,
        )
        self.compute_slots = nn.ModuleList(
            self._new_compute_slot() for _ in range(compute_slot_count)
        )
        self.binding_adapters = nn.ModuleList(
            self._new_binding_adapter() for _ in bindings
        )
        self._binding_compute_slots = list(bindings)

    def _new_compute_slot(self) -> EpisodicContextEncoder:
        return EpisodicContextEncoder(
            self.event_width,
            self.action_width,
            hidden=self.residual_context_hidden,
            context_width=self.residual_context_width,
        )

    def _new_binding_adapter(self) -> EpisodicIntentAdapter:
        return EpisodicIntentAdapter(
            self.context_width,
            self.intention_width,
            hidden=self.adapter_hidden,
        )

    @property
    def slot_count(self) -> int:
        """Return the number of logical capability bindings."""

        return len(self._binding_compute_slots)

    @property
    def compute_slot_count(self) -> int:
        """Return the number of physical recurrent compute modules."""

        return len(self.compute_slots)

    @property
    def binding_compute_slots(self) -> tuple[int, ...]:
        """Return the opaque logical-to-physical binding table."""

        return tuple(self._binding_compute_slots)

    def configuration(self) -> dict[str, object]:
        """Return the versioned compute-library and binding contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_REUSABLE_COMPUTE_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "shared_context_hidden": self.shared_context_hidden,
            "shared_context_width": self.shared_context_width,
            "residual_context_hidden": self.residual_context_hidden,
            "residual_context_width": self.residual_context_width,
            "adapter_hidden": self.adapter_hidden,
            "compute_slot_count": self.compute_slot_count,
            "binding_count": self.slot_count,
            "binding_compute_slots": self.binding_compute_slots,
            "state": "independent_external_state_per_binding_v1",
            "compute": "shared_context_plus_reusable_local_recurrent_module_v1",
            "binding": "opaque_binding_specific_intention_adapter_v1",
        }

    def freeze_shared_base(self) -> None:
        """Make the shared representation immutable for later growth."""

        for parameter in self.shared_context_encoder.parameters():
            parameter.requires_grad_(False)

    def freeze_compute_slot(self, compute_slot_index: int) -> None:
        """Protect one physical compute module from later updates."""

        if compute_slot_index < 0 or compute_slot_index >= self.compute_slot_count:
            raise IndexError("reusable compute slot is out of range")
        for parameter in self.compute_slots[compute_slot_index].parameters():
            parameter.requires_grad_(False)

    def freeze_binding(self, binding_index: int) -> None:
        """Protect one logical binding adapter from later updates."""

        if binding_index < 0 or binding_index >= self.slot_count:
            raise IndexError("reusable binding is out of range")
        for parameter in self.binding_adapters[binding_index].parameters():
            parameter.requires_grad_(False)

    def freeze_slot(self, binding_index: int) -> None:
        """Compatibility alias for protecting one logical binding."""

        self.freeze_binding(binding_index)

    def add_compute_slot(self) -> int:
        """Append one physical recurrent module without changing old modules."""

        compute_slot = self._new_compute_slot()
        reference = next(self.shared_context_encoder.parameters())
        compute_slot.to(device=reference.device, dtype=reference.dtype)
        self.compute_slots.append(compute_slot)
        return self.compute_slot_count - 1

    def add_binding(self, compute_slot_index: int) -> int:
        """Append a binding adapter pointing at an existing compute module."""

        if compute_slot_index < 0 or compute_slot_index >= self.compute_slot_count:
            raise IndexError("reusable compute slot is out of range")
        adapter = self._new_binding_adapter()
        reference = next(self.shared_context_encoder.parameters())
        adapter.to(device=reference.device, dtype=reference.dtype)
        self.binding_adapters.append(adapter)
        self._binding_compute_slots.append(int(compute_slot_index))
        return self.slot_count - 1

    def add_slot(self) -> int:
        """Append a new compute module and bind one capability to it."""

        return self.add_binding(self.add_compute_slot())

    def remove_binding(self, binding_index: int) -> None:
        """Discard the newest unpromoted binding without touching compute."""

        if binding_index != self.slot_count - 1:
            raise ValueError("only the newest reusable binding can be discarded")
        if self.slot_count < 2:
            raise ValueError("reusable compute library must retain one binding")
        self.binding_adapters.pop(binding_index)
        self._binding_compute_slots.pop()

    def binding_modules(
        self,
        binding_index: int,
    ) -> tuple[nn.Module, nn.Module]:
        """Return the physical compute and logical adapter for one binding."""

        if binding_index < 0 or binding_index >= self.slot_count:
            raise IndexError("reusable binding is out of range")
        compute_index = self._binding_compute_slots[binding_index]
        return self.compute_slots[compute_index], self.binding_adapters[binding_index]

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        if batch_size < 1:
            raise ValueError("reusable compute batch size must be positive")
        return ExternalCapabilityPipelineState(
            tuple(
                ExternalCapabilityState(
                    torch.cat(
                        (
                            self.shared_context_encoder.initial_state(
                                batch_size,
                                device=device,
                                dtype=dtype,
                            ),
                            self.compute_slots[compute_index].initial_state(
                                batch_size,
                                device=device,
                                dtype=dtype,
                            ),
                        ),
                        dim=-1,
                    )
                )
                for compute_index in self._binding_compute_slots
            )
        )

    def step(
        self,
        *,
        binding_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Execute one binding while preserving all binding states."""

        if binding_index < 0 or binding_index >= self.slot_count:
            raise IndexError("reusable binding is out of range")
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=(self.context_hidden,) * self.slot_count,
        )
        adapted, next_state = self.step_binding(
            binding_index=binding_index,
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state.programs[binding_index],
            present=present,
        )
        next_states = list(state.programs)
        next_states[binding_index] = next_state
        return adapted, ExternalCapabilityPipelineState(tuple(next_states))

    def step_binding(
        self,
        *,
        binding_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityState]:
        """Execute one binding using its own state and shared compute module."""

        if binding_index < 0 or binding_index >= self.slot_count:
            raise IndexError("reusable binding is out of range")
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for reusable compute library")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for reusable compute library")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for reusable compute library")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match reusable compute event")
        state.validate(batch_size=event.shape[0], hidden=self.context_hidden)
        shared_state, residual_state = torch.split(
            state.context,
            (self.shared_context_hidden, self.residual_context_hidden),
            dim=-1,
        )
        shared_context, next_shared = self.shared_context_encoder.step(
            event,
            action,
            outcome,
            shared_state,
            present,
        )
        compute_slot, adapter = self.binding_modules(binding_index)
        residual_context, next_residual = compute_slot.step(
            event,
            action,
            outcome,
            residual_state,
            present,
        )
        combined = torch.cat((shared_context.context, residual_context.context), dim=-1)
        adapted = adapter(intention, combined)
        return adapted, ExternalCapabilityState(
            torch.cat((next_shared, next_residual), dim=-1)
        )

    def step_slot(
        self,
        *,
        slot_index: int,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityState]:
        """Compatibility alias for executing one logical binding."""

        return self.step_binding(
            binding_index=slot_index,
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
        )


class ExternalCapabilityPipeline(nn.Module):
    """Compose zero or more replaceable capability programs in memory.

    The pipeline is an orchestration boundary, not a controller branch. Each
    program receives the same standardized event, opaque feedback, and scalar
    outcome, while the adapted intention from one program becomes the opaque
    input to the next. Every program retains its own recurrent state outside
    the controller, so the chain can grow, shrink, persist, or be rehydrated
    without resizing the controller or merging program memories.
    """

    def __init__(
        self,
        programs: Iterable[ExternalCapabilityProgram] = (),
        *,
        event_width: int | None = None,
        action_width: int | None = None,
        intention_width: int | None = None,
        hide_downstream_events: bool = False,
    ) -> None:
        super().__init__()
        members = tuple(programs)
        if members:
            dimensions = {
                "event_width": members[0].event_width,
                "action_width": members[0].action_width,
                "intention_width": members[0].intention_width,
            }
            for program in members[1:]:
                if any(
                    getattr(program, name) != value
                    for name, value in dimensions.items()
                ):
                    raise ValueError(
                        "pipeline programs must share interface dimensions"
                    )
        else:
            if None in (event_width, action_width, intention_width):
                raise ValueError(
                    "empty pipelines require event, action, and intention widths"
                )
            dimensions = {
                "event_width": int(event_width),
                "action_width": int(action_width),
                "intention_width": int(intention_width),
            }
        if min(dimensions.values()) < 1:
            raise ValueError("pipeline interface dimensions must be positive")
        self.event_width = dimensions["event_width"]
        self.action_width = dimensions["action_width"]
        self.intention_width = dimensions["intention_width"]
        self.hide_downstream_events = bool(hide_downstream_events)
        self.programs = nn.ModuleList(members)

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(program.context_hidden for program in self.programs)

    def configuration(self) -> dict[str, object]:
        """Return the versioned, order-sensitive composition contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_PIPELINE_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "program_count": len(self.programs),
            "program_schemas": tuple(
                program.configuration()["schema"] for program in self.programs
            ),
            "event_visibility": (
                "head_only" if self.hide_downstream_events else "all_programs"
            ),
            "state": "independent_external_recurrent_contexts_v1",
            "composition": "adapted_intention_serial_chain_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        return ExternalCapabilityPipelineState(
            tuple(
                program.initial_state(batch_size, device=device, dtype=dtype)
                for program in self.programs
            )
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Run one event through the chain while preserving state isolation."""

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for pipeline")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for pipeline")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for pipeline")
        intention.validate(width=self.intention_width)
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=self.hidden_sizes,
        )
        current = intention
        next_states: list[ExternalCapabilityState] = []
        for index, (program, program_state) in enumerate(
            zip(
                self.programs,
                state.programs,
                strict=True,
            )
        ):
            program_event = event
            program_present = present
            if self.hide_downstream_events and index > 0:
                program_event = torch.zeros_like(event)
                program_present = torch.zeros(
                    event.shape[0],
                    dtype=torch.bool,
                    device=event.device,
                )
            current, next_state = program.step(
                event=program_event,
                action=action,
                outcome=outcome,
                intention=current,
                state=program_state,
                present=program_present,
            )
            next_states.append(next_state)
        return current, ExternalCapabilityPipelineState(tuple(next_states))


class ExternalCapabilityComposition(nn.Module):
    """Bind independently learned external programs into a learned sequence.

    Every slot is evaluated from the current opaque intention and an external
    router chooses the slot for each composition step. The controller remains
    unaware of slot identity; program states, router weights, and binding
    decisions all live outside it. Soft routing keeps the boundary
    differentiable while scalar outcome training discovers which learned
    event cues should open each slot.
    """

    def __init__(
        self,
        programs: Iterable[ExternalCapabilityProgram] = (),
        *,
        event_width: int | None = None,
        action_width: int | None = None,
        intention_width: int | None = None,
        composition_steps: int = 2,
        router_hidden: int = 64,
    ) -> None:
        super().__init__()
        members = tuple(programs)
        if members:
            dimensions = {
                "event_width": members[0].event_width,
                "action_width": members[0].action_width,
                "intention_width": members[0].intention_width,
            }
            for program in members[1:]:
                if any(
                    getattr(program, name) != value
                    for name, value in dimensions.items()
                ):
                    raise ValueError(
                        "composition programs must share interface dimensions"
                    )
        else:
            if None in (event_width, action_width, intention_width):
                raise ValueError(
                    "empty compositions require event, action, and intention widths"
                )
            dimensions = {
                "event_width": int(event_width),
                "action_width": int(action_width),
                "intention_width": int(intention_width),
            }
        if len(members) < 2:
            raise ValueError("compositions require at least two programs")
        if composition_steps < 1 or router_hidden < 1:
            raise ValueError("composition steps and router hidden must be positive")
        if min(dimensions.values()) < 1:
            raise ValueError("composition interface dimensions must be positive")
        self.event_width = dimensions["event_width"]
        self.action_width = dimensions["action_width"]
        self.intention_width = dimensions["intention_width"]
        self.composition_steps = int(composition_steps)
        self.router_hidden = int(router_hidden)
        self.programs = nn.ModuleList(members)
        router_input = (
            self.event_width
            + self.action_width
            + 1
            + self.intention_width
        )
        self.router = nn.Sequential(
            nn.Linear(router_input, self.router_hidden),
            nn.GELU(),
            nn.Linear(
                self.router_hidden,
                self.composition_steps * len(self.programs),
            ),
        )

    @property
    def hidden_sizes(self) -> tuple[int, ...]:
        return tuple(program.context_hidden for program in self.programs)

    def configuration(self) -> dict[str, object]:
        """Return the versioned learned-binding contract."""

        return {
            "schema": EXTERNAL_CAPABILITY_COMPOSITION_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "program_count": len(self.programs),
            "composition_steps": self.composition_steps,
            "router_hidden": self.router_hidden,
            "program_schemas": tuple(
                program.configuration()["schema"] for program in self.programs
            ),
            "state": "independent_external_recurrent_contexts_v1",
            "routing": "learned_event_conditioned_soft_slot_binding_v1",
            "binding": "optional_opaque_external_slot_mask_v1",
            "execution": "masked_sparse_active_slots_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalCapabilityPipelineState:
        return ExternalCapabilityPipelineState(
            tuple(
                program.initial_state(batch_size, device=device, dtype=dtype)
                for program in self.programs
            )
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalCapabilityPipelineState,
        present: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
    ) -> tuple[IntentEvent, ExternalCapabilityPipelineState]:
        """Apply a learned slot sequence while keeping state external.

        ``slot_mask`` is an opaque memory-side binding.  It can restrict the
        slots eligible for this alias without exposing a task identifier to
        the controller or changing the learned event representation.
        """

        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for composition")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for composition")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for composition")
        intention.validate(width=self.intention_width)
        state.validate(
            batch_size=event.shape[0],
            hidden_sizes=self.hidden_sizes,
        )
        if slot_mask is not None:
            if slot_mask.ndim != 2 or slot_mask.shape != (
                event.shape[0],
                len(self.programs),
            ):
                raise ValueError("slot mask has the wrong shape for composition")
            if slot_mask.dtype is not torch.bool:
                raise TypeError("slot mask must be boolean")
            if not bool(slot_mask.any(dim=-1).all()):
                raise ValueError("slot mask must allow at least one slot per row")
            slot_mask = slot_mask.to(device=event.device)
        current = intention
        next_states = list(state.programs)
        if slot_mask is None:
            active_indices = tuple(range(len(self.programs)))
        else:
            active_indices = tuple(
                index
                for index in range(len(self.programs))
                if bool(slot_mask[:, index].any())
            )
        for step_index in range(self.composition_steps):
            router_input = torch.cat(
                (event, action, outcome.unsqueeze(1), current.payload),
                dim=-1,
            )
            route_logits = self.router(router_input).reshape(
                event.shape[0], self.composition_steps, len(self.programs)
            )[:, step_index]
            if slot_mask is not None:
                route_logits = route_logits.masked_fill(
                    ~slot_mask,
                    torch.finfo(route_logits.dtype).min,
                )
            weights = torch.softmax(route_logits, dim=-1)
            candidates: list[torch.Tensor] = []
            for index in active_indices:
                program = self.programs[index]
                program_state = next_states[index]
                adapted, next_state = program.step(
                    event=event,
                    action=action,
                    outcome=outcome,
                    intention=current,
                    state=program_state,
                    present=present,
                )
                candidates.append(adapted.payload)
                if slot_mask is not None:
                    enabled = slot_mask[:, index].unsqueeze(-1)
                    next_state = ExternalCapabilityState(
                        torch.where(
                            enabled,
                            next_state.context,
                            program_state.context,
                        )
                    )
                next_states[index] = next_state
            active_weights = weights[:, list(active_indices)]
            current = IntentEvent(
                torch.stack(candidates, dim=1)
                .mul(active_weights.unsqueeze(-1))
                .sum(dim=1)
            )
        return current, ExternalCapabilityPipelineState(tuple(next_states))
