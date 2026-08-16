"""Replaceable identity-assignment artifacts at the external runtime seam.

The controller never consumes this object.  A frontend or external self model
turns learned causal evidence into an opaque slot choice; the live navigation
adapter may then select among caller-owned goal fragments.  Ties are explicit
abstentions rather than guessed identities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA = "neural-computer.external-identity-assignment.v1"
EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA = (
    "neural-computer.external-causal-identity-artifact.v1"
)
EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V2_SCHEMA = (
    "neural-computer.persistent-causal-identity.v2"
)


@dataclass(frozen=True)
class ExternalIdentityAssignment:
    """One verified external slot assignment, or an explicit abstention."""

    selected_slot: torch.Tensor
    confidence: torch.Tensor
    abstained: torch.Tensor
    schema: str = EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA

    def validate(
        self,
        *,
        batch_size: int,
        slot_count: int,
    ) -> ExternalIdentityAssignment:
        if self.schema != EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA:
            raise ValueError("unsupported external identity-assignment schema")
        if slot_count < 1:
            raise ValueError("identity assignment requires at least one slot")
        if self.selected_slot.shape != (batch_size,) or self.selected_slot.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("selected identity slot must be an integer [batch]")
        if self.confidence.shape != (batch_size,):
            raise ValueError("identity confidence must have shape [batch]")
        if self.abstained.shape != (batch_size,) or self.abstained.dtype != torch.bool:
            raise ValueError("identity abstention must be boolean [batch]")
        if not bool(torch.isfinite(self.confidence).all()):
            raise ValueError("identity confidence must be finite")
        if bool(torch.any(self.confidence < 0.0)):
            raise ValueError("identity confidence cannot be negative")
        if bool(torch.any((self.selected_slot < 0) | (self.selected_slot >= slot_count))):
            raise ValueError("selected identity slot is outside the candidate set")
        return self


class ExternalCausalIdentityAssignment:
    """Gate learned causal slot evidence before external goal selection.

    ``evidence`` is an opaque score emitted by a replaceable self/assignment
    artifact.  This gate does not inspect coordinates, labels, rewards, or
    protocol actions.  It only checks the top-two margin and abstains when the
    evidence cannot distinguish candidates.
    """

    schema = EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA

    def __init__(
        self, *, margin: float = 0.15, minimum_evidence: float = 0.0
    ) -> None:
        if not torch.isfinite(torch.tensor(margin)) or margin < 0.0:
            raise ValueError("identity assignment margin must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(minimum_evidence)) or minimum_evidence < 0.0:
            raise ValueError(
                "identity assignment minimum evidence must be finite and nonnegative"
            )
        self.margin = float(margin)
        self.minimum_evidence = float(minimum_evidence)

    def configuration(self) -> dict[str, float | str]:
        return {
            "schema": self.schema,
            "behavior": "opaque-causal-evidence_top-slot_with_explicit_abstention_v1",
            "margin": self.margin,
            "minimum_evidence": self.minimum_evidence,
        }

    def resolve(self, evidence: torch.Tensor) -> ExternalIdentityAssignment:
        if evidence.ndim != 2 or evidence.shape[1] < 1:
            raise ValueError("identity evidence must have shape [batch, slots]")
        if not bool(torch.isfinite(evidence).all()):
            raise ValueError("identity evidence must be finite")
        batch_size, slot_count = evidence.shape
        values, indices = torch.topk(evidence, k=min(2, slot_count), dim=1)
        selected = indices[:, 0].to(dtype=torch.long)
        confidence = torch.softmax(evidence, dim=1).amax(dim=1)
        if slot_count == 1:
            margin = torch.full_like(confidence, float("inf"))
        else:
            margin = values[:, 0] - values[:, 1]
        abstained = (margin < self.margin) | (values[:, 0] < self.minimum_evidence)
        return ExternalIdentityAssignment(
            selected_slot=selected,
            confidence=confidence,
            abstained=abstained,
        ).validate(batch_size=batch_size, slot_count=slot_count)


class ExternalCausalIdentityArtifact:
    """Score action-conditioned dependence in bound learned event histories.

    The artifact is intentionally outside the controller.  It receives only a
    bounded history of learned event tensors and opaque action/intention
    features.  A centered cross-covariance score measures how much each track's
    event change is explained by those features; no coordinates, task labels,
    or verifier outcomes are accepted.
    """

    schema = EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA

    def __init__(self, *, minimum_history: int = 4, epsilon: float = 1e-8) -> None:
        if minimum_history < 2:
            raise ValueError("causal identity history must contain at least two events")
        if not torch.isfinite(torch.tensor(epsilon)) or epsilon <= 0.0:
            raise ValueError("causal identity epsilon must be positive and finite")
        self.minimum_history = int(minimum_history)
        self.epsilon = float(epsilon)

    def configuration(self) -> dict[str, float | int | str]:
        return {
            "schema": self.schema,
            "behavior": "centered-action-event-cross-covariance_v1",
            "minimum_history": self.minimum_history,
            "epsilon": self.epsilon,
        }

    def evidence(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        if event_history.ndim != 4:
            raise ValueError("event history must have shape [batch, time, tracks, width]")
        if action_history.ndim != 3:
            raise ValueError("action history must have shape [batch, time-1, width]")
        batch_size, time_steps, track_count, _ = event_history.shape
        if time_steps < self.minimum_history:
            raise ValueError("causal identity history is shorter than its minimum")
        if action_history.shape[0] != batch_size or action_history.shape[1] != time_steps - 1:
            raise ValueError("action history must align with event transitions")
        if track_count < 1 or action_history.shape[2] < 1:
            raise ValueError("causal identity history needs tracks and action features")
        if not bool(torch.isfinite(event_history).all()) or not bool(
            torch.isfinite(action_history).all()
        ):
            raise ValueError("causal identity histories must be finite")
        event_delta = event_history[:, 1:] - event_history[:, :-1]
        centered_actions = action_history - action_history.mean(dim=1, keepdim=True)
        centered_delta = event_delta - event_delta.mean(dim=1, keepdim=True)
        covariance = torch.einsum(
            "bta,btkd->bkad", centered_actions, centered_delta
        )
        action_energy = centered_actions.square().sum(dim=(1, 2))
        delta_energy = centered_delta.square().sum(dim=(1, 3))
        denominator = action_energy.unsqueeze(1) * delta_energy
        explained = covariance.square().sum(dim=(2, 3)) / denominator.clamp_min(
            self.epsilon
        )
        return explained.clamp_min(0.0).clamp_max(1.0).sqrt()


class PersistentCausalIdentityV2:
    """Persist action-conditioned self dynamics and rebind them to tracks.

    This artifact stores a compact causal signature, not a slot/object index.
    Each new episode is matched against that signature, so object replacement
    and slot permutation do not become persistent identity.  A high-confidence
    episode is the only event that updates the signature.  Contradictions,
    weak applicability, and missing evidence quarantine the model; while
    quarantined, statistics are frozen until fresh high-confidence episodes
    provide a replacement signature.

    The artifact is external to the controller and accepts only learned event
    tensors plus opaque action/intention features.  ``episode_id`` is an
    external lifetime token: passing it makes repeated live ticks idempotent
    for persistent-statistics updates.
    """

    schema = EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V2_SCHEMA

    def __init__(
        self,
        *,
        minimum_history: int = 4,
        margin: float = 0.15,
        minimum_evidence: float = 0.2,
        minimum_similarity: float = 0.65,
        recovery_episodes: int = 2,
        epsilon: float = 1e-8,
    ) -> None:
        if minimum_history < 2:
            raise ValueError("persistent identity history must contain at least two events")
        values = {
            "margin": margin,
            "minimum_evidence": minimum_evidence,
            "minimum_similarity": minimum_similarity,
            "epsilon": epsilon,
        }
        if any(not torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise ValueError("persistent identity parameters must be finite")
        if margin < 0.0 or minimum_evidence < 0.0:
            raise ValueError("persistent identity gates must be nonnegative")
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("persistent identity similarity must lie in [0, 1]")
        if epsilon <= 0.0:
            raise ValueError("persistent identity epsilon must be positive")
        if recovery_episodes < 1:
            raise ValueError("persistent identity recovery needs at least one episode")
        self.minimum_history = int(minimum_history)
        self.margin = float(margin)
        self.minimum_evidence = float(minimum_evidence)
        self.minimum_similarity = float(minimum_similarity)
        self.recovery_episodes = int(recovery_episodes)
        self.epsilon = float(epsilon)
        self.local_artifact = ExternalCausalIdentityArtifact(
            minimum_history=self.minimum_history,
            epsilon=self.epsilon,
        )
        self.assignment_gate = ExternalCausalIdentityAssignment(
            margin=self.margin,
            minimum_evidence=self.minimum_evidence,
        )
        self._prototype: torch.Tensor | None = None
        self._support = 0
        self._status = "uninitialized"
        self._reason = "no_persistent_model"
        self._quarantine_count = 0
        self._recovery_buffer: list[torch.Tensor] = []
        self._committed_episode_id: object | None = None
        self._buffered_episode_ids: set[object] = set()
        self._last_evidence: torch.Tensor | None = None
        self._last_similarity: torch.Tensor | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def support(self) -> int:
        return self._support

    @property
    def quarantine_count(self) -> int:
        return self._quarantine_count

    @property
    def last_evidence(self) -> torch.Tensor | None:
        return None if self._last_evidence is None else self._last_evidence.clone()

    @property
    def last_similarity(self) -> torch.Tensor | None:
        return None if self._last_similarity is None else self._last_similarity.clone()

    def configuration(self) -> dict[str, float | int | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "behavior": "persistent-action-conditioned-dynamics_rebind-quarantine-relearn_v2",
            "minimum_history": self.minimum_history,
            "margin": self.margin,
            "minimum_evidence": self.minimum_evidence,
            "minimum_similarity": self.minimum_similarity,
            "recovery_episodes": self.recovery_episodes,
            "epsilon": self.epsilon,
            "assignment_gate": self.assignment_gate.configuration(),
            "local_artifact": self.local_artifact.configuration(),
        }

    def reset(self) -> None:
        """Forget persistent statistics and return to an uninitialized state."""

        self._prototype = None
        self._support = 0
        self._status = "uninitialized"
        self._reason = "no_persistent_model"
        self._quarantine_count = 0
        self._recovery_buffer.clear()
        self._committed_episode_id = None
        self._buffered_episode_ids.clear()
        self._last_evidence = None
        self._last_similarity = None

    @staticmethod
    def _validate_present(
        event_present: torch.Tensor | None,
        *,
        batch_size: int,
        time_steps: int,
        track_count: int,
    ) -> torch.Tensor:
        if event_present is None:
            return torch.ones(
                batch_size,
                time_steps,
                track_count,
                dtype=torch.bool,
            )
        if event_present.shape != (batch_size, time_steps, track_count):
            raise ValueError("event presence must have shape [batch, time, tracks]")
        if event_present.dtype != torch.bool:
            raise ValueError("event presence must be boolean")
        return event_present

    def _signatures(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized action-conditioned delta signatures [tracks, actions, width]."""

        event_delta = event_history[:, 1:] - event_history[:, :-1]
        centered_actions = action_history - action_history.mean(dim=1, keepdim=True)
        centered_delta = event_delta - event_delta.mean(dim=1, keepdim=True)
        covariance = torch.einsum(
            "bta,btkd->bkad", centered_actions, centered_delta
        )
        action_energy = centered_actions.square().sum(dim=(1, 2))
        signatures = covariance / action_energy[:, None, None, None].clamp_min(
            self.epsilon
        )
        return signatures[0]

    def _empty_assignment(self, slot_count: int) -> ExternalIdentityAssignment:
        evidence = torch.zeros(1, slot_count)
        self._last_evidence = evidence
        return self.assignment_gate.resolve(evidence)

    def _quarantine(self, reason: str) -> None:
        if self._status != "quarantined":
            self._quarantine_count += 1
        self._status = "quarantined"
        self._reason = reason
        self._recovery_buffer.clear()
        self._buffered_episode_ids.clear()

    def _update_prototype(self, signature: torch.Tensor) -> None:
        signature = signature.detach().clone()
        if self._prototype is None:
            self._prototype = signature
        else:
            self._prototype = (
                self._prototype * self._support + signature
            ) / float(self._support + 1)
        self._support += 1
        self._status = "active"
        self._reason = "high_confidence_assignment"

    def _similarities(self, signatures: torch.Tensor) -> torch.Tensor:
        if self._prototype is None:
            return torch.ones(signatures.shape[0])
        prototype = self._prototype.to(device=signatures.device, dtype=signatures.dtype)
        flattened = signatures.flatten(start_dim=1)
        reference = prototype.flatten()
        reference_norm = reference.norm()
        norms = flattened.norm(dim=1)
        denominator = norms * reference_norm
        cosine = (flattened @ reference) / denominator.clamp_min(self.epsilon)
        return cosine.clamp_min(0.0).clamp_max(1.0)

    def resolve(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
        *,
        event_present: torch.Tensor | None = None,
        episode_id: object | None = None,
    ) -> ExternalIdentityAssignment:
        """Match the current tracks and update only after a gated assignment."""

        local_evidence = self.local_artifact.evidence(event_history, action_history)
        batch_size, time_steps, track_count, _ = event_history.shape
        if batch_size != 1:
            raise ValueError("persistent causal identity v2 currently requires batch size one")
        present = self._validate_present(
            event_present,
            batch_size=batch_size,
            time_steps=time_steps,
            track_count=track_count,
        )
        self._last_similarity = torch.zeros(track_count)
        if not bool(present.all()):
            self._last_evidence = torch.zeros_like(local_evidence)
            if self._prototype is not None:
                self._quarantine("missing_evidence")
            else:
                self._reason = "missing_evidence"
            return self._empty_assignment(track_count)

        signatures = self._signatures(event_history, action_history)
        if self._status == "quarantined":
            candidate = self.assignment_gate.resolve(local_evidence)
            self._last_evidence = local_evidence.detach().clone()
            if bool(candidate.abstained[0]):
                self._reason = "recovery_evidence_insufficient"
                self._recovery_buffer.clear()
                self._buffered_episode_ids.clear()
                return self._empty_assignment(track_count)
            if episode_id is None or episode_id not in self._buffered_episode_ids:
                self._recovery_buffer.append(
                    signatures[int(candidate.selected_slot[0].item())].detach().clone()
                )
                if episode_id is not None:
                    self._buffered_episode_ids.add(episode_id)
            if len(self._recovery_buffer) < self.recovery_episodes:
                self._reason = "quarantined_relearning"
                return self._empty_assignment(track_count)
            self._prototype = torch.stack(self._recovery_buffer).mean(dim=0)
            self._support = len(self._recovery_buffer)
            self._status = "active"
            self._reason = "relearned_requires_confirmation"
            self._recovery_buffer.clear()
            self._buffered_episode_ids.clear()
            self._committed_episode_id = episode_id
            return self._empty_assignment(track_count)

        similarities = self._similarities(signatures)
        self._last_similarity = similarities.detach().clone()
        if self._prototype is None:
            evidence = local_evidence
        else:
            evidence = local_evidence * similarities.unsqueeze(0)
        self._last_evidence = evidence.detach().clone()
        assignment = self.assignment_gate.resolve(evidence)
        if bool(assignment.abstained[0]):
            if self._prototype is not None:
                self._quarantine("low_applicability_or_margin")
            else:
                self._reason = "initial_evidence_insufficient"
            return assignment
        selected = int(assignment.selected_slot[0].item())
        if self._prototype is not None and float(similarities[selected]) < self.minimum_similarity:
            self._quarantine("causal_signature_contradiction")
            return self._empty_assignment(track_count)
        if episode_id is None or episode_id != self._committed_episode_id:
            self._update_prototype(signatures[selected])
            self._committed_episode_id = episode_id
        return assignment


__all__ = [
    "EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA",
    "EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA",
    "EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V2_SCHEMA",
    "ExternalCausalIdentityArtifact",
    "ExternalCausalIdentityAssignment",
    "ExternalIdentityAssignment",
    "PersistentCausalIdentityV2",
]
