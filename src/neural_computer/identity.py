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

    def __init__(self, *, margin: float = 0.15) -> None:
        if not torch.isfinite(torch.tensor(margin)) or margin < 0.0:
            raise ValueError("identity assignment margin must be finite and nonnegative")
        self.margin = float(margin)

    def configuration(self) -> dict[str, float | str]:
        return {
            "schema": self.schema,
            "behavior": "opaque-causal-evidence_top-slot_with_explicit_abstention_v1",
            "margin": self.margin,
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
        abstained = margin < self.margin
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


__all__ = [
    "EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA",
    "EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA",
    "ExternalCausalIdentityArtifact",
    "ExternalCausalIdentityAssignment",
    "ExternalIdentityAssignment",
]
