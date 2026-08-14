"""Opaque memory-side routing for verified external executive skills."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .executive_bank import ExternalExecutiveProgramArtifact

if TYPE_CHECKING:
    from .agent_brain_bank import ExternalAgentBrainBank

EXTERNAL_EXECUTIVE_SKILL_ROUTER_SCHEMA = (
    "neural-computer.external-executive-skill-router.v1"
)


def _validate_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return value


def _normalize_context(context: torch.Tensor, *, width: int) -> torch.Tensor:
    if not isinstance(context, torch.Tensor):
        raise TypeError("executive skill route context must be a tensor")
    if context.ndim != 1 or context.shape[0] != width:
        raise ValueError(f"executive skill route context must have shape [{width}]")
    if not bool(torch.isfinite(context).all()):
        raise ValueError("executive skill route context must be finite")
    value = context.detach().to(device="cpu", dtype=torch.float32)
    if float(torch.linalg.vector_norm(value)) <= 1e-8:
        raise ValueError("executive skill route context cannot be zero")
    return F.normalize(value, dim=0).contiguous()


@dataclass(frozen=True)
class ExternalExecutiveSkillSelection:
    """One opaque bank-slot choice with its logged behavior propensity."""

    slot: int
    propensity: float
    context: torch.Tensor
    artifact: ExternalExecutiveProgramArtifact
    bank_digest: str
    bank_version: int
    schema: str = EXTERNAL_EXECUTIVE_SKILL_ROUTER_SCHEMA

    def validate(self, *, context_width: int) -> ExternalExecutiveSkillSelection:
        if self.schema != EXTERNAL_EXECUTIVE_SKILL_ROUTER_SCHEMA:
            raise ValueError("unsupported executive skill selection schema")
        if not isinstance(self.slot, int) or isinstance(self.slot, bool) or self.slot < 0:
            raise ValueError("executive skill selection slot is invalid")
        if not 0.0 < self.propensity <= 1.0:
            raise ValueError("executive skill selection propensity is invalid")
        _normalize_context(self.context, width=context_width)
        if not isinstance(self.artifact, ExternalExecutiveProgramArtifact):
            raise TypeError("executive skill selection artifact is invalid")
        self.artifact.validate()
        _validate_digest(self.bank_digest, name="executive skill selection bank digest")
        if not isinstance(self.bank_version, int) or isinstance(self.bank_version, bool):
            raise TypeError("executive skill selection bank version is invalid")
        if self.bank_version < 0:
            raise ValueError("executive skill selection bank version cannot be negative")
        return self


class ExternalExecutiveSkillRouter:
    """Select and learn among admitted executive artifacts outside the controller."""

    schema = EXTERNAL_EXECUTIVE_SKILL_ROUTER_SCHEMA

    def __init__(
        self,
        bank: ExternalAgentBrainBank,
        *,
        context_width: int,
        matching_tolerance: float = 1e-4,
        generalization_tolerance: float = 0.0,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
        reversal_threshold: float | None = None,
        reversal_patience: int = 4,
    ) -> None:
        from .agent_brain_bank import ExternalAgentBrainBank

        if not isinstance(bank, ExternalAgentBrainBank):
            raise TypeError("executive skill router needs an AgentBrain bank")
        if not isinstance(context_width, int) or isinstance(context_width, bool) or context_width < 1:
            raise ValueError("executive skill router context width must be positive")
        self.bank = bank
        self.context_width = context_width
        resolved_reversal_threshold = (
            mastery_threshold if reversal_threshold is None else reversal_threshold
        )
        self.evidence = bank.executive_route_evidence(
            context_width,
            matching_tolerance=matching_tolerance,
            generalization_tolerance=generalization_tolerance,
            prior_strength=prior_strength,
            mastery_threshold=mastery_threshold,
            min_mastery_observations=min_mastery_observations,
            reversal_threshold=resolved_reversal_threshold,
            reversal_patience=reversal_patience,
        )
        self.unique_outcome_bits = 0

    @property
    def slot_count(self) -> int:
        return self.bank.executive_program_count

    @property
    def bank_digest(self) -> str:
        return self.bank.digest()

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "route_evidence_schema": self.evidence.schema,
            "selection": "opaque_context_balanced_behavior_probability_v1",
            "feedback": "attempted_scalar_outcome_only_v1",
        }

    def select(
        self,
        context: torch.Tensor,
        *,
        exploration: float = 0.0,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> ExternalExecutiveSkillSelection:
        if self.slot_count < 1:
            raise LookupError("executive skill bank is empty")
        key = _normalize_context(context, width=self.context_width)
        probabilities = self.evidence.behavior_probabilities(
            key.unsqueeze(0),
            exploration=exploration,
            strategy="balanced",
        )[0]
        slot = (
            int(torch.multinomial(probabilities, 1, generator=generator).item())
            if sample
            else int(probabilities.argmax().item())
        )
        selection = ExternalExecutiveSkillSelection(
            slot=slot,
            propensity=float(probabilities[slot].item()),
            context=key,
            artifact=self.bank.artifact("executive_program", slot),
            bank_digest=self.bank.digest(),
            bank_version=self.bank.version,
        )
        return selection.validate(context_width=self.context_width)

    def observe(
        self,
        selection: ExternalExecutiveSkillSelection,
        outcome: float | torch.Tensor,
    ) -> None:
        if not isinstance(selection, ExternalExecutiveSkillSelection):
            raise TypeError("executive skill route feedback needs a selection")
        selection.validate(context_width=self.context_width)
        if selection.slot >= self.slot_count:
            raise IndexError("executive skill selection slot is no longer present")
        current_artifact = self.bank.artifact("executive_program", selection.slot)
        if current_artifact.digest() != selection.artifact.digest():
            raise ValueError("executive skill selection artifact no longer matches the bank")
        value = torch.as_tensor(outcome, dtype=torch.float32).reshape(-1)
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or not bool(
            ((value >= 0.0) & (value <= 1.0)).all()
        ):
            raise ValueError("executive skill route outcome must be one scalar in [0, 1]")
        self.bank.observe_executive_route(selection.context, selection.slot, value[0])
        self.unique_outcome_bits += 1

    def state_payload(self) -> dict[str, object]:
        """Return a checksummed route binding for diagnostics and handoff."""

        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "context_width": self.context_width,
            "bank_digest": self.bank.digest(),
            "bank_version": self.bank.version,
            "evidence_digest": self.evidence.digest(),
        }
        unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["sha256"] = hashlib.sha256(unsigned).hexdigest()
        return payload


__all__ = [
    "EXTERNAL_EXECUTIVE_SKILL_ROUTER_SCHEMA",
    "ExternalExecutiveSkillRouter",
    "ExternalExecutiveSkillSelection",
]
