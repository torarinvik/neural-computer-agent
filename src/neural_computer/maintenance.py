"""Learned finite-budget maintenance for replaceable external memory.

The controller is not a memory allocator.  This module is the narrow policy
boundary that lets external memory learn when to grow, share equivalent
content, compress a retained representation, evict a disposable slot, or
defer.  It deliberately
does not choose semantic slots, inspect raw modalities, or commit mutations:
the caller supplies structural action availability and an independent
verifier remains authoritative at the transaction boundary.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

EXTERNAL_MEMORY_MAINTENANCE_POLICY_SCHEMA = (
    "neural-computer.external-memory-maintenance-policy.v2"
)
EXTERNAL_MEMORY_MAINTENANCE_PROPOSAL_SCHEMA = (
    "neural-computer.external-memory-maintenance-proposal.v2"
)
_LEGACY_POLICY_SCHEMA = "neural-computer.external-memory-maintenance-policy.v1"
_LEGACY_ACTIONS = ("grow", "share", "compress", "defer")
MAINTENANCE_ACTIONS = ("grow", "share", "compress", "evict", "defer")
MAINTENANCE_FEATURE_WIDTH = 12


def _scalar(value: torch.Tensor | float, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        values = value.detach().reshape(-1)
        if values.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        result = float(values[0])
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _state_digest(
    configuration: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(repr(dict(configuration)).encode("utf-8"))
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalMemoryMaintenanceProposal:
    """One policy proposal, before any external-memory mutation."""

    action: str
    action_index: int
    logits: torch.Tensor
    features: torch.Tensor
    available_actions: torch.Tensor
    selected_probability: float
    selected_propensity: float
    selection_mode: str
    reason: str
    schema: str = EXTERNAL_MEMORY_MAINTENANCE_PROPOSAL_SCHEMA

    def validate(self) -> ExternalMemoryMaintenanceProposal:
        if self.schema != EXTERNAL_MEMORY_MAINTENANCE_PROPOSAL_SCHEMA:
            raise ValueError("unsupported external-memory maintenance proposal schema")
        if self.action not in MAINTENANCE_ACTIONS:
            raise ValueError("external-memory maintenance action is unknown")
        if not 0 <= self.action_index < len(MAINTENANCE_ACTIONS):
            raise ValueError("external-memory maintenance action index is invalid")
        if self.action != MAINTENANCE_ACTIONS[self.action_index]:
            raise ValueError("external-memory maintenance action/index disagree")
        if self.logits.shape != (len(MAINTENANCE_ACTIONS),):
            raise ValueError("external-memory maintenance logits are mis-shaped")
        if self.features.shape != (MAINTENANCE_FEATURE_WIDTH,):
            raise ValueError("external-memory maintenance features are mis-shaped")
        if self.available_actions.shape != (len(MAINTENANCE_ACTIONS),):
            raise ValueError("external-memory maintenance availability is mis-shaped")
        if self.available_actions.dtype != torch.bool:
            raise TypeError("external-memory maintenance availability must be bool")
        if not bool(torch.isfinite(self.logits).all()) or not bool(
            torch.isfinite(self.features).all()
        ):
            raise ValueError("external-memory maintenance proposal is non-finite")
        if not bool(self.available_actions[self.action_index]):
            raise ValueError("external-memory maintenance action is unavailable")
        if not math.isfinite(self.selected_probability) or not 0.0 < self.selected_probability <= 1.0:
            raise ValueError("external-memory maintenance probability is invalid")
        if not math.isfinite(self.selected_propensity) or not 0.0 < self.selected_propensity <= 1.0:
            raise ValueError("external-memory maintenance propensity is invalid")
        if self.selection_mode not in {"greedy", "sampled"}:
            raise ValueError("external-memory maintenance selection mode is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("external-memory maintenance proposal reason is missing")
        return self


class ExternalMemoryMaintenancePolicy(nn.Module):
    """Learn a discrete maintenance choice from one scalar verifier outcome.

    ``features`` are generic storage facts only.  The action mask is supplied
    by the external memory implementation and may disable an operation that
    is not structurally possible.  Updates consume exactly one scalar outcome
    and retain no evidence, making this a replay-free, independently
    replaceable memory-side learner.
    """

    schema = EXTERNAL_MEMORY_MAINTENANCE_POLICY_SCHEMA

    def __init__(
        self,
        *,
        hidden_width: int = 32,
        learning_rate: float = 1e-2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_width < 1:
            raise ValueError("external-memory maintenance hidden width must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("external-memory maintenance learning rate is invalid")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("external-memory maintenance temperature is invalid")
        self.hidden_width = int(hidden_width)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(MAINTENANCE_FEATURE_WIDTH, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, len(MAINTENANCE_ACTIONS)),
        )

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "hidden_width": self.hidden_width,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "actions": MAINTENANCE_ACTIONS,
            "features": (
                "capacity_pressure",
                "logical_slot_fraction",
                "physical_slot_fraction",
                "alias_fraction",
                "mean_usage",
                "mean_age",
                "mean_prediction_error",
                "max_prediction_error",
                "binding_pressure",
                "provisional_pressure",
                "redundancy_pressure",
                "compression_opportunity",
            ),
            "selection": "masked_discrete_external_action_v1",
            "updates": "single_scalar_verifier_utility_without_replay_v1",
            "commit": "caller_owned_copy_on_write_retention_verification_v1",
        }

    @staticmethod
    def _validate_inputs(
        features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        if features.shape != (MAINTENANCE_FEATURE_WIDTH,):
            raise ValueError(
                f"maintenance features must have shape [{MAINTENANCE_FEATURE_WIDTH}]"
            )
        if action_mask.shape != (len(MAINTENANCE_ACTIONS),):
            raise ValueError("maintenance action mask has invalid shape")
        if action_mask.dtype != torch.bool:
            raise TypeError("maintenance action mask must be bool")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("maintenance features must be finite")
        if not bool(action_mask.any()):
            raise ValueError("maintenance action mask must leave one action available")

    @torch.no_grad()
    def propose(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> ExternalMemoryMaintenanceProposal:
        """Select one masked action, optionally by logged exploration."""

        if not isinstance(sample, bool):
            raise TypeError("maintenance proposal sample flag must be bool")
        self._validate_inputs(features, action_mask)
        logits = self.network(features)
        available = action_mask.detach().clone()
        masked_logits = logits.masked_fill(~available, -torch.inf)
        probabilities = torch.softmax(masked_logits / self.temperature, dim=-1)
        if sample:
            action_index = int(
                torch.multinomial(probabilities, 1, generator=generator)[0]
            )
            selection_mode = "sampled"
            propensity = float(probabilities[action_index])
        else:
            action_index = int(masked_logits.argmax())
            selection_mode = "greedy"
            propensity = 1.0
        return ExternalMemoryMaintenanceProposal(
            action=MAINTENANCE_ACTIONS[action_index],
            action_index=action_index,
            logits=logits.detach().clone(),
            features=features.detach().clone(),
            available_actions=available,
            selected_probability=float(probabilities[action_index]),
            selected_propensity=propensity,
            selection_mode=selection_mode,
            reason="masked external-memory maintenance action selected",
        ).validate()

    def adaptation_step(
        self,
        proposal: ExternalMemoryMaintenanceProposal,
        verifier_utility: torch.Tensor | float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Update from one verifier utility bit without replay."""

        proposal.validate()
        utility = _scalar(verifier_utility, name="maintenance verifier utility")
        if not 0.0 <= utility <= 1.0:
            raise ValueError("maintenance verifier utility must lie in [0, 1]")
        logits = self.network(proposal.features)
        masked_logits = logits.masked_fill(~proposal.available_actions, -torch.inf)
        log_probability = torch.log_softmax(masked_logits, dim=-1)[
            proposal.action_index
        ]
        advantage = utility - 0.5
        loss = -advantage * log_probability / max(proposal.selected_propensity, 1e-3)
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())

    def state_payload(self) -> dict[str, object]:
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
        for name, value in state.items():
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.contiguous().numpy().tobytes())
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": digest.hexdigest(),
        }

    def digest(self) -> str:
        return str(self.state_payload()["sha256"])

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalMemoryMaintenancePolicy:
        if not isinstance(payload, Mapping):
            raise TypeError("external-memory maintenance policy payload is not a mapping")
        payload_schema = payload.get("schema")
        if payload_schema not in {cls.schema, _LEGACY_POLICY_SCHEMA}:
            raise ValueError("unsupported external-memory maintenance policy payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("external-memory maintenance policy payload is incomplete")
        policy = cls(
            hidden_width=int(configuration["hidden_width"]),
            learning_rate=float(configuration["learning_rate"]),
            temperature=float(configuration["temperature"]),
        )
        current = policy.state_dict()
        if payload_schema == cls.schema:
            if dict(configuration) != policy.configuration():
                raise ValueError(
                    "external-memory maintenance policy configuration differs"
                )
            if set(state) != set(current):
                raise ValueError(
                    "external-memory maintenance policy state names differ"
                )
        else:
            if tuple(configuration.get("actions", ())) != _LEGACY_ACTIONS:
                raise ValueError("unsupported legacy maintenance action set")
            if payload.get("sha256") != _state_digest(configuration, state):
                raise ValueError("legacy maintenance policy checksum mismatch")
            expected_names = set(current)
            if set(state) != expected_names:
                raise ValueError("legacy maintenance policy state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("external-memory maintenance policy state is not tensor")
            if payload_schema == _LEGACY_POLICY_SCHEMA and name == "network.2.weight":
                if value.shape != (len(_LEGACY_ACTIONS), expected.shape[1]):
                    raise ValueError("legacy maintenance policy output state is incompatible")
                value = torch.cat((value, torch.zeros_like(expected[4:5])), dim=0)
            elif payload_schema == _LEGACY_POLICY_SCHEMA and name == "network.2.bias":
                if value.shape != (len(_LEGACY_ACTIONS),):
                    raise ValueError("legacy maintenance policy output bias is incompatible")
                value = torch.cat((value, torch.zeros_like(expected[4:5])), dim=0)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("external-memory maintenance policy state is incompatible")
            normalized[name] = value.detach().clone()
        policy.load_state_dict(normalized, strict=True)
        if payload_schema == cls.schema and payload.get("sha256") != policy.state_payload()["sha256"]:
            raise ValueError("external-memory maintenance policy checksum mismatch")
        return policy


__all__ = [
    "EXTERNAL_MEMORY_MAINTENANCE_POLICY_SCHEMA",
    "EXTERNAL_MEMORY_MAINTENANCE_PROPOSAL_SCHEMA",
    "MAINTENANCE_ACTIONS",
    "MAINTENANCE_FEATURE_WIDTH",
    "ExternalMemoryMaintenancePolicy",
    "ExternalMemoryMaintenanceProposal",
]
