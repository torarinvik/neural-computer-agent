"""Replay-free cost estimates for memory-side prior admission.

The controller does not decide whether an external file is worth reusing.  A
replaceable memory policy makes that decision from opaque context, verified
source coverage, and the current bank size.  This module keeps the policy's
mutable state outside the controller and updates it from the scalar cost of a
completed admission; no task labels or examples are retained.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA = (
    "neural-computer.external-routed-intention-cost-model.v1"
)
EXTERNAL_ROUTED_INTENTION_COST_LEDGER_SCHEMA = (
    "neural-computer.external-routed-intention-cost-ledger.v1"
)
EXTERNAL_ROUTED_INTENTION_COST_OBSERVATION_SCHEMA = (
    "neural-computer.external-routed-intention-cost-observation.v1"
)


def _digest_state(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExternalRoutedIntentionCostModelState:
    """Persistent sufficient statistics for transfer and fresh cost heads."""

    transfer_weights: torch.Tensor
    fresh_weights: torch.Tensor
    transfer_bias: torch.Tensor
    fresh_bias: torch.Tensor
    transfer_observations: torch.Tensor
    fresh_observations: torch.Tensor
    transfer_absolute_error: torch.Tensor
    fresh_absolute_error: torch.Tensor
    schema: str = EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA

    def _tensors(self) -> dict[str, torch.Tensor]:
        return {
            "transfer_weights": self.transfer_weights,
            "fresh_weights": self.fresh_weights,
            "transfer_bias": self.transfer_bias,
            "fresh_bias": self.fresh_bias,
            "transfer_observations": self.transfer_observations,
            "fresh_observations": self.fresh_observations,
            "transfer_absolute_error": self.transfer_absolute_error,
            "fresh_absolute_error": self.fresh_absolute_error,
        }

    def validate(
        self,
        *,
        feature_width: int,
        device: torch.device | None = None,
    ) -> ExternalRoutedIntentionCostModelState:
        if self.schema != EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA:
            raise ValueError("unsupported routed intention cost-model schema")
        if feature_width < 1:
            raise ValueError("routed intention cost-model feature width is invalid")
        expected = {
            "transfer_weights": (feature_width,),
            "fresh_weights": (feature_width,),
            "transfer_bias": (1,),
            "fresh_bias": (1,),
            "transfer_observations": (1,),
            "fresh_observations": (1,),
            "transfer_absolute_error": (1,),
            "fresh_absolute_error": (1,),
        }
        for name, value in self._tensors().items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name]:
                raise ValueError(f"routed intention cost-model {name} has wrong shape")
            if device is not None and value.device != device:
                raise ValueError("routed intention cost-model tensors use different devices")
        for name in (
            "transfer_weights",
            "fresh_weights",
            "transfer_bias",
            "fresh_bias",
            "transfer_absolute_error",
            "fresh_absolute_error",
        ):
            if not bool(torch.isfinite(self._tensors()[name]).all()):
                raise ValueError(f"routed intention cost-model {name} is not finite")
        for name in ("transfer_observations", "fresh_observations"):
            value = self._tensors()[name]
            if value.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"routed intention cost-model {name} must be integer")
            if bool((value < 0).any()):
                raise ValueError(f"routed intention cost-model {name} cannot be negative")
        if bool(
            (self.transfer_absolute_error < 0).any()
            or (self.fresh_absolute_error < 0).any()
        ):
            raise ValueError("routed intention cost-model errors cannot be negative")
        return self


@dataclass(frozen=True)
class ExternalRoutedIntentionCostEstimate:
    """A versioned, auditable estimate supplied to external admission policy."""

    transfer_cost: float
    fresh_cost: float
    transfer_observations: int
    fresh_observations: int
    schema: str = EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA

    def validate(self) -> ExternalRoutedIntentionCostEstimate:
        if self.schema != EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA:
            raise ValueError("unsupported routed intention cost estimate schema")
        for name, value in (
            ("transfer_cost", self.transfer_cost),
            ("fresh_cost", self.fresh_cost),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"routed intention {name} is invalid")
        for name, value in (
            ("transfer_observations", self.transfer_observations),
            ("fresh_observations", self.fresh_observations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"routed intention {name} is invalid")
        return self


@dataclass(frozen=True)
class ExternalRoutedIntentionCostObservationReceipt:
    """Receipt for one replay-free update of the selected cost branch."""

    selected_initialization: str
    observed_cost: float
    predicted_cost: float
    transfer_cost_after: float
    fresh_cost_after: float
    transfer_observations: int
    fresh_observations: int
    schema: str = EXTERNAL_ROUTED_INTENTION_COST_OBSERVATION_SCHEMA

    def validate(self) -> ExternalRoutedIntentionCostObservationReceipt:
        if self.schema != EXTERNAL_ROUTED_INTENTION_COST_OBSERVATION_SCHEMA:
            raise ValueError("unsupported routed intention cost observation schema")
        if self.selected_initialization not in {"transfer", "fresh"}:
            raise ValueError("routed intention cost observation branch is invalid")
        for name, value in (
            ("observed_cost", self.observed_cost),
            ("predicted_cost", self.predicted_cost),
            ("transfer_cost_after", self.transfer_cost_after),
            ("fresh_cost_after", self.fresh_cost_after),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"routed intention cost observation {name} is invalid")
        for name, value in (
            ("transfer_observations", self.transfer_observations),
            ("fresh_observations", self.fresh_observations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"routed intention cost observation {name} is invalid")
        return self


class ExternalRoutedIntentionCostModel:
    """Learn transfer/fresh admission cost from scalar continuation cost.

    The model is deliberately small and linear.  It is not a controller
    branch: it is replaceable memory-side accounting that predicts normalized
    continuation work from opaque context values, an observation mask, source
    coverage, and bank size.  Each completed admission updates only the
    branch that was actually selected, so learning remains replay-free.
    """

    schema = EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA

    def __init__(
        self,
        context_width: int,
        *,
        learning_rate: float = 0.35,
        initial_cost: float = 0.25,
    ) -> None:
        if context_width < 1:
            raise ValueError("routed intention cost-model context width must be positive")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("routed intention cost-model learning rate is invalid")
        if not 0.0 <= initial_cost <= 1.0 or not math.isfinite(initial_cost):
            raise ValueError("routed intention cost-model initial cost is invalid")
        self.context_width = int(context_width)
        self.learning_rate = float(learning_rate)
        self.initial_cost = float(initial_cost)

    @property
    def feature_width(self) -> int:
        return 2 * self.context_width + 2

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "feature_width": self.feature_width,
            "learning_rate": self.learning_rate,
            "initial_cost": self.initial_cost,
            "features": "masked_context_plus_mask_plus_source_coverage_plus_log_bank_size_v1",
            "learning": "normalized_replay_free_selected_branch_lms_v1",
            "ownership": "external_memory_policy_not_controller_state_v1",
        }

    def initial_state(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalRoutedIntentionCostModelState:
        if not dtype.is_floating_point:
            raise TypeError("routed intention cost-model dtype must be floating point")
        zero = torch.zeros(self.feature_width, device=device, dtype=dtype)
        initial = torch.full((1,), self.initial_cost, device=device, dtype=dtype)
        state = ExternalRoutedIntentionCostModelState(
            transfer_weights=zero.clone(),
            fresh_weights=zero.clone(),
            transfer_bias=initial.clone(),
            fresh_bias=initial.clone(),
            transfer_observations=torch.zeros(1, device=device, dtype=torch.int64),
            fresh_observations=torch.zeros(1, device=device, dtype=torch.int64),
            transfer_absolute_error=initial.new_zeros(1),
            fresh_absolute_error=initial.new_zeros(1),
        )
        return state.validate(feature_width=self.feature_width, device=torch.device(device))

    def _features(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        *,
        source_coverage: float,
        cell_count: int,
    ) -> torch.Tensor:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape != (1, self.context_width):
            raise ValueError("routed intention cost-model context has the wrong shape")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("routed intention cost-model context must be finite")
        if context_mask is None:
            context_mask = torch.ones_like(context, dtype=torch.bool)
        if context_mask.shape != context.shape or context_mask.dtype != torch.bool:
            raise ValueError("routed intention cost-model mask is invalid")
        if context_mask.device != context.device:
            raise ValueError("routed intention cost-model mask is on the wrong device")
        if not 0.0 <= source_coverage <= 1.0 or not math.isfinite(source_coverage):
            raise ValueError("routed intention cost-model source coverage is invalid")
        if not isinstance(cell_count, int) or isinstance(cell_count, bool) or cell_count < 1:
            raise ValueError("routed intention cost-model cell count is invalid")
        observed = context * context_mask.to(dtype=context.dtype)
        extras = context.new_tensor(
            [source_coverage, math.log1p(cell_count)],
        ).unsqueeze(0)
        return torch.cat((observed, context_mask.to(dtype=context.dtype), extras), dim=-1)

    def estimate(
        self,
        state: ExternalRoutedIntentionCostModelState,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        source_coverage: float = 1.0,
        cell_count: int = 1,
    ) -> ExternalRoutedIntentionCostEstimate:
        state.validate(feature_width=self.feature_width)
        features = self._features(
            context,
            context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
        )
        transfer = torch.clamp(
            state.transfer_bias + torch.mm(features, state.transfer_weights.unsqueeze(1)).squeeze(1),
            0.0,
            1.0,
        )
        fresh = torch.clamp(
            state.fresh_bias + torch.mm(features, state.fresh_weights.unsqueeze(1)).squeeze(1),
            0.0,
            1.0,
        )
        return ExternalRoutedIntentionCostEstimate(
            transfer_cost=float(transfer.item()),
            fresh_cost=float(fresh.item()),
            transfer_observations=int(state.transfer_observations.item()),
            fresh_observations=int(state.fresh_observations.item()),
        ).validate()

    def observe(
        self,
        state: ExternalRoutedIntentionCostModelState,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        source_coverage: float = 1.0,
        cell_count: int = 1,
        selected_initialization: str,
        observed_cost: float,
    ) -> ExternalRoutedIntentionCostModelState:
        """Update one branch from normalized work, without retaining the sample."""

        state.validate(feature_width=self.feature_width)
        if selected_initialization not in {"transfer", "fresh"}:
            raise ValueError("routed intention cost-model branch is invalid")
        if not 0.0 <= observed_cost <= 1.0 or not math.isfinite(observed_cost):
            raise ValueError("routed intention cost-model observed cost is invalid")
        features = self._features(
            context,
            context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
        ).squeeze(0)
        estimate = self.estimate(
            state,
            context,
            context_mask=context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
        )
        prediction = (
            estimate.transfer_cost
            if selected_initialization == "transfer"
            else estimate.fresh_cost
        )
        error = observed_cost - prediction
        gain = self.learning_rate * error / (1.0 + float(features.square().sum()))
        tensors = {name: value.detach().clone() for name, value in state._tensors().items()}
        prefix = selected_initialization
        tensors[f"{prefix}_weights"] += gain * features
        tensors[f"{prefix}_bias"] += gain
        tensors[f"{prefix}_observations"] += 1
        tensors[f"{prefix}_absolute_error"] += abs(error)
        return ExternalRoutedIntentionCostModelState(**tensors).validate(
            feature_width=self.feature_width,
            device=state.transfer_weights.device,
        )

    def state_payload(self, state: ExternalRoutedIntentionCostModelState) -> dict[str, Any]:
        state.validate(feature_width=self.feature_width)
        normalized = {
            name: value.detach().cpu().clone() for name, value in state._tensors().items()
        }
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": normalized,
            "sha256": _digest_state(normalized),
        }

    def state_from_payload(
        self,
        payload: Mapping[str, Any],
    ) -> ExternalRoutedIntentionCostModelState:
        if not isinstance(payload, Mapping) or payload.get("schema") != self.schema:
            raise ValueError("unsupported routed intention cost-model payload")
        if payload.get("configuration") != self.configuration():
            raise ValueError("routed intention cost-model configuration mismatch")
        raw_state = payload.get("state")
        if not isinstance(raw_state, Mapping):
            raise TypeError("routed intention cost-model state is missing")
        expected_names = (
            "transfer_weights",
            "fresh_weights",
            "transfer_bias",
            "fresh_bias",
            "transfer_observations",
            "fresh_observations",
            "transfer_absolute_error",
            "fresh_absolute_error",
        )
        if tuple(raw_state) != expected_names:
            raise ValueError("routed intention cost-model state names do not match")
        state = ExternalRoutedIntentionCostModelState(**{
            name: value.detach().clone()
            if isinstance(value, torch.Tensor)
            else value
            for name, value in raw_state.items()
        })
        state.validate(feature_width=self.feature_width)
        normalized = {name: value.detach().cpu().clone() for name, value in state._tensors().items()}
        if payload.get("sha256") != _digest_state(normalized):
            raise ValueError("routed intention cost-model checksum mismatch")
        return state


class ExternalRoutedIntentionCostLedger:
    """Shared mutable memory-side state for learned acquisition economics.

    The ledger is separate from the controller and from factual transition
    content. It predicts transfer/fresh continuation cost from an opaque
    candidate context, then updates only the branch selected by a verified
    admission. Streams may share one ledger while retaining their own pending
    evidence and candidates.
    """

    schema = EXTERNAL_ROUTED_INTENTION_COST_LEDGER_SCHEMA

    def __init__(
        self,
        model: ExternalRoutedIntentionCostModel,
        state: ExternalRoutedIntentionCostModelState | None = None,
        *,
        decision_weight: float = 1.0,
    ) -> None:
        if not isinstance(model, ExternalRoutedIntentionCostModel):
            raise TypeError("routed intention cost ledger requires its cost model")
        if state is None:
            state = model.initial_state()
        state.validate(feature_width=model.feature_width)
        if not math.isfinite(decision_weight) or decision_weight < 0.0:
            raise ValueError("routed intention cost ledger decision weight is invalid")
        self.model = model
        self.state = state
        self.decision_weight = float(decision_weight)

    @classmethod
    def create(
        cls,
        context_width: int,
        *,
        learning_rate: float = 0.35,
        initial_cost: float = 0.25,
        decision_weight: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalRoutedIntentionCostLedger:
        model = ExternalRoutedIntentionCostModel(
            context_width,
            learning_rate=learning_rate,
            initial_cost=initial_cost,
        )
        return cls(
            model,
            model.initial_state(device=device, dtype=dtype),
            decision_weight=decision_weight,
        )

    @property
    def context_width(self) -> int:
        return self.model.context_width

    def configuration(self) -> dict[str, int | float | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "model": self.model.configuration(),
            "decision_weight": self.decision_weight,
            "ownership": "shared_external_memory_policy_not_controller_state_v1",
        }

    def estimate(
        self,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        source_coverage: float = 1.0,
        cell_count: int = 1,
    ) -> ExternalRoutedIntentionCostEstimate:
        return self.model.estimate(
            self.state,
            context,
            context_mask=context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
        )

    def observe(
        self,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        source_coverage: float = 1.0,
        cell_count: int = 1,
        selected_initialization: str,
        observed_cost: float,
    ) -> ExternalRoutedIntentionCostObservationReceipt:
        before = self.estimate(
            context,
            context_mask=context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
        )
        self.state = self.model.observe(
            self.state,
            context,
            context_mask=context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
            selected_initialization=selected_initialization,
            observed_cost=observed_cost,
        )
        after = self.estimate(
            context,
            context_mask=context_mask,
            source_coverage=source_coverage,
            cell_count=cell_count,
        )
        predicted = (
            before.transfer_cost
            if selected_initialization == "transfer"
            else before.fresh_cost
        )
        return ExternalRoutedIntentionCostObservationReceipt(
            selected_initialization=selected_initialization,
            observed_cost=float(observed_cost),
            predicted_cost=predicted,
            transfer_cost_after=after.transfer_cost,
            fresh_cost_after=after.fresh_cost,
            transfer_observations=after.transfer_observations,
            fresh_observations=after.fresh_observations,
        ).validate()

    def state_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "model_state": self.model.state_payload(self.state),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalRoutedIntentionCostLedger:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported routed intention cost ledger payload")
        configuration = payload.get("configuration")
        model_payload = payload.get("model_state")
        if not isinstance(configuration, Mapping) or not isinstance(
            model_payload, Mapping
        ):
            raise TypeError("routed intention cost ledger payload is incomplete")
        model_configuration = configuration.get("model")
        if not isinstance(model_configuration, Mapping):
            raise TypeError("routed intention cost ledger model configuration is missing")
        model = ExternalRoutedIntentionCostModel(
            int(model_configuration["context_width"]),
            learning_rate=float(model_configuration["learning_rate"]),
            initial_cost=float(model_configuration["initial_cost"]),
        )
        if dict(model_configuration) != model.configuration():
            raise ValueError("routed intention cost ledger model configuration mismatch")
        state = model.state_from_payload(model_payload)
        ledger = cls(
            model,
            state,
            decision_weight=float(configuration["decision_weight"]),
        )
        if dict(configuration) != ledger.configuration():
            raise ValueError("routed intention cost ledger configuration mismatch")
        return ledger


__all__ = [
    "EXTERNAL_ROUTED_INTENTION_COST_LEDGER_SCHEMA",
    "EXTERNAL_ROUTED_INTENTION_COST_MODEL_SCHEMA",
    "EXTERNAL_ROUTED_INTENTION_COST_OBSERVATION_SCHEMA",
    "ExternalRoutedIntentionCostEstimate",
    "ExternalRoutedIntentionCostLedger",
    "ExternalRoutedIntentionCostModel",
    "ExternalRoutedIntentionCostModelState",
    "ExternalRoutedIntentionCostObservationReceipt",
]
