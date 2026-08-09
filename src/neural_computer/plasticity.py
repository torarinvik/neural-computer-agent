"""Trainer-only utilities for protected continual learning.

These helpers never enter the deployed controller boundary. A trainer may
accumulate gradients from verified rehearsal experiences, then remove only the
component of a new target update that would oppose that protected direction.
The operation is task-agnostic: it consumes parameter names and gradients,
not task IDs, semantic labels, or correct unattempted actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

NamedParameters = Sequence[tuple[str, torch.nn.Parameter]]
GradientMap = Mapping[str, torch.Tensor]

EXTERNAL_FAST_WEIGHT_SCHEMA = (
    "neural-computer.external-fast-weight-plasticity.v1"
)
EXTERNAL_OUTCOME_CREDIT_SCHEMA = (
    "neural-computer.external-outcome-credit-plasticity.v1"
)
EXTERNAL_OUTCOME_VALUE_SCHEMA = (
    "neural-computer.external-outcome-value-baseline.v1"
)


@dataclass(frozen=True)
class ExternalFastWeightState:
    """External per-capability state for one fast associative computation.

    ``weights`` and ``updates`` are state, not parameters of the controller or
    of the plasticity rule.  A memory backend may keep one state per opaque
    capability, append new states, or serialize them as independent files.
    """

    weights: torch.Tensor
    updates: torch.Tensor

    def validate(self, *, key_width: int, value_width: int) -> None:
        if self.weights.ndim != 3 or self.weights.shape[1:] != (
            key_width,
            value_width,
        ):
            raise ValueError("fast-weight state has the wrong weight shape")
        if self.updates.ndim != 1 or self.updates.shape[0] != self.weights.shape[0]:
            raise ValueError("fast-weight state has the wrong update shape")
        if self.updates.dtype not in (torch.int32, torch.int64):
            raise TypeError("fast-weight update counts must be integer tensors")
        if self.updates.device != self.weights.device:
            raise ValueError("fast-weight state tensors must share a device")
        if not bool(torch.isfinite(self.weights).all()):
            raise ValueError("fast-weight state must contain finite weights")
        if bool((self.updates < 0).any()):
            raise ValueError("fast-weight update counts cannot be negative")


class ExternalFastWeightPlasticity(nn.Module):
    """Learnable, outcome-gated delta-rule plasticity outside the controller.

    The rule reads an opaque query from an external fast-weight matrix and
    updates only that external state from an opaque value and a deterministic
    scalar outcome.  It is a bounded associative computation primitive: a
    successful value is written with a delta rule, while failed or missing
    evidence leaves the stored computation unchanged.  The learned write gate
    is independently replaceable and can be meta-trained without changing the
    frozen controller.
    """

    schema = EXTERNAL_FAST_WEIGHT_SCHEMA

    def __init__(
        self,
        key_width: int,
        value_width: int,
        *,
        hidden: int = 32,
        initial_learning_rate: float = 1.0,
    ) -> None:
        super().__init__()
        if min(key_width, value_width, hidden) < 1:
            raise ValueError("fast-weight dimensions must be positive")
        if not 0.0 < initial_learning_rate <= 1.0:
            raise ValueError("initial fast-weight learning rate must lie in (0, 1]")
        self.key_width = int(key_width)
        self.value_width = int(value_width)
        self.hidden = int(hidden)
        self.feature_width = key_width + 2 * value_width + 1
        self.write_gate = nn.Sequential(
            nn.Linear(self.feature_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start from an almost-open generic write gate.  Meta-training may
        # learn selective plasticity, but a new state is immediately useful.
        nn.init.zeros_(self.write_gate[-1].weight)
        nn.init.constant_(
            self.write_gate[-1].bias,
            float(torch.logit(torch.tensor(0.99))),
        )
        learning_rate = torch.tensor(float(initial_learning_rate)).clamp(
            1e-4, 1.0 - 1e-4
        )
        self.learning_rate_logit = nn.Parameter(torch.logit(learning_rate))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "key_width": self.key_width,
            "value_width": self.value_width,
            "hidden": self.hidden,
            "feature_width": self.feature_width,
            "update_rule": "outcome_gated_normalized_delta_fast_weight_v1",
            "state": "external_per_capability_tensor_state_v1",
            "learning_rate": float(self.learning_rate.detach().sigmoid()),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalFastWeightState:
        if batch_size < 1:
            raise ValueError("fast-weight batch size must be positive")
        return ExternalFastWeightState(
            weights=torch.zeros(
                batch_size,
                self.key_width,
                self.value_width,
                device=device,
                dtype=dtype,
            ),
            updates=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @property
    def learning_rate(self) -> torch.Tensor:
        return self.learning_rate_logit.sigmoid()

    def _validate_query(
        self, query: torch.Tensor, *, batch_size: int | None = None
    ) -> None:
        if query.ndim != 2 or query.shape[1] != self.key_width:
            raise ValueError("fast-weight query has the wrong shape")
        if batch_size is not None and query.shape[0] != batch_size:
            raise ValueError("fast-weight query batch does not match state")
        if not bool(torch.isfinite(query).all()):
            raise ValueError("fast-weight query must be finite")

    def _validate_value(
        self, value: torch.Tensor, *, batch_size: int | None = None
    ) -> None:
        if value.ndim != 2 or value.shape[1] != self.value_width:
            raise ValueError("fast-weight value has the wrong shape")
        if batch_size is not None and value.shape[0] != batch_size:
            raise ValueError("fast-weight value batch does not match state")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("fast-weight value must be finite")

    @staticmethod
    def _validate_outcome(outcome: torch.Tensor, batch_size: int) -> None:
        if outcome.ndim != 1 or outcome.shape[0] != batch_size:
            raise ValueError("fast-weight outcome must have shape [batch]")
        if not bool(torch.isfinite(outcome).all()):
            raise ValueError("fast-weight outcome must be finite")
        if bool(((outcome < 0.0) | (outcome > 1.0)).any()):
            raise ValueError("fast-weight outcome must lie in [0, 1]")

    def read(
        self,
        state: ExternalFastWeightState,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Read one opaque value from external state without mutating it."""

        state.validate(key_width=self.key_width, value_width=self.value_width)
        self._validate_query(query, batch_size=state.weights.shape[0])
        normalized_query = F.normalize(query, dim=-1, eps=1e-8)
        return torch.einsum("bk,bkv->bv", normalized_query, state.weights)

    def update(
        self,
        state: ExternalFastWeightState,
        query: torch.Tensor,
        value: torch.Tensor,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalFastWeightState:
        """Apply one outcome-only external-memory update.

        A positive outcome writes the value toward the current read using a
        normalized delta rule.  A zero outcome or absent evidence produces an
        exactly unchanged weight matrix, which makes retention measurable
        without replaying the old examples.
        """

        state.validate(key_width=self.key_width, value_width=self.value_width)
        batch_size = state.weights.shape[0]
        self._validate_query(query, batch_size=batch_size)
        self._validate_value(value, batch_size=batch_size)
        self._validate_outcome(outcome, batch_size)
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=query.device)
        if present.ndim != 1 or present.shape[0] != batch_size:
            raise ValueError("fast-weight presence must have shape [batch]")
        if present.dtype is not torch.bool:
            raise TypeError("fast-weight presence must be boolean")
        normalized_query = F.normalize(query, dim=-1, eps=1e-8)
        current = torch.einsum("bk,bkv->bv", normalized_query, state.weights)
        features = torch.cat(
            (query, value, current, outcome.unsqueeze(-1)), dim=-1
        )
        gate = torch.sigmoid(self.write_gate(features)).squeeze(-1)
        strength = gate * outcome * present.to(dtype=query.dtype)
        error = value - current
        delta = torch.einsum("bk,bv->bkv", normalized_query, error)
        delta = self.learning_rate.to(dtype=delta.dtype) * strength[:, None, None] * delta
        next_weights = state.weights + delta
        next_updates = state.updates + present.to(device=state.updates.device).long()
        next_state = ExternalFastWeightState(next_weights, next_updates)
        next_state.validate(key_width=self.key_width, value_width=self.value_width)
        return next_state

    def state_payload(self, state: ExternalFastWeightState) -> dict[str, object]:
        """Return a tensor-only versioned payload for an external memory file."""

        state.validate(key_width=self.key_width, value_width=self.value_width)
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "weights": state.weights.detach().cpu().clone(),
            "updates": state.updates.detach().cpu().clone(),
        }

    def state_from_payload(self, payload: Mapping[str, object]) -> ExternalFastWeightState:
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported fast-weight state schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("fast-weight state configuration is invalid")
        if (
            configuration.get("key_width") != self.key_width
            or configuration.get("value_width") != self.value_width
        ):
            raise ValueError("fast-weight state dimensions do not match")
        weights = payload.get("weights")
        updates = payload.get("updates")
        if not isinstance(weights, torch.Tensor) or not isinstance(
            updates, torch.Tensor
        ):
            raise TypeError("fast-weight state payload must contain tensors")
        state = ExternalFastWeightState(weights, updates)
        state.validate(key_width=self.key_width, value_width=self.value_width)
        return state


@dataclass(frozen=True)
class ExternalOutcomeCreditState:
    """External policy and eligibility state for delayed scalar outcomes.

    ``policy`` is the mutable capability-specific computation.  ``eligibility``
    is transient episode state that carries credit across time; neither tensor
    belongs to the frozen controller or to the plasticity-rule parameters.
    Counters are kept separately so accounting can distinguish decisions from
    received feedback without storing examples for replay.
    """

    policy: torch.Tensor
    eligibility: torch.Tensor
    baseline: torch.Tensor
    decisions: torch.Tensor
    feedbacks: torch.Tensor

    def validate(self, *, feature_width: int, action_count: int) -> None:
        expected = (feature_width, action_count)
        if self.policy.ndim != 3 or self.policy.shape[1:] != expected:
            raise ValueError("outcome-credit policy state has the wrong shape")
        if self.eligibility.shape != self.policy.shape:
            raise ValueError("outcome-credit eligibility has the wrong shape")
        batch_size = self.policy.shape[0]
        if self.baseline.shape != (batch_size,):
            raise ValueError("outcome-credit baseline has the wrong shape")
        for name, value in (
            ("decisions", self.decisions),
            ("feedbacks", self.feedbacks),
        ):
            if value.shape != (batch_size,):
                raise ValueError(f"outcome-credit {name} has the wrong shape")
            if value.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"outcome-credit {name} must be integer")
            if bool((value < 0).any()):
                raise ValueError(f"outcome-credit {name} cannot be negative")
        if self.baseline.device != self.policy.device:
            raise ValueError("outcome-credit state tensors must share a device")
        if self.eligibility.device != self.policy.device:
            raise ValueError("outcome-credit state tensors must share a device")
        if self.decisions.device != self.policy.device:
            raise ValueError("outcome-credit state tensors must share a device")
        if self.feedbacks.device != self.policy.device:
            raise ValueError("outcome-credit state tensors must share a device")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (self.policy, self.eligibility, self.baseline)
        ):
            raise ValueError("outcome-credit state must contain finite values")
        if bool(((self.baseline < 0.0) | (self.baseline > 1.0)).any()):
            raise ValueError("outcome-credit baseline must lie in [0, 1]")


class ExternalOutcomeCreditPlasticity(nn.Module):
    """Learn an external policy from delayed scalar outcomes.

    The rule is an eligibility-trace policy-gradient primitive.  It receives
    only a learned feature tensor, an opaque sampled choice, its exact logging
    propensity, and a deterministic scalar outcome.  The policy update is
    applied to external state, not to the controller or to this rule's
    parameters.  A terminal feedback clears the trace after credit is applied.

    This is deliberately a learning primitive, not a task solver: it does not
    receive correct unattempted choices, task IDs, semantic labels, or raw
    modality data.  A caller still needs a verifier and a memory-side policy
    for deciding how to allocate or compose capabilities.
    """

    schema = EXTERNAL_OUTCOME_CREDIT_SCHEMA

    def __init__(
        self,
        feature_width: int,
        action_count: int,
        *,
        initial_learning_rate: float = 0.1,
        initial_trace_decay: float = 0.9,
        initial_baseline_rate: float = 0.05,
        initial_baseline: float = 0.5,
    ) -> None:
        super().__init__()
        if min(feature_width, action_count) < 1:
            raise ValueError("outcome-credit dimensions must be positive")
        if not 0.0 < initial_learning_rate <= 1.0:
            raise ValueError("outcome-credit learning rate must lie in (0, 1]")
        if not 0.0 <= initial_trace_decay < 1.0:
            raise ValueError("outcome-credit trace decay must lie in [0, 1)")
        if not 0.0 < initial_baseline_rate <= 1.0:
            raise ValueError("outcome-credit baseline rate must lie in (0, 1]")
        if not 0.0 <= initial_baseline <= 1.0:
            raise ValueError("outcome-credit initial baseline must lie in [0, 1]")
        self.feature_width = int(feature_width)
        self.action_count = int(action_count)
        self.initial_baseline = float(initial_baseline)
        self._trace_decay_exact_zero = initial_trace_decay == 0.0
        self.learning_rate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(initial_learning_rate)).clamp(1e-4, 1.0 - 1e-4))
        )
        self.trace_decay_logit = nn.Parameter(
            torch.tensor(0.0)
            if self._trace_decay_exact_zero
            else torch.logit(torch.tensor(float(initial_trace_decay)).clamp(1e-4, 1.0 - 1e-4))
        )
        self.baseline_rate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(initial_baseline_rate)).clamp(1e-4, 1.0 - 1e-4))
        )

    @property
    def learning_rate(self) -> torch.Tensor:
        return self.learning_rate_logit.sigmoid()

    @property
    def trace_decay(self) -> torch.Tensor:
        if self._trace_decay_exact_zero:
            return torch.zeros_like(self.trace_decay_logit)
        return self.trace_decay_logit.sigmoid()

    @property
    def baseline_rate(self) -> torch.Tensor:
        return self.baseline_rate_logit.sigmoid()

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "feature_width": self.feature_width,
            "action_count": self.action_count,
            "update_rule": "importance_weighted_delayed_policy_gradient_v1",
            "trace": "decayed_log_probability_gradient_v1",
            "state": "external_capability_policy_and_eligibility_v1",
            "learning_rate": float(self.learning_rate.detach()),
            "trace_decay": float(self.trace_decay.detach()),
            "baseline_rate": float(self.baseline_rate.detach()),
            "initial_baseline": self.initial_baseline,
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalOutcomeCreditState:
        if batch_size < 1:
            raise ValueError("outcome-credit batch size must be positive")
        policy = torch.zeros(
            batch_size,
            self.feature_width,
            self.action_count,
            device=device,
            dtype=dtype,
        )
        return ExternalOutcomeCreditState(
            policy=policy,
            eligibility=torch.zeros_like(policy),
            baseline=torch.full(
                (batch_size,),
                self.initial_baseline,
                device=device,
                dtype=dtype,
            ),
            decisions=torch.zeros(batch_size, device=device, dtype=torch.long),
            feedbacks=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _validate_features(
        self,
        features: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError("outcome-credit features have the wrong shape")
        if batch_size is not None and features.shape[0] != batch_size:
            raise ValueError("outcome-credit feature batch does not match state")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("outcome-credit features must be finite")

    @staticmethod
    def _validate_presence(
        present: torch.Tensor,
        batch_size: int,
        *,
        device: torch.device,
    ) -> None:
        if present.shape != (batch_size,) or present.dtype is not torch.bool:
            raise ValueError("outcome-credit presence must be boolean [batch]")
        if present.device != device:
            raise ValueError("outcome-credit presence is on the wrong device")

    def _validate_state(self, state: ExternalOutcomeCreditState) -> None:
        state.validate(
            feature_width=self.feature_width,
            action_count=self.action_count,
        )

    def logits(
        self,
        state: ExternalOutcomeCreditState,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Read the current external policy without mutating state."""

        self._validate_state(state)
        self._validate_features(features, batch_size=state.policy.shape[0])
        return torch.einsum("bf,bfa->ba", features, state.policy)

    def record_decision(
        self,
        state: ExternalOutcomeCreditState,
        features: torch.Tensor,
        choice: torch.Tensor,
        propensity: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalOutcomeCreditState:
        """Append one choice's log-probability gradient to the trace."""

        self._validate_state(state)
        batch_size = state.policy.shape[0]
        self._validate_features(features, batch_size=batch_size)
        if choice.shape != (batch_size,) or choice.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("outcome-credit choices must be integer [batch]")
        if bool((choice < 0).any()) or bool((choice >= self.action_count).any()):
            raise ValueError("outcome-credit choice is outside the action space")
        if propensity.shape != (batch_size,) or not bool(torch.isfinite(propensity).all()):
            raise ValueError("outcome-credit propensity must be finite [batch]")
        if bool((propensity <= 0.0).any()) or bool((propensity > 1.0).any()):
            raise ValueError("outcome-credit propensity must lie in (0, 1]")
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=features.device)
        self._validate_presence(present, batch_size, device=features.device)
        probabilities = self.logits(state, features).softmax(dim=-1)
        selected_probability = probabilities.gather(1, choice.to(torch.long).unsqueeze(-1)).squeeze(-1)
        importance = selected_probability / propensity
        one_hot = F.one_hot(choice.to(torch.long), self.action_count).to(
            dtype=features.dtype
        )
        score_gradient = features.unsqueeze(-1) * (one_hot - probabilities)
        trace_decay = self.trace_decay.to(dtype=features.dtype)
        next_eligibility = (
            trace_decay * state.eligibility
            + importance[:, None, None] * score_gradient
        )
        active = present[:, None, None]
        next_eligibility = torch.where(active, next_eligibility, state.eligibility)
        next_decisions = state.decisions + present.to(device=state.decisions.device).long()
        next_state = ExternalOutcomeCreditState(
            policy=state.policy,
            eligibility=next_eligibility,
            baseline=state.baseline,
            decisions=next_decisions,
            feedbacks=state.feedbacks,
        )
        self._validate_state(next_state)
        return next_state

    def apply_feedback(
        self,
        state: ExternalOutcomeCreditState,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
        baseline_override: torch.Tensor | None = None,
    ) -> ExternalOutcomeCreditState:
        """Apply scalar feedback to all eligible decisions in the trace."""

        self._validate_state(state)
        batch_size = state.policy.shape[0]
        if outcome.shape != (batch_size,) or not bool(torch.isfinite(outcome).all()):
            raise ValueError("outcome-credit outcome must be finite [batch]")
        if bool(((outcome < 0.0) | (outcome > 1.0)).any()):
            raise ValueError("outcome-credit outcome must lie in [0, 1]")
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=outcome.device)
        self._validate_presence(present, batch_size, device=outcome.device)
        if terminal is None:
            terminal = torch.zeros(batch_size, dtype=torch.bool, device=outcome.device)
        self._validate_presence(terminal, batch_size, device=outcome.device)
        if baseline_override is not None:
            if (
                baseline_override.shape != (batch_size,)
                or not bool(torch.isfinite(baseline_override).all())
            ):
                raise ValueError("outcome-credit baseline override must be finite [batch]")
            if bool(((baseline_override < 0.0) | (baseline_override > 1.0)).any()):
                raise ValueError("outcome-credit baseline override must lie in [0, 1]")
            if baseline_override.device != outcome.device:
                raise ValueError("outcome-credit baseline override is on the wrong device")
        active = present[:, None, None]
        centered = outcome - (
            state.baseline if baseline_override is None else baseline_override
        )
        update = (
            self.learning_rate.to(dtype=state.policy.dtype)
            * centered[:, None, None]
            * state.eligibility
        )
        updated_policy = torch.where(active, state.policy + update, state.policy)
        baseline_rate = self.baseline_rate.to(dtype=state.baseline.dtype)
        updated_baseline = state.baseline + baseline_rate * (outcome - state.baseline)
        updated_baseline = torch.where(present, updated_baseline, state.baseline)
        clear = terminal[:, None, None] & active
        next_eligibility = torch.where(
            clear,
            torch.zeros_like(state.eligibility),
            state.eligibility,
        )
        next_feedbacks = state.feedbacks + present.to(device=state.feedbacks.device).long()
        next_state = ExternalOutcomeCreditState(
            policy=updated_policy,
            eligibility=next_eligibility,
            baseline=updated_baseline,
            decisions=state.decisions,
            feedbacks=next_feedbacks,
        )
        self._validate_state(next_state)
        return next_state

    def begin_episode(
        self,
        state: ExternalOutcomeCreditState,
    ) -> ExternalOutcomeCreditState:
        """Clear transient credit while preserving learned policy and baseline."""

        self._validate_state(state)
        next_state = ExternalOutcomeCreditState(
            policy=state.policy,
            eligibility=torch.zeros_like(state.eligibility),
            baseline=state.baseline,
            decisions=state.decisions,
            feedbacks=state.feedbacks,
        )
        self._validate_state(next_state)
        return next_state

    def state_payload(self, state: ExternalOutcomeCreditState) -> dict[str, object]:
        """Return a tensor-only versioned payload for external persistence."""

        self._validate_state(state)
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "policy": state.policy.detach().cpu().clone(),
            "eligibility": state.eligibility.detach().cpu().clone(),
            "baseline": state.baseline.detach().cpu().clone(),
            "decisions": state.decisions.detach().cpu().clone(),
            "feedbacks": state.feedbacks.detach().cpu().clone(),
        }

    def state_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> ExternalOutcomeCreditState:
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported outcome-credit state schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("outcome-credit state configuration is invalid")
        if (
            configuration.get("feature_width") != self.feature_width
            or configuration.get("action_count") != self.action_count
        ):
            raise ValueError("outcome-credit state dimensions do not match")
        tensors: dict[str, torch.Tensor] = {}
        for name in ("policy", "eligibility", "baseline", "decisions", "feedbacks"):
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"outcome-credit payload field {name!r} must be a tensor")
            tensors[name] = value
        state = ExternalOutcomeCreditState(**tensors)
        self._validate_state(state)
        return state


@dataclass(frozen=True)
class ExternalOutcomeValueState:
    """External feature-conditioned baseline state for delayed outcomes."""

    weights: torch.Tensor
    eligibility: torch.Tensor
    bias: torch.Tensor
    prediction_trace: torch.Tensor
    trace_mass: torch.Tensor
    decisions: torch.Tensor
    feedbacks: torch.Tensor

    def validate(self, *, feature_width: int) -> None:
        if self.weights.ndim != 2 or self.weights.shape[1] != feature_width:
            raise ValueError("outcome-value weights have the wrong shape")
        if self.eligibility.shape != self.weights.shape:
            raise ValueError("outcome-value eligibility has the wrong shape")
        batch_size = self.weights.shape[0]
        for name, value in (
            ("bias", self.bias),
            ("prediction_trace", self.prediction_trace),
            ("trace_mass", self.trace_mass),
            ("decisions", self.decisions),
            ("feedbacks", self.feedbacks),
        ):
            if value.shape != (batch_size,):
                raise ValueError(f"outcome-value {name} has the wrong shape")
        for name, value in (
            ("decisions", self.decisions),
            ("feedbacks", self.feedbacks),
        ):
            if value.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"outcome-value {name} must be integer")
            if bool((value < 0).any()):
                raise ValueError(f"outcome-value {name} cannot be negative")
        if bool((self.trace_mass < 0.0).any()):
            raise ValueError("outcome-value trace mass cannot be negative")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                self.weights,
                self.eligibility,
                self.bias,
                self.prediction_trace,
                self.trace_mass,
            )
        ):
            raise ValueError("outcome-value state must contain finite values")
        for value in (
            self.eligibility,
            self.bias,
            self.prediction_trace,
            self.trace_mass,
            self.decisions,
            self.feedbacks,
        ):
            if value.device != self.weights.device:
                raise ValueError("outcome-value state tensors must share a device")


class ExternalOutcomeValueBaseline(nn.Module):
    """Learn a memory-side value baseline from scalar outcomes only.

    This critic is an optional variance-reduction companion to
    :class:`ExternalOutcomeCreditPlasticity`.  Its weights, feature trace, and
    predicted-value trace are external state.  It receives no correct choice,
    task identity, or privileged state; a caller may pass the resulting
    trajectory baseline back to the policy credit rule.
    """

    schema = EXTERNAL_OUTCOME_VALUE_SCHEMA

    def __init__(
        self,
        feature_width: int,
        *,
        initial_learning_rate: float = 0.05,
        initial_trace_decay: float = 0.9,
        initial_value: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_width < 1:
            raise ValueError("outcome-value feature width must be positive")
        if not 0.0 < initial_learning_rate <= 1.0:
            raise ValueError("outcome-value learning rate must lie in (0, 1]")
        if not 0.0 <= initial_trace_decay < 1.0:
            raise ValueError("outcome-value trace decay must lie in [0, 1)")
        if not 0.0 < initial_value < 1.0:
            raise ValueError("outcome-value initial value must lie in (0, 1)")
        self.feature_width = int(feature_width)
        self.initial_value = float(initial_value)
        self._trace_decay_exact_zero = initial_trace_decay == 0.0
        self.learning_rate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(initial_learning_rate)).clamp(1e-4, 1.0 - 1e-4))
        )
        self.trace_decay_logit = nn.Parameter(
            torch.tensor(0.0)
            if self._trace_decay_exact_zero
            else torch.logit(torch.tensor(float(initial_trace_decay)).clamp(1e-4, 1.0 - 1e-4))
        )

    @property
    def learning_rate(self) -> torch.Tensor:
        return self.learning_rate_logit.sigmoid()

    @property
    def trace_decay(self) -> torch.Tensor:
        if self._trace_decay_exact_zero:
            return torch.zeros_like(self.trace_decay_logit)
        return self.trace_decay_logit.sigmoid()

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "feature_width": self.feature_width,
            "update_rule": "external_trace_value_baseline_v1",
            "trace": "decayed_feature_and_prediction_trace_v1",
            "learning_rate": float(self.learning_rate.detach()),
            "trace_decay": float(self.trace_decay.detach()),
            "initial_value": self.initial_value,
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalOutcomeValueState:
        if batch_size < 1:
            raise ValueError("outcome-value batch size must be positive")
        weights = torch.zeros(batch_size, self.feature_width, device=device, dtype=dtype)
        return ExternalOutcomeValueState(
            weights=weights,
            eligibility=torch.zeros_like(weights),
            bias=torch.full(
                (batch_size,),
                float(torch.logit(torch.tensor(self.initial_value))),
                device=device,
                dtype=dtype,
            ),
            prediction_trace=torch.zeros(batch_size, device=device, dtype=dtype),
            trace_mass=torch.zeros(batch_size, device=device, dtype=dtype),
            decisions=torch.zeros(batch_size, device=device, dtype=torch.long),
            feedbacks=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _validate_features(
        self,
        features: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_width:
            raise ValueError("outcome-value features have the wrong shape")
        if batch_size is not None and features.shape[0] != batch_size:
            raise ValueError("outcome-value feature batch does not match state")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("outcome-value features must be finite")

    @staticmethod
    def _validate_presence(
        present: torch.Tensor,
        batch_size: int,
        *,
        device: torch.device,
    ) -> None:
        if present.shape != (batch_size,) or present.dtype is not torch.bool:
            raise ValueError("outcome-value presence must be boolean [batch]")
        if present.device != device:
            raise ValueError("outcome-value presence is on the wrong device")

    def _validate_state(self, state: ExternalOutcomeValueState) -> None:
        state.validate(feature_width=self.feature_width)

    def predict(
        self,
        state: ExternalOutcomeValueState,
        features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_state(state)
        self._validate_features(features, batch_size=state.weights.shape[0])
        logits = torch.einsum("bf,bf->b", features, state.weights) + state.bias
        return logits.sigmoid()

    def record_decision(
        self,
        state: ExternalOutcomeValueState,
        features: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExternalOutcomeValueState]:
        """Record one feature and return its current value prediction."""

        self._validate_state(state)
        batch_size = state.weights.shape[0]
        self._validate_features(features, batch_size=batch_size)
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=features.device)
        self._validate_presence(present, batch_size, device=features.device)
        prediction = self.predict(state, features)
        decay = self.trace_decay.to(dtype=features.dtype)
        next_eligibility = decay * state.eligibility + features
        next_prediction_trace = decay * state.prediction_trace + prediction
        next_mass = decay * state.trace_mass + 1.0
        active = present[:, None]
        next_eligibility = torch.where(active, next_eligibility, state.eligibility)
        next_prediction_trace = torch.where(
            present, next_prediction_trace, state.prediction_trace
        )
        next_mass = torch.where(present, next_mass, state.trace_mass)
        next_state = ExternalOutcomeValueState(
            weights=state.weights,
            eligibility=next_eligibility,
            bias=state.bias,
            prediction_trace=next_prediction_trace,
            trace_mass=next_mass,
            decisions=state.decisions + present.to(device=state.decisions.device).long(),
            feedbacks=state.feedbacks,
        )
        self._validate_state(next_state)
        return prediction, next_state

    def episode_baseline(self, state: ExternalOutcomeValueState) -> torch.Tensor:
        """Return the trace-weighted baseline for the current episode."""

        self._validate_state(state)
        default = state.bias.sigmoid()
        return torch.where(
            state.trace_mass > 0.0,
            state.prediction_trace / state.trace_mass.clamp_min(1e-8),
            default,
        )

    def apply_feedback(
        self,
        state: ExternalOutcomeValueState,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
    ) -> ExternalOutcomeValueState:
        """Fit the external value trace to one scalar outcome."""

        self._validate_state(state)
        batch_size = state.weights.shape[0]
        if outcome.shape != (batch_size,) or not bool(torch.isfinite(outcome).all()):
            raise ValueError("outcome-value outcome must be finite [batch]")
        if bool(((outcome < 0.0) | (outcome > 1.0)).any()):
            raise ValueError("outcome-value outcome must lie in [0, 1]")
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=outcome.device)
        self._validate_presence(present, batch_size, device=outcome.device)
        if terminal is None:
            terminal = torch.zeros(batch_size, dtype=torch.bool, device=outcome.device)
        self._validate_presence(terminal, batch_size, device=outcome.device)
        baseline = self.episode_baseline(state)
        error = outcome - baseline
        update = self.learning_rate.to(dtype=state.weights.dtype) * error[:, None] * state.eligibility
        active = present[:, None]
        next_weights = torch.where(active, state.weights + update, state.weights)
        next_bias = torch.where(
            present,
            state.bias + self.learning_rate.to(dtype=state.bias.dtype) * error,
            state.bias,
        )
        clear_flat = terminal & present
        clear = clear_flat[:, None]
        next_eligibility = torch.where(
            clear, torch.zeros_like(state.eligibility), state.eligibility
        )
        next_prediction_trace = torch.where(
            clear_flat,
            torch.zeros_like(state.prediction_trace),
            state.prediction_trace,
        )
        next_mass = torch.where(
            clear_flat,
            torch.zeros_like(state.trace_mass),
            state.trace_mass,
        )
        next_state = ExternalOutcomeValueState(
            weights=next_weights,
            eligibility=next_eligibility,
            bias=next_bias,
            prediction_trace=next_prediction_trace,
            trace_mass=next_mass,
            decisions=state.decisions,
            feedbacks=state.feedbacks + present.to(device=state.feedbacks.device).long(),
        )
        self._validate_state(next_state)
        return next_state

    def begin_episode(self, state: ExternalOutcomeValueState) -> ExternalOutcomeValueState:
        self._validate_state(state)
        next_state = ExternalOutcomeValueState(
            weights=state.weights,
            eligibility=torch.zeros_like(state.eligibility),
            bias=state.bias,
            prediction_trace=torch.zeros_like(state.prediction_trace),
            trace_mass=torch.zeros_like(state.trace_mass),
            decisions=state.decisions,
            feedbacks=state.feedbacks,
        )
        self._validate_state(next_state)
        return next_state

    def state_payload(self, state: ExternalOutcomeValueState) -> dict[str, object]:
        self._validate_state(state)
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "weights": state.weights.detach().cpu().clone(),
            "eligibility": state.eligibility.detach().cpu().clone(),
            "bias": state.bias.detach().cpu().clone(),
            "prediction_trace": state.prediction_trace.detach().cpu().clone(),
            "trace_mass": state.trace_mass.detach().cpu().clone(),
            "decisions": state.decisions.detach().cpu().clone(),
            "feedbacks": state.feedbacks.detach().cpu().clone(),
        }

    def state_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> ExternalOutcomeValueState:
        if payload.get("schema") != self.schema:
            raise ValueError("unsupported outcome-value state schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("outcome-value state configuration is invalid")
        if configuration.get("feature_width") != self.feature_width:
            raise ValueError("outcome-value state dimensions do not match")
        tensors: dict[str, torch.Tensor] = {}
        for name in (
            "weights",
            "eligibility",
            "bias",
            "prediction_trace",
            "trace_mass",
            "decisions",
            "feedbacks",
        ):
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"outcome-value payload field {name!r} must be a tensor")
            tensors[name] = value
        state = ExternalOutcomeValueState(**tensors)
        self._validate_state(state)
        return state


@dataclass(frozen=True)
class MemoryWriteObservation:
    """Opaque controller-native context presented to an external writer.

    The observation contains no verifier labels or protocol fields.  It is
    the narrow boundary between a frozen processor and a separately
    trainable memory-side write policy: learned event/state tensors, the
    current memory read, and the opaque feedback that followed the event.
    """

    event: torch.Tensor
    hidden: torch.Tensor
    workspace_read: torch.Tensor
    query_key: torch.Tensor
    write_value: torch.Tensor
    controller_write_proposal: torch.Tensor
    controller_write_context: torch.Tensor
    controller_write_relevance: torch.Tensor
    memory_read_value: torch.Tensor
    memory_read_hit: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    propensity: torch.Tensor
    has_feedback: torch.Tensor

    def features(self) -> torch.Tensor:
        tensors = (
            self.event,
            self.hidden,
            self.workspace_read,
            self.query_key,
            self.write_value,
            self.controller_write_context,
            self.memory_read_value,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("memory write tensors must have shape [batch, width]")
        batch = tensors[0].shape[0]
        if any(tensor.shape[0] != batch for tensor in tensors):
            raise ValueError("memory write tensors must share a batch dimension")
        if self.memory_read_hit.ndim == 1:
            memory_read_hit = self.memory_read_hit.unsqueeze(-1)
        else:
            memory_read_hit = self.memory_read_hit
        scalars = (
            self.action,
            self.reward,
            self.propensity,
            self.has_feedback,
            self.controller_write_proposal,
            self.controller_write_relevance,
            memory_read_hit,
        )
        normalized_scalars: list[torch.Tensor] = []
        for tensor in scalars:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            if tensor.ndim != 2 or tensor.shape[0] != batch:
                raise ValueError("memory write scalar tensors must share a batch")
            normalized_scalars.append(tensor)
        features = torch.cat((*tensors, *normalized_scalars), dim=-1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("memory write observation contains non-finite values")
        return features


@dataclass(frozen=True)
class MemoryEvictionObservation:
    """Opaque write context paired with one candidate physical memory row."""

    write: MemoryWriteObservation
    candidate_key: torch.Tensor
    candidate_value: torch.Tensor
    candidate_strength: torch.Tensor
    candidate_timestamp: torch.Tensor
    candidate_occupied: torch.Tensor

    def features(self) -> torch.Tensor:
        base = self.write.features()
        tensors = (
            self.candidate_key,
            self.candidate_value,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("candidate tensors must have shape [batch, width]")
        if any(tensor.shape[0] != base.shape[0] for tensor in tensors):
            raise ValueError("candidate tensors must share the write batch")
        scalars: list[torch.Tensor] = []
        for tensor in (
            self.candidate_strength,
            self.candidate_timestamp,
            self.candidate_occupied,
        ):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            if tensor.ndim != 2 or tensor.shape != (base.shape[0], 1):
                raise ValueError("candidate scalars must have shape [batch, 1]")
            scalars.append(tensor.to(device=base.device, dtype=base.dtype))
        features = torch.cat(
            (base, tensors[0], tensors[1], *scalars),
            dim=-1,
        )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("memory eviction observation contains non-finite values")
        return features


class ExternalMemoryWritePolicy(nn.Module):
    """A replaceable writer that can learn while the controller is frozen.

    This module is intentionally independent of ``AmodalCognitiveController``
    parameters.  It consumes only :class:`MemoryWriteObservation` and emits a
    Bernoulli write probability; the memory backend remains responsible for
    committing the sampled decision.  It is useful both as a real memory-side
    growth component and as a causal test of whether forgetting is caused by
    changing the processor's generic write head.
    """

    schema = "neural-computer.external-memory-write-policy.v11"

    def __init__(
        self,
        *,
        event_width: int,
        hidden_width: int,
        workspace_width: int,
        key_width: int,
        value_width: int,
        memory_read_width: int,
        action_width: int,
        controller_write_context_width: int,
        controller_write_relevance_width: int,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        widths = (
            event_width,
            hidden_width,
            workspace_width,
            key_width,
            value_width,
            memory_read_width,
            action_width,
            controller_write_context_width,
            controller_write_relevance_width,
        )
        if min(widths) < 1:
            raise ValueError("external memory writer widths must be positive")
        self.event_width = event_width
        self.hidden_width = hidden_width
        self.workspace_width = workspace_width
        self.key_width = key_width
        self.value_width = value_width
        self.memory_read_width = memory_read_width
        self.action_width = action_width
        self.controller_write_context_width = controller_write_context_width
        self.controller_write_relevance_width = controller_write_relevance_width
        self.feature_width = sum(widths) + 5
        hidden_width_value = max(16, self.feature_width // 2) if hidden is None else hidden
        if hidden_width_value < 1:
            raise ValueError("external memory writer hidden width must be positive")
        self.hidden = hidden_width_value
        self.network = nn.Sequential(
            nn.Linear(self.feature_width, hidden_width_value),
            nn.GELU(),
            nn.Linear(hidden_width_value, 1),
        )
        self.value_network = nn.Sequential(
            nn.Linear(self.feature_width, hidden_width_value),
            nn.GELU(),
            nn.Linear(hidden_width_value, value_width),
        )
        # A zero residual starts from a stable generic relevance prior.  The
        # external writer can adapt that prior, but bounded residual plasticity
        # prevents a short run from erasing the separator it inherited.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        # Preserve the controller's mastered value representation until the
        # external memory receives a differentiable outcome signal.
        nn.init.zeros_(self.value_network[-1].weight)
        nn.init.zeros_(self.value_network[-1].bias)
        # A one-slot memory needs a decisive novelty/relevance boundary: a
        # moderately similar distractor must not overwrite a strongly bound
        # current event.  The prior is generic and frozen; only the external
        # residual adapts from outcome-only causal evidence.
        self.register_buffer("relevance_scale", torch.tensor(12.0))
        self.register_buffer("relevance_bias", torch.tensor(-9.0))
        self.residual_scale = 0.25
        self.value_residual_scale = 0.05

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "event_width": self.event_width,
            "hidden_width": self.hidden_width,
            "workspace_width": self.workspace_width,
            "key_width": self.key_width,
            "value_width": self.value_width,
            "memory_read_width": self.memory_read_width,
            "action_width": self.action_width,
            "controller_write_context_width": self.controller_write_context_width,
            "controller_write_relevance_width": self.controller_write_relevance_width,
            "relevance_prior": "frozen_affine_logit_v2",
            "residual_parameterization": "bounded_tanh_v1",
            "residual_scale": self.residual_scale,
            "value_parameterization": "bounded_tanh_residual_v1",
            "value_residual_scale": self.value_residual_scale,
            "hidden": self.hidden,
        }

    def forward(self, observation: MemoryWriteObservation) -> torch.Tensor:
        features = observation.features()
        expected = (
            self.event_width
            + self.hidden_width
            + self.workspace_width
            + self.key_width
            + self.value_width
            + self.memory_read_width
            + self.action_width
            + self.controller_write_context_width
            + self.controller_write_relevance_width
            + 5
        )
        if features.shape[1] != expected:
            raise ValueError(
                f"memory write observation width {features.shape[1]} != {expected}"
            )
        relevance = observation.controller_write_relevance.reshape(-1).to(
            device=features.device, dtype=features.dtype
        )
        if relevance.shape != (features.shape[0],):
            raise ValueError("controller write relevance has the wrong shape")
        prior = self.relevance_scale * relevance + self.relevance_bias
        residual = torch.tanh(self.network(features).squeeze(-1))
        return torch.sigmoid(prior + self.residual_scale * residual)

    def adapt_value(self, observation: MemoryWriteObservation) -> torch.Tensor:
        """Return a bounded, identity-initialized memory-value adaptation."""
        features = observation.features()
        expected = (
            self.event_width
            + self.hidden_width
            + self.workspace_width
            + self.key_width
            + self.value_width
            + self.memory_read_width
            + self.action_width
            + self.controller_write_context_width
            + self.controller_write_relevance_width
            + 5
        )
        if features.shape[1] != expected:
            raise ValueError(
                f"memory write observation width {features.shape[1]} != {expected}"
            )
        value = observation.write_value
        if value.ndim != 2 or value.shape[1] != self.value_width:
            raise ValueError("memory write value has the wrong shape")
        residual = torch.tanh(self.value_network(features))
        return value + self.value_residual_scale * residual


class ExternalMemoryEvictionPolicy(nn.Module):
    """A replaceable scorer for choosing which opaque row to overwrite.

    The policy is memory-side state: it consumes a generic write observation
    and one candidate row, then emits a comparable score. Candidate count and
    physical row identity are supplied by the backend, so the same scorer can
    rank any bounded capacity without adding a controller branch.
    """

    schema = "neural-computer.external-memory-eviction-policy.v1"

    def __init__(
        self,
        *,
        event_width: int,
        hidden_width: int,
        workspace_width: int,
        key_width: int,
        value_width: int,
        memory_read_width: int,
        action_width: int,
        controller_write_context_width: int,
        controller_write_relevance_width: int,
        candidate_key_width: int,
        candidate_value_width: int,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        widths = (
            event_width,
            hidden_width,
            workspace_width,
            key_width,
            value_width,
            memory_read_width,
            action_width,
            controller_write_context_width,
            controller_write_relevance_width,
            candidate_key_width,
            candidate_value_width,
        )
        if min(widths) < 1:
            raise ValueError("external memory eviction widths must be positive")
        self.event_width = event_width
        self.hidden_width = hidden_width
        self.workspace_width = workspace_width
        self.key_width = key_width
        self.value_width = value_width
        self.memory_read_width = memory_read_width
        self.action_width = action_width
        self.controller_write_context_width = controller_write_context_width
        self.controller_write_relevance_width = controller_write_relevance_width
        self.candidate_key_width = candidate_key_width
        self.candidate_value_width = candidate_value_width
        self.write_observation_width = sum(widths[:-2]) + 5
        self.feature_width = self.write_observation_width + candidate_key_width + candidate_value_width + 3
        hidden_width_value = max(16, self.feature_width // 2) if hidden is None else hidden
        if hidden_width_value < 1:
            raise ValueError("external memory eviction hidden width must be positive")
        self.hidden = hidden_width_value
        self.network = nn.Sequential(
            nn.Linear(self.feature_width, hidden_width_value),
            nn.GELU(),
            nn.Linear(hidden_width_value, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "event_width": self.event_width,
            "hidden_width": self.hidden_width,
            "workspace_width": self.workspace_width,
            "key_width": self.key_width,
            "value_width": self.value_width,
            "memory_read_width": self.memory_read_width,
            "action_width": self.action_width,
            "controller_write_context_width": self.controller_write_context_width,
            "controller_write_relevance_width": self.controller_write_relevance_width,
            "candidate_key_width": self.candidate_key_width,
            "candidate_value_width": self.candidate_value_width,
            "hidden": self.hidden,
        }

    def forward(self, observation: MemoryEvictionObservation) -> torch.Tensor:
        features = observation.features()
        if features.shape[1] != self.feature_width:
            raise ValueError(
                f"memory eviction observation width {features.shape[1]} != {self.feature_width}"
            )
        if observation.candidate_key.shape[1] != self.candidate_key_width:
            raise ValueError("candidate key has the wrong width")
        if observation.candidate_value.shape[1] != self.candidate_value_width:
            raise ValueError("candidate value has the wrong width")
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class CapabilityEvictionObservation:
    """Opaque context paired with one external capability candidate.

    ``context`` and ``candidate`` are learned tensors or scalar summaries
    assembled at the memory boundary.  No physical slot index, task name, or
    verifier target belongs in this observation.  The candidate axis is
    supplied by the caller so one policy can rank any bounded bank.
    """

    context: torch.Tensor
    candidate: torch.Tensor

    def features(self) -> torch.Tensor:
        if self.context.ndim != 2 or self.candidate.ndim != 2:
            raise ValueError("capability eviction tensors must be [batch, width]")
        if self.context.shape[0] != self.candidate.shape[0]:
            raise ValueError("capability eviction tensors must share a batch")
        features = torch.cat((self.context, self.candidate), dim=-1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("capability eviction observations must be finite")
        return features


class ExternalCapabilityEvictionPolicy(nn.Module):
    """Learn a generic disposable-capability score outside the controller.

    The policy is deliberately smaller than the controller and independently
    replaceable.  It consumes opaque current context and candidate summaries,
    emits a comparable score whose larger value means more disposable, and
    leaves protection masking to :class:`CapabilityRetentionLedger`.
    """

    schema = "neural-computer.external-capability-eviction-policy.v1"

    def __init__(
        self,
        *,
        context_width: int,
        candidate_width: int,
        hidden: int = 32,
    ) -> None:
        if min(context_width, candidate_width, hidden) < 1:
            raise ValueError("capability eviction widths must be positive")
        self.context_width = int(context_width)
        self.candidate_width = int(candidate_width)
        self.hidden = int(hidden)
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.context_width + self.candidate_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "candidate_width": self.candidate_width,
            "hidden": self.hidden,
        }

    def forward(self, observation: CapabilityEvictionObservation) -> torch.Tensor:
        features = observation.features()
        expected = self.context_width + self.candidate_width
        if features.shape[1] != expected:
            raise ValueError(
                f"capability eviction width {features.shape[1]} != {expected}"
            )
        return self.network(features).squeeze(-1)

    def score_candidates(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score ``[batch, candidates, width]`` without exposing row identity."""

        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError(
                f"context must have shape [batch, {self.context_width}]"
            )
        if candidates.ndim != 3 or candidates.shape[2] != self.candidate_width:
            raise ValueError(
                "candidates must have shape [batch, rows, candidate_width]"
            )
        repeated_context = context[:, None, :].expand(
            -1, candidates.shape[1], -1
        )
        flat = CapabilityEvictionObservation(
            context=repeated_context.reshape(-1, self.context_width),
            candidate=candidates.reshape(-1, self.candidate_width),
        )
        return self(flat).reshape(candidates.shape[:2])


def _validate_inputs(
    named_parameters: NamedParameters,
    reference_gradient: GradientMap,
    strength: float,
) -> None:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("projection strength must be within [0, 1]")
    if not named_parameters:
        raise ValueError("at least one named parameter is required")
    for name, parameter in named_parameters:
        reference = reference_gradient.get(name)
        if reference is None:
            raise ValueError(f"reference gradient is missing {name!r}")
        if reference.shape != parameter.shape:
            raise ValueError(f"reference gradient has the wrong shape for {name!r}")
        if reference.device != parameter.device:
            raise ValueError(f"reference gradient is on the wrong device for {name!r}")
        if reference.dtype != parameter.dtype:
            raise ValueError(f"reference gradient has the wrong dtype for {name!r}")


def _gradient_statistics(
    named_parameters: NamedParameters,
    reference_gradient: GradientMap,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = named_parameters[0][1].device
    target_norm_sq = torch.zeros((), device=device)
    reference_norm_sq = torch.zeros_like(target_norm_sq)
    dot = torch.zeros_like(target_norm_sq)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        reference = reference_gradient[name]
        dot = dot + torch.sum(parameter.grad * reference)
        target_norm_sq = target_norm_sq + torch.sum(parameter.grad.square())
        reference_norm_sq = reference_norm_sq + torch.sum(reference.square())
    return target_norm_sq, reference_norm_sq, dot


def project_gradient_against_reference(
    named_parameters: NamedParameters,
    reference_gradient: GradientMap,
    strength: float,
) -> tuple[bool, float | None, float | None]:
    """Remove an opposing component from the current parameter gradients.

    ``reference_gradient`` is normally the sum of gradients from verified
    rehearsal streams. Compatible target gradients are left unchanged. If
    the target update opposes the reference, ``strength=1`` removes the full
    opposing component and smaller values perform a partial projection.
    """
    _validate_inputs(named_parameters, reference_gradient, strength)
    target_norm_sq, reference_norm_sq, dot = _gradient_statistics(
        named_parameters, reference_gradient
    )
    if float(target_norm_sq) == 0.0 or float(reference_norm_sq) == 0.0:
        return False, None, None
    cosine = float(dot / torch.sqrt(target_norm_sq * reference_norm_sq))
    applied = float(dot) < 0.0 and strength > 0.0
    if applied:
        coefficient = strength * dot / reference_norm_sq
        with torch.no_grad():
            for name, parameter in named_parameters:
                if parameter.grad is not None:
                    parameter.grad.sub_(coefficient * reference_gradient[name])
    post_dot = torch.zeros_like(dot)
    for name, parameter in named_parameters:
        if parameter.grad is not None:
            post_dot = post_dot + torch.sum(
                parameter.grad * reference_gradient[name]
            )
    return applied, cosine, float(post_dot)


def project_parameter_update_against_reference(
    named_parameters: NamedParameters,
    parameters_before: Mapping[str, torch.Tensor],
    reference_gradient: GradientMap,
    strength: float,
) -> tuple[bool, float | None, float | None]:
    """Remove an opposing component from an applied optimizer update.

    This is a secondary safeguard for optimizers whose momentum or adaptive
    scaling changes the direction after gradient projection. It is kept
    separate so trainers can measure whether it is actually needed.
    """
    _validate_inputs(named_parameters, reference_gradient, strength)
    for name, parameter in named_parameters:
        before = parameters_before.get(name)
        if before is None:
            raise ValueError(f"parameters_before is missing {name!r}")
        if before.shape != parameter.shape:
            raise ValueError(f"parameters_before has the wrong shape for {name!r}")
        if before.device != parameter.device or before.dtype != parameter.dtype:
            raise ValueError(
                f"parameters_before has the wrong device or dtype for {name!r}"
            )

    update_norm_sq = torch.zeros((), device=named_parameters[0][1].device)
    reference_norm_sq = torch.zeros_like(update_norm_sq)
    dot = torch.zeros_like(update_norm_sq)
    for name, parameter in named_parameters:
        update = parameter.detach() - parameters_before[name]
        reference = reference_gradient[name]
        dot = dot + torch.sum(update * reference)
        update_norm_sq = update_norm_sq + torch.sum(update.square())
        reference_norm_sq = reference_norm_sq + torch.sum(reference.square())
    if float(update_norm_sq) == 0.0 or float(reference_norm_sq) == 0.0:
        return False, None, None
    cosine = float(dot / torch.sqrt(update_norm_sq * reference_norm_sq))
    applied = float(dot) > 0.0 and strength > 0.0
    if applied:
        coefficient = strength * dot / reference_norm_sq
        with torch.no_grad():
            for name, parameter in named_parameters:
                parameter.sub_(coefficient * reference_gradient[name])
    post_dot = torch.zeros_like(dot)
    for name, parameter in named_parameters:
        update = parameter.detach() - parameters_before[name]
        post_dot = post_dot + torch.sum(update * reference_gradient[name])
    return applied, cosine, float(post_dot)


def zero_gradient_map(named_parameters: NamedParameters) -> dict[str, torch.Tensor]:
    """Create a detached accumulator matching a trainable parameter set."""
    if not named_parameters:
        raise ValueError("at least one named parameter is required")
    return {
        name: torch.zeros_like(parameter)
        for name, parameter in named_parameters
    }


def accumulate_current_gradients(
    named_parameters: NamedParameters,
    accumulator: dict[str, torch.Tensor],
) -> None:
    """Add current gradients into a detached rehearsal accumulator."""
    for name, parameter in named_parameters:
        if name not in accumulator:
            raise ValueError(f"gradient accumulator is missing {name!r}")
        if parameter.grad is not None:
            accumulator[name].add_(parameter.grad.detach())
