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
from torch import nn

NamedParameters = Sequence[tuple[str, torch.nn.Parameter]]
GradientMap = Mapping[str, torch.Tensor]


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
