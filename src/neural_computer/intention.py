"""External memory for opaque intention discovery and reuse.

The controller emits intentions, but it should not have to remember every
intention it has ever tried.  This module is an append-only, protocol-agnostic
repertoire for those learned output vectors.  It stores only opaque vectors
and verifier statistics; model-based planning remains the authority that
chooses behavior for a goal.

The repertoire deliberately does not rank intentions by reward.  A reward
ranking would quietly become another policy.  It exposes the available
experience to factual search, while an ephemeral controller seed keeps novel
output content discoverable before it has been written to external memory.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

EXTERNAL_INTENTION_REPERTOIRE_SCHEMA = (
    "neural-computer.external-intention-repertoire.v1"
)
EXTERNAL_INTENTION_PROPOSAL_SCHEMA = "neural-computer.external-intention-proposal.v1"
EXTERNAL_INTENTION_OBSERVATION_SCHEMA = (
    "neural-computer.external-intention-observation.v1"
)
EXTERNAL_INTENTION_ADMISSION_SCHEMA = (
    "neural-computer.external-intention-admission.v1"
)
EXTERNAL_INTENTION_EXPLORATION_SCHEMA = (
    "neural-computer.external-intention-exploration.v1"
)
EXTERNAL_INTENTION_CONSOLIDATION_SCHEMA = (
    "neural-computer.external-intention-consolidation.v1"
)
EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA_V1 = (
    "neural-computer.external-outcome-intention-generator.v1"
)
EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA = (
    "neural-computer.external-outcome-intention-generator.v2"
)
EXTERNAL_INTENTION_GENERATION_PROPOSAL_SCHEMA = (
    "neural-computer.external-intention-generation-proposal.v1"
)


def _validate_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    width: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if width is not None and value.shape[-1] != width:
        raise ValueError(f"{name} has the wrong width")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ExternalIntentionProposal:
    """Runtime-sized opaque candidates available to factual model search."""

    intentions: torch.Tensor
    source_indices: tuple[int, ...]
    propensities: torch.Tensor
    exploration_mask: torch.Tensor
    version: int
    schema: str = EXTERNAL_INTENTION_PROPOSAL_SCHEMA

    def validate(self, *, width: int, batch: int | None = None) -> ExternalIntentionProposal:
        if self.schema != EXTERNAL_INTENTION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported intention-proposal schema")
        _validate_tensor(
            self.intentions,
            name="intention proposal",
            ndim=3,
            width=width,
        )
        candidate_count = self.intentions.shape[1]
        if candidate_count < 1:
            raise ValueError("intention proposal requires one candidate")
        if batch is not None and self.intentions.shape[0] != batch:
            raise ValueError("intention proposal batch differs")
        if len(self.source_indices) != candidate_count:
            raise ValueError("intention proposal source indices are misaligned")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < -1
            for index in self.source_indices
        ):
            raise ValueError("intention proposal source index is invalid")
        expected_shape = (self.intentions.shape[0], candidate_count)
        if self.propensities.shape != expected_shape:
            raise ValueError("intention proposal propensities have the wrong shape")
        if self.exploration_mask.shape != expected_shape or (
            self.exploration_mask.dtype != torch.bool
        ):
            raise ValueError("intention proposal exploration mask has the wrong shape")
        source_exploration = torch.tensor(
            [index == -1 for index in self.source_indices],
            dtype=torch.bool,
            device=self.exploration_mask.device,
        ).unsqueeze(0)
        if not bool(
            torch.equal(
                self.exploration_mask,
                source_exploration.expand_as(self.exploration_mask),
            )
        ):
            raise ValueError("intention proposal exploration flags are inconsistent")
        if not bool(torch.isfinite(self.propensities).all()) or bool(
            torch.any(self.propensities <= 0.0) or torch.any(self.propensities > 1.0)
        ):
            raise ValueError("intention proposal propensities must lie in (0, 1]")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("intention proposal version is invalid")
        return self


@dataclass(frozen=True)
class ExternalOutcomeIntentionGeneratorState:
    """Persistent external state for one or more opaque generator cells.

    The tensors are capability state, not controller parameters. Each batch
    row is an independently addressable external cell; a memory backend can
    keep one row per logical capability, append rows, and protect mastered
    rows while later rows continue adapting.
    """

    input_weights: torch.Tensor
    input_bias: torch.Tensor
    output_weights: torch.Tensor
    output_bias: torch.Tensor
    input_weight_eligibility: torch.Tensor
    input_bias_eligibility: torch.Tensor
    output_weight_eligibility: torch.Tensor
    output_bias_eligibility: torch.Tensor
    context_residual_weights: torch.Tensor
    context_residual_eligibility: torch.Tensor
    baseline: torch.Tensor
    decisions: torch.Tensor
    feedbacks: torch.Tensor
    protected: torch.Tensor

    def validate(
        self,
        *,
        context_width: int,
        intention_width: int,
        hidden_width: int,
        feature_width: int | None = None,
    ) -> None:
        expected_feature_width = context_width + 1 if feature_width is None else feature_width
        if expected_feature_width < context_width + 1:
            raise ValueError("intention generator feature width is invalid")
        if not isinstance(self.baseline, torch.Tensor) or self.baseline.ndim != 1:
            raise ValueError("intention generator baseline has the wrong shape")
        batch_size = self.baseline.shape[0]
        expected = {
            "input_weights": (batch_size, hidden_width, expected_feature_width),
            "input_bias": (batch_size, hidden_width),
            "output_weights": (batch_size, intention_width, hidden_width),
            "output_bias": (batch_size, intention_width),
            "input_weight_eligibility": (
                batch_size,
                hidden_width,
                expected_feature_width,
            ),
            "input_bias_eligibility": (batch_size, hidden_width),
            "output_weight_eligibility": (
                batch_size,
                intention_width,
                hidden_width,
            ),
            "output_bias_eligibility": (batch_size, intention_width),
            "context_residual_weights": (
                batch_size,
                intention_width,
                context_width + 1,
            ),
            "context_residual_eligibility": (
                batch_size,
                intention_width,
                context_width + 1,
            ),
            "baseline": (batch_size,),
            "decisions": (batch_size,),
            "feedbacks": (batch_size,),
            "protected": (batch_size,),
        }
        tensors = {
            "input_weights": self.input_weights,
            "input_bias": self.input_bias,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "input_weight_eligibility": self.input_weight_eligibility,
            "input_bias_eligibility": self.input_bias_eligibility,
            "output_weight_eligibility": self.output_weight_eligibility,
            "output_bias_eligibility": self.output_bias_eligibility,
            "context_residual_weights": self.context_residual_weights,
            "context_residual_eligibility": self.context_residual_eligibility,
            "baseline": self.baseline,
            "decisions": self.decisions,
            "feedbacks": self.feedbacks,
            "protected": self.protected,
        }
        if batch_size < 1:
            raise ValueError("intention generator state needs one cell")
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name]:
                raise ValueError(f"intention generator {name} has the wrong shape")
            if value.device != self.input_weights.device:
                raise ValueError("intention generator state tensors must share a device")
        for name in (
            "input_weights",
            "input_bias",
            "output_weights",
            "output_bias",
            "input_weight_eligibility",
            "input_bias_eligibility",
            "output_weight_eligibility",
            "output_bias_eligibility",
            "context_residual_weights",
            "context_residual_eligibility",
            "baseline",
        ):
            if not bool(torch.isfinite(tensors[name]).all()):
                raise ValueError(f"intention generator {name} must be finite")
        if self.decisions.dtype not in (torch.int32, torch.int64) or self.feedbacks.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("intention generator counters must be integer tensors")
        if self.protected.dtype != torch.bool:
            raise TypeError("intention generator protection mask must be boolean")
        if bool((self.decisions < 0).any()) or bool((self.feedbacks < 0).any()):
            raise ValueError("intention generator counters cannot be negative")
        if bool(((self.baseline < 0.0) | (self.baseline > 1.0)).any()):
            raise ValueError("intention generator baseline must lie in [0, 1]")


@dataclass(frozen=True)
class ExternalIntentionGenerationProposal:
    """A sampled continuous intention and its exact Gaussian log density."""

    intentions: torch.Tensor
    means: torch.Tensor
    features: torch.Tensor
    hidden: torch.Tensor
    noise: torch.Tensor
    log_propensities: torch.Tensor
    noise_scale: float
    schema: str = EXTERNAL_INTENTION_GENERATION_PROPOSAL_SCHEMA

    def validate(
        self,
        *,
        context_width: int,
        intention_width: int,
        hidden_width: int,
        batch: int | None = None,
        feature_width: int | None = None,
    ) -> ExternalIntentionGenerationProposal:
        if self.schema != EXTERNAL_INTENTION_GENERATION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported intention-generation proposal schema")
        if not isinstance(self.noise_scale, (float, int)) or not math.isfinite(
            float(self.noise_scale)
        ) or float(self.noise_scale) <= 0.0:
            raise ValueError("intention-generation noise scale is invalid")
        expected_feature_width = context_width + 1 if feature_width is None else feature_width
        if expected_feature_width < context_width + 1:
            raise ValueError("generated intention feature width is invalid")
        expected = {
            "intentions": (intention_width,),
            "means": (intention_width,),
            "features": (expected_feature_width,),
            "hidden": (hidden_width,),
            "noise": (intention_width,),
            "log_propensities": (),
        }
        tensors = {
            "intentions": self.intentions,
            "means": self.means,
            "features": self.features,
            "hidden": self.hidden,
            "noise": self.noise,
            "log_propensities": self.log_propensities,
        }
        if self.intentions.ndim != 2:
            raise ValueError("generated intentions must have shape [batch, width]")
        if batch is not None and self.intentions.shape[0] != batch:
            raise ValueError("generated intention batch differs")
        for name, value in tensors.items():
            if value.shape != (self.intentions.shape[0], *expected[name]):
                raise ValueError(f"generated intention {name} has the wrong shape")
            _validate_tensor(
                value,
                name=f"generated intention {name}",
                ndim=value.ndim,
            )
        return self


class ExternalOutcomeIntentionGenerator:
    """Generate new opaque intention content from scalar outcomes only.

    This is a memory-side stochastic neural program. It maps a learned opaque
    context to a sampled intention with a compact tanh/linear network. A
    caller supplies only the sampled proposal and a deterministic verifier
    outcome; score-function credit updates the external generator state
    without backpropagating through the verifier, touching the controller, or
    storing examples for replay.

    The generator is deliberately separate from ``ExternalIntentionRepertoire``:
    this class invents provisional content, while the repertoire remains the
    durable verified file store. A caller should admit a successful proposal
    through ``admit_verified`` and protect the generator cell or checkpoint
    the resulting state only after held-out retention passes.
    """

    schema = EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA

    def __init__(
        self,
        context_width: int,
        intention_width: int,
        *,
        hidden_width: int = 32,
        initial_learning_rate: float = 0.1,
        initial_trace_decay: float = 0.0,
        initial_baseline_rate: float = 0.05,
        initial_baseline: float = 0.5,
        noise_scale: float = 0.5,
        initial_parameter_scale: float = 0.05,
        context_masking: bool = False,
        mask_stable_content: bool = False,
        factorized_context_residual: bool = False,
    ) -> None:
        dimensions = (context_width, intention_width, hidden_width)
        if min(dimensions) < 1:
            raise ValueError("intention generator dimensions must be positive")
        if not 0.0 < initial_learning_rate <= 1.0:
            raise ValueError("intention generator learning rate is invalid")
        if not 0.0 <= initial_trace_decay < 1.0:
            raise ValueError("intention generator trace decay is invalid")
        if not 0.0 < initial_baseline_rate <= 1.0:
            raise ValueError("intention generator baseline rate is invalid")
        if not 0.0 <= initial_baseline <= 1.0:
            raise ValueError("intention generator baseline is invalid")
        if not math.isfinite(noise_scale) or noise_scale <= 0.0:
            raise ValueError("intention generator noise scale is invalid")
        if not math.isfinite(initial_parameter_scale) or initial_parameter_scale <= 0.0:
            raise ValueError("intention generator parameter scale is invalid")
        if not isinstance(context_masking, bool):
            raise TypeError("intention generator context masking must be boolean")
        if not isinstance(mask_stable_content, bool):
            raise TypeError("intention generator mask-stable content must be boolean")
        if mask_stable_content and not context_masking:
            raise ValueError("mask-stable content requires context masking")
        if not isinstance(factorized_context_residual, bool):
            raise TypeError("intention generator factorized residual must be boolean")
        self.context_width = int(context_width)
        self.intention_width = int(intention_width)
        self.hidden_width = int(hidden_width)
        self.initial_learning_rate = float(initial_learning_rate)
        self.initial_trace_decay = float(initial_trace_decay)
        self.initial_baseline_rate = float(initial_baseline_rate)
        self.initial_baseline = float(initial_baseline)
        self.noise_scale = float(noise_scale)
        self.initial_parameter_scale = float(initial_parameter_scale)
        self.context_masking = bool(context_masking)
        self.mask_stable_content = bool(mask_stable_content)
        self.factorized_context_residual = bool(factorized_context_residual)

    @property
    def feature_width(self) -> int:
        """Width of the external learner's explicit context feature vector."""

        return 2 * self.context_width + 1 if self.context_masking else self.context_width + 1

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "intention_width": self.intention_width,
            "hidden_width": self.hidden_width,
            "feature_width": self.feature_width,
            "context_masking": self.context_masking,
            "mask_stable_content": self.mask_stable_content,
            "factorized_context_residual": self.factorized_context_residual,
            "initial_learning_rate": self.initial_learning_rate,
            "initial_trace_decay": self.initial_trace_decay,
            "initial_baseline_rate": self.initial_baseline_rate,
            "initial_baseline": self.initial_baseline,
            "noise_scale": self.noise_scale,
            "initial_parameter_scale": self.initial_parameter_scale,
            "update_rule": "outcome_only_gaussian_score_credit_external_mlp_v1",
            "state": "external_generator_cells_and_eligibility_v1",
            "controller": "frozen_opaque_context_only_v1",
            "persistence": "tensor_only_versioned_payload_v1",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalOutcomeIntentionGeneratorState:
        if batch_size < 1:
            raise ValueError("intention generator batch size must be positive")
        feature_width = self.feature_width
        input_weights = self.initial_parameter_scale * torch.randn(
            batch_size,
            self.hidden_width,
            feature_width,
            device=device,
            dtype=dtype,
        )
        if self.context_masking:
            input_weights[:, :, self.context_width : 2 * self.context_width].zero_()
        output_weights = self.initial_parameter_scale * torch.randn(
            batch_size,
            self.intention_width,
            self.hidden_width,
            device=device,
            dtype=dtype,
        )
        state = ExternalOutcomeIntentionGeneratorState(
            input_weights=input_weights,
            input_bias=torch.zeros(batch_size, self.hidden_width, device=device, dtype=dtype),
            output_weights=output_weights,
            output_bias=torch.zeros(batch_size, self.intention_width, device=device, dtype=dtype),
            input_weight_eligibility=torch.zeros_like(input_weights),
            input_bias_eligibility=torch.zeros(batch_size, self.hidden_width, device=device, dtype=dtype),
            output_weight_eligibility=torch.zeros_like(output_weights),
            output_bias_eligibility=torch.zeros(batch_size, self.intention_width, device=device, dtype=dtype),
            context_residual_weights=torch.zeros(
                batch_size,
                self.intention_width,
                self.context_width + 1,
                device=device,
                dtype=dtype,
            ),
            context_residual_eligibility=torch.zeros(
                batch_size,
                self.intention_width,
                self.context_width + 1,
                device=device,
                dtype=dtype,
            ),
            baseline=torch.full(
                (batch_size,),
                self.initial_baseline,
                device=device,
                dtype=dtype,
            ),
            decisions=torch.zeros(batch_size, device=device, dtype=torch.long),
            feedbacks=torch.zeros(batch_size, device=device, dtype=torch.long),
            protected=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )
        self._validate_state(state)
        return state

    def _validate_state(self, state: ExternalOutcomeIntentionGeneratorState) -> None:
        state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            feature_width=self.feature_width,
        )
        if self.mask_stable_content:
            mask_weights = state.input_weights[
                :, :, self.context_width : 2 * self.context_width
            ]
            if bool(mask_weights.abs().max() > 0.0):
                raise ValueError("mask-stable content state has mutable mask weights")

    def residual_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return observed values plus bias for the factorized residual path."""

        return torch.cat((features[:, : self.context_width], features[:, -1:]), dim=-1)

    def _validate_context(
        self,
        context: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> None:
        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError("intention generator context has the wrong shape")
        if batch_size is not None and context.shape[0] != batch_size:
            raise ValueError("intention generator context batch differs")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("intention generator context must be finite")

    def context_features(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return learned features, observed values, and the explicit mask.

        Masked mode adds a learned-visible observation channel while preserving
        the original opaque value width.  The bias feature remains last so
        dense v1 callers retain their exact parameter layout.
        """

        self._validate_context(context, batch_size=batch_size)
        if context_mask is None:
            mask = torch.ones(
                context.shape,
                dtype=torch.bool,
                device=context.device,
            )
        else:
            if not self.context_masking:
                raise ValueError(
                    "context_mask requires intention generator context_masking=True"
                )
            if context_mask.shape != context.shape or context_mask.dtype != torch.bool:
                raise ValueError("intention generator context mask must be boolean [batch,width]")
            if context_mask.device != context.device:
                raise ValueError("intention generator context mask is on the wrong device")
            mask = context_mask
        observed_context = context * mask.to(dtype=context.dtype)
        if self.context_masking:
            features = torch.cat(
                (
                    observed_context,
                    mask.to(dtype=context.dtype),
                    torch.ones(
                        context.shape[0], 1, device=context.device, dtype=context.dtype
                    ),
                ),
                dim=-1,
            )
        else:
            features = torch.cat(
                (
                    observed_context if context_mask is not None else context,
                    torch.ones(
                        context.shape[0], 1, device=context.device, dtype=context.dtype
                    ),
                ),
                dim=-1,
            )
        return features, observed_context, mask

    def content_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return features allowed to influence the mutable content path.

        The full feature tensor remains in proposals so the memory/router can
        recover observation provenance.  In mask-stable mode the mask channel
        is structurally disconnected from hidden content computation; routing
        and retention still consume it separately.
        """

        if self.mask_stable_content:
            stable_features = features.clone()
            stable_features[:, self.context_width : 2 * self.context_width] = 0.0
            return stable_features
        return features

    def _validate_presence(
        self,
        present: torch.Tensor,
        batch_size: int,
        *,
        device: torch.device,
    ) -> None:
        if present.shape != (batch_size,) or present.dtype != torch.bool:
            raise ValueError("intention generator presence must be boolean [batch]")
        if present.device != device:
            raise ValueError("intention generator presence is on the wrong device")

    def mean(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read the current deterministic generator mean without mutation."""

        self._validate_state(state)
        features, _, _ = self.context_features(
            context,
            context_mask,
            batch_size=state.baseline.shape[0],
        )
        if context.device != state.input_weights.device:
            raise ValueError("intention generator context is on the wrong device")
        content_features = self.content_features(features)
        residual_features = self.residual_features(features)
        hidden = torch.tanh(
            torch.einsum("bf,bhf->bh", content_features, state.input_weights)
            + state.input_bias
        )
        result = torch.einsum("bh,boh->bo", hidden, state.output_weights) + state.output_bias
        if self.factorized_context_residual:
            result = result + torch.einsum(
                "bf,bof->bo", residual_features, state.context_residual_weights
            )
        return result

    def propose(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        context: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> ExternalIntentionGenerationProposal:
        """Sample one novel intention per external generator cell."""

        self._validate_state(state)
        batch_size = state.baseline.shape[0]
        features, _, _ = self.context_features(
            context,
            context_mask,
            batch_size=batch_size,
        )
        if context.device != state.input_weights.device:
            raise ValueError("intention generator context is on the wrong device")
        content_features = self.content_features(features)
        residual_features = self.residual_features(features)
        hidden = torch.tanh(
            torch.einsum("bf,bhf->bh", content_features, state.input_weights)
            + state.input_bias
        )
        means = torch.einsum("bh,boh->bo", hidden, state.output_weights) + state.output_bias
        if self.factorized_context_residual:
            means = means + torch.einsum(
                "bf,bof->bo", residual_features, state.context_residual_weights
            )
        noise = torch.randn(
            means.shape,
            device=means.device,
            dtype=means.dtype,
            generator=generator,
        )
        scale = torch.as_tensor(self.noise_scale, device=means.device, dtype=means.dtype)
        intentions = means + scale * noise
        log_propensities = -0.5 * torch.sum(
            noise.square() + math.log(2.0 * math.pi * self.noise_scale**2),
            dim=-1,
        )
        proposal = ExternalIntentionGenerationProposal(
            intentions=intentions,
            means=means,
            features=features,
            hidden=hidden,
            noise=noise,
            log_propensities=log_propensities,
            noise_scale=self.noise_scale,
        )
        return proposal.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            batch=batch_size,
            feature_width=self.feature_width,
        )

    def _proposal_gradients(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionGenerationProposal,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        score = proposal.noise / proposal.noise_scale
        output_weight_gradient = torch.einsum(
            "bo,bh->boh", score, proposal.hidden
        )
        output_bias_gradient = score
        hidden_score = torch.einsum(
            "bo,boh->bh", score, state.output_weights
        ) * (1.0 - proposal.hidden.square())
        input_weight_gradient = torch.einsum(
            "bh,bf->bhf", hidden_score, self.content_features(proposal.features)
        )
        input_bias_gradient = hidden_score
        residual_weight_gradient = torch.einsum(
            "bo,bf->bof",
            score,
            self.residual_features(proposal.features),
        )
        if not self.factorized_context_residual:
            residual_weight_gradient.zero_()
        return (
            input_weight_gradient,
            input_bias_gradient,
            output_weight_gradient,
            output_bias_gradient,
            residual_weight_gradient,
        )

    def record_decision(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionGenerationProposal,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Put sampled score gradients into an external eligibility trace."""

        self._validate_state(state)
        proposal.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            batch=state.baseline.shape[0],
            feature_width=self.feature_width,
        )
        if proposal.intentions.device != state.baseline.device:
            raise ValueError("intention generator proposal is on the wrong device")
        batch_size = state.baseline.shape[0]
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=state.baseline.device)
        self._validate_presence(present, batch_size, device=state.baseline.device)
        gradients = self._proposal_gradients(state, proposal)
        decay = self.initial_trace_decay
        active = present
        next_eligibilities = tuple(
            torch.where(
                active.reshape(batch_size, *([1] * (gradient.ndim - 1))),
                decay * previous + gradient,
                previous,
            )
            for previous, gradient in zip(
                (
                    state.input_weight_eligibility,
                    state.input_bias_eligibility,
                    state.output_weight_eligibility,
                    state.output_bias_eligibility,
                    state.context_residual_eligibility,
                ),
                gradients,
                strict=True,
            )
        )
        next_state = ExternalOutcomeIntentionGeneratorState(
            input_weights=state.input_weights,
            input_bias=state.input_bias,
            output_weights=state.output_weights,
            output_bias=state.output_bias,
            input_weight_eligibility=next_eligibilities[0],
            input_bias_eligibility=next_eligibilities[1],
            output_weight_eligibility=next_eligibilities[2],
            output_bias_eligibility=next_eligibilities[3],
            context_residual_weights=state.context_residual_weights,
            context_residual_eligibility=next_eligibilities[4],
            baseline=state.baseline,
            decisions=state.decisions + present.long(),
            feedbacks=state.feedbacks,
            protected=state.protected,
        )
        self._validate_state(next_state)
        return next_state

    def apply_feedback(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
        baseline_override: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Apply delayed scalar feedback without changing protected cells."""

        self._validate_state(state)
        batch_size = state.baseline.shape[0]
        if outcome.shape != (batch_size,) or not bool(torch.isfinite(outcome).all()):
            raise ValueError("intention generator outcome must be finite [batch]")
        if bool(((outcome < 0.0) | (outcome > 1.0)).any()):
            raise ValueError("intention generator outcome must lie in [0, 1]")
        if outcome.device != state.baseline.device:
            raise ValueError("intention generator outcome is on the wrong device")
        if present is None:
            present = torch.ones(batch_size, dtype=torch.bool, device=outcome.device)
        if terminal is None:
            terminal = torch.zeros(batch_size, dtype=torch.bool, device=outcome.device)
        self._validate_presence(present, batch_size, device=outcome.device)
        self._validate_presence(terminal, batch_size, device=outcome.device)
        if baseline_override is not None:
            if baseline_override.shape != (batch_size,) or not bool(
                torch.isfinite(baseline_override).all()
            ):
                raise ValueError("intention generator baseline override is invalid")
            if bool(((baseline_override < 0.0) | (baseline_override > 1.0)).any()):
                raise ValueError("intention generator baseline override must lie in [0, 1]")
            if baseline_override.device != outcome.device:
                raise ValueError("intention generator baseline override is on the wrong device")
        mutable = present & ~state.protected
        centered = outcome - (
            state.baseline if baseline_override is None else baseline_override
        )
        rate = self.initial_learning_rate
        update_scale = rate * centered * mutable.to(dtype=state.baseline.dtype)

        def update_tensor(value: torch.Tensor, eligibility: torch.Tensor) -> torch.Tensor:
            return value + update_scale.reshape(batch_size, *([1] * (eligibility.ndim - 1))) * eligibility

        next_state = ExternalOutcomeIntentionGeneratorState(
            input_weights=update_tensor(state.input_weights, state.input_weight_eligibility),
            input_bias=update_tensor(state.input_bias, state.input_bias_eligibility),
            output_weights=update_tensor(state.output_weights, state.output_weight_eligibility),
            output_bias=update_tensor(state.output_bias, state.output_bias_eligibility),
            context_residual_weights=update_tensor(
                state.context_residual_weights,
                state.context_residual_eligibility,
            ),
            input_weight_eligibility=torch.where(
                (terminal & present).reshape(batch_size, 1, 1),
                torch.zeros_like(state.input_weight_eligibility),
                state.input_weight_eligibility,
            ),
            input_bias_eligibility=torch.where(
                (terminal & present).reshape(batch_size, 1),
                torch.zeros_like(state.input_bias_eligibility),
                state.input_bias_eligibility,
            ),
            output_weight_eligibility=torch.where(
                (terminal & present).reshape(batch_size, 1, 1),
                torch.zeros_like(state.output_weight_eligibility),
                state.output_weight_eligibility,
            ),
            output_bias_eligibility=torch.where(
                (terminal & present).reshape(batch_size, 1),
                torch.zeros_like(state.output_bias_eligibility),
                state.output_bias_eligibility,
            ),
            context_residual_eligibility=torch.where(
                (terminal & present).reshape(batch_size, 1, 1),
                torch.zeros_like(state.context_residual_eligibility),
                state.context_residual_eligibility,
            ),
            baseline=torch.where(
                mutable,
                state.baseline + self.initial_baseline_rate * centered,
                state.baseline,
            ),
            decisions=state.decisions,
            feedbacks=state.feedbacks + present.long(),
            protected=state.protected,
        )
        self._validate_state(next_state)
        return next_state

    def begin_episode(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Clear transient credit while preserving learned external content."""

        self._validate_state(state)
        next_state = ExternalOutcomeIntentionGeneratorState(
            input_weights=state.input_weights,
            input_bias=state.input_bias,
            output_weights=state.output_weights,
            output_bias=state.output_bias,
            input_weight_eligibility=torch.zeros_like(state.input_weight_eligibility),
            input_bias_eligibility=torch.zeros_like(state.input_bias_eligibility),
            output_weight_eligibility=torch.zeros_like(state.output_weight_eligibility),
            output_bias_eligibility=torch.zeros_like(state.output_bias_eligibility),
            context_residual_weights=state.context_residual_weights,
            context_residual_eligibility=torch.zeros_like(
                state.context_residual_eligibility
            ),
            baseline=state.baseline,
            decisions=state.decisions,
            feedbacks=state.feedbacks,
            protected=state.protected,
        )
        self._validate_state(next_state)
        return next_state

    def append_cell(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        *,
        source_cell: int | None = None,
    ) -> tuple[ExternalOutcomeIntentionGeneratorState, int]:
        """Append a fresh or copy-on-write generator cell.

        Existing cells are copied unchanged. A copied cell inherits only the
        learned generator content and baseline; eligibility and counters are
        reset, and the new cell remains mutable until a verifier promotes it.
        """

        self._validate_state(state)
        batch_size = state.baseline.shape[0]
        if source_cell is not None and (
            not isinstance(source_cell, int)
            or isinstance(source_cell, bool)
            or not 0 <= source_cell < batch_size
        ):
            raise ValueError("intention generator source cell is out of range")

        source_row = None if source_cell is None else source_cell
        if source_row is None:
            new_input_weights = self.initial_parameter_scale * torch.randn_like(
                state.input_weights[:1]
            )
            if self.context_masking:
                new_input_weights[
                    :, :, self.context_width : 2 * self.context_width
                ].zero_()
            new_input_bias = torch.zeros_like(state.input_bias[:1])
            new_output_weights = self.initial_parameter_scale * torch.randn_like(
                state.output_weights[:1]
            )
            new_output_bias = torch.zeros_like(state.output_bias[:1])
            new_context_residual_weights = torch.zeros_like(
                state.context_residual_weights[:1]
            )
            new_baseline = torch.full_like(state.baseline[:1], self.initial_baseline)
        else:
            new_input_weights = state.input_weights[source_row : source_row + 1].clone()
            if self.context_masking:
                new_input_weights[
                    :, :, self.context_width : 2 * self.context_width
                ].zero_()
            new_input_bias = state.input_bias[source_row : source_row + 1].clone()
            new_output_weights = state.output_weights[source_row : source_row + 1].clone()
            new_output_bias = state.output_bias[source_row : source_row + 1].clone()
            new_context_residual_weights = state.context_residual_weights[
                source_row : source_row + 1
            ].clone()
            new_baseline = state.baseline[source_row : source_row + 1].clone()
        next_state = ExternalOutcomeIntentionGeneratorState(
            input_weights=torch.cat((state.input_weights, new_input_weights), dim=0),
            input_bias=torch.cat((state.input_bias, new_input_bias), dim=0),
            output_weights=torch.cat((state.output_weights, new_output_weights), dim=0),
            output_bias=torch.cat((state.output_bias, new_output_bias), dim=0),
            context_residual_weights=torch.cat(
                (state.context_residual_weights, new_context_residual_weights), dim=0
            ),
            context_residual_eligibility=torch.cat(
                (
                    state.context_residual_eligibility,
                    torch.zeros_like(state.context_residual_eligibility[:1]),
                ),
                dim=0,
            ),
            input_weight_eligibility=torch.cat(
                (state.input_weight_eligibility, torch.zeros_like(state.input_weight_eligibility[:1])),
                dim=0,
            ),
            input_bias_eligibility=torch.cat(
                (state.input_bias_eligibility, torch.zeros_like(state.input_bias_eligibility[:1])),
                dim=0,
            ),
            output_weight_eligibility=torch.cat(
                (state.output_weight_eligibility, torch.zeros_like(state.output_weight_eligibility[:1])),
                dim=0,
            ),
            output_bias_eligibility=torch.cat(
                (state.output_bias_eligibility, torch.zeros_like(state.output_bias_eligibility[:1])),
                dim=0,
            ),
            baseline=torch.cat((state.baseline, new_baseline), dim=0),
            decisions=torch.cat(
                (state.decisions, torch.zeros_like(state.decisions[:1])), dim=0
            ),
            feedbacks=torch.cat(
                (state.feedbacks, torch.zeros_like(state.feedbacks[:1])), dim=0
            ),
            protected=torch.cat(
                (state.protected, torch.zeros_like(state.protected[:1])), dim=0
            ),
        )
        self._validate_state(next_state)
        return next_state, batch_size

    def protect(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        cell_indices: torch.Tensor | list[int] | tuple[int, ...],
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Protect learned external cells from later feedback updates."""

        self._validate_state(state)
        if isinstance(cell_indices, torch.Tensor):
            if cell_indices.ndim != 1 or cell_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("intention generator cell indices must be integer [n]")
            indices = cell_indices.detach().to(device=state.baseline.device, dtype=torch.long)
        else:
            indices = torch.tensor(cell_indices, device=state.baseline.device, dtype=torch.long)
        if indices.ndim != 1 or bool((indices < 0).any()) or bool(
            (indices >= state.baseline.shape[0]).any()
        ):
            raise ValueError("intention generator cell index is out of range")
        protected = state.protected.clone()
        protected[indices] = True
        next_state = ExternalOutcomeIntentionGeneratorState(
            input_weights=state.input_weights,
            input_bias=state.input_bias,
            output_weights=state.output_weights,
            output_bias=state.output_bias,
            context_residual_weights=state.context_residual_weights,
            context_residual_eligibility=state.context_residual_eligibility,
            input_weight_eligibility=state.input_weight_eligibility,
            input_bias_eligibility=state.input_bias_eligibility,
            output_weight_eligibility=state.output_weight_eligibility,
            output_bias_eligibility=state.output_bias_eligibility,
            baseline=state.baseline,
            decisions=state.decisions,
            feedbacks=state.feedbacks,
            protected=protected,
        )
        self._validate_state(next_state)
        return next_state

    def state_payload(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
    ) -> dict[str, object]:
        """Serialize exact external generator state for a memory file."""

        self._validate_state(state)
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "input_weights": state.input_weights.detach().cpu().clone(),
            "input_bias": state.input_bias.detach().cpu().clone(),
            "output_weights": state.output_weights.detach().cpu().clone(),
            "output_bias": state.output_bias.detach().cpu().clone(),
            "context_residual_weights": state.context_residual_weights.detach().cpu().clone(),
            "context_residual_eligibility": state.context_residual_eligibility.detach().cpu().clone(),
            "input_weight_eligibility": state.input_weight_eligibility.detach().cpu().clone(),
            "input_bias_eligibility": state.input_bias_eligibility.detach().cpu().clone(),
            "output_weight_eligibility": state.output_weight_eligibility.detach().cpu().clone(),
            "output_bias_eligibility": state.output_bias_eligibility.detach().cpu().clone(),
            "baseline": state.baseline.detach().cpu().clone(),
            "decisions": state.decisions.detach().cpu().clone(),
            "feedbacks": state.feedbacks.detach().cpu().clone(),
            "protected": state.protected.detach().cpu().clone(),
        }

    def state_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> ExternalOutcomeIntentionGeneratorState:
        if payload.get("schema") not in (
            self.schema,
            EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA_V1,
        ):
            raise ValueError("unsupported intention-generator state schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("intention-generator state configuration is invalid")
        if any(
            configuration.get(name) != value
            for name, value in (
                ("context_width", self.context_width),
                ("intention_width", self.intention_width),
                ("hidden_width", self.hidden_width),
            )
        ):
            raise ValueError("intention-generator state dimensions do not match")
        if configuration.get("feature_width", self.context_width + 1) != self.feature_width:
            raise ValueError("intention-generator state feature width does not match")
        if configuration.get("context_masking", False) != self.context_masking:
            raise ValueError("intention-generator state masking mode does not match")
        if configuration.get("mask_stable_content", False) != self.mask_stable_content:
            raise ValueError("intention-generator state content mode does not match")
        payload_factorized_residual = bool(
            configuration.get("factorized_context_residual", False)
        )
        if payload_factorized_residual and not self.factorized_context_residual:
            raise ValueError("intention-generator state residual mode does not match")
        names = (
            "input_weights",
            "input_bias",
            "output_weights",
            "output_bias",
            "input_weight_eligibility",
            "input_bias_eligibility",
            "output_weight_eligibility",
            "output_bias_eligibility",
            "baseline",
            "decisions",
            "feedbacks",
            "protected",
        )
        tensors = {name: payload.get(name) for name in names}
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("intention-generator state payload must contain tensors")
        residual_shape = (
            tensors["baseline"].shape[0],
            self.intention_width,
            self.context_width + 1,
        )
        for name in ("context_residual_weights", "context_residual_eligibility"):
            value = payload.get(name)
            if value is None:
                tensors[name] = torch.zeros(
                    residual_shape,
                    device=tensors["input_weights"].device,
                    dtype=tensors["input_weights"].dtype,
                )
            elif isinstance(value, torch.Tensor):
                tensors[name] = value
            else:
                raise TypeError(f"intention-generator state field {name!r} must be a tensor")
        state = ExternalOutcomeIntentionGeneratorState(**tensors)
        self._validate_state(state)
        return state


@dataclass(frozen=True)
class ExternalIntentionObservationReceipt:
    """Auditable external-memory write for one observed intention batch."""

    entry_indices: tuple[int, ...]
    added: tuple[bool, ...]
    outcome_observed: bool
    version: int
    record_count: int
    content_digest: str
    schema: str = EXTERNAL_INTENTION_OBSERVATION_SCHEMA

    def validate(self) -> ExternalIntentionObservationReceipt:
        if self.schema != EXTERNAL_INTENTION_OBSERVATION_SCHEMA:
            raise ValueError("unsupported intention-observation schema")
        if len(self.entry_indices) != len(self.added) or not self.entry_indices:
            raise ValueError("intention-observation receipt is empty or misaligned")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in self.entry_indices
        ):
            raise ValueError("intention-observation entry index is invalid")
        if not all(isinstance(value, bool) for value in self.added):
            raise TypeError("intention-observation added flags must be boolean")
        if not isinstance(self.outcome_observed, bool):
            raise TypeError("intention-observation outcome flag must be boolean")
        if min(self.version, self.record_count) < 1:
            raise ValueError("intention-observation receipt version is invalid")
        if not isinstance(self.content_digest, str) or not self.content_digest:
            raise ValueError("intention-observation digest is missing")
        return self


@dataclass(frozen=True)
class ExternalIntentionAdmissionReceipt:
    """Copy-on-write admission result for one novel opaque intention."""

    accepted: bool
    entry_index: int | None
    source_record_count: int
    destination_record_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    schema: str = EXTERNAL_INTENTION_ADMISSION_SCHEMA

    def validate(self) -> ExternalIntentionAdmissionReceipt:
        if self.schema != EXTERNAL_INTENTION_ADMISSION_SCHEMA:
            raise ValueError("unsupported intention-admission schema")
        if min(self.source_record_count, self.destination_record_count) < 0:
            raise ValueError("intention-admission record counts cannot be negative")
        if self.accepted:
            if self.entry_index is None or self.entry_index < 0:
                raise ValueError("accepted intention admission has no entry index")
            if self.destination_record_count != self.source_record_count + 1:
                raise ValueError("accepted intention admission has wrong growth")
        elif self.entry_index is not None:
            raise ValueError("rejected intention admission has an entry index")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"intention-admission {name} is missing")
        return self


@dataclass(frozen=True)
class ExternalIntentionExplorationProposal:
    """Ephemeral candidates composed from verified opaque intention entries."""

    intentions: torch.Tensor
    source_pairs: tuple[tuple[int, int], ...]
    operations: tuple[str, ...]
    version: int
    schema: str = EXTERNAL_INTENTION_EXPLORATION_SCHEMA

    def validate(self, *, width: int) -> ExternalIntentionExplorationProposal:
        if self.schema != EXTERNAL_INTENTION_EXPLORATION_SCHEMA:
            raise ValueError("unsupported intention-exploration schema")
        _validate_tensor(
            self.intentions,
            name="intention exploration proposal",
            ndim=2,
            width=width,
        )
        count = self.intentions.shape[0]
        if len(self.source_pairs) != count or len(self.operations) != count:
            raise ValueError("intention exploration metadata is misaligned")
        for pair in self.source_pairs:
            if (
                len(pair) != 2
                or any(
                    not isinstance(index, int) or isinstance(index, bool) or index < 0
                    for index in pair
                )
                or pair[0] == pair[1]
            ):
                raise ValueError("intention exploration source pair is invalid")
        if any(not isinstance(operation, str) or not operation for operation in self.operations):
            raise ValueError("intention exploration operation is missing")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("intention exploration version is invalid")
        return self


@dataclass(frozen=True)
class ExternalIntentionConsolidationReceipt:
    """Retention-gated copy-on-write consolidation result."""

    accepted: bool
    retired_ids: tuple[int, ...]
    replacement_id: int | None
    source_record_count: int
    destination_record_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    version: int
    schema: str = EXTERNAL_INTENTION_CONSOLIDATION_SCHEMA

    def validate(self) -> ExternalIntentionConsolidationReceipt:
        if self.schema != EXTERNAL_INTENTION_CONSOLIDATION_SCHEMA:
            raise ValueError("unsupported intention-consolidation schema")
        if not self.retired_ids or len(set(self.retired_ids)) != len(self.retired_ids):
            raise ValueError("intention-consolidation retired IDs are invalid")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in self.retired_ids
        ):
            raise ValueError("intention-consolidation retired ID is invalid")
        if self.accepted:
            if (
                self.replacement_id is None
                or isinstance(self.replacement_id, bool)
                or self.replacement_id < 0
            ):
                raise ValueError("accepted intention consolidation has no replacement ID")
            if self.destination_record_count != self.source_record_count - len(
                self.retired_ids
            ) + 1:
                raise ValueError("accepted intention consolidation count is invalid")
        elif self.replacement_id is not None:
            raise ValueError("rejected intention consolidation has a replacement ID")
        if min(self.source_record_count, self.destination_record_count) < 1:
            raise ValueError("intention-consolidation record counts are invalid")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise ValueError("intention-consolidation version is invalid")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"intention-consolidation {name} is missing")
        return self


class ExternalIntentionCompositionExplorer:
    """Generate verifier-bound ephemeral intentions from retained experience.

    The explorer knows only vector algebra and opaque external entry indices.
    It never scores a candidate by reward and never mutates the repertoire;
    factual held-out verification remains the sole admission authority.
    """

    schema = EXTERNAL_INTENTION_EXPLORATION_SCHEMA
    _SUPPORTED_OPERATIONS = ("mean", "sum", "difference")

    def __init__(
        self,
        operations: tuple[str, ...] = ("mean", "sum", "difference"),
        *,
        merge_cosine: float = 0.999,
    ) -> None:
        if not operations:
            raise ValueError("intention explorer needs one operation")
        if any(operation not in self._SUPPORTED_OPERATIONS for operation in operations):
            raise ValueError("intention explorer operation is unsupported")
        if len(set(operations)) != len(operations):
            raise ValueError("intention explorer operations must be unique")
        if not -1.0 <= merge_cosine <= 1.0 or not math.isfinite(merge_cosine):
            raise ValueError("intention explorer merge cosine is invalid")
        self.operations = tuple(operations)
        self.merge_cosine = float(merge_cosine)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "operations": list(self.operations),
            "merge_cosine": self.merge_cosine,
            "behavior": "ephemeral_opaque_composition_before_heldout_admission_v1",
            "policy": "none_reward_ranking_disabled_v1",
        }

    @staticmethod
    def _similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        if float(left_norm) <= 1e-12 or float(right_norm) <= 1e-12:
            return 1.0 if torch.equal(left, right) else -1.0
        return float(torch.dot(left, right) / (left_norm * right_norm))

    def propose(
        self,
        repertoire: ExternalIntentionRepertoire,
        *,
        max_candidates: int | None = None,
    ) -> ExternalIntentionExplorationProposal:
        if not isinstance(repertoire, ExternalIntentionRepertoire):
            raise TypeError("intention exploration requires an external repertoire")
        if max_candidates is not None and (
            not isinstance(max_candidates, int)
            or isinstance(max_candidates, bool)
            or max_candidates < 1
        ):
            raise ValueError("maximum exploration candidates must be positive")
        stored = repertoire.statistics()["intentions"]
        candidates: list[torch.Tensor] = []
        pairs: list[tuple[int, int]] = []
        operations: list[str] = []
        for left_index in range(stored.shape[0]):
            for right_index in range(left_index + 1, stored.shape[0]):
                left = stored[left_index]
                right = stored[right_index]
                for operation in self.operations:
                    if operation == "mean":
                        candidate = 0.5 * (left + right)
                    elif operation == "sum":
                        candidate = left + right
                    else:
                        candidate = left - right
                    if not bool(torch.isfinite(candidate).all()):
                        continue
                    if any(
                        self._similarity(candidate, existing) >= self.merge_cosine
                        for existing in [*stored, *candidates]
                    ):
                        continue
                    candidates.append(candidate.detach().clone())
                    pairs.append(
                        (
                            repertoire.logical_id_at(left_index),
                            repertoire.logical_id_at(right_index),
                        )
                    )
                    operations.append(operation)
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
            pairs = pairs[:max_candidates]
            operations = operations[:max_candidates]
        intentions = (
            torch.stack(candidates)
            if candidates
            else torch.empty((0, repertoire.width), dtype=torch.float32)
        )
        return ExternalIntentionExplorationProposal(
            intentions=intentions,
            source_pairs=tuple(pairs),
            operations=tuple(operations),
            version=repertoire.version,
        ).validate(width=repertoire.width)


class ExternalIntentionRepertoire:
    """Append-only memory of opaque intention vectors.

    Entries are identified only by their position in this external store.  A
    cosine merge threshold prevents duplicate writes, while the original
    vector is retained so intention magnitude remains part of the learned
    representation.  ``observe`` accepts a scalar verifier outcome and its
    exact logging propensity, accumulating sufficient statistics without
    replaying old evidence or changing controller parameters.
    """

    schema = EXTERNAL_INTENTION_REPERTOIRE_SCHEMA

    def __init__(self, width: int, *, merge_cosine: float = 0.999) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("intention repertoire width must be positive")
        if not -1.0 <= merge_cosine <= 1.0 or not math.isfinite(merge_cosine):
            raise ValueError("intention repertoire merge cosine is invalid")
        self.width = int(width)
        self.merge_cosine = float(merge_cosine)
        self._intentions: list[torch.Tensor] = []
        self._attempts: list[int] = []
        self._outcome_counts: list[int] = []
        self._utility_sums: list[float] = []
        self._utility_square_sums: list[float] = []
        self._propensity_sums: list[float] = []
        self._inverse_propensity_utility_sums: list[float] = []
        self._last_propensities: list[float] = []
        self._last_seen: list[int] = []
        self._version = 0
        self._logical_ids: list[int] = []
        self._next_logical_id = 0
        self._aliases: dict[int, int] = {}

    @property
    def record_count(self) -> int:
        return len(self._intentions)

    @property
    def version(self) -> int:
        return self._version

    @property
    def logical_ids(self) -> tuple[int, ...]:
        """Return stable logical IDs in current physical proposal order."""

        return tuple(self._logical_ids)

    def logical_id_at(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.record_count:
            raise IndexError("intention physical index is out of range")
        return self._logical_ids[index]

    def resolve_logical_id(self, logical_id: int) -> int:
        """Resolve a retired logical ID to its retained replacement."""

        if not isinstance(logical_id, int) or isinstance(logical_id, bool) or logical_id < 0:
            raise ValueError("intention logical ID is invalid")
        seen: set[int] = set()
        current = logical_id
        while current in self._aliases:
            if current in seen:
                raise RuntimeError("intention logical-ID alias cycle detected")
            seen.add(current)
            current = self._aliases[current]
        return current

    def physical_index_for_id(self, logical_id: int) -> int:
        resolved = self.resolve_logical_id(logical_id)
        try:
            return self._logical_ids.index(resolved)
        except ValueError as error:
            raise KeyError(f"unknown intention logical ID: {logical_id}") from error

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "storage": "logical_addressed_opaque_intention_experience_v1",
            "proposal": (
                "verified_retrieval_default_plus_explicit_ephemeral_controller_seed_v1"
            ),
            "learning": "outcome_sufficient_statistics_without_replay_v1",
            "outcome_presence": "explicit_masked_scalar_verifier_presence_v1",
            "logical_addresses": "stable_ids_with_persisted_aliases_v1",
            "maintenance": "retention_gated_copy_on_write_consolidation_v1",
        }

    def _validate_batch(
        self,
        intentions: torch.Tensor,
        utility: torch.Tensor | float | None,
        propensity: torch.Tensor | float | None,
        timestamp: torch.Tensor | int | None,
        outcome_mask: torch.Tensor | bool | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if intentions.ndim == 1:
            intentions = intentions.unsqueeze(0)
        _validate_tensor(
            intentions,
            name="observed intention",
            ndim=2,
            width=self.width,
        )
        batch = intentions.shape[0]
        if batch < 1:
            raise ValueError("intention observation batch cannot be empty")

        def scalar_batch(
            value: torch.Tensor | float | None,
            *,
            name: str,
            default: float,
        ) -> torch.Tensor:
            if value is None:
                result = torch.full((batch,), default, dtype=torch.float64)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    result = value.reshape(1).expand(batch)
                elif value.shape == (batch,) or value.shape == (batch, 1):
                    result = value.reshape(batch)
                else:
                    raise ValueError(f"{name} must contain one value per intention")
                result = result.detach().to(device="cpu", dtype=torch.float64)
            else:
                result = torch.full((batch,), float(value), dtype=torch.float64)
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"{name} must be finite")
            return result

        utility_values = None if utility is None else scalar_batch(
            utility, name="intention utility", default=0.0
        )
        if outcome_mask is None:
            outcome_values = torch.full((batch,), utility_values is not None, dtype=torch.bool)
        elif isinstance(outcome_mask, torch.Tensor):
            if outcome_mask.ndim == 0:
                outcome_values = outcome_mask.reshape(1).expand(batch)
            elif outcome_mask.shape == (batch,) or outcome_mask.shape == (batch, 1):
                outcome_values = outcome_mask.reshape(batch)
            else:
                raise ValueError("intention outcome mask must contain one value per intention")
            if outcome_values.dtype != torch.bool:
                raise TypeError("intention outcome mask must be boolean")
            outcome_values = outcome_values.detach().to(device="cpu")
        elif isinstance(outcome_mask, bool):
            outcome_values = torch.full((batch,), bool(outcome_mask), dtype=torch.bool)
        else:
            raise TypeError("intention outcome mask must be boolean")
        if utility_values is None and bool(outcome_values.any()):
            raise ValueError("an intention outcome mask requires utility values")
        propensity_values = scalar_batch(
            propensity, name="intention logging propensity", default=1.0
        )
        if bool(torch.any(propensity_values <= 0.0)) or bool(
            torch.any(propensity_values > 1.0)
        ):
            raise ValueError("intention logging propensities must lie in (0, 1]")

        if timestamp is None:
            timestamp_values = torch.full(
                (batch,), self._version + 1, dtype=torch.int64
            )
        elif isinstance(timestamp, torch.Tensor):
            if timestamp.ndim == 0:
                timestamp_values = timestamp.reshape(1).expand(batch)
            elif timestamp.shape == (batch,) or timestamp.shape == (batch, 1):
                timestamp_values = timestamp.reshape(batch)
            else:
                raise ValueError("intention timestamps must contain one value per intention")
            if timestamp_values.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("intention timestamps must be integer tensors")
            timestamp_values = timestamp_values.detach().to(device="cpu", dtype=torch.int64)
        else:
            timestamp_values = torch.full((batch,), int(timestamp), dtype=torch.int64)
        if bool(torch.any(timestamp_values < 0)):
            raise ValueError("intention timestamps cannot be negative")
        return (
            intentions.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            utility_values,
            propensity_values,
            timestamp_values,
            outcome_values,
        )

    def _entry_similarity(self, left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        if float(left_norm) <= 1e-12 or float(right_norm) <= 1e-12:
            return 1.0 if torch.equal(left, right) else -1.0
        return float(torch.dot(left, right) / (left_norm * right_norm))

    def _find_entry(self, intention: torch.Tensor) -> int | None:
        for index, stored in enumerate(self._intentions):
            if self._entry_similarity(stored, intention) >= self.merge_cosine:
                return index
        return None

    def _prefix_digest(self, count: int) -> str:
        if not isinstance(count, int) or count < 0 or count > self.record_count:
            raise ValueError("intention prefix count is invalid")
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(str(self.width).encode("utf-8"))
        digest.update(str(count).encode("utf-8"))
        tensors = self.statistics()
        for name in sorted(tensors):
            value = tensors[name][:count].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def admit_verified(
        self,
        intention: torch.Tensor,
        verifier: Callable[[ExternalIntentionRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_verifier",
    ) -> ExternalIntentionAdmissionReceipt:
        """Admit one novel vector only through an isolated verifier transaction.

        ``verifier`` receives a copy containing the staged vector. It may run
        an independent held-out factual probe and may record the new outcome
        on that copy. Existing entries must remain byte-equivalent and the
        verifier may not add a second entry. Rejection, including a mutating
        verifier, leaves the live repertoire unchanged.
        """

        if not callable(verifier):
            raise TypeError("intention admission verifier must be callable")
        if not isinstance(reason, str) or not reason:
            raise ValueError("intention admission reason is missing")
        normalized, _utility, _propensity, _timestamp, _outcome_mask = self._validate_batch(
            intention,
            None,
            None,
            None,
            None,
        )
        if normalized.shape[0] != 1:
            raise ValueError("intention admission accepts one vector")
        candidate_intention = normalized[0]
        source_count = self.record_count
        source_digest = self.content_digest()
        if self._find_entry(candidate_intention) is not None:
            return ExternalIntentionAdmissionReceipt(
                accepted=False,
                entry_index=None,
                source_record_count=source_count,
                destination_record_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                destination_digest=source_digest,
                reason="intention already exists in verified repertoire",
            ).validate()

        candidate = ExternalIntentionRepertoire.from_payload(self.payload())
        candidate.observe(candidate_intention)
        candidate_digest = candidate.content_digest()
        accepted = bool(verifier(candidate))
        prefix_unchanged = candidate._prefix_digest(source_count) == self._prefix_digest(
            source_count
        )
        shape_unchanged = candidate.record_count == source_count + 1
        staged_vector_unchanged = shape_unchanged and torch.equal(
            candidate.statistics()["intentions"][source_count], candidate_intention
        )
        accepted = accepted and prefix_unchanged and shape_unchanged and staged_vector_unchanged
        if accepted:
            self._copy_from(candidate)
            destination_digest = self.content_digest()
            return ExternalIntentionAdmissionReceipt(
                accepted=True,
                entry_index=candidate.logical_id_at(candidate.record_count - 1),
                source_record_count=source_count,
                destination_record_count=self.record_count,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
                destination_digest=destination_digest,
                reason=reason,
            ).validate()
        return ExternalIntentionAdmissionReceipt(
            accepted=False,
            entry_index=None,
            source_record_count=source_count,
            destination_record_count=source_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=source_digest,
            reason=(
                "heldout verifier rejected or mutated retained intention state"
            ),
        ).validate()

    def _copy_from(self, other: ExternalIntentionRepertoire) -> None:
        if not isinstance(other, ExternalIntentionRepertoire):
            raise TypeError("intention repertoire replacement must use same type")
        if self.width != other.width or self.merge_cosine != other.merge_cosine:
            raise ValueError("intention repertoire replacement configuration differs")
        other.validate_state()
        self._intentions = [row.clone() for row in other._intentions]
        self._attempts = list(other._attempts)
        self._outcome_counts = list(other._outcome_counts)
        self._utility_sums = list(other._utility_sums)
        self._utility_square_sums = list(other._utility_square_sums)
        self._propensity_sums = list(other._propensity_sums)
        self._inverse_propensity_utility_sums = list(
            other._inverse_propensity_utility_sums
        )
        self._last_propensities = list(other._last_propensities)
        self._last_seen = list(other._last_seen)
        self._version = other._version
        self._logical_ids = list(other._logical_ids)
        self._next_logical_id = other._next_logical_id
        self._aliases = dict(other._aliases)

    def observe(
        self,
        intentions: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
        outcome_mask: torch.Tensor | bool | None = None,
    ) -> ExternalIntentionObservationReceipt:
        """Record opaque output experience without touching controller weights.

        ``outcome_mask`` keeps delayed or missing verifier evidence explicit in
        a batch.  An intention can therefore be attempted and retained before
        its scalar utility is available, without turning absence into a false
        negative.
        """

        (
            normalized_intentions,
            utility_values,
            propensity_values,
            timestamp_values,
            outcome_values,
        ) = self._validate_batch(
            intentions,
            utility,
            propensity,
            timestamp,
            outcome_mask,
        )
        located: list[int] = []
        added: list[bool] = []
        for row in normalized_intentions:
            index = self._find_entry(row)
            if index is None:
                self._intentions.append(row.clone())
                self._attempts.append(0)
                self._outcome_counts.append(0)
                self._utility_sums.append(0.0)
                self._utility_square_sums.append(0.0)
                self._propensity_sums.append(0.0)
                self._inverse_propensity_utility_sums.append(0.0)
                self._last_propensities.append(0.0)
                self._last_seen.append(0)
                index = len(self._intentions) - 1
                if index != len(self._logical_ids):
                    raise RuntimeError("intention storage appended out of order")
                self._logical_ids.append(self._next_logical_id)
                self._next_logical_id += 1
                added.append(True)
            else:
                added.append(False)
            located.append(index)

        self._version += 1
        for row_index, entry_index in enumerate(located):
            self._attempts[entry_index] += 1
            propensity_value = float(propensity_values[row_index])
            self._propensity_sums[entry_index] += propensity_value
            self._last_propensities[entry_index] = propensity_value
            self._last_seen[entry_index] = int(timestamp_values[row_index])
            if utility_values is not None and bool(outcome_values[row_index]):
                utility_value = float(utility_values[row_index])
                self._outcome_counts[entry_index] += 1
                self._utility_sums[entry_index] += utility_value
                self._utility_square_sums[entry_index] += utility_value * utility_value
                self._inverse_propensity_utility_sums[entry_index] += (
                    utility_value / propensity_value
                )
        self.validate_state()
        return ExternalIntentionObservationReceipt(
            entry_indices=tuple(located),
            added=tuple(added),
            outcome_observed=bool(outcome_values.any()),
            version=self._version,
            record_count=self.record_count,
            content_digest=self.content_digest(),
        ).validate()

    def _consolidation_candidate(
        self,
        retired_ids: tuple[int, ...],
        replacement_intention: torch.Tensor,
    ) -> tuple[ExternalIntentionRepertoire, int]:
        if len(retired_ids) < 2:
            raise ValueError("intention consolidation needs two records")
        if len(set(retired_ids)) != len(retired_ids):
            raise ValueError("intention consolidation IDs are duplicated")
        if any(
            not isinstance(logical_id, int)
            or isinstance(logical_id, bool)
            or logical_id < 0
            for logical_id in retired_ids
        ):
            raise ValueError("intention consolidation ID is invalid")
        if any(logical_id not in self._logical_ids for logical_id in retired_ids):
            raise ValueError("intention consolidation can retire live IDs only")
        normalized, _utility, _propensity, _timestamp, _outcome_mask = self._validate_batch(
            replacement_intention,
            None,
            None,
            None,
            None,
        )
        if normalized.shape[0] != 1:
            raise ValueError("intention consolidation accepts one vector")
        replacement = normalized[0]
        retired_indices = tuple(
            sorted(self._logical_ids.index(logical_id) for logical_id in retired_ids)
        )
        retired_index_set = set(retired_indices)
        retained_indices = tuple(
            index
            for index in range(self.record_count)
            if index not in retired_index_set
        )
        if any(
            self._entry_similarity(self._intentions[index], replacement)
            >= self.merge_cosine
            for index in retained_indices
        ):
            raise ValueError(
                "intention consolidation replacement duplicates a retained vector"
            )

        candidate = ExternalIntentionRepertoire.from_payload(self.payload())
        candidate._intentions = [
            self._intentions[index].clone() for index in retained_indices
        ] + [replacement.clone()]

        def aggregate(values: list[int | float]) -> list[int | float]:
            retained = [values[index] for index in retained_indices]
            combined = sum(values[index] for index in retired_indices)
            return retained + [combined]

        candidate._attempts = [int(value) for value in aggregate(self._attempts)]
        candidate._outcome_counts = [
            int(value) for value in aggregate(self._outcome_counts)
        ]
        candidate._utility_sums = [
            float(value) for value in aggregate(self._utility_sums)
        ]
        candidate._utility_square_sums = [
            float(value) for value in aggregate(self._utility_square_sums)
        ]
        candidate._propensity_sums = [
            float(value) for value in aggregate(self._propensity_sums)
        ]
        candidate._inverse_propensity_utility_sums = [
            float(value)
            for value in aggregate(self._inverse_propensity_utility_sums)
        ]
        latest_index = max(
            retired_indices,
            key=lambda index: (self._last_seen[index], index),
        )
        candidate._last_propensities = [
            self._last_propensities[index] for index in retained_indices
        ] + [self._last_propensities[latest_index]]
        candidate._last_seen = [self._last_seen[index] for index in retained_indices] + [
            max(self._last_seen[index] for index in retired_indices)
        ]
        candidate._version = self._version + 1
        replacement_id = min(retired_ids)
        candidate._logical_ids = [
            self._logical_ids[index] for index in retained_indices
        ] + [replacement_id]
        retired_set = set(retired_ids)
        candidate._aliases = {
            source: (
                replacement_id
                if self.resolve_logical_id(destination) in retired_set
                else self.resolve_logical_id(destination)
            )
            for source, destination in self._aliases.items()
        }
        for logical_id in retired_ids:
            if logical_id != replacement_id:
                candidate._aliases[logical_id] = replacement_id
        candidate._next_logical_id = max(self._next_logical_id, replacement_id + 1)
        candidate.validate_state()
        return candidate, replacement_id

    def consolidate_verified(
        self,
        retired_ids: tuple[int, ...] | list[int],
        replacement_intention: torch.Tensor,
        retention_probe: Callable[[ExternalIntentionRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_retention_probe",
    ) -> ExternalIntentionConsolidationReceipt:
        """Compact intention memory only after an isolated retention probe passes."""

        if not callable(retention_probe):
            raise TypeError("intention consolidation retention probe must be callable")
        if not isinstance(reason, str) or not reason:
            raise ValueError("intention consolidation reason is missing")
        normalized_ids = tuple(retired_ids)
        source_count = self.record_count
        source_digest = self.content_digest()
        candidate, replacement_id = self._consolidation_candidate(
            normalized_ids,
            replacement_intention,
        )
        candidate_digest = candidate.content_digest()
        accepted = bool(retention_probe(candidate))
        probe_unchanged = candidate.content_digest() == candidate_digest
        accepted = accepted and probe_unchanged
        if accepted:
            self._copy_from(candidate)
            return ExternalIntentionConsolidationReceipt(
                accepted=True,
                retired_ids=normalized_ids,
                replacement_id=replacement_id,
                source_record_count=source_count,
                destination_record_count=self.record_count,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
                destination_digest=self.content_digest(),
                reason=reason,
                version=self.version,
            ).validate()
        return ExternalIntentionConsolidationReceipt(
            accepted=False,
            retired_ids=normalized_ids,
            replacement_id=None,
            source_record_count=source_count,
            destination_record_count=source_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=source_digest,
            reason="heldout retention probe rejected or mutated candidate intention state",
            version=self.version,
        ).validate()

    def propose(
        self,
        seed_intention: torch.Tensor | None = None,
        *,
        max_candidates: int | None = None,
        include_seed: bool = True,
    ) -> ExternalIntentionProposal:
        """Expose stored experience plus an ephemeral controller seed.

        The proposal is a candidate set, not a decision.  It contains no
        reward-ranked ordering and does not mutate the repertoire.  A novel
        controller seed is marked as exploration until a later observation
        commits it to external memory.  Callers that already have verified
        candidates should set ``include_seed=False`` so an unverified vector
        cannot contaminate factual search; the policy-free runtime does this
        by default and falls back to the seed only for an empty repertoire.
        """

        if not isinstance(include_seed, bool):
            raise TypeError("intention proposal include_seed must be boolean")

        if seed_intention is None:
            seed_batch = None
            output_device = torch.device("cpu")
            output_dtype = torch.float32
        else:
            if seed_intention.ndim == 1:
                seed_batch = seed_intention.unsqueeze(0)
            elif seed_intention.ndim == 2:
                seed_batch = seed_intention
            else:
                raise ValueError("seed intention must be [width] or [batch,width]")
            _validate_tensor(
                seed_batch,
                name="seed intention",
                ndim=2,
                width=self.width,
            )
            output_device = seed_intention.device
            output_dtype = (
                seed_intention.dtype
                if seed_intention.is_floating_point()
                else torch.float32
            )

        stored = [row.clone() for row in self._intentions]
        rows: list[torch.Tensor] = []
        source_indices: list[int] = []
        exploration_flags: list[bool] = []
        if seed_batch is not None and include_seed:
            for seed in seed_batch.detach().to(device="cpu", dtype=torch.float32):
                matching_index = self._find_entry(seed)
                if matching_index is None:
                    if not any(
                        self._entry_similarity(existing, seed) >= self.merge_cosine
                        for existing in rows
                    ):
                        rows.append(seed.clone())
                        source_indices.append(-1)
                        exploration_flags.append(True)
                elif self._logical_ids[matching_index] not in source_indices:
                    rows.append(self._intentions[matching_index].clone())
                    source_indices.append(self._logical_ids[matching_index])
                    exploration_flags.append(False)
        for index, stored_row in enumerate(stored):
            logical_id = self._logical_ids[index]
            if logical_id not in source_indices:
                rows.append(stored_row)
                source_indices.append(logical_id)
                exploration_flags.append(False)
        if not rows:
            raise ValueError("intention repertoire cannot propose an empty set")
        if max_candidates is not None:
            if not isinstance(max_candidates, int) or max_candidates < 1:
                raise ValueError("maximum intention candidate count must be positive")
            if max_candidates < sum(exploration_flags):
                raise ValueError("maximum candidate count would discard an exploration seed")
            if len(rows) > max_candidates:
                keep = list(range(max_candidates))
                rows = [rows[index] for index in keep]
                source_indices = [source_indices[index] for index in keep]
                exploration_flags = [exploration_flags[index] for index in keep]
        candidates = torch.stack(rows).to(device=output_device, dtype=output_dtype)
        batch = 1 if seed_batch is None else seed_batch.shape[0]
        intentions = candidates.unsqueeze(0).expand(batch, -1, -1).clone()
        exploration_mask = torch.tensor(
            exploration_flags,
            dtype=torch.bool,
            device=output_device,
        ).unsqueeze(0).expand(batch, -1).clone()
        propensities = torch.full(
            (batch, candidates.shape[0]),
            1.0 / candidates.shape[0],
            dtype=output_dtype,
            device=output_device,
        )
        return ExternalIntentionProposal(
            intentions=intentions,
            source_indices=tuple(source_indices),
            propensities=propensities,
            exploration_mask=exploration_mask,
            version=self._version,
        ).validate(width=self.width, batch=batch)

    def statistics(self) -> dict[str, torch.Tensor]:
        """Return detached sufficient statistics for external diagnostics."""

        self.validate_state()
        return {
            "intentions": self._stack_intentions(),
            "attempts": torch.tensor(self._attempts, dtype=torch.long),
            "outcome_counts": torch.tensor(self._outcome_counts, dtype=torch.long),
            "utility_sums": torch.tensor(self._utility_sums, dtype=torch.float64),
            "utility_square_sums": torch.tensor(
                self._utility_square_sums, dtype=torch.float64
            ),
            "propensity_sums": torch.tensor(
                self._propensity_sums, dtype=torch.float64
            ),
            "inverse_propensity_utility_sums": torch.tensor(
                self._inverse_propensity_utility_sums, dtype=torch.float64
            ),
            "last_propensities": torch.tensor(
                self._last_propensities, dtype=torch.float64
            ),
            "last_seen": torch.tensor(self._last_seen, dtype=torch.long),
        }

    def _stack_intentions(self) -> torch.Tensor:
        if not self._intentions:
            return torch.empty((0, self.width), dtype=torch.float32)
        return torch.stack(self._intentions).detach().clone()

    def validate_state(self) -> None:
        count = self.record_count
        if not isinstance(self._version, int) or self._version < 0:
            raise ValueError("intention repertoire version is invalid")
        lengths = (
            len(self._attempts),
            len(self._outcome_counts),
            len(self._utility_sums),
            len(self._utility_square_sums),
            len(self._propensity_sums),
            len(self._inverse_propensity_utility_sums),
            len(self._last_propensities),
            len(self._last_seen),
        )
        if any(length != count for length in lengths):
            raise ValueError("intention repertoire statistics are misaligned")
        for row in self._intentions:
            _validate_tensor(row, name="stored intention", ndim=1, width=self.width)
        for name, values in (
            ("attempts", self._attempts),
            ("outcome counts", self._outcome_counts),
            ("last seen", self._last_seen),
        ):
            if any(not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"intention repertoire {name} are invalid")
        for name, values in (
            ("utility sums", self._utility_sums),
            ("utility square sums", self._utility_square_sums),
            ("propensity sums", self._propensity_sums),
            ("inverse-propensity utility sums", self._inverse_propensity_utility_sums),
            ("last propensities", self._last_propensities),
        ):
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"intention repertoire {name} are not finite")
        if any(
            outcome_count > attempt
            for outcome_count, attempt in zip(
                self._outcome_counts, self._attempts, strict=True
            )
        ):
            raise ValueError("intention outcome counts exceed attempts")
        if len(self._logical_ids) != count:
            raise ValueError("intention logical IDs are misaligned")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in self._logical_ids
        ):
            raise ValueError("intention logical IDs are invalid")
        if len(set(self._logical_ids)) != len(self._logical_ids):
            raise ValueError("intention logical IDs are duplicated")
        if (
            not isinstance(self._next_logical_id, int)
            or isinstance(self._next_logical_id, bool)
            or self._next_logical_id < 0
        ):
            raise ValueError("intention next logical ID is invalid")
        if self._next_logical_id <= max(self._logical_ids, default=-1):
            raise ValueError("intention next logical ID is stale")
        if any(
            not isinstance(source, int)
            or isinstance(source, bool)
            or source < 0
            or not isinstance(destination, int)
            or isinstance(destination, bool)
            or destination < 0
            for source, destination in self._aliases.items()
        ):
            raise ValueError("intention logical-ID aliases are invalid")
        if set(self._aliases) & set(self._logical_ids):
            raise ValueError("intention logical-ID aliases shadow live IDs")
        for source, destination in self._aliases.items():
            if self.resolve_logical_id(destination) not in self._logical_ids:
                raise ValueError("intention logical-ID alias target is not live")
            if source == destination:
                raise ValueError("intention logical-ID alias is self-referential")

    @staticmethod
    def _digest_payload(payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(payload):
            if name == "sha256":
                continue
            value = payload[name]
            digest.update(name.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        self.validate_state()
        payload: dict[str, Any] = {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "version": self._version,
            "logical_ids": list(self._logical_ids),
            "next_logical_id": self._next_logical_id,
            "aliases": dict(sorted(self._aliases.items())),
            **self.statistics(),
        }
        payload["sha256"] = self._digest_payload(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExternalIntentionRepertoire:
        if payload.get("schema") != EXTERNAL_INTENTION_REPERTOIRE_SCHEMA:
            raise ValueError("unsupported intention repertoire payload")
        width = payload.get("width")
        merge_cosine = payload.get("merge_cosine")
        version = payload.get("version")
        logical_ids_payload = payload.get("logical_ids")
        next_logical_id_payload = payload.get("next_logical_id")
        aliases_payload = payload.get("aliases", {})
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError("intention repertoire payload width is invalid")
        if not isinstance(merge_cosine, (int, float)) or not math.isfinite(
            float(merge_cosine)
        ):
            raise ValueError("intention repertoire payload merge cosine is invalid")
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("intention repertoire payload version is invalid")
        expected_digest = payload.get("sha256")
        if expected_digest != cls._digest_payload(payload):
            raise ValueError("intention repertoire payload checksum mismatch")
        required = (
            "intentions",
            "attempts",
            "outcome_counts",
            "utility_sums",
            "utility_square_sums",
            "propensity_sums",
            "inverse_propensity_utility_sums",
            "last_propensities",
            "last_seen",
        )
        if any(name not in payload for name in required):
            raise ValueError("intention repertoire payload is incomplete")
        repertoire = cls(width, merge_cosine=float(merge_cosine))
        intentions = payload["intentions"]
        if not isinstance(intentions, torch.Tensor) or intentions.ndim != 2:
            raise ValueError("intention repertoire payload vectors are invalid")
        if intentions.shape[1] != width:
            raise ValueError("intention repertoire payload vector width differs")
        count = intentions.shape[0]
        tensors = {
            name: payload[name]
            for name in required
            if name != "intentions"
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != count:
                raise ValueError(f"intention repertoire payload {name} is misaligned")
        logical_ids = (
            list(range(count))
            if logical_ids_payload is None
            else logical_ids_payload
        )
        next_logical_id = count if next_logical_id_payload is None else next_logical_id_payload
        if not isinstance(logical_ids, list) or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in logical_ids
        ):
            raise ValueError("intention repertoire payload logical IDs are invalid")
        if not isinstance(next_logical_id, int) or isinstance(next_logical_id, bool) or next_logical_id < 0:
            raise ValueError("intention repertoire payload next logical ID is invalid")
        if not isinstance(aliases_payload, Mapping):
            raise TypeError("intention repertoire payload aliases are invalid")
        aliases: dict[int, int] = {}
        for source, destination in aliases_payload.items():
            if (
                not isinstance(source, int)
                or isinstance(source, bool)
                or not isinstance(destination, int)
                or isinstance(destination, bool)
            ):
                raise TypeError("intention repertoire payload aliases are invalid")
            aliases[int(source)] = int(destination)
        repertoire._intentions = [
            row.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for row in intentions
        ]
        repertoire._attempts = [int(value) for value in tensors["attempts"].tolist()]
        repertoire._outcome_counts = [
            int(value) for value in tensors["outcome_counts"].tolist()
        ]
        repertoire._utility_sums = [
            float(value) for value in tensors["utility_sums"].tolist()
        ]
        repertoire._utility_square_sums = [
            float(value) for value in tensors["utility_square_sums"].tolist()
        ]
        repertoire._propensity_sums = [
            float(value) for value in tensors["propensity_sums"].tolist()
        ]
        repertoire._inverse_propensity_utility_sums = [
            float(value)
            for value in tensors["inverse_propensity_utility_sums"].tolist()
        ]
        repertoire._last_propensities = [
            float(value) for value in tensors["last_propensities"].tolist()
        ]
        repertoire._last_seen = [int(value) for value in tensors["last_seen"].tolist()]
        repertoire._version = version
        repertoire._logical_ids = list(logical_ids)
        repertoire._next_logical_id = next_logical_id
        repertoire._aliases = aliases
        repertoire.validate_state()
        return repertoire

    def content_digest(self) -> str:
        return self._digest_payload(self.payload_without_digest())

    def payload_without_digest(self) -> dict[str, Any]:
        self.validate_state()
        return {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "version": self._version,
            "logical_ids": list(self._logical_ids),
            "next_logical_id": self._next_logical_id,
            "aliases": dict(sorted(self._aliases.items())),
            **self.statistics(),
        }


__all__ = [
    "EXTERNAL_INTENTION_ADMISSION_SCHEMA",
    "EXTERNAL_INTENTION_CONSOLIDATION_SCHEMA",
    "EXTERNAL_INTENTION_EXPLORATION_SCHEMA",
    "EXTERNAL_INTENTION_GENERATION_PROPOSAL_SCHEMA",
    "EXTERNAL_INTENTION_OBSERVATION_SCHEMA",
    "EXTERNAL_INTENTION_PROPOSAL_SCHEMA",
    "EXTERNAL_INTENTION_REPERTOIRE_SCHEMA",
    "EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA",
    "EXTERNAL_OUTCOME_INTENTION_GENERATOR_SCHEMA_V1",
    "ExternalIntentionAdmissionReceipt",
    "ExternalIntentionCompositionExplorer",
    "ExternalIntentionConsolidationReceipt",
    "ExternalIntentionExplorationProposal",
    "ExternalIntentionGenerationProposal",
    "ExternalIntentionObservationReceipt",
    "ExternalIntentionProposal",
    "ExternalIntentionRepertoire",
    "ExternalOutcomeIntentionGenerator",
    "ExternalOutcomeIntentionGeneratorState",
]
