"""Learned opaque routing over independent external intention cells."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from .intention import ExternalOutcomeIntentionGeneratorState
from .intention_memory import (
    ExternalIntentionMemoryProposal,
    ExternalOutcomeIntentionMemory,
)

EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V1 = (
    "neural-computer.external-routed-intention-memory.v1"
)
EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V2 = (
    "neural-computer.external-routed-intention-memory.v2"
)
EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V3 = (
    "neural-computer.external-routed-intention-memory.v3"
)
EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V4 = (
    "neural-computer.external-routed-intention-memory.v4"
)
EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA = (
    "neural-computer.external-routed-intention-memory.v5"
)
EXTERNAL_ROUTED_INTENTION_PROPOSAL_SCHEMA = (
    "neural-computer.external-routed-intention-proposal.v1"
)
EXTERNAL_ROUTED_INTENTION_RETENTION_VERIFICATION_SCHEMA = (
    "neural-computer.external-routed-intention-retention-verification.v1"
)


@dataclass(frozen=True)
class ExternalRoutedIntentionMemoryState:
    """Persistent memory cells plus context-to-cell route state."""

    cells: ExternalOutcomeIntentionGeneratorState
    routing_keys: torch.Tensor
    routing_bias: torch.Tensor
    routing_baseline: torch.Tensor
    routing_decisions: torch.Tensor
    routing_feedbacks: torch.Tensor
    retention_observations: torch.Tensor
    retention_successes: torch.Tensor
    retention_prefix_minima: torch.Tensor
    retention_reversal_streaks: torch.Tensor
    retention_reversal_counts: torch.Tensor
    retention_mastered: torch.Tensor
    retention_context_prototypes: torch.Tensor
    retention_context_masses: torch.Tensor
    retention_context_observed_masses: torch.Tensor
    retention_context_mask_profiles: torch.Tensor

    def validate(
        self,
        *,
        context_width: int,
        intention_width: int,
        hidden_width: int,
        routing_feature_width: int | None = None,
    ) -> None:
        expected_routing_feature_width = (
            context_width if routing_feature_width is None else routing_feature_width
        )
        if expected_routing_feature_width < context_width:
            raise ValueError("routed intention routing feature width is invalid")
        self.cells.validate(
            context_width=context_width,
            intention_width=intention_width,
            hidden_width=hidden_width,
            feature_width=self.cells.input_weights.shape[-1],
        )
        cell_count = self.cells.baseline.shape[0]
        expected = {
            "routing_keys": (cell_count, expected_routing_feature_width),
            "routing_bias": (cell_count,),
            "routing_baseline": (1,),
            "routing_decisions": (cell_count,),
            "routing_feedbacks": (cell_count,),
            "retention_observations": (cell_count,),
            "retention_successes": (cell_count,),
            "retention_prefix_minima": (cell_count,),
            "retention_reversal_streaks": (cell_count,),
            "retention_reversal_counts": (cell_count,),
            "retention_mastered": (cell_count,),
            "retention_context_prototypes": (cell_count, context_width),
            "retention_context_masses": (cell_count,),
            "retention_context_observed_masses": (cell_count, context_width),
            "retention_context_mask_profiles": (cell_count, context_width),
        }
        tensors = {
            "routing_keys": self.routing_keys,
            "routing_bias": self.routing_bias,
            "routing_baseline": self.routing_baseline,
            "routing_decisions": self.routing_decisions,
            "routing_feedbacks": self.routing_feedbacks,
            "retention_observations": self.retention_observations,
            "retention_successes": self.retention_successes,
            "retention_prefix_minima": self.retention_prefix_minima,
            "retention_reversal_streaks": self.retention_reversal_streaks,
            "retention_reversal_counts": self.retention_reversal_counts,
            "retention_mastered": self.retention_mastered,
            "retention_context_prototypes": self.retention_context_prototypes,
            "retention_context_masses": self.retention_context_masses,
            "retention_context_observed_masses": self.retention_context_observed_masses,
            "retention_context_mask_profiles": self.retention_context_mask_profiles,
        }
        for name, value in tensors.items():
            if value.shape != expected[name]:
                raise ValueError(f"routed intention {name} has the wrong shape")
            if value.device != self.cells.input_weights.device:
                raise ValueError("routed intention state tensors must share a device")
        for name in (
            "routing_keys",
            "routing_bias",
            "routing_baseline",
            "retention_successes",
            "retention_prefix_minima",
            "retention_context_prototypes",
            "retention_context_masses",
            "retention_context_observed_masses",
            "retention_context_mask_profiles",
        ):
            if not bool(torch.isfinite(tensors[name]).all()):
                raise ValueError(f"routed intention {name} must be finite")
        for name in (
            "routing_decisions",
            "routing_feedbacks",
            "retention_observations",
            "retention_reversal_streaks",
            "retention_reversal_counts",
        ):
            if tensors[name].dtype not in (torch.int32, torch.int64):
                raise TypeError(f"routed intention {name} must be integer")
            if bool((tensors[name] < 0).any()):
                raise ValueError(f"routed intention {name} cannot be negative")
        if tensors["retention_mastered"].dtype != torch.bool:
            raise TypeError("routed intention retention mastered must be boolean")
        if bool(
            (self.routing_baseline < 0.0).any()
            or (self.routing_baseline > 1.0).any()
            or (self.retention_prefix_minima < 0.0).any()
            or (self.retention_prefix_minima > 1.0).any()
        ):
            raise ValueError("routed intention baseline must lie in [0, 1]")
        if bool(
            (
                self.retention_successes
                > self.retention_observations.to(
                    dtype=self.retention_successes.dtype
                )
            ).any()
        ):
            raise ValueError("routed intention retention successes exceed observations")
        if bool((self.retention_context_masses < 0.0).any()):
            raise ValueError("routed intention retention context masses cannot be negative")
        if bool((self.retention_context_observed_masses < 0.0).any()):
            raise ValueError(
                "routed intention observed context masses cannot be negative"
            )
        if bool(
            (self.retention_context_mask_profiles < 0.0).any()
            or (self.retention_context_mask_profiles > 1.0).any()
        ):
            raise ValueError("routed intention context mask profiles must lie in [0, 1]")


@dataclass(frozen=True)
class ExternalRoutedIntentionProposal:
    """A routed cell choice plus all candidate content and route credit."""

    candidates: ExternalIntentionMemoryProposal
    selected_cells: torch.Tensor
    selected_intentions: torch.Tensor
    route_probabilities: torch.Tensor
    route_log_propensities: torch.Tensor
    route_key_gradients: torch.Tensor
    route_bias_gradients: torch.Tensor
    exploration_bonus: torch.Tensor
    temperature: float
    schema: str = EXTERNAL_ROUTED_INTENTION_PROPOSAL_SCHEMA

    def validate(
        self,
        *,
        context_width: int,
        intention_width: int,
        hidden_width: int,
        cell_count: int,
        batch: int | None = None,
        routing_feature_width: int | None = None,
    ) -> ExternalRoutedIntentionProposal:
        if self.schema != EXTERNAL_ROUTED_INTENTION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported routed intention proposal schema")
        self.candidates.validate(
            context_width=context_width,
            intention_width=intention_width,
            hidden_width=hidden_width,
            batch=batch,
            cell_count=cell_count,
            feature_width=self.candidates.features.shape[-1],
        )
        candidate_batch = self.candidates.intentions.shape[0]
        if self.selected_cells.shape != (candidate_batch,) or self.selected_cells.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("routed intention selected cells have the wrong shape")
        if self.selected_intentions.shape != (candidate_batch, intention_width):
            raise ValueError("routed intention selected content has the wrong shape")
        if self.route_probabilities.shape != (candidate_batch, cell_count):
            raise ValueError("routed intention probabilities have the wrong shape")
        if self.route_log_propensities.shape != (candidate_batch,):
            raise ValueError("routed intention propensities have the wrong shape")
        expected_routing_feature_width = (
            context_width
            if routing_feature_width is None
            else routing_feature_width
        )
        if expected_routing_feature_width < context_width:
            raise ValueError("routed intention proposal routing width is invalid")
        if self.route_key_gradients.shape != (
            candidate_batch,
            cell_count,
            expected_routing_feature_width,
        ):
            raise ValueError("routed intention key gradients have the wrong shape")
        if self.route_bias_gradients.shape != (candidate_batch, cell_count):
            raise ValueError("routed intention bias gradients have the wrong shape")
        if self.exploration_bonus.shape != (candidate_batch, cell_count):
            raise ValueError("routed intention exploration has the wrong shape")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0.0:
            raise ValueError("routed intention temperature is invalid")
        if bool((self.selected_cells < 0).any()) or bool(
            (self.selected_cells >= cell_count).any()
        ):
            raise ValueError("routed intention selected cell is out of range")
        candidate_indices = torch.tensor(
            self.candidates.cell_indices,
            device=self.selected_cells.device,
            dtype=torch.long,
        )
        if not bool(torch.isin(self.selected_cells, candidate_indices).all()):
            raise ValueError("routed intention selected cell is absent from candidates")
        for name, value in (
            ("selected_intentions", self.selected_intentions),
            ("route_probabilities", self.route_probabilities),
            ("route_log_propensities", self.route_log_propensities),
            ("route_key_gradients", self.route_key_gradients),
            ("route_bias_gradients", self.route_bias_gradients),
            ("exploration_bonus", self.exploration_bonus),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"routed intention {name} must be finite")
        if bool((self.route_probabilities < 0.0).any()) or not bool(
            torch.allclose(
                self.route_probabilities.sum(dim=-1),
                torch.ones(candidate_batch, device=self.route_probabilities.device),
                atol=1e-5,
                rtol=1e-5,
            )
        ):
            raise ValueError("routed intention probabilities must sum to one")
        return self


@dataclass(frozen=True)
class ExternalRoutedRetentionVerification:
    """Held-out prefix evidence for verifier-gated cell protection."""

    cell_index: int
    outcomes: tuple[float, ...]
    prefix_minimum: float
    mean_outcome: float
    floor: float
    accepted: bool
    context_relevance_minimum: float | None
    reason: str
    schema: str = EXTERNAL_ROUTED_INTENTION_RETENTION_VERIFICATION_SCHEMA

    def validate(self) -> ExternalRoutedRetentionVerification:
        if self.schema != EXTERNAL_ROUTED_INTENTION_RETENTION_VERIFICATION_SCHEMA:
            raise ValueError("unsupported routed retention verification schema")
        if not isinstance(self.cell_index, int) or isinstance(self.cell_index, bool):
            raise TypeError("routed retention cell index must be an integer")
        if self.cell_index < 0:
            raise ValueError("routed retention cell index cannot be negative")
        if not self.outcomes or not all(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
            for value in self.outcomes
        ):
            raise ValueError("routed retention outcomes must lie in [0, 1]")
        for name, value in (
            ("prefix minimum", self.prefix_minimum),
            ("mean outcome", self.mean_outcome),
            ("floor", self.floor),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"routed retention {name} is invalid")
        if not math.isclose(
            self.prefix_minimum,
            min(self.outcomes),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ) or not math.isclose(
            self.mean_outcome,
            sum(self.outcomes) / len(self.outcomes),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("routed retention summary does not match outcomes")
        if not isinstance(self.accepted, bool):
            raise TypeError("routed retention acceptance must be boolean")
        if self.context_relevance_minimum is not None and (
            not math.isfinite(float(self.context_relevance_minimum))
            or not -1.0 <= float(self.context_relevance_minimum) <= 1.0
        ):
            raise ValueError("routed retention context relevance is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("routed retention verification reason is missing")
        return self


class ExternalOutcomeIntentionRouter:
    """Select external intention cells from opaque context and scalar outcomes."""

    schema = EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA

    def __init__(
        self,
        memory: ExternalOutcomeIntentionMemory,
        *,
        initial_learning_rate: float = 0.1,
        initial_baseline_rate: float = 0.05,
        initial_baseline: float = 0.5,
        temperature: float = 1.0,
        exploration_bonus: float = 0.75,
        initial_routing_scale: float = 0.01,
        unqualified_cell_probability: float = 0.25,
        mastery_threshold: float = 0.95,
        min_mastery_feedbacks: int = 8,
        reversal_threshold: float = 0.5,
        reversal_patience: int = 4,
        context_relevance_threshold: float = 0.9,
        verified_prototype_scale: float = 3.0,
        verified_context_coverage_scale: float = 6.0,
        reversal_context_coverage_threshold: float = 0.75,
        masked_reversal_quarantine_scale: float = 6.0,
        context_mask_profile_rate: float = 0.25,
        context_mask_profile_scale: float = 0.0,
    ) -> None:
        if not isinstance(memory, ExternalOutcomeIntentionMemory):
            raise TypeError("intention router requires external intention memory")
        if not 0.0 < initial_learning_rate <= 1.0:
            raise ValueError("intention router learning rate is invalid")
        if not 0.0 < initial_baseline_rate <= 1.0:
            raise ValueError("intention router baseline rate is invalid")
        if not 0.0 <= initial_baseline <= 1.0:
            raise ValueError("intention router baseline is invalid")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("intention router temperature is invalid")
        if not math.isfinite(exploration_bonus) or exploration_bonus < 0.0:
            raise ValueError("intention router exploration bonus is invalid")
        if not math.isfinite(initial_routing_scale) or initial_routing_scale <= 0.0:
            raise ValueError("intention router initialization scale is invalid")
        if not 0.0 <= unqualified_cell_probability <= 1.0:
            raise ValueError("intention router unqualified-cell probability is invalid")
        if not 0.0 <= mastery_threshold <= 1.0:
            raise ValueError("intention router mastery threshold is invalid")
        if min_mastery_feedbacks < 1:
            raise ValueError("intention router mastery observations must be positive")
        if not 0.0 <= reversal_threshold <= 1.0:
            raise ValueError("intention router reversal threshold is invalid")
        if reversal_patience < 1:
            raise ValueError("intention router reversal patience must be positive")
        if not -1.0 <= context_relevance_threshold <= 1.0:
            raise ValueError("intention router context relevance threshold is invalid")
        if not math.isfinite(verified_prototype_scale) or verified_prototype_scale < 0.0:
            raise ValueError("intention router verified prototype scale is invalid")
        if (
            not math.isfinite(verified_context_coverage_scale)
            or verified_context_coverage_scale < 0.0
        ):
            raise ValueError("intention router verified context coverage scale is invalid")
        if not 0.0 <= reversal_context_coverage_threshold <= 1.0:
            raise ValueError("intention router reversal context coverage threshold is invalid")
        if not math.isfinite(masked_reversal_quarantine_scale) or masked_reversal_quarantine_scale < 0.0:
            raise ValueError("intention router masked reversal quarantine scale is invalid")
        if not 0.0 < context_mask_profile_rate <= 1.0:
            raise ValueError("intention router context mask profile rate is invalid")
        if not math.isfinite(context_mask_profile_scale) or context_mask_profile_scale < 0.0:
            raise ValueError("intention router context mask profile scale is invalid")
        self.memory = memory
        self.initial_learning_rate = float(initial_learning_rate)
        self.initial_baseline_rate = float(initial_baseline_rate)
        self.initial_baseline = float(initial_baseline)
        self.temperature = float(temperature)
        self.exploration_bonus = float(exploration_bonus)
        self.initial_routing_scale = float(initial_routing_scale)
        self.unqualified_cell_probability = float(unqualified_cell_probability)
        self.mastery_threshold = float(mastery_threshold)
        self.min_mastery_feedbacks = int(min_mastery_feedbacks)
        self.reversal_threshold = float(reversal_threshold)
        self.reversal_patience = int(reversal_patience)
        self.context_relevance_threshold = float(context_relevance_threshold)
        self.verified_prototype_scale = float(verified_prototype_scale)
        self.verified_context_coverage_scale = float(verified_context_coverage_scale)
        self.reversal_context_coverage_threshold = float(
            reversal_context_coverage_threshold
        )
        self.masked_reversal_quarantine_scale = float(masked_reversal_quarantine_scale)
        self.context_mask_profile_rate = float(context_mask_profile_rate)
        self.context_mask_profile_scale = float(context_mask_profile_scale)

    @property
    def context_width(self) -> int:
        return self.memory.context_width

    @property
    def intention_width(self) -> int:
        return self.memory.intention_width

    @property
    def routing_feature_width(self) -> int:
        return (
            2 * self.context_width
            if self.memory.generator.context_masking
            else self.context_width
        )

    @property
    def hidden_width(self) -> int:
        return self.memory.hidden_width

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "memory": self.memory.configuration(),
            "routing": (
                "masked_opaque_context_and_observation_to_external_cell_mixture_v1"
                if self.memory.generator.context_masking
                else "opaque_context_to_external_cell_mixture_then_sparse_materialization_v4"
            ),
            "routing_feature_width": self.routing_feature_width,
            "credit": "outcome_only_route_score_gradient_v1",
            "temperature": self.temperature,
            "exploration_bonus": self.exploration_bonus,
            "initial_learning_rate": self.initial_learning_rate,
            "initial_baseline_rate": self.initial_baseline_rate,
            "initial_baseline": self.initial_baseline,
            "initial_routing_scale": self.initial_routing_scale,
            "unqualified_cell_probability": self.unqualified_cell_probability,
            "retention": {
                "mastery_threshold": self.mastery_threshold,
                "min_mastery_feedbacks": self.min_mastery_feedbacks,
                "reversal_threshold": self.reversal_threshold,
                "reversal_patience": self.reversal_patience,
                "context_relevance_threshold": self.context_relevance_threshold,
                "verified_prototype_scale": self.verified_prototype_scale,
                "verified_context_coverage_scale": self.verified_context_coverage_scale,
                "reversal_context_coverage_threshold": self.reversal_context_coverage_threshold,
                "masked_reversal_quarantine_scale": self.masked_reversal_quarantine_scale,
                "context_mask_profile_rate": self.context_mask_profile_rate,
                "context_mask_profile_scale": self.context_mask_profile_scale,
                "heldout_gate": "verifier_prefix_minimum_copy_on_write_v1",
            },
            "protection": "verified_cell_freezes_content_and_route_until_reversal_v1",
            "capacity": "append_only_external_cell_count_v1",
            "controller": "frozen_opaque_context_only_v1",
        }

    def initial_state(
        self,
        cell_count: int = 1,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalRoutedIntentionMemoryState:
        cells = self.memory.initial_state(cell_count, device=device, dtype=dtype)
        routing_keys = self.initial_routing_scale * torch.randn(
            cell_count,
            self.routing_feature_width,
            device=device,
            dtype=dtype,
        )
        if self.memory.generator.context_masking:
            routing_keys[:, self.context_width :].zero_()
        state = ExternalRoutedIntentionMemoryState(
            cells=cells,
            routing_keys=routing_keys,
            routing_bias=torch.zeros(cell_count, device=device, dtype=dtype),
            routing_baseline=torch.full(
                (1,), self.initial_baseline, device=device, dtype=dtype
            ),
            routing_decisions=torch.zeros(cell_count, device=device, dtype=torch.long),
            routing_feedbacks=torch.zeros(cell_count, device=device, dtype=torch.long),
            retention_observations=torch.zeros(
                cell_count, device=device, dtype=torch.long
            ),
            retention_successes=torch.zeros(cell_count, device=device, dtype=dtype),
            retention_prefix_minima=torch.ones(cell_count, device=device, dtype=dtype),
            retention_reversal_streaks=torch.zeros(
                cell_count, device=device, dtype=torch.long
            ),
            retention_reversal_counts=torch.zeros(
                cell_count, device=device, dtype=torch.long
            ),
            retention_mastered=torch.zeros(
                cell_count, device=device, dtype=torch.bool
            ),
            retention_context_prototypes=torch.zeros(
                cell_count, self.context_width, device=device, dtype=dtype
            ),
            retention_context_masses=torch.zeros(cell_count, device=device, dtype=dtype),
            retention_context_observed_masses=torch.zeros(
                cell_count, self.context_width, device=device, dtype=dtype
            ),
            retention_context_mask_profiles=torch.zeros(
                cell_count, self.context_width, device=device, dtype=dtype
            ),
        )
        self._validate_state(state)
        return state

    def _validate_state(self, state: ExternalRoutedIntentionMemoryState) -> None:
        state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            routing_feature_width=self.routing_feature_width,
        )

    def _context_view(
        self,
        context: torch.Tensor,
        state: ExternalRoutedIntentionMemoryState,
        context_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return observed context values and a boolean feature mask."""

        self._validate_context(context, state)
        if context_mask is None:
            mask = torch.ones(context.shape, dtype=torch.bool, device=context.device)
        else:
            if context_mask.shape != context.shape or context_mask.dtype != torch.bool:
                raise ValueError("intention router context mask must be boolean [batch,width]")
            if context_mask.device != context.device:
                raise ValueError("intention router context mask is on the wrong device")
            mask = context_mask
        return context * mask.to(dtype=context.dtype), mask

    def _routing_features(
        self,
        observed_context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.memory.generator.context_masking:
            return torch.cat(
                (observed_context, mask.to(dtype=observed_context.dtype)), dim=-1
            )
        return observed_context

    def _proposal_context_view(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover values and masks from dense or explicitly masked memory features."""

        if features.shape[-1] == self.context_width + 1:
            return features[:, : self.context_width], torch.ones(
                features.shape[0], self.context_width, dtype=torch.bool, device=features.device
            )
        if features.shape[-1] == 2 * self.context_width + 1:
            mask_values = features[:, self.context_width : 2 * self.context_width]
            if bool(((mask_values < 0.0) | (mask_values > 1.0)).any()):
                raise ValueError("routed intention proposal context mask is invalid")
            return features[:, : self.context_width], mask_values > 0.5
        raise ValueError("routed intention proposal has an unsupported feature width")

    @staticmethod
    def _masked_cosine(
        contexts: torch.Tensor,
        masks: torch.Tensor,
        prototypes: torch.Tensor,
        observed_masses: torch.Tensor,
    ) -> torch.Tensor:
        overlap = masks.to(dtype=contexts.dtype).unsqueeze(1) * (
            observed_masses > 0.0
        ).to(dtype=contexts.dtype).unsqueeze(0)
        query = contexts.unsqueeze(1) * overlap
        prototype = prototypes.unsqueeze(0) * overlap
        numerator = (query * prototype).sum(dim=-1)
        denominator = query.square().sum(dim=-1).sqrt() * prototype.square().sum(dim=-1).sqrt()
        cosine = numerator / denominator.clamp_min(1e-12)
        return torch.where(denominator > 0.0, cosine.clamp(-1.0, 1.0), 0.0)

    @staticmethod
    def _context_coverage(
        masks: torch.Tensor,
        observed_masses: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fraction of each query whose dimensions are verified."""

        query_width = masks.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
        covered_width = (
            masks.unsqueeze(1) & (observed_masses > 0.0).unsqueeze(0)
        ).to(dtype=torch.float32).sum(dim=-1)
        return covered_width / query_width.unsqueeze(-1)

    @staticmethod
    def _context_mask_profile_compatibility(
        masks: torch.Tensor,
        profiles: torch.Tensor,
    ) -> torch.Tensor:
        """Score how well each persistent evidence profile matches each query."""

        return 1.0 - (
            masks.to(dtype=profiles.dtype).unsqueeze(1) - profiles.unsqueeze(0)
        ).abs().mean(dim=-1)

    def _validate_context(
        self,
        context: torch.Tensor,
        state: ExternalRoutedIntentionMemoryState,
    ) -> None:
        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError("intention router context has the wrong shape")
        if context.device != state.routing_keys.device:
            raise ValueError("intention router context is on the wrong device")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("intention router context must be finite")

    def _validate_presence(
        self,
        present: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if present.shape != (batch,) or present.dtype != torch.bool:
            raise ValueError("intention router presence must be boolean [batch]")
        if present.device != device:
            raise ValueError("intention router presence is on the wrong device")

    def propose(
        self,
        state: ExternalRoutedIntentionMemoryState,
        context: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> ExternalRoutedIntentionProposal:
        """Sample an opaque route, then materialize only routed cells."""

        self._validate_state(state)
        observed_context, mask = self._context_view(context, state, context_mask)
        routing_features = self._routing_features(observed_context, mask)
        batch = context.shape[0]
        cell_count = state.cells.baseline.shape[0]
        exploration_bonus = self.exploration_bonus / torch.sqrt(
            1.0 + state.routing_decisions.to(dtype=context.dtype)
        )
        exploration_bonus = exploration_bonus.unsqueeze(0).expand(batch, -1)
        logits = (
            torch.einsum("bf,cf->bc", routing_features, state.routing_keys)
            + state.routing_bias.unsqueeze(0)
            + exploration_bonus
        ) / self.temperature
        verified = state.cells.protected & (state.retention_context_masses > 0.0)
        if bool(verified.any()):
            if self.verified_prototype_scale > 0.0:
                prototype_similarity = self._masked_cosine(
                    observed_context,
                    mask,
                    state.retention_context_prototypes,
                    state.retention_context_observed_masses,
                )
                logits = logits + (
                    self.verified_prototype_scale
                    * prototype_similarity
                    * verified.to(dtype=context.dtype).unsqueeze(0)
                ) / self.temperature
            if self.memory.generator.context_masking:
                coverage = self._context_coverage(
                    mask,
                    state.retention_context_observed_masses,
                ).to(dtype=context.dtype)
                if self.verified_context_coverage_scale > 0.0:
                    logits = logits + (
                        self.verified_context_coverage_scale
                        * torch.log(coverage.clamp_min(1e-3))
                        * verified.to(dtype=context.dtype).unsqueeze(0)
                    ) / self.temperature
                if self.masked_reversal_quarantine_scale > 0.0:
                    quarantined = (
                        verified
                        & (state.retention_reversal_counts > 0)
                    ).to(dtype=context.dtype)
                    logits = logits - (
                        self.masked_reversal_quarantine_scale
                        * quarantined.unsqueeze(0)
                    ) / self.temperature
        if self.memory.generator.context_masking and self.context_mask_profile_scale > 0.0:
            profile_active = state.retention_context_mask_profiles.sum(dim=-1) > 0.0
            if bool(profile_active.any()):
                profile_compatibility = self._context_mask_profile_compatibility(
                    mask,
                    state.retention_context_mask_profiles,
                ).to(dtype=context.dtype)
                logits = logits + (
                    self.context_mask_profile_scale
                    * torch.log(profile_compatibility.clamp_min(1e-3))
                    * profile_active.to(dtype=context.dtype).unsqueeze(0)
                ) / self.temperature
        base_probabilities = torch.softmax(logits, dim=-1)
        unqualified = (
            ~state.cells.protected
        ).to(dtype=context.dtype)
        unqualified_count = unqualified.sum()
        if (
            self.unqualified_cell_probability > 0.0
            and bool(unqualified_count > 0)
        ):
            unqualified_distribution = unqualified.unsqueeze(0) / unqualified_count
            probabilities = (
                (1.0 - self.unqualified_cell_probability) * base_probabilities
                + self.unqualified_cell_probability * unqualified_distribution
            )
        else:
            probabilities = base_probabilities
        selected_cells = torch.multinomial(
            probabilities,
            1,
            generator=generator,
        ).squeeze(-1)
        selected_cell_indices = tuple(
            sorted({int(index) for index in selected_cells.detach().cpu().tolist()})
        )
        candidates = self.memory.propose(
            state.cells,
            context,
            cell_indices=selected_cell_indices,
            generator=generator,
            context_mask=context_mask,
        )
        candidate_positions = {
            cell_index: position
            for position, cell_index in enumerate(candidates.cell_indices)
        }
        selected_positions = torch.tensor(
            [candidate_positions[int(index)] for index in selected_cells.detach().cpu()],
            device=context.device,
            dtype=torch.long,
        )
        row_indices = torch.arange(batch, device=context.device)
        selected_intentions = candidates.intentions[row_indices, selected_positions]
        one_hot = torch.nn.functional.one_hot(
            selected_cells,
            num_classes=cell_count,
        ).to(dtype=context.dtype)
        selected_base_probabilities = base_probabilities[row_indices, selected_cells]
        selected_probabilities = probabilities[row_indices, selected_cells]
        route_bias_gradients = (
            (1.0 - self.unqualified_cell_probability)
            * (
                selected_base_probabilities
                / selected_probabilities.clamp_min(1e-12)
            ).unsqueeze(-1)
            * (one_hot - base_probabilities)
            / self.temperature
        )
        route_key_gradients = routing_features.unsqueeze(1) * route_bias_gradients.unsqueeze(-1)
        route_propensities = probabilities[row_indices, selected_cells]
        proposal = ExternalRoutedIntentionProposal(
            candidates=candidates,
            selected_cells=selected_cells,
            selected_intentions=selected_intentions,
            route_probabilities=probabilities,
            route_log_propensities=route_propensities.log(),
            route_key_gradients=route_key_gradients,
            route_bias_gradients=route_bias_gradients,
            exploration_bonus=exploration_bonus,
            temperature=self.temperature,
        )
        return proposal.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            cell_count=cell_count,
            batch=batch,
            routing_feature_width=self.routing_feature_width,
        )

    def record_decision(
        self,
        state: ExternalRoutedIntentionMemoryState,
        proposal: ExternalRoutedIntentionProposal,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalRoutedIntentionMemoryState:
        self._validate_state_and_proposal(state, proposal)
        batch = proposal.selected_cells.shape[0]
        device = state.routing_keys.device
        if present is None:
            present = torch.ones(batch, dtype=torch.bool, device=device)
        self._validate_presence(present, batch=batch, device=device)
        cells = self.memory.record_decision(
            state.cells,
            proposal.candidates,
            proposal.selected_cells,
            present=present,
        )
        mutable = present & ~state.cells.protected[proposal.selected_cells]
        counts = torch.zeros_like(state.routing_decisions)
        if bool(mutable.any()):
            counts.index_add_(
                0,
                proposal.selected_cells[mutable],
                torch.ones_like(proposal.selected_cells[mutable]),
            )
        next_state = replace(
            state,
            cells=cells,
            routing_decisions=state.routing_decisions + counts,
        )
        self._validate_state(next_state)
        return next_state

    def apply_feedback(
        self,
        state: ExternalRoutedIntentionMemoryState,
        proposal: ExternalRoutedIntentionProposal,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
    ) -> ExternalRoutedIntentionMemoryState:
        """Credit the selected cell and route from one delayed scalar outcome."""

        self._validate_state_and_proposal(state, proposal)
        batch = proposal.selected_cells.shape[0]
        device = state.routing_keys.device
        if outcome.shape != (batch,) or outcome.device != device:
            raise ValueError("intention router outcomes must be [batch] on state device")
        if not bool(torch.isfinite(outcome).all()):
            raise ValueError("intention router outcomes must be finite")
        if bool(((outcome < 0.0) | (outcome > 1.0)).any()):
            raise ValueError("intention router outcomes must lie in [0, 1]")
        if present is None:
            present = torch.ones(batch, dtype=torch.bool, device=device)
        self._validate_presence(present, batch=batch, device=device)
        if terminal is not None:
            self._validate_presence(terminal, batch=batch, device=device)
        cells = self.memory.apply_feedback(
            state.cells,
            proposal.candidates,
            proposal.selected_cells,
            outcome,
            present=present,
            terminal=terminal,
        )
        proposal_contexts, proposal_masks = self._proposal_context_view(
            proposal.candidates.features
        )
        (
            protected,
            retention_observations,
            retention_successes,
            retention_prefix_minima,
            retention_reversal_streaks,
            retention_reversal_counts,
            retention_mastered,
            retention_context_prototypes,
            retention_context_masses,
            retention_context_observed_masses,
            retention_context_mask_profiles,
        ) = self._update_retention(
            state,
            proposal.selected_cells,
            outcome,
            present,
            proposal_contexts,
            proposal_masks,
        )
        cells = replace(cells, protected=protected)
        centered = outcome - state.routing_baseline[0]
        active = present
        route_mutable = active & ~protected[proposal.selected_cells]
        update_scale = self.initial_learning_rate * centered * route_mutable.to(
            dtype=state.routing_keys.dtype
        )
        route_key_delta = (
            proposal.route_key_gradients
            * update_scale.reshape(batch, 1, 1)
        ).sum(dim=0)
        route_bias_delta = (
            proposal.route_bias_gradients * update_scale.unsqueeze(-1)
        ).sum(dim=0)
        baseline_delta = torch.zeros_like(state.routing_baseline)
        if bool(active.any()):
            baseline_delta[0] = self.initial_baseline_rate * centered[active].mean()
        feedback_counts = torch.zeros_like(state.routing_feedbacks)
        if bool(active.any()):
            feedback_counts.index_add_(
                0,
                proposal.selected_cells[active],
                torch.ones_like(proposal.selected_cells[active]),
            )
        next_state = replace(
            state,
            cells=cells,
            routing_keys=state.routing_keys + route_key_delta,
            routing_bias=state.routing_bias + route_bias_delta,
            routing_baseline=(state.routing_baseline + baseline_delta).clamp(0.0, 1.0),
            routing_feedbacks=state.routing_feedbacks + feedback_counts,
            retention_observations=retention_observations,
            retention_successes=retention_successes,
            retention_prefix_minima=retention_prefix_minima,
            retention_reversal_streaks=retention_reversal_streaks,
            retention_reversal_counts=retention_reversal_counts,
            retention_mastered=retention_mastered,
            retention_context_prototypes=retention_context_prototypes,
            retention_context_masses=retention_context_masses,
            retention_context_observed_masses=retention_context_observed_masses,
            retention_context_mask_profiles=retention_context_mask_profiles,
        )
        self._validate_state(next_state)
        return next_state

    def _update_retention(
        self,
        state: ExternalRoutedIntentionMemoryState,
        selected_cells: torch.Tensor,
        outcome: torch.Tensor,
        present: torch.Tensor,
        contexts: torch.Tensor,
        context_masks: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Update stable-prefix mastery and hysteretic reversal state."""

        protected = state.cells.protected.clone()
        observations = state.retention_observations.clone()
        successes = state.retention_successes.clone()
        prefix_minima = state.retention_prefix_minima.clone()
        reversal_streaks = state.retention_reversal_streaks.clone()
        reversal_counts = state.retention_reversal_counts.clone()
        mastered = state.retention_mastered.clone()
        context_prototypes = state.retention_context_prototypes.clone()
        context_masses = state.retention_context_masses.clone()
        context_observed_masses = state.retention_context_observed_masses.clone()
        context_mask_profiles = state.retention_context_mask_profiles.clone()
        active = present & (selected_cells >= 0)
        for batch_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
            cell_index = int(selected_cells[batch_index].item())
            value = float(outcome[batch_index].item())
            context = contexts[batch_index]
            context_mask = context_masks[batch_index]
            if not bool(protected[cell_index]):
                context_mask_profiles[cell_index] = (
                    (1.0 - self.context_mask_profile_rate)
                    * context_mask_profiles[cell_index]
                    + self.context_mask_profile_rate
                    * context_mask.to(dtype=context_mask_profiles.dtype)
                )
            mass = float(context_masses[cell_index].item())
            if bool(protected[cell_index]) and mass > 0.0:
                if self.memory.generator.context_masking:
                    coverage = float(
                        self._context_coverage(
                            context_mask.unsqueeze(0),
                            context_observed_masses[cell_index].unsqueeze(0),
                        ).item()
                    )
                    if coverage < self.reversal_context_coverage_threshold:
                        continue
                relevance = self._masked_cosine(
                    context.unsqueeze(0),
                    context_mask.unsqueeze(0),
                    context_prototypes[cell_index].unsqueeze(0),
                    context_observed_masses[cell_index].unsqueeze(0),
                ).item()
                if relevance < self.context_relevance_threshold:
                    continue
            context_weight = max(value, 0.0)
            if context_weight > 0.0:
                next_mass = mass + context_weight
                observed = context_mask.to(dtype=context.dtype)
                previous_observed = context_observed_masses[cell_index]
                next_observed = previous_observed + context_weight * observed
                numerator = (
                    context_prototypes[cell_index] * previous_observed
                    + context * (context_weight * observed)
                )
                context_prototypes[cell_index] = torch.where(
                    next_observed > 0.0,
                    numerator / next_observed.clamp_min(1e-12),
                    context_prototypes[cell_index],
                )
                context_observed_masses[cell_index] = next_observed
                context_masses[cell_index] = next_mass
            observations[cell_index] += 1
            successes[cell_index] += value
            observation_count = int(observations[cell_index].item())
            if bool(protected[cell_index]):
                if observation_count >= self.min_mastery_feedbacks:
                    prefix_minima[cell_index] = torch.minimum(
                        prefix_minima[cell_index],
                        successes[cell_index] / observation_count,
                    )
                if value <= self.reversal_threshold:
                    reversal_streaks[cell_index] += 1
                else:
                    reversal_streaks[cell_index] = 0
                if int(reversal_streaks[cell_index].item()) >= self.reversal_patience:
                    reversal_counts[cell_index] += 1
                    reversal_streaks[cell_index] = 0
                    if not self.memory.generator.context_masking:
                        protected[cell_index] = False
                        observations[cell_index] = 0
                        successes[cell_index] = 0.0
                        prefix_minima[cell_index] = 1.0
                        mastered[cell_index] = False
                        context_prototypes[cell_index].zero_()
                        context_masses[cell_index] = 0.0
                        context_observed_masses[cell_index].zero_()
            elif not bool(mastered[cell_index]):
                current_mean = successes[cell_index] / observation_count
                if (
                    observation_count >= self.min_mastery_feedbacks
                    and float(current_mean.item()) >= self.mastery_threshold
                ):
                    mastered[cell_index] = True
                    protected[cell_index] = True
                    prefix_minima[cell_index] = current_mean
                    reversal_streaks[cell_index] = 0
        return (
            protected,
            observations,
            successes,
            prefix_minima,
            reversal_streaks,
            reversal_counts,
            mastered,
            context_prototypes,
            context_masses,
            context_observed_masses,
            context_mask_profiles,
        )

    def _validate_state_and_proposal(
        self,
        state: ExternalRoutedIntentionMemoryState,
        proposal: ExternalRoutedIntentionProposal,
    ) -> None:
        self._validate_state(state)
        proposal.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            cell_count=state.cells.baseline.shape[0],
            routing_feature_width=self.routing_feature_width,
        )
        if proposal.selected_cells.device != state.routing_keys.device:
            raise ValueError("routed intention proposal is on the wrong device")

    def append_cell(
        self,
        state: ExternalRoutedIntentionMemoryState,
        *,
        source_cell: int | None = None,
        copy_route: bool = True,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[ExternalRoutedIntentionMemoryState, int]:
        """Append content and a fresh/transfer route address copy-on-write."""

        self._validate_state(state)
        if not isinstance(copy_route, bool):
            raise TypeError("intention router copy_route must be boolean")
        if source_cell is not None and not 0 <= source_cell < state.routing_keys.shape[0]:
            raise ValueError("intention router source cell is out of range")
        if context_mask is not None:
            if context_mask.shape != (self.context_width,) or context_mask.dtype != torch.bool:
                raise ValueError("intention router append context mask must be boolean [width]")
            if context_mask.device != state.routing_keys.device:
                raise ValueError("intention router append context mask is on the wrong device")
        next_cells, new_index = self.memory.append_cell(
            state.cells,
            source_cell=source_cell,
        )
        device = state.routing_keys.device
        dtype = state.routing_keys.dtype
        source_observed_dimensions: torch.Tensor | None = None
        if source_cell is not None and self.memory.generator.context_masking:
            source_profile = state.retention_context_mask_profiles[source_cell]
            observed_masses = state.retention_context_observed_masses[source_cell]
            source_observed_dimensions = source_profile > 0.0
            if not bool(source_observed_dimensions.any()):
                source_observed_dimensions = observed_masses > 0.0
            input_weights = next_cells.input_weights.clone()
            unavailable = ~source_observed_dimensions
            input_weights[new_index, :, : self.context_width][:, unavailable] = 0.0
            input_weights[new_index, :, self.context_width : 2 * self.context_width][
                :, unavailable
            ] = 0.0
            next_cells = replace(next_cells, input_weights=input_weights)
        if source_cell is None:
            new_key = self.initial_routing_scale * torch.randn(
                1, self.routing_feature_width, device=device, dtype=dtype
            )
            if self.memory.generator.context_masking:
                new_key[:, self.context_width :].zero_()
            new_bias = torch.zeros(1, device=device, dtype=dtype)
        elif copy_route:
            new_key = state.routing_keys[source_cell : source_cell + 1].clone()
            if self.memory.generator.context_masking:
                new_key[:, self.context_width :].zero_()
                if source_observed_dimensions is not None:
                    new_key[:, : self.context_width][:, ~source_observed_dimensions] = 0.0
            new_bias = state.routing_bias[source_cell : source_cell + 1].clone()
        else:
            new_key = self.initial_routing_scale * torch.randn(
                1, self.routing_feature_width, device=device, dtype=dtype
            )
            if self.memory.generator.context_masking:
                new_key[:, self.context_width :].zero_()
            new_bias = torch.zeros(1, device=device, dtype=dtype)
        next_state = replace(
            state,
            cells=next_cells,
            routing_keys=torch.cat((state.routing_keys, new_key), dim=0),
            routing_bias=torch.cat((state.routing_bias, new_bias), dim=0),
            routing_decisions=torch.cat(
                (state.routing_decisions, torch.zeros(1, device=device, dtype=torch.long)),
                dim=0,
            ),
            routing_feedbacks=torch.cat(
                (state.routing_feedbacks, torch.zeros(1, device=device, dtype=torch.long)),
                dim=0,
            ),
            retention_observations=torch.cat(
                (
                    state.retention_observations,
                    torch.zeros(1, device=device, dtype=torch.long),
                ),
                dim=0,
            ),
            retention_successes=torch.cat(
                (state.retention_successes, torch.zeros(1, device=device, dtype=dtype)),
                dim=0,
            ),
            retention_prefix_minima=torch.cat(
                (state.retention_prefix_minima, torch.ones(1, device=device, dtype=dtype)),
                dim=0,
            ),
            retention_reversal_streaks=torch.cat(
                (
                    state.retention_reversal_streaks,
                    torch.zeros(1, device=device, dtype=torch.long),
                ),
                dim=0,
            ),
            retention_reversal_counts=torch.cat(
                (
                    state.retention_reversal_counts,
                    torch.zeros(1, device=device, dtype=torch.long),
                ),
                dim=0,
            ),
            retention_mastered=torch.cat(
                (
                    state.retention_mastered,
                    torch.zeros(1, device=device, dtype=torch.bool),
                ),
                dim=0,
            ),
            retention_context_prototypes=torch.cat(
                (
                    state.retention_context_prototypes,
                    torch.zeros(1, self.context_width, device=device, dtype=dtype),
                ),
                dim=0,
            ),
            retention_context_masses=torch.cat(
                (
                    state.retention_context_masses,
                    torch.zeros(1, device=device, dtype=dtype),
                ),
                dim=0,
            ),
            retention_context_observed_masses=torch.cat(
                (
                    state.retention_context_observed_masses,
                    torch.zeros(1, self.context_width, device=device, dtype=dtype),
                ),
                dim=0,
            ),
            retention_context_mask_profiles=torch.cat(
                (
                    state.retention_context_mask_profiles,
                    torch.zeros(1, self.context_width, device=device, dtype=dtype),
                ),
                dim=0,
            ),
        )
        self._validate_state(next_state)
        return next_state, new_index

    def protect(
        self,
        state: ExternalRoutedIntentionMemoryState,
        cell_indices: torch.Tensor | list[int] | tuple[int, ...],
    ) -> ExternalRoutedIntentionMemoryState:
        next_state = replace(
            state,
            cells=self.memory.protect(state.cells, cell_indices),
        )
        self._validate_state(next_state)
        return next_state

    def verify_and_protect(
        self,
        state: ExternalRoutedIntentionMemoryState,
        cell_index: int,
        context: torch.Tensor,
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        *,
        floor: float | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[
        ExternalRoutedIntentionMemoryState,
        ExternalRoutedRetentionVerification,
    ]:
        """Qualify a cell from held-out outcomes without learning from them.

        The caller owns the verifier and must provide fresh prefix outcomes.
        This transaction never updates generator content, route parameters,
        decisions, feedbacks, or eligibility traces. It only commits the
        protection bit when every held-out prefix outcome clears ``floor``.
        """

        self._validate_state(state)
        if not isinstance(cell_index, int) or isinstance(cell_index, bool):
            raise TypeError("routed retention cell index must be an integer")
        cell_count = state.cells.baseline.shape[0]
        if not 0 <= cell_index < cell_count:
            raise IndexError("routed retention cell index is out of range")
        if context.ndim not in (1, 2) or context.shape[-1] != self.context_width:
            raise ValueError("routed retention context has the wrong shape")
        if context.device != state.routing_keys.device:
            raise ValueError("routed retention context is on the wrong device")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("routed retention context must be finite")
        if isinstance(outcomes, torch.Tensor):
            if outcomes.ndim != 1 or outcomes.device != context.device:
                raise ValueError("routed retention outcomes must be a device-local vector")
            values = tuple(float(value) for value in outcomes.detach().cpu().tolist())
        else:
            values = tuple(float(value) for value in outcomes)
        if not values:
            raise ValueError("routed retention needs at least one held-out outcome")
        if context.ndim == 1:
            verifier_contexts = context.unsqueeze(0).expand(len(values), -1)
        else:
            if context.shape[0] != len(values):
                raise ValueError(
                    "routed retention contexts and outcomes must have equal length"
                )
            verifier_contexts = context
        if context_mask is None:
            verifier_masks = torch.ones(
                verifier_contexts.shape,
                dtype=torch.bool,
                device=context.device,
            )
        else:
            if context_mask.shape != context.shape or context_mask.dtype != torch.bool:
                raise ValueError("routed retention context mask must be boolean [batch,width]")
            if context_mask.device != context.device:
                raise ValueError("routed retention context mask is on the wrong device")
            verifier_masks = (
                context_mask.unsqueeze(0).expand(len(values), -1)
                if context.ndim == 1
                else context_mask
            )
        selected_floor = self.mastery_threshold if floor is None else float(floor)
        if not 0.0 <= selected_floor <= 1.0:
            raise ValueError("routed retention floor must lie in [0, 1]")
        context_relevance_minimum: float | None = None
        mass = float(state.retention_context_masses[cell_index].item())
        if mass > 0.0:
            context_relevance_minimum = float(
                self._masked_cosine(
                    verifier_contexts,
                    verifier_masks,
                    state.retention_context_prototypes[cell_index].unsqueeze(0),
                    state.retention_context_observed_masses[cell_index].unsqueeze(0),
                ).min().item()
            )
        prefix_minimum = min(values)
        mean_outcome = sum(values) / len(values)
        accepted = (
            len(values) >= self.min_mastery_feedbacks
            and prefix_minimum >= selected_floor
            and (
                context_relevance_minimum is None
                or context_relevance_minimum >= self.context_relevance_threshold
            )
        )
        if accepted:
            cells = self.memory.protect(state.cells, [cell_index])
            mastered = state.retention_mastered.clone()
            mastered[cell_index] = True
            prefix_minima = state.retention_prefix_minima.clone()
            prefix_minima[cell_index] = min(
                prefix_minimum,
                float(prefix_minima[cell_index]),
            )
            context_prototypes = state.retention_context_prototypes.clone()
            context_masses = state.retention_context_masses.clone()
            context_observed_masses = state.retention_context_observed_masses.clone()
            context_mask_profiles = state.retention_context_mask_profiles.clone()
            if mass <= 0.0:
                observed_weights = verifier_masks.to(dtype=verifier_contexts.dtype)
                observed_totals = observed_weights.sum(dim=0)
                context_prototypes[cell_index] = torch.where(
                    observed_totals > 0.0,
                    (verifier_contexts * observed_weights).sum(dim=0)
                    / observed_totals.clamp_min(1e-12),
                    torch.zeros_like(context_prototypes[cell_index]),
                )
                context_masses[cell_index] = float(len(values))
                context_observed_masses[cell_index] = observed_totals
                context_mask_profiles[cell_index] = verifier_masks.to(
                    dtype=context_mask_profiles.dtype
                ).mean(dim=0)
            next_state = replace(
                state,
                cells=cells,
                retention_mastered=mastered,
                retention_prefix_minima=prefix_minima,
                retention_context_prototypes=context_prototypes,
                retention_context_masses=context_masses,
                retention_context_observed_masses=context_observed_masses,
                retention_context_mask_profiles=context_mask_profiles,
            )
            reason = "heldout_prefix_floor_passed"
        else:
            next_state = state
            reason = "heldout_prefix_floor_failed"
            if len(values) < self.min_mastery_feedbacks:
                reason = "heldout_prefix_too_short"
            elif context_relevance_minimum is not None and (
                context_relevance_minimum < self.context_relevance_threshold
            ):
                reason = "heldout_context_not_relevant"
        receipt = ExternalRoutedRetentionVerification(
            cell_index=cell_index,
            outcomes=values,
            prefix_minimum=prefix_minimum,
            mean_outcome=mean_outcome,
            floor=selected_floor,
            accepted=accepted,
            context_relevance_minimum=context_relevance_minimum,
            reason=reason,
        ).validate()
        self._validate_state(next_state)
        return next_state, receipt

    def begin_episode(
        self,
        state: ExternalRoutedIntentionMemoryState,
    ) -> ExternalRoutedIntentionMemoryState:
        next_state = replace(state, cells=self.memory.begin_episode(state.cells))
        self._validate_state(next_state)
        return next_state

    def mean(
        self,
        state: ExternalRoutedIntentionMemoryState,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_state(state)
        self._context_view(context, state, context_mask)
        return self.memory.mean(state.cells, context, context_mask=context_mask)

    def state_payload(
        self,
        state: ExternalRoutedIntentionMemoryState,
    ) -> dict[str, object]:
        self._validate_state(state)
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "cells": self.memory.state_payload(state.cells),
            "routing_keys": state.routing_keys.detach().cpu().clone(),
            "routing_bias": state.routing_bias.detach().cpu().clone(),
            "routing_baseline": state.routing_baseline.detach().cpu().clone(),
            "routing_decisions": state.routing_decisions.detach().cpu().clone(),
            "routing_feedbacks": state.routing_feedbacks.detach().cpu().clone(),
            "retention_observations": state.retention_observations.detach().cpu().clone(),
            "retention_successes": state.retention_successes.detach().cpu().clone(),
            "retention_prefix_minima": state.retention_prefix_minima.detach().cpu().clone(),
            "retention_reversal_streaks": state.retention_reversal_streaks.detach().cpu().clone(),
            "retention_reversal_counts": state.retention_reversal_counts.detach().cpu().clone(),
            "retention_mastered": state.retention_mastered.detach().cpu().clone(),
            "retention_context_prototypes": state.retention_context_prototypes.detach().cpu().clone(),
            "retention_context_masses": state.retention_context_masses.detach().cpu().clone(),
            "retention_context_observed_masses": state.retention_context_observed_masses.detach().cpu().clone(),
            "retention_context_mask_profiles": state.retention_context_mask_profiles.detach().cpu().clone(),
        }

    def state_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> ExternalRoutedIntentionMemoryState:
        schema = payload.get("schema")
        if schema not in (
            self.schema,
            EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V4,
            EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V3,
            EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V2,
            EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V1,
        ):
            raise ValueError("unsupported routed intention state schema")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("routed intention state configuration is invalid")
        cells_payload = payload.get("cells")
        if not isinstance(cells_payload, Mapping):
            raise TypeError("routed intention cells payload is invalid")
        cells = self.memory.state_from_payload(cells_payload)
        names = (
            "routing_keys",
            "routing_bias",
            "routing_baseline",
            "routing_decisions",
            "routing_feedbacks",
        )
        values: dict[str, torch.Tensor] = {}
        for name in names:
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"routed intention payload field {name!r} must be a tensor")
            values[name] = value
        cell_count = cells.baseline.shape[0]
        if schema == EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V1:
            device = cells.input_weights.device
            dtype = cells.input_weights.dtype
            values.update(
                {
                    "retention_observations": torch.zeros(
                        cell_count, device=device, dtype=torch.long
                    ),
                    "retention_successes": torch.zeros(
                        cell_count, device=device, dtype=dtype
                    ),
                    "retention_prefix_minima": torch.ones(
                        cell_count, device=device, dtype=dtype
                    ),
                    "retention_reversal_streaks": torch.zeros(
                        cell_count, device=device, dtype=torch.long
                    ),
                    "retention_reversal_counts": torch.zeros(
                        cell_count, device=device, dtype=torch.long
                    ),
                    "retention_mastered": torch.zeros(
                        cell_count, device=device, dtype=torch.bool
                    ),
                    "retention_context_prototypes": torch.zeros(
                        cell_count, self.context_width, device=device, dtype=dtype
                    ),
                    "retention_context_masses": torch.zeros(
                        cell_count, device=device, dtype=dtype
                    ),
                    "retention_context_observed_masses": torch.zeros(
                        cell_count, self.context_width, device=device, dtype=dtype
                    ),
                    "retention_context_mask_profiles": torch.zeros(
                        cell_count, self.context_width, device=device, dtype=dtype
                    ),
                }
            )
        else:
            retention_names = (
                "retention_observations",
                "retention_successes",
                "retention_prefix_minima",
                "retention_reversal_streaks",
                "retention_reversal_counts",
                "retention_mastered",
                "retention_context_prototypes",
                "retention_context_masses",
                "retention_context_observed_masses",
                "retention_context_mask_profiles",
            )
            for name in retention_names:
                value = payload.get(name)
                if name == "retention_context_observed_masses" and value is None:
                    device = cells.input_weights.device
                    dtype = cells.input_weights.dtype
                    if schema == EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V1:
                        value = torch.zeros(
                            cell_count, self.context_width, device=device, dtype=dtype
                        )
                    elif schema in (
                        EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V2,
                        EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V3,
                    ):
                        value = values["retention_context_masses"].unsqueeze(-1).expand(
                            -1, self.context_width
                        ).clone()
                    else:
                        raise TypeError(
                            "routed intention payload field "
                            "'retention_context_observed_masses' must be a tensor"
                        )
                if name == "retention_context_mask_profiles" and value is None:
                    device = cells.input_weights.device
                    dtype = cells.input_weights.dtype
                    if schema == EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V4:
                        value = (
                            values["retention_context_observed_masses"] > 0.0
                        ).to(dtype=dtype)
                    elif schema in (
                        EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V2,
                        EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA_V3,
                    ):
                        value = (
                            values["retention_context_masses"] > 0.0
                        ).to(dtype=dtype).unsqueeze(-1).expand(-1, self.context_width).clone()
                    else:
                        raise TypeError(
                            "routed intention payload field "
                            "'retention_context_mask_profiles' must be a tensor"
                        )
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"routed intention payload field {name!r} must be a tensor"
                    )
                values[name] = value
        state = ExternalRoutedIntentionMemoryState(cells=cells, **values)
        self._validate_state(state)
        return state
