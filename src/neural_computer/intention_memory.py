"""Memory-sized outcome learning for opaque intention candidates.

The original outcome generator used one external state row per controller
batch row.  That is useful for a one-cell proof, but it cannot represent a
growing memory when the controller batch is one.  This module separates those
dimensions: one controller context queries every external generator cell, and
the caller later credits the exact selected cell from the planner's candidate
provenance.

The proposal stores its score gradients, so delayed feedback remains attached
to the proposal that caused it even if another cell learns before the outcome
arrives.  No raw modality data, task labels, unattempted-action labels, or
controller parameters enter this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from .intention import (
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionGeneratorState,
)

EXTERNAL_OUTCOME_INTENTION_MEMORY_SCHEMA = (
    "neural-computer.external-outcome-intention-memory.v1"
)
EXTERNAL_INTENTION_MEMORY_PROPOSAL_SCHEMA = (
    "neural-computer.external-intention-memory-proposal.v1"
)


def _finite(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite tensor")


@dataclass(frozen=True)
class ExternalIntentionMemoryProposal:
    """One provisional candidate per external cell and controller row."""

    intentions: torch.Tensor
    means: torch.Tensor
    features: torch.Tensor
    hidden: torch.Tensor
    noise: torch.Tensor
    log_propensities: torch.Tensor
    input_weight_gradients: torch.Tensor
    input_bias_gradients: torch.Tensor
    output_weight_gradients: torch.Tensor
    output_bias_gradients: torch.Tensor
    cell_indices: tuple[int, ...]
    noise_scale: float
    schema: str = EXTERNAL_INTENTION_MEMORY_PROPOSAL_SCHEMA

    def validate(
        self,
        *,
        context_width: int,
        intention_width: int,
        hidden_width: int,
        batch: int | None = None,
        cell_count: int | None = None,
    ) -> ExternalIntentionMemoryProposal:
        if self.schema != EXTERNAL_INTENTION_MEMORY_PROPOSAL_SCHEMA:
            raise ValueError("unsupported intention-memory proposal schema")
        if not math.isfinite(float(self.noise_scale)) or self.noise_scale <= 0.0:
            raise ValueError("intention-memory proposal noise scale is invalid")
        if self.intentions.ndim != 3:
            raise ValueError("intention-memory candidates must be [batch,cells,width]")
        proposal_batch, proposal_cells, proposal_width = self.intentions.shape
        if proposal_width != intention_width:
            raise ValueError("intention-memory candidate width is wrong")
        if batch is not None and proposal_batch != batch:
            raise ValueError("intention-memory proposal batch differs")
        if cell_count is not None and proposal_cells != cell_count:
            raise ValueError("intention-memory proposal cell count differs")
        if len(self.cell_indices) != proposal_cells or tuple(self.cell_indices) != tuple(
            range(proposal_cells)
        ):
            raise ValueError("intention-memory cell indices are not canonical")
        expected = {
            "means": (proposal_batch, proposal_cells, intention_width),
            "features": (proposal_batch, context_width + 1),
            "hidden": (proposal_batch, proposal_cells, hidden_width),
            "noise": (proposal_batch, proposal_cells, intention_width),
            "log_propensities": (proposal_batch, proposal_cells),
            "input_weight_gradients": (
                proposal_batch,
                proposal_cells,
                hidden_width,
                context_width + 1,
            ),
            "input_bias_gradients": (proposal_batch, proposal_cells, hidden_width),
            "output_weight_gradients": (
                proposal_batch,
                proposal_cells,
                intention_width,
                hidden_width,
            ),
            "output_bias_gradients": (
                proposal_batch,
                proposal_cells,
                intention_width,
            ),
        }
        for name, value in (
            ("means", self.means),
            ("features", self.features),
            ("hidden", self.hidden),
            ("noise", self.noise),
            ("log_propensities", self.log_propensities),
            ("input_weight_gradients", self.input_weight_gradients),
            ("input_bias_gradients", self.input_bias_gradients),
            ("output_weight_gradients", self.output_weight_gradients),
            ("output_bias_gradients", self.output_bias_gradients),
        ):
            if value.shape != expected[name]:
                raise ValueError(f"intention-memory {name} has the wrong shape")
            _finite(value, f"intention-memory {name}")
        return self


class ExternalOutcomeIntentionMemory:
    """Query and update an external generator with independent cell capacity."""

    schema = EXTERNAL_OUTCOME_INTENTION_MEMORY_SCHEMA

    def __init__(self, generator: ExternalOutcomeIntentionGenerator) -> None:
        if not isinstance(generator, ExternalOutcomeIntentionGenerator):
            raise TypeError("intention memory requires an external outcome generator")
        self.generator = generator

    @property
    def context_width(self) -> int:
        return self.generator.context_width

    @property
    def intention_width(self) -> int:
        return self.generator.intention_width

    @property
    def hidden_width(self) -> int:
        return self.generator.hidden_width

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "generator": self.generator.configuration(),
            "capacity": "independent_external_cells_runtime_variable_v1",
            "proposal": "one_opaque_candidate_per_cell_and_context_v1",
            "credit": "delayed_proposal_specific_gaussian_score_gradients_v1",
            "controller": "frozen_opaque_context_only_v1",
            "persistence": "generator_tensor_payload_v1",
        }

    def initial_state(
        self,
        cell_count: int = 1,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalOutcomeIntentionGeneratorState:
        return self.generator.initial_state(cell_count, device=device, dtype=dtype)

    def _validate_context(
        self,
        context: torch.Tensor,
        state: ExternalOutcomeIntentionGeneratorState,
    ) -> None:
        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError("intention-memory context has the wrong shape")
        if context.device != state.input_weights.device:
            raise ValueError("intention-memory context is on the wrong device")
        _finite(context, "intention-memory context")

    def _validate_selection(
        self,
        selected_cells: torch.Tensor,
        *,
        batch: int,
        cell_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if selected_cells.shape != (batch,) or selected_cells.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("intention-memory selected cells must be integer [batch]")
        if selected_cells.device != device:
            raise ValueError("intention-memory selected cells are on the wrong device")
        if bool((selected_cells < -1).any()) or bool(
            (selected_cells >= cell_count).any()
        ):
            raise ValueError("intention-memory selected cell is out of range")
        return selected_cells.to(dtype=torch.long)

    def _validate_presence(
        self,
        present: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if present.shape != (batch,) or present.dtype != torch.bool:
            raise ValueError("intention-memory presence must be boolean [batch]")
        if present.device != device:
            raise ValueError("intention-memory presence is on the wrong device")

    def propose(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        context: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> ExternalIntentionMemoryProposal:
        """Sample every external cell for every controller context."""

        state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
        )
        self._validate_context(context, state)
        batch = context.shape[0]
        cells = state.baseline.shape[0]
        features = torch.cat(
            (
                context,
                torch.ones(batch, 1, device=context.device, dtype=context.dtype),
            ),
            dim=-1,
        )
        hidden = torch.tanh(
            torch.einsum("bf,chf->bch", features, state.input_weights)
            + state.input_bias.unsqueeze(0)
        )
        means = torch.einsum("bch,coh->bco", hidden, state.output_weights)
        means = means + state.output_bias.unsqueeze(0)
        noise = torch.randn(
            means.shape,
            device=means.device,
            dtype=means.dtype,
            generator=generator,
        )
        scale = torch.as_tensor(
            self.generator.noise_scale,
            device=means.device,
            dtype=means.dtype,
        )
        intentions = means + scale * noise
        log_propensities = -0.5 * torch.sum(
            noise.square() + math.log(2.0 * math.pi * self.generator.noise_scale**2),
            dim=-1,
        )
        score = noise / scale
        output_weight_gradients = torch.einsum("bco,bch->bcoh", score, hidden)
        output_bias_gradients = score
        hidden_score = torch.einsum(
            "bco,coh->bch", score, state.output_weights
        ) * (1.0 - hidden.square())
        input_weight_gradients = torch.einsum(
            "bch,bf->bchf", hidden_score, features
        )
        input_bias_gradients = hidden_score
        proposal = ExternalIntentionMemoryProposal(
            intentions=intentions,
            means=means,
            features=features,
            hidden=hidden,
            noise=noise,
            log_propensities=log_propensities,
            input_weight_gradients=input_weight_gradients,
            input_bias_gradients=input_bias_gradients,
            output_weight_gradients=output_weight_gradients,
            output_bias_gradients=output_bias_gradients,
            cell_indices=tuple(range(cells)),
            noise_scale=self.generator.noise_scale,
        )
        return proposal.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            batch=batch,
            cell_count=cells,
        )

    def record_decision(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionMemoryProposal,
        selected_cells: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Record which opaque cell was actually attempted."""

        self._validate_state_and_proposal(state, proposal)
        batch = proposal.intentions.shape[0]
        device = state.baseline.device
        selected = self._validate_selection(
            selected_cells,
            batch=batch,
            cell_count=state.baseline.shape[0],
            device=device,
        )
        if present is None:
            present = torch.ones(batch, dtype=torch.bool, device=device)
        self._validate_presence(present, batch=batch, device=device)
        active = present & (selected >= 0)
        counts = torch.zeros_like(state.decisions)
        if bool(active.any()):
            counts.index_add_(0, selected[active], torch.ones_like(selected[active]))
        next_state = replace(state, decisions=state.decisions + counts)
        next_state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
        )
        return next_state

    def apply_feedback(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionMemoryProposal,
        selected_cells: torch.Tensor,
        outcome: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        terminal: torch.Tensor | None = None,
    ) -> ExternalOutcomeIntentionGeneratorState:
        """Apply delayed scalar outcome credit to only the selected cells."""

        self._validate_state_and_proposal(state, proposal)
        batch = proposal.intentions.shape[0]
        device = state.baseline.device
        if outcome.shape != (batch,) or outcome.device != device:
            raise ValueError("intention-memory outcomes must be [batch] on the state device")
        _finite(outcome, "intention-memory outcome")
        if bool(((outcome < 0.0) | (outcome > 1.0)).any()):
            raise ValueError("intention-memory outcomes must lie in [0, 1]")
        selected = self._validate_selection(
            selected_cells,
            batch=batch,
            cell_count=state.baseline.shape[0],
            device=device,
        )
        if present is None:
            present = torch.ones(batch, dtype=torch.bool, device=device)
        self._validate_presence(present, batch=batch, device=device)
        if terminal is not None:
            self._validate_presence(terminal, batch=batch, device=device)
        active = present & (selected >= 0)
        safe_selected = selected.clamp_min(0)
        mutable = active & ~state.protected[safe_selected]
        centered = outcome - state.baseline[safe_selected]
        update_scale = (
            self.generator.initial_learning_rate
            * centered
            * mutable.to(dtype=state.baseline.dtype)
        )

        def aggregate(gradients: torch.Tensor) -> torch.Tensor:
            selected_gradients = gradients[
                torch.arange(batch, device=device), safe_selected
            ]
            contribution = selected_gradients * update_scale.reshape(
                batch, *([1] * (selected_gradients.ndim - 1))
            )
            result = torch.zeros(
                (state.baseline.shape[0], *selected_gradients.shape[1:]),
                device=device,
                dtype=selected_gradients.dtype,
            )
            if bool(active.any()):
                result.index_add_(0, safe_selected[active], contribution[active])
            return result

        next_input_weights = state.input_weights + aggregate(
            proposal.input_weight_gradients
        )
        next_input_bias = state.input_bias + aggregate(proposal.input_bias_gradients)
        next_output_weights = state.output_weights + aggregate(
            proposal.output_weight_gradients
        )
        next_output_bias = state.output_bias + aggregate(proposal.output_bias_gradients)
        baseline_delta = torch.zeros_like(state.baseline)
        if bool(mutable.any()):
            baseline_delta.index_add_(
                0,
                safe_selected[mutable],
                self.generator.initial_baseline_rate * centered[mutable],
            )
        feedback_counts = torch.zeros_like(state.feedbacks)
        if bool(active.any()):
            feedback_counts.index_add_(
                0, safe_selected[active], torch.ones_like(safe_selected[active])
            )
        next_state = replace(
            state,
            input_weights=next_input_weights,
            input_bias=next_input_bias,
            output_weights=next_output_weights,
            output_bias=next_output_bias,
            baseline=(state.baseline + baseline_delta).clamp(0.0, 1.0),
            feedbacks=state.feedbacks + feedback_counts,
        )
        next_state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
        )
        return next_state

    def _validate_state_and_proposal(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        proposal: ExternalIntentionMemoryProposal,
    ) -> None:
        state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
        )
        proposal.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
            cell_count=state.baseline.shape[0],
        )
        if proposal.intentions.device != state.baseline.device:
            raise ValueError("intention-memory proposal is on the wrong device")

    def mean(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Read deterministic means for every external cell."""

        self._validate_state_and_context(state, context)
        features = torch.cat(
            (
                context,
                torch.ones(
                    context.shape[0], 1, device=context.device, dtype=context.dtype
                ),
            ),
            dim=-1,
        )
        hidden = torch.tanh(
            torch.einsum("bf,chf->bch", features, state.input_weights)
            + state.input_bias.unsqueeze(0)
        )
        return torch.einsum("bch,coh->bco", hidden, state.output_weights) + state.output_bias.unsqueeze(0)

    def _validate_state_and_context(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        context: torch.Tensor,
    ) -> None:
        state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
        )
        self._validate_context(context, state)

    def begin_episode(
        self, state: ExternalOutcomeIntentionGeneratorState
    ) -> ExternalOutcomeIntentionGeneratorState:
        return self.generator.begin_episode(state)

    def append_cell(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        *,
        source_cell: int | None = None,
    ) -> tuple[ExternalOutcomeIntentionGeneratorState, int]:
        return self.generator.append_cell(state, source_cell=source_cell)

    def protect(
        self,
        state: ExternalOutcomeIntentionGeneratorState,
        cell_indices: torch.Tensor | list[int] | tuple[int, ...],
    ) -> ExternalOutcomeIntentionGeneratorState:
        return self.generator.protect(state, cell_indices)

    def state_payload(
        self, state: ExternalOutcomeIntentionGeneratorState
    ) -> dict[str, object]:
        return self.generator.state_payload(state)

    def state_from_payload(
        self, payload: Mapping[str, object]
    ) -> ExternalOutcomeIntentionGeneratorState:
        return self.generator.state_from_payload(payload)
