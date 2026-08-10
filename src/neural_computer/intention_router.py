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

EXTERNAL_ROUTED_INTENTION_MEMORY_SCHEMA = (
    "neural-computer.external-routed-intention-memory.v1"
)
EXTERNAL_ROUTED_INTENTION_PROPOSAL_SCHEMA = (
    "neural-computer.external-routed-intention-proposal.v1"
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

    def validate(
        self,
        *,
        context_width: int,
        intention_width: int,
        hidden_width: int,
    ) -> None:
        self.cells.validate(
            context_width=context_width,
            intention_width=intention_width,
            hidden_width=hidden_width,
        )
        cell_count = self.cells.baseline.shape[0]
        expected = {
            "routing_keys": (cell_count, context_width),
            "routing_bias": (cell_count,),
            "routing_baseline": (1,),
            "routing_decisions": (cell_count,),
            "routing_feedbacks": (cell_count,),
        }
        tensors = {
            "routing_keys": self.routing_keys,
            "routing_bias": self.routing_bias,
            "routing_baseline": self.routing_baseline,
            "routing_decisions": self.routing_decisions,
            "routing_feedbacks": self.routing_feedbacks,
        }
        for name, value in tensors.items():
            if value.shape != expected[name]:
                raise ValueError(f"routed intention {name} has the wrong shape")
            if value.device != self.cells.input_weights.device:
                raise ValueError("routed intention state tensors must share a device")
        for name in ("routing_keys", "routing_bias", "routing_baseline"):
            if not bool(torch.isfinite(tensors[name]).all()):
                raise ValueError(f"routed intention {name} must be finite")
        for name in ("routing_decisions", "routing_feedbacks"):
            if tensors[name].dtype not in (torch.int32, torch.int64):
                raise TypeError(f"routed intention {name} must be integer")
            if bool((tensors[name] < 0).any()):
                raise ValueError(f"routed intention {name} cannot be negative")
        if bool(
            (self.routing_baseline < 0.0).any()
            or (self.routing_baseline > 1.0).any()
        ):
            raise ValueError("routed intention baseline must lie in [0, 1]")


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
    ) -> ExternalRoutedIntentionProposal:
        if self.schema != EXTERNAL_ROUTED_INTENTION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported routed intention proposal schema")
        self.candidates.validate(
            context_width=context_width,
            intention_width=intention_width,
            hidden_width=hidden_width,
            batch=batch,
            cell_count=cell_count,
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
        if self.route_key_gradients.shape != (
            candidate_batch,
            cell_count,
            context_width,
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
        self.memory = memory
        self.initial_learning_rate = float(initial_learning_rate)
        self.initial_baseline_rate = float(initial_baseline_rate)
        self.initial_baseline = float(initial_baseline)
        self.temperature = float(temperature)
        self.exploration_bonus = float(exploration_bonus)
        self.initial_routing_scale = float(initial_routing_scale)

    @property
    def context_width(self) -> int:
        return self.memory.context_width

    @property
    def intention_width(self) -> int:
        return self.memory.intention_width

    @property
    def hidden_width(self) -> int:
        return self.memory.hidden_width

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "memory": self.memory.configuration(),
            "routing": "opaque_context_to_external_cell_softmax_then_sparse_materialization_v1",
            "credit": "outcome_only_route_score_gradient_v1",
            "temperature": self.temperature,
            "exploration_bonus": self.exploration_bonus,
            "initial_learning_rate": self.initial_learning_rate,
            "initial_baseline_rate": self.initial_baseline_rate,
            "initial_baseline": self.initial_baseline,
            "initial_routing_scale": self.initial_routing_scale,
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
            self.context_width,
            device=device,
            dtype=dtype,
        )
        state = ExternalRoutedIntentionMemoryState(
            cells=cells,
            routing_keys=routing_keys,
            routing_bias=torch.zeros(cell_count, device=device, dtype=dtype),
            routing_baseline=torch.full(
                (1,), self.initial_baseline, device=device, dtype=dtype
            ),
            routing_decisions=torch.zeros(cell_count, device=device, dtype=torch.long),
            routing_feedbacks=torch.zeros(cell_count, device=device, dtype=torch.long),
        )
        self._validate_state(state)
        return state

    def _validate_state(self, state: ExternalRoutedIntentionMemoryState) -> None:
        state.validate(
            context_width=self.context_width,
            intention_width=self.intention_width,
            hidden_width=self.hidden_width,
        )

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
    ) -> ExternalRoutedIntentionProposal:
        """Sample an opaque route, then materialize only routed cells."""

        self._validate_state(state)
        self._validate_context(context, state)
        batch = context.shape[0]
        cell_count = state.cells.baseline.shape[0]
        exploration_bonus = self.exploration_bonus / torch.sqrt(
            1.0 + state.routing_decisions.to(dtype=context.dtype)
        )
        exploration_bonus = exploration_bonus.unsqueeze(0).expand(batch, -1)
        logits = (
            torch.einsum("bf,cf->bc", context, state.routing_keys)
            + state.routing_bias.unsqueeze(0)
            + exploration_bonus
        ) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
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
        route_bias_gradients = (one_hot - probabilities) / self.temperature
        route_key_gradients = context.unsqueeze(1) * route_bias_gradients.unsqueeze(-1)
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
        counts = torch.zeros_like(state.routing_decisions)
        if bool(present.any()):
            counts.index_add_(
                0,
                proposal.selected_cells[present],
                torch.ones_like(proposal.selected_cells[present]),
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
        centered = outcome - state.routing_baseline[0]
        active = present
        update_scale = self.initial_learning_rate * centered * active.to(
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
        )
        self._validate_state(next_state)
        return next_state

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
        )
        if proposal.selected_cells.device != state.routing_keys.device:
            raise ValueError("routed intention proposal is on the wrong device")

    def append_cell(
        self,
        state: ExternalRoutedIntentionMemoryState,
        *,
        source_cell: int | None = None,
    ) -> tuple[ExternalRoutedIntentionMemoryState, int]:
        """Append content and a fresh/transfer route address copy-on-write."""

        self._validate_state(state)
        next_cells, new_index = self.memory.append_cell(
            state.cells,
            source_cell=source_cell,
        )
        device = state.routing_keys.device
        dtype = state.routing_keys.dtype
        if source_cell is None:
            new_key = self.initial_routing_scale * torch.randn(
                1, self.context_width, device=device, dtype=dtype
            )
            new_bias = torch.zeros(1, device=device, dtype=dtype)
        else:
            if not 0 <= source_cell < state.routing_keys.shape[0]:
                raise ValueError("intention router source cell is out of range")
            new_key = state.routing_keys[source_cell : source_cell + 1].clone()
            new_bias = state.routing_bias[source_cell : source_cell + 1].clone()
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
    ) -> torch.Tensor:
        self._validate_state(state)
        self._validate_context(context, state)
        return self.memory.mean(state.cells, context)

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
        }

    def state_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> ExternalRoutedIntentionMemoryState:
        if payload.get("schema") != self.schema:
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
        state = ExternalRoutedIntentionMemoryState(cells=cells, **values)
        self._validate_state(state)
        return state
