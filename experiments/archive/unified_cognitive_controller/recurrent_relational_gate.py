"""Causal recurrent relational gate for task-agnostic episodic context."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RecurrentRelationalGate(nn.Module):
    """Read prior opaque events and emit a zero-init intention residual.

    The gate has no task/ring input.  It receives only an amodal event, the
    previous opaque action, and the previous scalar outcome.  ``history`` is a
    list of projected event snapshots retained by the caller, so the read is
    causal and runtime-variable.  The final projection is zero-initialized to
    make adding the branch behavior-preserving at construction.
    """

    def __init__(self, *, event_width: int, action_count: int,
                 hidden_width: int, intention_width: int,
                 max_history: int = 10) -> None:
        super().__init__()
        if min(event_width, action_count, hidden_width, intention_width) < 1:
            raise ValueError("gate dimensions must be positive")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.event_width = int(event_width)
        self.action_count = int(action_count)
        self.hidden_width = int(hidden_width)
        self.intention_width = int(intention_width)
        self.max_history = int(max_history)
        self.input_projection = nn.Linear(event_width + action_count + 2,
                                          hidden_width)
        self.position = nn.Parameter(torch.randn(max_history, hidden_width) * 0.02)
        self.query = nn.Linear(hidden_width, hidden_width, bias=False)
        self.key = nn.Linear(hidden_width * 3, hidden_width, bias=False)
        self.relation = nn.Sequential(
            nn.Linear(hidden_width * 4, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width))
        self.value = nn.Sequential(
            nn.Linear(hidden_width * 4, hidden_width), nn.GELU(),
            nn.Linear(hidden_width, hidden_width))
        self.update = nn.GRUCell(hidden_width * 2, hidden_width)
        self.output = nn.Linear(hidden_width, intention_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_width, device=device)

    def forward(
            self, event: torch.Tensor, state: torch.Tensor | None,
            history: list[torch.Tensor], previous_action: torch.Tensor,
            previous_reward: torch.Tensor,
            has_feedback: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if event.ndim != 2 or event.shape[-1] != self.event_width:
            raise ValueError("event must have shape [batch, event_width]")
        batch = event.shape[0]
        if state is None:
            state = self.initial_state(batch, device=event.device)
        action_index = previous_action.clamp_min(0).clamp_max(
            self.action_count - 1)
        action = F.one_hot(action_index, num_classes=self.action_count).float()
        token = torch.cat((event, action, previous_reward.unsqueeze(-1),
                           has_feedback.unsqueeze(-1)), dim=-1)
        current = self.input_projection(token)
        prior = history[-self.max_history:]
        if prior:
            prior_tensor = torch.stack(prior, dim=1)
            current_expanded = current.unsqueeze(1).expand_as(prior_tensor)
            position = self.position[:len(prior)].unsqueeze(0).expand(
                batch, -1, -1)
            relation_input = torch.cat((
                current_expanded, prior_tensor,
                current_expanded * prior_tensor, position), dim=-1)
            relation = self.relation(relation_input)
            address_input = torch.cat((
                current_expanded, prior_tensor,
                current_expanded * prior_tensor), dim=-1)
            weights = F.softmax(
                (self.query(current).unsqueeze(1) * self.key(address_input))
                .sum(-1) / self.hidden_width**0.5, dim=-1)
            values = self.value(torch.cat((
                current_expanded, prior_tensor, relation, position), dim=-1))
            context = (weights.unsqueeze(-1) * values).sum(1)
        else:
            context = torch.zeros_like(current)
        new_state = self.update(torch.cat((current, context), dim=-1), state)
        return self.output(new_state), new_state, current
