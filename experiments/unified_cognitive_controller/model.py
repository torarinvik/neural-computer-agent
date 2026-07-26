"""One recurrent controller coordinating sensory encoding and latent memory."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .environment import ACTIONS, NULL_ACTION


class VisionEventEncoder(nn.Module):
    """Small modality encoder; it receives pixels and emits one event latent."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(width),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.network(frames)


@dataclass
class ControllerState:
    hidden: torch.Tensor
    workspace: torch.Tensor

    def detach(self) -> "ControllerState":
        return ControllerState(self.hidden.detach(), self.workspace.detach())


@dataclass
class ControllerOutput:
    logits: torch.Tensor
    intention: torch.Tensor
    memory_key: torch.Tensor
    memory_value: torch.Tensor
    memory_write_strength: torch.Tensor
    workspace_read: torch.Tensor


class UnifiedCognitiveController(nn.Module):
    """A single task-agnostic controller with a differentiable RAM workspace.

    There are no primitive-specific heads.  The same encoder, recurrent cell,
    workspace operations, latent intention, and actuator adapter process every
    trial. Persistent-memory keys and values are exposed for the later disk
    rung, but disk writes are deliberately disabled in the first fast-learning
    experiment.
    """

    def __init__(
            self, width: int = 96, workspace_slots: int = 8,
            intention_width: int = 24,
            adaptive_memory_read: bool = False,
            adaptive_memory_read_hidden: int = 0,
            adaptive_memory_replace: bool = False,
            adaptive_memory_replace_hidden: int = 8) -> None:
        super().__init__()
        if (
                width < 16 or workspace_slots < 1 or intention_width < 2
                or adaptive_memory_read_hidden < 0
                or adaptive_memory_replace_hidden < 1):
            raise ValueError("controller dimensions are too small")
        self.width = width
        self.workspace_slots = workspace_slots
        self.intention_width = intention_width
        self.adaptive_memory_read = adaptive_memory_read
        self.adaptive_memory_read_hidden = adaptive_memory_read_hidden
        self.adaptive_memory_replace = adaptive_memory_replace
        self.adaptive_memory_replace_hidden = adaptive_memory_replace_hidden
        self.vision = VisionEventEncoder(width)
        self.action_embedding = nn.Embedding(ACTIONS + 1, width // 4)
        feedback_width = width // 4
        self.feedback_encoder = nn.Sequential(
            nn.Linear(2, feedback_width), nn.Tanh())
        self.read_query = nn.Linear(width * 2, width)
        controller_input = width * 3 + width // 4 + feedback_width
        self.controller = nn.GRUCell(controller_input, width)
        self.write_gate = nn.Linear(width * 3, 1)
        self.write_query = nn.Linear(width * 3, width)
        self.write_value = nn.Sequential(
            nn.Linear(width * 3, width), nn.Tanh())
        self.intention = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, intention_width),
            nn.Tanh(),
        )
        # This is the replaceable device/protocol adapter. The controller emits
        # an abstract intention before this final mapping.
        self.actuator = nn.Linear(intention_width, ACTIONS)
        self.memory_key = nn.Linear(width * 2, width)
        self.memory_value = nn.Linear(width * 2, width)
        self.memory_write = nn.Linear(width * 2, 1)
        if not adaptive_memory_read:
            self.memory_read_gate = None
        elif adaptive_memory_read_hidden == 0:
            self.memory_read_gate = nn.Linear(4, 1)
        else:
            self.memory_read_gate = nn.Sequential(
                nn.Linear(4, adaptive_memory_read_hidden),
                nn.GELU(),
                nn.Linear(adaptive_memory_read_hidden, 1),
            )
        self.memory_replacement_gate = (
            nn.Sequential(
                nn.Linear(5, adaptive_memory_replace_hidden),
                nn.GELU(),
                nn.Linear(adaptive_memory_replace_hidden, 1),
            )
            if adaptive_memory_replace else None)

    def memory_read_probability(
            self, features: torch.Tensor) -> torch.Tensor:
        """Return a generic read/no-read probability from memory statistics."""
        if self.memory_read_gate is None:
            raise RuntimeError("adaptive memory read is not enabled")
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("memory read features must have shape [batch, 4]")
        return torch.sigmoid(
            self.memory_read_gate(features)).squeeze(-1)

    def memory_replacement_scores(
            self, option_features: torch.Tensor) -> torch.Tensor:
        """Score skip/replace options from generic latent-memory statistics."""
        if self.memory_replacement_gate is None:
            raise RuntimeError("adaptive memory replacement is not enabled")
        if option_features.ndim != 3 or option_features.shape[-1] != 5:
            raise ValueError(
                "replacement features must have shape [batch, options, 5]")
        return self.memory_replacement_gate(
            option_features).squeeze(-1)

    def initial_state(
            self, batch_size: int, *, device: torch.device | str,
            dtype: torch.dtype = torch.float32) -> ControllerState:
        return ControllerState(
            torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            torch.zeros(
                batch_size, self.workspace_slots, self.width,
                device=device, dtype=dtype),
        )

    def step(
            self, frame: torch.Tensor, state: ControllerState,
            previous_action: torch.Tensor, previous_reward: torch.Tensor,
            has_feedback: torch.Tensor,
            retrieved_memory: torch.Tensor | None = None,
            *, disable_workspace: bool = False) -> tuple[
                ControllerOutput, ControllerState]:
        event = self.vision(frame)
        if retrieved_memory is None:
            retrieved_memory = torch.zeros_like(event)
        query = torch.nn.functional.normalize(
            self.read_query(torch.cat([event, state.hidden], dim=-1)),
            dim=-1)
        slots = torch.nn.functional.normalize(
            state.workspace, dim=-1)
        scores = torch.einsum("bw,bsw->bs", query, slots)
        weights = torch.softmax(scores, dim=-1)
        read = torch.einsum("bs,bsw->bw", weights, state.workspace)
        if disable_workspace:
            read = torch.zeros_like(read)

        action_embedding = self.action_embedding(previous_action)
        feedback = self.feedback_encoder(torch.stack([
            previous_reward, has_feedback], dim=-1))
        controller_input = torch.cat([
            event, read, retrieved_memory, action_embedding, feedback], dim=-1)
        hidden = self.controller(controller_input, state.hidden)

        write_context = torch.cat([event, hidden, read], dim=-1)
        write_query = torch.nn.functional.normalize(
            self.write_query(write_context), dim=-1)
        write_scores = torch.einsum("bw,bsw->bs", write_query, slots)
        write_weights = torch.softmax(write_scores, dim=-1)
        gate = torch.sigmoid(self.write_gate(write_context))
        candidate = self.write_value(write_context)
        update = gate.unsqueeze(-1) * write_weights.unsqueeze(-1)
        workspace = (
            state.workspace * (1.0 - update)
            + candidate.unsqueeze(1) * update)
        if disable_workspace:
            workspace = torch.zeros_like(workspace)

        combined = torch.cat([hidden, read, event], dim=-1)
        intention = self.intention(combined)
        memory_context = torch.cat([hidden, read], dim=-1)
        output = ControllerOutput(
            logits=self.actuator(intention),
            intention=intention,
            memory_key=self.memory_key(memory_context),
            memory_value=self.memory_value(memory_context),
            memory_write_strength=torch.sigmoid(
                self.memory_write(memory_context)).squeeze(-1),
            workspace_read=read,
        )
        return output, ControllerState(hidden, workspace)
