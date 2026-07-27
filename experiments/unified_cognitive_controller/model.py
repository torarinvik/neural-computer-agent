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
    # Per-slot opening of the successor residuals, shape [batch, slots]. It is
    # exposed so a rung can measure and price how far its own new slot fires
    # outside the events it was added for. Empty when no slot exists.
    skill_adapter_openings: torch.Tensor | None = None
    # Norm of the perturbation each successor slot actually adds to the
    # intention, shape [batch, slots]. The opening alone understates this: a
    # nearly shut gate on a large residual still moves the answer, so this is
    # the quantity a locality price has to act on.
    skill_adapter_residual_norms: torch.Tensor | None = None


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
            adaptive_memory_replace_hidden: int = 8,
            adaptive_memory_replace_features: int = 5,
            relation_adapter_width: int = 0,
            relation_adapter_gated: bool = False,
            action_adapter_width: int = 0,
            action_adapter_gated: bool = False,
            skill_adapter_widths: tuple[int, ...] = (),
            skill_adapter_gate_mode: str = "sigmoid",
            skill_adapter_gate_hidden: int = 0) -> None:
        super().__init__()
        if skill_adapter_gate_mode not in ("sigmoid", "relu"):
            raise ValueError(
                "skill adapter gate mode must be sigmoid or relu")
        if (
                width < 16 or workspace_slots < 1 or intention_width < 2
                or adaptive_memory_read_hidden < 0
                or adaptive_memory_replace_hidden < 1
                or adaptive_memory_replace_features < 5
                or relation_adapter_width < 0
                or action_adapter_width < 0
                or any(value < 1 for value in skill_adapter_widths)):
            raise ValueError("controller dimensions are too small")
        self.width = width
        self.workspace_slots = workspace_slots
        self.intention_width = intention_width
        self.adaptive_memory_read = adaptive_memory_read
        self.adaptive_memory_read_hidden = adaptive_memory_read_hidden
        self.adaptive_memory_replace = adaptive_memory_replace
        self.adaptive_memory_replace_hidden = adaptive_memory_replace_hidden
        self.adaptive_memory_replace_features = adaptive_memory_replace_features
        self.relation_adapter_width = relation_adapter_width
        self.relation_adapter_gated = relation_adapter_gated
        self.action_adapter_width = action_adapter_width
        self.action_adapter_gated = action_adapter_gated
        self.skill_adapter_widths = tuple(skill_adapter_widths)
        self.skill_adapter_gate_mode = skill_adapter_gate_mode
        self.skill_adapter_gate_hidden = skill_adapter_gate_hidden
        # Training-time leak below the rectifier's knee. A rectified gate that
        # shuts everywhere before it has learned where to open has no gradient
        # left and stays shut forever; a small leak keeps that recoverable. It
        # is a schedule, not architecture, so it is a plain runtime attribute
        # that no checkpoint stores, and it must be returned to zero before any
        # measurement: only an exact zero leaves an inherited skill untouched.
        self.skill_adapter_gate_leak = 0.0
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
        # Optional generic answer-path relation adapter. It combines the
        # controller state available before a query with that query's sensory
        # event; it has no task, context, or action-label input. A zero final
        # projection makes insertion exactly behavior-preserving.
        self.relation_adapter = (
            nn.Sequential(
                nn.Linear(width * 2, relation_adapter_width),
                nn.GELU(),
                nn.Linear(relation_adapter_width, intention_width),
            )
            if relation_adapter_width else None)
        if self.relation_adapter is not None:
            nn.init.zeros_(self.relation_adapter[-1].weight)
            nn.init.zeros_(self.relation_adapter[-1].bias)
        self.relation_adapter_gate = (
            nn.Linear(width * 2, 1)
            if relation_adapter_width and relation_adapter_gated else None)
        if self.relation_adapter_gate is not None:
            nn.init.zeros_(self.relation_adapter_gate.weight)
            nn.init.constant_(self.relation_adapter_gate.bias, -2.0)
        self.action_adapter = (
            nn.Sequential(
                nn.Linear(width * 2, action_adapter_width),
                nn.GELU(),
                nn.Linear(action_adapter_width, ACTIONS),
            )
            if action_adapter_width else None)
        if self.action_adapter is not None:
            nn.init.zeros_(self.action_adapter[-1].weight)
            nn.init.zeros_(self.action_adapter[-1].bias)
        self.action_adapter_gate = (
            nn.Linear(width * 2, 1) if action_adapter_width and action_adapter_gated else None)
        if self.action_adapter_gate is not None:
            nn.init.zeros_(self.action_adapter_gate.weight)
            # The residual output itself is exactly zero at insertion, so a
            # moderately open gate preserves behavior while avoiding the
            # near-zero gradients caused by a saturated closed gate.
            nn.init.constant_(self.action_adapter_gate.bias, -2.0)
        # Indexable successor slots for later primitives. The two adapters
        # above were each claimed by one earlier rung; this stack lets rung N
        # add exactly one fresh zero-output residual without renaming or
        # reshaping anything an already promoted checkpoint stored. An empty
        # stack contributes no state, so older checkpoints load unchanged.
        self.skill_adapters = nn.ModuleList()
        self.skill_adapter_gates = nn.ModuleList()
        for slot_width in self.skill_adapter_widths:
            adapter = nn.Sequential(
                nn.Linear(width * 2, slot_width),
                nn.GELU(),
                nn.Linear(slot_width, intention_width),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            self.skill_adapters.append(adapter)
            # A linear gate has to separate this slot's own events from every
            # other skill's with one hyperplane. A hidden layer lets the
            # boundary bend, which is what decides how cleanly a slot can be
            # both fully available to its operation and fully absent elsewhere.
            if skill_adapter_gate_hidden:
                gate = nn.Sequential(
                    nn.Linear(width * 2, skill_adapter_gate_hidden),
                    nn.GELU(),
                    nn.Linear(skill_adapter_gate_hidden, 1),
                )
                output_layer = gate[-1]
            else:
                gate = nn.Linear(width * 2, 1)
                output_layer = gate
            nn.init.zeros_(output_layer.weight)
            # A sigmoid gate can never be exactly shut, so a slot always
            # perturbs every event it sees a little. The rectified gate can
            # reach exact zero and therefore be genuinely inert outside the
            # events it was added for. Either way the zero-initialized output
            # layer makes insertion exactly behavior-preserving; the positive
            # rectified bias only keeps the gate's own gradient alive.
            nn.init.constant_(
                output_layer.bias,
                1.0 if skill_adapter_gate_mode == "relu" else -2.0)
            self.skill_adapter_gates.append(gate)
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
        self.memory_replacement_extra_gate = (
            nn.Linear(
                adaptive_memory_replace_features - 5, 1, bias=False)
            if (
                adaptive_memory_replace
                and adaptive_memory_replace_features > 5)
            else None)
        if self.memory_replacement_extra_gate is not None:
            nn.init.zeros_(self.memory_replacement_extra_gate.weight)

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
        if (
                option_features.ndim != 3
                or option_features.shape[-1]
                != self.adaptive_memory_replace_features):
            raise ValueError(
                "replacement features must have shape "
                f"[batch, options, {self.adaptive_memory_replace_features}]")
        # The leading slice is contiguous only when no extra feature is
        # present. Materializing it keeps the base-five scores bit-identical
        # across widths, so adding a zero-initialized statistic is exactly
        # behavior-preserving rather than merely close.
        scores = self.memory_replacement_gate(
            option_features[..., :5].contiguous()).squeeze(-1)
        if self.memory_replacement_extra_gate is not None:
            scores = scores + self.memory_replacement_extra_gate(
                option_features[..., 5:]).squeeze(-1)
        return scores

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
        if self.relation_adapter is not None:
            relation_features = torch.cat([state.hidden, event], dim=-1)
            relation_residual = self.relation_adapter(relation_features)
            if self.relation_adapter_gate is not None:
                relation_residual = relation_residual * torch.sigmoid(
                    self.relation_adapter_gate(relation_features))
            intention = intention + relation_residual
        skill_adapter_openings = None
        skill_adapter_residual_norms = None
        if len(self.skill_adapters):
            # Successor slots read the same generic prior-state/query-event
            # pair as the legacy adapters: no task, context, or action label.
            slot_features = torch.cat([state.hidden, event], dim=-1)
            openings = []
            residual_norms = []
            for adapter, gate in zip(
                    self.skill_adapters, self.skill_adapter_gates):
                score = gate(slot_features)
                opening = (
                    # leaky_relu at slope zero is exactly relu, so a finished
                    # anneal restores exact-zero gating bit for bit.
                    torch.nn.functional.leaky_relu(
                        score, self.skill_adapter_gate_leak)
                    if self.skill_adapter_gate_mode == "relu"
                    else torch.sigmoid(score))
                residual = adapter(slot_features) * opening
                openings.append(opening)
                residual_norms.append(
                    residual.norm(dim=-1, keepdim=True))
                intention = intention + residual
            skill_adapter_openings = torch.cat(openings, dim=-1)
            skill_adapter_residual_norms = torch.cat(residual_norms, dim=-1)
        memory_context = torch.cat([hidden, read], dim=-1)
        logits = self.actuator(intention)
        if self.action_adapter is not None:
            adapter_features = torch.cat([state.hidden, event], dim=-1)
            adapter_residual = self.action_adapter(adapter_features)
            if self.action_adapter_gate is not None:
                adapter_residual = adapter_residual * torch.sigmoid(
                    self.action_adapter_gate(adapter_features))
            logits = logits + adapter_residual
        output = ControllerOutput(
            logits=logits,
            intention=intention,
            memory_key=self.memory_key(memory_context),
            memory_value=self.memory_value(memory_context),
            memory_write_strength=torch.sigmoid(
                self.memory_write(memory_context)).squeeze(-1),
            workspace_read=read,
            skill_adapter_openings=skill_adapter_openings,
            skill_adapter_residual_norms=skill_adapter_residual_norms,
        )
        return output, ControllerState(hidden, workspace)
