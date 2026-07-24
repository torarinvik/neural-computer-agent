from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from experiments.syllogimous_latent_agent.model import AudioEncoder, VisionEncoder


ACTION_COUNT = 5


class SharedReasoningBlock(nn.Module):
    """A generic, weight-shared memory update with no task-specific operations."""

    def __init__(self, hidden: int, heads: int, expansion: int = 2):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * expansion), nn.GELU(),
            nn.Linear(hidden * expansion, hidden),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(memory)
        attended, _ = self.attention(normalized, self.memory_norm(memory),
                                     self.memory_norm(memory), need_weights=False)
        memory = memory + attended
        return memory + self.ff(self.ff_norm(memory))


@dataclass
class PolicyOutput:
    observation_logits: torch.Tensor
    answer_logits: torch.Tensor
    halt_logits: torch.Tensor
    values: torch.Tensor


class BitterLessonAgent(nn.Module):
    """Raw sensory policy with learned memory, reasoning, and adaptive halting.

    The only inputs at the model boundary are RGB frames, PCM samples, and a
    padding mask. There are deliberately no entity, relation, graph, or answer
    inputs and no task-specific algorithmic modules.
    """

    def __init__(self, hidden: int = 192, memory_slots: int = 16,
                 heads: int = 6, max_thought_steps: int = 8,
                 memory_core: str = "soft_slots",
                 thought_dynamics: str = "replace",
                 action_count: int = ACTION_COUNT):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if memory_slots < 2 or max_thought_steps < 1:
            raise ValueError("memory_slots >= 2 and max_thought_steps >= 1 required")
        if action_count < ACTION_COUNT:
            raise ValueError(f"action_count must be at least {ACTION_COUNT}")
        if memory_core not in {"soft_slots", "residual_slots", "residual_gru",
                               "event_transformer"}:
            raise ValueError(f"unknown memory core: {memory_core}")
        if thought_dynamics not in {"replace", "gated_residual"}:
            raise ValueError(f"unknown thought dynamics: {thought_dynamics}")
        if thought_dynamics == "gated_residual" and memory_core not in {
                "residual_gru", "event_transformer"}:
            raise ValueError("gated residual thoughts require a sequential memory core")
        self.hidden = hidden
        self.memory_slots = memory_slots
        self.max_thought_steps = max_thought_steps
        self.memory_core = memory_core
        self.thought_dynamics = thought_dynamics
        self.action_count = action_count
        self.vision = VisionEncoder(hidden)
        audio_hidden = max(16, hidden // 6)
        self.audio = AudioEncoder(audio_hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + audio_hidden, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.initial_memory = nn.Parameter(torch.randn(1, memory_slots, hidden) * 0.02)
        self.write_query = nn.Linear(hidden, hidden)
        self.write_gate = nn.Linear(hidden, memory_slots)
        self.event_cell = nn.GRUCell(hidden, hidden)
        self.reason = SharedReasoningBlock(hidden, heads)
        if memory_core in {"residual_gru", "event_transformer"}:
            self.thought_input = nn.Parameter(torch.randn(1, hidden) * 0.02)
            self.thought_cell = nn.GRUCell(hidden, hidden)
            if thought_dynamics == "gated_residual":
                self.thought_gate = nn.Linear(hidden * 2, hidden)
                nn.init.zeros_(self.thought_gate.weight)
                nn.init.constant_(self.thought_gate.bias, -3.0)
        if memory_core == "residual_gru":
            self.sequence_core = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        elif memory_core == "event_transformer":
            layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.0,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.sequence_core = nn.TransformerEncoder(layer, num_layers=2,
                                                       enable_nested_tensor=False)
        self.observation_head = nn.Linear(hidden, action_count)
        self.answer_head = nn.Linear(hidden, action_count)
        self.halt_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def _encode(self, frames: torch.Tensor, pcm: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        vision = self.vision(frames.reshape(batch * steps, *frames.shape[2:]))
        audio = self.audio(pcm.reshape(batch * steps, pcm.shape[-1]))
        fused = self.fusion(torch.cat((vision, audio), dim=-1))
        return fused.reshape(batch, steps, self.hidden)

    def forward(self, frames: torch.Tensor, pcm: torch.Tensor,
                mask: torch.Tensor) -> PolicyOutput:
        events = self._encode(frames, pcm)
        batch, steps, _ = events.shape
        memory = self.initial_memory.expand(batch, -1, -1).clone()
        controller = events.new_zeros(batch, self.hidden)
        observation_states = []
        for index in range(steps):
            proposed = self.event_cell(events[:, index], controller)
            active = mask[:, index, None]
            controller = torch.where(active, proposed, controller)
            weights = torch.softmax(self.write_gate(controller), dim=-1).unsqueeze(-1)
            content = self.write_query(controller).unsqueeze(1)
            proposed_memory = memory + weights * (content - memory)
            memory = torch.where(active[:, None], proposed_memory, memory)
            observation_states.append(controller)

        observation_states = torch.stack(observation_states, dim=1)
        if self.memory_core == "residual_gru":
            sequence_states, _ = self.sequence_core(events)
            rows = torch.arange(batch, device=events.device)
            controller = sequence_states[rows, mask.sum(1) - 1]
        elif self.memory_core == "event_transformer":
            causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool,
                                           device=events.device), diagonal=1)
            sequence_states = self.sequence_core(events, mask=causal,
                                                 src_key_padding_mask=~mask)
            rows = torch.arange(batch, device=events.device)
            controller = sequence_states[rows, mask.sum(1) - 1]
        elif self.memory_core == "residual_slots":
            memory = memory.clone()
            memory[:, 0] = memory[:, 0] + controller
        answer_logits, halt_logits, values = [], [], []
        for _ in range(self.max_thought_steps):
            if self.memory_core in {"soft_slots", "residual_slots"}:
                memory = self.reason(memory)
                controller = memory[:, 0]
            else:
                proposal = self.thought_cell(self.thought_input.expand(batch, -1), controller)
                if self.thought_dynamics == "gated_residual":
                    gate = torch.sigmoid(self.thought_gate(
                        torch.cat((controller, proposal), dim=-1)))
                    controller = controller + gate * (proposal - controller)
                else:
                    controller = proposal
            answer_logits.append(self.answer_head(controller))
            halt_logits.append(self.halt_head(controller).squeeze(-1))
            values.append(self.value_head(controller).squeeze(-1))
        return PolicyOutput(
            self.observation_head(observation_states),
            torch.stack(answer_logits, dim=1),
            torch.stack(halt_logits, dim=1),
            torch.stack(values, dim=1),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_config(scale: str) -> dict[str, int]:
    configs = {
        "1m": {"hidden": 96, "memory_slots": 8, "heads": 4, "max_thought_steps": 6},
        "2m": {"hidden": 160, "memory_slots": 12, "heads": 5, "max_thought_steps": 8},
        "5m": {"hidden": 384, "memory_slots": 16, "heads": 8, "max_thought_steps": 8},
        "20m": {"hidden": 896, "memory_slots": 32, "heads": 14, "max_thought_steps": 12},
    }
    if scale not in configs:
        raise ValueError(f"unknown scale {scale!r}; choose from {tuple(configs)}")
    return configs[scale].copy()
