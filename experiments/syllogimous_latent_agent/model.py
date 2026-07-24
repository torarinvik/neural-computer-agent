from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


ACTION_COUNT = 5


class PortableSpatialPool(nn.Module):
    """Preserve the 4x8 interface on MPS, whose adaptive kernel is restricted."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type == "mps":
            height, width = inputs.shape[-2:]
            rows = []
            for row in range(4):
                row_start = row * height // 4
                row_end = ((row + 1) * height + 3) // 4
                columns = []
                for column in range(8):
                    column_start = column * width // 8
                    column_end = ((column + 1) * width + 7) // 8
                    columns.append(inputs[..., row_start:row_end,
                                          column_start:column_end].mean((-2, -1)))
                rows.append(torch.stack(columns, dim=-1))
            return torch.stack(rows, dim=-2)
        return nn.functional.adaptive_avg_pool2d(inputs, (4, 8))


class VisionEncoder(nn.Module):
    def __init__(self, hidden: int = 384):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.GELU(),
            PortableSpatialPool(), nn.Flatten(),
            nn.Linear(192 * 4 * 8, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.features(frames)


class AudioEncoder(nn.Module):
    def __init__(self, output: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(128, output), nn.GELU(),
        )

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        return self.features(pcm.unsqueeze(1))


class RecursiveBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.net(self.norm(state))


class CachedGraphLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, 6, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Linear(hidden * 2, hidden))

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                memory_padding: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.query_norm(query)
        attended, _ = self.attention(normalized, self.memory_norm(memory),
                                     self.memory_norm(memory),
                                     key_padding_mask=memory_padding,
                                     need_weights=False)
        query = query + attended
        return query + self.ff(self.ff_norm(query))


@dataclass
class AgentOutput:
    logits: torch.Tensor
    halt_logits: torch.Tensor
    subject_logits: torch.Tensor
    relation_logits: torch.Tensor
    object_logits: torch.Tensor
    final_logits: torch.Tensor


class LatentAgent(nn.Module):
    """The forward boundary accepts only raw RGB, raw PCM, and a padding mask."""

    def __init__(self, core: str = "gru", hidden: int = 384,
                 recursive_steps: int = 4, max_events: int = 96,
                 entity_count: int = 64, use_positions: bool = True):
        super().__init__()
        if core not in {"gru", "graph", "graph_cached", "closure", "recursive"}:
            raise ValueError(f"unknown core: {core}")
        self.core_name = core
        self.hidden = hidden
        self.recursive_steps = recursive_steps
        self.entity_count = entity_count
        self.use_positions = use_positions
        self.vision = VisionEncoder(hidden)
        self.audio = AudioEncoder(64)
        self.fusion = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.LayerNorm(hidden), nn.GELU())
        if core == "gru":
            self.core = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        elif core == "graph":
            layer = nn.TransformerEncoderLayer(hidden, 6, hidden * 2, dropout=0.0,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.core = nn.TransformerEncoder(layer, num_layers=2,
                                              enable_nested_tensor=False)
            self.positions = nn.Parameter(torch.zeros(1, max_events, hidden))
        elif core == "graph_cached":
            self.core = nn.ModuleList([CachedGraphLayer(hidden), CachedGraphLayer(hidden)])
            if use_positions:
                self.positions = nn.Parameter(torch.zeros(1, max_events, hidden))
        elif core == "closure":
            self.core = nn.Identity()
            self.truth_scale = nn.Parameter(torch.tensor(8.0))
        else:
            self.memory_cell = nn.GRUCell(hidden, hidden)
            self.core = RecursiveBlock(hidden)
        self.action_head = nn.Linear(hidden, ACTION_COUNT)
        self.halt_head = nn.Linear(hidden, 1)
        self.subject_head = nn.Linear(hidden, entity_count)
        self.relation_head = nn.Linear(hidden, 8)
        self.object_head = nn.Linear(hidden, entity_count)
        self.final_head = nn.Linear(hidden, 1)

    def encode(self, frames: torch.Tensor, pcm: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        vision = self.vision(frames.reshape(batch * steps, *frames.shape[2:]))
        audio = self.audio(pcm.reshape(batch * steps, pcm.shape[-1]))
        return self.fusion(torch.cat((vision, audio), dim=-1)).reshape(batch, steps, self.hidden)

    def forward(self, frames: torch.Tensor, pcm: torch.Tensor,
                mask: torch.Tensor) -> AgentOutput:
        events = self.encode(frames, pcm)
        subject_logits = self.subject_head(events)
        relation_logits = self.relation_head(events)
        object_logits = self.object_head(events)
        final_logits = self.final_head(events).squeeze(-1)
        direct_logits = None
        if self.core_name == "gru":
            states, _ = self.core(events)
        elif self.core_name == "graph":
            steps = events.shape[1]
            causal = torch.triu(torch.ones(steps, steps, device=events.device, dtype=torch.bool), 1)
            states = self.core(events + self.positions[:, :steps], mask=causal,
                               src_key_padding_mask=~mask)
        elif self.core_name == "graph_cached":
            outputs = []
            memory = None
            memory_mask = None
            for index in range(events.shape[1]):
                query = events[:, index:index + 1]
                if self.use_positions:
                    query = query + self.positions[:, index:index + 1]
                current_mask = ~mask[:, index:index + 1]
                keys = query if memory is None else torch.cat((memory, query), dim=1)
                key_mask = current_mask if memory_mask is None else torch.cat((memory_mask, current_mask), dim=1)
                state = query
                for layer in self.core:
                    state = layer(state, keys, key_mask)
                outputs.append(state[:, 0])
                memory = state if memory is None else torch.cat((memory, state), dim=1)
                memory_mask = key_mask
            states = torch.stack(outputs, dim=1)
        elif self.core_name == "closure":
            batch, steps, _ = events.shape
            # Discrete graph construction must not send unstable answer gradients
            # into an untrained visual parser. The parser has exact auxiliary
            # supervision; closure consumes its predictions as a module boundary.
            subject = self._hard_distribution(subject_logits).detach()
            relation = self._hard_distribution(relation_logits).detach()
            obj = self._hard_distribution(object_logits).detach()
            forward = relation[..., 0::2].sum(-1)
            reverse = relation[..., 1::2].sum(-1)
            edge_forward = subject.unsqueeze(-1) * obj.unsqueeze(-2)
            edge_reverse = obj.unsqueeze(-1) * subject.unsqueeze(-2)
            edges = forward[..., None, None] * edge_forward + reverse[..., None, None] * edge_reverse
            lengths = mask.sum(1) - 1
            rows = torch.arange(batch, device=events.device)
            premise_mask = mask.clone()
            premise_mask[rows, lengths] = False
            closure = (edges * premise_mask[..., None, None]).sum(1).clamp(0.0, 1.0)
            iterations = max(1, (self.entity_count - 1).bit_length())
            for _ in range(iterations):
                closure = (closure + torch.bmm(closure, closure)).clamp(0.0, 1.0)
            query_subject = subject[rows, lengths]
            query_object = obj[rows, lengths]
            query_forward = forward[rows, lengths]
            query_reverse = reverse[rows, lengths]
            forward_truth = torch.einsum("bi,bij,bj->b", query_subject, closure, query_object)
            reverse_truth = torch.einsum("bi,bij,bj->b", query_object, closure, query_subject)
            truth = (query_forward * forward_truth + query_reverse * reverse_truth).clamp(0.0, 1.0)
            truth_score = self.truth_scale.clamp(1.0, 16.0) * (2.0 * truth - 1.0)
            gate = 8.0 * (2.0 * torch.sigmoid(final_logits) - 1.0)
            direct_logits = events.new_full((batch, steps, ACTION_COUNT), -16.0)
            direct_logits[..., 1] = -gate
            direct_logits[..., 2] = gate
            direct_logits[..., 3] = gate
            direct_logits[rows, lengths, 2] -= truth_score
            direct_logits[rows, lengths, 3] += truth_score
            states = events
        else:
            batch, steps, _ = events.shape
            memory = events.new_zeros(batch, self.hidden)
            outputs = []
            for index in range(steps):
                proposed = self.memory_cell(events[:, index], memory)
                memory = torch.where(mask[:, index, None], proposed, memory)
                reasoned = memory
                for _ in range(self.recursive_steps):
                    reasoned = self.core(reasoned)
                outputs.append(reasoned)
            states = torch.stack(outputs, dim=1)
        return AgentOutput(direct_logits if direct_logits is not None else self.action_head(states),
                           self.halt_head(states).squeeze(-1), subject_logits,
                           relation_logits, object_logits, final_logits)

    @staticmethod
    def _hard_distribution(logits: torch.Tensor) -> torch.Tensor:
        soft = torch.softmax(logits, dim=-1)
        hard = torch.nn.functional.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        return hard + soft - soft.detach()

    def init_stream_state(self) -> dict[str, torch.Tensor | int | None]:
        if self.core_name == "closure":
            return {"adjacency": None, "events": 0}
        if self.core_name != "graph_cached":
            raise ValueError("incremental cache is implemented for graph_cached and closure")
        return {"memory": None, "events": 0}

    def stream_step(self, frame: torch.Tensor, pcm: torch.Tensor,
                    state: dict[str, torch.Tensor | int | None]) -> tuple[AgentOutput, dict]:
        """Encode exactly one new sensory event and append one cached state."""
        if self.core_name == "closure":
            event = self.fusion(torch.cat((self.vision(frame), self.audio(pcm)), dim=-1)).unsqueeze(1)
            subject_logits = self.subject_head(event)
            relation_logits = self.relation_head(event)
            object_logits = self.object_head(event)
            final_logits = self.final_head(event).squeeze(-1)
            subject = self._hard_distribution(subject_logits).detach()[:, 0]
            relation = self._hard_distribution(relation_logits).detach()[:, 0]
            obj = self._hard_distribution(object_logits).detach()[:, 0]
            forward = relation[..., 0::2].sum(-1)
            reverse = relation[..., 1::2].sum(-1)
            edge = (forward[:, None, None] * subject.unsqueeze(-1) * obj.unsqueeze(-2)
                    + reverse[:, None, None] * obj.unsqueeze(-1) * subject.unsqueeze(-2))
            adjacency = state["adjacency"]
            if adjacency is None:
                adjacency = edge.new_zeros(edge.shape)
            is_final = final_logits[:, 0] > 0
            logits = event.new_full((event.shape[0], 1, ACTION_COUNT), -16.0)
            logits[..., 1] = 8.0
            if bool(is_final.any()):
                closure = adjacency
                for _ in range(max(1, (self.entity_count - 1).bit_length())):
                    closure = (closure + torch.bmm(closure, closure)).clamp(0.0, 1.0)
                forward_truth = torch.einsum("bi,bij,bj->b", subject, closure, obj)
                reverse_truth = torch.einsum("bi,bij,bj->b", obj, closure, subject)
                truth = (forward * forward_truth + reverse * reverse_truth).clamp(0.0, 1.0)
                score = self.truth_scale.clamp(1.0, 16.0) * (2.0 * truth - 1.0)
                logits[is_final, 0, 1] = -16.0
                logits[is_final, 0, 2] = -score[is_final]
                logits[is_final, 0, 3] = score[is_final]
            next_adjacency = torch.where(is_final[:, None, None], adjacency,
                                         (adjacency + edge).clamp(0.0, 1.0))
            output = AgentOutput(logits, self.halt_head(event).squeeze(-1),
                                 subject_logits, relation_logits, object_logits, final_logits)
            return output, {"adjacency": next_adjacency,
                            "events": int(state["events"]) + 1}
        if self.core_name != "graph_cached":
            raise ValueError("incremental cache is implemented for graph_cached and closure")
        event = self.fusion(torch.cat((self.vision(frame), self.audio(pcm)), dim=-1)).unsqueeze(1)
        index = int(state["events"])
        query = event
        if self.use_positions:
            query = query + self.positions[:, index:index + 1]
        previous = state["memory"]
        keys = query if previous is None else torch.cat((previous, query), dim=1)
        reasoned = query
        for layer in self.core:
            reasoned = layer(reasoned, keys)
        memory = reasoned if previous is None else torch.cat((previous, reasoned), dim=1)
        output = AgentOutput(self.action_head(reasoned), self.halt_head(reasoned).squeeze(-1),
                             self.subject_head(event), self.relation_head(event),
                             self.object_head(event), self.final_head(event).squeeze(-1))
        return output, {"memory": memory, "events": index + 1}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
