from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from experiments.syllogimous_latent_agent.model import AudioEncoder, VisionEncoder

from .memory import PersistentMemory


@dataclass
class NeuralComputerOutput:
    observation_logits: torch.Tensor
    answer_logits: torch.Tensor
    halt_logits: torch.Tensor
    values: torch.Tensor
    write_keys: torch.Tensor
    write_values: torch.Tensor
    write_logits: torch.Tensor
    write_strengths: torch.Tensor
    read_confidence: torch.Tensor
    workspace: torch.Tensor
    read_context: torch.Tensor
    write_source: torch.Tensor
    event_binding_residual: torch.Tensor


class FixedPairwiseTransfer(nn.Module):
    """Frozen audited relation-to-writer adapter with an exact zero start."""

    def __init__(self, pairwise_checkpoint: str, projection_checkpoint: str,
                 hidden: int, width: int = 64) -> None:
        super().__init__()
        pair_payload = torch.load(pairwise_checkpoint, map_location="cpu", weights_only=False)
        transfer_payload = torch.load(projection_checkpoint, map_location="cpu", weights_only=False)
        pair_state = pair_payload.get("model", pair_payload)
        if "projection" not in transfer_payload:
            raise ValueError("transfer checkpoint has no writer projection")
        self.mean = nn.Parameter(pair_payload["mean"].detach().float(), requires_grad=False)
        self.scale = nn.Parameter(pair_payload["scale"].detach().float(), requires_grad=False)
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(),
                                     nn.Linear(width, width), nn.LayerNorm(width))
        self.positions = nn.Parameter(torch.empty(1, 3, width), requires_grad=False)
        self.pair_head = nn.Sequential(nn.Linear(width * 9, width * 3), nn.GELU(),
                                       nn.LayerNorm(width * 3), nn.Linear(width * 3, width),
                                       nn.GELU(), nn.Linear(width, 2))
        self.projection = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.strength = nn.Parameter(torch.zeros(()))
        self.project.load_state_dict({k.removeprefix("project."): v for k, v in pair_state.items()
                                      if k.startswith("project.")})
        self.positions.data.copy_(pair_state["positions"])
        self.pair_head.load_state_dict({k.removeprefix("head."): v for k, v in pair_state.items()
                                        if k.startswith("head.")})
        self.projection.load_state_dict(transfer_payload["projection"])
        for module in (self.project, self.pair_head, self.projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, write_source: torch.Tensor,
                observation_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        snapshots, valid = EventSnapshotWriteBinder.recent_events(observation_states, mask)
        events = (self.project((snapshots - self.mean) / self.scale.clamp_min(1e-6))
                  + self.positions) * valid[:, :, None]
        relations = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            relations.extend((events[:, left] * events[:, right],
                              (events[:, left] - events[:, right]).abs()))
        latent = self.pair_head[:-1](torch.cat((*events.unbind(dim=1), *relations), dim=-1))
        return write_source + self.strength * self.projection(latent)


class EventSnapshotWriteBinder(nn.Module):
    """Generic three-event relation module for the persistent-memory write path."""

    def __init__(self, hidden: int, width: int = 64,
                 use_write_pairs: bool = False,
                 pairwise_transfer_checkpoint: tuple[str, str] | None = None) -> None:
        super().__init__()
        self.use_write_pairs = use_write_pairs
        self.pairwise_transfer = (FixedPairwiseTransfer(*pairwise_transfer_checkpoint, hidden, width)
                                  if pairwise_transfer_checkpoint else None)
        self.last_relation_features = None
        self.project = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width))
        self.positions = nn.Parameter(torch.randn(1, 3, width) * 0.02)
        self.relation = nn.Sequential(
            nn.Linear(width * (16 if use_write_pairs else 9), width * 3), nn.GELU(), nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, hidden))
        if use_write_pairs:
            self.write_project = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(),
                nn.Linear(width, width), nn.LayerNorm(width))
        self.gate = nn.Linear(hidden * 2, hidden)
        # Enabling the module must be exactly behavior-preserving before training.
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def recent_events(observation_states: torch.Tensor,
                      mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the last three valid sensory events, preserving their order."""
        batch, steps, hidden = observation_states.shape
        lengths = mask.long().sum(dim=1)
        offsets = torch.arange(3, device=observation_states.device)[None]
        starts = (lengths - 3).clamp_min(0)[:, None]
        indices = (starts + offsets).clamp_max(steps - 1)
        snapshots = observation_states.gather(
            1, indices[:, :, None].expand(batch, 3, hidden))
        valid = offsets < lengths.clamp_max(3)[:, None]
        return snapshots, valid

    def forward(self, write_source: torch.Tensor,
                observation_states: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        snapshots, valid = self.recent_events(observation_states, mask)
        events = (self.project(snapshots) + self.positions) * valid[:, :, None]
        relations = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            relations.extend((events[:, left] * events[:, right],
                              (events[:, left] - events[:, right]).abs()))
        if self.use_write_pairs:
            write = self.write_project(write_source)
            relations.extend((write,))
            for event in events.unbind(dim=1):
                relations.extend((write * event, (write - event).abs()))
        relation_features = torch.cat((*events.unbind(dim=1), *relations), dim=-1)
        self.last_relation_features = relation_features
        bound = self.relation(relation_features)
        self.last_bound = bound
        gate = torch.sigmoid(self.gate(torch.cat((write_source, bound), dim=-1)))
        output = write_source + gate * bound
        if self.pairwise_transfer is not None:
            output = self.pairwise_transfer(output, observation_states, mask)
        return output


class FactorizedEventAnswerRouter(nn.Module):
    """Generic learned routing over a support row and two query events."""

    def __init__(self, hidden: int, width: int = 64) -> None:
        super().__init__()
        for name in ("support", "first", "second"):
            self.register_buffer(name + "_mean", torch.zeros(1, hidden))
            self.register_buffer(name + "_scale", torch.ones(1, hidden))
        self.support = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU())

        def candidate_module():
            return nn.Sequential(
                nn.LayerNorm(hidden * 7),
                nn.Linear(hidden * 7, width), nn.GELU())

        self.first_candidate = candidate_module()
        self.second_candidate = candidate_module()
        self.rule_head = nn.Linear(width, 2)
        self.first_candidate_head = nn.Linear(width, 8)
        self.second_candidate_head = nn.Linear(width, 8)
        self.answer_gate = nn.Linear(2, 2)

    def forward(self, support, first, second, *,
                first_action_override=None, second_action_override=None,
                override_strength=0.0):
        support = (
            support - self.support_mean) / self.support_scale.clamp_min(1e-5)
        first = (first - self.first_mean) / self.first_scale.clamp_min(1e-5)
        second = (
            second - self.second_mean) / self.second_scale.clamp_min(1e-5)
        density_distance = torch.cat(
            (support, first, second), dim=-1).square().mean(-1)
        rule = self.support(support)

        def candidate(module, primary, other):
            return module(torch.cat((
                support, primary, other, support * primary, support * other,
                primary * other, (primary - other).abs()), dim=-1))

        first_latent = candidate(self.first_candidate, first, second)
        second_latent = candidate(self.second_candidate, second, first)
        rule_logits = self.rule_head(rule)
        first_logits = self.first_candidate_head(first_latent)
        second_logits = self.second_candidate_head(second_latent)
        if first_action_override is not None or second_action_override is not None:
            if first_action_override is None or second_action_override is None:
                raise ValueError("both candidate-action overrides are required")
            first_logits = (
                first_logits + override_strength *
                (first_action_override - first_logits))
            second_logits = (
                second_logits + override_strength *
                (second_action_override - second_logits))
        route = torch.softmax(self.answer_gate(rule_logits), dim=-1)
        soft_action = (
            route[:, :1] * first_logits + route[:, 1:] * second_logits)
        hard_route = nn.functional.one_hot(
            route.argmax(-1), num_classes=2).to(route)
        # Hard forward decision with a straight-through soft-route gradient.
        routed = hard_route + route - route.detach()
        hard_action = (
            routed[:, :1] * first_logits + routed[:, 1:] * second_logits)
        return {
            "action": soft_action,
            "hard_action": hard_action,
            "route": route,
            "rule": rule_logits,
            "first_action": first_logits,
            "second_action": second_logits,
            "density_distance": density_distance,
        }


class EventIndexedMemoryReader(nn.Module):
    """Content-based relation reader over a variable-size event memory.

    Rows are sensory-derived memories and queries are sensory/recurrent states.
    Shared row processing plus symmetric pooling makes the reader invariant to
    storage order. Normalization buffers travel with checkpoints.
    """

    def __init__(self, hidden: int, width: int = 64, actions: int = 8):
        super().__init__()
        self.register_buffer("rows_mean", torch.zeros(1, 1, hidden))
        self.register_buffer("rows_scale", torch.ones(1, 1, hidden))
        self.register_buffer("query_mean", torch.zeros(1, hidden))
        self.register_buffer("query_scale", torch.ones(1, hidden))
        self.row = nn.Sequential(nn.Linear(hidden, width), nn.GELU())
        self.query = nn.Sequential(nn.Linear(hidden, width), nn.GELU())
        self.relation = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU())
        self.answer = nn.Linear(width * 2, actions)

    def forward(self, rows: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 3 or query.ndim != 2:
            raise ValueError("expected rows [batch, events, hidden] and query [batch, hidden]")
        if rows.shape[0] != query.shape[0] or rows.shape[1] < 1:
            raise ValueError("reader requires a matching batch and at least one event")
        rows = (rows - self.rows_mean) / self.rows_scale.clamp_min(1e-5)
        query = (query - self.query_mean) / self.query_scale.clamp_min(1e-5)
        row = self.row(rows)
        query = self.query(query)
        expanded = query[:, None].expand_as(row)
        related = self.relation(torch.cat(
            (row, expanded, row * expanded), dim=-1))
        pooled = torch.cat(
            (related.mean(1), related.max(1).values), dim=-1)
        return self.answer(pooled)


class ContentAddressedEventMemoryReader(nn.Module):
    """Learn latent key/value slots per event and retrieve them by query."""

    def __init__(self, hidden: int, width: int = 64, slots: int = 2,
                 actions: int = 8):
        super().__init__()
        self.slots = slots
        self.width = width
        self.register_buffer("rows_mean", torch.zeros(1, 1, hidden))
        self.register_buffer("rows_scale", torch.ones(1, 1, hidden))
        self.register_buffer("query_mean", torch.zeros(1, hidden))
        self.register_buffer("query_scale", torch.ones(1, hidden))
        self.keys = nn.Linear(hidden, slots * width)
        self.values = nn.Linear(hidden, slots * width)
        self.query = nn.Sequential(
            nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, width))
        self.slot_bias = nn.Parameter(
            torch.randn(1, 1, slots, width) * .02)
        self.action = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, actions))

    def forward(self, rows: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 3 or query.ndim != 2:
            raise ValueError(
                "expected rows [batch, events, hidden] and query [batch, hidden]")
        if rows.shape[0] != query.shape[0] or rows.shape[1] < 1:
            raise ValueError("reader requires a matching batch and at least one event")
        rows = (rows - self.rows_mean) / self.rows_scale.clamp_min(1e-5)
        query = (query - self.query_mean) / self.query_scale.clamp_min(1e-5)
        batch, events, _ = rows.shape
        keys = self.keys(rows).reshape(
            batch, events, self.slots, self.width) + self.slot_bias
        values = self.values(rows).reshape(
            batch, events, self.slots, self.width)
        keys = keys.flatten(1, 2)
        values = values.flatten(1, 2)
        query = self.query(query)
        attention = torch.softmax(
            torch.einsum("bd,bsd->bs", query, keys) /
            self.width ** .5, dim=-1)
        return (
            attention[:, :, None] * self.action(values)
        ).sum(dim=1)


class NeuralComputerAgent(nn.Module):
    """Sensory controller with fast workspace and external persistent memory.

    Frames, PCM, and padding are the only environmental inputs. PersistentMemory
    is the agent's own state, never a game-state or semantic side channel.
    """

    def __init__(self, hidden: int = 160, workspace_slots: int = 12, heads: int = 5,
                 thought_steps: int = 8, action_count: int = 8, read_top_k: int = 4,
                 order_routing: bool = False, write_binding: bool = False,
                 event_binding: bool = False, event_binding_width: int = 64,
                 event_binding_write_pairs: bool = False,
                 event_binding_pairwise_transfer: tuple[str, str] | None = None,
                 latest_row_reader: bool = False,
                 latest_row_warmstart: str | None = None,
                 latest_row_answer_fusion: bool = False,
                 latest_row_answer_gate: bool = False,
                 latest_row_answer_entropy_threshold: float | None = None,
                 latest_row_answer_pairwise: bool = False,
                 latest_row_answer_event_binding: bool = False,
                 latest_row_answer_event_linear: bool = False,
                 latest_row_answer_factorized_router: bool = False,
                 latest_row_factorized_ood_threshold: float | None = None,
                 event_indexed_memory_reader: bool = False,
                 event_indexed_memory_reader_width: int = 64,
                 event_indexed_memory_reader_architecture: str = "relation"):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if workspace_slots < 2 or thought_steps < 1 or action_count < 5:
            raise ValueError("invalid neural-computer dimensions")
        self.hidden = hidden
        self.workspace_slots = workspace_slots
        self.thought_steps = thought_steps
        self.read_top_k = read_top_k
        self.order_routing = order_routing
        self.write_binding = write_binding
        self.event_binding = event_binding
        self.event_binding_write_pairs = event_binding_write_pairs
        self.event_binding_pairwise_transfer = event_binding_pairwise_transfer
        self.latest_row_reader = latest_row_reader
        self.latest_row_answer_fusion = latest_row_answer_fusion
        self.latest_row_answer_gate = latest_row_answer_gate
        self.latest_row_answer_entropy_threshold = latest_row_answer_entropy_threshold
        self.latest_row_answer_pairwise = latest_row_answer_pairwise
        self.latest_row_answer_event_binding = latest_row_answer_event_binding
        self.latest_row_answer_event_linear = latest_row_answer_event_linear
        self.latest_row_answer_factorized_router = (
            latest_row_answer_factorized_router)
        self.latest_row_factorized_ood_threshold = (
            latest_row_factorized_ood_threshold)
        self.event_indexed_memory_reader_enabled = event_indexed_memory_reader
        self.event_indexed_memory_reader_architecture = (
            event_indexed_memory_reader_architecture)
        self.vision = VisionEncoder(hidden)
        audio_hidden = max(16, hidden // 6)
        self.audio = AudioEncoder(audio_hidden)
        self.fusion = nn.Sequential(nn.Linear(hidden + audio_hidden, hidden),
                                    nn.LayerNorm(hidden), nn.GELU())
        self.initial_workspace = nn.Parameter(torch.randn(1, workspace_slots, hidden) * 0.02)
        self.read_query = nn.Linear(hidden, hidden)
        self.log_read_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.controller = nn.GRUCell(hidden * 3, hidden)
        self.workspace_key = nn.Linear(hidden, hidden)
        self.workspace_add = nn.Linear(hidden, hidden)
        self.workspace_erase = nn.Linear(hidden, hidden)
        self.workspace_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.thought_cell = nn.GRUCell(hidden * 2, hidden)
        if latest_row_reader:
            self.latest_row_project = nn.Linear(hidden, hidden)
            self.latest_row_gate = nn.Linear(hidden * 2, hidden)
            self.register_buffer("latest_row_mean", torch.zeros(1, hidden))
            self.register_buffer("latest_row_scale", torch.ones(1, hidden))
            # The optional channel starts as a small residual, not a replacement.
            nn.init.zeros_(self.latest_row_project.weight)
            nn.init.zeros_(self.latest_row_project.bias)
            nn.init.constant_(self.latest_row_gate.bias, -2.0)
            if latest_row_warmstart:
                warm = torch.load(latest_row_warmstart, map_location="cpu", weights_only=False)
                self.latest_row_project.load_state_dict(warm["projection"])
                self.latest_row_mean.copy_(warm["mean"])
                self.latest_row_scale.copy_(warm["scale"])
            if latest_row_answer_fusion:
                # Zero-start preserves the legacy answer path exactly.  This
                # tiny head is a diagnostic bridge, not a hidden game hook.
                self.latest_row_answer_fusion_head = nn.Linear(hidden * 2, action_count)
                nn.init.zeros_(self.latest_row_answer_fusion_head.weight)
                nn.init.zeros_(self.latest_row_answer_fusion_head.bias)
                if latest_row_answer_gate:
                    self.latest_row_answer_gate_head = nn.Linear(hidden * 2, 1)
                    nn.init.zeros_(self.latest_row_answer_gate_head.weight)
                    nn.init.constant_(self.latest_row_answer_gate_head.bias, 2.0)
            if latest_row_answer_pairwise:
                self.latest_row_answer_pairwise_head = nn.Sequential(
                    nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU(),
                    nn.Linear(hidden, action_count))
                nn.init.zeros_(self.latest_row_answer_pairwise_head[-1].weight)
                nn.init.zeros_(self.latest_row_answer_pairwise_head[-1].bias)
            if latest_row_answer_event_binding:
                if latest_row_answer_event_linear:
                    self.latest_row_answer_event_head = nn.Linear(
                        hidden * 7, action_count)
                    output = self.latest_row_answer_event_head
                else:
                    self.latest_row_answer_event_head = nn.Sequential(
                        nn.LayerNorm(hidden * 7),
                        nn.Linear(hidden * 7, hidden * 2), nn.GELU(),
                        nn.LayerNorm(hidden * 2),
                        nn.Linear(hidden * 2, hidden), nn.GELU(),
                        nn.Linear(hidden, action_count))
                    output = self.latest_row_answer_event_head[-1]
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
            if latest_row_answer_factorized_router:
                self.latest_row_factorized_router = FactorizedEventAnswerRouter(
                    hidden)
                self.latest_row_factorized_strength = nn.Parameter(
                    torch.zeros(()))
        if event_indexed_memory_reader:
            reader_class = {
                "relation": EventIndexedMemoryReader,
                "content-addressed": ContentAddressedEventMemoryReader,
            }.get(event_indexed_memory_reader_architecture)
            if reader_class is None:
                raise ValueError(
                    "unknown event-indexed memory reader architecture")
            self.event_indexed_memory_reader = reader_class(
                hidden, width=event_indexed_memory_reader_width,
                actions=action_count)
            # Exact no-op until a separately audited reader is installed.
            self.event_indexed_memory_reader_strength = nn.Parameter(
                torch.zeros(()))
        self.write_key = nn.Linear(hidden, hidden)
        self.write_value = nn.Linear(hidden, hidden)
        self.write_gate = nn.Linear(hidden, 1)
        nn.init.constant_(self.write_gate.bias, -2.0)
        if write_binding:
            self.write_binding_attention = nn.MultiheadAttention(
                hidden, heads, batch_first=True)
            self.write_binding_mlp = nn.Sequential(
                nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.write_binding_gate = nn.Linear(hidden * 2, hidden)
            nn.init.zeros_(self.write_binding_mlp[-1].weight)
            nn.init.zeros_(self.write_binding_mlp[-1].bias)
            nn.init.constant_(self.write_binding_gate.bias, -2.0)
        if event_binding:
            self.event_binding_module = EventSnapshotWriteBinder(
                hidden, width=event_binding_width,
                use_write_pairs=event_binding_write_pairs,
                pairwise_transfer_checkpoint=event_binding_pairwise_transfer)
        self.observation_head = nn.Linear(hidden, action_count)
        self.answer_head = nn.Linear(hidden, action_count)
        if order_routing:
            self.answer_route = nn.Sequential(
                nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.Tanh())
            self.answer_route_gate = nn.Linear(hidden * 3, hidden)
            nn.init.zeros_(self.answer_route[0].weight)
            nn.init.zeros_(self.answer_route[0].bias)
            nn.init.constant_(self.answer_route_gate.bias, -2.0)
        self.halt_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def _encode(self, frames: torch.Tensor, pcm: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        vision = self.vision(frames.reshape(batch * steps, *frames.shape[2:]))
        audio = self.audio(pcm.reshape(batch * steps, pcm.shape[-1]))
        return self.fusion(torch.cat((vision, audio), dim=-1)).reshape(batch, steps, -1)

    def sensory_summary(self, frames: torch.Tensor, pcm: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """Return a latent retrieval query derived only from the sensory stream."""
        events = self._encode(frames, pcm)
        weights = mask.to(events.dtype).unsqueeze(-1)
        return (events * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def retrieval_summary(self, frames: torch.Tensor, pcm: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """Build a recurrent sensory-only query before long-term retrieval.

        This mirrors the controller's perception loop with zero recalled content,
        so it cannot leak a long-term row into the decision about which row to load.
        """
        events = self._encode(frames, pcm)
        batch, steps, _ = events.shape
        workspace = self.initial_workspace.expand(batch, -1, -1).clone()
        controller = events.new_zeros(batch, self.hidden)
        query = events.new_zeros(batch, self.hidden)
        scale = self.hidden ** -0.5
        for index in range(steps):
            candidate_query = self.read_query(controller + events[:, index])
            workspace_summary = workspace.mean(dim=1)
            proposal = self.controller(
                torch.cat((events[:, index], torch.zeros_like(controller),
                           workspace_summary), dim=-1), controller)
            active = mask[:, index, None]
            query = torch.where(active, candidate_query, query)
            controller = torch.where(active, proposal, controller)
            weights = torch.softmax(
                torch.einsum("bd,bsd->bs", self.workspace_key(controller), workspace) * scale,
                dim=-1).unsqueeze(-1)
            erase = torch.sigmoid(self.workspace_erase(controller)).unsqueeze(1)
            addition = torch.tanh(self.workspace_add(controller)).unsqueeze(1)
            proposed_workspace = workspace * (1.0 - weights * erase) + weights * addition
            workspace = torch.where(active[:, None], proposed_workspace, workspace)
        return query

    def forward(self, frames: torch.Tensor, pcm: torch.Tensor, mask: torch.Tensor,
                memory: PersistentMemory,
                event_memory: PersistentMemory | None = None
                ) -> NeuralComputerOutput:
        if memory.width != self.hidden:
            raise ValueError("persistent memory width must match model hidden size")
        events = self._encode(frames, pcm)
        reader_memory = memory if event_memory is None else event_memory
        batch, steps, _ = events.shape
        workspace = self.initial_workspace.expand(batch, -1, -1).clone()
        controller = events.new_zeros(batch, self.hidden)
        observations, confidences = [], []
        scale = self.hidden ** -0.5
        for index in range(steps):
            query = self.read_query(controller + events[:, index])
            recalled, confidence = memory.read(
                query, self.read_top_k, self.log_read_scale.exp().clamp(max=100.0))
            workspace_summary = workspace.mean(dim=1)
            proposal = self.controller(
                torch.cat((events[:, index], recalled, workspace_summary), dim=-1), controller)
            active = mask[:, index, None]
            controller = torch.where(active, proposal, controller)
            weights = torch.softmax(
                torch.einsum("bd,bsd->bs", self.workspace_key(controller), workspace) * scale,
                dim=-1).unsqueeze(-1)
            erase = torch.sigmoid(self.workspace_erase(controller)).unsqueeze(1)
            addition = torch.tanh(self.workspace_add(controller)).unsqueeze(1)
            proposed_workspace = workspace * (1.0 - weights * erase) + weights * addition
            workspace = torch.where(active[:, None], proposed_workspace, workspace)
            observations.append(controller)
            confidences.append(confidence)

        observation_states = torch.stack(observations, dim=1)
        perceptual_state = controller
        pre_memory_event_states = events
        answer_logits, halt_logits, values = [], [], []
        for _ in range(self.thought_steps):
            attended, _ = self.workspace_attention(
                controller.unsqueeze(1), workspace, workspace, need_weights=False)
            query = self.read_query(controller)
            recalled, _ = memory.read(
                query, self.read_top_k, self.log_read_scale.exp().clamp(max=100.0))
            read_context = recalled
            if self.latest_row_reader and memory.count:
                # The audited relation is strongest in the writer key; keep
                # this channel separate from the consolidator/value mixture.
                latest_key = ((memory.keys[:, -1] - self.latest_row_mean.to(memory.keys)) /
                              self.latest_row_scale.to(memory.keys).clamp_min(1e-5))
                latest = self.latest_row_project(latest_key)
                latest_gate = torch.sigmoid(self.latest_row_gate(
                    torch.cat((controller, latest), dim=-1)))
                read_context = recalled + latest_gate * latest
            controller = self.thought_cell(
                torch.cat((attended.squeeze(1), read_context), dim=-1), controller)
            decision_state = controller
            if self.latest_row_reader and memory.count:
                # Preserve a direct path so the recurrent thought cell cannot
                # erase a newly learned memory feature before the answer head.
                decision_state = controller + latest_gate * latest
            recalled = decision_state
            if self.order_routing:
                route_input = torch.cat((controller, perceptual_state, recalled), dim=-1)
                decision_state = (controller + torch.sigmoid(self.answer_route_gate(route_input)) *
                                  self.answer_route(route_input))
            logits = self.answer_head(decision_state)
            if self.latest_row_reader and self.latest_row_answer_fusion and memory.count:
                fusion_input = torch.cat((controller, latest), dim=-1)
                fusion_logits = self.latest_row_answer_fusion_head(fusion_input)
                if self.latest_row_answer_gate:
                    fusion_logits = fusion_logits * torch.sigmoid(
                        self.latest_row_answer_gate_head(fusion_input))
                if self.latest_row_answer_entropy_threshold is not None:
                    probs = torch.softmax(fusion_logits, dim=-1)
                    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1, keepdim=True)
                    fusion_logits = fusion_logits * (
                        entropy > self.latest_row_answer_entropy_threshold).to(fusion_logits)
                logits = logits + fusion_logits
            if self.latest_row_reader and self.latest_row_answer_pairwise and memory.count:
                pairwise_input = torch.cat(
                    (perceptual_state, latest, perceptual_state * latest), dim=-1)
                logits = logits + self.latest_row_answer_pairwise_head(pairwise_input)
            if (self.latest_row_reader and self.latest_row_answer_event_binding and
                    memory.count):
                snapshots, valid = EventSnapshotWriteBinder.recent_events(
                    observation_states, mask)
                events = snapshots[:, :2] * valid[:, :2, None]
                first, second = events.unbind(dim=1)
                # Bind against the normalized raw writer key.  The generic
                # latest-row projection is useful for recurrent reading but
                # measurably attenuates the temporal rule at answer time.
                memory_feature = latest_key
                event_memory_input = torch.cat((
                    first, second, memory_feature,
                    first * second, (first - second).abs(),
                    first * memory_feature, second * memory_feature), dim=-1)
                logits = logits + self.latest_row_answer_event_head(event_memory_input)
            if (self.latest_row_reader and
                    self.latest_row_answer_factorized_router and memory.count):
                snapshots, valid = EventSnapshotWriteBinder.recent_events(
                    observation_states, mask)
                events = snapshots[:, :2] * valid[:, :2, None]
                reader_kwargs = {}
                if (self.event_indexed_memory_reader_enabled and
                        pre_memory_event_states.shape[1] >= 2 and
                        reader_memory.count):
                    reader_kwargs = {
                        "first_action_override":
                            self.event_indexed_memory_reader(
                                reader_memory.keys,
                                pre_memory_event_states[:, 0]),
                        "second_action_override":
                            self.event_indexed_memory_reader(
                                reader_memory.keys,
                                pre_memory_event_states[:, 1]),
                        "override_strength":
                            self.event_indexed_memory_reader_strength,
                    }
                routed = self.latest_row_factorized_router(
                    memory.keys[:, -1], events[:, 0], events[:, 1],
                    **reader_kwargs)
                activation = 1.0
                if self.latest_row_factorized_ood_threshold is not None:
                    activation = (
                        routed["density_distance"] <
                        self.latest_row_factorized_ood_threshold
                    ).to(logits)[:, None]
                logits = (
                    logits + self.latest_row_factorized_strength *
                    activation * routed["hard_action"])
            answer_logits.append(logits)
            halt_logits.append(self.halt_head(controller).squeeze(-1))
            values.append(self.value_head(controller).squeeze(-1))
        write_source = controller
        if self.write_binding:
            bound, _ = self.write_binding_attention(
                controller.unsqueeze(1), observation_states, observation_states,
                key_padding_mask=~mask, need_weights=False)
            bound = bound.squeeze(1)
            interaction = self.write_binding_mlp(
                torch.cat((controller, bound, controller * bound), dim=-1))
            gate = torch.sigmoid(self.write_binding_gate(
                torch.cat((controller, bound), dim=-1)))
            write_source = controller + gate * interaction
        event_binding_residual = torch.zeros_like(write_source)
        if self.event_binding:
            unbound_write_source = write_source
            write_source = self.event_binding_module(
                write_source, observation_states, mask)
            event_binding_residual = write_source - unbound_write_source
        return NeuralComputerOutput(
            self.observation_head(observation_states),
            torch.stack(answer_logits, dim=1),
            torch.stack(halt_logits, dim=1),
            torch.stack(values, dim=1),
            self.write_key(write_source), self.write_value(write_source),
            self.write_gate(write_source).squeeze(-1),
            torch.sigmoid(self.write_gate(write_source)).squeeze(-1),
            torch.stack(confidences, dim=1), workspace, recalled, write_source,
            event_binding_residual,
        )

    @torch.no_grad()
    def commit(self, memory: PersistentMemory, output: NeuralComputerOutput,
               *, threshold: float = 0.5) -> int:
        return memory.write(output.write_keys, output.write_values,
                            output.write_strengths, threshold=threshold)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
