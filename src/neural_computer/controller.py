"""Protocol-agnostic amodal cognitive controller.

This is the production controller boundary.  It consumes event tensors,
opaque feedback vectors, memory reads, and scalar outcomes.  It owns no camera,
actuator, action vocabulary, modality branch, or device protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .interface import (
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    EventTokenWindow,
    IntentEvent,
)
from .memory import ContentAddressedMemory, MemoryQuery, MemoryRead, MemoryWriteReceipt
from .policies import EventReliabilityPolicy


@dataclass(frozen=True)
class ControllerState:
    hidden: torch.Tensor
    workspace: torch.Tensor
    latest_event: torch.Tensor
    workspace_usage: torch.Tensor
    event_window: EventTokenWindow

    def detached(self) -> ControllerState:
        return ControllerState(
            hidden=self.hidden.detach(),
            workspace=self.workspace.detach(),
            latest_event=self.latest_event.detach(),
            workspace_usage=self.workspace_usage.detach(),
            event_window=EventTokenWindow(
                payload=self.event_window.payload.detach(),
                present=self.event_window.present.detach(),
                confidence=self.event_window.confidence.detach(),
                timestamp=self.event_window.timestamp.detach(),
                timestamp_present=self.event_window.timestamp_present.detach(),
                duration=self.event_window.duration.detach(),
                age=self.event_window.age.detach(),
                source_key=(
                    None
                    if self.event_window.source_key is None
                    else self.event_window.source_key.detach()
                ),
            ),
        )


@dataclass(frozen=True)
class ControllerOutput:
    intention: IntentEvent
    memory_key: torch.Tensor
    memory_value: torch.Tensor
    memory_write_strength: torch.Tensor
    workspace_read: torch.Tensor
    memory_query_key: torch.Tensor
    memory_read: MemoryRead | None
    memory_write_receipt: MemoryWriteReceipt | None
    event_attention: torch.Tensor
    event_reliability: torch.Tensor


class AmodalCognitiveController(nn.Module):
    """One controller for any number of learned event streams.

    ``feedback_width`` is the width of an opaque action/feedback encoder.  It
    is not the number of device actions and is deliberately independent of the
    number or type of output decoders.
    """

    def __init__(
        self,
        width: int = 96,
        workspace_slots: int = 8,
        intention_width: int = 24,
        feedback_width: int = 16,
        source_key_width: int = 0,
        event_window_capacity: int = 32,
        reliability_hidden: int = 32,
        memory_top_k: int = 1,
    ) -> None:
        super().__init__()
        if min(width, workspace_slots, intention_width, feedback_width) < 1:
            raise ValueError("controller dimensions must be positive")
        if min(event_window_capacity, reliability_hidden, memory_top_k) < 1:
            raise ValueError("event and memory capacities must be positive")
        if source_key_width < 0:
            raise ValueError("source_key_width cannot be negative")
        self.width = width
        self.workspace_slots = workspace_slots
        self.intention_width = intention_width
        self.feedback_width = feedback_width
        self.source_key_width = source_key_width
        self.event_window_capacity = event_window_capacity
        self.memory_top_k = memory_top_k

        feedback_features = feedback_width + 3
        feedback_hidden = max(8, width // 4)
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_features, feedback_hidden), nn.Tanh()
        )
        self.read_query = nn.Linear(width * 2, width)
        self.event_time_encoder = nn.Sequential(nn.Linear(4, width), nn.Tanh())
        self.event_relevance = nn.Linear(width * 2, 1)
        self.event_binding = nn.Linear(width * 2, width)
        self.memory_query = nn.Linear(width * 2, width)
        self.controller = nn.GRUCell(width * 3 + feedback_hidden, width)
        self.write_gate = nn.Linear(width * 3, 1)
        self.write_query = nn.Linear(width * 3, width)
        self.write_value = nn.Sequential(nn.Linear(width * 3, width), nn.Tanh())
        self.intention = nn.Sequential(
            nn.LayerNorm(width * 3), nn.Linear(width * 3, intention_width), nn.Tanh()
        )
        self.memory_key = nn.Linear(width * 2, width)
        self.memory_value = nn.Linear(width * 2, width)
        self.memory_write = nn.Linear(width * 2, 1)
        self.reliability_policy = EventReliabilityPolicy(
            width,
            source_key_width=source_key_width,
            hidden=reliability_hidden,
        )

    def configuration(self) -> dict[str, int | str]:
        """Return only constructor data needed to rebuild this component."""
        return {
            "schema": "neural-computer.controller.v1",
            "width": self.width,
            "workspace_slots": self.workspace_slots,
            "intention_width": self.intention_width,
            "feedback_width": self.feedback_width,
            "source_key_width": self.source_key_width,
            "event_window_capacity": self.event_window_capacity,
            "reliability_hidden": self.reliability_policy.network[0].out_features,
            "memory_top_k": self.memory_top_k,
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return ControllerState(
            hidden=torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            workspace=torch.zeros(
                batch_size, self.workspace_slots, self.width, device=device, dtype=dtype
            ),
            latest_event=torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            workspace_usage=torch.zeros(
                batch_size, self.workspace_slots, device=device, dtype=dtype
            ),
            event_window=EventTokenWindow.empty(
                batch_size,
                self.event_window_capacity,
                self.width,
                source_key_width=self.source_key_width,
                device=device,
                dtype=dtype,
            ),
        )

    def _collection(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent] | torch.Tensor,
    ) -> AmodalEventCollection:
        if isinstance(events, AmodalEventCollection):
            return events.validate(width=self.width)
        if isinstance(events, torch.Tensor):
            if events.ndim != 2 or events.shape[1] != self.width:
                raise ValueError(f"event must have shape [batch, {self.width}]")
            return AmodalEventCollection.from_events([AmodalEvent(events)])
        return AmodalEventCollection.from_events(events, width=self.width)

    def _append_event_window(
        self,
        previous: EventTokenWindow,
        collection: AmodalEventCollection,
        *,
        elapsed: torch.Tensor,
    ) -> EventTokenWindow:
        collection.validate(width=self.width)
        previous.validate(width=self.width, source_key_width=self.source_key_width)
        batch = collection.payload.shape[0]
        if previous.payload.shape[0] != batch:
            raise ValueError("event window batch does not match collection")
        if elapsed.shape not in ((batch,), (batch, 1)):
            raise ValueError("elapsed must have shape [batch] or [batch, 1]")
        elapsed = elapsed.reshape(batch).to(collection.payload)
        if torch.any(elapsed < 0):
            raise ValueError("elapsed cannot be negative")

        window = EventTokenWindow.empty(
            batch,
            self.event_window_capacity,
            self.width,
            source_key_width=self.source_key_width,
            device=collection.payload.device,
            dtype=collection.payload.dtype,
        )
        for row in range(batch):
            old_indices = torch.nonzero(previous.present[row], as_tuple=False).reshape(-1)
            new_indices = torch.nonzero(collection.present[row], as_tuple=False).reshape(-1)
            old_payload = previous.payload[row, old_indices]
            new_payload = collection.payload[row, new_indices]
            payload = torch.cat([old_payload, new_payload], dim=0)[-self.event_window_capacity :]
            if payload.numel() == 0:
                continue
            old_confidence = previous.confidence[row, old_indices] + 0.0
            new_confidence = collection.confidence[row, new_indices]
            confidence = torch.cat([old_confidence, new_confidence], dim=0)[-self.event_window_capacity :]
            age = torch.cat(
                [previous.age[row, old_indices] + elapsed[row], torch.zeros_like(new_confidence)]
            )[-self.event_window_capacity :]
            if collection.timestamp is None:
                new_timestamp = torch.zeros_like(new_confidence)
                new_timestamp_present = torch.zeros_like(new_indices, dtype=torch.bool)
            else:
                new_timestamp = collection.timestamp[row, new_indices]
                new_timestamp_present = torch.ones_like(new_indices, dtype=torch.bool)
            timestamp = torch.cat(
                [previous.timestamp[row, old_indices], new_timestamp]
            )[-self.event_window_capacity :]
            timestamp_present = torch.cat(
                [previous.timestamp_present[row, old_indices], new_timestamp_present]
            )[-self.event_window_capacity :]
            if collection.duration is None:
                new_duration = torch.zeros_like(new_confidence)
            else:
                new_duration = collection.duration[row, new_indices]
            duration = torch.cat(
                [previous.duration[row, old_indices], new_duration]
            )[-self.event_window_capacity :]
            keep = payload.shape[0]
            window.payload[row, :keep] = payload
            window.present[row, :keep] = True
            window.confidence[row, :keep] = confidence
            window.timestamp[row, :keep] = timestamp
            window.timestamp_present[row, :keep] = timestamp_present
            window.duration[row, :keep] = duration
            window.age[row, :keep] = age
            if self.source_key_width:
                if new_indices.numel() and collection.source_key is None:
                    raise ValueError("source_key is required by this controller")
                assert previous.source_key is not None and window.source_key is not None
                source_parts = [previous.source_key[row, old_indices]]
                if new_indices.numel():
                    assert collection.source_key is not None
                    source_parts.append(collection.source_key[row, new_indices])
                source_key = torch.cat(source_parts, dim=0)[-self.event_window_capacity :]
                window.source_key[row, :keep] = source_key
        return window.validate(width=self.width, source_key_width=self.source_key_width)

    def _bind_events(
        self,
        window: EventTokenWindow,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        window.validate(width=self.width, source_key_width=self.source_key_width)
        payload = window.payload
        batch, events, _ = payload.shape
        if events == 0:
            zeros = payload.new_zeros(batch, self.width)
            return (
                zeros,
                payload.new_zeros(batch, 1),
                None,
                payload.new_zeros(batch, events),
                payload.new_zeros(batch, events),
            )

        hidden_tokens = hidden.unsqueeze(1).expand(-1, events, -1)
        transport = torch.stack(
            [
                window.age,
                window.duration,
                window.timestamp_present.to(payload.dtype),
                window.confidence,
            ],
            dim=-1,
        )
        temporal_context = self.event_time_encoder(transport)
        temporal_payload = payload + temporal_context
        binding = torch.tanh(
            self.event_binding(torch.cat([temporal_payload, hidden_tokens], dim=-1))
        )
        scores = self.event_relevance(
            torch.cat([temporal_payload, hidden_tokens], dim=-1)
        ).squeeze(-1)
        trust = self.reliability_policy(window)
        confidence = (window.confidence * trust).to(payload.dtype).clamp_min(1e-8)
        scores = scores + confidence.log()
        scores = scores.masked_fill(~window.present, -torch.inf)
        valid = window.present.any(dim=1, keepdim=True)
        safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=1)
        weights = torch.where(window.present, weights, torch.zeros_like(weights))
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        bound = torch.einsum("be,bew->bw", weights, binding)
        timestamp_weights = weights * window.timestamp_present.to(weights.dtype)
        timestamp = (
            torch.einsum("be,be->b", timestamp_weights, window.timestamp)
            / timestamp_weights.sum(dim=1).clamp_min(1e-8)
            if window.timestamp_present.any()
            else None
        )
        combined_confidence = torch.einsum("be,be->b", weights, confidence)
        return bound, combined_confidence.unsqueeze(-1), timestamp, weights, trust

    def step(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent] | torch.Tensor,
        state: ControllerState,
        feedback: ControllerFeedback,
        memory: ContentAddressedMemory | None = None,
        *,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
    ) -> tuple[ControllerOutput, ControllerState]:
        collection = self._collection(events)
        batch = collection.payload.shape[0]
        feedback.validate(batch=batch, action_width=self.feedback_width)
        reward = feedback.reward.reshape(batch, 1).to(collection.payload.dtype)
        propensity = feedback.propensity.reshape(batch, 1).to(collection.payload.dtype)
        has_feedback = feedback.has_feedback.reshape(batch, 1).to(collection.payload.dtype)
        action = feedback.action.to(collection.payload.dtype)
        elapsed_tensor = torch.as_tensor(
            elapsed, device=collection.payload.device, dtype=collection.payload.dtype
        )
        if elapsed_tensor.ndim == 0:
            elapsed_tensor = elapsed_tensor.expand(batch)
        window = self._append_event_window(
            state.event_window, collection, elapsed=elapsed_tensor
        )
        event, confidence, timestamp, attention, reliability = self._bind_events(
            window, state.hidden
        )
        query = torch.nn.functional.normalize(
            self.read_query(torch.cat([event, state.hidden], dim=-1)), dim=-1
        )
        slots = torch.nn.functional.normalize(state.workspace, dim=-1)
        read_scores = torch.einsum("bw,bsw->bs", query, slots)
        read_weights = torch.softmax(read_scores, dim=-1)
        workspace_read = torch.einsum("bs,bsw->bw", read_weights, state.workspace)
        if disable_workspace:
            workspace_read = torch.zeros_like(workspace_read)

        feedback_vector = torch.cat([action, reward, propensity, has_feedback], dim=-1)
        feedback_embedding = self.feedback_encoder(feedback_vector)
        memory_query_key = self.memory_query(torch.cat([event, state.hidden], dim=-1))
        memory_read = memory.read(MemoryQuery(memory_query_key, self.memory_top_k)) if memory else None
        memory_context = (
            memory_read.value
            if memory_read is not None
            else collection.payload.new_zeros(batch, self.width)
        )
        controller_input = torch.cat(
            [event, workspace_read, memory_context, feedback_embedding], dim=-1
        )
        hidden = self.controller(controller_input, state.hidden)

        write_context = torch.cat([event, hidden, workspace_read], dim=-1)
        write_query = torch.nn.functional.normalize(self.write_query(write_context), dim=-1)
        write_scores = torch.einsum("bw,bsw->bs", write_query, slots)
        write_weights = torch.softmax(write_scores, dim=-1)
        write_gate = torch.sigmoid(self.write_gate(write_context))
        candidate = self.write_value(write_context).unsqueeze(1)
        update = write_gate.unsqueeze(-1) * write_weights.unsqueeze(-1)
        workspace = state.workspace * (1.0 - update) + candidate * update
        usage = 0.98 * state.workspace_usage + 0.02 * (read_weights + write_weights) * 0.5
        if disable_workspace:
            workspace = torch.zeros_like(workspace)
            usage = torch.zeros_like(usage)

        combined = torch.cat([hidden, workspace_read, event], dim=-1)
        intention = self.intention(combined)
        intent_event = IntentEvent(
            payload=intention,
            timestamp=timestamp,
            confidence=confidence,
        ).validate(width=self.intention_width)
        memory_context = torch.cat([hidden, workspace_read], dim=-1)
        output = ControllerOutput(
            intention=intent_event,
            memory_key=self.memory_key(memory_context),
            memory_value=self.memory_value(memory_context),
            memory_write_strength=torch.sigmoid(self.memory_write(memory_context)).squeeze(-1),
            workspace_read=workspace_read,
            memory_query_key=memory_query_key,
            memory_read=memory_read,
            memory_write_receipt=None,
            event_attention=attention,
            event_reliability=reliability,
        )
        memory_write_receipt = (
            memory.write(
                output.memory_key,
                output.memory_value,
                output.memory_write_strength,
                timestamp=timestamp,
            )
            if memory is not None
            else None
        )
        output = ControllerOutput(
            intention=output.intention,
            memory_key=output.memory_key,
            memory_value=output.memory_value,
            memory_write_strength=output.memory_write_strength,
            workspace_read=output.workspace_read,
            memory_query_key=output.memory_query_key,
            memory_read=output.memory_read,
            memory_write_receipt=memory_write_receipt,
            event_attention=output.event_attention,
            event_reliability=output.event_reliability,
        )
        if disable_workspace:
            workspace = torch.zeros_like(workspace)
        next_state = ControllerState(
            hidden=hidden,
            workspace=workspace,
            latest_event=event,
            workspace_usage=usage,
            event_window=window,
        )
        return output, next_state
