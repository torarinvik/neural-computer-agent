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
from .memory import MemoryBackend, MemoryQuery, MemoryRead, MemoryWriteReceipt
from .policies import EventReliabilityPolicy

EXECUTION_STATES = ("wait", "think", "commit")
EXECUTION_TRANSPORT_FEATURES = 3


@dataclass(frozen=True)
class ControllerState:
    hidden: torch.Tensor
    workspace: torch.Tensor
    latest_event: torch.Tensor
    workspace_usage: torch.Tensor
    event_window: EventTokenWindow
    source_trust: torch.Tensor

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
            source_trust=self.source_trust.detach(),
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
    execution_logits: torch.Tensor
    memory_write_log_probability: torch.Tensor | None = None
    memory_write_sample: torch.Tensor | None = None


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
        execution_hidden: int = 16,
        memory_value_feedback: bool = True,
        stable_memory_address: bool = True,
        memory_address_residual: bool = True,
    ) -> None:
        super().__init__()
        if min(width, workspace_slots, intention_width, feedback_width) < 1:
            raise ValueError("controller dimensions must be positive")
        if min(event_window_capacity, reliability_hidden, memory_top_k, execution_hidden) < 1:
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
        self.execution_hidden = execution_hidden
        self.memory_value_feedback_enabled = memory_value_feedback
        self.stable_memory_address = stable_memory_address
        self.memory_address_residual = memory_address_residual
        self.source_credit_decay = 0.9
        self.source_trust_binding_scale = 0.25

        feedback_features = feedback_width + 3
        feedback_hidden = max(8, width // 4)
        self.feedback_hidden = feedback_hidden
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_features, feedback_hidden), nn.Tanh()
        )
        self.read_query = nn.Linear(width * 2, width)
        self.event_time_encoder = nn.Sequential(nn.Linear(4, width), nn.Tanh())
        self.event_relevance = nn.Linear(width * 2, 1)
        self.event_binding = nn.Linear(width * 2, width)
        # Generic pairwise event binding lets a token condition its relevance
        # on other present tokens without naming a modality or task role.  The
        # output starts at zero so existing checkpoints retain their behavior
        # until this mechanism is deliberately trained.
        with torch.random.fork_rng(devices=[]):
            self.event_pair_query = nn.Linear(width, width, bias=False)
            self.event_pair_key = nn.Linear(width, width, bias=False)
            self.event_pair_value = nn.Linear(width, width, bias=False)
            self.event_pair_output = nn.Linear(width, width)
        self.event_pair_relevance = nn.Linear(width, 1)
        self.event_address_relevance = nn.Linear(width, 1)
        nn.init.zeros_(self.event_address_relevance.weight)
        nn.init.zeros_(self.event_address_relevance.bias)
        nn.init.zeros_(self.event_pair_output.weight)
        nn.init.zeros_(self.event_pair_output.bias)
        nn.init.zeros_(self.event_pair_relevance.weight)
        nn.init.zeros_(self.event_pair_relevance.bias)
        # Let prior opaque action/outcome feedback condition the next event
        # binding decision through a generic, protocol-agnostic path.  It is
        # zero-initialized so older checkpoints retain their behavior.
        self.event_feedback_relevance = nn.Bilinear(
            width + source_key_width, feedback_hidden, 1
        )
        nn.init.zeros_(self.event_feedback_relevance.weight)
        nn.init.zeros_(self.event_feedback_relevance.bias)
        self.event_feedback_source_relevance = nn.Bilinear(
            width + source_key_width,
            feedback_hidden + source_key_width,
            1,
        )
        nn.init.zeros_(self.event_feedback_source_relevance.weight)
        nn.init.zeros_(self.event_feedback_source_relevance.bias)
        self.source_credit_hidden = max(8, width // 4)
        self.source_credit_policy = (
            nn.Sequential(
                nn.Linear(
                    width + source_key_width + feedback_hidden,
                    self.source_credit_hidden,
                ),
                nn.Tanh(),
                nn.Linear(self.source_credit_hidden, source_key_width),
            )
            if source_key_width
            else None
        )
        if self.source_credit_policy is not None:
            # A neutral source-credit prior is essential: random output bias
            # otherwise becomes an unconditional preference for one source
            # before any outcome has been observed. Keep the feature-dependent
            # weights random so the head retains a useful gradient path.
            nn.init.zeros_(self.source_credit_policy[-1].bias)
        # The address is shared by writes and reads and is formed from an
        # event-window representation that does not include recurrent state or
        # feedback. This keeps an address stable across the state reset used
        # by memory recall while leaving values free to encode outcomes.
        self.memory_address = nn.Linear(width, width)
        self.controller = nn.GRUCell(width * 3 + feedback_hidden, width)
        self.write_gate = nn.Linear(width * 3, 1)
        self.write_query = nn.Linear(width * 3, width)
        self.write_value = nn.Sequential(nn.Linear(width * 3, width), nn.Tanh())
        self.intention = nn.Sequential(
            nn.LayerNorm(width * 3), nn.Linear(width * 3, intention_width), nn.Tanh()
        )
        self.memory_value = nn.Linear(width * 2, width)
        self.memory_value_feedback = (
            nn.Linear(feedback_hidden, width) if memory_value_feedback else None
        )
        if self.memory_value_feedback is not None:
            nn.init.zeros_(self.memory_value_feedback.weight)
            nn.init.zeros_(self.memory_value_feedback.bias)
        # Memory retention is a utility decision, not merely a value
        # projection. Give it a nonlinear, protocol-agnostic policy over the
        # current event, recurrent state, workspace context, and address.
        self.memory_write_hidden = max(8, width // 4)
        self.memory_write = nn.Linear(width * 2, 1)
        self.memory_write_policy = nn.Sequential(
            nn.Linear(width * 4, self.memory_write_hidden),
            nn.GELU(),
            nn.Linear(self.memory_write_hidden, 1),
        )
        # Keep the initial mean write propensity neutral while allowing
        # different event contexts to receive distinct, very small gradients
        # from the first outcome-only episode. A fully zero output layer would
        # make every context identical until its scalar bias moved.
        nn.init.normal_(self.memory_write_policy[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.memory_write_policy[-1].bias)
        self.execution_policy = nn.Sequential(
            nn.Linear(width * 3 + 1, execution_hidden),
            nn.GELU(),
            nn.Linear(execution_hidden, len(EXECUTION_STATES)),
        )
        self.execution_transport_policy = nn.Sequential(
            nn.Linear(EXECUTION_TRANSPORT_FEATURES, execution_hidden),
            nn.GELU(),
            nn.Linear(execution_hidden, len(EXECUTION_STATES)),
        )
        # Preserve the old one-pass runtime as the compatibility prior.  The
        # policy remains trainable, but an untrained controller starts by
        # committing rather than silently changing inference semantics.
        nn.init.zeros_(self.execution_policy[-1].weight)
        nn.init.zeros_(self.execution_policy[-1].bias)
        nn.init.zeros_(self.execution_transport_policy[-1].weight)
        nn.init.zeros_(self.execution_transport_policy[-1].bias)
        with torch.no_grad():
            self.execution_policy[-1].bias[EXECUTION_STATES.index("commit")] = 1.0
            self.execution_transport_policy[-1].bias[EXECUTION_STATES.index("commit")] = 1.0
        self.reliability_policy = EventReliabilityPolicy(
            width,
            source_key_width=source_key_width,
            hidden=reliability_hidden,
        )
        # Construct the optional timeout residual after the inherited
        # controller modules so its initialization cannot perturb their RNG
        # sequence. It is zero at construction and gated by event age below.
        with torch.random.fork_rng(devices=[]):
            self.execution_timeout_policy = nn.Sequential(
                nn.Linear(1, execution_hidden),
                nn.GELU(),
                nn.Linear(execution_hidden, len(EXECUTION_STATES)),
            )
        nn.init.zeros_(self.execution_timeout_policy[-1].weight)
        nn.init.zeros_(self.execution_timeout_policy[-1].bias)

    def configuration(self) -> dict[str, object]:
        """Return only constructor data needed to rebuild this component."""
        return {
            "schema": "neural-computer.controller.v27",
            "width": self.width,
            "workspace_slots": self.workspace_slots,
            "intention_width": self.intention_width,
            "feedback_width": self.feedback_width,
            "source_key_width": self.source_key_width,
            "event_window_capacity": self.event_window_capacity,
            "reliability_hidden": self.reliability_policy.network[0].out_features,
            "memory_top_k": self.memory_top_k,
            "execution_hidden": self.execution_hidden,
            "execution_states": EXECUTION_STATES,
            "execution_transport_policy": True,
            "execution_transport_features": EXECUTION_TRANSPORT_FEATURES,
            "execution_timeout_policy": True,
            "event_pair_attention": True,
            "event_pair_relevance": True,
            "event_address_relevance": True,
            "memory_address": (
                "latest_event_payload_residual_v2"
                if self.stable_memory_address and self.memory_address_residual
                else "latest_event_payload_v1"
                if self.stable_memory_address
                else "latest_event_token_v1"
            ),
            "memory_write_policy": "event_state_workspace_address_v1",
            "memory_write_sampling": "bernoulli_straight_through_v1",
            "memory_write_event_window": "latest_pair_context_match_max_v3",
            "memory_write_event_match": "latest_prior_cosine_and_max_v2",
            "memory_value_feedback": (
                "feedback_residual_v1"
                if self.memory_value_feedback_enabled
                else "none_v1"
            ),
            "memory_write_hidden": self.memory_write_hidden,
            "event_feedback_relevance": True,
            "event_feedback_source_relevance": True,
            "source_credit_state": True,
            "source_credit_decay": self.source_credit_decay,
            "source_credit_policy": self.source_credit_policy is not None,
            "source_credit_hidden": self.source_credit_hidden,
            "source_credit_projection": "vector_to_trust_space",
            "source_trust_binding": bool(self.source_key_width),
            "source_trust_binding_scale": self.source_trust_binding_scale,
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
            source_trust=torch.zeros(
                batch_size,
                self.source_key_width,
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
        feedback_embedding: torch.Tensor | None = None,
        feedback_source_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        window.validate(width=self.width, source_key_width=self.source_key_width)
        payload = window.payload
        batch, events, _ = payload.shape
        if feedback_embedding is None:
            feedback_embedding = payload.new_zeros(batch, self.feedback_hidden)
        if feedback_embedding.shape != (batch, self.feedback_hidden):
            raise ValueError("feedback embedding has the wrong shape")
        if feedback_source_context is None:
            feedback_source_context = payload.new_zeros(
                batch, self.feedback_hidden + self.source_key_width
            )
        if feedback_source_context.shape != (
            batch,
            self.feedback_hidden + self.source_key_width,
        ):
            raise ValueError("feedback source context has the wrong shape")
        if events == 0:
            zeros = payload.new_zeros(batch, self.width)
            return (
                zeros,
                payload.new_zeros(batch, 1),
                None,
                payload.new_zeros(batch, events),
                payload.new_zeros(batch, events),
                zeros,
                zeros,
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
        # A durable content address must be invariant between the write and
        # recall appearances of the same learned event. Event age, duration,
        # timestamp presence, and confidence are transport context and can
        # legitimately differ across those appearances, so they must not
        # perturb the address in the stable-address path.
        address_payload = payload if self.stable_memory_address else temporal_payload
        pair_query = self.event_pair_query(temporal_payload)
        pair_key = self.event_pair_key(temporal_payload)
        pair_value = self.event_pair_value(temporal_payload)
        # Keep the full token-token score matrix so each event can condition its
        # relevance and binding on every other present event.
        pair_scores = torch.einsum("bew,bfw->bef", pair_query, pair_key) / (self.width**0.5)
        pair_scores = pair_scores.masked_fill(~window.present[:, None, :], -torch.inf)
        valid = window.present.any(dim=1, keepdim=True)
        safe_pair_scores = torch.where(valid[:, :, None], pair_scores, torch.zeros_like(pair_scores))
        pair_weights = torch.softmax(safe_pair_scores, dim=-1)
        pair_weights = torch.where(window.present[:, None, :], pair_weights, torch.zeros_like(pair_weights))
        pair_context = torch.einsum("bef,bfw->bew", pair_weights, pair_value)
        latest_index = window.present.to(torch.long).sum(dim=1).clamp_min(1) - 1
        latest_pair_context = pair_context.gather(
            1,
            latest_index.view(batch, 1, 1).expand(-1, 1, self.width),
        ).squeeze(1)
        temporal_payload = temporal_payload + torch.tanh(self.event_pair_output(pair_context))
        positions = torch.arange(events, device=payload.device).view(1, events)
        prior_present = window.present & (positions != latest_index.view(batch, 1))
        prior_context = (
            (temporal_payload * prior_present.unsqueeze(-1).to(payload.dtype)).sum(dim=1)
            / prior_present.sum(dim=1, keepdim=True).clamp_min(1).to(payload.dtype)
        )
        latest_payload = temporal_payload.gather(
            1,
            latest_index.view(batch, 1, 1).expand(-1, 1, self.width),
        ).squeeze(1)
        latest_prior_match = latest_payload * prior_context
        latest_prior_similarity = (
            torch.nn.functional.normalize(latest_payload, dim=-1)
            * torch.nn.functional.normalize(prior_context, dim=-1)
        ).sum(dim=-1, keepdim=True)
        latest_normalized = torch.nn.functional.normalize(latest_payload, dim=-1)
        prior_normalized = torch.nn.functional.normalize(temporal_payload, dim=-1)
        latest_prior_scores = torch.einsum(
            "bw,bew->be", latest_normalized, prior_normalized
        )
        latest_prior_scores = latest_prior_scores.masked_fill(
            ~prior_present, -torch.inf
        )
        latest_prior_max_similarity = latest_prior_scores.max(dim=1).values
        latest_prior_max_similarity = torch.where(
            prior_present.any(dim=1),
            latest_prior_max_similarity,
            torch.zeros_like(latest_prior_max_similarity),
        ).unsqueeze(-1)
        # Content addresses must remain stable when an irrelevant prior token
        # is added to the event window. The latest learned event token is the
        # generic temporal binding unit; timing features remain part of the
        # token, while prior context continues to influence reasoning and the
        # write utility policy below.
        latest_address_payload = address_payload.gather(
            1,
            latest_index.view(batch, 1, 1).expand(-1, 1, self.width),
        ).squeeze(1)
        address_gain = 1.0 + 0.1 * torch.tanh(
            self.event_address_relevance(latest_address_payload)
        )
        address_event = latest_address_payload * address_gain
        binding = torch.tanh(
            self.event_binding(torch.cat([temporal_payload, hidden_tokens], dim=-1))
        )
        scores = self.event_relevance(
            torch.cat([temporal_payload, hidden_tokens], dim=-1)
        ).squeeze(-1) + self.event_pair_relevance(torch.tanh(pair_context)).squeeze(-1)
        feedback_tokens = feedback_embedding.unsqueeze(1).expand(-1, events, -1)
        feedback_payload = temporal_payload
        if self.source_key_width:
            if window.source_key is None:
                raise ValueError("source_key is required for feedback-conditioned binding")
            feedback_payload = torch.cat([feedback_payload, window.source_key], dim=-1)
        scores = scores + self.event_feedback_relevance(
            feedback_payload, feedback_tokens
        ).squeeze(-1)
        feedback_source_tokens = feedback_source_context.unsqueeze(1).expand(
            -1, events, -1
        )
        scores = scores + self.event_feedback_source_relevance(
            feedback_payload, feedback_source_tokens
        ).squeeze(-1)
        if self.source_key_width:
            source_trust = feedback_source_context[:, -self.source_key_width :]
            scores = scores + self.source_trust_binding_scale * torch.einsum(
                "bek,bk->be",
                window.source_key,
                torch.tanh(source_trust),
            )
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
        # The reduced event remains the controller's general semantic input,
        # while the write policy also receives a current-token-conditioned
        # view of the retained event window. This lets a generic policy learn
        # whether the current event agrees with prior context without adding a
        # modality-specific retention branch.
        write_event_context = (
            bound
            + torch.tanh(latest_pair_context)
            + latest_prior_match
            + latest_prior_similarity.expand_as(latest_payload)
            + latest_prior_max_similarity.expand_as(latest_payload)
        )
        return (
            bound,
            combined_confidence.unsqueeze(-1),
            timestamp,
            weights,
            trust,
            address_event,
            write_event_context,
        )

    def step(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent] | torch.Tensor,
        state: ControllerState,
        feedback: ControllerFeedback,
        memory: MemoryBackend | None = None,
        *,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
        sample_memory_writes: bool = False,
        memory_write_override: torch.Tensor | None = None,
        memory_write_uniform: torch.Tensor | None = None,
    ) -> tuple[ControllerOutput, ControllerState]:
        if memory is not None and not isinstance(memory, MemoryBackend):
            raise TypeError("memory must implement the MemoryBackend contract")
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
        feedback_vector = torch.cat([action, reward, propensity, has_feedback], dim=-1)
        feedback_embedding = self.feedback_encoder(feedback_vector)
        if self.source_key_width:
            if state.event_window.source_key is None:
                raise ValueError("source_key is required for source credit binding")
            source_keys = state.event_window.source_key
            source_credit_input = torch.cat(
                [
                    state.event_window.payload,
                    source_keys,
                    feedback_embedding.unsqueeze(1).expand(
                        -1, state.event_window.payload.shape[1], -1
                    ),
                ],
                dim=-1,
            )
            present = state.event_window.present.to(source_keys.dtype)
            source_credit_tokens = self.source_credit_policy(source_credit_input)
            source_keys = torch.nn.functional.normalize(source_keys, dim=-1)
            source_credit_delta = (
                source_credit_tokens * present.unsqueeze(-1) * source_keys
            ).sum(dim=1) / present.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            source_trust = (
                collection.payload.new_zeros(batch, 0)
            )
        if self.source_key_width:
            source_trust = (
                self.source_credit_decay * state.source_trust
                + has_feedback * source_credit_delta
            )
        feedback_source_context = torch.cat([feedback_embedding, source_trust], dim=-1)
        (
            event,
            confidence,
            timestamp,
            attention,
            reliability,
            address_event,
            write_event_context,
        ) = self._bind_events(
            window, state.hidden, feedback_embedding, feedback_source_context
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

        memory_query_key = self.memory_address(address_event)
        if self.memory_address_residual:
            # Preserve a direct event-identity path alongside the learned
            # address projection. This keeps exact repeated learned events
            # separable when nearby distractors compete for bounded memory,
            # while the projection remains free to learn cross-adapter
            # invariances. The path is protocol- and modality-agnostic.
            memory_query_key = memory_query_key + address_event
        memory_read = (
            memory.read(
                MemoryQuery(
                    memory_query_key,
                    self.memory_top_k,
                    scope=memory_scope,
                )
            )
            if memory
            else None
        )
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
        # Occupancy is transport metadata, not a modality or task label.  It
        # lets the execution policy distinguish a complete event window from
        # a partial one without adding a special reasoning branch.
        present = window.present.to(collection.payload.dtype)
        event_density = present.sum(dim=1, keepdim=True) / float(
            self.event_window_capacity
        )
        mean_event_age = (window.age * present).sum(dim=1, keepdim=True) / present.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        execution_features = torch.cat([combined, confidence * event_density], dim=-1)
        # Generic transport features provide a stable first-order cue. Latent
        # content may refine the choice, but cannot overwhelm the
        # complete/partial-window signal because of nuisance payload variation.
        content_logits = torch.tanh(self.execution_policy(execution_features))
        transport_features = torch.cat(
            [
                event_density,
                confidence,
                event_density * confidence,
            ],
            dim=-1,
        )
        execution_logits = (
            self.execution_transport_policy(transport_features)
            + 0.25 * content_logits
            + mean_event_age * self.execution_timeout_policy(mean_event_age)
        )
        intent_event = IntentEvent(
            payload=intention,
            timestamp=timestamp,
            confidence=confidence,
        ).validate(width=self.intention_width)
        memory_value_context = torch.cat([hidden, workspace_read], dim=-1)
        memory_write_context = torch.cat(
            [write_event_context, hidden, workspace_read, memory_query_key], dim=-1
        )
        memory_write_strength = torch.sigmoid(
            self.memory_write_policy(memory_write_context)
        ).squeeze(-1)
        memory_write_backend_strength = memory_write_strength
        memory_write_log_probability: torch.Tensor | None = None
        memory_write_sample: torch.Tensor | None = None
        if memory_write_uniform is not None:
            uniform = memory_write_uniform.reshape(-1).to(
                device=memory_write_strength.device,
                dtype=memory_write_strength.dtype,
            )
            if uniform.shape != memory_write_strength.shape:
                raise ValueError("memory write uniform has the wrong shape")
            if not bool(torch.isfinite(uniform).all()) or bool(
                torch.any((uniform < 0.0) | (uniform > 1.0))
            ):
                raise ValueError("memory write uniform must lie in [0, 1]")
            probability = memory_write_strength.clamp(1e-6, 1.0 - 1e-6)
            memory_write_sample = (uniform < probability).to(
                memory_write_strength.dtype
            )
            memory_write_log_probability = (
                memory_write_sample * probability.log()
                + (1.0 - memory_write_sample) * (1.0 - probability).log()
            )
            memory_write_backend_strength = (
                memory_write_sample
                + memory_write_strength
                - memory_write_strength.detach()
            )
        elif memory_write_override is not None:
            override = memory_write_override.reshape(-1).to(
                device=memory_write_strength.device,
                dtype=memory_write_strength.dtype,
            )
            if override.shape != memory_write_strength.shape:
                raise ValueError("memory write override has the wrong shape")
            if not bool(torch.isfinite(override).all()) or bool(
                torch.any((override < 0.0) | (override > 1.0))
            ):
                raise ValueError("memory write override must lie in [0, 1]")
            if not bool(torch.all((override == 0.0) | (override == 1.0))):
                raise ValueError("memory write override must be binary")
            probability = memory_write_strength.clamp(1e-6, 1.0 - 1e-6)
            memory_write_sample = override
            memory_write_log_probability = (
                memory_write_sample * probability.log()
                + (1.0 - memory_write_sample) * (1.0 - probability).log()
            )
            memory_write_backend_strength = (
                memory_write_sample
                + memory_write_strength
                - memory_write_strength.detach()
            )
        elif sample_memory_writes:
            probability = memory_write_strength.clamp(1e-6, 1.0 - 1e-6)
            memory_write_sample = torch.bernoulli(probability)
            memory_write_log_probability = (
                memory_write_sample * probability.log()
                + (1.0 - memory_write_sample) * (1.0 - probability).log()
            )
            # Preserve the sampled hard decision for durable commit while
            # retaining the probability gradient for the transaction-local
            # soft row. The direct log-probability credit is consumed by the
            # outcome-only training loop.
            memory_write_backend_strength = (
                memory_write_sample
                + memory_write_strength
                - memory_write_strength.detach()
            )
        memory_value = self.memory_value(memory_value_context)
        if self.memory_value_feedback_enabled and self.memory_value_feedback is not None:
            memory_value = memory_value + self.memory_value_feedback(feedback_embedding)
        output = ControllerOutput(
            intention=intent_event,
            memory_key=memory_query_key,
            memory_value=memory_value,
            memory_write_strength=memory_write_strength,
            workspace_read=workspace_read,
            memory_query_key=memory_query_key,
            memory_read=memory_read,
            memory_write_receipt=None,
            event_attention=attention,
            event_reliability=reliability,
            execution_logits=execution_logits,
            memory_write_log_probability=memory_write_log_probability,
            memory_write_sample=memory_write_sample,
        )
        memory_write_receipt = (
            memory.write(
                output.memory_key,
                output.memory_value,
                memory_write_backend_strength,
                timestamp=timestamp,
                scope=memory_scope,
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
            execution_logits=output.execution_logits,
            memory_write_log_probability=output.memory_write_log_probability,
            memory_write_sample=output.memory_write_sample,
        )
        if disable_workspace:
            workspace = torch.zeros_like(workspace)
        next_state = ControllerState(
            hidden=hidden,
            workspace=workspace,
            latest_event=event,
            workspace_usage=usage,
            event_window=window,
            source_trust=source_trust,
        )
        return output, next_state
