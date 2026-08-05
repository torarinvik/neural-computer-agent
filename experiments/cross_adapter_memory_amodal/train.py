"""Qualify scalar-outcome memory retrieval across replaceable event adapters."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AmodalOutputBus,
    ContentAddressedMemory,
    ControllerFeedback,
    OpaqueProtocolDecoder,
    PersistentContentAddressedMemory,
)

from .environment import CrossAdapterRecallVerifier


@dataclass(frozen=True)
class Accounting:
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    optimizer_updates: int
    alignment_optimizer_updates: int
    unlabeled_alignment_events: int
    replayed_examples: int
    verifier_outcome_events: int
    feedback_events: int
    wall_time_seconds: float
    mean_action_latency_ms: float
    stable_bits_to_threshold: int | None
    retention_on_mastered_primitives: float | None = None
    transfer_ratio_against_fresh_learner: float | None = None
    persistent_audit_verifier_bits: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(
    *,
    seed: int,
    batch_size: int,
    width: int = 16,
    slot_count: int = 2,
    memory_capacity: int = 2,
    memory_write_threshold: float = 0.0,
    event_window_capacity: int | None = None,
    memory_scope_capacity: int | None = None,
) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=2,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=max(
            slot_count + 1,
            2 if event_window_capacity is None else event_window_capacity,
        ),
        memory_top_k=1,
    )
    return AmodalControllerRuntime(
        controller,
        output_bus=AmodalOutputBus(
            {"protocol": OpaqueProtocolDecoder(4, 2, hidden=8)}
        ),
        memory=ContentAddressedMemory(
            width=width,
            capacity=memory_capacity,
            scope_capacity=(
                batch_size if memory_scope_capacity is None else memory_scope_capacity
            ),
            write_threshold=memory_write_threshold,
        ),
    )


class ReaderEventAdapter(nn.Module):
    """A generic learned raw-event-to-neural-IR replacement adapter."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(self, raw_event: torch.Tensor) -> torch.Tensor:
        if raw_event.ndim != 2 or raw_event.shape[1] != self.projection.in_features:
            raise ValueError("reader raw event has the wrong shape")
        return self.projection(raw_event)


def _feedback(
    batch_size: int,
    *,
    action: torch.Tensor | None = None,
    reward: torch.Tensor | None = None,
    propensity: torch.Tensor | None = None,
    has_feedback: torch.Tensor | None = None,
) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch_size, 2) if action is None else action,
        reward=torch.zeros(batch_size) if reward is None else reward,
        propensity=torch.ones(batch_size) if propensity is None else propensity,
        has_feedback=(
            torch.zeros(batch_size) if has_feedback is None else has_feedback
        ),
    )


def _event(payload: torch.Tensor) -> AmodalEventCollection:
    return AmodalEventCollection.from_events([AmodalEvent(payload)])


def _slot_order(
    verifier: CrossAdapterRecallVerifier,
    *,
    target_cue: bool,
    randomize_slot_order: bool = False,
) -> torch.Tensor:
    """Return a trainer-side permutation of opaque event rows.

    When randomized ordering is enabled, the target appears at a random
    position. With the ordinary cue enabled, the cue is shown first; without
    it, the same schedule is a causal no-cue control. This prevents a bounded
    store from solving retention by always writing the most recent event.
    """
    slots = torch.arange(verifier.slot_count).expand(verifier.batch_size, -1)
    if randomize_slot_order:
        return torch.argsort(
            torch.rand(
                verifier.batch_size,
                verifier.slot_count,
                device=verifier.device,
            ),
            dim=1,
        )
    if not target_cue:
        return slots
    target = verifier.query_slot
    distractors = slots[slots != target[:, None]].reshape(
        verifier.batch_size, verifier.slot_count - 1
    )
    return torch.cat((distractors, target[:, None]), dim=1)


def _probe(
    runtime: AmodalControllerRuntime,
    state: Any,
    payload: torch.Tensor,
    uniform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    # This is an action preview. The same payload is committed once, with
    # its scalar outcome, by _store_output below. Advancing the state here
    # would insert every probe event twice and evict an earlier cue from the
    # bounded event window before the retention decision is made.
    output, _ = runtime.controller.step(
        _event(payload), state, _feedback(payload.shape[0]), memory=None
    )
    distribution = Categorical(logits=runtime.output_bus(output.intention)["protocol"])
    if uniform is None:
        action = distribution.sample()
    else:
        uniform = uniform.reshape(-1).to(
            device=distribution.probs.device, dtype=distribution.probs.dtype
        )
        if uniform.shape != (payload.shape[0],):
            raise ValueError("probe sampling uniform has the wrong shape")
        action = (
            uniform[:, None] >= distribution.probs.cumsum(dim=-1)
        ).sum(dim=-1).clamp_max(distribution.probs.shape[-1] - 1)
    propensity = distribution.probs.gather(1, action[:, None]).squeeze(1).clamp_min(1e-6)
    return action, propensity.log(), state


def _store(
    runtime: AmodalControllerRuntime,
    state: Any,
    payload: torch.Tensor,
    action: torch.Tensor,
    propensity_log: torch.Tensor,
    reward: torch.Tensor,
    scope: torch.Tensor,
    *,
    memory_write_override: torch.Tensor | None = None,
    memory_write_uniform: torch.Tensor | None = None,
) -> Any:
    state, _ = _store_output(
        runtime,
        state,
        payload,
        action,
        propensity_log,
        reward,
        scope,
        memory_write_override=memory_write_override,
        memory_write_uniform=memory_write_uniform,
    )
    return state


def _store_output(
    runtime: AmodalControllerRuntime,
    state: Any,
    payload: torch.Tensor,
    action: torch.Tensor,
    propensity_log: torch.Tensor,
    reward: torch.Tensor,
    scope: torch.Tensor,
    *,
    memory_write_override: torch.Tensor | None = None,
    memory_write_uniform: torch.Tensor | None = None,
) -> tuple[Any, Any]:
    opaque_action = torch.nn.functional.one_hot(action, num_classes=2).to(
        torch.float32
    )
    output, state = runtime.step_events(
        _event(payload),
        state,
        _feedback(
            payload.shape[0],
            action=opaque_action,
            reward=reward,
            propensity=propensity_log.exp(),
            has_feedback=torch.ones(payload.shape[0]),
        ),
        memory_scope=scope,
        memory_write_override=memory_write_override,
        memory_write_uniform=memory_write_uniform,
    )
    return state, output


def _query(
    runtime: AmodalControllerRuntime,
    payload: torch.Tensor,
    scope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, _ = runtime.step_events(
        _event(payload),
        runtime.initial_state(payload.shape[0], device=payload.device),
        _feedback(payload.shape[0]),
        memory_scope=scope,
    )
    distribution = Categorical(logits=output.decoded["protocol"])
    action = distribution.sample()
    return action, distribution.log_prob(action)


def _episode(
    runtime: AmodalControllerRuntime,
    verifier: CrossAdapterRecallVerifier,
    writer_tokens: torch.Tensor,
    reader_tokens: torch.Tensor,
    scope: torch.Tensor,
    *,
    query_mode: str,
    reader_adapter: ReaderEventAdapter,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    baseline: float = 0.5,
    reward_shuffle: bool = False,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
) -> tuple[torch.Tensor, float]:
    if query_mode not in {"writer", "reader", "raw_reader"}:
        raise ValueError("unknown query mode")
    verifier.reset()
    runtime.memory.clear()
    state = runtime.initial_state(verifier.batch_size, device="cpu")
    if target_cue:
        _, state = runtime.controller.step(
            _event(writer_tokens[verifier.query_slot]),
            state,
            _feedback(verifier.batch_size),
            memory=None,
        )
    slot_order = _slot_order(
        verifier,
        target_cue=target_cue,
        randomize_slot_order=randomize_slot_order,
    )
    probe_logs: list[torch.Tensor] = []
    for slot in range(verifier.slot_count):
        slot_ids = slot_order[:, slot]
        payload = writer_tokens[slot_ids]
        action, propensity_log, state = _probe(runtime, state, payload)
        reward = verifier.score_probe(
            slot_ids, action
        )
        if reward_shuffle:
            reward = torch.randint(0, 2, reward.shape).to(torch.float32)
        state = _store(
            runtime, state, payload, action, propensity_log, reward, scope
        )
        probe_logs.append(propensity_log)

    slots = verifier.query_slot
    if query_mode == "writer":
        query_payload = writer_tokens[slots]
    elif query_mode == "reader":
        query_payload = reader_adapter(reader_tokens[slots])
    else:
        query_payload = reader_tokens[slots]
    action, query_log = _query(runtime, query_payload, scope)
    reward = verifier.score_recall(action)
    if reward_shuffle:
        reward = torch.randint(0, 2, reward.shape).to(torch.float32)
    if train:
        assert optimizer is not None
        log_probability = query_log + torch.stack(probe_logs, dim=1).sum(dim=1)
        loss = -((reward.detach() - baseline) * log_probability).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
    return reward, 0.95 * baseline + 0.05 * float(reward.mean())


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: CrossAdapterRecallVerifier,
    writer_tokens: torch.Tensor,
    reader_tokens: torch.Tensor,
    reader_adapter: ReaderEventAdapter,
    *,
    condition: str,
    episodes: int = 64,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
) -> float:
    valid = {
        "writer",
        "reader",
        "raw_reader",
        "clear",
        "corrupt",
        "swapped_slot",
        "random_action",
        "missing_cue",
        "swapped_cue",
    }
    if condition not in valid:
        raise ValueError("unknown cross-adapter condition")
    runtime.eval()
    reader_adapter.eval()
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    for _ in range(episodes):
        verifier.reset()
        runtime.memory.clear()
        state = runtime.initial_state(verifier.batch_size, device="cpu")
        if target_cue and condition != "missing_cue":
            cue_slots = verifier.query_slot
            if condition == "swapped_cue":
                cue_slots = (cue_slots + 1) % verifier.slot_count
            _, state = runtime.controller.step(
                _event(writer_tokens[cue_slots]),
                state,
                _feedback(verifier.batch_size),
                memory=None,
            )
        slot_order = _slot_order(
            verifier,
            target_cue=target_cue,
            randomize_slot_order=randomize_slot_order,
        )
        for slot in range(verifier.slot_count):
            slot_ids = slot_order[:, slot]
            payload = writer_tokens[slot_ids]
            action = slot_ids % 2
            reward = verifier.score_probe(
                slot_ids, action
            )
            state = _store(
                runtime,
                state,
                payload,
                action,
                torch.full((verifier.batch_size,), np.log(0.5)),
                reward,
                scope,
            )
        query_slots = verifier.query_slot
        if condition == "writer":
            query_payload = writer_tokens[query_slots]
        elif condition == "reader":
            query_payload = reader_adapter(reader_tokens[query_slots])
        elif condition == "raw_reader":
            query_payload = reader_tokens[query_slots]
        elif condition == "swapped_slot":
            query_payload = reader_adapter(
                reader_tokens[(query_slots + 1) % verifier.slot_count]
            )
        else:
            query_payload = reader_adapter(reader_tokens[query_slots])
        if condition == "clear":
            runtime.memory.clear()
        elif condition == "corrupt":
            runtime.memory.values.zero_()
        if condition == "random_action":
            action = torch.randint(0, 2, (verifier.batch_size,))
        else:
            action, _ = _query(runtime, query_payload, scope)
        total += float(verifier.score_recall(action).sum())
    runtime.train()
    reader_adapter.train()
    return total / (episodes * verifier.batch_size)


@torch.no_grad()
def evaluate_token_population(
    runtime: AmodalControllerRuntime,
    reader_adapter: ReaderEventAdapter,
    *,
    basis: torch.Tensor,
    width: int,
    slot_count: int,
    batch_size: int,
    seed: int,
    pairs: int = 4,
    episodes: int = 64,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
) -> dict[str, dict[str, float | list[float]]]:
    """Audit cross-adapter retrieval on fresh opaque token pairs."""
    if min(pairs, episodes) < 1:
        raise ValueError("population pairs and episodes must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scores = {"writer": [], "reader": [], "raw_reader": [], "swapped_slot": []}
    for pair_index in range(pairs):
        writer_tokens = torch.randn(slot_count, width, generator=generator)
        reader_tokens = writer_tokens @ basis
        verifier = CrossAdapterRecallVerifier(
            batch_size=batch_size, seed=seed + 1 + pair_index, slot_count=slot_count
        )
        for condition, values in scores.items():
            values.append(
                evaluate_condition(
                    runtime,
                    verifier,
                    writer_tokens,
                    reader_tokens,
                    reader_adapter,
                    condition=condition,
                    episodes=episodes,
                    target_cue=target_cue,
                    randomize_slot_order=randomize_slot_order,
                )
            )

    return {
        condition: {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
            "per_pair": values,
        }
        for condition, values in scores.items()
    }


@torch.no_grad()
def evaluate_persistent_reload(
    runtime: AmodalControllerRuntime,
    verifier: CrossAdapterRecallVerifier,
    writer_tokens: torch.Tensor,
    reader_tokens: torch.Tensor,
    reader_adapter: ReaderEventAdapter,
    *,
    episodes: int = 16,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
) -> dict[str, float | bool]:
    """Audit reader retrieval after a persistent snapshot reload."""
    original_memory = runtime.memory
    if original_memory is None:
        raise RuntimeError("persistent audit requires a memory backend")
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    recovered_total = 0.0
    corruption_rejected = False

    def query(memory: PersistentContentAddressedMemory) -> float:
        runtime.memory = memory
        output, _ = runtime.step_events(
            _event(reader_adapter(reader_tokens[verifier.query_slot])),
            runtime.initial_state(verifier.batch_size, device="cpu"),
            _feedback(verifier.batch_size),
            memory_scope=scope,
            memory_write_override=torch.zeros(verifier.batch_size),
        )
        action = output.decoded["protocol"].argmax(dim=1)
        return float(verifier.score_recall(action).sum())

    with tempfile.TemporaryDirectory(prefix="neural-computer-cross-adapter-") as directory:
        path = Path(directory) / "cross-adapter-memory.pt"
        persistent = PersistentContentAddressedMemory(
            width=original_memory.width,
            capacity=original_memory.capacity,
            path=path,
            write_threshold=original_memory.write_threshold,
            query_temperature=original_memory.query_temperature,
            write_match_threshold=original_memory.write_match_threshold,
            scope_capacity=original_memory.scope_capacity,
        )
        runtime.memory = persistent
        try:
            for _ in range(episodes):
                verifier.reset()
                runtime.memory = persistent
                persistent.clear()
                state = runtime.initial_state(verifier.batch_size, device="cpu")
                if target_cue:
                    _, state = runtime.controller.step(
                        _event(writer_tokens[verifier.query_slot]),
                        state,
                        _feedback(verifier.batch_size),
                        memory=None,
                    )
                slot_order = _slot_order(
                    verifier,
                    target_cue=target_cue,
                    randomize_slot_order=randomize_slot_order,
                )
                for slot in range(verifier.slot_count):
                    slot_ids = slot_order[:, slot]
                    payload = writer_tokens[slot_ids]
                    action = slot_ids % 2
                    reward = verifier.score_probe(
                        slot_ids, action
                    )
                    state = _store(
                        runtime,
                        state,
                        payload,
                        action,
                        torch.full((verifier.batch_size,), np.log(0.5)),
                        reward,
                        scope,
                    )
                reloaded = PersistentContentAddressedMemory(
                    width=original_memory.width,
                    capacity=original_memory.capacity,
                    path=path,
                    write_threshold=original_memory.write_threshold,
                    query_temperature=original_memory.query_temperature,
                    write_match_threshold=original_memory.write_match_threshold,
                    scope_capacity=original_memory.scope_capacity,
                )
                total += query(reloaded)
                reopened = PersistentContentAddressedMemory(
                    width=original_memory.width,
                    capacity=original_memory.capacity,
                    path=path,
                    write_threshold=original_memory.write_threshold,
                    query_temperature=original_memory.query_temperature,
                    write_match_threshold=original_memory.write_match_threshold,
                    scope_capacity=original_memory.scope_capacity,
                )
                recovered_total += query(reopened)

            good_payload = torch.load(path, map_location="cpu", weights_only=False)
            corrupted_payload = dict(good_payload)
            corrupted_state = dict(good_payload["state_dict"])
            values = corrupted_state["values"].clone()
            values.reshape(-1)[0] += 1.0
            corrupted_state["values"] = values
            corrupted_payload["state_dict"] = corrupted_state
            torch.save(corrupted_payload, path)
            try:
                PersistentContentAddressedMemory(
                    width=original_memory.width,
                    capacity=original_memory.capacity,
                    path=path,
                    write_threshold=original_memory.write_threshold,
                    query_temperature=original_memory.query_temperature,
                    write_match_threshold=original_memory.write_match_threshold,
                    scope_capacity=original_memory.scope_capacity,
                )
            except ValueError as error:
                corruption_rejected = "checksum" in str(error)
            runtime.memory = persistent
            persistent.snapshot(path)
            recovered = PersistentContentAddressedMemory(
                width=original_memory.width,
                capacity=original_memory.capacity,
                path=path,
                write_threshold=original_memory.write_threshold,
                query_temperature=original_memory.query_temperature,
                write_match_threshold=original_memory.write_match_threshold,
                scope_capacity=original_memory.scope_capacity,
            )
            query(recovered)
        finally:
            runtime.memory = original_memory
    return {
        "reload_intact_recall": total / (episodes * verifier.batch_size),
        "corruption_rejected": corruption_rejected,
        "recovery_intact_recall": recovered_total / (episodes * verifier.batch_size),
    }


def train_base(
    runtime: AmodalControllerRuntime,
    verifier: CrossAdapterRecallVerifier,
    writer_tokens: torch.Tensor,
    reader_tokens: torch.Tensor,
    reader_adapter: ReaderEventAdapter,
    *,
    steps: int,
    seed: int,
    basis: torch.Tensor,
    randomize_event_tokens: bool = False,
    token_reuse_steps: int = 4,
    reward_shuffle: bool = False,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
) -> int | None:
    seed_everything(seed)
    runtime.train()
    reader_adapter.eval()
    for parameter in reader_adapter.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(runtime.parameters(), lr=2e-3)
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    baseline = 0.5
    history: list[float] = []
    token_block: tuple[torch.Tensor, torch.Tensor] | None = None
    token_block_step = token_reuse_steps
    for step in range(1, steps + 1):
        if randomize_event_tokens and token_block_step >= token_reuse_steps:
            episode_writer_tokens = torch.randn_like(writer_tokens)
            token_block = (episode_writer_tokens, episode_writer_tokens @ basis)
            token_block_step = 0
        episode_writer_tokens, episode_reader_tokens = (
            token_block if token_block is not None else (writer_tokens, reader_tokens)
        )
        token_block_step += 1
        with runtime.memory.differentiable_transaction():
            _, baseline = _episode(
                runtime,
                verifier,
                episode_writer_tokens,
                episode_reader_tokens,
                scope,
                query_mode="writer",
                reader_adapter=reader_adapter,
                train=True,
                optimizer=optimizer,
                baseline=baseline,
                reward_shuffle=reward_shuffle,
                target_cue=target_cue,
                randomize_slot_order=randomize_slot_order,
            )
        if step == 1 or step % 32 == 0 or step == steps:
            history.append(
                evaluate_condition(
                    runtime,
                    CrossAdapterRecallVerifier(
                        batch_size=verifier.batch_size,
                        seed=seed + 1000 + step,
                        slot_count=verifier.slot_count,
                    ),
                    writer_tokens,
                    reader_tokens,
                    reader_adapter,
                    condition="writer",
                    episodes=32,
                    target_cue=target_cue,
                    randomize_slot_order=randomize_slot_order,
                )
            )
    for parameter in reader_adapter.parameters():
        parameter.requires_grad = True
    for index, score in enumerate(history):
        if score >= 0.8 and all(later >= 0.8 for later in history[index:]):
            return (index + 1) * 32 if index else 1
    return None


def train_writer_intervention(
    runtime: AmodalControllerRuntime,
    verifier: CrossAdapterRecallVerifier,
    tokens: torch.Tensor,
    *,
    steps: int,
    seed: int,
    target_cue: bool,
    randomize_slot_order: bool = False,
    reward_shuffle: bool = False,
    randomize_event_tokens: bool = False,
    token_reuse_steps: int = 4,
) -> dict[str, object]:
    """Train generic write utility with paired write/skip interventions.

    The duplicated rows share one hidden verifier world. One arm writes only
    at a selected position and the other skips it; the scalar recall
    difference trains the generic write policy. No target index or verifier
    bit enters the runtime.
    """
    if steps < 1:
        raise ValueError("writer intervention steps must be positive")
    if token_reuse_steps < 1:
        raise ValueError("token reuse steps must be positive")
    seed_everything(seed)
    runtime.train()
    for parameter in runtime.parameters():
        parameter.requires_grad = False
    for parameter in runtime.controller.memory_write_policy.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(
        runtime.controller.memory_write_policy.parameters(), lr=2e-3
    )
    capacity_one_tail_isolation = (
        runtime.memory is not None and getattr(runtime.memory, "capacity", None) == 1
    )
    batch = verifier.batch_size
    paired_batch = batch * 2
    scope = torch.arange(paired_batch, dtype=torch.long)
    history: list[float] = []
    write_utilities: list[float] = []
    token_block: torch.Tensor | None = None
    token_block_step = token_reuse_steps
    for step in range(1, steps + 1):
        if randomize_event_tokens and token_block_step >= token_reuse_steps:
            token_block = torch.randn_like(tokens)
            token_block_step = 0
        episode_tokens = token_block if token_block is not None else tokens
        token_block_step += 1
        verifier.reset()
        paired_verifier = verifier.duplicate_rows(2)
        runtime.memory.clear()
        state = runtime.initial_state(paired_batch, device="cpu")
        if target_cue:
            _, state = runtime.controller.step(
                _event(episode_tokens[paired_verifier.query_slot]),
                state,
                _feedback(paired_batch),
                memory=None,
            )
        arm = torch.arange(paired_batch, dtype=torch.long) % 2
        branch_slot = (torch.arange(batch) // verifier.slot_count) % verifier.slot_count
        all_slots = torch.arange(verifier.slot_count).expand(batch, -1)
        if randomize_slot_order:
            branch_position = torch.randint(0, verifier.slot_count, (batch,))
            ordered_rows: list[torch.Tensor] = []
            for row in range(batch):
                other = all_slots[row][all_slots[row] != branch_slot[row]]
                permutation = other[torch.randperm(other.numel())]
                ordered_rows.append(
                    torch.cat(
                        (
                            permutation[: branch_position[row]],
                            branch_slot[row : row + 1],
                            permutation[branch_position[row] :],
                        )
                    )
                )
            ordered_slots = torch.stack(ordered_rows)
        else:
            branch_position = torch.full(
                (batch,), verifier.slot_count - 1, dtype=torch.long
            )
            other_slots = all_slots[all_slots != branch_slot[:, None]].reshape(
                batch, verifier.slot_count - 1
            )
            ordered_slots = torch.cat((other_slots, branch_slot[:, None]), dim=1)
        paired_slots = ordered_slots.repeat_interleave(2, dim=0)
        paired_branch_position = branch_position.repeat_interleave(2)
        probe_logs: list[torch.Tensor] = []
        branch_strengths: list[torch.Tensor] = []
        with runtime.memory.differentiable_transaction():
            for position in range(verifier.slot_count):
                slot = paired_slots[:, position]
                payload = episode_tokens[slot]
                paired_uniform = torch.rand(batch).repeat_interleave(2)
                action, propensity_log, state = _probe(
                    runtime, state, payload, uniform=paired_uniform
                )
                reward = paired_verifier.score_probe(
                    slot, action
                )
                if reward_shuffle:
                    reward = torch.randint(0, 2, (batch,)).repeat_interleave(2).to(
                        torch.float32
                    )
                branch_position_mask = position == paired_branch_position
                if capacity_one_tail_isolation and position > 0:
                    # A one-slot store cannot identify the utility of an
                    # earlier candidate if a later write is allowed to erase
                    # both counterfactual arms.  Isolate the causal write by
                    # skipping the suffix after the selected position. This
                    # changes only the trainer's paired intervention; the
                    # deployed policy still sees ordinary sequential events.
                    suffix_uniform = torch.ones(paired_batch)
                    write_uniform = torch.where(
                        branch_position_mask & (arm == 0),
                        torch.zeros(paired_batch),
                        torch.where(
                            branch_position_mask & (arm == 1),
                            torch.ones(paired_batch),
                            suffix_uniform,
                        ),
                    )
                else:
                    write_uniform = torch.where(
                        branch_position_mask & (arm == 0),
                        torch.zeros(paired_batch),
                        torch.where(
                            branch_position_mask & (arm == 1),
                            torch.ones(paired_batch),
                            torch.rand(batch).repeat_interleave(2),
                        ),
                    )
                state, output = _store_output(
                    runtime,
                    state,
                    payload,
                    action,
                    propensity_log,
                    reward,
                    scope,
                    memory_write_uniform=write_uniform,
                )
                probe_logs.append(propensity_log)
                branch_strengths.append(output.controller.memory_write_strength)

            query_output, _ = runtime.step_events(
                _event(episode_tokens[paired_verifier.query_slot]),
                runtime.initial_state(paired_batch, device="cpu"),
                _feedback(paired_batch),
                memory_scope=scope,
                memory_write_override=torch.zeros(paired_batch),
            )
            distribution = Categorical(logits=query_output.decoded["protocol"])
            query_uniform = torch.rand(batch).repeat_interleave(2).to(
                device=distribution.probs.device, dtype=distribution.probs.dtype
            )
            action = (
                query_uniform[:, None] >= distribution.probs.cumsum(dim=-1)
            ).sum(dim=-1).clamp_max(distribution.probs.shape[-1] - 1)
            reward = paired_verifier.score_recall(action)
            if reward_shuffle:
                reward = torch.randint(0, 2, (batch,)).repeat_interleave(2).to(
                    torch.float32
                )
            pair_rewards = reward.reshape(batch, 2)
            pair_mean = pair_rewards.mean(dim=1).repeat_interleave(2)
            action_log_probability = distribution.log_prob(action) + torch.stack(
                probe_logs, dim=1
            ).sum(dim=1)
            action_loss = -(
                (reward.detach() - pair_mean.detach()) * action_log_probability
            ).mean()
            strength_by_arm = torch.stack(branch_strengths, dim=1).reshape(
                batch, 2, verifier.slot_count
            )
            strength = strength_by_arm.gather(
                2,
                branch_position[:, None, None].expand(-1, 2, -1),
            ).squeeze(-1)
            write_logit = torch.logit(
                strength[:, 0].clamp(1e-6, 1.0 - 1e-6)
            )
            write_utility = pair_rewards[:, 0] - pair_rewards[:, 1]
            write_credit_loss = -(write_utility.detach() * write_logit).mean()
            if capacity_one_tail_isolation:
                # The paired intervention is measuring retention utility, not
                # relearning the probe/query policy. Letting query REINFORCE
                # gradients flow through the one-slot memory gate creates a
                # strong unconditional-write pressure and drowns the
                # content-conditioned contrast.
                loss = write_credit_loss
            else:
                loss = action_loss + write_credit_loss
                loss = loss - 0.01 * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                runtime.controller.memory_write_policy.parameters(), max_norm=5.0
            )
            optimizer.step()
        write_utilities.append(float(write_utility.mean()))
        if step == 1 or step % 32 == 0 or step == steps:
            history.append(float(pair_rewards.mean()))
    for parameter in runtime.parameters():
        parameter.requires_grad = True
    return {
        "steps": steps,
        "optimizer_updates": steps,
        "unique_verifier_bits": steps * batch * (verifier.slot_count + 1),
        "unique_logical_lifetimes": steps * batch,
        "verifier_outcome_events": steps * paired_batch * (verifier.slot_count + 1),
        "feedback_events": steps * paired_batch * verifier.slot_count,
        "mean_write_utility": sum(write_utilities) / max(1, len(write_utilities)),
        "capacity_one_tail_isolation": capacity_one_tail_isolation,
        "capacity_one_action_credit_isolated": capacity_one_tail_isolation,
        "randomize_event_tokens": randomize_event_tokens,
        "token_reuse_steps": token_reuse_steps,
        "history": history,
    }


def train_reader_adapter(
    runtime: AmodalControllerRuntime,
    verifier: CrossAdapterRecallVerifier,
    writer_tokens: torch.Tensor,
    reader_tokens: torch.Tensor,
    reader_adapter: ReaderEventAdapter,
    *,
    steps: int,
    seed: int,
    basis: torch.Tensor,
    randomize_event_tokens: bool = False,
    token_reuse_steps: int = 4,
    reward_shuffle: bool = False,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
) -> int | None:
    seed_everything(seed)
    runtime.eval()
    for parameter in runtime.parameters():
        parameter.requires_grad = False
    reader_adapter.train()
    optimizer = torch.optim.Adam(reader_adapter.parameters(), lr=3e-3)
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    baseline = 0.5
    history: list[float] = []
    token_block: tuple[torch.Tensor, torch.Tensor] | None = None
    token_block_step = token_reuse_steps
    for step in range(1, steps + 1):
        if randomize_event_tokens and token_block_step >= token_reuse_steps:
            episode_writer_tokens = torch.randn_like(writer_tokens)
            token_block = (episode_writer_tokens, episode_writer_tokens @ basis)
            token_block_step = 0
        episode_writer_tokens, episode_reader_tokens = (
            token_block if token_block is not None else (writer_tokens, reader_tokens)
        )
        token_block_step += 1
        _, baseline = _episode(
            runtime,
            verifier,
            episode_writer_tokens,
            episode_reader_tokens,
            scope,
            query_mode="reader",
            reader_adapter=reader_adapter,
            train=True,
            optimizer=optimizer,
            baseline=baseline,
            reward_shuffle=reward_shuffle,
            target_cue=target_cue,
            randomize_slot_order=randomize_slot_order,
        )
        if step == 1 or step % 32 == 0 or step == steps:
            history.append(
                evaluate_condition(
                    runtime,
                    CrossAdapterRecallVerifier(
                        batch_size=verifier.batch_size,
                        seed=seed + 2000 + step,
                        slot_count=verifier.slot_count,
                    ),
                    writer_tokens,
                    reader_tokens,
                    reader_adapter,
                    condition="reader",
                    episodes=32,
                    target_cue=target_cue,
                    randomize_slot_order=randomize_slot_order,
                )
            )
    for parameter in runtime.parameters():
        parameter.requires_grad = True
    for index, score in enumerate(history):
        if score >= 0.8 and all(later >= 0.8 for later in history[index:]):
            return (index + 1) * 32 if index else 1
    return None


def align_reader_adapter(
    reader_adapter: ReaderEventAdapter,
    *,
    basis: torch.Tensor,
    steps: int,
    batch_size: int,
    seed: int,
) -> float:
    """Align paired opaque event streams without verifier-private labels."""
    if steps < 1:
        raise ValueError("alignment steps must be positive")
    seed_everything(seed)
    reader_adapter.train()
    optimizer = torch.optim.Adam(reader_adapter.parameters(), lr=3e-3)
    final_loss = 0.0
    for _ in range(steps):
        writer_events = torch.randn(batch_size, basis.shape[0])
        reader_raw = writer_events @ basis
        aligned = reader_adapter(reader_raw)
        loss = torch.nn.functional.mse_loss(aligned, writer_events)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def run_experiment(
    *,
    base_steps: int,
    adapter_steps: int,
    alignment_steps: int = 64,
    seed: int,
    batch_size: int = 16,
    slot_count: int = 2,
    memory_capacity: int = 2,
    memory_write_threshold: float = 0.0,
    event_window_capacity: int | None = None,
    writer_intervention_steps: int = 0,
    persistent_memory_audit: bool = False,
    randomize_event_tokens: bool = False,
    token_reuse_steps: int = 4,
    random_orthogonal_basis: bool = False,
    target_cue: bool = False,
    randomize_slot_order: bool = False,
    report_out: Path | None = None,
    reward_shuffle: bool = False,
) -> dict[str, Any]:
    if min(base_steps, adapter_steps, alignment_steps) < 1:
        raise ValueError("training steps must be positive")
    if slot_count < 2:
        raise ValueError("slot count must be at least two")
    if memory_capacity < 1:
        raise ValueError("memory capacity must be positive")
    if not 0.0 <= memory_write_threshold <= 1.0:
        raise ValueError("memory write threshold must lie in [0, 1]")
    if writer_intervention_steps < 0:
        raise ValueError("writer intervention steps cannot be negative")
    if token_reuse_steps < 1:
        raise ValueError("token reuse steps must be positive")
    seed_everything(seed)
    started = time.perf_counter()
    runtime = build_runtime(
        seed=seed,
        batch_size=batch_size,
        slot_count=slot_count,
        memory_capacity=memory_capacity,
        memory_write_threshold=memory_write_threshold,
        event_window_capacity=event_window_capacity,
        memory_scope_capacity=(
            batch_size * 2
            if writer_intervention_steps
            else batch_size
        ),
    )
    width = runtime.event_width
    writer_tokens = torch.randn(slot_count, width)
    if random_orthogonal_basis:
        basis_generator = torch.Generator(device="cpu")
        basis_generator.manual_seed(seed + 7500)
        basis, _ = torch.linalg.qr(
            torch.randn(width, width, generator=basis_generator)
        )
    else:
        basis = torch.zeros(width, width)
        for index in range(0, width - 1, 2):
            basis[index, index + 1] = 1.0
            basis[index + 1, index] = -1.0
        if width % 2:
            basis[-1, -1] = 1.0
    reader_tokens = writer_tokens @ basis
    reader_adapter = ReaderEventAdapter(width)
    verifier = CrossAdapterRecallVerifier(
        batch_size=batch_size, seed=seed + 10, slot_count=slot_count
    )

    def make_verifier(verifier_seed: int) -> CrossAdapterRecallVerifier:
        return CrossAdapterRecallVerifier(
            batch_size=batch_size, seed=verifier_seed, slot_count=slot_count
        )
    base_stable_bits = train_base(
        runtime,
        verifier,
        writer_tokens,
        reader_tokens,
        reader_adapter,
        steps=base_steps,
        seed=seed,
        basis=basis,
        randomize_event_tokens=randomize_event_tokens,
        token_reuse_steps=token_reuse_steps,
        reward_shuffle=reward_shuffle,
        target_cue=target_cue,
        randomize_slot_order=randomize_slot_order,
    )
    writer_intervention_accounting: dict[str, object] | None = None
    if writer_intervention_steps:
        writer_intervention_accounting = train_writer_intervention(
            runtime,
            verifier,
            writer_tokens,
            steps=writer_intervention_steps,
            seed=seed + 600,
            target_cue=target_cue,
            randomize_slot_order=randomize_slot_order,
            reward_shuffle=reward_shuffle,
            randomize_event_tokens=randomize_event_tokens,
            token_reuse_steps=token_reuse_steps,
        )
    reader_zero_shot = evaluate_condition(
        runtime,
        make_verifier(seed + 100),
        writer_tokens,
        reader_tokens,
        reader_adapter,
        condition="reader",
        target_cue=target_cue,
        randomize_slot_order=randomize_slot_order,
    )
    alignment_loss = align_reader_adapter(
        reader_adapter,
        basis=basis,
        steps=alignment_steps,
        batch_size=batch_size,
        seed=seed + 250,
    )
    adapter_stable_bits = train_reader_adapter(
        runtime,
        make_verifier(seed + 200),
        writer_tokens,
        reader_tokens,
        reader_adapter,
        steps=adapter_steps,
        seed=seed + 300,
        basis=basis,
        randomize_event_tokens=randomize_event_tokens,
        token_reuse_steps=token_reuse_steps,
        reward_shuffle=reward_shuffle,
        target_cue=target_cue,
        randomize_slot_order=randomize_slot_order,
    )
    conditions = {
        condition: evaluate_condition(
            runtime,
            make_verifier(seed + 400 + index),
            writer_tokens,
            reader_tokens,
            reader_adapter,
            condition=condition,
            target_cue=target_cue,
            randomize_slot_order=randomize_slot_order,
        )
        for index, condition in enumerate(
            (
                "writer",
                "reader",
                "raw_reader",
                "clear",
                "corrupt",
                "swapped_slot",
                "random_action",
                "missing_cue",
                "swapped_cue",
            )
        )
    }
    population = evaluate_token_population(
        runtime,
        reader_adapter,
        basis=basis,
        width=width,
        slot_count=slot_count,
        batch_size=batch_size,
        seed=seed + 5000,
        target_cue=target_cue,
        randomize_slot_order=randomize_slot_order,
    )
    population_gains = [
        reader - raw
        for reader, raw in zip(
            population["reader"]["per_pair"],
            population["raw_reader"]["per_pair"],
        )
    ]
    population["reader_vs_raw_gain"] = {
        "mean": sum(population_gains) / len(population_gains),
        "minimum": min(population_gains),
        "maximum": max(population_gains),
        "per_pair": population_gains,
    }
    persistent_memory: dict[str, float | bool] | None = None
    if persistent_memory_audit:
        persistent_memory = evaluate_persistent_reload(
            runtime,
            make_verifier(seed + 600),
            writer_tokens,
            reader_tokens,
            reader_adapter,
            target_cue=target_cue,
            randomize_slot_order=randomize_slot_order,
        )
    promotion = (
        not reward_shuffle
        and base_stable_bits is not None
        and adapter_stable_bits is not None
        and conditions["writer"] >= 0.80
        and conditions["reader"] >= 0.80
        and population["writer"]["minimum"] >= 0.90
        and population["reader"]["minimum"] >= 0.90
        and population["reader_vs_raw_gain"]["minimum"] >= 0.0
        and population["reader_vs_raw_gain"]["mean"] >= 0.15
        and population["swapped_slot"]["maximum"] <= 0.65
        and conditions["clear"] <= 0.65
        and conditions["corrupt"] <= 0.65
        and conditions["swapped_slot"] <= 0.65
        and conditions["reader"] - conditions["clear"] >= 0.25
        and (
            persistent_memory is None
            or (
                float(persistent_memory["reload_intact_recall"]) >= 0.80
                and bool(persistent_memory["corruption_rejected"])
                and float(persistent_memory["recovery_intact_recall"]) >= 0.80
            )
        )
    )
    persistent_audit_verifier_bits = (
        16 * batch_size * (slot_count + 2) if persistent_memory is not None else 0
    )
    writer_unique_bits = base_steps * batch_size * (slot_count + 2)
    if writer_intervention_accounting is not None:
        writer_unique_bits += int(
            writer_intervention_accounting["unique_verifier_bits"]
        )
    writer_outcome_events = base_steps * batch_size * (slot_count + 2)
    if writer_intervention_accounting is not None:
        writer_outcome_events += int(
            writer_intervention_accounting["verifier_outcome_events"]
        )
    writer_feedback_events = base_steps * batch_size * slot_count
    if writer_intervention_accounting is not None:
        writer_feedback_events += int(
            writer_intervention_accounting["feedback_events"]
        )
    writer_optimizer_updates = base_steps
    if writer_intervention_accounting is not None:
        writer_optimizer_updates += int(
            writer_intervention_accounting["optimizer_updates"]
        )
    writer_logical_lifetimes = base_steps * batch_size
    if writer_intervention_accounting is not None:
        writer_logical_lifetimes += int(
            writer_intervention_accounting["unique_logical_lifetimes"]
        )
    accounting = Accounting(
        unique_verifier_bits=writer_unique_bits
        + adapter_steps * batch_size * (slot_count + 2),
        unique_logical_lifetimes=writer_logical_lifetimes + adapter_steps * batch_size,
        optimizer_updates=writer_optimizer_updates + adapter_steps + alignment_steps,
        alignment_optimizer_updates=alignment_steps,
        unlabeled_alignment_events=alignment_steps * batch_size,
        replayed_examples=0,
        verifier_outcome_events=writer_outcome_events
        + adapter_steps * batch_size * (slot_count + 2),
        feedback_events=writer_feedback_events
        + adapter_steps * batch_size * slot_count,
        wall_time_seconds=time.perf_counter() - started,
        mean_action_latency_ms=0.0,
        stable_bits_to_threshold=adapter_stable_bits,
        persistent_audit_verifier_bits=persistent_audit_verifier_bits,
    )
    report = {
        "experiment": "outcome-only-cross-adapter-memory-retrieval",
        "seed": seed,
        "base_steps": base_steps,
        "adapter_steps": adapter_steps,
        "alignment_steps": alignment_steps,
        "batch_size": batch_size,
        "slot_count": slot_count,
        "memory_capacity": memory_capacity,
        "memory_write_threshold": memory_write_threshold,
        "event_window_capacity": runtime.controller.event_window_capacity,
        "learner_visible_inputs": [
            "opaque writer event",
            "opaque reader event",
            "opaque probe action",
            "scalar probe outcome",
            "scalar recall outcome",
        ],
        "adapter_boundary": {
            "writer": "fixed opaque event adapter",
            "reader": "trainable generic linear event adapter",
            "reader_raw_basis": "fixed random orthogonal latent basis",
            "controller_frozen_during_reader_phase": True,
            "memory_top_k": 1,
            "memory_capacity": memory_capacity,
            "memory_write_threshold": memory_write_threshold,
            "memory_read_match_threshold": runtime.memory.configuration()[
                "read_match_threshold"
            ],
            "memory_write_match_threshold": runtime.memory.configuration()[
                "write_match_threshold"
            ],
            "event_window_capacity": runtime.controller.event_window_capacity,
            "paired_unlabeled_alignment": True,
            "randomize_event_tokens": randomize_event_tokens,
            "token_reuse_steps": token_reuse_steps,
            "random_orthogonal_basis": random_orthogonal_basis,
            "target_cue": target_cue,
            "target_presentation": (
                "cued_row_random_position"
                if randomize_slot_order and target_cue
                else "random_position_without_cue"
                if randomize_slot_order
                else "cued_row_last"
                if target_cue
                else "fixed_slot_order"
            ),
            "writer_intervention_steps": writer_intervention_steps,
            "writer_intervention_protocol": (
                "counterfactual_leave_one_out_v2_token_diverse"
                if randomize_event_tokens
                else "counterfactual_leave_one_out_v1"
            ),
        },
        "conditions": conditions,
        "promotion_gate": {
            "writer_min": 0.80,
            "reader_min": 0.80,
            "raw_reader_max": 0.65,
            "clear_max": 0.65,
            "corrupt_max": 0.65,
            "swapped_slot_max": 0.65,
            "missing_cue_and_swapped_cue": "diagnostic_only; no cue-conditioned selection claim",
            "reader_clear_gap_min": 0.25,
            "reader_population_min": 0.90,
            "paired_reader_vs_raw_gain_min": 0.0,
            "mean_reader_vs_raw_gain_min": 0.15,
            "swapped_population_max": 0.65,
            "stable_prefix_required": True,
        },
        "zero_shot_reader_recall": reader_zero_shot,
        "reader_alignment_final_mse": alignment_loss,
        "fresh_token_population": population,
        "persistent_memory": persistent_memory,
        "base_stable_bits_to_threshold": base_stable_bits,
        "writer_intervention_accounting": writer_intervention_accounting,
        "adapter_stable_bits_to_threshold": adapter_stable_bits,
        "promoted": promotion,
        "claim_boundary": (
            "Synthetic outcome-only cross-adapter retrieval with randomized target "
            "position and a cue-assisted training curriculum; cue removal and "
            "cue swaps are reported diagnostics, not a cue-conditioned selection "
            "claim; bounded memory is qualified only under the stated read-match "
            "contract; no natural-modality alignment or general episodic-memory "
            "claim"
        ),
        "accounting": asdict(accounting),
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-steps", type=int, default=256)
    parser.add_argument("--adapter-steps", type=int, default=512)
    parser.add_argument("--alignment-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--slot-count", type=int, default=2)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--memory-write-threshold", type=float, default=0.0)
    parser.add_argument("--event-window-capacity", type=int)
    parser.add_argument("--writer-intervention-steps", type=int, default=0)
    parser.add_argument("--persistent-memory-audit", action="store_true")
    parser.add_argument("--randomize-event-tokens", action="store_true")
    parser.add_argument("--token-reuse-steps", type=int, default=4)
    parser.add_argument("--random-orthogonal-basis", action="store_true")
    parser.add_argument("--target-cue", action="store_true")
    parser.add_argument("--randomize-slot-order", action="store_true")
    parser.add_argument("--reward-shuffle", action="store_true")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
