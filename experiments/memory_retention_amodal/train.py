"""Qualify outcome-only memory retention through the canonical runtime."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from contextlib import nullcontext
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

from .environment import OutcomeOnlyRetentionVerifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(
    *,
    seed: int,
    batch_size: int,
    width: int = 16,
    memory_capacity: int = 1,
    event_window_capacity: int = 3,
    memory_scope_capacity: int | None = None,
    memory_value_feedback: bool = True,
    identity_memory_address: bool = False,
    stable_memory_address: bool = True,
) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=2,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=event_window_capacity,
        memory_top_k=1,
        memory_value_feedback=memory_value_feedback,
        stable_memory_address=stable_memory_address,
    )
    if identity_memory_address:
        with torch.no_grad():
            controller.memory_address.weight.copy_(torch.eye(width))
            controller.memory_address.bias.zero_()
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
            write_threshold=0.5,
        ),
    )


class OutcomeWriteCritic(nn.Module):
    """Training-only baseline for a stochastic memory write decision.

    The critic receives detached learned tensors from the controller after an
    opaque outcome has been processed. It is never attached to the runtime
    checkpoint or output bus, and it never receives verifier-private labels.
    """

    def __init__(self, width: int, hidden: int | None = None) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("critic width must be positive")
        hidden_width = max(8, width // 2) if hidden is None else hidden
        if hidden_width < 1:
            raise ValueError("critic hidden width must be positive")
        self.network = nn.Sequential(
            nn.Linear(width * 3 + 1, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.network[0].in_features:
            raise ValueError("critic features have the wrong shape")
        return self.network(features).squeeze(-1)


class OutcomeValueCritic(nn.Module):
    """Training-only scalar value baseline for parent action learning."""

    def __init__(self, width: int, hidden: int | None = None) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("value critic width must be positive")
        hidden_width = max(8, width // 2) if hidden is None else hidden
        if hidden_width < 1:
            raise ValueError("value critic hidden width must be positive")
        self.network = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.network[0].in_features:
            raise ValueError("value critic features have the wrong shape")
        return self.network(features).squeeze(-1)


def _empty(batch: int, width: int) -> AmodalEventCollection:
    return AmodalEventCollection.empty(batch, width, device="cpu")


def _event(token: torch.Tensor) -> AmodalEventCollection:
    return AmodalEventCollection.from_events([AmodalEvent(token)])


@torch.no_grad()
def evaluate_persistent_reload(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    episodes: int = 16,
) -> dict[str, float | bool]:
    """Audit reload, checksum rejection, and recovery of a disk-backed store."""
    original_memory = runtime.memory
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    recovered_total = 0.0
    corruption_rejected = False

    def query_reloaded(
        reloaded: PersistentContentAddressedMemory,
    ) -> float:
        runtime.memory = reloaded
        query_output, _ = runtime.step_events(
            _event(tokens[verifier.query_slot]),
            runtime.initial_state(verifier.batch_size, device="cpu"),
            _feedback(verifier.batch_size),
            memory_scope=scope,
        )
        distribution = Categorical(logits=query_output.decoded["protocol"])
        return float(verifier.score_recall(distribution.sample()).sum())

    with tempfile.TemporaryDirectory(prefix="neural-computer-memory-") as directory:
        path = Path(directory) / "retention-memory.pt"
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
                _, state = runtime.controller.step(
                    _event(tokens[verifier.query_slot]),
                    state,
                    _feedback(verifier.batch_size),
                    memory=None,
                )
                for slot in verifier.order.T:
                    action, propensity, _, state = _probe_without_writing(
                        runtime, state, _event(tokens[slot])
                    )
                    reward = verifier.score_probe(slot, action)
                    state, _, _, _, _ = _store_outcome(
                        runtime,
                        state,
                        action,
                        reward,
                        propensity,
                        scope,
                        sample_memory_writes=False,
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
                total += query_reloaded(reloaded)

            # Corrupt the serialized state without updating its checksum. A
            # new backend must reject the snapshot instead of serving altered
            # learned memory. The live backend then restores the known-good
            # snapshot atomically, and a fresh backend must recover it.
            good_payload = torch.load(path, map_location="cpu", weights_only=False)
            corrupted_payload = dict(good_payload)
            corrupted_state = dict(good_payload["state_dict"])
            corrupted_values = corrupted_state["values"].clone()
            corrupted_values.reshape(-1)[0] += 1.0
            corrupted_state["values"] = corrupted_values
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
            recovered_total = query_reloaded(recovered)
        finally:
            runtime.memory = original_memory
    return {
        "reload_intact_recall": total / (episodes * verifier.batch_size),
        "corruption_rejected": corruption_rejected,
        "recovery_intact_recall": recovered_total / verifier.batch_size,
    }


def _feedback(
    batch: int,
    *,
    action: torch.Tensor | None = None,
    reward: torch.Tensor | None = None,
    propensity: torch.Tensor | None = None,
    has_feedback: torch.Tensor | None = None,
) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, 2) if action is None else action,
        reward=torch.zeros(batch) if reward is None else reward,
        propensity=torch.ones(batch) if propensity is None else propensity,
        has_feedback=(
            torch.zeros(batch) if has_feedback is None else has_feedback
        ),
    )


def _probe_without_writing(
    runtime: AmodalControllerRuntime,
    state: Any,
    event: AmodalEventCollection,
    uniform: torch.Tensor | None = None,
    forced_action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    output, state = runtime.controller.step(
        event,
        state,
        _feedback(event.payload.shape[0]),
        memory=None,
    )
    distribution = Categorical(logits=runtime.output_bus(output.intention)["protocol"])
    safe_probs = distribution.probs.clamp_min(1e-6)
    safe_probs = safe_probs / safe_probs.sum(dim=-1, keepdim=True)
    if forced_action is not None:
        action = forced_action.reshape(-1).to(
            device=distribution.probs.device, dtype=torch.long
        )
        if action.shape != (event.payload.shape[0],):
            raise ValueError("forced action has the wrong shape")
        if bool(torch.any((action < 0) | (action >= distribution.probs.shape[-1]))):
            raise ValueError("forced action is outside the action vocabulary")
        propensity = safe_probs.gather(1, action[:, None]).squeeze(1)
        log_probability = propensity.log()
    elif uniform is None:
        action = distribution.sample()
        propensity = safe_probs.gather(1, action[:, None]).squeeze(1)
        log_probability = propensity.log()
    else:
        uniform = uniform.reshape(-1).to(
            device=distribution.probs.device, dtype=distribution.probs.dtype
        )
        if uniform.shape != (event.payload.shape[0],):
            raise ValueError("action sampling uniform has the wrong shape")
        if not bool(torch.isfinite(uniform).all()) or bool(
            torch.any((uniform < 0.0) | (uniform > 1.0))
        ):
            raise ValueError("action sampling uniform must lie in [0, 1]")
        action = (
            uniform[:, None] >= distribution.probs.cumsum(dim=-1)
        ).sum(dim=-1).clamp_max(distribution.probs.shape[-1] - 1)
        propensity = safe_probs.gather(1, action[:, None]).squeeze(1)
        log_probability = propensity.log()
    return action, propensity, log_probability, state


def _store_outcome(
    runtime: AmodalControllerRuntime,
    state: Any,
    action: torch.Tensor,
    reward: torch.Tensor,
    propensity: torch.Tensor,
    scope: torch.Tensor,
    sample_memory_writes: bool = False,
    memory_write_override: torch.Tensor | None = None,
    memory_write_uniform: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    opaque_action = torch.nn.functional.one_hot(action, num_classes=2).to(torch.float32)
    output, state = runtime.step_events(
        _empty(action.shape[0], runtime.event_width),
        state,
        _feedback(
            action.shape[0],
            action=opaque_action,
            reward=reward,
            propensity=propensity,
            has_feedback=torch.ones(action.shape[0]),
        ),
        memory_scope=scope,
        sample_memory_writes=sample_memory_writes,
        memory_write_override=memory_write_override,
        memory_write_uniform=memory_write_uniform,
    )
    receipt = output.controller.memory_write_receipt
    assert receipt is not None
    return (
        state,
        output.controller.memory_write_strength,
        receipt.committed,
        output.controller.memory_write_log_probability,
        torch.cat(
            [
                state.hidden,
                output.controller.memory_key,
                output.controller.memory_value,
                output.controller.memory_write_strength.unsqueeze(-1),
            ],
            dim=-1,
        ).detach(),
    )


def _paired_uniforms(batch: int, *, antithetic: bool = True) -> torch.Tensor:
    """Return shared or antithetic uniforms for two coupled arms per row."""
    base = torch.rand(batch)
    second = 1.0 - base if antithetic else base
    return torch.stack((base, second), dim=1).reshape(-1)


def _retention_slots(
    verifier: OutcomeOnlyRetentionVerifier,
    retention_order: str,
) -> torch.Tensor:
    """Return a verifier-controlled presentation order without exposing it.

    The order is trainer/environment state. The controller receives only the
    resulting opaque token events. ``balanced`` cycles the target position so
    the write policy cannot pass by always preferring an early or late event;
    it works for both the original two-slot rung and larger bounded banks.
    """
    if retention_order not in {"random", "balanced", "target_first", "target_last"}:
        raise ValueError("unknown retention order curriculum")
    batch = verifier.batch_size
    target = verifier.query_slot
    order = verifier.order
    if retention_order == "random":
        return order
    distractors = order.masked_select(order != target[:, None]).reshape(
        batch, verifier.slot_count - 1
    )
    if retention_order == "target_first":
        return torch.cat((target[:, None], distractors), dim=1)
    if retention_order == "target_last":
        return torch.cat((distractors, target[:, None]), dim=1)

    target_position = verifier.balanced_position
    positions = torch.arange(verifier.slot_count, device=target.device).expand(
        batch, -1
    )
    distractor_index = positions - (positions > target_position[:, None]).to(
        torch.long
    )
    distractor_index = distractor_index.clamp_max(verifier.slot_count - 2)
    balanced = distractors.gather(1, distractor_index)
    return torch.where(
        positions == target_position[:, None], target[:, None], balanced
    )


def _reset_memory_write_policy_output(runtime: AmodalControllerRuntime) -> None:
    """Restore a neutral write prior without resetting the parent controller."""
    output_layer = runtime.controller.memory_write_policy[-1]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        nn.init.normal_(output_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(output_layer.bias)


def _reset_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[nn.Parameter, ...],
) -> None:
    """Drop stale adaptive moments when a phase resets a policy head."""
    for parameter in parameters:
        optimizer.state.pop(parameter, None)


def _set_memory_write_policy_trainable(
    runtime: AmodalControllerRuntime,
    trainable: bool,
) -> None:
    """Route phase updates through or around the generic write policy."""
    for parameter in runtime.controller.memory_write_policy.parameters():
        parameter.requires_grad = trainable


def _counterfactual_retention_episode(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    baseline: float,
    reward_shuffle: bool,
    retention_order: str,
    memory_write_cost: float,
    differentiable_memory: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, float]:
    """Train two common-random-number arms against the same hidden world.

    The duplicated verifier rows exist only in this trainer. Each runtime arm
    gets its own memory scope and sees only ordinary opaque events, sampled
    actions, and scalar outcomes. Pair-centering the final reward supplies a
    counterfactual write credit signal without exposing the target bit or
    target slot to the controller.
    """
    verifier.reset()
    batch = verifier.batch_size
    paired_verifier = verifier.duplicate_rows(2)
    paired_batch = paired_verifier.batch_size
    scope = torch.arange(paired_batch, dtype=torch.long)
    runtime.memory.clear()
    state = runtime.initial_state(paired_batch, device="cpu")
    cue = tokens[paired_verifier.query_slot]
    _, state = runtime.controller.step(
        _event(cue), state, _feedback(paired_batch), memory=None
    )

    slots = _retention_slots(paired_verifier, retention_order)

    strengths: list[torch.Tensor] = []
    probe_log_probabilities: list[torch.Tensor] = []
    write_log_probabilities: list[torch.Tensor] = []
    context = (
        runtime.memory.differentiable_transaction()
        if differentiable_memory
        else nullcontext()
    )
    with context:
        for position in range(slots.shape[1]):
            slot = slots[:, position]
            action, propensity, probe_log_probability, state = _probe_without_writing(
                runtime,
                state,
                _event(tokens[slot]),
                uniform=_paired_uniforms(batch, antithetic=False),
            )
            reward = paired_verifier.score_probe(slot, action)
            if reward_shuffle:
                reward = torch.randint(0, 2, (batch,)).repeat_interleave(2).to(
                    torch.float32
                )
            (
                state,
                write_strength,
                committed,
                write_log_probability,
                _critic_features,
            ) = _store_outcome(
                runtime,
                state,
                action,
                reward,
                propensity,
                scope,
                sample_memory_writes=False,
                memory_write_uniform=_paired_uniforms(batch),
            )
            assert write_log_probability is not None
            strengths.append(write_strength)
            probe_log_probabilities.append(probe_log_probability)
            write_log_probabilities.append(write_log_probability)

        query_output, _ = runtime.step_events(
            _event(tokens[paired_verifier.query_slot]),
            runtime.initial_state(paired_batch, device="cpu"),
            _feedback(paired_batch),
            memory_scope=scope,
        )
        distribution = Categorical(logits=query_output.decoded["protocol"])
        uniform = _paired_uniforms(batch, antithetic=False).to(
            device=distribution.probs.device, dtype=distribution.probs.dtype
        )
        action = (
            uniform[:, None] >= distribution.probs.cumsum(dim=-1)
        ).sum(dim=-1).clamp_max(distribution.probs.shape[-1] - 1)
        reward = paired_verifier.score_recall(action)
        if reward_shuffle:
            reward = torch.randint(0, 2, (batch,)).repeat_interleave(2).to(
                torch.float32
            )
        recall_log_probability = distribution.log_prob(action)
        probe_log_probability = torch.stack(
            probe_log_probabilities, dim=1
        ).sum(dim=1)
        pair_rewards = reward.reshape(batch, 2)
        pair_advantage = (pair_rewards - pair_rewards.flip(dims=(1,))).reshape(-1)
        policy_log_probability = (
            recall_log_probability
            + probe_log_probability
            + torch.stack(write_log_probabilities, dim=1).sum(dim=1)
        )
        loss = -(
            pair_advantage.detach() * policy_log_probability
        ).mean() - 0.01 * distribution.entropy().mean()
        if memory_write_cost:
            loss = loss + memory_write_cost * torch.stack(strengths).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=5.0)
        optimizer.step()
    return (
        reward,
        strengths,
        committed,
        0.95 * baseline + 0.05 * float(reward.mean()),
    )


def _counterfactual_intervention_episode(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    baseline: float,
    reward_shuffle: bool,
    retention_order: str,
    memory_write_cost: float,
    differentiable_memory: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, float]:
    """Train the write policy from a paired write/skip intervention.

    One position is selected independently of the hidden target. Arm zero is
    forced to write there and arm one is forced to skip; all other positions
    skip for both arms. The controller receives no branch or verifier label.
    The scalar recall difference is a direct counterfactual utility signal for
    the generic write logit.
    """
    verifier.reset()
    batch = verifier.batch_size
    paired_verifier = verifier.duplicate_rows(2)
    paired_batch = paired_verifier.batch_size
    scope = torch.arange(paired_batch, dtype=torch.long)
    runtime.memory.clear()
    state = runtime.initial_state(paired_batch, device="cpu")
    _, state = runtime.controller.step(
        _event(tokens[paired_verifier.query_slot]),
        state,
        _feedback(paired_batch),
        memory=None,
    )

    slots = _retention_slots(paired_verifier, retention_order)

    branch_positions = (
        (torch.arange(batch) // slots.shape[1]) % slots.shape[1]
        if retention_order == "balanced"
        else torch.randint(0, slots.shape[1], (batch,))
    )
    paired_branch_positions = branch_positions.repeat_interleave(2)
    arm = torch.arange(paired_batch) % 2
    strengths: list[torch.Tensor] = []
    probe_log_probabilities: list[torch.Tensor] = []
    branch_logits: list[torch.Tensor] = []
    context = (
        runtime.memory.differentiable_transaction()
        if differentiable_memory
        else nullcontext()
    )
    with context:
        for position in range(slots.shape[1]):
            slot = slots[:, position]
            action, propensity, probe_log_probability, state = _probe_without_writing(
                runtime,
                state,
                _event(tokens[slot]),
                uniform=_paired_uniforms(batch, antithetic=False),
            )
            reward = paired_verifier.score_probe(slot, action)
            if reward_shuffle:
                reward = torch.randint(0, 2, (batch,)).repeat_interleave(2).to(
                    torch.float32
                )
            branch_mask = paired_branch_positions == position
            normal_uniform = _paired_uniforms(batch, antithetic=False)
            write_uniform = torch.where(
                branch_mask & (arm == 0),
                torch.zeros(paired_batch),
                torch.where(branch_mask, torch.ones(paired_batch), normal_uniform),
            )
            (
                state,
                write_strength,
                committed,
                _write_log_probability,
                _critic_features,
            ) = _store_outcome(
                runtime,
                state,
                action,
                reward,
                propensity,
                scope,
                sample_memory_writes=False,
                memory_write_uniform=write_uniform,
            )
            strengths.append(write_strength)
            probe_log_probabilities.append(probe_log_probability)
            branch_logits.append(
                torch.logit(write_strength.reshape(batch, 2)[:, 0].clamp(1e-6, 1.0 - 1e-6))
            )

        query_output, _ = runtime.step_events(
            _event(tokens[paired_verifier.query_slot]),
            runtime.initial_state(paired_batch, device="cpu"),
            _feedback(paired_batch),
            memory_scope=scope,
        )
        distribution = Categorical(logits=query_output.decoded["protocol"])
        uniform = _paired_uniforms(batch, antithetic=False).to(
            device=distribution.probs.device, dtype=distribution.probs.dtype
        )
        action = (
            uniform[:, None] >= distribution.probs.cumsum(dim=-1)
        ).sum(dim=-1).clamp_max(distribution.probs.shape[-1] - 1)
        reward = paired_verifier.score_recall(action)
        if reward_shuffle:
            reward = torch.randint(0, 2, (batch,)).repeat_interleave(2).to(
                torch.float32
            )
        pair_rewards = reward.reshape(batch, 2)
        pair_mean = pair_rewards.mean(dim=1).repeat_interleave(2)
        action_log_probability = distribution.log_prob(action) + torch.stack(
            probe_log_probabilities, dim=1
        ).sum(dim=1)
        action_loss = -(
            (reward.detach() - pair_mean.detach()) * action_log_probability
        ).mean()
        write_logit = torch.stack(branch_logits, dim=1).gather(
            1, branch_positions[:, None]
        ).squeeze(1)
        write_utility = pair_rewards[:, 0] - pair_rewards[:, 1]
        loss = action_loss - (write_utility.detach() * write_logit).mean()
        if memory_write_cost:
            loss = loss + memory_write_cost * torch.stack(strengths).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=5.0)
        optimizer.step()
    return (
        reward,
        strengths,
        committed,
        0.95 * baseline + 0.05 * float(reward.mean()),
    )


def _counterfactual_parent_episode(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    baseline: float,
    reward_shuffle: bool,
    memory_write_cost: float,
    differentiable_memory: bool,
    force_write: bool = False,
    include_probe_credit: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, float]:
    """Train the single-event parent from coupled forced-action outcomes.

    Each trainer-only pair uses the same hidden verifier world. One arm takes
    action zero and the other takes action one at both the probe and recall
    decisions. The controller receives only the resulting opaque action and
    scalar outcome streams. Pair reward differences provide an unbiased,
    low-variance action direction without exposing the correct action label.
    """
    verifier.reset()
    batch = verifier.batch_size
    paired_verifier = verifier.duplicate_rows(2)
    paired_batch = paired_verifier.batch_size
    scope = torch.arange(paired_batch, dtype=torch.long)
    forced_actions = torch.arange(paired_batch, dtype=torch.long) % 2
    runtime.memory.clear()
    state = runtime.initial_state(paired_batch, device="cpu")
    _, state = runtime.controller.step(
        _event(tokens[paired_verifier.query_slot]),
        state,
        _feedback(paired_batch),
        memory=None,
    )
    context = (
        runtime.memory.differentiable_transaction()
        if differentiable_memory
        else nullcontext()
    )
    strengths: list[torch.Tensor] = []
    probe_log_probabilities: list[torch.Tensor] = []
    write_log_probabilities: list[torch.Tensor] = []
    with context:
        action, propensity, probe_log_probability, state = _probe_without_writing(
            runtime,
            state,
            _event(tokens[paired_verifier.query_slot]),
            forced_action=forced_actions,
        )
        probe_reward = paired_verifier.score_probe(
            paired_verifier.query_slot, action
        )
        if reward_shuffle:
            probe_reward = torch.randint(0, 2, (paired_batch,)).to(torch.float32)
        write_override = (
            torch.ones(paired_batch) if force_write else None
        )
        write_uniform = None if force_write else _paired_uniforms(
            batch, antithetic=False
        )
        (
            state,
            write_strength,
            committed,
            write_log_probability,
            _critic_features,
        ) = _store_outcome(
            runtime,
            state,
            action,
            probe_reward,
            propensity,
            scope,
            sample_memory_writes=False,
            memory_write_override=write_override,
            memory_write_uniform=write_uniform,
        )
        strengths.append(write_strength)
        probe_log_probabilities.append(probe_log_probability)
        if write_log_probability is not None and not force_write:
            write_log_probabilities.append(write_log_probability)

        query_output, _ = runtime.step_events(
            _event(tokens[paired_verifier.query_slot]),
            runtime.initial_state(paired_batch, device="cpu"),
            _feedback(paired_batch),
            memory_scope=scope,
        )
        distribution = Categorical(logits=query_output.decoded["protocol"])
        recall_reward = paired_verifier.score_recall(forced_actions)
        if reward_shuffle:
            recall_reward = torch.randint(0, 2, (paired_batch,)).to(torch.float32)
        probe_pair = probe_reward.reshape(batch, 2)
        recall_pair = recall_reward.reshape(batch, 2)
        probe_log_pair = probe_log_probability.reshape(batch, 2)
        recall_log_pair = distribution.log_prob(forced_actions).reshape(batch, 2)
        probe_utility = probe_pair[:, 1] - probe_pair[:, 0]
        recall_utility = recall_pair[:, 1] - recall_pair[:, 0]
        loss_terms = [
            recall_utility.detach()
            * (recall_log_pair[:, 1] - recall_log_pair[:, 0])
        ]
        if include_probe_credit:
            loss_terms.insert(
                0,
                probe_utility.detach()
                * (probe_log_pair[:, 1] - probe_log_pair[:, 0]),
            )
        loss = -torch.stack(loss_terms, dim=0).sum(dim=0).mean()
        if write_log_probabilities:
            write_log_pair = torch.stack(write_log_probabilities, dim=1).reshape(
                batch, 2, -1
            )
            write_utility = recall_utility.detach().unsqueeze(1)
            loss = loss - (
                write_utility
                * (write_log_pair[:, 1, :] - write_log_pair[:, 0, :])
            ).mean()
        if memory_write_cost:
            loss = loss + memory_write_cost * torch.stack(strengths).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=5.0)
        optimizer.step()
    return (
        recall_reward,
        strengths,
        committed,
        0.95 * baseline + 0.05 * float(recall_reward.mean()),
    )


def _episode(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    scope: torch.Tensor,
    *,
    curriculum: str,
    reward_shuffle: bool,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    baseline: float = 0.5,
    reverse_order: bool = False,
    memory_condition: str = "intact",
    retention_order: str = "random",
    memory_write_cost: float = 0.0,
    stochastic_write_sampling: bool = True,
    differentiable_memory: bool = True,
    write_critic: OutcomeWriteCritic | None = None,
    critic_losses: list[float] | None = None,
    value_critic: OutcomeValueCritic | None = None,
    value_critic_losses: list[float] | None = None,
    diagnostics: dict[str, list[float]] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, float]:
    if curriculum not in {"single", "retention"}:
        raise ValueError("unknown retention curriculum")
    if memory_condition not in {
        "intact",
        "clear",
        "corrupt",
        "reverse_order",
        "random_action",
        "missing_write_cue",
        "missing_query_cue",
        "target_first",
        "target_last",
    }:
        raise ValueError("unknown memory condition")
    if retention_order not in {
        "random",
        "balanced",
        "target_first",
        "target_last",
    }:
        raise ValueError("unknown retention order curriculum")
    if memory_write_cost < 0.0:
        raise ValueError("memory write cost cannot be negative")
    verifier.reset()
    batch = verifier.batch_size
    runtime.memory.clear()
    state = runtime.initial_state(batch, device="cpu")
    if curriculum == "retention":
        cue = tokens[verifier.query_slot]
        if train or memory_condition != "missing_write_cue":
            _, state = runtime.controller.step(
                _event(cue), state, _feedback(batch), memory=None
            )
        slots = (
            verifier.order.flip(1)
            if reverse_order
            else _retention_slots(verifier, retention_order)
        )
    else:
        slots = verifier.query_slot[:, None]

    strengths: list[torch.Tensor] = []
    probe_log_probabilities: list[torch.Tensor] = []
    write_log_probabilities: list[torch.Tensor] = []
    critic_values: list[torch.Tensor] = []
    value_features: list[torch.Tensor] = []
    context = (
        runtime.memory.differentiable_transaction()
        if train and differentiable_memory
        else nullcontext()
        if train
        else torch.no_grad()
    )
    with context:
        for position in range(slots.shape[1]):
            slot = slots[:, position]
            action, propensity, probe_log_probability, state = _probe_without_writing(
                runtime, state, _event(tokens[slot])
            )
            if train and curriculum == "single" and value_critic is not None:
                value_features.append(state.hidden.detach())
            reward = verifier.score_probe(slot, action)
            if reward_shuffle:
                reward = torch.randint(0, 2, reward.shape).to(torch.float32)
            (
                state,
                write_strength,
                committed,
                write_log_probability,
                critic_features,
            ) = _store_outcome(
                runtime,
                state,
                action,
                reward,
                propensity,
                scope,
                sample_memory_writes=train and stochastic_write_sampling,
            )
            strengths.append(write_strength)
            probe_log_probabilities.append(probe_log_probability)
            if write_log_probability is not None:
                write_log_probabilities.append(write_log_probability)
            if write_critic is not None:
                write_value = write_critic(critic_features)
                critic_values.append(write_value)
            if diagnostics is not None:
                target = slot == verifier.query_slot
                for label, mask in (
                    ("target", target),
                    ("distractor", ~target),
                ):
                    selected_strength = write_strength.detach()[mask]
                    selected_commits = committed.detach()[mask].to(torch.float32)
                    diagnostics.setdefault(
                        f"{label}_strength", []
                    ).extend(float(value) for value in selected_strength)
                    diagnostics.setdefault(
                        f"{label}_commit", []
                    ).extend(float(value) for value in selected_commits)
        if not train and memory_condition == "clear":
            runtime.memory.clear()
        elif not train and memory_condition == "corrupt":
            runtime.memory.values.zero_()
        # Present the opaque target cue again at read time. Without this
        # second cue, resetting the recurrent state would also erase the
        # query key, forcing every target to share the same blank address and
        # making content-addressed selection impossible. The verifier still
        # withholds the target index and bit; the learner sees only the same
        # ordinary event token that was already part of the episode.
        query_events = (
            _empty(batch, runtime.event_width)
            if not train and memory_condition == "missing_query_cue"
            else _event(tokens[verifier.query_slot])
        )
        query_output, _ = runtime.step_events(
            query_events,
            runtime.initial_state(batch, device="cpu"),
            _feedback(batch),
            memory_scope=scope,
        )
        distribution = Categorical(logits=query_output.decoded["protocol"])
        action = (
            torch.randint(0, 2, (batch,))
            if not train and memory_condition == "random_action"
            else distribution.sample()
        )
        reward = verifier.score_recall(action)
        if train:
            recall_log_probability = distribution.log_prob(action)
            probe_log_probability = torch.stack(
                probe_log_probabilities, dim=1
            ).sum(dim=1)
            if value_critic is not None and curriculum == "single":
                policy_log_probability = recall_log_probability + probe_log_probability
                if write_log_probabilities:
                    policy_log_probability = policy_log_probability + torch.stack(
                        write_log_probabilities, dim=1
                    ).sum(dim=1)
                values = value_critic(
                    torch.stack(value_features, dim=1).mean(dim=1)
                )
                value_loss = 0.5 * (values - reward.detach()).pow(2).mean()
                loss = (
                    -((reward.detach() - values.detach()) * policy_log_probability).mean()
                    - 0.01 * distribution.entropy().mean()
                    + value_loss
                )
                if value_critic_losses is not None:
                    value_critic_losses.append(float(value_loss.detach()))
            elif write_critic is None:
                policy_log_probability = recall_log_probability + probe_log_probability
                if write_log_probabilities:
                    policy_log_probability = policy_log_probability + torch.stack(
                        write_log_probabilities, dim=1
                    ).sum(dim=1)
                loss = -(
                    (reward.detach() - baseline) * policy_log_probability
                ).mean() - 0.01 * distribution.entropy().mean()
            elif write_log_probabilities:
                loss = -(
                    (reward.detach() - baseline)
                    * (recall_log_probability + probe_log_probability)
                ).mean() - 0.01 * distribution.entropy().mean()
                values = torch.stack(critic_values, dim=1)
                log_probabilities = torch.stack(write_log_probabilities, dim=1)
                write_advantage = reward.detach().unsqueeze(1) - values.detach()
                write_policy_loss = -(write_advantage * log_probabilities).mean()
                critic_loss = 0.5 * (
                    values - reward.detach().unsqueeze(1)
                ).pow(2).mean()
                loss = loss + write_policy_loss + critic_loss
                if critic_losses is not None:
                    critic_losses.append(float(critic_loss.detach()))
            else:
                loss = -(
                    (reward.detach() - baseline)
                    * (recall_log_probability + probe_log_probability)
                ).mean() - 0.01 * distribution.entropy().mean()
            if memory_write_cost:
                loss = loss + memory_write_cost * torch.stack(strengths).mean()
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=5.0)
            optimizer.step()
    return (
        reward,
        strengths,
        committed,
        0.95 * baseline + 0.05 * float(reward.mean()),
    )


def train_curriculum(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    phase1_steps: int,
    phase2_steps: int,
    seed: int,
    reward_shuffle: bool,
    retention_order: str = "random",
    memory_write_cost: float = 0.0,
    parent_protection: str = "none",
    retention_warmup_steps: int = 0,
    stochastic_write_sampling: bool = True,
    differentiable_memory: bool = True,
    write_credit: str = "global_baseline",
    retention_write_policy_reset: bool = False,
    parent_rehearsal_steps: int = 0,
    parent_rehearsal_interval: int = 1,
    retention_early_stop_validations: int = 0,
    parent_credit: str = "policy_gradient",
    parent_write_policy_protection: str = "none",
    outcome_value_critic: bool = False,
    randomize_event_tokens: bool = False,
    retention_token_reuse_steps: int = 4,
    learning_rate: float = 2e-3,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    if min(phase1_steps, phase2_steps) < 1:
        raise ValueError("curriculum steps must be positive")
    seed_everything(seed)
    if parent_protection not in {"none", "write_policy"}:
        raise ValueError("unknown parent protection mode")
    if parent_write_policy_protection not in {"none", "freeze_parent_phase"}:
        raise ValueError("unknown parent write-policy protection mode")
    if write_credit not in {
        "global_baseline",
        "critic_v1",
        "counterfactual_v1",
        "counterfactual_v2",
    }:
        raise ValueError("unknown write credit mode")
    if write_credit == "counterfactual_v1" and not stochastic_write_sampling:
        raise ValueError("counterfactual write credit requires stochastic writes")
    if not 0 <= retention_warmup_steps <= phase2_steps:
        raise ValueError("retention warmup must be within phase2_steps")
    if parent_rehearsal_steps < 0:
        raise ValueError("parent rehearsal steps cannot be negative")
    if parent_rehearsal_interval < 1:
        raise ValueError("parent rehearsal interval must be positive")
    if retention_early_stop_validations < 0:
        raise ValueError("retention early-stop validations cannot be negative")
    if retention_token_reuse_steps < 1:
        raise ValueError("retention token reuse steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if parent_credit not in {
        "policy_gradient",
        "counterfactual_action",
        "counterfactual_action_fixed_write",
        "counterfactual_recall_fixed_write",
    }:
        raise ValueError("unknown parent credit mode")
    runtime.train()
    write_critic = (
        OutcomeWriteCritic(runtime.controller.width)
        if write_credit == "critic_v1"
        else None
    )
    value_critic = (
        OutcomeValueCritic(runtime.controller.width)
        if outcome_value_critic
        else None
    )
    memory_write_policy_parameters = tuple(
        runtime.controller.memory_write_policy.parameters()
    )
    optimizer_parameters = list(runtime.parameters())
    if write_critic is not None:
        optimizer_parameters.extend(write_critic.parameters())
    if value_critic is not None:
        optimizer_parameters.extend(value_critic.parameters())
    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    baseline = 0.5
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    strengths: list[float] = []
    critic_losses: list[float] = []
    value_critic_losses: list[float] = []
    commits = 0
    total_writes = 0
    phase1_updates = 0
    phase2_updates = 0
    parent_rehearsal_updates = 0
    parent_audit_count = 0
    parent_stable = False
    retention_blocked = False
    diagnostic_verifier_bits = 0
    diagnostic_logical_lifetimes = 0
    retention_validation_count = 0
    best_validation_metric: float | None = None
    best_validation_state: dict[str, torch.Tensor] | None = None
    best_validation_parent_eligible = False
    best_validation_parent_retention: float | None = None
    stable_parent_audits = 0
    retention_validation_streak = 0
    stop_after_retention = False
    retention_token_block: torch.Tensor | None = None
    retention_token_block_step = 0
    parent_audit_verifier = OutcomeOnlyRetentionVerifier(
        batch_size=verifier.batch_size,
        seed=seed + 100_000,
        slot_count=verifier.slot_count,
    )
    retention_write_policy_reset_done = False
    retention_validation_verifier = OutcomeOnlyRetentionVerifier(
        batch_size=verifier.batch_size,
        seed=seed + 200_000,
        slot_count=verifier.slot_count,
    )
    parent_rehearsal_verifier = OutcomeOnlyRetentionVerifier(
        batch_size=verifier.batch_size,
        seed=seed + 300_000,
        slot_count=verifier.slot_count,
    )
    curriculum_schedule = [("single", phase1_steps, retention_order)]
    if retention_warmup_steps:
        curriculum_schedule.append(
            ("retention_warmup", retention_warmup_steps, "target_last")
        )
    if phase2_steps > retention_warmup_steps:
        curriculum_schedule.append(
            ("retention", phase2_steps - retention_warmup_steps, retention_order)
        )
    for curriculum_label, steps, active_order in curriculum_schedule:
        curriculum = "single" if curriculum_label == "single" else "retention"
        if curriculum == "single":
            _set_memory_write_policy_trainable(
                runtime,
                parent_write_policy_protection != "freeze_parent_phase",
            )
        else:
            _set_memory_write_policy_trainable(runtime, True)
        if curriculum == "retention" and not parent_stable and not reward_shuffle:
            # Do not spend retention budget on an unqualified parent. The
            # reward-shuffled control is intentionally allowed through so it
            # can measure the null arm, but it can never promote.
            retention_blocked = True
            break
        if curriculum == "retention" and parent_protection == "write_policy":
            # Diagnostic protocol: preserve the mastered parent and expose
            # only the generic memory utility head to phase-transition
            # updates. This distinguishes credit-assignment failure from
            # destructive co-adaptation; it is not itself a capability claim.
            for name, parameter in runtime.named_parameters():
                parameter.requires_grad = name.startswith(
                    "controller.memory_write_policy."
                )
        if (
            curriculum == "retention"
            and retention_write_policy_reset
            and not retention_write_policy_reset_done
        ):
            _reset_memory_write_policy_output(runtime)
            _reset_optimizer_state(optimizer, memory_write_policy_parameters)
            retention_write_policy_reset_done = True
        for step in range(1, steps + 1):
            if curriculum_label == "retention_warmup":
                active_order = "target_first" if step % 2 else "target_last"
            # Fixed token identities make the narrow verifier vulnerable to a
            # lookup shortcut: the controller can memorize the two training
            # addresses instead of learning a reusable event/value interface.
            # Keep parent acquisition and the bounded retention warmup fixed
            # so the scalar policy and phase transition can qualify at the
            # established budget, then vary opaque payloads in the main
            # retention and rehearsal episodes. Reusing each random pair for
            # a short bounded block keeps the policy target stationary long
            # enough to learn while preserving token diversity across blocks;
            # the token remains unchanged within an episode's write and recall
            # appearances.
            if randomize_event_tokens and curriculum_label == "retention":
                if (
                    retention_token_block is None
                    or retention_token_block_step >= retention_token_reuse_steps
                ):
                    retention_token_block = torch.randn_like(tokens)
                    retention_token_block_step = 0
                episode_tokens = retention_token_block
                retention_token_block_step += 1
            else:
                episode_tokens = tokens
            if curriculum == "single" and parent_credit in {
                "counterfactual_action",
                "counterfactual_action_fixed_write",
                "counterfactual_recall_fixed_write",
            }:
                reward, gates, committed, baseline = _counterfactual_parent_episode(
                    runtime,
                    verifier,
                    episode_tokens,
                    optimizer=optimizer,
                    baseline=baseline,
                    reward_shuffle=reward_shuffle,
                    memory_write_cost=memory_write_cost,
                    differentiable_memory=True,
                    force_write=parent_credit
                    in {
                        "counterfactual_action_fixed_write",
                        "counterfactual_recall_fixed_write",
                    },
                    include_probe_credit=parent_credit
                    != "counterfactual_recall_fixed_write",
                )
            elif curriculum == "retention" and write_credit in {
                "counterfactual_v1",
                "counterfactual_v2",
            }:
                counterfactual_episode = (
                    _counterfactual_intervention_episode
                    if write_credit == "counterfactual_v2"
                    else _counterfactual_retention_episode
                )
                reward, gates, committed, baseline = counterfactual_episode(
                    runtime,
                    verifier,
                    episode_tokens,
                    optimizer=optimizer,
                    baseline=baseline,
                    reward_shuffle=reward_shuffle,
                    retention_order=active_order,
                    memory_write_cost=memory_write_cost,
                    differentiable_memory=differentiable_memory,
                )
            else:
                reward, gates, committed, baseline = _episode(
                    runtime,
                    verifier,
                    episode_tokens,
                    scope,
                    curriculum=curriculum,
                    reward_shuffle=reward_shuffle,
                    train=True,
                    optimizer=optimizer,
                    baseline=baseline,
                    retention_order=active_order,
                    memory_write_cost=memory_write_cost,
                    stochastic_write_sampling=stochastic_write_sampling,
                    differentiable_memory=(
                        differentiable_memory or curriculum == "single"
                    ),
                    write_critic=write_critic,
                    critic_losses=critic_losses,
                    value_critic=value_critic,
                    value_critic_losses=value_critic_losses,
                )
            strengths.extend(float(g.detach().mean()) for g in gates)
            commits += int(committed.sum())
            total_writes += committed.numel()
            if (
                curriculum == "retention"
                and parent_rehearsal_steps
                and step % parent_rehearsal_interval == 0
            ):
                for _ in range(parent_rehearsal_steps):
                    if parent_credit == "counterfactual_action":
                        (
                            _rehearsal_reward,
                            rehearsal_gates,
                            rehearsal_committed,
                            baseline,
                        ) = _counterfactual_parent_episode(
                            runtime,
                            parent_rehearsal_verifier,
                            episode_tokens,
                            optimizer=optimizer,
                            baseline=baseline,
                            reward_shuffle=False,
                            memory_write_cost=memory_write_cost,
                            differentiable_memory=True,
                            force_write=parent_credit
                            == "counterfactual_recall_fixed_write",
                            include_probe_credit=parent_credit
                            != "counterfactual_recall_fixed_write",
                        )
                    else:
                        (
                            _rehearsal_reward,
                            rehearsal_gates,
                            rehearsal_committed,
                            baseline,
                        ) = _episode(
                            runtime,
                            parent_rehearsal_verifier,
                            episode_tokens,
                            scope,
                            curriculum="single",
                            reward_shuffle=False,
                            train=True,
                            optimizer=optimizer,
                            baseline=baseline,
                            stochastic_write_sampling=stochastic_write_sampling,
                            differentiable_memory=True,
                            write_critic=write_critic,
                            critic_losses=critic_losses,
                            value_critic=value_critic,
                            value_critic_losses=value_critic_losses,
                        )
                    strengths.extend(
                        float(g.detach().mean()) for g in rehearsal_gates
                    )
                    commits += int(rehearsal_committed.sum())
                    total_writes += rehearsal_committed.numel()
                    parent_rehearsal_updates += 1
            if curriculum == "single":
                phase1_updates += 1
            else:
                phase2_updates += 1
            if step == 1 or step % 32 == 0 or step == steps:
                history.append(
                    {
                        "curriculum": curriculum_label,
                        "step": step,
                        "reward": float(reward.mean()),
                    }
                )
            if curriculum == "single" and step >= 32 and step % 32 == 0:
                parent_score = evaluate_parent_condition(
                    runtime,
                    parent_audit_verifier,
                    tokens,
                    episodes=8,
                )
                parent_audit_count += 1
                diagnostic_verifier_bits += 8 * verifier.batch_size * 2
                diagnostic_logical_lifetimes += 8 * verifier.batch_size
                stable_parent_audits = (
                    stable_parent_audits + 1 if parent_score >= 0.90 else 0
                )
                history.append(
                    {
                        "curriculum": "single_audit",
                        "step": step,
                        "reward": parent_score,
                    }
                )
                if stable_parent_audits >= 3:
                    parent_stable = True
                    break
            if (
                curriculum_label == "retention"
                and (step % 64 == 0 or step == steps)
            ):
                validation_intact = evaluate_condition(
                    runtime,
                    retention_validation_verifier,
                    tokens,
                    condition="intact",
                    episodes=16,
                )
                validation_clear = evaluate_condition(
                    runtime,
                    retention_validation_verifier,
                    tokens,
                    condition="clear",
                    episodes=16,
                )
                validation_target_first = evaluate_condition(
                    runtime,
                    retention_validation_verifier,
                    tokens,
                    condition="target_first",
                    episodes=16,
                )
                validation_target_last = evaluate_condition(
                    runtime,
                    retention_validation_verifier,
                    tokens,
                    condition="target_last",
                    episodes=16,
                )
                validation_missing_write = evaluate_condition(
                    runtime,
                    retention_validation_verifier,
                    tokens,
                    condition="missing_write_cue",
                    episodes=16,
                )
                validation_parent_retention = evaluate_parent_condition(
                    runtime,
                    parent_audit_verifier,
                    tokens,
                    episodes=8,
                )
                validation_order_score = min(
                    validation_target_first, validation_target_last
                )
                validation_gap = validation_order_score - validation_clear
                validation_cue_gain = (
                    validation_target_first - validation_missing_write
                )
                validation_metric = (
                    validation_order_score
                    + max(0.0, validation_gap)
                    + max(0.0, validation_cue_gain)
                )
                retention_validation_count += 1
                diagnostic_verifier_bits += (
                    16 * verifier.batch_size * (verifier.slot_count + 1) * 5
                )
                diagnostic_logical_lifetimes += 16 * verifier.batch_size * 5
                diagnostic_verifier_bits += 8 * verifier.batch_size * 2
                diagnostic_logical_lifetimes += 8 * verifier.batch_size
                validation_parent_eligible = validation_parent_retention >= 0.90
                if (
                    best_validation_metric is None
                    or validation_parent_eligible and not best_validation_parent_eligible
                    or (
                        validation_parent_eligible == best_validation_parent_eligible
                        and validation_metric > best_validation_metric
                    )
                ):
                    best_validation_metric = validation_metric
                    best_validation_parent_eligible = validation_parent_eligible
                    best_validation_parent_retention = validation_parent_retention
                    best_validation_state = {
                        name: value.detach().clone()
                        for name, value in runtime.state_dict().items()
                    }
                history.append(
                    {
                        "curriculum": "retention_validation",
                        "step": step,
                        "reward": validation_order_score,
                        "intact": validation_intact,
                        "target_first": validation_target_first,
                        "target_last": validation_target_last,
                        "missing_write_cue": validation_missing_write,
                        "causal_gap": validation_gap,
                        "cue_gain": validation_cue_gain,
                        "parent_retention": validation_parent_retention,
                        "parent_rehearsal_updates": parent_rehearsal_updates,
                    }
                )
                validation_passed = (
                    validation_order_score >= 0.70
                    and validation_gap >= 0.15
                    and validation_cue_gain >= 0.10
                    and validation_parent_retention >= 0.90
                )
                retention_validation_streak = (
                    retention_validation_streak + 1 if validation_passed else 0
                )
                if (
                    retention_early_stop_validations
                    and retention_validation_streak >= retention_early_stop_validations
                ):
                    stop_after_retention = True
                    break
        if stop_after_retention:
            break
    if best_validation_state is not None:
        runtime.load_state_dict(best_validation_state)
    for parameter in runtime.parameters():
        parameter.requires_grad = True
    counterfactual_arm_multiplier = (
        2 if write_credit in {"counterfactual_v1", "counterfactual_v2"} else 1
    )
    validation_records = [
        record
        for record in history
        if record.get("curriculum") == "retention_validation"
    ]
    stable_validation_step: int | None = None
    stable_rehearsal_updates = 0
    for index, record in enumerate(validation_records):
        if all(
            float(later["reward"]) >= 0.70
            and float(later["causal_gap"]) >= 0.15
            and float(later["cue_gain"]) >= 0.10
            and float(later["parent_retention"]) >= 0.90
            for later in validation_records[index:]
        ):
            stable_validation_step = int(record["step"])
            stable_rehearsal_updates = int(record.get("parent_rehearsal_updates", 0))
            break
    stable_bits_to_threshold = (
        phase1_updates * verifier.batch_size * 2
        + stable_rehearsal_updates * verifier.batch_size * 2
        + (retention_warmup_steps + stable_validation_step)
        * verifier.batch_size
        * (verifier.slot_count + 1)
        if stable_validation_step is not None
        else None
    )
    return history, {
        "unique_verifier_bits": phase1_updates * verifier.batch_size * 2
        + phase2_updates
        * verifier.batch_size
        * (verifier.slot_count + 1)
        + parent_rehearsal_updates * verifier.batch_size * 2,
        "unique_logical_lifetimes": (phase1_updates + phase2_updates)
        * verifier.batch_size
        + parent_rehearsal_updates * verifier.batch_size,
        "logical_lifetime_observations": phase1_updates * verifier.batch_size
        + phase2_updates * verifier.batch_size * counterfactual_arm_multiplier
        + parent_rehearsal_updates * verifier.batch_size,
        "optimizer_updates": phase1_updates
        + phase2_updates
        + parent_rehearsal_updates,
        "replayed_examples": 0,
        "verifier_outcome_events": phase1_updates * verifier.batch_size * 2
        + phase2_updates
        * verifier.batch_size
        * (verifier.slot_count + 1)
        * counterfactual_arm_multiplier,
        "parent_rehearsal_outcome_events": parent_rehearsal_updates
        * verifier.batch_size
        * 2,
        "feedback_events": phase1_updates * verifier.batch_size
        + phase2_updates
        * verifier.batch_size
        * verifier.slot_count
        * counterfactual_arm_multiplier,
        "parent_rehearsal_feedback_events": parent_rehearsal_updates
        * verifier.batch_size,
        "diagnostic_verifier_bits": diagnostic_verifier_bits,
        "diagnostic_logical_lifetimes": diagnostic_logical_lifetimes,
        "phase1_updates": phase1_updates,
        "phase2_updates": phase2_updates,
        "parent_rehearsal_updates": parent_rehearsal_updates,
        "parent_stable": parent_stable,
        "retention_blocked": retention_blocked,
        "retention_validation_count": retention_validation_count,
        "best_validation_metric": best_validation_metric,
        "best_validation_parent_retention": best_validation_parent_retention,
        "stable_validation_step": stable_validation_step,
        "stable_bits_to_threshold": stable_bits_to_threshold,
        "mean_memory_write_strength": sum(strengths) / max(1, len(strengths)),
        "committed_write_rate": commits / max(1, total_writes),
        "retention_order": retention_order,
        "memory_write_cost": memory_write_cost,
        "stochastic_write_sampling": stochastic_write_sampling,
        "differentiable_memory": differentiable_memory,
        "parent_differentiable_memory": True,
        "write_credit": write_credit,
        "parent_credit": parent_credit,
        "parent_write_policy_protection": parent_write_policy_protection,
        "retention_write_policy_optimizer_state_reset": (
            retention_write_policy_reset
        ),
        "counterfactual_arm_multiplier": counterfactual_arm_multiplier,
        "mean_write_critic_loss": (
            sum(critic_losses) / max(1, len(critic_losses))
            if critic_losses
            else None
        ),
        "mean_outcome_value_critic_loss": (
            sum(value_critic_losses) / max(1, len(value_critic_losses))
            if value_critic_losses
            else None
        ),
        "outcome_value_critic": outcome_value_critic,
        "parent_protection": parent_protection,
        "retention_warmup_steps": retention_warmup_steps,
        "retention_write_policy_reset": retention_write_policy_reset,
        "parent_rehearsal_steps": parent_rehearsal_steps,
        "parent_rehearsal_interval": parent_rehearsal_interval,
        "retention_early_stop_validations": retention_early_stop_validations,
        "retention_token_reuse_steps": retention_token_reuse_steps,
        "learning_rate": learning_rate,
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    condition: str,
    episodes: int = 64,
) -> float:
    valid = {
        "intact",
        "clear",
        "corrupt",
        "reverse_order",
        "random_action",
        "missing_write_cue",
        "missing_query_cue",
        "target_first",
        "target_last",
    }
    if condition not in valid:
        raise ValueError("unknown retention condition")
    runtime.eval()
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    for _ in range(episodes):
        reward, _, _, _ = _episode(
            runtime,
            verifier,
            tokens,
            scope,
            curriculum="retention",
            reward_shuffle=False,
            train=False,
            reverse_order=condition == "reverse_order",
            memory_condition=condition,
            retention_order=(
                "target_first"
                if condition == "target_first"
                else "target_last"
                if condition == "target_last"
                else "random"
            ),
        )
        total += float(reward.sum())
    runtime.train()
    return total / (episodes * verifier.batch_size)


@torch.no_grad()
def evaluate_unseen_token_population(
    runtime: AmodalControllerRuntime,
    *,
    batch_size: int,
    event_width: int,
    seed: int,
    slot_count: int = 2,
    pairs: int = 4,
    episodes: int = 16,
) -> dict[str, float | list[float]]:
    """Evaluate retention on several unseen opaque token populations.

    A single held-out pair is too weak: it can be lucky or expose a residual
    address shortcut.  This diagnostic creates token payloads privately and
    reports the per-pair scores; none of them enter learner-visible training.
    """
    if slot_count < 2 or pairs < 1 or episodes < 1:
        raise ValueError("slot count, token populations, and episodes must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scores: list[float] = []
    for pair_index in range(pairs):
        tokens = torch.randn(slot_count, event_width, generator=generator)
        verifier = OutcomeOnlyRetentionVerifier(
            batch_size=batch_size, seed=seed + 1 + pair_index, slot_count=slot_count
        )
        scores.append(
            evaluate_condition(
                runtime,
                verifier,
                tokens,
                condition="intact",
                episodes=episodes,
            )
        )
    return {
        "mean": sum(scores) / len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "per_pair": scores,
    }


@torch.no_grad()
def evaluate_parent_condition(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    episodes: int = 8,
) -> float:
    """Audit the mastered single-event parent before entering retention."""
    runtime.eval()
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    for _ in range(episodes):
        reward, _, _, _ = _episode(
            runtime,
            verifier,
            tokens,
            scope,
            curriculum="single",
            reward_shuffle=False,
            train=False,
        )
        total += float(reward.sum())
    runtime.train()
    return total / (episodes * verifier.batch_size)


@torch.no_grad()
def diagnose_write_policy(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    *,
    retention_order: str,
    episodes: int = 32,
) -> dict[str, float]:
    """Measure target-vs-distractor writes using verifier-private labels.

    This is a discarded diagnostic probe. The target label is used only after
    the controller has produced its write decision, never as an input or loss
    feature. It distinguishes a cue-conditioned utility policy from a generic
    first/last-position write shortcut.
    """
    if retention_order not in {"target_first", "target_last", "random", "balanced"}:
        raise ValueError("unknown retention order")
    runtime.eval()
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    diagnostics: dict[str, list[float]] = {}
    for _ in range(episodes):
        _episode(
            runtime,
            verifier,
            tokens,
            scope,
            curriculum="retention",
            reward_shuffle=False,
            train=False,
            retention_order=retention_order,
            diagnostics=diagnostics,
        )
    runtime.train()

    def mean(name: str) -> float:
        values = diagnostics.get(name, [])
        return sum(values) / max(1, len(values))

    return {
        "target_write_strength": mean("target_strength"),
        "distractor_write_strength": mean("distractor_strength"),
        "target_commit_rate": mean("target_commit"),
        "distractor_commit_rate": mean("distractor_commit"),
        "target_minus_distractor_strength": mean("target_strength")
        - mean("distractor_strength"),
        "target_minus_distractor_commit_rate": mean("target_commit")
        - mean("distractor_commit"),
    }


def run_experiment(
    *,
    phase1_steps: int,
    phase2_steps: int,
    seed: int,
    batch_size: int = 16,
    slot_count: int = 2,
    memory_capacity: int = 1,
    event_window_capacity: int = 3,
    report_out: Path | None = None,
    reward_shuffle: bool = False,
    retention_order: str = "random",
    memory_write_cost: float = 0.0,
    parent_protection: str = "none",
    retention_warmup_steps: int = 0,
    stochastic_write_sampling: bool = True,
    differentiable_memory: bool = True,
    write_credit: str = "global_baseline",
    retention_write_policy_reset: bool = False,
    parent_rehearsal_steps: int = 0,
    parent_rehearsal_interval: int = 1,
    retention_early_stop_validations: int = 0,
    parent_credit: str = "policy_gradient",
    parent_write_policy_protection: str = "none",
    transfer_phase1_steps: int = 0,
    transfer_phase2_steps: int = 0,
    transfer_fresh_seeds: int = 1,
    persistent_memory_audit: bool = False,
    outcome_value_critic: bool = False,
    memory_value_feedback: bool = True,
    identity_memory_address: bool = False,
    stable_memory_address: bool = True,
    randomize_event_tokens: bool = False,
    retention_token_reuse_steps: int = 4,
    learning_rate: float = 2e-3,
) -> dict[str, Any]:
    if (transfer_phase1_steps == 0) != (transfer_phase2_steps == 0):
        raise ValueError("transfer phase steps must both be zero or positive")
    if transfer_fresh_seeds < 1:
        raise ValueError("transfer fresh seed count must be positive")
    if slot_count < 2:
        raise ValueError("slot count must be at least two")
    if memory_capacity < 1:
        raise ValueError("memory capacity must be positive")
    if event_window_capacity < 1:
        raise ValueError("event window capacity must be positive")
    seed_everything(seed)
    runtime = build_runtime(
        seed=seed,
        batch_size=batch_size,
        memory_capacity=memory_capacity,
        event_window_capacity=max(event_window_capacity, slot_count + 1),
        memory_scope_capacity=(
            batch_size * 2
            if write_credit in {"counterfactual_v1", "counterfactual_v2"}
            else batch_size
        ),
        memory_value_feedback=memory_value_feedback,
        identity_memory_address=identity_memory_address,
        stable_memory_address=stable_memory_address,
    )
    tokens = torch.randn(slot_count, runtime.event_width)
    verifier = OutcomeOnlyRetentionVerifier(
        batch_size=batch_size, seed=seed + 10, slot_count=slot_count
    )
    history, accounting = train_curriculum(
        runtime,
        verifier,
        tokens,
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        seed=seed,
        reward_shuffle=reward_shuffle,
        retention_order=retention_order,
        memory_write_cost=memory_write_cost,
        parent_protection=parent_protection,
        retention_warmup_steps=retention_warmup_steps,
        stochastic_write_sampling=stochastic_write_sampling,
        differentiable_memory=differentiable_memory,
        write_credit=write_credit,
        retention_write_policy_reset=retention_write_policy_reset,
        parent_rehearsal_steps=parent_rehearsal_steps,
        parent_rehearsal_interval=parent_rehearsal_interval,
        retention_early_stop_validations=retention_early_stop_validations,
        parent_credit=parent_credit,
        parent_write_policy_protection=parent_write_policy_protection,
        outcome_value_critic=outcome_value_critic,
        randomize_event_tokens=randomize_event_tokens,
        retention_token_reuse_steps=retention_token_reuse_steps,
        learning_rate=learning_rate,
    )
    conditions = {
        condition: evaluate_condition(
            runtime,
            OutcomeOnlyRetentionVerifier(
                batch_size=batch_size,
                seed=seed + 200 + index,
                slot_count=slot_count,
            ),
            tokens,
            condition=condition,
        )
        for index, condition in enumerate(
            (
                "intact",
                "clear",
                "corrupt",
                "reverse_order",
                "random_action",
                "missing_write_cue",
                "missing_query_cue",
                "target_first",
                "target_last",
            )
        )
    }
    mastered_start = time.perf_counter()
    mastered_primitive_retention = evaluate_parent_condition(
        runtime,
        OutcomeOnlyRetentionVerifier(
            batch_size=batch_size,
            seed=seed + 300_000,
            slot_count=slot_count,
        ),
        tokens,
        episodes=16,
    )
    mastered_elapsed = time.perf_counter() - mastered_start
    unseen_token_population = evaluate_unseen_token_population(
        runtime,
        batch_size=batch_size,
        event_width=runtime.event_width,
        seed=seed + 400_000,
        slot_count=slot_count,
    )
    unseen_token_retention = float(unseen_token_population["mean"])
    unseen_token_retention_min = float(unseen_token_population["minimum"])
    unseen_token_retention_max = float(unseen_token_population["maximum"])
    unseen_token_pairs = len(unseen_token_population["per_pair"])
    accounting["diagnostic_verifier_bits"] = int(
        accounting["diagnostic_verifier_bits"]
    ) + unseen_token_pairs * 16 * batch_size * (slot_count + 1)
    accounting["diagnostic_logical_lifetimes"] = int(
        accounting["diagnostic_logical_lifetimes"]
    ) + unseen_token_pairs * 16 * batch_size
    accounting["mean_inference_latency_ms"] = (
        mastered_elapsed / (16 * batch_size) * 1000.0
    )
    accounting["retention_on_mastered_primitives"] = mastered_primitive_retention
    accounting["transfer_ratio_against_fresh_learner"] = None
    persistent_reload_intact: float | None = None
    persistent_corruption_rejected: bool | None = None
    persistent_recovery_intact: float | None = None
    if persistent_memory_audit:
        persistent_audit = evaluate_persistent_reload(
            runtime,
            OutcomeOnlyRetentionVerifier(
                batch_size=batch_size,
                seed=seed + 600_000,
                slot_count=slot_count,
            ),
            tokens,
            episodes=16,
        )
        persistent_reload_intact = float(persistent_audit["reload_intact_recall"])
        persistent_corruption_rejected = bool(
            persistent_audit["corruption_rejected"]
        )
        persistent_recovery_intact = float(
            persistent_audit["recovery_intact_recall"]
        )
        accounting["diagnostic_verifier_bits"] = int(
            accounting["diagnostic_verifier_bits"]
        ) + 16 * batch_size * 4
        accounting["diagnostic_logical_lifetimes"] = int(
            accounting["diagnostic_logical_lifetimes"]
        ) + 16 * batch_size
    transfer_report: dict[str, Any] | None = None
    if transfer_phase1_steps:
        transfer_generator = torch.Generator(device="cpu")
        transfer_generator.manual_seed(seed + 400_000)
        transfer_tokens_for_training = torch.randn(
            slot_count, runtime.event_width, generator=transfer_generator
        )
        transfer_verifier_seed = seed + 500_000
        transfer_verifier = OutcomeOnlyRetentionVerifier(
            batch_size=batch_size,
            seed=transfer_verifier_seed,
            slot_count=slot_count,
        )
        zero_shot_transfer = evaluate_condition(
            runtime,
            OutcomeOnlyRetentionVerifier(
                batch_size=batch_size,
                seed=seed + 500_001,
                slot_count=slot_count,
            ),
            transfer_tokens_for_training,
            condition="intact",
            episodes=16,
        )
        transferred_runtime = build_runtime(
            seed=seed + 500_002,
            batch_size=batch_size,
            memory_capacity=memory_capacity,
            event_window_capacity=max(event_window_capacity, slot_count + 1),
            memory_scope_capacity=batch_size * 2,
            memory_value_feedback=memory_value_feedback,
            identity_memory_address=identity_memory_address,
            stable_memory_address=stable_memory_address,
        )
        transferred_runtime.load_state_dict(runtime.state_dict())
        transfer_training_options = {
            "memory_write_cost": memory_write_cost,
            "parent_protection": parent_protection,
            "stochastic_write_sampling": stochastic_write_sampling,
            "differentiable_memory": differentiable_memory,
            "retention_write_policy_reset": retention_write_policy_reset,
            "parent_rehearsal_steps": parent_rehearsal_steps,
            "parent_rehearsal_interval": parent_rehearsal_interval,
            "retention_early_stop_validations": retention_early_stop_validations,
            "parent_credit": parent_credit,
            "parent_write_policy_protection": parent_write_policy_protection,
            "outcome_value_critic": outcome_value_critic,
            "randomize_event_tokens": randomize_event_tokens,
            "retention_token_reuse_steps": retention_token_reuse_steps,
            "learning_rate": learning_rate,
        }
        _, transferred_accounting = train_curriculum(
            transferred_runtime,
            transfer_verifier,
            transfer_tokens_for_training,
            phase1_steps=transfer_phase1_steps,
            phase2_steps=transfer_phase2_steps,
            seed=seed + 500_004,
            reward_shuffle=False,
            retention_order=retention_order,
            retention_warmup_steps=retention_warmup_steps,
            write_credit=write_credit,
            **transfer_training_options,
        )
        fresh_accountings: list[dict[str, object]] = []
        for fresh_index in range(transfer_fresh_seeds):
            fresh_runtime = build_runtime(
                seed=seed + 500_003 + fresh_index * 1_000,
                batch_size=batch_size,
                memory_capacity=memory_capacity,
                event_window_capacity=max(event_window_capacity, slot_count + 1),
                memory_scope_capacity=batch_size * 2,
                memory_value_feedback=memory_value_feedback,
                identity_memory_address=identity_memory_address,
                stable_memory_address=stable_memory_address,
            )
            fresh_verifier = OutcomeOnlyRetentionVerifier(
                batch_size=batch_size,
                seed=transfer_verifier_seed,
                slot_count=slot_count,
            )
            _, fresh_accounting = train_curriculum(
                fresh_runtime,
                fresh_verifier,
                transfer_tokens_for_training,
                phase1_steps=transfer_phase1_steps,
                phase2_steps=transfer_phase2_steps,
                seed=seed + 500_004 + fresh_index * 1_000,
                reward_shuffle=False,
                retention_order=retention_order,
                retention_warmup_steps=retention_warmup_steps,
                write_credit=write_credit,
                **transfer_training_options,
            )
            fresh_accountings.append(fresh_accounting)
        fresh_accounting = fresh_accountings[0]
        transferred_stable_bits = transferred_accounting["stable_bits_to_threshold"]
        fresh_stable_bits_by_seed = [
            accounting["stable_bits_to_threshold"]
            for accounting in fresh_accountings
        ]
        fresh_parent_stable_by_seed = [
            bool(accounting["parent_stable"]) for accounting in fresh_accountings
        ]
        all_fresh_stable = all(
            stable_bits is not None for stable_bits in fresh_stable_bits_by_seed
        )
        transfer_ratio = (
            sum(
                float(stable_bits) / float(transferred_stable_bits)
                for stable_bits in fresh_stable_bits_by_seed
            )
            / len(fresh_stable_bits_by_seed)
            if transferred_stable_bits is not None
            and all_fresh_stable
            and transferred_stable_bits > 0
            else None
        )
        if not bool(transferred_accounting["parent_stable"]):
            transfer_status = "transferred_parent_not_qualified"
        elif not all(fresh_parent_stable_by_seed):
            transfer_status = "fresh_parent_not_qualified"
        elif not all_fresh_stable:
            transfer_status = "fresh_retention_not_stable"
        else:
            transfer_status = "qualified"
        positive_transfer_gain = (
            transfer_ratio is not None and transfer_ratio > 1.0
        )
        accounting["transfer_ratio_against_fresh_learner"] = transfer_ratio
        transfer_report = {
            "unseen_event_token_zero_shot_intact": zero_shot_transfer,
            "phase1_steps": transfer_phase1_steps,
            "phase2_steps": transfer_phase2_steps,
            "fresh_initialization_seeds": transfer_fresh_seeds,
            "fresh_parent_stable_by_seed": fresh_parent_stable_by_seed,
            "fresh_stable_bits_to_thresholds": fresh_stable_bits_by_seed,
            "transfer_status": transfer_status,
            "transferred_stable_bits_to_threshold": transferred_stable_bits,
            "fresh_stable_bits_to_threshold": (
                fresh_stable_bits_by_seed[0]
                if transfer_fresh_seeds == 1
                else None
            ),
            "ratio_fresh_over_transferred_bits": transfer_ratio,
            "positive_transfer_gain": positive_transfer_gain,
            "transferred_accounting": transferred_accounting,
            "fresh_accounting": fresh_accounting,
            "fresh_accountings": fresh_accountings,
        }
    promotion = (
        not reward_shuffle
        and retention_order in {"random", "balanced"}
        and parent_protection == "none"
        and phase2_steps > retention_warmup_steps
        and bool(accounting["parent_stable"])
        and not bool(accounting["retention_blocked"])
        and accounting["stable_bits_to_threshold"] is not None
        and float(accounting["retention_on_mastered_primitives"]) >= 0.90
        and unseen_token_retention_min >= 0.70
        and conditions["intact"] >= 0.70
        and conditions["clear"] <= 0.60
        and conditions["corrupt"] <= 0.60
        and conditions["reverse_order"] >= 0.65
        and conditions["target_first"] >= 0.70
        and conditions["target_last"] >= 0.70
        and conditions["target_first"] - conditions["missing_write_cue"] >= 0.10
        and conditions["intact"] - conditions["clear"] >= 0.15
        and conditions["intact"] - conditions["corrupt"] >= 0.15
    )
    report: dict[str, Any] = {
        "experiment": "outcome-only-cue-guided-memory-retention",
        "seed": seed,
        "phase1_steps": phase1_steps,
        "phase2_steps": phase2_steps,
        "batch_size": batch_size,
        "slot_count": slot_count,
        "memory_capacity": memory_capacity,
        "event_window_capacity": max(event_window_capacity, slot_count + 1),
        "reward_shuffle_control": reward_shuffle,
        "learner_visible_inputs": [
            "opaque target cue event",
            "opaque slot event",
            "opaque probe action",
            "scalar probe outcome",
            "scalar recall outcome",
        ],
        "memory_training": {
            "capacity": memory_capacity,
            "slot_count": slot_count,
            "event_window_capacity": max(event_window_capacity, slot_count + 1),
            "scope_capacity": runtime.memory.scope_capacity,
            "write_threshold": 0.5,
            "differentiable_transaction": differentiable_memory,
            "parent_differentiable_transaction": True,
            "controller_runtime": runtime.controller.configuration()["schema"],
            "retention_order": retention_order,
            "memory_write_cost": memory_write_cost,
            "stochastic_write_sampling": stochastic_write_sampling,
            "write_credit": write_credit,
            "counterfactual_arm_multiplier": accounting[
                "counterfactual_arm_multiplier"
            ],
            "parent_protection": parent_protection,
            "retention_warmup_steps": retention_warmup_steps,
            "retention_write_policy_reset": retention_write_policy_reset,
            "parent_rehearsal_steps": parent_rehearsal_steps,
            "parent_rehearsal_interval": parent_rehearsal_interval,
            "parent_credit": parent_credit,
            "parent_write_policy_protection": parent_write_policy_protection,
            "retention_early_stop_validations": retention_early_stop_validations,
            "transfer_phase1_steps": transfer_phase1_steps,
            "transfer_phase2_steps": transfer_phase2_steps,
            "transfer_fresh_seeds": transfer_fresh_seeds,
            "outcome_value_critic": outcome_value_critic,
            "memory_value_feedback": memory_value_feedback,
            "identity_memory_address": identity_memory_address,
            "stable_memory_address": stable_memory_address,
            "randomize_event_tokens": randomize_event_tokens,
            "retention_token_reuse_steps": retention_token_reuse_steps,
            "learning_rate": learning_rate,
            "persistent_memory_audit": persistent_memory_audit,
            "persistent_reload_intact": persistent_reload_intact,
            "persistent_corruption_rejected": persistent_corruption_rejected,
            "persistent_recovery_intact": persistent_recovery_intact,
            "parent_stable": accounting["parent_stable"],
            "retention_blocked": accounting["retention_blocked"],
            "retention_validation_count": accounting["retention_validation_count"],
            "best_validation_metric": accounting["best_validation_metric"],
            "retention_on_mastered_primitives": mastered_primitive_retention,
            "retention_on_unseen_event_tokens": unseen_token_retention,
            "retention_on_unseen_event_tokens_min": unseen_token_retention_min,
            "retention_on_unseen_event_tokens_max": unseen_token_retention_max,
            "unseen_event_token_pairs": unseen_token_pairs,
            "retention_on_unseen_event_tokens_by_pair": unseen_token_population[
                "per_pair"
            ],
            "transfer_ratio_against_fresh_learner": accounting[
                "transfer_ratio_against_fresh_learner"
            ],
            "claim_boundary": "retention qualification only; no promotion without population replication",
        },
        "history": history,
        "conditions": conditions,
        "promotion_gate": {
            "intact_min": 0.70,
            "clear_max": 0.60,
            "corrupt_max": 0.60,
            "reverse_order_min": 0.65,
            "causal_gap_min": 0.15,
            "cue_gain_min": 0.10,
            "stable_validation_required": True,
            "retention_on_mastered_primitives_min": 0.90,
            "retention_on_unseen_event_tokens_min": 0.70,
        },
        "promoted": promotion,
        "transfer": transfer_report,
        "persistent_memory": {
            "audit_enabled": persistent_memory_audit,
            "reload_intact_recall": persistent_reload_intact,
            "corruption_rejected": persistent_corruption_rejected,
            "recovery_intact_recall": persistent_recovery_intact,
            "storage": "temporary_atomic_snapshot",
        },
        "accounting": accounting,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-steps", type=int, default=64)
    parser.add_argument("--phase2-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--slot-count", type=int, default=2)
    parser.add_argument("--memory-capacity", type=int, default=1)
    parser.add_argument("--event-window-capacity", type=int, default=3)
    parser.add_argument("--reward-shuffle", action="store_true")
    parser.add_argument(
        "--retention-order",
        choices=("random", "balanced", "target_first", "target_last"),
        default="random",
    )
    parser.add_argument("--memory-write-cost", type=float, default=0.0)
    parser.add_argument(
        "--deterministic-memory-writes",
        action="store_false",
        dest="stochastic_write_sampling",
        default=True,
    )
    parser.add_argument(
        "--write-credit",
        choices=(
            "global_baseline",
            "critic_v1",
            "counterfactual_v1",
            "counterfactual_v2",
        ),
        default="global_baseline",
    )
    parser.add_argument(
        "--hard-memory-training",
        action="store_false",
        dest="differentiable_memory",
        default=True,
    )
    parser.add_argument(
        "--reset-retention-write-policy",
        action="store_true",
        dest="retention_write_policy_reset",
        default=False,
    )
    parser.add_argument("--parent-rehearsal-steps", type=int, default=0)
    parser.add_argument("--parent-rehearsal-interval", type=int, default=1)
    parser.add_argument("--retention-early-stop-validations", type=int, default=0)
    parser.add_argument(
        "--parent-credit",
        choices=(
            "policy_gradient",
            "counterfactual_action",
            "counterfactual_action_fixed_write",
            "counterfactual_recall_fixed_write",
        ),
        default="policy_gradient",
    )
    parser.add_argument(
        "--parent-write-policy-protection",
        choices=("none", "freeze_parent_phase"),
        default="none",
    )
    parser.add_argument("--transfer-phase1-steps", type=int, default=0)
    parser.add_argument("--transfer-phase2-steps", type=int, default=0)
    parser.add_argument("--transfer-fresh-seeds", type=int, default=1)
    parser.add_argument("--outcome-value-critic", action="store_true")
    parser.add_argument(
        "--memory-value-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--identity-memory-address", action="store_true")
    parser.add_argument("--randomize-event-tokens", action="store_true")
    parser.add_argument("--retention-token-reuse-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--persistent-memory-audit", action="store_true")
    parser.add_argument(
        "--parent-protection",
        choices=("none", "write_policy"),
        default="none",
    )
    parser.add_argument("--retention-warmup-steps", type=int, default=0)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
