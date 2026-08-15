"""Outcome-only training utilities for the canonical Neural Workshop pilot."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_computer import AdaptiveOnlineEpisodicRelationReader, ContentAddressedMemory

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent


@dataclass(frozen=True)
class RewardOnlyUpdate:
    """One fresh-lifetime optimizer update."""

    update: int
    loss: float
    eligible_accuracy: float
    unique_verifier_bits: int
    replayed_examples: int
    selected_slot_fraction: float = 0.0


def evaluate_policy(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    batch_size: int,
    seeds: tuple[int, ...],
    steps: int | None = None,
    time_shuffle: bool = False,
    reset_history: bool = False,
    record_retention: bool = False,
    exploration_probability: float = 0.0,
    learned_route: bool = False,
    persistent_route: bool = False,
    context_route: bool = False,
    cue_symbol: int | None = None,
    record_context_route: bool = False,
) -> float:
    """Evaluate fresh lifetimes under a causal control condition.

    Controls and ordinary diagnostics do not enter the retention ledger by
    default. A caller should record only a deliberate post-acquisition
    retention audit.
    """

    if not seeds:
        raise ValueError("evaluation needs at least one seed")
    scores: list[torch.Tensor] = []
    for seed in seeds:
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            seed=seed,
            time_shuffle=time_shuffle,
            cue_symbol=cue_symbol,
        )
        with torch.no_grad():
            rollout = agent.rollout(
                verifier,
                sample=False,
                reset_history=reset_history,
                record_retention=record_retention,
                exploration_probability=exploration_probability,
                learned_route=learned_route,
                persistent_route=persistent_route,
                context_route=context_route,
                record_context_route=record_context_route,
            )
        scores.append(rollout.eligible_accuracy.mean())
    return float(torch.stack(scores).mean())


def audit_retention(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    batch_size: int,
    seeds: tuple[int, ...],
    steps: int | None = None,
) -> list[float]:
    """Measure fresh post-acquisition lifetimes and write the retention audit.

    This is intentionally separate from causal controls. Only successful
    candidate evaluations should advance the ledger's stable-prefix gate.
    """

    if not seeds:
        raise ValueError("retention audit needs at least one seed")
    scores: list[float] = []
    for seed in seeds:
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            seed=seed,
        )
        with torch.no_grad():
            rollout = agent.rollout(
                verifier,
                sample=False,
                record_retention=True,
            )
        scores.append(float(rollout.eligible_accuracy.mean()))
    return scores


def freeze_shared_path(agent: CanonicalBrainWorkshopAgent) -> None:
    """Freeze the controller and frontend, leaving external growth trainable."""

    for parameter in agent.parameters():
        parameter.requires_grad_(False)


def train_reward_only(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    updates: int,
    batch_size: int,
    steps: int | None = None,
    seed: int = 0,
    learning_rate: float = 3e-3,
    train_output: bool = True,
    context_route: bool = False,
    cue_symbol: int | None = None,
    record_context_route: bool = False,
) -> list[RewardOnlyUpdate]:
    """Train only external episodic/output state from fresh scalar outcomes.

    The controller, event frontend, and action-feedback encoder remain frozen.
    Each update creates new verifier lifetimes and clears transient memory;
    neither old trajectories nor verifier targets are replayed into the loss.
    """

    if min(n_back, updates, batch_size) < 1:
        raise ValueError("n-back, updates, and batch size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    freeze_shared_path(agent)
    for parameter in agent.external_reader.parameters():
        parameter.requires_grad_(True)
    for parameter in agent.intent_adapter.parameters():
        parameter.requires_grad_(True)
    if train_output:
        for parameter in agent.keypress_decoder.parameters():
            parameter.requires_grad_(True)
    trainable = [
        parameter
        for module in (
            agent.external_reader,
            agent.intent_adapter,
            agent.keypress_decoder if train_output else None,
        )
        if module is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[RewardOnlyUpdate] = []
    for update in range(updates):
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            seed=seed + update,
            cue_symbol=cue_symbol,
        )
        rollout = agent.rollout(
            verifier,
            sample=True,
            record_retention=False,
            context_route=context_route,
            record_context_route=record_context_route,
        )
        eligible = rollout.eligible.to(rollout.rewards.dtype)
        log_propensity = rollout.propensities.clamp_min(1e-8).log()
        advantage = (rollout.rewards - 0.5).detach()
        denominator = eligible.sum().clamp_min(1.0)
        loss = -((advantage * log_propensity * eligible).sum() / denominator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            RewardOnlyUpdate(
                update=update + 1,
                loss=float(loss.detach()),
                eligible_accuracy=float(rollout.eligible_accuracy.mean().detach()),
                unique_verifier_bits=batch_size * verifier.eligible_trials,
                replayed_examples=rollout.replayed_examples,
                selected_slot_fraction=float(
                    (rollout.selected_slots == 0).to(torch.float32).mean()
                ),
            )
        )
    return history


def _train_relation_extension(
    agent: CanonicalBrainWorkshopAgent,
    *,
    slot: int,
    verifier_n_back: int,
    updates: int,
    batch_size: int,
    steps: int | None = None,
    seed: int = 0,
    learning_rate: float = 3e-3,
    exploration_probability: float = 0.25,
    forced_slot: int | None = None,
    learned_route: bool = False,
    persistent_route: bool = False,
    context_route: bool = False,
    cue_symbol: int | None = None,
    record_context_route: bool = False,
) -> tuple[int, list[RewardOnlyUpdate]]:
    """Train one external relation extension from fresh scalar outcomes."""

    if min(verifier_n_back, updates, batch_size) < 1:
        raise ValueError("verifier n-back, updates, and batch size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    freeze_shared_path(agent)
    extension = agent.extensions[slot - 1]
    for parameter in extension.parameters():
        parameter.requires_grad_(True)
    if not learned_route:
        for parameter in extension.route_score.parameters():
            parameter.requires_grad_(False)
    decoder = agent.extension_decoder(slot)
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    trainable = [
        parameter
        for module in (extension, decoder)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[RewardOnlyUpdate] = []
    for update in range(updates):
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=verifier_n_back,
            steps=steps,
            seed=seed + update,
            cue_symbol=cue_symbol,
        )
        rollout = agent.rollout(
            verifier,
            sample=True,
            record_retention=False,
            exploration_probability=exploration_probability,
            forced_slot=forced_slot,
            learned_route=learned_route,
            persistent_route=persistent_route,
            context_route=context_route,
            record_context_route=record_context_route,
        )
        eligible = rollout.eligible.to(rollout.rewards.dtype)
        log_propensity = rollout.propensities.clamp_min(1e-8).log()
        advantage = (rollout.rewards - 0.5).detach()
        denominator = eligible.sum().clamp_min(1.0)
        loss = -((advantage * log_propensity * eligible).sum() / denominator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            RewardOnlyUpdate(
                update=update + 1,
                loss=float(loss.detach()),
                eligible_accuracy=float(rollout.eligible_accuracy.mean().detach()),
                unique_verifier_bits=batch_size * verifier.eligible_trials,
                replayed_examples=rollout.replayed_examples,
                selected_slot_fraction=float(
                    (rollout.selected_slots == slot).to(torch.float32).mean()
                ),
            )
        )
    return slot, history


def train_isolated_relation_capability(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    updates: int,
    batch_size: int,
    steps: int | None = None,
    seed: int = 0,
    learning_rate: float = 3e-3,
    exploration_probability: float = 0.25,
    learned_route: bool = False,
    persistent_route: bool = False,
    context_route: bool = False,
    cue_symbol: int | None = None,
    record_context_route: bool = False,
) -> tuple[int, list[RewardOnlyUpdate]]:
    """Append and train a benchmark-sized relation capability.

    The extension is selected only after the currently selected slot produces
    an eligible scalar failure. The trainer never receives a correct action,
    task ID, or unattempted-action outcome, and old slots receive no optimizer
    updates or replayed examples.
    """

    slot = agent.add_relation_capability(n_back=n_back, seed=seed)
    return _train_relation_extension(
        agent,
        slot=slot,
        verifier_n_back=n_back,
        updates=updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        learning_rate=learning_rate,
        exploration_probability=exploration_probability,
        learned_route=learned_route,
        persistent_route=persistent_route,
        context_route=context_route,
        cue_symbol=cue_symbol,
        record_context_route=record_context_route,
    )


def train_adaptive_relation_capability(
    agent: CanonicalBrainWorkshopAgent,
    *,
    verifier_n_back: int,
    memory_capacity: int,
    updates: int,
    batch_size: int,
    steps: int | None = None,
    seed: int = 0,
    learning_rate: float = 3e-3,
    exploration_probability: float = 0.25,
    learned_route: bool = False,
    persistent_route: bool = False,
    context_route: bool = False,
    cue_symbol: int | None = None,
    record_context_route: bool = False,
) -> tuple[int, list[RewardOnlyUpdate]]:
    """Acquire a capability with only a generic bounded event window.

    ``verifier_n_back`` configures the benchmark's hidden verifier and is not
    passed to the external capability or the deployed controller. The
    capability is provisioned solely with ``memory_capacity`` and learns from
    fresh opaque actions plus deterministic scalar outcomes.
    """

    if memory_capacity < 1:
        raise ValueError("adaptive capability memory capacity must be positive")
    slot = agent.add_adaptive_relation_capability(
        memory_capacity=memory_capacity,
        seed=seed,
    )
    return _train_relation_extension(
        agent,
        slot=slot,
        verifier_n_back=verifier_n_back,
        updates=updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        learning_rate=learning_rate,
        exploration_probability=exploration_probability,
        learned_route=learned_route,
        persistent_route=persistent_route,
        context_route=context_route,
        cue_symbol=cue_symbol,
        record_context_route=record_context_route,
    )


def train_existing_adaptive_relation_capability(
    agent: CanonicalBrainWorkshopAgent,
    *,
    slot: int,
    verifier_n_back: int,
    updates: int,
    batch_size: int,
    steps: int | None = None,
    seed: int = 0,
    learning_rate: float = 3e-3,
    forced_slot: bool = True,
    cue_symbol: int | None = None,
) -> list[RewardOnlyUpdate]:
    """Continue a grown adaptive slot from fresh verifier lifetimes.

    The slot is selected as an opaque candidate; ``verifier_n_back`` remains
    private benchmark configuration and is never passed into the capability.
    """

    if slot < 1 or slot > len(agent.extensions):
        raise IndexError("adaptive capability slot is outside the bank")
    if not isinstance(
        agent.extensions[slot - 1].reader,
        AdaptiveOnlineEpisodicRelationReader,
    ):
        raise TypeError("existing capability is not adaptive")
    _, history = _train_relation_extension(
        agent,
        slot=slot,
        verifier_n_back=verifier_n_back,
        updates=updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        learning_rate=learning_rate,
        exploration_probability=0.0,
        forced_slot=slot if forced_slot else None,
        cue_symbol=cue_symbol,
    )
    return history
