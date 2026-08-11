"""Acquire multiple temporal addresses under one rendered context.

The verifier keeps the same episode cue while its first learned event token
selects which temporal relation is required.  An external context-keyed route
table must therefore learn two offsets in one memory system: offset four for
one query token and offset five for another.  The source readout is frozen
before the second address is acquired, so the new capability is external
state growth rather than controller or readout fine-tuning.

Only learned event tensors, opaque attempted actions, and scalar verifier
outcomes cross the learner boundary.  Query values, relation depths, and
target actions remain private verifier state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from neural_computer import (
    ControllerFeedback,
    ExternalTemporalHistoryMemory,
    PersistentOpaqueContextRouteEvidence,
)

from .external_temporal_offset_growth import (
    ACTION_COUNT,
    EVENT_WIDTH,
    MASTERY_THRESHOLD,
    ExternalTemporalCapabilityFile,
    _build,
)

QUERY_ADDRESS_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-query-address-growth.v1"
)
DATA_STEPS = 14
MAX_OFFSET = 8
CUE_SYMBOL = 12
SOURCE_QUERY = 0
TARGET_QUERY = 1
UNKNOWN_QUERY = 2
SOURCE_DEPTH = 4
TARGET_DEPTH = 5
ROUTE_SELECTION_THRESHOLD = 0.99


class QueryConditionalNBackVerifier:
    """Private verifier with a constant cue and a rendered query token."""

    action_count = ACTION_COUNT

    def __init__(
        self,
        *,
        query_symbol: int,
        depth: int,
        batch_size: int,
        data_steps: int,
        seed: int,
    ) -> None:
        if query_symbol < 0 or depth < 1 or data_steps <= depth:
            raise ValueError("query verifier dimensions are invalid")
        if batch_size < 1:
            raise ValueError("query verifier batch size must be positive")
        self.query_symbol = int(query_symbol)
        self.depth = int(depth)
        self.batch_size = int(batch_size)
        self.data_steps = int(data_steps)
        self._generator = torch.Generator().manual_seed(seed)
        self._symbols = torch.randint(
            0,
            4,
            (batch_size, data_steps),
            generator=self._generator,
        )
        self._targets = torch.stack(
            tuple(
                self._symbols[:, position]
                == self._symbols[:, position - depth]
                for position in range(depth, data_steps)
            ),
            dim=1,
        )
        self._position = 0

    @property
    def done(self) -> bool:
        return self._position >= self.data_steps + 2

    @property
    def eligible_trials(self) -> int:
        return self.data_steps - self.depth

    @property
    def position(self) -> int:
        return self._position

    def observation(self) -> torch.Tensor:
        if self.done:
            raise RuntimeError("query verifier has no observations remaining")
        if self._position == 0:
            return torch.full(
                (self.batch_size,), CUE_SYMBOL, dtype=torch.long
            )
        if self._position == 1:
            return torch.full(
                (self.batch_size,),
                self.query_symbol,
                dtype=torch.long,
            )
        return self._symbols[:, self._position - 2].clone()

    def score(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape != (self.batch_size,) or action.dtype is not torch.long:
            raise ValueError("query verifier actions must be int64 [batch]")
        if bool(torch.any((action < 0) | (action >= ACTION_COUNT))):
            raise ValueError("query verifier action is outside the keypress domain")
        if self._position < 2:
            reward = torch.zeros(self.batch_size)
            eligible = torch.zeros(self.batch_size, dtype=torch.bool)
        else:
            data_position = self._position - 2
            if data_position < self.depth:
                reward = torch.zeros(self.batch_size)
                eligible = torch.zeros(self.batch_size, dtype=torch.bool)
            else:
                target_index = data_position - self.depth
                reward = (action == self._targets[:, target_index].long()).float()
                eligible = torch.ones(self.batch_size, dtype=torch.bool)
        self._position += 1
        return reward, eligible


@dataclass(frozen=True)
class QueryEpisode:
    loss: torch.Tensor
    accuracy: torch.Tensor
    context: torch.Tensor
    selected_offset: int
    eligible_bits: int


@dataclass(frozen=True)
class GlobalSourceEpisode:
    loss: torch.Tensor
    accuracy: torch.Tensor
    accuracy_by_row: torch.Tensor
    context: torch.Tensor
    selected_offsets: torch.Tensor
    eligible_bits: int


def _digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _choose_offset(
    evidence: PersistentOpaqueContextRouteEvidence,
    context: torch.Tensor,
    *,
    seed: int,
    explore: bool,
) -> int:
    preferred_slot = int(evidence.preferred_order(context)[0])
    if explore:
        generator = torch.Generator().manual_seed(seed + 72_901)
        if bool(torch.rand((), generator=generator) < 0.5):
            preferred_slot = int(
                torch.randint(MAX_OFFSET, (), generator=generator).item()
            )
    return preferred_slot + 1


def _episode(
    system,
    file: ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    query_symbol: int,
    depth: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    train: bool,
    explore: bool,
    forced_offset: int | None = None,
    reset_memory_each_step: bool = False,
) -> QueryEpisode:
    verifier = QueryConditionalNBackVerifier(
        query_symbol=query_symbol,
        depth=depth,
        batch_size=batch_size,
        data_steps=data_steps,
        seed=seed,
    )
    agent = system.agent
    controller_state = agent.initial_state(batch_size, device="cpu")
    feedback = agent.initial_feedback(batch_size, device="cpu")
    scope = torch.arange(batch_size, dtype=torch.long)
    history = ExternalTemporalHistoryMemory(EVENT_WIDTH, scope_capacity=batch_size)
    selected_offset: int | None = None
    context: torch.Tensor | None = None
    selected_logits: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    while not verifier.done:
        with torch.no_grad():
            collection = agent.runtime.encode_streams(
                {"stimulus": verifier.observation()}
            )
            controller_output, controller_state = agent.runtime.step_events(
                collection, controller_state, feedback
            )
        del controller_output
        event = collection.payload[:, 0].detach()
        history.append(event, scope=scope)
        if verifier.position == 1:
            context = event[0].clone()
            if forced_offset is None:
                selected_offset = _choose_offset(
                    evidence,
                    context,
                    seed=seed,
                    explore=explore,
                )
            else:
                if not 1 <= forced_offset <= MAX_OFFSET:
                    raise ValueError("forced temporal offset is outside the domain")
                selected_offset = forced_offset
        if selected_offset is None:
            read_offset = 1
        else:
            read_offset = selected_offset
        if reset_memory_each_step:
            history.clear()
        offsets = torch.full(
            (batch_size, 1), read_offset, dtype=torch.long
        )
        read = history.read_relative(offsets, scope=scope)
        retrieved = read.values[:, 0]
        present = read.present[:, 0].to(event.dtype).unsqueeze(-1)
        logits = file.readout(torch.cat((event, retrieved, present), dim=-1))
        probabilities = logits.softmax(dim=-1)
        if train:
            action = torch.multinomial(probabilities, 1).squeeze(-1)
        else:
            action = logits.argmax(dim=-1)
        propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        reward, is_eligible = verifier.score(action)
        selected_logits.append(logits.gather(1, action[:, None]).squeeze(1))
        rewards.append(reward)
        eligible.append(is_eligible)
        feedback = ControllerFeedback(
            action=agent.keypress_encoder(action),
            reward=reward,
            propensity=propensity,
            has_feedback=torch.ones(batch_size),
        )
    if context is None or selected_offset is None:
        raise RuntimeError("query episode did not expose a route context")
    reward_tensor = torch.stack(rewards, dim=1)
    eligible_tensor = torch.stack(eligible, dim=1)
    selected_tensor = torch.stack(selected_logits, dim=1)
    denominator = eligible_tensor.sum().clamp_min(1.0)
    accuracy = (reward_tensor * eligible_tensor).sum() / denominator
    if train:
        loss = F.binary_cross_entropy_with_logits(
            selected_tensor[eligible_tensor], reward_tensor[eligible_tensor]
        )
    else:
        loss = torch.zeros((), dtype=accuracy.dtype)
    return QueryEpisode(
        loss=loss,
        accuracy=accuracy,
        context=context,
        selected_offset=selected_offset,
        eligible_bits=int(eligible_tensor.sum().item()),
    )


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _global_source_episode(
    system,
    file: ExternalTemporalCapabilityFile,
    *,
    query_symbol: int,
    depth: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    train: bool,
    entropy_weight: float,
) -> GlobalSourceEpisode:
    """Train one source file with its generic scalar-credit offset policy."""

    verifier = QueryConditionalNBackVerifier(
        query_symbol=query_symbol,
        depth=depth,
        batch_size=batch_size,
        data_steps=data_steps,
        seed=seed,
    )
    agent = system.agent
    controller_state = agent.initial_state(batch_size, device="cpu")
    feedback = agent.initial_feedback(batch_size, device="cpu")
    scope = torch.arange(batch_size, dtype=torch.long)
    history = ExternalTemporalHistoryMemory(EVENT_WIDTH, scope_capacity=batch_size)
    selected_offsets: torch.Tensor | None = None
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    context: torch.Tensor | None = None
    selected_logits: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    while not verifier.done:
        with torch.no_grad():
            collection = agent.runtime.encode_streams(
                {"stimulus": verifier.observation()}
            )
            controller_output, controller_state = agent.runtime.step_events(
                collection, controller_state, feedback
            )
        del controller_output
        event = collection.payload[:, 0].detach()
        history.append(event, scope=scope)
        if verifier.position == 1:
            context = event[0].clone()
        if verifier.position >= 1:
            selected_offsets, log_probability, entropy = file.offset_selector(
                batch_size,
                sample=train,
            )
        else:
            selected_offsets = torch.ones(batch_size, dtype=torch.long)
            log_probability = torch.zeros(batch_size)
            entropy = torch.zeros(())
        log_probabilities.append(log_probability)
        entropies.append(entropy)
        offsets = selected_offsets
        read = history.read_relative(offsets[:, None], scope=scope)
        retrieved = read.values[:, 0]
        present = read.present[:, 0].to(event.dtype).unsqueeze(-1)
        logits = file.readout(torch.cat((event, retrieved, present), dim=-1))
        probabilities = logits.softmax(dim=-1)
        if train:
            action = torch.multinomial(probabilities, 1).squeeze(-1)
        else:
            action = logits.argmax(dim=-1)
        propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        reward, is_eligible = verifier.score(action)
        selected_logits.append(logits.gather(1, action[:, None]).squeeze(1))
        rewards.append(reward)
        eligible.append(is_eligible)
        feedback = ControllerFeedback(
            action=agent.keypress_encoder(action),
            reward=reward,
            propensity=propensity,
            has_feedback=torch.ones(batch_size),
        )
    if context is None or selected_offsets is None:
        raise RuntimeError("global source episode did not expose an address")
    reward_tensor = torch.stack(rewards, dim=1)
    eligible_tensor = torch.stack(eligible, dim=1)
    selected_tensor = torch.stack(selected_logits, dim=1)
    log_probability_tensor = torch.stack(log_probabilities, dim=1)
    entropy_tensor = torch.stack(entropies)
    row_denominator = eligible_tensor.sum(dim=1).clamp_min(1.0)
    accuracy_by_row = (
        (reward_tensor * eligible_tensor).sum(dim=1) / row_denominator
    )
    accuracy = accuracy_by_row.mean()
    if train:
        action_loss = F.binary_cross_entropy_with_logits(
            selected_tensor[eligible_tensor], reward_tensor[eligible_tensor]
        )
        offset_loss = -(
            ((reward_tensor - 0.5).detach() * log_probability_tensor * eligible_tensor)
            .sum()
            / eligible_tensor.sum().clamp_min(1.0)
        )
        entropy_loss = (
            (entropy_tensor[None, :] * eligible_tensor).sum()
            / eligible_tensor.sum().clamp_min(1.0)
        )
        loss = action_loss + offset_loss - entropy_weight * entropy_loss
    else:
        loss = torch.zeros((), dtype=accuracy.dtype)
    return GlobalSourceEpisode(
        loss=loss,
        accuracy=accuracy,
        accuracy_by_row=accuracy_by_row,
        context=context,
        selected_offsets=selected_offsets,
        eligible_bits=int(eligible_tensor.sum().item()),
    )


def _train_source(
    system,
    file: ExternalTemporalCapabilityFile,
    *,
    updates: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    learning_rate: float,
    entropy_weight: float,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(file.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        episode = _global_source_episode(
            system,
            file,
            query_symbol=SOURCE_QUERY,
            depth=SOURCE_DEPTH,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + update,
            train=True,
            entropy_weight=entropy_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        episode.loss.backward()
        nn.utils.clip_grad_norm_(file.parameters(), max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "accuracy": float(episode.accuracy.detach()),
                "selected_offset": int(torch.mode(episode.selected_offsets).values),
                "unique_verifier_bits": batch_size * episode.eligible_bits // batch_size,
                "replayed_examples": 0,
            }
        )
    return history


def _calibrate_source_route(
    system,
    file: ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    lifetimes: int,
    batch_size: int,
    data_steps: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Transfer only scalar outcomes from the frozen source address policy."""

    history: list[dict[str, float | int]] = []
    for lifetime in range(lifetimes):
        episode = _global_source_episode(
            system,
            file,
            query_symbol=SOURCE_QUERY,
            depth=SOURCE_DEPTH,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + lifetime,
            train=False,
            entropy_weight=0.0,
        )
        contexts = episode.context.expand(batch_size, -1)
        evidence.observe_batch(
            contexts,
            episode.selected_offsets - 1,
            episode.accuracy_by_row,
        )
        history.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(episode.accuracy),
                "selected_offset": int(torch.mode(episode.selected_offsets).values),
                "unique_verifier_bits": episode.eligible_bits,
                "replayed_examples": 0,
            }
        )
    return history


def _train_target_route(
    system,
    file: ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    updates: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    shuffled_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    delayed_outcome: float | None = None
    for update in range(1, updates + 1):
        episode = _episode(
            system,
            file,
            evidence,
            query_symbol=TARGET_QUERY,
            depth=TARGET_DEPTH,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + update,
            train=False,
            explore=True,
        )
        observed = 0.5 if delayed_outcome is None else delayed_outcome
        evidence.observe(
            episode.context,
            episode.selected_offset - 1,
            observed if shuffled_outcomes else episode.accuracy,
        )
        delayed_outcome = float(episode.accuracy)
        history.append(
            {
                "update": update,
                "accuracy": float(episode.accuracy),
                "selected_offset": episode.selected_offset,
                "observed_outcome": observed
                if shuffled_outcomes
                else float(episode.accuracy),
                "unique_verifier_bits": episode.eligible_bits,
                "replayed_examples": 0,
            }
        )
    return history


@torch.no_grad()
def _evaluate(
    system,
    file: ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    query_symbol: int,
    depth: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
    forced_offset: int | None = None,
    reset_memory_each_step: bool = False,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for lifetime in range(lifetimes):
        episode = _episode(
            system,
            file,
            evidence,
            query_symbol=query_symbol,
            depth=depth,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + lifetime,
            train=False,
            explore=False,
            forced_offset=forced_offset,
            reset_memory_each_step=reset_memory_each_step,
        )
        rows.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(episode.accuracy),
                "selected_offset": episode.selected_offset,
                "unique_verifier_bits": episode.eligible_bits,
                "replayed_examples": 0,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.route_calibration_lifetimes,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("query-address budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("query-address optimization parameters are invalid")
    if args.data_steps <= TARGET_DEPTH:
        raise ValueError("data steps must include n-back-5 target trials")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    file = ExternalTemporalCapabilityFile()
    evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    for _ in range(MAX_OFFSET):
        evidence.append_slot()
    source_history = _train_source(
        system,
        file,
        updates=args.source_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
    )
    source_route_history = _calibrate_source_route(
        system,
        file,
        evidence,
        lifetimes=args.route_calibration_lifetimes,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 15_000,
    )
    source_before = _evaluate(
        system,
        file,
        evidence,
        query_symbol=SOURCE_QUERY,
        depth=SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
    )
    source_file_digest = file.digest()
    readout_before_growth = file.digest()
    for parameter in file.parameters():
        parameter.requires_grad_(False)
    target_history = _train_target_route(
        system,
        file,
        evidence,
        updates=args.target_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 30_000,
    )
    source_after = _evaluate(
        system,
        file,
        evidence,
        query_symbol=SOURCE_QUERY,
        depth=SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
    )
    target_after = _evaluate(
        system,
        file,
        evidence,
        query_symbol=TARGET_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 40_000,
        lifetimes=args.retention_lifetimes,
    )
    unknown = _evaluate(
        system,
        file,
        evidence,
        query_symbol=UNKNOWN_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 50_000,
        lifetimes=args.retention_lifetimes,
    )
    wrong_offset = _evaluate(
        system,
        file,
        evidence,
        query_symbol=TARGET_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 60_000,
        lifetimes=args.retention_lifetimes,
        forced_offset=1,
    )
    missing_history = _evaluate(
        system,
        file,
        evidence,
        query_symbol=TARGET_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 70_000,
        lifetimes=args.retention_lifetimes,
        reset_memory_each_step=True,
    )
    shuffled_evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    for _ in range(MAX_OFFSET):
        shuffled_evidence.append_slot()
    shuffled_history = _train_target_route(
        system,
        file,
        shuffled_evidence,
        updates=args.target_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 80_000,
        shuffled_outcomes=True,
    )
    shuffled = _evaluate(
        system,
        file,
        shuffled_evidence,
        query_symbol=TARGET_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 90_000,
        lifetimes=args.retention_lifetimes,
    )
    restored = PersistentOpaqueContextRouteEvidence.from_payload(evidence.payload())
    restored_target = _evaluate(
        system,
        file,
        restored,
        query_symbol=TARGET_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 40_000,
        lifetimes=args.retention_lifetimes,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    target_offsets = [int(row["selected_offset"]) for row in target_after]
    source_offsets = [int(row["selected_offset"]) for row in source_after]
    shuffled_offsets = [int(row["selected_offset"]) for row in shuffled]
    gates = {
        "source_mastered_before_growth": _stable(source_before),
        "target_mastered_after_growth": _stable(target_after),
        "source_retained_after_growth": _stable(source_after),
        "source_offset_is_four": min(source_offsets) == SOURCE_DEPTH,
        "target_offset_is_five": min(target_offsets) == TARGET_DEPTH,
        "unknown_query_does_not_claim_mastery": not _stable(unknown),
        "wrong_offset_rejects_mastery": not _stable(wrong_offset),
        "missing_history_rejects_mastery": not _stable(missing_history),
        "shuffled_outcome_rejects_target_offset": min(shuffled_offsets)
        != TARGET_DEPTH,
        "route_reload_exact": target_after == restored_target,
        "readout_frozen_during_growth": readout_before_growth == file.digest(),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    source_bits = args.batch_size * args.source_updates * (
        args.data_steps - SOURCE_DEPTH
    )
    calibration_bits = args.batch_size * args.route_calibration_lifetimes * (
        args.data_steps - SOURCE_DEPTH
    )
    target_bits = args.batch_size * args.target_updates * (
        args.data_steps - TARGET_DEPTH
    )
    control_bits = target_bits
    audit_rows = (
        *source_before,
        *source_after,
        *target_after,
        *unknown,
        *wrong_offset,
        *missing_history,
        *shuffled,
    )
    report = {
        "schema": QUERY_ADDRESS_SCHEMA,
        "claim_boundary": (
            "Outcome-only acquisition of multiple query-conditioned temporal "
            "offsets under one rendered cue with a frozen readout; not "
            "content search, learned compression, unrestricted memory growth, "
            "arbitrary new computation, or general continual learning."
        ),
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "readout": "external_temporal_capability_file_frozen_before_target",
            "address_memory": "persistent_opaque_context_route_evidence_v1",
            "address_key": "learned_query_event_tensor",
            "route_feedback": "terminal_scalar_episode_accuracy_only",
            "cue_symbol": CUE_SYMBOL,
            "source_query": SOURCE_QUERY,
            "target_query": TARGET_QUERY,
            "source_offset": SOURCE_DEPTH,
            "target_offset": TARGET_DEPTH,
            "max_offset": MAX_OFFSET,
        },
        "seed": args.seed,
        "source_history_tail": source_history[-5:],
        "source_route_history_tail": source_route_history[-5:],
        "target_history_tail": target_history[-5:],
        "shuffled_history_tail": shuffled_history[-5:],
        "evaluation": {
            "source_before": source_before,
            "source_after": source_after,
            "target_after": target_after,
            "unknown_query": unknown,
            "wrong_offset": wrong_offset,
            "missing_history": missing_history,
            "shuffled_outcome": shuffled,
            "source_file_digest": source_file_digest,
            "route_payload": evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": source_bits + calibration_bits + target_bits,
            "control_verifier_bits": control_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"]) for row in audit_rows
            ),
            "unique_logical_lifetimes": args.batch_size
            * (
                args.source_updates
                + args.route_calibration_lifetimes
                + args.target_updates
            ),
            "optimizer_updates": args.source_updates,
            "route_memory_updates": args.route_calibration_lifetimes
            + args.target_updates,
            "control_route_memory_updates": args.target_updates,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": source_bits + calibration_bits + target_bits
            if all(gates.values())
            else None,
        },
        "status": "promoted_temporal_query_address_growth"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=512)
    parser.add_argument("--target-updates", type=int, default=512)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
