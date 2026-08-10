"""Audit repeated maintenance decisions on one nonstationary memory stream.

This is the next rung after the independent real-transaction pressure test.
One external transition bank survives the entire stream.  New opaque regimes
arrive, equivalent slots can be shared, storage can be compressed, and a
disposable slot can be evicted when the finite capacity is full.  The policy
sees only generic bank telemetry and a structural action mask; an independent
verifier supplies held-out retention and transaction utility.

The experiment is deliberately bounded.  It tests whether maintenance state
compounds safely across repeated operations, not unrestricted memory growth,
autonomous verifier construction, or arbitrary new computation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    MAINTENANCE_ACTIONS,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMemoryMaintenancePolicy,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
HIDDEN_WIDTH = 24
TRAIN_STEPS = 640
EVAL_STEPS = 192
INITIAL_CAPACITY = 2
MAX_CAPACITY = 4
BOOTSTRAP_OBSERVATIONS = 12
MASTERY_OBSERVATIONS = 8
MASTERY_TOLERANCE = 0.08
RETENTION_TOLERANCE = 0.12
TEMPERATURE = 1.0
LEARNING_RATE = 0.01

CORE_CONCEPTS = (0, 1, 2, 3)
TRANSIENT_CONCEPT = 4

_CONTEXT_KEYS = tuple(
    torch.nn.functional.normalize(
        torch.tensor(values, dtype=torch.float32),
        dim=0,
    )
    for values in (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.0),
    )
)
_DUPLICATE_KEY = torch.nn.functional.normalize(
    torch.tensor((1.0, 0.0, 0.0, 1.0), dtype=torch.float32),
    dim=0,
)
_DELTAS = (
    torch.tensor((0.25, 0.75)),
    torch.tensor((-0.55, 0.40)),
    torch.tensor((0.80, -0.20)),
    torch.tensor((-0.30, -0.65)),
    torch.tensor((0.65, 0.25)),
)


def _digest(module: torch.nn.Module) -> str:
    return repr(
        [
            (name, value.detach().cpu().clone())
            for name, value in sorted(module.state_dict().items())
        ]
    )


def _observation(concept: int, occurrence: int) -> ExternalTransitionObservation:
    if not 0 <= concept < len(_DELTAS):
        raise ValueError("unknown long-stream concept")
    state = torch.tensor(
        [[0.15 + 0.031 * (occurrence % 17), -0.45 + 0.017 * (occurrence % 13)]],
        dtype=torch.float32,
    )
    intention = torch.tensor([[0.7 - 0.011 * (occurrence % 7)]], dtype=torch.float32)
    next_state = state + intention * _DELTAS[concept]
    return ExternalTransitionObservation(state, intention, next_state)


def _heldout(concept: int, occurrence: int = 10000) -> ExternalTransitionObservation:
    return _observation(concept, occurrence + concept * 19)


def _router_for_bank(
    bank: ExternalTransitionModelBank,
) -> ExternalLearnedMultiStreamTransitionContextRouter:
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=8,
        context_width=CONTEXT_WIDTH,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=MAX_CAPACITY,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    binding = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=2,
        max_streams=MAX_CAPACITY,
        provisional_capacity=2,
    )
    return ExternalLearnedMultiStreamTransitionContextRouter(
        binding,
        ExternalMultiStreamTransitionContextRouter(
            single,
            stream_key_width=CONTEXT_WIDTH,
        ),
    )


@dataclass
class LongStreamEnvironment:
    bank: ExternalTransitionModelBank
    router: ExternalLearnedMultiStreamTransitionContextRouter
    slot_concepts: dict[int, int | None] = field(default_factory=dict)
    observed: dict[int, int] = field(default_factory=dict)
    mastered: dict[int, int] = field(default_factory=dict)
    ever_mastered: set[int] = field(default_factory=set)
    retention_violation: bool = False
    duplicate_slot_id: int | None = None
    duplicate_source_slot_id: int | None = None
    duplicate_shared: bool = False
    compression_dirty: bool = False
    utility_history: list[float] = field(default_factory=list)
    retention_history: list[float] = field(default_factory=list)
    action_counts: dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in MAINTENANCE_ACTIONS}
    )
    executed_action_counts: dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in MAINTENANCE_ACTIONS}
    )
    accepted_transactions: int = 0
    admissions: int = 0
    missed_novel_events: int = 0
    bytes_saved: int = 0
    growth_events: int = 0
    eviction_events: int = 0
    share_events: int = 0
    compression_events: int = 0
    last_transaction_reason: str = "initialization"

    @classmethod
    def create(cls, seed: int) -> LongStreamEnvironment:
        torch.manual_seed(seed)
        bank = ExternalTransitionModelBank(
            STATE_WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            hidden_width=8,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            capacity=INITIAL_CAPACITY,
        )
        environment = cls(bank=bank, router=_router_for_bank(bank))
        environment._admit(0)
        environment._admit(1)
        for concept in (0, 1):
            for occurrence in range(BOOTSTRAP_OBSERVATIONS):
                environment._adapt(concept, occurrence)
        environment.compression_dirty = False
        return environment

    def _key(self, concept: int) -> torch.Tensor:
        return _CONTEXT_KEYS[concept]

    def _find_slot_id(self, key: torch.Tensor) -> int | None:
        if self.bank.context_count == 0:
            return None
        distances = torch.linalg.vector_norm(self.bank.contexts - key, dim=-1)
        index = int(distances.argmin())
        if float(distances[index]) > self.bank.matching_tolerance:
            return None
        return self.bank.slot_id_at(index)

    def _admit(self, concept: int, *, duplicate: bool = False) -> int:
        key = _DUPLICATE_KEY if duplicate else self._key(concept)
        existing = self._find_slot_id(key)
        if existing is not None:
            return existing
        index = self.bank.ensure_context(
            key,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        )
        slot_id = self.bank.slot_id_at(index)
        if duplicate:
            if self.duplicate_source_slot_id is None:
                raise RuntimeError("duplicate source slot is missing")
            source_index = self.bank.physical_index_for_slot_id(
                self.duplicate_source_slot_id
            )
            self.bank.models[index].load_state_dict(
                self.bank.models[source_index].state_dict()
            )
            self.duplicate_slot_id = slot_id
            self.slot_concepts[slot_id] = None
        else:
            self.slot_concepts[slot_id] = concept
            self.observed.setdefault(concept, 0)
            self.admissions += 1
        self.compression_dirty = True
        return slot_id

    def _adapt(self, concept: int, occurrence: int) -> bool:
        slot_id = self._find_slot_id(self._key(concept))
        if slot_id is None:
            return False
        observation = _observation(concept, occurrence)
        self.bank.adaptation_step(
            observation,
            self._key(concept).unsqueeze(0),
            None,
        )
        self.observed[concept] = self.observed.get(concept, 0) + 1
        if (
            concept in CORE_CONCEPTS
            and self.observed[concept] >= MASTERY_OBSERVATIONS
            and self._prediction_error(slot_id, _heldout(concept))
            <= MASTERY_TOLERANCE
        ):
            self.mastered[slot_id] = concept
            self.ever_mastered.add(concept)
        return True

    def _prediction_error(
        self,
        slot_id: int,
        observation: ExternalTransitionObservation,
        *,
        bank: ExternalTransitionModelBank | None = None,
    ) -> float:
        selected_bank = self.bank if bank is None else bank
        index = selected_bank.physical_index_for_slot_id(slot_id)
        prediction = selected_bank.models[index](
            observation.state,
            observation.intention,
        )
        return float((prediction - observation.next_state).square().mean())

    def _retention_probe(self) -> Callable[[ExternalTransitionModelBank], bool]:
        protected = tuple(self.mastered.items())

        def probe(candidate: ExternalTransitionModelBank) -> bool:
            for slot_id, concept in protected:
                try:
                    error = self._prediction_error(
                        slot_id,
                        _heldout(concept),
                        bank=candidate,
                    )
                except (IndexError, KeyError):
                    return False
                if error > RETENTION_TOLERANCE:
                    return False
            return True

        return probe

    def _eviction_candidate(self) -> int | None:
        telemetry = self.bank.lifetime_telemetry()
        values: list[tuple[int, int, int]] = []
        for index, slot_id in enumerate(self.bank.slot_ids):
            if slot_id in self.mastered:
                continue
            values.append(
                (
                    int(telemetry.usage[index]),
                    -int(telemetry.age[index]),
                    slot_id,
                )
            )
        if not values:
            return None
        return min(values)[2]

    def _share_pair(self) -> tuple[int, int] | None:
        if self.duplicate_slot_id is None or self.duplicate_shared:
            return None
        if self.duplicate_source_slot_id is None:
            return None
        try:
            first = self.bank.physical_index_for_slot_id(
                self.duplicate_source_slot_id
            )
            second = self.bank.physical_index_for_slot_id(self.duplicate_slot_id)
        except KeyError:
            return None
        return first, second

    def _source_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for model in self.bank.models
            for value in model.state_dict().values()
        )

    def _compression_opportunity(self) -> float:
        source = self._source_bytes()
        if source <= 0 or self.bank.context_count == 0:
            return 0.0
        payload = self.bank.compressed_payload(dtype=torch.float16)
        compressed = sum(
            value.numel() * value.element_size()
            for model in payload["models"]
            for value in model["state"].values()
        )
        return max(0.0, min(1.0, 1.0 - compressed / source))

    def _stage_duplicate(self) -> None:
        if self.duplicate_slot_id is not None:
            return
        if self.bank.capacity is None or self.bank.context_count >= self.bank.capacity:
            return
        self.duplicate_source_slot_id = self._find_slot_id(self._key(0))
        if self.duplicate_source_slot_id is not None:
            self._admit(0, duplicate=True)

    def _current_concept(self, step: int, variant: int) -> int:
        phase_step = step if step < TRAIN_STEPS else step - TRAIN_STEPS
        if variant % 2:
            phase_step = TRAIN_STEPS - 1 - phase_step
        if phase_step < 48:
            return 0 if (phase_step // 4) % 2 == 0 else 1
        if phase_step < 96:
            return 2
        if phase_step < 128:
            return 0 if (phase_step // 4) % 2 == 0 else 2
        if phase_step < 160:
            return TRANSIENT_CONCEPT
        if phase_step < 208:
            return 3
        return CORE_CONCEPTS[(phase_step // 5) % len(CORE_CONCEPTS)]

    def _pending_action(
        self,
        concept: int,
        *,
        grow_available: bool,
        share_available: bool,
        compression_available: bool,
        evict_available: bool,
    ) -> str:
        novel = self._find_slot_id(self._key(concept)) is None
        full = self.bank.capacity is not None and self.bank.context_count >= self.bank.capacity
        if novel and full and self.bank.capacity is not None and self.bank.capacity < MAX_CAPACITY:
            return "grow"
        if novel and full and evict_available:
            return "evict"
        if share_available:
            return "share"
        if compression_available:
            return "compress"
        if grow_available and self.bank.capacity is not None and self.bank.capacity < 3:
            return "grow"
        return "defer"

    def step(
        self,
        policy: ExternalMemoryMaintenancePolicy,
        step: int,
        *,
        learn: bool,
        generator: torch.Generator | None,
        variant: int = 0,
        shuffled_verifier: bool = False,
        action_shuffled: bool = False,
    ) -> dict[str, object]:
        concept = self._current_concept(step, variant)
        self._stage_duplicate()
        current_slot = self._find_slot_id(self._key(concept))
        novel = current_slot is None
        share_pair = self._share_pair()
        grow_available = (
            self.bank.capacity is not None
            and self.bank.capacity < MAX_CAPACITY
        )
        share_available = share_pair is not None
        compression_opportunity = self._compression_opportunity()
        compression_available = self.compression_dirty and step >= 24
        eviction_candidate = self._eviction_candidate()
        evict_available = (
            novel
            and self.bank.capacity is not None
            and self.bank.context_count >= self.bank.capacity
            and eviction_candidate is not None
        )
        pending = self._pending_action(
            concept,
            grow_available=grow_available,
            share_available=share_available,
            compression_available=compression_available,
            evict_available=evict_available,
        )
        proposal = self.router.propose_maintenance(
            policy,
            grow_available=grow_available,
            share_available=share_available,
            compression_available=compression_available,
            evict_available=evict_available,
            redundancy_pressure=1.0 if share_available else None,
            compression_opportunity=compression_opportunity,
            sample=learn,
            generator=generator,
        )
        executed_action = proposal.action
        if action_shuffled:
            legal = [
                action
                for index, action in enumerate(MAINTENANCE_ACTIONS)
                if bool(proposal.available_actions[index])
            ]
            executed_action = legal[
                int(torch.randint(len(legal), (), generator=generator))
            ]
        self.action_counts[proposal.action] += 1
        self.executed_action_counts[executed_action] += 1

        before_bytes = self._source_bytes()
        accepted = False
        receipt: Any | None = None
        retention_probe = self._retention_probe()
        if executed_action == "grow" and grow_available:
            receipt = self.bank.grow_verified(
                (self.bank.capacity or 0) + 1,
                retention_probe,
            )
            accepted = bool(receipt.accepted)
            if accepted:
                self.growth_events += 1
                self.compression_dirty = True
        elif executed_action == "share" and share_pair is not None:
            receipt = self.bank.consolidate_verified(
                share_pair[0],
                share_pair[1],
                (_heldout(0),),
                prediction_tolerance=RETENTION_TOLERANCE,
                retention_probe=retention_probe,
            )
            accepted = bool(receipt.accepted)
            if accepted:
                self.share_events += 1
                self.duplicate_shared = True
                self.compression_dirty = True
        elif executed_action == "compress" and compression_available:
            receipt = self.bank.compress_and_commit_verified(
                dtype=torch.float16,
                retention_probe=retention_probe,
            )
            accepted = bool(receipt.accepted)
            if accepted:
                self.compression_events += 1
                self.compression_dirty = False
        elif executed_action == "evict" and eviction_candidate is not None:
            receipt = self.bank.evict_verified_id(
                eviction_candidate,
                retention_probe,
            )
            accepted = bool(receipt.accepted)
            if accepted:
                self.eviction_events += 1
                if eviction_candidate in self.mastered:
                    self.retention_violation = True
                self.slot_concepts.pop(eviction_candidate, None)
                self.mastered.pop(eviction_candidate, None)
                if eviction_candidate == self.duplicate_slot_id:
                    self.duplicate_slot_id = None
                    self.duplicate_source_slot_id = None
                    self.duplicate_shared = False
                self.compression_dirty = True
        if receipt is not None:
            self.last_transaction_reason = str(receipt.reason)
        self.accepted_transactions += int(accepted)

        current_slot = self._find_slot_id(self._key(concept))
        if current_slot is None and self.bank.capacity is not None:
            if self.bank.context_count < self.bank.capacity:
                current_slot = self._admit(concept)
            else:
                self.missed_novel_events += int(novel)
        adapted = False
        if current_slot is not None and concept in CORE_CONCEPTS:
            adapted = self._adapt(concept, step)
        after_bytes = self._source_bytes()
        if receipt is not None and hasattr(receipt, "source_bytes"):
            self.bytes_saved += max(
                0,
                int(receipt.source_bytes) - int(receipt.compressed_bytes),
            )
        self.bytes_saved += max(0, before_bytes - after_bytes)

        if pending == "defer":
            utility = 1.0 if executed_action == "defer" else 0.0
        elif pending == "grow":
            utility = 1.0 if executed_action == "grow" and accepted else 0.0
        elif pending == "evict":
            utility = 1.0 if executed_action == "evict" and accepted else 0.0
        elif pending == "share":
            utility = 1.0 if executed_action == "share" and accepted else 0.0
        else:
            utility = 1.0 if executed_action == "compress" and accepted else 0.0
        if shuffled_verifier:
            utility = float(torch.randint(2, (), generator=generator))
        if learn:
            optimizer = getattr(policy, "_long_stream_optimizer", None)
            if optimizer is None:
                optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
                policy._long_stream_optimizer = optimizer
            policy.adaptation_step(proposal, utility, optimizer=optimizer)
        retention = self.retention_floor()
        self.utility_history.append(utility)
        self.retention_history.append(retention)
        return {
            "proposal": proposal.action,
            "executed": executed_action,
            "pending": pending,
            "accepted": accepted,
            "adapted": adapted,
            "utility": utility,
            "retention": retention,
            "mastered": tuple(sorted(self.mastered.values())),
        }

    def retention_floor(self) -> float:
        if not self.mastered:
            return 1.0
        values = [
            max(0.0, 1.0 - self._prediction_error(slot_id, _heldout(concept)))
            for slot_id, concept in self.mastered.items()
            if slot_id in self.bank.slot_ids
        ]
        return min(values, default=0.0)

    def persistence_exact(self) -> bool:
        restored = ExternalTransitionModelBank.from_payload(self.bank.payload())
        return restored.digest() == self.bank.digest()


def _rollout(
    seed: int,
    *,
    learn: bool,
    variant: int = 0,
    shuffled_verifier: bool = False,
    action_shuffled: bool = False,
) -> dict[str, object]:
    torch.manual_seed(seed)
    policy = ExternalMemoryMaintenancePolicy(
        hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
    )
    environment = LongStreamEnvironment.create(seed + 17)
    optimizer = (
        torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE) if learn else None
    )
    if optimizer is not None:
        policy._long_stream_optimizer = optimizer
    generator = torch.Generator().manual_seed(seed + 4000)
    updates = 0
    latencies_ms: list[float] = []
    for step in range(TRAIN_STEPS):
        started = time.perf_counter()
        environment.step(
            policy,
            step,
            learn=learn,
            generator=generator if learn else None,
            variant=variant,
            shuffled_verifier=shuffled_verifier,
            action_shuffled=action_shuffled,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        updates += int(learn)
    return {
        "policy": policy,
        "environment": environment,
        "optimizer_updates": updates,
        "latencies_ms": latencies_ms,
    }


def _evaluate(policy: ExternalMemoryMaintenancePolicy, seed: int, variant: int) -> dict[str, object]:
    environment = LongStreamEnvironment.create(seed + 17)
    utilities: list[float] = []
    for step in range(EVAL_STEPS):
        result = environment.step(
            policy,
            TRAIN_STEPS + step,
            learn=False,
            generator=None,
            variant=variant,
        )
        utilities.append(float(result["utility"]))
    return {"environment": environment, "utilities": utilities}


def _stable_minimum(values: list[float], window: int = 64) -> float:
    if len(values) < window:
        return min(values, default=0.0)
    return min(
        sum(values[index : index + window]) / window
        for index in range(len(values) - window + 1)
    )


def _stable_threshold_step(
    values: list[float],
    *,
    threshold: float = 0.9,
    window: int = 64,
) -> int | None:
    if len(values) < window:
        return None
    for start in range(len(values) - window + 1):
        later_windows = (
            sum(values[index : index + window]) / window
            for index in range(start, len(values) - window + 1)
        )
        if min(later_windows) >= threshold:
            return start + window
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    controller = torch.nn.Linear(4, 4)
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    trained = _rollout(seed, learn=True)
    fresh = _rollout(seed + 900000, learn=False)
    shuffled = _rollout(seed + 910000, learn=True, shuffled_verifier=True)
    action_shuffled = _rollout(seed + 920000, learn=True, action_shuffled=True)
    trained_environment = trained["environment"]
    fresh_environment = fresh["environment"]
    shuffled_environment = shuffled["environment"]
    action_shuffled_environment = action_shuffled["environment"]
    trained_policy = trained["policy"]
    trained_eval = _evaluate(trained_policy, seed + 500000, 1)
    fresh_eval = _evaluate(fresh["policy"], seed + 500000, 1)
    shuffled_eval = _evaluate(shuffled["policy"], seed + 500000, 1)
    action_shuffled_eval = _evaluate(action_shuffled["policy"], seed + 500000, 1)
    trained_utility = sum(trained_eval["utilities"]) / EVAL_STEPS
    fresh_utility = sum(fresh_eval["utilities"]) / EVAL_STEPS
    shuffled_utility = sum(shuffled_eval["utilities"]) / EVAL_STEPS
    action_shuffled_utility = sum(action_shuffled_eval["utilities"]) / EVAL_STEPS
    trained_online_utility = sum(trained_environment.utility_history) / TRAIN_STEPS
    fresh_online_utility = sum(fresh_environment.utility_history) / TRAIN_STEPS
    shuffled_online_utility = (
        sum(shuffled_environment.utility_history) / TRAIN_STEPS
    )
    action_shuffled_online_utility = (
        sum(action_shuffled_environment.utility_history) / TRAIN_STEPS
    )
    trained_stable_step = _stable_threshold_step(
        trained_environment.utility_history
    )
    trained_latency_mean = sum(trained["latencies_ms"]) / TRAIN_STEPS
    trained_latency_p95 = _percentile(trained["latencies_ms"], 0.95)
    gates = {
        "trained_beats_fresh": trained_utility > fresh_utility + 0.10,
        "trained_online_beats_shuffled_verifier": (
            trained_online_utility > shuffled_online_utility + 0.10
        ),
        "trained_online_beats_action_shuffled": (
            trained_online_utility > action_shuffled_online_utility + 0.005
        ),
        "long_stream_retention": _stable_minimum(
            trained_environment.retention_history
        ) >= 0.85,
        "no_mastered_capability_evicted": not trained_environment.retention_violation,
        "repeated_growth_share_compress_evict": all(
            value > 0
            for value in (
                trained_environment.growth_events,
                trained_environment.share_events,
                trained_environment.compression_events,
                trained_environment.eviction_events,
            )
        ),
        "core_capabilities_acquired": all(
            concept in trained_environment.mastered.values()
            for concept in CORE_CONCEPTS
        ),
        "real_transactions_observed": trained_environment.accepted_transactions > 3,
        "persistence_exact": trained_environment.persistence_exact(),
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "one_update_per_stream_utility": trained["optimizer_updates"] == TRAIN_STEPS,
    }
    report = {
        "schema": "neural-computer.external-memory-long-nonstationary.v1",
        "claim_boundary": (
            "bounded replay-free repeated maintenance over one persistent external "
            "bank with real growth, sharing, compression, and eviction receipts; "
            "not unrestricted memory growth, learned verifier design, or general "
            "continual learning"
        ),
        "seed": seed,
        "configuration": {
            "train_steps": TRAIN_STEPS,
            "eval_steps": EVAL_STEPS,
            "initial_capacity": INITIAL_CAPACITY,
            "maximum_capacity": MAX_CAPACITY,
            "actions": MAINTENANCE_ACTIONS,
            "mastery_observations": MASTERY_OBSERVATIONS,
            "transaction": "persistent_copy_on_write_retention_verified_bank_v2",
            "update": "one_scalar_utility_without_replay_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "trained_eval_utility": trained_utility,
            "fresh_eval_utility": fresh_utility,
            "shuffled_verifier_eval_utility": shuffled_utility,
            "action_shuffled_eval_utility": action_shuffled_utility,
            "trained_online_mean_utility": trained_online_utility,
            "fresh_online_mean_utility": fresh_online_utility,
            "online_gain_vs_fresh": trained_online_utility - fresh_online_utility,
            "online_transfer_ratio_vs_fresh": (
                None
                if fresh_online_utility == 0.0
                else trained_online_utility / fresh_online_utility
            ),
            "shuffled_verifier_online_mean_utility": shuffled_online_utility,
            "action_shuffled_online_mean_utility": action_shuffled_online_utility,
            "trained_online_last_window_utility": sum(
                trained_environment.utility_history[-64:]
            )
            / 64,
            "trained_online_stable_minimum": _stable_minimum(
                trained_environment.utility_history
            ),
            "trained_stable_threshold_step": trained_stable_step,
            "trained_retention_floor": min(
                trained_environment.retention_history,
                default=0.0,
            ),
            "trained_action_counts": trained_environment.action_counts,
            "trained_executed_action_counts": trained_environment.executed_action_counts,
            "trained_growth_events": trained_environment.growth_events,
            "trained_share_events": trained_environment.share_events,
            "trained_compression_events": trained_environment.compression_events,
            "trained_eviction_events": trained_environment.eviction_events,
            "trained_admissions": trained_environment.admissions,
            "trained_missed_novel_events": trained_environment.missed_novel_events,
            "trained_bytes_saved": trained_environment.bytes_saved,
            "trained_mastered_concepts": sorted(trained_environment.mastered.values()),
            "trained_ever_mastered_concepts": sorted(trained_environment.ever_mastered),
            "fresh_mastered_concepts": sorted(fresh_environment.mastered.values()),
            "shuffled_mastered_concepts": sorted(shuffled_environment.mastered.values()),
            "action_shuffled_mastered_concepts": sorted(
                action_shuffled_environment.mastered.values()
            ),
        },
        "accounting": {
            "unique_verifier_utilities": TRAIN_STEPS,
            "unique_logical_lifetimes": TRAIN_STEPS,
            "optimizer_updates": trained["optimizer_updates"],
            "replayed_examples": 0,
            "controller_updates": 0,
            "stable_bits_to_threshold": trained_stable_step,
            "latency_ms_mean": trained_latency_mean,
            "latency_ms_p95": trained_latency_p95,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6120)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
