"""Audit two-row content binding and batch-isolated memory scopes."""

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
from torch.distributions import Categorical

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AmodalOutputBus,
    ContentAddressedMemory,
    ControllerFeedback,
    MemoryBackend,
    OpaqueProtocolDecoder,
)

from .environment import TwoSlotBindingVerifier


@dataclass(frozen=True)
class Accounting:
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    optimizer_updates: int
    replayed_examples: int
    verifier_outcome_events: int
    feedback_events: int
    mean_memory_write_strength: float
    committed_write_rate: float
    wall_time_seconds: float
    mean_action_latency_ms: float
    stable_bits_to_threshold: int | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(*, seed: int, batch_size: int, width: int = 16) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=2,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
        memory_top_k=1,
    )
    return AmodalControllerRuntime(
        controller,
        output_bus=AmodalOutputBus(
            {"protocol": OpaqueProtocolDecoder(4, 2, hidden=8)}
        ),
        memory=ContentAddressedMemory(
            width=width,
            capacity=2,
            scope_capacity=batch_size,
            write_threshold=0.0,
        ),
    )


def _feedback(
    batch_size: int,
    *,
    action: torch.Tensor | None = None,
    reward: torch.Tensor | None = None,
    propensity: torch.Tensor | None = None,
    has_feedback: torch.Tensor | None = None,
) -> ControllerFeedback:
    return ControllerFeedback(
        action=(
            torch.zeros(batch_size, 2) if action is None else action
        ),
        reward=(
            torch.zeros(batch_size) if reward is None else reward
        ),
        propensity=(
            torch.ones(batch_size) if propensity is None else propensity
        ),
        has_feedback=(
            torch.zeros(batch_size) if has_feedback is None else has_feedback
        ),
    )


def _event(token: torch.Tensor, batch_size: int) -> AmodalEventCollection:
    return AmodalEventCollection.from_events(
        [AmodalEvent(token.expand(batch_size, -1))]
    )


def _store_pair(
    runtime: AmodalControllerRuntime,
    verifier: TwoSlotBindingVerifier,
    tokens: torch.Tensor,
    scope: torch.Tensor,
    *,
    state: Any,
) -> Any:
    for slot in (0, 1):
        action = torch.full((verifier.batch_size,), slot, dtype=torch.long)
        reward = verifier.score_probe(slot, action)
        feedback = _feedback(
            verifier.batch_size,
            action=torch.nn.functional.one_hot(action, num_classes=2).to(torch.float32),
            reward=reward,
            propensity=torch.full((verifier.batch_size,), 0.5),
            has_feedback=torch.ones(verifier.batch_size),
        )
        _, state = runtime.step_events(
            _event(tokens[slot], verifier.batch_size),
            state,
            feedback,
            memory_scope=scope,
        )
    return state


def _query(
    runtime: AmodalControllerRuntime,
    verifier: TwoSlotBindingVerifier,
    tokens: torch.Tensor,
    scope: torch.Tensor,
    *,
    query_scope: torch.Tensor | None = None,
    query_slot: torch.Tensor | None = None,
    sample: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = verifier.query_slot if query_slot is None else query_slot
    token_rows = tokens[slots]
    output, _ = runtime.step_events(
        _event(token_rows, verifier.batch_size),
        runtime.initial_state(verifier.batch_size, device=token_rows.device),
        _feedback(verifier.batch_size),
        memory_scope=scope if query_scope is None else query_scope,
    )
    distribution = Categorical(logits=output.decoded["protocol"])
    action = distribution.sample() if sample else distribution.probs.argmax(dim=-1)
    return action, distribution


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: TwoSlotBindingVerifier,
    tokens: torch.Tensor,
    *,
    steps: int,
    seed: int,
    learning_rate: float = 2e-3,
    entropy_weight: float = 0.01,
    eval_every: int = 32,
    threshold: float = 0.8,
    reward_shuffle: bool = False,
) -> tuple[list[dict[str, float | int]], Accounting]:
    if steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(seed)
    runtime.train()
    optimizer = torch.optim.Adam(runtime.parameters(), lr=learning_rate)
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    baseline = 0.5
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    action_time = 0.0
    write_strength_total = 0.0
    committed_writes = 0

    for step in range(1, steps + 1):
        verifier.reset()
        assert isinstance(runtime.memory, MemoryBackend)
        runtime.memory.clear()
        state = runtime.initial_state(verifier.batch_size, device="cpu")
        tick_started = time.perf_counter()
        with runtime.memory.differentiable_transaction():
            for slot in (0, 1):
                action = torch.full((verifier.batch_size,), slot, dtype=torch.long)
                probe_reward = verifier.score_probe(slot, action)
                reward = (
                    torch.randint(0, 2, probe_reward.shape).to(torch.float32)
                    if reward_shuffle
                    else probe_reward
                )
                feedback = _feedback(
                    verifier.batch_size,
                    action=torch.nn.functional.one_hot(action, num_classes=2).to(
                        torch.float32
                    ),
                    reward=reward,
                    propensity=torch.full((verifier.batch_size,), 0.5),
                    has_feedback=torch.ones(verifier.batch_size),
                )
                output, state = runtime.step_events(
                    _event(tokens[slot], verifier.batch_size),
                    state,
                    feedback,
                    memory_scope=scope,
                )
                write_strength_total += float(
                    output.controller.memory_write_strength.detach().mean()
                )
                committed_writes += int(
                    output.controller.memory_write_receipt.committed.sum()
                )
            query_action, distribution = _query(
                runtime,
                verifier,
                tokens,
                scope,
                sample=True,
            )
            query_reward = verifier.score_recall(query_action)
            loss = -(
                (query_reward.detach() - baseline)
                * distribution.log_prob(query_action)
            ).mean() - entropy_weight * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runtime.parameters(), max_norm=5.0)
            optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(query_reward.mean())
        action_time += time.perf_counter() - tick_started

        if step == 1 or step % eval_every == 0 or step == steps:
            score = evaluate_condition(
                runtime,
                TwoSlotBindingVerifier(
                    batch_size=verifier.batch_size, seed=seed + 1000 + step
                ),
                tokens,
                condition="intact",
                batches=16,
            )
            history.append({"step": step, "intact_reward": score})

    stable_bits = None
    for index, point in enumerate(history):
        if point["intact_reward"] >= threshold and all(
            later["intact_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(point["step"] * verifier.batch_size * 3)
            break
    return history, Accounting(
        unique_verifier_bits=steps * verifier.batch_size * 3,
        unique_logical_lifetimes=steps * verifier.batch_size,
        optimizer_updates=steps,
        replayed_examples=0,
        verifier_outcome_events=steps * verifier.batch_size * 3,
        feedback_events=steps * verifier.batch_size * 2,
        mean_memory_write_strength=write_strength_total / (steps * 2),
        committed_write_rate=committed_writes / (steps * 2 * verifier.batch_size),
        wall_time_seconds=time.perf_counter() - started,
        mean_action_latency_ms=action_time / steps / (verifier.batch_size * 3) * 1000.0,
        stable_bits_to_threshold=stable_bits,
    )


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: TwoSlotBindingVerifier,
    tokens: torch.Tensor,
    *,
    condition: str,
    batches: int,
) -> float:
    valid = {"intact", "clear", "corrupt", "swapped_slot", "swapped_scope", "random_action"}
    if condition not in valid:
        raise ValueError(f"unknown binding condition {condition!r}")
    runtime.eval()
    assert isinstance(runtime.memory, ContentAddressedMemory)
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    count = 0
    for _ in range(batches):
        verifier.reset()
        runtime.memory.clear()
        _store_pair(
            runtime,
            verifier,
            tokens,
            scope,
            state=runtime.initial_state(verifier.batch_size, device="cpu"),
        )
        if condition in {"clear"}:
            runtime.memory.clear()
        elif condition == "corrupt":
            runtime.memory.values.zero_()
        query_scope = scope
        query_slot = verifier.query_slot
        if condition == "swapped_slot":
            query_slot = 1 - query_slot
        elif condition == "swapped_scope":
            query_scope = scope.roll(1)
        if condition == "random_action":
            action = torch.randint(0, 2, (verifier.batch_size,))
        else:
            action, _ = _query(
                runtime,
                verifier,
                tokens,
                scope,
                query_scope=query_scope,
                query_slot=query_slot,
                sample=False,
            )
        total += float(verifier.score_recall(action).sum())
        count += verifier.batch_size
    runtime.train()
    return total / count


def run_experiment(
    *,
    steps: int,
    seed: int,
    batch_size: int = 4,
    report_out: Path | None = None,
    reward_shuffle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    runtime = build_runtime(seed=seed, batch_size=batch_size)
    tokens = torch.randn(2, runtime.event_width)
    verifier = TwoSlotBindingVerifier(batch_size=batch_size, seed=seed + 10)
    history, accounting = train_steps(
        runtime,
        verifier,
        tokens,
        steps=steps,
        seed=seed,
        reward_shuffle=reward_shuffle,
    )
    conditions = {
        condition: evaluate_condition(
            runtime,
            TwoSlotBindingVerifier(batch_size=batch_size, seed=seed + 200 + index),
            tokens,
            condition=condition,
            batches=32,
        )
        for index, condition in enumerate(
            ("intact", "clear", "corrupt", "swapped_slot", "swapped_scope", "random_action")
        )
    }
    promotion = (
        not reward_shuffle
        and conditions["intact"] >= 0.80
        and conditions["clear"] <= 0.65
        and conditions["corrupt"] <= 0.65
        and conditions["swapped_slot"] <= 0.65
        and conditions["swapped_scope"] <= 0.65
        and conditions["intact"] - conditions["clear"] >= 0.25
        and conditions["intact"] - conditions["corrupt"] >= 0.25
    )
    report: dict[str, Any] = {
        "experiment": "outcome-only-two-slot-binding",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "reward_shuffle_control": reward_shuffle,
        "learner_visible_inputs": [
            "opaque slot event",
            "opaque probe action",
            "scalar probe outcome",
            "scalar recall outcome",
        ],
        "memory_training": {
            "capacity": 2,
            "scope_capacity": batch_size,
            "write_threshold": 0.0,
            "forced_write_control": True,
            "differentiable_transaction": True,
            "claim_boundary": "v18 shared-address fixed-write two-slot binding and batch-isolation diagnostic only",
        },
        "history": history,
        "conditions": conditions,
        "promotion_gate": {
            "intact_min": 0.80,
            "clear_max": 0.65,
            "corrupt_max": 0.65,
            "swapped_slot_max": 0.65,
            "swapped_scope_max": 0.65,
            "causal_gap_min": 0.25,
        },
        "promoted": promotion,
        "accounting": asdict(accounting),
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reward-shuffle", action="store_true")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
