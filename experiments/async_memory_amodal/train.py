"""Train and audit delayed outcome-only amodal control."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalOutputBus,
    ControllerFeedback,
    ContentAddressedMemory,
    OpaqueProtocolDecoder,
    save_runtime,
)

from .environment import DelayedComplementVerifier


class TaggedEventEncoder(nn.Module):
    """Independent raw frontend with generic provenance metadata."""

    def __init__(self, raw_width: int, event_width: int, source_key: list[float]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(raw_width, event_width),
            nn.Tanh(),
            nn.Linear(event_width, event_width),
            nn.Tanh(),
        )
        self.register_buffer(
            "source_key", torch.tensor(source_key, dtype=torch.float32).reshape(1, -1)
        )

    def forward(self, raw: torch.Tensor) -> AmodalEvent:
        if raw.ndim != 2:
            raise ValueError("raw stream must have shape [batch, width]")
        batch = raw.shape[0]
        return AmodalEvent(
            payload=self.network(raw),
            source_key=self.source_key.expand(batch, -1),
            confidence=torch.ones(batch, device=raw.device, dtype=raw.dtype),
            duration=torch.ones(batch, device=raw.device, dtype=raw.dtype),
        )


@dataclass(frozen=True)
class Accounting:
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    optimizer_updates: int
    replayed_examples: int
    feedback_events: int
    diagnostic_lifetimes_charged_to_budget: int
    wall_time_seconds: float
    mean_action_latency_ms: float
    stable_bits_to_threshold: int | None
    retention_on_mastered_primitives: float | None
    transfer_ratio_against_fresh_learner: float | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(
    *, seed: int = 0, width: int = 32, with_memory: bool = False
) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=2,
        intention_width=16,
        feedback_width=8,
        source_key_width=2,
        event_window_capacity=4,
        reliability_hidden=16,
        memory_top_k=1,
    )
    memory = (
        ContentAddressedMemory(width, capacity=64, write_threshold=0.5)
        if with_memory
        else None
    )
    return AmodalControllerRuntime(
        controller,
        encoders={
            "a": TaggedEventEncoder(DelayedComplementVerifier.raw_width, width, [1.0, 0.0]),
            "b": TaggedEventEncoder(DelayedComplementVerifier.raw_width, width, [0.0, 1.0]),
        },
        output_bus=AmodalOutputBus(
            {"protocol": OpaqueProtocolDecoder(16, DelayedComplementVerifier.action_count, hidden=16)}
        ),
        memory=memory,
    )


def zero_feedback(batch: int, width: int, device: torch.device) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, width, device=device),
        reward=torch.zeros(batch, device=device),
        propensity=torch.ones(batch, device=device),
        has_feedback=torch.zeros(batch, device=device),
    )


def opaque_action(actions: torch.Tensor, width: int) -> torch.Tensor:
    result = torch.zeros(actions.shape[0], width, device=actions.device)
    result.scatter_(1, (actions % width)[:, None], 1.0)
    return result


def _timed_events(
    runtime: AmodalControllerRuntime,
    streams: Mapping[str, torch.Tensor],
    *,
    timestamp: float,
) -> list[AmodalEvent]:
    events: list[AmodalEvent] = []
    for name, raw in streams.items():
        event = runtime.encoders[name](raw)
        events.append(
            AmodalEvent(
                payload=event.payload,
                source_key=event.source_key,
                timestamp=torch.full(
                    (raw.shape[0],), timestamp, device=raw.device, dtype=raw.dtype
                ),
                duration=event.duration,
                confidence=event.confidence,
            )
        )
    return events


def _tick(
    runtime: AmodalControllerRuntime,
    state: Any,
    streams: Mapping[str, torch.Tensor],
    feedback: ControllerFeedback,
    *,
    timestamp: float,
) -> tuple[Any, Any]:
    batch = feedback.reward.shape[0]
    events = _timed_events(runtime, streams, timestamp=timestamp)
    if events:
        output, state = runtime.step_events(events, state, feedback, elapsed=1.0)
    else:
        output, state = runtime.step_streams(
            {},
            state,
            feedback,
            batch_size=batch,
            device=feedback.reward.device,
            elapsed=1.0,
        )
    return output, state


def _episode(
    runtime: AmodalControllerRuntime,
    verifier: DelayedComplementVerifier,
    *,
    batch_size: int,
    device: torch.device,
    shuffle_second: bool = False,
    feedback_after: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = runtime.initial_state(batch_size, device=device)
    feedback = zero_feedback(batch_size, runtime.controller.feedback_width, device)
    first = verifier.start(batch_size)
    _, state = _tick(runtime, state, first, feedback, timestamp=0.0)
    second = dict(verifier.next_arrival())
    if shuffle_second and second:
        second["b"] = second["b"][torch.randperm(batch_size, device=device)]
    _, state = _tick(runtime, state, second, feedback, timestamp=1.0)
    output, state = _tick(runtime, state, {}, feedback, timestamp=2.0)
    distribution = Categorical(logits=output.decoded["protocol"])
    actions = distribution.sample()
    propensity = distribution.probs.gather(1, actions[:, None]).squeeze(1)
    reward = verifier.score(actions)
    if feedback_after:
        delayed_feedback = ControllerFeedback(
            action=opaque_action(actions, runtime.controller.feedback_width),
            reward=reward,
            propensity=propensity,
            has_feedback=torch.ones(batch_size, device=device),
        )
        with torch.no_grad():
            _tick(runtime, state.detached(), {}, delayed_feedback, timestamp=3.0)
    return actions, distribution.entropy(), propensity, reward


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: DelayedComplementVerifier,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 3e-3,
    entropy_weight: float = 0.01,
    eval_every: int = 32,
    threshold: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], Accounting]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    seed_everything(seed)
    device = torch.device(device)
    runtime.to(device).train()
    optimizer = torch.optim.Adam(runtime.parameters(), lr=learning_rate)
    baseline = 0.25
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    action_time = 0.0
    diagnostic_lifetimes = 0
    for step in range(1, steps + 1):
        tick_start = time.perf_counter()
        _, entropy, propensity, reward = _episode(
            runtime, verifier, batch_size=batch_size, device=device
        )
        action_time += time.perf_counter() - tick_start
        advantage = reward - baseline
        loss = -(
            advantage.detach() * propensity.clamp_min(1e-8).log()
        ).mean() - entropy_weight * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runtime.parameters(), max_norm=5.0)
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(reward.mean())
        if step == 1 or step % eval_every == 0 or step == steps:
            score = evaluate_condition(
                runtime,
                DelayedComplementVerifier(seed=seed + 1000 + step, device=device),
                condition="fused",
                batches=4,
                batch_size=batch_size,
                device=device,
            )
            diagnostic_lifetimes += 4 * batch_size
            history.append({"step": step, "fused_reward": score})
    elapsed = time.perf_counter() - start
    stable_bits = None
    for index, point in enumerate(history):
        if point["fused_reward"] >= threshold and all(
            later["fused_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(point["step"] * batch_size * 2)
            break
    lifetimes = steps * batch_size
    return history, Accounting(
        unique_verifier_bits=lifetimes * 2,
        unique_logical_lifetimes=lifetimes,
        optimizer_updates=steps,
        replayed_examples=0,
        feedback_events=lifetimes,
        diagnostic_lifetimes_charged_to_budget=diagnostic_lifetimes,
        wall_time_seconds=elapsed,
        mean_action_latency_ms=action_time / lifetimes * 1000.0,
        stable_bits_to_threshold=stable_bits,
        retention_on_mastered_primitives=None,
        transfer_ratio_against_fresh_learner=None,
    )


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: DelayedComplementVerifier,
    *,
    condition: str,
    batches: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> float:
    device = torch.device(device)
    runtime.eval()
    total = 0.0
    count = 0
    for _ in range(batches):
        if condition == "fused":
            _, _, _, reward = _episode(runtime, verifier, batch_size=batch_size, device=device)
        elif condition == "missing_second":
            verifier.missing_second = True
            _, _, _, reward = _episode(runtime, verifier, batch_size=batch_size, device=device)
        elif condition == "contradictory":
            verifier.contradictory = True
            _, _, _, reward = _episode(runtime, verifier, batch_size=batch_size, device=device)
        elif condition == "shuffled_partner":
            _, _, _, reward = _episode(
                runtime, verifier, batch_size=batch_size, device=device, shuffle_second=True
            )
        elif condition == "random_action":
            state = runtime.initial_state(batch_size, device=device)
            feedback = zero_feedback(batch_size, runtime.controller.feedback_width, device)
            _tick(runtime, state, verifier.start(batch_size), feedback, timestamp=0.0)
            verifier.next_arrival()
            reward = verifier.score(torch.randint(0, verifier.action_count, (batch_size,), device=device))
        else:
            raise ValueError(f"unknown condition {condition!r}")
        total += float(reward.sum())
        count += batch_size
        verifier.missing_second = False
        verifier.contradictory = False
    return total / count


def run_experiment(
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
    checkpoint_out: Path | None = None,
    report_out: Path | None = None,
    with_memory: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device)
    runtime = build_runtime(seed=seed, with_memory=with_memory).to(device)
    history, accounting = train_steps(
        runtime,
        DelayedComplementVerifier(seed=seed + 10, device=device),
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    condition_names = [
        "fused",
        "missing_second",
        "contradictory",
        "shuffled_partner",
        "random_action",
    ]
    conditions = {
        name: evaluate_condition(
            runtime,
            DelayedComplementVerifier(seed=seed + 200 + index, device=device),
            condition=name,
            batches=8,
            batch_size=batch_size,
            device=device,
        )
        for index, name in enumerate(condition_names)
    }
    memory_clear_reward = None
    memory_occupied_after_training = None
    if runtime.memory is not None:
        memory_occupied_after_training = int(runtime.memory.occupied.sum().item())
        runtime.memory.clear()
        memory_clear_reward = evaluate_condition(
            runtime,
            DelayedComplementVerifier(seed=seed + 900, device=device),
            condition="fused",
            batches=8,
            batch_size=batch_size,
            device=device,
        )
    audit_lifetimes = (len(condition_names) * 8 + (8 if runtime.memory is not None else 0)) * batch_size
    promotion = (
        conditions["fused"] >= 0.80
        and conditions["missing_second"] <= 0.65
        and conditions["contradictory"] <= 0.25
        and conditions["shuffled_partner"] <= 0.65
        and conditions["fused"] - conditions["shuffled_partner"] >= 0.25
    )
    if checkpoint_out is not None:
        save_runtime(
            runtime,
            checkpoint_out,
            provenance={
                "experiment": "delayed-outcome-only-amodal",
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "promoted": promotion,
            },
        )
    report = {
        "experiment": "delayed-outcome-only-amodal",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "arrival_schedule": ["a@0", "b@1", "action@2", "feedback@3"],
        "learner_visible_feedback": ["scalar verifier reward", "opaque sampled action", "exact logging propensity"],
        "history": history,
        "conditions": conditions,
        "promotion_gate": {
            "fused_min": 0.80,
            "partial_evidence_max": 0.65,
            "contradictory_max": 0.25,
            "shuffle_gap_min": 0.25,
        },
        "promoted": promotion,
        "accounting": {
            **asdict(accounting),
            "audit_lifetimes": audit_lifetimes,
            "diagnostic_lifetimes_charged_to_budget": (
                accounting.diagnostic_lifetimes_charged_to_budget + audit_lifetimes
            ),
        },
        "external_memory": {
            "enabled": with_memory,
            "occupied_rows_after_training": memory_occupied_after_training,
            "memory_clear_fused_reward": memory_clear_reward,
            "claim": "integration/corruption diagnostic only; this task does not require persistent memory",
        },
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--with-memory", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
