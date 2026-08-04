"""Train and audit scalar-outcome source-trust calibration."""

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
    OpaqueProtocolDecoder,
)

from .environment import CalibrationConflictVerifier


class FrozenEventEncoder(nn.Module):
    """Independent frozen frontend producing opaque standardized events."""

    def __init__(self, raw_width: int, event_width: int, source_key: list[float]) -> None:
        super().__init__()
        self.network = nn.Linear(raw_width, event_width)
        with torch.no_grad():
            self.network.weight.zero_()
            self.network.bias.zero_()
            for index in range(min(raw_width, event_width)):
                self.network.weight[index, index] = 1.0
        self.register_buffer(
            "source_key", torch.tensor(source_key, dtype=torch.float32).reshape(1, -1)
        )

    def forward(self, raw: torch.Tensor) -> AmodalEvent:
        return AmodalEvent(
            payload=self.network(raw),
            source_key=self.source_key.expand(raw.shape[0], -1),
            confidence=torch.ones(raw.shape[0], device=raw.device, dtype=raw.dtype),
        )


@dataclass(frozen=True)
class RunAccounting:
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    optimizer_updates: int
    replayed_examples: int
    diagnostic_lifetimes_charged_to_budget: int
    wall_time_seconds: float
    mean_inference_latency_ms: float
    stable_bits_to_threshold: int | None
    retention_on_mastered_primitives: float | None
    transfer_ratio_against_fresh_learner: float | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(*, seed: int = 0, width: int = 32) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=1,
        intention_width=16,
        feedback_width=8,
        source_key_width=2,
        event_window_capacity=2,
        reliability_hidden=16,
        memory_top_k=1,
    )
    # Keep the feedback/source interaction generic, but avoid a completely
    # silent credit path in this outcome-only bootstrap rung. Production
    # checkpoint compatibility remains zero-initialized in the core module.
    with torch.no_grad():
        controller.event_feedback_relevance.weight.normal_(mean=0.0, std=0.02)
        controller.event_feedback_source_relevance.weight.normal_(mean=0.0, std=0.02)
    encoders = {
        name: FrozenEventEncoder(
            CalibrationConflictVerifier.raw_width,
            width,
            [1.0 if index == position else 0.0 for position in range(2)],
        )
        for index, name in enumerate(CalibrationConflictVerifier.stream_names)
    }
    return AmodalControllerRuntime(
        controller,
        encoders=encoders,
        output_bus=AmodalOutputBus(
            {"protocol": OpaqueProtocolDecoder(16, 2, hidden=16)}
        ),
    )


def _feedback(
    previous_action: torch.Tensor | None,
    previous_reward: torch.Tensor | None,
    previous_propensity: torch.Tensor | None,
    *,
    batch_size: int,
    width: int,
    device: torch.device,
    no_feedback: bool = False,
) -> ControllerFeedback:
    if previous_action is None or no_feedback:
        return ControllerFeedback(
            action=torch.zeros(batch_size, width, device=device),
            reward=torch.zeros(batch_size, device=device),
            propensity=torch.ones(batch_size, device=device),
            has_feedback=torch.zeros(batch_size, device=device),
        )
    action = torch.zeros(batch_size, width, device=device)
    action[:, :2] = torch.nn.functional.one_hot(
        previous_action.to(torch.long), num_classes=2
    ).to(action.dtype)
    return ControllerFeedback(
        action=action,
        reward=previous_reward,
        propensity=previous_propensity,
        has_feedback=torch.ones_like(previous_reward),
    )


def _sample_action(
    runtime: AmodalControllerRuntime,
    streams: Mapping[str, torch.Tensor],
    state: Any,
    feedback: ControllerFeedback,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    output, next_state = runtime.step_streams(streams, state, feedback)
    distribution = Categorical(logits=output.decoded["protocol"])
    action = distribution.sample()
    propensity = distribution.probs.gather(1, action[:, None]).squeeze(1)
    return (
        action,
        distribution.log_prob(action),
        distribution.entropy(),
        propensity,
        output.intention.payload,
        next_state,
    )


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: CalibrationConflictVerifier,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 1e-3,
    entropy_weight: float = 0.01,
    reward_shuffle: bool = False,
    eval_every: int = 64,
    threshold: float = 0.7,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], RunAccounting]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    seed_everything(seed)
    device = torch.device(device)
    runtime.to(device).train()
    for encoder in runtime.encoders.values():
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = [
        parameter for parameter in runtime.parameters() if parameter.requires_grad
    ]
    critic = nn.Sequential(
        nn.Linear(runtime.intention_width, 16), nn.Tanh(), nn.Linear(16, 1)
    ).to(device)
    trainable_parameters += list(critic.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    baseline = 0.5
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    total_latency = 0.0
    diagnostic_lifetimes = 0
    for step in range(1, steps + 1):
        sequence = verifier.sample(
            batch_size,
            force_role=0 if step <= max(1, steps // 4) else None,
        )
        state = runtime.initial_state(batch_size, device=device)
        previous_action = previous_reward = previous_propensity = None
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        value_features: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        for tick, streams in enumerate(sequence.streams):
            tick_start = time.perf_counter()
            feedback = _feedback(
                previous_action,
                previous_reward,
                previous_propensity,
                batch_size=batch_size,
                width=runtime.controller.feedback_width,
                device=device,
            )
            action, log_prob, entropy, propensity, features, state = _sample_action(
                runtime, streams, state, feedback
            )
            total_latency += time.perf_counter() - tick_start
            reward = verifier.outcome(action, sequence.targets[tick])
            visible_reward = (
                reward[torch.randperm(batch_size, device=device)]
                if reward_shuffle
                else reward
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
            value_features.append(features)
            rewards.append(reward)
            previous_action, previous_reward, previous_propensity = (
                action.detach(), visible_reward.detach(), propensity.detach()
            )
        reward_tensor = torch.stack(rewards)
        returns = torch.zeros_like(reward_tensor)
        running = torch.zeros(batch_size, device=device)
        for tick in range(verifier.sequence_length - 1, -1, -1):
            running = reward_tensor[tick] + 0.95 * running
            returns[tick] = running
        values = critic(torch.stack(value_features)).squeeze(-1)
        advantage = returns.detach() - values.detach()
        loss = -(
            torch.stack(log_probs) * advantage
        ).mean() + 0.5 * (values - returns.detach()).square().mean() - entropy_weight * torch.stack(entropies).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(reward_tensor.mean())
        if step == 1 or step % eval_every == 0 or step == steps:
            scores = evaluate_condition(
                runtime,
                CalibrationConflictVerifier(
                    seed=seed + 1000 + step,
                    device=device,
                    sequence_length=verifier.sequence_length,
                ),
                condition="clean",
                batches=4,
                batch_size=batch_size,
                device=device,
            )
            diagnostic_lifetimes += 4 * batch_size * verifier.sequence_length
            history.append({"step": step, "post_calibration_reward": scores["post_calibration"]})
    elapsed = time.perf_counter() - start
    stable_bits: int | None = None
    for index, point in enumerate(history):
        if point["post_calibration_reward"] >= threshold and all(
            later["post_calibration_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(
                point["step"] * batch_size * verifier.sequence_length * verifier.bit_count
            )
            break
    lifetimes = steps * batch_size * verifier.sequence_length
    return history, RunAccounting(
        unique_verifier_bits=lifetimes * verifier.bit_count,
        unique_logical_lifetimes=lifetimes,
        optimizer_updates=steps,
        replayed_examples=0,
        diagnostic_lifetimes_charged_to_budget=diagnostic_lifetimes,
        wall_time_seconds=elapsed,
        mean_inference_latency_ms=total_latency / lifetimes * 1000.0,
        stable_bits_to_threshold=stable_bits,
        retention_on_mastered_primitives=None,
        transfer_ratio_against_fresh_learner=None,
    )


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: CalibrationConflictVerifier,
    *,
    condition: str,
    batches: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    device = torch.device(device)
    runtime.eval()
    total = post_total = 0.0
    count = post_count = 0
    for _ in range(batches):
        sequence = verifier.sample(batch_size)
        state = runtime.initial_state(batch_size, device=device)
        previous_action = previous_reward = previous_propensity = None
        for tick, streams in enumerate(sequence.streams):
            feedback = _feedback(
                previous_action,
                previous_reward,
                previous_propensity,
                batch_size=batch_size,
                width=runtime.controller.feedback_width,
                device=device,
                no_feedback=condition == "no_feedback",
            )
            output, state = runtime.step_streams(streams, state, feedback)
            actions = output.decoded["protocol"].argmax(dim=-1)
            if condition == "action_shuffled":
                actions = actions[torch.randperm(batch_size, device=device)]
            elif condition in {"intention_shuffled", "intention_zero"}:
                payload = (
                    output.intention.payload[torch.randperm(batch_size, device=device)]
                    if condition == "intention_shuffled"
                    else torch.zeros_like(output.intention.payload)
                )
                intervention = output.intention.__class__(
                    payload=payload,
                    timestamp=output.intention.timestamp,
                    confidence=output.intention.confidence,
                    target_key=output.intention.target_key,
                )
                actions = runtime.output_bus(intervention)["protocol"].argmax(dim=-1)
            reward = verifier.outcome(actions, sequence.targets[tick])
            total += float(reward.sum())
            count += batch_size
            if tick > 0:
                post_total += float(reward.sum())
                post_count += batch_size
            visible_reward = reward
            if condition == "feedback_shuffled":
                visible_reward = reward[torch.randperm(batch_size, device=device)]
            previous_action, previous_reward, previous_propensity = (
                actions,
                visible_reward,
                torch.ones_like(reward),
            )
    return {
        "overall": total / count,
        "post_calibration": post_total / post_count,
    }


def run_experiment(
    *,
    steps: int,
    batch_size: int,
    seed: int,
    sequence_length: int = 4,
    device: torch.device | str = "cpu",
    report_out: Path | None = None,
    reward_shuffle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device)
    runtime = build_runtime(seed=seed).to(device)
    verifier = CalibrationConflictVerifier(
        seed=seed + 10, device=device, sequence_length=sequence_length
    )
    history, accounting = train_steps(
        runtime,
        verifier,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        reward_shuffle=reward_shuffle,
        device=device,
    )
    condition_names = [
        "clean",
        "stream_order_shuffled",
        "no_feedback",
        "feedback_shuffled",
        "action_shuffled",
        "intention_shuffled",
        "intention_zero",
    ]
    conditions: dict[str, dict[str, float]] = {}
    for index, condition in enumerate(condition_names):
        conditions[condition] = evaluate_condition(
            runtime,
            CalibrationConflictVerifier(
                seed=seed + 200 + index,
                device=device,
                sequence_length=sequence_length,
                stream_order_shuffle=condition == "stream_order_shuffled",
            ),
            condition=condition,
            batches=8,
            batch_size=batch_size,
            device=device,
        )
    audit_lifetimes = len(condition_names) * 8 * batch_size * sequence_length
    promotion = (
        conditions["clean"]["post_calibration"] >= 0.75
        and conditions["stream_order_shuffled"]["post_calibration"] >= 0.70
        and conditions["no_feedback"]["post_calibration"] <= 0.65
        and conditions["feedback_shuffled"]["post_calibration"] <= 0.65
        and conditions["action_shuffled"]["post_calibration"] <= 0.65
        and conditions["intention_shuffled"]["post_calibration"] <= 0.65
        and conditions["intention_zero"]["post_calibration"] <= 0.65
    )
    report = {
        "experiment": "outcome-only-source-trust-calibration",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "frontends": "frozen independent standardized neural-IR encoders",
        "learner_visible_feedback": [
            "standardized event tokens",
            "previous opaque action",
            "previous scalar verifier outcome",
        ],
        "hidden_from_learner": ["stable reliable source", "target action"],
        "history": history,
        "conditions": conditions,
        "promotion_gate": {
            "clean_post_calibration_min": 0.75,
            "stream_order_post_calibration_min": 0.70,
            "no_feedback_and_shuffled_feedback_max": 0.65,
            "intervention_max": 0.65,
        },
        "promoted": promotion,
        "accounting": {
            **asdict(accounting),
            "audit_lifetimes": audit_lifetimes,
            "diagnostic_lifetimes_charged_to_budget": (
                accounting.diagnostic_lifetimes_charged_to_budget + audit_lifetimes
            ),
        },
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--reward-shuffle", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
