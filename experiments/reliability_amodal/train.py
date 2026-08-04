"""Train and audit outcome-only contradictory-evidence handling."""

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

from .environment import RedundantComplementVerifier


class TaggedEventEncoder(nn.Module):
    """Independent raw frontend with opaque learned provenance."""

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
        return AmodalEvent(
            payload=self.network(raw),
            source_key=self.source_key.expand(raw.shape[0], -1),
            # Corruption is deliberately unmarked at the event boundary.
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
    """Build one controller with four independently replaceable encoders."""
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=2,
        intention_width=16,
        feedback_width=8,
        source_key_width=4,
        event_window_capacity=8,
        reliability_hidden=16,
        memory_top_k=1,
    )
    encoders = {}
    for index, name in enumerate(RedundantComplementVerifier.stream_names):
        key = [1.0 if index == position else 0.0 for position in range(4)]
        encoders[name] = TaggedEventEncoder(
            RedundantComplementVerifier.raw_width, width, key
        )
    return AmodalControllerRuntime(
        controller,
        encoders=encoders,
        output_bus=AmodalOutputBus(
            {
                "protocol": OpaqueProtocolDecoder(
                    16, RedundantComplementVerifier.action_count, hidden=16
                )
            }
        ),
    )


def _feedback(batch_size: int, width: int, device: torch.device) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch_size, width, device=device),
        reward=torch.zeros(batch_size, device=device),
        propensity=torch.ones(batch_size, device=device),
        has_feedback=torch.zeros(batch_size, device=device),
    )


def _sample_actions(
    runtime: AmodalControllerRuntime,
    streams: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    batch_size = next(iter(streams.values())).shape[0]
    output, _ = runtime.step_streams(
        streams,
        runtime.initial_state(batch_size, device=device),
        _feedback(batch_size, runtime.controller.feedback_width, device),
    )
    distribution = Categorical(logits=output.decoded["protocol"])
    actions = distribution.sample()
    propensities = distribution.probs.gather(1, actions[:, None]).squeeze(1)
    return actions, distribution.log_prob(actions), distribution.entropy(), propensities, output


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: RedundantComplementVerifier,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 3e-3,
    entropy_weight: float = 0.01,
    reward_shuffle: bool = False,
    freeze_encoders: bool = False,
    eval_every: int = 32,
    threshold: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], RunAccounting]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    seed_everything(seed)
    device = torch.device(device)
    runtime.to(device).train()
    if freeze_encoders:
        for encoder in runtime.encoders.values():
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
    trainable_parameters = [
        parameter for parameter in runtime.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    baseline = 0.25
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    total_latency = 0.0
    diagnostic_lifetimes = 0
    for step in range(1, steps + 1):
        streams = verifier.reset(batch_size)
        tick_start = time.perf_counter()
        actions, _, entropy, propensities, _ = _sample_actions(
            runtime, streams, device=device
        )
        reward = verifier.step(actions)
        total_latency += time.perf_counter() - tick_start
        learner_reward = (
            reward[torch.randperm(batch_size, device=device)] if reward_shuffle else reward
        )
        advantage = learner_reward - baseline
        loss = -(
            advantage.detach() * propensities.clamp_min(1e-8).log()
        ).mean() - entropy_weight * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(learner_reward.mean())
        if step == 1 or step % eval_every == 0 or step == steps:
            score = evaluate_condition(
                runtime,
                RedundantComplementVerifier(seed=seed + 1000 + step, device=device),
                condition="clean",
                batches=4,
                batch_size=batch_size,
                device=device,
            )
            diagnostic_lifetimes += 4 * batch_size
            history.append({"step": step, "clean_reward": score})
    elapsed = time.perf_counter() - start
    stable_bits: int | None = None
    for index, point in enumerate(history):
        if point["clean_reward"] >= threshold and all(
            later["clean_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(
                point["step"] * batch_size * RedundantComplementVerifier.bit_count
            )
            break
    lifetimes = steps * batch_size
    return history, RunAccounting(
        unique_verifier_bits=lifetimes * RedundantComplementVerifier.bit_count,
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


def _condition_verifier(seed: int, condition: str, device: torch.device) -> RedundantComplementVerifier:
    if condition in {
        "clean",
        "single_high",
        "single_low",
        "action_shuffled",
        "intention_shuffled",
        "intention_zero",
        "random_action",
    }:
        return RedundantComplementVerifier(seed=seed, device=device)
    if condition == "one_corruption":
        return RedundantComplementVerifier(
            seed=seed, device=device, corruption_probability=1.0
        )
    if condition == "one_noisy_source_corruption":
        return RedundantComplementVerifier(
            seed=seed, device=device, force_flip_mask=(False, True, False)
        )
    if condition == "one_missing":
        return RedundantComplementVerifier(
            seed=seed, device=device, missing_probability=1.0
        )
    if condition == "trusted_source_conflict":
        return RedundantComplementVerifier(
            seed=seed, device=device, force_flip_mask=(False, True, True)
        )
    if condition == "low_reliability_source_conflict":
        return RedundantComplementVerifier(
            seed=seed, device=device, force_flip_mask=(True, False, False)
        )
    if condition == "all_low_missing":
        return RedundantComplementVerifier(seed=seed, device=device, drop_all_low=True)
    if condition == "all_low_inverted":
        return RedundantComplementVerifier(
            seed=seed, device=device, invert_all_low=True
        )
    if condition == "stream_order_shuffled":
        return RedundantComplementVerifier(
            seed=seed, device=device, stream_order_shuffle=True
        )
    raise ValueError(f"unknown verifier condition {condition!r}")


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: RedundantComplementVerifier,
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
        streams = verifier.reset(batch_size)
        if condition == "all_low_missing":
            streams.pop("b", None)
            streams.pop("c", None)
            streams.pop("d", None)
        elif condition == "single_high":
            streams = {"a": streams["a"]}
        elif condition == "single_low":
            streams = {"b": streams["b"]}
        elif condition in {
            "clean",
            "one_corruption",
            "one_noisy_source_corruption",
            "one_missing",
            "trusted_source_conflict",
            "low_reliability_source_conflict",
            "all_low_inverted",
            "stream_order_shuffled",
            "action_shuffled",
            "intention_shuffled",
            "intention_zero",
        }:
            pass
        elif condition == "random_action":
            reward = verifier.step(
                torch.randint(0, verifier.action_count, (batch_size,), device=device)
            )
            total += float(reward.sum())
            count += batch_size
            continue
        else:
            raise ValueError(f"unknown audit condition {condition!r}")

        actions, _, _, _, output = _sample_actions(runtime, streams, device=device)
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
        reward = verifier.step(actions)
        total += float(reward.sum())
        count += batch_size
    return total / count


def run_experiment(
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
    report_out: Path | None = None,
    reward_shuffle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device)
    runtime = build_runtime(seed=seed).to(device)
    fresh_runtime = build_runtime(seed=seed + 1).to(device)
    passive = evaluate_condition(
        fresh_runtime,
        RedundantComplementVerifier(seed=seed + 50, device=device),
        condition="clean",
        batches=4,
        batch_size=batch_size,
        device=device,
    )
    verifier = RedundantComplementVerifier(
        seed=seed + 10,
        device=device,
        source_flip_probabilities=(0.05, 0.35, 0.35),
        missing_probability=0.33,
    )
    history, accounting = train_steps(
        runtime,
        verifier,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        reward_shuffle=reward_shuffle,
        freeze_encoders=True,
        device=device,
    )
    condition_names = [
        "clean",
        "one_corruption",
        "one_noisy_source_corruption",
        "one_missing",
        "trusted_source_conflict",
        "low_reliability_source_conflict",
        "all_low_missing",
        "single_high",
        "single_low",
        "all_low_inverted",
        "stream_order_shuffled",
        "action_shuffled",
        "intention_shuffled",
        "intention_zero",
        "random_action",
    ]
    conditions: dict[str, float] = {}
    for index, condition in enumerate(condition_names):
        verifier_for_condition = _condition_verifier(seed + 200 + index, condition, device)
        conditions[condition] = evaluate_condition(
            runtime,
            verifier_for_condition,
            condition=condition,
            batches=8,
            batch_size=batch_size,
            device=device,
        )
    audit_lifetimes = len(condition_names) * 8 * batch_size + 4 * batch_size
    promotion = (
        conditions["clean"] >= 0.85
        and conditions["one_noisy_source_corruption"] >= 0.85
        and conditions["trusted_source_conflict"] >= 0.80
        and conditions["stream_order_shuffled"] >= 0.85
        and conditions["all_low_missing"] <= 0.70
        and conditions["low_reliability_source_conflict"] <= 0.20
        and conditions["action_shuffled"] <= 0.40
        and conditions["intention_shuffled"] <= 0.40
        and conditions["intention_zero"] <= 0.40
    )
    report = {
        "experiment": "outcome-only-redundant-reliability",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "raw_streams": "one high-bit stream plus three redundant low-bit streams",
        "frontends": "frozen random neural-IR encoders; reliability is learned in the controller",
        "learner_visible_feedback": [
            "rendered event tokens",
            "scalar verifier reward",
            "exact sampled-action propensity",
        ],
        "history": history,
        "passive_fresh_clean_reward": passive,
        "conditions": conditions,
        "promotion_gate": {
            "clean_min": 0.85,
            "one_noisy_source_corruption_min": 0.85,
            "trusted_source_conflict_min": 0.80,
            "stream_order_shuffled_min": 0.85,
            "all_low_missing_max": 0.70,
            "intervention_max": 0.40,
        },
        "promoted": promotion,
        "accounting": {
            **asdict(accounting),
            "audit_lifetimes": audit_lifetimes,
            "diagnostic_lifetimes_charged_to_budget": (
                accounting.diagnostic_lifetimes_charged_to_budget + audit_lifetimes
            ),
        },
        "memory_corruption_control": "not_applicable: isolated reliability rung has no external memory component",
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--reward-shuffle", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
