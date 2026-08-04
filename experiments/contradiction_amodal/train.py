"""Train and audit adaptive outcome-only contradiction resolution."""

from __future__ import annotations

import argparse
import json
import random
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
    AmodalOutputBus,
    ControllerFeedback,
    OpaqueProtocolDecoder,
)

from .environment import ConflictSequence, SequentialConflictVerifier


class FrozenEventEncoder(nn.Module):
    """Independent raw frontend with generic learned provenance."""

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
    # The production controller keeps these interactions zero-initialized for
    # checkpoint compatibility. This experiment deliberately opens the
    # generic outcome/source route so the reversal rung tests learned temporal
    # trust rather than a permanently silent feedback path.
    with torch.no_grad():
        controller.event_feedback_relevance.weight.normal_(mean=0.0, std=0.02)
        controller.event_feedback_source_relevance.weight.normal_(mean=0.0, std=0.02)
    encoders = {}
    for index, name in enumerate(SequentialConflictVerifier.stream_names):
        encoders[name] = FrozenEventEncoder(
            SequentialConflictVerifier.raw_width,
            width,
            [1.0 if index == position else 0.0 for position in range(2)],
        )
    return AmodalControllerRuntime(
        controller,
        encoders=encoders,
        output_bus=AmodalOutputBus(
            {
                "protocol": OpaqueProtocolDecoder(
                    16, SequentialConflictVerifier.action_count, hidden=16
                )
            }
        ),
    )


def _feedback(
    *,
    previous_action: torch.Tensor | None,
    previous_reward: torch.Tensor | None,
    previous_propensity: torch.Tensor | None,
    feedback_width: int,
    batch_size: int | None = None,
    device: torch.device | None = None,
    no_feedback: bool = False,
) -> ControllerFeedback:
    if previous_action is None or no_feedback:
        batch_size = (
            previous_reward.shape[0]
            if previous_reward is not None
            else previous_action.shape[0]
            if previous_action is not None
            else batch_size
        )
        if batch_size is None:
            raise ValueError("initial feedback requires batch_size")
        device = (
            previous_reward.device
            if previous_reward is not None
            else previous_action.device
            if previous_action is not None
            else device
        )
        if device is None:
            raise ValueError("initial feedback requires device")
        return ControllerFeedback(
            action=torch.zeros(batch_size, feedback_width, device=device),
            reward=torch.zeros(batch_size, device=device),
            propensity=torch.ones(batch_size, device=device),
            has_feedback=torch.zeros(batch_size, device=device),
        )
    action = torch.zeros(
        previous_action.shape[0], feedback_width, device=previous_action.device
    )
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
    streams: dict[str, torch.Tensor],
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
    verifier: SequentialConflictVerifier,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 3e-3,
    entropy_weight: float = 0.01,
    reward_shuffle: bool = False,
    freeze_encoders: bool = True,
    curriculum: bool = True,
    eval_every: int = 32,
    threshold: float = 0.72,
    return_discount: float = 0.95,
    random_reversal_probability: float = 0.0,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], RunAccounting]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if not 0.0 <= return_discount <= 1.0:
        raise ValueError("return_discount must be in [0, 1]")
    if not 0.0 <= random_reversal_probability <= 1.0:
        raise ValueError("random_reversal_probability must be in [0, 1]")
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
        if curriculum and step <= max(1, steps // 4):
            roles = torch.zeros(
                batch_size, verifier.block_count, dtype=torch.long, device=device
            )
            sequence = verifier.sample(batch_size, roles=roles)
        elif curriculum and step <= max(2, steps // 2):
            roles = torch.arange(
                verifier.block_count, device=device, dtype=torch.long
            ).remainder(2)
            roles = roles.unsqueeze(0).expand(batch_size, -1)
            sequence = verifier.sample(batch_size, roles=roles)
        elif curriculum:
            if (
                random_reversal_probability > 0.0
                and verifier.block_count >= 3
                and bool(torch.rand((), device=device) < random_reversal_probability)
            ):
                sequence = verifier.sample_random_reversal(batch_size)
            else:
                sequence = verifier.sample_markov_roles(batch_size)
        else:
            sequence = verifier.sample(batch_size)
        state = runtime.initial_state(batch_size, device=device)
        previous_action = previous_reward = previous_propensity = None
        feedback = _feedback(
            previous_action=None,
            previous_reward=None,
            previous_propensity=None,
            feedback_width=runtime.controller.feedback_width,
            batch_size=batch_size,
            device=device,
        )
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        value_features: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        for tick, streams in enumerate(sequence.streams):
            tick_start = time.perf_counter()
            action, log_prob, entropy, propensity, features, state = _sample_action(
                runtime, dict(streams), state, feedback
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
                action.detach(),
                visible_reward.detach(),
                propensity.detach(),
            )
            feedback = _feedback(
                previous_action=previous_action,
                previous_reward=previous_reward,
                previous_propensity=previous_propensity,
                feedback_width=runtime.controller.feedback_width,
            )
        reward_tensor = torch.stack(rewards)
        returns = torch.zeros_like(reward_tensor)
        running = torch.zeros(batch_size, device=device)
        for tick in range(verifier.sequence_length - 1, -1, -1):
            running = reward_tensor[tick] + return_discount * running
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
            score = evaluate_condition(
                runtime,
                SequentialConflictVerifier(
                    seed=seed + 1000 + step,
                    device=device,
                    sequence_length=verifier.sequence_length,
                    block_length=verifier.block_length,
                ),
                condition="markov",
                batches=2,
                batch_size=batch_size,
                device=device,
            )["overall"]
            diagnostic_lifetimes += 2 * batch_size * verifier.sequence_length
            history.append({"step": step, "markov_reward": score})
    elapsed = time.perf_counter() - start
    stable_bits: int | None = None
    for index, point in enumerate(history):
        if point["markov_reward"] >= threshold and all(
            later["markov_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(
                point["step"]
                * batch_size
                * verifier.sequence_length
                * verifier.bit_count
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
    verifier: SequentialConflictVerifier,
    *,
    condition: str,
    batches: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    device = torch.device(device)
    runtime.eval()
    total_reward = 0.0
    total_count = 0
    post_transition_reward = 0.0
    post_transition_count = 0
    for batch_index in range(batches):
        if condition == "reversal":
            switch = verifier.block_count // 2
            roles = torch.zeros(
                batch_size, verifier.block_count, dtype=torch.long, device=device
            )
            roles[:, switch:] = 1
            sequence = verifier.sample(batch_size, roles=roles)
        elif condition == "markov":
            sequence = verifier.sample_markov_roles(batch_size)
        else:
            sequence = verifier.sample_markov_roles(batch_size)
        role_steps = sequence.roles.repeat_interleave(verifier.block_length, dim=1)
        role_changes = role_steps[:, 1:] != role_steps[:, :-1]
        has_change = role_changes.any(dim=1)
        first_change = torch.where(
            has_change,
            role_changes.to(torch.long).argmax(dim=1) + 1,
            torch.full((batch_size,), verifier.sequence_length, device=device),
        )
        state = runtime.initial_state(batch_size, device=device)
        previous_action = previous_reward = previous_propensity = None
        feedback = _feedback(
            previous_action=None,
            previous_reward=None,
            previous_propensity=None,
            feedback_width=runtime.controller.feedback_width,
            batch_size=batch_size,
            device=device,
        )
        for tick, streams in enumerate(sequence.streams):
            output, state = runtime.step_streams(dict(streams), state, feedback)
            action = output.decoded["protocol"].argmax(dim=-1)
            if condition == "action_shuffled":
                action = action[torch.randperm(batch_size, device=device)]
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
                action = runtime.output_bus(intervention)["protocol"].argmax(dim=-1)
            reward = verifier.outcome(action, sequence.targets[tick])
            total_reward += float(reward.sum())
            total_count += batch_size
            post_mask = torch.full(
                (batch_size,), tick, device=device, dtype=torch.long
            ) > first_change
            if post_mask.any():
                post_transition_reward += float(reward[post_mask].sum())
                post_transition_count += int(post_mask.sum())
            visible_reward = reward
            if condition == "feedback_shuffled":
                visible_reward = reward[torch.randperm(batch_size, device=device)]
            if condition == "no_feedback":
                feedback = _feedback(
                    previous_action=None,
                    previous_reward=None,
                    previous_propensity=None,
                    feedback_width=runtime.controller.feedback_width,
                    batch_size=batch_size,
                    device=device,
                    no_feedback=True,
                )
            else:
                previous_action = action
                previous_reward = visible_reward
                previous_propensity = torch.ones_like(reward)
                feedback = _feedback(
                    previous_action=previous_action,
                    previous_reward=previous_reward,
                    previous_propensity=previous_propensity,
                    feedback_width=runtime.controller.feedback_width,
                )
    return {
        "overall": total_reward / max(total_count, 1),
        "post_transition": post_transition_reward / max(post_transition_count, 1),
    }


def run_experiment(
    *,
    steps: int,
    batch_size: int,
    seed: int,
    sequence_length: int = 32,
    block_length: int = 8,
    device: torch.device | str = "cpu",
    report_out: Path | None = None,
    reward_shuffle: bool = False,
    learning_rate: float = 1e-3,
    return_discount: float = 0.95,
    random_reversal_probability: float = 0.0,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device)
    runtime = build_runtime(seed=seed).to(device)
    passive = evaluate_condition(
        build_runtime(seed=seed + 1).to(device),
        SequentialConflictVerifier(
            seed=seed + 50,
            device=device,
            sequence_length=sequence_length,
            block_length=block_length,
        ),
        condition="markov",
        batches=2,
        batch_size=batch_size,
        device=device,
    )["overall"]
    verifier = SequentialConflictVerifier(
        seed=seed + 10,
        device=device,
        sequence_length=sequence_length,
        block_length=block_length,
    )
    history, accounting = train_steps(
        runtime,
        verifier,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        reward_shuffle=reward_shuffle,
        freeze_encoders=True,
        learning_rate=learning_rate,
        return_discount=return_discount,
        random_reversal_probability=random_reversal_probability,
        device=device,
    )
    condition_names = [
        "markov",
        "reversal",
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
            SequentialConflictVerifier(
                seed=seed + 200 + index,
                device=device,
                sequence_length=sequence_length,
                block_length=block_length,
                stream_order_shuffle=condition == "stream_order_shuffled",
            ),
            condition=condition,
            batches=4,
            batch_size=batch_size,
            device=device,
        )
    audit_lifetimes = len(condition_names) * 4 * batch_size * sequence_length
    promotion = (
        conditions["markov"]["overall"] >= 0.60
        and conditions["markov"]["post_transition"] >= 0.60
        and conditions["reversal"]["post_transition"] >= 0.75
        and conditions["stream_order_shuffled"]["overall"] >= 0.70
        and conditions["no_feedback"]["post_transition"] <= 0.65
        and conditions["feedback_shuffled"]["post_transition"] <= 0.65
        and conditions["action_shuffled"]["overall"] <= 0.65
        and conditions["intention_shuffled"]["overall"] <= 0.65
        and conditions["intention_zero"]["overall"] <= 0.65
    )
    report = {
        "experiment": "outcome-only-sequential-contradiction-resolution",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "block_length": block_length,
        "learning_rate": learning_rate,
        "return_discount": return_discount,
        "random_reversal_probability": random_reversal_probability,
        "raw_streams": "two contradictory candidate events with a hidden reversible reliable source",
        "frontends": "frozen independent standardized neural-IR encoders; source trust is inferred by the controller",
        "learner_visible_feedback": [
            "rendered event tokens",
            "previous opaque action",
            "previous scalar verifier outcome",
            "exact sampled-action propensity during training",
        ],
        "training_curriculum": "one fixed source, then alternating roles, then stochastic Markov source roles",
        "history": history,
        "passive_fresh_markov_reward": passive,
        "conditions": conditions,
        "promotion_gate": {
            "markov_overall_min": 0.60,
            "markov_post_transition_min": 0.60,
            "reversal_post_transition_min": 0.75,
            "stream_order_shuffled_overall_min": 0.70,
            "no_feedback_post_transition_max": 0.65,
            "feedback_shuffled_post_transition_max": 0.65,
            "intervention_overall_max": 0.65,
        },
        "promoted": promotion,
        "accounting": {
            **asdict(accounting),
            "audit_lifetimes": audit_lifetimes,
            "diagnostic_lifetimes_charged_to_budget": (
                accounting.diagnostic_lifetimes_charged_to_budget + audit_lifetimes
            ),
        },
        "memory_corruption_control": "not_applicable: isolated sequential contradiction rung has no external memory component",
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--reward-shuffle", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--return-discount", type=float, default=0.95)
    parser.add_argument("--random-reversal-probability", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
