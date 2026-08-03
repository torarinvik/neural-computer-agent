"""Train and audit the promoted outcome-only multimodal experiment.

This module is intentionally a policy-gradient experiment.  The learner never
receives the verifier target, a correct action, or a semantic task label.  Its
only learning signal is the scalar reward returned after sampling an opaque
four-way protocol action.
"""

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
    save_runtime,
)

from .environment import OutcomeOnlyComplementVerifier


class TaggedEventEncoder(nn.Module):
    """Independent raw frontend that emits a learned event with provenance."""

    def __init__(self, raw_width: int, event_width: int, source_key: list[float]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(raw_width, event_width),
            nn.Tanh(),
            nn.Linear(event_width, event_width),
            nn.Tanh(),
        )
        key = torch.tensor(source_key, dtype=torch.float32).reshape(1, -1)
        self.register_buffer("source_key", key)

    def forward(self, raw: torch.Tensor) -> AmodalEvent:
        if raw.ndim != 2:
            raise ValueError("raw stream must have shape [batch, width]")
        batch = raw.shape[0]
        return AmodalEvent(
            payload=self.network(raw),
            source_key=self.source_key.expand(batch, -1),
            timestamp=torch.zeros(batch, device=raw.device, dtype=raw.dtype),
            duration=torch.ones(batch, device=raw.device, dtype=raw.dtype),
            confidence=torch.ones(batch, device=raw.device, dtype=raw.dtype),
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
    """Build the canonical runtime used by training and checkpoint loading."""
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
    runtime = AmodalControllerRuntime(
        controller,
        encoders={
            "a": TaggedEventEncoder(OutcomeOnlyComplementVerifier.raw_width, width, [1.0, 0.0]),
            "b": TaggedEventEncoder(OutcomeOnlyComplementVerifier.raw_width, width, [0.0, 1.0]),
        },
        output_bus=AmodalOutputBus(
            {"protocol": OpaqueProtocolDecoder(16, OutcomeOnlyComplementVerifier.action_count, hidden=16)}
        ),
    )
    return runtime


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
    state = runtime.initial_state(batch_size, device=device)
    output, _ = runtime.step_streams(
        streams,
        state,
        _feedback(batch_size, runtime.controller.feedback_width, device),
    )
    logits = output.decoded["protocol"]
    distribution = Categorical(logits=logits)
    actions = distribution.sample()
    propensities = distribution.probs.gather(1, actions[:, None]).squeeze(1)
    return actions, distribution.log_prob(actions), distribution.entropy(), propensities, output


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyComplementVerifier,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 3e-3,
    entropy_weight: float = 0.01,
    reward_shuffle: bool = False,
    eval_every: int = 32,
    threshold: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], RunAccounting]:
    """Run one bandit learner using reward and logging propensity only."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    seed_everything(seed)
    device = torch.device(device)
    runtime.to(device)
    runtime.train()
    optimizer = torch.optim.Adam(runtime.parameters(), lr=learning_rate)
    baseline = 0.25
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    total_latency = 0.0
    unique_lifetimes = 0
    diagnostic_lifetimes = 0
    for step in range(1, steps + 1):
        streams = verifier.reset(batch_size)
        tick_start = time.perf_counter()
        actions, _, entropy, propensities, _ = _sample_actions(runtime, streams, device=device)
        reward = verifier.step(actions)
        total_latency += time.perf_counter() - tick_start
        unique_lifetimes += batch_size
        learner_reward = reward
        if reward_shuffle:
            learner_reward = reward[torch.randperm(batch_size, device=device)]
        advantage = learner_reward - baseline
        exact_log_propensity = propensities.clamp_min(1e-8).log()
        loss = -(advantage.detach() * exact_log_propensity).mean() - entropy_weight * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runtime.parameters(), max_norm=5.0)
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(learner_reward.mean())

        if step == 1 or step % eval_every == 0 or step == steps:
            score = evaluate_condition(runtime, verifier, condition="fused", batches=4, batch_size=batch_size, device=device)
            diagnostic_lifetimes += 4 * batch_size
            history.append({"step": step, "fused_reward": score})

    elapsed = time.perf_counter() - start
    stable_bits: int | None = None
    for index, point in enumerate(history):
        if point["fused_reward"] >= threshold and all(
            later["fused_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(point["step"] * batch_size * OutcomeOnlyComplementVerifier.bit_count)
            break
    accounting = RunAccounting(
        unique_verifier_bits=unique_lifetimes * OutcomeOnlyComplementVerifier.bit_count,
        unique_logical_lifetimes=unique_lifetimes,
        optimizer_updates=steps,
        replayed_examples=0,
        diagnostic_lifetimes_charged_to_budget=diagnostic_lifetimes,
        wall_time_seconds=elapsed,
        mean_inference_latency_ms=(total_latency / unique_lifetimes) * 1000.0,
        stable_bits_to_threshold=stable_bits,
        retention_on_mastered_primitives=None,
        transfer_ratio_against_fresh_learner=None,
    )
    return history, accounting


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyComplementVerifier,
    *,
    condition: str,
    batches: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate using rewards only; interventions happen at the opaque IR."""
    device = torch.device(device)
    runtime.eval()
    total = 0.0
    count = 0
    for _ in range(batches):
        streams = dict(verifier.reset(batch_size))
        if condition == "a_only" or condition == "missing_b":
            streams.pop("b")
        elif condition == "b_only" or condition == "missing_a":
            streams.pop("a")
        elif condition == "shuffled_partner":
            streams["b"] = streams["b"][torch.randperm(batch_size, device=device)]
        elif condition in {"fused", "action_shuffled", "intention_shuffled", "intention_zero"}:
            pass
        elif condition == "random_action":
            streams = streams
        else:
            raise ValueError(f"unknown evaluation condition {condition!r}")

        if condition == "random_action":
            actions = torch.randint(0, verifier.action_count, (batch_size,), device=device)
        else:
            actions, _, _, _, output = _sample_actions(runtime, streams, device=device)
            if condition == "action_shuffled":
                actions = actions[torch.randperm(batch_size, device=device)]
            elif condition in {"intention_shuffled", "intention_zero"}:
                if condition == "intention_shuffled":
                    payload = output.intention.payload[torch.randperm(batch_size, device=device)]
                else:
                    payload = torch.zeros_like(output.intention.payload)
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
    checkpoint_out: Path | None = None,
    report_out: Path | None = None,
    reward_shuffle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device)
    runtime = build_runtime(seed=seed)
    verifier = OutcomeOnlyComplementVerifier(seed=seed + 100, device=device)
    fresh_runtime = build_runtime(seed=seed + 1).to(device)
    fresh_verifier = OutcomeOnlyComplementVerifier(seed=seed + 101, device=device)
    passive = evaluate_condition(fresh_runtime, fresh_verifier, condition="fused", batches=8, batch_size=batch_size, device=device)
    history, accounting = train_steps(
        runtime,
        verifier,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        reward_shuffle=reward_shuffle,
        device=device,
    )
    audit_conditions = [
        "fused",
        "a_only",
        "b_only",
        "missing_a",
        "missing_b",
        "shuffled_partner",
        "action_shuffled",
        "intention_shuffled",
        "intention_zero",
        "random_action",
    ]
    controls = {
        name: evaluate_condition(
            runtime,
            OutcomeOnlyComplementVerifier(seed=seed + 200 + index, device=device),
            condition=name,
            batches=8,
            batch_size=batch_size,
            device=device,
        )
        for index, name in enumerate(audit_conditions)
    }
    reversal = evaluate_condition(
        runtime,
        OutcomeOnlyComplementVerifier(seed=seed + 300, reverse=True, device=device),
        condition="fused",
        batches=8,
        batch_size=batch_size,
        device=device,
    )
    audit_lifetimes = (8 + len(audit_conditions) * 8 + 8) * batch_size
    promotion = (
        controls["fused"] >= 0.80
        and max(controls["a_only"], controls["b_only"]) <= 0.65
        and controls["shuffled_partner"] <= 0.65
        and controls["fused"] - controls["shuffled_partner"] >= 0.25
        and controls["action_shuffled"] <= 0.40
        and controls["intention_shuffled"] <= 0.40
        and controls["intention_zero"] <= 0.40
    )
    if checkpoint_out is not None:
        save_runtime(
            runtime,
            checkpoint_out,
            provenance={
                "experiment": "outcome-only-complement",
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "reward_shuffle": reward_shuffle,
                "promoted": promotion,
            },
        )
    report = {
        "experiment": "outcome-only-complement",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "reward_shuffle_control": reward_shuffle,
        "raw_streams": {"a": "hidden high bit + independent distractors", "b": "hidden low bit + independent distractors"},
        "learner_visible_feedback": ["scalar verifier reward", "exact sampled-action propensity"],
        "history": history,
        "passive_fresh_reward": passive,
        "conditions": controls,
        "reversal_reward": reversal,
        "promotion_gate": {
            "fused_min": 0.80,
            "single_stream_max": 0.65,
            "partial_evidence_max": 0.65,
            "shuffle_gap_min": 0.25,
            "intention_intervention_max": 0.40,
        },
        "promoted": promotion,
        "accounting": {
            **asdict(accounting),
            "audit_lifetimes": audit_lifetimes,
            "diagnostic_lifetimes_charged_to_budget": (
                accounting.diagnostic_lifetimes_charged_to_budget + audit_lifetimes
            ),
        },
        "memory_corruption_control": "not_applicable: isolated controller run has no external memory component",
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--reward-shuffle", action="store_true")
    args = parser.parse_args()
    report = run_experiment(**vars(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
