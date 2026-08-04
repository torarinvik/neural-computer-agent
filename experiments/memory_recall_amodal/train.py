"""Train and audit scalar-outcome recall through the canonical memory boundary."""

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
from torch import nn
from torch.distributions import Categorical

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEventCollection,
    AmodalOutputBus,
    ContentAddressedMemory,
    ControllerFeedback,
    MemoryBackend,
    OpaqueProtocolDecoder,
    PersistentContentAddressedMemory,
)

from .environment import OutcomeRecallVerifier


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


def build_runtime(*, seed: int = 0, width: int = 16) -> AmodalControllerRuntime:
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
            width=width, capacity=1, write_threshold=0.5
        ),
    )


def _feedback(
    runtime: AmodalControllerRuntime,
    *,
    action: torch.Tensor | None = None,
    reward: torch.Tensor | None = None,
    propensity: torch.Tensor | None = None,
    has_feedback: torch.Tensor | None = None,
) -> ControllerFeedback:
    device = next(runtime.parameters()).device
    return ControllerFeedback(
        action=torch.zeros(1, 2, device=device) if action is None else action,
        reward=torch.zeros(1, device=device) if reward is None else reward,
        propensity=(
            torch.ones(1, device=device) if propensity is None else propensity
        ),
        has_feedback=(
            torch.zeros(1, device=device)
            if has_feedback is None
            else has_feedback
        ),
    )


def _empty(runtime: AmodalControllerRuntime) -> AmodalEventCollection:
    return AmodalEventCollection.empty(1, runtime.event_width, device=next(runtime.parameters()).device)


def _sample_probe(
    runtime: AmodalControllerRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, _ = runtime.step_events(
        _empty(runtime),
        runtime.initial_state(1, device=next(runtime.parameters()).device),
        _feedback(runtime),
    )
    distribution = Categorical(logits=output.decoded["protocol"])
    action = distribution.sample()
    propensity = distribution.probs.gather(1, action[:, None]).squeeze(1)
    return action, propensity


def _store_feedback(
    runtime: AmodalControllerRuntime,
    action: torch.Tensor,
    reward: torch.Tensor,
    propensity: torch.Tensor,
) -> Any:
    opaque_action = torch.nn.functional.one_hot(action, num_classes=2).to(torch.float32)
    return runtime.step_events(
        _empty(runtime),
        runtime.initial_state(1, device=next(runtime.parameters()).device),
        _feedback(
            runtime,
            action=opaque_action,
            reward=reward,
            propensity=propensity,
            has_feedback=torch.ones(1, device=reward.device),
        ),
    )[0]


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeRecallVerifier,
    *,
    steps: int,
    seed: int,
    learning_rate: float = 2e-3,
    entropy_weight: float = 0.01,
    reward_shuffle: bool = False,
    eval_every: int = 32,
    threshold: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], Accounting]:
    if steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(seed)
    device = torch.device(device)
    runtime.to(device).train()
    optimizer = torch.optim.Adam(runtime.parameters(), lr=learning_rate)
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
        tick_started = time.perf_counter()
        probe_action, probe_propensity = _sample_probe(runtime)
        probe_reward = verifier.score_probe(probe_action)
        store_reward = (
            torch.randint(0, 2, probe_reward.shape, device=device).to(torch.float32)
            if reward_shuffle
            else probe_reward
        )
        runtime.memory.clear()
        with runtime.memory.differentiable_transaction():
            store_output = _store_feedback(
                runtime, probe_action, store_reward, probe_propensity
            )
            write_strength_total += float(
                store_output.controller.memory_write_strength.item()
            )
            committed_writes += int(
                store_output.controller.memory_write_receipt.committed.item()
            )
            query_output, _ = runtime.step_events(
                _empty(runtime),
                runtime.initial_state(1, device=device),
                _feedback(runtime),
            )
            distribution = Categorical(logits=query_output.decoded["protocol"])
            query_action = distribution.sample()
            query_reward = verifier.score_recall(query_action)
            loss = -(
                (query_reward.detach() - baseline)
                * distribution.log_prob(query_action)
            ).mean() - entropy_weight * distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runtime.parameters(), max_norm=5.0)
            optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(query_reward)
        action_time += time.perf_counter() - tick_started

        if step == 1 or step % eval_every == 0 or step == steps:
            score = evaluate_condition(
                runtime,
                OutcomeRecallVerifier(seed=seed + 1000 + step, device=device),
                condition="intact",
                episodes=128,
                device=device,
            )
            history.append({"step": step, "intact_reward": score})

    stable_bits = None
    for index, point in enumerate(history):
        if point["intact_reward"] >= threshold and all(
            later["intact_reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(point["step"] * 2)
            break
    return history, Accounting(
        unique_verifier_bits=steps * 2,
        unique_logical_lifetimes=steps,
        optimizer_updates=steps,
        replayed_examples=0,
        verifier_outcome_events=steps * 2,
        feedback_events=steps,
        mean_memory_write_strength=write_strength_total / steps,
        committed_write_rate=committed_writes / steps,
        diagnostic_lifetimes_charged_to_budget=0,
        wall_time_seconds=time.perf_counter() - started,
        mean_action_latency_ms=action_time / steps * 1000.0,
        stable_bits_to_threshold=stable_bits,
        retention_on_mastered_primitives=None,
        transfer_ratio_against_fresh_learner=None,
    )


@torch.no_grad()
def evaluate_condition(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeRecallVerifier,
    *,
    condition: str,
    episodes: int,
    device: torch.device | str = "cpu",
) -> float:
    if condition not in {"intact", "clear", "corrupt", "fresh", "replacement", "random_action"}:
        raise ValueError(f"unknown memory condition {condition!r}")
    device = torch.device(device)
    runtime.eval()
    assert isinstance(runtime.memory, ContentAddressedMemory)
    total = 0.0
    with tempfile.TemporaryDirectory() as directory:
        for episode in range(episodes):
            verifier.reset()
            runtime.memory.clear()
            probe_action = torch.tensor([episode % 2], device=device)
            probe_propensity = torch.full((1,), 0.5, device=device)
            probe_reward = verifier.score_probe(probe_action)
            runtime.memory.clear()
            _store_feedback(runtime, probe_action, probe_reward, probe_propensity)
            memory: MemoryBackend = runtime.memory
            if condition in {"clear", "fresh"}:
                runtime.memory.clear()
            elif condition == "corrupt":
                runtime.memory.values[0].zero_()
            elif condition == "replacement":
                path = Path(directory) / "memory.pt"
                runtime.memory.snapshot(path)
                memory = PersistentContentAddressedMemory(
                    width=runtime.event_width,
                    capacity=1,
                    path=path,
                    write_threshold=runtime.memory.write_threshold,
                )
            if condition == "random_action":
                query_action = torch.randint(0, 2, (1,), device=device)
            else:
                query_output = runtime.controller.step(
                    _empty(runtime),
                    runtime.initial_state(1, device=device),
                    _feedback(runtime),
                    memory,
                )[0]
                query_action = runtime.output_bus(query_output.intention)["protocol"].argmax(dim=-1)
            total += float(verifier.score_recall(query_action).item())
    runtime.train()
    return total / episodes


def run_experiment(
    *,
    steps: int,
    seed: int,
    device: torch.device | str = "cpu",
    report_out: Path | None = None,
    reward_shuffle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(device)
    runtime = build_runtime(seed=seed).to(device)
    history, accounting = train_steps(
        runtime,
        OutcomeRecallVerifier(seed=seed + 10, device=device),
        steps=steps,
        seed=seed,
        reward_shuffle=reward_shuffle,
        device=device,
    )
    conditions = {
        condition: evaluate_condition(
            runtime,
            OutcomeRecallVerifier(seed=seed + 200 + index, device=device),
            condition=condition,
            episodes=512,
            device=device,
        )
        for index, condition in enumerate(
            ("intact", "clear", "corrupt", "fresh", "replacement", "random_action")
        )
    }
    promotion = (
        not reward_shuffle
        and conditions["intact"] >= 0.80
        and conditions["replacement"] >= 0.80
        and conditions["clear"] <= 0.65
        and conditions["corrupt"] <= 0.65
        and conditions["intact"] - conditions["clear"] >= 0.25
        and conditions["intact"] - conditions["corrupt"] >= 0.25
    )
    accounting_payload = asdict(accounting)
    accounting_payload["diagnostic_lifetimes_charged_to_budget"] = (
        len(history) * 128 + 6 * 512
    )
    accounting_payload["retention_on_mastered_primitives"] = conditions["replacement"]
    report: dict[str, Any] = {
        "experiment": "outcome-only-memory-recall",
        "seed": seed,
        "steps": steps,
        "batch_size": 1,
        "reward_shuffle_control": reward_shuffle,
        "learner_visible_inputs": ["opaque probe action", "scalar probe reward", "scalar recall reward"],
        "memory_training": {
            "write_threshold": 0.5,
            "differentiable_write_strength_path": True,
            "differentiable_transaction": True,
            "capacity": 1,
            "claim_boundary": "scalar outcome recall only; learned skip policy, multi-row retention, and content binding remain unqualified",
        },
        "history": history,
        "conditions": conditions,
        "promotion_gate": {
            "intact_min": 0.80,
            "replacement_min": 0.80,
            "clear_max": 0.65,
            "corrupt_max": 0.65,
            "causal_gap_min": 0.25,
        },
        "promoted": promotion,
        "accounting": accounting_payload,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--reward-shuffle", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
