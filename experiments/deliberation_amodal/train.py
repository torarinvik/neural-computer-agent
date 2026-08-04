"""Train and audit a one-controller variable-deliberation policy.

The execution head is trained with the same outcome-only policy-gradient
signal as the opaque action decoder.  No difficulty label, target, or correct
action is passed to the controller.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AmodalOutputBus,
    ControllerFeedback,
    OpaqueProtocolDecoder,
)

from .environment import BalancedDeliberationCurriculum, VariableDeliberationVerifier


EXECUTION_COST = {"wait": 0.20, "think": 0.35, "commit": 0.0}


class TaggedEventEncoder(nn.Module):
    """Independent raw frontend that emits an opaque event token."""

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

    def forward(self, raw: torch.Tensor):
        if raw.ndim != 2:
            raise ValueError("raw stream must have shape [batch, width]")
        return AmodalEvent(
            payload=self.network(raw),
            source_key=self.source_key.expand(raw.shape[0], -1),
            confidence=raw[:, -1].clamp_min(0.05),
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


class ExecutionValueBaseline(nn.Module):
    """Private state-conditioned baseline trained only from scalar utility."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, execution_logits: torch.Tensor) -> torch.Tensor:
        return self.network(execution_logits).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(*, seed: int = 0, width: int = 32) -> AmodalControllerRuntime:
    """Build the canonical one-controller runtime for this experiment."""
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
        execution_hidden=16,
    )
    return AmodalControllerRuntime(
        controller,
        encoders={
            "a": TaggedEventEncoder(VariableDeliberationVerifier.raw_width, width, [1.0, 0.0]),
            "b": TaggedEventEncoder(VariableDeliberationVerifier.raw_width, width, [0.0, 1.0]),
        },
        output_bus=AmodalOutputBus(
            {"protocol": OpaqueProtocolDecoder(16, 4, hidden=16)}
        ),
    )


def _feedback(width: int, device: torch.device) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, width, device=device),
        reward=torch.zeros(1, device=device),
        propensity=torch.ones(1, device=device),
        has_feedback=torch.zeros(1, device=device),
    )


def _quiet_feedback(width: int, device: torch.device) -> ControllerFeedback:
    return _feedback(width, device)


def _step_streams(
    runtime: AmodalControllerRuntime,
    streams: Mapping[str, torch.Tensor],
    state,
    feedback: ControllerFeedback,
    *,
    device: torch.device,
    elapsed: float = 1.0,
):
    if streams:
        return runtime.step_streams(streams, state, feedback, elapsed=elapsed)
    return runtime.step_streams(
        {},
        state,
        feedback,
        elapsed=elapsed,
        batch_size=1,
        device=device,
    )


def _sample_rollout(
    runtime: AmodalControllerRuntime,
    verifier: VariableDeliberationVerifier,
    *,
    device: torch.device,
    mode_override: str | None = None,
):
    """Sample one execution decision and one opaque action."""
    streams = verifier.reset(1)
    state = runtime.initial_state(1, device=device)
    initial, state = _step_streams(
        runtime, streams, state, _feedback(runtime.controller.feedback_width, device), device=device
    )
    execution_distribution = Categorical(logits=initial.execution_logits[0])
    execution_entropy = (
        execution_distribution.entropy()
        if mode_override is None
        else initial.execution_logits.new_zeros(())
    )
    if mode_override is None:
        execution_index = execution_distribution.sample()
        mode = ("wait", "think", "commit")[int(execution_index)]
        execution_log_prob = execution_distribution.log_prob(execution_index)
    elif mode_override == "transport":
        if "b" in streams:
            mode = "commit"
        elif float(streams["a"][:, -1].mean()) < 0.5:
            mode = "think"
        else:
            mode = "wait"
        execution_log_prob = initial.execution_logits.new_zeros(())
    else:
        if mode_override not in EXECUTION_COST:
            raise ValueError(f"unknown execution mode {mode_override!r}")
        mode = mode_override
        execution_log_prob = initial.execution_logits.new_zeros(())

    execution_cost = EXECUTION_COST[mode]
    final = initial
    if mode == "wait":
        if "b" in streams:
            # A complete initial window already contains the partner event.
            # Waiting must not reinterpret it as a missing-stream timeout.
            final = initial
        else:
            released = verifier.release_delayed()
            if released:
                final, _ = _step_streams(
                    runtime,
                    released,
                    state,
                    _quiet_feedback(runtime.controller.feedback_width, device),
                    device=device,
                )
            else:
                # A bounded timeout is a second decision from the same controller,
                # not a hidden missing-stream label. The timeout tick is quiet and
                # preserves the partial event window plus its learned age.
                timeout, state = _step_streams(
                    runtime,
                    {},
                    state,
                    _quiet_feedback(runtime.controller.feedback_width, device),
                    device=device,
                )
                if mode_override == "transport":
                    timeout_mode = "commit"
                    timeout_log_prob = timeout.execution_logits.new_zeros(())
                    timeout_entropy = timeout.execution_logits.new_zeros(())
                else:
                    timeout_distribution = Categorical(logits=timeout.execution_logits[0])
                    timeout_index = timeout_distribution.sample()
                    timeout_mode = ("wait", "think", "commit")[int(timeout_index)]
                    timeout_log_prob = timeout_distribution.log_prob(timeout_index)
                    timeout_entropy = timeout_distribution.entropy()
                execution_log_prob = execution_log_prob + timeout_log_prob
                execution_entropy = execution_entropy + timeout_entropy
                execution_cost += EXECUTION_COST[timeout_mode]
                final = timeout
                if timeout_mode == "think":
                    quiet = AmodalEventCollection.empty(1, runtime.event_width, device=device)
                    _, state = runtime.step_events(
                        quiet,
                        state,
                        _quiet_feedback(runtime.controller.feedback_width, device),
                        elapsed=1.0,
                    )
                    final, _ = _step_streams(
                        runtime,
                        verifier.release_delayed(after_think=True),
                        state,
                        _quiet_feedback(runtime.controller.feedback_width, device),
                        device=device,
                    )
    elif mode == "think":
        quiet = AmodalEventCollection.empty(1, runtime.event_width, device=device)
        final, state = runtime.step_events(
            quiet,
            state,
            _quiet_feedback(runtime.controller.feedback_width, device),
            elapsed=1.0,
        )
        released = verifier.release_delayed(after_think=True)
        if released:
            final, _ = _step_streams(
                runtime,
                released,
                state,
                _quiet_feedback(runtime.controller.feedback_width, device),
                device=device,
            )

    action_distribution = Categorical(logits=final.decoded["protocol"][0])
    action = action_distribution.sample()
    reward = verifier.step(action.reshape(1))
    action_log_prob = action_distribution.log_prob(action)
    entropy = execution_entropy + action_distribution.entropy()
    return (
        mode,
        reward,
        execution_log_prob,
        action_log_prob,
        entropy,
        initial.execution_logits[0],
        execution_cost,
    )


def train_steps(
    runtime: AmodalControllerRuntime,
    verifier: VariableDeliberationVerifier,
    *,
    steps: int,
    seed: int,
    learning_rate: float = 3e-3,
    execution_learning_rate: float = 3e-3,
    warmup_steps: int = 4096,
    entropy_weight: float = 0.01,
    eval_every: int = 32,
    train_timeout_policy: bool = False,
    threshold: float = 0.8,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int]], RunAccounting]:
    if steps < 1 or warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps cannot be negative")
    seed_everything(seed)
    device = torch.device(device)
    runtime.to(device).train()
    base_parameters = [
        parameter
        for name, parameter in runtime.named_parameters()
        if not name.startswith("controller.execution_policy.")
        and not name.startswith("controller.execution_transport_policy.")
        and not name.startswith("controller.execution_timeout_policy.")
    ]
    execution_parameters = [
        *runtime.controller.execution_policy.parameters(),
        *runtime.controller.execution_transport_policy.parameters(),
    ]
    timeout_parameters = list(runtime.controller.execution_timeout_policy.parameters())
    for parameter in timeout_parameters:
        parameter.requires_grad_(train_timeout_policy)
    if train_timeout_policy:
        execution_parameters.extend(timeout_parameters)
    optimizer = torch.optim.Adam(base_parameters, lr=learning_rate)
    baseline = 0.35
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    total_latency = 0.0
    diagnostic_lifetimes = 0

    # Stabilize the action/intention path with an observable transport-only
    # schedule. Complete windows commit immediately; partial windows wait for
    # their next event. The execution head receives no gradient here.
    for _ in range(warmup_steps):
        tick_start = time.perf_counter()
        mode, reward, _, action_log_prob, entropy, _, execution_cost = _sample_rollout(
            runtime,
            verifier,
            device=device,
            mode_override="transport",
        )
        total_latency += time.perf_counter() - tick_start
        utility = reward - execution_cost
        advantage = utility - baseline
        loss = -(advantage.detach() * action_log_prob) - entropy_weight * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_parameters, max_norm=5.0)
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * float(utility.item())

    # With the opaque action path stabilized, train only the execution head
    # from verifier outcomes. This isolates execution credit assignment from
    # representation drift in the controller and decoder.
    for parameter in base_parameters:
        parameter.requires_grad_(False)
    value_baseline = ExecutionValueBaseline().to(device)
    execution_optimizer = torch.optim.Adam(
        [*execution_parameters, *value_baseline.parameters()],
        lr=execution_learning_rate,
    )
    execution_updates = 0

    for step in range(1, steps + 1):
        tick_start = time.perf_counter()
        mode, reward, execution_log_prob, action_log_prob, entropy, execution_context, execution_cost = _sample_rollout(
            runtime, verifier, device=device
        )
        total_latency += time.perf_counter() - tick_start
        utility = reward - execution_cost
        value = value_baseline(execution_context.detach())
        advantage = utility - value
        policy_loss = -(advantage.detach() * execution_log_prob) - entropy_weight * entropy
        value_loss = 0.5 * (utility.detach() - value).square()
        loss = policy_loss + value_loss
        execution_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(execution_parameters, max_norm=5.0)
        execution_optimizer.step()
        execution_updates += 1

        if step == 1 or step % eval_every == 0 or step == steps:
            metrics = evaluate_metrics(
                runtime, verifier, episodes=32, condition="adaptive", device=device
            )
            diagnostic_lifetimes += 32
            history.append({"step": step, **metrics})

    elapsed = time.perf_counter() - start
    stable_bits: int | None = None
    for index, point in enumerate(history):
        if point["reward"] >= threshold and all(
            later["reward"] >= threshold for later in history[index:]
        ):
            stable_bits = int(
                (warmup_steps + point["step"])
                * VariableDeliberationVerifier.bit_count
            )
            break
    lifetimes = warmup_steps + steps
    accounting = RunAccounting(
        unique_verifier_bits=lifetimes * VariableDeliberationVerifier.bit_count,
        unique_logical_lifetimes=lifetimes,
        optimizer_updates=warmup_steps + execution_updates,
        replayed_examples=0,
        diagnostic_lifetimes_charged_to_budget=diagnostic_lifetimes,
        wall_time_seconds=elapsed,
        mean_inference_latency_ms=(total_latency / lifetimes) * 1000.0,
        stable_bits_to_threshold=stable_bits,
        retention_on_mastered_primitives=None,
        transfer_ratio_against_fresh_learner=None,
    )
    for parameter in base_parameters:
        parameter.requires_grad_(True)
    return history, accounting


@torch.no_grad()
def _evaluate_episode(
    runtime: AmodalControllerRuntime,
    verifier: VariableDeliberationVerifier,
    *,
    condition: str,
    device: torch.device,
) -> tuple[float, int, str]:
    streams = verifier.reset(1)
    state = runtime.initial_state(1, device=device)
    feedback = _feedback(runtime.controller.feedback_width, device)
    initial, state = _step_streams(runtime, streams, state, feedback, device=device)

    if condition == "adaptive":
        # This invokes the production bounded deliberation API.  Delayed
        # evidence is released only after the selected wait/think decision.
        result = runtime.deliberate(
            runtime.encode_streams(streams),
            runtime.initial_state(1, device=device),
            feedback,
            think_budget=1,
        )
        final = result.output
        state = result.state
        think_ticks = result.think_ticks
        initial_decision = result.initial_decision
        needs_next_event = result.decision == "wait" or think_ticks > 0
    elif condition in {"commit_immediate", "wait_fixed", "think_fixed"}:
        mode = {
            "commit_immediate": "commit",
            "wait_fixed": "wait",
            "think_fixed": "think",
        }[condition]
        result = runtime.deliberate(
            runtime.encode_streams(streams),
            runtime.initial_state(1, device=device),
            feedback,
            think_budget=1,
            execution_mode=mode,
        )
        final = result.output
        state = result.state
        think_ticks = result.think_ticks
        initial_decision = result.initial_decision
        needs_next_event = mode in {"wait", "think"}
    elif condition == "missing_delayed":
        final = initial
        think_ticks = 0
        initial_decision = "commit"
        needs_next_event = False
    elif condition == "random_action":
        action = torch.randint(0, verifier.action_count, (1,), device=device)
        return float(verifier.step(action).item()), 0, "commit"
    else:
        raise ValueError(f"unknown evaluation condition {condition!r}")

    if needs_next_event:
        final, _ = _step_streams(
            runtime,
            verifier.release_delayed(
                after_think=initial_decision == "think" or think_ticks > 0
            ),
            state,
            _quiet_feedback(runtime.controller.feedback_width, device),
            device=device,
        )
    action = final.decoded["protocol"].argmax(dim=-1)
    return float(verifier.step(action).item()), think_ticks, initial_decision


@torch.no_grad()
def evaluate_metrics(
    runtime: AmodalControllerRuntime,
    verifier: VariableDeliberationVerifier,
    *,
    episodes: int,
    condition: str,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    device = torch.device(device)
    runtime.eval()
    rewards: list[float] = []
    utilities: list[float] = []
    ticks: list[int] = []
    decisions: list[str] = []
    for _ in range(episodes):
        reward, think_ticks, decision = _evaluate_episode(
            runtime, verifier, condition=condition, device=device
        )
        rewards.append(reward)
        ticks.append(think_ticks)
        decisions.append(decision)
        utilities.append(reward - EXECUTION_COST[decision])
    mean_reward = float(np.mean(rewards))
    mean_utility = float(np.mean(utilities))
    mean_think_ticks = float(np.mean(ticks))
    return {
        "reward": mean_reward,
        "utility": mean_utility,
        "mean_think_ticks": mean_think_ticks,
        "wait_fraction": float(decisions.count("wait") / episodes),
        "think_fraction": float(decisions.count("think") / episodes),
        "commit_fraction": float(decisions.count("commit") / episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--balanced-curriculum", action="store_true")
    parser.add_argument("--easy-probability", type=float, default=0.5)
    parser.add_argument("--think-probability", type=float, default=0.0)
    args = parser.parse_args()
    verifier = (
        BalancedDeliberationCurriculum(seed=args.seed, device=args.device)
        if args.balanced_curriculum
        else VariableDeliberationVerifier(
            seed=args.seed,
            device=args.device,
            easy_probability=args.easy_probability,
            think_probability=args.think_probability,
        )
    )
    runtime = build_runtime(seed=args.seed)
    history, accounting = train_steps(
        runtime,
        verifier,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        device=args.device,
    )
    audits = {}
    for name, easy_probability, think_probability in (
        ("mixed", 0.5, 0.25),
        ("complete", 1.0, 0.0),
        ("delayed", 0.0, 0.0),
        ("think_required", 0.0, 1.0),
    ):
        audit_verifier = VariableDeliberationVerifier(
            seed=args.seed + 1000,
            device=args.device,
            easy_probability=easy_probability,
            think_probability=think_probability,
        )
        audits[name] = {
            condition: evaluate_metrics(
                runtime,
                audit_verifier,
                episodes=128,
                condition=condition,
                device=args.device,
            )
            for condition in (
                "adaptive",
                "commit_immediate",
                "wait_fixed",
                "think_fixed",
                "missing_delayed",
                "random_action",
            )
        }
    print(json.dumps({"history": history, "accounting": asdict(accounting), "audits": audits}, indent=2))


if __name__ == "__main__":
    main()
