"""Train a metadata-only wait policy from scalar verifier utility."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .amodal_runtime import AmodalEventWindowBuffer
from .amodal_wait_policy import AmodalWaitDecisionPolicy, arrival_features
from .audit_amodal_adaptive_wait import (
    _event,
    _load_runtime,
    _prepare,
    _sample_delays,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _sampled_rollout(
    runtime,
    policy: AmodalWaitDecisionPolicy,
    encoded_a: torch.Tensor,
    encoded_b: torch.Tensor,
    labels: torch.Tensor,
    delays: torch.Tensor,
    *,
    deadline: int,
    latency_cost: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one reward episode batch and retain log-probability gradients."""
    count, trials, _ = encoded_a.shape
    device = labels.device
    policy_generator = torch.Generator(device=device).manual_seed(seed)
    rewards = []
    log_probabilities = []
    entropies = []
    for episode in range(count):
        buffer = AmodalEventWindowBuffer(("stream_a", "stream_b"))
        state = runtime.initial_state(1, device=device)
        previous_action = torch.full(
            (1,), NULL_ACTION, dtype=torch.long, device=device
        )
        previous_reward = torch.zeros(1, device=device)
        actions = torch.full((trials,), -1, dtype=torch.long, device=device)
        latencies = torch.zeros(trials, device=device)
        processed = 0
        history: list[bool] = []
        episode_log_probs = []
        episode_entropies = []
        for trial in range(trials):
            base = trial * 10
            delay = int(delays[episode, trial])
            released = False

            def consume(window, release_clock: int) -> None:
                nonlocal state, previous_action, previous_reward, processed, released
                window_trial = int(round(window.timestamp / 10.0))
                if window_trial != trial:
                    raise AssertionError("window released at the wrong trial")
                feedback = torch.full((1,), float(processed > 0), device=device)
                output, state = runtime.step_events(
                    window.collection,
                    state,
                    previous_action,
                    previous_reward * feedback,
                    feedback,
                )
                previous_action = output.decoded["action"].argmax(dim=-1)
                previous_reward = (
                    previous_action == labels[episode, trial].reshape(1)
                ).float()
                actions[trial] = previous_action[0]
                latencies[trial] = float(release_clock - base)
                processed += 1
                released = True

            arrivals = {
                "stream_a": _event(
                    runtime, "stream_a", encoded_a[episode, trial], base
                )
            }
            if delay == 0:
                arrivals["stream_b"] = _event(
                    runtime, "stream_b", encoded_b[episode, trial], base
                )
            for window in buffer.push(arrivals):
                consume(window, base)
            for offset in range(deadline + 1):
                if released:
                    break
                if delay > 0 and delay == offset:
                    for window in buffer.push(
                        {
                            "stream_b": _event(
                                runtime,
                                "stream_b",
                                encoded_b[episode, trial],
                                base,
                            )
                        }
                    ):
                        consume(window, base + offset)
                    if released:
                        break
                status = buffer.pending_status(current_timestamp=base + offset)
                if not status:
                    raise AssertionError("pending window disappeared")
                current = status[0]
                if offset >= deadline:
                    consume(buffer.release_pending(current.timestamp), base + offset)
                    break
                features = arrival_features(
                    current, history, deadline=float(deadline)
                ).to(device)
                probability = policy(features.unsqueeze(0))[0].clamp(1e-5, 1 - 1e-5)
                wait = torch.rand((), generator=policy_generator, device=device) < probability
                log_probability = torch.where(
                    wait, probability.log(), torch.log1p(-probability)
                )
                entropy = -(
                    probability * probability.log()
                    + (1 - probability) * torch.log1p(-probability)
                )
                episode_log_probs.append(log_probability)
                episode_entropies.append(entropy)
                if not bool(wait):
                    consume(buffer.release_pending(current.timestamp), base + offset)
            if not released:
                raise AssertionError("reward policy failed to release a window")
            history.append(delay >= 0)
            del history[:-4]
        if processed != trials or buffer.pending_timestamps:
            raise AssertionError("reward policy left pending or missing windows")
        accuracy = (actions[1:] == labels[episode, 1:]).float().mean()
        mean_latency = latencies[1:].mean()
        rewards.append(accuracy - latency_cost * mean_latency)
        log_probabilities.append(
            torch.stack(episode_log_probs).sum()
            if episode_log_probs
            else torch.zeros((), device=device)
        )
        entropies.append(
            torch.stack(episode_entropies).mean()
            if episode_entropies
            else torch.zeros((), device=device)
        )
    return (
        torch.stack(rewards),
        torch.stack(log_probabilities),
        torch.stack(entropies),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--init-predictor", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=994001)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--latency-cost", type=float, default=0.03)
    parser.add_argument("--deadline", type=int, default=2)
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 2 or args.deadline < 1:
        raise ValueError("invalid reward training configuration")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime, _ = _load_runtime(args.controller, args.input_bus, args.audio, device)
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    policy = AmodalWaitDecisionPolicy(args.hidden).to(device)
    if args.init_predictor is not None:
        initializer = torch.load(
            args.init_predictor, map_location=device, weights_only=False
        )
        if initializer.get("schema") != "amodal-arrival-predictor-v1":
            raise ValueError("init-predictor must be an arrival predictor artifact")
        if int(initializer["hidden"]) != args.hidden:
            raise ValueError("init-predictor hidden width disagrees")
        policy.load_state_dict(initializer["state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    baseline = 0.75
    curve = []
    start = time.perf_counter()
    policy.train()
    for update in range(1, args.updates + 1):
        appearance = ("bars", "diamonds", "dot_pairs")[(update - 1) % 3]
        batch = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + update,
            heldout=False,
            task="pair_relation",
            appearance=appearance,
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        delays, _ = _sample_delays(
            args.batch_size,
            seed=args.seed + update + 50_000,
            device=device,
        )
        encoded_a, encoded_b = _prepare(runtime, first, second)
        rewards, log_probs, entropies = _sampled_rollout(
            runtime,
            policy,
            encoded_a,
            encoded_b,
            batch.correct_actions,
            delays,
            deadline=args.deadline,
            latency_cost=args.latency_cost,
            seed=args.seed + update + 100_000,
        )
        mean_reward = float(rewards.mean())
        baseline = 0.9 * baseline + 0.1 * mean_reward
        advantage = rewards - baseline
        loss = -(advantage.detach() * log_probs).mean() - args.entropy_coefficient * entropies.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        curve.append(
            {
                "update": update,
                "appearance": appearance,
                "reward": mean_reward,
                "baseline": baseline,
                "loss": float(loss.detach()),
                "entropy": float(entropies.mean().detach()),
            }
        )
    policy.eval()
    payload = {
        "schema": "amodal-wait-decision-policy-v1",
        "feature_count": 5,
        "hidden": args.hidden,
        "state_dict": {
            name: value.detach().cpu() for name, value in policy.state_dict().items()
        },
        "training": {
            "method": "reinforce-verifier-utility",
            "labels_used": [],
            "initializer": str(args.init_predictor) if args.init_predictor else None,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "latency_cost": args.latency_cost,
            "deadline": args.deadline,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report = {
        "schema": "amodal-wait-reward-training-v1",
        "labels_used": [],
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "learning_rate": args.learning_rate,
            "entropy_coefficient": args.entropy_coefficient,
            "latency_cost": args.latency_cost,
            "deadline": args.deadline,
            "device": str(device),
            "init_predictor": str(args.init_predictor) if args.init_predictor else None,
        },
        "curve": curve,
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final_reward": curve[-1]["reward"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
