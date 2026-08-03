"""Learn a global wait threshold online from scalar verifier utility."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .audit_amodal_adaptive_wait import (
    _load_predictor,
    _load_runtime,
    _prepare,
    _rollout,
    _sample_delays,
)
from .environment import generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _select_arm(counts: list[int], means: list[float], round_index: int, exploration: float) -> int:
    for index, count in enumerate(counts):
        if count == 0:
            return index
    return max(
        range(len(counts)),
        key=lambda index: means[index]
        + exploration * math.sqrt(math.log(round_index + 1) / counts[index]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=995001)
    parser.add_argument("--rounds", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latency-cost", type=float, default=0.03)
    parser.add_argument("--deadline", type=int, default=2)
    parser.add_argument("--exploration", type=float, default=0.05)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
    )
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
    if args.rounds < len(args.thresholds) or args.batch_size < 2:
        raise ValueError("rounds must cover every threshold and batch-size must be positive")
    device = torch.device(args.device)
    runtime, _ = _load_runtime(args.controller, args.input_bus, args.audio, device)
    predictor = _load_predictor(args.predictor, device)
    counts = [0] * len(args.thresholds)
    means = [0.0] * len(args.thresholds)
    curve = []
    start = time.perf_counter()
    for round_index in range(1, args.rounds + 1):
        arm = _select_arm(counts, means, round_index, args.exploration)
        appearance_index = (round_index - 1) % 3
        appearance = ("bars", "diamonds", "dot_pairs")[appearance_index]
        batch = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + round_index,
            heldout=False,
            task="pair_relation",
            appearance=appearance,
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        delays, _ = _sample_delays(
            args.batch_size,
            seed=args.seed + round_index + 50_000,
            device=device,
        )
        encoded_a, encoded_b = _prepare(runtime, first, second)
        result = _rollout(
            runtime,
            predictor,
            encoded_a,
            encoded_b,
            batch.correct_actions,
            delays,
            mode="adaptive",
            threshold=args.thresholds[arm],
            deadline=args.deadline,
            latency_cost=args.latency_cost,
        )
        reward = float(result["verified_utility"])
        counts[arm] += 1
        means[arm] += (reward - means[arm]) / counts[arm]
        curve.append(
            {
                "round": round_index,
                "appearance": appearance,
                "arm": arm,
                "threshold": args.thresholds[arm],
                "reward": reward,
                "running_mean": means[arm],
            }
        )
    selected_arm = max(range(len(means)), key=means.__getitem__)
    report = {
        "schema": "amodal-wait-utility-bandit-v1",
        "claim": "UCB learns a timing threshold from scalar verifier utility only.",
        "labels_used": [],
        "configuration": {
            "seed": args.seed,
            "rounds": args.rounds,
            "batch_size": args.batch_size,
            "thresholds": list(args.thresholds),
            "latency_cost": args.latency_cost,
            "deadline": args.deadline,
            "exploration": args.exploration,
            "device": str(device),
        },
        "arm_counts": counts,
        "arm_mean_utility": means,
        "selected_arm": selected_arm,
        "selected_threshold": args.thresholds[selected_arm],
        "curve": curve,
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_threshold": report["selected_threshold"], "means": means}))


if __name__ == "__main__":
    main()
