"""Compare one physical utility horse race with its tensor equivalent."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import torch

from .audit_frequency_recency_replacement import (
    _physical_policy,
    _retarget_future,
)
from .audit_multifeature_utility import _materialize_histories
from .legacy_model import UnifiedCognitiveController
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import _bank_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--banks", type=int, default=32)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if model.adaptive_memory_replace_features < 7:
        raise ValueError("checkpoint has no reliability feature")
    initial_weight = (
        model.memory_replacement_extra_gate.weight.detach().clone())
    weights = (0.3, 0.3, 0.4)
    started = time.perf_counter()
    data = frequency_recency_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=args.seed, device=device, write_threshold=0.5,
        noise_scale=0.04, recency_weight=weights[0],
        frequency_weight=weights[1], reliability_weight=weights[2])
    generated_seconds = time.perf_counter() - started

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        history_directory = root / "histories"
        history_directory.mkdir()
        history_started = time.perf_counter()
        (
            memories, access, successes, failures,
            persisted_exact, requested_exact,
        ) = _materialize_histories(
            model, data, history_directory, device=device)
        history_seconds = time.perf_counter() - history_started
        normalized_access = (
            torch.log1p(access.to(data["bank_ages"].dtype))
            / math.log(10.0))
        reliability = (
            (successes.to(data["bank_ages"].dtype) + 1.0)
            / (successes + failures + 2).to(data["bank_ages"].dtype))
        visible_utility = (
            weights[0] * data["bank_ages"] / args.bank_capacity
            + weights[1] * normalized_access
            + weights[2] * reliability)
        realized_utility = visible_utility + data["utility_noise"]
        target_slot = realized_utility.argmin(-1)
        target_action = target_slot + 1
        future_batch, future_queries = _retarget_future(
            data, target_slot)
        features = data["option_features"].clone()
        features[:, 1:, 5] = normalized_access - 0.5
        features[:, 1:, 6] = reliability - 0.5
        realized_data = dict(data)
        realized_data.update({
            "option_features": features,
            "target_action": target_action,
            "future_batch": future_batch,
            "future_queries": future_queries,
        })

        direction = torch.ones_like(initial_weight)
        candidates = {}
        for name, sign in (("plus", 1.0), ("center", 0.0),
                           ("minus", -1.0)):
            with torch.no_grad():
                model.memory_replacement_extra_gate.weight.copy_(
                    initial_weight
                    + sign * args.perturbation * direction)
                actions = model.memory_replacement_scores(
                    features).argmax(-1)
                tensor_started = time.perf_counter()
                tensor_reward = float(_bank_reward(
                    model, realized_data, actions, device=device).mean())
                tensor_seconds = time.perf_counter() - tensor_started
                physical_directory = root / name
                physical_directory.mkdir()
                physical_started = time.perf_counter()
                physical = _physical_policy(
                    model, memories, realized_data, actions,
                    future_batch, future_queries, physical_directory,
                    device=device)
                physical_seconds = time.perf_counter() - physical_started
                candidates[name] = {
                    "weights": (
                        model.memory_replacement_extra_gate.weight
                        .flatten().tolist()),
                    "target_eviction_rate": float(
                        (actions == target_action).float().mean()),
                    "tensor_reward": tensor_reward,
                    "physical_reward": physical["accuracy"],
                    "absolute_reward_difference": abs(
                        tensor_reward - physical["accuracy"]),
                    "tensor_seconds": tensor_seconds,
                    "physical_seconds": physical_seconds,
                    "before_rows": physical["before_rows"],
                    "after_rows": physical["after_rows"],
                    "capacity_growth": physical["capacity_growth"],
                }
        with torch.no_grad():
            model.memory_replacement_extra_gate.weight.copy_(initial_weight)

    tensor_winner = max(
        candidates, key=lambda name: candidates[name]["tensor_reward"])
    physical_winner = max(
        candidates, key=lambda name: candidates[name]["physical_reward"])
    maximum_difference = max(
        candidate["absolute_reward_difference"]
        for candidate in candidates.values())
    gate = {
        "same_winner": tensor_winner == physical_winner,
        "maximum_reward_difference_at_most_1_point":
            maximum_difference <= 0.01,
        "all_histories_persisted": persisted_exact == args.banks,
        "all_rows_remained_bounded": all(
            candidate["before_rows"]
            == candidate["after_rows"]
            == args.banks * args.bank_capacity
            and candidate["capacity_growth"] == 0
            for candidate in candidates.values()),
        "weights_restored_exactly": torch.equal(
            model.memory_replacement_extra_gate.weight,
            initial_weight),
    }
    gate["accepted"] = all(gate.values())
    report = {
        "schema": "unified-controller-physical-online-parity-v1",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "banks": args.banks,
        "bank_capacity": args.bank_capacity,
        "weights": {
            "recency": weights[0],
            "frequency": weights[1],
            "reliability": weights[2],
        },
        "generated_contexts": data["generated_contexts"],
        "requested_histories_reproduced_exactly": requested_exact,
        "histories_survived_save_reload_exactly": persisted_exact,
        "tensor_winner": tensor_winner,
        "physical_winner": physical_winner,
        "candidates": candidates,
        "timing": {
            "generation_seconds": generated_seconds,
            "history_materialization_seconds": history_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "gate": gate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
