"""Find the least optional thought that preserves magnitude mastery.

Experience efficiency is settled first by the acquisition run. This audit then
holds the learned checkpoint and external evidence fixed while varying only
extra recurrent controller steps. Accuracy and causal counterfactual gates are
hard constraints; elapsed compute breaks ties only among passing budgets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audit_pair_relation_compute import _rollout_with_extra_thought
from .audit_pair_relation_repertoire import _load
from .environment import generate_lifetimes


@torch.no_grad()
def evaluate_budget(
        model, *, count: int, trials: int, seed: int,
        extra_thought_steps: int, device: torch.device,
        ) -> dict[str, float | int | bool]:
    normal = generate_lifetimes(
        count, trials, seed=seed, heldout=True,
        task="visible_pair_magnitude", appearance="bars",
        support_trials=1, device=device)
    reversed_batch = generate_lifetimes(
        count, trials, seed=seed, heldout=True,
        task="visible_pair_magnitude", appearance="bars",
        support_trials=1, reverse_contexts=True, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    normal_result = _rollout_with_extra_thought(
        model, normal, extra_thought_steps=extra_thought_steps)
    reversed_result = _rollout_with_extra_thought(
        model, reversed_batch, extra_thought_steps=extra_thought_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    normal_accuracy = float(normal_result["rewards"].float().mean())
    reversed_accuracy = float(
        reversed_result["rewards"].float().mean())
    flip_rate = float(
        (normal_result["actions"] != reversed_result["actions"])
        .float().mean())
    return {
        "normal_accuracy": normal_accuracy,
        "counterfactual_accuracy": reversed_accuracy,
        "prediction_flip_rate": flip_rate,
        "extra_thought_steps_per_event": extra_thought_steps,
        "controller_steps_per_event": 1 + extra_thought_steps,
        "controller_steps_per_lifetime":
            trials * (1 + extra_thought_steps),
        "logical_lifetimes": count,
        "verifier_bits": count * trials * 2,
        "seconds": elapsed,
        "mastery": (
            normal_accuracy >= 0.90
            and reversed_accuracy >= 0.90
            and flip_rate >= 0.80),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23461)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--extra-thought-budgets", default="0,1,2,4,8")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    budgets = tuple(
        int(value) for value in args.extra_thought_budgets.split(","))
    if (
            not budgets or any(value < 0 for value in budgets)
            or len(set(budgets)) != len(budgets)):
        raise ValueError("thought budgets must be unique nonnegative integers")
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and divisible by two")

    device = torch.device(args.device)
    model = _load(args.checkpoint, device)
    results = {
        str(budget): evaluate_budget(
            model, count=args.count, trials=args.trials,
            seed=args.seed + 100_000 * budget,
            extra_thought_steps=budget, device=device)
        for budget in budgets
    }
    passing = [budget for budget in budgets if results[str(budget)]["mastery"]]
    minimum = min(passing) if passing else None
    report = {
        "schema": "pair-magnitude-compute-audit-v1",
        "claim_boundary": (
            "The checkpoint and external event stream are fixed. Extra "
            "thought reprocesses the same pixels without a new action, "
            "outcome, task identity or verifier fact."),
        "checkpoint": str(args.checkpoint),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "trials": args.trials,
            "extra_thought_budgets": budgets,
            "device": str(device),
        },
        "results": results,
        "minimum_extra_thought_budget_preserving_mastery": minimum,
        "minimum_controller_steps_per_event": (
            1 + minimum if minimum is not None else None),
        "already_compiled_to_physical_minimum": minimum == 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "minimum_extra_thought_budget_preserving_mastery": minimum,
        "already_compiled_to_physical_minimum": minimum == 0,
        "accuracies": {
            budget: round(float(value["normal_accuracy"]), 6)
            for budget, value in results.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
