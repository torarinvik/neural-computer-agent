"""Conservative incumbent/challenger adaptation from attempted outcomes only."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .probe_requery_operation import requery_batch
from .train import evaluate, seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import (
    ComputeAdvantageHead,
    attempted_advantage_target,
)
from .train_thought_compute_transfer import _metrics


def paired_ips_improvement(
        incumbent_actions: torch.Tensor,
        challenger_actions: torch.Tensor,
        attempted_actions: torch.Tensor,
        attempted_utilities: torch.Tensor,
        *, propensity: float = 0.5, z: float = 1.96,
        ) -> dict[str, float]:
    """Return paired IPS challenger-minus-incumbent evidence."""
    if not 0 < propensity < 1:
        raise ValueError("propensity must be between zero and one")
    if not (
            incumbent_actions.shape == challenger_actions.shape
            == attempted_actions.shape == attempted_utilities.shape):
        raise ValueError("all logged tensors must have the same shape")
    weights = torch.where(
        attempted_actions.bool(),
        torch.full_like(attempted_utilities, 1.0 / propensity),
        torch.full_like(attempted_utilities, 1.0 / (1.0 - propensity)))
    # A common reward baseline is an unbiased control variate for a policy
    # difference because the two action-match indicators have equal expected
    # mass under the logger. It removes irrelevant reward-level variance.
    centered_utilities = attempted_utilities - attempted_utilities.mean()
    paired = weights * centered_utilities * (
        (attempted_actions == challenger_actions).to(attempted_utilities.dtype)
        - (attempted_actions == incumbent_actions).to(
            attempted_utilities.dtype))
    mean = float(paired.mean())
    standard_error = (
        float(paired.std(unbiased=True)) / math.sqrt(paired.numel())
        if paired.numel() > 1 else float("inf"))
    return {
        "estimated_improvement": mean,
        "standard_error": standard_error,
        "lower_95": mean - z * standard_error,
        "upper_95": mean + z * standard_error,
        "records": paired.numel(),
        "reward_baseline": float(attempted_utilities.mean()),
    }


def _load_head(path: Path, device: torch.device) -> ComputeAdvantageHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = ComputeAdvantageHead(int(payload["head_hidden"])).to(device)
    head.load_state_dict(payload["head_state_dict"])
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7951)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--promotion-every", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    mastered = _load_head(args.head_checkpoint, device)
    hidden = mastered.network[1].out_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = ComputeAdvantageHead(hidden).to(device)
    arms = {}
    for name, initial in (("mastered", mastered), ("gap", reset)):
        arms[name] = {
            "incumbent": copy.deepcopy(initial),
            "challenger": copy.deepcopy(initial),
            "naive": copy.deepcopy(initial),
            "optimizer": None,
            "naive_optimizer": None,
            "logged": [],
            "promotions": [],
            "history": [],
        }
        arms[name]["optimizer"] = torch.optim.AdamW(
            arms[name]["challenger"].parameters(),
            lr=args.learning_rate, weight_decay=1e-4)
        arms[name]["naive_optimizer"] = torch.optim.AdamW(
            arms[name]["naive"].parameters(),
            lr=args.learning_rate, weight_decay=1e-4)

    test_features, test_first, test_second, _ = requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)

    def record(step: int) -> None:
        for arm in arms.values():
            arm["history"].append({
                "step": step,
                "unique_verifier_bits": step * args.batch_size,
                "incumbent": _metrics(
                    arm["incumbent"], test_features, test_first, test_second,
                    thought_cost=args.requery_cost),
                "challenger": _metrics(
                    arm["challenger"], test_features, test_first, test_second,
                    thought_cost=args.requery_cost),
                "naive": _metrics(
                    arm["naive"], test_features, test_first, test_second,
                    thought_cost=args.requery_cost),
            })

    record(0)
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    baselines = {name: [0.0, 0] for name in arms}
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        features, first, second, _ = requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold)
        attempted = torch.randint(
            0, 2, (args.batch_size,), generator=action_generator,
            device=device)
        utilities = torch.where(
            attempted.bool(), second - args.requery_cost, first)
        for name, arm in arms.items():
            baselines[name][0] += float(utilities.sum())
            baselines[name][1] += utilities.numel()
            targets = attempted_advantage_target(
                attempted, utilities,
                baseline=baselines[name][0] / baselines[name][1],
                propensity=0.5)
            for key, optimizer_key in (
                    ("challenger", "optimizer"),
                    ("naive", "naive_optimizer")):
                loss = nn.functional.smooth_l1_loss(
                    arm[key](features), targets)
                arm[optimizer_key].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(arm[key].parameters(), 1.0)
                arm[optimizer_key].step()
            arm["logged"].append((
                features.detach(), attempted.detach(), utilities.detach()))

        if step % args.promotion_every == 0:
            for arm in arms.values():
                features_log = torch.cat([row[0] for row in arm["logged"]])
                attempted_log = torch.cat([row[1] for row in arm["logged"]])
                utilities_log = torch.cat([row[2] for row in arm["logged"]])
                with torch.no_grad():
                    incumbent_actions = (
                        arm["incumbent"](features_log) > 0).long()
                    challenger_actions = (
                        arm["challenger"](features_log) > 0).long()
                evidence = paired_ips_improvement(
                    incumbent_actions, challenger_actions,
                    attempted_log, utilities_log)
                promoted = evidence["lower_95"] > 0
                if promoted:
                    arm["incumbent"].load_state_dict(
                        arm["challenger"].state_dict())
                    # Earlier records compared against the previous incumbent;
                    # restart the paired audit only after an actual promotion.
                    arm["logged"].clear()
                arm["promotions"].append({
                    "step": step, **evidence, "promoted": promoted})
        if step % 2 == 0 or step == args.steps:
            record(step)

    final = {
        name: arm["history"][-1] for name, arm in arms.items()}
    mastered_start = arms["mastered"]["history"][0]["incumbent"]
    mastered_final = final["mastered"]["incumbent"]
    gap_start = arms["gap"]["history"][0]["incumbent"]
    gap_final = final["gap"]["incumbent"]
    gate = {
        "mastered_incumbent_accuracy_retained":
            mastered_final["compute_choice_accuracy"] >= 0.65,
        "mastered_incumbent_utility_not_degraded": (
            mastered_final["verified_utility"]
            >= mastered_start["verified_utility"] - 0.005),
        "mastered_naive_is_destructive_control": (
            final["mastered"]["naive"]["verified_utility"]
            < mastered_final["verified_utility"] - 0.005),
        "gap_arm_promoted_with_positive_lower_bound": any(
            row["promoted"] and row["lower_95"] > 0
            for row in arms["gap"]["promotions"]),
        "gap_incumbent_utility_improved": (
            gap_final["verified_utility"]
            > gap_start["verified_utility"] + 0.02),
    }
    binary = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    gate["binary_retained"] = binary["gate"]["accepted"]
    gate["four_rule_retained"] = four_rule["gate"]["accepted"]
    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, arm in arms.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(arm["incumbent"].state_dict(), path)
            restored = ComputeAdvantageHead(hidden).to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            persistence[name] = torch.equal(
                arm["incumbent"](test_features),
                restored(test_features))
    gate["all_round_trips_exact"] = all(persistence.values())
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "safe-requery-adaptation-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "head_checkpoint": str(args.head_checkpoint),
            "report": str(args.report),
        },
        "learner_visible": [
            "four_generic_memory_statistics", "attempted_action",
            "attempted_action_scalar_outcome", "logging_propensity_0_5",
        ],
        "hidden_from_learner_and_promotion": [
            "unattempted_outcome", "correct_compute_action",
            "correct_answer", "semantic_task_identity",
            "private_evaluation_metrics",
        ],
        "arms": {
            name: {
                "history": arm["history"],
                "promotion_evidence": arm["promotions"],
            }
            for name, arm in arms.items()
        },
        "final": final,
        "gate": gate,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "persistence_exact": persistence,
        "accounting": {
            "learner_visible_unique_verifier_bits":
                args.steps * args.batch_size,
            "optimizer_updates_per_challenger": args.steps,
            "private_test_both_action_bits": args.test_contexts * 2,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "promotions": {
            name: arm["promotions"] for name, arm in arms.items()},
        "final": final,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
