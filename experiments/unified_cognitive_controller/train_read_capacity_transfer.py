"""Measure gradual transfer of a compute allocator to a changed memory bank."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .train import evaluate, seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import (
    ComputeAdvantageHead,
    advantage_policy_metrics,
    attempted_advantage_target,
)
from .train_shadow_compute_critic import _logged_batch, controlled_features


ARMS = (
    "inherited", "reset", "reward_shuffled",
    "feature_shuffled", "missing_evidence")


def _stable_bits(history: list[dict[str, float]]) -> int | None:
    def passes(row: dict[str, float]) -> bool:
        return (
            row["compute_choice_accuracy"] >= 0.65
            and row["shadow_verified_utility"]
            >= row["strongest_fixed_utility"] + 0.05
            and row["captured_oracle_gap_fraction"] >= 0.20)
    for index, row in enumerate(history):
        if passes(row) and all(passes(later) for later in history[index:]):
            return int(row["unique_verifier_bits"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--advantage-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-checkpoint", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7821)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=256)
    parser.add_argument("--bank-capacity", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--read-cost", type=float, default=0.01)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    if (
            args.batch_size % args.bank_capacity
            or args.test_contexts % args.bank_capacity
            or args.batch_size % 2 or args.test_contexts % 2):
        raise ValueError("counts must be even and divide by bank capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    payload = torch.load(
        args.advantage_checkpoint, map_location=device,
        weights_only=False)
    hidden = int(payload["head_hidden"])
    inherited = ComputeAdvantageHead(hidden).to(device)
    inherited.load_state_dict(payload["head_state_dict"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = ComputeAdvantageHead(hidden).to(device)
    heads = {
        "inherited": inherited,
        "reset": reset,
        "reward_shuffled": copy.deepcopy(inherited),
        "feature_shuffled": copy.deepcopy(inherited),
        "missing_evidence": copy.deepcopy(inherited),
    }
    if args.previous_checkpoint is not None:
        previous_payload = torch.load(
            args.previous_checkpoint, map_location=device,
            weights_only=False)
        previous = ComputeAdvantageHead(hidden).to(device)
        previous.load_state_dict(previous_payload["head_state_dict"])
        heads["previous_inherited"] = previous
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, head in heads.items()}
    test_features, _, _, test_no, test_read = _logged_batch(
        controller, count=args.test_contexts,
        capacity=args.bank_capacity, seed=args.seed + 90_000_000,
        device=device, write_threshold=0.5)
    histories = {name: [] for name in heads}
    gradient_norms = {name: [] for name in heads}
    utility_sum = 0.0
    utility_count = 0
    started = time.perf_counter()

    def record(step: int) -> None:
        reverse = torch.arange(
            args.test_contexts - 1, -1, -1, device=device)
        for name, head in heads.items():
            histories[name].append({
                "step": step,
                "unique_verifier_bits": step * args.batch_size,
                **advantage_policy_metrics(
                    head, controlled_features(
                        test_features, (
                            "intact" if name in (
                                "inherited", "reset",
                                "previous_inherited")
                            else name),
                        permutation=reverse),
                    test_no, test_read, read_cost=args.read_cost),
            })

    record(0)
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    feature_generator = torch.Generator(device=device).manual_seed(
        args.seed + 72_000_000)
    for step in range(1, args.steps + 1):
        features, actions, outcomes, _, _ = _logged_batch(
            controller, count=args.batch_size,
            capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, write_threshold=0.5)
        utility = outcomes - args.read_cost * actions
        utility_sum += float(utility.sum())
        utility_count += utility.numel()
        targets = attempted_advantage_target(
            actions, utility, baseline=utility_sum / utility_count,
            propensity=0.5)
        reward_permutation = torch.randperm(
            args.batch_size, generator=reward_generator, device=device)
        feature_permutation = torch.randperm(
            args.batch_size, generator=feature_generator, device=device)
        for name, head in heads.items():
            control_name = (
                "intact" if name in (
                    "inherited", "reset", "previous_inherited")
                else name)
            active = controlled_features(
                features, control_name,
                permutation=feature_permutation)
            active_targets = (
                targets[reward_permutation]
                if name == "reward_shuffled" else targets)
            loss = nn.functional.smooth_l1_loss(
                head(active), active_targets)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)))
            optimizers[name].step()
        if step % args.evaluate_every == 0 or step == args.steps:
            record(step)

    final = {name: rows[-1] for name, rows in histories.items()}
    stable = {name: _stable_bits(rows) for name, rows in histories.items()}
    inherited_final = final["inherited"]
    control_cost = min(
        inherited_final["shadow_verified_utility"]
        - final[name]["shadow_verified_utility"]
        for name in (
            "reward_shuffled", "feature_shuffled",
            "missing_evidence"))
    inherited_bits = stable["inherited"]
    reset_bits = stable["reset"]
    previous_bits = stable.get("previous_inherited")
    gate = {
        "inherited_choice_at_least_0_65":
            inherited_final["compute_choice_accuracy"] >= 0.65,
        "inherited_beats_fixed_by_0_05":
            inherited_final["shadow_verified_utility"]
            >= inherited_final["strongest_fixed_utility"] + 0.05,
        "inherited_captures_20_percent_gap":
            inherited_final["captured_oracle_gap_fraction"] >= 0.20,
        "inherited_strictly_faster_than_reset": (
            inherited_bits is not None
            and (reset_bits is None or inherited_bits < reset_bits)),
        "controls_cost_at_least_0_02": control_cost >= 0.02,
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
    }
    if args.previous_checkpoint is not None:
        gate["inherited_strictly_faster_than_previous"] = (
            inherited_bits is not None
            and (
                previous_bits is None
                or inherited_bits < previous_bits))
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
        for name, head in heads.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(head.state_dict(), path)
            restored = ComputeAdvantageHead(hidden).to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            persistence[name] = torch.equal(
                head(test_features), restored(test_features))
    gate["all_round_trips_exact"] = all(persistence.values())
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "read-capacity-transfer-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "advantage_checkpoint": str(args.advantage_checkpoint),
            "previous_checkpoint": (
                str(args.previous_checkpoint)
                if args.previous_checkpoint is not None else None),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "learner_visible": [
            "four_generic_read_statistics", "attempted_action",
            "logging_propensity_0_5",
            "attempted_action_scalar_outcome", "read_cost",
        ],
        "hidden_from_learner": [
            "bank_capacity", "unattempted_outcome",
            "correct_compute_action", "correct_answer",
            "semantic_task_identity",
        ],
        "histories": histories,
        "stable_unique_verifier_bits": stable,
        "final_metrics": final,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "gate": gate,
        "accounting": {
            "learner_visible_unique_lifetimes":
                args.steps * args.batch_size,
            "learner_visible_unique_verifier_bits":
                args.steps * args.batch_size,
            "optimizer_updates_per_arm": args.steps,
            "replayed_examples": 0,
            "private_test_both_action_bits": args.test_contexts * 2,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "shadow-compute-advantage-head-v1",
            "head_hidden": hidden,
            "head_state_dict": {
                name: value.detach().cpu()
                for name, value in inherited.state_dict().items()},
            "source_seed": args.seed,
            "source_bank_capacity": args.bank_capacity,
            "source_training_lifetimes":
                args.steps * args.batch_size,
            "source_training_verifier_bits":
                args.steps * args.batch_size,
            "source_optimizer_updates": args.steps,
            "source_report": str(args.report),
            "parent_advantage_checkpoint":
                str(args.advantage_checkpoint),
        }, args.checkpoint_out)
    print(json.dumps({
        "report": str(args.report),
        "stable_unique_verifier_bits": stable,
        "final_metrics": final,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
