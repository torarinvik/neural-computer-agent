"""Transfer attempted-action compute value to a second-ranked memory re-query."""
from __future__ import annotations

import argparse
import copy
import json
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


def _stable_bits(history: list[dict[str, float]]) -> int | None:
    def passes(row: dict[str, float]) -> bool:
        return (
            row["compute_choice_accuracy"] >= 0.65
            and row["verified_utility"]
            >= row["strongest_fixed_utility"] + 0.03
            and row["captured_oracle_gap_fraction"] >= 0.20)
    for index, row in enumerate(history):
        if passes(row) and all(passes(later) for later in history[index:]):
            return int(row["unique_verifier_bits"])
    return None


def _features(
        values: torch.Tensor, name: str,
        permutation: torch.Tensor | None = None) -> torch.Tensor:
    if name in (
            "inherited", "inherited_trunk", "previous_inherited",
            "previous_trunk", "reset", "reward_shuffled"):
        return values
    if name == "missing_evidence":
        return torch.zeros_like(values)
    if name == "feature_shuffled":
        if permutation is None:
            raise ValueError("feature shuffle requires permutation")
        return values[permutation]
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--advantage-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7891)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--evaluate-every", type=int, default=2)
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
    payload = torch.load(
        args.advantage_checkpoint, map_location=device,
        weights_only=False)
    previous_payload = torch.load(
        args.previous_checkpoint, map_location=device,
        weights_only=False)
    hidden = int(payload["head_hidden"])
    inherited = ComputeAdvantageHead(hidden).to(device)
    inherited.load_state_dict(payload["head_state_dict"])
    previous = ComputeAdvantageHead(hidden).to(device)
    previous.load_state_dict(previous_payload["head_state_dict"])
    inherited_trunk = copy.deepcopy(inherited)
    nn.init.zeros_(inherited_trunk.network[-1].weight)
    nn.init.zeros_(inherited_trunk.network[-1].bias)
    previous_trunk = copy.deepcopy(previous)
    nn.init.zeros_(previous_trunk.network[-1].weight)
    nn.init.zeros_(previous_trunk.network[-1].bias)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = ComputeAdvantageHead(hidden).to(device)
    heads = {
        "inherited": inherited,
        "inherited_trunk": inherited_trunk,
        "previous_inherited": previous,
        "previous_trunk": previous_trunk,
        "reset": reset,
        "reward_shuffled": copy.deepcopy(inherited_trunk),
        "feature_shuffled": copy.deepcopy(inherited_trunk),
        "missing_evidence": copy.deepcopy(inherited_trunk),
    }
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, head in heads.items()}
    test_features, test_first, test_second, _ = requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)
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
                **_metrics(
                    head, _features(test_features, name, reverse),
                    test_first, test_second,
                    thought_cost=args.requery_cost),
            })

    record(0)
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    feature_generator = torch.Generator(device=device).manual_seed(
        args.seed + 72_000_000)
    for step in range(1, args.steps + 1):
        features, first, second, _ = requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold)
        actions = torch.randint(
            0, 2, (args.batch_size,), generator=action_generator,
            device=device)
        outcomes = torch.where(actions.bool(), second, first)
        utility = outcomes - args.requery_cost * actions
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
            active_targets = (
                targets[reward_permutation]
                if name == "reward_shuffled" else targets)
            loss = nn.functional.smooth_l1_loss(
                head(_features(features, name, feature_permutation)),
                active_targets)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)))
            optimizers[name].step()
        if step % args.evaluate_every == 0 or step == args.steps:
            record(step)

    final = {name: rows[-1] for name, rows in histories.items()}
    stable = {name: _stable_bits(rows) for name, rows in histories.items()}
    inherited_final = final["inherited_trunk"]
    inherited_bits = stable["inherited_trunk"]
    reset_bits = stable["reset"]
    previous_bits = stable["previous_trunk"]
    evidence_cost = min(
        inherited_final["verified_utility"]
        - final[name]["verified_utility"]
        for name in ("feature_shuffled", "missing_evidence"))
    reward_cost = (
        inherited_final["verified_utility"]
        - final["reward_shuffled"]["verified_utility"])
    gate = {
        "inherited_trunk_choice_at_least_0_65":
            inherited_final["compute_choice_accuracy"] >= 0.65,
        "inherited_trunk_beats_fixed_by_0_03":
            inherited_final["verified_utility"]
            >= inherited_final["strongest_fixed_utility"] + 0.03,
        "inherited_trunk_captures_20_percent_gap":
            inherited_final["captured_oracle_gap_fraction"] >= 0.20,
        "inherited_trunk_strictly_faster_than_reset": (
            inherited_bits is not None
            and (reset_bits is None or inherited_bits < reset_bits)),
        "inherited_trunk_strictly_faster_than_previous_trunk": (
            inherited_bits is not None
            and (previous_bits is None or inherited_bits < previous_bits)),
        "evidence_controls_cost_at_least_0_02": evidence_cost >= 0.02,
        "reward_shuffle_costs_0_02_when_learning_required": (
            inherited_bits == 0 or reward_cost >= 0.02),
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
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
        "schema": "requery-transfer-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "advantage_checkpoint": str(args.advantage_checkpoint),
            "previous_checkpoint": str(args.previous_checkpoint),
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
        },
        "learner_visible": [
            "four_generic_memory_statistics", "attempted_action",
            "logging_propensity_0_5",
            "attempted_action_scalar_outcome", "requery_cost",
        ],
        "hidden_from_learner": [
            "unattempted_outcome", "correct_compute_action",
            "correct_answer", "semantic_task_identity",
        ],
        "histories": histories,
        "stable_unique_verifier_bits": stable,
        "final_metrics": final,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "control_diagnostics": {
            "evidence_control_minimum_utility_cost": evidence_cost,
            "reward_shuffle_utility_cost": reward_cost,
            "reward_shuffle_required": inherited_bits != 0,
        },
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
                for name, value in inherited_trunk.state_dict().items()},
            "source_seed": args.seed,
            "source_operation": "second_ranked_requery",
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
