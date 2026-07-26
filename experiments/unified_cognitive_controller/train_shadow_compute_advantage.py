"""Estimate extra-compute advantage from attempted actions only."""
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
from .train_shadow_compute_critic import (
    ARMS,
    _logged_batch,
    controlled_features,
)


class ComputeAdvantageHead(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def attempted_advantage_target(
        actions: torch.Tensor, observed_utility: torch.Tensor, *,
        baseline: float, propensity: float) -> torch.Tensor:
    if not 0 < propensity < 1:
        raise ValueError("propensity must be strictly between zero and one")
    sign = actions.to(observed_utility.dtype) * 2.0 - 1.0
    return sign * (observed_utility - baseline) / propensity


@torch.no_grad()
def advantage_policy_metrics(
        head: ComputeAdvantageHead, features: torch.Tensor,
        no_read: torch.Tensor, read: torch.Tensor, *, read_cost: float,
        shuffle_features: bool = False, seed: int = 0,
        ) -> dict[str, float]:
    active = features
    if shuffle_features:
        generator = torch.Generator(
            device=features.device).manual_seed(seed)
        active = features[torch.randperm(
            features.shape[0], generator=generator,
            device=features.device)]
    predicted_advantage = head(active)
    chosen = (predicted_advantage > 0).to(torch.long)
    actual = torch.stack((no_read, read - read_cost), dim=1)
    true_advantage = actual[:, 1] - actual[:, 0]
    oracle = actual.argmax(-1)
    chosen_utility = actual.gather(1, chosen[:, None]).squeeze(1)
    strongest_fixed = max(
        float(actual[:, 0].mean()), float(actual[:, 1].mean()))
    oracle_utility = float(actual.max(-1).values.mean())
    achieved = float(chosen_utility.mean())
    gap = oracle_utility - strongest_fixed
    return {
        "compute_choice_accuracy": float((chosen == oracle).float().mean()),
        "shadow_verified_utility": achieved,
        "always_no_read_utility": float(actual[:, 0].mean()),
        "always_read_utility": float(actual[:, 1].mean()),
        "strongest_fixed_utility": strongest_fixed,
        "oracle_utility": oracle_utility,
        "available_oracle_gap": gap,
        "captured_oracle_gap_fraction": (
            (achieved - strongest_fixed) / gap if gap > 1e-8 else 0.0),
        "read_rate": float(chosen.float().mean()),
        "advantage_mse_private_audit": float(
            (predicted_advantage - true_advantage).square().mean()),
        "advantage_sign_accuracy_private_audit": float(
            ((predicted_advantage > 0) == (true_advantage > 0))
            .float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7421)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=126)
    parser.add_argument("--bank-capacity", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--read-cost", type=float, default=0.01)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    if (
            args.batch_size % args.bank_capacity
            or args.test_contexts % args.bank_capacity):
        raise ValueError("counts must divide by bank capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    model = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    test_features, _, _, test_no, test_read = _logged_batch(
        model, count=args.test_contexts, capacity=args.bank_capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        prototype = ComputeAdvantageHead(hidden=args.head_hidden).to(device)
    heads = {name: copy.deepcopy(prototype) for name in ARMS}
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, head in heads.items()}
    histories = {name: [] for name in ARMS}
    gradient_norms = {name: [] for name in ARMS}
    utility_sum = 0.0
    utility_count = 0
    started = time.perf_counter()

    def record(step: int) -> None:
        reverse = torch.arange(
            test_features.shape[0] - 1, -1, -1, device=device)
        for name, head in heads.items():
            metrics = advantage_policy_metrics(
                head, controlled_features(
                    test_features, name, permutation=reverse),
                test_no, test_read, read_cost=args.read_cost)
            histories[name].append({
                "step": step,
                "unique_lifetimes": step * args.batch_size,
                "unique_verifier_bits": step * args.batch_size,
                **metrics,
            })

    record(0)
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    feature_generator = torch.Generator(device=device).manual_seed(
        args.seed + 72_000_000)
    for step in range(1, args.steps + 1):
        features, actions, outcomes, _, _ = _logged_batch(
            model, count=args.batch_size, capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold)
        observed_utility = outcomes - args.read_cost * actions
        utility_sum += float(observed_utility.sum())
        utility_count += observed_utility.numel()
        baseline = utility_sum / utility_count
        targets = attempted_advantage_target(
            actions, observed_utility, baseline=baseline,
            propensity=0.5)
        reward_permutation = torch.randperm(
            args.batch_size, generator=reward_generator, device=device)
        feature_permutation = torch.randperm(
            args.batch_size, generator=feature_generator, device=device)
        for name, head in heads.items():
            active = controlled_features(
                features, name, permutation=feature_permutation)
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
    intact = final["intact"]
    shuffled = advantage_policy_metrics(
        heads["intact"], test_features, test_no, test_read,
        read_cost=args.read_cost, shuffle_features=True,
        seed=args.seed + 79_000_000)
    control_margin = min(
        intact["shadow_verified_utility"]
        - final[name]["shadow_verified_utility"]
        for name in ("reward_shuffled", "feature_shuffled",
                     "missing_evidence"))
    fixed_gain = (
        intact["shadow_verified_utility"]
        - intact["strongest_fixed_utility"])
    shuffle_cost = (
        intact["shadow_verified_utility"]
        - shuffled["shadow_verified_utility"])
    stable = all(
        row["shadow_verified_utility"]
        >= row["strongest_fixed_utility"] + 0.05
        for row in histories["intact"][-2:])

    inference_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            heads["intact"](test_features)
    latency = (
        (time.perf_counter() - inference_started)
        / (100 * args.test_contexts))
    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, head in heads.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(head.state_dict(), path)
            restored = ComputeAdvantageHead(hidden=args.head_hidden).to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            with torch.no_grad():
                persistence[name] = torch.equal(
                    head(test_features), restored(test_features))
    binary = evaluate(
        model, count=128, trials=6, seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=128, trials=6, seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    gate = {
        "compute_choice_accuracy_at_least_0_65":
            intact["compute_choice_accuracy"] >= 0.65,
        "utility_beats_strongest_fixed_by_0_05": fixed_gain >= 0.05,
        "captures_at_least_20_percent_oracle_gap":
            intact["captured_oracle_gap_fraction"] >= 0.20,
        "beats_every_control_by_0_02_utility": control_margin >= 0.02,
        "evidence_shuffle_costs_at_least_0_02_utility":
            shuffle_cost >= 0.02,
        "improvement_stable_at_last_two_prefixes": stable,
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
        "all_round_trips_exact": all(persistence.values()),
        "head_latency_positive": latency > 0,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "shadow-compute-advantage-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "learner_visible": [
            "four_generic_read_statistics",
            "opaque_attempted_compute_action",
            "exact_logging_propensity_0_5",
            "attempted_action_scalar_verified_outcome",
            "generic_normalized_read_cost",
        ],
        "hidden_from_learner": [
            "unattempted_outcome", "correct_compute_action",
            "true_compute_advantage", "stored_context_metadata",
            "semantic_task_identity",
        ],
        "critic_can_influence_compute_or_answers": False,
        "training_objective":
            "inverse_propensity_attempted_action_advantage",
        "histories": histories,
        "final_metrics": final,
        "evidence_shuffled_audit": shuffled,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "gate": gate,
        "accounting": {
            "unique_training_lifetimes": args.steps * args.batch_size,
            "unique_training_verifier_bits": args.steps * args.batch_size,
            "optimizer_updates_per_head": args.steps,
            "replayed_examples": 0,
            "heldout_private_both_action_bits": 2 * args.test_contexts,
            "wall_seconds": time.perf_counter() - started,
            "mean_head_inference_latency_seconds": latency,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "final_metrics": final,
        "evidence_shuffled_audit": shuffled,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
