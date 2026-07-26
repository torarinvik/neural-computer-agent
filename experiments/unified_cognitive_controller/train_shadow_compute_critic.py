"""Learn the verified value of one extra memory read without controlling it."""
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
from .train_adaptive_memory_read import _batch, _outcomes
from .train_passive_replacement_critic import expected_calibration_error
from .train_redundancy_transfer import build_transfer_arms


ARMS = ("intact", "reward_shuffled", "feature_shuffled", "missing_evidence")


class ShadowComputeCritic(nn.Module):
    """Two opaque compute-action values from four generic read statistics."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def selected_compute_loss(
        logits: torch.Tensor, actions: torch.Tensor,
        outcomes: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(1, actions[:, None]).squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(selected, outcomes)


def controlled_features(
        features: torch.Tensor, arm: str, *,
        permutation: torch.Tensor | None = None) -> torch.Tensor:
    if arm in ("intact", "reward_shuffled"):
        return features
    if arm == "missing_evidence":
        return torch.zeros_like(features)
    if arm == "feature_shuffled":
        if permutation is None:
            raise ValueError("feature-shuffled arm requires a permutation")
        return features[permutation]
    raise ValueError(f"unknown critic arm: {arm}")


def _base_logits(
        reward_sums: torch.Tensor, action_counts: torch.Tensor) -> torch.Tensor:
    rates = torch.where(
        action_counts > 0,
        reward_sums / action_counts.clamp_min(1),
        torch.full_like(reward_sums, 0.5)).clamp(1e-5, 1 - 1e-5)
    return (rates / (1 - rates)).log()


@torch.no_grad()
def _attempt_metrics(
        critic: ShadowComputeCritic, features: torch.Tensor,
        actions: torch.Tensor, outcomes: torch.Tensor,
        base_logits: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(base_logits + critic(features))
    selected = probabilities.gather(1, actions[:, None]).squeeze(1)
    baseline = torch.sigmoid(base_logits)[actions]
    return {
        "brier": float((selected - outcomes).square().mean()),
        "action_rate_brier": float((baseline - outcomes).square().mean()),
        "ece": expected_calibration_error(selected, outcomes),
        "mean_prediction": float(selected.mean()),
        "mean_outcome": float(outcomes.mean()),
    }


@torch.no_grad()
def _shadow_metrics(
        critic: ShadowComputeCritic, features: torch.Tensor,
        no_read_outcomes: torch.Tensor, read_outcomes: torch.Tensor,
        base_logits: torch.Tensor, *, read_cost: float,
        shuffle_features: bool = False, seed: int = 0,
        ) -> dict[str, float]:
    active = features
    if shuffle_features:
        generator = torch.Generator(
            device=features.device).manual_seed(seed)
        active = features[torch.randperm(
            features.shape[0], generator=generator,
            device=features.device)]
    probabilities = torch.sigmoid(base_logits + critic(active))
    predicted_utility = probabilities.clone()
    predicted_utility[:, 1] -= read_cost
    chosen = predicted_utility.argmax(-1)
    actual = torch.stack((
        no_read_outcomes,
        read_outcomes - read_cost), dim=1)
    oracle = actual.argmax(-1)
    chosen_utility = actual.gather(1, chosen[:, None]).squeeze(1)
    no_utility = actual[:, 0]
    read_utility = actual[:, 1]
    strongest_fixed = max(float(no_utility.mean()), float(read_utility.mean()))
    oracle_utility = float(actual.max(-1).values.mean())
    available = oracle_utility - strongest_fixed
    achieved = float(chosen_utility.mean())
    return {
        "compute_choice_accuracy": float((chosen == oracle).float().mean()),
        "shadow_verified_utility": achieved,
        "always_no_read_utility": float(no_utility.mean()),
        "always_read_utility": float(read_utility.mean()),
        "strongest_fixed_utility": strongest_fixed,
        "oracle_utility": oracle_utility,
        "available_oracle_gap": available,
        "captured_oracle_gap_fraction": (
            (achieved - strongest_fixed) / available
            if available > 1e-8 else 0.0),
        "read_rate": float((chosen == 1).float().mean()),
    }


@torch.no_grad()
def _logged_batch(
        model, *, count: int, capacity: int, seed: int,
        device: torch.device, write_threshold: float,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor]:
    batch, read, features, _ = _batch(
        model, count=count, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold)
    generator = torch.Generator(device=device).manual_seed(seed + 55_019)
    actions = torch.randint(
        0, 2, (count,), generator=generator, device=device)
    memory = torch.where(
        actions[:, None].to(torch.bool), read, torch.zeros_like(read))
    outcomes = _outcomes(model, batch, memory, device=device)
    no_read = _outcomes(
        model, batch, torch.zeros_like(read), device=device)
    read_outcomes = _outcomes(model, batch, read, device=device)
    return features, actions, outcomes, no_read, read_outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7411)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=126)
    parser.add_argument("--bank-capacity", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--read-cost", type=float, default=0.01)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    if (
            args.batch_size % args.bank_capacity
            or args.test_contexts % args.bank_capacity):
        raise ValueError("batch and test counts must divide by bank capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    model = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]

    test_features, test_actions, test_outcomes, test_no, test_read = (
        _logged_batch(
            model, count=args.test_contexts,
            capacity=args.bank_capacity, seed=args.seed + 90_000_000,
            device=device, write_threshold=args.write_threshold))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        prototype = ShadowComputeCritic().to(device)
    critics = {name: copy.deepcopy(prototype) for name in ARMS}
    optimizers = {
        name: torch.optim.AdamW(
            critic.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, critic in critics.items()}
    reward_sums = torch.zeros(2, device=device)
    action_counts = torch.zeros(2, device=device)
    histories = {name: [] for name in ARMS}
    gradient_norms = {name: [] for name in ARMS}
    started = time.perf_counter()

    def record(step: int) -> None:
        bases = _base_logits(reward_sums, action_counts)
        for name, critic in critics.items():
            histories[name].append({
                "step": step,
                "unique_lifetimes": step * args.batch_size,
                "unique_verifier_bits": step * args.batch_size,
                **_attempt_metrics(
                    critic,
                    controlled_features(
                        test_features, name,
                        permutation=torch.arange(
                            test_features.shape[0] - 1, -1, -1,
                            device=device)),
                    test_actions, test_outcomes, bases),
            })

    record(0)
    reward_shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    feature_shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 72_000_000)
    for step in range(1, args.steps + 1):
        features, actions, outcomes, _, _ = _logged_batch(
            model, count=args.batch_size, capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold)
        reward_sums.scatter_add_(0, actions, outcomes)
        action_counts.scatter_add_(
            0, actions, torch.ones_like(outcomes))
        bases = _base_logits(reward_sums, action_counts)
        reward_permutation = torch.randperm(
            args.batch_size, generator=reward_shuffle_generator,
            device=device)
        feature_permutation = torch.randperm(
            args.batch_size, generator=feature_shuffle_generator,
            device=device)
        for name, critic in critics.items():
            critic.train()
            active = controlled_features(
                features, name, permutation=feature_permutation)
            targets = (
                outcomes[reward_permutation]
                if name == "reward_shuffled" else outcomes)
            loss = selected_compute_loss(
                bases + critic(active), actions, targets)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)))
            optimizers[name].step()
        if step % args.evaluate_every == 0 or step == args.steps:
            record(step)

    bases = _base_logits(reward_sums, action_counts)
    final = {name: rows[-1] for name, rows in histories.items()}
    shadow = _shadow_metrics(
        critics["intact"], test_features, test_no, test_read, bases,
        read_cost=args.read_cost)
    evidence_shuffled = _shadow_metrics(
        critics["intact"], test_features, test_no, test_read, bases,
        read_cost=args.read_cost, shuffle_features=True,
        seed=args.seed + 79_000_000)

    inference_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            critics["intact"](test_features)
    critic_latency = (
        (time.perf_counter() - inference_started)
        / (100 * args.test_contexts))

    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, critic in critics.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(critic.state_dict(), path)
            restored = ShadowComputeCritic().to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            with torch.no_grad():
                persistence[name] = torch.equal(
                    critic(test_features), restored(test_features))

    binary = evaluate(
        model, count=128, trials=6, seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=128, trials=6, seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    intact_gain = (
        final["intact"]["action_rate_brier"]
        - final["intact"]["brier"])
    control_margin = min(
        final[name]["brier"] - final["intact"]["brier"]
        for name in ("reward_shuffled", "feature_shuffled",
                     "missing_evidence"))
    fixed_gain = (
        shadow["shadow_verified_utility"]
        - shadow["strongest_fixed_utility"])
    shuffle_cost = (
        shadow["shadow_verified_utility"]
        - evidence_shuffled["shadow_verified_utility"])
    stable = all(
        row["brier"] < row["action_rate_brier"]
        for row in histories["intact"][-2:])
    gate = {
        "intact_beats_action_rate_brier_by_0_005":
            intact_gain >= 0.005,
        "intact_beats_every_control_by_0_002":
            control_margin >= 0.002,
        "compute_choice_accuracy_at_least_0_65":
            shadow["compute_choice_accuracy"] >= 0.65,
        "utility_beats_strongest_fixed_by_0_05": fixed_gain >= 0.05,
        "captures_at_least_20_percent_oracle_gap":
            shadow["captured_oracle_gap_fraction"] >= 0.20,
        "evidence_shuffle_costs_at_least_0_02_utility":
            shuffle_cost >= 0.02,
        "intact_ece_at_most_0_10": final["intact"]["ece"] <= 0.10,
        "improvement_stable_at_last_two_prefixes": stable,
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
        "all_round_trips_exact": all(persistence.values()),
        "critic_latency_positive": critic_latency > 0,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "shadow-compute-allocation-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "learner_visible": [
            "generic_read_confidence", "generic_top_two_margin",
            "generic_selected_strength", "generic_bank_occupancy",
            "opaque_attempted_compute_action",
            "exact_logging_propensity_0_5",
            "later_scalar_verified_outcome",
        ],
        "hidden_from_learner": [
            "correct_compute_action", "unattempted_action_outcome",
            "stored_or_absent_context_metadata", "semantic_task_identity",
        ],
        "critic_can_influence_compute_or_answers": False,
        "semantic_or_correct_action_labels_used_for_training": False,
        "histories": histories,
        "final_metrics": final,
        "shadow_policy_audit": shadow,
        "evidence_shuffled_audit": evidence_shuffled,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "gate": gate,
        "accounting": {
            "unique_training_lifetimes": args.steps * args.batch_size,
            "unique_training_verifier_bits": args.steps * args.batch_size,
            "optimizer_updates_per_critic": args.steps,
            "replayed_examples": 0,
            "heldout_logged_lifetimes_and_bits": args.test_contexts,
            "private_both_action_audit_bits": 2 * args.test_contexts,
            "wall_seconds": time.perf_counter() - started,
            "mean_critic_inference_latency_seconds": critic_latency,
            "normalized_read_cost": args.read_cost,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "final_metrics": final,
        "shadow_policy_audit": shadow,
        "evidence_shuffled_audit": evidence_shuffled,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
