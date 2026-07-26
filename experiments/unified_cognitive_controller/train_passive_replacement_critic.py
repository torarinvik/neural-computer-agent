"""Passive attempted-action critic for memory-replacement outcomes.

The critic cannot alter the controller or its sampled actions.  It observes
only generic controller-created memory statistics, the attempted option,
its exact logging propensity, and the later scalar verifier outcome.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .train import evaluate, seed_everything
from .train_memory_replacement import _bank_reward
from .train_redundancy_transfer import (
    build_transfer_arms,
    redundancy_utility_batch,
)


ARM_NAMES = ("intact", "reward_shuffled", "missing_action", "missing_context")
CONTEXT_FEATURES = (0, 1, 2, 5, 6, 7)
CRITIC_INPUT_WIDTH = 18 + 8 + 3


class PassiveReplacementCritic(nn.Module):
    """A small scalar predictor with no connection to the action path."""

    def __init__(self, width: int = CRITIC_INPUT_WIDTH, hidden: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Begin as an exact empirical-rate predictor.  The critic must earn
        # every feature-dependent deviation from that passive baseline.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def per_bank_context(option_features: torch.Tensor) -> torch.Tensor:
    """Generic per-bank context; never a semantic task identifier."""
    if option_features.ndim != 3 or option_features.shape[-1] != 8:
        raise ValueError("expected [banks, options, 8] features")
    rows = option_features[:, 1:, CONTEXT_FEATURES]
    return torch.cat((
        rows.mean(1),
        rows.std(1, unbiased=False),
        (rows.abs().mean(1) > 1e-6).to(rows.dtype),
    ), dim=-1)


def attempted_action_features(
        option_features: torch.Tensor,
        scores: torch.Tensor,
        actions: torch.Tensor,
        propensities: torch.Tensor,
        ) -> torch.Tensor:
    """Describe only the action that was actually attempted."""
    selected = torch.gather(
        option_features, 1,
        actions[:, None, None].expand(-1, 1, option_features.shape[-1]),
    ).squeeze(1)
    selected_score = scores.gather(1, actions[:, None]).squeeze(1)
    alternatives = scores.masked_fill(
        torch.nn.functional.one_hot(
            actions, scores.shape[1]).to(torch.bool), -torch.inf)
    margin = selected_score - alternatives.max(-1).values
    return torch.cat((
        per_bank_context(option_features),
        selected,
        propensities[:, None],
        propensities.clamp_min(1e-8).log()[:, None],
        margin[:, None],
    ), dim=-1)


def apply_evidence_control(features: torch.Tensor, arm: str) -> torch.Tensor:
    controlled = features.clone()
    if arm == "missing_context":
        controlled[:, :18] = 0
    elif arm == "missing_action":
        controlled[:, 18:] = 0
    elif arm not in ("intact", "reward_shuffled"):
        raise ValueError(f"unknown critic arm: {arm}")
    return controlled


def exploration_probabilities(
        scores: torch.Tensor, *, epsilon: float,
        temperature: float) -> torch.Tensor:
    if not 0.0 <= epsilon <= 1.0 or temperature <= 0:
        raise ValueError("invalid exploration configuration")
    probabilities = torch.softmax(scores / temperature, dim=-1)
    return (1.0 - epsilon) * probabilities + epsilon / scores.shape[-1]


def expected_calibration_error(
        probabilities: torch.Tensor, outcomes: torch.Tensor,
        bins: int = 10) -> float:
    error = probabilities.new_zeros(())
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (
            (probabilities >= low)
            & (probabilities <= high if index == bins - 1
               else probabilities < high))
        if mask.any():
            error += mask.float().mean() * (
                probabilities[mask].mean() - outcomes[mask].mean()).abs()
    return float(error)


def concordance(
        predictions: torch.Tensor, outcomes: torch.Tensor) -> float:
    """Pairwise ranking accuracy, ignoring equal-outcome pairs."""
    target_delta = outcomes[:, None] - outcomes[None, :]
    prediction_delta = predictions[:, None] - predictions[None, :]
    mask = target_delta > 1e-8
    if not mask.any():
        return 0.5
    correct = prediction_delta[mask] > 0
    ties = prediction_delta[mask].abs() <= 1e-8
    return float(correct.float().mean() + 0.5 * ties.float().mean())


@torch.no_grad()
def critic_metrics(
        critic: PassiveReplacementCritic, features: torch.Tensor,
        outcomes: torch.Tensor, *, constant: float) -> dict[str, float]:
    critic.eval()
    clipped = min(max(constant, 1e-5), 1.0 - 1e-5)
    base_logit = torch.tensor(
        clipped / (1.0 - clipped), device=features.device,
        dtype=features.dtype).log()
    probabilities = torch.sigmoid(base_logit + critic(features))
    return {
        "brier": float((probabilities - outcomes).square().mean()),
        "constant_brier": float(
            (torch.full_like(outcomes, constant) - outcomes).square().mean()),
        "ece": expected_calibration_error(probabilities, outcomes),
        "concordance": concordance(probabilities, outcomes),
        "mean_prediction": float(probabilities.mean()),
        "mean_outcome": float(outcomes.mean()),
    }


@torch.no_grad()
def collect_attempts(
        policy, reward_model, *, banks: int, capacity: int, seed: int,
        device: torch.device, epsilon: float, temperature: float,
        write_threshold: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = redundancy_utility_batch(
        reward_model, banks=banks, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold, noise_scale=0.0,
        weights=(0.0, 0.0, 0.0, 1.0))
    scores = policy.memory_replacement_scores(data["option_features"])
    probabilities = exploration_probabilities(
        scores, epsilon=epsilon, temperature=temperature)
    generator = torch.Generator(device=device).manual_seed(seed + 81_337)
    actions = torch.multinomial(
        probabilities, 1, generator=generator).squeeze(1)
    propensities = probabilities.gather(1, actions[:, None]).squeeze(1)
    outcomes = _bank_reward(
        reward_model, data, actions, device=device)
    features = attempted_action_features(
        data["option_features"], scores, actions, propensities)
    return features, outcomes, actions, propensities


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-banks", type=int, default=64)
    parser.add_argument("--test-banks", type=int, default=128)
    parser.add_argument("--bank-capacity", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.batch_banks, args.test_banks) < 1:
        raise ValueError("training budgets must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    arms = build_transfer_arms(
        parent, selected, device=device, fresh_seed=args.seed + 1)
    policy = arms["selected_experience"]
    reward_model = policy

    test_features, test_outcomes, test_actions, test_propensities = (
        collect_attempts(
            policy, reward_model, banks=args.test_banks,
            capacity=args.bank_capacity, seed=args.seed + 90_000_000,
            device=device, epsilon=args.epsilon,
            temperature=args.temperature,
            write_threshold=args.write_threshold))

    critics = {}
    optimizers = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        prototype = PassiveReplacementCritic().to(device)
    for name in ARM_NAMES:
        critics[name] = copy.deepcopy(prototype)
        optimizers[name] = torch.optim.AdamW(
            critics[name].parameters(), lr=args.learning_rate,
            weight_decay=1e-4)

    histories: dict[str, list[dict[str, float]]] = {
        name: [] for name in ARM_NAMES}
    train_outcome_sum = 0.0
    train_examples = 0
    action_digests = []
    propensity_digests = []
    gradient_norms = {name: [] for name in ARM_NAMES}
    started = time.perf_counter()

    def record(step: int) -> None:
        constant = (
            train_outcome_sum / train_examples
            if train_examples else float(test_outcomes.mean()))
        for name, critic in critics.items():
            histories[name].append({
                "step": step,
                "unique_logical_lifetimes":
                    step * args.batch_banks,
                "unique_verifier_bits":
                    step * args.batch_banks * args.bank_capacity,
                **critic_metrics(
                    critic, apply_evidence_control(test_features, name),
                    test_outcomes, constant=constant),
            })

    record(0)
    shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 73_000_000)
    for step in range(1, args.steps + 1):
        features, outcomes, actions, propensities = collect_attempts(
            policy, reward_model, banks=args.batch_banks,
            capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, epsilon=args.epsilon,
            temperature=args.temperature,
            write_threshold=args.write_threshold)
        train_outcome_sum += float(outcomes.sum())
        train_examples += outcomes.numel()
        empirical_rate = min(max(
            train_outcome_sum / train_examples, 1e-5), 1.0 - 1e-5)
        base_logit = torch.tensor(
            empirical_rate / (1.0 - empirical_rate), device=device,
            dtype=features.dtype).log()
        action_digests.append(_tensor_digest(actions))
        propensity_digests.append(_tensor_digest(propensities))
        permutation = torch.randperm(
            outcomes.numel(), generator=shuffle_generator, device=device)
        for name, critic in critics.items():
            critic.train()
            controlled = apply_evidence_control(features, name)
            targets = (
                outcomes[permutation]
                if name == "reward_shuffled" else outcomes)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                base_logit + critic(controlled), targets)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)))
            optimizers[name].step()
        if step % args.evaluate_every == 0 or step == args.steps:
            record(step)

    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, critic in critics.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(critic.state_dict(), path)
            restored = PassiveReplacementCritic().to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            controlled = apply_evidence_control(test_features, name)
            with torch.no_grad():
                persistence[name] = torch.equal(
                    critic(controlled), restored(controlled))

    binary = evaluate(
        policy, count=128, trials=6, seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        policy, count=128, trials=6, seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    final = {name: history[-1] for name, history in histories.items()}
    intact_advantage = (
        final["intact"]["constant_brier"] - final["intact"]["brier"])
    control_advantage = min(
        final[name]["brier"] - final["intact"]["brier"]
        for name in ("reward_shuffled", "missing_action", "missing_context"))
    stable_last_two = (
        len(histories["intact"]) >= 2
        and all(
            row["brier"] < row["constant_brier"]
            for row in histories["intact"][-2:]))
    gate = {
        "logging_distribution_has_action_coverage": (
            int(torch.unique(test_actions).numel()) >= 3
            and float(test_propensities.min()) > 0),
        "outcomes_have_variation": float(test_outcomes.std()) >= 0.05,
        "intact_beats_constant_brier_by_0_005":
            intact_advantage >= 0.005,
        "intact_beats_every_control_by_0_002":
            control_advantage >= 0.002,
        "intact_concordance_at_least_0_55":
            final["intact"]["concordance"] >= 0.55,
        "intact_ece_at_most_0_10": final["intact"]["ece"] <= 0.10,
        "improvement_stable_at_last_two_prefixes": stable_last_two,
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
        "all_critic_round_trips_exact": all(persistence.values()),
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    gate["accepted_for_three_minute_promotion"] = all(gate.values())
    elapsed = time.perf_counter() - started
    report = {
        "schema": "passive-replacement-critic-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "learner_visible": [
            "generic_per_bank_memory_statistics",
            "attempted_option_statistics",
            "attempted_action_logging_propensity",
            "attempted_action_policy_margin",
            "later_scalar_verified_outcome",
        ],
        "hidden_from_learner": [
            "optimal_replacement_action", "unattempted_action_outcomes",
            "utility_or_task_identifier", "future_query_identity",
        ],
        "critic_can_influence_actions": False,
        "semantic_or_correct_action_labels_used_for_training": False,
        "histories": histories,
        "final_metrics": final,
        "controls": {
            "reward_shuffled": "training outcomes permuted within each batch",
            "missing_action": "attempted option, propensity, and margin removed",
            "missing_context": "generic bank context removed",
        },
        "action_trace_digest": action_digests,
        "propensity_trace_digest": propensity_digests,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "gate": gate,
        "accounting": {
            "unique_logical_lifetimes": (
                args.steps * args.batch_banks),
            "unique_verifier_bits": (
                args.steps * args.batch_banks * args.bank_capacity),
            "optimizer_updates_per_critic": args.steps,
            "replayed_examples": 0,
            "training_examples_per_critic": (
                args.steps * args.batch_banks),
            "heldout_logical_lifetimes": args.test_banks,
            "heldout_verifier_bits": (
                args.test_banks * args.bank_capacity),
            "wall_seconds": elapsed,
            "mean_decision_latency_seconds": None,
            "stable_bits_to_threshold": None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "wall_seconds": elapsed,
        "final_metrics": final,
        "gate": gate,
    }, indent=2))


if __name__ == "__main__":
    main()
