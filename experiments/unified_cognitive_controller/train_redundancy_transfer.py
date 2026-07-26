"""Measure forward transfer to a held-out redundancy memory utility."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import (
    _bank_reward,
    _gather_rows,
    _select_batch,
)
from .strategy_memory import physical_context_key


UTILITY_NAMES = ("recency", "frequency", "reliability", "novelty")
ARM_NAMES = (
    "selected_experience",
    "selected_strategy_memory",
    "shared_parent",
    "architecture_reset",
    "replacement_policy_reset",
    "fresh_matched",
    "selected_reward_shuffled",
)


def _expanded_eight_feature_controller(
        configuration: dict[str, object],
        state: dict[str, torch.Tensor],
        *, device: torch.device,
        reset_residual: bool = False,
        fresh_policy: bool = False,
        fresh_seed: int = 0,
        ) -> UnifiedCognitiveController:
    """Build one matched eight-feature policy without changing old scores."""
    expanded = dict(configuration)
    expanded["adaptive_memory_replace"] = True
    expanded["adaptive_memory_replace_features"] = 8
    if fresh_policy:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(fresh_seed)
            return UnifiedCognitiveController(**expanded).to(device)

    model = UnifiedCognitiveController(**expanded).to(device)
    source = {
        name: value.to(device)
        for name, value in state.items()
        if name != "memory_replacement_extra_gate.weight"
    }
    missing, unexpected = model.load_state_dict(source, strict=False)
    if missing != ["memory_replacement_extra_gate.weight"] or unexpected:
        raise ValueError(
            f"unexpected expanded-controller mismatch: {missing=}, "
            f"{unexpected=}")
    old = state.get("memory_replacement_extra_gate.weight")
    with torch.no_grad():
        model.memory_replacement_extra_gate.weight.zero_()
        if old is not None and not reset_residual:
            columns = min(
                old.shape[1],
                model.memory_replacement_extra_gate.weight.shape[1])
            model.memory_replacement_extra_gate.weight[
                :, :columns].copy_(old[:, :columns].to(device))
    return model


def build_transfer_arms(
        parent_payload: dict[str, object],
        selected_payload: dict[str, object],
        *, device: torch.device,
        fresh_seed: int,
        ) -> dict[str, UnifiedCognitiveController]:
    """Create the pre-registered arms on one matched interface."""
    if parent_payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError("unsupported parent checkpoint schema")
    if selected_payload.get("schema") != (
            "unified-controller-physical-prefix-state-v1"):
        raise ValueError("unsupported selected prefix-state schema")
    configuration = dict(parent_payload["model_configuration"])
    selected_state = selected_payload["model_state_dict"]
    parent_state = parent_payload["state_dict"]
    replacement_policy_reset = _expanded_eight_feature_controller(
        configuration, selected_state, device=device,
        reset_residual=True)
    with torch.no_grad():
        for parameter in (
                replacement_policy_reset.memory_replacement_gate
                .parameters()):
            parameter.zero_()
    arms = {
        "selected_experience": _expanded_eight_feature_controller(
            configuration, selected_state, device=device),
        "selected_strategy_memory": _expanded_eight_feature_controller(
            configuration, selected_state, device=device),
        "shared_parent": _expanded_eight_feature_controller(
            configuration, parent_state, device=device),
        "architecture_reset": _expanded_eight_feature_controller(
            configuration, selected_state, device=device,
            reset_residual=True),
        "replacement_policy_reset": replacement_policy_reset,
        "fresh_matched": _expanded_eight_feature_controller(
            configuration, {}, device=device, fresh_policy=True,
            fresh_seed=fresh_seed),
        "selected_reward_shuffled": _expanded_eight_feature_controller(
            configuration, selected_state, device=device),
    }
    for model in arms.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return arms


@torch.no_grad()
def initialize_from_saved_strategy(
        model: UnifiedCognitiveController,
        selected_payload: dict[str, object],
        option_features: torch.Tensor,
        ) -> dict[str, object]:
    """Retrieve one old latent strategy for the new visible context."""
    strategy = selected_payload.get("strategy_memory")
    encoder_state = selected_payload.get("context_encoder_state_dict")
    run_state = selected_payload.get("run_state", {})
    if strategy is None or encoder_state is None:
        raise ValueError("selected checkpoint has no strategy-memory state")
    count = int(strategy["count"])
    if count < 1:
        raise ValueError("selected strategy memory is empty")
    reward_signature = run_state["previous_reward_signature"].to(
        option_features.device)
    query = physical_context_key(option_features, reward_signature)
    scales = encoder_state["log_scale"].to(
        option_features.device).exp()
    keys = strategy["keys"][:count].to(option_features.device)
    encoded_keys = torch.nn.functional.normalize(keys * scales, dim=-1)
    encoded_query = torch.nn.functional.normalize(query * scales, dim=0)
    similarities = encoded_keys @ encoded_query
    slot = int(similarities.argmax())
    value = strategy["values"][slot].to(option_features.device)
    model.memory_replacement_extra_gate.weight.zero_()
    model.memory_replacement_extra_gate.weight[0, :2].copy_(value)
    return {
        "slot": slot,
        "similarity": float(similarities[slot]),
        "retrieved_residual": value.tolist(),
    }


@torch.no_grad()
def redundancy_utility_batch(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        noise_scale: float,
        weights: tuple[float, float, float, float],
        ) -> dict[str, object]:
    """Add a row-novelty utility to the already mastered utility arena."""
    if model.adaptive_memory_replace_features != 8:
        raise ValueError("redundancy utility requires exactly eight features")
    if len(weights) != 4 or any(weight < 0.0 for weight in weights):
        raise ValueError("four nonnegative utility weights are required")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("at least one utility weight must be positive")
    normalized_weights = tuple(weight / total for weight in weights)
    data = frequency_recency_batch(
        model, banks=banks, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold,
        noise_scale=noise_scale,
        recency_weight=max(weights[0], 1e-12),
        frequency_weight=max(weights[1], 1e-12),
        reliability_weight=max(weights[2], 0.0))

    keys = torch.nn.functional.normalize(data["bank_keys"], dim=-1)
    similarities = torch.einsum("biw,bjw->bij", keys, keys)
    diagonal = torch.eye(
        capacity, device=device, dtype=torch.bool).unsqueeze(0)
    nearest_similarity = similarities.masked_fill(
        diagonal, -torch.inf).max(-1).values
    row_novelty = ((1.0 - nearest_similarity) / 2.0).clamp(0.0, 1.0)
    centered_novelty = row_novelty - 0.5

    option_features = data["option_features"].clone()
    option_features[:, 1:, 7] = centered_novelty
    normalized_age = data["bank_ages"] / capacity
    normalized_access = (
        torch.log1p(data["bank_access_counts"].to(keys.dtype))
        / math.log(10.0))
    visible_utility = (
        normalized_weights[0] * normalized_age
        + normalized_weights[1] * normalized_access
        + normalized_weights[2] * data["bank_reliability"]
        + normalized_weights[3] * row_novelty)
    realized_utility = visible_utility + data["utility_noise"]
    target_slot = realized_utility.argmin(-1)
    target_action = target_slot + 1
    visible_oracle_action = visible_utility.argmin(-1) + 1

    permutation = data["slot_to_logical"]
    target_logical = torch.gather(
        permutation, 1, target_slot.unsqueeze(1)).squeeze(1)
    logical = torch.arange(capacity, device=device).expand(banks, -1)
    retained_logical = logical[
        logical != target_logical.unsqueeze(1)].reshape(banks, capacity - 1)
    future_logical = torch.cat((
        retained_logical,
        torch.full(
            (banks, 1), capacity, device=device, dtype=torch.long),
    ), dim=1)
    base = (
        torch.arange(banks, device=device).unsqueeze(1)
        * (capacity + 1))
    future_indices = (base + future_logical).reshape(-1)

    result = dict(data)
    result.update({
        "option_features": option_features,
        "future_batch": _select_batch(
            data["source_batch"], future_indices),
        "future_queries": _gather_rows(
            data["query_group"], future_logical),
        "target_action": target_action,
        "visible_oracle_action": visible_oracle_action,
        "row_novelty": row_novelty,
        "nearest_row_similarity": nearest_similarity,
        "utility_weights": dict(zip(
            UTILITY_NAMES, normalized_weights, strict=True)),
    })
    return result


@torch.no_grad()
def _policy_metrics(
        policy: UnifiedCognitiveController,
        reward_model: UnifiedCognitiveController,
        datasets: list[dict[str, object]],
        *, device: torch.device,
        shuffle_novelty: bool = False,
        ) -> dict[str, float]:
    rewards = []
    targets = []
    for data in datasets:
        features = data["option_features"].clone()
        if shuffle_novelty:
            features[:, 1:, 7] = features[:, 1:, 7].roll(1, dims=1)
        actions = policy.memory_replacement_scores(features).argmax(-1)
        rewards.append(float(_bank_reward(
            reward_model, data, actions, device=device).mean()))
        targets.append(float(
            (actions == data["target_action"]).float().mean()))
    return {
        "verified_reward": sum(rewards) / len(rewards),
        "target_rate_diagnostic": sum(targets) / len(targets),
    }


@torch.no_grad()
def _control_metrics(
        reward_model: UnifiedCognitiveController,
        datasets: list[dict[str, object]], *,
        device: torch.device,
        seed: int,
        ) -> dict[str, float]:
    reward_lists: dict[str, list[float]] = {
        name: [] for name in (
            "random", "fixed", "skip", "recency", "frequency",
            "reliability", "redundancy", "visible_oracle", "oracle")}
    generator = torch.Generator(device=device).manual_seed(seed)
    for data in datasets:
        banks = data["bank_keys"].shape[0]
        capacity = data["bank_keys"].shape[1]
        policies = {
            "random": torch.randint(
                0, capacity + 1, (banks,),
                generator=generator, device=device),
            "fixed": torch.ones(banks, dtype=torch.long, device=device),
            "skip": torch.zeros(banks, dtype=torch.long, device=device),
            "recency": data["bank_ages"].argmin(-1) + 1,
            "frequency": data["bank_access_counts"].argmin(-1) + 1,
            "reliability": data["bank_reliability"].argmin(-1) + 1,
            "redundancy": data["row_novelty"].argmin(-1) + 1,
            "visible_oracle": data["visible_oracle_action"],
            "oracle": data["target_action"],
        }
        for name, actions in policies.items():
            reward_lists[name].append(float(_bank_reward(
                reward_model, data, actions, device=device).mean()))
    return {
        name: sum(values) / len(values)
        for name, values in reward_lists.items()
    }


def _stable_crossing(
        values: list[tuple[int, float]], threshold: float,
        ) -> int | None:
    """Return the first verifier-bit prefix staying above the threshold."""
    for index, (bits, value) in enumerate(values):
        if value >= threshold and all(
                later >= threshold for _, later in values[index:]):
            return bits
    return None


def _shared_backbone_exact(
        first: UnifiedCognitiveController,
        second: UnifiedCognitiveController,
        ) -> bool:
    for name, value in first.state_dict().items():
        if name.startswith("memory_replacement_"):
            continue
        if not torch.equal(value, second.state_dict()[name]):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-banks", type=int, default=32)
    parser.add_argument("--test-banks", type=int, default=128)
    parser.add_argument("--test-seeds", type=int, default=2)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=0.04)
    parser.add_argument("--recency-weight", type=float, default=0.30)
    parser.add_argument("--frequency-weight", type=float, default=0.30)
    parser.add_argument("--reliability-weight", type=float, default=0.30)
    parser.add_argument("--novelty-weight", type=float, default=0.10)
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    if (
            args.steps < 1 or args.batch_banks < 2
            or args.test_banks < 2 or args.test_seeds < 1
            or args.bank_capacity < 3 or args.evaluate_every < 1):
        raise ValueError("budgets and dimensions are too small")
    if args.perturbation <= 0.0 or args.step_size <= 0.0:
        raise ValueError("search scales must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    arms = build_transfer_arms(
        parent, selected, device=device, fresh_seed=args.seed + 123)
    reward_model = arms["selected_experience"]
    weights = (
        args.recency_weight, args.frequency_weight,
        args.reliability_weight, args.novelty_weight)
    evaluation_data = [
        redundancy_utility_batch(
            reward_model, banks=args.test_banks,
            capacity=args.bank_capacity,
            seed=args.seed + 90_000_000 + index * 100_003,
            device=device, write_threshold=args.write_threshold,
            noise_scale=args.noise_scale, weights=weights)
        for index in range(args.test_seeds)
    ]
    strategy_retrieval = initialize_from_saved_strategy(
        arms["selected_strategy_memory"], selected,
        evaluation_data[0]["option_features"])
    old_evaluation_data = [
        frequency_recency_batch(
            reward_model, banks=args.test_banks,
            capacity=args.bank_capacity,
            seed=args.seed + 92_000_000 + index * 100_003,
            device=device, write_threshold=args.write_threshold,
            noise_scale=args.noise_scale,
            recency_weight=1 / 3, frequency_weight=1 / 3,
            reliability_weight=1 / 3)
        for index in range(args.test_seeds)
    ]
    old_utility_prefix_zero = {
        name: _policy_metrics(
            model, reward_model, old_evaluation_data, device=device)
        for name, model in arms.items()
        if name != "fresh_matched"
    }
    controls = _control_metrics(
        reward_model, evaluation_data, device=device,
        seed=args.seed + 91_000_000)
    # Redundancy is the hand-coded oracle for the new primitive. It remains an
    # upper-bound diagnostic rather than a baseline the learner must beat.
    nonlearned = [
        controls[name] for name in (
            "random", "fixed", "skip", "recency", "frequency",
            "reliability")]
    baseline = max(nonlearned)
    available_gap = controls["visible_oracle"] - baseline
    fractions = (0.25, 0.50, 0.75)
    thresholds = {
        str(fraction): baseline + fraction * available_gap
        for fraction in fractions}

    direction_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    histories: dict[str, list[dict[str, object]]] = {
        name: [] for name in ARM_NAMES}
    generated_contexts = 0
    candidate_bits_per_arm = 0
    started = time.perf_counter()

    def record(prefix: int) -> None:
        bits = prefix * args.batch_banks * args.bank_capacity * 2
        for name, model in arms.items():
            metrics = _policy_metrics(
                model, reward_model, evaluation_data, device=device)
            histories[name].append({
                "step": prefix,
                "candidate_verifier_bits": bits,
                **metrics,
                "residual_weights": (
                    model.memory_replacement_extra_gate.weight
                    .detach().flatten().tolist()),
            })

    record(0)
    for step in range(1, args.steps + 1):
        data = redundancy_utility_batch(
            reward_model, banks=args.batch_banks,
            capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, write_threshold=args.write_threshold,
            noise_scale=args.noise_scale, weights=weights)
        generated_contexts += int(data["generated_contexts"])
        candidate_bits_per_arm += (
            2 * args.batch_banks * args.bank_capacity)
        direction = torch.randint(
            0, 2, (3,), generator=direction_generator,
            device=device, dtype=torch.long).to(torch.float32) * 2.0 - 1.0
        for name, model in arms.items():
            current = (
                model.memory_replacement_extra_gate.weight
                .detach().clone())
            candidate_rewards = []
            for sign in (1.0, -1.0):
                model.memory_replacement_extra_gate.weight.copy_(
                    current + sign * args.perturbation
                    * direction.unsqueeze(0))
                actions = model.memory_replacement_scores(
                    data["option_features"]).argmax(-1)
                candidate_rewards.append(_bank_reward(
                    reward_model, data, actions, device=device))
            rewards = torch.stack(candidate_rewards)
            if name == "selected_reward_shuffled":
                swaps = torch.randint(
                    0, 2, (args.batch_banks,),
                    generator=shuffle_generator, device=device,
                    dtype=torch.bool)
                rewards[:, swaps] = rewards.flip(0)[:, swaps]
            means = rewards.mean(1)
            winner = 1.0 if means[0] >= means[1] else -1.0
            model.memory_replacement_extra_gate.weight.copy_(
                current + winner * args.step_size
                * direction.unsqueeze(0))
        if step % args.evaluate_every == 0 or step == args.steps:
            record(step)

    curve_summaries = {}
    for name, history in histories.items():
        points = [
            (row["candidate_verifier_bits"], row["verified_reward"])
            for row in history]
        final = _policy_metrics(
            arms[name], reward_model, evaluation_data,
            device=device, shuffle_novelty=True)
        curve_summaries[name] = {
            "prefix_zero_verified_reward":
                history[0]["verified_reward"],
            "final_verified_reward": history[-1]["verified_reward"],
            "verified_reward_auc": sum(
                row["verified_reward"] - baseline for row in history),
            "novelty_shuffled_final": final,
            "stable_bits_to_gap_fraction": {
                fraction: _stable_crossing(points, threshold)
                for fraction, threshold in thresholds.items()},
        }

    selected_curve = curve_summaries["selected_strategy_memory"]
    selected_crossings = selected_curve["stable_bits_to_gap_fraction"]
    faster_thresholds = []
    for fraction, crossing in selected_crossings.items():
        controls_crossings = [
            curve_summaries[name]["stable_bits_to_gap_fraction"][fraction]
            for name in (
                "selected_experience", "shared_parent",
                "architecture_reset", "replacement_policy_reset",
                "fresh_matched")]
        if crossing is not None and all(
                other is None or crossing < other
                for other in controls_crossings):
            faster_thresholds.append(fraction)

    old_utility = {
        name: _policy_metrics(
            model, reward_model, old_evaluation_data, device=device)
        for name, model in arms.items()
        if name != "fresh_matched"
    }

    binary = evaluate(
        arms["selected_experience"], count=512, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        arms["selected_experience"], count=512, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    reward_shuffle = curve_summaries["selected_reward_shuffled"]
    novelty_drop = (
        selected_curve["final_verified_reward"]
        - selected_curve["novelty_shuffled_final"]["verified_reward"])
    causal_atom_arms = [
        name for name in ARM_NAMES
        if (
            name != "selected_reward_shuffled"
            and curve_summaries[name]["final_verified_reward"]
            >= baseline + 0.005
            and curve_summaries[name]["final_verified_reward"]
            - curve_summaries[name]["novelty_shuffled_final"]
            ["verified_reward"] >= 0.02)
    ]
    gate = {
        "visible_oracle_gap_at_least_2_points":
            available_gap >= 0.02,
        "some_intact_arm_finishes_above_baseline":
            max(
                curve_summaries[name]["final_verified_reward"]
                for name in ARM_NAMES
                if name != "selected_reward_shuffled")
            >= baseline + 0.005,
        "selected_reaches_a_threshold_strictly_first":
            bool(faster_thresholds),
        "selected_old_utility_within_2_points_of_prefix_zero": (
            old_utility["selected_strategy_memory"]["verified_reward"]
            >= (
                old_utility_prefix_zero["selected_strategy_memory"]
                ["verified_reward"] - 0.02)),
        "causal_control_does_not_match_selected": (
            novelty_drop >= 0.005
            or reward_shuffle["verified_reward_auc"]
            < selected_curve["verified_reward_auc"]),
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    gate["accepted_for_three_minute_promotion"] = all(gate.values())
    transfer_ratios = {}
    for fraction, selected_bits in selected_crossings.items():
        fresh_bits = curve_summaries[
            "fresh_matched"]["stable_bits_to_gap_fraction"][fraction]
        transfer_ratios[fraction] = (
            fresh_bits / selected_bits
            if selected_bits not in (None, 0) and fresh_bits is not None
            else None)

    report = {
        "schema": "unified-controller-redundancy-transfer-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "utility_weights": dict(zip(
            UTILITY_NAMES,
            [weight / sum(weights) for weight in weights],
            strict=True)),
        "learner_visible": [
            "controller_created_latent_memory_statistics",
            "row_age", "ordinary_retrieval_access_count",
            "empirical_row_reliability", "row_novelty",
            "later_scalar_verified_outcomes",
        ],
        "hidden_from_learner": [
            "utility_weights", "realized_future_utility",
            "future_query_identity", "optimal_replacement_action",
        ],
        "semantic_or_utility_labels_used_for_training": False,
        "training_signal":
            "matched_symmetric_verified_horse_race",
        "matched_backbone_checks": {
            name: _shared_backbone_exact(
                arms["selected_experience"], arms[name])
            for name in (
                "shared_parent", "architecture_reset",
                "selected_strategy_memory", "selected_reward_shuffled")
        },
        "selected_strategy_retrieval": strategy_retrieval,
        "control_verified_rewards": controls,
        "strongest_nonlearned_control_reward": baseline,
        "visible_oracle_available_gap": available_gap,
        "verified_reward_thresholds": thresholds,
        "histories": histories,
        "curve_summaries": curve_summaries,
        "selected_strictly_faster_thresholds": faster_thresholds,
        "transfer_ratio_fresh_bits_over_selected_bits": transfer_ratios,
        "old_utility_prefix_zero": old_utility_prefix_zero,
        "old_utility_final": old_utility,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "accounting": {
            "unique_generated_logical_lifetimes_per_arm":
                generated_contexts,
            "candidate_verifier_bits_per_arm":
                candidate_bits_per_arm,
            "black_box_updates_per_arm": args.steps,
            "replayed_examples": 0,
            "heldout_evaluation_lifetimes":
                args.test_banks * args.test_seeds,
            "heldout_evaluation_verifier_bits_per_prefix":
                args.test_banks * args.bank_capacity * args.test_seeds,
        },
        "gate": gate,
        "redundancy_atom_mechanistic_gate": {
            "causal_above_baseline_arms": causal_atom_arms,
            "accepted": bool(causal_atom_arms),
        },
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "controls": controls,
        "curve_summaries": curve_summaries,
        "gate": gate,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
