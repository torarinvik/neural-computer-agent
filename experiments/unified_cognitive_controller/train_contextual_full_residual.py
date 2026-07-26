"""Learn a context-retrieved full-feature replacement-policy residual."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .strategy_memory import LatentStrategyMemory
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import _bank_reward
from .train_redundancy_transfer import (
    _control_metrics,
    _stable_crossing,
    build_transfer_arms,
    redundancy_utility_batch,
)


ARM_NAMES = ("contextual", "reward_shuffled", "global")
CONTEXT_FEATURES = (0, 1, 2, 5, 6, 7)


def full_residual_context_key(
        option_features: torch.Tensor) -> torch.Tensor:
    """Summarize generic visible statistics without a task identifier."""
    if option_features.ndim != 3 or option_features.shape[-1] != 8:
        raise ValueError("expected [banks, options, 8] features")
    rows = option_features[:, 1:, CONTEXT_FEATURES]
    summary = torch.cat((
        rows.mean(dim=(0, 1)),
        rows.std(dim=(0, 1), unbiased=False),
        (rows.abs().mean(dim=(0, 1)) > 1e-6).to(rows.dtype),
    ))
    return torch.nn.functional.normalize(summary, dim=0)


def residual_scores(
        model, option_features: torch.Tensor,
        residual: torch.Tensor) -> torch.Tensor:
    if residual.shape != (8,):
        raise ValueError("full-feature residual must have width eight")
    return (
        model.memory_replacement_scores(option_features)
        + torch.einsum("bof,f->bo", option_features, residual))


def adapter_scores(
        model, option_features: torch.Tensor,
        adapter: torch.Tensor, *, mode: str,
        exact_zero_bypass: bool = True) -> torch.Tensor:
    """Apply either the broad diagnostic or a low-dimensional arbitrator."""
    base = model.memory_replacement_scores(option_features)
    if exact_zero_bypass and int(torch.count_nonzero(adapter)) == 0:
        return base
    if mode == "full_residual":
        return residual_scores(model, option_features, adapter)
    if mode != "suppress_novelty" or adapter.shape != (2,):
        raise ValueError("unsupported adapter mode or width")
    old_policy_scale = adapter[0].clamp(-6.0, 3.0).exp()
    novelty = option_features[..., 7]
    return old_policy_scale * base + adapter[1] * novelty


def retrieve_residual(
        memory: LatentStrategyMemory,
        key: torch.Tensor, *, threshold: float,
        ) -> tuple[torch.Tensor, bool, int | None, float]:
    result = memory.retrieve(key, torch.zeros(
        memory.value_width, device=key.device, dtype=key.dtype))
    accepted = result.slot is not None and result.similarity >= threshold
    return (
        result.value if accepted else torch.zeros_like(result.value),
        accepted,
        result.slot if accepted else None,
        result.similarity,
    )


@torch.no_grad()
def _metrics(
        model, datasets: list[dict[str, object]], *,
        device: torch.device,
        residual: torch.Tensor | None = None,
        memory: LatentStrategyMemory | None = None,
        threshold: float = 0.982,
        shuffle_novelty: bool = False,
        adapter_mode: str = "full_residual",
        ) -> dict[str, object]:
    rewards = []
    targets = []
    accepted = []
    similarities = []
    for data in datasets:
        features = data["option_features"].clone()
        if shuffle_novelty:
            features[:, 1:, 7] = features[:, 1:, 7].roll(1, dims=1)
        active = residual
        if memory is not None:
            active, used, _, similarity = retrieve_residual(
                memory, full_residual_context_key(features),
                threshold=threshold)
            accepted.append(used)
            similarities.append(similarity)
        if active is None:
            active = torch.zeros(
                8 if adapter_mode == "full_residual" else 2,
                device=device)
        actions = adapter_scores(
            model, features, active, mode=adapter_mode).argmax(-1)
        rewards.append(float(_bank_reward(
            model, data, actions, device=device).mean()))
        targets.append(float(
            (actions == data["target_action"]).float().mean()))
    return {
        "verified_reward": sum(rewards) / len(rewards),
        "target_rate_diagnostic": sum(targets) / len(targets),
        "retrieval_acceptance_rate": (
            sum(accepted) / len(accepted) if accepted else None),
        "mean_retrieval_similarity": (
            sum(similarities) / len(similarities)
            if similarities else None),
    }


def _memory_equal(
        left: LatentStrategyMemory,
        right: LatentStrategyMemory) -> bool:
    return (
        left.count == right.count
        and all(torch.equal(
            getattr(left, field), getattr(right, field))
            for field in (
                "keys", "values", "usage",
                "success", "failure"))
    )


def select_verified_candidate(
        means: torch.Tensor, *, center_index: int = 1,
        tolerance: float = 1e-6) -> int:
    """Prefer no update unless another candidate is verifiably better."""
    if means.ndim != 1 or not 0 <= center_index < means.shape[0]:
        raise ValueError("candidate means or center index are invalid")
    best = int(means.argmax())
    if means[best] <= means[center_index] + tolerance:
        return center_index
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7311)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-banks", type=int, default=16)
    parser.add_argument("--test-banks", type=int, default=32)
    parser.add_argument("--bank-capacity", type=int, default=3)
    parser.add_argument("--strategy-capacity", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.982)
    parser.add_argument("--direction-proposals", type=int, default=16)
    parser.add_argument("--perturbation", type=float, default=8.0)
    parser.add_argument("--step-size", type=float, default=4.0)
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument(
        "--adapter-mode",
        choices=("full_residual", "suppress_novelty"),
        default="full_residual")
    parser.add_argument(
        "--learning-rule", choices=("spsa", "reinforce"),
        default="spsa")
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=2.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--normalize-policy-gradient", action="store_true")
    args = parser.parse_args()
    if (
            args.steps < 1 or args.batch_banks < 2
            or args.test_banks < 2 or args.bank_capacity < 3
            or args.strategy_capacity < 1
            or args.direction_proposals < 1
            or args.evaluate_every < 1):
        raise ValueError("invalid experiment dimensions")
    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("similarity threshold must be in [0, 1]")
    if args.perturbation <= 0.0 or args.step_size <= 0.0:
        raise ValueError("search scales must be positive")
    if (
            args.learning_rate <= 0.0
            or args.policy_temperature <= 0.0
            or args.entropy_bonus < 0.0):
        raise ValueError("policy-gradient scales are invalid")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    transfer_arms = build_transfer_arms(
        parent, selected, device=device, fresh_seed=args.seed + 101)
    model = transfer_arms["selected_experience"]
    adapter_width = 8 if args.adapter_mode == "full_residual" else 2
    memories = {
        name: LatentStrategyMemory(
            capacity=args.strategy_capacity,
            key_width=18, value_width=adapter_width, device=device)
        for name in ("contextual", "reward_shuffled")
    }
    global_residual = torch.zeros(adapter_width, device=device)
    eval_data = [
        redundancy_utility_batch(
            model, banks=args.test_banks,
            capacity=args.bank_capacity,
            seed=args.seed + 90_000_000 + index * 100_003,
            device=device, write_threshold=0.5,
            noise_scale=0.0, weights=(0.0, 0.0, 0.0, 1.0))
        for index in range(2)
    ]
    old_contexts = {
        "old_equal": (0.5, 0.5, 0.0),
        "reliability_dominant": (0.3, 0.3, 0.4),
        "old_return": (0.5, 0.5, 0.0),
    }
    old_data = {
        name: frequency_recency_batch(
            model, banks=args.test_banks,
            capacity=args.bank_capacity,
            seed=args.seed + 92_000_000 + index * 100_003,
            device=device, write_threshold=0.5, noise_scale=0.04,
            recency_weight=weights[0],
            frequency_weight=weights[1],
            reliability_weight=weights[2])
        for index, (name, weights) in enumerate(old_contexts.items())
    }
    controls = _control_metrics(
        model, eval_data, device=device, seed=args.seed + 91_000_000)
    baseline = max(controls[name] for name in (
        "random", "fixed", "skip", "recency",
        "frequency", "reliability"))
    available_gap = controls["visible_oracle"] - baseline
    thresholds = {
        str(fraction): baseline + fraction * available_gap
        for fraction in (0.25, 0.5, 0.75)}
    direction_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    action_generators = {
        name: torch.Generator(device=device).manual_seed(
            args.seed + 72_000_000)
        for name in ARM_NAMES}
    histories = {name: [] for name in ARM_NAMES}
    generated_contexts = 0
    candidate_bits = 0
    persistence_checks = 0
    persistence_exact = 0
    action_divergent_rounds = {name: 0 for name in ARM_NAMES}
    training_trace = []
    started = time.perf_counter()

    def record(step: int) -> None:
        bits = (
            step * args.batch_banks * args.bank_capacity
            * (3 if args.learning_rule == "spsa" else 1))
        for name in ARM_NAMES:
            if name == "global":
                metrics = _metrics(
                    model, eval_data, device=device,
                    residual=global_residual,
                    threshold=args.similarity_threshold,
                    adapter_mode=args.adapter_mode)
                count = None
            else:
                metrics = _metrics(
                    model, eval_data, device=device,
                    memory=memories[name],
                    threshold=args.similarity_threshold,
                    adapter_mode=args.adapter_mode)
                count = memories[name].count
            histories[name].append({
                "step": step,
                "candidate_verifier_bits": bits,
                **metrics,
                "strategy_count": count,
                "global_residual": (
                    global_residual.tolist()
                    if name == "global" else None),
            })

    record(0)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for step in range(1, args.steps + 1):
            data = redundancy_utility_batch(
                model, banks=args.batch_banks,
                capacity=args.bank_capacity,
                seed=args.seed * 1_000_000 + step,
                device=device, write_threshold=0.5,
                noise_scale=0.0, weights=(0.0, 0.0, 0.0, 1.0))
            generated_contexts += int(data["generated_contexts"])
            candidate_bits += (
                (3 if args.learning_rule == "spsa" else 1)
                * args.batch_banks * args.bank_capacity)
            features = data["option_features"]
            key = full_residual_context_key(features)
            proposals = [
                torch.randint(
                    0, 2, (adapter_width,), generator=direction_generator,
                    device=device, dtype=torch.long).float() * 2.0 - 1.0
                for _ in range(args.direction_proposals)
            ]
            for name in ARM_NAMES:
                accepted = False
                slot = None
                if name == "global":
                    current = global_residual.clone()
                else:
                    current, accepted, slot, _ = retrieve_residual(
                        memories[name], key,
                        threshold=args.similarity_threshold)
                if args.learning_rule == "spsa":
                    screened = []
                    for direction in proposals:
                        positive = adapter_scores(
                            model, features,
                            current + args.perturbation * direction,
                            mode=args.adapter_mode).argmax(-1)
                        negative = adapter_scores(
                            model, features,
                            current - args.perturbation * direction,
                            mode=args.adapter_mode).argmax(-1)
                        screened.append((
                            int((positive != negative).sum()), direction))
                    disagreement, direction = max(
                        screened, key=lambda item: item[0])
                    signs = (1.0, 0.0, -1.0)
                    rewards = []
                    for sign in signs:
                        candidate = (
                            current + sign * args.perturbation * direction)
                        actions = adapter_scores(
                            model, features, candidate,
                            mode=args.adapter_mode).argmax(-1)
                        rewards.append(_bank_reward(
                            model, data, actions, device=device))
                    reward_tensor = torch.stack(rewards)
                    aligned = reward_tensor.clone()
                    if name == "reward_shuffled":
                        for bank in range(args.batch_banks):
                            order = torch.randperm(
                                3, generator=shuffle_generator, device=device)
                            aligned[:, bank] = reward_tensor[order, bank]
                    means = aligned.mean(1)
                    winner = select_verified_candidate(means)
                    updated = (
                        current
                        + signs[winner] * args.step_size * direction)
                    perceived_improvement = float(
                        means[winner] - means[1])
                    trace_rewards = [
                        float(value) for value in reward_tensor.mean(1)]
                    aligned_trace_rewards = [
                        float(value) for value in means]
                    winner_sign = signs[winner]
                else:
                    trainable = current.detach().clone().requires_grad_(True)
                    logits = adapter_scores(
                        model, features, trainable,
                        mode=args.adapter_mode,
                        exact_zero_bypass=False)
                    probabilities = (
                        logits / args.policy_temperature).softmax(-1)
                    actions = torch.multinomial(
                        probabilities, 1,
                        generator=action_generators[name]).squeeze(1)
                    rewards = _bank_reward(
                        model, data, actions, device=device)
                    aligned = rewards.clone()
                    if name == "reward_shuffled":
                        aligned = aligned[torch.randperm(
                            args.batch_banks,
                            generator=shuffle_generator, device=device)]
                    log_probabilities = (
                        logits / args.policy_temperature
                    ).log_softmax(-1)
                    attempted_log_probability = torch.gather(
                        log_probabilities, 1,
                        actions.unsqueeze(1)).squeeze(1)
                    advantage = aligned - aligned.mean()
                    entropy = -(
                        probabilities * log_probabilities).sum(-1).mean()
                    loss = -(
                        advantage.detach()
                        * attempted_log_probability).mean()
                    loss = loss - args.entropy_bonus * entropy
                    gradient, = torch.autograd.grad(loss, trainable)
                    gradient = gradient.clamp(-1.0, 1.0)
                    if args.normalize_policy_gradient:
                        gradient = gradient / gradient.norm().clamp_min(1e-8)
                    updated = (
                        current - args.learning_rate * gradient.detach())
                    disagreement = int(gradient.norm() > 1e-9)
                    perceived_improvement = float(
                        (aligned - aligned.mean()).abs().mean())
                    trace_rewards = [float(rewards.mean())]
                    aligned_trace_rewards = [float(aligned.mean())]
                    winner_sign = None
                action_divergent_rounds[name] += int(disagreement > 0)
                if name == "global":
                    global_residual.copy_(updated)
                else:
                    memories[name].upsert(
                        key, updated,
                        verified_improvement=perceived_improvement,
                        preferred_slot=slot if accepted else None)
                    path = root / f"{name}-{step}.pt"
                    memories[name].save(path)
                    restored = LatentStrategyMemory.load(
                        path, device=device)
                    persistence_checks += 1
                    persistence_exact += int(
                        _memory_equal(memories[name], restored))
                    memories[name] = restored
                training_trace.append({
                    "step": step,
                    "arm": name,
                    "retrieval_accepted": accepted,
                    "retrieval_slot": slot,
                    "action_disagreement_count": disagreement,
                    "candidate_mean_rewards": trace_rewards,
                    "aligned_candidate_mean_rewards":
                        aligned_trace_rewards,
                    "winner_sign": winner_sign,
                    "current_residual_norm": float(current.norm()),
                    "updated_residual_norm": float(updated.norm()),
                })
            if step % args.evaluate_every == 0 or step == args.steps:
                record(step)

    summaries = {}
    for name in ARM_NAMES:
        history = histories[name]
        points = [
            (row["candidate_verifier_bits"], row["verified_reward"])
            for row in history]
        if name == "global":
            shuffled = _metrics(
                model, eval_data, device=device,
                residual=global_residual, shuffle_novelty=True,
                adapter_mode=args.adapter_mode)
        else:
            shuffled = _metrics(
                model, eval_data, device=device,
                memory=memories[name],
                threshold=args.similarity_threshold,
                shuffle_novelty=True,
                adapter_mode=args.adapter_mode)
        summaries[name] = {
            "prefix_zero_verified_reward":
                history[0]["verified_reward"],
            "final_verified_reward": history[-1]["verified_reward"],
            "novelty_shuffled_final": shuffled,
            "verified_reward_auc": sum(
                row["verified_reward"] - baseline for row in history),
            "stable_bits_to_gap_fraction": {
                fraction: _stable_crossing(points, threshold)
                for fraction, threshold in thresholds.items()},
        }

    old_context_audit = {}
    for name, data in old_data.items():
        key = full_residual_context_key(data["option_features"])
        residual, accepted, slot, similarity = retrieve_residual(
            memories["contextual"], key,
            threshold=args.similarity_threshold)
        base_scores = model.memory_replacement_scores(
            data["option_features"])
        contextual_scores = adapter_scores(
            model, data["option_features"], residual,
            mode=args.adapter_mode)
        old_context_audit[name] = {
            "retrieval_accepted": accepted,
            "retrieval_slot": slot,
            "similarity": similarity,
            "scores_bit_identical": torch.equal(
                base_scores, contextual_scores),
        }

    binary = evaluate(
        model, count=512, trials=6, seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=512, trials=6, seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    intact = summaries["contextual"]
    shuffled = summaries["reward_shuffled"]
    novelty_drop = (
        intact["final_verified_reward"]
        - intact["novelty_shuffled_final"]["verified_reward"])
    gate = {
        "visible_oracle_gap_at_least_5_points":
            available_gap >= 0.05,
        "contextual_reaches_stable_25_percent_gap":
            intact["stable_bits_to_gap_fraction"]["0.25"] is not None,
        "contextual_finishes_2_points_above_unchanged":
            intact["final_verified_reward"]
            >= histories["contextual"][0]["verified_reward"] + 0.02,
        "novelty_shuffle_costs_at_least_4_points":
            novelty_drop >= 0.04,
        "aligned_auc_beats_reward_shuffle":
            intact["verified_reward_auc"]
            > shuffled["verified_reward_auc"],
        "heldout_new_contexts_activate_retrieval":
            histories["contextual"][-1][
                "retrieval_acceptance_rate"] == 1.0,
        "all_old_contexts_reject_and_remain_exact":
            all(
                not audit["retrieval_accepted"]
                and audit["scores_bit_identical"]
                for audit in old_context_audit.values()),
        "every_strategy_save_reload_exact":
            persistence_checks > 0
            and persistence_checks == persistence_exact,
        "every_intact_round_had_learning_signal":
            action_divergent_rounds["contextual"] == args.steps,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    gate["accepted_for_three_minute_promotion"] = all(gate.values())
    report = {
        "schema": "unified-controller-contextual-full-residual-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "report": str(args.report),
        },
        "semantic_or_utility_labels_used_for_training": False,
        "training_signal":
            "three_candidate_scalar_verified_horse_race",
        "context_key":
            "mean_std_and_activity_of_six_generic_visible_row_statistics",
        "adapter_mode": args.adapter_mode,
        "controls": controls,
        "strongest_nonredundancy_control": baseline,
        "available_oracle_gap": available_gap,
        "thresholds": thresholds,
        "histories": histories,
        "training_trace": training_trace,
        "summaries": summaries,
        "old_context_audit": old_context_audit,
        "action_divergent_rounds": action_divergent_rounds,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "accounting": {
            "unique_generated_logical_lifetimes_per_arm":
                generated_contexts,
            "candidate_verifier_bits_per_arm": candidate_bits,
            "black_box_updates_per_arm": args.steps,
            "reward_free_direction_proposals_per_arm":
                (
                    args.steps * args.direction_proposals
                    if args.learning_rule == "spsa" else 0),
            "replayed_examples": 0,
            "strategy_save_reload_checks": persistence_checks,
        },
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate,
        "summaries": summaries,
        "old_context_audit": old_context_audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
