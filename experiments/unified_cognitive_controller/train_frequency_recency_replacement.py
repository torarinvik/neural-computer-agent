"""Reward-train replacement from noisy frequency and recency utility."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_memory_replacement import (
    _bank_reward,
    _gather_rows,
    _select_batch,
    _written_contexts,
    evaluate_replacement,
    replacement_batch,
)


@torch.no_grad()
def frequency_recency_batch(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        noise_scale: float = 0.04,
        recency_weight: float = 0.5,
        frequency_weight: float = 0.5) -> dict[str, object]:
    """Create a bank where neither oldest-only nor frequency-only is optimal."""
    if (
            recency_weight < 0.0 or frequency_weight < 0.0
            or recency_weight + frequency_weight <= 0.0):
        raise ValueError("utility weights must be nonnegative with positive sum")
    context_count = banks * (capacity + 1)
    batch, keys, values, strengths, query_keys, generated = _written_contexts(
        model, count=context_count, seed=seed, device=device,
        write_threshold=write_threshold)
    width = keys.shape[-1]
    key_group = keys.reshape(banks, capacity + 1, width)
    value_group = values.reshape(banks, capacity + 1, width)
    strength_group = strengths.reshape(banks, capacity + 1)
    query_group = query_keys.reshape(banks, capacity + 1, width)

    generator = torch.Generator().manual_seed(seed + 30_000_000)
    permutation = torch.stack([
        torch.randperm(capacity, generator=generator)
        for _ in range(banks)
    ]).to(device)
    bank_keys = _gather_rows(key_group[:, :capacity], permutation)
    bank_values = _gather_rows(value_group[:, :capacity], permutation)
    bank_strengths = torch.gather(
        strength_group[:, :capacity], 1, permutation)
    logical_age = torch.arange(
        1, capacity + 1, device=device,
        dtype=keys.dtype).expand(banks, -1)
    bank_ages = torch.gather(logical_age, 1, permutation)

    # These counts stand for actual prior content-addressed reads. The physical
    # disk audit generates them through record_access=True; this tensor arena
    # avoids thousands of tiny file-backed calls while exposing the same field.
    logical_access = torch.randint(
        0, 10, (banks, capacity), generator=generator,
        dtype=torch.long).to(device)
    bank_access = torch.gather(logical_access, 1, permutation)
    normalized_access = (
        torch.log1p(bank_access.to(keys.dtype)) / math.log(10.0))
    centered_access = normalized_access - 0.5
    normalized_age = bank_ages / capacity
    noise = (
        torch.rand(
            banks, capacity, generator=generator,
            dtype=keys.dtype).to(device) * 2.0 - 1.0
    ) * noise_scale
    total_weight = recency_weight + frequency_weight
    normalized_recency_weight = recency_weight / total_weight
    normalized_frequency_weight = frequency_weight / total_weight
    visible_utility = (
        normalized_recency_weight * normalized_age
        + normalized_frequency_weight * normalized_access)
    realized_utility = visible_utility + noise
    target_slot = realized_utility.argmin(-1)
    target_action = target_slot + 1
    visible_oracle_action = visible_utility.argmin(-1) + 1

    candidate_key = key_group[:, capacity]
    candidate_value = value_group[:, capacity]
    candidate_strength = strength_group[:, capacity]
    candidate_similarity = torch.einsum(
        "bw,bcw->bc",
        torch.nn.functional.normalize(candidate_key, dim=-1),
        torch.nn.functional.normalize(bank_keys, dim=-1))
    row_features = torch.stack((
        normalized_age,
        bank_strengths,
        candidate_similarity,
        candidate_strength.unsqueeze(1).expand(-1, capacity),
        torch.zeros_like(normalized_age),
        centered_access,
    ), dim=-1)
    skip_features = torch.zeros(
        banks, 1, 6, device=device, dtype=keys.dtype)
    skip_features[:, 0, 3] = candidate_strength
    skip_features[:, 0, 4] = 1.0
    option_features = torch.cat((skip_features, row_features), dim=1)

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
    future_batch = _select_batch(batch, future_indices)
    future_queries = _gather_rows(query_group, future_logical)
    return {
        "bank_keys": bank_keys,
        "bank_values": bank_values,
        "bank_strengths": bank_strengths,
        "bank_ages": bank_ages,
        "bank_access_counts": bank_access,
        "candidate_key": candidate_key,
        "candidate_value": candidate_value,
        "candidate_strength": candidate_strength,
        "option_features": option_features,
        "future_batch": future_batch,
        "future_queries": future_queries,
        "target_action": target_action,
        "visible_oracle_action": visible_oracle_action,
        "generated_contexts": generated,
        # Private verifier/audit construction state. These fields never enter
        # the learner; the disk audit uses them to rebuild future queries after
        # ordinary physical retrievals produce the realized access history.
        "source_batch": batch,
        "query_group": query_group,
        "slot_to_logical": permutation,
        "utility_noise": noise,
        "recency_weight": normalized_recency_weight,
        "frequency_weight": normalized_frequency_weight,
    }


@torch.no_grad()
def evaluate_frequency_recency(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        noise_scale: float, recency_weight: float = 0.5,
        frequency_weight: float = 0.5) -> dict[str, object]:
    model.eval()
    data = frequency_recency_batch(
        model, banks=banks, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold,
        noise_scale=noise_scale,
        recency_weight=recency_weight,
        frequency_weight=frequency_weight)
    scores = model.memory_replacement_scores(data["option_features"])
    learned = scores.argmax(-1)
    generator = torch.Generator(device=device).manual_seed(
        seed + 40_000_000)
    random = torch.randint(
        0, capacity + 1, (banks,), generator=generator, device=device)
    skip = torch.zeros(banks, dtype=torch.long, device=device)
    fixed = torch.ones(banks, dtype=torch.long, device=device)
    recency = data["bank_ages"].argmin(-1) + 1
    frequency = data["bank_access_counts"].argmin(-1) + 1
    oracle = data["target_action"]
    visible_oracle = data["visible_oracle_action"]

    age_shuffled_features = data["option_features"].clone()
    age_shuffled_features[:, 1:, 0] = (
        age_shuffled_features[:, 1:, 0].roll(1, dims=1))
    age_shuffled = model.memory_replacement_scores(
        age_shuffled_features).argmax(-1)
    frequency_shuffled_features = data["option_features"].clone()
    frequency_shuffled_features[:, 1:, 5] = (
        frequency_shuffled_features[:, 1:, 5].roll(1, dims=1))
    frequency_shuffled = model.memory_replacement_scores(
        frequency_shuffled_features).argmax(-1)

    policies = {
        "learned": learned,
        "random": random,
        "fixed": fixed,
        "skip": skip,
        "recency": recency,
        "frequency": frequency,
        "visible_oracle": visible_oracle,
        "oracle": oracle,
        "age_shuffled": age_shuffled,
        "frequency_shuffled": frequency_shuffled,
    }
    accuracies = {
        name: float(_bank_reward(
            model, data, action, device=device).mean())
        for name, action in policies.items()
    }
    target_rates = {
        name: float((action == oracle).float().mean())
        for name, action in policies.items()
    }
    strongest_single = max(
        accuracies["recency"], accuracies["frequency"],
        accuracies["random"], accuracies["fixed"], accuracies["skip"])

    def captures_oracle_gap(control: float) -> bool:
        available = max(0.0, accuracies["visible_oracle"] - control)
        captured = accuracies["learned"] - control
        return captured + 1e-6 >= 0.75 * available

    report = {
        "accuracy": accuracies["learned"],
        "target_eviction_rate": target_rates["learned"],
        "policy_accuracies": accuracies,
        "policy_target_eviction_rates": target_rates,
        "replace_rate": float((learned > 0).float().mean()),
        "generated_contexts": data["generated_contexts"],
    }
    report["gate"] = {
        "visible_oracle_target_at_least_85":
            target_rates["visible_oracle"] >= 0.85,
        "learned_target_at_least_75":
            target_rates["learned"] >= 0.75,
        "within_3_points_of_visible_oracle":
            accuracies["learned"] >= accuracies["visible_oracle"] - 0.03,
        "captures_75_percent_of_best_control_gap":
            captures_oracle_gap(strongest_single),
        "age_feature_is_causal":
            accuracies["age_shuffled"]
            <= accuracies["learned"] - 0.04,
        "frequency_feature_is_causal":
            accuracies["frequency_shuffled"]
            <= accuracies["learned"] - 0.04,
        "oracle_is_solved": accuracies["oracle"] >= 0.85,
    }
    report["gate"]["accepted"] = all(report["gate"].values())
    return report


def _load_expanded_model(
        payload: dict[str, object], *, gate_hidden: int,
        device: torch.device) -> tuple[
            UnifiedCognitiveController, dict[str, object]]:
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_replace"] = True
    configuration["adaptive_memory_replace_hidden"] = gate_hidden
    configuration["adaptive_memory_replace_features"] = 6
    model = UnifiedCognitiveController(**configuration).to(device)
    state = dict(payload["state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if (
            not all(
                item.startswith("memory_replacement_")
                for item in missing)
            or unexpected):
        raise ValueError(
            f"unexpected checkpoint mismatch: {missing=}, {unexpected=}")
    return model, configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-banks", type=int, default=128)
    parser.add_argument("--test-banks", type=int, default=1024)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--replacement-cost", type=float, default=0.01)
    parser.add_argument("--noise-scale", type=float, default=0.04)
    parser.add_argument("--recency-weight", type=float, default=0.5)
    parser.add_argument("--frequency-weight", type=float, default=0.5)
    parser.add_argument("--exploration-temperature", type=float, default=4.0)
    parser.add_argument(
        "--rehearsal-temperature", type=float, default=2.0)
    parser.add_argument(
        "--rehearsal-every", type=int, default=0,
        help="Use one recency rehearsal update every N steps; zero disables it.")
    parser.add_argument("--gate-hidden", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    if args.bank_capacity < 3:
        raise ValueError("bank capacity must be at least three")
    if (
            args.exploration_temperature < 1.0
            or args.rehearsal_temperature < 1.0):
        raise ValueError("training temperatures must be at least one")
    if args.rehearsal_every < 0:
        raise ValueError("rehearsal cadence cannot be negative")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = _load_expanded_model(
        payload, gate_hidden=args.gate_hidden, device=device)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.memory_replacement_gate is not None
    assert model.memory_replacement_extra_gate is not None
    for parameter in model.memory_replacement_extra_gate.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.memory_replacement_extra_gate.parameters(),
        lr=args.learning_rate)

    preflight = evaluate_frequency_recency(
        model, banks=min(args.test_banks, 512),
        capacity=args.bank_capacity, seed=args.seed + 80_000_000,
        device=device, write_threshold=args.write_threshold,
        noise_scale=args.noise_scale,
        recency_weight=args.recency_weight,
        frequency_weight=args.frequency_weight)
    print(json.dumps({
        "preflight": preflight,
    }, sort_keys=True), flush=True)

    history = []
    generated_contexts = 0
    query_bits = 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        new_utility_step = (
            args.rehearsal_every == 0
            or step % args.rehearsal_every != 0)
        if new_utility_step:
            data = frequency_recency_batch(
                model, banks=args.batch_banks,
                capacity=args.bank_capacity,
                seed=args.seed * 1_000_000 + step,
                device=device, write_threshold=args.write_threshold,
                noise_scale=args.noise_scale,
                recency_weight=args.recency_weight,
                frequency_weight=args.frequency_weight)
        else:
            data = replacement_batch(
                model, banks=args.batch_banks,
                capacity=args.bank_capacity,
                seed=args.seed * 1_000_000 + step,
                device=device, write_threshold=args.write_threshold)
        generated_contexts += int(data["generated_contexts"])
        query_bits += args.batch_banks * args.bank_capacity
        logits = model.memory_replacement_scores(data["option_features"])
        temperature = (
            args.exploration_temperature if new_utility_step
            else args.rehearsal_temperature)
        distribution = torch.distributions.Categorical(
            logits=logits / temperature)
        actions = distribution.sample()
        with torch.no_grad():
            accuracy = _bank_reward(
                model, data, actions, device=device)
        reward = (
            accuracy
            - args.replacement_cost * (actions > 0).to(accuracy.dtype))
        advantage = reward.detach() - reward.detach().mean()
        loss = -(
            advantage * distribution.log_prob(actions)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.memory_replacement_extra_gate.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "future_accuracy": float(accuracy.mean()),
                "target_eviction_rate": float(
                    (actions == data["target_action"]).float().mean()),
                "replace_rate": float((actions > 0).float().mean()),
                "policy_entropy": float(
                    distribution.entropy().mean().detach()),
                "frequency_weight": float(
                    model.memory_replacement_extra_gate.weight[
                        0, 0].detach()),
                "training_stream": (
                    "frequency_recency" if new_utility_step
                    else "recency_rehearsal"),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    utility_report = evaluate_frequency_recency(
        model, banks=args.test_banks, capacity=args.bank_capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold,
        noise_scale=args.noise_scale,
        recency_weight=args.recency_weight,
        frequency_weight=args.frequency_weight)
    recency_report = evaluate_replacement(
        model, banks=args.test_banks, capacity=args.bank_capacity,
        seed=args.seed + 90_500_000, device=device,
        write_threshold=args.write_threshold)
    binary = evaluate(
        model, count=2048, trials=6, seed=args.seed + 91_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=2048, trials=6, seed=args.seed + 92_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    only_replacement_changed = all(
        name.startswith("memory_replacement_")
        for name in changed)
    recency_retention_gate = {
        "accuracy_at_least_95": recency_report["accuracy"] >= 0.95,
        "target_eviction_at_least_95":
            recency_report["target_eviction_rate"] >= 0.95,
        "capacity_aware_age_causality":
            recency_report["shuffled_age_feature_accuracy"]
            <= (
                recency_report["accuracy"]
                - 0.4 / args.bank_capacity + 1e-6),
    }
    recency_retention_gate["accepted"] = all(
        recency_retention_gate.values())
    admitted = (
        utility_report["gate"]["accepted"]
        and recency_retention_gate["accepted"]
        and binary["gate"]["accepted"]
        and four_rule["gate"]["accepted"]
        and only_replacement_changed)
    report = {
        "schema": "unified-controller-frequency-recency-replacement-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "learner_visible": [
            "controller_created_key_value_latents",
            "row_age", "ordinary_retrieval_access_count",
            "row_write_strength", "candidate_row_similarity",
            "skip_option", "later_scalar_verified_outcomes",
        ],
        "hidden_from_learner": [
            "realized_future_utility",
            "future_query_identity",
            "optimal_eviction_action",
        ],
        "semantic_or_utility_labels_used_for_training": False,
        "training_signal":
            "future_verified_success_minus_generic_replacement_cost",
        "accounting": {
            "generated_support_contexts": generated_contexts,
            "future_query_verifier_bits": query_bits,
            "unique_verifier_bits": generated_contexts + query_bits,
            "unique_logical_lifetimes": generated_contexts,
            "optimizer_updates": args.steps,
            "replayed_examples": 0,
            "training_seconds": (
                history[-1]["elapsed_seconds"] if history else 0.0),
        },
        "preflight": preflight,
        "history": history,
        "frequency_recency_evaluation": utility_report,
        "recency_retention": recency_report,
        "recency_retention_gate": recency_retention_gate,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "only_memory_replacement_gate_changed":
            only_replacement_changed,
        "all_admission_gates_passed": admitted,
        "total_seconds": time.perf_counter() - started,
    }
    if admitted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "frequency_recency_evaluation": utility_report,
        "recency_retained": recency_retention_gate["accepted"],
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
