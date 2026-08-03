"""Reward-select a generic retrieval usage-prior scale on physical memory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audit_controller_memory_volatility import audit
from .legacy_model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import evaluate_frequency_recency


CANDIDATE_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def expand_with_adaptive_usage_prior(
        payload: dict[str, object], *, device: torch.device
        ) -> tuple[UnifiedCognitiveController, dict[str, object]]:
    configuration = dict(payload["model_configuration"])
    if int(configuration["adaptive_memory_replace_features"]) != 8:
        raise ValueError("usage-prior adaptation requires the volatility parent")
    configuration["adaptive_memory_usage_prior"] = True
    model = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False)
    if missing != ["memory_usage_prior_scale"] or unexpected:
        raise ValueError(
            f"unexpected adaptive-prior mismatch: {missing=}, {unexpected=}")
    assert model.memory_usage_prior_scale is not None
    with torch.no_grad():
        model.memory_usage_prior_scale.fill_(1.0)
    return model, configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=17400)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--banks-per-candidate", type=int, default=16)
    parser.add_argument("--test-banks", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument("--shuffle-candidate-rewards", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = expand_with_adaptive_usage_prior(
        payload, device=device)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    started = time.perf_counter()
    totals = torch.zeros(len(CANDIDATE_SCALES))
    history = []
    reward_generator = torch.Generator().manual_seed(args.seed + 84_000_000)
    for round_index in range(args.rounds):
        seed = args.seed * 1_000_000 + round_index
        rewards = []
        for scale in CANDIDATE_SCALES:
            result = audit(
                model, banks=args.banks_per_candidate,
                capacity=args.capacity, seed=seed, device=device,
                write_threshold=args.write_threshold,
                equal_strength=False, usage_prior_scale=scale)
            # Training consumes only the scalar pixel-task verifier score.
            rewards.append(float(result["accuracy"]))
        credited = torch.tensor(rewards)
        if args.shuffle_candidate_rewards:
            credited = credited[torch.randperm(
                credited.numel(), generator=reward_generator)]
        totals += credited
        history.append({
            "round": round_index + 1,
            "candidate_scales": list(CANDIDATE_SCALES),
            "verified_rewards": rewards,
            "credited_rewards": credited.tolist(),
            "leader": CANDIDATE_SCALES[int(totals.argmax())],
            "elapsed_seconds": time.perf_counter() - started,
        })
    selected_index = int(totals.argmax())
    selected_scale = CANDIDATE_SCALES[selected_index]
    with torch.no_grad():
        model.memory_usage_prior_scale.fill_(selected_scale)

    parent = audit(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold, equal_strength=False,
        usage_prior_scale=1.0)
    learned = audit(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold, equal_strength=False,
        usage_prior_scale=selected_scale)
    shuffled = audit(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold,
        intervention="shuffle_volatility", equal_strength=False,
        usage_prior_scale=selected_scale)
    reversed_histories = audit(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold,
        intervention="reverse_histories", equal_strength=False,
        usage_prior_scale=selected_scale)
    old_utility = evaluate_frequency_recency(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 92_000_000, device=device,
        write_threshold=args.write_threshold, noise_scale=0.04,
        recency_weight=0.3, frequency_weight=0.3,
        reliability_weight=0.4)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    total_seconds = time.perf_counter() - started
    verifier_bits = (
        args.rounds * len(CANDIDATE_SCALES)
        * args.banks_per_candidate * (args.capacity // 2 + 1))
    report = {
        "schema": "unified-controller-memory-usage-prior-race-v1",
        "configuration": {
            **vars(args),
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
            "candidate_scales": list(CANDIDATE_SCALES),
        },
        "model_configuration": configuration,
        "history": history,
        "selected_scale": selected_scale,
        "candidate_mean_verified_rewards": (
            totals.div(args.rounds).tolist()),
        "parent_scale_one": parent,
        "learned": learned,
        "volatility_shuffled": shuffled,
        "outcome_histories_reversed": reversed_histories,
        "retention": {
            "old_utility": old_utility,
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "changed_parameters": changed,
        "accounting": {
            "unique_logical_contexts": (
                args.rounds * args.banks_per_candidate
                * (args.capacity + 1)),
            "counterfactual_candidate_evaluations":
                args.rounds * len(CANDIDATE_SCALES),
            "unique_verifier_bits": verifier_bits,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "total_seconds": total_seconds,
        },
        "gates": {
            "selected_content_first_scale":
                selected_scale <= 0.25,
            "learned_valid_replacement_at_least_95_percent":
                learned["valid_replacement_rate"] >= 0.95,
            "learned_accuracy_at_least_90_percent":
                learned["accuracy"] >= 0.90,
            "improves_parent_valid_replacement_by_15_points":
                learned["valid_replacement_rate"]
                >= parent["valid_replacement_rate"] + 0.15,
            "volatility_shuffle_costs_30_points":
                learned["valid_replacement_rate"]
                >= shuffled["valid_replacement_rate"] + 0.30,
            "outcome_reversal_flips_90_percent":
                reversed_histories["stable_eviction_rate"] >= 0.90,
            "physical_state_persists":
                learned["all_history_fields_persist_exactly"],
            "physical_capacity_bounded":
                learned["all_banks_remain_bounded"],
            "old_utility_retained":
                old_utility["accuracy"] >= 0.90,
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_usage_prior_changed":
                changed == ["memory_usage_prior_scale"],
            "under_five_minutes": total_seconds <= 300.0,
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "history": history,
        "selected_scale": selected_scale,
        "candidate_mean_verified_rewards":
            report["candidate_mean_verified_rewards"],
        "parent_scale_one": parent,
        "learned": learned,
        "volatility_shuffled": shuffled,
        "outcome_histories_reversed": reversed_histories,
        "accounting": report["accounting"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
