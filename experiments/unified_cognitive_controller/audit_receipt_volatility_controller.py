"""Audit learned volatility after receipt-attributed physical histories.

The learned controller was previously trained on generic outcome-order
features.  This audit checks that those features remain causal when histories
are produced by real ``DiskLatentMemory`` retrievals with unequal admission
strengths.  The controller never sees stable/decoy labels; those are verifier
state used only to score the audit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .train import evaluate
from .train_controller_memory_volatility import volatility_batch
from .train_memory_replacement import _bank_reward


@torch.no_grad()
def _physical_volatility(
        model: UnifiedCognitiveController, data: dict[str, object], *,
        policy: str, device: torch.device,
        ) -> torch.Tensor:
    if policy not in {"receipt", "ordinary", "shuffled_receipt"}:
        raise ValueError("unknown attribution policy")
    banks, capacity = data["bank_keys"].shape[:2]
    result = []
    for bank in range(banks):
        memory = DiskLatentMemory(
            model.width, capacity=capacity, device=device)
        memory.commit(
            data["bank_keys"][bank], data["bank_values"][bank],
            data["bank_strengths"][bank], threshold=0.0)
        queries = torch.stack([
            data["query_group"][bank, int(data["slot_to_logical"][bank, slot])]
            for slot in range(capacity)])
        stable = data["stable_mask"][bank]
        failures_then_successes = torch.where(
            stable, torch.zeros(capacity, device=device),
            torch.ones(capacity, device=device))
        successes_then_failures = 1.0 - failures_then_successes
        for outcomes in (
                failures_then_successes, failures_then_successes,
                failures_then_successes, failures_then_successes,
                failures_then_successes, successes_then_failures,
                successes_then_failures, successes_then_failures,
                successes_then_failures, successes_then_failures):
            if policy == "ordinary":
                memory.store.record_outcomes(
                    queries, outcomes, update_volatility=True,
                    success_protection_rate=0.2,
                    failure_thaw_rate=0.25, stale_thaw_rate=0.0,
                    usage_prior_scale=1.0)
            else:
                _, _, receipts = memory.retrieve_with_receipt(
                    queries, top_k=1, confidence_mode="cosine",
                    usage_prior_scale=0.0)
                if policy == "shuffled_receipt":
                    receipts = receipts.roll(1)
                memory.record_outcomes_from_receipts(
                    receipts, outcomes, update_volatility=True,
                    success_protection_rate=0.2,
                    failure_thaw_rate=0.25, stale_thaw_rate=0.0)
        result.append(memory.store.volatility)
    return torch.stack(result)


@torch.no_grad()
def _evaluate_policy(
        model: UnifiedCognitiveController, data: dict[str, object], *,
        policy: str, device: torch.device) -> dict[str, float]:
    volatility = _physical_volatility(
        model, data, policy=policy, device=device)
    options = data["option_features"].clone()
    options[:, 1:, 7] = volatility
    actions = model.memory_replacement_scores(options).argmax(-1)
    reward = _bank_reward(model, data, actions, device=device)
    stable_selected = torch.zeros_like(actions, dtype=torch.bool)
    replacing = actions > 0
    stable_selected[replacing] = data["stable_mask"][
        replacing, actions[replacing] - 1]
    oracle_actions = (~data["stable_mask"]).to(torch.float32).argmax(-1) + 1
    return {
        "accuracy": float(reward.mean()),
        "stable_eviction_rate": float(stable_selected.float().mean()),
        "replace_rate": float(replacing.float().mean()),
        "oracle_action_rate": float((actions == oracle_actions).float().mean()),
        "stable_volatility": float(
            volatility[data["stable_mask"]].mean()),
        "decoy_volatility": float(
            volatility[~data["stable_mask"]].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17107)
    parser.add_argument("--banks", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--retention-count", type=int, default=128)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.banks < 1 or args.capacity < 4 or args.capacity % 2:
        raise ValueError("banks must be positive and capacity even >= 4")
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if model.adaptive_memory_replace_features != 8:
        raise ValueError("receipt audit requires the learned eight-feature model")
    started = time.perf_counter()
    data = volatility_batch(
        model, banks=args.banks, capacity=args.capacity, seed=args.seed,
        device=device, write_threshold=0.5)
    policies = {
        name: _evaluate_policy(
            model, data, policy=name, device=device)
        for name in ("receipt", "ordinary", "shuffled_receipt")
    }
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    receipt = policies["receipt"]
    shuffled = policies["shuffled_receipt"]
    ordinary = policies["ordinary"]
    total_seconds = time.perf_counter() - started
    report = {
        "schema": "unified-controller-receipt-volatility-audit-v1",
        "checkpoint": str(args.checkpoint),
        "configuration": vars(args) | {"device": str(device)},
        "semantic_or_task_labels_used_for_training": False,
        "learner_visible": [
            "latent keys and values", "retrieval confidence",
            "write strength", "physical read receipt", "volatility"],
        "policies": policies,
        "retention": {"binary_mapping": binary, "four_rule": four_rule},
        "accounting": {
            "banks": args.banks,
            "physical_reads": args.banks * args.capacity * 10,
            "verifier_bits": args.banks * args.capacity * 10,
            "total_seconds": total_seconds,
        },
        "gates": {
            "receipt_accuracy_at_least_95": receipt["accuracy"] >= 0.95,
            "receipt_stable_eviction_at_most_10_percent":
                receipt["stable_eviction_rate"] <= 0.10,
            "receipt_beats_ordinary":
                receipt["accuracy"] >= ordinary["accuracy"],
            "shuffled_receipt_costs_at_least_6_points":
                receipt["accuracy"] >= shuffled["accuracy"] + 0.06,
            "receipt_volatility_separates_stable_and_decoy":
                receipt["stable_volatility"]
                < receipt["decoy_volatility"],
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "under_five_minute_cap": total_seconds <= 300.0,
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
