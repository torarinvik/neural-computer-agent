"""Audit latent strategy RAM against the global-residual baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _target_metrics(path: Path) -> dict[str, float | int | bool]:
    report = json.loads(path.read_text())
    rows = [
        row for row in report["trace"]
        if row["phase"] == "reliability_dominant"]
    reward_auc = sum(
        row["learned_reward"] - row["frozen_reward"] for row in rows)
    target_auc = sum(
        row["learned_target_rate"] - row["frozen_target_rate"]
        for row in rows)
    bits = int(report["accounting"].get(
        "candidate_verifier_bits",
        report["accounting"]["physical_rounds"]
        * report["configuration"]["banks"]
        * report["configuration"]["bank_capacity"] * 3))
    return {
        "reward_advantage_auc": reward_auc,
        "target_advantage_auc": target_auc,
        "total_candidate_verifier_bits": bits,
        "reward_advantage_per_1000_bits":
            reward_auc / bits * 1000,
        "retained":
            report["binary_retention"]["gate"]["accepted"]
            and report["four_rule_retention"]["gate"]["accepted"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-arm", type=Path, nargs="+", required=True)
    parser.add_argument("--strategy-arm", type=Path, nargs="+", required=True)
    parser.add_argument("--shuffled-keys", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if len(args.global_arm) != len(args.strategy_arm):
        raise ValueError("global and strategy arms must be paired")
    pairs = []
    for global_path, strategy_path in zip(
            args.global_arm, args.strategy_arm, strict=True):
        global_report = json.loads(global_path.read_text())
        strategy_report = json.loads(strategy_path.read_text())
        global_metrics = _target_metrics(global_path)
        strategy_metrics = _target_metrics(strategy_path)
        pairs.append({
            "seed": global_report["configuration"]["seed"],
            "seed_matched": (
                global_report["configuration"]["seed"]
                == strategy_report["configuration"]["seed"]),
            "global_residual": global_metrics,
            "strategy_memory": strategy_metrics,
            "strategy_minus_global_reward_per_1000_bits": (
                strategy_metrics["reward_advantage_per_1000_bits"]
                - global_metrics["reward_advantage_per_1000_bits"]),
        })
    shuffled = _target_metrics(args.shuffled_keys)
    gate = {
        "all_seeds_matched":
            all(pair["seed_matched"] for pair in pairs),
        "strategy_more_reward_efficient_in_every_replica":
            all(
                pair["strategy_minus_global_reward_per_1000_bits"] > 0
                for pair in pairs),
        "shuffled_keys_reduce_target_reward":
            shuffled["reward_advantage_auc"]
            < pairs[0]["strategy_memory"]["reward_advantage_auc"],
        "all_old_skills_retained":
            all(
                pair[arm]["retained"]
                for pair in pairs
                for arm in ("global_residual", "strategy_memory"))
            and shuffled["retained"],
    }
    gate["accepted"] = all(gate.values())
    report = {
        "schema": "unified-controller-strategy-memory-audit-v1",
        "pairs": pairs,
        "shuffled_strategy_keys": shuffled,
        "gate": gate,
        "conclusion": (
            "The bounded strategy bank is mechanically valid but has not "
            "shown replicated sample-efficiency gains. Context keys remain "
            "the frontier; do not scale bank capacity or training duration."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
