"""Compare persistent and fresh physical-memory learners at matched budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _arm_metrics(report: dict[str, object]) -> dict[str, float]:
    trace = report["trace"]
    reward_advantages = [
        row["learned_reward"] - row["frozen_reward"] for row in trace]
    target_advantages = [
        row["learned_target_rate"] - row["frozen_target_rate"]
        for row in trace]
    reliability = [
        row for row in trace if row["phase"] == "reliability_dominant"]
    old_return = [
        row for row in trace if row["phase"] == "old_return"]
    return {
        "verified_reward_advantage_auc": sum(reward_advantages),
        "mean_verified_reward_advantage":
            sum(reward_advantages) / len(reward_advantages),
        "target_rate_advantage_auc": sum(target_advantages),
        "mean_target_rate_advantage":
            sum(target_advantages) / len(target_advantages),
        "reliability_first_round_reward_advantage":
            reliability[0]["learned_reward"]
            - reliability[0]["frozen_reward"],
        "reliability_last_round_reward_advantage":
            reliability[-1]["learned_reward"]
            - reliability[-1]["frozen_reward"],
        "old_return_reward_advantage":
            sum(
                row["learned_reward"] - row["frozen_reward"]
                for row in old_return) / len(old_return),
        "wall_seconds": report["total_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent", type=Path, nargs="+", required=True)
    parser.add_argument("--fresh", type=Path, nargs="+", required=True)
    parser.add_argument("--shuffled-control", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if len(args.persistent) != len(args.fresh):
        raise ValueError("persistent and fresh arms must be paired")

    pairs = []
    for persistent_path, fresh_path in zip(
            args.persistent, args.fresh, strict=True):
        persistent = _load(persistent_path)
        fresh = _load(fresh_path)
        persistent_configuration = persistent["configuration"]
        fresh_configuration = fresh["configuration"]
        matched = all(
            persistent_configuration[key] == fresh_configuration[key]
            for key in (
                "seed", "banks", "bank_capacity", "rounds_per_phase",
                "perturbation", "step_size"))
        persistent_metrics = _arm_metrics(persistent)
        fresh_metrics = _arm_metrics(fresh)
        candidate_bits = (
            persistent_configuration["banks"]
            * persistent_configuration["bank_capacity"]
            * persistent["accounting"]["physical_rounds"] * 3)
        pairs.append({
            "seed": persistent_configuration["seed"],
            "matched_configuration": matched,
            "candidate_verifier_bits_per_arm": candidate_bits,
            "persistent": persistent_metrics,
            "fresh": fresh_metrics,
            "persistent_minus_fresh": {
                key: persistent_metrics[key] - fresh_metrics[key]
                for key in (
                    "verified_reward_advantage_auc",
                    "mean_verified_reward_advantage",
                    "target_rate_advantage_auc",
                    "mean_target_rate_advantage",
                    "reliability_first_round_reward_advantage",
                    "reliability_last_round_reward_advantage",
                    "old_return_reward_advantage",
                )},
        })

    shuffled = _load(args.shuffled_control)
    persistent_reward_wins = [
        pair["persistent_minus_fresh"]
        ["verified_reward_advantage_auc"] > 0.0
        for pair in pairs]
    persistent_target_wins = [
        pair["persistent_minus_fresh"]["target_rate_advantage_auc"] > 0.0
        for pair in pairs]
    gate = {
        "all_pair_configurations_matched":
            all(pair["matched_configuration"] for pair in pairs),
        "equal_candidate_verifier_bits_per_arm":
            all(
                pair["candidate_verifier_bits_per_arm"]
                == pairs[0]["candidate_verifier_bits_per_arm"]
                for pair in pairs),
        "persistent_reward_advantage_beats_fresh_in_every_replica":
            all(persistent_reward_wins),
        "persistent_target_advantage_beats_fresh_in_every_replica":
            all(persistent_target_wins),
        "all_unshuffled_arms_retain_old_tasks":
            all(
                _load(path)["binary_retention"]["gate"]["accepted"]
                and _load(path)["four_rule_retention"]["gate"]["accepted"]
                for path in args.persistent + args.fresh),
        "shuffled_reward_control_rejected":
            not shuffled["gate"]["accepted"],
    }
    gate["accepted"] = all(gate.values())
    report = {
        "schema":
            "unified-controller-persistent-vs-fresh-efficiency-v1",
        "metric_definition": (
            "area under (learned verified reward minus frozen verified "
            "reward on the same physical states), at equal candidate "
            "verifier-bit budgets"),
        "pairs": pairs,
        "aggregate": {
            "persistent_reward_advantage_auc_mean": sum(
                pair["persistent"]["verified_reward_advantage_auc"]
                for pair in pairs) / len(pairs),
            "fresh_reward_advantage_auc_mean": sum(
                pair["fresh"]["verified_reward_advantage_auc"]
                for pair in pairs) / len(pairs),
            "persistent_target_advantage_auc_mean": sum(
                pair["persistent"]["target_rate_advantage_auc"]
                for pair in pairs) / len(pairs),
            "fresh_target_advantage_auc_mean": sum(
                pair["fresh"]["target_rate_advantage_auc"]
                for pair in pairs) / len(pairs),
        },
        "gate": gate,
        "honest_boundary": (
            "This isolates a short-horizon memory-lifetime advantage; it "
            "does not yet prove transfer to a novel cognitive primitive."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "aggregate": report["aggregate"],
        "gate": gate,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
