"""Compare gradual-relation transfer with controller and memory ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metrics(path: Path) -> dict[str, float | int | None]:
    report = json.loads(path.read_text())
    rows = [
        row for row in report["trace"]
        if row["phase"] == "reliability_dominant"]
    reward = [
        row["learned_reward"] - row["frozen_reward"] for row in rows]
    target = [
        row["learned_target_rate"] - row["frozen_target_rate"]
        for row in rows]

    def first_crossing(values: list[float], threshold: float) -> int | None:
        for index, value in enumerate(values, start=1):
            if value >= threshold:
                return index
        return None

    return {
        "reward_advantage_auc": sum(reward),
        "target_advantage_auc": sum(target),
        "first_round_reaching_10_point_target_advantage":
            first_crossing(target, 0.10),
        "candidate_verifier_bits_per_round": (
            report["configuration"]["banks"]
            * report["configuration"]["bank_capacity"] * 3),
        "binary_retained":
            report["binary_retention"]["gate"]["accepted"],
        "four_rule_retained":
            report["four_rule_retention"]["gate"]["accepted"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intact", type=Path, nargs="+", required=True)
    parser.add_argument("--cold", type=Path, nargs="+", required=True)
    parser.add_argument("--shuffled", type=Path, nargs="+", required=True)
    parser.add_argument("--empty", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not (
            len(args.intact) == len(args.cold)
            == len(args.shuffled)):
        raise ValueError("intact, cold, and shuffled arms must be paired")

    pairs = []
    for intact_path, cold_path, shuffled_path in zip(
            args.intact, args.cold, args.shuffled, strict=True):
        intact_report = json.loads(intact_path.read_text())
        cold_report = json.loads(cold_path.read_text())
        shuffled_report = json.loads(shuffled_path.read_text())
        seeds = {
            intact_report["configuration"]["seed"],
            cold_report["configuration"]["seed"],
            shuffled_report["configuration"]["seed"],
        }
        intact = _metrics(intact_path)
        cold = _metrics(cold_path)
        shuffled = _metrics(shuffled_path)
        pairs.append({
            "seed": next(iter(seeds)) if len(seeds) == 1 else None,
            "seed_matched": len(seeds) == 1,
            "intact": intact,
            "cold_controller": cold,
            "shuffled_visible_history": shuffled,
            "intact_minus_cold": {
                "reward_advantage_auc":
                    intact["reward_advantage_auc"]
                    - cold["reward_advantage_auc"],
                "target_advantage_auc":
                    intact["target_advantage_auc"]
                    - cold["target_advantage_auc"],
            },
            "intact_minus_shuffled": {
                "reward_advantage_auc":
                    intact["reward_advantage_auc"]
                    - shuffled["reward_advantage_auc"],
                "target_advantage_auc":
                    intact["target_advantage_auc"]
                    - shuffled["target_advantage_auc"],
            },
        })

    all_metrics = [
        pair[arm]
        for pair in pairs
        for arm in (
            "intact", "cold_controller", "shuffled_visible_history")]
    empty = _metrics(args.empty) if args.empty else None
    gate = {
        "all_seeds_matched":
            all(pair["seed_matched"] for pair in pairs),
        "intact_history_beats_shuffled_reward_in_every_replica":
            all(
                pair["intact_minus_shuffled"]
                ["reward_advantage_auc"] > 0
                for pair in pairs),
        "intact_history_beats_shuffled_target_in_every_replica":
            all(
                pair["intact_minus_shuffled"]
                ["target_advantage_auc"] > 0
                for pair in pairs),
        "intact_controller_never_worse_than_cold":
            all(
                pair["intact_minus_cold"]["reward_advantage_auc"] >= 0
                and pair["intact_minus_cold"]["target_advantage_auc"] >= 0
                for pair in pairs),
        "intact_controller_strictly_beats_cold_in_at_least_one_replica":
            any(
                pair["intact_minus_cold"]["reward_advantage_auc"] > 0
                and pair["intact_minus_cold"]["target_advantage_auc"] > 0
                for pair in pairs),
        "all_old_skills_retained":
            all(
                item["binary_retained"] and item["four_rule_retained"]
                for item in all_metrics)
            and (
                empty is None
                or empty["binary_retained"] and empty["four_rule_retained"]),
    }
    gate["accepted"] = all(gate.values())
    report = {
        "schema": "unified-controller-gradual-transfer-audit-v1",
        "pairs": pairs,
        "empty_visible_history_supporting_arm": empty,
        "gate": gate,
        "conclusion": (
            "Correct accumulated physical history causally accelerates the "
            "related reliability transfer. Reusing learned controller "
            "weights helped one seed and tied one seed, so faster ignition "
            "from weight reuse is not yet independently replicated."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
