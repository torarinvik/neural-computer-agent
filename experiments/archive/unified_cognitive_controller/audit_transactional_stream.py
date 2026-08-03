"""Audit verifier-gated memory updates across a persistent task stream.

This is the next rung after the one-update transaction audit.  Each arm keeps
the same physical banks alive while candidate skills arrive from several
independent verifier-generated streams.  Accepted candidates become old-skill
verifiers for later updates; rejected candidates do not.  The adversarial arm
deliberately targets a stable row so the safety gate has a causal control.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_receipt_volatility_controller import _physical_volatility
from .audit_transactional_plasticity import _history_memory, _score
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .train_controller_memory_volatility import volatility_batch
from .train_memory_replacement import _select_batch


def _load_model(checkpoint: Path, device: torch.device) -> UnifiedCognitiveController:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if model.adaptive_memory_replace_features != 8:
        raise ValueError("transactional stream requires the eight-feature head")
    return model


@torch.no_grad()
def _stream_data(
        model: UnifiedCognitiveController, *, rounds: int, banks: int,
        capacity: int, seed: int, device: torch.device) -> list[dict[str, object]]:
    result = []
    for round_index in range(rounds):
        data = volatility_batch(
            model, banks=banks, capacity=capacity,
            seed=seed + round_index * 100_003, device=device,
            write_threshold=0.5)
        volatility = _physical_volatility(
            model, data, policy="receipt", device=device)
        options = data["option_features"].clone()
        options[:, 1:, 7] = volatility
        data["physical_volatility"] = volatility
        data["learned_actions"] = model.memory_replacement_scores(
            options).argmax(-1)
        data["adversarial_actions"] = (
            data["stable_mask"].to(torch.float32).argmax(-1) + 1)
        result.append(data)
    return result


@torch.no_grad()
def run_audit(
        checkpoint: Path, *, seed: int, rounds: int, banks: int,
        capacity: int, device: torch.device) -> dict[str, object]:
    model = _load_model(checkpoint, device)
    stream = _stream_data(
        model, rounds=rounds, banks=banks, capacity=capacity,
        seed=seed, device=device)
    horizon = capacity // 2 + 1
    old_count = capacity // 2
    rows: list[dict[str, object]] = []
    final_memories: list[tuple[str, DiskLatentMemory, list]] = []

    for arm in ("learned", "adversarial"):
        for bank in range(banks):
            memory = _history_memory(model, stream[0], bank, device=device)
            initial_start = bank * horizon
            initial_indices = torch.arange(
                initial_start, initial_start + old_count, device=device,
                dtype=torch.long)
            initial_batch = _select_batch(
                stream[0]["future_batch"], initial_indices)
            initial_queries = stream[0]["future_queries"][bank, :old_count]

            def initial_verifier(
                    store: DiskLatentMemory,
                    batch=initial_batch,
                    queries=initial_queries) -> float:
                return _score(model, store, batch, queries, device=device)

            old_verifiers = [initial_verifier]
            for round_index, data in enumerate(stream):
                actions = data[f"{arm}_actions"]
                action = int(actions[bank])
                if action == 0:
                    rows.append({
                        "arm": arm, "bank": bank, "round": round_index,
                        "action": 0, "skipped": True,
                    })
                    continue
                start = bank * horizon
                candidate_index = torch.tensor(
                    [start + old_count], device=device)
                candidate_batch = _select_batch(
                    data["future_batch"], candidate_index)
                candidate_queries = data["future_queries"][
                    bank, old_count:old_count + 1]

                def candidate_verifier(
                        store: DiskLatentMemory,
                        batch=candidate_batch,
                        queries=candidate_queries) -> float:
                    return _score(
                        model, store, batch, queries,
                        device=device)

                before_old = [verifier(memory) for verifier in old_verifiers]
                before_candidate = candidate_verifier(memory)
                unguarded = memory.clone()
                unguarded.elastic_replace(
                    action - 1, data["candidate_key"][bank],
                    data["candidate_value"][bank],
                    data["candidate_strength"][bank])
                unguarded_old = [verifier(unguarded)
                                 for verifier in old_verifiers]
                result = memory.transactional_replace(
                    action - 1, data["candidate_key"][bank],
                    data["candidate_value"][bank],
                    data["candidate_strength"][bank],
                    [
                        (lambda store, verifier=verifier: verifier(store))
                        for verifier in old_verifiers
                    ],
                    candidate_verifier, required_candidate_gain=0.0,
                    rejection_penalty=0.01)
                after_old = [verifier(result.memory)
                             for verifier in old_verifiers]
                after_candidate = candidate_verifier(result.memory)
                unguarded_forgets = any(
                    after < before - 1e-7
                    for before, after in zip(before_old, unguarded_old))
                guarded_forgets = any(
                    after < before - 1e-7
                    for before, after in zip(before_old, after_old))
                rows.append({
                    "arm": arm, "bank": bank, "round": round_index,
                    "action": action, "committed": result.committed,
                    "old_task_count_before": len(old_verifiers),
                    "old_retention_before": min(before_old, default=1.0),
                    "old_retention_after": min(after_old, default=1.0),
                    "before_candidate": before_candidate,
                    "after_candidate": after_candidate,
                    "candidate_gain": result.candidate_gain,
                    "unguarded_forgets": unguarded_forgets,
                    "guarded_forgets": guarded_forgets,
                })
                if result.committed:
                    memory = result.memory
                    old_verifiers.append(candidate_verifier)
            final_memories.append((arm, memory, old_verifiers))

    attempted = [row for row in rows if not row.get("skipped")]
    summaries = {}
    for arm in ("learned", "adversarial"):
        arm_rows = [row for row in attempted if row["arm"] == arm]
        commits = [row for row in arm_rows if row["committed"]]
        summaries[arm] = {
            "proposals": len(arm_rows),
            "commits": len(commits),
            "rollbacks": len(arm_rows) - len(commits),
            "strict_positive_commits": sum(
                float(row["candidate_gain"]) > 0.0 for row in commits),
            "nonnegative_commits": sum(
                float(row["candidate_gain"]) >= 0.0 for row in commits),
            "unguarded_forgetting": sum(
                bool(row["unguarded_forgets"]) for row in arm_rows),
            "guarded_forgetting": sum(
                bool(row["guarded_forgets"]) for row in arm_rows),
            "max_old_task_count": max(
                (int(row["old_task_count_before"]) for row in arm_rows),
                default=0),
            "mean_candidate_gain": (
                sum(float(row["candidate_gain"]) for row in commits)
                / max(1, len(commits))),
        }
    corruption = {"learned": [], "adversarial": []}
    for arm, memory, verifiers in final_memories:
        corrupted = memory.clone()
        if corrupted.count > 1:
            corrupted.store.values.copy_(torch.roll(
                corrupted.store.values, shifts=1, dims=0))
        clean = [verifier(memory) for verifier in verifiers]
        damaged = [verifier(corrupted) for verifier in verifiers]
        corruption[arm].append({
            "clean_min": min(clean, default=1.0),
            "corrupted_min": min(damaged, default=1.0),
            "drop": max(0.0, min(clean, default=1.0)
                        - min(damaged, default=1.0)),
        })
    corruption_summary = {
        arm: {
            "banks": len(rows_for_arm),
            "banks_with_drop": sum(row["drop"] > 1e-7
                                    for row in rows_for_arm),
            "mean_drop": sum(row["drop"] for row in rows_for_arm)
            / max(1, len(rows_for_arm)),
        }
        for arm, rows_for_arm in corruption.items()
    }
    disk_exact = True
    with tempfile.TemporaryDirectory() as temporary:
        for index, (_, memory, _) in enumerate(final_memories):
            path = Path(temporary) / f"stream-{index}.pt"
            memory.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            disk_exact = disk_exact and torch.equal(
                restored.store.keys, memory.store.keys)
            disk_exact = disk_exact and torch.equal(
                restored.store.values, memory.store.values)
            disk_exact = disk_exact and torch.equal(
                restored.store.volatility, memory.store.volatility)
    learned = summaries["learned"]
    adversarial = summaries["adversarial"]
    report = {
        "schema": "transactional-plasticity-stream-audit-v1",
        "checkpoint": str(checkpoint),
        "configuration": {
            "seed": seed, "rounds": rounds, "banks": banks,
            "capacity": capacity, "device": str(device),
        },
        "semantic_or_task_labels_used_for_training": False,
        "same_physical_banks_across_rounds": True,
        "summary": summaries,
        "accounting": {
            "proposals": len(attempted),
            "rounds": rounds,
            "total_seconds": None,
        },
        "disk_round_trip_exact": disk_exact,
        "corruption": corruption_summary,
        "rows": rows,
        "gates": {
            "learned_arm_attempted": learned["proposals"] > 0,
            "learned_arm_commits_across_multiple_rounds": (
                learned["commits"] >= banks * 2),
            "learned_arm_no_guarded_forgetting": (
                learned["guarded_forgetting"] == 0),
            "adversarial_arm_causes_unguarded_regression": (
                adversarial["unguarded_forgetting"] > 0),
            "adversarial_arm_rolls_back_regressions": (
                adversarial["rollbacks"] >= adversarial["unguarded_forgetting"]
                and adversarial["rollbacks"] > 0),
            "adversarial_arm_no_guarded_forgetting": (
                adversarial["guarded_forgetting"] == 0),
            "learned_memory_corruption_degrades_retained_skill": (
                corruption_summary["learned"]["banks_with_drop"] > 0),
            "disk_round_trip_exact": disk_exact,
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19200)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.rounds < 2 or args.banks < 2 or args.capacity < 4 or args.capacity % 2:
        raise ValueError("rounds>=2, banks>=2, and even capacity>=4 required")
    started = time.perf_counter()
    report = run_audit(
        args.checkpoint, seed=args.seed, rounds=args.rounds,
        banks=args.banks, capacity=args.capacity,
        device=torch.device(args.device))
    report["accounting"]["total_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "summary": report["summary"],
        "accounting": report["accounting"],
        "gates": report["gates"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
