"""Integrate the learned plasticity head with verifier-gated transactions."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_selective_disk import _gated_retrieve
from .audit_receipt_volatility_controller import _physical_volatility
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .train_adaptive_memory_read import _outcomes
from .train_controller_memory_volatility import volatility_batch
from .train_memory_replacement import _select_batch


@torch.no_grad()
def _history_memory(
        model: UnifiedCognitiveController, data: dict[str, object], bank: int,
        *, device: torch.device) -> DiskLatentMemory:
    """Build one physical bank and apply its receipt-attributed history."""
    capacity = data["bank_keys"].shape[1]
    memory = DiskLatentMemory(model.width, capacity=capacity, device=device)
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
        _, _, receipts = memory.retrieve_with_receipt(
            queries, top_k=1, confidence_mode="cosine",
            usage_prior_scale=0.0)
        memory.record_outcomes_from_receipts(
            receipts, outcomes, update_volatility=True,
            success_protection_rate=0.2, failure_thaw_rate=0.25,
            stale_thaw_rate=0.0)
    return memory


@torch.no_grad()
def _score(
        model: UnifiedCognitiveController, memory: DiskLatentMemory,
        batch, queries: torch.Tensor, *, device: torch.device) -> float:
    retrieved = _gated_retrieve(
        model, memory, queries, read_threshold=None,
        usage_prior_scale=model.effective_memory_usage_prior_scale())
    return float(_outcomes(model, batch, retrieved, device=device).mean())


@torch.no_grad()
def run_audit(
        checkpoint: Path, *, seed: int, banks: int, capacity: int,
        device: torch.device) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if model.adaptive_memory_replace_features != 8:
        raise ValueError("transactional audit requires the eight-feature head")
    data = volatility_batch(
        model, banks=banks, capacity=capacity, seed=seed, device=device,
        write_threshold=0.5)
    volatility = _physical_volatility(
        model, data, policy="receipt", device=device)
    options = data["option_features"].clone()
    options[:, 1:, 7] = volatility
    actions = model.memory_replacement_scores(options).argmax(-1)
    # Verifier-side adversarial control: deliberately target a stable row.
    # This is not a learner-visible label or training signal.
    adversarial_actions = data["stable_mask"].to(torch.float32).argmax(-1) + 1
    horizon = capacity // 2 + 1
    old_count = capacity // 2
    rows: list[dict[str, object]] = []
    committed_memories: list[DiskLatentMemory] = []
    proposal_actions = {
        "learned": actions,
        "adversarial": adversarial_actions,
    }
    for arm, arm_actions in proposal_actions.items():
        for bank in range(banks):
            action = int(arm_actions[bank])
            if action == 0:
                rows.append({"arm": arm, "bank": bank, "action": 0,
                             "skipped": True})
                continue
            # Rebuild from the same pre-update state for each arm so the
            # adversarial control cannot contaminate the learned arm.
            memory = _history_memory(model, data, bank, device=device)
            start = bank * horizon
            old_indices = torch.arange(
                start, start + old_count, device=device, dtype=torch.long)
            candidate_index = torch.tensor([start + old_count], device=device)
            old_batch = _select_batch(data["future_batch"], old_indices)
            candidate_batch = _select_batch(data["future_batch"], candidate_index)
            old_queries = data["future_queries"][bank, :old_count]
            candidate_queries = data["future_queries"][bank, old_count:old_count + 1]

            def old_verifier(store: DiskLatentMemory) -> float:
                return _score(model, store, old_batch, old_queries, device=device)

            def candidate_verifier(store: DiskLatentMemory) -> float:
                return _score(
                    model, store, candidate_batch, candidate_queries,
                    device=device)

            unguarded = memory.clone()
            unguarded.elastic_replace(
                action - 1, data["candidate_key"][bank],
                data["candidate_value"][bank],
                data["candidate_strength"][bank])
            unguarded_old = old_verifier(unguarded)
            unguarded_candidate = candidate_verifier(unguarded)
            result = memory.transactional_replace(
                action - 1, data["candidate_key"][bank],
                data["candidate_value"][bank],
                data["candidate_strength"][bank],
                [old_verifier], candidate_verifier,
                required_candidate_gain=0.0,
                rejection_penalty=0.01)
            guarded_old = old_verifier(result.memory)
            guarded_candidate = candidate_verifier(result.memory)
            rows.append({
                "arm": arm,
                "bank": bank,
                "action": action,
                "committed": result.committed,
                "before_old": result.before_retention[0],
                "after_old": guarded_old,
                "before_candidate": result.before_candidate,
                "after_candidate": guarded_candidate,
                "candidate_gain": result.candidate_gain,
                "unguarded_old": unguarded_old,
                "unguarded_candidate": unguarded_candidate,
                "unguarded_forgets":
                    unguarded_old < result.before_retention[0],
                "guarded_forgets":
                    guarded_old < result.before_retention[0],
            })
            if result.committed:
                committed_memories.append(result.memory)

    attempted = [row for row in rows if not row.get("skipped")]
    committed = [row for row in attempted if row["committed"]]
    summaries: dict[str, dict[str, object]] = {}
    for arm in proposal_actions:
        arm_attempted = [row for row in attempted if row["arm"] == arm]
        arm_committed = [row for row in arm_attempted if row["committed"]]
        arm_unguarded = [row for row in arm_attempted
                         if row["unguarded_forgets"]]
        arm_guarded = [row for row in arm_attempted
                       if row["guarded_forgets"]]
        arm_positive = [row for row in arm_committed
                        if float(row["candidate_gain"]) >= 0.0]
        summaries[arm] = {
            "proposals": len(arm_attempted),
            "skips": banks - len(arm_attempted),
            "commits": len(arm_committed),
            "rollbacks": len(arm_attempted) - len(arm_committed),
            "unguarded_forgetting": len(arm_unguarded),
            "guarded_forgetting": len(arm_guarded),
            "positive_commits": len(arm_positive),
            "mean_committed_candidate_gain": (
                sum(float(row["candidate_gain"]) for row in arm_positive)
                / max(1, len(arm_positive))),
        }
    disk_exact = True
    if committed_memories:
        with tempfile.TemporaryDirectory() as temporary:
            for index, memory in enumerate(committed_memories[:8]):
                path = Path(temporary) / f"commit-{index}.pt"
                memory.save(path)
                restored = DiskLatentMemory.load(path, device=device)
                disk_exact = disk_exact and torch.equal(
                    restored.store.keys, memory.store.keys)
                disk_exact = disk_exact and torch.equal(
                    restored.store.values, memory.store.values)
                disk_exact = disk_exact and torch.equal(
                    restored.store.volatility, memory.store.volatility)
    return {
        "schema": "transactional-plasticity-audit-v1",
        "checkpoint": str(checkpoint),
        "configuration": {
            "seed": seed, "banks": banks, "capacity": capacity,
            "device": str(device),
        },
        "semantic_or_task_labels_used_for_training": False,
        "learner_visible": [
            "latent keys and values", "physical receipt history",
            "volatility", "verifier scores"],
        "accounting": {
            "proposals": len(attempted),
            "skips": sum(bool(row.get("skipped")) for row in rows),
            "commits": len(committed),
            "rollbacks": len(attempted) - len(committed),
            "per_arm": summaries,
        },
        "summary": summaries,
        "rows": rows,
        "disk_round_trip_exact": disk_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19100)
    parser.add_argument("--banks", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.banks < 2 or args.capacity < 4 or args.capacity % 2:
        raise ValueError("banks must be >=2 and capacity even >=4")
    started = time.perf_counter()
    report = run_audit(
        args.checkpoint, seed=args.seed, banks=args.banks,
        capacity=args.capacity, device=torch.device(args.device))
    report["accounting"]["total_seconds"] = time.perf_counter() - started
    attempted = report["accounting"]["proposals"]
    committed = report["accounting"]["commits"]
    learned = report["summary"]["learned"]
    adversarial = report["summary"]["adversarial"]
    report["gates"] = {
        "learned_proposals_were_attempted": learned["proposals"] > 0,
        "adversarial_proposals_were_attempted": adversarial["proposals"] > 0,
        "transaction_rejected_at_least_one_forgetting_update":
            adversarial["unguarded_forgetting"] > 0
            and adversarial["rollbacks"] > 0,
        "no_guarded_update_forgets_old_skill": (
            learned["guarded_forgetting"] == 0
            and adversarial["guarded_forgetting"] == 0),
        "learned_arm_has_positive_commit": learned["positive_commits"] > 0,
        "adversarial_arm_has_rollback": adversarial["rollbacks"] > 0,
        "committed_memory_round_trip_exact": report["disk_round_trip_exact"],
        "under_five_minute_cap":
            report["accounting"]["total_seconds"] <= 300.0,
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "accounting": report["accounting"],
        "summary": report["summary"],
        "gates": report["gates"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
