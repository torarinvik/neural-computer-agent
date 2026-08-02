"""Tiny verifier-driven commit/rollback probe for long-term memory updates."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from .memory import DiskLatentMemory


def _similarity(
        memory: DiskLatentMemory, query: torch.Tensor,
        expected: torch.Tensor) -> float:
    read, _ = memory.retrieve(
        query.unsqueeze(0), top_k=1, confidence_mode="cosine",
        usage_prior_scale=0.0)
    return float(torch.nn.functional.cosine_similarity(
        read, expected.unsqueeze(0)).item())


def run_probe() -> dict[str, object]:
    old_keys = torch.eye(4)[:2]
    decoy_key = torch.tensor([0.0, 0.0, 0.0, 1.0])
    new_key = torch.tensor([0.0, 0.0, 1.0, 0.0])

    def make_memory() -> DiskLatentMemory:
        memory = DiskLatentMemory(width=4, capacity=3)
        memory.commit(
            torch.stack((old_keys[0], old_keys[1], decoy_key)),
            torch.stack((old_keys[0], old_keys[1], decoy_key)),
            torch.ones(3), threshold=0.0)
        return memory

    def old_verifier(store: DiskLatentMemory) -> float:
        return sum(
            _similarity(store, key, key) for key in old_keys) / len(old_keys)

    def new_verifier(store: DiskLatentMemory) -> float:
        return _similarity(store, new_key, new_key)

    harmful = make_memory()
    rejected = harmful.transactional_replace(
        0, new_key, new_key, 1.0, [old_verifier], new_verifier,
        required_candidate_gain=0.5, rejection_penalty=0.25)

    safe = make_memory()
    accepted = safe.transactional_replace(
        2, new_key, new_key, 1.0, [old_verifier], new_verifier,
        required_candidate_gain=0.5)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "committed.pt"
        accepted.memory.save(path)
        restored = DiskLatentMemory.load(path)
        disk_round_trip_exact = (
            torch.equal(restored.store.keys, accepted.memory.store.keys)
            and torch.equal(restored.store.values, accepted.memory.store.values)
            and torch.equal(
                restored.store.volatility, accepted.memory.store.volatility))

    result = {
        "schema": "memory-transaction-probe-v1",
        "learner_visible": ["latent keys", "latent values", "verifier scores"],
        "semantic_task_labels_visible_to_memory": False,
        "rejected_harmful_update": {
            "committed": rejected.committed,
            "before_retention": rejected.before_retention,
            "after_retention": rejected.after_retention,
            "candidate_gain": rejected.candidate_gain,
            "maximum_retention_drop": rejected.maximum_retention_drop,
            "rollback_exact": torch.equal(
                rejected.memory.store.keys, harmful.store.keys)
                and torch.equal(
                    rejected.memory.store.values, harmful.store.values),
        },
        "accepted_safe_update": {
            "committed": accepted.committed,
            "before_retention": accepted.before_retention,
            "after_retention": accepted.after_retention,
            "candidate_gain": accepted.candidate_gain,
            "maximum_retention_drop": accepted.maximum_retention_drop,
            "disk_round_trip_exact": disk_round_trip_exact,
        },
    }
    result["gates"] = {
        "harmful_update_rejected": not rejected.committed,
        "harmful_update_rolled_back_exactly": result[
            "rejected_harmful_update"]["rollback_exact"],
        "safe_update_committed": accepted.committed,
        "safe_update_has_positive_candidate_gain":
            accepted.candidate_gain >= 0.5,
        "safe_update_preserves_old_skills":
            accepted.maximum_retention_drop == 0.0,
        "committed_state_persists_exactly": disk_round_trip_exact,
    }
    result["gates"]["accepted"] = all(result["gates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_probe()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
