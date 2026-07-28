"""Causal atom for verified-use memory volatility.

The learner-facing state is deliberately generic: latent keys/values, access
counts, and scalar verifier outcomes.  The private audit labels rows only so we
can measure whether useful memories survived a non-stationary replacement
phase.  No semantic task identity or correct action is exposed to the memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .memory import DiskLatentMemory


def _orthogonal_rows(count: int, width: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(width, count, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q.T


def _is_recalled(
        memory: DiskLatentMemory, key: torch.Tensor, value: torch.Tensor) -> bool:
    recalled, _ = memory.retrieve(
        key.unsqueeze(0), top_k=1, confidence_mode="cosine")
    return bool(torch.nn.functional.cosine_similarity(
        recalled, value.unsqueeze(0)).item() > 0.95)


def _access_only_volatility(memory: DiskLatentMemory) -> None:
    valid = memory.store.valid
    access = memory.store.access_count[valid].to(memory.store.volatility.dtype)
    memory.store.volatility[valid] = 1.0 / (1.0 + access)


def _prepare_trial(
        seed: int, policy: str
        ) -> tuple[
            DiskLatentMemory, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Generator]:
    if policy not in {"uniform", "access", "verified", "shuffled_verified"}:
        raise ValueError("unknown volatility policy")
    generator = torch.Generator().manual_seed(seed)
    capacity, width, incoming = 8, 16, 4
    rows = _orthogonal_rows(capacity + incoming, width, generator)
    values = _orthogonal_rows(capacity + incoming, width, generator)
    permutation = torch.randperm(capacity, generator=generator)
    stable = permutation[:3]
    decoy = permutation[3:6]
    stale = permutation[6:]

    memory = DiskLatentMemory(width=width, capacity=capacity)
    memory.commit(
        rows[:capacity], values[:capacity], torch.ones(capacity),
        threshold=0.0)

    # Stale rows were once useful, then disappear from experience.
    memory.store.record_outcomes(
        rows[stale].repeat(3, 1), torch.ones(stale.numel() * 3),
        update_volatility=policy in {"verified", "shuffled_verified"},
        success_protection_rate=0.12, failure_thaw_rate=0.25,
        stale_thaw_rate=0.0)
    # Stable and decoy rows are equally frequent. Only verifier outcomes
    # distinguish genuinely useful retrievals from a frequently accessed trap.
    for _ in range(24):
        queried = torch.cat((rows[stable], rows[decoy]))
        memory.retrieve(
            queried, top_k=1, confidence_mode="cosine",
            record_access=True)
        outcomes = torch.cat((torch.ones(3), torch.zeros(3)))
        memory.store.record_outcomes(
            queried, outcomes,
            update_volatility=policy in {"verified", "shuffled_verified"},
            success_protection_rate=0.12, failure_thaw_rate=0.25,
            stale_thaw_rate=0.08)

    if policy == "uniform":
        memory.store.volatility[memory.store.valid] = 1.0
    elif policy == "access":
        _access_only_volatility(memory)
    elif policy == "shuffled_verified":
        valid = memory.store.valid.nonzero(as_tuple=False).squeeze(1)
        shuffled = valid[torch.randperm(valid.numel(), generator=generator)]
        memory.store.volatility[valid] = memory.store.volatility[shuffled].clone()
    return memory, rows, values, stable, decoy, stale, generator


def run_trial(seed: int, policy: str) -> dict[str, float | int | str]:
    (
        memory, rows, values, stable, decoy, stale, generator,
    ) = _prepare_trial(seed, policy)
    capacity, incoming = 8, 4

    volatility_before = memory.store.volatility.clone()
    eligible = set(range(capacity))
    rewritten: list[int] = []
    for offset in range(incoming):
        candidates = torch.tensor(sorted(eligible))
        candidate_volatility = memory.store.volatility[candidates]
        best = candidate_volatility.max()
        tied = candidates[candidate_volatility == best]
        chosen = int(tied[torch.randint(
            tied.numel(), (), generator=generator)])
        memory.elastic_replace(
            chosen, rows[capacity + offset], values[capacity + offset], 1.0)
        rewritten.append(chosen)
        eligible.remove(chosen)

    stable_retained = sum(
        _is_recalled(memory, rows[index], values[index])
        for index in stable.tolist())
    new_acquired = sum(
        _is_recalled(memory, rows[capacity + offset], values[capacity + offset])
        for offset in range(incoming))
    decoys_rewritten = len(set(rewritten).intersection(decoy.tolist()))
    score = (stable_retained + new_acquired) / (stable.numel() + incoming)
    return {
        "seed": seed,
        "policy": policy,
        "score": score,
        "stable_retained": stable_retained / stable.numel(),
        "new_acquired": new_acquired / incoming,
        "decoys_rewritten": decoys_rewritten / decoy.numel(),
        "stable_mean_volatility": float(volatility_before[stable].mean()),
        "decoy_mean_volatility": float(volatility_before[decoy].mean()),
        "stale_mean_volatility": float(volatility_before[stale].mean()),
    }


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    policies = sorted({str(row["policy"]) for row in rows})
    summary: dict[str, object] = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            key: sum(float(row[key]) for row in selected) / len(selected)
            for key in (
                "score", "stable_retained", "new_acquired",
                "decoys_rewritten", "stable_mean_volatility",
                "decoy_mean_volatility", "stale_mean_volatility")
        }
    verified = summary["verified"]
    assert isinstance(verified, dict)
    controls = [
        summary[name]["score"]  # type: ignore[index]
        for name in ("uniform", "access", "shuffled_verified")
    ]
    return {
        "by_policy": summary,
        "gates": {
            "verified_retains_at_least_95_percent_stable":
                verified["stable_retained"] >= 0.95,
            "verified_acquires_at_least_95_percent_new":
                verified["new_acquired"] >= 0.95,
            "verified_beats_every_control_by_10_points":
                verified["score"] >= max(controls) + 0.10,
            "verified_protects_useful_more_than_frequent_failures":
                verified["stable_mean_volatility"]
                < verified["decoy_mean_volatility"] * 0.5,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = [
        run_trial(args.seed + offset, policy)
        for offset in range(args.seeds)
        for policy in ("uniform", "access", "verified", "shuffled_verified")
    ]
    report = {
        "schema": "verified-memory-volatility-probe-v1",
        "experience": {
            "semantic_task_ids_visible_to_memory": False,
            "correct_actions_visible_to_memory": False,
            "generic_signals": [
                "latent key", "latent value", "access", "scalar verifier outcome"],
        },
        "trials": rows,
        **summarize(rows),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_policy"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
