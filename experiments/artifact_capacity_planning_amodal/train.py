"""Audit learned admission planning before protected artifact growth."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from neural_computer import (
    CapabilityRetentionLedger,
    ExecutableArtifactMemory,
    MemoryCandidates,
    OpaqueCapacityPlanner,
    RetentionPolicyConfig,
)


def _state(
    *,
    seed: int,
    batch: int,
    capacity: int,
    width: int,
) -> tuple[
    MemoryCandidates,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if capacity < 2:
        raise ValueError("capacity planner pressure tests require capacity >= 2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(
        torch.randn(batch, capacity, width, generator=generator), dim=-1
    )
    values = torch.randn(batch, capacity, width, generator=generator)
    strengths = 0.1 + 0.9 * torch.rand(batch, capacity, generator=generator)
    timestamps = torch.rand(batch, capacity, generator=generator) * capacity
    occupied = torch.zeros(batch, capacity, dtype=torch.bool)
    protected = torch.zeros(batch, capacity, dtype=torch.bool)
    consolidation_available = torch.rand(batch, generator=generator) > 0.5
    for index in range(batch):
        count = (
            capacity
            if index % 2 == 0
            else 1 + int(torch.randint(capacity - 1, (1,), generator=generator))
        )
        occupied[index, :count] = True
        if count == capacity:
            mode = (index // 2) % 4
            if mode == 0:
                protected[index, :count] = True
                consolidation_available[index] = True
            elif mode == 1:
                protected[index, :count] = True
                consolidation_available[index] = False
            else:
                protected[index, 0] = False
                protected[index, 1:count] = True
                if mode == 2:
                    values[index, 1] = values[index, 0] + 0.01 * torch.randn(
                        width, generator=generator
                    )
                consolidation_available[index] = bool(
                    torch.randint(2, (1,), generator=generator)
                )
        else:
            protected[index, :count] = (
                torch.rand(count, generator=generator) > 0.5
            )
    bank = MemoryCandidates(
        keys=keys,
        values=values,
        strengths=strengths,
        timestamps=timestamps,
        occupied=occupied,
    )
    incoming_key = F.normalize(
        torch.randn(batch, width, generator=generator), dim=-1
    )
    incoming_value = torch.randn(batch, width, generator=generator)
    return bank, incoming_key, incoming_value, protected, consolidation_available


def _action_targets(
    bank: MemoryCandidates,
    protected: torch.Tensor,
    consolidation_available: torch.Tensor,
) -> torch.Tensor:
    count = bank.occupied.sum(dim=-1)
    has_free = count < bank.keys.shape[1]
    has_eviction = (bank.occupied & ~protected).any(dim=-1)
    has_consolidation = bank.occupied.sum(dim=-1) >= 2
    normalized_values = F.normalize(bank.values, dim=-1)
    similarity = normalized_values @ normalized_values.transpose(-1, -2)
    upper = torch.triu(
        torch.ones(bank.keys.shape[1], bank.keys.shape[1], dtype=torch.bool),
        diagonal=1,
    )
    valid_pairs = bank.occupied[:, :, None] & bank.occupied[:, None, :]
    best_similarity = similarity.masked_fill(
        ~upper | ~valid_pairs, -torch.inf
    ).amax(dim=(-1, -2))
    targets = torch.full_like(count, 3)
    targets = torch.where(has_consolidation & consolidation_available, 2, targets)
    targets = torch.where(has_eviction, 1, targets)
    targets = torch.where(
        has_eviction
        & has_consolidation
        & consolidation_available
        & (best_similarity >= 0.90),
        2,
        targets,
    )
    targets = torch.where(has_free, 0, targets)
    return targets


def _train_planner(
    *,
    seed: int,
    updates: int,
    batch: int,
    width: int,
    capacity: int,
    shuffle_utility: bool,
) -> tuple[OpaqueCapacityPlanner, dict[str, int | float]]:
    torch.manual_seed(seed)
    planner = OpaqueCapacityPlanner(width=width, hidden=64)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed + 99)
    last_loss = 0.0
    for update in range(updates):
        bank, incoming_key, incoming_value, protected, available = _state(
            seed=seed + update * 17,
            batch=batch,
            capacity=capacity,
            width=width,
        )
        targets = _action_targets(bank, protected, available)
        if shuffle_utility:
            targets = targets[torch.randperm(batch, generator=generator)]
        output = planner(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=available,
        )
        action_loss = F.cross_entropy(output.action_logits, targets)
        valid_evictions = output.valid_evictions.any(dim=-1)
        eviction_loss = torch.zeros((), dtype=torch.float32)
        if bool(valid_evictions.any()) and not shuffle_utility:
            disposable = bank.strengths.masked_fill(
                ~bank.occupied | protected, torch.inf
            )
            target_rows = disposable.argmin(dim=-1)
            rows = torch.arange(batch)[valid_evictions]
            eviction_loss = F.cross_entropy(
                output.eviction_scores[rows], target_rows[rows]
            )
        valid_pairs = output.valid_pairs.any(dim=(-1, -2))
        pair_loss = torch.zeros((), dtype=torch.float32)
        if bool(valid_pairs.any()) and not shuffle_utility:
            similarity = F.normalize(bank.values, dim=-1) @ F.normalize(
                bank.values, dim=-1
            ).transpose(-1, -2)
            upper = torch.triu(
                torch.ones(capacity, capacity, dtype=torch.bool), diagonal=1
            )
            pair_targets = similarity.masked_fill(
                ~upper | ~output.valid_pairs, -torch.inf
            ).reshape(batch, -1).argmax(dim=-1)
            rows = torch.arange(batch)[valid_pairs]
            pair_loss = F.cross_entropy(
                output.pair_scores[rows].reshape(rows.numel(), -1),
                pair_targets[rows],
            )
        loss = action_loss + 0.25 * eviction_loss + 0.25 * pair_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    planner.eval()
    return planner, {
        "optimizer_updates": updates,
        "unique_logical_lifetimes": updates * batch,
        "unique_verifier_bits": updates * batch * 4,
        "last_loss": last_loss,
    }


@torch.no_grad()
def _action_accuracy(
    planner: OpaqueCapacityPlanner,
    *,
    seed: int,
    batches: int,
    batch: int,
    width: int,
    capacity: int,
    ambiguous_only: bool = False,
) -> float:
    correct = 0
    total = 0
    for index in range(batches):
        bank, incoming_key, incoming_value, protected, available = _state(
            seed=seed + index * 17,
            batch=batch,
            capacity=capacity,
            width=width,
        )
        output = planner(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=available,
        )
        predictions = output.action_logits.masked_fill(
            ~output.available_actions, -torch.inf
        ).argmax(dim=-1)
        targets = _action_targets(bank, protected, available)
        mask = torch.ones(batch, dtype=torch.bool)
        if ambiguous_only:
            mask = (
                (bank.occupied.sum(dim=-1) == capacity)
                & (bank.occupied & ~protected).any(dim=-1)
                & available
            )
        correct += int(((predictions == targets) & mask).sum())
        total += int(mask.sum())
    if total == 0:
        raise RuntimeError("capacity planner audit contained no ambiguous states")
    return correct / total


def _artifact(value: float) -> dict[str, torch.Tensor]:
    return {
        "growth.weight": torch.tensor([[value, -value]], dtype=torch.float32),
        "growth.bias": torch.tensor([value], dtype=torch.float32),
    }


def _growth_audit(root: Path, *, width: int, retention_probes: int) -> dict[str, object]:
    for directory in (root / "source", root / "grown"):
        if directory.exists():
            shutil.rmtree(directory)
    ledger = CapabilityRetentionLedger(
        width,
        config=RetentionPolicyConfig(
            mastery_threshold=0.70,
            min_mastery_observations=retention_probes,
        ),
    )
    source = ExecutableArtifactMemory(
        root / "source",
        width=width,
        capacity=2,
        retention_ledger=ledger,
    )
    keys = [torch.eye(width)[index] for index in range(3)]
    for index in range(2):
        source.put(keys[index], _artifact(float(index + 1)))
        for _ in range(retention_probes):
            source.observe_retention(keys[index], 1.0)
    source_before = {
        "capacity": source.capacity,
        "occupied": list(source.occupied),
        "version": source.version,
        "paths": list(source.paths),
        "hashes": list(source.artifact_sha256),
    }
    planner = OpaqueCapacityPlanner(width=width, hidden=64).eval()
    candidates = source.planner_candidates()
    protected = torch.tensor(
        [[source._row_is_protected(index) for index in range(source.capacity)]],
        dtype=torch.bool,
    )
    plan = planner.propose(
        candidates,
        keys[-1].unsqueeze(0),
        torch.ones(1, width),
        protected,
        consolidation_available=torch.zeros(1, dtype=torch.bool),
    )
    if plan.action != "grow":
        raise RuntimeError(f"protected admission planner proposed {plan.action!r}")
    grown = source.grow(root / "grown", capacity=3)
    grown.put(keys[-1], _artifact(3.0))
    reloaded = ExecutableArtifactMemory.load(root / "grown")
    reloaded.validate()
    promoted = [reloaded.promote(key)[0].index for key in keys]
    source_after = {
        "capacity": source.capacity,
        "occupied": list(source.occupied),
        "version": source.version,
        "paths": list(source.paths),
        "hashes": list(source.artifact_sha256),
    }
    return {
        "planner_action": plan.action,
        "source_before": source_before,
        "source_after": source_after,
        "reloaded_promoted_indices": promoted,
        "retention_transferred": all(
            reloaded.retention.is_protected(key) for key in keys[:2]
        ),
        "new_artifact_admitted": len(grown.occupied) == 3,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    planner, accounting = _train_planner(
        seed=args.seed,
        updates=args.updates,
        batch=args.batch_size,
        width=args.width,
        capacity=args.capacity,
        shuffle_utility=False,
    )
    shuffled, shuffled_accounting = _train_planner(
        seed=args.seed + 10_000,
        updates=args.updates,
        batch=args.batch_size,
        width=args.width,
        capacity=args.capacity,
        shuffle_utility=True,
    )
    learned_accuracy = _action_accuracy(
        planner,
        seed=args.seed + 20_000,
        batches=args.audit_batches,
        batch=args.batch_size,
        width=args.width,
        capacity=args.capacity,
    )
    shuffled_accuracy = _action_accuracy(
        shuffled,
        seed=args.seed + 20_000,
        batches=args.audit_batches,
        batch=args.batch_size,
        width=args.width,
        capacity=args.capacity,
    )
    learned_ambiguous_accuracy = _action_accuracy(
        planner,
        seed=args.seed + 20_000,
        batches=args.audit_batches,
        batch=args.batch_size,
        width=args.width,
        capacity=args.capacity,
        ambiguous_only=True,
    )
    shuffled_ambiguous_accuracy = _action_accuracy(
        shuffled,
        seed=args.seed + 20_000,
        batches=args.audit_batches,
        batch=args.batch_size,
        width=args.width,
        capacity=args.capacity,
        ambiguous_only=True,
    )
    bank, incoming_key, incoming_value, protected, available = _state(
        seed=args.seed + 30_000,
        batch=1,
        capacity=args.capacity,
        width=args.width,
    )
    permutation = torch.randperm(args.capacity, generator=torch.Generator().manual_seed(args.seed + 31_000))
    permuted_bank = MemoryCandidates(
        keys=bank.keys[:, permutation],
        values=bank.values[:, permutation],
        strengths=bank.strengths[:, permutation],
        timestamps=bank.timestamps[:, permutation],
        occupied=bank.occupied[:, permutation],
    )
    original_plan = planner.propose(
        bank, incoming_key, incoming_value, protected, consolidation_available=available
    )
    permuted_plan = planner.propose(
        permuted_bank,
        incoming_key,
        incoming_value,
        protected[:, permutation],
        consolidation_available=available,
    )
    growth = _growth_audit(
        args.report_out.parent / "capacity_growth",
        width=args.width,
        retention_probes=args.retention_probes,
    )
    report: dict[str, object] = {
        "schema": "neural-computer.artifact-capacity-planning-report.v1",
        "claim_boundary": (
            "An outcome-trained opaque admission planner selects generic memory "
            "actions under capacity pressure; protection and behavior verification "
            "remain explicit transaction gates. This is not unrestricted memory "
            "growth, arbitrary computation, or general continual learning."
        ),
        "seed": args.seed,
        "width": args.width,
        "capacity": args.capacity,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learned_action_accuracy": learned_accuracy,
        "reward_shuffled_action_accuracy": shuffled_accuracy,
        "learned_ambiguous_action_accuracy": learned_ambiguous_accuracy,
        "reward_shuffled_ambiguous_action_accuracy": shuffled_ambiguous_accuracy,
        "permutation_plan": {
            "original_action": original_plan.action,
            "permuted_action": permuted_plan.action,
            "original_eviction_index": original_plan.eviction_index,
            "permuted_eviction_index": permuted_plan.eviction_index,
            "permutation": permutation.tolist(),
        },
        "protected_growth": growth,
        "accounting": {
            "optimizer_updates": accounting["optimizer_updates"]
            + shuffled_accounting["optimizer_updates"],
            "unique_logical_lifetimes": accounting["unique_logical_lifetimes"]
            + shuffled_accounting["unique_logical_lifetimes"],
            "unique_verifier_bits": accounting["unique_verifier_bits"]
            + shuffled_accounting["unique_verifier_bits"],
            "retention_observations": args.retention_probes * 2,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "learned_action_mastered": learned_accuracy >= 0.85,
            "learned_ambiguous_choice_mastered": learned_ambiguous_accuracy >= 0.80,
            "reward_shuffled_near_chance": shuffled_ambiguous_accuracy <= 0.60,
            "candidate_permutation_invariant": (
                original_plan.action == permuted_plan.action
            ),
            "full_protected_plans_growth": growth["planner_action"] == "grow",
            "source_immutable": growth["source_before"] == growth["source_after"],
            "retention_transferred": growth["retention_transferred"],
            "new_artifact_admitted": growth["new_artifact_admitted"],
            "reloaded_all_artifacts": growth["reloaded_promoted_indices"]
            == [0, 1, 2],
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audit-batches", type=int, default=8)
    parser.add_argument("--retention-probes", type=int, default=8)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "learned_action_accuracy": report["learned_action_accuracy"],
                "reward_shuffled_action_accuracy": report[
                    "reward_shuffled_action_accuracy"
                ],
                "learned_ambiguous_action_accuracy": report[
                    "learned_ambiguous_action_accuracy"
                ],
                "reward_shuffled_ambiguous_action_accuracy": report[
                    "reward_shuffled_ambiguous_action_accuracy"
                ],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
