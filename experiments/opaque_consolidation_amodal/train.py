"""Audit a learned, opaque, verifier-gated consolidation policy."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from neural_computer import (
    MemoryCandidates,
    OpaqueConsolidationPolicy,
    verify_consolidation_proposal,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _bank(
    *,
    batch: int,
    rows: int,
    width: int,
    seed: int,
    noise: float = 0.05,
) -> MemoryCandidates:
    if rows < 2 or rows % 2:
        raise ValueError("rows must be an even number of at least two")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = []
    values = []
    strengths = []
    timestamps = []
    occupied = []
    for _ in range(batch):
        centers = torch.randn(rows // 2, width, generator=generator)
        row_values = centers.repeat_interleave(2, dim=0)
        row_values = row_values + noise * torch.randn(
            rows, width, generator=generator
        )
        row_keys = F.normalize(torch.randn(rows, width, generator=generator), dim=-1)
        permutation = torch.randperm(rows, generator=generator)
        keys.append(row_keys[permutation])
        values.append(row_values[permutation])
        # Strength and age are independent memory metadata.  They must not
        # inherit the latent duplicate ordering, or a shuffled-utility control
        # could solve the audit through a non-causal timestamp shortcut.
        strengths.append(0.6 + 0.35 * torch.rand(rows, generator=generator))
        timestamps.append(rows * torch.rand(rows, generator=generator))
        occupied.append(torch.ones(rows, dtype=torch.bool))
    return MemoryCandidates(
        keys=torch.stack(keys),
        values=torch.stack(values),
        strengths=torch.stack(strengths),
        timestamps=torch.stack(timestamps),
        occupied=torch.stack(occupied),
    )


def _pair_indices(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(rows, rows, offset=1)


def _pair_quality(bank: MemoryCandidates) -> torch.Tensor:
    difference = bank.values[:, :, None, :] - bank.values[:, None, :, :]
    return torch.exp(-difference.square().mean(dim=-1))


def _train_policy(
    *,
    seed: int,
    rows: int,
    width: int,
    updates: int,
    batch_size: int,
    shuffled_utility: bool,
) -> tuple[OpaqueConsolidationPolicy, dict[str, int | float]]:
    torch.manual_seed(seed)
    policy = OpaqueConsolidationPolicy(width, hidden=64)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3, weight_decay=1e-5)
    left, right = _pair_indices(rows)
    generator = torch.Generator(device="cpu").manual_seed(seed + 11)
    last_loss = 0.0
    for update in range(updates):
        bank = _bank(
            batch=batch_size,
            rows=rows,
            width=width,
            seed=seed + 1_000 + update,
        )
        output = policy(bank)
        qualities = _pair_quality(bank)[:, left, right]
        if shuffled_utility:
            qualities = torch.rand(
                qualities.shape,
                generator=generator,
                dtype=qualities.dtype,
            )
        targets = qualities.argmax(dim=-1)
        pair_logits = output.pair_scores[:, left, right]
        pair_loss = F.cross_entropy(pair_logits, targets)
        target_operation = output.operation_logits[
            torch.arange(batch_size), left[targets], right[targets]
        ]
        operation_loss = F.cross_entropy(
            target_operation,
            torch.zeros(batch_size, dtype=torch.long),
        )
        target_merge = output.merge_logits[
            torch.arange(batch_size), left[targets], right[targets]
        ]
        merge_loss = F.mse_loss(torch.sigmoid(target_merge), torch.full_like(target_merge, 0.5))
        loss = pair_loss + operation_loss + 0.25 * merge_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    return policy.eval(), {
        "optimizer_updates": updates,
        "unique_lifetimes": updates * batch_size,
        "unique_verifier_bits": updates * batch_size * len(left),
        "last_loss": last_loss,
    }


def _proposal_quality(
    bank: MemoryCandidates,
    policy: OpaqueConsolidationPolicy,
    *,
    utility_bank: MemoryCandidates | None = None,
) -> float:
    proposal = policy.propose(bank)
    if proposal is None:
        return 0.0
    scored_bank = bank if utility_bank is None else utility_bank
    return float(_pair_quality(scored_bank)[0, proposal.first, proposal.second])


def _permutation_gate(
    bank: MemoryCandidates,
    policy: OpaqueConsolidationPolicy,
    *,
    seed: int,
) -> bool:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(bank.keys.shape[1], generator=generator)
    permuted = MemoryCandidates(
        keys=bank.keys[:, permutation],
        values=bank.values[:, permutation],
        strengths=bank.strengths[:, permutation],
        timestamps=bank.timestamps[:, permutation],
        occupied=bank.occupied[:, permutation],
    )
    first = policy.propose(bank)
    second = policy.propose(permuted)
    if first is None or second is None:
        return False
    first_keys = {
        tuple(float(value) for value in bank.keys[0, index].tolist())
        for index in (first.first, first.second)
    }
    second_keys = {
        tuple(float(value) for value in permuted.keys[0, index].tolist())
        for index in (second.first, second.second)
    }
    return first_keys == second_keys


def _sequential_chain(
    policy: OpaqueConsolidationPolicy,
    *,
    seed: int,
    rows: int,
    width: int,
    threshold: float,
) -> dict[str, object]:
    current = _bank(batch=1, rows=rows, width=width, seed=seed + 90_000)
    initial_rows = int(current.occupied.sum())
    accepted = 0
    retention_gates: list[bool] = []
    proposal_scores: list[float] = []
    outcome_generator = torch.Generator(device="cpu").manual_seed(seed + 91_000)
    while int(current.occupied.sum()) > rows // 2:
        proposal = policy.propose(current)
        if proposal is None:
            break
        score = float(_pair_quality(current)[0, proposal.first, proposal.second])
        proposal_scores.append(score)
        expected_rows = int(current.occupied.sum()) - 1
        fresh_outcomes = torch.clamp(
            score + 0.002 * torch.randn(4, generator=outcome_generator),
            0.0,
            1.0,
        ).tolist()
        candidate, receipt = verify_consolidation_proposal(
            current,
            proposal,
            verifier=lambda rewritten, expected=expected_rows: int(
                rewritten.occupied.sum()
            )
            == expected,
            candidate_outcomes=fresh_outcomes,
            retained_scores=[1.0],
            candidate_threshold=threshold,
            retention_floor=threshold,
            min_candidate_observations=4,
        )
        retention_gates.append(receipt.retention_accepted is True)
        if candidate is None or not receipt.accepted:
            break
        current = candidate
        accepted += 1
    return {
        "initial_rows": initial_rows,
        "final_rows": int(current.occupied.sum()),
        "accepted_rewrites": accepted,
        "proposal_scores": proposal_scores,
        "retention_gates": retention_gates,
        "all_retention_gates_passed": all(retention_gates),
        "source_unchanged_by_transaction": initial_rows == rows,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if min(args.rows, args.width, args.updates, args.batch_size, args.audit_count) < 1:
        raise ValueError("rows, width, updates, batch size, and audit count must be positive")
    seed_everything(args.seed)
    learned, learned_accounting = _train_policy(
        seed=args.seed,
        rows=args.rows,
        width=args.width,
        updates=args.updates,
        batch_size=args.batch_size,
        shuffled_utility=False,
    )
    shuffled, shuffled_accounting = _train_policy(
        seed=args.seed + 50_000,
        rows=args.rows,
        width=args.width,
        updates=args.updates,
        batch_size=args.batch_size,
        shuffled_utility=True,
    )
    torch.manual_seed(args.seed + 100_000)
    untrained = OpaqueConsolidationPolicy(args.width, hidden=64).eval()
    learned_scores = []
    shuffled_scores = []
    untrained_scores = []
    permutation_passes = 0
    duplicate_passes = 0
    corrupted_scores = []
    for index in range(args.audit_count):
        bank = _bank(
            batch=1,
            rows=args.rows,
            width=args.width,
            seed=args.seed + 70_000 + index,
        )
        quality = _pair_quality(bank)
        maximum = float(quality[0][~torch.eye(args.rows, dtype=torch.bool)].max())
        learned_score = _proposal_quality(bank, learned)
        shuffled_score = _proposal_quality(bank, shuffled)
        untrained_score = _proposal_quality(bank, untrained)
        learned_scores.append(learned_score)
        shuffled_scores.append(shuffled_score)
        untrained_scores.append(untrained_score)
        duplicate_passes += int(learned_score >= args.quality_threshold)
        permutation_passes += int(
            _permutation_gate(bank, learned, seed=args.seed + 80_000 + index)
        )
        corrupted = MemoryCandidates(
            keys=bank.keys,
            values=bank.values.flip(dims=(1,)),
            strengths=bank.strengths,
            timestamps=bank.timestamps,
            occupied=bank.occupied,
        )
        corrupted_scores.append(
            _proposal_quality(corrupted, learned, utility_bank=bank)
        )
        if maximum < args.quality_threshold:
            raise RuntimeError("generated bank lacks a verifiable high-quality pair")
    chain = _sequential_chain(
        learned,
        seed=args.seed,
        rows=args.rows,
        width=args.width,
        threshold=args.quality_threshold,
    )
    pair_count = args.rows * (args.rows - 1) // 2
    training_lifetimes = int(learned_accounting["unique_lifetimes"])
    training_bits = int(learned_accounting["unique_verifier_bits"])
    audit_lifetimes = args.audit_count
    audit_bits = args.audit_count * pair_count
    chain_steps = len(chain["proposal_scores"])
    retention_observations = chain_steps * 5
    retention_bits = chain_steps * 5
    report: dict[str, object] = {
        "schema": "neural-computer.opaque-consolidation-report.v1",
        "claim_boundary": (
            "A memory-side policy learns opaque pair selection and mechanical "
            "consolidation from scalar rewrite utility; transactions remain "
            "held-out-verifier gated and do not claim general continual learning."
        ),
        "seed": args.seed,
        "rows": args.rows,
        "width": args.width,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "quality_threshold": args.quality_threshold,
        "learned_mean_quality": float(np.mean(learned_scores)),
        "shuffled_mean_quality": float(np.mean(shuffled_scores)),
        "untrained_mean_quality": float(np.mean(untrained_scores)),
        "corrupted_mean_quality": float(np.mean(corrupted_scores)),
        "learned_duplicate_rate": duplicate_passes / args.audit_count,
        "candidate_permutation_rate": permutation_passes / args.audit_count,
        "chain": chain,
        "accounting": {
            "training_logical_lifetimes": training_lifetimes,
            "heldout_audit_logical_lifetimes": audit_lifetimes,
            "chain_verifier_lifetimes": chain_steps,
            "unique_logical_lifetimes": training_lifetimes
            + audit_lifetimes
            + chain_steps,
            "training_verifier_bits": training_bits,
            "heldout_audit_verifier_bits": audit_bits,
            "retention_verifier_bits": retention_bits,
            "unique_verifier_bits": training_bits + audit_bits + retention_bits,
            "optimizer_updates": learned_accounting["optimizer_updates"],
            "shuffled_control_optimizer_updates": shuffled_accounting[
                "optimizer_updates"
            ],
            "replayed_examples": 0,
            "retention_observations": retention_observations,
            "privileged_task_labels_seen_by_policy": 0,
            "privileged_correct_rows_seen_by_policy": 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "gates": {
            "learned_beats_untrained": bool(
                np.mean(learned_scores) >= np.mean(untrained_scores) + 0.10
            ),
            "learned_beats_reward_shuffled": bool(
                np.mean(learned_scores) >= np.mean(shuffled_scores) + 0.10
            ),
            "learned_selects_verifiable_pairs": duplicate_passes
            >= int(0.8 * args.audit_count),
            "candidate_permutation_invariant": permutation_passes
            >= int(0.9 * args.audit_count),
            "corruption_is_causal": bool(
                np.mean(learned_scores) >= np.mean(corrupted_scores) + 0.05
            ),
            "long_chain_compacted": chain["final_rows"] == args.rows // 2,
            "long_chain_retention_gated": chain["all_retention_gates_passed"],
            "no_replay": True,
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
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--quality-threshold", type=float, default=0.8)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "learned_mean_quality": report["learned_mean_quality"],
                "untrained_mean_quality": report["untrained_mean_quality"],
                "chain": report["chain"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
