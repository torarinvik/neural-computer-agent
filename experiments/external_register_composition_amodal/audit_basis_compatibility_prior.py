"""Audit opaque compatibility screening against a scalar verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_computer import (
    ExternalRegisterBasisCompatibilityPrior,
    ExternalRegisterComputeBasis,
)


def _verifier(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Frozen environment: return scalar outcomes, never semantic labels."""

    normalized_query = torch.nn.functional.normalize(query, dim=-1)
    normalized_keys = torch.nn.functional.normalize(keys, dim=-1)
    similarity = torch.einsum("bd,kd->bk", normalized_query, normalized_keys)
    return torch.sigmoid(4.0 * similarity)


def _trial_count(
    order: tuple[int, ...],
    outcomes: torch.Tensor,
    *,
    threshold: float,
) -> tuple[int, bool]:
    for count, index in enumerate(order, start=1):
        if float(outcomes[index].detach()) >= threshold:
            return count, True
    return len(order), False


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.basis_count, args.audit_queries, args.updates) < 1:
        raise ValueError("counts must be positive")
    torch.manual_seed(args.seed)
    basis_slots = tuple(
        ExternalRegisterComputeBasis(args.width, args.width, hidden=args.hidden)
        for _ in range(args.basis_count)
    )
    with torch.no_grad():
        for slot in basis_slots:
            slot.signature.copy_(torch.randn(args.width))
    keys = ExternalRegisterBasisCompatibilityPrior.basis_keys(basis_slots)
    prior = ExternalRegisterBasisCompatibilityPrior(
        args.width,
        latent_width=args.latent_width,
        hidden=args.hidden,
    )
    prior.enable()
    optimizer = torch.optim.AdamW(prior.parameters(), lr=args.learning_rate)
    for update in range(args.updates):
        queries = torch.randn(args.batch_size, args.width)
        outcomes = _verifier(queries, keys)
        loss, informative = prior.outcome_ranking_loss(queries, keys, outcomes)
        if informative == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    query_batch = torch.randn(args.audit_queries, args.width)
    outcome_batch = _verifier(query_batch, keys)
    learned_counts: list[int] = []
    cold_counts: list[int] = []
    learned_passes = 0
    cold_passes = 0
    for query_index in range(args.audit_queries):
        permutation = torch.randperm(args.basis_count)
        permuted_keys = keys[permutation]
        learned_order_in_permutation = prior(
            query_batch[query_index : query_index + 1],
            permuted_keys,
        )[0].argsort(descending=True, stable=True)
        learned_order = tuple(
            int(permutation[index]) for index in learned_order_in_permutation
        )
        cold_order = tuple(int(index) for index in permutation)
        learned_count, learned_pass = _trial_count(
            learned_order,
            outcome_batch[query_index],
            threshold=args.threshold,
        )
        cold_count, cold_pass = _trial_count(
            cold_order,
            outcome_batch[query_index],
            threshold=args.threshold,
        )
        learned_counts.append(learned_count)
        cold_counts.append(cold_count)
        learned_passes += int(learned_pass)
        cold_passes += int(cold_pass)
    learned_mean = sum(learned_counts) / len(learned_counts)
    cold_mean = sum(cold_counts) / len(cold_counts)
    admissible_indices = [
        index
        for index, count in enumerate(cold_counts)
        if float(outcome_batch[index].max().detach()) >= args.threshold
    ]
    learned_admissible_mean = (
        sum(learned_counts[index] for index in admissible_indices)
        / len(admissible_indices)
        if admissible_indices
        else None
    )
    cold_admissible_mean = (
        sum(cold_counts[index] for index in admissible_indices)
        / len(admissible_indices)
        if admissible_indices
        else None
    )
    report = {
        "schema": "neural-computer.external-register-compatibility-prior-audit.v1",
        "claim_boundary": (
            "A memory-side opaque screen may reduce candidate trial count on a "
            "held-out scalar-verifier distribution; it never replaces fresh admission."
        ),
        "seed": args.seed,
        "basis_count": args.basis_count,
        "train_queries": args.updates * args.batch_size,
        "audit_queries": args.audit_queries,
        "updates": args.updates,
        "threshold": args.threshold,
        "learned": {
            "mean_trials": learned_mean,
            "pass_rate": learned_passes / args.audit_queries,
            "mean_trials_on_admissible": learned_admissible_mean,
        },
        "cold_append_order": {
            "mean_trials": cold_mean,
            "pass_rate": cold_passes / args.audit_queries,
            "mean_trials_on_admissible": cold_admissible_mean,
        },
        "admissible_queries": len(admissible_indices),
        "accounting": {
            "replayed_examples": 0,
            "optimizer_updates": args.updates,
            "unique_verifier_outcomes": (
                args.updates * args.batch_size * args.basis_count
                + args.audit_queries * args.basis_count
            ),
        },
        "gates": {
            "learned_screen_preserves_verifier_admissibility": (
                learned_passes == cold_passes
            ),
            "learned_reduces_trials_on_admissible": (
                learned_admissible_mean is not None
                and cold_admissible_mean is not None
                and learned_admissible_mean < cold_admissible_mean
            ),
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
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--latent-width", type=int, default=12)
    parser.add_argument("--basis-count", type=int, default=5)
    parser.add_argument("--audit-queries", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
