"""Pressure-test reversal-safe eviction and replacement of alignment cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.external_register_composition_amodal.train import (
    _batch,
    _module_digest,
    _rollout,
    _stable_bits,
)
from experiments.outcome_only_alignment_cell_stream.train import (
    ExternalAlignmentCellBank,
    TRANSFORM_SEEDS,
    _score,
    _source_setup,
    _train_cell,
)
from experiments.outcome_only_online_alignment_growth.train import (
    ExternalAlignmentKeyBank,
    ONLINE_TRANSFORM_SEED,
    _estimate_key,
    _evaluate_key_router,
)
from neural_computer import CapabilityRetentionLedger, RetentionPolicyConfig


OPERATION = "reverse"
REPLACEMENT_TRANSFORM_SEED = 812_345
BASE_TRANSFORM_SEEDS = (*TRANSFORM_SEEDS, ONLINE_TRANSFORM_SEED)
MASTERY_THRESHOLD = 0.8


def _collect_outcome(
    parent,
    machine,
    decoder,
    bridge,
    *,
    transform_seed: int,
    seed: int,
    batch_size: int,
    span: int,
) -> float:
    batch = _batch(
        OPERATION,
        count=batch_size,
        span=span,
        seed=seed,
    )
    with torch.no_grad():
        _, rewards = _rollout(
            parent,
            machine,
            decoder,
            batch,
            (machine.instructions[0],),
            basis_slots=(0,),
            train_decoder=False,
            credit_mode="attempted_bce",
            event_bridge=bridge,
            bridge_event_mode="composed_orthogonal",
            bridge_state_mode="zero",
            bridge_transform_seed=transform_seed,
        )
    return float(rewards.mean())


def _observe_key(
    ledger: CapabilityRetentionLedger,
    key: torch.Tensor,
    *,
    parent,
    machine,
    decoder,
    bridge,
    transform_seed: int,
    seed: int,
    observations: int,
    batch_size: int,
    span: int,
) -> list[float]:
    outcomes = []
    for index in range(observations):
        outcome = _collect_outcome(
            parent,
            machine,
            decoder,
            bridge,
            transform_seed=transform_seed,
            seed=seed + index * 1009,
            batch_size=batch_size,
            span=span,
        )
        outcomes.append(outcome)
        ledger.observe(key, outcome)
    return outcomes


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    if min(
        args.parent_updates,
        args.source_updates,
        args.source_restarts,
        args.bridge_updates,
        args.batch_size,
        args.span,
        args.audit_count,
        args.eval_every,
    ) < 1:
        raise ValueError("all update, batch, span, and audit counts must be positive")
    (
        parent,
        machine,
        decoder,
        parent_digest_before,
        machine_digest_before,
        decoder_digest_before,
        source_accuracy,
        source_attempts,
    ) = _source_setup(args)
    bank = ExternalAlignmentCellBank()
    key_bank = ExternalAlignmentKeyBank(64)
    cell_reports = []
    for index, transform_seed in enumerate(BASE_TRANSFORM_SEEDS):
        bridge, progress = _train_cell(
            parent,
            machine,
            decoder,
            transform_seed=transform_seed,
            updates=args.bridge_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 40_000 + index * 100_003,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            shuffle_outcomes=False,
        )
        shuffled_bridge, shuffled_progress = _train_cell(
            parent,
            machine,
            decoder,
            transform_seed=transform_seed,
            updates=args.bridge_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 60_000 + index * 100_003,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            shuffle_outcomes=True,
        )
        logical_id = f"cell_{index}"
        bank.add(logical_id, bridge)
        bank.freeze(logical_id)
        key_bank.add(
            logical_id,
            _estimate_key(
                parent,
                transform_seed,
                seed=args.seed + 130_000 + index * 10_003,
            ),
        )
        cell_reports.append(
            {
                "logical_id": logical_id,
                "transform_seed": transform_seed,
                "target_after": _score(
                    parent,
                    machine,
                    decoder,
                    bridge,
                    transform_seed=transform_seed,
                    count=args.audit_count,
                    span=args.span,
                    seed=args.seed + 50_000 + index * 1009,
                ),
                "shuffled_after": _score(
                    parent,
                    machine,
                    decoder,
                    shuffled_bridge,
                    transform_seed=transform_seed,
                    count=args.audit_count,
                    span=args.span,
                    seed=args.seed + 70_000 + index * 1009,
                ),
                "stable_bits_to_threshold": _stable_bits(
                    progress,
                    threshold=MASTERY_THRESHOLD,
                    bits_per_update=args.batch_size * args.span,
                ),
                "shuffled_stable_bits_to_threshold": _stable_bits(
                    shuffled_progress,
                    threshold=MASTERY_THRESHOLD,
                    bits_per_update=args.batch_size * args.span,
                ),
            }
        )

    ledger = CapabilityRetentionLedger(
        width=64,
        config=RetentionPolicyConfig(
            mastery_threshold=MASTERY_THRESHOLD,
            min_mastery_observations=4,
            reversal_threshold=0.75,
            reversal_patience=4,
            recent_window=4,
        ),
    )
    protected_outcomes = {}
    for index, transform_seed in enumerate(BASE_TRANSFORM_SEEDS):
        protected_outcomes[f"cell_{index}"] = _observe_key(
            ledger,
            key_bank.keys[index],
            parent=parent,
            machine=machine,
            decoder=decoder,
            bridge=bank.cell(f"cell_{index}"),
            transform_seed=transform_seed,
            seed=args.seed + 200_000 + index * 10_003,
            observations=4,
            batch_size=args.batch_size,
            span=args.span,
        )
    statuses_protected = [
        ledger.status(key_bank.keys[index]).__dict__
        for index in range(len(BASE_TRANSFORM_SEEDS))
    ]

    replacement_bridge, replacement_progress = _train_cell(
        parent,
        machine,
        decoder,
        transform_seed=REPLACEMENT_TRANSFORM_SEED,
        updates=args.bridge_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 340_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        shuffle_outcomes=False,
    )
    shuffled_replacement_bridge, shuffled_replacement_progress = _train_cell(
        parent,
        machine,
        decoder,
        transform_seed=REPLACEMENT_TRANSFORM_SEED,
        updates=args.bridge_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 360_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        shuffle_outcomes=True,
    )
    replacement_key = _estimate_key(
        parent,
        REPLACEMENT_TRANSFORM_SEED,
        seed=args.seed + 440_000,
    )
    stale_key = key_bank.keys[3].detach().clone()
    stale_outcomes = _observe_key(
        ledger,
        stale_key,
        parent=parent,
        machine=machine,
        decoder=decoder,
        bridge=bank.cell("cell_3"),
        transform_seed=REPLACEMENT_TRANSFORM_SEED,
        seed=args.seed + 460_000,
        observations=4,
        batch_size=args.batch_size,
        span=args.span,
    )
    statuses_reversed = [
        ledger.status(key_bank.keys[index]).__dict__
        for index in range(len(BASE_TRANSFORM_SEEDS))
    ]
    learned_scores = torch.tensor(
        [1.0 - float(status["recent_mean"]) for status in statuses_reversed]
    )
    eviction_index = ledger.choose_eviction_index(key_bank.keys, learned_scores)
    evicted_id = f"cell_{eviction_index}" if eviction_index is not None else None
    if eviction_index == 3:
        bank.remove("cell_3")
        key_bank.remove("cell_3")
        bank.add("cell_3", replacement_bridge)
        bank.freeze("cell_3")
        key_bank.add("cell_3", replacement_key)
    replacement_outcomes = _observe_key(
        ledger,
        replacement_key,
        parent=parent,
        machine=machine,
        decoder=decoder,
        bridge=replacement_bridge,
        transform_seed=REPLACEMENT_TRANSFORM_SEED,
        seed=args.seed + 480_000,
        observations=4,
        batch_size=args.batch_size,
        span=args.span,
    )
    replacement_routing = _evaluate_key_router(
        parent,
        machine,
        decoder,
        bank,
        key_bank,
        transform_seeds=(*TRANSFORM_SEEDS[:3], REPLACEMENT_TRANSFORM_SEED),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 490_000,
    )
    source_after = _score(
        parent,
        machine,
        decoder,
        bank.cell("cell_0"),
        transform_seed=TRANSFORM_SEEDS[0],
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 100_000,
    )
    parent_digest_after = _module_digest(parent.controller)
    machine_digest_after = _module_digest(machine)
    decoder_digest_after = _module_digest(decoder)
    retained_rows = replacement_routing["rows"][:3]
    gates = {
        "source_mastered": source_accuracy >= MASTERY_THRESHOLD,
        "base_cells_mastered": all(
            float(row["target_after"]) >= MASTERY_THRESHOLD for row in cell_reports
        ),
        "base_shuffled_rejected": all(
            float(row["shuffled_after"]) < MASTERY_THRESHOLD
            for row in cell_reports
        ),
        "all_base_cells_protected": all(
            bool(status["protected"]) for status in statuses_protected
        ),
        "stale_cell_reversed": not bool(statuses_reversed[3]["protected"]),
        "eviction_selected_stale_cell": eviction_index == 3,
        "replacement_mastered": (
            _score(
                parent,
                machine,
                decoder,
                replacement_bridge,
                transform_seed=REPLACEMENT_TRANSFORM_SEED,
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 500_000,
            )
            >= MASTERY_THRESHOLD
        ),
        "replacement_shuffled_rejected": (
            _score(
                parent,
                machine,
                decoder,
                shuffled_replacement_bridge,
                transform_seed=REPLACEMENT_TRANSFORM_SEED,
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 510_000,
            )
            < MASTERY_THRESHOLD
        ),
        "replacement_key_routing_mastered": bool(
            replacement_routing["action_mastery"]
        )
        and float(replacement_routing["routing_accuracy"]) == 1.0,
        "protected_cells_retained": all(
            bool(row["routing_correct"])
            and float(row["action_accuracy"]) >= MASTERY_THRESHOLD
            for row in retained_rows
        ),
        "source_retained": source_after >= source_accuracy - 0.02,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "frozen_source_machine": machine_digest_before == machine_digest_after,
        "frozen_source_decoder": decoder_digest_before == decoder_digest_after,
        "reversal_has_scalar_evidence": len(stale_outcomes) == 4,
    }
    report = {
        "schema": "neural-computer.outcome-only-alignment-lifecycle-pressure-report.v1",
        "claim_boundary": (
            "A scalar-outcome retention ledger can release a reversed external "
            "alignment cell and replace it transactionally while protected "
            "cells remain; this is not general continual learning."
        ),
        "seed": args.seed,
        "configuration": {
            "base_transform_seeds": list(BASE_TRANSFORM_SEEDS),
            "replacement_transform_seed": REPLACEMENT_TRANSFORM_SEED,
            "retention_ledger": ledger.configuration(),
            "capacity": len(BASE_TRANSFORM_SEEDS),
            "bridge_updates": args.bridge_updates,
            "batch_size": args.batch_size,
            "span": args.span,
            "audit_count": args.audit_count,
        },
        "results": {
            "source_accuracy": source_accuracy,
            "source_attempts": source_attempts,
            "source_after": source_after,
            "base_cells": cell_reports,
            "protected_outcomes": protected_outcomes,
            "statuses_protected": statuses_protected,
            "stale_outcomes": stale_outcomes,
            "statuses_reversed": statuses_reversed,
            "learned_eviction_scores": learned_scores.tolist(),
            "eviction_index": eviction_index,
            "evicted_id": evicted_id,
            "replacement_outcomes": replacement_outcomes,
            "replacement_routing": replacement_routing,
            "replacement_progress": replacement_progress,
            "shuffled_replacement_progress": shuffled_replacement_progress,
            "key_bank": key_bank.configuration(),
        },
        "accounting": {
            "unique_verifier_bits": (
                args.source_updates * args.batch_size * args.span * 2
                + len(BASE_TRANSFORM_SEEDS) * args.bridge_updates * args.batch_size * args.span * 2
                + args.bridge_updates * args.batch_size * args.span * 2
                + 4 * len(BASE_TRANSFORM_SEEDS) * args.batch_size * args.span
                + 4 * args.batch_size * args.span
            ),
            "unique_logical_lifetimes": (
                args.source_updates * args.batch_size
                + (len(BASE_TRANSFORM_SEEDS) + 1) * args.bridge_updates * args.batch_size * 2
                + 4 * len(BASE_TRANSFORM_SEEDS)
                + 4
            ),
            "optimizer_updates": (
                args.source_updates
                + (len(BASE_TRANSFORM_SEEDS) + 1) * args.bridge_updates * 2
            ),
            "replayed_examples": 0,
        },
        "digests": {
            "parent_before": parent_digest_before,
            "parent_after": parent_digest_after,
            "source_machine_before": machine_digest_before,
            "source_machine_after": machine_digest_after,
            "source_decoder_before": decoder_digest_before,
            "source_decoder_after": decoder_digest_after,
            "key_bank": _module_digest(key_bank),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "elapsed_seconds": perf_counter() - started,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--source-restarts", type=int, default=2)
    parser.add_argument("--bridge-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
