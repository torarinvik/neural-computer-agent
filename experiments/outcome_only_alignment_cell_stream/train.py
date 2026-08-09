"""Pressure-test persistent external alignment cells under scalar-only credit.

Each opaque frontend transform receives a fresh external event-bridge cell.
The source controller, external register, and decoder are frozen before the
stream begins. Cells are trained sequentially from sampled verifier outcomes,
frozen after admission, and revisited without replay while later cells grow.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from experiments.external_register_composition_amodal.train import (
    ACTION_WIDTH,
    EVENT_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _batch,
    _module_digest,
    _new_machine,
    _rollout,
    _stable_bits,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.outcome_only_event_alignment.train import (
    SOURCE_BANK_WIDTH,
    _train_source,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import AmodalEventBridge, OpaqueProtocolDecoder


OPERATION = "reverse"
MASTERY_THRESHOLD = 0.8
TRANSFORM_SEEDS = (4_271_991, 98_271, 193_817)


class ExternalAlignmentCellBank(nn.Module):
    """A growable external bank of independently replaceable bridge cells."""

    schema = "neural-computer.external-alignment-cell-bank.v1"

    def __init__(self) -> None:
        super().__init__()
        self.cells = nn.ModuleDict()

    @staticmethod
    def _key(logical_id: str) -> str:
        if not logical_id or "." in logical_id:
            raise ValueError("alignment logical IDs must be non-empty and dot-free")
        return logical_id

    def add(self, logical_id: str, bridge: AmodalEventBridge) -> None:
        key = self._key(logical_id)
        if key in self.cells:
            raise ValueError(f"alignment cell already exists: {logical_id}")
        self.cells[key] = bridge

    def cell(self, logical_id: str) -> AmodalEventBridge:
        key = self._key(logical_id)
        try:
            return self.cells[key]
        except KeyError as error:
            raise KeyError(f"unknown alignment cell: {logical_id}") from error

    def freeze(self, logical_id: str) -> None:
        for parameter in self.cell(logical_id).parameters():
            parameter.requires_grad_(False)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "cell_count": len(self.cells),
            "logical_ids": tuple(self.cells.keys()),
            "cell_interface": "neural-computer.event-bridge.v1",
        }


def _source_setup(args: argparse.Namespace):
    parent = _runtime(seed=args.seed, growth=False)
    _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=3e-3,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
    )
    parent.eval()
    parent_digest_before = _module_digest(parent.controller)

    source_parent = _new_machine(SOURCE_BANK_WIDTH)
    for _ in range(SOURCE_BANK_WIDTH):
        source_parent.add_basis_slot()
    best_machine_state: dict[str, torch.Tensor] | None = None
    best_decoder_state: dict[str, torch.Tensor] | None = None
    source_accuracy = float("-inf")
    source_attempts: list[float] = []
    for attempt in range(args.source_restarts):
        machine = copy.deepcopy(source_parent)
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        _train_source(
            parent,
            machine,
            decoder,
            updates=args.source_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 10_000 + attempt * 1_000_003,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
        )
        score = _accuracy(
            parent,
            machine,
            decoder,
            operation=OPERATION,
            instructions=(machine.instructions[0],),
            basis_slots=(0,),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 20_000,
            credit_mode="paired_counterfactual",
        )
        source_attempts.append(score)
        if score > source_accuracy:
            source_accuracy = score
            best_machine_state = {
                name: value.detach().clone()
                for name, value in machine.state_dict().items()
            }
            best_decoder_state = {
                name: value.detach().clone()
                for name, value in decoder.state_dict().items()
            }
    assert best_machine_state is not None and best_decoder_state is not None
    machine = _new_machine(SOURCE_BANK_WIDTH)
    for _ in range(SOURCE_BANK_WIDTH):
        machine.add_basis_slot()
    machine.load_state_dict(best_machine_state, strict=True)
    decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    decoder.load_state_dict(best_decoder_state, strict=True)
    machine_digest_before = _module_digest(machine)
    decoder_digest_before = _module_digest(decoder)
    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return (
        parent,
        machine,
        decoder,
        parent_digest_before,
        machine_digest_before,
        decoder_digest_before,
        source_accuracy,
        source_attempts,
    )


def _score(
    parent,
    machine,
    decoder,
    bridge,
    *,
    transform_seed: int,
    count: int,
    span: int,
    seed: int,
) -> float:
    return _accuracy(
        parent,
        machine,
        decoder,
        operation=OPERATION,
        instructions=(machine.instructions[0],),
        basis_slots=(0,),
        count=count,
        span=span,
        seed=seed,
        credit_mode="paired_counterfactual",
        event_bridge=bridge,
        bridge_event_mode="composed_orthogonal",
        bridge_state_mode="zero",
        bridge_transform_seed=transform_seed,
    )


def _train_cell(
    parent,
    machine,
    decoder,
    *,
    transform_seed: int,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    learning_rate: float,
    eval_every: int,
    audit_count: int,
    shuffle_outcomes: bool,
) -> tuple[AmodalEventBridge, list[dict[str, float | int]]]:
    bridge = AmodalEventBridge(
        EVENT_WIDTH,
        parent.controller.width,
        EVENT_WIDTH,
        hidden=64,
    )
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    progress: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        batch = _batch(
            OPERATION,
            count=batch_size,
            span=span,
            seed=seed + update * 10_007,
        )
        loss, _ = _rollout(
            parent,
            machine,
            decoder,
            batch,
            (machine.instructions[0],),
            basis_slots=(0,),
            train_decoder=True,
            shuffle_outcomes=shuffle_outcomes,
            credit_mode="attempted_bce",
            event_bridge=bridge,
            bridge_event_mode="composed_orthogonal",
            bridge_state_mode="zero",
            bridge_transform_seed=transform_seed,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            progress.append(
                {
                    "update": update,
                    "heldout_accuracy": _score(
                        parent,
                        machine,
                        decoder,
                        bridge,
                        transform_seed=transform_seed,
                        count=audit_count,
                        span=span,
                        seed=seed + 100_000 + update,
                    ),
                }
            )
    return bridge, progress


def _return_scores(
    bank: ExternalAlignmentCellBank,
    transform_seeds: tuple[int, ...],
    parent,
    machine,
    decoder,
    *,
    count: int,
    span: int,
    seed: int,
) -> list[float]:
    return [
        _score(
            parent,
            machine,
            decoder,
            bank.cell(f"cell_{index}"),
            transform_seed=transform_seed,
            count=count,
            span=span,
            seed=seed + index * 101,
        )
        for index, transform_seed in enumerate(transform_seeds)
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    if args.cell_count > len(TRANSFORM_SEEDS):
        raise ValueError("cell count exceeds the registered transform stream")
    if min(
        args.cell_count,
        args.parent_updates,
        args.source_updates,
        args.source_restarts,
        args.bridge_updates,
        args.batch_size,
        args.span,
        args.audit_count,
        args.eval_every,
    ) < 1:
        raise ValueError("all update, cell, batch, span, and audit counts must be positive")

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
    cell_reports: list[dict[str, object]] = []
    return_history: list[list[float]] = []
    normal_stable_bits: list[int | None] = []
    shuffled_stable_bits: list[int | None] = []
    for index in range(args.cell_count):
        logical_id = f"cell_{index}"
        transform_seed = TRANSFORM_SEEDS[index]
        fresh_bridge = AmodalEventBridge(
            EVENT_WIDTH,
            parent.controller.width,
            EVENT_WIDTH,
            hidden=64,
        )
        target_before = _score(
            parent,
            machine,
            decoder,
            fresh_bridge,
            transform_seed=transform_seed,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 30_000 + index * 1009,
        )
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
        target_after = _score(
            parent,
            machine,
            decoder,
            bridge,
            transform_seed=transform_seed,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 50_000 + index * 1009,
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
        shuffled_after = _score(
            parent,
            machine,
            decoder,
            shuffled_bridge,
            transform_seed=transform_seed,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 70_000 + index * 1009,
        )
        bank.add(logical_id, bridge)
        bank.freeze(logical_id)
        returns = _return_scores(
            bank,
            TRANSFORM_SEEDS[: index + 1],
            parent,
            machine,
            decoder,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 80_000,
        )
        return_history.append(returns)
        normal_stable_bits.append(
            _stable_bits(
                progress,
                threshold=MASTERY_THRESHOLD,
                bits_per_update=args.batch_size * args.span,
            )
        )
        shuffled_stable_bits.append(
            _stable_bits(
                shuffled_progress,
                threshold=MASTERY_THRESHOLD,
                bits_per_update=args.batch_size * args.span,
            )
        )
        cell_reports.append(
            {
                "logical_id": logical_id,
                "transform_seed": transform_seed,
                "target_before": target_before,
                "target_after": target_after,
                "shuffled_after": shuffled_after,
                "stable_bits_to_threshold": normal_stable_bits[-1],
                "shuffled_stable_bits_to_threshold": shuffled_stable_bits[-1],
                "progress": progress,
                "shuffled_progress": shuffled_progress,
                "return_scores_after_admission": returns,
            }
        )

    baseline_returns = return_history[-1]
    corrupted_bank = copy.deepcopy(bank)
    corruption_index = min(1, args.cell_count - 1)
    corrupted_cell = corrupted_bank.cell(f"cell_{corruption_index}")
    with torch.no_grad():
        for parameter in corrupted_cell.parameters():
            parameter.zero_()
    corrupted_returns = _return_scores(
        corrupted_bank,
        TRANSFORM_SEEDS[: args.cell_count],
        parent,
        machine,
        decoder,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 80_000,
    )
    source_after = _accuracy(
        parent,
        machine,
        decoder,
        operation=OPERATION,
        instructions=(machine.instructions[0],),
        basis_slots=(0,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 100_000,
        credit_mode="paired_counterfactual",
    )
    parent_digest_after = _module_digest(parent.controller)
    machine_digest_after = _module_digest(machine)
    decoder_digest_after = _module_digest(decoder)
    other_indices = [
        index for index in range(args.cell_count) if index != corruption_index
    ]
    gates = {
        "source_mastered": source_accuracy >= MASTERY_THRESHOLD,
        "every_cell_target_mastered": all(
            float(report["target_after"]) >= MASTERY_THRESHOLD
            for report in cell_reports
        ),
        "every_cell_representation_shifted": all(
            float(report["target_before"]) < MASTERY_THRESHOLD
            for report in cell_reports
        ),
        "shuffled_outcomes_rejected": all(
            float(report["shuffled_after"]) < MASTERY_THRESHOLD
            for report in cell_reports
        ),
        "stable_prefix_for_every_cell": all(
            value is not None for value in normal_stable_bits
        ),
        "no_shuffled_stable_prefix": all(
            value is None for value in shuffled_stable_bits
        ),
        "return_retention": all(
            score >= MASTERY_THRESHOLD
            for scores in return_history
            for score in scores
        ),
        "single_cell_corruption_is_local": all(
            abs(corrupted_returns[index] - baseline_returns[index]) < 1e-7
            for index in other_indices
        ),
        "corrupted_cell_degrades": (
            corrupted_returns[corruption_index]
            < baseline_returns[corruption_index] - 0.05
        ),
        "frozen_parent": parent_digest_before == parent_digest_after,
        "frozen_source_machine": machine_digest_before == machine_digest_after,
        "frozen_source_decoder": decoder_digest_before == decoder_digest_after,
        "source_retained": source_after >= source_accuracy - 0.02,
    }
    report = {
        "schema": "neural-computer.outcome-only-alignment-cell-stream-report.v1",
        "claim_boundary": (
            "A growable external bank can acquire and retain bounded event-"
            "alignment cells from scalar outcomes while the computation core "
            "stays frozen; this is not general continual learning."
        ),
        "seed": args.seed,
        "configuration": {
            "operation": OPERATION,
            "cell_count": args.cell_count,
            "transform_seeds": list(TRANSFORM_SEEDS[: args.cell_count]),
            "bridge_event_mode": "composed_orthogonal",
            "bridge_state_mode": "zero",
            "parent_updates": args.parent_updates,
            "source_updates": args.source_updates,
            "source_restarts": args.source_restarts,
            "bridge_updates_per_cell": args.bridge_updates,
            "batch_size": args.batch_size,
            "span": args.span,
            "audit_count": args.audit_count,
        },
        "results": {
            "source_accuracy": source_accuracy,
            "source_attempts": source_attempts,
            "source_after": source_after,
            "cells": cell_reports,
            "return_history": return_history,
            "baseline_returns": baseline_returns,
            "corrupted_cell_index": corruption_index,
            "corrupted_returns": corrupted_returns,
            "bank_configuration": bank.configuration(),
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2 * 2
                + args.source_updates * args.batch_size * args.span * 2
                + args.cell_count * args.bridge_updates * args.batch_size * args.span * 2
            ),
            "unique_logical_lifetimes": (
                args.source_updates * args.batch_size
                + args.cell_count * args.bridge_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.source_updates
                + args.cell_count * args.bridge_updates * 2
            ),
            "replayed_examples": 0,
            "stable_bits_to_threshold": normal_stable_bits,
            "shuffled_stable_bits_to_threshold": shuffled_stable_bits,
        },
        "digests": {
            "parent_before": parent_digest_before,
            "parent_after": parent_digest_after,
            "source_machine_before": machine_digest_before,
            "source_machine_after": machine_digest_after,
            "source_decoder_before": decoder_digest_before,
            "source_decoder_after": decoder_digest_after,
            "bank": _module_digest(bank),
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
    parser.add_argument("--cell-count", type=int, default=3)
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
