"""Pressure-test outcome-only recovery from a changed event representation.

The parent controller, external register, and output decoder are trained on a
source event space and then frozen. A cyclic permutation is applied to the
learned event tensor before a replaceable external bridge. The bridge alone is
trained from sampled scalar verifier outcomes; no correct action, task label,
or representation correspondence enters the optimizer. Matched shuffled-
outcome, source-retention, and frozen-parent controls define the claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import AmodalEventBridge, OpaqueProtocolDecoder

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
    _train_stage,
)


OPERATION = "reverse"
SOURCE_BANK_WIDTH = 3
MASTERY_THRESHOLD = 0.8


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train_source(
    parent,
    machine,
    decoder,
    *,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    eval_every: int,
    audit_count: int,
) -> list[dict[str, float | int]]:
    return _train_stage(
        parent,
        machine,
        decoder,
        operation=OPERATION,
        instructions=(machine.instructions[0],),
        basis_slots=(0,),
        updates=updates,
        batch_size=batch_size,
        span=span,
        seed=seed,
        trainable=[*machine.parameters(), *decoder.parameters()],
        credit_mode="paired_counterfactual",
        eval_every=eval_every,
        audit_count=audit_count,
        audit_seed=seed + 100_000,
        fixed_audit_seed=True,
        restore_best_checkpoint=True,
    )


def _score(
    parent,
    machine,
    decoder,
    bridge,
    *,
    count: int,
    span: int,
    seed: int,
    bridge_event_mode: str = "cyclic_permutation",
    shuffle_outcomes: bool = False,
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
        shuffle_outcomes=shuffle_outcomes,
        event_bridge=bridge,
        bridge_event_mode=bridge_event_mode,
        bridge_state_mode="zero",
    )


def _train_bridge(
    parent,
    machine,
    decoder,
    bridge,
    *,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    learning_rate: float,
    eval_every: int,
    audit_count: int,
    shuffle_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=learning_rate, weight_decay=1e-5
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
            bridge_event_mode="cyclic_permutation",
            bridge_state_mode="zero",
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            accuracy = _score(
                parent,
                machine,
                decoder,
                bridge,
                count=audit_count,
                span=span,
                seed=seed + 100_000 + update,
            )
            progress.append({"update": update, "heldout_accuracy": accuracy})
    return progress


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
    source_attempts: list[float] = []
    best_machine_state: dict[str, torch.Tensor] | None = None
    best_decoder_state: dict[str, torch.Tensor] | None = None
    best_source_progress: list[dict[str, float | int]] | None = None
    source_accuracy = float("-inf")
    for attempt in range(args.source_restarts):
        candidate_machine = copy.deepcopy(source_parent)
        candidate_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        candidate_progress = _train_source(
            parent,
            candidate_machine,
            candidate_decoder,
            updates=args.source_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 10_000 + attempt * 1_000_003,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
        )
        candidate_accuracy = _accuracy(
            parent,
            candidate_machine,
            candidate_decoder,
            operation=OPERATION,
            instructions=(candidate_machine.instructions[0],),
            basis_slots=(0,),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 20_000,
            credit_mode="paired_counterfactual",
        )
        source_attempts.append(candidate_accuracy)
        if candidate_accuracy > source_accuracy:
            source_accuracy = candidate_accuracy
            best_machine_state = {
                name: value.detach().clone()
                for name, value in candidate_machine.state_dict().items()
            }
            best_decoder_state = {
                name: value.detach().clone()
                for name, value in candidate_decoder.state_dict().items()
            }
            best_source_progress = candidate_progress
    assert best_machine_state is not None and best_decoder_state is not None
    assert best_source_progress is not None
    machine = _new_machine(SOURCE_BANK_WIDTH)
    for _ in range(SOURCE_BANK_WIDTH):
        machine.add_basis_slot()
    machine.load_state_dict(best_machine_state, strict=True)
    decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    decoder.load_state_dict(best_decoder_state, strict=True)
    source_progress = best_source_progress
    source_machine_digest = _digest(machine)
    source_decoder_digest = _digest(decoder)
    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    bridge = AmodalEventBridge(
        EVENT_WIDTH,
        parent.controller.width,
        EVENT_WIDTH,
        hidden=64,
    )
    target_before = _score(
        parent,
        machine,
        decoder,
        bridge,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 30_000,
    )
    bridge_progress = _train_bridge(
        parent,
        machine,
        decoder,
        bridge,
        updates=args.bridge_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 40_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
    )
    target_after = _score(
        parent,
        machine,
        decoder,
        bridge,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 50_000,
    )
    shuffled_bridge = AmodalEventBridge(
        EVENT_WIDTH,
        parent.controller.width,
        EVENT_WIDTH,
        hidden=64,
    )
    shuffled_progress = _train_bridge(
        parent,
        machine,
        decoder,
        shuffled_bridge,
        updates=args.bridge_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 60_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        shuffle_outcomes=True,
    )
    shuffled_accuracy = _score(
        parent,
        machine,
        decoder,
        shuffled_bridge,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 70_000,
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
        seed=args.seed + 80_000,
        credit_mode="paired_counterfactual",
    )
    parent_digest_after = _module_digest(parent.controller)
    source_machine_after = _digest(machine)
    source_decoder_after = _digest(decoder)
    stable_bits = _stable_bits(
        bridge_progress,
        threshold=MASTERY_THRESHOLD,
        bits_per_update=args.batch_size * args.span,
    )
    shuffled_stable_bits = _stable_bits(
        shuffled_progress,
        threshold=MASTERY_THRESHOLD,
        bits_per_update=args.batch_size * args.span,
    )
    gates = {
        "source_mastered": source_accuracy >= MASTERY_THRESHOLD,
        "representation_change_is_nontrivial": target_before < MASTERY_THRESHOLD,
        "bridge_relearns_from_scalar_outcomes": (
            target_after >= MASTERY_THRESHOLD and stable_bits is not None
        ),
        "shuffled_outcomes_rejected": shuffled_accuracy < MASTERY_THRESHOLD,
        "source_retained": source_after >= source_accuracy - 0.02,
        "source_machine_unchanged": source_machine_digest == source_machine_after,
        "source_decoder_unchanged": source_decoder_digest == source_decoder_after,
        "frozen_parent": parent_digest_before == parent_digest_after,
    }
    report = {
        "schema": "neural-computer.outcome-only-event-alignment-report.v1",
        "claim_boundary": (
            "A frozen external computation can recover one deterministic event-"
            "space permutation from sampled scalar outcomes; this is not a "
            "general alignment or continual-learning promotion."
        ),
        "seed": args.seed,
        "configuration": {
            "operation": OPERATION,
            "source_bank_width": SOURCE_BANK_WIDTH,
            "event_width": EVENT_WIDTH,
            "bridge_event_mode": "cyclic_permutation",
            "bridge_state_mode": "zero",
            "parent_updates": args.parent_updates,
            "source_updates": args.source_updates,
            "source_restarts": args.source_restarts,
            "bridge_updates": args.bridge_updates,
            "batch_size": args.batch_size,
            "span": args.span,
            "audit_count": args.audit_count,
        },
        "results": {
            "source_accuracy": source_accuracy,
            "source_attempts": source_attempts,
            "target_before_bridge": target_before,
            "target_after_bridge": target_after,
            "shuffled_outcome_accuracy": shuffled_accuracy,
            "source_after": source_after,
        },
        "learning_curves": {
            "source": source_progress,
            "bridge": bridge_progress,
            "shuffled_bridge": shuffled_progress,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2 * 2
                + args.source_updates * args.batch_size * args.span * 2
                + args.bridge_updates * args.batch_size * args.span
                + args.bridge_updates * args.batch_size * args.span
            ),
            "unique_logical_lifetimes": (
                args.source_updates * args.batch_size
                + args.bridge_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.source_updates
                + args.bridge_updates * 2
            ),
            "replayed_examples": 0,
            "stable_bits_to_threshold": stable_bits,
            "shuffled_stable_bits_to_threshold": shuffled_stable_bits,
            "target_bridge_verifier_bits": (
                args.bridge_updates * args.batch_size * args.span
            ),
        },
        "digests": {
            "parent_before": parent_digest_before,
            "parent_after": parent_digest_after,
            "source_machine_before": source_machine_digest,
            "source_machine_after": source_machine_after,
            "source_decoder_before": source_decoder_digest,
            "source_decoder_after": source_decoder_after,
            "bridge": _digest(bridge),
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
