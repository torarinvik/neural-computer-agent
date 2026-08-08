"""Audit verifier-gated reuse of a mastered external register basis slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import OpaqueProtocolDecoder

from .train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _module_digest,
    _new_machine,
    _stable_bits,
    _train_stage,
)

TARGET_OPERATION = "rotate"
THRESHOLD = 0.8


def _freeze_machine(machine) -> None:
    for parameter in machine.parameters():
        parameter.requires_grad_(False)


def _train_rotate(
    parent,
    machine,
    decoder,
    *,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    trainable,
    basis_slot: int,
    shuffled: bool = False,
):
    return _train_stage(
        parent,
        machine,
        decoder,
        operation=TARGET_OPERATION,
        instructions=(machine.instructions[-1],),
        basis_slots=(basis_slot,),
        updates=updates,
        batch_size=batch_size,
        span=span,
        seed=seed,
        trainable=list(trainable),
        credit_mode="paired_counterfactual",
        shuffle_outcomes=shuffled,
        eval_every=32,
        audit_count=64,
        audit_seed=seed + 1_000_000,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
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

    inherited = _new_machine(1, operator_mode=args.operator_mode)
    basis_slot = inherited.add_basis_slot()
    first_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_rotate(
        parent,
        inherited,
        first_decoder,
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 20_000,
        trainable=[*inherited.parameters(), *first_decoder.parameters()],
        basis_slot=basis_slot,
    )
    first_accuracy = _accuracy(
        parent,
        inherited,
        first_decoder,
        operation=TARGET_OPERATION,
        instructions=(inherited.instructions[0],),
        basis_slots=(basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 21_000,
        credit_mode="paired_counterfactual",
    )
    admission = inherited.select_basis_slot(
        {basis_slot: (first_accuracy, first_accuracy)}, threshold=THRESHOLD
    )
    if admission.action != "reuse":
        raise RuntimeError("mastered basis was not admitted for reuse")

    inherited.freeze_basis_slot(basis_slot)
    _freeze_machine(inherited)
    inherited.add_instruction(type(inherited.instructions[0])(inherited.instruction_width))
    second_instruction = inherited.instructions[-1]
    second_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    second_progress = _train_rotate(
        parent,
        inherited,
        second_decoder,
        updates=args.reuse_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 30_000,
        trainable=[second_instruction.code, *second_decoder.parameters()],
        basis_slot=basis_slot,
    )
    second_accuracy = _accuracy(
        parent,
        inherited,
        second_decoder,
        operation=TARGET_OPERATION,
        instructions=(second_instruction,),
        basis_slots=(basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 31_000,
        credit_mode="paired_counterfactual",
    )
    first_after = _accuracy(
        parent,
        inherited,
        first_decoder,
        operation=TARGET_OPERATION,
        instructions=(inherited.instructions[0],),
        basis_slots=(basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 32_000,
        credit_mode="paired_counterfactual",
    )
    reuse_state = {
        name: value.detach().clone() for name, value in inherited.state_dict().items()
    }
    basis_digest = _module_digest(inherited.basis_slots[basis_slot])
    shuffled = _new_machine(2, operator_mode=args.operator_mode)
    shuffled.add_basis_slot()
    shuffled.load_state_dict(reuse_state, strict=True)
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _freeze_machine(shuffled)
    _train_rotate(
        parent,
        shuffled,
        shuffled_decoder,
        updates=args.reuse_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 40_000,
        trainable=[shuffled.instructions[-1].code, *shuffled_decoder.parameters()],
        basis_slot=basis_slot,
        shuffled=True,
    )
    shuffled_accuracy = _accuracy(
        parent,
        shuffled,
        shuffled_decoder,
        operation=TARGET_OPERATION,
        instructions=(shuffled.instructions[-1],),
        basis_slots=(basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 41_000,
        credit_mode="paired_counterfactual",
        shuffle_outcomes=True,
    )
    missing_accuracy = _accuracy(
        parent,
        inherited,
        second_decoder,
        operation=TARGET_OPERATION,
        instructions=(second_instruction,),
        basis_slots=(basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 42_000,
        credit_mode="paired_counterfactual",
        evidence_present=False,
    )
    
    reloaded = _new_machine(2, operator_mode=args.operator_mode)
    reloaded.add_basis_slot()
    reloaded.load_state_dict(reuse_state, strict=True)
    reload_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reload_decoder.load_state_dict(second_decoder.state_dict(), strict=True)
    reload_accuracy = _accuracy(
        parent,
        reloaded,
        reload_decoder,
        operation=TARGET_OPERATION,
        instructions=(reloaded.instructions[-1],),
        basis_slots=(basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 31_000,
        credit_mode="paired_counterfactual",
    )
    inherited_stable = _stable_bits(
        second_progress,
        threshold=THRESHOLD,
        bits_per_update=args.batch_size * args.span * 2,
    )
    fresh = _new_machine(1, operator_mode=args.operator_mode)
    fresh_basis = fresh.add_basis_slot()
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_progress = _train_rotate(
        parent,
        fresh,
        fresh_decoder,
        updates=args.reuse_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 50_000,
        trainable=[*fresh.parameters(), *fresh_decoder.parameters()],
        basis_slot=fresh_basis,
    )
    fresh_accuracy = _accuracy(
        parent,
        fresh,
        fresh_decoder,
        operation=TARGET_OPERATION,
        instructions=(fresh.instructions[-1],),
        basis_slots=(fresh_basis,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 51_000,
        credit_mode="paired_counterfactual",
    )
    parent_digest_after = _module_digest(parent.controller)
    gates = {
        "basis_admitted_by_fresh_probe": admission.action == "reuse",
        "first_mastered": first_accuracy >= THRESHOLD,
        "reused_mastered": second_accuracy >= THRESHOLD,
        "first_retained": first_after >= THRESHOLD,
        "frozen_basis": basis_digest == _module_digest(inherited.basis_slots[basis_slot]),
        "reward_shuffled_rejected": shuffled_accuracy < THRESHOLD,
        "missing_evidence_rejected": missing_accuracy < THRESHOLD,
        "reload_exact": abs(reload_accuracy - second_accuracy) < 1e-12,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "no_replayed_examples": True,
    }
    return {
        "schema": "neural-computer.external-register-basis-reuse-report.v1",
        "claim_boundary": "A mastered external basis slot is reused by a fresh opaque instruction without replaying prior examples or updating the slot.",
        "seed": args.seed,
        "operator_mode": args.operator_mode,
        "first_accuracy": first_accuracy,
        "reused_accuracy": second_accuracy,
        "fresh_accuracy": fresh_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "missing_evidence_accuracy": missing_accuracy,
        "reload_accuracy": reload_accuracy,
        "stable_bits": {
            "reused": inherited_stable,
            "fresh": _stable_bits(
                fresh_progress,
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * args.span * 2,
            ),
        },
        "admission": {
            "action": admission.action,
            "slot": admission.compute_slot_index,
            "candidate_scores": admission.candidate_scores,
        },
        "accounting": {
            "unique_verifier_bits": (
                (args.parent_updates * args.batch_size * 2)
                + (args.primitive_updates + args.reuse_updates * 2)
                * args.batch_size
                * args.span
                * 2
            ),
            "optimizer_updates": args.parent_updates
            + args.primitive_updates
            + args.reuse_updates * 2,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--primitive-updates", type=int, default=192)
    parser.add_argument("--reuse-updates", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument(
        "--operator-mode",
        choices=("factorized_bounded_residual", "factorized_protected_meta"),
        default="factorized_bounded_residual",
    )
    args = parser.parse_args()
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
