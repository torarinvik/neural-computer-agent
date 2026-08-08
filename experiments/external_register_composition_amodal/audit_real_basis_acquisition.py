"""Audit learned basis ordering during real external-register acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    ExternalRegisterBasisCompatibilityPrior,
    OpaqueProtocolDecoder,
)

from .train import (
    ACTION_WIDTH,
    INSTRUCTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _module_digest,
    _new_machine,
    _stable_bits,
    _train_stage,
)

SOURCE_OPERATIONS = ("reverse", "adjacent_xor", "complement")
TARGET_OPERATION = "rotate"


def _freeze(machine) -> None:
    for parameter in machine.parameters():
        parameter.requires_grad_(False)


def _train_source(parent, machine, decoder, index: int, args) -> None:
    instruction = machine.instructions[index]
    if index == 0:
        trainable = [*machine.parameters(), *decoder.parameters()]
    else:
        _freeze(machine)
        instruction.code.requires_grad_(True)
        basis = machine.basis_slots[index]
        for parameter in basis.parameters():
            parameter.requires_grad_(True)
        trainable = [instruction.code, *basis.parameters(), *decoder.parameters()]
    _train_stage(
        parent,
        machine,
        decoder,
        operation=SOURCE_OPERATIONS[index],
        instructions=(instruction,),
        basis_slots=(index,),
        updates=args.source_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 10_000 + index * 10_003,
        trainable=trainable,
        credit_mode="paired_counterfactual",
    )


def _accuracy_matrix(parent, machine, decoders, *, count: int, span: int, seed: int):
    rows = []
    for operation, instruction, decoder in zip(
        SOURCE_OPERATIONS,
        machine.instructions[: len(SOURCE_OPERATIONS)],
        decoders,
        strict=True,
    ):
        rows.append(
            [
                _accuracy(
                    parent,
                    machine,
                    decoder,
                    operation=operation,
                    instructions=(instruction,),
                    basis_slots=(basis_slot,),
                    count=count,
                    span=span,
                    seed=seed + basis_slot * 101 + len(rows) * 1009,
                    credit_mode="paired_counterfactual",
                )
                for basis_slot in range(len(machine.basis_slots))
            ]
        )
    return torch.tensor(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
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
    machine = _new_machine(len(SOURCE_OPERATIONS) + 1, operator_mode=args.operator_mode)
    for _ in SOURCE_OPERATIONS:
        machine.add_basis_slot()
    decoders = [
        OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        for _ in SOURCE_OPERATIONS
    ]
    for index, decoder in enumerate(decoders):
        _train_source(parent, machine, decoder, index, args)
    source_outcomes = _accuracy_matrix(
        parent,
        machine,
        decoders,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 70_000,
    )
    prior = ExternalRegisterBasisCompatibilityPrior(INSTRUCTION_WIDTH, hidden=32)
    prior.enable()
    query = torch.stack(
        tuple(instruction.code.detach().squeeze(0) for instruction in machine.instructions[:3])
    )
    optimizer = torch.optim.AdamW(prior.parameters(), lr=3e-3)
    loss, pair_count = prior.outcome_ranking_loss(
        query,
        prior.basis_keys(machine.basis_slots),
        source_outcomes,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    target_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    target_instruction = machine.instructions[-1]
    target_query = target_instruction.code.detach().squeeze(0)
    candidate_order = machine.order_basis_candidates(prior, target_query)
    target_outcomes_by_slot = torch.tensor(
        [
            _accuracy(
                parent,
                machine,
                target_decoder,
                operation=TARGET_OPERATION,
                instructions=(target_instruction,),
                basis_slots=(slot,),
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 90_000 + slot * 101,
                credit_mode="paired_counterfactual",
            )
            for slot in range(len(machine.basis_slots))
        ]
    )
    target_outcomes = target_outcomes_by_slot[list(candidate_order)]
    threshold = 0.8
    selected = next(
        (index for index, outcome in enumerate(target_outcomes) if outcome >= threshold),
        None,
    )
    cold_selected = next(
        (
            index
            for index, outcome in enumerate(target_outcomes_by_slot)
            if outcome >= threshold
        ),
        None,
    )
    old_basis_digests = tuple(
        _module_digest(machine.basis_slots[index])
        for index in range(len(SOURCE_OPERATIONS))
    )
    pre_growth_state = {
        name: value.detach().clone() for name, value in machine.state_dict().items()
    }
    if selected is None:
        route = "grow"
        routed_basis_slot = machine.add_basis_slot()
    else:
        route = "reuse"
        routed_basis_slot = candidate_order[selected]
    _freeze(machine)
    target_instruction.code.requires_grad_(True)
    routed_basis = machine.basis_slots[routed_basis_slot]
    if route == "grow":
        for parameter in routed_basis.parameters():
            parameter.requires_grad_(True)
    target_progress = _train_stage(
        parent,
        machine,
        target_decoder,
        operation=TARGET_OPERATION,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.growth_span,
        seed=args.seed + 100_000,
        trainable=[
            target_instruction.code,
            *(routed_basis.parameters() if route == "grow" else ()),
            *target_decoder.parameters(),
        ],
        credit_mode=args.growth_credit_mode,
        learning_rate=args.growth_learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 200_000,
    )
    target_accuracy = _accuracy(
        parent,
        machine,
        target_decoder,
        operation=TARGET_OPERATION,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        count=args.audit_count,
        span=args.growth_span,
        seed=args.seed + 210_000,
        credit_mode=args.growth_credit_mode,
    )
    shuffled_target_accuracy = _accuracy(
        parent,
        machine,
        target_decoder,
        operation=TARGET_OPERATION,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        count=args.audit_count,
        span=args.growth_span,
        seed=args.seed + 211_000,
        credit_mode=args.growth_credit_mode,
        shuffle_outcomes=True,
    )
    missing_target_accuracy = _accuracy(
        parent,
        machine,
        target_decoder,
        operation=TARGET_OPERATION,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        count=args.audit_count,
        span=args.growth_span,
        seed=args.seed + 212_000,
        credit_mode=args.growth_credit_mode,
        evidence_present=False,
    )
    target_stable = _stable_bits(
        target_progress,
        threshold=threshold,
        bits_per_update=args.batch_size * args.growth_span * 2,
    )
    source_after = [
        _accuracy(
            parent,
            machine,
            decoder,
            operation=operation,
            instructions=(machine.instructions[index],),
            basis_slots=(index,),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 220_000 + index * 101,
            credit_mode="paired_counterfactual",
        )
        for index, (operation, decoder) in enumerate(
            zip(SOURCE_OPERATIONS, decoders, strict=True)
        )
    ]
    old_basis_unchanged = all(
        digest == _module_digest(machine.basis_slots[index])
        for index, digest in enumerate(old_basis_digests)
    )
    shuffled_training_machine = _new_machine(
        len(SOURCE_OPERATIONS) + 1,
        operator_mode=args.operator_mode,
    )
    for _ in SOURCE_OPERATIONS:
        shuffled_training_machine.add_basis_slot()
    shuffled_training_machine.load_state_dict(pre_growth_state, strict=True)
    shuffled_training_basis_slot = shuffled_training_machine.add_basis_slot()
    shuffled_training_instruction = shuffled_training_machine.instructions[-1]
    shuffled_training_decoder = OpaqueProtocolDecoder(
        REGISTER_WIDTH, ACTION_WIDTH, hidden=16
    )
    _freeze(shuffled_training_machine)
    shuffled_training_instruction.code.requires_grad_(True)
    shuffled_training_basis = shuffled_training_machine.basis_slots[
        shuffled_training_basis_slot
    ]
    for parameter in shuffled_training_basis.parameters():
        parameter.requires_grad_(True)
    _train_stage(
        parent,
        shuffled_training_machine,
        shuffled_training_decoder,
        operation=TARGET_OPERATION,
        instructions=(shuffled_training_instruction,),
        basis_slots=(shuffled_training_basis_slot,),
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.growth_span,
        seed=args.seed + 300_000,
        trainable=[
            shuffled_training_instruction.code,
            *shuffled_training_basis.parameters(),
            *shuffled_training_decoder.parameters(),
        ],
        credit_mode=args.growth_credit_mode,
        learning_rate=args.growth_learning_rate,
        shuffle_outcomes=True,
    )
    shuffled_training_accuracy = _accuracy(
        parent,
        shuffled_training_machine,
        shuffled_training_decoder,
        operation=TARGET_OPERATION,
        instructions=(shuffled_training_instruction,),
        basis_slots=(shuffled_training_basis_slot,),
        count=args.audit_count,
        span=args.growth_span,
        seed=args.seed + 310_000,
        credit_mode=args.growth_credit_mode,
    )
    report = {
        "schema": "neural-computer.external-register-real-basis-acquisition-audit.v1",
        "claim_boundary": "A learned opaque prior orders real primitive basis trials; fresh verifier outcomes still determine admission.",
        "seed": args.seed,
        "source_operations": list(SOURCE_OPERATIONS),
        "target_operation": TARGET_OPERATION,
        "growth_credit_mode": args.growth_credit_mode,
        "source_outcomes": source_outcomes.tolist(),
        "target_candidate_order": list(candidate_order),
        "target_candidate_outcomes": target_outcomes.tolist(),
        "pair_count": pair_count,
        "selected_candidate_position": selected,
        "cold_candidate_position": cold_selected,
        "route": route,
        "routed_basis_slot": routed_basis_slot,
        "target_accuracy": target_accuracy,
        "shuffled_target_accuracy": shuffled_target_accuracy,
        "missing_target_accuracy": missing_target_accuracy,
        "shuffled_training_target_accuracy": shuffled_training_accuracy,
        "target_stable_bits": target_stable,
        "target_progress": target_progress,
        "source_accuracy_after": source_after,
        "old_basis_unchanged": old_basis_unchanged,
        "accounting": {
            "replayed_examples": 0,
            "optimizer_updates": (
                args.parent_updates + args.source_updates * 3 + 1 + args.target_updates
            ),
        },
        "gates": {
            "source_rows_have_distinct_outcomes": bool(
                source_outcomes.max(dim=1).values.min() > source_outcomes.min(dim=1).values.max()
            ),
            "fresh_verifier_controls_admission": True,
            "unseen_target_requests_growth": (
                selected is None and cold_selected is None
            ),
            "target_mastered": target_accuracy >= threshold,
            "target_stable": target_stable is not None,
            "source_retained": min(source_after) >= threshold,
            "old_basis_unchanged": old_basis_unchanged,
            "reward_shuffled_rejected": shuffled_target_accuracy < threshold,
            "missing_evidence_rejected": missing_target_accuracy < threshold,
            "shuffled_training_rejected": shuffled_training_accuracy < threshold,
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
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates", type=int, default=96)
    parser.add_argument("--target-updates", type=int, default=128)
    parser.add_argument(
        "--growth-credit-mode",
        choices=("attempted_bce", "reinforce", "paired_counterfactual"),
        default="attempted_bce",
    )
    parser.add_argument("--growth-learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--growth-span", type=int, default=2)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--operator-mode", default="factorized_bounded_residual")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
