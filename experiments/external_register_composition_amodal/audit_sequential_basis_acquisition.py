"""Audit sequential acquisition of two unseen external capabilities."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    AmodalEventBridge,
    ExternalRegisterInstruction,
    OpaqueProtocolDecoder,
)

from .audit_real_basis_acquisition import (
    SOURCE_OPERATIONS,
    _accuracy_matrix,
    _freeze,
    _train_source,
)
from .train import (
    ACTION_WIDTH,
    EVENT_WIDTH,
    INSTRUCTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _new_machine,
    _train_stage,
)

TARGET_OPERATIONS = ("rotate", "prefix_parity", "global_parity")
MASTERY_THRESHOLD = 0.8


def _target_spec(
    operation: str,
    instruction,
    decoder,
    basis_slot: int,
    event_bridge: AmodalEventBridge,
) -> tuple[str, object, object, int, AmodalEventBridge]:
    return operation, instruction, decoder, basis_slot, event_bridge


def _score_spec(
    parent,
    machine,
    spec,
    *,
    count: int,
    span: int,
    seed: int,
    reverse_operations: bool = False,
    reverse_sequence: bool = False,
) -> float:
    operation, instruction, decoder, basis_slot, event_bridge = spec
    return _accuracy(
        parent,
        machine,
        decoder,
        operation=operation,
        instructions=(instruction,),
        basis_slots=(basis_slot,),
        count=count,
        span=span,
        seed=seed,
        credit_mode="attempted_bce",
        event_bridge=event_bridge,
        reverse_operations=reverse_operations,
        reverse_sequence=reverse_sequence,
    )


def _acquire_target(
    parent,
    machine,
    retained_specs: list[tuple[str, object, object, int, AmodalEventBridge]],
    *,
    target_index: int,
    operation: str,
    args: argparse.Namespace,
) -> tuple[dict[str, object], tuple[str, object, object, int, AmodalEventBridge] | None]:
    target_instruction = machine.instructions[target_index]
    target_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    target_bridge = AmodalEventBridge(
        EVENT_WIDTH,
        parent.controller.width,
        EVENT_WIDTH,
        hidden=64,
    )
    pre_target_machine = copy.deepcopy(machine)
    pre_target_instruction = target_instruction.code.detach().clone()
    retained_before = [
        _score_spec(
            parent,
            machine,
            spec,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + target_index * 100_000 + 500_000 + index * 1_009,
        )
        for index, spec in enumerate(retained_specs)
    ]
    routed_basis_slot = machine.add_basis_slot()
    routed_basis = machine.basis_slots[routed_basis_slot]
    _freeze(machine)
    target_instruction.code.requires_grad_(True)
    for parameter in routed_basis.parameters():
        parameter.requires_grad_(True)

    trainable = [
        target_instruction.code,
        *routed_basis.parameters(),
        *target_decoder.parameters(),
        *target_bridge.parameters(),
    ]
    warmup = _train_stage(
        parent,
        machine,
        target_decoder,
        operation=operation,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        updates=args.growth_warmup_updates,
        batch_size=args.batch_size,
        span=args.growth_span,
        seed=args.seed + target_index * 100_000 + 100_000,
        trainable=trainable,
        credit_mode=args.growth_credit_mode,
        learning_rate=args.growth_learning_rate,
        event_bridge=target_bridge,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + target_index * 100_000 + 200_000,
    )
    for parameter in target_decoder.parameters():
        parameter.requires_grad_(False)
    focus_trainable = [target_instruction.code, *routed_basis.parameters()]
    focus_trainable.extend(target_bridge.parameters())
    focus = _train_stage(
        parent,
        machine,
        target_decoder,
        operation=operation,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        updates=args.growth_basis_focus_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + target_index * 100_000 + 105_000,
        trainable=focus_trainable,
        credit_mode=args.growth_credit_mode,
        learning_rate=args.growth_learning_rate,
        event_bridge=target_bridge,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + target_index * 100_000 + 215_000,
    )
    for parameter in target_decoder.parameters():
        parameter.requires_grad_(True)
    full = _train_stage(
        parent,
        machine,
        target_decoder,
        operation=operation,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + target_index * 100_000 + 110_000,
        trainable=trainable,
        credit_mode=args.growth_credit_mode,
        learning_rate=args.growth_learning_rate,
        event_bridge=target_bridge,
        restore_best_checkpoint=True,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + target_index * 100_000 + 220_000,
    )

    candidate_spec = _target_spec(
        operation,
        target_instruction,
        target_decoder,
        routed_basis_slot,
        target_bridge,
    )
    probe_scores = [
        _score_spec(
            parent,
            machine,
            candidate_spec,
            count=args.consolidation_audit_count,
            span=args.span,
            seed=args.seed + target_index * 100_000 + 400_000 + probe * 1_009,
        )
        for probe in range(args.consolidation_probes)
    ]
    candidate_score = _score_spec(
        parent,
        machine,
        candidate_spec,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + target_index * 100_000 + 410_000,
    )
    missing_score = _accuracy(
        parent,
        machine,
        target_decoder,
        operation=operation,
        instructions=(target_instruction,),
        basis_slots=(routed_basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + target_index * 100_000 + 412_000,
        credit_mode=args.growth_credit_mode,
        evidence_present=False,
        event_bridge=target_bridge,
    )

    shuffled_machine = pre_target_machine
    shuffled_basis_slot = shuffled_machine.add_basis_slot()
    shuffled_instruction = shuffled_machine.instructions[target_index]
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    shuffled_bridge = AmodalEventBridge(
        EVENT_WIDTH,
        parent.controller.width,
        EVENT_WIDTH,
        hidden=64,
    )
    _freeze(shuffled_machine)
    shuffled_instruction.code.requires_grad_(True)
    shuffled_basis = shuffled_machine.basis_slots[shuffled_basis_slot]
    for parameter in shuffled_basis.parameters():
        parameter.requires_grad_(True)
    shuffled_trainable = [
        shuffled_instruction.code,
        *shuffled_basis.parameters(),
        *shuffled_decoder.parameters(),
        *shuffled_bridge.parameters(),
    ]
    _train_stage(
        parent,
        shuffled_machine,
        shuffled_decoder,
        operation=operation,
        instructions=(shuffled_instruction,),
        basis_slots=(shuffled_basis_slot,),
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + target_index * 100_000 + 300_000,
        trainable=shuffled_trainable,
        credit_mode=args.growth_credit_mode,
        learning_rate=args.growth_learning_rate,
        shuffle_outcomes=True,
        event_bridge=shuffled_bridge,
    )
    shuffled_score = _accuracy(
        parent,
        shuffled_machine,
        shuffled_decoder,
        operation=operation,
        instructions=(shuffled_instruction,),
        basis_slots=(shuffled_basis_slot,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + target_index * 100_000 + 310_000,
        credit_mode=args.growth_credit_mode,
        event_bridge=shuffled_bridge,
    )

    retained_after = [
        _score_spec(
            parent,
            machine,
            spec,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + target_index * 100_000 + 500_000 + index * 1_009,
        )
        for index, spec in enumerate(retained_specs)
    ]
    retention_deltas = [
        after - before
        for before, after in zip(retained_before, retained_after, strict=True)
    ]
    candidate_stable = bool(
        len(probe_scores) >= args.consolidation_probes
        and min(probe_scores) >= MASTERY_THRESHOLD
    )
    retention_passed = bool(
        retained_after and min(retained_after) >= MASTERY_THRESHOLD
    )
    accepted = bool(
        candidate_stable
        and retention_passed
        and min(retention_deltas, default=0.0)
        >= -args.retention_regression_tolerance
        and shuffled_score < MASTERY_THRESHOLD
        and missing_score < MASTERY_THRESHOLD
    )
    rollback = False
    if accepted:
        machine.freeze_basis_slot(routed_basis_slot)
        target_instruction.code.requires_grad_(False)
    else:
        machine.remove_basis_slot(routed_basis_slot)
        with torch.no_grad():
            target_instruction.code.copy_(pre_target_instruction)
        rollback = True
    result = {
        "operation": operation,
        "target_index": target_index,
        "candidate_accuracy": candidate_score,
        "consolidation_probe_scores": probe_scores,
        "candidate_stable": candidate_stable,
        "missing_evidence_accuracy": missing_score,
        "shuffled_training_accuracy": shuffled_score,
        "retained_before": retained_before,
        "retained_after": retained_after,
        "retention_deltas": retention_deltas,
        "retention_passed": retention_passed,
        "accepted": accepted,
        "rollback_applied": rollback,
        "warmup_progress": warmup,
        "basis_focus_progress": focus,
        "target_progress": full,
    }
    return result, candidate_spec if accepted else None


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    if args.candidate_restarts < 1:
        raise ValueError("candidate restarts must be positive")
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
    machine = _new_machine(
        len(SOURCE_OPERATIONS) + 1,
        operator_mode=args.operator_mode,
        basis_hidden=args.basis_hidden,
        basis_microsteps=args.basis_microsteps,
        basis_event_read_mode=args.basis_event_read_mode,
        event_width=EVENT_WIDTH,
        event_input_mode="frontend",
    )
    for _ in SOURCE_OPERATIONS:
        machine.add_basis_slot()
    source_decoders = [
        OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        for _ in SOURCE_OPERATIONS
    ]
    for index, decoder in enumerate(source_decoders):
        _train_source(parent, machine, decoder, index, args)
    source_outcomes = _accuracy_matrix(
        parent,
        machine,
        source_decoders,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 70_000,
    )
    # Source decoders do not use event bridges; retained source specs use the
    # direct standardized frontend event boundary.
    retained_specs = []
    for index, (operation, decoder) in enumerate(
        zip(SOURCE_OPERATIONS, source_decoders, strict=True)
    ):
        retained_specs.append(
            (operation, machine.instructions[index], decoder, index, None)
        )

    # _score_spec accepts an event bridge, while source events use the direct
    # frontend boundary. Keep this explicit so target bridges cannot leak into
    # inherited source evaluation.
    def score_retained(spec, seed: int) -> float:
        operation, instruction, decoder, basis_slot, bridge = spec
        return _accuracy(
            parent,
            machine,
            decoder,
            operation=operation,
            instructions=(instruction,),
            basis_slots=(basis_slot,),
            count=args.audit_count,
            span=args.span,
            seed=seed,
            credit_mode="paired_counterfactual",
        )

    initial_source_scores = [
        score_retained(spec, args.seed + 70_000 + index * 101 + index * 1009)
        for index, spec in enumerate(retained_specs)
    ]
    targets: list[dict[str, object]] = []
    if min(initial_source_scores, default=0.0) >= MASTERY_THRESHOLD:
        for target_stage_index, operation in enumerate(TARGET_OPERATIONS):
            if target_stage_index:
                machine.add_instruction(ExternalRegisterInstruction(INSTRUCTION_WIDTH))
            target_index = len(machine.instructions) - 1
            attempts: list[dict[str, object]] = []
            accepted_spec = None
            result = None
            for attempt in range(args.candidate_restarts):
                attempt_args = copy.copy(args)
                attempt_args.seed = args.seed + attempt * 1_000_000
                attempt_result, attempt_spec = _acquire_target(
                    parent,
                    machine,
                    retained_specs,
                    target_index=target_index,
                    operation=operation,
                    args=attempt_args,
                )
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "accepted": attempt_result["accepted"],
                        "candidate_accuracy": attempt_result["candidate_accuracy"],
                        "consolidation_probe_scores": attempt_result[
                            "consolidation_probe_scores"
                        ],
                        "retained_after": attempt_result["retained_after"],
                    }
                )
                result = attempt_result
                accepted_spec = attempt_spec
                if accepted_spec is not None:
                    break
            assert result is not None
            result["candidate_attempt_count"] = len(attempts)
            result["candidate_attempts"] = attempts
            targets.append(result)
            if accepted_spec is None:
                break
            retained_specs.append(accepted_spec)
    report = {
        "schema": "neural-computer.external-register-sequential-basis-acquisition-audit.v1",
        "seed": args.seed,
        "source_operations": list(SOURCE_OPERATIONS),
        "target_operations": list(TARGET_OPERATIONS),
        "source_updates": args.source_updates,
        "target_updates": args.target_updates,
        "source_outcomes": source_outcomes.tolist(),
        "initial_source_scores": initial_source_scores,
        "targets": targets,
        "promoted_target_count": sum(bool(target["accepted"]) for target in targets),
        "accounting": {
            "replayed_examples": 0,
            "parent_unique_verifier_bits": args.parent_updates * args.batch_size * 2,
            "unique_verifier_bits": 0,
        },
    }
    reversal_probes = []
    if report["promoted_target_count"] == len(TARGET_OPERATIONS):
        for index, spec in enumerate(retained_specs):
            if index < len(SOURCE_OPERATIONS):
                continue
            reversal_probes.append(
                {
                    "operation": spec[0],
                    "normal_accuracy": _score_spec(
                        parent,
                        machine,
                        spec,
                        count=args.consolidation_audit_count,
                        span=args.span,
                        seed=args.seed + 900_000 + index * 1_009,
                    ),
                    "sequence_reversal_accuracy": _score_spec(
                        parent,
                        machine,
                        spec,
                        count=args.consolidation_audit_count,
                        span=args.span,
                        seed=args.seed + 900_000 + index * 1_009,
                        reverse_sequence=True,
                    ),
                }
            )
    report["reversal_probes"] = reversal_probes
    report["sequence_reversal_pressure_detected"] = bool(
        reversal_probes
        and all(
            probe["normal_accuracy"] >= MASTERY_THRESHOLD
            and probe["sequence_reversal_accuracy"] < (
                probe["normal_accuracy"] - 0.1
            )
            for probe in reversal_probes
        )
    )
    reversal_bits = (
        len(reversal_probes)
        * args.consolidation_audit_count
        * args.span
        * 4
    )
    reversal_lifetimes = (
        len(reversal_probes) * args.consolidation_audit_count * 2
    )
    target_count = len(targets)
    candidate_attempt_count = sum(
        int(target["candidate_attempt_count"]) for target in targets
    )
    source_selection_evaluations = (
        (args.source_updates + args.eval_every - 1) // args.eval_every
    )
    source_train_bits = (
        args.source_updates
        * len(SOURCE_OPERATIONS)
        * args.batch_size
        * args.span
        * 2
    )
    source_selection_bits = (
        source_selection_evaluations
        * len(SOURCE_OPERATIONS)
        * args.source_selection_audit_count
        * args.span
        * 2
    )
    target_stage_bits = (
        candidate_attempt_count
        * (
            args.growth_warmup_updates * args.growth_span
            + (args.growth_basis_focus_updates + args.target_updates) * args.span
        )
        * args.batch_size
        * 2
    )
    shuffled_control_bits = (
        candidate_attempt_count
        * args.target_updates
        * args.batch_size
        * args.span
        * 2
    )
    consolidation_bits = (
        candidate_attempt_count
        * args.consolidation_probes
        * args.consolidation_audit_count
        * args.span
        * 2
    )
    report["accounting"].update(
        {
            "source_train_verifier_bits": source_train_bits,
            "source_selection_verifier_bits": source_selection_bits,
            "target_stage_verifier_bits": target_stage_bits,
            "shuffled_control_verifier_bits": shuffled_control_bits,
            "consolidation_verifier_bits": consolidation_bits,
            "reversal_verifier_bits": reversal_bits,
            "unique_verifier_bits": (
                report["accounting"]["parent_unique_verifier_bits"]
                + source_train_bits
                + source_selection_bits
                + target_stage_bits
                + shuffled_control_bits
                + consolidation_bits
                + reversal_bits
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.source_updates * len(SOURCE_OPERATIONS)
                + candidate_attempt_count
                * (
                    args.growth_warmup_updates
                    + args.growth_basis_focus_updates
                    + args.target_updates
                    + args.target_updates
                )
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.source_updates
                * len(SOURCE_OPERATIONS)
                * args.batch_size
                + source_selection_evaluations
                * len(SOURCE_OPERATIONS)
                * args.source_selection_audit_count
                + candidate_attempt_count
                * (
                    (args.growth_warmup_updates + args.growth_basis_focus_updates
                     + args.target_updates)
                    * args.batch_size
                    + args.target_updates * args.batch_size
                    + args.consolidation_probes * args.consolidation_audit_count
                )
                + reversal_lifetimes
            ),
        }
    )
    report["promoted"] = len(targets) == len(TARGET_OPERATIONS) and all(
        bool(target["accepted"]) for target in targets
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--growth-warmup-updates", type=int, default=64)
    parser.add_argument("--growth-basis-focus-updates", type=int, default=64)
    parser.add_argument("--target-updates", type=int, default=512)
    parser.add_argument("--candidate-restarts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--growth-span", type=int, default=2)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--source-selection-audit-count", type=int, default=64)
    parser.add_argument(
        "--restore-best-source-checkpoint", action="store_true", default=True
    )
    parser.add_argument("--consolidation-probes", type=int, default=4)
    parser.add_argument("--consolidation-audit-count", type=int, default=64)
    parser.add_argument("--operator-mode", default="factorized_bounded_residual")
    parser.add_argument("--growth-credit-mode", default="attempted_bce")
    parser.add_argument("--growth-learning-rate", type=float, default=1e-3)
    parser.add_argument("--basis-hidden", type=int, default=64)
    parser.add_argument("--basis-microsteps", type=int, default=1)
    parser.add_argument("--basis-event-read-mode", default="flattened_window")
    parser.add_argument("--retention-regression-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
