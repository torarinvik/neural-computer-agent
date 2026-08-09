"""Audit interleaved acquisition of multiple mutable external capabilities.

Two fresh capabilities are trained in alternating updates after the source
memory is frozen.  Each candidate owns its instruction, compute basis,
decoder, and event bridge, but both candidates share the frozen controller and
inherited external memory.  The candidates are admitted transactionally only
when both pass the same causal and retention gates used by sequential growth.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import permutations
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
    _freeze,
    _train_source,
)
from .train import (
    ACTION_WIDTH,
    EVENT_WIDTH,
    INSTRUCTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _batch,
    _new_machine,
    _rollout,
)

TARGET_OPERATIONS = ("complement_rotate", "prefix_parity")
COMPOSITION_PROGRAMS = tuple(permutations(SOURCE_OPERATIONS))
MASTERY_THRESHOLD = 0.8


def _score(
    parent,
    machine,
    candidate: dict[str, object],
    *,
    count: int,
    span: int,
    seed: int,
    shuffle_outcomes: bool = False,
    evidence_present: bool = True,
) -> float:
    return _accuracy(
        parent,
        machine,
        candidate["decoder"],
        operation=candidate["operation"],
        instructions=candidate["instructions"],
        basis_slots=candidate["basis_slots"],
        count=count,
        span=span,
        seed=seed,
        credit_mode="attempted_bce",
        shuffle_outcomes=shuffle_outcomes,
        evidence_present=evidence_present,
        event_bridge=candidate["bridge"],
        generated_composition_ids=candidate.get("generated_composition_ids"),
        generated_compositions=candidate.get("generated_compositions"),
    )


def _train_interleaved_phase(
    parent,
    machine,
    candidates: list[dict[str, object]],
    *,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    learning_rate: float,
    eval_every: int,
    audit_count: int,
    audit_seed: int,
    decoder_trainable: bool,
    shuffle_outcomes: bool = False,
) -> None:
    """Train each candidate on alternating local updates.

    Optimizers remain separate and persistent for the whole phase.  Thus the
    alternation is not simulated by repeatedly reinitializing an optimizer,
    and each candidate's moment estimates remain isolated from the other.
    """
    optimizers = []
    for candidate in candidates:
        decoder = candidate["decoder"]
        trainable = []
        if candidate.get("mutable", True):
            basis = machine.basis_slots[candidate["basis_slot"]]
            trainable.extend([candidate["instruction"].code, *basis.parameters()])
        trainable.extend(candidate["bridge"].parameters())
        if decoder_trainable:
            trainable.extend(decoder.parameters())
        optimizers.append(torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5))
        candidate["phase_trainable"] = trainable
        candidate["phase_optimizer"] = optimizers[-1]
        candidate["phase_progress"] = []
        candidate["phase_best"] = None
        candidate["phase_best_accuracy"] = float("-inf")

    for global_update in range(1, updates * len(candidates) + 1):
        index = (global_update - 1) % len(candidates)
        local_update = (global_update - 1) // len(candidates) + 1
        candidate = candidates[index]
        batch = _batch(
            candidate["operation"],
            count=batch_size,
            span=span,
            seed=seed + index * 100_003 + local_update * 10_007,
            generated_composition_ids=candidate.get("generated_composition_ids"),
            generated_compositions=candidate.get("generated_compositions"),
        )
        loss, _ = _rollout(
            parent,
            machine,
            candidate["decoder"],
            batch,
            candidate["instructions"],
            basis_slots=candidate["basis_slots"],
            train_decoder=True,
            credit_mode="attempted_bce",
            shuffle_outcomes=shuffle_outcomes,
            event_bridge=candidate["bridge"],
        )
        optimizer = candidate["phase_optimizer"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate["phase_trainable"], 1.0)
        optimizer.step()
        if local_update % eval_every == 0 or local_update == updates:
            accuracy = _score(
                parent,
                machine,
                candidate,
                count=audit_count,
                span=span,
                seed=audit_seed + index * 1_009,
            )
            candidate["phase_progress"].append(
                {"update": local_update, "heldout_accuracy": accuracy}
            )
            if accuracy > candidate["phase_best_accuracy"]:
                candidate["phase_best_accuracy"] = accuracy
                candidate["phase_best"] = [
                    parameter.detach().clone()
                    for parameter in candidate["phase_trainable"]
                ]

    for candidate in candidates:
        if candidate["phase_best"] is not None:
            with torch.no_grad():
                for parameter, snapshot in zip(
                    candidate["phase_trainable"],
                    candidate["phase_best"],
                    strict=True,
                ):
                    parameter.copy_(snapshot)


def _prepare_candidates(
    parent,
    machine,
    operations: tuple[str, ...],
    *,
    include_composition: bool,
) -> list[dict[str, object]]:
    candidates = []
    for index, operation in enumerate(operations):
        instruction = machine.instructions[len(SOURCE_OPERATIONS) + index]
        basis_slot = machine.add_basis_slot()
        candidates.append(
            {
                "operation": operation,
                "instruction": instruction,
                "basis_slot": basis_slot,
                "instructions": (instruction,),
                "basis_slots": (basis_slot,),
                "mutable": True,
                "decoder": OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16),
                "bridge": AmodalEventBridge(
                    EVENT_WIDTH, parent.controller.width, EVENT_WIDTH, hidden=64
                ),
            }
        )
    if include_composition:
        # Each candidate has no new instruction or basis.  It is a fresh
        # decoder/bridge learning to expose a different generated program
        # executed by the frozen source capabilities, making the test about
        # compositional reuse rather than another direct primitive.
        for program in COMPOSITION_PROGRAMS:
            source_indices = tuple(
                SOURCE_OPERATIONS.index(primitive) for primitive in program
            )
            candidates.append(
                {
                    "operation": "generated_composition",
                    "instructions": tuple(
                        machine.instructions[index] for index in source_indices
                    ),
                    "basis_slots": source_indices,
                    "generated_composition_ids": (0,),
                    "generated_compositions": (program,),
                    "composition_program": program,
                    "mutable": False,
                    "decoder": OpaqueProtocolDecoder(
                        REGISTER_WIDTH, ACTION_WIDTH, hidden=16
                    ),
                    "bridge": AmodalEventBridge(
                        EVENT_WIDTH, parent.controller.width, EVENT_WIDTH, hidden=64
                    ),
                }
            )
    return candidates


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    target_operations = tuple(
        operation for operation in args.target_operations.split(",") if operation
    )
    if not target_operations:
        raise ValueError("at least one target operation is required")
    if args.source_restarts < 1:
        raise ValueError("source restarts must be positive")
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
        len(SOURCE_OPERATIONS),
        operator_mode=args.operator_mode,
    )
    for _ in SOURCE_OPERATIONS:
        machine.add_basis_slot()
    source_decoders = []
    source_attempt_counts = []
    for index in range(len(SOURCE_OPERATIONS)):
        source_parent = copy.deepcopy(machine)
        best_machine_state = None
        best_decoder_state = None
        best_score = float("-inf")
        attempts = []
        for attempt in range(args.source_restarts):
            candidate_machine = copy.deepcopy(source_parent)
            candidate_decoder = OpaqueProtocolDecoder(
                REGISTER_WIDTH, ACTION_WIDTH, hidden=16
            )
            attempt_args = copy.copy(args)
            attempt_args.seed = args.seed + index * 100_003 + attempt * 1_000_000
            _train_source(
                parent, candidate_machine, candidate_decoder, index, attempt_args
            )
            candidate_score = _accuracy(
                parent,
                candidate_machine,
                candidate_decoder,
                operation=SOURCE_OPERATIONS[index],
                instructions=(candidate_machine.instructions[index],),
                basis_slots=(index,),
                count=args.source_selection_audit_count,
                span=args.span,
                seed=args.seed + 60_000 + index * 101 + index * 1009,
                credit_mode="paired_counterfactual",
            )
            attempts.append(candidate_score)
            if candidate_score > best_score:
                best_score = candidate_score
                best_machine_state = {
                    name: value.detach().clone()
                    for name, value in candidate_machine.state_dict().items()
                }
                best_decoder_state = {
                    name: value.detach().clone()
                    for name, value in candidate_decoder.state_dict().items()
                }
        assert best_machine_state is not None and best_decoder_state is not None
        machine.load_state_dict(best_machine_state, strict=True)
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        decoder.load_state_dict(best_decoder_state, strict=True)
        source_decoders.append(decoder)
        source_attempt_counts.append(attempts)

    source_specs = [
        (operation, machine.instructions[index], decoder, index)
        for index, (operation, decoder) in enumerate(
            zip(SOURCE_OPERATIONS, source_decoders, strict=True)
        )
    ]

    def source_score(spec, index: int) -> float:
        operation, instruction, decoder, basis_slot = spec
        return _accuracy(
            parent,
            machine,
            decoder,
            operation=operation,
            instructions=(instruction,),
            basis_slots=(basis_slot,),
            count=args.source_selection_audit_count,
            span=args.span,
            seed=args.seed + 60_000 + index * 101 + index * 1009,
            credit_mode="paired_counterfactual",
        )

    source_before = [source_score(spec, index) for index, spec in enumerate(source_specs)]
    source_attempt_count = sum(len(attempts) for attempts in source_attempt_counts)
    # Do not allocate future target instructions until source acquisition is
    # complete.  The first source phase trains machine parameters jointly; if
    # future capacity exists already, it becomes an accidental optimizer
    # participant and source quality depends on the number of capabilities we
    # happened to reserve for later.
    for _ in target_operations:
        machine.add_instruction(ExternalRegisterInstruction(INSTRUCTION_WIDTH))
    pre_growth_machine = copy.deepcopy(machine)
    candidates = _prepare_candidates(
        parent,
        machine,
        target_operations,
        include_composition=args.include_composition,
    )
    retained_before = [
        source_score(spec, index) for index, spec in enumerate(source_specs)
    ]
    _freeze(machine)
    for candidate in candidates:
        if candidate.get("mutable", True):
            candidate["instruction"].code.requires_grad_(True)
            for parameter in machine.basis_slots[candidate["basis_slot"]].parameters():
                parameter.requires_grad_(True)

    _train_interleaved_phase(
        parent,
        machine,
        candidates,
        updates=args.warmup_updates,
        batch_size=args.batch_size,
        span=args.warmup_span,
        seed=args.seed + 100_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 200_000,
        decoder_trainable=True,
    )
    for candidate in candidates:
        for parameter in candidate["decoder"].parameters():
            parameter.requires_grad_(False)
    _train_interleaved_phase(
        parent,
        machine,
        candidates,
        updates=args.focus_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 300_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 400_000,
        decoder_trainable=False,
    )
    for candidate in candidates:
        for parameter in candidate["decoder"].parameters():
            parameter.requires_grad_(True)
    _train_interleaved_phase(
        parent,
        machine,
        candidates,
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 500_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 600_000,
        decoder_trainable=True,
    )

    shuffled_machine = copy.deepcopy(pre_growth_machine)
    shuffled_candidates = _prepare_candidates(
        parent,
        shuffled_machine,
        target_operations,
        include_composition=args.include_composition,
    )
    _freeze(shuffled_machine)
    for candidate in shuffled_candidates:
        if candidate.get("mutable", True):
            candidate["instruction"].code.requires_grad_(True)
            for parameter in shuffled_machine.basis_slots[candidate["basis_slot"]].parameters():
                parameter.requires_grad_(True)
    for phase_updates, phase_span, phase_seed, decoder_trainable in (
        (args.warmup_updates, args.warmup_span, args.seed + 800_000, True),
        (args.focus_updates, args.span, args.seed + 900_000, False),
        (args.target_updates, args.span, args.seed + 1_000_000, True),
    ):
        _train_interleaved_phase(
            parent,
            shuffled_machine,
            shuffled_candidates,
            updates=phase_updates,
            batch_size=args.batch_size,
            span=phase_span,
            seed=phase_seed,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            audit_seed=phase_seed + 100_000,
            decoder_trainable=decoder_trainable,
            shuffle_outcomes=True,
        )

    records = []
    for index, candidate in enumerate(candidates):
        retained_after = [
            source_score(spec, offset)
            for offset, spec in enumerate(source_specs)
        ]
        score = _score(
            parent,
            machine,
            candidate,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 700_000 + index * 1_009,
        )
        missing = _score(
            parent,
            machine,
            candidate,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 720_000 + index * 1_009,
            evidence_present=False,
        )
        shuffled = _score(
            parent,
            shuffled_machine,
            shuffled_candidates[index],
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 740_000 + index * 1_009,
        )
        deltas = [after - before for before, after in zip(retained_before, retained_after, strict=True)]
        probe = _score(
            parent,
            machine,
            candidate,
            count=args.consolidation_audit_count,
            span=args.span,
            seed=args.seed + 730_000 + index * 1_009,
        )
        records.append(
            {
                "operation": candidate["operation"],
                "composition_program": list(candidate["composition_program"])
                if candidate.get("composition_program") is not None
                else None,
                "candidate_accuracy": score,
                "consolidation_probe_accuracy": probe,
                "shuffled_training_accuracy": shuffled,
                "missing_evidence_accuracy": missing,
                "retained_before": retained_before,
                "retained_after": retained_after,
                "retention_deltas": deltas,
                "candidate_progress": candidate["phase_progress"],
                "accepted": bool(
                    probe >= MASTERY_THRESHOLD
                    and min(retained_after, default=0.0) >= MASTERY_THRESHOLD
                    and min(deltas, default=0.0) >= -args.retention_regression_tolerance
                    and shuffled < MASTERY_THRESHOLD
                    and missing < MASTERY_THRESHOLD
                ),
            }
        )

    accepted = all(record["accepted"] for record in records)
    if accepted:
        for candidate in candidates:
            if candidate.get("mutable", True):
                machine.freeze_basis_slot(candidate["basis_slot"])
                candidate["instruction"].code.requires_grad_(False)
    else:
        # The transaction is all-or-nothing: concurrent mutable growth cannot
        # leave one candidate admitted when the paired pressure test fails.
        machine = pre_growth_machine

    candidate_count = len(candidates)
    source_selection_evaluations = (
        (args.source_updates + args.eval_every - 1) // args.eval_every
    )
    phase_audit_lifetimes = candidate_count * (
        (args.warmup_updates + args.eval_every - 1) // args.eval_every * args.audit_count
        + (args.focus_updates + args.eval_every - 1) // args.eval_every * args.audit_count
        + (args.target_updates + args.eval_every - 1) // args.eval_every * args.audit_count
    )
    normal_target_training_lifetimes = candidate_count * (
        args.warmup_updates * args.batch_size
        + (args.focus_updates + args.target_updates) * args.batch_size
    )
    shuffled_target_training_lifetimes = normal_target_training_lifetimes
    audit_lifetimes = (
        candidate_count * args.audit_count * 3
        + candidate_count * args.consolidation_audit_count
        + len(source_specs) * args.source_selection_audit_count * 3
        + source_selection_evaluations * source_attempt_count * args.source_selection_audit_count
        + phase_audit_lifetimes * 2
    )
    logical_lifetimes = (
        args.parent_updates * args.batch_size
        + args.source_updates * source_attempt_count * args.batch_size
        + normal_target_training_lifetimes
        + shuffled_target_training_lifetimes
        + audit_lifetimes
    )
    parent_training_bits = args.parent_updates * args.batch_size * 2 * 2
    source_training_bits = (
        args.source_updates * source_attempt_count * args.batch_size * args.span * 2
    )
    source_selection_bits = (
        source_selection_evaluations
        * source_attempt_count
        * args.source_selection_audit_count
        * args.span
        * 2
    )
    normal_target_training_bits = candidate_count * (
        args.warmup_updates * args.batch_size * args.warmup_span * 2
        + (args.focus_updates + args.target_updates) * args.batch_size * args.span * 2
    )
    shuffled_target_training_bits = normal_target_training_bits
    progress_audit_bits = candidate_count * 2 * (
        (args.warmup_updates + args.eval_every - 1) // args.eval_every
        * args.audit_count * args.warmup_span * 2
        + (args.focus_updates + args.eval_every - 1) // args.eval_every
        * args.audit_count * args.span * 2
        + (args.target_updates + args.eval_every - 1) // args.eval_every
        * args.audit_count * args.span * 2
    )
    suite_and_control_bits = (
        candidate_count * args.audit_count * 3 * args.span * 2
        + candidate_count * args.consolidation_audit_count * args.span * 2
        + len(source_specs) * args.source_selection_audit_count * 3 * args.span * 2
    )
    unique_verifier_bits = (
        parent_training_bits
        + source_training_bits
        + source_selection_bits
        + normal_target_training_bits
        + shuffled_target_training_bits
        + progress_audit_bits
        + suite_and_control_bits
    )
    report = {
        "schema": "neural-computer.external-register-interleaved-basis-acquisition-audit.v1",
        "seed": args.seed,
        "source_operations": list(SOURCE_OPERATIONS),
        "target_operations": list(target_operations),
        "composition_programs": [list(program) for program in COMPOSITION_PROGRAMS]
        if args.include_composition
        else [],
        "source_before": source_before,
        "source_attempt_scores": source_attempt_counts,
        "retained_before": retained_before,
        "targets": records,
        "interleaving": {
            "schedule": "round_robin_per_local_update",
            "candidate_count": len(candidates),
            "separate_optimizer_state": True,
            "transactional_admission": True,
        },
        "accounting": {
            "replayed_examples": 0,
            "parent_training_verifier_bits": parent_training_bits,
            "source_training_verifier_bits": source_training_bits,
            "source_selection_verifier_bits": source_selection_bits,
            "normal_target_training_verifier_bits": normal_target_training_bits,
            "shuffled_target_training_verifier_bits": shuffled_target_training_bits,
            "progress_audit_verifier_bits": progress_audit_bits,
            "suite_and_control_verifier_bits": suite_and_control_bits,
            "unique_verifier_bits": unique_verifier_bits,
            "optimizer_updates": (
                args.parent_updates
                + args.source_updates * source_attempt_count
                + candidate_count * (args.warmup_updates + args.focus_updates + args.target_updates)
                + candidate_count * (args.warmup_updates + args.focus_updates + args.target_updates)
            ),
            "unique_logical_lifetimes": logical_lifetimes,
        },
        "promoted": accepted,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--source-restarts", type=int, default=1)
    parser.add_argument("--warmup-updates", type=int, default=64)
    parser.add_argument("--focus-updates", type=int, default=64)
    parser.add_argument("--target-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--warmup-span", type=int, default=2)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--source-selection-audit-count", type=int, default=64)
    parser.add_argument("--consolidation-audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument(
        "--restore-best-source-checkpoint", action="store_true", default=True
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--operator-mode", default="factorized_low_rank")
    parser.add_argument(
        "--target-operations", default=",".join(TARGET_OPERATIONS)
    )
    parser.add_argument(
        "--include-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="interleave a fresh decoder/bridge for a frozen source-program composition",
    )
    parser.add_argument("--retention-regression-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
