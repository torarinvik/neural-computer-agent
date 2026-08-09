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
    CapabilityConditionedEventBridge,
    CanonicalRegisterReadout,
    ConditionedOpaqueProtocolDecoder,
    ExternalRegisterInstruction,
    OpaqueProtocolDecoder,
    ExternalSequenceMemory,
    ExternalSequenceOperatorMemory,
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
    _stable_bits,
)

TARGET_OPERATIONS = ("complement_rotate", "prefix_parity")
COMPOSITION_PROGRAMS = tuple(permutations(SOURCE_OPERATIONS))
MASTERY_THRESHOLD = 0.8
SOURCE_CHECKPOINT_SCHEMA = (
    "neural-computer.external-register-source-checkpoint.v1"
)


def _instruction_context(instructions) -> torch.Tensor:
    return torch.stack(
        tuple(instruction.code.detach().squeeze(0) for instruction in instructions)
    ).mean(dim=0)


def _positive_transfer(
    inherited_stable_bits: int | None,
    fresh_stable_bits: int | None,
) -> bool:
    """Return true only when inherited capacity reaches mastery sooner."""

    return bool(
        inherited_stable_bits is not None
        and fresh_stable_bits is not None
        and inherited_stable_bits < fresh_stable_bits
    )


def _save_source_checkpoint(
    path: Path,
    *,
    parent,
    machine,
    source_decoders: list[OpaqueProtocolDecoder],
    source_operations: tuple[str, ...],
    source_scores: list[float],
) -> None:
    """Persist only a verified mastered source bank for later diagnostics."""

    if min(source_scores, default=0.0) < MASTERY_THRESHOLD:
        raise ValueError("refusing to save an unmastered source checkpoint")
    payload = {
        "schema": SOURCE_CHECKPOINT_SCHEMA,
        "operator_mode": machine.operator_mode,
        "operator_rank": machine.operator_rank,
        "source_operations": list(source_operations),
        "source_scores": list(source_scores),
        "machine_configuration": machine.configuration(),
        "parent_state": {
            name: value.detach().cpu().clone()
            for name, value in parent.state_dict().items()
        },
        "machine_state": {
            name: value.detach().cpu().clone()
            for name, value in machine.state_dict().items()
        },
        "source_decoder_states": [
            {
                name: value.detach().cpu().clone()
                for name, value in decoder.state_dict().items()
            }
            for decoder in source_decoders
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_source_checkpoint(
    path: Path,
    *,
    parent,
    machine,
    source_operations: tuple[str, ...],
) -> list[OpaqueProtocolDecoder]:
    """Load and validate a previously verified source bank."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SOURCE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported source checkpoint schema")
    if payload.get("operator_mode") != machine.operator_mode:
        raise ValueError("source checkpoint operator mode does not match")
    if payload.get("operator_rank") != machine.operator_rank:
        raise ValueError("source checkpoint operator rank does not match")
    if tuple(payload.get("source_operations", ())) != source_operations:
        raise ValueError("source checkpoint source order does not match")
    source_scores = tuple(float(value) for value in payload.get("source_scores", ()))
    if len(source_scores) != len(source_operations):
        raise ValueError("source checkpoint decoder count does not match")
    if min(source_scores, default=0.0) < MASTERY_THRESHOLD:
        raise ValueError("source checkpoint is below the mastery threshold")
    parent.load_state_dict(payload["parent_state"], strict=True)
    machine.load_state_dict(payload["machine_state"], strict=True)
    decoder_states = payload["source_decoder_states"]
    if len(decoder_states) != len(source_operations):
        raise ValueError("source checkpoint decoder count does not match")
    decoders = []
    for state in decoder_states:
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        decoder.load_state_dict(state, strict=True)
        decoders.append(decoder)
    return decoders


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
        register_readout=candidate.get("readout"),
        preserve_execution_trace=candidate.get("preserve_execution_trace", False),
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
            instructions = (
                candidate["mutable_instructions"]
                if "mutable_instructions" in candidate
                else (candidate["instruction"],)
            )
            basis_slots = (
                candidate["mutable_basis_slots"]
                if "mutable_basis_slots" in candidate
                else (candidate["basis_slot"],)
            )
            trainable.extend(instruction.code for instruction in instructions)
            for basis_slot in basis_slots:
                trainable.extend(machine.basis_slots[basis_slot].parameters())
        if not candidate.get("bridge_frozen", False):
            trainable.extend(candidate["bridge"].parameters())
        # A frozen composition candidate may have no mutable interpreter or
        # bridge. In that case the decoder is its only adaptation surface and
        # must remain trainable during the focus phase.
        if decoder_trainable or not trainable:
            for parameter in decoder.parameters():
                parameter.requires_grad_(True)
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
            register_readout=candidate.get("readout"),
            preserve_execution_trace=candidate.get("preserve_execution_trace", False),
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
    source_operations: tuple[str, ...] = SOURCE_OPERATIONS,
    include_composition: bool,
    decoder_prior_state: dict[str, torch.Tensor] | None = None,
    composition_programs: tuple[tuple[str, ...], ...] = COMPOSITION_PROGRAMS,
    bridge_prior_state: dict[str, torch.Tensor] | None = None,
    conditioned_bridge_prior: bool = False,
    readout_prior_state: dict[str, torch.Tensor] | None = None,
    preserve_composition_trace: bool = False,
) -> list[dict[str, object]]:
    candidates = []
    for index, operation in enumerate(operations):
        instruction = machine.instructions[len(source_operations) + index]
        basis_slot = machine.add_basis_slot()
        candidates.append(
            {
                "operation": operation,
                "instruction": instruction,
                "basis_slot": basis_slot,
                "instructions": (instruction,),
                "basis_slots": (basis_slot,),
                "mutable": True,
                "mutable_instructions": (instruction,),
                "mutable_basis_slots": (basis_slot,),
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
        for program in composition_programs:
            source_indices = tuple(
                source_operations.index(primitive) for primitive in program
            )
            decoder = OpaqueProtocolDecoder(
                REGISTER_WIDTH * (len(program) if preserve_composition_trace else 1),
                ACTION_WIDTH,
                hidden=16,
            )
            if decoder_prior_state is not None:
                decoder.load_state_dict(decoder_prior_state, strict=True)
            bridge = (
                CapabilityConditionedEventBridge(
                    EVENT_WIDTH,
                    parent.controller.width,
                    EVENT_WIDTH,
                    INSTRUCTION_WIDTH,
                    hidden=64,
                )
                if conditioned_bridge_prior
                else AmodalEventBridge(
                    EVENT_WIDTH, parent.controller.width, EVENT_WIDTH, hidden=64
                )
            )
            if bridge_prior_state is not None:
                bridge.load_state_dict(bridge_prior_state, strict=True)
                if conditioned_bridge_prior:
                    bridge.set_context(
                        _instruction_context(
                            tuple(machine.instructions[index] for index in source_indices)
                        )
                    )
                for parameter in bridge.parameters():
                    parameter.requires_grad_(False)
            readout = CanonicalRegisterReadout(
                REGISTER_WIDTH, REGISTER_WIDTH, hidden=64
            )
            if readout_prior_state is not None:
                readout.load_state_dict(readout_prior_state, strict=True)
                for parameter in readout.parameters():
                    parameter.requires_grad_(False)
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
                    "decoder": decoder,
                    "bridge": bridge,
                    "bridge_frozen": bridge_prior_state is not None,
                    "readout": readout,
                    "preserve_execution_trace": preserve_composition_trace,
                }
            )
    return candidates


def _enable_candidate_capacity(machine, candidate: dict[str, object]) -> None:
    if not candidate.get("mutable", True):
        return
    instructions = (
        candidate["mutable_instructions"]
        if "mutable_instructions" in candidate
        else (candidate["instruction"],)
    )
    basis_slots = (
        candidate["mutable_basis_slots"]
        if "mutable_basis_slots" in candidate
        else (candidate["basis_slot"],)
    )
    for instruction in instructions:
        instruction.code.requires_grad_(True)
    for basis_slot in basis_slots:
        for parameter in machine.basis_slots[basis_slot].parameters():
            parameter.requires_grad_(True)


def _train_candidate_schedule(
    parent,
    machine,
    candidate: dict[str, object],
    *,
    args: argparse.Namespace,
    seed_base: int,
) -> None:
    _freeze(machine)
    _enable_candidate_capacity(machine, candidate)
    _train_interleaved_phase(
        parent,
        machine,
        [candidate],
        updates=args.warmup_updates,
        batch_size=args.batch_size,
        span=args.warmup_span,
        seed=seed_base,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed_base + 100_000,
        decoder_trainable=True,
    )
    for parameter in candidate["decoder"].parameters():
        parameter.requires_grad_(False)
    _train_interleaved_phase(
        parent,
        machine,
        [candidate],
        updates=args.focus_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed_base + 200_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed_base + 300_000,
        decoder_trainable=False,
    )
    for parameter in candidate["decoder"].parameters():
        parameter.requires_grad_(True)
    _train_interleaved_phase(
        parent,
        machine,
        [candidate],
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed_base + 400_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed_base + 500_000,
        decoder_trainable=True,
    )


def _source_bridge_accuracy(
    parent,
    machine,
    source_spec,
    *,
    bridge,
    conditioned: bool,
    count: int,
    span: int,
    seed: int,
) -> float:
    operation, instruction, decoder, basis_slot = source_spec
    if conditioned:
        bridge.set_context(_instruction_context((instruction,)))
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
        credit_mode="paired_counterfactual",
        event_bridge=bridge,
    )


def _fresh_source_curriculum(parent, *, args, seed_base: int):
    """Acquire a matched fresh source bank before target adaptation."""
    machine = _new_machine(
        len(args.source_operations),
        operator_mode=args.operator_mode,
        operator_rank=args.operator_rank,
    )
    for _ in args.source_operations:
        machine.add_basis_slot()
    if args.joint_source_updates:
        _train_joint_source_bank(
            parent,
            machine,
            args=args,
            seed_base=seed_base,
        )
        return machine
    for index in range(len(args.source_operations)):
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        source_args = copy.copy(args)
        source_args.seed = seed_base + index * 100_003
        _train_source(
            parent,
            machine,
            decoder,
            index,
            source_args,
            source_operations=args.source_operations,
        )
    return machine


def _train_joint_source_bank(parent, machine, *, args, seed_base: int):
    """Train all source instructions against one balanced shared operator.

    This is an explicit calibration upper bound, not a continual-learning
    result: it uses all source procedures during one acquisition phase and is
    reported separately from sequential source learning.
    """
    decoders = [
        OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        for _ in args.source_operations
    ]
    # Source acquisition must train the base operator from its random
    # initialization.  Protection begins only at the later sequence-
    # calibration boundary, where the mastered base is frozen and the
    # zero-initialized meta residual is opened.
    trainable = list(machine.parameters())
    trainable.extend(
        parameter for decoder in decoders for parameter in decoder.parameters()
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-5)
    best_state = None
    best_memory_state = None
    best_operator_memory_state = None
    best_decoders = None
    best_floor = float("-inf")
    for update in range(1, args.joint_source_updates + 1):
        index = (update - 1) % len(args.source_operations)
        batch = _batch(
            args.source_operations[index],
            count=args.batch_size,
            span=args.span,
            seed=seed_base + update * 10_007,
        )
        loss, _ = _rollout(
            parent,
            machine,
            decoders[index],
            batch,
            (machine.instructions[index],),
            basis_slots=(index,),
            train_decoder=True,
            credit_mode="paired_counterfactual",
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if update % args.eval_every == 0 or update == args.joint_source_updates:
            scores = [
                _accuracy(
                    parent,
                    machine,
                    decoder,
                    operation=operation,
                    instructions=(machine.instructions[index_value],),
                    basis_slots=(index_value,),
                    count=args.source_selection_audit_count,
                    span=args.span,
                    seed=seed_base + 70_000 + index_value * 1_009,
                    credit_mode="paired_counterfactual",
                )
                for index_value, (operation, decoder) in enumerate(
                    zip(args.source_operations, decoders, strict=True)
                )
            ]
            floor = min(scores)
            if floor > best_floor:
                best_floor = floor
                best_state = {
                    name: value.detach().clone()
                    for name, value in machine.state_dict().items()
                }
                best_decoders = [
                    {
                        name: value.detach().clone()
                        for name, value in decoder.state_dict().items()
                    }
                    for decoder in decoders
                ]
    if best_state is None or best_decoders is None:
        raise RuntimeError("joint source calibration did not produce a checkpoint")
    machine.load_state_dict(best_state, strict=True)
    for decoder, state in zip(decoders, best_decoders, strict=True):
        decoder.load_state_dict(state, strict=True)
    return decoders


def _train_sequence_calibration(
    parent,
    machine,
    *,
    args: argparse.Namespace,
    composition_programs: tuple[tuple[str, ...], ...],
    seed_base: int,
    source_decoders: list[OpaqueProtocolDecoder] | None = None,
    sequence_memory: ExternalSequenceMemory | None = None,
    sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
    sequence_readout: CanonicalRegisterReadout | None = None,
) -> list[dict[str, object]]:
    """Calibrate the shared operator on opaque multi-instruction chains.

    This is deliberately an explicit upper bound.  It uses verifier outcomes
    from generated source programs before target acquisition and is therefore
    not evidence for replay-free continual learning by itself.  The matched
    fresh control receives the same calibration budget.
    """

    if args.sequence_calibration_updates <= 0:
        return []
    programs = tuple(composition_programs)
    if not programs:
        raise ValueError("sequence calibration requires composition programs")
    decoder_input_width = REGISTER_WIDTH * (
        len(programs[0]) if args.preserve_composition_trace else 1
    )
    if args.shared_sequence_decoder and args.preserve_composition_trace:
        if any(len(program) != len(programs[0]) for program in programs):
            raise ValueError("shared sequence decoder requires equal trace widths")
    shared_decoder = (
        ConditionedOpaqueProtocolDecoder(
            decoder_input_width,
            INSTRUCTION_WIDTH,
            ACTION_WIDTH,
            hidden=32,
        )
        if args.conditioned_sequence_decoder
        else OpaqueProtocolDecoder(decoder_input_width, ACTION_WIDTH, hidden=16)
    )
    decoders = (
        [shared_decoder for _ in programs]
        if args.shared_sequence_decoder
        else [
            OpaqueProtocolDecoder(
                REGISTER_WIDTH * (len(program) if args.preserve_composition_trace else 1),
                ACTION_WIDTH,
                hidden=16,
            )
            for program in programs
        ]
    )
    use_operator_router = bool(
        sequence_operator_memory is not None
        and args.use_operator_sequence_router
    )

    def route_query(program: tuple[str, ...]) -> torch.Tensor:
        instructions = tuple(
            machine.instructions[args.source_operations.index(name)]
            for name in program
        )
        codes = torch.stack(
            tuple(instruction.code.detach().squeeze(0) for instruction in instructions)
        ).unsqueeze(0)
        return sequence_operator_memory.encode_program(codes).squeeze(0)

    protected_meta_mode = args.operator_mode in (
        "factorized_protected_meta",
        "factorized_protected_bounded_meta",
    )
    if protected_meta_mode:
        for parameter in machine.parameters():
            parameter.requires_grad_(False)
        if sequence_operator_memory is not None:
            trainable = list(sequence_operator_memory.parameters())
        else:
            trainable = (
                list(sequence_memory.parameters())
                if sequence_memory is not None
                else [
                parameter
                for name, parameter in machine.named_parameters()
                if name.startswith("operator_meta_")
                ]
            )
    else:
        trainable = list(machine.parameters())
    if sequence_readout is not None:
        trainable.extend(sequence_readout.parameters())
    seen_decoder_parameters: set[int] = set()
    for decoder in decoders:
        for parameter in decoder.parameters():
            if id(parameter) not in seen_decoder_parameters:
                trainable.append(parameter)
                seen_decoder_parameters.add(id(parameter))
    if sequence_memory is not None and not protected_meta_mode:
        trainable.extend(sequence_memory.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-5)
    best_state = None
    best_decoder_states = None
    best_readout_state = None
    best_floor = float("-inf")
    progress: list[dict[str, object]] = []
    for update in range(1, args.sequence_calibration_updates + 1):
        index = (update - 1) % len(programs)
        program = programs[index]
        source_indices = tuple(args.source_operations.index(name) for name in program)
        batch = _batch(
            "generated_composition",
            count=args.batch_size,
            span=args.span,
            seed=seed_base + update * 10_007,
            generated_composition_ids=(0,),
            generated_compositions=(program,),
        )
        loss, _ = _rollout(
            parent,
            machine,
            decoders[index],
            batch,
            tuple(machine.instructions[value] for value in source_indices),
            basis_slots=source_indices,
            train_decoder=True,
            credit_mode="paired_counterfactual",
            preserve_execution_trace=args.preserve_composition_trace,
            meta_context=(
                sequence_memory.slots[index]
                if sequence_memory is not None
                else None
            ),
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=(
                index
                if sequence_operator_memory is not None and not use_operator_router
                else None
            ),
            sequence_operator_route_query=(
                route_query(program) if use_operator_router else None
            ),
            route_probe=(args.use_route_outcome_credit and use_operator_router),
            register_readout=sequence_readout,
            decoder_context=(route_query(program) if args.conditioned_sequence_decoder else None),
        )
        if use_operator_router and args.route_assignment_loss_weight > 0.0:
            all_route_weights = torch.stack(
                tuple(
                    sequence_operator_memory.route_weights(
                        route_query(value).unsqueeze(0)
                    )
                    for value in programs
                ),
                dim=0,
            )
            column_balance = (
                all_route_weights.mean(dim=0) - (1.0 / len(programs))
            ).square().mean()
            entropy = -(
                all_route_weights.clamp_min(torch.finfo(all_route_weights.dtype).tiny)
                * all_route_weights.clamp_min(torch.finfo(all_route_weights.dtype).tiny).log()
            ).sum(dim=-1).mean()
            loss = loss + args.route_assignment_loss_weight * (
                column_balance + args.route_assignment_entropy_weight * entropy
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if update % args.eval_every == 0 or update == args.sequence_calibration_updates:
            sequence_scores = [
                _accuracy(
                    parent,
                    machine,
                    decoders[index_value],
                    operation="generated_composition",
                    instructions=tuple(
                        machine.instructions[value]
                        for value in tuple(
                            args.source_operations.index(name) for name in program_value
                        )
                    ),
                    basis_slots=tuple(
                        args.source_operations.index(name) for name in program_value
                    ),
                    count=args.source_selection_audit_count,
                    span=args.span,
                    seed=seed_base + 70_000 + index_value * 1_009,
                    credit_mode="paired_counterfactual",
                    generated_composition_ids=(0,),
                    generated_compositions=(program_value,),
                    preserve_execution_trace=args.preserve_composition_trace,
                    meta_context=(
                        sequence_memory.slots[index_value]
                        if sequence_memory is not None
                        else None
                    ),
                    sequence_operator_memory=sequence_operator_memory,
                    sequence_operator_slot=(
                        index_value
                        if sequence_operator_memory is not None and not use_operator_router
                        else None
                    ),
                    sequence_operator_route_query=(
                        route_query(program_value) if use_operator_router else None
                    ),
                    register_readout=sequence_readout,
                    decoder_context=(
                        route_query(program_value)
                        if args.conditioned_sequence_decoder
                        else None
                    ),
                )
                for index_value, program_value in enumerate(programs)
            ]
            source_floor = None
            if source_decoders is not None:
                source_floor = min(
                    _accuracy(
                        parent,
                        machine,
                        decoder,
                        operation=operation,
                        instructions=(machine.instructions[index_value],),
                        basis_slots=(index_value,),
                        count=args.source_selection_audit_count,
                        span=args.span,
                        seed=seed_base + 80_000 + index_value * 1_009,
                        credit_mode="paired_counterfactual",
                    )
                    for index_value, (operation, decoder) in enumerate(
                        zip(args.source_operations, source_decoders, strict=True)
                    )
                )
            floor = min(sequence_scores)
            if source_floor is not None:
                floor = min(floor, source_floor)
            progress.append(
                {
                    "update": update,
                    "sequence_floor": floor,
                    "source_floor": source_floor,
                    "sequence_scores": sequence_scores,
                }
            )
            if floor > best_floor:
                best_floor = floor
                best_state = {
                    name: value.detach().clone()
                    for name, value in machine.state_dict().items()
                }
                if sequence_memory is not None:
                    best_memory_state = {
                        name: value.detach().clone()
                        for name, value in sequence_memory.state_dict().items()
                    }
                if sequence_operator_memory is not None:
                    best_operator_memory_state = {
                        name: value.detach().clone()
                        for name, value in sequence_operator_memory.state_dict().items()
                    }
                best_decoder_states = [
                    {
                        name: value.detach().clone()
                        for name, value in decoder.state_dict().items()
                    }
                    for decoder in decoders
                ]
                if sequence_readout is not None:
                    best_readout_state = {
                        name: value.detach().clone()
                        for name, value in sequence_readout.state_dict().items()
                    }
    if best_state is None:
        raise RuntimeError("sequence calibration did not produce a checkpoint")
    machine.load_state_dict(best_state, strict=True)
    if sequence_memory is not None and best_memory_state is not None:
        sequence_memory.load_state_dict(best_memory_state, strict=True)
    if (
        sequence_operator_memory is not None
        and best_operator_memory_state is not None
    ):
        sequence_operator_memory.load_state_dict(
            best_operator_memory_state, strict=True
        )
    if best_decoder_states is not None:
        for decoder, state in zip(decoders, best_decoder_states, strict=True):
            decoder.load_state_dict(state, strict=True)
    if sequence_readout is not None and best_readout_state is not None:
        sequence_readout.load_state_dict(best_readout_state, strict=True)
    return progress

def _train_shared_bridge_prior(
    parent,
    machine,
    source_specs,
    *,
    args: argparse.Namespace,
) -> tuple[AmodalEventBridge, list[dict[str, object]]]:
    """Learn one reusable event interface from mastered source outcomes."""
    bridge = (
        CapabilityConditionedEventBridge(
            EVENT_WIDTH,
            parent.controller.width,
            EVENT_WIDTH,
            INSTRUCTION_WIDTH,
            hidden=64,
        )
        if args.reuse_conditioned_bridge_prior
        else AmodalEventBridge(
            EVENT_WIDTH, parent.controller.width, EVENT_WIDTH, hidden=64
        )
    )
    _freeze(machine)
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    for _, _, decoder, _ in source_specs:
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    best_state = None
    best_floor = float("-inf")
    progress = []
    for update in range(1, args.bridge_prior_updates + 1):
        index = (update - 1) % len(source_specs)
        operation, instruction, decoder, basis_slot = source_specs[index]
        if args.reuse_conditioned_bridge_prior:
            bridge.set_context(_instruction_context((instruction,)))
        batch = _batch(
            operation,
            count=args.batch_size,
            span=args.span,
            seed=args.seed + 1_900_000 + update * 10_007,
        )
        loss, _ = _rollout(
            parent,
            machine,
            decoder,
            batch,
            (instruction,),
            basis_slots=(basis_slot,),
            train_decoder=False,
            credit_mode="attempted_bce",
            event_bridge=bridge,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        if update % args.eval_every == 0 or update == args.bridge_prior_updates:
            scores = [
                _source_bridge_accuracy(
                    parent,
                    machine,
                    source_spec=(
                        operation_name,
                        instruction_value,
                        decoder,
                        slot,
                    ),
                    bridge=bridge,
                    count=args.source_selection_audit_count,
                    span=args.span,
                    seed=args.seed + 60_000 + source_index * 101 + source_index * 1009,
                    conditioned=args.reuse_conditioned_bridge_prior,
                )
                for source_index, (
                    operation_name,
                    instruction_value,
                    decoder,
                    slot,
                ) in enumerate(source_specs)
            ]
            floor = min(scores)
            progress.append({"update": update, "source_scores": scores})
            if floor > best_floor:
                best_floor = floor
                best_state = {
                    name: value.detach().clone()
                    for name, value in bridge.state_dict().items()
                }
    if best_state is not None:
        bridge.load_state_dict(best_state, strict=True)
    for _, _, decoder, _ in source_specs:
        for parameter in decoder.parameters():
            parameter.requires_grad_(True)
    return bridge, progress


def _train_canonical_readout_prior(
    parent,
    machine,
    source_specs,
    *,
    args: argparse.Namespace,
) -> tuple[CanonicalRegisterReadout, dict[str, torch.Tensor], list[dict[str, object]]]:
    """Fit one shared register-to-output convention on mastered source skills."""
    readout = CanonicalRegisterReadout(REGISTER_WIDTH, REGISTER_WIDTH, hidden=64)
    decoders = [
        OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        for _ in source_specs
    ]
    for decoder, (_, _, source_decoder, _) in zip(decoders, source_specs, strict=True):
        decoder.load_state_dict(source_decoder.state_dict(), strict=True)
    _freeze(machine)
    optimizer = torch.optim.AdamW(
        list(readout.parameters())
        + [parameter for decoder in decoders for parameter in decoder.parameters()],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    best_state = None
    best_floor = float("-inf")
    progress = []
    for update in range(1, args.readout_prior_updates + 1):
        index = (update - 1) % len(source_specs)
        operation, instruction, _, basis_slot = source_specs[index]
        batch = _batch(
            operation,
            count=args.batch_size,
            span=args.span,
            seed=args.seed + 2_300_000 + update * 10_007,
        )
        loss, _ = _rollout(
            parent,
            machine,
            decoders[index],
            batch,
            (instruction,),
            basis_slots=(basis_slot,),
            train_decoder=True,
            credit_mode="attempted_bce",
            register_readout=readout,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(readout.parameters())
            + [parameter for decoder in decoders for parameter in decoder.parameters()],
            1.0,
        )
        optimizer.step()
        if update % args.eval_every == 0 or update == args.readout_prior_updates:
            scores = [
                _accuracy(
                    parent,
                    machine,
                    decoder,
                    operation=operation_name,
                    instructions=(instruction_value,),
                    basis_slots=(slot,),
                    count=args.source_selection_audit_count,
                    span=args.span,
                    seed=args.seed + 70_000 + source_index * 101,
                    credit_mode="paired_counterfactual",
                    register_readout=readout,
                )
                for source_index, (
                    operation_name,
                    instruction_value,
                    _,
                    slot,
                ) in enumerate(source_specs)
                for decoder in (decoders[source_index],)
            ]
            floor = min(scores)
            progress.append({"update": update, "source_scores": scores})
            if floor > best_floor:
                best_floor = floor
                best_state = {
                    name: value.detach().clone()
                    for name, value in readout.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("canonical readout prior did not produce a checkpoint")
    readout.load_state_dict(best_state, strict=True)
    return readout, best_state, progress


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    target_operations = tuple(
        operation for operation in args.target_operations.split(",") if operation
    )
    if not target_operations:
        raise ValueError("at least one target operation is required")
    source_operations = tuple(
        operation for operation in args.source_order.split(",") if operation
    )
    if len(source_operations) != len(SOURCE_OPERATIONS) or set(source_operations) != set(
        SOURCE_OPERATIONS
    ):
        raise ValueError("source order must be a permutation of the source operations")
    args.source_operations = source_operations
    if args.source_restarts < 1:
        raise ValueError("source restarts must be positive")
    if args.joint_source_updates < 0:
        raise ValueError("joint source updates cannot be negative")
    if args.sequence_calibration_updates < 0:
        raise ValueError("sequence calibration updates cannot be negative")
    if args.route_assignment_loss_weight < 0.0:
        raise ValueError("route assignment loss weight cannot be negative")
    if args.route_assignment_entropy_weight < 0.0:
        raise ValueError("route assignment entropy weight cannot be negative")
    if args.source_checkpoint_in and args.joint_source_updates:
        raise ValueError("source checkpoint input cannot be combined with source calibration")
    if args.use_sequence_memory and args.use_operator_sequence_memory:
        raise ValueError("choose value or operator sequence memory")
    if args.use_operator_sequence_router and not args.use_operator_sequence_memory:
        raise ValueError("operator sequence routing requires operator sequence memory")
    if args.use_route_outcome_credit and not args.use_operator_sequence_router:
        raise ValueError("route outcome credit requires operator sequence routing")
    if args.conditioned_sequence_decoder and not args.use_operator_sequence_router:
        raise ValueError("conditioned sequence decoder requires operator sequence routing")
    if args.conditioned_sequence_decoder and not args.shared_sequence_decoder:
        raise ValueError("conditioned sequence decoder requires shared sequence decoder")
    if args.reuse_shared_bridge_prior and args.reuse_conditioned_bridge_prior:
        raise ValueError("choose one bridge prior mode")
    args.reuse_shared_bridge_prior = (
        args.reuse_shared_bridge_prior or args.reuse_conditioned_bridge_prior
    )
    if not 1 <= args.composition_program_count <= len(COMPOSITION_PROGRAMS):
        raise ValueError("composition program count is outside the available grammar")
    composition_programs = COMPOSITION_PROGRAMS[: args.composition_program_count]
    if args.target_composition_start is None:
        target_composition_programs = composition_programs
    else:
        target_start = args.target_composition_start
        target_stop = target_start + args.composition_program_count
        if target_start < 0 or target_stop > len(COMPOSITION_PROGRAMS):
            raise ValueError("target composition range is outside the available grammar")
        target_composition_programs = COMPOSITION_PROGRAMS[target_start:target_stop]
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
        len(args.source_operations),
        operator_mode=args.operator_mode,
        operator_rank=args.operator_rank,
    )
    for _ in args.source_operations:
        machine.add_basis_slot()
    source_decoders = []
    source_attempt_counts = []
    source_checkpoint_loaded = args.source_checkpoint_in is not None
    if source_checkpoint_loaded:
        source_decoders = _load_source_checkpoint(
            args.source_checkpoint_in,
            parent=parent,
            machine=machine,
            source_operations=args.source_operations,
        )
        source_attempt_counts = [["loaded_checkpoint"] for _ in source_decoders]
    elif args.joint_source_updates:
        source_decoders = _train_joint_source_bank(
            parent,
            machine,
            args=args,
            seed_base=args.seed + 1_500_000,
        )
        source_attempt_counts = [
            [
                _accuracy(
                    parent,
                    machine,
                    decoder,
                    operation=operation,
                    instructions=(machine.instructions[index],),
                    basis_slots=(index,),
                    count=args.source_selection_audit_count,
                    span=args.span,
                    seed=args.seed + 60_000 + index * 101 + index * 1009,
                    credit_mode="paired_counterfactual",
                )
            ]
            for index, (operation, decoder) in enumerate(
                zip(args.source_operations, source_decoders, strict=True)
            )
        ]
    for index in (
        range(len(args.source_operations))
        if not args.joint_source_updates and not source_checkpoint_loaded
        else ()
    ):
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
                parent,
                candidate_machine,
                candidate_decoder,
                index,
                attempt_args,
                source_operations=args.source_operations,
            )
            candidate_score = _accuracy(
                parent,
                candidate_machine,
                candidate_decoder,
                operation=args.source_operations[index],
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
            zip(args.source_operations, source_decoders, strict=True)
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
    if args.source_checkpoint_out:
        _save_source_checkpoint(
            args.source_checkpoint_out,
            parent=parent,
            machine=machine,
            source_decoders=source_decoders,
            source_operations=args.source_operations,
            source_scores=source_before,
        )
    sequence_calibration_progress: list[dict[str, object]] = []
    sequence_memory = None
    sequence_operator_memory = None
    sequence_readout = None
    if args.use_sequence_memory and args.sequence_calibration_updates:
        sequence_memory = ExternalSequenceMemory(REGISTER_WIDTH)
        for _ in composition_programs:
            sequence_memory.add_slot()
    if args.use_operator_sequence_memory and args.sequence_calibration_updates:
        sequence_operator_memory = ExternalSequenceOperatorMemory(
            REGISTER_WIDTH,
            INSTRUCTION_WIDTH,
            operator_rank=args.operator_rank,
            router_temperature=args.operator_router_temperature,
        )
        for _ in composition_programs:
            sequence_operator_memory.add_slot()
    if args.use_shared_sequence_readout and args.sequence_calibration_updates:
        sequence_readout = CanonicalRegisterReadout(
            REGISTER_WIDTH, REGISTER_WIDTH, hidden=64
        )
    sequence_calibration_source_after = list(source_before)
    sequence_calibration_accepted = False
    if args.sequence_calibration_updates:
        machine_before_sequence_calibration = copy.deepcopy(machine)
        sequence_calibration_progress = _train_sequence_calibration(
            parent,
            machine,
            args=args,
            composition_programs=composition_programs,
            seed_base=args.seed + 1_700_000,
            source_decoders=source_decoders,
            sequence_memory=sequence_memory,
            sequence_operator_memory=sequence_operator_memory,
            sequence_readout=sequence_readout,
        )
        sequence_calibration_source_after = [
            source_score(spec, index) for index, spec in enumerate(source_specs)
        ]
        sequence_calibration_accepted = min(
            sequence_calibration_source_after, default=0.0
        ) >= min(source_before, default=0.0) - args.retention_regression_tolerance
        if not sequence_calibration_accepted:
            machine.load_state_dict(
                machine_before_sequence_calibration.state_dict(), strict=True
            )
            sequence_calibration_source_after = list(source_before)
    source_attempt_count = sum(len(attempts) for attempts in source_attempt_counts)
    # Do not allocate future target instructions until source acquisition is
    # complete.  The first source phase trains machine parameters jointly; if
    # future capacity exists already, it becomes an accidental optimizer
    # participant and source quality depends on the number of capabilities we
    # happened to reserve for later.
    for _ in target_operations:
        machine.add_instruction(ExternalRegisterInstruction(INSTRUCTION_WIDTH))
    pre_growth_machine = copy.deepcopy(machine)
    bridge_prior_state = None
    bridge_prior_progress = []
    bridge_prior_source_after = source_before
    if args.reuse_shared_bridge_prior:
        shared_bridge, bridge_prior_progress = _train_shared_bridge_prior(
            parent,
            machine,
            source_specs,
            args=args,
        )
        bridge_prior_state = {
            name: value.detach().clone()
            for name, value in shared_bridge.state_dict().items()
        }
        bridge_prior_source_after = [
            source_score(spec, index) for index, spec in enumerate(source_specs)
        ]
    readout_prior_state = None
    readout_prior_progress = []
    if args.reuse_canonical_readout_prior:
        _, readout_prior_state, readout_prior_progress = _train_canonical_readout_prior(
            parent,
            machine,
            source_specs,
            args=args,
        )
    decoder_prior_state = None
    if args.reuse_decoder_prior:
        decoder_prior_state = {
            name: value.detach().clone()
            for name, value in source_decoders[0].state_dict().items()
        }
    candidates = _prepare_candidates(
        parent,
        machine,
        target_operations,
        include_composition=args.include_composition,
        source_operations=args.source_operations,
        decoder_prior_state=decoder_prior_state,
        composition_programs=target_composition_programs,
        bridge_prior_state=bridge_prior_state,
        conditioned_bridge_prior=args.reuse_conditioned_bridge_prior,
        preserve_composition_trace=args.preserve_composition_trace,
        readout_prior_state=readout_prior_state,
    )
    retained_before = [
        source_score(spec, index) for index, spec in enumerate(source_specs)
    ]
    _freeze(machine)
    for candidate in candidates:
        _enable_candidate_capacity(machine, candidate)

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
        decoder_prior_state=decoder_prior_state,
        composition_programs=target_composition_programs,
        bridge_prior_state=bridge_prior_state,
        conditioned_bridge_prior=args.reuse_conditioned_bridge_prior,
        source_operations=args.source_operations,
        preserve_composition_trace=args.preserve_composition_trace,
    )
    _freeze(shuffled_machine)
    for candidate in shuffled_candidates:
        _enable_candidate_capacity(shuffled_machine, candidate)
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

    transfer_records = []
    for index, candidate in enumerate(candidates):
        if candidate["operation"] != "generated_composition":
            continue
        program = tuple(candidate["composition_program"])
        fresh_source_training_bits = 0
        if args.curriculum_fresh_control:
            fresh_machine = _fresh_source_curriculum(
                parent,
                args=args,
                seed_base=args.seed + 2_000_000 + index * 20_003,
            )
            fresh_source_training_bits = (
                (
                    args.joint_source_updates
                    if args.joint_source_updates
                    else len(args.source_operations) * args.source_updates
                )
                * args.batch_size * args.span * 2
            )
        else:
            fresh_machine = _new_machine(
                len(args.source_operations),
                operator_mode=args.operator_mode,
                operator_rank=args.operator_rank,
            )
            for _ in args.source_operations:
                fresh_machine.add_basis_slot()
        fresh_sequence_calibration_progress = _train_sequence_calibration(
            parent,
            fresh_machine,
            args=args,
            composition_programs=composition_programs,
            seed_base=args.seed + 2_400_000 + index * 20_003,
        )
        source_indices = tuple(
            args.source_operations.index(primitive) for primitive in program
        )
        fresh_bridge = (
            CapabilityConditionedEventBridge(
                EVENT_WIDTH,
                parent.controller.width,
                EVENT_WIDTH,
                INSTRUCTION_WIDTH,
                hidden=64,
            )
            if args.reuse_conditioned_bridge_prior
            else AmodalEventBridge(
                EVENT_WIDTH, parent.controller.width, EVENT_WIDTH, hidden=64
            )
        )
        if bridge_prior_state is not None:
            fresh_bridge.load_state_dict(bridge_prior_state, strict=True)
            if args.reuse_conditioned_bridge_prior:
                fresh_bridge.set_context(
                    _instruction_context(
                        tuple(
                            fresh_machine.instructions[index]
                            for index in source_indices
                        )
                    )
                )
            for parameter in fresh_bridge.parameters():
                parameter.requires_grad_(False)
        fresh_candidate = {
            "operation": "generated_composition",
            "instructions": tuple(
                fresh_machine.instructions[source_index]
                for source_index in source_indices
            ),
            "basis_slots": source_indices,
            "mutable": True,
            "mutable_instructions": tuple(fresh_machine.instructions),
            "mutable_basis_slots": tuple(range(len(args.source_operations))),
            "generated_composition_ids": (0,),
            "generated_compositions": (program,),
            "composition_program": program,
            "decoder": OpaqueProtocolDecoder(
                REGISTER_WIDTH * (
                    len(program) if args.preserve_composition_trace else 1
                ),
                ACTION_WIDTH,
                hidden=16,
            ),
            "bridge": fresh_bridge,
            "bridge_frozen": bridge_prior_state is not None,
            "preserve_execution_trace": args.preserve_composition_trace,
        }
        _train_candidate_schedule(
            parent,
            fresh_machine,
            fresh_candidate,
            args=args,
            seed_base=args.seed + 1_200_000 + index * 10_009,
        )
        inherited_stable = _stable_bits(
            candidate["phase_progress"],
            threshold=MASTERY_THRESHOLD,
            bits_per_update=args.batch_size * args.span * 2,
        )
        fresh_stable = _stable_bits(
            fresh_candidate["phase_progress"],
            threshold=MASTERY_THRESHOLD,
            bits_per_update=args.batch_size * args.span * 2,
        )
        fresh_accuracy = _score(
            parent,
            fresh_machine,
            fresh_candidate,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 1_800_000 + index * 1_009,
        )
        transfer_records.append(
            {
                "composition_program": list(program),
                "inherited_stable_bits": inherited_stable,
                "fresh_stable_bits": fresh_stable,
                "fresh_accuracy": fresh_accuracy,
                "fresh_source_training_verifier_bits": fresh_source_training_bits,
                "fresh_sequence_calibration_verifier_bits": (
                    args.sequence_calibration_updates
                    * args.batch_size
                    * args.span
                    * 2
                    if fresh_sequence_calibration_progress
                    else 0
                ),
                "positive_transfer": _positive_transfer(
                    inherited_stable,
                    fresh_stable,
                ),
            }
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
                for basis_slot in candidate.get(
                    "mutable_basis_slots", (candidate["basis_slot"],)
                ):
                    machine.freeze_basis_slot(basis_slot)
                for instruction in candidate.get(
                    "mutable_instructions", (candidate["instruction"],)
                ):
                    instruction.code.requires_grad_(False)
    else:
        # The transaction is all-or-nothing: concurrent mutable growth cannot
        # leave one candidate admitted when the paired pressure test fails.
        machine = pre_growth_machine

    candidate_count = len(candidates)
    source_selection_updates = (
        args.joint_source_updates
        if args.joint_source_updates
        else args.source_updates
    )
    source_selection_evaluations = (
        (source_selection_updates + args.eval_every - 1) // args.eval_every
    )
    source_acquisition_updates = (
        args.joint_source_updates
        if args.joint_source_updates
        else args.source_updates * source_attempt_count
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
    fresh_transfer_candidate_count = len(transfer_records)
    fresh_transfer_training_lifetimes = fresh_transfer_candidate_count * (
        args.warmup_updates * args.batch_size
        + (args.focus_updates + args.target_updates) * args.batch_size
    )
    fresh_curriculum_source_lifetimes = sum(
        int(row["fresh_source_training_verifier_bits"]) // (args.span * 2)
        for row in transfer_records
    )
    sequence_calibration_lifetimes = (
        args.sequence_calibration_updates * args.batch_size
        if sequence_calibration_progress
        else 0
    )
    fresh_sequence_calibration_lifetimes = sum(
        int(row["fresh_sequence_calibration_verifier_bits"]) // (args.span * 2)
        for row in transfer_records
    )
    audit_lifetimes = (
        candidate_count * args.audit_count * 3
        + candidate_count * args.consolidation_audit_count
        + len(source_specs) * args.source_selection_audit_count * 3
        + source_selection_evaluations * source_attempt_count * args.source_selection_audit_count
        + phase_audit_lifetimes * 2
    )
    logical_lifetimes = (
        args.parent_updates * args.batch_size
        + source_acquisition_updates * args.batch_size
        + (
            args.bridge_prior_updates * args.batch_size
            if args.reuse_shared_bridge_prior
            else 0
        )
        + normal_target_training_lifetimes
        + shuffled_target_training_lifetimes
        + fresh_transfer_training_lifetimes
        + fresh_curriculum_source_lifetimes
        + sequence_calibration_lifetimes
        + fresh_sequence_calibration_lifetimes
        + audit_lifetimes
    )
    parent_training_bits = args.parent_updates * args.batch_size * 2 * 2
    source_training_bits = source_acquisition_updates * args.batch_size * args.span * 2
    source_selection_bits = (
        source_selection_evaluations
        * source_attempt_count
        * args.source_selection_audit_count
        * args.span
        * 2
    )
    bridge_prior_training_bits = (
        args.bridge_prior_updates * args.batch_size * args.span * 2
        if args.reuse_shared_bridge_prior
        else 0
    )
    readout_prior_training_bits = (
        args.readout_prior_updates * args.batch_size * args.span * 2
        if args.reuse_canonical_readout_prior
        else 0
    )
    normal_target_training_bits = candidate_count * (
        args.warmup_updates * args.batch_size * args.warmup_span * 2
        + (args.focus_updates + args.target_updates) * args.batch_size * args.span * 2
    )
    shuffled_target_training_bits = normal_target_training_bits
    fresh_transfer_training_bits = fresh_transfer_candidate_count * (
        args.warmup_updates * args.batch_size * args.warmup_span * 2
        + (args.focus_updates + args.target_updates)
        * args.batch_size
        * args.span
        * 2
    )
    fresh_curriculum_source_bits = sum(
        int(row["fresh_source_training_verifier_bits"])
        for row in transfer_records
    )
    sequence_calibration_bits = (
        args.sequence_calibration_updates * args.batch_size * args.span * 2
        if sequence_calibration_progress
        else 0
    )
    if args.use_route_outcome_credit and sequence_calibration_progress:
        sequence_calibration_bits *= args.composition_program_count
    fresh_sequence_calibration_bits = sum(
        int(row["fresh_sequence_calibration_verifier_bits"])
        for row in transfer_records
    )
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
        + bridge_prior_training_bits
        + readout_prior_training_bits
        + normal_target_training_bits
        + shuffled_target_training_bits
        + fresh_transfer_training_bits
        + fresh_curriculum_source_bits
        + sequence_calibration_bits
        + fresh_sequence_calibration_bits
        + progress_audit_bits
        + suite_and_control_bits
    )
    report = {
        "schema": "neural-computer.external-register-interleaved-basis-acquisition-audit.v1",
        "seed": args.seed,
        "source_operations": list(args.source_operations),
        "source_acquisition_mode": (
            "balanced_joint_calibration_upper_bound"
            if args.joint_source_updates
            else "sequential_with_restarts"
        ),
        "joint_source_updates": args.joint_source_updates,
        "sequence_calibration_updates": args.sequence_calibration_updates,
        "sequence_calibration_trainable_surface": (
            (
                (
                    "sequence_operator_memory_slots_plus_learned_router_plus_outcome_probe_plus_conditioned_shared_decoder_plus_readout"
                    if args.use_route_outcome_credit
                    and args.conditioned_sequence_decoder
                    and args.use_shared_sequence_readout
                    else "sequence_operator_memory_slots_plus_learned_router_plus_outcome_probe_plus_conditioned_shared_decoder"
                    if args.use_route_outcome_credit and args.conditioned_sequence_decoder
                    else "sequence_operator_memory_slots_plus_learned_router_plus_outcome_probe_plus_shared_decoder"
                    if args.use_route_outcome_credit and args.shared_sequence_decoder
                    else "sequence_operator_memory_slots_plus_learned_router_plus_shared_decoder_plus_readout"
                    if args.shared_sequence_decoder and args.use_shared_sequence_readout
                    else "sequence_operator_memory_slots_plus_learned_router_plus_shared_decoder"
                    if args.shared_sequence_decoder
                    else "sequence_operator_memory_slots_plus_learned_router_plus_outcome_probe_plus_readout"
                    if args.use_route_outcome_credit and args.use_shared_sequence_readout
                    else "sequence_operator_memory_slots_plus_learned_router_plus_outcome_probe_plus_temporary_decoders"
                    if args.use_route_outcome_credit
                    else "sequence_operator_memory_slots_plus_learned_router_plus_readout"
                    if args.use_shared_sequence_readout
                    else "sequence_operator_memory_slots_plus_learned_router_plus_temporary_decoders"
                )
                if args.use_operator_sequence_router
                else "sequence_operator_memory_slots_plus_temporary_decoders"
            )
            if args.use_operator_sequence_memory
            else (
                "sequence_memory_slots_plus_temporary_decoders"
                if args.use_sequence_memory
                else "operator_meta_residual_plus_temporary_decoders"
            )
        ),
        "sequence_calibration_accepted": sequence_calibration_accepted,
        "sequence_calibration_source_before": source_before,
        "sequence_calibration_source_after": sequence_calibration_source_after,
        "sequence_calibration_progress": sequence_calibration_progress,
        "sequence_memory": (
            sequence_memory.configuration()
            if sequence_memory is not None
            else None
        ),
        "sequence_operator_memory": (
            sequence_operator_memory.configuration()
            if sequence_operator_memory is not None
            else None
        ),
        "sequence_readout": (
            sequence_readout.configuration() if sequence_readout is not None else None
        ),
        "route_credit": (
            "counterfactual_scalar_outcome_per_operator_slot"
            if args.use_route_outcome_credit
            else None
        ),
        "route_assignment_regularization": (
            {
                "loss_weight": args.route_assignment_loss_weight,
                "entropy_weight": args.route_assignment_entropy_weight,
            }
            if args.route_assignment_loss_weight > 0.0
            else None
        ),
        "operator_mode": args.operator_mode,
        "operator_rank": args.operator_rank,
        "role_binding": (
            machine.configuration().get("role_count")
            if args.operator_mode in (
                "factorized_shared_role_bound",
                "factorized_shared_relational",
                "factorized_shared_stable_relational",
            )
            else None
        ),
        "preserve_composition_trace": args.preserve_composition_trace,
        "target_operations": list(target_operations),
        "composition_programs": [list(program) for program in composition_programs]
        if args.include_composition
        else [],
        "target_composition_programs": [list(program) for program in target_composition_programs]
        if args.include_composition
        else [],
        "decoder_prior": "source_decoder_0"
        if args.reuse_decoder_prior
        else None,
        "shared_bridge_prior": (
            "conditioned_outcome_trained_source_bridge"
            if args.reuse_conditioned_bridge_prior
            else "outcome_trained_source_bridge"
        )
        if args.reuse_shared_bridge_prior
        else None,
        "source_before": source_before,
        "source_checkpoint_loaded": source_checkpoint_loaded,
        "source_checkpoint_out": (
            str(args.source_checkpoint_out)
            if args.source_checkpoint_out is not None
            else None
        ),
        "source_attempt_scores": source_attempt_counts,
        "retained_before": retained_before,
        "bridge_prior_source_after": bridge_prior_source_after,
        "bridge_prior_progress": bridge_prior_progress,
        "canonical_readout_prior": (
            "source_trained_frozen_register_readout"
            if args.reuse_canonical_readout_prior
            else None
        ),
        "canonical_readout_prior_progress": readout_prior_progress,
        "targets": records,
        "fresh_transfer": transfer_records,
        "transfer_promoted": bool(
            transfer_records
            and all(row["positive_transfer"] for row in transfer_records)
        ),
        "curriculum_fresh_control": args.curriculum_fresh_control,
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
            "joint_source_calibration_verifier_bits": (
                source_training_bits if args.joint_source_updates else 0
            ),
            "bridge_prior_training_verifier_bits": bridge_prior_training_bits,
            "canonical_readout_prior_training_verifier_bits": readout_prior_training_bits,
            "normal_target_training_verifier_bits": normal_target_training_bits,
            "shuffled_target_training_verifier_bits": shuffled_target_training_bits,
            "fresh_transfer_training_verifier_bits": fresh_transfer_training_bits,
            "fresh_curriculum_source_verifier_bits": fresh_curriculum_source_bits,
            "sequence_calibration_verifier_bits": sequence_calibration_bits,
            "fresh_sequence_calibration_verifier_bits": fresh_sequence_calibration_bits,
            "progress_audit_verifier_bits": progress_audit_bits,
            "suite_and_control_verifier_bits": suite_and_control_bits,
            "unique_verifier_bits": unique_verifier_bits,
            "optimizer_updates": (
                args.parent_updates
                + source_acquisition_updates
                + args.sequence_calibration_updates
                + (
                    args.bridge_prior_updates
                    if args.reuse_shared_bridge_prior
                    else 0
                )
                + (
                    args.readout_prior_updates
                    if args.reuse_canonical_readout_prior
                    else 0
                )
                + candidate_count * (args.warmup_updates + args.focus_updates + args.target_updates)
                + candidate_count * (args.warmup_updates + args.focus_updates + args.target_updates)
                + fresh_transfer_candidate_count
                * (args.warmup_updates + args.focus_updates + args.target_updates)
                + fresh_transfer_candidate_count * args.sequence_calibration_updates
                + (
                    fresh_transfer_candidate_count
                    * (
                        args.joint_source_updates
                        if args.joint_source_updates
                        else len(args.source_operations) * args.source_updates
                    )
                    if args.curriculum_fresh_control
                    else 0
                )
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
    parser.add_argument(
        "--joint-source-updates",
        type=int,
        default=0,
        help="balanced source-bank calibration updates (diagnostic upper bound)",
    )
    parser.add_argument(
        "--source-checkpoint-in",
        type=Path,
        default=None,
        help="load a previously verified mastered source bank",
    )
    parser.add_argument(
        "--source-checkpoint-out",
        type=Path,
        default=None,
        help="save the source bank only when every source passes mastery",
    )
    parser.add_argument("--source-restarts", type=int, default=1)
    parser.add_argument(
        "--sequence-calibration-updates",
        type=int,
        default=0,
        help="verifier-trained opaque composition calibration updates before target acquisition",
    )
    parser.add_argument(
        "--use-sequence-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train one append-only external state slot per calibration ordering",
    )
    parser.add_argument(
        "--use-operator-sequence-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train one append-only low-rank operator slot per calibration ordering",
    )
    parser.add_argument(
        "--use-operator-sequence-router",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="learn opaque soft addressing over external operator slots",
    )
    parser.add_argument(
        "--operator-router-temperature",
        type=float,
        default=0.25,
        help="softmax temperature for learned external-operator addressing",
    )
    parser.add_argument(
        "--use-route-outcome-credit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train routing from counterfactual scalar outcomes of every operator slot",
    )
    parser.add_argument(
        "--shared-sequence-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use one reusable decoder across all sequence-calibration programs",
    )
    parser.add_argument(
        "--use-shared-sequence-readout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train one canonical register readout across sequence-calibration programs",
    )
    parser.add_argument(
        "--conditioned-sequence-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use one shared decoder conditioned on opaque program context",
    )
    parser.add_argument(
        "--route-assignment-loss-weight",
        type=float,
        default=0.0,
        help="memory-side weight for balanced low-entropy context-to-slot assignment",
    )
    parser.add_argument(
        "--route-assignment-entropy-weight",
        type=float,
        default=0.01,
        help="relative entropy penalty inside route assignment regularization",
    )
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
    parser.add_argument("--operator-rank", type=int, default=8)
    parser.add_argument(
        "--target-operations", default=",".join(TARGET_OPERATIONS)
    )
    parser.add_argument(
        "--source-order",
        default=",".join(SOURCE_OPERATIONS),
        help="acquisition order for the opaque source operations",
    )
    parser.add_argument(
        "--include-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="interleave a fresh decoder/bridge for a frozen source-program composition",
    )
    parser.add_argument(
        "--preserve-composition-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="feed composition decoders the ordered intermediate register bank",
    )
    parser.add_argument(
        "--reuse-decoder-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="initialize fresh candidate decoders from mastered source decoder 0",
    )
    parser.add_argument(
        "--composition-program-count",
        type=int,
        default=len(COMPOSITION_PROGRAMS),
        help="number of deterministic source-program permutations to interleave",
    )
    parser.add_argument(
        "--target-composition-start",
        type=int,
        default=None,
        help="optional held-out target permutation start; calibration keeps the first programs",
    )
    parser.add_argument(
        "--reuse-shared-bridge-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train and freeze one event bridge from mastered source outcomes",
    )
    parser.add_argument(
        "--reuse-conditioned-bridge-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="condition a reusable bridge on opaque source-program vectors",
    )
    parser.add_argument(
        "--reuse-canonical-readout-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="freeze a source-trained canonical register readout for compositions",
    )
    parser.add_argument(
        "--curriculum-fresh-control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="acquire source primitives before training each fresh target control",
    )
    parser.add_argument("--bridge-prior-updates", type=int, default=128)
    parser.add_argument("--readout-prior-updates", type=int, default=256)
    parser.add_argument("--retention-regression-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
