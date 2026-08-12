"""Audit four-instruction external-register growth on a runtime grammar.

The verifier-private grammar supplies one four-primitive program and its
reversed-order control. Primitive procedures are acquired sequentially with a
frozen parent and no replay. Program names and grammar identifiers stay in the
verifier; the register receives only learned events, opaque feedback, and
memory-selected instruction vectors.
"""

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
from neural_computer import OpaqueProtocolDecoder, PersistentOpaqueStateStore

from .train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    GeneratedCompositionGrammar,
    _accuracy,
    _module_digest,
    _new_machine,
    _stable_bits,
    _train_stage,
)

FOUR_PRIMITIVES = ("reverse", "adjacent_xor", "complement", "prefix_parity")
FOUR_PROGRAM: tuple[str, ...] = FOUR_PRIMITIVES
REVERSED_FOUR_PROGRAM: tuple[str, ...] = tuple(reversed(FOUR_PROGRAM))
FOUR_GRAMMAR: GeneratedCompositionGrammar = (FOUR_PROGRAM,)
REVERSED_FOUR_GRAMMAR: GeneratedCompositionGrammar = (REVERSED_FOUR_PROGRAM,)
FOUR_COMPOSITION_IDS = (0,)
REVERSED_FOUR_COMPOSITION_IDS = (0,)


def _shared_interpreter_parameters(machine):
    instruction_parameter_ids = {
        id(instruction.code) for instruction in machine.instructions
    }
    return [
        parameter
        for parameter in machine.parameters()
        if id(parameter) not in instruction_parameter_ids
    ]


def _pretrain_shared_interpreter(parent, machine, args: argparse.Namespace) -> None:
    """Train only the shared operator blueprint, then discard its probes."""

    shared_parameters = _shared_interpreter_parameters(machine)
    for index, operation in enumerate(FOUR_PRIMITIVES):
        instruction = machine.instructions[index]
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        _train_stage(
            parent,
            machine,
            decoder,
            operation=operation,
            instructions=(instruction,),
            updates=args.interpreter_pretrain_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 20_000 + index * 10_000,
            trainable=[*shared_parameters, instruction.code, *decoder.parameters()],
            credit_mode=args.credit_mode,
        )
    with torch.no_grad():
        for instruction in machine.instructions:
            instruction.code.normal_(mean=0.0, std=0.02)


def _pretrain_compositional_interpreter(
    parent, machine, args: argparse.Namespace
) -> None:
    """Expose the shared blueprint to a composition before resetting codes."""

    decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=tuple(machine.instructions),
        updates=args.interpreter_composition_pretrain_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 60_000,
        trainable=[*machine.parameters(), *decoder.parameters()],
        credit_mode=args.credit_mode,
    )


def _primitive_retention(
    parent,
    machine,
    decoders: dict[str, OpaqueProtocolDecoder],
    instructions,
    *,
    count: int,
    span: int,
    seed: int,
    credit_mode: str,
) -> dict[str, float]:
    return {
        operation: _accuracy(
            parent,
            machine,
            decoders[operation],
            operation=operation,
            instructions=(instructions[index],),
            count=count,
            span=span,
            seed=seed + index,
            credit_mode=credit_mode,
        )
        for index, operation in enumerate(decoders)
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
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

    machine = _new_machine(
        len(FOUR_PRIMITIVES), operator_mode=args.operator_mode
    )
    blueprint_state = None
    if args.interpreter_pretrain_updates > 0:
        _pretrain_shared_interpreter(parent, machine, args)
    if args.interpreter_composition_pretrain_updates > 0:
        _pretrain_compositional_interpreter(parent, machine, args)
    if (
        args.interpreter_pretrain_updates > 0
        or args.interpreter_composition_pretrain_updates > 0
    ):
        blueprint_state = {
            name: value.detach().clone()
            for name, value in machine.state_dict().items()
            if not name.startswith("instructions.")
        }
    instructions = tuple(machine.instructions)
    decoders: dict[str, OpaqueProtocolDecoder] = {}
    retention_by_stage: list[dict[str, float]] = []
    primitive_final: dict[str, float] = {}

    for index, (operation, instruction) in enumerate(
        zip(FOUR_PRIMITIVES, instructions, strict=True)
    ):
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        decoders[operation] = decoder
        if index == 0:
            for parameter in machine.parameters():
                parameter.requires_grad_(True)
            trainable = list(machine.parameters()) + list(decoder.parameters())
        else:
            for parameter in machine.parameters():
                parameter.requires_grad_(False)
            instruction.code.requires_grad_(True)
            trainable = [instruction.code, *decoder.parameters()]
        _train_stage(
            parent,
            machine,
            decoder,
            operation=operation,
            instructions=(instruction,),
            updates=args.primitive_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 70_000 + index * 10_000,
            trainable=trainable,
            credit_mode=args.credit_mode,
        )
        primitive_final[operation] = _accuracy(
            parent,
            machine,
            decoder,
            operation=operation,
            instructions=(instruction,),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 71_000 + index * 10_000,
            credit_mode=args.credit_mode,
        )
        retention_by_stage.append(
            _primitive_retention(
                parent,
                machine,
                decoders,
                instructions,
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 72_000 + index * 10_000,
                credit_mode=args.credit_mode,
            )
        )

    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    composition_instructions = instructions
    composition_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    composition_progress = _train_stage(
        parent,
        machine,
        composition_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=composition_instructions,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 110_000,
        trainable=list(composition_decoder.parameters()),
        credit_mode=args.credit_mode,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 111_000,
    )
    composition_accuracy = _accuracy(
        parent,
        machine,
        composition_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=composition_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 112_000,
        credit_mode=args.credit_mode,
    )

    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        shuffled_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=composition_instructions,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 113_000,
        trainable=list(shuffled_decoder.parameters()),
        credit_mode=args.credit_mode,
        shuffle_outcomes=True,
    )
    shuffled_accuracy = _accuracy(
        parent,
        machine,
        shuffled_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=composition_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 114_000,
        credit_mode=args.credit_mode,
    )

    fresh_machine = _new_machine(
        len(FOUR_PRIMITIVES), operator_mode=args.operator_mode
    )
    if blueprint_state is not None:
        missing, unexpected = fresh_machine.load_state_dict(
            blueprint_state, strict=False
        )
        if unexpected or any(not name.startswith("instructions.") for name in missing):
            raise RuntimeError(
                "shared interpreter blueprint did not load cleanly: "
                f"missing={missing}, unexpected={unexpected}"
            )
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_progress = _train_stage(
        parent,
        fresh_machine,
        fresh_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=tuple(fresh_machine.instructions),
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 115_000,
        trainable=list(fresh_machine.parameters()) + list(fresh_decoder.parameters()),
        credit_mode=args.credit_mode,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 117_000,
    )
    fresh_accuracy = _accuracy(
        parent,
        fresh_machine,
        fresh_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=tuple(fresh_machine.instructions),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 116_000,
        credit_mode=args.credit_mode,
    )
    missing_evidence_accuracy = _accuracy(
        parent,
        machine,
        composition_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=composition_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 118_000,
        credit_mode=args.credit_mode,
        evidence_present=False,
    )

    reversed_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reversed_instructions = tuple(reversed(instructions))
    _train_stage(
        parent,
        machine,
        reversed_decoder,
        operation="generated_composition",
        generated_composition_ids=REVERSED_FOUR_COMPOSITION_IDS,
        generated_compositions=REVERSED_FOUR_GRAMMAR,
        instructions=reversed_instructions,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 119_000,
        trainable=list(reversed_decoder.parameters()),
        credit_mode=args.credit_mode,
    )
    reversed_order_accuracy = _accuracy(
        parent,
        machine,
        reversed_decoder,
        operation="generated_composition",
        generated_composition_ids=REVERSED_FOUR_COMPOSITION_IDS,
        generated_compositions=REVERSED_FOUR_GRAMMAR,
        instructions=reversed_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 120_000,
        credit_mode=args.credit_mode,
    )

    reloaded_machine = _new_machine(
        len(FOUR_PRIMITIVES), operator_mode=args.operator_mode
    )
    reloaded_machine.load_state_dict(machine.state_dict(), strict=True)
    reloaded_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reloaded_decoder.load_state_dict(composition_decoder.state_dict(), strict=True)
    reload_accuracy = _accuracy(
        parent,
        reloaded_machine,
        reloaded_decoder,
        operation="generated_composition",
        generated_composition_ids=FOUR_COMPOSITION_IDS,
        generated_compositions=FOUR_GRAMMAR,
        instructions=tuple(reloaded_machine.instructions),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 112_000,
        credit_mode=args.credit_mode,
    )
    parent_digest_after = _module_digest(parent.controller)

    persistence_dir = args.report_out.parent / "persistence"
    store = PersistentOpaqueStateStore(
        persistence_dir / "machine.pt",
        configuration=machine.configuration(),
    )
    machine_store_digest = store.save_module(machine)
    intact_payload = (persistence_dir / "machine.pt").read_bytes()
    payload = torch.load(
        persistence_dir / "machine.pt",
        map_location="cpu",
        weights_only=False,
    )
    state_dict = dict(payload["state_dict"])
    first_name = next(iter(state_dict))
    corrupted = state_dict[first_name].clone()
    corrupted.reshape(-1)[0] += 1.0
    state_dict[first_name] = corrupted
    payload["state_dict"] = state_dict
    torch.save(payload, persistence_dir / "machine.pt")
    corruption_rejected = False
    try:
        store.load()
    except ValueError as error:
        corruption_rejected = "checksum mismatch" in str(error)
    (persistence_dir / "machine.pt").write_bytes(intact_payload)

    bits_per_update = args.batch_size * args.span * 2
    composition_stable_bits = _stable_bits(
        composition_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    fresh_stable_bits = _stable_bits(
        fresh_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    retention = retention_by_stage[-1]
    promotion_gates = {
        "composition_stable": composition_stable_bits is not None,
        "fresh_stable": fresh_stable_bits is not None,
        "positive_stable_transfer": (
            composition_stable_bits is not None
            and fresh_stable_bits is not None
            and fresh_stable_bits > composition_stable_bits
        ),
        "retained_all_primitives": all(
            value >= args.mastery_threshold for value in retention.values()
        ),
        "reward_shuffled_rejected": shuffled_accuracy < args.mastery_threshold,
        "missing_evidence_rejected": missing_evidence_accuracy < args.mastery_threshold,
        "reload_exact": (
            _module_digest(machine) == _module_digest(reloaded_machine)
            and _module_digest(composition_decoder)
            == _module_digest(reloaded_decoder)
        ),
        "corruption_rejected": corruption_rejected,
        "frozen_parent": parent_digest_before == parent_digest_after,
    }
    promotion_accepted = all(promotion_gates.values())
    transfer_ratio = (
        float(fresh_stable_bits) / float(composition_stable_bits)
        if composition_stable_bits and fresh_stable_bits
        else None
    )
    report = {
        "schema": "neural-computer.external-register-four-instruction-report.v1",
        "claim_boundary": "Four opaque external instructions compose one verifier-private runtime-grammar program through one frozen parent and factorized register; this is not general continual learning.",
        "seed": args.seed,
        "program": list(FOUR_PROGRAM),
        "reversed_program": list(REVERSED_FOUR_PROGRAM),
        "parent": {
            "updates": args.parent_updates,
            "final_heldout_accuracy": float(parent_progress[-1]["heldout_accuracy"]),
        },
        "primitive_operations": list(FOUR_PRIMITIVES),
        "primitive_updates": args.primitive_updates,
        "composition_updates": args.composition_updates,
        "credit_mode": args.credit_mode,
        "execution_mode": "read_execute",
        "operator_mode": args.operator_mode,
        "interpreter_pretrain_updates": args.interpreter_pretrain_updates,
        "interpreter_composition_pretrain_updates": (
            args.interpreter_composition_pretrain_updates
        ),
        "batch_size": args.batch_size,
        "span": args.span,
        "audit_count": args.audit_count,
        "primitive_final_accuracy": primitive_final,
        "primitive_retention_by_stage": retention_by_stage,
        "results": {
            "composition": composition_accuracy,
            "reversed_order_composition": reversed_order_accuracy,
            "reward_shuffled_composition": shuffled_accuracy,
            "fresh_composition": fresh_accuracy,
            "missing_evidence_composition": missing_evidence_accuracy,
        },
        "learning_curves": {
            "composition": composition_progress,
            "fresh": fresh_progress,
        },
        "persistence": {
            "machine_digest": _module_digest(machine),
            "reloaded_machine_digest": _module_digest(reloaded_machine),
            "decoder_digest": _module_digest(composition_decoder),
            "reloaded_decoder_digest": _module_digest(reloaded_decoder),
            "reload_exact": promotion_gates["reload_exact"],
            "reloaded_composition": reload_accuracy,
            "machine_store_digest": machine_store_digest,
            "corruption_rejected": corruption_rejected,
        },
        "frozen_core": {
            "parent_digest_before": parent_digest_before,
            "parent_digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": args.interpreter_pretrain_updates
            * args.batch_size
            * args.span
            * len(FOUR_PRIMITIVES)
            + args.interpreter_composition_pretrain_updates
            * args.batch_size
            * args.span
            * (len(FOUR_PRIMITIVES) + 1)
            + args.primitive_updates
            * args.batch_size
            * args.span
            * len(FOUR_PRIMITIVES)
            + args.composition_updates * args.batch_size * args.span * (len(FOUR_PRIMITIVES) + 1),
            "unique_logical_lifetimes": args.interpreter_pretrain_updates
            * args.batch_size
            * len(FOUR_PRIMITIVES)
            + args.interpreter_composition_pretrain_updates
            * args.batch_size
            * (len(FOUR_PRIMITIVES) + 1)
            + args.primitive_updates
            * args.batch_size
            * len(FOUR_PRIMITIVES)
            + args.composition_updates * args.batch_size * (len(FOUR_PRIMITIVES) + 1),
            "optimizer_updates": args.parent_updates
            + args.interpreter_pretrain_updates * len(FOUR_PRIMITIVES)
            + args.interpreter_composition_pretrain_updates
            * (len(FOUR_PRIMITIVES) + 1)
            + args.primitive_updates * len(FOUR_PRIMITIVES)
            + args.composition_updates * (len(FOUR_PRIMITIVES) + 1),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
            "stable_bits_to_threshold": composition_stable_bits,
            "composition_stable_bits_to_threshold": composition_stable_bits,
            "fresh_stable_bits_to_threshold": fresh_stable_bits,
            "retention_on_mastered_primitives": min(retention.values()),
            "transfer_ratio_against_fresh_learner": transfer_ratio,
        },
        "promotion": {
            "accepted": promotion_accepted,
            "gates": promotion_gates,
            "reason": (
                "narrow_four_instruction_composition_promoted"
                if promotion_accepted
                else "one_or_more_registered_gates_failed"
            ),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--primitive-updates", type=int, default=128)
    parser.add_argument("--composition-updates", type=int, default=128)
    parser.add_argument("--interpreter-pretrain-updates", type=int, default=0)
    parser.add_argument(
        "--interpreter-composition-pretrain-updates", type=int, default=0
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.8)
    parser.add_argument(
        "--operator-mode",
        choices=(
            "factorized_low_rank",
            "factorized_film",
            "factorized_hybrid",
            "factorized_bounded_residual",
        ),
        default="factorized_low_rank",
    )
    parser.add_argument(
        "--credit-mode",
        choices=("paired_counterfactual", "attempted_bce"),
        default="paired_counterfactual",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
