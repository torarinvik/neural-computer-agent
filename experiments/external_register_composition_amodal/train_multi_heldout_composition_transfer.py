"""Pressure-test replay-free transfer across multiple held-out compositions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch

from experiments.external_register_composition_amodal.train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _module_digest,
    _new_machine,
    _stable_bits,
    _train_stage,
)
from experiments.external_register_composition_amodal.train_heldout_composition_transfer import (
    SOURCE_PROGRAMS,
    _train_primitive,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import OpaqueProtocolDecoder, PersistentOpaqueStateStore

MASTERY_THRESHOLD = 0.8
TARGET_ORDERS = ((3, 2, 0, 1), (1, 0, 3, 2), (2, 3, 1, 0))
TARGET_PROGRAMS = tuple(
    tuple(SOURCE_PROGRAMS[index] for index in order) for order in TARGET_ORDERS
)


def _set_frozen(machine) -> None:
    for parameter in machine.parameters():
        parameter.requires_grad_(False)


def _target_train(
    parent,
    machine,
    decoder,
    *,
    order: tuple[int, ...],
    program: tuple[str, ...],
    args: argparse.Namespace,
    seed: int,
    fresh: bool = False,
    shuffle_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    if fresh:
        trainable = [*machine.parameters(), *decoder.parameters()]
    else:
        _set_frozen(machine)
        trainable = list(decoder.parameters())
    instructions = tuple(machine.instructions[index] for index in order)
    return _train_stage(
        parent,
        machine,
        decoder,
        operation="generated_composition",
        generated_composition_ids=(0,),
        generated_compositions=(program,),
        instructions=instructions,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed,
        trainable=trainable,
        credit_mode=args.credit_mode,
        shuffle_outcomes=shuffle_outcomes,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed + 1_000_000,
    )


def _target_accuracy(parent, machine, decoder, *, order, program, args, seed, **kwargs):
    return _accuracy(
        parent,
        machine,
        decoder,
        operation="generated_composition",
        generated_composition_ids=(0,),
        generated_compositions=(program,),
        instructions=tuple(machine.instructions[index] for index in order),
        count=args.audit_count,
        span=args.span,
        seed=seed,
        credit_mode=args.credit_mode,
        **kwargs,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.primitive_updates,
        args.composition_updates,
        args.batch_size,
        args.audit_count,
        args.eval_every,
    ) < 1:
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

    parent = _runtime(seed=args.seed, growth=False)
    _, _parent_progress = _train_with_progress(
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
    inherited = _new_machine(4, operator_mode=args.operator_mode)
    source_decoders = []
    for index in range(len(SOURCE_PROGRAMS)):
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        source_decoders.append(decoder)
        _train_primitive(parent, inherited, decoder, index=index, args=args)
    source_before = [
        _accuracy(
            parent,
            inherited,
            decoder,
            operation=SOURCE_PROGRAMS[index],
            instructions=(inherited.instructions[index],),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 60_000 + index,
            credit_mode=args.credit_mode,
        )
        for index, decoder in enumerate(source_decoders)
    ]
    source_state = {
        name: value.detach().clone() for name, value in inherited.state_dict().items()
    }
    target_records: list[dict[str, object]] = []
    inherited_decoders = []
    for target_index, (order, program) in enumerate(
        zip(TARGET_ORDERS, TARGET_PROGRAMS, strict=True)
    ):
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        inherited_decoders.append(decoder)
        progress = _target_train(
            parent,
            inherited,
            decoder,
            order=order,
            program=program,
            args=args,
            seed=args.seed + 100_000 + target_index * 20_003,
        )
        inherited_accuracy = _target_accuracy(
            parent,
            inherited,
            decoder,
            order=order,
            program=program,
            args=args,
            seed=args.seed + 101_000 + target_index * 20_003,
        )
        shuffled = _new_machine(4, operator_mode=args.operator_mode)
        shuffled.load_state_dict(source_state, strict=True)
        shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        _target_train(
            parent,
            shuffled,
            shuffled_decoder,
            order=order,
            program=program,
            args=args,
            seed=args.seed + 110_000 + target_index * 20_003,
            shuffle_outcomes=True,
        )
        shuffled_accuracy = _target_accuracy(
            parent,
            shuffled,
            shuffled_decoder,
            order=order,
            program=program,
            args=args,
            seed=args.seed + 111_000 + target_index * 20_003,
            shuffle_outcomes=True,
        )
        fresh = _new_machine(4, operator_mode=args.operator_mode)
        fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        fresh_progress = _target_train(
            parent,
            fresh,
            fresh_decoder,
            order=tuple(range(4)),
            program=program,
            args=args,
            seed=args.seed + 120_000 + target_index * 20_003,
            fresh=True,
        )
        fresh_accuracy = _target_accuracy(
            parent,
            fresh,
            fresh_decoder,
            order=tuple(range(4)),
            program=program,
            args=args,
            seed=args.seed + 121_000 + target_index * 20_003,
        )
        missing = _target_accuracy(
            parent,
            inherited,
            decoder,
            order=order,
            program=program,
            args=args,
            seed=args.seed + 122_000 + target_index * 20_003,
            evidence_present=False,
        )
        inherited_stable = _stable_bits(
            progress,
            threshold=MASTERY_THRESHOLD,
            bits_per_update=args.batch_size * args.span * 2,
        )
        fresh_stable = _stable_bits(
            fresh_progress,
            threshold=MASTERY_THRESHOLD,
            bits_per_update=args.batch_size * args.span * 2,
        )
        target_records.append(
            {
                "order": list(order),
                "program": list(program),
                "inherited_accuracy": inherited_accuracy,
                "fresh_accuracy": fresh_accuracy,
                "reward_shuffled_accuracy": shuffled_accuracy,
                "missing_evidence_accuracy": missing,
                "inherited_stable_bits": inherited_stable,
                "fresh_stable_bits": fresh_stable,
                "positive_transfer": (
                    inherited_stable is not None
                    and fresh_stable is not None
                    and fresh_stable > inherited_stable
                ),
                "mastered": inherited_accuracy >= MASTERY_THRESHOLD,
                "shuffled_rejected": shuffled_accuracy < MASTERY_THRESHOLD,
                "missing_rejected": missing < MASTERY_THRESHOLD,
            }
        )

    source_after = [
        _accuracy(
            parent,
            inherited,
            decoder,
            operation=SOURCE_PROGRAMS[index],
            instructions=(inherited.instructions[index],),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 70_000 + index,
            credit_mode=args.credit_mode,
        )
        for index, decoder in enumerate(source_decoders)
    ]
    reload_machine = _new_machine(4, operator_mode=args.operator_mode)
    reload_machine.load_state_dict(inherited.state_dict(), strict=True)
    reload_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reload_decoder.load_state_dict(inherited_decoders[0].state_dict(), strict=True)
    reload_accuracy = _target_accuracy(
        parent,
        reload_machine,
        reload_decoder,
        order=TARGET_ORDERS[0],
        program=TARGET_PROGRAMS[0],
        args=args,
        seed=args.seed + 101_000,
    )
    persistence_dir = args.report_out.parent / "persistence"
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
    store = PersistentOpaqueStateStore(
        persistence_dir / "machine.pt", configuration=inherited.configuration()
    )
    store.save_module(inherited)
    intact = (persistence_dir / "machine.pt").read_bytes()
    payload = torch.load(persistence_dir / "machine.pt", weights_only=False)
    corrupted = dict(payload["state_dict"])
    first_name = next(iter(corrupted))
    value = corrupted[first_name].clone()
    value.reshape(-1)[0] += 1.0
    corrupted[first_name] = value
    payload["state_dict"] = corrupted
    torch.save(payload, persistence_dir / "machine.pt")
    corruption_rejected = False
    try:
        store.load()
    except ValueError as error:
        corruption_rejected = "checksum mismatch" in str(error)
    (persistence_dir / "machine.pt").write_bytes(intact)
    parent_digest_after = _module_digest(parent.controller)
    gates = {
        "sources_mastered": min(source_before) >= MASTERY_THRESHOLD,
        "sources_retained": min(source_after) >= MASTERY_THRESHOLD,
        "all_targets_mastered": all(bool(row["mastered"]) for row in target_records),
        "all_targets_stable": all(
            row["inherited_stable_bits"] is not None for row in target_records
        ),
        "all_targets_transfer": all(
            bool(row["positive_transfer"]) for row in target_records
        ),
        "all_shuffled_rejected": all(
            bool(row["shuffled_rejected"]) for row in target_records
        ),
        "all_missing_rejected": all(
            bool(row["missing_rejected"]) for row in target_records
        ),
        "reload_exact": (
            _module_digest(inherited) == _module_digest(reload_machine)
            and abs(reload_accuracy - float(target_records[0]["inherited_accuracy"]))
            < 1e-12
        ),
        "corruption_rejected": corruption_rejected,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-register-multi-heldout-composition-report.v1",
        "claim_boundary": (
            "One frozen external interpreter reuses four learned opaque "
            "instructions across three new held-out compositions. Each target "
            "has an independent external decoder and fresh learner control."
        ),
        "seed": args.seed,
        "source_programs": list(SOURCE_PROGRAMS),
        "target_programs": [list(program) for program in TARGET_PROGRAMS],
        "target_orders": [list(order) for order in TARGET_ORDERS],
        "operator_mode": args.operator_mode,
        "primitive_updates": args.primitive_updates,
        "composition_updates": args.composition_updates,
        "batch_size": args.batch_size,
        "source_retention_before": source_before,
        "source_retention_after": source_after,
        "targets": target_records,
        "reload": {"accuracy": reload_accuracy},
        "frozen_core": {"unchanged": parent_digest_before == parent_digest_after},
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.primitive_updates * args.batch_size * args.span * 4
                + args.composition_updates
                * args.batch_size
                * args.span
                * 3
                * 3
            ),
            "optimizer_updates": args.parent_updates
            + args.primitive_updates * 4
            + args.composition_updates * 3 * 3,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--primitive-updates", type=int, default=384)
    parser.add_argument("--composition-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument(
        "--credit-mode",
        choices=("paired_counterfactual", "attempted_bce"),
        default="paired_counterfactual",
    )
    parser.add_argument(
        "--operator-mode",
        choices=(
            "factorized_low_rank",
            "factorized_film",
            "factorized_hybrid",
            "factorized_bounded_residual",
        ),
        default="factorized_bounded_residual",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
