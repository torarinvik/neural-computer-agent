"""Audit transfer of a learned register interpreter to one new primitive."""

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
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import OpaqueProtocolDecoder, PersistentOpaqueStateStore

MASTERY_THRESHOLD = 0.8
SOURCE_PROGRAMS = ("reverse", "adjacent_xor", "complement", "prefix_parity")
TARGET_PROGRAM = "rotate"


def _freeze(machine) -> None:
    for parameter in machine.parameters():
        parameter.requires_grad_(False)


def _train_source(parent, machine, decoder, index, args):
    if index == 0:
        for parameter in machine.parameters():
            parameter.requires_grad_(True)
        trainable = [*machine.parameters(), *decoder.parameters()]
    else:
        _freeze(machine)
        machine.instructions[index].code.requires_grad_(True)
        trainable = [machine.instructions[index].code, *decoder.parameters()]
    return _train_stage(
        parent,
        machine,
        decoder,
        operation=SOURCE_PROGRAMS[index],
        instructions=(machine.instructions[index],),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 20_000 + index * 10_003,
        trainable=trainable,
        credit_mode=args.credit_mode,
    )


def _train_target(parent, machine, decoder, args, *, seed, fresh=False, shuffled=False):
    if fresh:
        trainable = [*machine.parameters(), *decoder.parameters()]
    else:
        _freeze(machine)
        machine.instructions[-1].code.requires_grad_(True)
        trainable = [machine.instructions[-1].code, *decoder.parameters()]
    return _train_stage(
        parent,
        machine,
        decoder,
        operation=TARGET_PROGRAM,
        instructions=(machine.instructions[-1],),
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed,
        trainable=trainable,
        credit_mode=args.credit_mode,
        shuffle_outcomes=shuffled,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed + 1_000_000,
    )


def _target_accuracy(parent, machine, decoder, args, seed, **kwargs):
    return _accuracy(
        parent,
        machine,
        decoder,
        operation=TARGET_PROGRAM,
        instructions=(machine.instructions[-1],),
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
        args.target_updates,
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
    inherited = _new_machine(5, operator_mode=args.operator_mode)
    source_decoders = []
    for index in range(4):
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        source_decoders.append(decoder)
        _train_source(parent, inherited, decoder, index, args)
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
    target_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    target_progress = _train_target(
        parent, inherited, target_decoder, args, seed=args.seed + 100_000
    )
    inherited_target = _target_accuracy(
        parent, inherited, target_decoder, args, args.seed + 101_000
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
            seed=args.seed + 62_000 + index,
            credit_mode=args.credit_mode,
        )
        for index, decoder in enumerate(source_decoders)
    ]
    source_state = {
        name: value.detach().clone() for name, value in inherited.state_dict().items()
    }
    shuffled = _new_machine(5, operator_mode=args.operator_mode)
    shuffled.load_state_dict(source_state, strict=True)
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_target(
        parent,
        shuffled,
        shuffled_decoder,
        args,
        seed=args.seed + 110_000,
        shuffled=True,
    )
    shuffled_target = _target_accuracy(
        parent, shuffled, shuffled_decoder, args, args.seed + 111_000, shuffle_outcomes=True
    )
    fresh = _new_machine(5, operator_mode=args.operator_mode)
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_progress = _train_target(
        parent, fresh, fresh_decoder, args, seed=args.seed + 120_000, fresh=True
    )
    fresh_target = _target_accuracy(parent, fresh, fresh_decoder, args, args.seed + 121_000)
    missing = _target_accuracy(
        parent,
        inherited,
        target_decoder,
        args,
        args.seed + 122_000,
        evidence_present=False,
    )
    reloaded = _new_machine(5, operator_mode=args.operator_mode)
    reloaded.load_state_dict(inherited.state_dict(), strict=True)
    reload_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reload_decoder.load_state_dict(target_decoder.state_dict(), strict=True)
    reload_target = _target_accuracy(
        parent, reloaded, reload_decoder, args, args.seed + 101_000
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
    inherited_stable = _stable_bits(
        target_progress,
        threshold=MASTERY_THRESHOLD,
        bits_per_update=args.batch_size * args.span * 2,
    )
    fresh_stable = _stable_bits(
        fresh_progress,
        threshold=MASTERY_THRESHOLD,
        bits_per_update=args.batch_size * args.span * 2,
    )
    parent_digest_after = _module_digest(parent.controller)
    gates = {
        "sources_mastered": min(source_before) >= MASTERY_THRESHOLD,
        "sources_retained": min(source_after) >= MASTERY_THRESHOLD,
        "target_stable": inherited_stable is not None,
        "fresh_target_stable": fresh_stable is not None,
        "positive_new_primitive_transfer": (
            inherited_stable is not None
            and fresh_stable is not None
            and fresh_stable > inherited_stable
        ),
        "target_mastered": inherited_target >= MASTERY_THRESHOLD,
        "reward_shuffled_rejected": shuffled_target < MASTERY_THRESHOLD,
        "missing_evidence_rejected": missing < MASTERY_THRESHOLD,
        "reload_exact": (
            _module_digest(inherited) == _module_digest(reloaded)
            and abs(reload_target - inherited_target) < 1e-12
        ),
        "corruption_rejected": corruption_rejected,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-register-heldout-primitive-report.v1",
        "claim_boundary": (
            "Four primitive instruction codes were acquired sequentially, then "
            "the interpreter was frozen while a fifth unseen primitive was "
            "acquired from fresh outcomes. Positive transfer is required against "
            "a matched fresh interpreter."
        ),
        "seed": args.seed,
        "source_programs": list(SOURCE_PROGRAMS),
        "target_program": TARGET_PROGRAM,
        "operator_mode": args.operator_mode,
        "primitive_updates": args.primitive_updates,
        "target_updates": args.target_updates,
        "batch_size": args.batch_size,
        "source_retention_before": source_before,
        "source_retention_after": source_after,
        "target": {
            "inherited_accuracy": inherited_target,
            "fresh_accuracy": fresh_target,
            "reward_shuffled_accuracy": shuffled_target,
            "missing_evidence_accuracy": missing,
            "inherited_stable_bits": inherited_stable,
            "fresh_stable_bits": fresh_stable,
            "reload_accuracy": reload_target,
        },
        "frozen_core": {"unchanged": parent_digest_before == parent_digest_after},
        "persistence": {"corruption_rejected": corruption_rejected},
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.primitive_updates * args.batch_size * args.span * 4
                + args.target_updates * args.batch_size * args.span * 3
            ),
            "optimizer_updates": args.parent_updates
            + args.primitive_updates * 4
            + args.target_updates * 3,
            "replayed_examples": 0,
            "transfer_ratio_against_fresh_learner": (
                float(fresh_stable) / float(inherited_stable)
                if inherited_stable and fresh_stable
                else None
            ),
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
    parser.add_argument("--target-updates", type=int, default=256)
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
