"""Audit transfer of a frozen external interpreter to a new opaque procedure.

Four verifier-private procedures are acquired sequentially.  The shared
register interpreter is then frozen and a fifth, previously unseen procedure
is acquired by training only a new opaque instruction vector and decoder from
fresh outcomes.  A matched fresh interpreter of the same size is trained on
the same target.  This distinguishes reusable computation from merely
executing a known instruction chain more reliably.
"""

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
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    generate_runtime_program_grammar,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import OpaqueProtocolDecoder, PersistentOpaqueStateStore

MASTERY_THRESHOLD = 0.8
SOURCE_COUNT = 4


def _set_all_frozen(machine) -> None:
    for parameter in machine.parameters():
        parameter.requires_grad_(False)


def _train_source(
    parent,
    machine,
    decoder,
    *,
    instruction_index: int,
    program_id: int,
    grammar,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    """Acquire one source procedure without replaying earlier procedures."""

    if instruction_index == 0:
        for parameter in machine.parameters():
            parameter.requires_grad_(True)
        trainable = [*machine.parameters(), *decoder.parameters()]
    else:
        _set_all_frozen(machine)
        instruction = machine.instructions[instruction_index]
        instruction.code.requires_grad_(True)
        trainable = [instruction.code, *decoder.parameters()]
    return _train_stage(
        parent,
        machine,
        decoder,
        operation="generated_composition",
        instructions=(machine.instructions[instruction_index],),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 20_000 + instruction_index * 10_003,
        trainable=trainable,
        credit_mode=args.credit_mode,
        generated_composition_ids=(program_id,),
        generated_compositions=grammar,
    )


def _train_target(
    parent,
    machine,
    decoder,
    *,
    target_id: int,
    grammar,
    args: argparse.Namespace,
    seed_offset: int,
    shuffle_outcomes: bool = False,
    fresh_machine: bool = False,
) -> list[dict[str, float | int]]:
    if not fresh_machine:
        _set_all_frozen(machine)
        machine.instructions[-1].code.requires_grad_(True)
        trainable = [machine.instructions[-1].code, *decoder.parameters()]
    else:
        trainable = [*machine.parameters(), *decoder.parameters()]
    return _train_stage(
        parent,
        machine,
        decoder,
        operation="generated_composition",
        instructions=(machine.instructions[-1],),
        updates=args.target_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + seed_offset,
        trainable=trainable,
        credit_mode=args.credit_mode,
        shuffle_outcomes=shuffle_outcomes,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + seed_offset + 1_000_000,
        generated_composition_ids=(target_id,),
        generated_compositions=grammar,
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
    if args.program_count < SOURCE_COUNT + 1:
        raise ValueError("program-count must include four sources and one target")

    grammar = generate_runtime_program_grammar(
        seed=args.program_seed,
        count=args.program_count,
        depth=args.program_depth,
        primitive_family=args.primitive_family,
    )
    source_ids = tuple(range(SOURCE_COUNT))
    target_id = SOURCE_COUNT
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

    inherited = _new_machine(SOURCE_COUNT + 1, operator_mode=args.operator_mode)
    source_decoders = []
    source_progress = []
    for index, program_id in enumerate(source_ids):
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        source_decoders.append(decoder)
        source_progress.append(
            _train_source(
                parent,
                inherited,
                decoder,
                instruction_index=index,
                program_id=program_id,
                grammar=grammar,
                args=args,
            )
        )

    source_retention_before = [
        _accuracy(
            parent,
            inherited,
            source_decoders[index],
            operation="generated_composition",
            instructions=(inherited.instructions[index],),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 60_000 + index,
            credit_mode=args.credit_mode,
            generated_composition_ids=(program_id,),
            generated_compositions=grammar,
        )
        for index, program_id in enumerate(source_ids)
    ]

    source_state = {
        name: value.detach().clone() for name, value in inherited.state_dict().items()
    }
    target_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    target_progress = _train_target(
        parent,
        inherited,
        target_decoder,
        target_id=target_id,
        grammar=grammar,
        args=args,
        seed_offset=100_000,
    )
    inherited_target = _accuracy(
        parent,
        inherited,
        target_decoder,
        operation="generated_composition",
        instructions=(inherited.instructions[-1],),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 101_000,
        credit_mode=args.credit_mode,
        generated_composition_ids=(target_id,),
        generated_compositions=grammar,
    )
    source_retention_after = [
        _accuracy(
            parent,
            inherited,
            source_decoders[index],
            operation="generated_composition",
            instructions=(inherited.instructions[index],),
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 62_000 + index,
            credit_mode=args.credit_mode,
            generated_composition_ids=(program_id,),
            generated_compositions=grammar,
        )
        for index, program_id in enumerate(source_ids)
    ]

    shuffled = _new_machine(SOURCE_COUNT + 1, operator_mode=args.operator_mode)
    shuffled.load_state_dict(source_state, strict=True)
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_target(
        parent,
        shuffled,
        shuffled_decoder,
        target_id=target_id,
        grammar=grammar,
        args=args,
        seed_offset=120_000,
        shuffle_outcomes=True,
    )
    shuffled_target = _accuracy(
        parent,
        shuffled,
        shuffled_decoder,
        operation="generated_composition",
        instructions=(shuffled.instructions[-1],),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 121_000,
        credit_mode=args.credit_mode,
        generated_composition_ids=(target_id,),
        generated_compositions=grammar,
        shuffle_outcomes=True,
    )

    fresh = _new_machine(SOURCE_COUNT + 1, operator_mode=args.operator_mode)
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_progress = _train_target(
        parent,
        fresh,
        fresh_decoder,
        target_id=target_id,
        grammar=grammar,
        args=args,
        seed_offset=130_000,
        fresh_machine=True,
    )
    fresh_target = _accuracy(
        parent,
        fresh,
        fresh_decoder,
        operation="generated_composition",
        instructions=(fresh.instructions[-1],),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 131_000,
        credit_mode=args.credit_mode,
        generated_composition_ids=(target_id,),
        generated_compositions=grammar,
    )
    missing_evidence = _accuracy(
        parent,
        inherited,
        target_decoder,
        operation="generated_composition",
        instructions=(inherited.instructions[-1],),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 132_000,
        credit_mode=args.credit_mode,
        evidence_present=False,
        generated_composition_ids=(target_id,),
        generated_compositions=grammar,
    )

    reloaded = _new_machine(SOURCE_COUNT + 1, operator_mode=args.operator_mode)
    reloaded.load_state_dict(inherited.state_dict(), strict=True)
    reloaded_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reloaded_decoder.load_state_dict(target_decoder.state_dict(), strict=True)
    reload_target = _accuracy(
        parent,
        reloaded,
        reloaded_decoder,
        operation="generated_composition",
        instructions=(reloaded.instructions[-1],),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 101_000,
        credit_mode=args.credit_mode,
        generated_composition_ids=(target_id,),
        generated_compositions=grammar,
    )

    persistence_dir = args.report_out.parent / "persistence"
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
    store = PersistentOpaqueStateStore(
        persistence_dir / "machine.pt", configuration=inherited.configuration()
    )
    store.save_module(inherited)
    intact_payload = (persistence_dir / "machine.pt").read_bytes()
    payload = torch.load(persistence_dir / "machine.pt", weights_only=False)
    first_name = next(iter(payload["state_dict"]))
    corrupted = dict(payload["state_dict"])
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
    (persistence_dir / "machine.pt").write_bytes(intact_payload)

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
        "sources_mastered_before_target": min(source_retention_before)
        >= MASTERY_THRESHOLD,
        "sources_retained_after_target": min(source_retention_after)
        >= MASTERY_THRESHOLD,
        "target_stable": inherited_stable is not None,
        "fresh_target_stable": fresh_stable is not None,
        "positive_heldout_transfer": (
            inherited_stable is not None
            and fresh_stable is not None
            and fresh_stable > inherited_stable
        ),
        "target_mastered": inherited_target >= MASTERY_THRESHOLD,
        "reward_shuffled_rejected": shuffled_target < MASTERY_THRESHOLD,
        "missing_evidence_rejected": missing_evidence < MASTERY_THRESHOLD,
        "reload_exact": (
            _module_digest(inherited) == _module_digest(reloaded)
            and _module_digest(target_decoder) == _module_digest(reloaded_decoder)
            and abs(reload_target - inherited_target) < 1e-12
        ),
        "corruption_rejected": corruption_rejected,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-register-heldout-transfer-report.v1",
        "claim_boundary": (
            "Four verifier-private procedures were acquired sequentially. A "
            "frozen shared register interpreter then acquired a fifth unseen "
            "procedure using only a new opaque instruction code and fresh "
            "outcomes. Positive transfer is required against a matched fresh "
            "interpreter; this is not assumed from target mastery alone."
        ),
        "seed": args.seed,
        "program_seed": args.program_seed,
        "primitive_family": args.primitive_family,
        "program_depth": args.program_depth,
        "programs": [list(program) for program in grammar],
        "source_ids": list(source_ids),
        "target_id": target_id,
        "operator_mode": args.operator_mode,
        "primitive_updates": args.primitive_updates,
        "target_updates": args.target_updates,
        "batch_size": args.batch_size,
        "span": args.span,
        "audit_count": args.audit_count,
        "source_retention_before_target": source_retention_before,
        "source_retention_after_target": source_retention_after,
        "target": {
            "inherited_accuracy": inherited_target,
            "fresh_accuracy": fresh_target,
            "reward_shuffled_accuracy": shuffled_target,
            "missing_evidence_accuracy": missing_evidence,
            "inherited_stable_bits": inherited_stable,
            "fresh_stable_bits": fresh_stable,
            "reload_accuracy": reload_target,
        },
        "parent": {"final_accuracy": float(parent_progress[-1]["heldout_accuracy"])},
        "frozen_core": {
            "before": parent_digest_before,
            "after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "persistence": {"corruption_rejected": corruption_rejected},
        "accounting": {
            "unique_verifier_bits": (
                args.primitive_updates * args.batch_size * args.span * SOURCE_COUNT
                + args.target_updates * args.batch_size * args.span * 2 * 3
                + args.parent_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.primitive_updates * SOURCE_COUNT
                + args.target_updates * 3
            ),
            "replayed_examples": 0,
            "stable_bits_to_threshold": inherited_stable,
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
    parser.add_argument("--program-seed", type=int, default=77_031)
    parser.add_argument("--program-count", type=int, default=5)
    parser.add_argument("--program-depth", type=int, default=4)
    parser.add_argument(
        "--primitive-family", choices=("registry", "opaque_rule"), default="registry"
    )
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--primitive-updates", type=int, default=128)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--credit-mode", choices=("paired_counterfactual", "attempted_bce"), default="paired_counterfactual")
    parser.add_argument(
        "--operator-mode",
        choices=("factorized_low_rank", "factorized_film", "factorized_hybrid", "factorized_bounded_residual"),
        default="factorized_bounded_residual",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
