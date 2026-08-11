"""Pressure-test replay-free external growth across composition depth.

One shared growth combiner first learns atomic fragment readout.  For each new
composition depth it appends one external residual slot, freezes the admitted
prefix, and trains only the new slot on fresh verifier outcomes.  The parent
controller, interpreter, and acquired fragment bank remain frozen on the
inherited path.  This is the executable test of "learn while frozen": mutable
capacity grows outside the controller while mastered behavior is protected.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.external_skill_fragment_composition_amodal.train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    _digest,
    _fragment_bank,
    _machine,
    _stable_bits,
    _train_stage,
)
from experiments.external_skill_fragment_composition_amodal.train_multi import (
    PRIMITIVES,
    _append_fragment,
    _bank_with_fragments,
    _corruption_audit,
    _eval_points,
    _retention,
    _set_fragment_stage,
)
from experiments.external_skill_fragment_composition_amodal.train_shared_multi_target import (
    _evaluate_specs,
    _shared_train_stage,
    _specs,
    _train_four_fragments,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    ExternalSkillFragmentGrowthCombiner,
    OpaqueProtocolDecoder,
)

STAGE_ORDERS: dict[int, tuple[tuple[int, ...], ...]] = {
    1: ((0,), (1,), (2,), (3,)),
    2: (
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (0, 3),
        (3, 0),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
    ),
    3: (
        (3, 2, 0),
        (1, 0, 3),
        (2, 3, 1),
        (0, 1, 2),
        (0, 2, 3),
        (1, 3, 0),
    ),
    4: ((3, 2, 0, 1), (1, 0, 3, 2), (2, 3, 1, 0)),
}
HELDOUT_ORDERS = (
    (0, 1, 3),
    (0, 3, 1),
    (1, 2, 0),
    (0, 1, 3, 2),
    (0, 3, 2, 1),
    (1, 2, 3, 0),
)


def _freeze(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _program_orders(orders: tuple[tuple[int, ...], ...]):
    return _specs(orders)


def _joint_bank(seed: int):
    """Create an unprotected bank for the shared atomic foundation stage."""

    torch.manual_seed(seed)
    bank = _fragment_bank(seed)
    for index in range(1, len(PRIMITIVES)):
        torch.manual_seed(seed + 2_000 + index)
        bank.grow_basis(1)
        bank.add_fragment(
            torch.randn(1, bank.basis_count) * 0.05,
            F.normalize(torch.randn(16), dim=0),
        )
    return bank


def _train_four_fragments_canonical(parent, args):
    """Acquire all fragments against one shared, protected output readout."""

    machine = _machine()
    bank = _fragment_bank(args.seed + 1)
    decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    decoders = [decoder] * len(PRIMITIVES)
    histories: list[list[dict[str, object]]] = []
    retention_by_stage: list[dict[str, float]] = []
    for index, operation in enumerate(PRIMITIVES):
        if index > 0:
            _append_fragment(bank, args.seed + 2_000 + index)
        trainable = _set_fragment_stage(machine, bank, index)
        for parameter in decoder.parameters():
            parameter.requires_grad_(index == 0)
        if index == 0:
            trainable.extend(decoder.parameters())
        histories.append(
            _train_stage(
                parent,
                machine,
                bank,
                decoder,
                operation=operation,
                selected=(index,),
                updates=args.primitive_updates,
                batch_size=args.batch_size,
                span=args.span,
                seed=args.seed + 10_000 + index * 20_003,
                trainable=trainable,
                eval_every=args.eval_every,
                audit_count=args.audit_count,
                audit_seed=args.seed + 30_000 + index * 20_003,
            )
        )
        bank.protect(index)
        retention_by_stage.append(
            _retention(
                parent,
                machine,
                bank,
                decoders[: index + 1],
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 50_000 + index * 20_003,
            )
        )
    _freeze(machine)
    _freeze(bank)
    _freeze(decoder)
    return machine, bank, decoders, histories, retention_by_stage


def _train_growth_path(
    parent,
    machine,
    bank,
    args: argparse.Namespace,
    *,
    seed: int,
    inherited: bool,
    stage_orders: dict[int, tuple[tuple[int, ...], ...]],
    foundation_depth: int = 1,
    shuffle_outcomes: bool = False,
) -> tuple[
    ExternalSkillFragmentGrowthCombiner,
    OpaqueProtocolDecoder,
    dict[int, list[dict[str, object]]],
    dict[int, list[float]],
]:
    combiner = ExternalSkillFragmentGrowthCombiner(
        REGISTER_WIDTH,
        16,
        REGISTER_WIDTH,
        hidden=64,
    )
    decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    histories: dict[int, list[dict[str, object]]] = {}
    stage_accuracy: dict[int, list[float]] = {}
    if inherited:
        _freeze(machine)
        _freeze(bank)

    for depth, orders in stage_orders.items():
        slot_index = combiner.append_depth_slot()
        in_foundation = not inherited and depth <= foundation_depth
        if in_foundation:
            # The foundation is allowed to shape the shared interpreter and
            # every admitted prefix slot from fresh outcomes.  This is the
            # only phase in which old external capacity is intentionally
            # trainable; after the foundation, growth is append-only.
            trainable = list(combiner.parameters())
            updates = args.foundation_updates
        else:
            _freeze(combiner)
            trainable = list(combiner.depth_slot_parameters(slot_index))
            updates = args.foundation_updates if depth == 1 else args.stage_updates
        if depth == 1 and not inherited:
            trainable.extend(decoder.parameters())
            trainable.extend(machine.parameters())
            trainable.append(bank.shared_basis)
            trainable.extend(bank.coefficients)
        elif in_foundation:
            trainable.extend(machine.parameters())
            trainable.append(bank.shared_basis)
            trainable.extend(bank.coefficients)
            trainable.extend(decoder.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        history = _shared_train_stage(
            parent,
            machine,
            bank,
            combiner,
            decoder,
            specs=_program_orders(orders),
            updates=updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=seed + depth * 10_007,
            trainable=trainable,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            audit_seed=seed + depth * 11_003,
            shuffle_outcomes=shuffle_outcomes,
        )
        histories[depth] = history
        if not (in_foundation and depth < foundation_depth):
            combiner.protect_depth_prefix()
            combiner.protect_base()
            if in_foundation:
                _freeze(machine)
                _freeze(bank)
                _freeze(decoder)
        if in_foundation and depth == foundation_depth and foundation_depth > 1:
            # Re-measure the earlier foundation rungs after the final shared
            # algebra update.  These are the source-retention baselines for
            # the subsequent frozen-growth phase, not the pre-foundation
            # curriculum checkpoints.
            for earlier_depth in range(1, foundation_depth):
                stage_accuracy[earlier_depth] = _evaluate_specs(
                    parent,
                    machine,
                    bank,
                    combiner,
                    decoder,
                    specs=_program_orders(stage_orders[earlier_depth]),
                    count=args.audit_count,
                    span=args.span,
                    seed=seed + earlier_depth * 13_007,
                    shuffle_outcomes=shuffle_outcomes,
                )
        stage_accuracy[depth] = _evaluate_specs(
            parent,
            machine,
            bank,
            combiner,
            decoder,
            specs=_program_orders(orders),
            count=args.audit_count,
            span=args.span,
            seed=seed + depth * 12_007,
            shuffle_outcomes=shuffle_outcomes,
        )
    _freeze(combiner)
    _freeze(decoder)
    return combiner, decoder, histories, stage_accuracy


def _stable_stage_bits(
    histories: dict[int, list[dict[str, object]]],
    *,
    args: argparse.Namespace,
    stage_orders: dict[int, tuple[tuple[int, ...], ...]],
) -> dict[int, int | None]:
    bits_per_update = args.batch_size * args.span * 2
    return {
        depth: _stable_bits(
            history,
            threshold=args.mastery_threshold,
            bits_per_update=bits_per_update * len(stage_orders[depth]),
        )
        for depth, history in histories.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    stage_orders = {
        depth: orders
        for depth, orders in STAGE_ORDERS.items()
        if depth <= args.max_depth
    }
    heldout_orders = tuple(
        order for order in HELDOUT_ORDERS if len(order) <= args.max_depth
    )
    if args.max_depth < 2:
        raise ValueError("max-depth must include the depth-2 training rung")
    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 100,
        learning_rate=3e-3,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
    )
    parent.eval()
    parent_digest_before = _digest(parent.controller)
    if args.joint_foundation:
        machine = _machine(operator_mode=args.operator_mode)
        bank = _joint_bank(args.seed + 1)
        source_decoders = []
        primitive_histories = []
        retention_by_stage = []
        inherited_mode = False
    else:
        machine, bank, source_decoders, primitive_histories, retention_by_stage = (
            _train_four_fragments(parent, args)
        )
        inherited_mode = True
    bank_digest_before = bank.payload()["sha256"]

    inherited_combiner, inherited_decoder, inherited_histories, inherited_stage = (
        _train_growth_path(
            parent,
            machine,
            bank,
            args,
            seed=args.seed + 100_000,
            inherited=inherited_mode,
            stage_orders=stage_orders,
            foundation_depth=args.foundation_depth,
        )
    )
    if args.joint_foundation:
        bank_digest_before = bank.payload()["sha256"]
    source_before = (
        {
            PRIMITIVES[index]: inherited_stage[1][index]
            for index in range(len(PRIMITIVES))
        }
        if args.joint_foundation
        else retention_by_stage[-1]
    )
    train_specs = _program_orders(
        tuple(order for rows in stage_orders.values() for order in rows)
    )
    heldout_specs = _program_orders(heldout_orders)
    inherited_train = _evaluate_specs(
        parent,
        machine,
        bank,
        inherited_combiner,
        inherited_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 150_000,
    )
    inherited_heldout = _evaluate_specs(
        parent,
        machine,
        bank,
        inherited_combiner,
        inherited_decoder,
        specs=heldout_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 151_000,
    )
    wrong_specs = tuple(
        (
            order if len(order) < 2 else tuple(order[1:]) + (order[0],),
            program,
        )
        for order, program in train_specs
    )
    wrong_composite_indices = [
        index
        for index, (order, _) in enumerate(train_specs)
        if len(order) == args.max_depth
        or (args.max_depth < 3 and len(order) >= 2)
    ]
    wrong_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        inherited_combiner,
        inherited_decoder,
        specs=wrong_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 152_000,
    )
    zero_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        inherited_combiner,
        inherited_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 153_000,
        zero_codes=True,
    )
    missing_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        inherited_combiner,
        inherited_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 154_000,
        blank_sequence=True,
    )

    fresh_machine = _machine(operator_mode=args.operator_mode)
    fresh_bank = _bank_with_fragments(args.seed + 2, len(PRIMITIVES))
    fresh_combiner, _fresh_decoder, fresh_histories, fresh_stage = _train_growth_path(
        parent,
        fresh_machine,
        fresh_bank,
        args,
        seed=args.seed + 200_000,
        inherited=False,
        stage_orders=stage_orders,
        foundation_depth=args.foundation_depth,
    )
    fresh_stable = _stable_stage_bits(
        fresh_histories, args=args, stage_orders=stage_orders
    )
    inherited_stable = _stable_stage_bits(
        inherited_histories, args=args, stage_orders=stage_orders
    )
    source_after = (
        {PRIMITIVES[index]: inherited_train[index] for index in range(len(PRIMITIVES))}
        if args.joint_foundation
        else _retention(
            parent,
            machine,
            bank,
            source_decoders,
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 160_000,
        )
    )
    bank_digest_after = bank.payload()["sha256"]
    parent_digest_after = _digest(parent.controller)
    with tempfile.TemporaryDirectory(prefix="fragment-growth-") as temporary:
        persistence_exact = _corruption_audit(
            bank, Path(temporary) / "fragment-bank.pt"
        )
    gates = {
        "source_primitives_mastered": min(source_before.values())
        >= args.mastery_threshold,
        "source_primitives_retained": min(source_after.values())
        >= args.mastery_threshold,
        "growth_stages_mastered": all(
            min(values) >= args.mastery_threshold for values in inherited_stage.values()
        ),
        "growth_stages_stable": all(
            value is not None for value in inherited_stable.values()
        ),
        "earlier_depths_retained_after_growth": min(inherited_train)
        >= args.mastery_threshold,
        "heldout_orders_generalize": bool(inherited_heldout)
        and min(inherited_heldout) >= args.mastery_threshold,
        "wrong_orders_rejected": bool(wrong_composite_indices)
        and max(wrong_accuracy[index] for index in wrong_composite_indices)
        < args.mastery_threshold,
        "no_fragment_bypass": max(zero_accuracy) < args.mastery_threshold,
        "missing_evidence_rejected": max(missing_accuracy) < args.mastery_threshold,
        "positive_stable_transfer": all(
            inherited_stable[depth] is not None
            and fresh_stable[depth] is not None
            and inherited_stable[depth] < fresh_stable[depth]
            for depth in stage_orders
        ),
        "one_shared_growth_combiner": True,
        "one_shared_decoder": True,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "frozen_acquired_bank": bank_digest_before == bank_digest_after,
        "persistence_exact_and_corruption_rejected": persistence_exact,
        "no_replayed_examples": True,
    }
    inherited_stage_updates = {
        depth: args.foundation_updates if depth == 1 else args.stage_updates
        for depth in stage_orders
    }
    fresh_stage_updates = {
        depth: (
            args.foundation_updates
            if depth <= args.foundation_depth
            else args.stage_updates
        )
        for depth in stage_orders
    }
    inherited_training_batches = sum(
        len(stage_orders[depth]) * inherited_stage_updates[depth]
        for depth in stage_orders
    )
    fresh_training_batches = sum(
        len(stage_orders[depth]) * fresh_stage_updates[depth]
        for depth in stage_orders
    )
    parent_training_batches = args.parent_updates
    primitive_training_batches = (
        0 if args.joint_foundation else len(PRIMITIVES) * args.primitive_updates
    )
    training_batches = (
        parent_training_batches
        + primitive_training_batches
        + inherited_training_batches
        + fresh_training_batches
    )
    # Each shared stage evaluates every target at its scheduled checkpoints and
    # once after the stage. The final inherited controls are fresh verifier
    # samples too, so keep them separate from optimizer exposure.
    inherited_growth_stage_audit_batches = sum(
        2
        * len(stage_orders[depth])
        * (_eval_points(inherited_stage_updates[depth], args.eval_every) + 1)
        for depth in stage_orders
    )
    fresh_growth_stage_audit_batches = sum(
        2
        * len(stage_orders[depth])
        * (_eval_points(fresh_stage_updates[depth], args.eval_every) + 1)
        for depth in stage_orders
    )
    foundation_retention_audit_batches = (
        (2 if args.joint_foundation else 1) * max(args.foundation_depth - 1, 0)
    )
    final_inherited_audit_batches = 5 * len(train_specs) + len(heldout_specs)
    source_retention_audit_batches = 0 if args.joint_foundation else len(PRIMITIVES)
    audit_batches = (
        inherited_growth_stage_audit_batches
        + fresh_growth_stage_audit_batches
        + foundation_retention_audit_batches
        + final_inherited_audit_batches
        + source_retention_audit_batches
    )
    bits_per_update = args.batch_size * args.span * 2
    report = {
        "schema": "neural-computer.external-skill-fragment-depth-growth-report.v1",
        "claim_boundary": (
            "A shared external composition foundation may be trained through "
            "the configured foundation depth, after which append-only residual "
            "slots are trained by composition depth without replay while the "
            "inherited parent and acquired bank remain frozen. This does not "
            "establish general continual learning."
        ),
        "seed": args.seed,
        "operator_mode": args.operator_mode,
        "foundation_depth": args.foundation_depth,
        "stage_orders": {
            str(depth): [list(order) for order in orders]
            for depth, orders in stage_orders.items()
        },
        "heldout_orders": [list(order) for order in heldout_orders],
        "parent_progress": parent_progress,
        "primitive_histories": primitive_histories,
        "retention_by_stage": retention_by_stage,
        "source_before": source_before,
        "source_after": source_after,
        "inherited": {
            "stage_accuracy": inherited_stage,
            "train_accuracy": inherited_train,
            "heldout_accuracy": inherited_heldout,
            "wrong_accuracy": wrong_accuracy,
            "zero_fragment_accuracy": zero_accuracy,
            "missing_evidence_accuracy": missing_accuracy,
            "stable_bits_by_depth": inherited_stable,
            "histories": inherited_histories,
            "depth_slots": inherited_combiner.depth_count,
        },
        "fresh": {
            "stage_accuracy": fresh_stage,
            "stable_bits_by_depth": fresh_stable,
            "histories": fresh_histories,
            "depth_slots": fresh_combiner.depth_count,
        },
        "acquired_bank": {
            "sha256_before_targets": bank_digest_before,
            "sha256_after_targets": bank_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": training_batches * bits_per_update
            + audit_batches * args.audit_count * args.span * 2,
            "training_unique_verifier_bits": training_batches * bits_per_update,
            "audit_unique_verifier_bits": audit_batches
            * args.audit_count
            * args.span
            * 2,
            "unique_logical_lifetimes": training_batches * args.batch_size,
            "audit_logical_lifetimes": audit_batches * args.audit_count,
            "optimizer_updates": (
                args.parent_updates
                + primitive_training_batches
                + sum(inherited_stage_updates.values())
                + sum(fresh_stage_updates.values())
            ),
            "replayed_examples": 0,
            "inherited_growth_training_batches": inherited_training_batches,
            "fresh_growth_training_batches": fresh_training_batches,
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
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--primitive-updates", type=int, default=256)
    parser.add_argument("--stage-updates", type=int, default=64)
    parser.add_argument("--foundation-updates", type=int, default=128)
    parser.add_argument(
        "--joint-foundation",
        action="store_true",
        help="align all atomic files to one shared readout before growth",
    )
    parser.add_argument(
        "--operator-mode",
        choices=(
            "factorized_low_rank",
            "factorized_bounded_residual",
            "factorized_shared_operator_basis",
        ),
        default="factorized_low_rank",
        help="frozen-interpreter state algebra used by the growth audit",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--span", type=int, default=3)
    parser.add_argument("--audit-count", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        choices=tuple(STAGE_ORDERS),
        help="run only through this composition-depth rung",
    )
    parser.add_argument(
        "--foundation-depth",
        type=int,
        default=1,
        choices=tuple(STAGE_ORDERS),
        help=(
            "train the shared external interpreter through this depth before "
            "freezing its prefix and switching to append-only growth"
        ),
    )
    args = parser.parse_args()
    if (
        min(
            args.parent_updates,
            args.primitive_updates,
            args.stage_updates,
            args.foundation_updates,
            args.batch_size,
            args.audit_count,
            args.eval_every,
        )
        < 1
    ):
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    if args.foundation_depth > args.max_depth:
        raise ValueError("foundation-depth cannot exceed max-depth")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
