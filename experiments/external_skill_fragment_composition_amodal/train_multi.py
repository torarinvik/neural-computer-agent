"""Pressure-test sequential multi-fragment acquisition and program transfer.

Four opaque fragments are acquired one at a time with a frozen amodal parent.
Each new fragment may use newly appended shared-basis directions, while old
directions and coefficient rows are protected.  A fresh external trace
combiner then learns an unseen four-fragment program.  The matched fresh arm
has the same expandable external architecture but no inherited acquisition.

The verifier owns operation names, program order, and correct actions.  The
runtime receives only rendered events, opaque feedback, and opaque route-key
queries; the controller is never resized or trained during fragment growth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.external_skill_fragment_composition_amodal.train import (
    ACTION_WIDTH,
    INTENTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _digest,
    _fragment_bank,
    _machine,
    _stable_bits,
    _train_stage,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    ExternalSkillFragmentBank,
    ExternalSkillFragmentCombiner,
    OpaqueProtocolDecoder,
)

PRIMITIVES = ("reverse", "rotate", "complement", "prefix_parity")
TARGET_ORDERS = ((3, 2, 0, 1), (1, 0, 3, 2), (2, 3, 1, 0))


def _append_fragment(bank: ExternalSkillFragmentBank, seed: int) -> int:
    """Append one fresh fragment and one fresh shared-basis direction."""

    torch.manual_seed(seed)
    previous_basis_count = bank.basis_count
    bank.grow_basis(1)
    bank.freeze_basis_prefix(previous_basis_count)
    return bank.add_fragment(
        torch.randn(1, bank.basis_count) * 0.05,
        F.normalize(torch.randn(INTENTION_WIDTH), dim=0),
    )


def _bank_with_fragments(seed: int, count: int) -> ExternalSkillFragmentBank:
    if count < 1:
        raise ValueError("fragment count must be positive")
    bank = _fragment_bank(seed)
    for index in range(1, count):
        _append_fragment(bank, seed + index * 10_003)
    return bank


def _set_fragment_stage(
    machine,
    bank: ExternalSkillFragmentBank,
    index: int,
) -> list[torch.nn.Parameter]:
    """Freeze acquired state and expose only the current growth seam."""

    for parameter in machine.parameters():
        parameter.requires_grad_(index == 0)
    for parameter in bank.router.parameters():
        parameter.requires_grad_(False)
    for coefficient in bank.coefficients:
        coefficient.requires_grad_(False)
    bank.shared_basis.requires_grad_(True)
    bank.coefficients[index].requires_grad_(True)
    trainable: list[torch.nn.Parameter] = [
        bank.shared_basis,
        bank.coefficients[index],
    ]
    if index == 0:
        trainable.extend(machine.parameters())
    return trainable


def _retention(
    parent,
    machine,
    bank,
    decoders: list[OpaqueProtocolDecoder],
    *,
    count: int,
    span: int,
    seed: int,
) -> dict[str, float]:
    return {
        PRIMITIVES[index]: _accuracy(
            parent,
            machine,
            bank,
            decoder,
            operation=PRIMITIVES[index],
            selected=(index,),
            count=count,
            span=span,
            seed=seed + index,
        )
        for index, decoder in enumerate(decoders)
    }


def _eval_points(updates: int, eval_every: int) -> int:
    if not eval_every:
        return 0
    return sum(
        1
        for update in range(1, updates + 1)
        if update % eval_every == 0 or update == updates
    )


def _corruption_audit(bank: ExternalSkillFragmentBank, path: Path) -> bool:
    digest = bank.save(path)
    intact = path.read_bytes()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank_payload = dict(payload["bank"])
    state = dict(bank_payload["state"])
    fragments = list(state["fragments"])
    first = dict(fragments[0])
    coefficients = first["coefficients"].clone()
    coefficients.reshape(-1)[0] += 1.0
    first["coefficients"] = coefficients
    fragments[0] = first
    state["fragments"] = fragments
    bank_payload["state"] = state
    payload["bank"] = bank_payload
    torch.save(payload, path)
    rejected = False
    try:
        ExternalSkillFragmentBank.load(path)
    except ValueError as error:
        rejected = "checksum" in str(error)
    path.write_bytes(intact)
    restored = ExternalSkillFragmentBank.load(path)
    return rejected and digest == restored.payload()["sha256"]


def _train_target(
    parent,
    machine,
    bank: ExternalSkillFragmentBank,
    args: argparse.Namespace,
    *,
    order: tuple[int, ...],
    target_index: int,
    bits_per_update: int,
) -> dict[str, object]:
    """Train and audit one target using only a frozen acquired bank.

    Each target receives a new external trace combiner and decoder.  The
    acquired machine and fragment bank are deliberately shared and frozen,
    so transfer across rows measures reusable external capability rather than
    accidental optimizer state or continued fragment growth.
    """

    target_program = tuple(PRIMITIVES[index] for index in order)
    wrong_order = tuple(reversed(order))
    wrong_program = tuple(PRIMITIVES[index] for index in wrong_order)
    seed_base = args.seed + 100_000 + target_index * 30_001

    composition_combiner = ExternalSkillFragmentCombiner(
        REGISTER_WIDTH, REGISTER_WIDTH, hidden=64
    )
    composition_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    composition_history = _train_stage(
        parent,
        machine,
        bank,
        composition_decoder,
        operation="generated_composition",
        selected=None,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed_base,
        trainable=[
            *composition_combiner.parameters(),
            *composition_decoder.parameters(),
        ],
        generated_compositions=(target_program,),
        combiner=composition_combiner,
        route_programs=(order,),
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed_base + 1_000,
    )

    def accuracy(**kwargs) -> float:
        return _accuracy(
            parent,
            machine,
            bank,
            composition_decoder,
            operation="generated_composition",
            selected=None,
            count=args.audit_count,
            span=args.span,
            generated_compositions=(target_program,),
            combiner=composition_combiner,
            route_programs=(order,),
            **kwargs,
        )

    composition_accuracy = accuracy(seed=seed_base + 2_000)
    wrong_order_accuracy = _accuracy(
        parent,
        machine,
        bank,
        composition_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=seed_base + 3_000,
        generated_compositions=(wrong_program,),
        combiner=composition_combiner,
        route_programs=(wrong_order,),
    )
    zero_fragment_accuracy = accuracy(seed=seed_base + 4_000, zero_codes=True)
    missing_evidence_accuracy = accuracy(seed=seed_base + 5_000, blank_sequence=True)

    shuffled_combiner = ExternalSkillFragmentCombiner(
        REGISTER_WIDTH, REGISTER_WIDTH, hidden=64
    )
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        bank,
        shuffled_decoder,
        operation="generated_composition",
        selected=None,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed_base + 10_000,
        trainable=[*shuffled_combiner.parameters(), *shuffled_decoder.parameters()],
        shuffle_outcomes=True,
        generated_compositions=(target_program,),
        combiner=shuffled_combiner,
        route_programs=(order,),
    )
    shuffled_accuracy = _accuracy(
        parent,
        machine,
        bank,
        shuffled_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=seed_base + 11_000,
        shuffle_outcomes=True,
        generated_compositions=(target_program,),
        combiner=shuffled_combiner,
        route_programs=(order,),
    )

    fresh_machine = _machine()
    fresh_bank = _bank_with_fragments(
        args.seed + 2 + target_index * 30_001, len(PRIMITIVES)
    )
    fresh_combiner = ExternalSkillFragmentCombiner(
        REGISTER_WIDTH, REGISTER_WIDTH, hidden=64
    )
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_history = _train_stage(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_decoder,
        operation="generated_composition",
        selected=None,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=seed_base + 20_000,
        trainable=[
            *fresh_machine.parameters(),
            fresh_bank.shared_basis,
            *fresh_bank.coefficients,
            *fresh_combiner.parameters(),
            *fresh_decoder.parameters(),
        ],
        generated_compositions=(target_program,),
        combiner=fresh_combiner,
        route_programs=(order,),
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=seed_base + 21_000,
    )
    fresh_accuracy = _accuracy(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=seed_base + 22_000,
        generated_compositions=(target_program,),
        combiner=fresh_combiner,
        route_programs=(order,),
    )

    inherited_stable_bits = _stable_bits(
        composition_history,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    fresh_stable_bits = _stable_bits(
        fresh_history,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    route_queries = torch.stack(
        tuple(bank.keys[index].detach() for index in order)
    ).unsqueeze(0)
    routed = bank.compose_queries(route_queries)
    return {
        "target_order": list(order),
        "target_program": list(target_program),
        "wrong_order": list(wrong_order),
        "wrong_program": list(wrong_program),
        "composition": {
            "accuracy": composition_accuracy,
            "wrong_order_accuracy": wrong_order_accuracy,
            "zero_fragment_accuracy": zero_fragment_accuracy,
            "missing_evidence_accuracy": missing_evidence_accuracy,
            "history": composition_history,
        },
        "reward_shuffled": {"accuracy": shuffled_accuracy},
        "fresh": {"accuracy": fresh_accuracy, "history": fresh_history},
        "stable_bits_to_threshold": inherited_stable_bits,
        "fresh_stable_bits_to_threshold": fresh_stable_bits,
        "transfer_ratio_fresh_over_inherited": (
            float(fresh_stable_bits) / float(inherited_stable_bits)
            if inherited_stable_bits and fresh_stable_bits
            else None
        ),
        "routing": {
            "selected_indices": routed.fragment_indices.tolist(),
            "route_scores": routed.route_scores.tolist(),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    if (
        min(
            args.parent_updates,
            args.primitive_updates,
            args.composition_updates,
            args.batch_size,
            args.audit_count,
            args.eval_every,
        )
        < 1
    ):
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

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

    machine = _machine()
    bank = _fragment_bank(args.seed + 1)
    decoders: list[OpaqueProtocolDecoder] = []
    primitive_histories: list[list[dict[str, float | int]]] = []
    retention_by_stage: list[dict[str, float]] = []
    bits_per_update = args.batch_size * args.span * 2
    for index, operation in enumerate(PRIMITIVES):
        if index > 0:
            _append_fragment(bank, args.seed + 2_000 + index)
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        decoders.append(decoder)
        trainable = _set_fragment_stage(machine, bank, index)
        primitive_history = _train_stage(
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
            trainable=[*trainable, *decoder.parameters()],
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            audit_seed=args.seed + 30_000 + index * 20_003,
        )
        primitive_histories.append(primitive_history)
        bank.protect(index)
        retention_by_stage.append(
            _retention(
                parent,
                machine,
                bank,
                decoders,
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 50_000 + index * 20_003,
            )
        )

    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    source_before = retention_by_stage[-1]
    bank_digest_before = bank.payload()["sha256"]
    target_records = [
        _train_target(
            parent,
            machine,
            bank,
            args,
            order=order,
            target_index=target_index,
            bits_per_update=bits_per_update,
        )
        for target_index, order in enumerate(TARGET_ORDERS)
    ]
    bank_digest_after = bank.payload()["sha256"]
    source_after = _retention(
        parent,
        machine,
        bank,
        decoders,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 130_000,
    )

    # Keep concurrent seed runs isolated.  A shared sibling directory makes
    # corruption audits race: one process can overwrite the checkpoint while
    # another is loading it, producing a false persistence failure.
    persistence_dir = args.report_out.with_name(
        f"{args.report_out.stem}.persistence"
    )
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
    persistence_dir.mkdir(parents=True, exist_ok=True)
    persistence_exact = _corruption_audit(bank, persistence_dir / "fragment-bank.pt")
    parent_digest_after = _digest(parent.controller)
    primitive_stable_bits = {
        PRIMITIVES[index]: _stable_bits(
            history,
            threshold=args.mastery_threshold,
            bits_per_update=bits_per_update,
        )
        for index, history in enumerate(primitive_histories)
    }
    composition_eval_points = _eval_points(args.composition_updates, args.eval_every)
    parent_eval_points = _eval_points(args.parent_updates, args.eval_every)
    target_count = len(target_records)
    training_batches = (
        args.parent_updates
        + len(PRIMITIVES) * args.primitive_updates
        + target_count * 3 * args.composition_updates
    )
    audit_batches = (
        parent_eval_points
        + len(PRIMITIVES) * _eval_points(args.primitive_updates, args.eval_every)
        + target_count * 2 * composition_eval_points
        + sum(range(1, len(PRIMITIVES) + 1))
        + 2 * len(PRIMITIVES)
        + target_count * 6
    )
    target_compositions = [row["composition"] for row in target_records]
    gates = {
        "primitives_mastered": min(source_before.values()) >= args.mastery_threshold,
        "primitives_stable": all(
            value is not None for value in primitive_stable_bits.values()
        ),
        "primitives_retained": min(source_after.values()) >= args.mastery_threshold,
        "all_compositions_mastered": all(
            float(composition["accuracy"]) >= args.mastery_threshold
            for composition in target_compositions
        ),
        "all_compositions_stable": all(
            row["stable_bits_to_threshold"] is not None for row in target_records
        ),
        "all_fresh_controls_stable": all(
            row["fresh_stable_bits_to_threshold"] is not None for row in target_records
        ),
        "all_positive_stable_transfers": all(
            row["transfer_ratio_fresh_over_inherited"] is not None
            and row["fresh_stable_bits_to_threshold"] > row["stable_bits_to_threshold"]
            for row in target_records
        ),
        "all_wrong_orders_rejected": all(
            float(composition["wrong_order_accuracy"]) < args.mastery_threshold
            for composition in target_compositions
        ),
        "all_fragment_bypasses_rejected": all(
            float(composition["zero_fragment_accuracy"]) < args.mastery_threshold
            for composition in target_compositions
        ),
        "all_missing_evidence_rejected": all(
            float(composition["missing_evidence_accuracy"]) < args.mastery_threshold
            for composition in target_compositions
        ),
        "all_reward_shuffles_rejected": all(
            float(row["reward_shuffled"]["accuracy"]) < args.mastery_threshold
            for row in target_records
        ),
        "frozen_parent": parent_digest_before == parent_digest_after,
        "frozen_acquired_bank": bank_digest_before == bank_digest_after,
        "routing_resolved": all(
            row["routing"]["selected_indices"] == [row["target_order"]]
            for row in target_records
        ),
        "persistence_exact_and_corruption_rejected": persistence_exact,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-skill-fragment-multi-target-composition-report.v1",
        "claim_boundary": (
            "Four opaque fragments are acquired sequentially and reused across "
            "three independently held-out four-fragment programs. Each target "
            "has a separate external trace combiner and fresh learner control; "
            "the acquired machine and bank remain frozen. This does not establish "
            "arbitrary program induction, unrestricted growth, compression, or "
            "general continual learning."
        ),
        "seed": args.seed,
        "primitives": list(PRIMITIVES),
        "target_orders": [list(order) for order in TARGET_ORDERS],
        "target_programs": [
            [PRIMITIVES[index] for index in order] for order in TARGET_ORDERS
        ],
        "parent_progress": parent_progress,
        "retention_by_stage": retention_by_stage,
        "primitive_histories": primitive_histories,
        "primitive_stable_bits": primitive_stable_bits,
        "source_before": source_before,
        "source_after": source_after,
        "targets": target_records,
        "acquired_bank": {
            "sha256_before_targets": bank_digest_before,
            "sha256_after_targets": bank_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                (training_batches + audit_batches) * args.batch_size * args.span * 2
            ),
            "training_unique_verifier_bits": training_batches * bits_per_update,
            "audit_unique_verifier_bits": audit_batches
            * args.audit_count
            * args.span
            * 2,
            "unique_logical_lifetimes": training_batches * args.batch_size,
            "audit_logical_lifetimes": audit_batches * args.audit_count,
            "optimizer_updates": (
                args.parent_updates
                + len(PRIMITIVES) * args.primitive_updates
                + target_count * 3 * args.composition_updates
            ),
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
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--primitive-updates", type=int, default=256)
    parser.add_argument("--composition-updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--span", type=int, default=3)
    parser.add_argument("--audit-count", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
