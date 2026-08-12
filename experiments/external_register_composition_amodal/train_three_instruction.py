"""Audit three-instruction external-register growth and order sensitivity.

This is the next rung after the promoted reverse→complement result. The
verifier-private generated grammar supplies a rendered reverse→complement→
rotate target (grammar entry 6); the controller receives only its learned
events and opaque feedback. The grammar identifier never enters the model.
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
from neural_computer import (
    OpaqueProtocolDecoder,
    PersistentOpaqueStateStore,
)

from .train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    _accuracy,
    _module_digest,
    _new_machine,
    _stable_bits,
    _train_stage,
)

TRIPLE_COMPOSITION_IDS = (6,)
REVERSED_TRIPLE_COMPOSITION_IDS = (7,)


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

    machine = _new_machine(3)
    reverse_instruction, complement_instruction, rotate_instruction = tuple(
        machine.instructions
    )
    reverse_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        reverse_decoder,
        operation="reverse",
        instructions=(reverse_instruction,),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 70_000,
        trainable=list(machine.parameters()) + list(reverse_decoder.parameters()),
        credit_mode=args.credit_mode,
    )
    reverse_before = _accuracy(
        parent,
        machine,
        reverse_decoder,
        operation="reverse",
        instructions=(reverse_instruction,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 71_000,
        credit_mode=args.credit_mode,
    )

    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    complement_instruction.code.requires_grad_(True)
    complement_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        complement_decoder,
        operation="complement",
        instructions=(complement_instruction,),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 80_000,
        trainable=[complement_instruction.code, *complement_decoder.parameters()],
        credit_mode=args.credit_mode,
    )
    reverse_after_second = _accuracy(
        parent,
        machine,
        reverse_decoder,
        operation="reverse",
        instructions=(reverse_instruction,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 81_000,
        credit_mode=args.credit_mode,
    )
    complement_before_third = _accuracy(
        parent,
        machine,
        complement_decoder,
        operation="complement",
        instructions=(complement_instruction,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 82_000,
        credit_mode=args.credit_mode,
    )

    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    rotate_instruction.code.requires_grad_(True)
    rotate_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        rotate_decoder,
        operation="rotate",
        instructions=(rotate_instruction,),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 83_000,
        trainable=[rotate_instruction.code, *rotate_decoder.parameters()],
        credit_mode=args.credit_mode,
    )
    reverse_after_third = _accuracy(
        parent,
        machine,
        reverse_decoder,
        operation="reverse",
        instructions=(reverse_instruction,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 84_000,
        credit_mode=args.credit_mode,
    )
    complement_after_third = _accuracy(
        parent,
        machine,
        complement_decoder,
        operation="complement",
        instructions=(complement_instruction,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 85_000,
        credit_mode=args.credit_mode,
    )
    rotate_accuracy = _accuracy(
        parent,
        machine,
        rotate_decoder,
        operation="rotate",
        instructions=(rotate_instruction,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 86_000,
        credit_mode=args.credit_mode,
    )

    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    composition_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    composition_progress = _train_stage(
        parent,
        machine,
        composition_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=(
            reverse_instruction,
            complement_instruction,
            rotate_instruction,
        ),
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 90_000,
        trainable=list(composition_decoder.parameters()),
        credit_mode=args.credit_mode,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 91_000,
    )
    composition_instructions = (
        reverse_instruction,
        complement_instruction,
        rotate_instruction,
    )
    composition_accuracy = _accuracy(
        parent,
        machine,
        composition_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=composition_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 92_000,
        credit_mode=args.credit_mode,
    )

    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    _train_stage(
        parent,
        machine,
        shuffled_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=composition_instructions,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 93_000,
        trainable=list(shuffled_decoder.parameters()),
        credit_mode=args.credit_mode,
        shuffle_outcomes=True,
    )
    shuffled_accuracy = _accuracy(
        parent,
        machine,
        shuffled_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=composition_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 94_000,
        credit_mode=args.credit_mode,
    )

    fresh_machine = _new_machine(3)
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_progress = _train_stage(
        parent,
        fresh_machine,
        fresh_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=tuple(fresh_machine.instructions),
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 95_000,
        trainable=list(fresh_machine.parameters()) + list(fresh_decoder.parameters()),
        credit_mode=args.credit_mode,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 97_000,
    )
    fresh_accuracy = _accuracy(
        parent,
        fresh_machine,
        fresh_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=tuple(fresh_machine.instructions),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 96_000,
        credit_mode=args.credit_mode,
    )
    missing_evidence_accuracy = _accuracy(
        parent,
        machine,
        composition_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=composition_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 98_000,
        credit_mode=args.credit_mode,
        evidence_present=False,
    )

    reversed_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reversed_instructions = (
        rotate_instruction,
        complement_instruction,
        reverse_instruction,
    )
    _train_stage(
        parent,
        machine,
        reversed_decoder,
        operation="generated_composition",
        generated_composition_ids=REVERSED_TRIPLE_COMPOSITION_IDS,
        instructions=reversed_instructions,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 99_000,
        trainable=list(reversed_decoder.parameters()),
        credit_mode=args.credit_mode,
    )
    reversed_order_accuracy = _accuracy(
        parent,
        machine,
        reversed_decoder,
        operation="generated_composition",
        generated_composition_ids=REVERSED_TRIPLE_COMPOSITION_IDS,
        instructions=reversed_instructions,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 100_000,
        credit_mode=args.credit_mode,
    )

    reloaded_machine = _new_machine(3)
    reloaded_machine.load_state_dict(machine.state_dict(), strict=True)
    reloaded_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reloaded_decoder.load_state_dict(composition_decoder.state_dict(), strict=True)
    reload_accuracy = _accuracy(
        parent,
        reloaded_machine,
        reloaded_decoder,
        operation="generated_composition",
        generated_composition_ids=TRIPLE_COMPOSITION_IDS,
        instructions=tuple(reloaded_machine.instructions),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 92_000,
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
    promotion_gates = {
        "composition_stable": composition_stable_bits is not None,
        "fresh_stable": fresh_stable_bits is not None,
        "positive_stable_transfer": (
            composition_stable_bits is not None
            and fresh_stable_bits is not None
            and fresh_stable_bits > composition_stable_bits
        ),
        "retained_reverse": reverse_after_third >= args.mastery_threshold,
        "retained_complement": complement_after_third >= args.mastery_threshold,
        "retained_rotate": rotate_accuracy >= args.mastery_threshold,
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
        "schema": "neural-computer.external-register-three-instruction-report.v1",
        "claim_boundary": "Three opaque external instructions compose a verifier-private rendered reverse-complement-rotate program through one frozen parent and factorized register; this is not general continual learning.",
        "seed": args.seed,
        "triple_composition_ids": list(TRIPLE_COMPOSITION_IDS),
        "reversed_triple_composition_ids": list(REVERSED_TRIPLE_COMPOSITION_IDS),
        "parent": {
            "updates": args.parent_updates,
            "final_heldout_accuracy": float(parent_progress[-1]["heldout_accuracy"]),
        },
        "primitive_updates": args.primitive_updates,
        "composition_updates": args.composition_updates,
        "credit_mode": args.credit_mode,
        "execution_mode": "read_execute",
        "batch_size": args.batch_size,
        "span": args.span,
        "audit_count": args.audit_count,
        "results": {
            "reverse_before_second_instruction": reverse_before,
            "reverse_after_second_instruction": reverse_after_second,
            "reverse_after_third_instruction": reverse_after_third,
            "complement_before_third_instruction": complement_before_third,
            "complement_after_third_instruction": complement_after_third,
            "rotate": rotate_accuracy,
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
            "unique_verifier_bits": args.primitive_updates
            * args.batch_size
            * args.span
            * 3
            + args.composition_updates * args.batch_size * args.span * 4,
            "unique_logical_lifetimes": args.primitive_updates
            * args.batch_size
            * 3
            + args.composition_updates * args.batch_size * 4,
            "optimizer_updates": args.parent_updates
            + args.primitive_updates * 3
            + args.composition_updates * 4,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
            "stable_bits_to_threshold": composition_stable_bits,
            "composition_stable_bits_to_threshold": composition_stable_bits,
            "fresh_stable_bits_to_threshold": fresh_stable_bits,
            "retention_on_mastered_primitives": min(
                reverse_after_third,
                complement_after_third,
                rotate_accuracy,
            ),
            "transfer_ratio_against_fresh_learner": transfer_ratio,
        },
        "promotion": {
            "accepted": promotion_accepted,
            "gates": promotion_gates,
            "reason": (
                "narrow_three_instruction_composition_promoted"
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.8)
    parser.add_argument(
        "--credit-mode",
        choices=("paired_counterfactual", "attempted_bce"),
        default="paired_counterfactual",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
