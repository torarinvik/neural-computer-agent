"""Pressure-test a factorized external register on rendered events.

The parent controller is trained once and then frozen. External instruction
data and external decoders are acquired outside it. The controller receives no
operation labels or verifier targets; only the local scalar outcome is fed
back through the opaque action record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _feedback,
    _runtime,
)
from neural_computer import (
    ExternalCapabilityRegisterMachine,
    ExternalRegisterInstruction,
    OpaqueProtocolDecoder,
    PersistentOpaqueStateStore,
    paired_counterfactual_ranking_loss,
)

ACTION_WIDTH = 2
EVENT_WIDTH = 32
INTENTION_WIDTH = 16
REGISTER_WIDTH = 32
INSTRUCTION_WIDTH = 16


def _batch(operation: str, *, count: int, span: int, seed: int):
    return generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation=operation,
    )


def _new_machine() -> ExternalCapabilityRegisterMachine:
    return ExternalCapabilityRegisterMachine(
        EVENT_WIDTH,
        ACTION_WIDTH,
        INTENTION_WIDTH,
        REGISTER_WIDTH,
        INSTRUCTION_WIDTH,
        interpreter_hidden=64,
        operator_rank=8,
        instructions=(
            ExternalRegisterInstruction(INSTRUCTION_WIDTH),
            ExternalRegisterInstruction(INSTRUCTION_WIDTH),
        ),
    )


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rollout(
    parent,
    machine: ExternalCapabilityRegisterMachine,
    decoder: OpaqueProtocolDecoder,
    batch,
    instructions: tuple[ExternalRegisterInstruction, ...],
    *,
    train_decoder: bool,
    shuffle_outcomes: bool = False,
    credit_mode: str = "paired_counterfactual",
    evidence_present: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = batch.input_frames.device
    batch_size = batch.batch_size
    parent_state = parent.initial_state(batch_size, device=device)
    register_state = machine.initial_state(batch_size, device=device)
    zeros = torch.zeros(batch_size, device=device)
    previous_action = torch.zeros(batch_size, ACTION_WIDTH, device=device)
    previous_reward = zeros
    previous_propensity = torch.ones(batch_size, device=device)
    previous_has_feedback = zeros
    present = torch.full(
        (batch_size,),
        evidence_present,
        dtype=torch.bool,
        device=device,
    )
    encoder = parent.encoders["vision"]

    def tick(frame: torch.Tensor, feedback) -> torch.Tensor:
        nonlocal parent_state, register_state
        with torch.no_grad():
            event = encoder(frame)
            output, parent_state = parent.step_streams(
                {"vision": frame},
                parent_state,
                feedback,
            )
        register, register_state = machine.step_register(
            event=event,
            action=previous_action,
            outcome=previous_reward,
            intention=output.intention,
            state=register_state,
            present=present,
            instructions=instructions,
        )
        return decoder(register)

    quiet = _feedback(
        previous_action,
        previous_reward,
        previous_propensity,
        previous_has_feedback,
    )
    for frame in batch.input_frames.transpose(0, 1):
        tick(frame, quiet)
    for frame in batch.distractor_frames.transpose(0, 1):
        tick(frame, quiet)

    losses: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    for frame, correct in zip(
        batch.query_frames.transpose(0, 1),
        batch.correct_actions.transpose(0, 1),
        strict=True,
    ):
        feedback = _feedback(
            previous_action,
            previous_reward,
            previous_propensity,
            previous_has_feedback,
        )
        logits = tick(frame, feedback)
        probabilities = logits.softmax(dim=-1)
        action = (
            torch.multinomial(probabilities * 0.9 + 0.05, 1).squeeze(1)
            if train_decoder
            else logits.argmax(dim=-1)
        )
        reward = (action == correct).to(logits.dtype)
        delivered = reward.roll(1) if shuffle_outcomes else reward
        if credit_mode == "paired_counterfactual":
            attempted = torch.tensor(
                [[0, 1]], dtype=torch.long, device=device
            ).expand(batch_size, -1)
            utilities = (
                attempted == correct.unsqueeze(1)
            ).to(logits.dtype)
            if shuffle_outcomes:
                utilities = utilities.roll(1, dims=0)
            loss, _ = paired_counterfactual_ranking_loss(
                logits,
                attempted,
                utilities,
            )
        elif credit_mode == "attempted_bce":
            selected = logits.gather(1, action.unsqueeze(1)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(selected, delivered)
        else:
            raise ValueError(f"unknown credit mode: {credit_mode!r}")
        losses.append(loss)
        rewards.append(reward)
        previous_action = F.one_hot(action, ACTION_WIDTH).to(logits.dtype)
        previous_reward = delivered
        previous_propensity = probabilities.gather(
            1,
            action.unsqueeze(1),
        ).squeeze(1).detach().clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        previous_has_feedback = torch.ones_like(previous_reward)
    return torch.stack(losses).mean(), torch.stack(rewards, dim=1)


def _train_stage(
    parent,
    machine: ExternalCapabilityRegisterMachine,
    decoder: OpaqueProtocolDecoder,
    *,
    operation: str,
    instructions: tuple[ExternalRegisterInstruction, ...],
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    trainable: list[torch.nn.Parameter],
    credit_mode: str,
    shuffle_outcomes: bool = False,
    eval_every: int = 0,
    audit_count: int = 0,
    audit_seed: int = 0,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=1e-5)
    progress: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        batch = _batch(
            operation,
            count=batch_size,
            span=span,
            seed=seed + update * 10_007,
        )
        loss, _ = _rollout(
            parent,
            machine,
            decoder,
            batch,
            instructions,
            train_decoder=True,
            shuffle_outcomes=shuffle_outcomes,
            credit_mode=credit_mode,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if eval_every > 0 and (
            update % eval_every == 0 or update == updates
        ):
            progress.append(
                {
                    "update": update,
                    "heldout_accuracy": _accuracy(
                        parent,
                        machine,
                        decoder,
                        operation=operation,
                        instructions=instructions,
                        count=audit_count,
                        span=span,
                        seed=audit_seed + update,
                        credit_mode=credit_mode,
                    ),
                }
            )
    return progress


def _stable_bits(
    progress: list[dict[str, float | int]],
    *,
    threshold: float,
    bits_per_update: int,
) -> int | None:
    for index, row in enumerate(progress):
        if all(
            float(later["heldout_accuracy"]) >= threshold
            for later in progress[index:]
        ):
            return int(row["update"]) * bits_per_update
    return None


@torch.no_grad()
def _accuracy(
    parent,
    machine: ExternalCapabilityRegisterMachine,
    decoder: OpaqueProtocolDecoder,
    *,
    operation: str,
    instructions: tuple[ExternalRegisterInstruction, ...],
    count: int,
    span: int,
    seed: int,
    shuffle_outcomes: bool = False,
    credit_mode: str = "paired_counterfactual",
    evidence_present: bool = True,
) -> float:
    batch = _batch(operation, count=count, span=span, seed=seed)
    return float(
        _rollout(
            parent,
            machine,
            decoder,
            batch,
            instructions,
            train_decoder=False,
            shuffle_outcomes=shuffle_outcomes,
            credit_mode=credit_mode,
            evidence_present=evidence_present,
        )[1].mean()
    )


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

    machine = _new_machine()
    reverse_instruction, complement_instruction = tuple(machine.instructions)
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
    reverse_after = _accuracy(
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
    complement_accuracy = _accuracy(
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
    composition_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    composition_progress = _train_stage(
        parent,
        machine,
        composition_decoder,
        operation="complement_reverse",
        instructions=(reverse_instruction, complement_instruction),
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
    composition_accuracy = _accuracy(
        parent,
        machine,
        composition_decoder,
        operation="complement_reverse",
        instructions=(reverse_instruction, complement_instruction),
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
        operation="complement_reverse",
        instructions=(reverse_instruction, complement_instruction),
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
        operation="complement_reverse",
        instructions=(reverse_instruction, complement_instruction),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 94_000,
        credit_mode=args.credit_mode,
    )

    fresh_machine = _new_machine()
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_progress = _train_stage(
        parent,
        fresh_machine,
        fresh_decoder,
        operation="complement_reverse",
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
        operation="complement_reverse",
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
        operation="complement_reverse",
        instructions=(reverse_instruction, complement_instruction),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 98_000,
        credit_mode=args.credit_mode,
        evidence_present=False,
    )
    reloaded_machine = _new_machine()
    reloaded_machine.load_state_dict(machine.state_dict(), strict=True)
    reloaded_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reloaded_decoder.load_state_dict(composition_decoder.state_dict(), strict=True)
    reload_accuracy = _accuracy(
        parent,
        reloaded_machine,
        reloaded_decoder,
        operation="complement_reverse",
        instructions=tuple(reloaded_machine.instructions),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 92_000,
        credit_mode=args.credit_mode,
    )
    parent_digest_after = _module_digest(parent.controller)
    persistence_dir = args.report_out.parent / "persistence"
    machine_store = PersistentOpaqueStateStore(
        persistence_dir / "machine.pt",
        configuration=machine.configuration(),
    )
    machine_store_digest = machine_store.save_module(machine)
    intact_machine_payload = (persistence_dir / "machine.pt").read_bytes()
    corrupted_payload = torch.load(
        persistence_dir / "machine.pt",
        map_location="cpu",
        weights_only=False,
    )
    corrupted_state = dict(corrupted_payload["state_dict"])
    first_name = next(iter(corrupted_state))
    corrupted_value = corrupted_state[first_name].clone()
    corrupted_value.reshape(-1)[0] += 1.0
    corrupted_state[first_name] = corrupted_value
    corrupted_payload["state_dict"] = corrupted_state
    torch.save(corrupted_payload, persistence_dir / "machine.pt")
    corruption_rejected = False
    try:
        machine_store.load()
    except ValueError as error:
        corruption_rejected = "checksum mismatch" in str(error)
    (persistence_dir / "machine.pt").write_bytes(intact_machine_payload)
    composition_stable_bits = _stable_bits(
        composition_progress,
        threshold=args.mastery_threshold,
        bits_per_update=args.batch_size * args.span * 2,
    )
    fresh_stable_bits = _stable_bits(
        fresh_progress,
        threshold=args.mastery_threshold,
        bits_per_update=args.batch_size * args.span * 2,
    )
    promotion_gates = {
        "composition_stable": composition_stable_bits is not None,
        "fresh_stable": fresh_stable_bits is not None,
        "positive_stable_transfer": (
            composition_stable_bits is not None
            and fresh_stable_bits is not None
            and fresh_stable_bits > composition_stable_bits
        ),
        "retained_reverse": reverse_after >= args.mastery_threshold,
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
        "schema": "neural-computer.external-register-rendered-composition-report.v1",
        "claim_boundary": "Rendered-event pressure test of a factorized external register; not a promotion unless the primitive and composition gates pass the full control ladder.",
        "seed": args.seed,
        "parent": {
            "updates": args.parent_updates,
            "final_heldout_accuracy": float(parent_progress[-1]["heldout_accuracy"]),
        },
        "primitive_updates": args.primitive_updates,
        "composition_updates": args.composition_updates,
        "credit_mode": args.credit_mode,
        "batch_size": args.batch_size,
        "span": args.span,
        "audit_count": args.audit_count,
        "results": {
            "reverse_before_second_instruction": reverse_before,
            "reverse_after_second_instruction": reverse_after,
            "complement": complement_accuracy,
            "composition": composition_accuracy,
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
            "unique_verifier_bits": (
                args.primitive_updates * args.batch_size * args.span * 2
                + args.primitive_updates * args.batch_size * args.span * 2
                + args.composition_updates * args.batch_size * args.span * 2
                + args.composition_updates * args.batch_size * args.span * 2
                + args.composition_updates * args.batch_size * args.span * 2
            ),
            "unique_logical_lifetimes": (
                args.primitive_updates * args.batch_size * 2
                + args.composition_updates * args.batch_size * 3
            ),
            "optimizer_updates": args.parent_updates
            + args.primitive_updates * 2
            + args.composition_updates * 3,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
            "stable_bits_to_threshold": composition_stable_bits,
            "composition_stable_bits_to_threshold": composition_stable_bits,
            "fresh_stable_bits_to_threshold": fresh_stable_bits,
            "retention_on_mastered_primitives": reverse_after,
            "transfer_ratio_against_fresh_learner": transfer_ratio,
        },
        "promotion": {
            "accepted": promotion_accepted,
            "gates": promotion_gates,
            "reason": (
                "narrow_factorized_composition_transfer_promoted"
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
