"""Append one protected successor slot to a frozen sequence controller.

The parent checkpoint supplies the already learned computation and skill
slots.  A new zero-output slot is appended, then only that slot is trained on
the next sequence span while older spans are interleaved as verifier-driven
rehearsal.  The script is deliberately small: it is an admission probe for a
real cumulative skill artifact, not a claim of mastery or a production
trainer.  No semantic span labels or correct unattempted actions are exposed
to the model; span names are used only by the verifier-side curriculum and
audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .audit_sequence_multi_skill_bank import _model
from .train_pair_numerosity_transfer import _build_student
from .train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
    rollout_sequence_memory,
)


def _audit(model, *, spans: tuple[int, ...], count: int, distractors: int,
           seed: int, device: torch.device) -> dict[str, dict[str, float]]:
    return {
        str(span): evaluate_sequence_memory(
            model, count=count, span=span, distractors=distractors,
            seed=seed + index * 100_003, operation="mixed", device=device)
        for index, span in enumerate(spans)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93501)
    parser.add_argument("--span", type=int, required=True)
    parser.add_argument("--slot-width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-per-batch", type=int, default=1)
    parser.add_argument("--rehearse-spans", default="9,10")
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument(
        "--target-position-curriculum", default="",
        help="comma-separated target position blends to cycle gradually")
    parser.add_argument(
        "--target-distractor-curriculum", default="",
        help="comma-separated target distractor counts to cycle gradually")
    parser.add_argument(
        "--distill-old-weight", type=float, default=0.0,
        help=(
            "task-agnostic replay weight: match the frozen parent's logits "
            "on rehearsal spans while learning the new slot"))
    parser.add_argument(
        "--old-residual-penalty", type=float, default=0.0,
        help=(
            "penalize the appended slot's residual magnitude on inherited "
            "rehearsal streams; no task label is used"))
    parser.add_argument(
        "--critic-aux-weight", type=float, default=0.0,
        help=(
            "temporary action-conditioned success-prediction loss on the "
            "new slot hidden state; the critic is discarded after training"))
    parser.add_argument(
        "--read-prior-slot", action="store_true",
        help=(
            "give only the appended slot a generic read of the immediately "
            "preceding successor slot"))
    parser.add_argument("--position-augmentation", action="store_true")
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.span < 1 or args.slot_width < 1:
        raise ValueError("span and slot width must be positive")
    if min(args.steps, args.batch_size, args.epochs_per_batch,
           args.test_count) < 1:
        raise ValueError("training and test counts must be positive")
    if args.batch_size % 2 or args.test_count % 2:
        raise ValueError("batch and test counts must be even")
    rehearsal_spans = tuple(
        int(value) for value in args.rehearse_spans.split(",") if value)
    target_position_curriculum = tuple(
        float(value) for value in args.target_position_curriculum.split(",")
        if value)
    target_distractor_curriculum = tuple(
        int(value) for value in args.target_distractor_curriculum.split(",")
        if value)
    if any(not 0.0 <= value <= 1.0 for value in target_position_curriculum):
        raise ValueError("target position blends must be within [0, 1]")
    if any(value < 0 for value in target_distractor_curriculum):
        raise ValueError("target distractor counts must be nonnegative")
    if any(span < 1 for span in rehearsal_spans):
        raise ValueError("rehearsal spans must be positive")
    if args.span in rehearsal_spans:
        raise ValueError("target span must not be repeated as rehearsal")
    if args.distill_old_weight < 0.0:
        raise ValueError("distill-old-weight must be nonnegative")
    if args.old_residual_penalty < 0.0:
        raise ValueError("old-residual-penalty must be nonnegative")
    if args.critic_aux_weight < 0.0:
        raise ValueError("critic-aux-weight must be nonnegative")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    parent_payload = torch.load(
        args.parent, map_location=device, weights_only=False)
    parent_model = _model(parent_payload, device)
    student, configuration, slot, prefixes = _build_student(
        parent_payload, device=device, slot_width=args.slot_width,
        continue_last_slot=False,
        configuration_overrides=(
            {
                "skill_adapter_reads_prior": True,
                "skill_adapter_reads_prior_from": len(
                    parent_payload["model_configuration"].get(
                        "skill_adapter_widths", ())),
                "skill_adapter_prior_read_limit": 1,
            }
            if args.read_prior_slot else None))
    student.eval()
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    trainable = [
        parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("appended slot produced no trainable parameters")
    critic = None
    if args.critic_aux_weight:
        critic = nn.Linear(
            student.skill_adapters[slot][2].in_features, 2).to(device)
    optimizer_parameters = list(trainable)
    if critic is not None:
        optimizer_parameters += list(critic.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate, weight_decay=1e-5)

    # The constructor initializes the new slot's final projection at zero.
    # Check the central invariant before spending any training experience.
    initial_batch = generate_sequence_memory_batch(
        args.batch_size, span=args.span, distractors=args.distractors,
        seed=args.seed + 1_000, operation="mixed",
        position_augmentation=args.position_augmentation, device=device)
    with torch.no_grad():
        parent_probe = rollout_sequence_memory(
            parent_model, initial_batch, sample_actions=False,
            exploration=args.exploration)
        student_probe = rollout_sequence_memory(
            student, initial_batch, sample_actions=False,
            exploration=args.exploration)
    initial_logits_exact = torch.equal(
        parent_probe["logits"], student_probe["logits"])
    initial_workspace_exact = torch.equal(
        parent_probe["final_workspace"], student_probe["final_workspace"])
    if not (initial_logits_exact and initial_workspace_exact):
        raise AssertionError(
            "new successor slot changed behavior before training")

    inherited_spans = tuple(sorted(set(rehearsal_spans)))
    audit_spans = tuple((*inherited_spans, args.span))
    parent_audit = _audit(
        parent_model, spans=audit_spans, count=args.test_count,
        distractors=args.distractors, seed=args.seed + 2_000_000,
        device=device)
    schedule = (args.span, *rehearsal_spans)
    history: list[dict[str, float | int]] = []
    seen_bits = 0
    target_updates = 0
    slot_hidden: list[torch.Tensor] = []

    def capture_hidden(_module, _inputs, output) -> None:
        slot_hidden.append(output)

    hidden_handle = (
        student.skill_adapters[slot][1].register_forward_hook(capture_hidden)
        if critic is not None else None)
    for step in range(1, args.steps + 1):
        train_span = schedule[(step - 1) % len(schedule)]
        is_target = train_span == args.span
        if is_target:
            position_blend = (
                target_position_curriculum[
                    target_updates % len(target_position_curriculum)]
                if target_position_curriculum else 0.0)
            train_distractors = (
                target_distractor_curriculum[
                    target_updates % len(target_distractor_curriculum)]
                if target_distractor_curriculum else args.distractors)
            train_position_augmentation = (
                args.position_augmentation
                if not target_position_curriculum else False)
            target_updates += 1
        else:
            position_blend = 0.0
            train_distractors = args.distractors
            train_position_augmentation = args.position_augmentation
        batch = generate_sequence_memory_batch(
            args.batch_size, span=train_span, distractors=train_distractors,
            seed=args.seed + step * 10_007, operation="mixed",
            position_blend=position_blend,
            position_augmentation=train_position_augmentation, device=device)
        result = None
        for epoch in range(args.epochs_per_batch):
            # Rerendering each epoch keeps the tiny run from memorizing one
            # particular RGB realization of a verifier-generated lifetime.
            epoch_batch = batch if epoch == 0 else generate_sequence_memory_batch(
                args.batch_size, span=train_span,
                distractors=train_distractors,
                seed=args.seed + step * 10_007 + epoch * 1_000_003,
                operation="mixed",
                position_blend=position_blend,
                position_augmentation=train_position_augmentation,
                sequence_override=batch.sequence,
                operation_bits_override=batch.operation_bits, device=device)
            if critic is not None:
                slot_hidden.clear()
            result = rollout_sequence_memory(
                student, epoch_batch, sample_actions=True,
                exploration=args.exploration)
            loss = result["loss"]
            if critic is not None:
                if len(slot_hidden) < train_span:
                    raise RuntimeError(
                        "critic hook captured too few successor events")
                query_hidden = torch.stack(slot_hidden[-train_span:], dim=1)
                critic_logits = critic(query_hidden)
                attempted = result["actions"]
                outcomes = result["rewards"]
                selected = critic_logits.gather(
                    2, attempted.unsqueeze(-1)).squeeze(-1)
                loss = loss + args.critic_aux_weight * (
                    F.binary_cross_entropy_with_logits(selected, outcomes))
                slot_hidden.clear()
            if (args.distill_old_weight or args.old_residual_penalty) \
                    and rehearsal_spans:
                # Distillation is deliberately computed from fresh RGB
                # streams and the parent's emitted logits.  It supplies no
                # verifier labels; it only protects the behavior already
                # encoded in the frozen parent while the new slot learns.
                for distill_index, distill_span in enumerate(rehearsal_spans):
                    distill_batch = generate_sequence_memory_batch(
                        args.batch_size, span=distill_span,
                        distractors=args.distractors,
                        seed=(args.seed + step * 20_011
                              + epoch * 1_000_003 + distill_index * 7_919),
                        operation="mixed",
                        position_augmentation=args.position_augmentation,
                        device=device)
                    old_result = rollout_sequence_memory(
                        student, distill_batch, sample_actions=False,
                        exploration=args.exploration,
                        return_slot_activity=True)
                    if args.distill_old_weight:
                        with torch.no_grad():
                            teacher = rollout_sequence_memory(
                                parent_model, distill_batch,
                                sample_actions=False,
                                exploration=args.exploration)["logits"]
                        loss = loss + args.distill_old_weight * F.mse_loss(
                            old_result["logits"], teacher)
                    if args.old_residual_penalty:
                        activity = old_result.get(
                            "skill_adapter_residual_norms")
                        if activity is None:
                            raise RuntimeError(
                                "slot activity was not returned for penalty")
                        loss = loss + args.old_residual_penalty * (
                            activity[..., -1].square().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            result["total_loss"] = loss.detach()
        assert result is not None
        seen_bits += args.batch_size * train_span
        history.append({
            "update": step,
            "train_span": train_span,
            "train_distractors": train_distractors,
            "train_position_blend": position_blend,
            "verifier_bits": seen_bits,
            "training_accuracy": float(result["rewards"].mean()),
            "loss": float(result["total_loss"]),
            "behavior_loss": float(result["loss"].detach()),
        })

    student.eval()
    child_audit = _audit(
        student, spans=audit_spans, count=args.test_count,
        distractors=args.distractors, seed=args.seed + 2_000_000,
        device=device)
    # The newly appended slot should be causally necessary for the target, but
    # its zeroed version must leave inherited spans governed by the parent.
    zeroed = _model(
        {"model_configuration": configuration,
         "state_dict": student.state_dict()}, device)
    for name, parameter in zeroed.named_parameters():
        if name.startswith(prefixes):
            parameter.data.zero_()
    zeroed.eval()
    zeroed_audit = _audit(
        zeroed, spans=audit_spans, count=args.test_count,
        distractors=args.distractors, seed=args.seed + 2_000_000,
        device=device)
    retention_deltas = {
        str(span): child_audit[str(span)]["accuracy"]
        - parent_audit[str(span)]["accuracy"]
        for span in inherited_spans
    }
    target_gain = (
        child_audit[str(args.span)]["accuracy"]
        - parent_audit[str(args.span)]["accuracy"])
    report = {
        "schema": "sequence-skill-slot-extension-v1",
        "claim_boundary": (
            "The parent controller is frozen. Only a new generic successor "
            "slot is trained. Span identities are verifier-side curriculum "
            "metadata, and correct actions are never supplied to the model."),
        "parent": str(args.parent),
        "checkpoint_out": str(args.checkpoint_out),
        "seed": args.seed,
        "target_span": args.span,
        "inherited_rehearsal_spans": list(inherited_spans),
        "slot": slot,
        "slot_width": args.slot_width,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "epochs_per_batch": args.epochs_per_batch,
        "verifier_bits": seen_bits,
        "optimizer_lifetime_exposures": (
            seen_bits * args.epochs_per_batch),
        "position_augmentation": args.position_augmentation,
        "target_position_curriculum": target_position_curriculum,
        "target_distractor_curriculum": target_distractor_curriculum,
        "distill_old_weight": args.distill_old_weight,
        "old_residual_penalty": args.old_residual_penalty,
        "critic_aux_weight": args.critic_aux_weight,
        "critic_discarded": critic is not None,
        "read_prior_slot": args.read_prior_slot,
        "initial_logits_exact": initial_logits_exact,
        "initial_workspace_exact": initial_workspace_exact,
        "parent_audit": parent_audit,
        "child_audit": child_audit,
        "zeroed_slot_audit": zeroed_audit,
        "retention_deltas": retention_deltas,
        "target_gain": target_gain,
        "history": history,
        "gates": {
            "behavior_preserved_at_insertion": (
                initial_logits_exact and initial_workspace_exact),
            "inherited_retention_within_two_points": all(
                delta >= -0.02 for delta in retention_deltas.values()),
            "target_slot_causally_used": (
                zeroed_audit[str(args.span)]["accuracy"]
                < child_audit[str(args.span)]["accuracy"] - 0.05),
        },
    }
    report["accepted_smoke"] = all(report["gates"].values())
    if hidden_handle is not None:
        hidden_handle.remove()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": configuration,
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()},
        "source_parent": str(args.parent),
        "sequence_skill_slot_extension_report": report,
        "admission_status": (
            "candidate" if report["accepted_smoke"] else "rejected_smoke"),
    }, args.checkpoint_out)
    print(json.dumps({
        "accepted_smoke": report["accepted_smoke"],
        "target_parent": parent_audit[str(args.span)]["accuracy"],
        "target_child": child_audit[str(args.span)]["accuracy"],
        "target_zeroed": zeroed_audit[str(args.span)]["accuracy"],
        "retention_deltas": retention_deltas,
        "verifier_bits": seen_bits,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
