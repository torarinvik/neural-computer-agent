"""Extend learned relative magnitude across a gradual contour morph.

The experienced arm keeps the learned magnitude slot. The reset arm replaces
exactly that slot with its behavior-preserving initialization. Both receive
identical rendered events, opaque attempted actions, scalar outcomes, and
opaque behavioral replay. This measures whether earlier magnitude experience
reduces the evidence needed for the next appearance rung.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path

import torch
from torch import nn

from .environment import generate_lifetimes
from .model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_fourth_primitive_transfer import (
    _headline_accuracy, _load, _operation_cue_ablation_accuracy,
    _replay_loss_and_leakage)
from .train_pair_relation_appearance_bridge import (
    _pair_loss, _reset_slot, _slot_prefixes)


NON_MAGNITUDE_REPLAY_SPECS = (
    ("pair_relation", "bars", None),
    ("pair_relation", "diamonds", None),
    ("pair_relation", "dot_pairs", None),
    ("binary_mapping", "bars", None),
    ("visible_context", "bars", None),
    ("visible_context_xor", "bars", None),
)


def _replay_specs(
        inherited_magnitude_blends: tuple[float, ...],
        ) -> tuple[tuple[str, str, float | None], ...]:
    if not inherited_magnitude_blends:
        raise ValueError("at least one inherited magnitude blend is required")
    if len(set(inherited_magnitude_blends)) != len(
            inherited_magnitude_blends):
        raise ValueError("inherited magnitude blends must be unique")
    return (
        tuple(
            ("visible_pair_magnitude", "bars", blend)
            for blend in inherited_magnitude_blends)
        + NON_MAGNITUDE_REPLAY_SPECS)


def _target_evaluation(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        blend: float, device: torch.device,
        ) -> dict[str, object]:
    return evaluate(
        model, count=count, trials=6, seed=seed, device=device,
        task="visible_pair_magnitude", feedback_trials=1,
        appearance="bars", appearance_blend=blend)


def _shuffle_verifier_outcomes(
        batch, *, seed: int,
        ):
    """Break the pixel/outcome relation without changing either marginal."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        batch.batch_size, generator=generator).to(
            batch.correct_actions.device)
    return replace(
        batch,
        correct_actions=batch.correct_actions[permutation])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--seed", type=int, default=21501)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--epochs-per-batch", type=int, default=1,
        help=(
            "optimizer passes over each uniquely generated batch; increasing "
            "this spends compute without consuming additional verifier bits"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=int, default=4)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--locality-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--bridge-slot-width", type=int, default=64)
    parser.add_argument(
        "--plasticity-mode", choices=("append", "refine"),
        default="append",
        help=(
            "append freezes every mastered slot and adds a zero-output "
            "successor; refine expands the latest bridge in place without "
            "increasing parameter count"))
    parser.add_argument("--exploration", type=float, default=0.5)
    parser.add_argument(
        "--shuffle-new-verifier-outcomes", action="store_true",
        help=(
            "causal control: permute verifier outcomes across otherwise "
            "unchanged new-task lifetimes"))
    parser.add_argument("--blend-start", type=float, default=0.125)
    parser.add_argument("--blend-end", type=float, default=0.25)
    parser.add_argument(
        "--inherited-magnitude-blends", default="0.0",
        help=(
            "comma-separated mastered contour blends to rehearse and gate; "
            "the next rung should include bars and every trained predecessor"))
    parser.add_argument(
        "--initialization", choices=("experienced", "reset"),
        default="experienced")
    parser.add_argument("--gate-leak-initial", type=float, default=0.05)
    parser.add_argument("--test-lifetimes", type=int, default=2048)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(
            args.steps, args.epochs_per_batch, args.batch_size,
            args.replay_batch_size,
            args.test_lifetimes) < 1:
        raise ValueError("counts and steps must be positive")
    if args.batch_size % 2 or args.test_lifetimes % 2:
        raise ValueError("batch and test counts must be divisible by two")
    if not 0.0 <= args.blend_start <= args.blend_end <= 1.0:
        raise ValueError("blend curriculum must be ordered within [0, 1]")
    if args.retention_weight <= 0 or args.locality_weight < 0:
        raise ValueError("loss weights are out of range")
    if (
            args.learning_rate <= 0 or args.bridge_slot_width < 1
            or not 0.0 <= args.gate_leak_initial):
        raise ValueError("learning rate and gate leak are out of range")
    inherited_magnitude_blends = tuple(
        float(value)
        for value in args.inherited_magnitude_blends.split(",")
        if value)
    if (
            not inherited_magnitude_blends
            or 0.0 not in inherited_magnitude_blends
            or any(
                not 0.0 <= value <= args.blend_start
                for value in inherited_magnitude_blends)):
        raise ValueError(
            "inherited magnitude blends must include 0 and lie within "
            "[0, blend-start]")
    replay_specs = _replay_specs(inherited_magnitude_blends)

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload, teacher = _load(args.parent, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    configuration = dict(payload["model_configuration"])
    inherited_slots = tuple(configuration.get("skill_adapter_widths", ()))
    if len(inherited_slots) < 2:
        raise ValueError(
            "magnitude bridge requires relation and magnitude slots")
    inherited_magnitude_slot = len(inherited_slots) - 1
    if args.plasticity_mode == "append":
        configuration["skill_adapter_widths"] = (
            *inherited_slots, args.bridge_slot_width)
        slot = len(inherited_slots)
    else:
        slot = inherited_magnitude_slot
    prefixes = _slot_prefixes(slot)
    student = UnifiedCognitiveController(**configuration).to(device)
    if args.plasticity_mode == "append":
        missing, unexpected = student.load_state_dict(
            payload["state_dict"], strict=False)
        expected_missing = {
            name for name in student.state_dict()
            if name.startswith(prefixes)}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"new bridge slot mismatch: missing={missing}, "
                f"unexpected={unexpected}")
    else:
        student.load_state_dict(payload["state_dict"])
    if args.initialization == "reset":
        _reset_slot(
            student, configuration, slot=inherited_magnitude_slot)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    frozen_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if not name.startswith(prefixes)}
    slot_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if name.startswith(prefixes)}
    parameters = [
        parameter for parameter in student.parameters()
        if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("magnitude bridge has no plastic parameter")
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=1e-5)

    started = time.perf_counter()
    history = []
    optimizer_updates = args.steps * args.epochs_per_batch
    optimizer_update = 0
    for update in range(1, args.steps + 1):
        progress = (
            1.0 if args.steps == 1 else (update - 1) / (args.steps - 1))
        blend = (
            args.blend_start
            + progress * (args.blend_end - args.blend_start))
        new_batch = generate_lifetimes(
            args.batch_size, 6,
            seed=args.seed * 10_000_000 + update,
            task="visible_pair_magnitude", appearance="bars",
            appearance_blend=blend,
            position_holdout=bool(update % 2),
            support_trials=1, device=device)
        if args.shuffle_new_verifier_outcomes:
            new_batch = _shuffle_verifier_outcomes(
                new_batch, seed=args.seed * 30_000_000 + update)
        replay_batches = [
            generate_lifetimes(
                args.replay_batch_size, 6,
                seed=(
                    args.seed * (20_000_000 + 10_000_000 * index)
                    + update),
                task=task, appearance=appearance,
                appearance_blend=appearance_blend,
                support_trials=1, device=device)
            for index, (task, appearance, appearance_blend)
            in enumerate(replay_specs)
        ]
        for epoch in range(1, args.epochs_per_batch + 1):
            optimizer_update += 1
            student.train()
            student.skill_adapter_gate_leak = (
                args.gate_leak_initial
                * max(
                    0.0,
                    1.0
                    - (optimizer_update - 1)
                    / max(1, optimizer_updates - 1)))
            skill_loss, observed_accuracy = _pair_loss(
                student, new_batch, exploration=args.exploration)
            replay_results = [
                _replay_loss_and_leakage(
                    student, teacher, batch, slot=slot,
                    feedback_trials=1, shuffled_teacher=False)
                for batch in replay_batches
            ]
            replay_losses = [value for value, _, _ in replay_results]
            residual_norms = [value for _, value, _ in replay_results]
            retention_loss = torch.stack(replay_losses).mean()
            # Every inherited magnitude contour belongs to the concept being
            # extended and may remain active. Other streams are interference
            # surfaces on which the new successor should stay quiet.
            unrelated_residuals = [
                residual for residual, (task, _, _) in zip(
                    residual_norms, replay_specs, strict=True)
                if task != "visible_pair_magnitude"]
            locality = torch.stack(unrelated_residuals).mean()
            loss = (
                skill_loss
                + args.retention_weight * retention_loss
                + args.locality_weight * locality)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if optimizer_update in (1, optimizer_updates):
                history.append({
                    "batch": update,
                    "epoch": epoch,
                    "optimizer_update": optimizer_update,
                    "appearance_blend": blend,
                    "new_batch_accuracy": observed_accuracy,
                    "skill_loss": float(skill_loss.detach()),
                    "retention_loss": float(retention_loss.detach()),
                    "locality": float(locality.detach()),
                    "total_loss": float(loss.detach()),
                })

    student.skill_adapter_gate_leak = 0.0
    target = _target_evaluation(
        student, count=args.test_lifetimes,
        seed=args.seed + 90_000_000, blend=args.blend_end,
        device=device)
    magnitude_retention = {
        str(blend): _target_evaluation(
            student, count=args.test_lifetimes,
            seed=args.seed + 91_000_000 + 10_000 * index,
            blend=blend, device=device)
        for index, blend in enumerate(inherited_magnitude_blends)
    }
    bars = magnitude_retention[str(0.0)]
    diamond = _target_evaluation(
        student, count=args.test_lifetimes,
        seed=args.seed + 92_000_000, blend=1.0, device=device)
    relation = {
        appearance: evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 93_000_000 + 10_000 * index,
            device=device, task="pair_relation", feedback_trials=1,
            appearance=appearance)
        for index, appearance in enumerate(
            ("bars", "diamonds", "dot_pairs"))
    }
    unrelated = {
        task: evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 94_000_000 + 10_000 * index,
            device=device, task=task, feedback_trials=1,
            appearance=appearance)
        for index, (task, appearance, _) in enumerate(
            spec for spec in NON_MAGNITUDE_REPLAY_SPECS
            if spec[0] not in ("pair_relation",))
    }
    missing_second = _operation_cue_ablation_accuracy(
        student, count=args.test_lifetimes,
        seed=args.seed + 90_000_000, device=device,
        support_trials=1, new_task="visible_pair_magnitude",
        appearance="bars", appearance_blend=args.blend_end)
    student.skill_adapter_ablate_prior_read = True
    prior_ablated = _target_evaluation(
        student, count=args.test_lifetimes,
        seed=args.seed + 90_000_000, blend=args.blend_end,
        device=device)
    student.skill_adapter_ablate_prior_read = False
    target_accuracy = _headline_accuracy(target)
    prior_accuracy = _headline_accuracy(prior_ablated)
    prior_advantage = target_accuracy - prior_accuracy
    gates = {
        "target_mastered": target["gate"]["accepted"],
        "magnitude_repertoire_retained": all(
            value["gate"]["accepted"]
            for value in magnitude_retention.values()),
        "second_object_causally_required":
            missing_second <= target_accuracy - 0.15,
        "prior_relation_read_causally_used": prior_advantage >= 0.05,
        "relation_repertoire_retained": all(
            value["gate"]["accepted"] for value in relation.values()),
        "unrelated_repertoire_retained": all(
            value["gate"]["accepted"] for value in unrelated.values()),
    }
    accepted = all(gates.values())
    frozen_bit_identical = all(
        torch.equal(
            before, student.state_dict()[name].detach().cpu())
        for name, before in frozen_initial.items())
    slot_change = sum(
        float(
            (student.state_dict()[name].detach().cpu() - before)
            .square().sum())
        for name, before in slot_initial.items()) ** 0.5
    total_unique_lifetimes = args.steps * (
        args.batch_size + len(replay_specs) * args.replay_batch_size)
    optimizer_lifetime_exposures = (
        total_unique_lifetimes * args.epochs_per_batch)
    report = {
        "schema": "pair-magnitude-appearance-bridge-v1",
        "claim_boundary": (
            "The learner receives pixels, its own opaque attempted actions, "
            "scalar outcomes, and opaque frozen-controller rehearsal. It "
            "receives no semantic task/relation label, correct unattempted "
            "action, or hidden generator state."),
        "configuration": {
            **vars(args),
            "parent": str(args.parent),
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
        },
        "history": history,
        "accounting": {
            "new_unique_lifetimes": args.steps * args.batch_size,
            "new_verifier_bits": args.steps * args.batch_size * 6,
            "replay_streams": len(replay_specs),
            "replay_specs": replay_specs,
            "replay_unique_lifetimes": (
                args.steps * len(replay_specs) * args.replay_batch_size),
            "total_unique_lifetimes": total_unique_lifetimes,
            "total_unique_verifier_bits": total_unique_lifetimes * 6,
            "optimizer_lifetime_exposures":
                optimizer_lifetime_exposures,
            "optimizer_updates": optimizer_updates,
        },
        "evaluations": {
            "target_blend": target,
            "magnitude_repertoire_retention": magnitude_retention,
            "full_diamond_transfer": diamond,
            "prior_read_ablated_target": prior_ablated,
            "pair_relation_retention": relation,
            "unrelated_retention": unrelated,
        },
        "headline_accuracy": {
            "target_blend": target_accuracy,
            "bars_magnitude_retention": _headline_accuracy(bars),
            "full_diamond_transfer": _headline_accuracy(diamond),
            "prior_read_ablated_target": prior_accuracy,
            "prior_read_advantage": prior_advantage,
            "missing_second_object": missing_second,
        },
        "gates": gates,
        "all_gates_passed": accepted,
        "frozen_base_bit_identical": frozen_bit_identical,
        "slot_l2_change": slot_change,
        "total_seconds": time.perf_counter() - started,
    }
    if accepted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": student.state_dict(),
            "source_report": str(args.report),
            "admission_status": "pair_magnitude_appearance_bridge",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "initialization": args.initialization,
        "all_gates_passed": accepted,
        "target": target_accuracy,
        "bars": _headline_accuracy(bars),
        "full_diamond": _headline_accuracy(diamond),
        "prior_advantage": prior_advantage,
        "missing_second": missing_second,
        "seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
