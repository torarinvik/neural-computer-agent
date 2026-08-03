"""Learn and progressively refine discrete numerosity from pixels.

The deployed learner receives RGB frames, its own sampled opaque actions,
scalar outcomes, and frozen-controller behavioral rehearsal. The first
numerosity rung appends one zero-output successor that may read the immediately
preceding magnitude slot. Later rungs refine that same slot and rehearse the
last promoted numerosity frontier rather than growing one slot per increment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn

from .environment import generate_lifetimes
from .legacy_model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_fourth_primitive_transfer import (
    _headline_accuracy, _load, _operation_cue_ablation_accuracy,
    _replay_loss_and_leakage)
from .train_pair_magnitude_appearance_bridge import (
    _annealed_gate_leak, _replay_specs, _sensory_summary,
    _shuffle_verifier_outcomes)
from .train_pair_relation_appearance_bridge import (
    _pair_loss, _slot_prefixes)


def _target_evaluation(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        mass_control: float, appearance_blend: float,
        device: torch.device,
        ) -> dict[str, object]:
    return evaluate(
        model, count=count, trials=6, seed=seed, device=device,
        task="visible_pair_numerosity", feedback_trials=1,
        numerosity_mass_control=mass_control,
        numerosity_appearance_blend=appearance_blend)


def _retained_within_parent_floor(
        candidate: dict[str, dict[str, object]],
        parent: dict[str, dict[str, object]], *,
        tolerance: float = 0.02) -> bool:
    """Require every matched inherited stream to stay near its parent."""
    if candidate.keys() != parent.keys():
        raise ValueError("candidate and parent retention streams differ")
    return all(
        _headline_accuracy(candidate[name])
        >= _headline_accuracy(parent[name]) - tolerance
        for name in candidate)


def _build_student(
        payload: dict[str, object], *, device: torch.device,
        slot_width: int, continue_last_slot: bool,
        configuration_overrides: dict[str, object] | None = None,
        ) -> tuple[
            UnifiedCognitiveController, dict[str, object], int,
            tuple[str, ...]]:
    """Build either the first numerosity slot or a same-slot continuation."""
    configuration = dict(payload["model_configuration"])
    inherited_slots = tuple(configuration.get("skill_adapter_widths", ()))
    if not inherited_slots:
        raise ValueError("numerosity transfer requires inherited skill slots")
    if continue_last_slot:
        slot = len(inherited_slots) - 1
        if inherited_slots[slot] != slot_width:
            raise ValueError(
                "continuation slot width must match the existing final slot")
    else:
        slot = len(inherited_slots)
        configuration["skill_adapter_widths"] = (
            *inherited_slots, slot_width)
    if configuration_overrides:
        configuration.update(configuration_overrides)
    prefixes = _slot_prefixes(slot)
    student = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = student.load_state_dict(
        payload["state_dict"], strict=continue_last_slot)
    expected_missing = (
        set() if continue_last_slot else {
            name for name in student.state_dict()
            if name.startswith(prefixes)})
    if set(missing) != expected_missing or unexpected:
        mode = "continuation" if continue_last_slot else "new slot"
        raise RuntimeError(
            f"{mode} mismatch: missing={missing}, unexpected={unexpected}")
    return student, configuration, slot, prefixes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--candidate-checkpoint-out", type=Path)
    parser.add_argument("--seed", type=int, default=23401)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--epochs-per-batch", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=int, default=12)
    parser.add_argument("--retention-weight", type=float, default=8.0)
    parser.add_argument("--locality-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--slot-width", type=int, default=64)
    parser.add_argument("--exploration", type=float, default=0.5)
    parser.add_argument("--mass-control-start", type=float, default=0.0)
    parser.add_argument("--mass-control-end", type=float, default=0.0)
    parser.add_argument(
        "--numerosity-appearance-start", type=float, default=1.0)
    parser.add_argument(
        "--numerosity-appearance-end", type=float, default=1.0)
    parser.add_argument(
        "--inherited-magnitude-blends",
        default=(
            "0.0,0.15625,0.203125,0.20703125,"
            "0.208984375,0.21484375,0.2265625"))
    parser.add_argument(
        "--continue-last-slot", action="store_true",
        help=(
            "refine the parent's final numerosity slot instead of appending "
            "another slot"))
    parser.add_argument(
        "--inherited-numerosity-blends", default="",
        help=(
            "comma-separated promoted numerosity appearance frontiers to "
            "rehearse; required for same-slot continuation"))
    parser.add_argument(
        "--ablate-inherited-read", action="store_true",
        help=(
            "zero only the new slot's inherited magnitude input; capacity "
            "and all older slot behavior remain unchanged"))
    parser.add_argument(
        "--shuffle-new-verifier-outcomes", action="store_true")
    parser.add_argument("--gate-leak-initial", type=float, default=0.05)
    parser.add_argument("--gate-leak-anneal-updates", type=int, default=16)
    parser.add_argument("--test-lifetimes", type=int, default=16384)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(
            args.steps, args.epochs_per_batch, args.batch_size,
            args.replay_batch_size, args.test_lifetimes) < 1:
        raise ValueError("counts and steps must be positive")
    if args.batch_size % 2 or args.test_lifetimes % 2:
        raise ValueError("batch and test counts must be divisible by two")
    if not (
            0.0 <= args.mass_control_start
            <= args.mass_control_end <= 1.0):
        raise ValueError("mass-control curriculum is invalid")
    if not (
            0.0 <= args.numerosity_appearance_start
            <= args.numerosity_appearance_end <= 1.0):
        raise ValueError("numerosity appearance curriculum is invalid")
    if (
            args.retention_weight <= 0 or args.locality_weight < 0
            or args.learning_rate <= 0 or args.slot_width < 1
            or args.gate_leak_initial < 0
            or args.gate_leak_anneal_updates < 0):
        raise ValueError("optimization configuration is invalid")
    inherited_magnitude_blends = tuple(
        float(value)
        for value in args.inherited_magnitude_blends.split(",")
        if value)
    if (
            not inherited_magnitude_blends
            or 0.0 not in inherited_magnitude_blends
            or len(set(inherited_magnitude_blends))
            != len(inherited_magnitude_blends)
            or any(not 0.0 <= value <= 1.0
                   for value in inherited_magnitude_blends)):
        raise ValueError("inherited magnitude contours are invalid")
    inherited_numerosity_blends = tuple(
        float(value)
        for value in args.inherited_numerosity_blends.split(",")
        if value)
    if (
            len(set(inherited_numerosity_blends))
            != len(inherited_numerosity_blends)
            or any(not 0.0 <= value <= 1.0
                   for value in inherited_numerosity_blends)):
        raise ValueError("inherited numerosity frontiers are invalid")
    if args.continue_last_slot and not inherited_numerosity_blends:
        raise ValueError(
            "same-slot continuation requires inherited numerosity rehearsal")
    if not args.continue_last_slot and inherited_numerosity_blends:
        raise ValueError(
            "numerosity rehearsal is only valid for same-slot continuation")
    if any(
            value > args.numerosity_appearance_start
            for value in inherited_numerosity_blends):
        raise ValueError(
            "inherited numerosity frontier exceeds the acquisition start")
    replay_specs = _replay_specs(inherited_magnitude_blends)

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload, teacher = _load(args.parent, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student, configuration, slot, prefixes = _build_student(
        payload, device=device, slot_width=args.slot_width,
        continue_last_slot=args.continue_last_slot)
    if args.ablate_inherited_read:
        student.skill_adapter_ablate_prior_read_slot = slot
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    frozen_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if not name.startswith(prefixes)}
    parameters = [
        parameter for parameter in student.parameters()
        if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=1e-5)

    started = time.perf_counter()
    history = []
    optimizer_updates = args.steps * args.epochs_per_batch
    leak_updates = (
        args.gate_leak_anneal_updates or optimizer_updates)
    optimizer_update = 0
    for update in range(1, args.steps + 1):
        progress = (
            1.0 if args.steps == 1
            else (update - 1) / (args.steps - 1))
        control = (
            args.mass_control_start
            + progress
            * (args.mass_control_end - args.mass_control_start))
        numerosity_appearance = (
            args.numerosity_appearance_start
            + progress * (
                args.numerosity_appearance_end
                - args.numerosity_appearance_start))
        new_batch = generate_lifetimes(
            args.batch_size, 6,
            seed=args.seed * 10_000_000 + update,
            task="visible_pair_numerosity",
            numerosity_mass_control=control,
            numerosity_appearance_blend=numerosity_appearance,
            position_holdout=bool(update % 2),
            support_trials=1, device=device)
        sensory_summary = _sensory_summary(student, new_batch)
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
        replay_batches.extend(
            generate_lifetimes(
                args.replay_batch_size, 6,
                seed=(
                    args.seed * (160_000_000 + 10_000_000 * index)
                    + update),
                task="visible_pair_numerosity",
                numerosity_appearance_blend=blend,
                support_trials=1, device=device)
            for index, blend in enumerate(
                inherited_numerosity_blends))
        for epoch in range(1, args.epochs_per_batch + 1):
            optimizer_update += 1
            student.train()
            student.skill_adapter_gate_leak = _annealed_gate_leak(
                args.gate_leak_initial,
                optimizer_update=optimizer_update,
                anneal_updates=leak_updates)
            skill_loss, observed_accuracy, diagnostics = _pair_loss(
                student, new_batch, exploration=args.exploration,
                return_diagnostics=True)
            replay_results = [
                _replay_loss_and_leakage(
                    student, teacher, batch, slot=slot,
                    feedback_trials=1, shuffled_teacher=False)
                for batch in replay_batches
            ]
            retention_loss = torch.stack([
                value for value, _, _ in replay_results]).mean()
            locality = torch.stack([
                residual for _, residual, _ in replay_results]).mean()
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
                    "mass_control": control,
                    "numerosity_appearance_blend": numerosity_appearance,
                    "new_batch_accuracy": observed_accuracy,
                    "skill_loss": float(skill_loss.detach()),
                    "retention_loss": float(retention_loss.detach()),
                    "locality": float(locality.detach()),
                    "total_loss": float(loss.detach()),
                    "sensory_summary": sensory_summary,
                    "learner_diagnostics": diagnostics,
                })

    student.skill_adapter_gate_leak = 0.0
    target_seed = args.seed + 90_000_000
    target = _target_evaluation(
        student, count=args.test_lifetimes, seed=target_seed,
        mass_control=args.mass_control_end,
        appearance_blend=args.numerosity_appearance_end,
        device=device)
    magnitude_retention = {
        str(blend): evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 91_000_000 + 10_000 * index,
            device=device, task="visible_pair_magnitude",
            feedback_trials=1, appearance="bars",
            appearance_blend=blend)
        for index, blend in enumerate(inherited_magnitude_blends)
    }
    relation_retention = {
        appearance: evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 92_000_000 + 10_000 * index,
            device=device, task="pair_relation",
            feedback_trials=1, appearance=appearance)
        for index, appearance in enumerate(
            ("bars", "diamonds", "dot_pairs"))
    }
    unrelated_retention = {
        task: evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 93_000_000 + 10_000 * index,
            device=device, task=task, feedback_trials=1)
        for index, task in enumerate(
            ("binary_mapping", "visible_context", "visible_context_xor"))
    }
    numerosity_retention = {
        str(blend): _target_evaluation(
            student, count=args.test_lifetimes,
            seed=args.seed + 94_000_000 + 10_000 * index,
            mass_control=0.0, appearance_blend=blend, device=device)
        for index, blend in enumerate(inherited_numerosity_blends)
    }
    parent_magnitude = {
        str(blend): evaluate(
            teacher, count=args.test_lifetimes, trials=6,
            seed=args.seed + 91_000_000 + 10_000 * index,
            device=device, task="visible_pair_magnitude",
            feedback_trials=1, appearance="bars",
            appearance_blend=blend)
        for index, blend in enumerate(inherited_magnitude_blends)
    }
    parent_relation = {
        appearance: evaluate(
            teacher, count=args.test_lifetimes, trials=6,
            seed=args.seed + 92_000_000 + 10_000 * index,
            device=device, task="pair_relation",
            feedback_trials=1, appearance=appearance)
        for index, appearance in enumerate(
            ("bars", "diamonds", "dot_pairs"))
    }
    parent_unrelated = {
        task: evaluate(
            teacher, count=args.test_lifetimes, trials=6,
            seed=args.seed + 93_000_000 + 10_000 * index,
            device=device, task=task, feedback_trials=1)
        for index, task in enumerate(
            ("binary_mapping", "visible_context", "visible_context_xor"))
    }
    parent_numerosity = {
        str(blend): _target_evaluation(
            teacher, count=args.test_lifetimes,
            seed=args.seed + 94_000_000 + 10_000 * index,
            mass_control=0.0, appearance_blend=blend, device=device)
        for index, blend in enumerate(inherited_numerosity_blends)
    }

    missing_second = _operation_cue_ablation_accuracy(
        student, count=args.test_lifetimes, seed=target_seed,
        device=device, support_trials=1,
        new_task="visible_pair_numerosity",
        numerosity_mass_control=args.mass_control_end,
        numerosity_appearance_blend=args.numerosity_appearance_end)
    previous_ablation = student.skill_adapter_ablate_prior_read_slot
    student.skill_adapter_ablate_prior_read_slot = slot
    prior_ablated = _target_evaluation(
        student, count=args.test_lifetimes, seed=target_seed,
        mass_control=args.mass_control_end,
        appearance_blend=args.numerosity_appearance_end,
        device=device)
    student.skill_adapter_ablate_prior_read_slot = previous_ablation

    target_accuracy = _headline_accuracy(target)
    prior_accuracy = _headline_accuracy(prior_ablated)
    gates = {
        "target_mastered": target["gate"]["accepted"],
        "second_count_field_causally_required":
            missing_second <= target_accuracy - 0.15,
        "magnitude_repertoire_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                magnitude_retention, parent_magnitude),
        "relation_repertoire_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                relation_retention, parent_relation),
        "unrelated_repertoire_retained_within_2pp_of_parent":
            _retained_within_parent_floor(
                unrelated_retention, parent_unrelated),
    }
    if inherited_numerosity_blends:
        gates[
            "numerosity_frontier_retained_within_2pp_of_parent"
        ] = _retained_within_parent_floor(
            numerosity_retention, parent_numerosity)
    accepted = all(gates.values())
    frozen_bit_identical = all(
        torch.equal(
            before, student.state_dict()[name].detach().cpu())
        for name, before in frozen_initial.items())
    replay_streams = (
        len(replay_specs) + len(inherited_numerosity_blends))
    total_unique_lifetimes = args.steps * (
        args.batch_size + replay_streams * args.replay_batch_size)
    replay_spec_records = [
        {
            "task": task,
            "appearance": appearance,
            "appearance_blend": appearance_blend,
        }
        for task, appearance, appearance_blend in replay_specs
    ] + [
        {
            "task": "visible_pair_numerosity",
            "numerosity_appearance_blend": blend,
        }
        for blend in inherited_numerosity_blends
    ]
    report = {
        "schema": "pair-numerosity-transfer-v1",
        "claim_boundary": (
            "The learner receives RGB frames, its own opaque sampled actions, "
            "scalar attempted-action outcomes, its latent state, and opaque "
            "behavioral rehearsal. Count, order, layouts, semantic labels, "
            "and correct unattempted actions remain verifier-private."),
        "configuration": {
            **vars(args),
            "parent": str(args.parent),
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "candidate_checkpoint_out": (
                str(args.candidate_checkpoint_out)
                if args.candidate_checkpoint_out is not None else None),
            "resolved_gate_leak_anneal_updates": leak_updates,
        },
        "history": history,
        "accounting": {
            "new_unique_lifetimes": args.steps * args.batch_size,
            "new_verifier_bits": args.steps * args.batch_size * 6,
            "replay_streams": replay_streams,
            "replay_specs": replay_spec_records,
            "replay_unique_lifetimes":
                args.steps * replay_streams * args.replay_batch_size,
            "total_unique_lifetimes": total_unique_lifetimes,
            "total_unique_verifier_bits": total_unique_lifetimes * 6,
            "optimizer_updates": optimizer_updates,
            "optimizer_lifetime_exposures":
                total_unique_lifetimes * args.epochs_per_batch,
        },
        "evaluations": {
            "target": target,
            "prior_read_ablated_target": prior_ablated,
            "magnitude_repertoire_retention": magnitude_retention,
            "relation_repertoire_retention": relation_retention,
            "unrelated_repertoire_retention": unrelated_retention,
            "numerosity_frontier_retention": numerosity_retention,
            "frozen_parent_magnitude_baseline": parent_magnitude,
            "frozen_parent_relation_baseline": parent_relation,
            "frozen_parent_unrelated_baseline": parent_unrelated,
            "frozen_parent_numerosity_baseline": parent_numerosity,
        },
        "headline_accuracy": {
            "target": target_accuracy,
            "prior_read_ablated_target": prior_accuracy,
            "prior_read_advantage": target_accuracy - prior_accuracy,
            "missing_second_object": missing_second,
        },
        "gates": gates,
        "all_gates_passed": accepted,
        "frozen_base_bit_identical": frozen_bit_identical,
        "parameters": sum(
            parameter.numel() for parameter in student.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters),
        "total_seconds": time.perf_counter() - started,
    }
    checkpoint_payload = {
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": configuration,
        "state_dict": student.state_dict(),
        "source_report": str(args.report),
        "admission_status": (
            "pair_numerosity_transfer"
            if accepted else "unpromoted_pair_numerosity_prefix"),
    }
    if accepted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    if args.candidate_checkpoint_out is not None:
        args.candidate_checkpoint_out.parent.mkdir(
            parents=True, exist_ok=True)
        torch.save(checkpoint_payload, args.candidate_checkpoint_out)
        report["candidate_checkpoint_saved"] = True
    else:
        report["candidate_checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_passed": accepted,
        "ablate_inherited_read": args.ablate_inherited_read,
        "target": target_accuracy,
        "prior_read_ablated_target": prior_accuracy,
        "prior_read_advantage": target_accuracy - prior_accuracy,
        "missing_second": missing_second,
        "seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
