"""Extend one learned visual relation across a new contour family.

This is a matched transfer experiment.  The experienced arm retains the
parent's pair-relation slot; the reset arm replaces exactly that slot with its
behavior-preserving initialization.  Both arms then receive identical diamond
pixels, opaque attempted actions, scalar outcomes, and identical behavioral
rehearsal.  No semantic relation labels or correct unattempted actions enter
training.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from .environment import CognitiveLifetimeBatch, generate_lifetimes
from .model import UnifiedCognitiveController
from .train import (
    attempted_success_loss, evaluate, rollout, seed_everything)
from .train_fourth_primitive_transfer import (
    _headline_accuracy, _load, _replay_loss_and_leakage)


UNRELATED_REPLAY_SPECS = (
    ("binary_mapping", "bars"),
    ("visible_context", "bars"),
    ("visible_context_xor", "bars"),
)


def _replay_specs(
        new_appearance: str,
        inherited_relation_appearances: tuple[str, ...] | None = None,
        ) -> tuple[tuple[str, str], ...]:
    """Return every inherited relation form plus unrelated skills.

    The original bridge only inherited bars.  Once diamonds are promoted, a
    dot-pair bridge must rehearse both earlier renderings or it can replace the
    diamond form while appearing to preserve the abstract relation.
    """
    relation_appearances = (
        inherited_relation_appearances
        if inherited_relation_appearances is not None
        else (
            ("bars",)
            if new_appearance == "diamonds"
            else ("bars", "diamonds")))
    if not relation_appearances:
        raise ValueError("at least one inherited relation appearance is required")
    if len(set(relation_appearances)) != len(relation_appearances):
        raise ValueError("inherited relation appearances must be unique")
    return (
        tuple(("pair_relation", appearance)
              for appearance in relation_appearances)
        + UNRELATED_REPLAY_SPECS)


def _slot_prefixes(slot: int) -> tuple[str, ...]:
    return (
        f"skill_adapters.{slot}.",
        f"skill_adapter_gates.{slot}.",
        f"skill_adapter_gate_refiners.{slot}.",
        f"skill_adapter_gate_extensions.{slot}.",
        f"skill_adapter_read_projections.{slot}.")


def _reset_slot(
        model: UnifiedCognitiveController, configuration: dict[str, object],
        *, slot: int) -> None:
    """Replace exactly one slot with a fresh zero-output initialization."""
    fresh = UnifiedCognitiveController(**configuration)
    prefixes = _slot_prefixes(slot)
    fresh_state = fresh.state_dict()
    state = model.state_dict()
    for name in state:
        if name.startswith(prefixes):
            state[name] = fresh_state[name].to(
                device=state[name].device, dtype=state[name].dtype)
    model.load_state_dict(state)


def _pair_loss(
        model: UnifiedCognitiveController, batch, *,
        exploration: float) -> tuple[torch.Tensor, float]:
    """Attempted-action loss with every independently sampled event priced."""
    result = rollout(
        model, batch, sample_actions=True, exploration=exploration,
        feedback_trials=1)
    losses = [
        attempted_success_loss(
            result["logits"][:, trial],
            result["actions"][:, trial],
            result["rewards"][:, trial])
        for trial in range(batch.trials)
    ]
    return (
        torch.stack(losses).mean(),
        float(result["rewards"].float().mean()))


def _concatenate(
        batches: list[CognitiveLifetimeBatch]) -> CognitiveLifetimeBatch:
    if not batches:
        raise ValueError("at least one batch is required")
    contexts = [batch.context_ids for batch in batches]
    if any(value is None for value in contexts):
        context_ids = None
    else:
        context_ids = torch.cat([
            value for value in contexts if value is not None])
    return CognitiveLifetimeBatch(
        frames=torch.cat([batch.frames for batch in batches]),
        correct_actions=torch.cat([
            batch.correct_actions for batch in batches]),
        stimulus_identities=torch.cat([
            batch.stimulus_identities for batch in batches]),
        rule_bits=torch.cat([batch.rule_bits for batch in batches]),
        seeds=torch.cat([batch.seeds for batch in batches]),
        context_ids=context_ids)


def _unrelated_locality(
        residual_norms: list[torch.Tensor],
        replay_specs: tuple[tuple[str, str], ...]) -> torch.Tensor:
    """Price disturbance only outside the relation being extended.

    Relation streams are earlier appearances where the slot must remain active.
    Pricing those residuals asks the skill to erase itself.  Only unrelated
    inherited skills have silence as the correct non-interference target.
    """
    if len(residual_norms) != len(replay_specs):
        raise ValueError("residuals and replay specifications must align")
    unrelated = [
        residual for residual, (task, _) in zip(
            residual_norms, replay_specs, strict=True)
        if task != "pair_relation"]
    if not unrelated:
        raise ValueError("bridge requires relation plus unrelated replay")
    return torch.stack(unrelated).mean()


def _prioritized_replay_loss(
        replay_losses: list[torch.Tensor], *,
        temperature: float,
        base_weights: tuple[float, ...] | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Concentrate rehearsal on the behavior currently drifting most.

    Detached weights prevent the optimizer from changing the prioritizer
    rather than repairing behavior.  Temperature zero preserves the historical
    uniform mean exactly.
    """
    if not replay_losses:
        raise ValueError("at least one replay loss is required")
    losses = torch.stack(replay_losses)
    if temperature < 0:
        raise ValueError("replay priority temperature must not be negative")
    if base_weights is None:
        prior = torch.ones_like(losses)
    else:
        if (
                len(base_weights) != len(replay_losses)
                or any(value <= 0 for value in base_weights)):
            raise ValueError(
                "replay base weights must align and be positive")
        prior = losses.new_tensor(base_weights)
    if temperature == 0:
        weights = prior / prior.sum()
    else:
        weights = torch.softmax(
            losses.detach() / temperature + prior.log(), dim=0)
    return (weights * losses).sum(), weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument(
        "--candidate-checkpoint-out", type=Path,
        help=(
            "save an explicitly unpromoted training-only checkpoint for an "
            "external held-out population selector"))
    parser.add_argument(
        "--skip-final-evaluation", action="store_true",
        help=(
            "skip the expensive causal suite; requires "
            "--candidate-checkpoint-out and never promotes the candidate"))
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--retention-weight", type=float, default=2.0)
    parser.add_argument("--locality-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--consolidation-steps", type=int, default=0)
    parser.add_argument(
        "--consolidation-retention-weight", type=float, default=1.0)
    parser.add_argument(
        "--consolidation-locality-weight", type=float, default=0.1)
    parser.add_argument(
        "--consolidation-replay-temperature", type=float, default=0.0,
        help=(
            "positive values prioritize replay streams with the largest "
            "current behavioral divergence; zero keeps the uniform mean"))
    parser.add_argument(
        "--consolidation-replay-weights",
        help=(
            "positive comma-separated base weights aligned with the replay "
            "specification; normalized to keep total loss scale fixed"))
    parser.add_argument(
        "--gate-refiner-width", type=int, default=0,
        help=("zero-output nonlinear correction to the existing slot gate; "
              "zero preserves the parent architecture"))
    parser.add_argument(
        "--acquisition-refiner-only", action="store_true",
        help=(
            "freeze relation content and learn only where the existing "
            "nonlinear gate refiner should open"))
    parser.add_argument(
        "--gate-extension-width", type=int, default=0,
        help=(
            "insert a zero-output additive gate branch on the selected slot"))
    parser.add_argument(
        "--acquisition-gate-extension-only", action="store_true",
        help=(
            "freeze relation content and established gates; train only the "
            "new additive gate extension"))
    parser.add_argument(
        "--gate-leak-initial", type=float, default=0.0,
        help=("temporary negative-side slope for a shut rectified skill gate; "
              "anneals back to exact zero before evaluation"))
    parser.add_argument(
        "--gate-leak-anneal-fraction", type=float, default=0.5)
    parser.add_argument("--exploration", type=float, default=0.1)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument(
        "--new-appearance",
        choices=("diamonds", "dot_pairs"), default="diamonds")
    parser.add_argument(
        "--inherited-relation-appearances", nargs="+",
        choices=("bars", "diamonds"), default=None,
        help=(
            "relation appearances already mastered by the parent; defaults "
            "to bars for a diamond bridge and bars+diamonds for dot pairs"))
    parser.add_argument(
        "--blend-start", type=float, default=1.0,
        help=("bars-to-diamonds pixel blend at the first update; 1 keeps the "
              "direct diamond jump"))
    parser.add_argument(
        "--blend-end", type=float, default=1.0,
        help="bars-to-diamonds pixel blend at the final update")
    parser.add_argument(
        "--curriculum-mode", choices=("blend", "mixture"),
        default="blend",
        help=("blend morphs every contour; mixture gradually replaces whole "
              "bar lifetimes with diamond lifetimes"))
    parser.add_argument(
        "--initialization", choices=("experienced", "reset"),
        default="experienced")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(
            args.steps, args.batch_size, args.replay_batch_size,
            args.test_lifetimes) < 1:
        raise ValueError("counts and steps must be positive")
    if args.batch_size % 2 or args.test_lifetimes % 2:
        raise ValueError("batch and test counts must be divisible by two")
    if args.retention_weight <= 0 or args.locality_weight < 0:
        raise ValueError("loss weights are out of range")
    if args.consolidation_steps < 0:
        raise ValueError("consolidation steps must not be negative")
    if (
            args.consolidation_retention_weight <= 0
            or args.consolidation_locality_weight < 0
            or args.consolidation_replay_temperature < 0):
        raise ValueError("consolidation loss weights are out of range")
    if args.gate_refiner_width < 0:
        raise ValueError("gate refiner width must not be negative")
    if args.gate_extension_width < 0:
        raise ValueError("gate extension width must not be negative")
    if (
            args.acquisition_refiner_only
            and args.acquisition_gate_extension_only):
        raise ValueError(
            "acquisition cannot target two gate branches at once")
    if not 0.0 <= args.blend_start <= args.blend_end <= 1.0:
        raise ValueError(
            "blend curriculum must satisfy 0 <= start <= end <= 1")
    if (
            args.new_appearance != "diamonds"
            and (
                args.curriculum_mode == "mixture"
                or args.blend_start != 1.0
                or args.blend_end != 1.0)):
        raise ValueError(
            "appearance curricula currently target diamonds")
    if args.gate_leak_initial < 0.0:
        raise ValueError("gate leak must not be negative")
    if not 0.0 < args.gate_leak_anneal_fraction <= 1.0:
        raise ValueError("gate leak anneal fraction must be within (0, 1]")
    if args.skip_final_evaluation and args.candidate_checkpoint_out is None:
        raise ValueError(
            "skipping final evaluation requires a candidate checkpoint path")

    seed_everything(args.seed)
    device = torch.device(args.device)
    inherited_relation_appearances = (
        tuple(args.inherited_relation_appearances)
        if args.inherited_relation_appearances is not None else None)
    replay_specs = _replay_specs(
        args.new_appearance, inherited_relation_appearances)
    inherited_relation_appearances = tuple(
        appearance for task, appearance in replay_specs
        if task == "pair_relation")
    relation_replay_count = sum(
        task == "pair_relation" for task, _ in replay_specs)
    consolidation_replay_weights = (
        tuple(
            float(value)
            for value in args.consolidation_replay_weights.split(","))
        if args.consolidation_replay_weights is not None else None)
    if (
            consolidation_replay_weights is not None
            and (
                len(consolidation_replay_weights) != len(replay_specs)
                or any(
                    value <= 0 for value
                    in consolidation_replay_weights))):
        raise ValueError(
            "consolidation replay weights must align and be positive")
    payload, teacher = _load(args.parent, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    configuration = dict(payload["model_configuration"])
    slots = tuple(configuration.get("skill_adapter_widths", ()))
    if not slots:
        raise ValueError("appearance bridge requires a learned skill slot")
    slot = len(slots) - 1
    prefixes = _slot_prefixes(slot)
    existing_refiners = tuple(
        configuration.get("skill_adapter_gate_refiner_widths", ()))
    if len(existing_refiners) < len(slots):
        existing_refiners = existing_refiners + (
            0,) * (len(slots) - len(existing_refiners))
    parent_refiner_width = existing_refiners[slot]
    inserting_refiner = (
        args.gate_refiner_width > 0 and parent_refiner_width == 0)
    if (
            args.gate_refiner_width > 0
            and parent_refiner_width not in (0, args.gate_refiner_width)):
        raise ValueError(
            "requested gate refiner width conflicts with the parent")
    if inserting_refiner:
        existing_refiners = (
            existing_refiners[:slot]
            + (args.gate_refiner_width,)
            + existing_refiners[slot + 1:])
        configuration["skill_adapter_gate_refiner_widths"] = (
            existing_refiners)
    refiner_width = existing_refiners[slot]
    if args.acquisition_refiner_only and not refiner_width:
        raise ValueError(
            "refiner-only acquisition requires an existing gate refiner")
    existing_extensions = tuple(
        configuration.get("skill_adapter_gate_extension_widths", ()))
    if len(existing_extensions) < len(slots):
        existing_extensions = existing_extensions + (
            0,) * (len(slots) - len(existing_extensions))
    parent_extension_width = existing_extensions[slot]
    inserting_extension = (
        args.gate_extension_width > 0 and parent_extension_width == 0)
    if (
            args.gate_extension_width > 0
            and parent_extension_width not in (
                0, args.gate_extension_width)):
        raise ValueError(
            "requested gate extension width conflicts with the parent")
    if inserting_extension:
        existing_extensions = (
            existing_extensions[:slot]
            + (args.gate_extension_width,)
            + existing_extensions[slot + 1:])
        configuration["skill_adapter_gate_extension_widths"] = (
            existing_extensions)
    extension_width = existing_extensions[slot]
    if args.acquisition_gate_extension_only and not extension_width:
        raise ValueError(
            "extension-only acquisition requires a gate extension")

    student = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = student.load_state_dict(
        payload["state_dict"], strict=False)
    expected_missing = {
        name for name in student.state_dict()
        if (
            (
                inserting_refiner
                and name.startswith(
                    f"skill_adapter_gate_refiners.{slot}."))
            or (
                inserting_extension
                and name.startswith(
                    f"skill_adapter_gate_extensions.{slot}.")))}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"unexpected refiner insertion mismatch: "
            f"missing={missing}, unexpected={unexpected}")
    if args.initialization == "reset":
        _reset_slot(student, configuration, slot=slot)
    acquisition_prefixes = (
        (f"skill_adapter_gate_refiners.{slot}.",)
        if args.acquisition_refiner_only
        else (
            (f"skill_adapter_gate_extensions.{slot}.",)
            if args.acquisition_gate_extension_only else prefixes))
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(acquisition_prefixes))
    frozen_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if not name.startswith(acquisition_prefixes)}
    slot_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if name.startswith(prefixes)}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-5)

    started = time.perf_counter()
    history = []
    leak_updates = max(
        1, int(round(args.steps * args.gate_leak_anneal_fraction)))
    for update in range(1, args.steps + 1):
        student.train()
        student.skill_adapter_gate_leak = (
            args.gate_leak_initial
            * max(0.0, 1.0 - (update - 1) / leak_updates))
        blend_fraction = (
            1.0 if args.steps == 1 else (update - 1) / (args.steps - 1))
        appearance_blend = (
            args.blend_start
            + blend_fraction * (args.blend_end - args.blend_start))
        if args.curriculum_mode == "blend":
            new_batch = generate_lifetimes(
                args.batch_size, 6,
                seed=args.seed * 10_000_000 + update,
                task="pair_relation",
                appearance=args.new_appearance,
                appearance_blend=(
                    appearance_blend
                    if args.new_appearance == "diamonds" else None),
                support_trials=1, device=device)
        else:
            # Replace whole sensory episodes rather than interpolate pixels.
            # Counts stay even so each constituent generator remains balanced.
            diamond_count = int(round(
                args.batch_size * appearance_blend / 2.0)) * 2
            diamond_count = min(args.batch_size, max(0, diamond_count))
            bar_count = args.batch_size - diamond_count
            parts = []
            if bar_count:
                parts.append(generate_lifetimes(
                    bar_count, 6,
                    seed=args.seed * 10_000_000 + update,
                    task="pair_relation", appearance="bars",
                    support_trials=1, device=device))
            if diamond_count:
                parts.append(generate_lifetimes(
                    diamond_count, 6,
                    seed=args.seed * 10_000_000 + 1_000_000 + update,
                    task="pair_relation", appearance="diamonds",
                    support_trials=1, device=device))
            new_batch = _concatenate(parts)
        replay_batches = [
            generate_lifetimes(
                args.replay_batch_size, 6,
                seed=(
                    args.seed * (20_000_000 + 10_000_000 * index)
                    + update),
                task=task, appearance=appearance,
                support_trials=1, device=device)
            for index, (task, appearance) in enumerate(replay_specs)
        ]
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
        retention_loss, replay_weights = _prioritized_replay_loss(
            replay_losses, temperature=0.0)
        locality = _unrelated_locality(residual_norms, replay_specs)
        loss = (
            skill_loss
            + args.retention_weight * retention_loss
            + args.locality_weight * locality)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if update in (1, args.steps):
            history.append({
                "phase": "acquisition",
                "update": update,
                "new_batch_accuracy": observed_accuracy,
                "appearance_blend": appearance_blend,
                "gate_leak": student.skill_adapter_gate_leak,
                "diamond_lifetime_fraction": (
                    float(diamond_count / args.batch_size)
                    if args.curriculum_mode == "mixture"
                    else appearance_blend),
                "skill_loss": float(skill_loss.detach()),
                "retention_loss": float(retention_loss.detach()),
                "replay_weights": [
                    float(value) for value in replay_weights.detach()],
                "unrelated_event_residual_norm": float(locality.detach()),
                "relation_residual_norms": [
                    float(value.detach())
                    for value in residual_norms[:relation_replay_count]],
                "total_loss": float(loss.detach()),
            })

    if args.consolidation_steps:
        # Freeze the broadened relation and learn only where it should speak.
        # Joint optimization otherwise changes content faster than the gate can
        # learn the relation-family boundary.
        if not (
                extension_width
                if args.acquisition_gate_extension_only
                else refiner_width):
            raise ValueError(
                "staged consolidation requires a nonlinear gate branch")
        # The old linear gate offers a cheap but invalid solution: close on
        # every event. Freeze it and train only the nonlinear correction, so
        # localization has to depend on event structure.
        gate_prefixes = (
            (f"skill_adapter_gate_extensions.{slot}.",)
            if args.acquisition_gate_extension_only
            else (f"skill_adapter_gate_refiners.{slot}.",))
        acquisition_teacher = copy.deepcopy(student).eval()
        for parameter in acquisition_teacher.parameters():
            parameter.requires_grad_(False)
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(name.startswith(gate_prefixes))
        gate_parameters = [
            parameter for parameter in student.parameters()
            if parameter.requires_grad]
        if not gate_parameters:
            raise RuntimeError("consolidation has no trainable gate parameters")
        consolidation_optimizer = torch.optim.AdamW(
            gate_parameters, lr=args.learning_rate, weight_decay=1e-5)
        for consolidation_update in range(1, args.consolidation_steps + 1):
            student.train()
            student.skill_adapter_gate_leak = 0.0
            update_seed = 1_000_000 + consolidation_update
            new_batch = generate_lifetimes(
                args.batch_size, 6,
                seed=args.seed * 10_000_000 + update_seed,
                task="pair_relation", appearance=args.new_appearance,
                support_trials=1, device=device)
            replay_batches = [
                generate_lifetimes(
                    args.replay_batch_size, 6,
                    seed=(
                        args.seed
                        * (20_000_000 + 10_000_000 * index)
                        + update_seed),
                    task=task, appearance=appearance,
                    support_trials=1, device=device)
                for index, (task, appearance) in enumerate(replay_specs)
            ]
            # Consolidate the controller's own verified successful behavior.
            # This is opaque self-distillation, not a semantic relation label
            # or correct-action target.
            skill_loss, _, observed_accuracy = _replay_loss_and_leakage(
                student, acquisition_teacher, new_batch, slot=slot,
                feedback_trials=1, shuffled_teacher=False)
            replay_results = [
                _replay_loss_and_leakage(
                    student, teacher, batch, slot=slot,
                    feedback_trials=1, shuffled_teacher=False)
                for batch in replay_batches
            ]
            replay_losses = [value for value, _, _ in replay_results]
            residual_norms = [value for _, value, _ in replay_results]
            retention_loss, replay_weights = _prioritized_replay_loss(
                replay_losses,
                temperature=args.consolidation_replay_temperature,
                base_weights=consolidation_replay_weights)
            locality = _unrelated_locality(residual_norms, replay_specs)
            loss = (
                skill_loss
                + args.consolidation_retention_weight * retention_loss
                + args.consolidation_locality_weight * locality)
            consolidation_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(gate_parameters, 1.0)
            consolidation_optimizer.step()
            if consolidation_update in (1, args.consolidation_steps):
                history.append({
                    "phase": "consolidation",
                    "update": consolidation_update,
                    "new_batch_accuracy": observed_accuracy,
                    "skill_loss": float(skill_loss.detach()),
                    "retention_loss": float(retention_loss.detach()),
                    "replay_weights": [
                        float(value) for value in replay_weights.detach()],
                    "unrelated_event_residual_norm":
                        float(locality.detach()),
                    "relation_residual_norms": [
                        float(value.detach())
                        for value in residual_norms[:relation_replay_count]],
                    "total_loss": float(loss.detach()),
                })

    # All capability and retention claims use the deployed exact-zero gate.
    student.skill_adapter_gate_leak = 0.0
    if args.skip_final_evaluation:
        # Population arms are not promoted individually.  Avoid running seven
        # complete causal suites per arm: a compact held-out selector chooses
        # one candidate, and only that winner receives the full audit.
        frozen_bit_identical = all(
            torch.equal(
                frozen_initial[name],
                student.state_dict()[name].detach().cpu())
            for name in frozen_initial)
        slot_change = sum(
            float(
                (student.state_dict()[name].detach().cpu() - before)
                .square().sum())
            for name, before in slot_initial.items()) ** 0.5
        accounting = {
            "new_unique_lifetimes":
                (args.steps + args.consolidation_steps) * args.batch_size,
            "new_verifier_bits":
                (args.steps + args.consolidation_steps)
                * args.batch_size * 6,
            "replay_lifetimes_per_stream":
                (args.steps + args.consolidation_steps)
                * args.replay_batch_size,
            "replay_streams": len(replay_specs),
            "replay_specs": replay_specs,
            "total_verifier_bits":
                (args.steps + args.consolidation_steps) * (
                    args.batch_size
                    + len(replay_specs) * args.replay_batch_size) * 6,
            "optimizer_updates": args.steps + args.consolidation_steps,
        }
        candidate_path = args.candidate_checkpoint_out
        assert candidate_path is not None
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": student.state_dict(),
            "source_report": str(args.report),
            "admission_status": "unpromoted_population_candidate",
        }, candidate_path)
        report = {
            "schema": "pair-relation-appearance-bridge-candidate-v1",
            "claim_boundary": (
                "This training-only population arm has no capability claim. "
                "It received pixels, opaque attempted actions, scalar "
                "outcomes, and opaque behavior rehearsal, without semantic "
                "labels or correct unattempted actions."),
            "configuration": {
                **vars(args),
                "parent": str(args.parent),
                "report": str(args.report),
                "checkpoint_out": (
                    str(args.checkpoint_out)
                    if args.checkpoint_out is not None else None),
                "candidate_checkpoint_out": str(candidate_path),
                "parent_gate_refiner_width": parent_refiner_width,
                "effective_gate_refiner_width": refiner_width,
                "inserted_gate_refiner": inserting_refiner,
                "parent_gate_extension_width": parent_extension_width,
                "effective_gate_extension_width": extension_width,
                "inserted_gate_extension": inserting_extension,
                "acquisition_trainable_prefixes": acquisition_prefixes,
            },
            "history": history,
            "accounting": accounting,
            "frozen_base_bit_identical": frozen_bit_identical,
            "slot_l2_change": slot_change,
            "total_seconds": time.perf_counter() - started,
            "candidate_checkpoint_saved": True,
            "required_gates_passed": None,
            "checkpoint_saved": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "initialization": args.initialization,
            "candidate_checkpoint_saved": True,
            "evaluation_skipped": True,
            "seconds": report["total_seconds"],
        }, sort_keys=True))
        return

    evaluations = {
        "new_appearance": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 90_000_000, device=device,
            task="pair_relation", feedback_trials=1,
            appearance=args.new_appearance),
        "bars_retention": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 91_000_000, device=device,
            task="pair_relation", feedback_trials=1,
            appearance="bars"),
        "diamonds_retention": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 91_500_000, device=device,
            task="pair_relation", feedback_trials=1,
            appearance="diamonds"),
        "dot_pair_transfer": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 92_000_000, device=device,
            task="pair_relation", feedback_trials=1,
            appearance="dot_pairs"),
        **{
            f"{task}_retention": evaluate(
                student, count=args.test_lifetimes, trials=6,
                seed=args.seed + 93_000_000 + index,
                device=device, task=task, feedback_trials=1,
                appearance=appearance)
            for index, (task, appearance) in enumerate(
                UNRELATED_REPLAY_SPECS)
        },
    }
    required = (
        "new_appearance", "bars_retention",
        "binary_mapping_retention", "visible_context_retention",
        "visible_context_xor_retention")
    if "diamonds" in inherited_relation_appearances:
        required = (*required, "diamonds_retention")
    accepted = all(evaluations[name]["gate"]["accepted"] for name in required)
    frozen_bit_identical = all(
        torch.equal(
            frozen_initial[name],
            student.state_dict()[name].detach().cpu())
        for name in frozen_initial)
    slot_change = sum(
        float(
            (student.state_dict()[name].detach().cpu() - before)
            .square().sum())
        for name, before in slot_initial.items()) ** 0.5
    report = {
        "schema": "pair-relation-appearance-bridge-v1",
        "claim_boundary": (
            "The learner receives rendered streams, its own opaque attempted "
            "actions, scalar outcomes, and opaque behavior rehearsal. It "
            "receives no semantic relation/task label or correct unattempted "
            "action."),
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
            "parent_gate_refiner_width": parent_refiner_width,
            "effective_gate_refiner_width": refiner_width,
            "inserted_gate_refiner": inserting_refiner,
            "parent_gate_extension_width": parent_extension_width,
            "effective_gate_extension_width": extension_width,
            "inserted_gate_extension": inserting_extension,
            "acquisition_trainable_prefixes": acquisition_prefixes,
        },
        "history": history,
        "accounting": {
            "new_unique_lifetimes": (
                args.steps + args.consolidation_steps) * args.batch_size,
            "new_verifier_bits": (
                args.steps + args.consolidation_steps)
                * args.batch_size * 6,
            "replay_lifetimes_per_stream":
                (args.steps + args.consolidation_steps)
                * args.replay_batch_size,
            "replay_streams": len(replay_specs),
            "replay_specs": replay_specs,
            "total_verifier_bits": (
                args.steps + args.consolidation_steps) * (
                    args.batch_size
                    + len(replay_specs) * args.replay_batch_size) * 6,
            "optimizer_updates": args.steps + args.consolidation_steps,
        },
        "evaluations": evaluations,
        "headline_accuracy": {
            name: _headline_accuracy(value)
            for name, value in evaluations.items()},
        "required_gates_passed": accepted,
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
            "admission_status": "pair_relation_appearance_bridge",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "initialization": args.initialization,
        "accepted": accepted,
        "new_appearance": report["headline_accuracy"]["new_appearance"],
        "bars": report["headline_accuracy"]["bars_retention"],
        "dot_pairs": report["headline_accuracy"]["dot_pair_transfer"],
        "seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
