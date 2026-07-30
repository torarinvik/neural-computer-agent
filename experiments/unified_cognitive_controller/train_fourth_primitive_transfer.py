"""Acquire a fourth primitive while consolidating every earlier behavior.

This is the third rung's procedure applied one level up. Earlier rungs claimed
the action and relation residuals, so this one appends a fresh successor slot
from the indexable skill-adapter stack: the whole inherited controller, both
legacy adapters included, is frozen, and each new primitive costs exactly one
new zero-output residual while never editing an old one.

The learner receives only rendered frames, its own opaque actions, and scalar
verifier outcomes on the new task. Retention targets are the frozen starting
controller's opaque action distributions on newly rendered replay lifetimes;
semantic task state and correct unattempted actions are never targets.

Every arm replays the same three earlier tasks for the same experience, but a
retention gate is required only where the frozen parent itself passes: an arm
cannot be asked to retain a skill it never had. That keeps the experience
accounting identical across arms while keeping each arm's gates honest.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .distill_visible_context import _distillation_loss, _trajectory_loss
from .environment import (
    ACTIONS, NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes)
from .model import UnifiedCognitiveController
from .train import attempted_success_loss, evaluate, rollout, seed_everything


DEFAULT_NEW_TASK = "contextual_composition"
PAIR_RELATION_APPEARANCES = ("bars", "diamonds", "dot_pairs")
DEFAULT_REPLAY_TASKS = (
    "binary_mapping", "visible_context", "visible_context_xor")
# Kept for importers and tests that pin the fourth rung's own pair.
NEW_TASK = DEFAULT_NEW_TASK
REPLAY_TASKS = DEFAULT_REPLAY_TASKS


def _plastic_prefixes(slot: int) -> tuple[str, ...]:
    """Name only the slot this rung appends, so earlier slots stay frozen.

    The read projection belongs to the slot that reads through it, so it is
    plastic with the rest of the slot; every earlier slot's projection stays
    frozen along with the adapter it feeds.
    """
    return (
        f"skill_adapters.{slot}.",
        f"skill_adapter_gates.{slot}.",
        f"skill_adapter_read_projections.{slot}.",
        f"skill_adapter_intention_interactions.{slot}.",
        f"skill_adapter_outer_event_projections.{slot}.",
        f"skill_adapter_outer_intention_projections.{slot}.")


def _prior_slot_prefixes(
        inherited_slot_count: int, thawed_slot_count: int,
        ) -> tuple[str, ...]:
    """Name the most recent inherited slots allowed to change.

    A fixed volatility is only an existence test for learned metaplasticity.
    Keeping this helper index-based makes the no-thaw default exactly reproduce
    the established frozen-parent path.
    """
    if thawed_slot_count < 0:
        raise ValueError("thawed slot count must not be negative")
    first = max(0, inherited_slot_count - thawed_slot_count)
    return tuple(
        prefix
        for slot in range(first, inherited_slot_count)
        for prefix in (
            f"skill_adapters.{slot}.",
            f"skill_adapter_gates.{slot}.",
            f"skill_adapter_read_projections.{slot}."))


def _relative_state_drift(
        model: nn.Module, initial: dict[str, torch.Tensor],
        ) -> float:
    """Relative L2 movement of the intentionally plastic inherited state."""
    if not initial:
        return 0.0
    changed = 0.0
    scale = 0.0
    current = model.state_dict()
    for name, before in initial.items():
        after = current[name].detach().cpu()
        changed += float((after - before).square().sum())
        scale += float(before.square().sum())
    return (changed / max(scale, 1e-24)) ** 0.5


def _alignment_volatility(alignment: float, maximum: float) -> float:
    """Smoothly thaw when acquisition and retention gradients agree.

    Orthogonal evidence gets half the available plasticity. Agreement moves
    toward the maximum; conflict approaches hard freezing. This consumes no
    extra verifier outcomes and has no semantic task input.
    """
    return maximum / (1.0 + math.exp(-8.0 * alignment))


def _gradient_alignment(
        new_loss: torch.Tensor, retention_loss: torch.Tensor,
        parameters: list[nn.Parameter],
        ) -> float:
    """Cosine agreement between acquisition and retention on one slot."""
    if not parameters:
        return 0.0
    new_grads = torch.autograd.grad(
        new_loss, parameters, retain_graph=True, allow_unused=True)
    retention_grads = torch.autograd.grad(
        retention_loss, parameters, retain_graph=True, allow_unused=True)
    dot = new_loss.new_zeros(())
    new_norm = new_loss.new_zeros(())
    retention_norm = new_loss.new_zeros(())
    for new_grad, retention_grad in zip(new_grads, retention_grads):
        if new_grad is None or retention_grad is None:
            continue
        dot = dot + (new_grad * retention_grad).sum()
        new_norm = new_norm + new_grad.square().sum()
        retention_norm = retention_norm + retention_grad.square().sum()
    denominator = (new_norm * retention_norm).sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((dot / denominator).clamp(-1.0, 1.0))


def _importance_volatility(
        importance: torch.Tensor, maximum: float, strength: float,
        ) -> torch.Tensor:
    """Protect frequently retention-critical hidden units.

    Importance is normalized within the slot, so the control depends on
    relative use rather than an arbitrary loss scale. An unused unit stays
    maximally plastic; a heavily used one changes slowly.
    """
    normalized = importance / importance.mean().clamp_min(1e-12)
    return maximum / (1.0 + strength * normalized)


@torch.no_grad()
def _blend_unit_update(
        parameter: nn.Parameter, before: torch.Tensor,
        volatility: torch.Tensor,
        ) -> None:
    """Scale an Adam update by one volatility scalar per output unit."""
    shape = (volatility.numel(),) + (1,) * (parameter.ndim - 1)
    scale = volatility.reshape(shape)
    parameter.copy_(before + scale * (parameter - before))


def _replay_loss_and_leakage(
        student: UnifiedCognitiveController,
        teacher: UnifiedCognitiveController, batch, *, slot: int,
        feedback_trials: int,
        shuffled_teacher: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Distil old behavior and measure how far the new slot opens doing it.

    Same trajectory distillation as the third rung, but it also returns the
    mean opening of this rung's slot on these replayed events. A slot that
    stays shut here cannot disturb the skill being replayed, so the opening is
    a direct, label-free price for interference: the verifier already chose
    which generator produced this batch, and the controller still sees no task
    identity.
    """
    count = batch.batch_size
    device = batch.frames.device
    student_state = student.initial_state(count, device=device)
    teacher_state = teacher.initial_state(count, device=device)
    null = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    student_action = teacher_action = null
    student_reward = teacher_reward = torch.zeros(count, device=device)
    losses = []
    openings = []
    query_rewards = []
    for trial in range(batch.trials):
        feedback = torch.full_like(
            student_reward, float(0 < trial <= feedback_trials))
        student_output, student_state = student.step(
            batch.frames[:, trial], student_state, student_action,
            student_reward * feedback, feedback)
        with torch.no_grad():
            teacher_output, teacher_state = teacher.step(
                batch.frames[:, trial], teacher_state, teacher_action,
                teacher_reward * feedback, feedback)
            target = (
                teacher_output.logits.roll(1, dims=0) if shuffled_teacher
                else teacher_output.logits)
        losses.append(_distillation_loss(student_output.logits, target))
        # The disturbance to price is the residual the slot actually adds, not
        # the gate opening: penalising the opening alone lets the adapter grow
        # its output to compensate, which is what happened when the opening
        # fell fortyfold and retention still got worse.
        assert student_output.skill_adapter_residual_norms is not None
        openings.append(
            student_output.skill_adapter_residual_norms[:, slot])
        student_action = student_output.logits.detach().argmax(-1)
        teacher_action = teacher_output.logits.argmax(-1)
        student_reward = (
            student_action == batch.correct_actions[:, trial]).float()
        teacher_reward = (
            teacher_action == batch.correct_actions[:, trial]).float()
        if trial >= feedback_trials:
            query_rewards.append(student_reward)
    student_accuracy = float(torch.stack(query_rewards).mean())
    return (
        torch.stack(losses).mean(), torch.stack(openings).mean(),
        student_accuracy)


@torch.no_grad()
def _slot_opening(
        model: UnifiedCognitiveController, *, slot: int, task: str,
        count: int, seed: int, support_trials: int,
        device: torch.device,
        numerosity_appearance_blend: float = 1.0,
        operation_cue_scale: float = 1.0,
        operation_cue_trials: int | None = None,
        operation_cue_prestimulus: bool = False,
        ) -> tuple[float, float]:
    """Mean opening of one slot, and how often it is exactly shut."""
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task=task,
        support_trials=support_trials,
        numerosity_appearance_blend=numerosity_appearance_blend,
        operation_cue_scale=operation_cue_scale,
        operation_cue_trials=operation_cue_trials,
        operation_cue_prestimulus=operation_cue_prestimulus,
        device=device)
    state = model.initial_state(count, device=device)
    action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    reward = torch.zeros(count, device=device)
    openings = []
    for trial in range(batch.trials):
        if batch.prestimulus_frames is not None:
            _, state = model.step(
                batch.prestimulus_frames[:, trial], state, action,
                torch.zeros_like(reward), torch.zeros_like(reward))
        feedback = torch.full_like(
            reward, float(0 < trial <= support_trials))
        output, state = model.step(
            batch.frames[:, trial], state, action, reward * feedback,
            feedback)
        assert output.skill_adapter_openings is not None
        openings.append(output.skill_adapter_openings[:, slot])
        action = output.logits.argmax(-1)
        reward = (action == batch.correct_actions[:, trial]).float()
    stacked = torch.stack(openings)
    # The fraction of events the slot is exactly shut on is the property that
    # matters: only an exact zero leaves an inherited skill untouched, and a
    # sigmoid gate can never produce one.
    return float(stacked.mean()), float((stacked == 0).float().mean())


@torch.no_grad()
def _slot_residual_norm(
        model: UnifiedCognitiveController, *, slot: int, task: str,
        count: int, seed: int, support_trials: int,
        device: torch.device,
        numerosity_appearance_blend: float = 1.0,
        operation_cue_scale: float = 1.0,
        operation_cue_trials: int | None = None,
        operation_cue_prestimulus: bool = False,
        ) -> float:
    """Mean norm of the perturbation one slot actually adds to the intention.

    The gate opening alone is not the disturbance: a nearly shut gate on a
    large residual still moves the answer, so this is the quantity a locality
    price has to act on.
    """
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task=task,
        support_trials=support_trials,
        numerosity_appearance_blend=numerosity_appearance_blend,
        operation_cue_scale=operation_cue_scale,
        operation_cue_trials=operation_cue_trials,
        operation_cue_prestimulus=operation_cue_prestimulus,
        device=device)
    state = model.initial_state(count, device=device)
    action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    reward = torch.zeros(count, device=device)
    norms = []
    for trial in range(batch.trials):
        if batch.prestimulus_frames is not None:
            _, state = model.step(
                batch.prestimulus_frames[:, trial], state, action,
                torch.zeros_like(reward), torch.zeros_like(reward))
        feedback = torch.full_like(
            reward, float(0 < trial <= support_trials))
        output, state = model.step(
            batch.frames[:, trial], state, action, reward * feedback,
            feedback)
        # Read the norm the controller itself computed, so this never has to
        # reimplement the gate and cannot drift from the active gate mode.
        assert output.skill_adapter_residual_norms is not None
        norms.append(output.skill_adapter_residual_norms[:, slot])
        action = output.logits.argmax(-1)
        reward = (action == batch.correct_actions[:, trial]).float()
    return float(torch.stack(norms).mean())


@torch.no_grad()
def _curve_accuracy(
        model: UnifiedCognitiveController, *, task: str, count: int,
        seed: int, support_trials: int, device: torch.device,
        numerosity_appearance_blend: float = 1.0,
        operation_cue_scale: float = 1.0,
        operation_cue_trials: int | None = None,
        operation_cue_prestimulus: bool = False,
        ) -> float:
    """One rollout's post-support accuracy: the cheapest honest curve sample."""
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task=task,
        support_trials=support_trials,
        numerosity_appearance_blend=numerosity_appearance_blend,
        operation_cue_scale=operation_cue_scale,
        operation_cue_trials=operation_cue_trials,
        operation_cue_prestimulus=operation_cue_prestimulus,
        device=device)
    result = rollout(
        model, batch, sample_actions=False, feedback_trials=support_trials)
    return float(result["rewards"][:, support_trials:].float().mean())


@torch.no_grad()
def _operation_counterfactual_metrics(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, numerosity_appearance_blend: float,
        operation_cue_scale: float = 1.0,
        ) -> dict[str, float]:
    """Audit cue causality with independent, history-free event replays.

    Each two-event lifetime contains one request for each operation. Flattening
    those events and resetting controller state prevents reward history or
    recurrent timing from explaining a flip. The counterfactual preserves
    every stimulus pixel and complements only the public operation symbol and
    verifier answer.
    """
    def actions_and_accuracy(reverse: bool) -> tuple[torch.Tensor, float]:
        batch = generate_lifetimes(
            count, 2, seed=seed, heldout=True,
            task="visible_pair_numerosity_operation",
            reverse_operations=reverse,
            numerosity_appearance_blend=numerosity_appearance_blend,
            operation_cue_scale=operation_cue_scale,
            operation_cue_prestimulus=True,
            support_trials=1, device=device)
        assert batch.prestimulus_frames is not None
        events = count * 2
        state = model.initial_state(events, device=device)
        null_action = torch.full(
            (events,), NULL_ACTION, dtype=torch.long, device=device)
        zeros = torch.zeros(events, device=device)
        _, state = model.step(
            batch.prestimulus_frames.flatten(0, 1),
            state, null_action, zeros, zeros)
        output, _ = model.step(
            batch.frames.flatten(0, 1),
            state, null_action, zeros, zeros)
        actions = output.logits.argmax(-1)
        answers = batch.correct_actions.flatten()
        return actions.cpu(), float((actions == answers).float().mean())

    normal_actions, normal_accuracy = actions_and_accuracy(False)
    reversed_actions, reversed_accuracy = actions_and_accuracy(True)
    return {
        "normal_accuracy": normal_accuracy,
        "reversed_operation_accuracy": reversed_accuracy,
        "prediction_flip_rate": float(
            (normal_actions != reversed_actions).float().mean()),
        "paired_mean_accuracy":
            (normal_accuracy + reversed_accuracy) / 2.0,
    }


def _headline_accuracy(evaluation: dict) -> float:
    """The comparable accuracy for either task family.

    Tasks whose answer is fully visible report one overall accuracy; hidden
    rule tasks such as the new composition are only identifiable after their
    support outcome, so their headline number is post-feedback accuracy.
    """
    if "overall_accuracy" in evaluation:
        return float(evaluation["overall_accuracy"])
    return float(evaluation["normal"]["post_feedback_accuracy"])


def _replay_appearance(task: str, policy: str, update: int) -> str:
    """Protect a visual repertoire without buying more replay experience."""
    if task != "pair_relation":
        return "bars"
    if policy == "cycle":
        return PAIR_RELATION_APPEARANCES[
            (update - 1) % len(PAIR_RELATION_APPEARANCES)]
    return policy


def _load(
        path: Path, device: torch.device,
        ) -> tuple[dict[str, object], UnifiedCognitiveController]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError("unsupported controller checkpoint")
    configuration = payload.get("model_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("controller checkpoint lacks model configuration")
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"])
    return payload, model


def _new_skill_loss(
        model: UnifiedCognitiveController, batch, *,
        exploration: float, support_trials: int,
        learning_rule: str = "bce",
        independent_events: bool = False,
        independent_event_share: float = 0.0,
        independent_action_augmentation: bool = False,
        return_accuracy: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, float]:
    if independent_event_share:
        if independent_events:
            raise ValueError(
                "an all-independent batch cannot also request a mixed share")
        if not 0.0 < independent_event_share < 1.0:
            raise ValueError(
                "independent event share must lie strictly within (0, 1)")
        independent_count = int(round(
            batch.batch_size * independent_event_share))
        independent_count = min(
            batch.batch_size - 1, max(1, independent_count))

        def subset(start: int, stop: int):
            return replace(
                batch,
                frames=batch.frames[start:stop],
                correct_actions=batch.correct_actions[start:stop],
                stimulus_identities=batch.stimulus_identities[start:stop],
                rule_bits=batch.rule_bits[start:stop],
                seeds=batch.seeds[start:stop],
                context_ids=(
                    batch.context_ids[start:stop]
                    if batch.context_ids is not None else None),
                prestimulus_frames=(
                    batch.prestimulus_frames[start:stop]
                    if batch.prestimulus_frames is not None else None))

        independent_result = _new_skill_loss(
            model, subset(0, independent_count),
            exploration=exploration, support_trials=support_trials,
            learning_rule=learning_rule,
            independent_events=True,
            independent_action_augmentation=(
                independent_action_augmentation),
            return_accuracy=return_accuracy)
        recurrent_result = _new_skill_loss(
            model, subset(independent_count, batch.batch_size),
            exploration=exploration, support_trials=support_trials,
            learning_rule=learning_rule,
            independent_events=False,
            independent_action_augmentation=False,
            return_accuracy=return_accuracy)
        weight = independent_count / batch.batch_size
        if return_accuracy:
            independent_loss, independent_accuracy = independent_result
            recurrent_loss, recurrent_accuracy = recurrent_result
            return (
                weight * independent_loss + (1.0 - weight) * recurrent_loss,
                weight * independent_accuracy
                + (1.0 - weight) * recurrent_accuracy)
        return (
            weight * independent_result
            + (1.0 - weight) * recurrent_result)

    if independent_events:
        events = batch.batch_size * batch.trials
        state = model.initial_state(
            events, device=batch.frames.device, dtype=batch.frames.dtype)
        previous_action = torch.full(
            (events,), NULL_ACTION, dtype=torch.long,
            device=batch.frames.device)
        if independent_action_augmentation:
            # Previous action is an irrelevant nuisance for a fully visible
            # operation. Sampling all opaque actions plus the true reset token
            # prevents a fresh-state learner from keying on NULL and failing
            # when embedded in an uninterrupted real-time stream.
            previous_action = torch.randint(
                0, ACTIONS + 1, (events,), device=batch.frames.device)
        zeros = torch.zeros(events, device=batch.frames.device)
        if batch.prestimulus_frames is not None:
            _, state = model.step(
                batch.prestimulus_frames.flatten(0, 1), state,
                previous_action, zeros, zeros)
        output, _ = model.step(
            batch.frames.flatten(0, 1), state,
            previous_action, zeros, zeros)
        probabilities = output.logits.softmax(dim=-1)
        behavior = (
            probabilities * (1.0 - exploration)
            + exploration / ACTIONS)
        actions = torch.multinomial(behavior, 1).squeeze(1)
        outcomes = (
            actions == batch.correct_actions.flatten()).to(
                output.logits.dtype)
        if learning_rule == "bce":
            loss = attempted_success_loss(
                output.logits, actions, outcomes)
        elif learning_rule == "policy_gradient":
            if exploration != 1.0:
                raise ValueError(
                    "policy-gradient control requires exact uniform logging")
            loss = _attempted_policy_gradient_loss(
                output.logits, actions, outcomes)
        else:
            raise ValueError(
                f"unknown new-skill learning rule {learning_rule}")
        if return_accuracy:
            return loss, float(outcomes.mean())
        return loss

    result = rollout(
        model, batch, sample_actions=True, exploration=exploration,
        feedback_trials=support_trials)
    losses = []
    for trial in range(batch.trials):
        if learning_rule == "bce":
            loss = attempted_success_loss(
                result["logits"][:, trial],
                result["actions"][:, trial],
                result["rewards"][:, trial])
        elif learning_rule == "policy_gradient":
            if exploration != 1.0:
                raise ValueError(
                    "policy-gradient control requires exact uniform logging")
            loss = _attempted_policy_gradient_loss(
                result["logits"][:, trial],
                result["actions"][:, trial],
                result["rewards"][:, trial])
        else:
            raise ValueError(f"unknown new-skill learning rule {learning_rule}")
        # Support trials precede the outcomes that identify the hidden rule,
        # so their gradient is deliberately discounted.
        losses.append(loss * (0.20 if trial < support_trials else 1.0))
    loss = torch.stack(losses).mean()
    if return_accuracy:
        accuracy = float(
            result["rewards"][:, support_trials:].float().mean())
        return loss, accuracy
    return loss


def _attempted_policy_gradient_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor,
        ) -> torch.Tensor:
    """Uniform-logging bandit loss from attempted scalar outcomes only."""
    selected_log_probability = logits.log_softmax(-1).gather(
        1, attempted.unsqueeze(1)).squeeze(1)
    # The task-agnostic binary-chance baseline turns both observed successes
    # and failures into useful gradients. No propensity model or unattempted
    # action target is needed under exact 0.5/0.5 logging.
    advantage = outcomes - 0.5
    return -(advantage.detach() * selected_log_probability).mean()


def _shuffle_verifier_outcomes(batch, *, seed: int):
    """Break the sensory/outcome relation while preserving its marginal."""
    generator = torch.Generator().manual_seed(seed)
    permutations = [
        torch.randperm(batch.batch_size, generator=generator)]
    permutations.extend(
        torch.arange(batch.batch_size).roll(offset)
        for offset in range(1, batch.batch_size))
    row_permutation = next(
        (
            candidate
            for candidate in permutations
            if not torch.equal(
                batch.correct_actions[
                    candidate.to(batch.correct_actions.device)],
                batch.correct_actions)
        ),
        None)
    if row_permutation is not None:
        return replace(
            batch,
            correct_actions=batch.correct_actions[
                row_permutation.to(batch.correct_actions.device)])
    flattened = batch.correct_actions.flatten()
    cell_permutations = [
        torch.randperm(flattened.numel(), generator=generator)]
    cell_permutations.extend(
        torch.arange(flattened.numel()).roll(offset)
        for offset in range(1, flattened.numel()))
    cell_permutation = next(
        (
            candidate
            for candidate in cell_permutations
            if not torch.equal(
                flattened[candidate.to(flattened.device)], flattened)
        ),
        None)
    if cell_permutation is None:
        raise ValueError("verifier shuffle cannot change a constant batch")
    return replace(
        batch,
        correct_actions=flattened[
            cell_permutation.to(flattened.device)].reshape_as(
                batch.correct_actions))


@torch.no_grad()
def _operation_cue_ablation_accuracy(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, support_trials: int,
        new_task: str = DEFAULT_NEW_TASK,
        appearance: str = "bars",
        appearance_blend: float | None = None,
        numerosity_mass_control: float = 0.0,
        numerosity_appearance_blend: float = 1.0,
        operation_cue_scale: float = 1.0,
        operation_cue_trials: int | None = None,
        operation_cue_prestimulus: bool = False) -> float:
    """Rerender the same public events without the operation-mode symbol.

    The composition cue is the only pixel difference from the direct-context
    rendering, so this isolates whether the requested operation is read off
    the frame rather than guessed from the rest of the scene.
    """
    marked = generate_lifetimes(
        count, 6, seed=seed, heldout=True,
        task=new_task, appearance=appearance,
        appearance_blend=appearance_blend,
        numerosity_mass_control=numerosity_mass_control,
        numerosity_appearance_blend=numerosity_appearance_blend,
        operation_cue_scale=operation_cue_scale,
        operation_cue_trials=operation_cue_trials,
        operation_cue_prestimulus=operation_cue_prestimulus,
        support_trials=support_trials, device=device)
    if new_task in (
            "pair_relation", "pair_magnitude",
            "visible_pair_magnitude", "visible_pair_numerosity",
            "visible_pair_numerosity_smaller",
            "visible_pair_numerosity_operation",
            "visible_numerosity_equality"):
        # Remove only the second held-out-position object. Its mask is confined
        # to rows 13:28, columns 5:20; filling that box from a clean corner
        # preserves the per-frame background while deleting the relational
        # evidence. This is a valid pixel ablation, not a hidden-state edit.
        ablated_frames = marked.frames.clone()
        background = marked.frames[:, :, :, -1:, -1:]
        if new_task == "pair_relation":
            ablated_frames[:, :, :, 13:28, 5:20] = background
        elif new_task in (
                "visible_pair_numerosity",
                "visible_pair_numerosity_smaller",
                "visible_pair_numerosity_operation",
                "visible_numerosity_equality"):
            # Delete the complete right count field while preserving the left.
            ablated_frames[:, :, :, :, 16:32] = background
        else:
            # Magnitude's held-out second object is centred at (16, 26).
            # This box covers its largest dilation without touching the first
            # object centred at (16, 6).
            ablated_frames[:, :, :, 16:32, 4:29] = background
        ablated = CognitiveLifetimeBatch(
            frames=ablated_frames,
            correct_actions=marked.correct_actions,
            stimulus_identities=marked.stimulus_identities,
            rule_bits=marked.rule_bits,
            seeds=marked.seeds,
            context_ids=marked.context_ids)
        result = rollout(
            model, ablated, sample_actions=False,
            feedback_trials=support_trials)
        return float(result["rewards"].float().mean())
    unmarked = generate_lifetimes(
        count, 6, seed=seed, heldout=True,
        task="visible_context", support_trials=support_trials,
        device=device)
    if (
            not torch.equal(
                marked.stimulus_identities, unmarked.stimulus_identities)
            or not torch.equal(marked.context_ids, unmarked.context_ids)
            or not torch.equal(marked.rule_bits, unmarked.rule_bits)):
        raise RuntimeError("operation-cue rerender changed task content")
    if torch.equal(marked.frames, unmarked.frames):
        raise RuntimeError("operation cue is not visible in the rendering")
    ablated = CognitiveLifetimeBatch(
        frames=unmarked.frames,
        correct_actions=marked.correct_actions,
        stimulus_identities=marked.stimulus_identities,
        rule_bits=marked.rule_bits,
        seeds=marked.seeds,
        context_ids=marked.context_ids)
    result = rollout(
        model, ablated, sample_actions=False,
        feedback_trials=support_trials)
    return float(result["rewards"].float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--seeds",
        help=("comma separated seeds to run in this one process. Interpreter, "
              "torch and CUDA start-up dominate a short rung, so a sweep costs "
              "far less as one process per arm than one per run. --report and "
              "--checkpoint-out are then templates taking {seed}"))
    parser.add_argument(
        "--eval-mode", choices=("full", "curve"), default="full",
        help=("full runs every audit; curve records only headline accuracies. "
              "Interior points of a budget sweep only contribute a curve "
              "sample, and the full suite costs about six rollouts per task"))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--steps-grid",
        help=("comma separated budgets to sweep in this one process. The "
              "parent's own retention audit does not depend on the budget, so "
              "sweeping inside one process computes it once instead of once "
              "per point. --report may take {steps} as well as {seed}"))
    parser.add_argument("--new-batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--skill-adapter-width", type=int, default=64)
    parser.add_argument(
        "--new-support-trials", type=int, default=1,
        help=("support outcomes on the new task only; the graduated rungs "
              "start above one and the final rung must reach one. Replay "
              "always uses the one support the teacher consolidated at."))
    parser.add_argument(
        "--final-support-trials", type=int,
        help=("support outcomes to finish on and to evaluate at; defaults to "
              "--new-support-trials. Set it lower to reduce support during "
              "the rung, which is how the original binary mapping was reached"))
    parser.add_argument(
        "--support-switch-fraction", type=float, default=0.5,
        help="fraction of the budget spent before dropping to the final support")
    parser.add_argument(
        "--gate-warmup-fraction", type=float, default=0.0,
        help=("fraction of the budget during which the new slot's gate is held "
              "open and frozen. A rectified gate that shuts before its adapter "
              "has any reason to be open is left without gradient and stays "
              "shut; this stops it closing on an adapter that still outputs "
              "zero"))
    parser.add_argument(
        "--gate-leak-initial", type=float, default=0.0,
        help=("initial leak below a rectified gate's knee, annealed to exactly "
              "zero; keeps a gate that shuts everywhere recoverable"))
    parser.add_argument(
        "--gate-leak-anneal-fraction", type=float, default=0.25,
        help="fraction of the budget over which the leak reaches exactly zero")
    parser.add_argument(
        "--new-task", default=DEFAULT_NEW_TASK,
        help="the primitive this rung acquires")
    parser.add_argument(
        "--new-numerosity-appearance-blend", type=float, default=1.0,
        help=(
            "bar-to-dot appearance blend for a new numerosity task; this is "
            "sensory difficulty only and never reveals the answer"))
    parser.add_argument(
        "--new-operation-cue-scale", type=float, default=1.0,
        help=(
            "salience of the public operation cue. This is an explicit "
            "perceptual curriculum axis, not a semantic label"))
    parser.add_argument(
        "--new-operation-cue-trials", type=int, default=0,
        help=(
            "number of initial real-time events carrying the operation cue; "
            "zero keeps it visible on every event"))
    parser.add_argument(
        "--new-operation-cue-prestimulus", action="store_true",
        help=(
            "show the public cue as a separate sensory frame immediately "
            "before each action-bearing stimulus"))
    parser.add_argument(
        "--replay-numerosity-appearance-blend", type=float, default=1.0,
        help=(
            "bar-to-dot appearance blend for replayed numerosity skills; "
            "matching the inherited frontier preserves the actual skill"))
    parser.add_argument(
        "--new-position-augmentation", action="store_true",
        help=(
            "alternate the new task across the train and held-out nuisance "
            "positions while retaining the training palette; this separates "
            "position invariance from held-out-colour generalization"))
    parser.add_argument(
        "--replay-tasks", default=",".join(DEFAULT_REPLAY_TASKS),
        help="comma separated primitives the parent already holds")
    parser.add_argument(
        "--replay-support-trials",
        help=("comma separated support counts, one per replay task; defaults "
              "to one each. A skill must be replayed and audited at the "
              "support it was acquired at, or its retention is unmeasurable"))
    parser.add_argument(
        "--pair-relation-replay-appearance",
        choices=(*PAIR_RELATION_APPEARANCES, "cycle"),
        default="bars",
        help=(
            "appearance used for pair_relation replay. 'cycle' rotates "
            "through its full learned contour repertoire at unchanged cost"))
    parser.add_argument(
        "--slot-reads-prior", action="store_true",
        help=("let this rung's slot read what earlier slots computed, while "
              "leaving their writes gated. Without it an exactly shut gate "
              "makes every deeper ancestry hand the new slot bit-identical "
              "features, so there is nothing to inherit"))
    parser.add_argument(
        "--read-bottleneck", type=int, default=0,
        help=("compress everything the slot reads to this width. One prior slot "
              "helped raw; two, or the legacy pair, hurt, so a wide read looks "
              "like dilution rather than information"))
    parser.add_argument(
        "--prior-read-limit", type=int, default=0,
        help=("read only this many immediately preceding skill slots; zero "
              "reads all earlier slots. One ancestor improved absolute learning "
              "while a second added no transfer gain, so one tests local reuse"))
    parser.add_argument(
        "--read-parent-intention", action="store_true",
        help=(
            "let the newly appended slot read the parent's accumulated latent "
            "intention, enabling learned transformations of an existing "
            "decision without exposing action labels or logits"))
    parser.add_argument(
        "--read-event-snapshot", action="store_true",
        help=(
            "let the new slot read the generic one-event sensory RAM trace; "
            "each controller step overwrites it, without a cue or task flag"))
    parser.add_argument(
        "--ablate-event-snapshot", action="store_true",
        help=(
            "matched control: keep the event-snapshot interface and parameter "
            "count but zero only its content for the appended slot"))
    parser.add_argument(
        "--multiply-parent-intention", action="store_true",
        help=(
            "append a generic learned state-by-intention interaction to the "
            "new slot. This supplies a bilinear binding bias without task IDs, "
            "semantic labels, or verifier-private values"))
    parser.add_argument(
        "--outer-product-parent-intention", action="store_true",
        help=(
            "append every pairwise product between low-dimensional learned "
            "event and inherited-intention projections"))
    parser.add_argument(
        "--outer-product-width", type=int, default=8,
        help="projection width on each side of the generic outer product")
    parser.add_argument(
        "--ablate-outer-product", action="store_true",
        help=(
            "matched capacity control: keep the widened module but zero only "
            "the outer-product content"))
    parser.add_argument(
        "--canonicalize-action-adapter", action="store_true",
        help=(
            "map a legacy action-logit residual through the learned actuator "
            "right inverse so later slots see the complete amodal intention"))
    parser.add_argument(
        "--prior-slot-volatility", type=float, default=0.0,
        help=("learning-rate multiplier in [0,1] for inherited skill slots. "
              "Zero preserves hard freezing; positive values are a diagnostic "
              "existence test for usage-aware learned plasticity"))
    parser.add_argument(
        "--volatility-policy", choices=("fixed", "gradient_alignment"),
        default="fixed",
        help=("fixed uses --prior-slot-volatility directly; gradient_alignment "
              "treats it as a maximum and reduces plasticity when acquisition "
              "and retention gradients conflict"))
    parser.add_argument(
        "--volatility-alignment-decay", type=float, default=0.9,
        help="EMA decay for the gradient-alignment volatility controller")
    parser.add_argument(
        "--unit-volatility-policy",
        choices=("none", "retention_importance", "shuffled_importance"),
        default="none",
        help=("adapt only the prior slot's latent hidden units. Retention "
              "gradient magnitude protects useful units; shuffled_importance "
              "keeps the same scalar distribution but breaks attribution"))
    parser.add_argument(
        "--unit-volatility-max", type=float, default=0.2,
        help="maximum Adam update fraction for an unused hidden unit")
    parser.add_argument(
        "--unit-importance-strength", type=float, default=4.0,
        help="how strongly normalized retention importance suppresses updates")
    parser.add_argument(
        "--unit-importance-decay", type=float, default=0.9,
        help="EMA decay for per-hidden-unit retention importance")
    parser.add_argument(
        "--volatile-prior-slots", type=int, default=1,
        help=("number of immediately preceding inherited skill slots governed "
              "by --prior-slot-volatility"))
    parser.add_argument(
        "--telemetry-every", type=int, default=0,
        help=("record already-observed training reward and loss every N updates; "
              "zero records only the first and final update"))
    parser.add_argument(
        "--read-legacy-adapters", action="store_true",
        help=("also let this rung's slot read the two legacy adapters, which is "
              "where rungs two and three consolidated. Only the slot this rung "
              "adds reads them, so earlier slots keep their input width and "
              "their checkpoints still load"))
    parser.add_argument(
        "--ablate-prior-read", action="store_true",
        help=("control for --slot-reads-prior: keep the wider input and zero "
              "its content, so a speedup from inherited information is "
              "distinguishable from one from extra capacity"))
    parser.add_argument(
        "--slot-gate-hidden", type=int, default=0,
        help=("hidden units in the new slot's gate; zero keeps the single "
              "hyperplane, which limits how cleanly a slot can separate its "
              "own events from every earlier skill's"))
    parser.add_argument(
        "--slot-gate-mode", choices=("sigmoid", "relu"), default="sigmoid",
        help=("rectified gates can shut exactly, so a slot can be genuinely "
              "inert outside its own operation; sigmoid never reaches zero"))
    parser.add_argument(
        "--replay-selection", choices=("all", "gate"), default="all",
        help=("all replays every earlier skill each update; gate replays only "
              "the skills this slot can still reach. Replay is what grows once "
              "interference is gone -- it is linear per rung and quadratic over "
              "a ladder -- and the slot's own gate already measures which old "
              "skills it can disturb at all"))
    parser.add_argument(
        "--replay-gate-threshold", type=float, default=0.98,
        help="skip replay of a skill the slot is exactly shut on this often")
    parser.add_argument(
        "--replay-selection-every", type=int, default=64,
        help="updates between remeasuring which skills the slot can reach")
    parser.add_argument(
        "--retention-control-gain", type=float, default=0.0,
        help=("proportional gain on each old skill's shortfall against the "
              "level it was inherited at; zero reproduces the fixed price"))
    parser.add_argument(
        "--retention-tracking-decay", type=float, default=0.9,
        help="smoothing on the measured replay accuracy the control acts on")
    parser.add_argument(
        "--locality-weight", type=float, default=0.0,
        help=("price the new slot's mean opening on replayed old-skill "
              "events; zero reproduces the unpriced rung"))
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--final-learning-rate", type=float,
        help=(
            "optional cosine-decayed learning rate reached at the final "
            "update; decay begins at --learning-rate-decay-start"))
    parser.add_argument(
        "--learning-rate-decay-start", type=float, default=0.5,
        help="fraction of updates completed before cosine decay begins")
    parser.add_argument(
        "--shuffle-retention-teacher", action="store_true",
        help="negative control: mismatch teacher behavior across lifetimes")
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--exploration", type=float, default=0.1)
    parser.add_argument(
        "--new-learning-rule", choices=("bce", "policy_gradient"),
        default="bce",
        help=(
            "loss for attempted actions. policy_gradient uses exact uniform "
            "logging and a task-agnostic chance baseline"))
    parser.add_argument(
        "--new-independent-events", action="store_true",
        help=(
            "train every new-task cue/stimulus pair from a fresh active state; "
            "the same attempted-action outcomes are used, but episode history "
            "cannot substitute for the current sensory operation"))
    parser.add_argument(
        "--new-independent-event-updates", type=int, default=0,
        help=(
            "number of initial updates trained as independent events before "
            "consolidating on the ordinary recurrent stream; zero disables "
            "this staged curriculum"))
    parser.add_argument(
        "--new-independent-event-share", type=float, default=0.0,
        help=(
            "fraction of each new-task batch trained from fresh active state; "
            "the remaining lifetimes use the ordinary recurrent stream, so "
            "the verifier-outcome budget is unchanged"))
    parser.add_argument(
        "--new-independent-action-augmentation", action="store_true",
        help=(
            "randomize the learner-visible previous opaque action on fresh-"
            "state events, making current sensory evidence invariant to this "
            "generic real-time history channel"))
    parser.add_argument(
        "--shuffle-new-verifier-outcomes", action="store_true",
        help=(
            "negative control: permute new-task verifier outcomes across "
            "otherwise unchanged rendered lifetimes"))
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(args.steps, args.new_batch_size, args.replay_batch_size) < 1:
        raise ValueError("steps and batch sizes must be positive")
    if args.retention_weight <= 0:
        raise ValueError("retention weight must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if (
            args.final_learning_rate is not None
            and not 0 < args.final_learning_rate <= args.learning_rate):
        raise ValueError(
            "final learning rate must be positive and no larger than initial")
    if not 0.0 <= args.learning_rate_decay_start < 1.0:
        raise ValueError(
            "learning-rate decay start must be within [0, 1)")
    if args.skill_adapter_width < 1:
        raise ValueError("the new plastic slot must have positive width")
    if args.prior_read_limit < 0:
        raise ValueError("prior read limit must not be negative")
    if not 0.0 <= args.prior_slot_volatility <= 1.0:
        raise ValueError("prior slot volatility must be within [0, 1]")
    if not 0.0 <= args.volatility_alignment_decay < 1.0:
        raise ValueError("volatility alignment decay must be within [0, 1)")
    if not 0.0 < args.unit_volatility_max <= 1.0:
        raise ValueError("unit volatility maximum must be within (0, 1]")
    if args.unit_importance_strength < 0.0:
        raise ValueError("unit importance strength must not be negative")
    if not 0.0 <= args.unit_importance_decay < 1.0:
        raise ValueError("unit importance decay must be within [0, 1)")
    if (
            args.unit_volatility_policy != "none"
            and args.prior_slot_volatility > 0.0):
        raise ValueError(
            "slot-level and unit-level volatility are separate experiments")
    if args.volatile_prior_slots < 1:
        raise ValueError("volatile prior slots must be positive")
    if args.telemetry_every < 0:
        raise ValueError("telemetry interval must not be negative")
    if not 1 <= args.new_support_trials < 6:
        raise ValueError("new-task support trials must be between 1 and 5")
    final_support_trials = (
        args.new_support_trials if args.final_support_trials is None
        else args.final_support_trials)
    if not 1 <= final_support_trials <= args.new_support_trials:
        raise ValueError(
            "final support trials must be between one and the starting value")
    if not 0.0 <= args.support_switch_fraction <= 1.0:
        raise ValueError("support switch fraction must be within [0, 1]")
    if args.gate_leak_initial < 0.0:
        raise ValueError("gate leak must not be negative")
    if not 0.0 < args.gate_leak_anneal_fraction <= 1.0:
        raise ValueError("gate leak anneal fraction must be within (0, 1]")
    support_switch_update = max(
        1, int(round(args.steps * args.support_switch_fraction)))
    leak_anneal_updates = max(
        1, int(round(args.steps * args.gate_leak_anneal_fraction)))
    if not 0.0 <= args.gate_warmup_fraction < 1.0:
        raise ValueError("gate warmup fraction must be within [0, 1)")
    gate_warmup_updates = int(round(args.steps * args.gate_warmup_fraction))
    new_task = args.new_task
    if args.new_independent_event_updates < 0:
        raise ValueError(
            "independent-event update count must not be negative")
    if (
            args.new_independent_events
            and (
                args.new_independent_event_updates
                or args.new_independent_event_share)):
        raise ValueError(
            "choose either all-independent training or a staged independent "
            "configuration")
    if (
            args.new_independent_event_updates
            and args.new_independent_event_share):
        raise ValueError(
            "staged and within-batch independent-event curricula are "
            "mutually exclusive")
    if not 0.0 <= args.new_independent_event_share < 1.0:
        raise ValueError(
            "independent-event share must lie within [0, 1)")
    if (
            args.new_independent_action_augmentation
            and not (
                args.new_independent_events
                or args.new_independent_event_updates
                or args.new_independent_event_share)):
        raise ValueError(
            "independent action augmentation requires an independent-event "
            "training mode")
    if args.new_independent_event_updates >= args.steps:
        raise ValueError(
            "a staged independent-event phase must leave at least one "
            "recurrent consolidation update")
    if (
            (
                args.new_independent_events
                or args.new_independent_event_updates
                or args.new_independent_event_share)
            and new_task != "visible_pair_numerosity_operation"):
        raise ValueError(
            "independent-event training currently requires the fully visible "
            "conditional numerosity operation")
    if not 0.0 <= args.new_numerosity_appearance_blend <= 1.0:
        raise ValueError("new numerosity appearance blend must be within [0, 1]")
    if not 0.0 <= args.new_operation_cue_scale <= 1.0:
        raise ValueError("new operation cue scale must be within [0, 1]")
    if not 0 <= args.new_operation_cue_trials <= 6:
        raise ValueError("new operation cue trials must be between zero and 6")
    new_operation_cue_trials = (
        None
        if args.new_operation_cue_trials == 0
        else args.new_operation_cue_trials)
    if not 0.0 <= args.replay_numerosity_appearance_blend <= 1.0:
        raise ValueError(
            "replay numerosity appearance blend must be within [0, 1]")
    if (
            new_task not in (
                "visible_pair_numerosity",
                "visible_pair_numerosity_smaller",
                "visible_pair_numerosity_operation",
                "visible_numerosity_equality")
            and args.new_numerosity_appearance_blend != 1.0):
        raise ValueError(
            "numerosity appearance blend requires a numerosity new task")
    replay_tasks = tuple(
        name for name in args.replay_tasks.split(",") if name)
    if not replay_tasks:
        raise ValueError("a rung must replay at least one earlier primitive")
    if new_task in replay_tasks:
        raise ValueError("the new primitive cannot also be a replay task")
    if args.replay_support_trials:
        replay_support = tuple(
            int(value) for value in args.replay_support_trials.split(","))
        if len(replay_support) != len(replay_tasks):
            raise ValueError(
                "replay support counts must match the replay task count")
        if any(not 1 <= value < 6 for value in replay_support):
            raise ValueError("replay support counts must be between 1 and 5")
    else:
        replay_support = tuple(1 for _ in replay_tasks)
    replay_support_by_task = dict(zip(replay_tasks, replay_support))
    seeds = (
        [int(v) for v in args.seeds.split(",") if v] if args.seeds
        else [args.seed])
    steps_grid = (
        [int(v) for v in args.steps_grid.split(",") if v] if args.steps_grid
        else [args.steps])
    if len(steps_grid) > 1 and "{steps}" not in str(args.report):
        raise ValueError(
            "--report must contain {steps} when a budget grid is requested")
    if len(seeds) > 1 and "{seed}" not in str(args.report):
        raise ValueError(
            "--report must contain {seed} when several seeds are requested")

    device = torch.device(args.device)
    summaries = []
    parent_audit_cache: dict[tuple[int, str], dict] = {}
    for seed, steps in ((s, b) for s in seeds for b in steps_grid):
        args.steps = steps
        support_switch_update = max(
            1, int(round(steps * args.support_switch_fraction)))
        leak_anneal_updates = max(
            1, int(round(steps * args.gate_leak_anneal_fraction)))
        gate_warmup_updates = int(round(steps * args.gate_warmup_fraction))
        report_path = Path(
            str(args.report).format(seed=seed, steps=steps))
        checkpoint_out = (
            None if args.checkpoint_out is None
            else Path(str(args.checkpoint_out).format(
                seed=seed, steps=steps)))
        seed_everything(seed)
        payload, teacher = _load(args.parent, device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        configuration = dict(payload["model_configuration"])
        inherited_slots = tuple(configuration.get("skill_adapter_widths", ()))
        new_slot = len(inherited_slots)
        plastic_prefixes = _plastic_prefixes(new_slot)
        thawed_prefixes = (
            _prior_slot_prefixes(new_slot, args.volatile_prior_slots)
            if args.prior_slot_volatility > 0.0 else ())
        unit_thawed_names = (
            {
                f"skill_adapters.{new_slot - 1}.0.weight",
                f"skill_adapters.{new_slot - 1}.0.bias",
            }
            if args.unit_volatility_policy != "none" and new_slot > 0
            else set())
        if args.unit_volatility_policy != "none" and not unit_thawed_names:
            raise ValueError("unit volatility requires an inherited skill slot")
        trainable_prefixes = plastic_prefixes + thawed_prefixes
        def is_trainable(name: str) -> bool:
            return (
                name.startswith(trainable_prefixes)
                or name in unit_thawed_names)
        configuration["skill_adapter_widths"] = (
            inherited_slots + (args.skill_adapter_width,))
        configuration["action_adapter_into_intention"] = (
            bool(configuration.get("action_adapter_into_intention", False))
            or args.canonicalize_action_adapter)
        configuration["skill_adapter_gate_mode"] = args.slot_gate_mode
        inherited_gate_hidden = int(
            configuration.get("skill_adapter_gate_hidden", 0))
        inherited_gate_hidden_from = configuration.get(
            "skill_adapter_gate_hidden_from")
        if inherited_gate_hidden and args.slot_gate_hidden != inherited_gate_hidden:
            raise ValueError(
                "an inherited hidden gate must keep its established width")
        configuration["skill_adapter_gate_hidden"] = args.slot_gate_hidden
        configuration["skill_adapter_gate_hidden_from"] = (
            inherited_gate_hidden_from
            if inherited_gate_hidden_from is not None
            else new_slot if args.slot_gate_hidden else None)
        inherited_reads_prior = bool(
            configuration.get("skill_adapter_reads_prior", False))
        inherited_reads_prior_from = configuration.get(
            "skill_adapter_reads_prior_from")
        if inherited_reads_prior and not args.slot_reads_prior:
            raise ValueError(
                "a readable parent must append a readable slot; use "
                "--ablate-prior-read for the matched no-content control")
        configuration["skill_adapter_reads_prior"] = (
            inherited_reads_prior or args.slot_reads_prior)
        configuration["skill_adapter_read_bottleneck"] = args.read_bottleneck
        configuration["skill_adapter_prior_read_limit"] = args.prior_read_limit
        configuration["skill_adapter_reads_prior_from"] = (
            inherited_reads_prior_from
            if inherited_reads_prior_from is not None
            else new_slot if args.slot_reads_prior else None)
        inherited_legacy_read_from = configuration.get(
            "skill_adapter_legacy_read_from")
        configuration["skill_adapter_legacy_read_from"] = (
            inherited_legacy_read_from
            if inherited_legacy_read_from is not None
            else new_slot if args.read_legacy_adapters else None)
        inherited_intention_read_from = configuration.get(
            "skill_adapter_reads_intention_from")
        if (
                inherited_intention_read_from is not None
                and not args.read_parent_intention):
            raise ValueError(
                "an intention-reading parent must append an intention-reading "
                "slot; use --ablate-prior-read for the no-content control")
        configuration["skill_adapter_reads_intention_from"] = (
            inherited_intention_read_from
            if inherited_intention_read_from is not None
            else new_slot if args.read_parent_intention else None)
        inherited_event_snapshot_read_from = configuration.get(
            "skill_adapter_reads_event_snapshot_from")
        if (
                inherited_event_snapshot_read_from is not None
                and not args.read_event_snapshot):
            raise ValueError(
                "an event-snapshot-reading parent must append the same "
                "generic sensory RAM interface")
        configuration["skill_adapter_reads_event_snapshot_from"] = (
            inherited_event_snapshot_read_from
            if inherited_event_snapshot_read_from is not None
            else new_slot if args.read_event_snapshot else None)
        inherited_multiplicative_read_from = configuration.get(
            "skill_adapter_multiplies_intention_from")
        if (
                inherited_multiplicative_read_from is not None
                and not args.multiply_parent_intention):
            raise ValueError(
                "a multiplicative-intention parent must append the same "
                "generic binding interface")
        if args.multiply_parent_intention and not args.read_parent_intention:
            raise ValueError(
                "multiplicative intention binding requires "
                "--read-parent-intention")
        if (
                args.outer_product_parent_intention
                and (
                    not args.read_parent_intention
                    or args.outer_product_width < 1)):
            raise ValueError(
                "outer-product intention binding requires "
                "--read-parent-intention and positive width")
        if (
                args.ablate_outer_product
                and not args.outer_product_parent_intention):
            raise ValueError(
                "outer-product ablation requires "
                "--outer-product-parent-intention")
        if args.ablate_event_snapshot and not args.read_event_snapshot:
            raise ValueError(
                "event-snapshot ablation requires --read-event-snapshot")
        configuration["skill_adapter_multiplies_intention_from"] = (
            inherited_multiplicative_read_from
            if inherited_multiplicative_read_from is not None
            else new_slot if args.multiply_parent_intention else None)
        inherited_outer_read_from = configuration.get(
            "skill_adapter_outer_multiplies_intention_from")
        if (
                inherited_outer_read_from is not None
                and not args.outer_product_parent_intention):
            raise ValueError(
                "an outer-product parent must append the same generic "
                "binding interface")
        inherited_outer_width = int(
            configuration.get("skill_adapter_outer_interaction_width", 0))
        if (
                inherited_outer_width
                and args.outer_product_width != inherited_outer_width):
            raise ValueError(
                "an inherited outer-product width cannot change")
        configuration["skill_adapter_outer_multiplies_intention_from"] = (
            inherited_outer_read_from
            if inherited_outer_read_from is not None
            else new_slot if args.outer_product_parent_intention else None)
        configuration["skill_adapter_outer_interaction_width"] = (
            inherited_outer_width
            if inherited_outer_width
            else (
                args.outer_product_width
                if args.outer_product_parent_intention else 0))
        student = UnifiedCognitiveController(**configuration).to(device)
        # Remove inherited content only from the newly appended slot. A global
        # ablation would also alter readable parent slots and make the
        # supposedly matched control start from a different controller.
        student.skill_adapter_ablate_prior_read = False
        student.skill_adapter_ablate_prior_read_slot = (
            new_slot if args.ablate_prior_read else None)
        student.skill_adapter_ablate_event_snapshot_slot = (
            new_slot if args.ablate_event_snapshot else None)
        student.skill_adapter_ablate_outer_interaction_slot = (
            new_slot if args.ablate_outer_product else None)
        missing, unexpected = student.load_state_dict(
            teacher.state_dict(), strict=False)
        expected_missing = {
            name for name in student.state_dict()
            if name.startswith(plastic_prefixes)}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"unexpected modular insertion mismatch: "
                f"missing={missing}, unexpected={unexpected}")
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(is_trainable(name))
        inherited_frozen_adapters = sorted({
            name.split(".")[0] for name in teacher.state_dict()
            if "adapter" in name})
        frozen_initial = {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()
            if not is_trainable(name)
        }
        thawed_initial = {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()
            if name.startswith(thawed_prefixes) or name in unit_thawed_names
        }
        plastic_parameters = sum(
            parameter.numel() for name, parameter in student.named_parameters()
            if is_trainable(name))
        frozen_parameters = sum(
            parameter.numel() for name, parameter in student.named_parameters()
            if not is_trainable(name))
        new_parameters = [
            parameter for name, parameter in student.named_parameters()
            if name.startswith(plastic_prefixes)]
        parameter_groups = [{
            "params": new_parameters,
            "lr": args.learning_rate,
        }]
        if thawed_prefixes:
            thawed_parameters = [
                parameter for name, parameter in student.named_parameters()
                if name.startswith(thawed_prefixes)]
            parameter_groups.append({
                "params": thawed_parameters,
                "lr": args.learning_rate * args.prior_slot_volatility,
            })
        else:
            thawed_parameters = []
        unit_thawed_parameters = [
            parameter for name, parameter in student.named_parameters()
            if name in unit_thawed_names]
        if unit_thawed_parameters:
            parameter_groups.append({
                "params": unit_thawed_parameters,
                "lr": args.learning_rate,
            })
        optimizer = torch.optim.AdamW(
            parameter_groups, lr=args.learning_rate, weight_decay=1e-5)
        initial_group_lrs = [
            float(group["lr"]) for group in optimizer.param_groups]
        unit_layer = (
            student.skill_adapters[new_slot - 1][0]
            if unit_thawed_parameters else None)
        if unit_layer is not None and not isinstance(unit_layer, nn.Linear):
            raise TypeError("skill adapter hidden projection must be linear")
        unit_importance = (
            torch.zeros(unit_layer.out_features, device=device)
            if unit_layer is not None else None)
        current_unit_volatility = (
            torch.full(
                (unit_layer.out_features,), args.unit_volatility_max,
                device=device)
            if unit_layer is not None else None)

        # A retention gate is only required where the frozen parent already had
        # the skill, so arms with different histories stay comparable.
        # Evaluated on exactly the seeds the student's retention will use. A set
        # point measured on different lifetimes than the outcome would make the
        # per-rung degradation a difference of two noisy numbers, and evaluation
        # noise here is the same size as the effect being measured.
        # The parent is frozen and its audit does not depend on this rung's
        # budget, so it is computed once per seed and reused across the grid.
        # Curve mode never needs it at all.
        if args.eval_mode == "curve":
            parent_evaluations = {}
        else:
            parent_evaluations = {}
            for index, task in enumerate(replay_tasks):
                key = (seed, task)
                if key not in parent_audit_cache:
                    parent_audit_cache[key] = evaluate(
                        teacher, count=args.test_lifetimes, trials=6,
                        seed=seed + 91_000_000 + index, device=device,
                        task=task,
                        feedback_trials=replay_support_by_task[task],
                        numerosity_appearance_blend=(
                            args.replay_numerosity_appearance_blend
                            if task == "visible_pair_numerosity" else 1.0))
                parent_evaluations[task] = parent_audit_cache[key]
        parent_retention = {
            task: evaluation["gate"]["accepted"]
            for task, evaluation in parent_evaluations.items()
        }
        if args.eval_mode == "curve":
            parent_retention = {task: False for task in replay_tasks}
        # The level each old skill arrives at is the level it should leave at. These
        # are the controller's set points; nothing here reaches the learner, which
        # still sees only frames, its own actions, and scalar outcomes.
        retention_set_point = {
            task: _headline_accuracy(evaluation)
            for task, evaluation in parent_evaluations.items()
        }
        if args.eval_mode == "curve":
            # No parent audit was run, so there is no set point to hold. A zero
            # target leaves the controller inert at its base weight rather than
            # silently steering against a missing measurement.
            retention_set_point = {task: 0.0 for task in replay_tasks}
        tracked_accuracy = dict(retention_set_point)

        started = time.perf_counter()
        history: list[dict[str, float | int]] = []
        tracked_new_accuracy = 0.0
        tracked_gradient_alignment = 0.0
        current_volatility = args.prior_slot_volatility
        volatility_sum = 0.0
        selected_replay = {task: True for task in replay_tasks}
        replay_batches_spent = 0
        for update in range(1, args.steps + 1):
            student.train()
            if args.final_learning_rate is not None:
                decay_start = max(
                    1, int(round(
                        args.steps * args.learning_rate_decay_start)))
                progress = min(
                    1.0,
                    max(0.0, (update - decay_start)
                        / max(1, args.steps - decay_start)))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                ratio = (
                    args.final_learning_rate / args.learning_rate
                    + (1.0 - args.final_learning_rate / args.learning_rate)
                    * cosine)
                for group, initial_lr in zip(
                        optimizer.param_groups, initial_group_lrs):
                    group["lr"] = initial_lr * ratio
            # Anneal the rectifier's leak linearly to exactly zero, then hold it
            # there for the rest of the rung so the gate hardens into a gate that
            # can be exactly shut.
            student.skill_adapter_gate_leak = (
                args.gate_leak_initial
                * max(0.0, 1.0 - (update - 1) / leak_anneal_updates))
            # Hold the gate open until the adapter is worth gating.
            for name, parameter in student.named_parameters():
                if name.startswith(f"skill_adapter_gates.{new_slot}."):
                    parameter.requires_grad_(update > gate_warmup_updates)
            support_trials = (
                args.new_support_trials if update < support_switch_update
                else final_support_trials)
            new_batch = generate_lifetimes(
                args.new_batch_size, 6,
                seed=seed * 10_000_000 + update,
                task=new_task,
                numerosity_appearance_blend=(
                    args.new_numerosity_appearance_blend),
                operation_cue_scale=args.new_operation_cue_scale,
                operation_cue_trials=new_operation_cue_trials,
                operation_cue_prestimulus=(
                    args.new_operation_cue_prestimulus),
                position_holdout=(
                    bool(update % 2)
                    if args.new_position_augmentation else None),
                support_trials=support_trials,
                device=device)
            if args.shuffle_new_verifier_outcomes:
                new_batch = _shuffle_verifier_outcomes(
                    new_batch,
                    seed=seed * 70_000_000 + update)
            if (args.replay_selection == "gate"
                    and update > gate_warmup_updates
                    and (update - 1) % args.replay_selection_every == 0):
                # A slot exactly shut on a skill's events cannot perturb it, so
                # replaying that skill is buying a guarantee already held.
                student.eval()
                for task in replay_tasks:
                    _, shut = _slot_opening(
                        student, slot=new_slot, task=task, count=64,
                        seed=seed + 95_000_000 + hash(task) % 1000,
                        support_trials=replay_support_by_task[task],
                        device=device)
                    selected_replay[task] = shut < args.replay_gate_threshold
                student.train()
            active_replay = [
                (index, task) for index, task in enumerate(replay_tasks)
                if selected_replay[task]]
            replay_batches = [
                generate_lifetimes(
                    args.replay_batch_size, 6,
                    seed=seed * (20_000_000 + 10_000_000 * index) + update,
                    task=task, support_trials=replay_support_by_task[task],
                    appearance=_replay_appearance(
                        task, args.pair_relation_replay_appearance, update),
                    numerosity_appearance_blend=(
                        args.replay_numerosity_appearance_blend
                        if task == "visible_pair_numerosity" else 1.0),
                    device=device)
                for index, task in active_replay
            ]
            replay_batches_spent += len(replay_batches)
            skill_loss, new_batch_accuracy = _new_skill_loss(
                student, new_batch, exploration=args.exploration,
                support_trials=support_trials,
                learning_rule=args.new_learning_rule,
                independent_events=(
                    args.new_independent_events
                    or update <= args.new_independent_event_updates),
                independent_event_share=args.new_independent_event_share,
                independent_action_augmentation=(
                    args.new_independent_action_augmentation),
                return_accuracy=True)
            tracked_new_accuracy = (
                0.9 * tracked_new_accuracy + 0.1 * new_batch_accuracy
                if update > 1 else new_batch_accuracy)
            replay_results = [
                _replay_loss_and_leakage(
                    student, teacher, batch, slot=new_slot,
                    feedback_trials=replay_support_by_task[task],
                    shuffled_teacher=args.shuffle_retention_teacher)
                for (_, task), batch in zip(active_replay, replay_batches)
            ]
            replay_losses = [value for value, _, _ in replay_results]
            leakages = [value for _, value, _ in replay_results]
            leakage = (
                torch.stack(leakages).mean() if leakages
                else torch.zeros((), device=device))
            # Proportional set-point control on the retention price. A skill at or
            # above the level it was inherited at costs nothing extra, so pressure
            # never competes with new learning unless a skill is actually slipping.
            weights = []
            deficits = []
            active_tasks = [task for _, task in active_replay]
            for position, task in enumerate(active_tasks):
                measured = replay_results[position][2]
                tracked_accuracy[task] = (
                    args.retention_tracking_decay * tracked_accuracy[task]
                    + (1.0 - args.retention_tracking_decay) * measured)
                deficit = max(
                    0.0, retention_set_point[task] - tracked_accuracy[task])
                deficits.append(deficit)
                weights.append(
                    args.retention_weight + args.retention_control_gain * deficit)
            retention_loss = sum(
                weight * value for weight, value in zip(weights, replay_losses))
            loss = skill_loss + retention_loss
            if args.locality_weight:
                # Price opening the new slot on events that belong to old skills.
                loss = loss + args.locality_weight * leakage
            gradient_alignment = 0.0
            if (
                    thawed_parameters
                    and args.volatility_policy == "gradient_alignment"):
                gradient_alignment = _gradient_alignment(
                    skill_loss, retention_loss, thawed_parameters)
                tracked_gradient_alignment = (
                    args.volatility_alignment_decay
                    * tracked_gradient_alignment
                    + (1.0 - args.volatility_alignment_decay)
                    * gradient_alignment
                    if update > 1 else gradient_alignment)
                current_volatility = _alignment_volatility(
                    tracked_gradient_alignment,
                    args.prior_slot_volatility)
                optimizer.param_groups[1]["lr"] = (
                    args.learning_rate * current_volatility)
            unit_before: tuple[torch.Tensor, torch.Tensor] | None = None
            if unit_layer is not None:
                assert unit_importance is not None
                retention_grads = torch.autograd.grad(
                    retention_loss,
                    (unit_layer.weight, unit_layer.bias),
                    retain_graph=True, allow_unused=True)
                weight_grad, bias_grad = retention_grads
                observed_importance = torch.zeros_like(unit_importance)
                if weight_grad is not None:
                    observed_importance.add_(
                        weight_grad.detach().square().mean(dim=1))
                if bias_grad is not None:
                    observed_importance.add_(
                        bias_grad.detach().square())
                unit_importance.mul_(args.unit_importance_decay).add_(
                    observed_importance,
                    alpha=1.0 - args.unit_importance_decay)
                current_unit_volatility = _importance_volatility(
                    unit_importance, args.unit_volatility_max,
                    args.unit_importance_strength)
                if args.unit_volatility_policy == "shuffled_importance":
                    current_unit_volatility = current_unit_volatility.roll(1)
                current_volatility = float(
                    current_unit_volatility.mean())
                unit_before = (
                    unit_layer.weight.detach().clone(),
                    unit_layer.bias.detach().clone())
            volatility_sum += current_volatility
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if unit_before is not None:
                assert current_unit_volatility is not None
                _blend_unit_update(
                    unit_layer.weight, unit_before[0],
                    current_unit_volatility)
                _blend_unit_update(
                    unit_layer.bias, unit_before[1],
                    current_unit_volatility)
            if (
                    update in (1, args.steps)
                    or (args.telemetry_every
                        and update % args.telemetry_every == 0)):
                history.append({
                    "update": update,
                    "skill_loss": float(skill_loss.detach()),
                    "new_batch_accuracy": new_batch_accuracy,
                    "tracked_new_accuracy": tracked_new_accuracy,
                    "gradient_alignment": gradient_alignment,
                    "tracked_gradient_alignment": tracked_gradient_alignment,
                    "prior_slot_volatility": current_volatility,
                    "unit_volatility_min": (
                        float(current_unit_volatility.min())
                        if current_unit_volatility is not None else 0.0),
                    "unit_volatility_max": (
                        float(current_unit_volatility.max())
                        if current_unit_volatility is not None else 0.0),
                    **{
                        f"{task}_distillation_loss": float(value.detach())
                        for task, value in zip(active_tasks, replay_losses)
                    },
                    **{
                        f"{task}_slot_opening": float(value.detach())
                        for task, value in zip(active_tasks, leakages)
                    },
                    **{
                        f"{task}_retention_weight": weight
                        for task, weight in zip(active_tasks, weights)
                    },
                    **{
                        f"{task}_retention_deficit": deficit
                        for task, deficit in zip(active_tasks, deficits)
                    },
                    "replay_slot_opening": float(leakage.detach()),
                    "total_loss": float(loss.detach()),
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]),
                })

        # Nothing is measured through a leaky gate: the exact-zero property is the
        # whole mechanism, so the anneal must be finished before any evaluation.
        student.skill_adapter_gate_leak = 0.0
        if args.eval_mode == "curve":
            # A budget sweep's interior points feed a learning curve and
            # nothing else. Retention still gets one rollout each so a
            # catastrophe cannot hide behind a cheap measurement.
            headline = {
                "new_skill": _curve_accuracy(
                    student, task=new_task, count=args.test_lifetimes,
                    seed=seed + 90_000_000,
                    support_trials=final_support_trials, device=device,
                    numerosity_appearance_blend=(
                        args.new_numerosity_appearance_blend),
                    operation_cue_scale=args.new_operation_cue_scale,
                    operation_cue_trials=new_operation_cue_trials,
                    operation_cue_prestimulus=(
                        args.new_operation_cue_prestimulus)),
                **{
                    f"{task}_retention": _curve_accuracy(
                        student, task=task, count=args.test_lifetimes,
                        seed=seed + 91_000_000 + index,
                        support_trials=replay_support_by_task[task],
                        device=device,
                        numerosity_appearance_blend=(
                            args.replay_numerosity_appearance_blend
                            if task == "visible_pair_numerosity" else 1.0))
                    for index, task in enumerate(replay_tasks)
                },
            }
            if args.new_operation_cue_prestimulus:
                # Preserve the extra recurrent timestep while removing only
                # the operation symbol.  Comparing this against ``new_skill``
                # rules out a timing/presence shortcut that the older
                # no-prestimulus control could not distinguish.
                headline["new_skill_blank_prestimulus"] = _curve_accuracy(
                    student, task=new_task, count=args.test_lifetimes,
                    seed=seed + 90_000_000,
                    support_trials=final_support_trials, device=device,
                    numerosity_appearance_blend=(
                        args.new_numerosity_appearance_blend),
                    operation_cue_scale=0.0,
                    operation_cue_trials=new_operation_cue_trials,
                    operation_cue_prestimulus=True)
                headline["new_skill_without_operation_cue"] = _curve_accuracy(
                    student, task=new_task, count=args.test_lifetimes,
                    seed=seed + 90_000_000,
                    support_trials=final_support_trials, device=device,
                    numerosity_appearance_blend=(
                        args.new_numerosity_appearance_blend),
                    operation_cue_scale=args.new_operation_cue_scale,
                    operation_cue_trials=new_operation_cue_trials,
                    operation_cue_prestimulus=False)
            operation_counterfactual = (
                _operation_counterfactual_metrics(
                    student, count=args.test_lifetimes,
                    seed=seed + 92_000_000, device=device,
                    numerosity_appearance_blend=(
                        args.new_numerosity_appearance_blend))
                if new_task == "visible_pair_numerosity_operation"
                else None)
            blank_operation_counterfactual = (
                _operation_counterfactual_metrics(
                    student, count=args.test_lifetimes,
                    seed=seed + 92_000_000, device=device,
                    numerosity_appearance_blend=(
                        args.new_numerosity_appearance_blend),
                    operation_cue_scale=0.0)
                if new_task == "visible_pair_numerosity_operation"
                else None)
            report = {
                "schema": "fourth-primitive-transfer-curve-v1",
                "eval_mode": "curve",
                "new_task": new_task,
                "replay_tasks": list(replay_tasks),
                "plastic_module": f"skill_adapters.{new_slot}",
                "thawed_prior_prefixes": list(thawed_prefixes),
                "unit_thawed_parameter_names": sorted(unit_thawed_names),
                "thawed_prior_relative_drift": _relative_state_drift(
                    student, thawed_initial),
                "mean_prior_slot_volatility": (
                    volatility_sum / max(1, args.steps)),
                "final_unit_volatility": (
                    current_unit_volatility.detach().cpu().tolist()
                    if current_unit_volatility is not None else []),
                "configuration": {
                    **vars(args), "seed": seed,
                    "parent": str(args.parent), "report": str(report_path),
                    "checkpoint_out": (
                        str(checkpoint_out)
                        if checkpoint_out is not None else None)},
                "history": history,
                "accounting": {
                    "new_unique_lifetimes": args.steps * args.new_batch_size,
                    "new_verifier_bits":
                        args.steps * args.new_batch_size * 6,
                    "optimizer_updates": args.steps,
                    "sensory_frames_per_action": (
                        2 if args.new_operation_cue_prestimulus else 1),
                    "new_sensory_frames": (
                        args.steps * args.new_batch_size * 6
                        * (
                            2
                            if args.new_operation_cue_prestimulus else 1)),
                },
                "headline_accuracy": headline,
                "operation_counterfactual": operation_counterfactual,
                "blank_operation_counterfactual":
                    blank_operation_counterfactual,
                "frozen_base_bit_identical": all(
                    torch.equal(frozen_initial[name], value.detach().cpu())
                    for name, value in student.state_dict().items()
                    if name in frozen_initial),
                "checkpoint_saved": checkpoint_out is not None,
                "total_seconds": time.perf_counter() - started,
            }
            if checkpoint_out is not None:
                checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema": "unified-cognitive-controller-v1",
                    "model_configuration": configuration,
                    "state_dict": student.state_dict(),
                    "source_report": str(report_path),
                    # Curve-mode candidates have not passed the >=90% mastery
                    # gates and must never be mistaken for promoted parents.
                    "admission_status":
                        "unpromoted-causal-operation-research-candidate",
                }, checkpoint_out)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            summaries.append(report_path)
            print(json.dumps({
                "seed": seed, "steps": args.steps,
                "new_skill": headline["new_skill"],
                "total_seconds": report["total_seconds"]}, sort_keys=True))
            continue
        evaluations = {
            "new_skill": evaluate(
                student, count=args.test_lifetimes, trials=6,
                seed=seed + 90_000_000, device=device,
                task=new_task, feedback_trials=final_support_trials,
                numerosity_appearance_blend=(
                    args.new_numerosity_appearance_blend),
                operation_cue_scale=args.new_operation_cue_scale,
                operation_cue_trials=new_operation_cue_trials,
                operation_cue_prestimulus=(
                    args.new_operation_cue_prestimulus)),
            **{
                f"{task}_retention": evaluate(
                    student, count=args.test_lifetimes, trials=6,
                    seed=seed + 91_000_000 + index, device=device,
                    task=task, feedback_trials=replay_support_by_task[task],
                    numerosity_appearance_blend=(
                        args.replay_numerosity_appearance_blend
                        if task == "visible_pair_numerosity" else 1.0))
                for index, task in enumerate(replay_tasks)
            },
        }
        relation_repertoire_evaluations = {}
        if (
                "pair_relation" in replay_tasks
                and args.pair_relation_replay_appearance == "cycle"):
            relation_repertoire_evaluations = {
                appearance: evaluate(
                    student, count=args.test_lifetimes, trials=6,
                    seed=seed + 92_000_000 + 10_000 * index,
                    device=device, task="pair_relation",
                    feedback_trials=replay_support_by_task["pair_relation"],
                    appearance=appearance)
                for index, appearance in enumerate(
                    PAIR_RELATION_APPEARANCES)
            }
        cue_ablation_accuracy = _operation_cue_ablation_accuracy(
            student, count=args.test_lifetimes,
            seed=seed + 90_000_000, device=device,
            support_trials=final_support_trials, new_task=new_task,
            numerosity_appearance_blend=(
                args.new_numerosity_appearance_blend),
            operation_cue_scale=args.new_operation_cue_scale,
            operation_cue_trials=new_operation_cue_trials,
            operation_cue_prestimulus=(
                args.new_operation_cue_prestimulus))
        _openings = {
            new_task: _slot_opening(
                student, slot=new_slot, task=new_task,
                count=args.test_lifetimes, seed=seed + 93_000_000,
                support_trials=final_support_trials, device=device,
                numerosity_appearance_blend=(
                    args.new_numerosity_appearance_blend),
                operation_cue_scale=args.new_operation_cue_scale,
                operation_cue_trials=new_operation_cue_trials,
                operation_cue_prestimulus=(
                    args.new_operation_cue_prestimulus)),
            **{
                task: _slot_opening(
                    student, slot=new_slot, task=task,
                    count=args.test_lifetimes,
                    seed=seed + 94_000_000 + index,
                    support_trials=replay_support_by_task[task], device=device,
                    numerosity_appearance_blend=(
                        args.replay_numerosity_appearance_blend
                        if task == "visible_pair_numerosity" else 1.0))
                for index, task in enumerate(replay_tasks)
            },
        }
        slot_opening = {task: value for task, (value, _) in _openings.items()}
        slot_shut_fraction = {
            task: shut for task, (_, shut) in _openings.items()}
        slot_residual_norm = {
            new_task: _slot_residual_norm(
                student, slot=new_slot, task=new_task,
                count=args.test_lifetimes, seed=seed + 93_000_000,
                support_trials=final_support_trials, device=device,
                numerosity_appearance_blend=(
                    args.new_numerosity_appearance_blend),
                operation_cue_scale=args.new_operation_cue_scale,
                operation_cue_trials=new_operation_cue_trials,
                operation_cue_prestimulus=(
                    args.new_operation_cue_prestimulus)),
            **{
                task: _slot_residual_norm(
                    student, slot=new_slot, task=task,
                    count=args.test_lifetimes,
                    seed=seed + 94_000_000 + index,
                    support_trials=replay_support_by_task[task], device=device,
                    numerosity_appearance_blend=(
                        args.replay_numerosity_appearance_blend
                        if task == "visible_pair_numerosity" else 1.0))
                for index, task in enumerate(replay_tasks)
            },
        }
        new_skill_accuracy = _headline_accuracy(evaluations["new_skill"])
        cue_causally_used = cue_ablation_accuracy <= new_skill_accuracy - 0.15
        required_retention = {
            f"{task}_retention": parent_retention[task] for task in replay_tasks}
        retention_gates_passed = all(
            evaluations[name]["gate"]["accepted"]
            for name, required in required_retention.items() if required)
        relation_repertoire_retained = (
            not relation_repertoire_evaluations
            or all(
                evaluation["gate"]["accepted"]
                for evaluation in relation_repertoire_evaluations.values()))
        accepted = (
            evaluations["new_skill"]["gate"]["accepted"]
            and retention_gates_passed
            and relation_repertoire_retained
            and cue_causally_used
            and slot_shut_fraction[new_task] < 1.0)
        frozen_base_identical = all(
            torch.equal(frozen_initial[name], value.detach().cpu())
            for name, value in student.state_dict().items()
            if name in frozen_initial)
        total_lifetimes = args.steps * (
            args.new_batch_size + len(replay_tasks) * args.replay_batch_size)
        total_verifier_bits = total_lifetimes * 6
        report = {
            "schema": "fourth-primitive-transfer-v1",
            "claim_boundary": (
                "New behavior learned from attempted-action verifier outcomes; "
                "retention uses only a frozen learned controller's opaque action "
                "distributions on newly rendered replay lifetimes. Retention "
                "gates are required only for skills the frozen parent had."),
            "semantic_labels_used_for_training": False,
            "unattempted_correct_actions_used_as_targets": False,
            "new_task": new_task,
            "replay_tasks": list(replay_tasks),
            "replay_support_trials": replay_support_by_task,
            "plastic_module": f"skill_adapters.{new_slot}",
            "thawed_prior_prefixes": list(thawed_prefixes),
            "unit_thawed_parameter_names": sorted(unit_thawed_names),
            "thawed_prior_relative_drift": _relative_state_drift(
                student, thawed_initial),
            "mean_prior_slot_volatility": (
                volatility_sum / max(1, args.steps)),
            "final_unit_volatility": (
                current_unit_volatility.detach().cpu().tolist()
                if current_unit_volatility is not None else []),
            "inherited_frozen_adapters": inherited_frozen_adapters,
            "inherited_skill_adapter_slots": list(inherited_slots),
            "plastic_parameters": plastic_parameters,
            "frozen_parameters": frozen_parameters,
            "configuration": {
                **vars(args),
                "parent": str(args.parent),
                "report": str(report_path),
                "checkpoint_out": (
                    str(checkpoint_out)
                    if checkpoint_out is not None else None),
            },
            "history": history,
            "replay_selection": {
                "mode": args.replay_selection,
                "final_selection": dict(selected_replay),
                "replay_batches_spent": replay_batches_spent,
                "replay_batches_if_all": args.steps * len(replay_tasks),
                "replay_fraction_spent": (
                    replay_batches_spent / max(1, args.steps * len(replay_tasks))),
            },
            "accounting": {
                "new_unique_lifetimes": args.steps * args.new_batch_size,
                "new_verifier_bits":
                    args.steps * args.new_batch_size * 6,
                "sensory_frames_per_action": (
                    2 if args.new_operation_cue_prestimulus else 1),
                "new_sensory_frames": (
                    args.steps * args.new_batch_size * 6
                    * (2 if args.new_operation_cue_prestimulus else 1)),
                "replay_lifetimes_per_task": args.steps * args.replay_batch_size,
                "total_unique_lifetimes": total_lifetimes,
                "total_verifier_bits": total_verifier_bits,
                "optimizer_updates": args.steps,
            },
            "parent_retention_gates": parent_retention,
            "retention_set_point": retention_set_point,
            "retention_delta_against_set_point": {
                task: (
                    _headline_accuracy(evaluations[f"{task}_retention"])
                    - retention_set_point[task])
                for task in replay_tasks
            },
            "final_tracked_replay_accuracy": dict(tracked_accuracy),
            "required_retention_gates": required_retention,
            "evaluations": evaluations,
            "pair_relation_repertoire_retention":
                relation_repertoire_evaluations,
            "pair_relation_repertoire_retained":
                relation_repertoire_retained,
            "headline_accuracy": {
                "new_skill": new_skill_accuracy,
                **{
                    f"{task}_retention": _headline_accuracy(
                        evaluations[f"{task}_retention"])
                    for task in replay_tasks
                },
                **{
                    f"pair_relation_{appearance}_retention":
                        _headline_accuracy(evaluation)
                    for appearance, evaluation
                    in relation_repertoire_evaluations.items()
                },
            },
            "slot_opening": slot_opening,
            "slot_shut_fraction": slot_shut_fraction,
            # A slot shut on every one of its own task's events learned nothing and
            # can no longer recover: the rectifier has no gradient below zero. It
            # is a training failure, not a result, and must never be read as
            # perfect retention.
            "slot_dead": slot_shut_fraction[new_task] >= 1.0,
            "support_schedule": {
                "initial": args.new_support_trials,
                "final": final_support_trials,
                "switch_update": support_switch_update,
            },
            "gate_warmup_updates": gate_warmup_updates,
            "gate_leak_schedule": {
                "initial": args.gate_leak_initial,
                "anneal_updates": leak_anneal_updates,
                "leak_at_evaluation": student.skill_adapter_gate_leak,
            },
            "slot_residual_norm": slot_residual_norm,
            "slot_selectivity": (
                slot_opening[new_task]
                - max(slot_opening[task] for task in replay_tasks)),
            "operation_cue_ablation_accuracy": cue_ablation_accuracy,
            "operation_cue_causally_used": cue_causally_used,
            "all_gates_passed": accepted,
            "frozen_base_bit_identical": frozen_base_identical,
            "total_seconds": time.perf_counter() - started,
        }
        if accepted and checkpoint_out is not None:
            checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema": "unified-cognitive-controller-v1",
                "model_configuration": configuration,
                "state_dict": student.state_dict(),
                "source_report": str(report_path),
                "admission_status": "four_skill_compounding_transfer",
            }, checkpoint_out)
            report["checkpoint_saved"] = True
        else:
            report["checkpoint_saved"] = False
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        summaries.append(report_path)
        print(json.dumps({
            "seed": seed,
            "all_gates_passed": accepted,
            "new_skill": new_skill_accuracy,
            "cue_ablation": cue_ablation_accuracy,
            "retention": {
                task: evaluations[f"{task}_retention"]["gate"]["accepted"]
                for task in replay_tasks
            },
            "total_unique_lifetimes": total_lifetimes,
            "total_seconds": report["total_seconds"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
