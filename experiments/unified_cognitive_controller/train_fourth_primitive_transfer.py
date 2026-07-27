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
import json
import time
from pathlib import Path

import torch
from torch import nn

from .distill_visible_context import _distillation_loss, _trajectory_loss
from .environment import (
    NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes)
from .model import UnifiedCognitiveController
from .train import attempted_success_loss, evaluate, rollout, seed_everything


DEFAULT_NEW_TASK = "contextual_composition"
DEFAULT_REPLAY_TASKS = (
    "binary_mapping", "visible_context", "visible_context_xor")
# Kept for importers and tests that pin the fourth rung's own pair.
NEW_TASK = DEFAULT_NEW_TASK
REPLAY_TASKS = DEFAULT_REPLAY_TASKS


def _plastic_prefixes(slot: int) -> tuple[str, ...]:
    """Name only the slot this rung appends, so earlier slots stay frozen."""
    return (f"skill_adapters.{slot}.", f"skill_adapter_gates.{slot}.")


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
        device: torch.device) -> tuple[float, float]:
    """Mean opening of one slot, and how often it is exactly shut."""
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task=task,
        support_trials=support_trials, device=device)
    state = model.initial_state(count, device=device)
    action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    reward = torch.zeros(count, device=device)
    openings = []
    for trial in range(batch.trials):
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
        device: torch.device) -> float:
    """Mean norm of the perturbation one slot actually adds to the intention.

    The gate opening alone is not the disturbance: a nearly shut gate on a
    large residual still moves the answer, so this is the quantity a locality
    price has to act on.
    """
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task=task,
        support_trials=support_trials, device=device)
    state = model.initial_state(count, device=device)
    action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    reward = torch.zeros(count, device=device)
    norms = []
    for trial in range(batch.trials):
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


def _headline_accuracy(evaluation: dict) -> float:
    """The comparable accuracy for either task family.

    Tasks whose answer is fully visible report one overall accuracy; hidden
    rule tasks such as the new composition are only identifiable after their
    support outcome, so their headline number is post-feedback accuracy.
    """
    if "overall_accuracy" in evaluation:
        return float(evaluation["overall_accuracy"])
    return float(evaluation["normal"]["post_feedback_accuracy"])


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
        exploration: float, support_trials: int) -> torch.Tensor:
    result = rollout(
        model, batch, sample_actions=True, exploration=exploration,
        feedback_trials=support_trials)
    losses = []
    for trial in range(batch.trials):
        loss = attempted_success_loss(
            result["logits"][:, trial],
            result["actions"][:, trial],
            result["rewards"][:, trial])
        # Support trials precede the outcomes that identify the hidden rule,
        # so their gradient is deliberately discounted.
        losses.append(loss * (0.20 if trial < support_trials else 1.0))
    return torch.stack(losses).mean()


@torch.no_grad()
def _operation_cue_ablation_accuracy(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, support_trials: int,
        new_task: str = DEFAULT_NEW_TASK) -> float:
    """Rerender the same public events without the operation-mode symbol.

    The composition cue is the only pixel difference from the direct-context
    rendering, so this isolates whether the requested operation is read off
    the frame rather than guessed from the rest of the scene.
    """
    marked = generate_lifetimes(
        count, 6, seed=seed, heldout=True,
        task=new_task, support_trials=support_trials, device=device)
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
    parser.add_argument("--steps", type=int, default=64)
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
        "--replay-tasks", default=",".join(DEFAULT_REPLAY_TASKS),
        help="comma separated primitives the parent already holds")
    parser.add_argument(
        "--replay-support-trials",
        help=("comma separated support counts, one per replay task; defaults "
              "to one each. A skill must be replayed and audited at the "
              "support it was acquired at, or its retention is unmeasurable"))
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
        "--shuffle-retention-teacher", action="store_true",
        help="negative control: mismatch teacher behavior across lifetimes")
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--exploration", type=float, default=0.1)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(args.steps, args.new_batch_size, args.replay_batch_size) < 1:
        raise ValueError("steps and batch sizes must be positive")
    if args.retention_weight <= 0:
        raise ValueError("retention weight must be positive")
    if args.skill_adapter_width < 1:
        raise ValueError("the new plastic slot must have positive width")
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

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload, teacher = _load(args.parent, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    configuration = dict(payload["model_configuration"])
    inherited_slots = tuple(configuration.get("skill_adapter_widths", ()))
    new_slot = len(inherited_slots)
    plastic_prefixes = _plastic_prefixes(new_slot)
    configuration["skill_adapter_widths"] = (
        inherited_slots + (args.skill_adapter_width,))
    configuration["skill_adapter_gate_mode"] = args.slot_gate_mode
    configuration["skill_adapter_gate_hidden"] = args.slot_gate_hidden
    student = UnifiedCognitiveController(**configuration).to(device)
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
        parameter.requires_grad_(name.startswith(plastic_prefixes))
    inherited_frozen_adapters = sorted({
        name.split(".")[0] for name in teacher.state_dict()
        if "adapter" in name})
    frozen_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if not name.startswith(plastic_prefixes)
    }
    plastic_parameters = sum(
        parameter.numel() for name, parameter in student.named_parameters()
        if name.startswith(plastic_prefixes))
    frozen_parameters = sum(
        parameter.numel() for name, parameter in student.named_parameters()
        if not name.startswith(plastic_prefixes))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-5)

    # A retention gate is only required where the frozen parent already had
    # the skill, so arms with different histories stay comparable.
    # Evaluated on exactly the seeds the student's retention will use. A set
    # point measured on different lifetimes than the outcome would make the
    # per-rung degradation a difference of two noisy numbers, and evaluation
    # noise here is the same size as the effect being measured.
    parent_evaluations = {
        task: evaluate(
            teacher, count=args.test_lifetimes, trials=6,
            seed=args.seed + 91_000_000 + index, device=device,
            task=task, feedback_trials=replay_support_by_task[task])
        for index, task in enumerate(replay_tasks)
    }
    parent_retention = {
        task: evaluation["gate"]["accepted"]
        for task, evaluation in parent_evaluations.items()
    }
    # The level each old skill arrives at is the level it should leave at. These
    # are the controller's set points; nothing here reaches the learner, which
    # still sees only frames, its own actions, and scalar outcomes.
    retention_set_point = {
        task: _headline_accuracy(evaluation)
        for task, evaluation in parent_evaluations.items()
    }
    tracked_accuracy = dict(retention_set_point)

    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    for update in range(1, args.steps + 1):
        student.train()
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
            seed=args.seed * 10_000_000 + update,
            task=new_task, support_trials=support_trials,
            device=device)
        replay_batches = [
            generate_lifetimes(
                args.replay_batch_size, 6,
                seed=args.seed * (20_000_000 + 10_000_000 * index) + update,
                task=task, support_trials=replay_support_by_task[task],
                device=device)
            for index, task in enumerate(replay_tasks)
        ]
        skill_loss = _new_skill_loss(
            student, new_batch, exploration=args.exploration,
            support_trials=support_trials)
        replay_results = [
            _replay_loss_and_leakage(
                student, teacher, batch, slot=new_slot,
                feedback_trials=replay_support_by_task[task],
                shuffled_teacher=args.shuffle_retention_teacher)
            for task, batch in zip(replay_tasks, replay_batches)
        ]
        replay_losses = [value for value, _, _ in replay_results]
        leakages = [value for _, value, _ in replay_results]
        leakage = torch.stack(leakages).mean()
        # Proportional set-point control on the retention price. A skill at or
        # above the level it was inherited at costs nothing extra, so pressure
        # never competes with new learning unless a skill is actually slipping.
        weights = []
        deficits = []
        for index, task in enumerate(replay_tasks):
            measured = replay_results[index][2]
            tracked_accuracy[task] = (
                args.retention_tracking_decay * tracked_accuracy[task]
                + (1.0 - args.retention_tracking_decay) * measured)
            deficit = max(
                0.0, retention_set_point[task] - tracked_accuracy[task])
            deficits.append(deficit)
            weights.append(
                args.retention_weight + args.retention_control_gain * deficit)
        loss = skill_loss + sum(
            weight * value for weight, value in zip(weights, replay_losses))
        if args.locality_weight:
            # Price opening the new slot on events that belong to old skills.
            loss = loss + args.locality_weight * leakage
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if update in (1, args.steps):
            history.append({
                "update": update,
                "skill_loss": float(skill_loss.detach()),
                **{
                    f"{task}_distillation_loss": float(value.detach())
                    for task, value in zip(replay_tasks, replay_losses)
                },
                **{
                    f"{task}_slot_opening": float(value.detach())
                    for task, value in zip(replay_tasks, leakages)
                },
                **{
                    f"{task}_retention_weight": weight
                    for task, weight in zip(replay_tasks, weights)
                },
                **{
                    f"{task}_retention_deficit": deficit
                    for task, deficit in zip(replay_tasks, deficits)
                },
                "replay_slot_opening": float(leakage.detach()),
                "total_loss": float(loss.detach()),
            })

    # Nothing is measured through a leaky gate: the exact-zero property is the
    # whole mechanism, so the anneal must be finished before any evaluation.
    student.skill_adapter_gate_leak = 0.0
    evaluations = {
        "new_skill": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 90_000_000, device=device,
            task=new_task, feedback_trials=final_support_trials),
        **{
            f"{task}_retention": evaluate(
                student, count=args.test_lifetimes, trials=6,
                seed=args.seed + 91_000_000 + index, device=device,
                task=task, feedback_trials=replay_support_by_task[task])
            for index, task in enumerate(replay_tasks)
        },
    }
    cue_ablation_accuracy = _operation_cue_ablation_accuracy(
        student, count=args.test_lifetimes,
        seed=args.seed + 90_000_000, device=device,
        support_trials=final_support_trials, new_task=new_task)
    _openings = {
        new_task: _slot_opening(
            student, slot=new_slot, task=new_task,
            count=args.test_lifetimes, seed=args.seed + 93_000_000,
            support_trials=final_support_trials, device=device),
        **{
            task: _slot_opening(
                student, slot=new_slot, task=task,
                count=args.test_lifetimes,
                seed=args.seed + 94_000_000 + index,
                support_trials=replay_support_by_task[task], device=device)
            for index, task in enumerate(replay_tasks)
        },
    }
    slot_opening = {task: value for task, (value, _) in _openings.items()}
    slot_shut_fraction = {
        task: shut for task, (_, shut) in _openings.items()}
    slot_residual_norm = {
        new_task: _slot_residual_norm(
            student, slot=new_slot, task=new_task,
            count=args.test_lifetimes, seed=args.seed + 93_000_000,
            support_trials=final_support_trials, device=device),
        **{
            task: _slot_residual_norm(
                student, slot=new_slot, task=task,
                count=args.test_lifetimes,
                seed=args.seed + 94_000_000 + index,
                support_trials=replay_support_by_task[task], device=device)
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
    accepted = (
        evaluations["new_skill"]["gate"]["accepted"]
        and retention_gates_passed
        and cue_causally_used
        and slot_shut_fraction[new_task] < 1.0)
    frozen_base_identical = all(
        torch.equal(frozen_initial[name], value.detach().cpu())
        for name, value in student.state_dict().items()
        if name in frozen_initial)
    total_lifetimes = args.steps * (
        args.new_batch_size + len(replay_tasks) * args.replay_batch_size)
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
        "inherited_frozen_adapters": inherited_frozen_adapters,
        "inherited_skill_adapter_slots": list(inherited_slots),
        "plastic_parameters": plastic_parameters,
        "frozen_parameters": frozen_parameters,
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
            "new_unique_lifetimes": args.steps * args.new_batch_size,
            "replay_lifetimes_per_task": args.steps * args.replay_batch_size,
            "total_unique_lifetimes": total_lifetimes,
            "total_verifier_bits": total_lifetimes * 6,
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
        "headline_accuracy": {
            "new_skill": new_skill_accuracy,
            **{
                f"{task}_retention": _headline_accuracy(
                    evaluations[f"{task}_retention"])
                for task in replay_tasks
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
    if accepted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": student.state_dict(),
            "source_report": str(args.report),
            "admission_status": "four_skill_compounding_transfer",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
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
