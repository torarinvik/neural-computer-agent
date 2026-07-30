"""Meta-train the unified controller on hidden two-choice cognitive lifetimes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .environment import ACTIONS, NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes
from .model import UnifiedCognitiveController


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attempted_success_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor) -> torch.Tensor:
    """Bandit loss using only the attempted action and its scalar outcome."""
    selected = logits.gather(1, attempted.unsqueeze(1)).squeeze(1)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        selected, outcomes)


def rollout(
        model: UnifiedCognitiveController, batch: CognitiveLifetimeBatch, *,
        sample_actions: bool, exploration: float = 0.10,
        feedback_trials: int = 1,
        reset_workspace_each_trial: bool = False,
        shuffle_feedback: bool = False,
        drop_feedback_at: int | None = None,
        disable_workspace: bool = False) -> dict[str, torch.Tensor]:
    if feedback_trials < 0 or feedback_trials >= batch.trials:
        raise ValueError(
            "feedback_trials must be between zero and trials - 1")
    if drop_feedback_at is not None and not 1 <= drop_feedback_at <= feedback_trials:
        raise ValueError("drop_feedback_at must identify a delivered support outcome")
    state = model.initial_state(
        batch.batch_size, device=batch.frames.device,
        dtype=batch.frames.dtype)
    previous_action = torch.full(
        (batch.batch_size,), NULL_ACTION, device=batch.frames.device,
        dtype=torch.long)
    previous_reward = torch.zeros(
        batch.batch_size, device=batch.frames.device)
    actions, rewards, logits_by_trial = [], [], []
    for trial in range(batch.trials):
        if reset_workspace_each_trial and trial:
            state = model.initial_state(
                batch.batch_size, device=batch.frames.device,
                dtype=batch.frames.dtype)
        if batch.prestimulus_frames is not None:
            # A cue frame is ordinary sensory time: no action is taken and no
            # verifier outcome is delivered until the following stimulus.
            _, state = model.step(
                batch.prestimulus_frames[:, trial], state,
                previous_action, torch.zeros_like(previous_reward),
                torch.zeros_like(previous_reward),
                disable_workspace=disable_workspace)
        has_feedback = torch.full_like(
            previous_reward,
            float(0 < trial <= feedback_trials))
        delivered_reward = previous_reward * has_feedback
        if drop_feedback_at == trial:
            has_feedback = torch.zeros_like(has_feedback)
            delivered_reward = torch.zeros_like(delivered_reward)
        if shuffle_feedback and bool(has_feedback[0]):
            delivered_reward = delivered_reward.roll(1)
        output, state = model.step(
            batch.frames[:, trial], state, previous_action,
            delivered_reward, has_feedback,
            disable_workspace=disable_workspace)
        probabilities = torch.softmax(output.logits, dim=-1)
        if sample_actions:
            behavior = (
                probabilities * (1.0 - exploration)
                + exploration / ACTIONS)
            action = torch.multinomial(behavior, 1).squeeze(1)
        else:
            action = output.logits.argmax(dim=-1)
        reward = (
            action == batch.correct_actions[:, trial]).to(
                output.logits.dtype)
        actions.append(action)
        rewards.append(reward)
        logits_by_trial.append(output.logits)
        previous_action = action
        previous_reward = reward
    return {
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "logits": torch.stack(logits_by_trial, dim=1),
        "final_workspace": state.workspace,
        "final_hidden": state.hidden,
    }


def _metrics(
        result: dict[str, torch.Tensor], *,
        query_start: int = 1) -> dict[str, object]:
    accuracy = result["rewards"].float().mean(dim=0)
    if query_start < 1 or query_start >= accuracy.numel():
        raise ValueError("query_start must identify a nonempty query suffix")
    query_accuracy = accuracy[query_start:].mean()
    return {
        "accuracy_by_trial": [float(value) for value in accuracy],
        "zero_shot_accuracy": float(accuracy[0]),
        "post_feedback_accuracy": float(query_accuracy),
        "query_accuracy": float(query_accuracy),
        "final_accuracy": float(accuracy[-1]),
        "learning_gain": float(query_accuracy - accuracy[0]),
    }


@torch.no_grad()
def evaluate(
        model: UnifiedCognitiveController, *, count: int, trials: int,
        seed: int, device: torch.device,
        task: str, feedback_trials: int,
        appearance: str = "bars",
        appearance_blend: float | None = None,
        numerosity_mass_control: float = 0.0,
        numerosity_appearance_blend: float = 1.0,
        operation_cue_scale: float = 1.0,
        operation_cue_trials: int | None = None,
        operation_cue_prestimulus: bool = False) -> dict[str, object]:
    model.eval()
    normal_batch = generate_lifetimes(
        count, trials, seed=seed, heldout=True, task=task,
        appearance=appearance, appearance_blend=appearance_blend,
        numerosity_mass_control=numerosity_mass_control,
        numerosity_appearance_blend=numerosity_appearance_blend,
        operation_cue_scale=operation_cue_scale,
        operation_cue_trials=operation_cue_trials,
        operation_cue_prestimulus=operation_cue_prestimulus,
        support_trials=feedback_trials, device=device)
    reversed_batch = generate_lifetimes(
        count, trials, seed=seed, heldout=True,
        reverse_rules=(task not in (
            "visible_identity", "pair_relation",
            "pair_magnitude", "visible_pair_magnitude",
            "visible_pair_numerosity", "visible_pair_numerosity_smaller",
            "visible_numerosity_equality",
            "visible_context", "visible_context_xor")),
        reverse_stimuli=(task == "visible_identity"),
        reverse_contexts=(task in (
            "pair_relation", "pair_magnitude",
            "visible_pair_magnitude",
            "visible_pair_numerosity", "visible_pair_numerosity_smaller",
            "visible_numerosity_equality",
            "visible_context", "visible_context_xor")),
        task=task, appearance=appearance,
        appearance_blend=appearance_blend,
        numerosity_mass_control=numerosity_mass_control,
        numerosity_appearance_blend=numerosity_appearance_blend,
        operation_cue_scale=operation_cue_scale,
        operation_cue_trials=operation_cue_trials,
        operation_cue_prestimulus=operation_cue_prestimulus,
        support_trials=feedback_trials, device=device)
    normal = rollout(
        model, normal_batch, sample_actions=False,
        feedback_trials=feedback_trials)
    reversed_result = rollout(
        model, reversed_batch, sample_actions=False,
        feedback_trials=feedback_trials)
    reset = rollout(
        model, normal_batch, sample_actions=False,
        feedback_trials=feedback_trials,
        reset_workspace_each_trial=True)
    shuffled = rollout(
        model, normal_batch, sample_actions=False,
        feedback_trials=feedback_trials, shuffle_feedback=True)
    second_support_removed = None
    if task in (
            "contextual_mapping", "contextual_override",
            "contextual_composition") and feedback_trials >= 2:
        second_support_removed = rollout(
            model, normal_batch, sample_actions=False,
            feedback_trials=feedback_trials,
            drop_feedback_at=feedback_trials)
    disabled = rollout(
        model, normal_batch, sample_actions=False,
        feedback_trials=feedback_trials, disable_workspace=True)
    blank_batch = CognitiveLifetimeBatch(
        frames=torch.zeros_like(normal_batch.frames),
        correct_actions=normal_batch.correct_actions,
        stimulus_identities=normal_batch.stimulus_identities,
        rule_bits=normal_batch.rule_bits,
        seeds=normal_batch.seeds,
        context_ids=normal_batch.context_ids)
    blank = rollout(
        model, blank_batch, sample_actions=False,
        feedback_trials=feedback_trials)
    # Hidden-rule tasks are only identifiable after all support outcomes have
    # arrived. Counterfactual sensitivity belongs on the query suffix, not on
    # support actions the controller could not yet infer.
    flip_start = (
        0 if task in (
            "visible_identity", "pair_relation",
            "visible_pair_numerosity",
            "visible_context", "visible_context_xor")
        else feedback_trials)
    flip_rate = (
        normal["actions"][:, flip_start:]
        != reversed_result["actions"][:, flip_start:]
    ).float().mean()
    report = {
        "normal": _metrics(normal, query_start=feedback_trials),
        "reversed_rule": _metrics(
            reversed_result, query_start=feedback_trials),
        "active_state_reset": _metrics(
            reset, query_start=feedback_trials),
        "feedback_shuffled": _metrics(
            shuffled, query_start=feedback_trials),
        "workspace_disabled": _metrics(
            disabled, query_start=feedback_trials),
        "blank_vision": _metrics(
            blank, query_start=feedback_trials),
        "post_feedback_prediction_flip_rate": float(flip_rate),
    }
    if second_support_removed is not None:
        report["second_support_feedback_removed"] = _metrics(
            second_support_removed, query_start=feedback_trials)
    if normal_batch.context_ids is not None:
        # Per-context accuracy needs only the normal rollout, so it is
        # available at every support budget, including the one-support rungs.
        query_contexts = normal_batch.context_ids[:, feedback_trials:]
        normal_rewards = normal["rewards"][:, feedback_trials:]
        report["normal_query_accuracy_by_context"] = {
            str(context): float(normal_rewards[query_contexts == context].float().mean())
            for context in (0, 1)
        }
    if task in (
            "visible_identity", "pair_relation",
            "visible_pair_magnitude",
            "visible_pair_numerosity", "visible_pair_numerosity_smaller",
            "visible_numerosity_equality",
            "visible_context", "visible_context_xor"):
        normal_accuracy = float(normal["rewards"].mean())
        reversed_accuracy = float(reversed_result["rewards"].mean())
        blank_accuracy = float(blank["rewards"].mean())
        report["overall_accuracy"] = normal_accuracy
        report["counterfactual_overall_accuracy"] = reversed_accuracy
        report["blank_vision_overall_accuracy"] = blank_accuracy
        report["gate"] = {
            "normal_accuracy_at_least_90": normal_accuracy >= 0.90,
            "counterfactual_accuracy_at_least_90":
                reversed_accuracy >= 0.90,
            "pixel_counterfactual_flip_at_least_80":
                float(flip_rate) >= 0.80,
            "blank_vision_at_chance": blank_accuracy <= 0.60,
        }
        report["gate"]["accepted"] = all(report["gate"].values())
        report["gate"]["primitive_mastery"] = report["gate"]["accepted"]
        return report
    normal_post = report["normal"]["post_feedback_accuracy"]
    reversed_post = report["reversed_rule"]["post_feedback_accuracy"]
    blank_post = report["blank_vision"]["post_feedback_accuracy"]
    zero_shot = report["normal"]["zero_shot_accuracy"]
    report["gate"] = {
        "zero_shot_near_chance": 0.40 <= zero_shot <= 0.60,
        "normal_few_shot_at_least_85": normal_post >= 0.85,
        "reversed_few_shot_at_least_85": reversed_post >= 0.85,
        "counterfactual_flip_at_least_80": float(flip_rate) >= 0.80,
        "vision_causally_used": blank_post <= normal_post - 0.15,
        "active_state_reset_hurts": (
            report["active_state_reset"]["post_feedback_accuracy"]
            <= normal_post - 0.15),
        "shuffled_feedback_hurts": (
            report["feedback_shuffled"]["post_feedback_accuracy"]
            <= normal_post - 0.15),
    }
    if task in (
            "contextual_mapping", "contextual_override",
            "contextual_composition"):
        report["gate"]["both_contexts_mastered"] = all(
            accuracy >= 0.85
            for accuracy in report["normal_query_accuracy_by_context"].values())
        if task == "contextual_mapping":
            # Only this task gates on the second support outcome, which the
            # one-support rungs never render.
            assert second_support_removed is not None
            report["gate"]["second_support_evidence_hurts"] = (
                report["second_support_feedback_removed"]["post_feedback_accuracy"]
                <= normal_post - 0.15)
    report["gate"]["few_shot_breakthrough"] = all(
        report["gate"].values())
    report["gate"]["accepted"] = report["gate"]["few_shot_breakthrough"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument(
        "--candidate-checkpoint-out", type=Path,
        help=("save an explicitly unpromoted continuation checkpoint even "
              "when the behavioral admission gate has not passed"))
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument(
        "--task", choices=(
            "constant_action", "visible_identity", "pair_relation",
            "binary_mapping",
            "visible_context", "visible_context_xor", "four_rule",
            "contextual_mapping",
            "contextual_override", "contextual_composition", "context_rule_xor",
            "context_identity_and", "context_identity_or"),
        default="constant_action")
    parser.add_argument(
        "--appearance", choices=("bars", "diamonds", "dot_pairs"),
        default="bars")
    parser.add_argument(
        "--rehearsal-task", choices=(
            "constant_action", "visible_identity", "pair_relation",
            "binary_mapping",
            "visible_context", "visible_context_xor", "four_rule",
            "contextual_mapping",
            "contextual_override", "contextual_composition", "context_rule_xor",
            "context_identity_and", "context_identity_or"))
    parser.add_argument(
        "--rehearsal-every", type=int, default=2,
        help="use one rehearsal batch every N optimizer steps")
    parser.add_argument(
        "--rehearsal-feedback-trials", type=int,
        help=(
            "support outcomes for rehearsal lifetimes; defaults to the "
            "main task's feedback count"))
    parser.add_argument(
        "--rehearsal-appearance",
        choices=("bars", "diamonds", "dot_pairs"),
        help="renderer appearance for rehearsal lifetimes")
    parser.add_argument(
        "--retention-task", choices=(
            "constant_action", "visible_identity", "pair_relation",
            "binary_mapping",
            "visible_context", "visible_context_xor", "four_rule",
            "contextual_mapping",
            "contextual_override", "contextual_composition", "context_rule_xor",
            "context_identity_and", "context_identity_or"))
    parser.add_argument(
        "--retention-feedback-trials", type=int,
        help=(
            "support outcomes used by the retention audit; defaults to the "
            "main task's feedback count"))
    parser.add_argument(
        "--retention-appearance",
        choices=("bars", "diamonds", "dot_pairs"),
        help="renderer appearance used by the retention audit")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument(
        "--feedback-trials", type=int, default=1,
        help="early support trials whose outcomes enter controller state")
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--intention-width", type=int, default=24)
    parser.add_argument(
        "--relation-adapter-width", type=int,
        help=("insert a zero-initialized generic prior-state/query-event "
              "relation residual; omit to preserve checkpoint architecture"))
    parser.add_argument(
        "--relation-adapter-gated", action="store_true",
        help="learn a sensory gate for an inserted relation residual")
    parser.add_argument(
        "--train-relation-adapter-only", action="store_true",
        help="freeze inherited parameters and optimize only the relation residual")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if args.batch_size % 2 or args.test_lifetimes % 2:
        raise ValueError("batch sizes must be divisible by two")
    if args.rehearsal_every < 1:
        raise ValueError("rehearsal_every must be positive")
    for name in (
            "rehearsal_feedback_trials", "retention_feedback_trials"):
        value = getattr(args, name)
        if value is not None and not 0 <= value < args.trials:
            raise ValueError(
                f"{name} must be between zero and trials - 1")
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint_payload = None
    model_configuration: dict[str, object] = {
        "width": args.width,
        "workspace_slots": args.workspace_slots,
        "intention_width": args.intention_width,
    }
    if args.checkpoint_in is not None:
        checkpoint_payload = torch.load(
            args.checkpoint_in, map_location=device, weights_only=False)
        if checkpoint_payload.get("schema") != "unified-cognitive-controller-v1":
            raise ValueError("unsupported controller checkpoint")
        checkpoint_configuration = checkpoint_payload.get("model_configuration")
        if not isinstance(checkpoint_configuration, dict):
            raise ValueError("controller checkpoint lacks model configuration")
        for key, expected in model_configuration.items():
            if checkpoint_configuration.get(key) != expected:
                raise ValueError(
                    f"checkpoint {key} does not match requested configuration")
        # Preserve optional architectural fields (adaptive reads/replacement,
        # etc.) rather than silently constructing a different controller.
        model_configuration = dict(checkpoint_configuration)
    if args.relation_adapter_width is not None:
        model_configuration["relation_adapter_width"] = args.relation_adapter_width
        model_configuration["relation_adapter_gated"] = (
            args.relation_adapter_gated)
    model = UnifiedCognitiveController(**model_configuration).to(device)
    initialization = "fresh"
    if checkpoint_payload is not None:
        missing, unexpected = model.load_state_dict(
            checkpoint_payload["state_dict"], strict=False)
        allowed_missing = (
            {name for name in model.state_dict()
             if name.startswith((
                 "relation_adapter.", "relation_adapter_gate."))}
            if args.relation_adapter_width is not None
            and args.relation_adapter_width > 0
            and "relation_adapter_width" not in checkpoint_configuration
            else set())
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "checkpoint/model architecture mismatch: "
                f"missing={missing}, unexpected={unexpected}")
        initialization = str(args.checkpoint_in)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    if args.train_relation_adapter_only:
        if model.relation_adapter is None:
            raise ValueError("--train-relation-adapter-only needs a relation adapter")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith((
                "relation_adapter.", "relation_adapter_gate.")))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=1e-5)
    optimizer_initialized_from = "fresh"
    if checkpoint_payload is not None and isinstance(
            checkpoint_payload.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        optimizer_initialized_from = "checkpoint"
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        is_rehearsal = (
            args.rehearsal_task is not None
            and step % args.rehearsal_every == 0)
        batch_task = args.rehearsal_task if is_rehearsal else args.task
        batch_feedback_trials = (
            args.rehearsal_feedback_trials
            if is_rehearsal
            and args.rehearsal_feedback_trials is not None
            else args.feedback_trials)
        batch_appearance = (
            args.rehearsal_appearance
            if is_rehearsal and args.rehearsal_appearance is not None
            else args.appearance)
        batch = generate_lifetimes(
            args.batch_size, args.trials,
            seed=args.seed * 1_000_000 + step, task=batch_task,
            appearance=batch_appearance,
            support_trials=batch_feedback_trials,
            device=device)
        result = rollout(
            model, batch, sample_actions=True,
            exploration=args.exploration,
            feedback_trials=batch_feedback_trials)
        losses = []
        for trial in range(args.trials):
            loss = attempted_success_loss(
                result["logits"][:, trial],
                result["actions"][:, trial],
                result["rewards"][:, trial])
            # Trial zero is irreducibly chance; retain a small calibration term
            # while concentrating learning pressure after evidence arrives.
            losses.append(loss * (0.20 if trial == 0 else 1.0))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "batch_task": batch_task,
                "batch_feedback_trials": batch_feedback_trials,
                "batch_appearance": batch_appearance,
                "loss": float(loss.detach()),
                **_metrics(result, query_start=batch_feedback_trials),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    evaluation = evaluate(
        model, count=args.test_lifetimes, trials=args.trials,
        seed=args.seed + 90_000_000, device=device, task=args.task,
        feedback_trials=args.feedback_trials, appearance=args.appearance)
    retention_evaluation = None
    if args.retention_task is not None:
        retention_feedback_trials = (
            args.retention_feedback_trials
            if args.retention_feedback_trials is not None
            else args.feedback_trials)
        retention_appearance = (
            args.retention_appearance
            if args.retention_appearance is not None
            else args.appearance)
        retention_evaluation = evaluate(
            model, count=args.test_lifetimes, trials=args.trials,
            seed=args.seed + 91_000_000, device=device,
            task=args.retention_task,
            feedback_trials=retention_feedback_trials,
            appearance=retention_appearance)
    admitted = (
        evaluation["gate"]["accepted"]
        and (
            retention_evaluation is None
            or retention_evaluation["gate"]["accepted"]))
    report = {
        "schema": "unified-cognitive-controller-hidden-rule-v1",
        "claim_boundary": (
            "Within-lifetime adaptation to a hidden binary visual-action "
            "bijection using one controller and differentiable workspace."),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "within_lifetime_weight_updates": False,
        "learner_visible": [
            "rendered_rgb_frame", "own_previous_opaque_action",
            "scalar_verified_outcome", "own_latent_workspace",
        ],
        "verifier_private": [
            "stimulus_identity", "hidden_rule_bit", "correct_action",
            "counterfactual_pairing",
        ],
        "configuration": vars(args) | {
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "candidate_checkpoint_out": (
                str(args.candidate_checkpoint_out)
                if args.candidate_checkpoint_out is not None else None),
            "checkpoint_in": (
                str(args.checkpoint_in)
                if args.checkpoint_in is not None else None),
        },
        "initialization": initialization,
        "optimizer_initialization": optimizer_initialized_from,
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable_parameters),
        "history": history,
        "evaluation": evaluation,
        "retention_evaluation": retention_evaluation,
        "all_admission_gates_passed": admitted,
        "total_seconds": time.perf_counter() - started,
        "weights_changed": any(
            not torch.equal(initial[name], value.detach().cpu())
            for name, value in model.state_dict().items()),
    }
    if admitted and (
            args.checkpoint_out is not None):
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": model_configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    if args.candidate_checkpoint_out is not None:
        args.candidate_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": model_configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
            "admission_status": "unpromoted_candidate",
            "all_admission_gates_passed": admitted,
            "optimizer_state_dict": optimizer.state_dict(),
        }, args.candidate_checkpoint_out)
        report["candidate_checkpoint_saved"] = True
    else:
        report["candidate_checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "evaluation": evaluation,
        "parameters": report["parameters"],
        "total_seconds": report["total_seconds"],
        "checkpoint_saved": report["checkpoint_saved"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
