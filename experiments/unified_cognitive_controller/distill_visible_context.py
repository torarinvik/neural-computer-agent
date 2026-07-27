"""Merge an acquired visual-action primitive into one controller without labels.

The student starts as the mature controller plus a zero-output generic action
residual.  It observes only RGB frames.  On context frames it matches an
independently acquired controller's opaque action distribution; on old binary
frames it matches the mature controller's distribution.  No verifier rule,
context ID, or correct action is used as a distillation target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, generate_lifetimes
from .model import UnifiedCognitiveController
from .train import evaluate


def _load(path: Path, device: torch.device) -> tuple[dict, UnifiedCognitiveController]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(**payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.kl_div(
        torch.log_softmax(student_logits, dim=-1),
        torch.softmax(teacher_logits, dim=-1), reduction="batchmean")


def _trajectory_loss(
        student: UnifiedCognitiveController, teacher: UnifiedCognitiveController,
        batch, *, feedback_trials: int, shuffled_teacher: bool) -> torch.Tensor:
    """Distil behavior along real separate action/outcome trajectories."""
    count = batch.batch_size
    device = batch.frames.device
    student_state = student.initial_state(count, device=device)
    teacher_state = teacher.initial_state(count, device=device)
    null = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    student_action = teacher_action = null
    student_reward = teacher_reward = torch.zeros(count, device=device)
    losses = []
    for trial in range(batch.trials):
        feedback = torch.full_like(student_reward, float(0 < trial <= feedback_trials))
        student_output, student_state = student.step(
            batch.frames[:, trial], student_state, student_action,
            student_reward * feedback, feedback)
        with torch.no_grad():
            teacher_output, teacher_state = teacher.step(
                batch.frames[:, trial], teacher_state, teacher_action,
                teacher_reward * feedback, feedback)
            target = teacher_output.logits.roll(1, dims=0) if shuffled_teacher else teacher_output.logits
        losses.append(_distillation_loss(student_output.logits, target))
        student_action = student_output.logits.detach().argmax(-1)
        teacher_action = teacher_output.logits.argmax(-1)
        student_reward = (student_action == batch.correct_actions[:, trial]).float()
        teacher_reward = (teacher_action == batch.correct_actions[:, trial]).float()
    return torch.stack(losses).mean()


def _train(
        student: UnifiedCognitiveController, specialist: UnifiedCognitiveController,
        parent: UnifiedCognitiveController, *, steps: int, batch_size: int,
        seed: int, device: torch.device, shuffled_specialist: bool,
        context_weight: float,
        ) -> list[float]:
    assert student.action_adapter is not None
    parameters = list(student.action_adapter.parameters())
    if student.action_adapter_gate is not None:
        parameters += list(student.action_adapter_gate.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    losses = []
    for update in range(steps):
        context = generate_lifetimes(
            batch_size, 3, seed=seed * 10_000 + update,
            task="visible_context", device=device)
        binary = generate_lifetimes(
            batch_size, 3, seed=seed * 20_000 + update,
            task="binary_mapping", device=device)
        loss = (
            context_weight * _trajectory_loss(
                student, specialist, context, feedback_trials=1,
                shuffled_teacher=shuffled_specialist)
            + _trajectory_loss(
                student, parent, binary, feedback_trials=1,
                shuffled_teacher=False))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _report(
        student: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device) -> dict[str, object]:
    context = evaluate(
        student, count=count, trials=6, seed=seed, device=device,
        task="visible_context", feedback_trials=1)
    binary = evaluate(
        student, count=count, trials=6, seed=seed + 1,
        device=device, task="binary_mapping", feedback_trials=1)
    return {
        "visible_context": context,
        "binary_retention": binary,
        "accepted": (
            context["gate"]["accepted"] and binary["gate"]["accepted"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--student-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--action-adapter-width", type=int, default=64)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.context_weight <= 0:
        raise ValueError("context_weight must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parent_payload, parent = _load(args.parent, device)
    _, specialist = _load(args.specialist, device)
    if args.student_in is None:
        config = dict(parent_payload["model_configuration"])
        config["action_adapter_width"] = args.action_adapter_width
        config["action_adapter_gated"] = args.gated
        student = UnifiedCognitiveController(**config).to(device)
        missing, unexpected = student.load_state_dict(parent.state_dict(), strict=False)
        expected_missing = {
            name for name in student.state_dict()
            if name.startswith(("action_adapter.", "action_adapter_gate."))}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(f"unexpected merge mismatch: missing={missing}, unexpected={unexpected}")
    else:
        student_payload = torch.load(args.student_in, map_location=device, weights_only=False)
        config = dict(student_payload["model_configuration"])
        student = UnifiedCognitiveController(**config).to(device)
        student.load_state_dict(student_payload["state_dict"])
    base = {name: value.detach().clone() for name, value in student.state_dict().items()
            if not name.startswith("action_adapter.")}
    losses = _train(
        student, specialist, parent, steps=args.steps, batch_size=args.batch_size,
        seed=args.seed, device=device, shuffled_specialist=False,
        context_weight=args.context_weight)
    evaluation = _report(student, count=args.test_lifetimes, seed=args.seed + 9_000_000, device=device)

    shuffled = UnifiedCognitiveController(**config).to(device)
    shuffled.load_state_dict(parent.state_dict(), strict=False)
    shuffled_losses = _train(
        shuffled, specialist, parent, steps=args.steps, batch_size=args.batch_size,
        seed=args.seed, device=device, shuffled_specialist=True,
        context_weight=args.context_weight)
    shuffled_evaluation = _report(
        shuffled, count=args.test_lifetimes, seed=args.seed + 9_000_000, device=device)
    base_unchanged = all(
        torch.equal(base[name], value)
        for name, value in student.state_dict().items()
        if name in base)
    report = {
        "schema": "visible-context-distillation-v1",
        "claim_boundary": (
            "Behavioral distillation between two learned controllers; no "
            "semantic labels or verifier correct actions are target inputs."),
        "parent": str(args.parent),
        "specialist": str(args.specialist),
        "student_in": str(args.student_in) if args.student_in is not None else None,
        "steps": args.steps,
        "gated": args.gated,
        "context_weight": args.context_weight,
        "batch_size": args.batch_size,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "base_parameters_bit_identical": base_unchanged,
        "evaluation": evaluation,
        "shuffled_specialist_loss_first": shuffled_losses[0],
        "shuffled_specialist_loss_last": shuffled_losses[-1],
        "shuffled_specialist_evaluation": shuffled_evaluation,
        "shuffled_control_rejected": not shuffled_evaluation["accepted"],
    }
    if evaluation["accepted"] and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": config,
            "state_dict": student.state_dict(),
            "source_report": str(args.report),
            "admission_status": "integrated_context_and_binary",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "loss_first": report["loss_first"], "loss_last": report["loss_last"],
        "evaluation": evaluation, "shuffled_control_rejected": report["shuffled_control_rejected"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
