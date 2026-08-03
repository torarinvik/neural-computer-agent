"""Acquire one new primitive while consolidating two retained behaviors.

The learner receives only rendered frames, its own opaque actions, and scalar
verifier outcomes on the new task. Retention targets are the frozen starting
controller's opaque action distributions on newly rendered replay lifetimes;
semantic task state and correct unattempted actions are never targets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .distill_visible_context import _trajectory_loss
from .environment import CognitiveLifetimeBatch, generate_lifetimes
from .legacy_model import UnifiedCognitiveController
from .train import attempted_success_loss, evaluate, rollout, seed_everything


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
        exploration: float) -> torch.Tensor:
    result = rollout(
        model, batch, sample_actions=True, exploration=exploration,
        feedback_trials=1)
    losses = []
    for trial in range(batch.trials):
        loss = attempted_success_loss(
            result["logits"][:, trial],
            result["actions"][:, trial],
            result["rewards"][:, trial])
        losses.append(loss * (0.20 if trial == 0 else 1.0))
    return torch.stack(losses).mean()


@torch.no_grad()
def _operation_cue_ablation_accuracy(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device) -> float:
    """Rerender the same public events without the operation-mode symbol."""
    marked = generate_lifetimes(
        count, 6, seed=seed, heldout=True,
        task="visible_context_xor", support_trials=1, device=device)
    unmarked = generate_lifetimes(
        count, 6, seed=seed, heldout=True,
        task="visible_context", support_trials=1, device=device)
    if (
            not torch.equal(
                marked.stimulus_identities, unmarked.stimulus_identities)
            or not torch.equal(marked.context_ids, unmarked.context_ids)):
        raise RuntimeError("operation-cue rerender changed task content")
    ablated = CognitiveLifetimeBatch(
        frames=unmarked.frames,
        correct_actions=marked.correct_actions,
        stimulus_identities=marked.stimulus_identities,
        rule_bits=marked.rule_bits,
        seeds=marked.seeds,
        context_ids=marked.context_ids)
    result = rollout(
        model, ablated, sample_actions=False, feedback_trials=1)
    return float(result["rewards"].float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--new-batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--relation-adapter-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--shuffle-retention-teacher", action="store_true",
        help="negative control: mismatch teacher behavior across lifetimes")
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--exploration", type=float, default=0.1)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(args.steps, args.new_batch_size, args.replay_batch_size) < 1:
        raise ValueError("steps and batch sizes must be positive")
    if args.retention_weight <= 0:
        raise ValueError("retention weight must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload, teacher = _load(args.parent, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    configuration = dict(payload["model_configuration"])
    if configuration.get("relation_adapter_width", 0):
        raise ValueError("parent already contains a relation adapter")
    configuration["relation_adapter_width"] = args.relation_adapter_width
    configuration["relation_adapter_gated"] = True
    student = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = student.load_state_dict(
        teacher.state_dict(), strict=False)
    expected_missing = {
        name for name in student.state_dict()
        if name.startswith((
            "relation_adapter.", "relation_adapter_gate."))}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"unexpected modular insertion mismatch: "
            f"missing={missing}, unexpected={unexpected}")
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith((
            "relation_adapter.", "relation_adapter_gate.")))
    frozen_initial = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
        if not name.startswith((
            "relation_adapter.", "relation_adapter_gate."))
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-5)
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    for update in range(1, args.steps + 1):
        student.train()
        new_batch = generate_lifetimes(
            args.new_batch_size, 6,
            seed=args.seed * 10_000_000 + update,
            task="visible_context_xor", support_trials=1, device=device)
        binary_replay = generate_lifetimes(
            args.replay_batch_size, 6,
            seed=args.seed * 20_000_000 + update,
            task="binary_mapping", support_trials=1, device=device)
        context_replay = generate_lifetimes(
            args.replay_batch_size, 6,
            seed=args.seed * 30_000_000 + update,
            task="visible_context", support_trials=1, device=device)
        skill_loss = _new_skill_loss(
            student, new_batch, exploration=args.exploration)
        binary_loss = _trajectory_loss(
            student, teacher, binary_replay, feedback_trials=1,
            shuffled_teacher=args.shuffle_retention_teacher)
        context_loss = _trajectory_loss(
            student, teacher, context_replay, feedback_trials=1,
            shuffled_teacher=args.shuffle_retention_teacher)
        loss = (
            skill_loss
            + args.retention_weight * (binary_loss + context_loss))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if update in (1, args.steps):
            history.append({
                "update": update,
                "skill_loss": float(skill_loss.detach()),
                "binary_distillation_loss": float(binary_loss.detach()),
                "context_distillation_loss": float(context_loss.detach()),
                "total_loss": float(loss.detach()),
            })

    evaluations = {
        "new_skill": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 90_000_000, device=device,
            task="visible_context_xor", feedback_trials=1),
        "binary_retention": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 91_000_000, device=device,
            task="binary_mapping", feedback_trials=1),
        "context_retention": evaluate(
            student, count=args.test_lifetimes, trials=6,
            seed=args.seed + 92_000_000, device=device,
            task="visible_context", feedback_trials=1),
    }
    cue_ablation_accuracy = _operation_cue_ablation_accuracy(
        student, count=args.test_lifetimes,
        seed=args.seed + 90_000_000, device=device)
    cue_causally_used = (
        cue_ablation_accuracy
        <= evaluations["new_skill"]["overall_accuracy"] - 0.15)
    accepted = all(
        evaluation["gate"]["accepted"]
        for evaluation in evaluations.values()) and cue_causally_used
    frozen_base_identical = all(
        torch.equal(frozen_initial[name], value.detach().cpu())
        for name, value in student.state_dict().items()
        if name in frozen_initial)
    total_lifetimes = args.steps * (
        args.new_batch_size + 2 * args.replay_batch_size)
    report = {
        "schema": "compounding-transfer-v1",
        "claim_boundary": (
            "New behavior learned from attempted-action verifier outcomes; "
            "retention uses only a frozen learned controller's opaque action "
            "distributions on newly rendered replay lifetimes."),
        "semantic_labels_used_for_training": False,
        "unattempted_correct_actions_used_as_targets": False,
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
            "binary_replay_lifetimes": (
                args.steps * args.replay_batch_size),
            "context_replay_lifetimes": (
                args.steps * args.replay_batch_size),
            "total_unique_lifetimes": total_lifetimes,
            "total_verifier_bits": total_lifetimes * 6,
            "optimizer_updates": args.steps,
        },
        "evaluations": evaluations,
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
            "admission_status": "three_skill_compounding_transfer",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_passed": accepted,
        "new_skill": evaluations["new_skill"]["overall_accuracy"],
        "binary": evaluations["binary_retention"]["normal"][
            "query_accuracy"],
        "context": evaluations["context_retention"]["overall_accuracy"],
        "total_unique_lifetimes": total_lifetimes,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
