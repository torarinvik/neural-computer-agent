"""Reward-train one row-local volatility gate on a trained sequence skill.

This is deliberately a tiny integration test.  The inherited controller and
the learned sequence adapter stay frozen; only the zero-initialized scalar
that biases writes toward high-volatility workspace rows is plastic.  The
controller receives only RGB events, opaque attempted actions, and the scalar
outcome of that attempt.  Shuffled outcomes are a matched causal control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .train import seed_everything
from .train_sequence_working_memory import (
    evaluate_sequence_memory, generate_sequence_memory_batch,
    rollout_sequence_memory)
from .model import UnifiedCognitiveController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--seed", type=int, default=33001)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--span", type=int, default=9)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--test-episodes", type=int, default=1024)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("steps and batch-size must be positive and even")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.parent, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"], workspace_volatility=True)
    model = UnifiedCognitiveController(**configuration).to(device)
    result = model.load_state_dict(payload["state_dict"], strict=False)
    if result.unexpected_keys or result.missing_keys != [
            "workspace_volatility_write_scale"]:
        raise RuntimeError(
            f"parent/configuration mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.workspace_volatility_write_scale is not None
    model.workspace_volatility_write_scale.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [model.workspace_volatility_write_scale],
        lr=args.learning_rate, weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    started = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        batch = generate_sequence_memory_batch(
            args.batch_size, span=args.span, distractors=args.distractors,
            seed=args.seed + step * args.batch_size, operation="mixed",
            position_augmentation=True, device=device)
        rollout = rollout_sequence_memory(
            model, batch, sample_actions=True, exploration=args.exploration,
            shuffle_outcomes=args.shuffle_outcomes, loss_mode="bce")
        optimizer.zero_grad(set_to_none=True)
        rollout["loss"].backward()
        torch.nn.utils.clip_grad_norm_([model.workspace_volatility_write_scale], 1.0)
        optimizer.step()
        history.append({
            "step": step,
            "loss": float(rollout["loss"].detach()),
            "training_accuracy": float(rollout["rewards"].mean()),
            "volatility_write_scale": float(
                model.workspace_volatility_write_scale.detach()),
        })
    model.eval()
    audit = evaluate_sequence_memory(
        model, count=args.test_episodes, span=args.span,
        distractors=args.distractors, seed=args.seed + 10_000_000,
        operation="mixed", device=device)
    report = {
        "schema": "sequence-workspace-volatility-v1",
        "claim_boundary": (
            "Only the row-local write scale changes; the inherited sequence "
            "controller and adapter remain frozen. Learner-visible inputs are "
            "RGB events, opaque actions, and scalar attempted outcomes."),
        "parent": str(args.parent),
        "configuration": configuration,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "span": args.span,
        "distractors": args.distractors,
        "outcomes_shuffled": args.shuffle_outcomes,
        "history": history,
        "audit": audit,
        "wall_seconds": perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint_out is not None and not args.shuffle_outcomes:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "accuracy": audit["accuracy"],
        "reverse_operation_accuracy": audit["reverse_operation_accuracy"],
        "blank_sequence_accuracy": audit["blank_sequence_accuracy"],
        "all_memory_reset_accuracy": audit["all_memory_reset_accuracy"],
        "volatility_write_scale": history[-1]["volatility_write_scale"],
        "outcomes_shuffled": args.shuffle_outcomes,
        "wall_seconds": report["wall_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
