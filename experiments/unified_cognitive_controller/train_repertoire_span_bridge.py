"""Add the first visual sequence-memory rung without dropping the repertoire.

Every optimizer update aggregates one new span-two batch with binary binding,
four-rule composition, pair relation, and persistent-recall rehearsal.  The
task scheduler is verifier-side; the controller receives no task identifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from .audit_pair_relation_repertoire import _load
from .train import attempted_success_loss, evaluate, seed_everything
from .train_persistent_memory import (
    _rehearsal_loss,
    evaluate_persistent,
    persistent_rollout,
)
from .train_procedural_shape_span import (
    evaluate_procedural_shape_span,
    generate_procedural_shape_batch,
    nuisance_from_level,
    rollout_procedural_shape_span,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=122_001)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=1024)
    parser.add_argument("--memory-capacity", type=int, default=8)
    parser.add_argument("--randomness", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--target-weight", type=float, default=1.0,
        help="relative span loss weight inside the complete repertoire update")
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--memory-temperature", type=float, default=50.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default=(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.batch_size % args.memory_capacity:
        raise ValueError("batch size must contain complete memory banks")
    if args.target_weight <= 0:
        raise ValueError("target weight must be positive")
    if args.test_count % 1024:
        raise ValueError("test count must be a positive multiple of 1024")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model = _load(args.checkpoint_in, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    nuisance = nuisance_from_level(args.randomness)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    history = []
    started = perf_counter()

    for step in range(1, args.steps + 1):
        data_seed = args.seed * 1_000_000 + step * 10
        target_batch = generate_procedural_shape_batch(
            args.batch_size, span=2, vocabulary=2, seed=data_seed,
            nuisance=nuisance, objective="recognition", query_count=2,
            device=device)
        target = rollout_procedural_shape_span(
            model, target_batch, sample_actions=True,
            exploration=args.exploration)
        persistent = persistent_rollout(
            model, count=args.batch_size, capacity=args.memory_capacity,
            seed=data_seed + 1, device=device, sample_actions=True,
            exploration=args.exploration, memory_mode="soft",
            memory_temperature=args.memory_temperature)
        persistent_loss = attempted_success_loss(
            persistent["logits"], persistent["actions"],
            persistent["outcomes"])
        binary_loss = _rehearsal_loss(
            model, task="binary_mapping", feedback_trials=1,
            count=args.batch_size, seed=data_seed + 2, device=device,
            exploration=args.exploration)
        four_rule_loss = _rehearsal_loss(
            model, task="four_rule", feedback_trials=2,
            count=args.batch_size, seed=data_seed + 3, device=device,
            exploration=args.exploration)
        appearance = ("bars", "diamonds", "dot_pairs")[(step - 1) % 3]
        relation_loss = _rehearsal_loss(
            model, task="pair_relation", feedback_trials=1,
            count=args.batch_size, seed=data_seed + 4, device=device,
            exploration=args.exploration, appearance=appearance)
        losses = torch.stack((
            target["loss"] * args.target_weight,
            persistent_loss, binary_loss, four_rule_loss, relation_loss))
        loss = losses.sum() / (args.target_weight + 4.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "target_accuracy": float(target["rewards"].mean()),
                "persistent_accuracy": float(persistent["outcomes"].mean()),
                "persistent_retrieval_top1": float(
                    persistent["retrieval_top1"].mean()),
                "losses": {
                    "target": float(target["loss"].detach()),
                    "persistent": float(persistent_loss.detach()),
                    "binary": float(binary_loss.detach()),
                    "four_rule": float(four_rule_loss.detach()),
                    "relation": float(relation_loss.detach()),
                },
                "elapsed_seconds": perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    model.eval()
    span = evaluate_procedural_shape_span(
        model, count=args.test_count, span=2, vocabulary=2,
        seed=args.seed + 90_000_000, nuisance=nuisance, device=device,
        objective="recognition", query_count=2)
    binary = evaluate(
        model, count=args.test_count, trials=6,
        seed=args.seed + 91_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.test_count, trials=6,
        seed=args.seed + 92_000_000, device=device,
        task="four_rule", feedback_trials=2)
    relations = {
        appearance: evaluate(
            model, count=args.test_count, trials=6,
            seed=args.seed + 93_000_000 + index * 10_000,
            device=device, task="pair_relation", feedback_trials=1,
            appearance=appearance)
        for index, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
    }
    persistent = evaluate_persistent(
        model, count=args.test_count, capacity=args.memory_capacity,
        seed=args.seed + 94_000_000, device=device)
    gates = {
        "span2_accuracy_at_least_90": float(span["accuracy"]) >= 0.90,
        "span2_memory_reset_at_chance":
            float(span["all_memory_reset_accuracy"]) <= 0.60,
        "span2_blank_presentation_at_chance":
            float(span["blank_presentation_accuracy"]) <= 0.60,
        "span2_candidate_flips_at_least_80": float(
            span["candidate_prediction_flip_rate_on_changed"]) >= 0.80,
        "span2_reversal_flips_at_least_80": float(
            span["reverse_prediction_flip_rate_on_changed"]) >= 0.80,
        "binary_retained": bool(binary["gate"]["accepted"]),
        "four_rule_retained": bool(four_rule["gate"]["accepted"]),
        "relation_retained": all(
            result["gate"]["accepted"] for result in relations.values()),
        "persistent_retained": bool(persistent["gate"]["accepted"]),
    }
    gates["accepted"] = all(gates.values())
    report = {
        "schema": "one-controller-span2-repertoire-bridge-v1",
        "learner_visible": [
            "rendered_rgb", "own_opaque_action", "scalar_outcome",
            "own_latent_state", "content_addressed_latent_read"],
        "semantic_task_ids_visible_to_learner": False,
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": str(args.checkpoint_out),
            "report": str(args.report)},
        "accounting": {
            "optimizer_updates": args.steps,
            "unique_span_lifetimes": args.steps * args.batch_size,
            "unique_span_verifier_bits": args.steps * args.batch_size * 2,
            "replayed_span_examples": 0,
        },
        "history": history,
        "span2": span,
        "binary": binary,
        "four_rule": four_rule,
        "relations": relations,
        "persistent": persistent,
        "gates": gates,
        "weights_changed": any(
            not torch.equal(initial[name], value.detach().cpu())
            for name, value in model.state_dict().items()),
        "total_seconds": perf_counter() - started,
    }
    if gates["accepted"]:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": payload["model_configuration"],
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accepted": gates["accepted"],
        "gates": gates,
        "span2_accuracy": span["accuracy"],
        "span2_memory_reset_accuracy": span["all_memory_reset_accuracy"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
