"""Tiny verifier-driven replay pilot for operation/intention binding.

This is deliberately a disposable adaptation arm.  The parent controller is
frozen and receives one fresh generic skill slot that reads the current event
and the inherited amodal intention.  Training uses only the controller's own
greedy action and the verifier's scalar success; the normal and operation-
counterfactual episodes are replayed as a small experience pool.  Old span
three streams are rehearsed after every pool pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .legacy_model import UnifiedCognitiveController
from .train import configure_compute, seed_everything
from .train_procedural_shape_span import (
    evaluate_procedural_shape_span,
    generate_procedural_shape_batch,
    nuisance_from_level,
    rollout_procedural_shape_span,
)


OLD_STREAMS = (
    dict(span=3, query_count=1, new_slot_difficulty=0.0,
         previous_query_stage=-1, next_query_stage=2),
    dict(span=3, query_count=3, new_slot_difficulty=1.0,
         previous_query_stage=-1, next_query_stage=1),
    dict(span=3, query_count=3, new_slot_difficulty=1.0,
         previous_query_stage=2, next_query_stage=-1),
)


def load_student(parent: Path, device: torch.device) -> tuple[
        UnifiedCognitiveController, dict[str, object]]:
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    inherited = tuple(configuration.get("skill_adapter_widths", ()))
    if inherited:
        raise ValueError(
            "this pilot expects the canonical parent without an operation slot")
    configuration.update({
        "skill_adapter_widths": (64,),
        "skill_adapter_reads_intention_from": 0,
        # Keep the new gate open while its zero-output adapter learns.  The
        # final projection is still zero-initialized, so insertion is exact.
        "skill_adapter_gate_mode": "relu",
    })
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("skill_adapters.0")
            or name.startswith("skill_adapter_gates.0"))
    return model, configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=80101)
    parser.add_argument("--pool-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replay-updates", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--target-conflict-weight", type=float, default=20.0,
        help=(
            "verifier-side emphasis on operation changes whose target identity "
            "differs from the direct cue; this is the binding signal"))
    parser.add_argument("--target-nuisance", type=float, default=0.8)
    parser.add_argument("--old-nuisance", type=float, default=0.1358)
    parser.add_argument("--eval-count", type=int, default=2048)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if args.pool_batches < 1 or args.replay_updates < 1:
        raise ValueError("pool size and replay updates must be positive")
    if args.batch_size < 1024 or args.batch_size % 1024:
        raise ValueError("batch size must be a multiple of 1024")
    if args.eval_count < 1024 or args.eval_count % 1024:
        raise ValueError("eval count must be a multiple of 1024")

    compute = configure_compute(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, configuration = load_student(args.parent, device)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-5)

    target_nuisance = nuisance_from_level(args.target_nuisance)
    old_nuisance = nuisance_from_level(args.old_nuisance)
    target_pool = []
    for pool_index in range(args.pool_batches):
        seed = args.seed + pool_index * 1_000_003
        common = dict(
            count=args.batch_size, span=5, vocabulary=2, seed=seed,
            nuisance=target_nuisance, objective="recognition", query_count=1,
            next_query_stage=2, next_query_anchor_focus=3,
            next_query_target_aligned=True, device=device)
        target_pool.append((
            generate_procedural_shape_batch(**common),
            generate_procedural_shape_batch(
                **common, flip_query_operations=True)))
    rehearsals = [
        generate_procedural_shape_batch(
            args.batch_size, vocabulary=2,
            seed=args.seed + 10_000 + index,
            nuisance=old_nuisance, objective="recognition", device=device,
            **stream)
        for index, stream in enumerate(OLD_STREAMS)]

    losses: list[float] = []

    def update(batch: object, *, thought_steps: int,
               conflict_weight: float = 1.0) -> None:
        model.train()
        result = rollout_procedural_shape_span(
            model, batch, sample_actions=False,
            query_thought_steps=thought_steps,
            complete_binary_outcomes=True,
            next_conflict_novelty_weight=conflict_weight,
            next_nonconflict_novelty_weight=(
                2.0 if conflict_weight > 1.0 else 1.0))
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.75)
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    for replay in range(args.replay_updates):
        for normal, counterfactual in target_pool:
            update(normal, thought_steps=1,
                   conflict_weight=args.target_conflict_weight)
            update(counterfactual, thought_steps=1,
                   conflict_weight=args.target_conflict_weight)
        for rehearsal in rehearsals:
            update(rehearsal, thought_steps=0)
        if (replay + 1) % max(1, args.replay_updates // 4) == 0:
            print(json.dumps({
                "replay": replay + 1,
                "loss_last": losses[-1],
                "loss_mean_last_pass": sum(losses[-(2 * args.pool_batches + 3):])
                / (2 * args.pool_batches + 3),
            }), flush=True)

    model.eval()
    target = evaluate_procedural_shape_span(
        model, count=args.eval_count, span=5, vocabulary=2,
        seed=args.seed + 20_000_000, nuisance=target_nuisance,
        device=device, objective="recognition", query_count=1,
        next_query_stage=2, next_query_anchor_focus=3,
        next_query_target_aligned=True, query_thought_steps=1)
    old = evaluate_procedural_shape_span(
        model, count=args.eval_count, span=3, vocabulary=2,
        seed=args.seed + 21_000_000, nuisance=old_nuisance,
        device=device, objective="recognition", query_count=3,
        next_query_stage=1, query_thought_steps=0)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "provenance": "operation-binding-replay-pilot",
        "promotion_status": "candidate-requires-full-audit",
        "model_configuration": configuration,
        "state_dict": model.state_dict(),
        "parent_checkpoint": str(args.parent),
        "training": vars(args),
    }, args.checkpoint_out)
    report = {
        "schema": "operation-binding-replay-pilot-v1",
        "compute": compute,
        "pool_batches": args.pool_batches,
        "pool_unique_lifetimes": args.pool_batches * args.batch_size,
        "replay_updates": args.replay_updates,
        "target_experiences": (
            args.pool_batches * args.batch_size * args.replay_updates * 2),
        "learner_visible_verifier_bits": 0,
        "sample_actions": False,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "target": target,
        "old_span3": old,
        "checkpoint": str(args.checkpoint_out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "replay_updates": args.replay_updates,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "operation_flip_accuracy": target[
            "operation_flip_accuracy"],
        "operation_flip_rate": target[
            "operation_prediction_flip_rate_on_changed"],
        "old_span3": old["accuracy"],
    }), flush=True)


if __name__ == "__main__":
    main()
