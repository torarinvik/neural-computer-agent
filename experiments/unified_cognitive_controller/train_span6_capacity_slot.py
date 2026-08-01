"""Reward-only capacity bridge from span five to a direct sixth-item query."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import configure_compute, seed_everything
from .train_procedural_shape_span import (
    evaluate_procedural_shape_span,
    generate_procedural_shape_batch,
    nuisance_from_level,
    rollout_procedural_shape_span,
)


def load_student(parent: Path, device: torch.device):
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    inherited = tuple(configuration.get("skill_adapter_widths", ()))
    if inherited and inherited != (64,):
        raise ValueError("capacity bridge accepts only one width-64 slot")
    if not inherited:
        configuration.update({
            "skill_adapter_widths": (64,),
            "skill_adapter_reads_intention_from": 0,
            "skill_adapter_gate_mode": "relu",
        })
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"], strict=bool(inherited))
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
    parser.add_argument("--seed", type=int, default=85101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--target-nuisance", type=float, default=0.8)
    parser.add_argument("--independence", type=float, default=0.0)
    parser.add_argument("--independence-weight", type=float, default=20.0)
    parser.add_argument("--eval-count", type=int, default=4096)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1024 or args.batch_size % 1024:
        raise ValueError("batch size must be a multiple of 1024")
    if not 0.0 <= args.independence <= 1.0:
        raise ValueError("independence must be within [0, 1]")

    compute = configure_compute(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, configuration = load_student(args.parent, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-5)
    target_nuisance = nuisance_from_level(args.target_nuisance)
    old_nuisance = nuisance_from_level(0.1358)
    losses: list[float] = []

    def update(batch, thought_steps: int, new_slot_weight: float = 1.0) -> None:
        model.train()
        result = rollout_procedural_shape_span(
            model, batch, sample_actions=False,
            query_thought_steps=thought_steps,
            complete_binary_outcomes=True,
            new_slot_novelty_weight=new_slot_weight)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.75)
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    for step in range(args.passes):
        target = generate_procedural_shape_batch(
            args.batch_size, span=6, vocabulary=2,
            seed=args.seed + step * 1000, nuisance=target_nuisance,
            objective="recognition", query_count=1,
            new_slot_difficulty=args.independence,
            direct_query_ordinal=5, allow_partial_balance=True,
            device=device)
        update(target, thought_steps=0,
               new_slot_weight=args.independence_weight)
        # The learned slot must remain inert on the already-mastered streams.
        span5 = generate_procedural_shape_batch(
            args.batch_size, span=5, vocabulary=2,
            seed=args.seed + 10_000 + step, nuisance=target_nuisance,
            objective="recognition", query_count=1,
            next_query_stage=2, next_query_anchor_focus=3,
            next_query_target_aligned=True, device=device)
        old = generate_procedural_shape_batch(
            args.batch_size, span=3, vocabulary=2,
            seed=args.seed + 20_000 + step, nuisance=old_nuisance,
            objective="recognition", query_count=3,
            next_query_stage=1, device=device)
        update(span5, thought_steps=1)
        update(old, thought_steps=0)
        if (step + 1) % max(1, args.passes // 4) == 0:
            print(json.dumps({
                "pass": step + 1,
                "loss_last": losses[-1],
                "loss_target": losses[-3],
            }), flush=True)

    model.eval()
    target = evaluate_procedural_shape_span(
        model, count=args.eval_count, span=6, vocabulary=2,
        seed=args.seed + 30_000_000, nuisance=target_nuisance,
        device=device, objective="recognition", query_count=1,
        new_slot_difficulty=args.independence, direct_query_ordinal=5,
        query_thought_steps=0)
    span5 = evaluate_procedural_shape_span(
        model, count=args.eval_count, span=5, vocabulary=2,
        seed=args.seed + 31_000_000, nuisance=target_nuisance,
        device=device, objective="recognition", query_count=1,
        next_query_stage=2, next_query_anchor_focus=3,
        next_query_target_aligned=True, query_thought_steps=1)
    old = evaluate_procedural_shape_span(
        model, count=args.eval_count, span=3, vocabulary=2,
        seed=args.seed + 32_000_000, nuisance=old_nuisance,
        device=device, objective="recognition", query_count=3,
        next_query_stage=1, query_thought_steps=0)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "provenance": "span6-direct-capacity-slot-pilot",
        "promotion_status": "candidate-requires-full-audit",
        "model_configuration": configuration,
        "state_dict": model.state_dict(),
        "parent_checkpoint": str(args.parent),
        "training": vars(args),
    }, args.checkpoint_out)
    report = {
        "schema": "span6-direct-capacity-slot-pilot-v1",
        "compute": compute,
        "passes": args.passes,
        "independence": args.independence,
        "independence_weight": args.independence_weight,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "target_span6": target,
        "retention_span5": span5,
        "retention_span3": old,
        "checkpoint": str(args.checkpoint_out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passes": args.passes,
        "target_span6": target["accuracy"],
        "target_reset": target["all_memory_reset_accuracy"],
        "span5": span5["accuracy"],
        "span3": old["accuracy"],
    }), flush=True)


if __name__ == "__main__":
    main()
