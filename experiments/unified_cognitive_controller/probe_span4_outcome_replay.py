"""Diagnostic: can replayed scalar outcomes train a four-item reader?

Each update is computed only from the controller's chosen binary action and
its scalar success outcome.  Repeating an already observed batch spends more
compute, not more verifier outcomes.  This is a disposable credit-assignment
experiment: its weights are not promoted into the controller checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import seed_everything
from .train_procedural_shape_span import (
    evaluate_procedural_shape_span,
    generate_procedural_shape_batch,
    nuisance_from_level,
    rollout_procedural_shape_span,
)


def _load(checkpoint: Path, device: torch.device) -> UnifiedCognitiveController:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration.update({"action_adapter_width": 64, "action_adapter_gated": False})
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("action_adapter"))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44901)
    parser.add_argument("--unique-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-updates", type=int, default=16)
    parser.add_argument(
        "--replay-chunk", type=int, default=0,
        help="target updates between rehearsal passes; zero means all at once")
    parser.add_argument(
        "--rehearsal-updates", type=int, default=0,
        help="outcome-replay updates per old-skill stream and target batch")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 256 or args.batch_size % 256:
        raise ValueError("batch size must be a multiple of 256")
    if args.replay_chunk < 0:
        raise ValueError("replay chunk cannot be negative")
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = _load(args.checkpoint, device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1e-5)
    nuisance = nuisance_from_level(0.135)
    losses = []

    def update(
            batch, *, thought_steps: int, repeat: int) -> None:
        for _ in range(repeat):
            result = rollout_procedural_shape_span(
                model, batch, sample_actions=True, exploration=0.10,
                query_thought_steps=thought_steps, complete_binary_outcomes=True,
                next_conflict_novelty_weight=3.0,
                next_nonconflict_novelty_weight=2.0)
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            optimizer.step()
            losses.append(float(result["loss"].detach()))

    old_streams = (
        dict(span=3, query_count=1, new_slot_difficulty=0.0,
             previous_query_stage=-1, next_query_stage=2),
        dict(span=3, query_count=3, new_slot_difficulty=1.0,
             previous_query_stage=-1, next_query_stage=1),
        dict(span=3, query_count=3, new_slot_difficulty=1.0,
             previous_query_stage=2, next_query_stage=-1),
    )
    for batch_index in range(args.unique_batches):
        batch = generate_procedural_shape_batch(
            args.batch_size, span=4, vocabulary=2,
            seed=args.seed + batch_index, nuisance=nuisance,
            objective="recognition", query_count=1, next_query_stage=2,
            next_query_anchor_focus=2, next_query_target_aligned=True,
            device=device)
        rehearsals = [
            generate_procedural_shape_batch(
                args.batch_size, vocabulary=2,
                seed=args.seed + 10_000 + batch_index * 10 + stream_index,
                nuisance=nuisance, objective="recognition", device=device,
                **stream)
            for stream_index, stream in enumerate(old_streams)]
        chunk = args.replay_chunk or args.replay_updates
        for begin in range(0, args.replay_updates, chunk):
            update(
                batch, thought_steps=1,
                repeat=min(chunk, args.replay_updates - begin))
            for rehearsal in rehearsals:
                update(
                    rehearsal, thought_steps=0, repeat=args.rehearsal_updates)
    span4 = evaluate_procedural_shape_span(
        model, count=1024, span=4, vocabulary=2, seed=args.seed + 100_000,
        nuisance=nuisance, device=device, objective="recognition",
        query_count=1, next_query_stage=2, next_query_anchor_focus=2,
        next_query_target_aligned=True, query_thought_steps=1)
    old_kwargs = dict(
        count=768, span=3, vocabulary=2, seed=args.seed + 200_000,
        nuisance=nuisance, device=device, objective="recognition",
        query_count=3, next_query_stage=1)
    old = evaluate_procedural_shape_span(
        model, **old_kwargs, query_thought_steps=0)
    old_thought = evaluate_procedural_shape_span(
        model, **old_kwargs, query_thought_steps=1)
    report = {
        "schema": "span-four-outcome-replay-diagnostic-v1",
        "diagnostic_only": True,
        "agent_weights_promoted": False,
        "learner_visible_information": "RGB, own binary action, scalar outcome",
        "unique_target_outcomes": args.unique_batches * args.batch_size,
        "unique_rehearsal_outcomes": (
            args.unique_batches * len(old_streams) * args.batch_size),
        "replay_optimizer_updates": args.unique_batches * args.replay_updates,
        "rehearsal_optimizer_updates": (
            args.unique_batches * len(old_streams) * args.rehearsal_updates
            * math.ceil(args.replay_updates / (args.replay_chunk or args.replay_updates))),
        "loss_first": losses[0], "loss_last": losses[-1],
        "span4": span4, "old_span3": old, "old_span3_thought1": old_thought,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "unique_target_outcomes": report["unique_target_outcomes"],
        "unique_rehearsal_outcomes": report["unique_rehearsal_outcomes"],
        "replay_optimizer_updates": report["replay_optimizer_updates"],
        "loss_first": report["loss_first"], "loss_last": report["loss_last"],
        "span4_next_conflict": span4["next_conflict_accuracy"],
        "span4_next": span4["accuracy_by_operation"][2],
        "old_span3": old["accuracy"],
    }), flush=True)


if __name__ == "__main__":
    main()
