"""Outcome-replay training arm for the four-item reader.

Each update is computed only from the controller's chosen binary action and
its scalar success outcome.  Repeating an already observed batch spends more
compute, not more verifier outcomes.  The default report is diagnostic; an
optional candidate checkpoint can be written for the promotion ladder.
"""
from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=44901)
    parser.add_argument("--unique-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-span", type=int, default=4)
    parser.add_argument("--target-anchor-focus", type=int, default=-1)
    parser.add_argument("--replay-updates", type=int, default=16)
    parser.add_argument(
        "--replay-chunk", type=int, default=0,
        help="target updates between rehearsal passes; zero means all at once")
    parser.add_argument(
        "--rehearsal-updates", type=int, default=0,
        help="outcome-replay updates per old-skill stream and target batch")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--nuisance-levels", default="0.135",
        help="comma-separated render nuisance levels, cycled per target batch")
    parser.add_argument(
        "--max-nuisance-step", type=float, default=0.0001,
        help="maximum adjacent curriculum increment; microscopic by default")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 256 or args.batch_size % 256:
        raise ValueError("batch size must be a multiple of 256")
    if args.replay_chunk < 0:
        raise ValueError("replay chunk cannot be negative")
    if args.target_span < 3:
        raise ValueError("target span must be at least 3")
    target_anchor_focus = (
        args.target_anchor_focus if args.target_anchor_focus >= 0
        else args.target_span - 2)
    if not 0 <= target_anchor_focus < args.target_span - 1:
        raise ValueError("target anchor focus must identify a non-final item")
    nuisance_levels = tuple(
        float(value.strip()) for value in args.nuisance_levels.split(",")
        if value.strip())
    if not nuisance_levels or any(value < 0.0 for value in nuisance_levels):
        raise ValueError("nuisance levels must contain nonnegative floats")
    if args.max_nuisance_step <= 0.0:
        raise ValueError("max nuisance step must be positive")
    if any(
            right < left or right - left > args.max_nuisance_step + 1e-12
            for left, right in zip(nuisance_levels, nuisance_levels[1:])):
        raise ValueError(
            "nuisance curriculum must be nondecreasing with microscopic "
            f"steps <= {args.max_nuisance_step:g}")
    compute = configure_compute(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    parent_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_configuration = dict(parent_payload["model_configuration"])
    model_configuration.update({"action_adapter_width": 64, "action_adapter_gated": False})
    model = _load(args.checkpoint, device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1e-5)
    nuisance = nuisance_from_level(nuisance_levels[0])
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
        batch_nuisance = nuisance_from_level(
            nuisance_levels[batch_index % len(nuisance_levels)])
        batch = generate_procedural_shape_batch(
            args.batch_size, span=args.target_span, vocabulary=2,
            seed=args.seed + batch_index, nuisance=batch_nuisance,
            objective="recognition", query_count=1, next_query_stage=2,
            next_query_anchor_focus=target_anchor_focus,
            next_query_target_aligned=True,
            device=device)
        rehearsals = [
            generate_procedural_shape_batch(
                args.batch_size, vocabulary=2,
                seed=args.seed + 10_000 + batch_index * 10 + stream_index,
                nuisance=batch_nuisance, objective="recognition", device=device,
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
    target = evaluate_procedural_shape_span(
        model, count=max(1024, args.batch_size), span=args.target_span,
        vocabulary=2, seed=args.seed + 100_000,
        nuisance=nuisance, device=device, objective="recognition",
        query_count=1, next_query_stage=2,
        next_query_anchor_focus=target_anchor_focus,
        next_query_target_aligned=True, query_thought_steps=1)
    old_kwargs = dict(
        count=768, span=3, vocabulary=2, seed=args.seed + 200_000,
        nuisance=nuisance, device=device, objective="recognition",
        query_count=3, next_query_stage=1)
    old = evaluate_procedural_shape_span(
        model, **old_kwargs, query_thought_steps=0)
    old_thought = evaluate_procedural_shape_span(
        model, **old_kwargs, query_thought_steps=1)
    candidate_checkpoint = None
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            # Keep the canonical loader schema; provenance below distinguishes
            # this candidate from an ordinary fully trained checkpoint.
            "schema": "unified-cognitive-controller-v1",
            "provenance": "span-replay-candidate",
            "model_configuration": model_configuration,
            "state_dict": model.state_dict(),
            "parent_checkpoint": str(args.checkpoint),
            "training": vars(args),
            "audit_snapshot": {"target": target, "old_span3": old},
        }, args.checkpoint_out)
        candidate_checkpoint = str(args.checkpoint_out)
    report = {
        "schema": "span-four-outcome-replay-training-v1",
        "diagnostic_only": args.checkpoint_out is None,
        "agent_weights_promoted": False,
        "candidate_checkpoint": candidate_checkpoint,
        "compute": compute,
        "learner_visible_information": "RGB, own binary action, scalar outcome",
        "unique_target_outcomes": args.unique_batches * args.batch_size,
        "unique_rehearsal_outcomes": (
            args.unique_batches * len(old_streams) * args.batch_size),
        "replay_optimizer_updates": args.unique_batches * args.replay_updates,
        "rehearsal_optimizer_updates": (
            args.unique_batches * len(old_streams) * args.rehearsal_updates
            * math.ceil(args.replay_updates / (args.replay_chunk or args.replay_updates))),
        "loss_first": losses[0], "loss_last": losses[-1],
        "target_span": args.target_span,
        "target_anchor_focus": target_anchor_focus,
        "training_nuisance_levels": nuisance_levels,
        "evaluation_nuisance_level": nuisance_levels[0],
        "target": target, "old_span3": old,
        "old_span3_thought1": old_thought,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "unique_target_outcomes": report["unique_target_outcomes"],
        "unique_rehearsal_outcomes": report["unique_rehearsal_outcomes"],
        "replay_optimizer_updates": report["replay_optimizer_updates"],
        "loss_first": report["loss_first"], "loss_last": report["loss_last"],
        "target_next_conflict": target["next_conflict_accuracy"],
        "target_next": target["accuracy_by_operation"][2],
        "old_span3": old["accuracy"],
    }), flush=True)


if __name__ == "__main__":
    main()
