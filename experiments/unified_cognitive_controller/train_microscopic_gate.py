"""Adaptive microscopic nuisance curriculum for the span-five reader.

Each nuisance level is trained only until its held-out capability gates pass.
The curriculum then advances by exactly 0.0001.  This keeps difficulty growth
and compute growth coupled to measured capability rather than a fixed update
budget.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .probe_span4_outcome_replay import _load
from .train import configure_compute, seed_everything
from .train_procedural_shape_span import (
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
OLD_NUISANCE_LEVEL = 0.1358


@torch.no_grad()
def gate_eval(model, *, level: float, seed: int, device: torch.device,
              batch_size: int, repeats: int) -> dict[str, object]:
    """Cheap held-out gate; full adversarial audit remains a final gate."""
    measurements = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat * 1_000_003
        target_nuisance = nuisance_from_level(level)
        # Retention must measure forgetting at the mastered old-task
        # difficulty, not the newly introduced nuisance.  Otherwise the gate
        # confounds perceptual difficulty with catastrophic forgetting.
        old_nuisance = nuisance_from_level(OLD_NUISANCE_LEVEL)
        target_batch = generate_procedural_shape_batch(
            batch_size, span=5, vocabulary=2, seed=repeat_seed,
            nuisance=target_nuisance, heldout=True, objective="recognition",
            query_count=1, next_query_stage=2, next_query_anchor_focus=3,
            next_query_target_aligned=True, device=device)
        target = rollout_procedural_shape_span(
            model, target_batch, sample_actions=False, query_thought_steps=1)
        conflict = (
            (target_batch.query_operations == 2)
            & (torch.gather(target_batch.sequence_identities, 1,
                            target_batch.query_ordinals)
               != torch.gather(target_batch.sequence_identities, 1,
                               target_batch.query_cue_ordinals)))
        # Use a much larger old-skill sample than the target gate.  Near the
        # 95% retention boundary, small episode counts have enough binomial
        # noise to trigger needless destructive retries.  This remains
        # verifier-only evaluation; it does not add training signal.
        old_batch = generate_procedural_shape_batch(
            max(4096, batch_size), span=3, vocabulary=2,
            seed=repeat_seed + 100_000,
            nuisance=old_nuisance, heldout=True, objective="recognition",
            query_count=3, next_query_stage=1, device=device)
        old = rollout_procedural_shape_span(
            model, old_batch, sample_actions=False, query_thought_steps=0)
        measurements.append({
            "overall": float(target["rewards"].mean()),
            "strict_conflict": float(target["rewards"][conflict].mean()),
            "old_span3": float(old["rewards"].mean()),
        })
    overall = min(item["overall"] for item in measurements)
    strict = min(item["strict_conflict"] for item in measurements)
    old_accuracy = min(item["old_span3"] for item in measurements)
    return {
        "overall": overall,
        "strict_conflict": strict,
        "old_span3": old_accuracy,
        "repeats": measurements,
        "passed": overall >= 0.90 and strict >= 0.85 and old_accuracy >= 0.95,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=45601)
    parser.add_argument("--start-level", type=float, default=0.1359)
    parser.add_argument("--end-level", type=float, default=0.1361)
    parser.add_argument("--step", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-updates-per-level", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--gate-repeats", type=int, default=2)
    parser.add_argument("--replay-chunk", type=int, default=2)
    parser.add_argument("--rehearsal-updates", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--target-loss-weight", type=float, default=1.0,
        help=(
            "weight for the new-level behavioral loss; set to 0 for a "
            "retention-only repair while keeping the same replay loop"))
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if abs(args.step - 0.0001) > 1e-12:
        raise ValueError("microscopic curriculum step must be exactly 0.0001")
    if args.end_level < args.start_level:
        raise ValueError("end level must not be below start level")
    if args.batch_size < 1024 or args.batch_size % 1024:
        raise ValueError("span-five batch size must be a multiple of 1024")
    if args.max_updates_per_level <= 0 or args.eval_every <= 0:
        raise ValueError("update and evaluation intervals must be positive")
    if args.gate_repeats < 2:
        raise ValueError("gate repeats must be at least two at the frontier")
    levels = [
        round(args.start_level + index * args.step, 4)
        for index in range(
            int(round((args.end_level - args.start_level) / args.step)) + 1)]
    compute = configure_compute(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    parent_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    configuration = dict(parent_payload["model_configuration"])
    model = _load(args.checkpoint, device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1e-5)
    history = []
    total_updates = 0

    def update(batch, *, thought_steps: int, repeat: int,
               loss_weight: float = 1.0) -> None:
        nonlocal total_updates
        model.train()
        for _ in range(repeat):
            result = rollout_procedural_shape_span(
                model, batch, sample_actions=True, exploration=0.10,
                query_thought_steps=thought_steps,
                complete_binary_outcomes=True,
                next_conflict_novelty_weight=3.0,
                next_nonconflict_novelty_weight=2.0)
            optimizer.zero_grad(set_to_none=True)
            (result["loss"] * loss_weight).backward()
            optimizer.step()
            total_updates += 1

    for level_index, level in enumerate(levels):
        nuisance = nuisance_from_level(level)
        old_nuisance = nuisance_from_level(OLD_NUISANCE_LEVEL)
        target_batch = generate_procedural_shape_batch(
            args.batch_size, span=5, vocabulary=2,
            seed=args.seed + level_index, nuisance=nuisance,
            objective="recognition", query_count=1, next_query_stage=2,
            next_query_anchor_focus=3, next_query_target_aligned=True,
            device=device)
        rehearsals = [
            generate_procedural_shape_batch(
                args.batch_size, vocabulary=2,
                seed=args.seed + 10_000 + level_index * 10 + stream_index,
                nuisance=old_nuisance, objective="recognition", device=device,
                **stream)
            for stream_index, stream in enumerate(OLD_STREAMS)]
        before = gate_eval(
            model, level=level, seed=args.seed + 100_000 + level_index,
            device=device, batch_size=args.batch_size,
            repeats=args.gate_repeats)
        updates_at_level = 0
        after = before
        while not bool(after["passed"]):
            if updates_at_level >= args.max_updates_per_level:
                break
            repeat = min(args.replay_chunk,
                         args.max_updates_per_level - updates_at_level)
            update(target_batch, thought_steps=1, repeat=repeat,
                   loss_weight=args.target_loss_weight)
            updates_at_level += repeat
            for rehearsal in rehearsals:
                update(rehearsal, thought_steps=0,
                       repeat=args.rehearsal_updates)
            if updates_at_level % args.eval_every == 0:
                after = gate_eval(
                    model, level=level,
                    seed=args.seed + 100_000 + level_index,
                    device=device, batch_size=args.batch_size,
                    repeats=args.gate_repeats)
        history.append({
            "level": level, "before": before, "after": after,
            "updates_at_level": updates_at_level,
            "passed": bool(after["passed"]),
        })
        if not bool(after["passed"]):
            break

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "provenance": "span5-adaptive-microscopic-curriculum",
        "promotion_status": "candidate-requires-full-adversarial-audit",
        "model_configuration": configuration,
        "state_dict": model.state_dict(),
        "parent_checkpoint": str(args.checkpoint),
        "training": vars(args),
        "history": history,
    }, args.checkpoint_out)
    report = {
        "schema": "span5-adaptive-microscopic-curriculum-v1",
        "compute": compute, "levels": history,
        "total_optimizer_updates": total_updates,
        "candidate_checkpoint": str(args.checkpoint_out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
if __name__ == "__main__":
    main()
