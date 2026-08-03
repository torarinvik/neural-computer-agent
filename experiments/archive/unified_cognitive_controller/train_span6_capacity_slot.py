"""Reward-only capacity bridge from span five to a direct sixth-item query."""
from __future__ import annotations

import argparse
from dataclasses import replace
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


def _subset_batch(batch, mask: torch.Tensor):
    """Keep verifier-selected episodes without exposing their labels."""
    fields = {
        name: getattr(batch, name)[mask]
        for name in (
            "presentation_frames", "query_frames", "correct_actions",
            "sequence_identities", "candidate_identities", "query_ordinals",
            "query_cue_ordinals", "query_operations", "new_slot_independent",
            "logical_lifetime_ids")}
    return replace(batch, **fields)


def _concat_batches(batches):
    if not batches:
        return None
    first = batches[0]
    fields = {
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in (
            "presentation_frames", "query_frames", "correct_actions",
            "sequence_identities", "candidate_identities", "query_ordinals",
            "query_cue_ordinals", "query_operations", "new_slot_independent",
            "logical_lifetime_ids")}
    return replace(first, **fields)


def load_student(
        parent: Path, device: torch.device, *, train_actuator: bool = False,
        read_event_snapshot: bool = False, slot_width: int = 64,
        workspace_slots: int | None = None, train_memory: bool = False,
        workspace_addressing: bool = False,
        train_action_adapter: bool = False):
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    inherited = tuple(configuration.get("skill_adapter_widths", ()))
    if inherited and inherited != (slot_width,):
        raise ValueError("capacity bridge accepts only one slot of the requested width")
    if not inherited:
        configuration.update({
            "skill_adapter_widths": (slot_width,),
            "skill_adapter_reads_intention_from": 0,
            "skill_adapter_gate_mode": "relu",
        })
        if read_event_snapshot:
            configuration["skill_adapter_reads_event_snapshot_from"] = 0
    if workspace_slots is not None:
        configuration["workspace_slots"] = workspace_slots
    if workspace_addressing:
        configuration["workspace_slot_addressing"] = True
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"], strict=bool(inherited))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("skill_adapters.0")
            or name.startswith("skill_adapter_gates.0")
            or (train_actuator and name.startswith("actuator."))
            or (train_action_adapter and name.startswith("action_adapter.")))
        if train_memory:
            parameter.requires_grad_(
                parameter.requires_grad
                or name.startswith("read_query.")
                or name.startswith("write_query.")
                or name.startswith("write_gate."))
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
    parser.add_argument(
        "--frontier-batch-size", type=int, default=0,
        help=("optional extra batch containing only independent sixth-item "
              "examples; used to test stratified sample efficiency"))
    parser.add_argument("--frontier-weight", type=float, default=1.0)
    parser.add_argument("--hard-example-focal-gamma", type=float, default=0.0)
    parser.add_argument("--slot-width", type=int, default=64)
    parser.add_argument("--gate-leak", type=float, default=0.0)
    parser.add_argument("--workspace-slots", type=int, default=0)
    parser.add_argument("--train-memory", action="store_true")
    parser.add_argument("--workspace-addressing", action="store_true")
    parser.add_argument(
        "--hard-replay-size", type=int, default=0,
        help="replay up to this many missed independent episodes per pass")
    parser.add_argument(
        "--transfer-span", type=int, default=0,
        help="optional easier direct-final-item span to rehearse as transfer")
    parser.add_argument(
        "--transfer-ordinal", type=int, default=-1,
        help="optional second direct retrieval ordinal within span six")
    parser.add_argument(
        "--counterfactual-pair", action="store_true",
        help="replay each target batch with candidates flipped")
    parser.add_argument(
        "--pretrain-spans", default="",
        help="comma-separated direct-retrieval spans to rehearse first")
    parser.add_argument("--pretrain-passes", type=int, default=0)
    parser.add_argument(
        "--train-actuator", action="store_true",
        help="also fine-tune the generic action reader with rehearsal")
    parser.add_argument("--train-action-adapter", action="store_true")
    parser.add_argument(
        "--read-event-snapshot", action="store_true",
        help="let a new slot read the raw current event embedding")
    parser.add_argument("--eval-count", type=int, default=4096)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1024 or args.batch_size % 1024:
        raise ValueError("batch size must be a multiple of 1024")
    if not 0.0 <= args.independence <= 1.0:
        raise ValueError("independence must be within [0, 1]")
    if args.frontier_batch_size < 0 or (
            args.frontier_batch_size and args.frontier_batch_size % 1024):
        raise ValueError("frontier batch size must be zero or a multiple of 1024")
    if args.frontier_weight < 1.0:
        raise ValueError("frontier weight must be at least one")
    if args.hard_example_focal_gamma < 0.0:
        raise ValueError("hard-example focal gamma must be non-negative")
    if args.transfer_span not in (0, 3, 4, 5):
        raise ValueError("transfer span must be zero, 3, 4, or 5")
    if args.transfer_ordinal not in (-1, 0, 1, 2, 3, 4):
        raise ValueError("transfer ordinal must be -1 or an ordinal in [0, 4]")
    if args.slot_width < 8:
        raise ValueError("slot width must be at least eight")
    if not 0.0 <= args.gate_leak <= 1.0:
        raise ValueError("gate leak must be within [0, 1]")
    if args.workspace_slots < 0:
        raise ValueError("workspace slots must be non-negative")
    pretrain_spans = tuple(
        int(value) for value in args.pretrain_spans.split(",")
        if value.strip())
    if any(span not in (3, 4, 5) for span in pretrain_spans):
        raise ValueError("pretrain spans must be 3, 4, or 5")
    if args.pretrain_passes < 0:
        raise ValueError("pretrain passes must be non-negative")
    if args.hard_replay_size < 0:
        raise ValueError("hard replay size must be non-negative")

    compute = configure_compute(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, configuration = load_student(
        args.parent, device, train_actuator=args.train_actuator,
        read_event_snapshot=args.read_event_snapshot, slot_width=args.slot_width,
        workspace_slots=(args.workspace_slots or None),
        train_memory=args.train_memory,
        workspace_addressing=args.workspace_addressing,
        train_action_adapter=args.train_action_adapter)
    model.skill_adapter_gate_leak = args.gate_leak
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-5)
    target_nuisance = nuisance_from_level(args.target_nuisance)
    old_nuisance = nuisance_from_level(0.1358)
    losses: list[float] = []
    hard_buffer = []

    def update(batch, thought_steps: int, new_slot_weight: float = 1.0):
        model.train()
        result = rollout_procedural_shape_span(
            model, batch, sample_actions=False,
            query_thought_steps=thought_steps,
            complete_binary_outcomes=True,
            new_slot_novelty_weight=new_slot_weight,
            hard_example_focal_gamma=args.hard_example_focal_gamma)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.75)
        optimizer.step()
        losses.append(float(result["loss"].detach()))
        return result

    for step in range(args.pretrain_passes):
        for span in pretrain_spans:
            pretrain = generate_procedural_shape_batch(
                args.batch_size, span=span, vocabulary=2,
                seed=args.seed + 800_000 + step * 10 + span,
                nuisance=target_nuisance, objective="recognition",
                query_count=1, new_slot_difficulty=1.0,
                direct_query_ordinal=span - 1,
                allow_partial_balance=True, device=device)
            update(pretrain, thought_steps=0, new_slot_weight=1.0)

    for step in range(args.passes):
        target = generate_procedural_shape_batch(
            args.batch_size, span=6, vocabulary=2,
            seed=args.seed + step * 1000, nuisance=target_nuisance,
            objective="recognition", query_count=1,
            new_slot_difficulty=args.independence,
            direct_query_ordinal=5, allow_partial_balance=True,
            device=device)
        target_result = update(
            target, thought_steps=0,
            new_slot_weight=args.independence_weight)
        if args.hard_replay_size:
            missed = (
                ~target_result["rewards"][:, 0].bool()
                & target.new_slot_independent)
            if bool(missed.any()):
                hard_buffer.append(_subset_batch(target, missed))
            replay = _concat_batches(hard_buffer)
            if replay is not None and replay.batch_size > args.hard_replay_size:
                replay = _subset_batch(
                    replay,
                    torch.arange(
                        replay.batch_size - args.hard_replay_size,
                        replay.batch_size, device=device))
            if replay is not None and replay.batch_size:
                update(replay, thought_steps=0,
                       new_slot_weight=args.independence_weight)
        if args.counterfactual_pair:
            counterfactual = generate_procedural_shape_batch(
                args.batch_size, span=6, vocabulary=2,
                seed=args.seed + step * 1000, nuisance=target_nuisance,
                objective="recognition", query_count=1,
                new_slot_difficulty=args.independence,
                direct_query_ordinal=5, flip_candidates=True,
                allow_partial_balance=True, device=device)
            update(counterfactual, thought_steps=0,
                   new_slot_weight=args.independence_weight)
        if args.frontier_batch_size:
            frontier = generate_procedural_shape_batch(
                args.frontier_batch_size, span=6, vocabulary=2,
                seed=args.seed + 500_000 + step, nuisance=target_nuisance,
                objective="recognition", query_count=1,
                new_slot_difficulty=1.0,
                direct_query_ordinal=5, allow_partial_balance=True,
                device=device)
            update(frontier, thought_steps=0,
                   new_slot_weight=args.frontier_weight)
        if args.transfer_span:
            transfer = generate_procedural_shape_batch(
                args.batch_size, span=args.transfer_span, vocabulary=2,
                seed=args.seed + 600_000 + step,
                nuisance=target_nuisance, objective="recognition",
                query_count=1, new_slot_difficulty=1.0,
                direct_query_ordinal=args.transfer_span - 1,
                allow_partial_balance=True, device=device)
            update(transfer, thought_steps=0, new_slot_weight=1.0)
        if args.transfer_ordinal >= 0:
            transfer = generate_procedural_shape_batch(
                args.batch_size, span=6, vocabulary=2,
                seed=args.seed + 700_000 + step, nuisance=target_nuisance,
                objective="recognition", query_count=1,
                new_slot_difficulty=1.0,
                direct_query_ordinal=args.transfer_ordinal,
                allow_partial_balance=True, device=device)
            update(transfer, thought_steps=0, new_slot_weight=1.0)
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
    model.skill_adapter_gate_leak = 0.0
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
        "frontier_batch_size": args.frontier_batch_size,
        "frontier_weight": args.frontier_weight,
        "hard_example_focal_gamma": args.hard_example_focal_gamma,
        "transfer_span": args.transfer_span,
        "transfer_ordinal": args.transfer_ordinal,
        "slot_width": args.slot_width,
        "gate_leak": args.gate_leak,
        "hard_replay_size": args.hard_replay_size,
        "counterfactual_pair": args.counterfactual_pair,
        "pretrain_spans": pretrain_spans,
        "pretrain_passes": args.pretrain_passes,
        "train_actuator": args.train_actuator,
        "read_event_snapshot": args.read_event_snapshot,
        "workspace_slots": args.workspace_slots,
        "train_memory": args.train_memory,
        "workspace_addressing": args.workspace_addressing,
        "train_action_adapter": args.train_action_adapter,
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
