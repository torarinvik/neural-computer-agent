"""Test a frozen spatial controller/consolidator on shape attention."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import torch

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent

from .consolidator import LatentConsolidator
from .environment import (generate_attention_lifetime,
                          generate_compositional_temporal_attention_lifetime,
                          generate_shape_attention_lifetime,
                          generate_temporal_attention_lifetime,
                          generate_temporal_first_lifetime,
                          generate_temporal_grounding_lifetime,
                          generate_temporal_last_lifetime)
from .train import seed_everything
from .train_consolidator import run_compaction_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eval-lifetimes", type=int, default=256)
    parser.add_argument("--lifetime-offset", type=int, default=0,
                        help="offset inside the deterministic held-out seed range")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=("intact", "empty", "shuffled", "garbage"),
                        default="intact")
    parser.add_argument("--primitive", choices=("spatial", "shape", "temporal"),
                        default="shape")
    parser.add_argument("--reverse-temporal-query", action="store_true")
    parser.add_argument("--counterfactual-temporal-labels", action="store_true")
    parser.add_argument("--temporal-stage",
                        choices=("meta", "grounded", "first", "last",
                                 "compositional"),
                        default="meta")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path)
    parser.add_argument("--projection-transfer-checkpoint", type=Path)
    parser.add_argument("--transfer-strength", type=float, default=0.0)
    parser.add_argument("--answer-entropy-threshold", type=float)
    parser.add_argument("--preserve-raw-write", action="store_true")
    parser.add_argument("--preserve-first-raw-write", action="store_true")
    parser.add_argument("--preserve-study-raw-writes", action="store_true")
    parser.add_argument("--per-lifetime-report", type=Path,
                        help="also save one metric record per evaluated lifetime")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.controller_checkpoint, map_location=device, weights_only=False)
    config = payload.get("controller_arguments", payload.get("arguments"))
    entropy_threshold = (args.answer_entropy_threshold
                         if args.answer_entropy_threshold is not None else
                         config.get("latest_row_answer_entropy_threshold"))
    transfer_paths = None
    if args.pairwise_transfer_checkpoint or args.projection_transfer_checkpoint:
        if not (args.pairwise_transfer_checkpoint and args.projection_transfer_checkpoint):
            raise ValueError("both transfer checkpoints are required")
        transfer_paths = (str(args.pairwise_transfer_checkpoint),
                          str(args.projection_transfer_checkpoint))
    model = NeuralComputerAgent(
        config["hidden"], config["workspace_slots"], config["heads"],
        config["thought_steps"], action_count=8, read_top_k=config["read_top_k"],
        order_routing=config.get("order_routing", False),
        write_binding=config.get("write_binding", False),
        event_binding=(config.get("event_binding", False) or transfer_paths is not None),
        event_binding_width=config.get("event_binding_width", 64),
        event_binding_write_pairs=config.get("event_binding_write_pairs", False),
        event_binding_pairwise_transfer=transfer_paths,
        latest_row_reader=config.get("latest_row_reader", False),
        latest_row_answer_fusion=config.get("latest_row_answer_fusion", False),
        latest_row_answer_gate=config.get("latest_row_answer_gate", False),
        latest_row_answer_entropy_threshold=entropy_threshold,
        latest_row_answer_pairwise=config.get(
            "latest_row_answer_pairwise", False),
        latest_row_answer_event_binding=config.get(
            "latest_row_answer_event_binding", False),
        latest_row_answer_event_linear=config.get(
            "latest_row_answer_event_linear", False),
        latest_row_answer_factorized_router=config.get(
            "latest_row_answer_factorized_router", False),
        latest_row_factorized_ood_threshold=config.get(
            "latest_row_factorized_ood_threshold"),
        event_indexed_memory_reader=config.get(
            "event_indexed_memory_reader", False),
        event_indexed_memory_reader_width=config.get(
            "event_indexed_memory_reader_width", 64),
        event_indexed_memory_reader_architecture=config.get(
            "event_indexed_memory_reader_architecture",
            "relation")).to(device)
    model.load_state_dict(payload["model"], strict=transfer_paths is None)
    if transfer_paths is not None:
        model.event_binding_module.pairwise_transfer.strength.data.fill_(args.transfer_strength)
    model.eval()
    if args.consolidator_checkpoint is None:
        if "consolidator" not in payload:
            raise ValueError("a consolidator checkpoint is required for controller-only files")
        compact_payload = payload
    else:
        compact_payload = torch.load(
            args.consolidator_checkpoint, map_location=device, weights_only=False)
    consolidator = LatentConsolidator(model.hidden, heads=config["heads"]).to(device)
    consolidator.load_state_dict(compact_payload["consolidator"])
    consolidator.eval()
    totals, seen = {}, 0
    per_lifetime = []
    start = 5_000_000 + args.seed * 10_000 + args.lifetime_offset
    temporal_generator = {
        "meta": generate_temporal_attention_lifetime,
        "grounded": generate_temporal_grounding_lifetime,
        "first": generate_temporal_first_lifetime,
        "last": generate_temporal_last_lifetime,
        # Match the direct-color feedback surface used by the promoted temporal
        # router. This first transfer test changes logical composition and
        # identity count, not the perceptual feedback primitive at the same time.
        "compositional": partial(
            generate_compositional_temporal_attention_lifetime,
            feedback_mode="color-button"),
    }[args.temporal_stage]
    generators = {
        "spatial": generate_attention_lifetime,
        "shape": generate_shape_attention_lifetime,
        "temporal": temporal_generator,
    }
    generator = generators[args.primitive]
    with torch.no_grad():
        step = 1 if args.per_lifetime_report is not None else args.batch_size
        for offset in range(0, args.eval_lifetimes, step):
            count = min(step, args.eval_lifetimes - offset)
            lifetimes = [generator(
                start + offset + index, heldout=True, query_count=args.query_count)
                for index in range(count)]
            if args.reverse_temporal_query:
                if args.primitive != "temporal":
                    raise ValueError("query reversal is only defined for temporal attention")
                reversed_lifetimes = []
                for item in lifetimes:
                    queries = []
                    for order, episode in zip(item.query_features,
                                              item.future_queries):
                        changes = {"frames": episode.frames[::-1].copy(),
                                   "pcm": episode.pcm[::-1].copy()}
                        if args.counterfactual_temporal_labels:
                            answer = item.color_mapping[order[1 - item.rule]]
                            changes["actions"] = episode.actions * 0 + answer
                        queries.append(replace(episode, **changes))
                    reversed_lifetimes.append(replace(
                        item, future_queries=tuple(queries)))
                lifetimes = reversed_lifetimes
            _, metrics = run_compaction_batch(
                model, consolidator, lifetimes, device, train=False,
                condition=args.condition, preserve_raw_write=args.preserve_raw_write,
                preserve_first_raw_write=args.preserve_first_raw_write,
                preserve_study_raw_writes=args.preserve_study_raw_writes)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
            if args.per_lifetime_report is not None:
                per_lifetime.append({"lifetime_index": offset,
                                     "lifetime_seed": start + offset,
                                     "metrics": metrics})
            seen += count
    result = {key: value / seen for key, value in totals.items()}
    report = {"schema": f"cross-primitive-{args.primitive}-attention-v1",
              "primitive": args.primitive, "sensory_only": True,
              "weights_frozen": True, "condition": args.condition,
              "reverse_temporal_query": args.reverse_temporal_query,
              "counterfactual_temporal_labels": args.counterfactual_temporal_labels,
              "pairwise_transfer": bool(transfer_paths),
              "transfer_strength": args.transfer_strength,
              "preserve_raw_write": args.preserve_raw_write,
              "preserve_first_raw_write": args.preserve_first_raw_write,
              "preserve_study_raw_writes": args.preserve_study_raw_writes,
              "evaluation": result, "seed": args.seed,
              "lifetime_offset": args.lifetime_offset}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.per_lifetime_report is not None:
        args.per_lifetime_report.parent.mkdir(parents=True, exist_ok=True)
        args.per_lifetime_report.write_text(
            json.dumps({"schema": "cross-primitive-per-lifetime-v1",
                        "primitive": args.primitive, "seed": args.seed,
                        "reverse_temporal_query": args.reverse_temporal_query,
                        "counterfactual_temporal_labels":
                            args.counterfactual_temporal_labels,
                        "records": per_lifetime}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
