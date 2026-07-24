"""Decode the demonstrated first/last rule from frozen compact memory."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import torch

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent

from .consolidator import LatentConsolidator
from .environment import (
    SHOTS, generate_compositional_temporal_attention_lifetime,
    generate_temporal_attention_lifetime)
from .probe_temporal_order import _fit_probe
from .train import _append, seed_everything
from .train_consolidator import _initial_memory


def _load(controller_path, consolidator_path, device, *, transfer_paths=None,
          transfer_strength=0.0):
    payload = torch.load(controller_path, map_location=device, weights_only=False)
    config = payload.get("controller_arguments", payload.get("arguments"))
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
        latest_row_answer_entropy_threshold=config.get(
            "latest_row_answer_entropy_threshold"),
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
        model.event_binding_module.pairwise_transfer.strength.data.fill_(transfer_strength)
    compact_payload = torch.load(consolidator_path, map_location=device, weights_only=False)
    consolidator = LatentConsolidator(model.hidden, heads=config["heads"]).to(device)
    consolidator.load_state_dict(compact_payload["consolidator"])
    return model.eval(), consolidator.eval()


@torch.no_grad()
def _extract(model, consolidator, *, start, lifetimes, batch_size, heldout,
             feedback_mode, device, preserve_raw_write=False,
             generator=generate_temporal_attention_lifetime):
    collected = {shots: {"object_1_state": [], "object_2_state": [],
                         "feedback_state": [], "support_sequence": [],
                         "prewrite_state": [],
                         "raw_write_key": [], "raw_write_value": [],
                         "raw_write_row": [], "raw_write_first": [],
                         "raw_write_history": [], "memory_key": [],
                         "memory_value": [], "memory_row": [], "recalled": [],
                         "latest_row_feature": []}
                 for shots in SHOTS}
    labels = []
    captured: dict[str, torch.Tensor] = {}
    write_handle = model.write_key.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("state", inputs[0].detach()))
    observation_handle = model.observation_head.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("observations", inputs[0].detach()))
    latest_handle = None
    if hasattr(model, "latest_row_project"):
        latest_handle = model.latest_row_project.register_forward_hook(
            lambda _module, _inputs, output:
            captured.__setitem__("latest_row_feature", output.detach()))
    for offset in range(0, lifetimes, batch_size):
        count = min(batch_size, lifetimes - offset)
        items = [generator(
            start + offset + index, heldout=heldout,
            feedback_mode=feedback_mode) for index in range(count)]
        labels.append(torch.tensor([item.rule for item in items], dtype=torch.long))
        memory = _initial_memory(model, items, device)
        support_states = torch.zeros(count, 3, model.hidden, device=device)
        prewrite_state = torch.zeros(count, model.hidden, device=device)
        raw_key = torch.zeros(count, model.hidden, device=device)
        raw_value = torch.zeros_like(raw_key)
        raw_write_rows = [
            torch.zeros(count, model.hidden * 2, device=device)
            for _ in range(max(SHOTS))
        ]
        cursor = 0
        for shots in SHOTS:
            while cursor < shots:
                appended, _, _ = _append(
                    model, [item.supports[cursor] for item in items], memory, device)
                raw_key = appended.keys[:, -1]
                raw_value = appended.values[:, -1]
                raw_write_rows[cursor] = torch.cat((raw_key, raw_value), dim=-1)
                raw_strength = appended.strengths[:, -1]
                prewrite_state = captured["state"]
                support_states = captured["observations"]
                memory = consolidator(appended)
                if preserve_raw_write:
                    memory = memory.append(
                        raw_key, raw_value, raw_strength,
                        torch.ones_like(raw_strength))
                cursor += 1
            episodes = [item.future_queries[0] for item in items]
            from experiments.syllogimous_latent_agent.data import collate_episodes
            batch = collate_episodes(episodes)
            query = model.retrieval_summary(
                batch["frames"].to(device), batch["pcm"].to(device),
                batch["mask"].to(device))
            recalled, _ = memory.read(
                query, model.read_top_k, model.log_read_scale.exp().clamp(max=100.0))
            key = memory.keys[:, -1]
            value = memory.values[:, -1]
            features = {"object_1_state": support_states[:, 0],
                        "object_2_state": support_states[:, 1],
                        "feedback_state": support_states[:, 2],
                        "support_sequence": support_states.flatten(1),
                        "prewrite_state": prewrite_state,
                        "raw_write_key": raw_key, "raw_write_value": raw_value,
                        "raw_write_row": torch.cat((raw_key, raw_value), dim=-1),
                        "raw_write_first": raw_write_rows[0],
                        "raw_write_history": torch.cat(raw_write_rows, dim=-1),
                        "memory_key": key, "memory_value": value,
                        "memory_row": torch.cat((key, value), dim=-1),
                        "recalled": recalled,
                        "latest_row_feature": captured.get(
                            "latest_row_feature", torch.zeros_like(recalled))}
            for name, feature in features.items():
                collected[shots][name].append(feature.cpu())
    write_handle.remove()
    observation_handle.remove()
    if latest_handle is not None:
        latest_handle.remove()
    return ({shots: {name: torch.cat(parts) for name, parts in taps.items()}
             for shots, taps in collected.items()}, torch.cat(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--feedback-mode", choices=("white-button", "color-button"),
                        default="white-button")
    parser.add_argument(
        "--temporal-stage", choices=("atom", "compositional"),
        default="atom")
    parser.add_argument(
        "--atom-color-ids", default="0,1",
        help="Two displayed color IDs for the atom's otherwise unchanged palette.")
    parser.add_argument("--shots", default=",".join(str(value) for value in SHOTS))
    parser.add_argument("--taps", default="raw_write_row,memory_row,recalled")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path)
    parser.add_argument("--projection-transfer-checkpoint", type=Path)
    parser.add_argument("--transfer-strength", type=float, default=0.0)
    parser.add_argument("--preserve-raw-write", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    transfer_paths = None
    if args.pairwise_transfer_checkpoint or args.projection_transfer_checkpoint:
        if not (args.pairwise_transfer_checkpoint and args.projection_transfer_checkpoint):
            raise ValueError("both transfer checkpoints are required")
        transfer_paths = (str(args.pairwise_transfer_checkpoint),
                          str(args.projection_transfer_checkpoint))
    model, consolidator = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfer_paths, transfer_strength=args.transfer_strength)
    generator = {
        "atom": generate_temporal_attention_lifetime,
        "compositional":
            generate_compositional_temporal_attention_lifetime,
    }[args.temporal_stage]
    if args.temporal_stage == "atom":
        atom_color_ids = tuple(int(value) for value in args.atom_color_ids.split(","))
        if len(atom_color_ids) != 2:
            raise ValueError("atom-color-ids must contain exactly two integers")
        generator = partial(generator, color_ids=atom_color_ids)
    selected_shots = tuple(int(value) for value in args.shots.split(","))
    if not set(selected_shots).issubset(SHOTS):
        raise ValueError(f"shots must be selected from {SHOTS}")
    selected_taps = tuple(value for value in args.taps.split(",") if value)
    train_x, train_y = _extract(
        model, consolidator, start=11_000_000, lifetimes=args.train_lifetimes,
        batch_size=args.batch_size, heldout=False,
        feedback_mode=args.feedback_mode, device=device,
        preserve_raw_write=args.preserve_raw_write,
        generator=generator)
    test_x, test_y = _extract(
        model, consolidator, start=13_000_000, lifetimes=args.test_lifetimes,
        batch_size=args.batch_size, heldout=True,
        feedback_mode=args.feedback_mode, device=device,
        preserve_raw_write=args.preserve_raw_write,
        generator=generator)
    results = {}
    for shots in selected_shots:
        results[str(shots)] = {}
        unknown_taps = set(selected_taps) - set(train_x[shots])
        if unknown_taps:
            raise ValueError(f"unknown taps: {sorted(unknown_taps)}")
        for tap in selected_taps:
            results[str(shots)][tap] = {
                "linear": _fit_probe(train_x[shots][tap], train_y,
                                     test_x[shots][tap], test_y, nonlinear=False,
                                     device=device, seed=args.seed),
                "mlp": _fit_probe(train_x[shots][tap], train_y,
                                  test_x[shots][tap], test_y, nonlinear=True,
                                  device=device, seed=args.seed),
            }
    report = {"schema": "temporal-rule-memory-probe-v1", "controller_frozen": True,
              "consolidator_frozen": True, "visual_only": True,
              "feedback_mode": args.feedback_mode,
              "temporal_stage": args.temporal_stage,
              "pairwise_transfer": bool(transfer_paths),
              "transfer_strength": args.transfer_strength,
              "selected_shots": selected_shots, "selected_taps": selected_taps,
              "train_examples": int(train_y.numel()), "test_examples": int(test_y.numel()),
              "train_rule_rate": float(train_y.float().mean()),
              "test_rule_rate": float(test_y.float().mean()), "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
