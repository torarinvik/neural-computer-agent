"""Probe both sides of the four-identity mapping interface.

Verifier labels train throwaway probes only. The frozen agent sees the normal
visual stream, never colors, mappings, actions, or game state directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import collate_episodes
from experiments.syllogimous_neural_computer.training_memory import (
    DifferentiableBatchMemory)

from .environment import generate_compositional_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _forward, seed_everything


@torch.no_grad()
def _extract(model, *, start, lifetimes, batch_size, heldout, device,
             query_tap: str = "in-situ-recurrent",
             reader_query_surface: str = "audit-card"):
    if query_tap not in ("in-situ-recurrent", "pre-memory-sensory"):
        raise ValueError(f"unknown query tap {query_tap!r}")
    if reader_query_surface not in (
            "audit-card", "temporal-event", "mixed"):
        raise ValueError(
            f"unknown reader query surface {reader_query_surface!r}")
    study_rows = {0: [], 1: []}
    study_targets = {(card, slot): [] for card in range(2) for slot in range(2)}
    query_states, query_colors = [], []
    reader_rows, reader_queries, reader_actions = [], [], []
    captured = []
    handle = model.observation_head.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach()))
    try:
        for offset in range(0, lifetimes, batch_size):
            count = min(batch_size, lifetimes - offset)
            items = [
                generate_compositional_temporal_attention_lifetime(
                    start + offset + index, heldout=heldout, query_count=1,
                    feedback_mode="color-button")
                for index in range(count)]
            empty = DifferentiableBatchMemory(
                count, model.hidden, device=device)
            batch_rows = []
            query_memory = empty
            for card in range(2):
                output, _ = _forward(
                    model, [item.studies[card] for item in items],
                    query_memory, device)
                row = output.write_keys.detach().cpu()
                batch_rows.append(row)
                study_rows[card].append(row)
                for slot in range(2):
                    color = card * 2 + slot
                    study_targets[(card, slot)].append(torch.tensor(
                        [item.color_mapping[color] for item in items],
                        dtype=torch.long))
                query_memory = query_memory.append(
                    output.write_keys, output.write_values,
                    output.write_strengths,
                    torch.ones_like(output.write_strengths))
            for color in range(4):
                captured.clear()
                query_episodes = [
                    item.old_audit_queries[color] for item in items]
                _forward(
                    model, query_episodes, query_memory, device)
                if query_tap == "in-situ-recurrent":
                    query_state = captured[-1][:, 0]
                else:
                    batch = collate_episodes(query_episodes)
                    query_state = model.sensory_summary(
                        batch["frames"].to(device),
                        batch["pcm"].to(device),
                        batch["mask"].to(device))
                query_states.append(query_state.cpu())
                query_colors.append(torch.full(
                    (count,), color, dtype=torch.long))
                if reader_query_surface in ("audit-card", "mixed"):
                    reader_rows.append(torch.stack(batch_rows, dim=1))
                    reader_queries.append(query_state.cpu())
                    reader_actions.append(torch.tensor(
                        [item.color_mapping[color] for item in items],
                        dtype=torch.long))
            if reader_query_surface in ("temporal-event", "mixed"):
                temporal_episodes = [
                    item.future_queries[0] for item in items]
                batch = collate_episodes(temporal_episodes)
                event_states = model._encode(
                    batch["frames"].to(device), batch["pcm"].to(device))
                for position in range(2):
                    reader_rows.append(torch.stack(batch_rows, dim=1))
                    reader_queries.append(event_states[:, position].cpu())
                    reader_actions.append(torch.tensor([
                        item.color_mapping[
                            item.query_features[0][position]]
                        for item in items], dtype=torch.long))
    finally:
        handle.remove()
    return {
        "study_rows": {
            card: torch.cat(parts) for card, parts in study_rows.items()},
        "study_targets": {
            key: torch.cat(parts) for key, parts in study_targets.items()},
        "query_states": torch.cat(query_states),
        "query_colors": torch.cat(query_colors),
        "reader_rows": torch.cat(reader_rows),
        "reader_queries": torch.cat(reader_queries),
        "reader_actions": torch.cat(reader_actions),
    }


def _fit(train_x, train_y, test_x, test_y, *, classes, seed, device):
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    probe = nn.Sequential(
        nn.Linear(train_x.shape[1], 64), nn.GELU(), nn.Linear(64, classes)
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=3e-3, weight_decay=1e-3)
    best = 0.0
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            best = max(best, float((
                probe(test_x).argmax(-1) == test_y).float().mean()))
    with torch.no_grad():
        return {
            "train_accuracy": float((
                probe(train_x).argmax(-1) == train_y).float().mean()),
            "test_accuracy": float((
                probe(test_x).argmax(-1) == test_y).float().mean()),
            "best_test_accuracy": best,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--train-lifetimes", type=int, default=512)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=569)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    model, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    train = _extract(
        model, start=59_000_000, lifetimes=args.train_lifetimes,
        batch_size=args.batch_size, heldout=False, device=device)
    test = _extract(
        model, start=61_000_000, lifetimes=args.test_lifetimes,
        batch_size=args.batch_size, heldout=True, device=device)
    study = {}
    for card in range(2):
        for slot in range(2):
            key = (card, slot)
            study[f"card_{card + 1}_slot_{slot + 1}_action"] = _fit(
                train["study_rows"][card], train["study_targets"][key],
                test["study_rows"][card], test["study_targets"][key],
                classes=8, seed=args.seed + card * 2 + slot,
                device=device)
    report = {
        "schema": "mapping-representation-probe-v1",
        "weights_frozen": True,
        "sensory_only": True,
        "diagnostic_labels_visible_to_probes_only": True,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "study_write_action_probes": study,
        "query_color_identity_probe": _fit(
            train["query_states"], train["query_colors"],
            test["query_states"], test["query_colors"],
            classes=4, seed=args.seed + 10, device=device),
        "chance": {"study_action": 1 / 8, "query_color": 1 / 4},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
