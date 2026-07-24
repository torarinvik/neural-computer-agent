"""Diagnose whether frozen controller latents retain visual stream order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import collate_episodes
from experiments.syllogimous_neural_computer.model import NeuralComputerAgent
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory

from .environment import generate_temporal_attention_lifetime
from .train import seed_everything


def _load_model(path: Path, device: torch.device) -> NeuralComputerAgent:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("controller_arguments", payload.get("arguments"))
    model = NeuralComputerAgent(
        config["hidden"], config["workspace_slots"], config["heads"],
        config["thought_steps"], action_count=8, read_top_k=config["read_top_k"],
        order_routing=config.get("order_routing", False),
        write_binding=config.get("write_binding", False),
        event_binding=config.get("event_binding", False),
        event_binding_width=config.get("event_binding_width", 64),
        event_binding_write_pairs=config.get("event_binding_write_pairs", False)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def _features(model: NeuralComputerAgent, *, start: int, lifetimes: int,
              batch_size: int, heldout: bool, device: torch.device):
    taps: dict[str, list[torch.Tensor]] = {
        "pooled_sensory": [], "recurrent_retrieval": [], "workspace": [],
        "write_key": [], "write_value": [],
    }
    labels = []
    episodes = []
    pending_labels = []
    for seed in range(start, start + lifetimes):
        lifetime = generate_temporal_attention_lifetime(seed, heldout=heldout)
        for order, episode in zip(lifetime.query_features, lifetime.future_queries):
            episodes.append(episode)
            pending_labels.append(order[0])
            if len(episodes) == batch_size:
                _extract_batch(model, episodes, pending_labels, taps, labels, device)
                episodes, pending_labels = [], []
    if episodes:
        _extract_batch(model, episodes, pending_labels, taps, labels, device)
    return {key: torch.cat(value) for key, value in taps.items()}, torch.cat(labels)


def _extract_batch(model, episodes, batch_labels, taps, labels, device):
    batch = collate_episodes(episodes)
    frames = batch["frames"].to(device)
    pcm = batch["pcm"].to(device)
    mask = batch["mask"].to(device)
    memory = DifferentiableBatchMemory(len(episodes), model.hidden, device=device)
    output = model(frames, pcm, mask, memory)
    values = {
        "pooled_sensory": model.sensory_summary(frames, pcm, mask),
        "recurrent_retrieval": model.retrieval_summary(frames, pcm, mask),
        "workspace": output.workspace.mean(dim=1),
        "write_key": output.write_keys,
        "write_value": output.write_values,
    }
    for key, value in values.items():
        taps[key].append(value.detach().cpu())
    labels.append(torch.tensor(batch_labels, dtype=torch.long))


def _fit_probe(train_x, train_y, test_x, test_y, *, nonlinear: bool,
               device: torch.device, seed: int):
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    probe = (nn.Sequential(nn.Linear(train_x.shape[1], 64), nn.GELU(), nn.Linear(64, 2))
             if nonlinear else nn.Linear(train_x.shape[1], 2)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-3)
    best = 0.0
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            best = max(best, float((probe(test_x).argmax(-1) == test_y).float().mean()))
    with torch.no_grad():
        train_accuracy = float((probe(train_x).argmax(-1) == train_y).float().mean())
        test_accuracy = float((probe(test_x).argmax(-1) == test_y).float().mean())
    return {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
            "best_test_accuracy": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)
    train_x, train_y = _features(
        model, start=7_000_000, lifetimes=args.train_lifetimes,
        batch_size=args.batch_size, heldout=False, device=device)
    test_x, test_y = _features(
        model, start=9_000_000, lifetimes=args.test_lifetimes,
        batch_size=args.batch_size, heldout=True, device=device)
    balance = {"train_first_color_rate": float(train_y.float().mean()),
               "test_first_color_rate": float(test_y.float().mean())}
    results = {}
    for tap in train_x:
        results[tap] = {
            "linear": _fit_probe(train_x[tap], train_y, test_x[tap], test_y,
                                 nonlinear=False, device=device, seed=args.seed),
            "mlp": _fit_probe(train_x[tap], train_y, test_x[tap], test_y,
                              nonlinear=True, device=device, seed=args.seed),
        }
    report = {"schema": "temporal-order-decode-probe-v1", "checkpoint": str(args.checkpoint),
              "controller_frozen": True, "visual_only": True,
              "train_examples": int(train_y.numel()), "test_examples": int(test_y.numel()),
              "balance": balance, "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
