from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.event_stream_snake.model import EventSnakeController, FrozenSmolActionListener

from .environment import MODALITIES, make_reflex_dataset


class Arrays(Dataset):
    def __init__(self, values: dict[str, np.ndarray]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values["action"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.as_tensor(value[index]) for key, value in self.values.items()}


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train_supervised(model: EventSnakeController, data: dict[str, np.ndarray],
                     device: torch.device, epochs: int, batch_size: int,
                     learning_rate: float, emission_weight: float) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    loader = DataLoader(Arrays(data), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            logits, audit = model(batch["frames"].float().to(device),
                                  batch["audio"].float().to(device))
            target = batch["action"].long().to(device)
            emissions = (audit["vision_probability"].sum(1)
                         + audit["audio_probability"].sum(1)).mean()
            emitted_fraction = emissions / (batch["frames"].shape[1] * 2)
            loss = nn.functional.cross_entropy(logits, target) + emission_weight * emitted_fraction
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite reflex loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if epoch == 0 or epoch + 1 == epochs:
            print(f"supervised_epoch={epoch + 1}", flush=True)


def immediate_reward_updates(model: EventSnakeController, device: torch.device,
                             batches: int, batch_size: int, sequence: int, size: int,
                             seed: int, learning_rate: float,
                             emission_weight: float) -> None:
    """One-step policy gradient: every sensory event immediately earns ±1."""
    if batches <= 0:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    for update in range(batches):
        data = make_reflex_dataset(batch_size, sequence, size, seed + 10000 + update)
        frames = torch.from_numpy(data["frames"]).float().to(device)
        audio = torch.from_numpy(data["audio"]).float().to(device)
        target = torch.from_numpy(data["action"]).long().to(device)
        model.train()
        logits, audit = model(frames, audio)
        probabilities = logits.softmax(-1)
        correct_probability = probabilities.gather(1, target[:, None]).squeeze(1)
        # Exact expected reward for the one-step ±1 game. This has far less
        # variance than sampling an action while preserving the same objective.
        expected_reward = 2.0 * correct_probability - 1.0
        emissions = (audit["vision_probability"].sum(1)
                     + audit["audio_probability"].sum(1)).mean()
        emitted_fraction = emissions / (sequence * 2)
        loss = -expected_reward.mean() + emission_weight * emitted_fraction
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if update == 0 or update + 1 == batches:
            print(f"reward_update={update + 1} expected_reward="
                  f"{expected_reward.mean().item():.3f}", flush=True)


@torch.no_grad()
def evaluate(model: EventSnakeController, data: dict[str, np.ndarray],
             device: torch.device, batch_size: int) -> dict:
    model.eval()
    predictions = []
    vision = audio = 0.0
    for batch in DataLoader(Arrays(data), batch_size=batch_size):
        logits, audit = model(batch["frames"].float().to(device),
                              batch["audio"].float().to(device))
        predictions.extend(logits.argmax(-1).cpu().tolist())
        vision += float(audit["vision_emissions"].sum().item())
        audio += float(audit["audio_emissions"].sum().item())
    prediction = np.asarray(predictions)
    correct = prediction == data["action"]
    by_modality = {
        name: float(correct[data["modality"] == index].mean())
        for index, name in enumerate(MODALITIES)
    }
    by_kind = {
        "target": float(correct[data["hazard"] == 0].mean()),
        "hazard": float(correct[data["hazard"] == 1].mean()),
    }
    count = len(correct)
    return {
        "accuracy": float(correct.mean()),
        "mean_reward": float(np.where(correct, 1.0, -1.0).mean()),
        "by_modality": by_modality,
        "by_kind": by_kind,
        "vision_tokens_per_trial": vision / count,
        "audio_tokens_per_trial": audio / count,
        "token_reduction_fraction": 1.0 - (vision + audio) / (count * data["frames"].shape[1] * 2),
    }


@torch.no_grad()
def latency_audit(model: EventSnakeController, data: dict[str, np.ndarray],
                  device: torch.device, runs: int) -> dict[str, float]:
    model.eval()
    values = []
    token_counts = []
    for index in range(min(runs, len(data["action"]))):
        frames = torch.from_numpy(data["frames"][index:index + 1]).float().to(device)
        audio = torch.from_numpy(data["audio"][index:index + 1]).float().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        _, audit = model(frames, audio, compact=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
        token_counts.append(float(audit["vision_emissions"].sum()
                                  + audit["audio_emissions"].sum()))
    array = np.asarray(values)
    return {
        "runs": len(array),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "tokens_per_trial": float(np.mean(token_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dense", "fixed", "learned"), default="learned")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--reward-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=12)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--emission-weight", type=float, default=0.001)
    parser.add_argument("--latency-runs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=Path("/tmp/event_stream_reflex"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    train = make_reflex_dataset(args.samples, args.sequence, args.size, args.seed)
    test = make_reflex_dataset(args.test_samples, args.sequence, args.size, args.seed + 1)
    listener = FrozenSmolActionListener(args.model, args.local_files_only).to(device)
    model = EventSnakeController(args.size + 2, listener, args.mode).to(device)
    train_supervised(model, train, device, args.epochs, args.batch_size,
                     args.lr, args.emission_weight)
    before = evaluate(model, test, device, args.batch_size)
    immediate_reward_updates(
        model, device, args.reward_batches if args.mode == "learned" else 0,
        args.batch_size, args.sequence, args.size, args.seed,
        args.lr * 0.2, args.emission_weight)
    result = {
        "config": vars(args) | {"out": str(args.out), "device": str(device)},
        "before_reward": before,
        "after_reward": evaluate(model, test, device, args.batch_size),
        "latency": latency_audit(model, test, device, args.latency_runs),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_seed{args.seed}"
    torch.save({"model": model.state_dict(), "result": result}, args.out / f"{stem}.pt")
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
