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

from experiments.event_stream_reflex.train import choose_device
from experiments.event_stream_snake.model import FrozenSmolActionListener

from .environment import MODALITIES, make_dataset
from .model import SelectiveController


class Arrays(Dataset):
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data

    def __len__(self):
        return len(self.data["action"])

    def __getitem__(self, index):
        return {key: torch.as_tensor(value[index]) for key, value in self.data.items()}


def optimize(model: SelectiveController, data: dict[str, np.ndarray], device: torch.device,
             epochs: int, batch_size: int, lr: float, emission_weight: float) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    for epoch in range(epochs):
        if model.adapter.mode == "learned":
            model.adapter.gate_override = "fixed" if epoch < max(2, epochs // 2) else None
        model.train()
        for batch in DataLoader(Arrays(data), batch_size=batch_size, shuffle=True):
            frames = batch["frames"].float().to(device)
            audio = batch["audio"].float().to(device)
            target = batch["action"].long().to(device)
            logits, audit = model(frames, audio)
            probability = logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
            expected_reward = 2 * probability - 1
            emissions = (audit["vision_probability"].sum(1) + audit["audio_probability"].sum(1)).mean()
            efficiency = emissions / (frames.shape[1] * 2)
            # Cross entropy prevents reward saturation; expected reward expresses
            # the immediate game objective, with efficiency deliberately tiny.
            loss = nn.functional.cross_entropy(logits, target) - expected_reward.mean()
            loss = loss + emission_weight * efficiency
            if model.adapter.mode == "learned":
                # Stabilize the content gate with sensor-derived event targets;
                # these are raw delta/energy thresholds, never game labels.
                if model.adapter.gate_override is None:
                    vision_weight = 1.0 + 4.0 * audit["vision_event_target"]
                    audio_weight = 1.0 + 4.0 * audit["audio_event_target"]
                    gate_loss = nn.functional.binary_cross_entropy(
                        audit["vision_probability"], audit["vision_event_target"], weight=vision_weight)
                    gate_loss = gate_loss + nn.functional.binary_cross_entropy(
                        audit["audio_probability"], audit["audio_event_target"], weight=audio_weight)
                    loss = loss + 0.1 * gate_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if epoch == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch + 1}", flush=True)


@torch.no_grad()
def evaluate(model: SelectiveController, data: dict[str, np.ndarray], device: torch.device,
             batch_size: int) -> dict:
    model.eval()
    predictions, vision_masks, audio_masks = [], [], []
    for batch in DataLoader(Arrays(data), batch_size=batch_size):
        logits, audit = model(batch["frames"].float().to(device), batch["audio"].float().to(device))
        predictions.append(logits.argmax(-1).cpu().numpy())
        vision_masks.append((audit["vision_probability"] >= 0.5).cpu().numpy()[..., 0])
        audio_masks.append((audit["audio_probability"] >= 0.5).cpu().numpy()[..., 0])
    prediction = np.concatenate(predictions)
    vision = np.concatenate(vision_masks)
    audio = np.concatenate(audio_masks)
    correct = prediction == data["action"]
    emitted = np.stack((vision, audio), -1)
    relevant = data["relevant_ticks"] > 0.5
    true_emissions = int((emitted & relevant).sum())
    total_emissions = int(emitted.sum())
    return {
        "accuracy": float(correct.mean()),
        "mean_reward": float(np.where(correct, 1.0, -1.0).mean()),
        "by_modality": {name: float(correct[data["modality"] == i].mean()) for i, name in enumerate(MODALITIES)},
        "by_kind": {
            "target": float(correct[data["hazard"] == 0].mean()),
            "hazard": float(correct[data["hazard"] == 1].mean()),
        },
        "tokens_per_window": float(emitted.sum((1, 2)).mean()),
        "token_reduction_fraction": float(1 - emitted.mean()),
        "relevant_event_recall": float((emitted & relevant).sum() / max(1, relevant.sum())),
        "emission_precision": float(true_emissions / max(1, total_emissions)),
    }


@torch.no_grad()
def latency(model: SelectiveController, data: dict[str, np.ndarray], device: torch.device,
            runs: int) -> dict:
    model.eval()
    values = []
    for index in range(min(runs, len(data["action"]))):
        frames = torch.from_numpy(data["frames"][index:index + 1]).to(device)
        audio = torch.from_numpy(data["audio"][index:index + 1]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        model(frames, audio, compact=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {"mean_ms": float(np.mean(values)), "p95_ms": float(np.percentile(values, 95)), "runs": len(values)}


@torch.no_grad()
def cache_benchmark(model: SelectiveController, data: dict[str, np.ndarray],
                    device: torch.device, windows: int = 16) -> dict:
    """Compare repeated causal prefill with persistent transformer KV state."""
    model.eval()
    frames = torch.from_numpy(data["frames"][:1]).to(device)
    audio = torch.from_numpy(data["audio"][:1]).to(device)
    tokens, mask, _ = model.adapter(frames, audio, compact=True)
    event = tokens[:, -1:]
    listener = model.listener
    prompt = listener.prompt.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    prefix = prompt
    for _ in range(windows):
        prefix = torch.cat((prefix, event), 1)
        listener.text_model(inputs_embeds=prefix, use_cache=False, return_dict=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    uncached = (time.perf_counter_ns() - started) / 1e6
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = listener.text_model(inputs_embeds=prompt, use_cache=True, return_dict=True)
    cache = output.past_key_values
    for _ in range(windows):
        output = listener.text_model(inputs_embeds=event, past_key_values=cache,
                                     use_cache=True, return_dict=True)
        cache = output.past_key_values
    if device.type == "cuda":
        torch.cuda.synchronize()
    cached = (time.perf_counter_ns() - started) / 1e6
    return {
        "windows": windows,
        "uncached_total_ms": uncached,
        "cached_total_ms": cached,
        "speedup": uncached / cached,
        "scope": "listener-only causal event benchmark",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dense", "fixed", "learned", "random"), default="learned")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--emission-weight", type=float, default=0.001)
    parser.add_argument("--random-keep", type=float, default=0.1)
    parser.add_argument("--latency-runs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=Path("/tmp/continuous_reflex"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train = make_dataset(args.samples, args.sequence, args.size, args.seed, "train", args.distractors)
    heldout = make_dataset(args.test_samples, args.sequence, args.size, args.seed + 1, "heldout", args.distractors)
    listener = FrozenSmolActionListener(args.model, args.local_files_only).to(device)
    model = SelectiveController(args.size + 2, listener, args.mode, args.random_keep).to(device)
    optimize(model, train, device, args.epochs, args.batch_size, args.lr, args.emission_weight)
    result = {
        "config": vars(args) | {"out": str(args.out), "device": str(device)},
        "train": evaluate(model, train, device, args.batch_size),
        "heldout": evaluate(model, heldout, device, args.batch_size),
        "latency": latency(model, heldout, device, args.latency_runs),
        "cache_benchmark": cache_benchmark(model, heldout, device),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_seed{args.seed}"
    torch.save({"model": model.state_dict(), "result": result}, args.out / f"{stem}.pt")
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
