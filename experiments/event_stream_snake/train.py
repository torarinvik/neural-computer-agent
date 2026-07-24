from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.sensory_codec.snake import SnakeEnv

from .environment import event_audio, make_event_dataset
from .model import EventSnakeController, FrozenSmolActionListener
from .runtime import ACTION_TOKENS, AVPacket, EventActionAgent


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


def objective(action_loss: torch.Tensor, mean_emissions: torch.Tensor,
              emission_weight: float) -> torch.Tensor:
    """Accuracy is primary; emissions provide only a small efficiency tie-break."""
    return action_loss + emission_weight * mean_emissions


def train_supervised(model: EventSnakeController, data: dict[str, np.ndarray],
                     device: torch.device, epochs: int, batch_size: int,
                     learning_rate: float, emission_weight: float) -> None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    loader = DataLoader(Arrays(data), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            frames = batch["frames"].float().to(device)
            audio = batch["audio"].float().to(device)
            target = batch["action"].long().to(device)
            logits, audit = model(frames, audio)
            emissions = (audit["vision_probability"].sum(1)
                         + audit["audio_probability"].sum(1)).mean()
            loss = objective(nn.functional.cross_entropy(logits, target),
                             emissions, emission_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite supervised event-stream loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if epoch == 0 or epoch + 1 == epochs:
            print(f"supervised_epoch={epoch + 1}", flush=True)


def reinforce(model: EventSnakeController, device: torch.device, episodes: int,
              size: int, sequence: int, sensor_ticks: int, horizon: int,
              learning_rate: float, emission_weight: float, seed: int) -> None:
    """Short-segment policy gradients; the frozen listener receives no updates."""
    if episodes <= 0:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        env = SnakeEnv(size=size, seed=seed + 1000 + episode)
        first = env.observe()
        frames = deque((first.copy() for _ in range(sequence)), maxlen=sequence)
        audio = deque((event_audio(False) for _ in range(sequence)), maxlen=sequence)
        model.train()
        segment_losses: list[torch.Tensor] = []
        for step in range(horizon):
            frame_tensor = torch.from_numpy(np.stack(frames)[None]).float().to(device)
            audio_tensor = torch.from_numpy(np.stack(audio)[None]).float().to(device)
            logits, audit = model(frame_tensor, audio_tensor)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite policy logits")
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            reward, done = env.step(int(action.item()))
            emission_proxy = (audit["vision_probability"].mean()
                              + audit["audio_probability"].mean())
            # Reward is dominated by survival/apples/death. The efficiency term
            # is deliberately too small to make early death attractive.
            advantage = float(reward) - 0.01
            segment_losses.append(-advantage * distribution.log_prob(action).mean()
                                  - 0.001 * distribution.entropy().mean()
                                  + emission_weight * emission_proxy)
            current = env.observe()
            sound = event_audio(env.state.ate_last)
            for tick in range(sensor_ticks):
                frames.append(current.copy())
                audio.append(sound.copy() if tick == 0 else event_audio(False))
            if len(segment_losses) == 8 or done or step + 1 == horizon:
                optimizer.zero_grad()
                torch.stack(segment_losses).mean().backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                segment_losses.clear()
            if done:
                break
        if episode == 0 or episode + 1 == episodes:
            print(f"rl_episode={episode + 1}", flush=True)


@torch.no_grad()
def offline_audit(model: EventSnakeController, data: dict[str, np.ndarray],
                  device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    correct = total = 0
    vision = audio = 0.0
    loader = DataLoader(Arrays(data), batch_size=batch_size)
    for batch in loader:
        logits, audit = model(batch["frames"].float().to(device),
                              batch["audio"].float().to(device))
        target = batch["action"].long().to(device)
        correct += int((logits.argmax(-1) == target).sum().item())
        total += len(target)
        vision += float(audit["vision_emissions"].sum().item())
        audio += float(audit["audio_emissions"].sum().item())
    sequence = data["frames"].shape[1]
    return {
        "action_accuracy": correct / total,
        "vision_tokens_per_decision": vision / total,
        "audio_tokens_per_decision": audio / total,
        "dense_tokens_per_decision": float(sequence * 2),
        "token_reduction_fraction": 1.0 - (vision + audio) / (total * sequence * 2),
    }


@torch.no_grad()
def rollout_audit(model: EventSnakeController, device: torch.device, episodes: int,
                  horizon: int, size: int, sequence: int, sensor_ticks: int,
                  seed: int) -> dict[str, float]:
    model.eval()
    apples = steps = deaths = vision = audio = 0.0
    latencies: list[float] = []
    agent = EventActionAgent(model, device)
    for episode in range(episodes):
        env = SnakeEnv(size=size, seed=seed + episode)
        first = env.observe()
        frames = deque((first.copy() for _ in range(sequence)), maxlen=sequence)
        sounds = deque((event_audio(False) for _ in range(sequence)), maxlen=sequence)
        for _ in range(horizon):
            packet = AVPacket(np.stack(frames), np.stack(sounds))
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            token, counts = agent.emit(packet)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            action = ACTION_TOKENS.index(token)
            reward, done = env.step(action)
            apples += float(reward >= 1.0)
            steps += 1
            vision += counts.get("vision_emissions", 0.0)
            audio += counts.get("audio_emissions", 0.0)
            current = env.observe()
            sound = event_audio(env.state.ate_last)
            for tick in range(sensor_ticks):
                frames.append(current.copy())
                sounds.append(sound.copy() if tick == 0 else event_audio(False))
            if done:
                deaths += 1
                break
    values = np.asarray(latencies)
    return {
        "episodes": episodes,
        "apples_per_episode": apples / episodes,
        "steps_per_episode": steps / episodes,
        "death_rate": deaths / episodes,
        "vision_tokens_per_action": vision / max(1, steps),
        "audio_tokens_per_action": audio / max(1, steps),
        "mean_latency_ms": float(values.mean()),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dense", "fixed", "learned"), default="learned")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=750)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--rl-episodes", type=int, default=20)
    parser.add_argument("--rollout-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence", type=int, default=12)
    parser.add_argument("--sensor-ticks", type=int, default=3)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--emission-weight", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=Path("/tmp/event_stream_snake"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    train = make_event_dataset(args.samples, args.sequence, args.size, args.seed,
                               args.sensor_ticks)
    test = make_event_dataset(args.test_samples, args.sequence, args.size, args.seed + 1,
                              args.sensor_ticks)
    listener = FrozenSmolActionListener(args.model, args.local_files_only).to(device)
    model = EventSnakeController(args.size + 2, listener, args.mode).to(device)
    train_supervised(model, train, device, args.epochs, args.batch_size, args.lr,
                     args.emission_weight)
    before_rl = offline_audit(model, test, device, args.batch_size)
    reinforce(model, device, args.rl_episodes if args.mode == "learned" else 0,
              args.size, args.sequence, args.sensor_ticks, args.horizon,
              args.lr * 0.2, args.emission_weight, args.seed)
    result = {
        "config": vars(args) | {"out": str(args.out), "device": str(device)},
        "offline_before_rl": before_rl,
        "offline_after_rl": offline_audit(model, test, device, args.batch_size),
        "rollout": rollout_audit(model, device, args.rollout_episodes, args.horizon,
                                  args.size, args.sequence, args.sensor_ticks,
                                  args.seed + 10000),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mode}_seed{args.seed}"
    torch.save({"model": model.state_dict(), "result": result}, args.out / f"{stem}.pt")
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
