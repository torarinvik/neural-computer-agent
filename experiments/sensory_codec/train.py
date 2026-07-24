#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .games import CAPABILITY_FAMILIES, GAME_NAMES, make_game, make_multigame_dataset
from .model import CodecModel, DirectModel, Listener, LLMCodecModel, SmolLLMListener, TASK_DIMS
from .runtime import SensoryPacket, VisualActionAgent, parse_action
from .traps import modality_trap_audit


class Arrays(Dataset):
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data["frames"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.as_tensor(value[index]) for key, value in self.data.items()}


def task_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
              variant: str) -> torch.Tensor:
    loss = nn.functional.cross_entropy(outputs["action"], batch["action"].long())
    if variant != "gameplay":
        for key in ("horizontal", "vertical", "direction", "ate"):
            loss = loss + nn.functional.cross_entropy(outputs[key], batch[key].long())
        loss = loss + nn.functional.binary_cross_entropy_with_logits(outputs["danger"], batch["danger"].float())
    return loss


@torch.no_grad()
def metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = {key: 0.0 for key in TASK_DIMS}
    total = 0
    gate_sum = code_abs = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs, code, gate = model(batch["frames"].float(), batch["audio"].float(),
                                    batch["text"].long())
        count = len(code)
        total += count
        for key in ("action", "horizontal", "vertical", "direction", "ate"):
            correct[key] += (outputs[key].argmax(-1) == batch[key]).float().sum().item()
        danger_prediction = (outputs["danger"].sigmoid() >= 0.5)
        correct["danger"] += (danger_prediction == batch["danger"].bool()).float().mean(1).sum().item()
        gate_sum += gate.mean(1).sum().item()
        code_abs += code.abs().mean(1).sum().item()
    result = {f"{key}_accuracy": value / total for key, value in correct.items()}
    result.update(gate_mean=gate_sum / total, code_abs_mean=code_abs / total)
    return result


def metrics_by_game(model: nn.Module, data: dict[str, np.ndarray], device: torch.device,
                    batch_size: int) -> dict[str, dict[str, float]]:
    result = {}
    for game_id in np.unique(data["game"]):
        mask = data["game"] == game_id
        subset = {key: value[mask] for key, value in data.items()}
        result[GAME_NAMES[int(game_id)]] = metrics(
            model, DataLoader(Arrays(subset), batch_size=batch_size), device)
    return result


def modality_ablations(model: nn.Module, data: dict[str, np.ndarray], device: torch.device,
                       batch_size: int) -> dict[str, dict[str, float]]:
    """Score causal reliance by zeroing one raw modality before streaming."""
    results = {}
    for modality, key in (("vision", "frames"), ("audio", "audio"), ("text", "text")):
        altered = {name: value.copy() for name, value in data.items()}
        altered[key].fill(0)
        results[f"without_{modality}"] = metrics(
            model, DataLoader(Arrays(altered), batch_size=batch_size), device)
    return results


def modality_ablations_by_game(model: nn.Module, data: dict[str, np.ndarray],
                               device: torch.device, batch_size: int) -> dict:
    result = {}
    for game_id in np.unique(data["game"]):
        mask = data["game"] == game_id
        subset = {key: value[mask] for key, value in data.items()}
        result[GAME_NAMES[int(game_id)]] = modality_ablations(
            model, subset, device, batch_size)
    return result


def train_listener(listener: Listener, data: dict[str, np.ndarray], device: torch.device,
                   epochs: int, batch_size: int, lr: float) -> None:
    listener.to(device).train()
    optimizer = torch.optim.AdamW(listener.parameters(), lr=lr)
    loader = DataLoader(Arrays(data), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = listener(batch["semantic"].float())
            loss = task_loss(outputs, batch, "grounded")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_model(model: nn.Module, data: dict[str, np.ndarray], device: torch.device,
                optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,
                variant: str, compact_weight: float, latency_weight: float,
                progress_prefix: str = "epoch") -> None:
    loader = DataLoader(Arrays(data), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs, _, gate = model(batch["frames"].float(), batch["audio"].float(),
                                     batch["text"].long())
            loss = task_loss(outputs, batch, variant)
            if variant == "compressed":
                loss = loss + compact_weight * gate.mean()
            # Wall time is not differentiable. Mean routing activity is the
            # supervised-stage surrogate; actual latency is measured below and
            # should replace this term once routing is trained with RL.
            loss = loss + latency_weight * gate.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 0 or epoch + 1 == epochs:
            print(f"{progress_prefix}={epoch + 1}", flush=True)


@torch.no_grad()
def collect_on_policy(model: nn.Module, device: torch.device, games: tuple[str, ...],
                      samples: int, sequence: int, size: int, seed: int) -> dict[str, np.ndarray]:
    """Collect teacher labels at states reached by the learned policy (DAgger)."""
    rng = np.random.default_rng(seed)
    frames, audio, text, semantic, game_ids = [], [], [], [], []
    label_names = ("action", "horizontal", "vertical", "direction", "danger", "ate")
    labels: dict[str, list] = {key: [] for key in label_names}
    model.eval()
    agent = VisualActionAgent(model, device)
    game_index = 0
    per_episode = max(1, min(32, samples // (len(games) * 4)))
    while len(frames) < samples:
        game = games[game_index % len(games)]
        game_index += 1
        env = make_game(game, size, seed + 997 * game_index)
        first = env.observe(target_visible=True, detail_visible=True,
                            theme=int(rng.integers(2)))
        history: deque[np.ndarray] = deque((first.copy() for _ in range(sequence)),
                                           maxlen=sequence)
        first_audio = env.raw_audio()
        first_text = env.raw_text()
        audio_history: deque[np.ndarray] = deque((first_audio.copy() for _ in range(sequence)),
                                                 maxlen=sequence)
        text_history: deque[np.ndarray] = deque((first_text.copy() for _ in range(sequence)),
                                                maxlen=sequence)
        episode_examples = 0
        for _ in range(250):
            stacked = np.stack(history)
            current = env.labels()
            frames.append(stacked)
            audio.append(np.stack(audio_history))
            text.append(np.stack(text_history))
            semantic.append(env.semantic_vector())
            game_ids.append(GAME_NAMES.index(game))
            for key in label_names:
                labels[key].append(current[key])
            episode_examples += 1
            packet = SensoryPacket(stacked, np.stack(audio_history), np.stack(text_history))
            learner_action = parse_action(agent.emit(packet))
            _, done = env.step(learner_action)
            if len(frames) >= samples or done or episode_examples >= per_episode:
                break
            history.append(env.observe(
                target_visible=rng.random() > 0.34,
                detail_visible=rng.random() > 0.18,
                theme=int(rng.integers(2)),
            ))
            audio_history.append(env.raw_audio())
            text_history.append(env.raw_text())
    result = {
        "frames": np.stack(frames),
        "audio": np.stack(audio),
        "text": np.stack(text),
        "semantic": np.stack(semantic),
        "game": np.asarray(game_ids, dtype=np.int64),
    }
    result.update({key: np.asarray(value) for key, value in labels.items()})
    return result


def concatenate(*datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([dataset[key] for dataset in datasets], axis=0)
            for key in datasets[0]}


def temporal_audit(model: nn.Module, data: dict[str, np.ndarray], device: torch.device,
                   limit: int = 256) -> dict[str, float]:
    """Representation sensitivity to destroyed order and repeated final frames."""
    model.eval()
    frames = torch.from_numpy(data["frames"][:limit]).float().to(device)
    audio = torch.from_numpy(data["audio"][:limit]).float().to(device)
    text = torch.from_numpy(data["text"][:limit]).long().to(device)
    with torch.no_grad():
        _, original, _ = model(frames, audio, text)
        _, reversed_code, _ = model(frames.flip(1), audio.flip(1), text.flip(1))
        frozen = frames[:, -1:].expand_as(frames)
        frozen_audio = audio[:, -1:].expand_as(audio)
        frozen_text = text[:, -1:].expand_as(text)
        _, frozen_code, _ = model(frozen, frozen_audio, frozen_text)
        no_target = frames.clone()
        no_target[:, :, 3] = 0
        _, no_target_code, _ = model(no_target, audio, text)
        no_detail = frames.clone()
        no_detail[:, :, 1] = 0
        _, no_detail_code, _ = model(no_detail, audio, text)
        _, no_audio_code, _ = model(frames, torch.zeros_like(audio), text)
        _, no_text_code, _ = model(frames, audio, torch.zeros_like(text))
    return {
        "reverse_code_distance": (original - reversed_code).square().mean().sqrt().item(),
        "frozen_code_distance": (original - frozen_code).square().mean().sqrt().item(),
        "target_erasure_code_distance": (original - no_target_code).square().mean().sqrt().item(),
        "detail_erasure_code_distance": (original - no_detail_code).square().mean().sqrt().item(),
        "audio_erasure_code_distance": (original - no_audio_code).square().mean().sqrt().item(),
        "text_erasure_code_distance": (original - no_text_code).square().mean().sqrt().item(),
    }


@torch.no_grad()
def rollout_audit(model: nn.Module, device: torch.device, size: int, sequence: int,
                  seed: int, game: str, episodes: int = 30,
                  horizon: int = 250) -> dict[str, float]:
    """Let the learned policy actually play; imitation accuracy is not enough."""
    rng = np.random.default_rng(seed)
    events = steps = deaths = 0
    model.eval()
    agent = VisualActionAgent(model, device)
    for episode in range(episodes):
        env = make_game(game, size=size, seed=seed + episode)
        first = env.observe(target_visible=True, detail_visible=True, theme=episode % 2)
        history = [first.copy() for _ in range(sequence)]
        audio_history = [env.raw_audio().copy() for _ in range(sequence)]
        text_history = [env.raw_text().copy() for _ in range(sequence)]
        for _ in range(horizon):
            packet = SensoryPacket(np.stack(history), np.stack(audio_history),
                                   np.stack(text_history))
            action = parse_action(agent.emit(packet))
            reward, done = env.step(action)
            steps += 1
            events += int(reward >= 1.0)
            if done:
                deaths += 1
                break
            history.pop(0)
            history.append(env.observe(
                target_visible=rng.random() > 0.34,
                detail_visible=rng.random() > 0.18,
                theme=episode % 2,
            ))
            audio_history.pop(0)
            audio_history.append(env.raw_audio())
            text_history.pop(0)
            text_history.append(env.raw_text())
    return {
        "episodes": episodes,
        "mean_steps": steps / episodes,
        "mean_events": events / episodes,
        "death_rate": deaths / episodes,
    }


def reference_rollouts(size: int, seed: int, game: str, episodes: int = 30,
                       horizon: int = 250) -> dict[str, dict[str, float]]:
    results = {}
    for policy in ("teacher", "random"):
        rng = np.random.default_rng(seed + (0 if policy == "teacher" else 5000))
        events = steps = deaths = 0
        for episode in range(episodes):
            env = make_game(game, size=size, seed=seed + episode)
            for _ in range(horizon):
                action = env.teacher_action() if policy == "teacher" else int(rng.integers(4))
                reward, done = env.step(action)
                steps += 1
                events += int(reward >= 1.0)
                if done:
                    deaths += 1
                    break
        results[policy] = {
            "mean_steps": steps / episodes,
            "mean_events": events / episodes,
            "death_rate": deaths / episodes,
        }
    return results


def synchronize(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def latency_audit(model: nn.Module, data: dict[str, np.ndarray], device: torch.device,
                  runs: int, target_ms: float, latency_weight: float,
                  action_accuracy: float) -> dict[str, float]:
    """Measure pixels/PCM/characters through action-token emission end to end."""
    if runs <= 0:
        return {}
    agent = VisualActionAgent(model, device)
    count = min(runs, len(data["frames"]))

    def packet(index: int) -> SensoryPacket:
        return SensoryPacket(data["frames"][index], data["audio"][index], data["text"][index])

    for index in range(min(5, count)):
        agent.emit(packet(index))
    synchronize(device)
    durations = []
    for index in range(count):
        synchronize(device)
        started = time.perf_counter_ns()
        agent.emit(packet(index))
        synchronize(device)
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    values = np.asarray(durations, dtype=np.float64)
    mean_ms = float(values.mean())
    latency_penalty = latency_weight * math.log1p(mean_ms / max(target_ms, 1e-9))
    return {
        "runs": count,
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "actions_per_second": 1000.0 / max(mean_ms, 1e-9),
        "target_ms": target_ms,
        "latency_weight": latency_weight,
        "latency_penalty": latency_penalty,
        "accuracy_dominant_score": action_accuracy - latency_penalty,
    }


def parse_games(value: str) -> tuple[str, ...]:
    games = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = sorted(set(games) - set(GAME_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown games: {', '.join(unknown)}")
    if not games:
        raise argparse.ArgumentTypeError("provide at least one game")
    return games


def select_device(requested: str) -> torch.device:
    """Choose an accelerator explicitly, preferring CUDA for cloud training."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Screenwatch sensory-codec v0 experiment")
    parser.add_argument("--variant", choices=("gameplay", "grounded", "compressed"), default="grounded")
    parser.add_argument("--listener", choices=("grounded", "random", "direct", "llm", "llm_random"), default="grounded")
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sensory-tokens", type=int, default=4)
    parser.add_argument("--vision-streamer", choices=("tiny", "smol"), default="tiny")
    parser.add_argument("--vision-model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--vision-size", type=int, default=128)
    parser.add_argument("--randomize-vision", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--sequence", type=int, default=6)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--games", type=parse_games, default=GAME_NAMES,
                        help="comma-separated training games (default: all)")
    parser.add_argument("--holdout-game", choices=GAME_NAMES,
                        help="exclude this game from training and evaluate only on it")
    parser.add_argument("--holdout-capability", choices=tuple(CAPABILITY_FAMILIES),
                        help="exclude every game in this capability family and evaluate on them")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--listener-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--compact-weight", type=float, default=0.03)
    parser.add_argument("--latency-weight", type=float, default=0.002,
                        help="small routing/selection penalty; accuracy loss remains dominant")
    parser.add_argument("--latency-target-ms", type=float, default=50.0)
    parser.add_argument("--latency-runs", type=int, default=64)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-samples", type=int, default=2000)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--rollout-episodes", type=int, default=30)
    parser.add_argument("--rollout-horizon", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sensory_codec"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.holdout_game and args.holdout_capability:
        parser.error("choose either --holdout-game or --holdout-capability")
    excluded = set(CAPABILITY_FAMILIES.get(args.holdout_capability, ()))
    if args.holdout_game:
        excluded.add(args.holdout_game)
    train_games = tuple(game for game in args.games if game not in excluded)
    if not train_games:
        parser.error("--holdout-game removed every training game")
    if args.holdout_capability:
        test_games = tuple(game for game in args.games if game in excluded)
    else:
        test_games = (args.holdout_game,) if args.holdout_game else args.games
    train = make_multigame_dataset(args.samples, args.sequence, args.size, args.seed,
                                   train_games, partial=True, themes=(0,))
    test = make_multigame_dataset(args.test_samples, args.sequence, args.size, args.seed + 1,
                                  test_games, partial=True, themes=(0, 1))
    board_pixels = args.size + 2
    streamer_kwargs = {
        "vision_backend": args.vision_streamer,
        "vision_model": args.vision_model,
        "vision_size": args.vision_size,
        "local_files_only": args.local_files_only,
        "randomize_vision": args.randomize_vision,
    }

    listener_metrics = None
    if args.listener == "direct":
        model: nn.Module = DirectModel(board_pixels, streamer_kwargs=streamer_kwargs)
    elif args.listener in ("llm", "llm_random"):
        llm_listener = SmolLLMListener(
            args.llm_model, sensory_tokens=args.sensory_tokens,
            local_files_only=args.local_files_only,
            randomize_backbone=args.listener == "llm_random")
        model = LLMCodecModel(board_pixels, llm_listener, streamer_kwargs=streamer_kwargs)
    else:
        listener = Listener()
        if args.listener == "grounded":
            train_listener(listener, train, device, args.listener_epochs, args.batch_size, args.lr)
            # Record whether the privileged-state listener learned its own task first.
            listener.eval()
            with torch.no_grad():
                semantic = torch.from_numpy(test["semantic"]).float().to(device)
                predicted = listener(semantic)["action"].argmax(-1).cpu().numpy()
                listener_metrics = float((predicted == test["action"]).mean())
        model = CodecModel(board_pixels, listener, frozen_listener=True,
                           streamer_kwargs=streamer_kwargs)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    test_loader = DataLoader(Arrays(test), batch_size=args.batch_size)
    train_model(model, train, device, optimizer, args.epochs, args.batch_size,
                args.variant, args.compact_weight, args.latency_weight)
    dagger_sizes = []
    aggregate_train = train
    for round_index in range(args.dagger_rounds):
        on_policy = collect_on_policy(
            model, device, train_games, args.dagger_samples, args.sequence,
            args.size, args.seed + 10000 * (round_index + 1))
        aggregate_train = concatenate(aggregate_train, on_policy)
        dagger_sizes.append(len(on_policy["frames"]))
        train_model(model, aggregate_train, device, optimizer, args.dagger_epochs,
                    args.batch_size, args.variant, args.compact_weight,
                    args.latency_weight,
                    progress_prefix=f"dagger{round_index + 1}_epoch")

    test_metrics = metrics(model, test_loader, device)

    result = {
        "config": vars(args) | {"out": str(args.out), "device": str(device),
                                "games": list(args.games), "train_games": list(train_games),
                                "test_games": list(test_games)},
        "listener_privileged_action_accuracy": listener_metrics,
        "dagger_collected_samples": dagger_sizes,
        "test": test_metrics,
        "test_by_game": metrics_by_game(model, test, device, args.batch_size),
        "modality_ablations": modality_ablations(model, test, device, args.batch_size),
        "modality_ablations_by_game": modality_ablations_by_game(
            model, test, device, args.batch_size),
        "temporal_audit": temporal_audit(model, test, device),
        "latency_audit": latency_audit(
            model, test, device, args.latency_runs, args.latency_target_ms,
            args.latency_weight, test_metrics["action_accuracy"]),
        "modality_trap_audit": modality_trap_audit(
            model, device, args.sequence, args.size),
        "rollout_audit": {
            game: rollout_audit(model, device, args.size, args.sequence,
                                args.seed + 1000 + 100 * index, game,
                                episodes=args.rollout_episodes,
                                horizon=args.rollout_horizon)
            for index, game in enumerate(test_games)
        },
        "rollout_references": {
            game: reference_rollouts(args.size, args.seed + 1000 + 100 * index, game,
                                     episodes=args.rollout_episodes,
                                     horizon=args.rollout_horizon)
            for index, game in enumerate(test_games)
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.holdout_capability:
        suite = f"holdout-capability-{args.holdout_capability}"
    else:
        suite = f"holdout-{args.holdout_game}" if args.holdout_game else "-".join(args.games)
    dagger = f"_dagger{args.dagger_rounds}" if args.dagger_rounds else ""
    latency = f"_latency{args.latency_weight:g}" if args.latency_weight else ""
    vision = f"_vision-{args.vision_streamer}{'-random' if args.randomize_vision else ''}"
    listener_model = f"_{Path(args.llm_model).name}" if args.listener.startswith("llm") else ""
    stem = (f"{args.variant}_{args.listener}{listener_model}_{suite}_seed{args.seed}"
            f"{vision}{dagger}{latency}")
    torch.save({"model": model.state_dict(), "result": result}, args.out / f"{stem}.pt")
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
