#!/usr/bin/env python3
"""Train a sensory gate with frozen VLM gameplay reward."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .adapter_rl import AdapterConfig, StreamerGate, packet_features, freeze_listener
from .checkpointing import save_adapter
from .environment import Action, RealtimeEpisode, generate_question
from .host_client import HostPacket
from .run_vlm_episode import action_from_text
from .vlm_policy import SmolVLMPolicy

def masked_packet(packet: HostPacket, decisions: torch.Tensor) -> HostPacket:
    vision = bool(decisions[0].item())
    audio = bool(decisions[1].item())
    frame = packet.frame if vision else np.zeros_like(packet.frame)
    pcm = packet.pcm if audio else np.zeros_like(packet.pcm)
    return HostPacket(packet.timestamp_ms, packet.width, packet.height, frame, pcm, packet.sample_rate)

def train(model: SmolVLMPolicy, *, episodes: int, premises: int, deadline_ms: int,
          seed: int, config: AdapterConfig, checkpoint: Path) -> dict:
    listener = model.model
    freeze_listener(listener)
    gate = StreamerGate(feature_dim=8, modalities=2)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4)
    history = []
    for episode_index in range(episodes):
        question = generate_question(seed + episode_index, premises=premises)
        episode = RealtimeEpisode(question, deadline_ms=deadline_ms)
        result = episode.step(Action.WAIT)
        previous = None
        log_probs = []
        emitted = []
        inference_ms = 0.0
        while not result.done:
            observation = result.observation
            packet = HostPacket(observation.timestamp_ms, 640, 400, observation.frame,
                                np.clip(observation.pcm * 32767, -32768, 32767).astype(np.int16), 16000)
            features = packet_features(packet, previous).unsqueeze(0)
            decisions, log_prob = gate(features)
            previous = observation.frame.astype("float32") / 255.0
            if bool(decisions.any().item()):
                started = time.perf_counter()
                text = model(masked_packet(packet, decisions[0]))
                inference_ms += (time.perf_counter() - started) * 1000
                emitted.append(float(decisions.sum().item()))
                result = episode.step(action_from_text(text))
            else:
                emitted.append(0.0)
                result = episode.step(Action.WAIT)
            log_probs.append(log_prob.squeeze(0))
        task_reward = 1.0 if result.outcome == "right" else -1.0
        if result.outcome == "right":
            task_reward += 0.05 * max(0.0, (deadline_ms - result.observation.timestamp_ms) / deadline_ms)
        reward = config.reward_weight * task_reward
        mean_emitted = sum(emitted) / max(1, len(emitted))
        rate = mean_emitted / max(1, config.modalities)
        silence_penalty = config.silence_weight * max(0.0, config.min_emission_rate - rate)
        shaped = reward - config.latency_weight * mean_emitted - silence_penalty
        loss = -((shaped - 0.0) * torch.stack(log_probs).sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"episode": episode_index, "reward": reward, "task_reward": task_reward,
                        "shaped_reward": shaped,
                        "outcome": result.outcome, "latency_ms": result.observation.timestamp_ms,
                        "inference_ms": inference_ms, "mean_emitted_modalities": mean_emitted,
                        "silence_penalty": silence_penalty})
    summary = {"schema": "syllogimous.adapter-training.v1", "episodes": episodes,
        "config": {"variant": config.variant, "latency_weight": config.latency_weight,
                          "reward_weight": config.reward_weight,
                          "modalities": config.modalities,
                          "min_emission_rate": config.min_emission_rate,
                          "silence_weight": config.silence_weight},
               "history": history}
    save_adapter(checkpoint, gate, config=summary["config"], metrics={
        "mean_reward": sum(x["reward"] for x in history) / max(1, len(history)),
        "accuracy": sum(x["outcome"] == "right" for x in history) / max(1, len(history)),
    }, seed=seed)
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--premises", type=int, default=2)
    parser.add_argument("--deadline-ms", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-weight", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=Path, default=Path("adapter.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("adapter_training.json"))
    args = parser.parse_args()
    model = SmolVLMPolicy.from_pretrained(args.model, device=args.device,
                                          local_files_only=args.local_files_only)
    result = train(model, episodes=args.episodes, premises=args.premises,
                   deadline_ms=args.deadline_ms, seed=args.seed,
                   config=AdapterConfig("learned-gate", latency_weight=args.latency_weight),
                   checkpoint=args.checkpoint)
    args.metrics.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"episodes": args.episodes, "checkpoint": str(args.checkpoint),
                      "metrics": str(args.metrics)}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
