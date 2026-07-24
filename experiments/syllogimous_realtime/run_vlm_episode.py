#!/usr/bin/env python3
"""Score a real packet-only VLM policy inside the causal reference episode."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .environment import Action, RealtimeEpisode, generate_question
from .evaluation import EpisodeRecord, write_summary
from .host_client import HostPacket, normalize_action
from .vlm_policy import SmolVLMPolicy

def to_host_packet(observation) -> HostPacket:
    pcm = np.clip(observation.pcm * 32767.0, -32768, 32767).astype(np.int16)
    return HostPacket(observation.timestamp_ms, 640, 400, observation.frame, pcm, 16_000)

def action_from_text(text: str) -> Action:
    return {"WAIT": Action.WAIT, "NEXT": Action.NEXT, "PREVIOUS": Action.PREVIOUS,
            "TRUE": Action.TRUE, "FALSE": Action.FALSE}[normalize_action(text)]

def run(model: SmolVLMPolicy, *, episodes: int, premises: int,
        deadline_ms: int, seed: int,
        packet_gate: Callable[[HostPacket], bool] | None = None,
        difficulty: str = "standard", stream_interval_ms: int = 33
        ) -> tuple[list[EpisodeRecord], list[list[dict]]]:
    if stream_interval_ms <= 0:
        raise ValueError("stream_interval_ms must be positive")
    rows = []
    traces = []
    for offset in range(episodes):
        question = generate_question(seed + offset, premises=premises)
        episode = RealtimeEpisode(question, deadline_ms=deadline_ms)
        result = episode.step(Action.WAIT)
        inference_total = 0.0
        trace = []
        while not result.done:
            packet = to_host_packet(result.observation)
            if packet_gate is not None and not packet_gate(packet):
                trace.append({"timestamp_ms": result.observation.timestamp_ms,
                              "raw": "", "action": "WAIT", "suppressed": True})
                # A rejected packet still advances the real stream clock.  Without
                # this sleep a learned gate can spin thousands of free WAIT actions
                # and make its latency/token metrics meaningless.
                time.sleep(stream_interval_ms / 1000.0)
                result = episode.step(Action.WAIT)
                continue
            started = time.perf_counter()
            text = model(packet)
            inference_total += (time.perf_counter() - started) * 1000
            action = action_from_text(text)
            trace.append({"timestamp_ms": packet.timestamp_ms,
                          "raw": model.last_generated, "action": action.name})
            result = episode.step(action)
        outcome = "correct" if result.outcome == "right" else result.outcome
        rows.append(EpisodeRecord(seed + offset, question.family, difficulty, outcome,
                                  result.observation.timestamp_ms, deadline_ms,
                                  inference_ms=int(inference_total),
                                  sensory_tokens=sum(1 for item in trace if not item.get("suppressed"))))
        traces.append(trace)
    return rows, traces

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--premises", type=int, default=2)
    parser.add_argument("--deadline-ms", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("vlm_episode_metrics.json"))
    args = parser.parse_args()
    model = SmolVLMPolicy.from_pretrained(args.model, device=args.device,
                                          local_files_only=args.local_files_only,
                                          image_size=args.image_size,
                                          max_new_tokens=args.max_new_tokens,
                                          dtype=args.dtype)
    rows, traces = run(model, episodes=args.episodes, premises=args.premises,
                       deadline_ms=args.deadline_ms, seed=args.seed)
    from .evaluation import summarize
    summary = summarize(rows)
    summary["action_traces"] = traces
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "episodes": args.episodes}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
