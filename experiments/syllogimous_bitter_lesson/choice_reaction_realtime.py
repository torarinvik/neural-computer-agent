from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.syllogimous_latent_agent.data import collate_episodes, PublicEpisode

from .choice_reaction import ReactionDifficulty, generate_choice_reaction_episode
from .model import BitterLessonAgent


@dataclass(frozen=True)
class StreamPacket:
    timestamp_ns: int
    frame: np.ndarray
    pcm: np.ndarray
    stimulus_visible: bool


@dataclass(frozen=True)
class ReactionResult:
    correct: bool
    action: int
    target: int
    response_ms: float
    reward: float
    deadline_missed: bool


class ChoiceReactionStream:
    """Private verifier that exposes only timestamped RGB/PCM packets."""

    def __init__(self, seed: int, difficulty: ReactionDifficulty, *,
                 heldout: bool = True, frame_interval_ms: float = 16.667):
        self._episode = generate_choice_reaction_episode(seed, difficulty,
                                                         heldout=heldout)
        self._target = int(self._episode.actions[-1])
        self._cursor = 0
        self.frame_interval_ms = frame_interval_ms

    @property
    def done_streaming(self) -> bool:
        return self._cursor >= self._episode.length

    def next_packet(self, realtime: bool = False) -> StreamPacket:
        if self.done_streaming:
            raise StopIteration
        if realtime and self._cursor:
            time.sleep(self.frame_interval_ms / 1000.0)
        index = self._cursor
        self._cursor += 1
        return StreamPacket(time.perf_counter_ns(), self._episode.frames[index],
                            self._episode.pcm[index],
                            stimulus_visible=index == self._episode.length - 1)

    def verify(self, action: int, stimulus_timestamp_ns: int, deadline_ms: float,
               speed_bonus: float) -> ReactionResult:
        response_ms = (time.perf_counter_ns() - stimulus_timestamp_ns) / 1_000_000
        correct = action == self._target
        missed = response_ms > deadline_ms
        speed = max(0.0, 1.0 - response_ms / deadline_ms)
        reward = (-1.0 if not correct else 1.0 + speed_bonus * speed)
        return ReactionResult(correct, action, self._target, response_ms, reward, missed)


class PixelStreamPolicy:
    """Accumulate public packets and emit an action without environment hooks."""

    def __init__(self, model: BitterLessonAgent, device: torch.device,
                 thought_depth: int = 1):
        self.model = model.eval()
        self.device = device
        self.thought_depth = thought_depth
        self.frames: list[np.ndarray] = []
        self.pcm: list[np.ndarray] = []

    def observe(self, packet: StreamPacket) -> None:
        self.frames.append(packet.frame)
        self.pcm.append(packet.pcm)

    @torch.inference_mode()
    def act(self) -> int:
        length = len(self.frames)
        unused = np.full(length, -1, dtype=np.int64)
        public = PublicEpisode(np.stack(self.frames), np.stack(self.pcm),
                               np.zeros(length, dtype=np.int64),
                               unused, unused.copy(), unused.copy(), length, 0)
        batch = collate_episodes([public])
        frames = batch["frames"].to(self.device)
        pcm = batch["pcm"].to(self.device)
        mask = batch["mask"].to(self.device)
        output = self.model(frames, pcm, mask)
        depth = min(self.thought_depth, output.answer_logits.shape[1]) - 1
        # Materialize the scalar here: the motor decision is not considered
        # emitted until accelerator work has completed.
        return int(output.answer_logits[0, depth].argmax().item())


def load_policy(checkpoint: Path, device: torch.device) -> BitterLessonAgent:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    model = BitterLessonAgent(**metadata["config"]).to(device)
    model.load_state_dict(payload["model"])
    return model


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    model = load_policy(args.checkpoint, device)
    difficulty = ReactionDifficulty(args.choices, args.distractors,
                                    args.delay_frames, args.audio_distractors,
                                    args.target_like_distractors,
                                    args.temporal_distractors)
    # Accelerator graph compilation and allocator setup are deployment startup
    # costs, not human-visible reaction latency. Exercise the identical public
    # stream path before starting the scored trials.
    for warmup in range(args.warmup_trials):
        stream = ChoiceReactionStream(args.seed - args.warmup_trials + warmup,
                                      difficulty,
                                      frame_interval_ms=1000.0 / args.fps)
        policy = PixelStreamPolicy(model, device, args.thought_depth)
        while not stream.done_streaming:
            policy.observe(stream.next_packet(realtime=False))
        policy.act()
    results: list[ReactionResult] = []
    wall_start = time.perf_counter()
    for trial in range(args.trials):
        stream = ChoiceReactionStream(args.seed + trial, difficulty,
                                      frame_interval_ms=1000.0 / args.fps)
        policy = PixelStreamPolicy(model, device, args.thought_depth)
        stimulus_timestamp = 0
        while not stream.done_streaming:
            packet = stream.next_packet(realtime=args.realtime)
            policy.observe(packet)
            if packet.stimulus_visible:
                stimulus_timestamp = packet.timestamp_ns
        action = policy.act()
        results.append(stream.verify(action, stimulus_timestamp, args.deadline_ms,
                                     args.speed_bonus))
    wall_seconds = time.perf_counter() - wall_start
    latencies = [result.response_ms for result in results]
    correct = sum(result.correct for result in results)
    return {
        "schema": "choice-reaction-realtime-v1",
        "trials": args.trials,
        "accuracy": correct / args.trials,
        "choices": args.choices,
        "distractors": args.distractors,
        "delay_frames": args.delay_frames,
        "audio_distractors": args.audio_distractors,
        "target_like_distractors": args.target_like_distractors,
        "temporal_distractors": args.temporal_distractors,
        "fps": args.fps,
        "realtime_frame_pacing": args.realtime,
        "deadline_ms": args.deadline_ms,
        "deadline_miss_rate": sum(result.deadline_missed for result in results) / args.trials,
        "mean_reward": sum(result.reward for result in results) / args.trials,
        "response_ms": {
            "mean": float(np.mean(latencies)),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
        },
        "wall_trials_per_second": args.trials / wall_seconds,
        "thought_depth": args.thought_depth,
        "warmup_trials": args.warmup_trials,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--warmup-trials", type=int, default=10)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--delay-frames", type=int, default=0)
    parser.add_argument("--audio-distractors", type=int, default=0)
    parser.add_argument("--target-like-distractors", type=int, default=0)
    parser.add_argument("--temporal-distractors", type=int, default=0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--speed-bonus", type=float, default=0.05)
    parser.add_argument("--thought-depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--device", default=("mps" if torch.backends.mps.is_available()
                                             else "cpu"))
    args = parser.parse_args()
    if (args.trials < 1 or args.warmup_trials < 0 or args.thought_depth < 1 or
            args.fps <= 0 or args.deadline_ms <= 0):
        raise ValueError("trials, thought depth, fps, and deadline must be positive")
    report = benchmark(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
