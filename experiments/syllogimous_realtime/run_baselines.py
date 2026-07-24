#!/usr/bin/env python3
"""Run comparable causal baselines without exposing evaluator state to policies."""
from __future__ import annotations

import argparse
import random
import hashlib
from dataclasses import dataclass

from .baselines import Baseline
from .environment import Action, RealtimeEpisode, generate_question
from .evaluation import EpisodeRecord, summarize, write_summary

class Clock:
    def __init__(self): self.ns = 0
    def __call__(self): return self.ns
    def advance_ms(self, ms: int): self.ns += ms * 1_000_000

@dataclass
class Policy:
    premises: int
    rng: random.Random
    baseline: Baseline
    step: int = 0

    def act(self, packet, answer: bool | None = None) -> Action:
        """Packet is the only model input; answer is test-harness-only oracle data."""
        if self.step < self.premises:
            result = Action.NEXT
        elif self.baseline is Baseline.TEXT_ORACLE:
            result = Action.TRUE if answer else Action.FALSE
        else:
            # These controls intentionally use only the received modality.  They
            # are not semantic OCR models, but they are genuine packet-only
            # policies rather than hidden-answer placeholders.
            if self.baseline is Baseline.VISION_ONLY:
                data = packet.frame.tobytes()
            elif self.baseline is Baseline.VISION_AUDIO:
                data = packet.frame.tobytes() + packet.pcm.tobytes()
            elif self.baseline is Baseline.FULL_STREAM:
                digest = hashlib.sha256(packet.frame.tobytes() + packet.pcm.tobytes()).digest()
                data = digest + int(packet.timestamp_ms).to_bytes(8, "little", signed=False)
            else:
                data = self.rng.getrandbits(64).to_bytes(8, "little")
            bit = hashlib.blake2b(data, digest_size=1).digest()[0] & 1
            result = Action.TRUE if bit else Action.FALSE
        self.step += 1
        return result

def run(baseline: Baseline, episodes: int, premises: int, deadline_ms: int,
        inference_ms: int, seed: int) -> list[EpisodeRecord]:
    records = []
    for offset in range(episodes):
        episode_seed = seed + offset
        question = generate_question(episode_seed, premises=premises)
        clock = Clock()
        episode = RealtimeEpisode(question, deadline_ms=deadline_ms, clock_ns=clock)
        policy = Policy(premises, random.Random(episode_seed + 17), baseline)
        result = None
        while result is None or not result.done:
            clock.advance_ms(inference_ms)
            observation = result.observation if result is not None else episode.step(Action.WAIT).observation
            result = episode.step(policy.act(observation, question.answer if baseline is Baseline.TEXT_ORACLE else None))
        outcome = "correct" if result.outcome == "right" else result.outcome
        records.append(EpisodeRecord(episode_seed, "chain", "standard", outcome,
                                     observation.timestamp_ms, deadline_ms,
                                     inference_ms=inference_ms))
    return records

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=[x.value for x in Baseline], default=Baseline.RANDOM.value)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--premises", type=int, default=6)
    parser.add_argument("--deadline-ms", type=int, default=8_000)
    parser.add_argument("--inference-ms", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="baseline_metrics.json")
    args = parser.parse_args()
    records = run(Baseline(args.baseline), args.episodes, args.premises,
                  args.deadline_ms, args.inference_ms, args.seed)
    write_summary(records, args.output)
    print(summarize(records)["overall"])
    return 0

if __name__ == "__main__": raise SystemExit(main())
