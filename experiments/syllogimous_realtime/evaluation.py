"""Evaluation records and correctness-dominant real-time metrics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
from pathlib import Path

@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    family: str
    difficulty: str
    outcome: str                 # correct, wrong, timeout
    elapsed_ms: int
    deadline_ms: int
    inference_ms: int = 0
    sensory_tokens: int = 0

def reward(outcome: str, elapsed_ms: int, deadline_ms: int) -> float:
    if outcome == "correct":
        return 1.0 + 0.05 * max(0.0, min(1.0, (deadline_ms - elapsed_ms) / deadline_ms))
    return -1.0

def summarize(records: list[EpisodeRecord]) -> dict:
    groups: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        groups[(record.family, record.difficulty)].append(record)
    def one(items: list[EpisodeRecord]) -> dict:
        n = len(items)
        correct = sum(x.outcome == "correct" for x in items)
        return {
            "episodes": n,
            "accuracy": correct / n if n else 0.0,
            "timeout_rate": sum(x.outcome == "timeout" for x in items) / n if n else 0.0,
            "mean_latency_ms": sum(x.elapsed_ms for x in items) / n if n else 0.0,
            "p95_latency_ms": sorted(x.elapsed_ms for x in items)[max(0, int(n * .95) - 1)] if n else 0,
            "mean_reward": sum(reward(x.outcome, x.elapsed_ms, x.deadline_ms) for x in items) / n if n else 0.0,
            "mean_inference_ms": sum(x.inference_ms for x in items) / n if n else 0.0,
            "mean_sensory_tokens": sum(x.sensory_tokens for x in items) / n if n else 0.0,
        }
    return {"schema": "syllogimous.metrics.v1", "episodes": len(records),
            "overall": one(records),
            "by_family_difficulty": {f"{family}/{difficulty}": one(items)
                                      for (family, difficulty), items in sorted(groups.items())}}

def write_summary(records: list[EpisodeRecord], path: str | Path) -> None:
    Path(path).write_text(json.dumps(summarize(records), indent=2, sort_keys=True) + "\n")
