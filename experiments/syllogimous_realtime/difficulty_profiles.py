"""Difficulty policy shared by generation/evaluation tooling.

Difficulty changes the amount of information the player must retain, not the
answer contract.  The evaluator may use these profiles to choose premise
counts, Boolean depth, distractors, and sensory interference independently.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    premise_count: int
    boolean_depth: int
    distractor_count: int
    visual_change_threshold: int
    audio_silence_ms: int
    deadline_ms: int
    interference_permille: int

PROFILES = {
    "intro": DifficultyProfile("intro", 3, 1, 0, 0, 250, 30_000, 0),
    "standard": DifficultyProfile("standard", 6, 2, 2, 1, 180, 20_000, 100),
    "hard": DifficultyProfile("hard", 10, 4, 5, 2, 120, 15_000, 250),
    "max": DifficultyProfile("max", 16, 4, 10, 4, 80, 10_000, 450),
}

def profile(name: str) -> DifficultyProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown difficulty: {name}") from exc

def manifest() -> dict[str, dict]:
    return {name: asdict(value) for name, value in PROFILES.items()}
