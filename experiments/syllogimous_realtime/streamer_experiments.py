"""Frozen-LLM streamer-adapter experiment matrix."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class StreamerVariant:
    name: str
    vision_mode: str
    audio_mode: str
    train_gate: bool
    train_representation: bool
    latency_weight: float = 0.01
    reward_weight: float = 1.0

VARIANTS = (
    StreamerVariant("dense", "dense", "dense", False, True),
    StreamerVariant("fixed-gate", "fixed-threshold", "fixed-threshold", False, True),
    StreamerVariant("random-control", "random", "random", False, False),
    StreamerVariant("learned-gate", "learned-threshold", "learned-threshold", True, True),
)

ABLATIONS = ("rgb-only", "audio-only", "rgb+audio", "full-sparse", "dense-control")
FRAME_THRESHOLDS = (0, 1, 2, 4, 8)
AUDIO_SILENCE_THRESHOLDS_MS = (40, 80, 120, 180, 250)

def manifest() -> dict:
    return {"llm_frozen": True, "variants": [asdict(x) for x in VARIANTS],
            "ablations": list(ABLATIONS), "frame_thresholds": list(FRAME_THRESHOLDS),
            "audio_silence_thresholds_ms": list(AUDIO_SILENCE_THRESHOLDS_MS),
            "accuracy_dominates_latency": True}
