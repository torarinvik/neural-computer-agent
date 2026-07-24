"""Named baseline policies; no baseline receives evaluator state."""
from __future__ import annotations
from enum import Enum

class Baseline(str, Enum):
    RANDOM = "random-policy"
    TEXT_ORACLE = "text-only-oracle"
    VISION_ONLY = "vision-only"
    VISION_AUDIO = "vision-plus-audio"
    FULL_STREAM = "full-streaming-model"

BASELINE_INPUTS = {
    Baseline.RANDOM: "none",
    Baseline.TEXT_ORACLE: "rendered-text-equivalent",
    Baseline.VISION_ONLY: "rgb",
    Baseline.VISION_AUDIO: "rgb+pcm",
    Baseline.FULL_STREAM: "event-sparse-rgb+event-sparse-pcm",
}
