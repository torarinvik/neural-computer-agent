"""Causal sensory sparsification based only on RGB/PCM changes."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .host_client import HostPacket

@dataclass
class EventGate:
    frame_threshold: float = 0.0
    audio_rms_threshold: float = 0.0
    audio_silence_ms: int = 0
    mode: str = "fixed-gate"
    _last_frame: np.ndarray | None = None
    _last_audio_event_ms: int = -10**12
    _rng: np.random.Generator | None = None

    def __post_init__(self):
        if self.mode not in {"dense", "fixed-gate", "random-control", "learned-gate"}:
            raise ValueError(f"unknown gate mode: {self.mode}")
        if self.mode == "random-control":
            self._rng = np.random.default_rng(0)

    def accept(self, packet: HostPacket) -> bool:
        """Return whether this packet should be sent to the model."""
        if self.mode == "dense":
            self._last_frame = packet.frame.copy()
            return True
        if self.mode == "random-control":
            return bool(self._rng.random() < 0.25)
        frame_changed = self._last_frame is None
        if self._last_frame is not None:
            difference = np.mean(np.abs(packet.frame.astype(np.float32) - self._last_frame))
            frame_changed = difference > self.frame_threshold
        if frame_changed:
            self._last_frame = packet.frame.copy()
        rms = float(np.sqrt(np.mean(np.square(packet.pcm.astype(np.float32))))) if packet.pcm.size else 0.0
        audio_event = rms > self.audio_rms_threshold and (
            packet.timestamp_ms - self._last_audio_event_ms >= self.audio_silence_ms
        )
        if audio_event:
            self._last_audio_event_ms = packet.timestamp_ms
        return frame_changed or audio_event
