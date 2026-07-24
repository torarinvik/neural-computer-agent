from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import EventSnakeController


ACTION_TOKENS = ("<UP>", "<RIGHT>", "<DOWN>", "<LEFT>")


@dataclass(frozen=True, slots=True)
class AVPacket:
    vision: np.ndarray
    audio: np.ndarray

    def validate(self) -> None:
        if self.vision.ndim != 4 or self.vision.shape[1] != 4:
            raise ValueError("vision must be [time,4,height,width]")
        if self.audio.ndim != 2 or self.audio.shape != (len(self.vision), 64):
            raise ValueError("audio must be synchronized [time,64] PCM")


@dataclass(frozen=True, slots=True)
class EventActionAgent:
    model: EventSnakeController
    device: torch.device

    @torch.no_grad()
    def emit(self, packet: AVPacket) -> tuple[str, dict[str, float]]:
        if not isinstance(packet, AVPacket):
            raise TypeError("runtime accepts only AVPacket")
        packet.validate()
        frames = torch.from_numpy(packet.vision[None]).float().to(self.device)
        audio = torch.from_numpy(packet.audio[None]).float().to(self.device)
        logits, audit = self.model(frames, audio, compact=True)
        action = int(logits.argmax(-1).item())
        counts = {key: float(value.sum().item()) for key, value in audit.items()
                  if key.endswith("emissions")}
        return ACTION_TOKENS[action], counts
