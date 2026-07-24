from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


ACTION_TOKENS = ("<UP>", "<RIGHT>", "<DOWN>", "<LEFT>")


def parse_action(token: str) -> int:
    try:
        return ACTION_TOKENS.index(token)
    except ValueError as error:
        raise ValueError(f"invalid action emission {token!r}") from error


@dataclass(frozen=True, slots=True)
class SensoryPacket:
    """The complete runtime input; every field is raw observable evidence."""

    vision: np.ndarray
    audio: np.ndarray
    text: np.ndarray

    def validate(self) -> None:
        if self.vision.ndim != 4 or self.vision.shape[1] != 4:
            raise ValueError("vision must be [time, 4, height, width]")
        if self.audio.ndim != 2 or self.audio.shape[0] != self.vision.shape[0]:
            raise ValueError("audio must be [time, samples]")
        if self.text.ndim != 2 or self.text.shape[0] != self.vision.shape[0]:
            raise ValueError("text must be [time, characters]")

    @classmethod
    def vision_only(cls, vision: np.ndarray, audio_samples: int = 64,
                    text_characters: int = 32) -> "SensoryPacket":
        time = vision.shape[0]
        return cls(vision, np.zeros((time, audio_samples), dtype=np.float32),
                   np.zeros((time, text_characters), dtype=np.int64))


@dataclass(frozen=True, slots=True)
class VisualActionAgent:
    """Runtime sensory firewall: pixels in, one action token out.

    The wrapper deliberately has no environment, reward, game id, teacher, label,
    or simulator-state argument. Privileged state is legal for dataset authorship
    and scoring, never for runtime inference.
    """

    model: nn.Module
    device: torch.device

    @torch.no_grad()
    def emit(self, packet: SensoryPacket) -> str:
        if not isinstance(packet, SensoryPacket):
            raise TypeError("runtime accepts only a SensoryPacket")
        packet.validate()
        vision = torch.from_numpy(packet.vision[None]).float().to(self.device)
        audio = torch.from_numpy(packet.audio[None]).float().to(self.device)
        text = torch.from_numpy(packet.text[None]).long().to(self.device)
        outputs, _, _ = self.model(vision, audio, text)
        action = int(outputs["action"].argmax(-1).item())
        return ACTION_TOKENS[action]
