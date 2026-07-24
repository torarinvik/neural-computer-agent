from __future__ import annotations

import torch
from torch import nn

from experiments.event_stream_snake.model import FrozenSmolActionListener


class SelectiveAdapter(nn.Module):
    def __init__(self, pixels: int, width: int, mode: str, random_keep: float = 0.1):
        super().__init__()
        if mode not in ("dense", "fixed", "learned", "random"):
            raise ValueError(mode)
        self.mode = mode
        self.gate_override: str | None = None
        self.random_keep = random_keep
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(), nn.Flatten())
        with torch.no_grad():
            vision_width = self.vision_encoder(torch.zeros(1, 4, pixels, pixels)).shape[-1]
        self.audio_encoder = nn.Sequential(nn.Linear(64, 64), nn.GELU())
        self.vision_projector = nn.Sequential(nn.Linear(vision_width + 2, 128), nn.GELU(), nn.Linear(128, width))
        self.audio_projector = nn.Sequential(nn.Linear(66, 128), nn.GELU(), nn.Linear(128, width))
        self.vision_gate = nn.Sequential(nn.Linear(vision_width + 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.audio_gate = nn.Sequential(nn.Linear(66, 64), nn.GELU(), nn.Linear(64, 1))
        # Start open so the representation can learn before the small emission
        # term teaches the content-aware gates to become selective.
        nn.init.constant_(self.vision_gate[-1].bias, 1.0)
        nn.init.constant_(self.audio_gate[-1].bias, 0.5)

    @staticmethod
    def straight_through(probability: torch.Tensor) -> torch.Tensor:
        hard = (probability >= 0.5).to(probability.dtype)
        return hard.detach() - probability.detach() + probability

    def forward(self, frames: torch.Tensor, audio: torch.Tensor,
                compact: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, channels, height, width = frames.shape
        encoded = self.vision_encoder(frames.reshape(batch * time, channels, height, width)).reshape(batch, time, -1)
        delta = torch.zeros(batch, time, 1, device=frames.device)
        delta[:, 0] = 1.0
        delta[:, 1:] = (frames[:, 1:] - frames[:, :-1]).abs().mean((2, 3, 4)).unsqueeze(-1)
        position = torch.linspace(0, 1, time, device=frames.device).view(1, time, 1).expand(batch, -1, -1)
        vision_input = torch.cat((encoded, delta, position), -1)
        audio_encoded = self.audio_encoder(audio)
        energy = audio.square().mean(-1, keepdim=True).sqrt()
        audio_input = torch.cat((audio_encoded, energy, position), -1)
        active_mode = self.gate_override or self.mode
        if active_mode == "dense":
            vision_probability = torch.ones_like(delta)
            audio_probability = torch.ones_like(energy)
        elif active_mode == "fixed":
            vision_probability = (delta >= 0.002).float()
            audio_probability = (energy >= 0.02).float()
        elif active_mode == "random":
            # Deterministic pseudo-random masks make evaluation reproducible and
            # independent of privileged labels.
            frame_mean = frames.mean((2, 3, 4), keepdim=False).unsqueeze(-1)
            vision_random = torch.frac(torch.abs(
                torch.sin(position * 12.9898 + frame_mean * 78.233) * 43758.5453))
            audio_random = torch.frac(torch.abs(
                torch.sin(position * 39.3467 + energy * 11.135) * 24634.6345))
            vision_probability = (vision_random < self.random_keep).float()
            audio_probability = (audio_random < self.random_keep).float()
        else:
            vision_probability = torch.sigmoid(self.vision_gate(vision_input))
            audio_probability = torch.sigmoid(self.audio_gate(audio_input))
        if active_mode == "learned":
            vision_mask = self.straight_through(vision_probability) if self.training else (vision_probability >= 0.5).float()
            audio_mask = self.straight_through(audio_probability) if self.training else (audio_probability >= 0.5).float()
        else:
            vision_mask, audio_mask = vision_probability, audio_probability
        vision_tokens = torch.tanh(self.vision_projector(vision_input)) * vision_mask
        audio_tokens = torch.tanh(self.audio_projector(audio_input)) * audio_mask
        tokens = torch.stack((vision_tokens, audio_tokens), 2).reshape(batch, time * 2, -1)
        hard_mask = torch.stack((vision_mask, audio_mask), 2).reshape(batch, time * 2) >= 0.5
        if compact:
            if batch != 1:
                raise ValueError("compact inference requires batch one")
            selected = tokens[0, hard_mask[0]]
            if len(selected) == 0:
                selected = tokens[0, :1]
            tokens = selected[None]
            hard_mask = torch.ones((1, len(selected)), dtype=torch.bool, device=frames.device)
        return tokens, hard_mask, {
            "vision_probability": vision_probability,
            "audio_probability": audio_probability,
            "vision_event_target": (delta >= 0.002).float(),
            "audio_event_target": (energy >= 0.02).float(),
            "vision_emissions": vision_mask.detach().sum(1),
            "audio_emissions": audio_mask.detach().sum(1),
        }


class SelectiveController(nn.Module):
    def __init__(self, pixels: int, listener: FrozenSmolActionListener, mode: str,
                 random_keep: float = 0.1):
        super().__init__()
        self.listener = listener
        self.adapter = SelectiveAdapter(pixels, listener.width, mode, random_keep)

    def forward(self, frames: torch.Tensor, audio: torch.Tensor,
                compact: bool = False):
        tokens, mask, audit = self.adapter(frames, audio, compact)
        return self.listener(tokens, mask), audit

    def train(self, mode: bool = True):
        super().train(mode)
        self.listener.eval()
        return self
