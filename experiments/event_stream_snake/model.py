from __future__ import annotations

import torch
from torch import nn


class EventAdapter(nn.Module):
    """Trainable audiovisual encoder with dense, fixed, or learned emission gates."""

    def __init__(self, board_pixels: int, output_width: int, mode: str = "learned",
                 vision_threshold: float = 0.002, audio_threshold: float = 0.02):
        super().__init__()
        if mode not in ("dense", "fixed", "learned"):
            raise ValueError(mode)
        self.mode = mode
        self.vision_threshold = vision_threshold
        self.audio_threshold = audio_threshold
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(), nn.Flatten())
        with torch.no_grad():
            vision_width = self.vision_encoder(
                torch.zeros(1, 4, board_pixels, board_pixels)).shape[-1]
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_width + 2, 128), nn.GELU(), nn.Linear(128, output_width))
        self.audio_encoder = nn.Sequential(nn.Linear(64, 64), nn.GELU())
        self.audio_projector = nn.Sequential(
            nn.Linear(66, 128), nn.GELU(), nn.Linear(128, output_width))
        self.vision_gate = nn.Sequential(nn.Linear(vision_width + 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.audio_gate = nn.Sequential(nn.Linear(66, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.constant_(self.vision_gate[-1].bias, 1.0)
        nn.init.constant_(self.audio_gate[-1].bias, -2.0)
        self.learned_vision_threshold = nn.Parameter(torch.tensor(vision_threshold))
        self.learned_audio_threshold = nn.Parameter(torch.tensor(audio_threshold))
        self.heartbeat = nn.Parameter(torch.zeros(output_width))

    @staticmethod
    def _straight_through(probability: torch.Tensor) -> torch.Tensor:
        hard = (probability >= 0.5).to(probability.dtype)
        return hard.detach() - probability.detach() + probability

    def forward(self, frames: torch.Tensor, audio: torch.Tensor,
                compact: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        visual = self.vision_encoder(flat).reshape(batch, time, -1)
        delta = torch.zeros(batch, time, 1, device=frames.device)
        delta[:, 0] = 1.0
        delta[:, 1:] = (frames[:, 1:] - frames[:, :-1]).abs().mean((2, 3, 4), keepdim=False).unsqueeze(-1)
        position = torch.linspace(0, 1, time, device=frames.device).view(1, time, 1).expand(batch, -1, -1)
        visual_input = torch.cat((visual, delta, position), dim=-1)
        audio_features = self.audio_encoder(audio.float())
        energy = audio.square().mean(-1, keepdim=True).sqrt()
        audio_input = torch.cat((audio_features, energy, position), dim=-1)
        if self.mode == "dense":
            vision_probability = torch.ones_like(delta)
            audio_probability = torch.ones_like(energy)
        elif self.mode == "fixed":
            vision_probability = (delta >= self.vision_threshold).float()
            audio_probability = (energy >= self.audio_threshold).float()
        else:
            # Learn the two interpretable event thresholds rather than an
            # unconstrained gate network. Sharp sigmoids provide training
            # gradients and converge to the hard deployment rule.
            vision_threshold = self.learned_vision_threshold.clamp(0.0, 0.25)
            audio_threshold = self.learned_audio_threshold.clamp(0.0, 1.0)
            vision_probability = torch.sigmoid((delta - vision_threshold) * 2000.0)
            audio_probability = torch.sigmoid((energy - audio_threshold) * 200.0)
        if self.mode == "learned":
            # Only two scalar thresholds receive straight-through gate gradients;
            # representations see the same hard events in training and inference.
            vision_gate = (self._straight_through(vision_probability) if self.training
                           else (vision_probability >= 0.5).float())
            audio_gate = (self._straight_through(audio_probability) if self.training
                          else (audio_probability >= 0.5).float())
        else:
            vision_gate, audio_gate = vision_probability, audio_probability
        # Keep learned soft events on a bounded manifold; unconstrained projector
        # magnitudes can destabilize gradients through a frozen transformer.
        vision_tokens = torch.tanh(self.vision_projector(visual_input)) * vision_gate
        audio_tokens = torch.tanh(self.audio_projector(audio_input)) * audio_gate
        # Interleave modalities in sensor-time order.
        tokens = torch.stack((vision_tokens, audio_tokens), dim=2).reshape(batch, time * 2, -1)
        hard_mask = torch.stack((vision_gate, audio_gate), dim=2).reshape(batch, time * 2) >= 0.5
        if compact:
            if batch != 1:
                raise ValueError("physical compaction is for single-stream inference")
            selected = tokens[0, hard_mask[0]]
            if len(selected) == 0:
                selected = self.heartbeat[None]
            tokens = selected[None]
            mask = torch.ones(1, len(selected), dtype=torch.bool, device=frames.device)
        else:
            mask = hard_mask
        audit = {
            "vision_probability": vision_probability,
            "audio_probability": audio_probability,
            "vision_event_target": (delta >= self.vision_threshold).float(),
            "audio_event_target": (energy >= self.audio_threshold).float(),
            "learned_vision_threshold": self.learned_vision_threshold.detach(),
            "learned_audio_threshold": self.learned_audio_threshold.detach(),
            "vision_emissions": vision_gate.detach().sum(1),
            "audio_emissions": audio_gate.detach().sum(1),
        }
        return tokens, mask, audit


class FrozenSmolActionListener(nn.Module):
    """Frozen SmolVLM2 text core consuming only adapter event embeddings."""

    def __init__(self, model_name: str, local_files_only: bool = False):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        loaded = AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=local_files_only, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.text_model = loaded.model.text_model
        self.lm_head = loaded.lm_head
        del loaded
        for parameter in self.text_model.parameters():
            parameter.requires_grad = False
        for parameter in self.lm_head.parameters():
            parameter.requires_grad = False
        prompt_ids = tokenizer.encode(
            "Observe the sensory events. Choose one action: UP RIGHT DOWN LEFT\nAction:",
            add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            prompt = self.text_model.get_input_embeddings()(prompt_ids).detach()
        self.register_buffer("prompt", prompt)
        action_ids = []
        for action in ("UP", "RIGHT", "DOWN", "LEFT"):
            ids = tokenizer.encode(action, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{action} is not one token: {ids}")
            action_ids.append(ids[0])
        self.register_buffer("action_ids", torch.tensor(action_ids))
        self.width = self.text_model.config.hidden_size

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        batch = len(events)
        prompt = self.prompt.expand(batch, -1, -1)
        embeddings = torch.cat((events.to(prompt.dtype), prompt), dim=1)
        prompt_mask = torch.ones(batch, prompt.shape[1], dtype=torch.bool, device=events.device)
        mask = torch.cat((event_mask, prompt_mask), dim=1)
        hidden = self.text_model(inputs_embeds=embeddings, attention_mask=mask,
                                 use_cache=False, return_dict=True).last_hidden_state[:, -1]
        return self.lm_head(hidden).index_select(-1, self.action_ids).float()


class EventSnakeController(nn.Module):
    def __init__(self, board_pixels: int, listener: FrozenSmolActionListener,
                 mode: str = "learned"):
        super().__init__()
        self.listener = listener
        self.adapter = EventAdapter(board_pixels, listener.width, mode=mode)

    def forward(self, frames: torch.Tensor, audio: torch.Tensor,
                compact: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        events, mask, audit = self.adapter(frames, audio, compact=compact)
        return self.listener(events, mask), audit

    def train(self, mode: bool = True):
        super().train(mode)
        # The listener is frozen and should never enable training-time dropout.
        self.listener.eval()
        return self
