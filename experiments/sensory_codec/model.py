from __future__ import annotations

import torch
from torch import nn


TASK_DIMS = {
    "action": 4,
    "horizontal": 3,
    "vertical": 3,
    "direction": 4,
    "danger": 4,
    "ate": 2,
}


class TaskHeads(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(width, dim) for name, dim in TASK_DIMS.items()})

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(hidden) for name, head in self.heads.items()}


class Listener(nn.Module):
    """Small listener whose input has the same shape as the learned sensory code."""

    def __init__(self, code_dim: int = 16, width: int = 96):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(code_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.tasks = TaskHeads(width)

    def forward(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.tasks(self.backbone(code))


class Streamer(nn.Module):
    def __init__(self, board_pixels: int, code_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            encoded = self.encoder(torch.zeros(1, 4, board_pixels, board_pixels)).shape[-1]
        self.temporal = nn.GRU(encoded, hidden, batch_first=True)
        self.code = nn.Linear(hidden, code_dim)
        self.gate = nn.Linear(hidden, code_dim)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels, height, width = frames.shape
        features = self.encoder(frames.reshape(batch * time, channels, height, width))
        recurrent, _ = self.temporal(features.reshape(batch, time, -1))
        final = recurrent[:, -1]
        gate = torch.sigmoid(self.gate(final))
        return torch.tanh(self.code(final)) * gate, gate


class SmolVisionStreamer(nn.Module):
    """Frozen SmolVLM2 video tower followed by a trainable temporal bottleneck."""

    def __init__(self, model_name: str, code_dim: int = 16, hidden: int = 128,
                 image_size: int = 128, local_files_only: bool = False,
                 randomize_backbone: bool = False, vision_batch: int = 32):
        super().__init__()
        from transformers import AutoModelForImageTextToText

        loaded = AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=local_files_only, dtype=torch.float16)
        self.vision_model = loaded.model.vision_model
        del loaded
        if randomize_backbone:
            def reset(module: nn.Module) -> None:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self.vision_model.apply(reset)
        for parameter in self.vision_model.parameters():
            parameter.requires_grad = False
        width = self.vision_model.config.hidden_size
        self.temporal = nn.GRU(width, hidden, batch_first=True)
        self.code = nn.Linear(hidden, code_dim)
        self.gate = nn.Linear(hidden, code_dim)
        self.image_size = image_size
        self.vision_batch = vision_batch

    def _rgb(self, frames: torch.Tensor) -> torch.Tensor:
        # Fixed rendering is part of the sensory adapter, not privileged state:
        # red=structure/detail, green=controlled agent, blue=target.
        red = torch.clamp(frames[:, 0] + 0.5 * frames[:, 1], 0, 1)
        rgb = torch.stack((red, frames[:, 2], frames[:, 3]), dim=1)
        rgb = nn.functional.interpolate(rgb, size=(self.image_size, self.image_size),
                                        mode="nearest")
        return rgb * 2.0 - 1.0

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels, height, width = frames.shape
        flattened = frames.reshape(batch * time, channels, height, width)
        rgb = self._rgb(flattened)
        dtype = next(self.vision_model.parameters()).dtype
        pooled = []
        with torch.no_grad():
            for chunk in rgb.split(self.vision_batch):
                hidden = self.vision_model(chunk.to(dtype)).last_hidden_state
                pooled.append(hidden.mean(dim=1).float())
        features = torch.cat(pooled).reshape(batch, time, -1)
        recurrent, _ = self.temporal(features)
        final = recurrent[:, -1]
        gate = torch.sigmoid(self.gate(final))
        return torch.tanh(self.code(final)) * gate, gate


class MultimodalStreamer(nn.Module):
    """Independent vision/audio/text encoders with a small fused state."""

    def __init__(self, board_pixels: int, code_dim: int = 16, hidden: int = 128,
                 audio_samples: int = 64, text_characters: int = 32,
                 vision_backend: str = "tiny", vision_model: str | None = None,
                 vision_size: int = 128, local_files_only: bool = False,
                 randomize_vision: bool = False):
        super().__init__()
        if vision_backend == "smol":
            if not vision_model:
                raise ValueError("vision_model is required for the smol vision backend")
            self.vision = SmolVisionStreamer(
                vision_model, code_dim=code_dim, hidden=hidden, image_size=vision_size,
                local_files_only=local_files_only, randomize_backbone=randomize_vision)
        elif vision_backend == "tiny":
            self.vision = Streamer(board_pixels, code_dim=code_dim, hidden=hidden)
        else:
            raise ValueError(f"unknown vision backend {vision_backend!r}")
        self.audio_samples = audio_samples
        self.text_characters = text_characters
        self.audio_encoder = nn.Sequential(nn.Linear(audio_samples, 64), nn.ReLU())
        self.audio_temporal = nn.GRU(64, 64, batch_first=True)
        self.audio_code = nn.Linear(64, code_dim)
        self.audio_gate = nn.Linear(64, code_dim)
        self.char_embedding = nn.Embedding(256, 24, padding_idx=0)
        self.text_temporal = nn.GRU(24, 64, batch_first=True)
        self.text_code = nn.Linear(64, code_dim)
        self.text_gate = nn.Linear(64, code_dim)
        self.fusion = nn.Sequential(nn.Linear(code_dim * 3, 64), nn.GELU(),
                                    nn.Linear(64, code_dim))

    def forward(self, frames: torch.Tensor, audio: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        vision_code, vision_gate = self.vision(frames)
        batch, time = frames.shape[:2]
        if audio is None:
            audio = frames.new_zeros((batch, time, self.audio_samples))
        audio_features = self.audio_encoder(audio.float())
        audio_state, _ = self.audio_temporal(audio_features)
        audio_final = audio_state[:, -1]
        audio_gate = torch.sigmoid(self.audio_gate(audio_final))
        audio_code = torch.tanh(self.audio_code(audio_final)) * audio_gate
        if text is None:
            text = torch.zeros((batch, time, self.text_characters), dtype=torch.long,
                               device=frames.device)
        characters = self.char_embedding(text.long().clamp(0, 255)).mean(dim=2)
        text_state, _ = self.text_temporal(characters)
        text_final = text_state[:, -1]
        text_gate = torch.sigmoid(self.text_gate(text_final))
        text_code = torch.tanh(self.text_code(text_final)) * text_gate
        modalities = {"vision": vision_code, "audio": audio_code, "text": text_code}
        fused = torch.tanh(self.fusion(torch.cat(tuple(modalities.values()), dim=-1)))
        gate = torch.stack((vision_gate, audio_gate, text_gate)).mean(0)
        return fused, gate, modalities


class CodecModel(nn.Module):
    def __init__(self, board_pixels: int, listener: Listener, frozen_listener: bool = True,
                 code_dim: int = 16, streamer_kwargs: dict | None = None):
        super().__init__()
        self.streamer = MultimodalStreamer(board_pixels, code_dim=code_dim,
                                           **(streamer_kwargs or {}))
        self.listener = listener
        if frozen_listener:
            for parameter in self.listener.parameters():
                parameter.requires_grad = False

    def forward(self, frames: torch.Tensor, audio: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        code, gate, _ = self.streamer(frames, audio, text)
        return self.listener(code), code, gate


class DirectModel(nn.Module):
    """Essential control: the same streamer with an ordinary trainable head."""

    def __init__(self, board_pixels: int, code_dim: int = 16,
                 streamer_kwargs: dict | None = None):
        super().__init__()
        self.streamer = MultimodalStreamer(board_pixels, code_dim=code_dim,
                                           **(streamer_kwargs or {}))
        self.tasks = TaskHeads(code_dim)

    def forward(self, frames: torch.Tensor, audio: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        code, gate, _ = self.streamer(frames, audio, text)
        return self.tasks(code), code, gate


class SmolLLMListener(nn.Module):
    """Frozen language-model listener with a trainable sensory-token projector.

    The vision tower is deliberately discarded. The language model receives only
    learned streamer tokens followed by one fixed, game-agnostic action prompt.
    """

    def __init__(self, model_name: str, code_dim: int = 16, sensory_tokens: int = 4,
                 local_files_only: bool = False, randomize_backbone: bool = False):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        loaded = AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=local_files_only, dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.text_model = loaded.model.text_model
        self.lm_head = loaded.lm_head
        del loaded
        if randomize_backbone:
            def reset(module: nn.Module) -> None:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self.text_model.apply(reset)
            self.lm_head.apply(reset)
        for parameter in self.text_model.parameters():
            parameter.requires_grad = False
        for parameter in self.lm_head.parameters():
            parameter.requires_grad = False
        hidden = self.text_model.config.hidden_size
        self.sensory_tokens = sensory_tokens
        self.modality_projectors = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(code_dim, hidden), nn.GELU(),
                                nn.Linear(hidden, sensory_tokens * hidden))
            for name in ("vision", "audio", "text")
        })
        self.auxiliary = TaskHeads(code_dim)
        prompt_ids = tokenizer.encode(
            "Choose one action: UP RIGHT DOWN LEFT\nAction:", add_special_tokens=True,
            return_tensors="pt")
        with torch.no_grad():
            prompt = self.text_model.get_input_embeddings()(prompt_ids).detach()
        self.register_buffer("prompt_embeddings", prompt, persistent=True)
        embedding = self.text_model.get_input_embeddings()
        for name in ("vision", "audio", "text"):
            for boundary, literal in (("open", f"<{name}>"), ("close", f"</{name}>")):
                ids = tokenizer.encode(literal, add_special_tokens=False, return_tensors="pt")
                with torch.no_grad():
                    value = embedding(ids).detach()
                self.register_buffer(f"{name}_{boundary}", value, persistent=True)
        action_ids = []
        for token in ("UP", "RIGHT", "DOWN", "LEFT"):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"action {token!r} is not one tokenizer token: {encoded}")
            action_ids.append(encoded[0])
        self.register_buffer("action_token_ids", torch.tensor(action_ids), persistent=True)
        self.randomized_backbone = randomize_backbone

    def forward(self, code: torch.Tensor,
                modalities: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        batch = len(code)
        prompt = self.prompt_embeddings.expand(batch, -1, -1)
        if modalities is None:
            modalities = {name: code for name in ("vision", "audio", "text")}
        segments = []
        for name in ("vision", "audio", "text"):
            opened = getattr(self, f"{name}_open").expand(batch, -1, -1)
            closed = getattr(self, f"{name}_close").expand(batch, -1, -1)
            sensory = self.modality_projectors[name](modalities[name]).reshape(
                batch, self.sensory_tokens, -1).to(prompt.dtype)
            segments.extend((opened, sensory, closed))
        embeddings = torch.cat((*segments, prompt), dim=1)
        hidden = self.text_model(inputs_embeds=embeddings, use_cache=False,
                                 return_dict=True).last_hidden_state[:, -1]
        vocabulary_logits = self.lm_head(hidden)
        outputs = self.auxiliary(code)
        outputs["action"] = vocabulary_logits.index_select(-1, self.action_token_ids)
        return outputs


class LLMCodecModel(nn.Module):
    def __init__(self, board_pixels: int, listener: SmolLLMListener, code_dim: int = 16,
                 streamer_kwargs: dict | None = None):
        super().__init__()
        self.streamer = MultimodalStreamer(board_pixels, code_dim=code_dim,
                                           **(streamer_kwargs or {}))
        self.listener = listener

    def forward(self, frames: torch.Tensor, audio: torch.Tensor | None = None,
                text: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        code, gate, modalities = self.streamer(frames, audio, text)
        return self.listener(code, modalities), code, gate
