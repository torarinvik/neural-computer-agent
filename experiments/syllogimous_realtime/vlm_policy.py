"""Optional Hugging Face VLM policy for the audiovisual-only host client."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from PIL import Image

from .host_client import HostPacket

ACTION_INSTRUCTION = (
    "You are playing a timed reasoning game. You receive only the current "
    "screen image. Respond with exactly one action: WAIT, NEXT, PREVIOUS, TRUE, or FALSE."
)
ACTION_PROMPT = "<image>\n" + ACTION_INSTRUCTION
ACTION_RE = re.compile(r"\b(WAIT|NEXT|PREVIOUS|TRUE|FALSE)\b", re.IGNORECASE)

def extract_action(text: str) -> str:
    match = ACTION_RE.search(text or "")
    return match.group(1).upper() if match else "WAIT"

@dataclass
class SmolVLMPolicy:
    processor: Any
    model: Any
    device: str
    max_new_tokens: int = 4
    image_size: int | None = None
    last_generated: str = ""

    @classmethod
    def from_pretrained(cls, model_name: str, *, device: str = "auto",
                        local_files_only: bool = False,
                        image_size: int | None = None,
                        max_new_tokens: int = 4,
                        dtype: str = "auto") -> "SmolVLMPolicy":
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for VLM inference") from exc
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        if image_size is not None:
            if image_size < 64 or image_size > 2048:
                raise ValueError("image_size must be between 64 and 2048")
            processor.image_processor.max_image_size = {"longest_edge": image_size}
        if dtype == "float32":
            model_dtype = torch.float32
        elif dtype == "float16":
            model_dtype = torch.float16
        elif device in ("cpu", "mps"):
            model_dtype = torch.float32
        else:
            model_dtype = "auto"
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=local_files_only,
            torch_dtype=model_dtype, device_map=device,
        )
        model.eval()
        resolved = str(next(model.parameters()).device)
        return cls(processor, model, resolved, max_new_tokens=max_new_tokens, image_size=image_size)

    def __call__(self, packet: HostPacket) -> str:
        import torch
        image = Image.fromarray(packet.frame, mode="RGB")
        if hasattr(self.processor, "apply_chat_template"):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": ACTION_INSTRUCTION}
            ]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            prompt = ACTION_PROMPT
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        model_dtype = next(self.model.parameters()).dtype
        inputs = {key: (value.to(self.device, dtype=model_dtype)
                        if hasattr(value, "to") and torch.is_floating_point(value)
                        else (value.to(self.device) if hasattr(value, "to") else value))
                  for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                            do_sample=False)
        # ``generate`` returns prompt + continuation.  Decoding the whole
        # sequence would see the prompt's action vocabulary (which starts with
        # WAIT) and could silently turn every model response into WAIT.
        prompt_length = inputs["input_ids"].shape[-1]
        continuation = generated[:, prompt_length:]
        text = self.processor.batch_decode(continuation, skip_special_tokens=True)[0]
        if not text.strip():
            text = ""
        self.last_generated = text
        return extract_action(text)

class AudioOnlyPolicy:
    """A causal PCM-only control; it never inspects RGB pixels."""
    def __call__(self, packet: HostPacket) -> str:
        rms = float(np.sqrt(np.mean(np.square(packet.pcm.astype(np.float32))))) if packet.pcm.size else 0.0
        return "NEXT" if rms > 3_000 else "WAIT"
