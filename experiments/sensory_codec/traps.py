from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .games import make_game
from .runtime import SensoryPacket


def _signal_packet(direction: int, sequence: int = 6, size: int = 10) -> SensoryPacket:
    env = make_game("signal", size=size, seed=900 + direction)
    env.signal_direction = direction
    env._set_signal_target()  # authored trap state; never passed across the runtime boundary
    vision = env.observe(target_visible=True, detail_visible=True)
    audio = env.raw_audio()
    text = env.raw_text()
    return SensoryPacket(
        np.stack([vision.copy() for _ in range(sequence)]),
        np.stack([audio.copy() for _ in range(sequence)]),
        np.stack([text.copy() for _ in range(sequence)]),
    )


@torch.no_grad()
def _infer(model: nn.Module, device: torch.device,
           packet: SensoryPacket) -> tuple[int, np.ndarray, np.ndarray]:
    packet.validate()
    vision = torch.from_numpy(packet.vision[None]).float().to(device)
    audio = torch.from_numpy(packet.audio[None]).float().to(device)
    text = torch.from_numpy(packet.text[None]).long().to(device)
    outputs, code, _ = model(vision, audio, text)
    probabilities = outputs["action"].float().softmax(-1)[0].cpu().numpy()
    return int(probabilities.argmax()), probabilities, code[0].float().cpu().numpy()


def _entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-9, 1.0)
    return float(-(safe * np.log(safe)).sum() / math.log(len(safe)))


@torch.no_grad()
def modality_trap_audit(model: nn.Module, device: torch.device,
                        sequence: int = 6, size: int = 10) -> dict:
    """Counterfactual grounding exam using only typed raw sensory packets."""
    model.eval()
    bases = [_signal_packet(direction, sequence, size) for direction in range(4)]
    categories: dict[str, list[bool]] = {
        "congruent": [], "audio_only": [], "text_only": [],
        "corrupt_text": [], "corrupt_audio": [],
        "stale_audio": [], "stale_text": [],
    }
    congruent_entropies, missing_entropies = [], []
    audio_pair_distances, text_pair_distances = [], []

    audio_only_codes, text_only_codes = [], []
    for direction, base in enumerate(bases):
        predicted, probabilities, _ = _infer(model, device, base)
        categories["congruent"].append(predicted == direction)
        congruent_entropies.append(_entropy(probabilities))

        zero_text = SensoryPacket(base.vision, base.audio, np.zeros_like(base.text))
        predicted, _, code = _infer(model, device, zero_text)
        categories["audio_only"].append(predicted == direction)
        audio_only_codes.append(code)

        zero_audio = SensoryPacket(base.vision, np.zeros_like(base.audio), base.text)
        predicted, _, code = _infer(model, device, zero_audio)
        categories["text_only"].append(predicted == direction)
        text_only_codes.append(code)

        corrupt_text = base.text.copy()
        nonzero = np.argwhere(corrupt_text[-1] != 0)
        if len(nonzero):
            corrupt_text[:, int(nonzero[-1, 0])] = ord("?")
        predicted, _, _ = _infer(model, device, SensoryPacket(base.vision, base.audio, corrupt_text))
        categories["corrupt_text"].append(predicted == direction)

        phase = np.arange(base.audio.shape[-1], dtype=np.float32)
        noise = (0.7 * np.sin(2 * np.pi * 17 * phase / len(phase))).astype(np.float32)
        corrupt_audio = np.broadcast_to(noise, base.audio.shape).copy()
        predicted, _, _ = _infer(model, device, SensoryPacket(base.vision, corrupt_audio, base.text))
        categories["corrupt_audio"].append(predicted == direction)

        other = bases[(direction + 1) % 4]
        stale_audio = other.audio.copy()
        stale_audio[-1] = base.audio[-1]
        predicted, _, _ = _infer(model, device, SensoryPacket(base.vision, stale_audio, base.text))
        categories["stale_audio"].append(predicted == direction)

        stale_text = other.text.copy()
        stale_text[-1] = base.text[-1]
        predicted, _, _ = _infer(model, device, SensoryPacket(base.vision, base.audio, stale_text))
        categories["stale_text"].append(predicted == direction)

        missing = SensoryPacket(base.vision, np.zeros_like(base.audio), np.zeros_like(base.text))
        _, missing_probabilities, _ = _infer(model, device, missing)
        missing_entropies.append(_entropy(missing_probabilities))

    for index in range(4):
        following = (index + 1) % 4
        audio_pair_distances.append(float(np.linalg.norm(
            audio_only_codes[index] - audio_only_codes[following])))
        text_pair_distances.append(float(np.linalg.norm(
            text_only_codes[index] - text_only_codes[following])))

    accuracy = {name: float(np.mean(values)) for name, values in categories.items()}
    return {
        "accuracy": accuracy,
        "corruption_mean_accuracy": float(np.mean([
            accuracy["corrupt_text"], accuracy["corrupt_audio"]])),
        "stale_mean_accuracy": float(np.mean([
            accuracy["stale_audio"], accuracy["stale_text"]])),
        "missing_entropy": float(np.mean(missing_entropies)),
        "congruent_entropy": float(np.mean(congruent_entropies)),
        "missing_entropy_delta": float(np.mean(missing_entropies) - np.mean(congruent_entropies)),
        "audio_minimal_pair_code_distance": float(np.mean(audio_pair_distances)),
        "text_minimal_pair_code_distance": float(np.mean(text_pair_distances)),
        "notes": {
            "corrupt_text": "audio remains valid; one visible command character is invalidated",
            "corrupt_audio": "text remains valid; waveform is replaced by out-of-family interference",
            "stale": "old history conflicts but the newest sample is correct",
            "missing_entropy": "should rise when both command channels disappear",
        },
    }
