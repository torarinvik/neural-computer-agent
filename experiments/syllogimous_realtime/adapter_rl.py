"""Frozen-listener streamer-gate RL.

The learner sees sensory tensors and the text action emitted by a frozen
listener.  The rollout callback is the only place where environment reward is
observed; no question or evaluator state is passed to the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

@dataclass(frozen=True)
class AdapterConfig:
    variant: str
    latency_weight: float = 0.01
    reward_weight: float = 1.0
    temperature: float = 1.0
    modalities: int = 2
    min_emission_rate: float = 0.10
    # Large enough that an all-silent rollout is worse than emitting one
    # modality, while still far below the correctness reward weight.
    silence_weight: float = 0.20

class StreamerGate(nn.Module):
    def __init__(self, feature_dim: int = 8, modalities: int = 2):
        super().__init__()
        self.logits = nn.Sequential(nn.Linear(feature_dim, 32), nn.GELU(), nn.Linear(32, modalities))

    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(features) / temperature
        distribution = torch.distributions.Bernoulli(logits=logits)
        decisions = distribution.sample()
        return decisions, distribution.log_prob(decisions).sum(-1)

def packet_features(packet, previous_frame=None) -> torch.Tensor:
    """Eight causal features from raw RGB/PCM; no evaluator metadata."""
    frame = packet.frame.astype("float32") / 255.0
    pcm = packet.pcm.astype("float32")
    delta = 0.0 if previous_frame is None else float(abs(frame - previous_frame).mean())
    rms = float((pcm * pcm).mean() ** 0.5) / 32768.0 if pcm.size else 0.0
    peak = float(abs(pcm).max()) / 32768.0 if pcm.size else 0.0
    return torch.tensor([
        float(frame.mean()), float(frame.std()), delta,
        float((frame > 0.5).mean()), rms, peak,
        float((pcm > 0).mean()) if pcm.size else 0.0,
        float(packet.timestamp_ms % 1000) / 1000.0,
    ], dtype=torch.float32)

def freeze_listener(listener: nn.Module) -> None:
    listener.eval()
    for parameter in listener.parameters():
        parameter.requires_grad_(False)

def adapter_objective(reward: torch.Tensor, log_prob: torch.Tensor,
                      emitted_tokens: torch.Tensor, config: AdapterConfig) -> torch.Tensor:
    """Policy-gradient loss with accuracy-dominant latency and anti-silence shaping.

    ``emitted_tokens`` is the number of modality decisions emitted for each
    rollout.  A small coverage floor prevents a sparse policy from earning a
    deceptively good objective by suppressing every packet; task reward remains
    the dominant term.
    """
    shaped = shape_reward(reward, emitted_tokens, config)
    advantage = (shaped - shaped.detach().mean())
    return -(advantage * log_prob).mean()


def shape_reward(reward: torch.Tensor, emitted_tokens: torch.Tensor,
                 config: AdapterConfig) -> torch.Tensor:
    """Apply correctness-dominant latency and anti-silence shaping."""
    shaped = config.reward_weight * reward - config.latency_weight * emitted_tokens
    rate = emitted_tokens / max(1, config.modalities)
    return shaped - config.silence_weight * torch.relu(config.min_emission_rate - rate)

def train_gate(gate: StreamerGate, listener: nn.Module,
               rollout: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
               batches: list[torch.Tensor], config: AdapterConfig, steps: int = 1_000) -> list[float]:
    """Train only ``gate``. ``rollout`` returns (reward, emitted_token_count)."""
    if config.variant == "random-control":
        return []
    freeze_listener(listener)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=3e-4)
    history: list[float] = []
    for step in range(steps):
        features = batches[step % len(batches)]
        decisions, log_prob = gate(features, config.temperature)
        reward, emitted = rollout(decisions.detach())
        loss = adapter_objective(reward, log_prob, emitted, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
