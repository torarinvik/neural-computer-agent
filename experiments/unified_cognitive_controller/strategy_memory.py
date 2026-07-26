"""Small bounded RAM bank for task-agnostic latent utility strategies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


def physical_context_key(
        option_features: torch.Tensor,
        reward_signature: torch.Tensor | None = None) -> torch.Tensor:
    """Summarize visible physical state without task or utility labels."""
    if option_features.ndim != 3 or option_features.shape[-1] < 7:
        raise ValueError("expected [banks, options, >=7] physical features")
    rows = option_features[:, 1:, :]
    selected = rows[..., (0, 1, 2, 5, 6)]
    parts = [
        selected.mean(dim=(0, 1)),
        selected.std(dim=(0, 1), unbiased=False),
    ]
    if reward_signature is not None:
        if reward_signature.shape != (3,):
            raise ValueError("reward signature must contain three candidates")
        centered = reward_signature - reward_signature.mean()
        parts.append(torch.nn.functional.normalize(centered, dim=0))
    summary = torch.cat(parts)
    return torch.nn.functional.normalize(summary, dim=0)


@dataclass
class StrategyRetrieval:
    value: torch.Tensor
    slot: int | None
    similarity: float


class VerifierTrainedContextEncoder(nn.Module):
    """Tiny task-agnostic key metric trained from scalar verifier outcomes.

    The encoder learns only a positive weight per physical context dimension.
    This intentionally low-capacity first experiment tests whether addressing,
    rather than a larger strategy bank, is the current bottleneck.
    """

    def __init__(self, width: int = 13) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("context width must be positive")
        self.width = width
        self.log_scale = nn.Parameter(torch.zeros(width))

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.shape[-1] != self.width:
            raise ValueError("context has wrong width")
        scaled = contexts * self.log_scale.exp()
        return torch.nn.functional.normalize(scaled, dim=-1)

    def reinforce(
            self, query: torch.Tensor, stored_keys: torch.Tensor, *,
            selected_slot: int, verified_improvement: float,
            optimizer: torch.optim.Optimizer,
            temperature: float = 0.25) -> float:
        """Increase/decrease selected-key probability from verified utility."""
        if stored_keys.ndim != 2 or stored_keys.shape[-1] != self.width:
            raise ValueError("stored context keys have wrong shape")
        if not 0 <= selected_slot < stored_keys.shape[0]:
            raise IndexError("selected context slot out of range")
        logits = (
            self(stored_keys) @ self(query)
        ) / temperature
        log_probability = logits.log_softmax(dim=0)[selected_slot]
        advantage = torch.as_tensor(
            verified_improvement, device=query.device, dtype=query.dtype)
        loss = -(advantage * log_probability)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())


class LatentStrategyMemory:
    """A capacity-bounded content-addressed strategy working set."""

    schema = "unified-controller-latent-strategy-memory-v1"

    def __init__(
            self, *, capacity: int, key_width: int = 10,
            value_width: int = 2, device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32) -> None:
        if capacity < 1 or key_width < 1 or value_width < 1:
            raise ValueError("strategy-memory dimensions must be positive")
        self.capacity = capacity
        self.key_width = key_width
        self.value_width = value_width
        self.keys = torch.zeros(
            capacity, key_width, device=device, dtype=dtype)
        self.values = torch.zeros(
            capacity, value_width, device=device, dtype=dtype)
        self.usage = torch.zeros(
            capacity, device=device, dtype=torch.long)
        self.success = torch.zeros(
            capacity, device=device, dtype=torch.long)
        self.failure = torch.zeros(
            capacity, device=device, dtype=torch.long)
        self.count = 0

    def retrieve(
            self, key: torch.Tensor,
            fallback: torch.Tensor,
            encoder: VerifierTrainedContextEncoder | None = None,
            ) -> StrategyRetrieval:
        if key.shape != (self.key_width,):
            raise ValueError("strategy key has wrong width")
        if fallback.shape != (self.value_width,):
            raise ValueError("fallback strategy has wrong width")
        if self.count == 0:
            return StrategyRetrieval(fallback.clone(), None, 0.0)
        if encoder is None:
            stored = torch.nn.functional.normalize(
                self.keys[:self.count], dim=-1)
            query = torch.nn.functional.normalize(key, dim=0)
        else:
            stored = encoder(self.keys[:self.count]).detach()
            query = encoder(key).detach()
        similarities = stored @ query
        slot = int(similarities.argmax())
        self.usage[slot] += 1
        return StrategyRetrieval(
            self.values[slot].clone(), slot,
            float(similarities[slot]))

    def upsert(
            self, key: torch.Tensor, value: torch.Tensor, *,
            verified_improvement: float) -> int:
        if key.shape != (self.key_width,):
            raise ValueError("strategy key has wrong width")
        if value.shape != (self.value_width,):
            raise ValueError("strategy value has wrong width")
        if self.count < self.capacity:
            slot = self.count
            self.count += 1
        else:
            reliability = (
                (self.success.to(value.dtype) + 1)
                / (self.success + self.failure + 2).to(value.dtype))
            utility = self.usage.to(value.dtype) + reliability
            slot = int(utility[:self.count].argmin())
        self.keys[slot] = key
        self.values[slot] = value
        if verified_improvement > 0:
            self.success[slot] += 1
        else:
            self.failure[slot] += 1
        return slot

    def save(self, path: Path) -> None:
        torch.save({
            "schema": self.schema,
            "capacity": self.capacity,
            "key_width": self.key_width,
            "value_width": self.value_width,
            "count": self.count,
            "keys": self.keys,
            "values": self.values,
            "usage": self.usage,
            "success": self.success,
            "failure": self.failure,
        }, path)

    @classmethod
    def load(
            cls, path: Path, *,
            device: torch.device | str = "cpu") -> "LatentStrategyMemory":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload["schema"] != cls.schema:
            raise ValueError("unsupported strategy-memory schema")
        memory = cls(
            capacity=payload["capacity"],
            key_width=payload["key_width"],
            value_width=payload["value_width"],
            device=device, dtype=payload["keys"].dtype)
        memory.count = int(payload["count"])
        for field in (
                "keys", "values", "usage", "success", "failure"):
            getattr(memory, field).copy_(payload[field].to(device))
        return memory
