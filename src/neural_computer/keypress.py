"""Replaceable keypress input/output adapters.

Key indices are owned by the external keyboard backend.  The controller sees
only the learned feedback embedding from :class:`KeypressEncoder` and emits
only an opaque intention.  :class:`KeypressDecoder` is the replaceable output
adapter that turns that intention into key-index logits; it does not add a
keypress-specific reasoning branch to the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .interface import IntentEvent

KEYPRESS_ENCODER_SCHEMA = "neural-computer.keypress-encoder.v1"
KEYPRESS_DECODER_SCHEMA = "neural-computer.keypress-decoder.v1"


@dataclass(frozen=True)
class KeypressDecision:
    """A sampled or deterministic external keypress decision."""

    key_index: torch.Tensor
    logits: torch.Tensor
    propensity: torch.Tensor

    @property
    def action(self) -> torch.Tensor:
        """Expose the generic live-decoder action ABI."""

        return self.key_index

    def validate(self, *, key_count: int, batch: int | None = None) -> KeypressDecision:
        if self.logits.ndim != 2 or self.logits.shape[1] != key_count:
            raise ValueError("keypress logits have the wrong shape")
        if self.key_index.ndim != 1 or self.key_index.shape[0] != self.logits.shape[0]:
            raise ValueError("keypress indices must align with logits")
        if self.key_index.dtype != torch.long:
            raise ValueError("keypress indices must be int64")
        if batch is not None and self.key_index.shape[0] != batch:
            raise ValueError("keypress batch does not match logits")
        if bool(torch.any(self.key_index < 0)) or bool(
            torch.any(self.key_index >= key_count)
        ):
            raise ValueError("keypress index is outside the decoder key count")
        if self.propensity.shape != self.key_index.shape:
            raise ValueError("keypress propensity must align with indices")
        if not bool(torch.isfinite(self.logits).all()) or not bool(
            torch.isfinite(self.propensity).all()
        ):
            raise ValueError("keypress decision must be finite")
        if bool(torch.any((self.propensity <= 0.0) | (self.propensity > 1.0))):
            raise ValueError("keypress propensity must lie in (0, 1]")
        return self


class KeypressEncoder(nn.Module):
    """Encode an external key index into opaque controller feedback."""

    schema = KEYPRESS_ENCODER_SCHEMA

    def __init__(self, key_count: int, feedback_width: int) -> None:
        super().__init__()
        if min(key_count, feedback_width) < 1:
            raise ValueError("keypress encoder dimensions must be positive")
        self.key_count = int(key_count)
        self.feedback_width = int(feedback_width)
        self.embedding = nn.Embedding(self.key_count, self.feedback_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "key_count": self.key_count,
            "feedback_width": self.feedback_width,
        }

    def forward(self, key_index: torch.Tensor) -> torch.Tensor:
        if key_index.ndim != 1:
            raise ValueError("keypress input must have shape [batch]")
        if key_index.dtype != torch.long:
            raise ValueError("keypress input indices must be int64")
        if bool(torch.any(key_index < 0)) or bool(
            torch.any(key_index >= self.key_count)
        ):
            raise ValueError("keypress input index is outside the key count")
        return self.embedding(key_index)


class KeypressDecoder(nn.Module):
    """Decode an opaque intention into logits over external key indices."""

    schema = KEYPRESS_DECODER_SCHEMA

    def __init__(
        self,
        intention_width: int,
        key_count: int,
        *,
        hidden: int = 0,
    ) -> None:
        super().__init__()
        if min(intention_width, key_count) < 1 or hidden < 0:
            raise ValueError("keypress decoder dimensions are invalid")
        self.intention_width = int(intention_width)
        self.key_count = int(key_count)
        self.hidden = int(hidden)
        self.network = (
            nn.Sequential(
                nn.Linear(self.intention_width, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.key_count),
            )
            if hidden
            else nn.Linear(self.intention_width, self.key_count)
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "intention_width": self.intention_width,
            "key_count": self.key_count,
            "hidden": self.hidden,
            "output": "key-index-logits-v1",
        }

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if payload.ndim != 2 or payload.shape[1] < self.intention_width:
            raise ValueError("intention payload is too narrow for keypress decoder")
        logits = self.network(payload[:, : self.intention_width])
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("keypress decoder produced non-finite logits")
        return logits

    def decide(
        self,
        intention: IntentEvent | torch.Tensor,
        *,
        sample: bool = True,
    ) -> KeypressDecision:
        """Sample or greedily select a key while recording exact propensity."""

        return self.decide_from_logits(self(intention), sample=sample)

    def decide_from_logits(
        self,
        logits: torch.Tensor,
        *,
        sample: bool = True,
    ) -> KeypressDecision:
        """Select a key from logits already produced by the output bus."""

        if logits.ndim != 2 or logits.shape[1] != self.key_count:
            raise ValueError("keypress logits have the wrong shape")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("keypress logits must be finite")
        distribution = Categorical(logits=logits)
        key_index = distribution.sample() if sample else logits.argmax(dim=-1)
        propensity = distribution.probs.gather(1, key_index.unsqueeze(-1)).squeeze(-1)
        return KeypressDecision(key_index, logits, propensity).validate(
            key_count=self.key_count
        )
