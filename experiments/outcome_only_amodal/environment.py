"""A small verifier whose learner-visible feedback is reward only.

The verifier deliberately keeps the target and correct action private.  Each
raw stream contains one bit of a hidden four-way decision plus independent
distractors.  The trainer can therefore learn the complement operation only
through two partial rendered streams and scalar verifier outcomes.
"""

from __future__ import annotations

from typing import Mapping

import torch


class OutcomeOnlyComplementVerifier:
    """Two-stream, four-action bandit with hidden two-bit targets."""

    action_count = 4
    raw_width = 4
    bit_count = 2

    def __init__(
        self,
        *,
        seed: int = 0,
        reverse: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.reverse = bool(reverse)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._target: torch.Tensor | None = None

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        """Render two partial streams without exposing the hidden target."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        target = torch.randint(
            0,
            self.action_count,
            (batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._target = target
        high = target // 2
        low = target % 2
        high_sign = high.to(torch.float32) * 2.0 - 1.0
        low_sign = low.to(torch.float32) * 2.0 - 1.0
        distractor_a = torch.randn(
            batch_size, 2, generator=self._generator, device=self.device
        )
        distractor_b = torch.randn(
            batch_size, 2, generator=self._generator, device=self.device
        )
        return {
            "a": torch.cat([high_sign[:, None], -high_sign[:, None], distractor_a], dim=1),
            "b": torch.cat([low_sign[:, None], -low_sign[:, None], distractor_b], dim=1),
        }

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Return only deterministic scalar verifier outcomes."""
        if self._target is None:
            raise RuntimeError("reset must be called before step")
        if actions.ndim != 1 or actions.shape != self._target.shape:
            raise ValueError("actions must have shape [batch]")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actions must be integer protocol outputs")
        expected = 3 - self._target if self.reverse else self._target
        return (actions.to(self.device) == expected).to(torch.float32)

