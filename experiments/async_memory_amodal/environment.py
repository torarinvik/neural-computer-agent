"""Outcome-only asynchronous verifier.

The verifier reveals one bit on the first arrival and the complementary bit on
the second arrival. The target stays private; callers receive raw streams and
scalar rewards only. Missing and contradictory conditions are implemented as
verifier controls, never exposed as labels to the learner.
"""

from __future__ import annotations

from typing import Mapping

import torch


class DelayedComplementVerifier:
    """Two asynchronous partial streams followed by a delayed action reward."""

    action_count = 4
    raw_width = 4

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        reverse: bool = False,
        contradictory: bool = False,
        missing_second: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.reverse = bool(reverse)
        self.contradictory = bool(contradictory)
        self.missing_second = bool(missing_second)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._target: torch.Tensor | None = None
        self._first: torch.Tensor | None = None
        self._second: torch.Tensor | None = None

    def start(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        """Return the first partial stream without revealing the target."""
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
        high_sign = high.to(torch.float32) * 2.0 - 1.0
        distractors = torch.randn(
            batch_size, 2, generator=self._generator, device=self.device
        )
        self._first = torch.cat([high_sign[:, None], -high_sign[:, None], distractors], dim=1)
        return {"a": self._first}

    def next_arrival(self) -> Mapping[str, torch.Tensor]:
        """Return the delayed complementary stream, or an explicit omission."""
        if self._target is None:
            raise RuntimeError("start must be called before next_arrival")
        low = self._target % 2
        if self.contradictory:
            low = 1 - low
        low_sign = low.to(torch.float32) * 2.0 - 1.0
        distractors = torch.randn(
            self._target.shape[0], 2, generator=self._generator, device=self.device
        )
        self._second = torch.cat([low_sign[:, None], -low_sign[:, None], distractors], dim=1)
        if self.missing_second:
            return {}
        return {"b": self._second}

    def score(self, actions: torch.Tensor) -> torch.Tensor:
        """Return only scalar verifier outcomes for the delayed action."""
        if self._target is None:
            raise RuntimeError("start must be called before score")
        if actions.ndim != 1 or actions.shape != self._target.shape:
            raise ValueError("actions must have shape [batch]")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actions must be integer protocol outputs")
        expected = 3 - self._target if self.reverse else self._target
        return (actions.to(self.device) == expected).to(torch.float32)

