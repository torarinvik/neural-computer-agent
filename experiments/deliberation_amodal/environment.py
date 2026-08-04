"""A tiny verifier for learning when to wait, think, or commit.

The learner sees only rendered event streams and the scalar result of its
opaque action. Complete, delayed, and think-required episodes vary whether the
partner is available immediately or released after a bounded internal tick.
The hidden target and the correct action never enter the learner-visible
interface.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


class VariableDeliberationVerifier:
    """Two-stream, four-action verifier with a delayed partner stream."""

    action_count = 4
    raw_width = 4
    bit_count = 2

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        easy_probability: float = 0.5,
        think_probability: float = 0.0,
        missing_probability: float = 0.0,
    ) -> None:
        if not 0.0 <= easy_probability <= 1.0:
            raise ValueError("easy_probability must be in [0, 1]")
        if not 0.0 <= think_probability <= 1.0:
            raise ValueError("think_probability must be in [0, 1]")
        if not 0.0 <= missing_probability <= 1.0:
            raise ValueError("missing_probability must be in [0, 1]")
        if easy_probability + think_probability + missing_probability > 1.0:
            raise ValueError(
                "easy_probability + think_probability + missing_probability "
                "must be <= 1"
            )
        self.device = torch.device(device)
        self.easy_probability = float(easy_probability)
        self.think_probability = float(think_probability)
        self.missing_probability = float(missing_probability)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._target: torch.Tensor | None = None
        self._delayed: dict[str, torch.Tensor] = {}
        self._think_required = False
        self._missing = False

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        """Render the currently available streams without exposing the target."""
        if batch_size != 1:
            raise ValueError("the first deliberation benchmark uses one trajectory per rollout")
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
            batch_size, 1, generator=self._generator, device=self.device
        )
        distractor_b = torch.randn(
            batch_size, 1, generator=self._generator, device=self.device
        )
        draw = torch.rand((), generator=self._generator, device=self.device)
        think_required = bool(draw < self.think_probability)
        complete = bool(
            draw < self.think_probability + self.easy_probability
        ) and not think_required
        missing = bool(
            draw
            >= self.think_probability + self.easy_probability
        ) and bool(
            draw
            < self.think_probability
            + self.easy_probability
            + self.missing_probability
        )
        quality = torch.full(
            (batch_size, 1),
            0.25 if think_required else 1.0,
            device=self.device,
        )
        partner_quality = torch.ones_like(quality)
        streams = {
            "a": torch.cat(
                [high_sign[:, None], -high_sign[:, None], distractor_a, quality], dim=1
            )
        }
        b = torch.cat(
            [low_sign[:, None], -low_sign[:, None], distractor_b, partner_quality], dim=1
        )
        self._think_required = think_required
        self._missing = missing
        if complete:
            streams["b"] = b
            self._delayed = {}
        else:
            self._delayed = {"b": b}
        return streams

    def release_delayed(self, *, after_think: bool = False) -> Mapping[str, torch.Tensor]:
        """Release the next scheduled event, if one exists."""
        if self._missing or (self._think_required and not after_think):
            return {}
        released = self._delayed
        self._delayed = {}
        return released

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Return only the deterministic scalar verifier outcome."""
        if self._target is None:
            raise RuntimeError("reset must be called before step")
        if actions.shape != self._target.shape or actions.ndim != 1:
            raise ValueError("actions must have shape [1]")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actions must be integer protocol outputs")
        return (actions.to(self.device) == self._target).to(torch.float32)


class BalancedDeliberationCurriculum:
    """Rotate complete, delayed, and think-required episodes evenly.

    This is a trainer-side exposure schedule. The controller receives only the
    rendered streams and verifier outcomes; no curriculum index is emitted.
    """

    action_count = VariableDeliberationVerifier.action_count
    raw_width = VariableDeliberationVerifier.raw_width
    bit_count = VariableDeliberationVerifier.bit_count

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self._verifier = VariableDeliberationVerifier(seed=seed, device=device)
        self._episode = 0

    @property
    def device(self) -> torch.device:
        return self._verifier.device

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        mode = self._episode % 3
        self._episode += 1
        if mode == 0:
            self._verifier.easy_probability = 1.0
            self._verifier.think_probability = 0.0
        elif mode == 1:
            self._verifier.easy_probability = 0.0
            self._verifier.think_probability = 0.0
        else:
            self._verifier.easy_probability = 0.0
            self._verifier.think_probability = 1.0
        return self._verifier.reset(batch_size)

    def release_delayed(self, *, after_think: bool = False) -> Mapping[str, torch.Tensor]:
        return self._verifier.release_delayed(after_think=after_think)

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        return self._verifier.step(actions)


class BalancedAsyncDeliberationCurriculum:
    """Rotate complete, delayed, missing, and think-required episodes evenly."""

    action_count = VariableDeliberationVerifier.action_count
    raw_width = VariableDeliberationVerifier.raw_width
    bit_count = VariableDeliberationVerifier.bit_count

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self._verifier = VariableDeliberationVerifier(seed=seed, device=device)
        self._episode = 0

    @property
    def device(self) -> torch.device:
        return self._verifier.device

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        mode = self._episode % 4
        self._episode += 1
        self._verifier.easy_probability = float(mode == 0)
        self._verifier.think_probability = float(mode == 3)
        self._verifier.missing_probability = float(mode == 2)
        return self._verifier.reset(batch_size)

    def release_delayed(self, *, after_think: bool = False) -> Mapping[str, torch.Tensor]:
        return self._verifier.release_delayed(after_think=after_think)

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        return self._verifier.step(actions)
