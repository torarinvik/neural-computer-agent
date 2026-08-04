"""Verifier for reusing source trust learned from one scalar outcome."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationSequence:
    streams: tuple[Mapping[str, torch.Tensor], ...]
    targets: torch.Tensor
    roles: torch.Tensor


class CalibrationConflictVerifier:
    """Two contradictory streams with one hidden stable reliable source."""

    action_count = 2
    raw_width = 4
    bit_count = 1
    stream_names = ("b", "c")

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        sequence_length: int = 4,
        stream_order_shuffle: bool = False,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.stream_order_shuffle = bool(stream_order_shuffle)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._order_generator = torch.Generator(device=self.device)
        self._order_generator.manual_seed(seed + 1_000_003)

    def sample(
        self, batch_size: int, *, force_role: int | None = None
    ) -> CalibrationSequence:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if force_role is None:
            roles = torch.randint(
                0,
                2,
                (batch_size,),
                generator=self._generator,
                device=self.device,
            )
        else:
            if force_role not in (0, 1):
                raise ValueError("force_role must be 0 or 1")
            roles = torch.full(
                (batch_size,), force_role, dtype=torch.long, device=self.device
            )
        targets = torch.randint(
            0,
            self.action_count,
            (self.sequence_length, batch_size),
            generator=self._generator,
            device=self.device,
        )
        streams: list[Mapping[str, torch.Tensor]] = []
        for tick in range(self.sequence_length):
            target = targets[tick]
            b_bit = torch.where(roles == 0, target, 1 - target)
            c_bit = 1 - b_bit
            rendered = {"b": self._render(b_bit), "c": self._render(c_bit)}
            if self.stream_order_shuffle:
                if bool(
                    torch.randint(
                        0, 2, (), generator=self._order_generator, device=self.device
                    )
                ):
                    rendered = {"c": rendered["c"], "b": rendered["b"]}
            streams.append(rendered)
        return CalibrationSequence(tuple(streams), targets, roles)

    def _render(self, bit: torch.Tensor) -> torch.Tensor:
        sign = bit.to(torch.float32) * 2.0 - 1.0
        noise = 0.05 * torch.randn(
            bit.shape[0], 2, generator=self._generator, device=self.device
        )
        return torch.cat([sign[:, None], -sign[:, None], noise], dim=1)

    def outcome(self, actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if actions.shape != targets.shape:
            raise ValueError("actions and targets must have the same shape")
        return (actions.to(self.device) == targets).to(torch.float32)
