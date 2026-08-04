"""A verifier for context-conditioned contradiction arbitration."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class ContextConflictVerifier:
    """Context stream selects one of two contradictory candidate streams."""

    action_count = 2
    raw_width = 4
    bit_count = 1
    stream_names = ("a", "b", "c")

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        force_context: int | None = None,
        shuffle_assignment: bool = False,
        invert_context: bool = False,
        stream_order_shuffle: bool = False,
    ) -> None:
        if force_context is not None and force_context not in (0, 1):
            raise ValueError("force_context must be 0 or 1")
        self.device = torch.device(device)
        self.force_context = force_context
        self.shuffle_assignment = bool(shuffle_assignment)
        self.invert_context = bool(invert_context)
        self.stream_order_shuffle = bool(stream_order_shuffle)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._order_generator = torch.Generator(device=self.device)
        self._order_generator.manual_seed(seed + 1_000_003)
        self._target: torch.Tensor | None = None

    def reset(
        self, batch_size: int, *, context_values: torch.Tensor | None = None
    ) -> Mapping[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if context_values is None:
            if self.force_context is None:
                context = torch.randint(
                    0,
                    2,
                    (batch_size,),
                    generator=self._generator,
                    device=self.device,
                )
            else:
                context = torch.full(
                    (batch_size,), self.force_context, dtype=torch.long, device=self.device
                )
        else:
            context = context_values.to(device=self.device, dtype=torch.long)
            if context.shape != (batch_size,) or torch.any((context < 0) | (context > 1)):
                raise ValueError("context_values must contain batch-sized binary values")
        target = torch.randint(
            0,
            self.action_count,
            (batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._target = target
        relevant_index = context.clone()
        if self.shuffle_assignment:
            relevant_index = torch.randint(
                0,
                2,
                (batch_size,),
                generator=self._generator,
                device=self.device,
            )
        b_bit = torch.where(relevant_index == 0, target, 1 - target)
        c_bit = 1 - b_bit
        streams: dict[str, torch.Tensor] = {
            "a": self._render(1 - context if self.invert_context else context),
            "b": self._render(b_bit),
            "c": self._render(c_bit),
        }
        if self.stream_order_shuffle:
            order = bool(
                torch.randint(
                    0, 2, (), generator=self._order_generator, device=self.device
                )
            )
            if order:
                streams = {"c": streams["c"], "a": streams["a"], "b": streams["b"]}
        return streams

    def _render(self, bit: torch.Tensor) -> torch.Tensor:
        sign = bit.to(torch.float32) * 2.0 - 1.0
        noise = 0.05 * torch.randn(
            bit.shape[0], 2, generator=self._generator, device=self.device
        )
        return torch.cat([sign[:, None], -sign[:, None], noise], dim=1)

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        if self._target is None:
            raise RuntimeError("reset must be called before step")
        if actions.shape != self._target.shape:
            raise ValueError("actions must have shape [batch]")
        return (actions.to(self.device) == self._target).to(torch.float32)


class BalancedContextConflictCurriculum(ContextConflictVerifier):
    """Keep both hidden context classes in every policy-gradient update."""

    def reset(
        self, batch_size: int, *, context_values: torch.Tensor | None = None
    ) -> Mapping[str, torch.Tensor]:
        if context_values is None:
            context_values = torch.arange(batch_size, device=self.device).remainder(2)
            context_values = context_values[
                torch.randperm(batch_size, generator=self._generator, device=self.device)
            ]
        return super().reset(batch_size, context_values=context_values)
