"""A verifier for outcome-only redundant and contradictory event streams."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class RedundantComplementVerifier:
    """One high-bit stream plus three redundant low-bit streams.

    A corruption flips one low-bit stream without changing its confidence or
    adding a learner-visible label.  The target and corruption choice remain
    private; the learner receives only rendered streams and the scalar result
    of its opaque action.
    """

    action_count = 4
    raw_width = 4
    bit_count = 2
    stream_names = ("a", "b", "c", "d")

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        corruption_probability: float = 0.0,
        missing_probability: float = 0.0,
        source_flip_probabilities: tuple[float, float, float] | None = None,
        force_flip_mask: tuple[bool, bool, bool] | None = None,
        drop_all_low: bool = False,
        invert_all_low: bool = False,
        stream_order_shuffle: bool = False,
    ) -> None:
        values = {
            "corruption_probability": corruption_probability,
            "missing_probability": missing_probability,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if corruption_probability + missing_probability > 1.0:
            raise ValueError(
                "corruption_probability + missing_probability must be <= 1"
            )
        if source_flip_probabilities is not None and (
            len(source_flip_probabilities) != 3
            or any(not 0.0 <= value <= 1.0 for value in source_flip_probabilities)
        ):
            raise ValueError("source_flip_probabilities must contain three values in [0, 1]")
        if force_flip_mask is not None and len(force_flip_mask) != 3:
            raise ValueError("force_flip_mask must contain three boolean values")
        if drop_all_low and missing_probability:
            raise ValueError("drop_all_low cannot be combined with missing_probability")
        self.device = torch.device(device)
        self.corruption_probability = float(corruption_probability)
        self.missing_probability = float(missing_probability)
        self.source_flip_probabilities = source_flip_probabilities
        self.force_flip_mask = force_flip_mask
        self.drop_all_low = bool(drop_all_low)
        self.invert_all_low = bool(invert_all_low)
        self.stream_order_shuffle = bool(stream_order_shuffle)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._target: torch.Tensor | None = None

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        """Render a variable-cardinality event set without exposing labels."""
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
        streams: dict[str, torch.Tensor] = {
            "a": torch.cat(
                [
                    high_sign[:, None],
                    -high_sign[:, None],
                    torch.randn(
                        batch_size, 2, generator=self._generator, device=self.device
                    ),
                ],
                dim=1,
            )
        }

        draw = torch.rand((), generator=self._generator, device=self.device)
        corrupted = bool(draw < self.corruption_probability)
        missing = bool(
            draw >= self.corruption_probability
            and draw < self.corruption_probability + self.missing_probability
        )
        corrupted_index = (
            int(torch.randint(0, 3, (), generator=self._generator, device=self.device))
            if corrupted
            else -1
        )
        missing_index = (
            int(torch.randint(0, 3, (), generator=self._generator, device=self.device))
            if missing
            else -1
        )

        for index, name in enumerate(("b", "c", "d")):
            bit = low
            if self.source_flip_probabilities is not None:
                flip = bool(
                    torch.rand((), generator=self._generator, device=self.device)
                    < self.source_flip_probabilities[index]
                )
            else:
                flip = index == corrupted_index
            if self.force_flip_mask is not None and self.force_flip_mask[index]:
                flip = True
            if self.invert_all_low or flip:
                bit = 1 - bit
            sign = bit.to(torch.float32) * 2.0 - 1.0
            streams[name] = torch.cat(
                [
                    sign[:, None],
                    -sign[:, None],
                    torch.randn(
                        batch_size, 2, generator=self._generator, device=self.device
                    ),
                ],
                dim=1,
            )
            if self.drop_all_low or index == missing_index:
                streams.pop(name)

        if self.stream_order_shuffle:
            names = list(streams)
            order = torch.randperm(len(names), generator=self._generator).tolist()
            # Change only mapping insertion order; source names retain their
            # own payloads so this is a true permutation control.
            streams = {names[index]: streams[names[index]] for index in order}
        return streams

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Return only the deterministic scalar verifier outcome."""
        if self._target is None:
            raise RuntimeError("reset must be called before step")
        if actions.ndim != 1 or actions.shape != self._target.shape:
            raise ValueError("actions must have shape [batch]")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actions must be integer protocol outputs")
        return (actions.to(self.device) == self._target).to(torch.float32)
