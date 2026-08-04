"""A sequential verifier for adaptive contradiction resolution.

The verifier renders two opaque candidate events whose payloads always
disagree.  One source is privately designated reliable for a short block of
steps, then the designation may reverse.  The learner must use only its prior
opaque action and scalar verifier outcome to update its source belief.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch


@dataclass(frozen=True)
class ConflictSequence:
    """Rendered streams and private targets for one batched trajectory."""

    streams: tuple[Mapping[str, torch.Tensor], ...]
    targets: torch.Tensor
    roles: torch.Tensor


class SequentialConflictVerifier:
    """Two contradictory sources with a hidden, reversible trust relation."""

    action_count = 2
    raw_width = 4
    bit_count = 1
    stream_names = ("b", "c")

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        sequence_length: int = 32,
        block_length: int = 8,
        stream_order_shuffle: bool = False,
        candidate_swap: bool = False,
    ) -> None:
        if sequence_length < 2 or block_length < 1:
            raise ValueError("sequence_length must be >= 2 and block_length positive")
        if sequence_length % block_length:
            raise ValueError("sequence_length must be divisible by block_length")
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.block_length = block_length
        self.block_count = sequence_length // block_length
        self.stream_order_shuffle = bool(stream_order_shuffle)
        self.candidate_swap = bool(candidate_swap)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._role_generator = torch.Generator(device=self.device)
        self._role_generator.manual_seed(seed + 2_000_003)
        self._order_generator = torch.Generator(device=self.device)
        self._order_generator.manual_seed(seed + 1_000_003)

    def sample(
        self,
        batch_size: int,
        *,
        roles: torch.Tensor | None = None,
        alternating_roles: bool = False,
        random_start: bool = False,
    ) -> ConflictSequence:
        """Render one sequence while keeping targets/roles private to training."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if roles is None:
            if alternating_roles:
                base_roles = torch.arange(
                    self.block_count, device=self.device, dtype=torch.long
                ).remainder(2)
                if random_start:
                    starts = torch.randint(
                        0,
                        2,
                        (batch_size, 1),
                        generator=self._generator,
                        device=self.device,
                    )
                else:
                    starts = torch.zeros(
                        batch_size, 1, dtype=torch.long, device=self.device
                    )
                block_roles = (base_roles.unsqueeze(0) + starts).remainder(2)
            else:
                block_roles = torch.randint(
                    0,
                    2,
                    (batch_size, self.block_count),
                    generator=self._generator,
                    device=self.device,
                )
        else:
            block_roles = roles.to(device=self.device, dtype=torch.long)
            if block_roles.shape != (batch_size, self.block_count):
                raise ValueError("roles must have shape [batch, block_count]")
            if torch.any((block_roles < 0) | (block_roles > 1)):
                raise ValueError("roles must contain only 0 or 1")

        targets = torch.randint(
            0,
            self.action_count,
            (self.sequence_length, batch_size),
            generator=self._generator,
            device=self.device,
        )
        streams: list[Mapping[str, torch.Tensor]] = []
        for step in range(self.sequence_length):
            target = targets[step]
            reliable_source = block_roles[:, step // self.block_length]
            b_bit = torch.where(reliable_source == 0, target, 1 - target)
            c_bit = 1 - b_bit
            rendered = {
                "b": self._render(b_bit),
                "c": self._render(c_bit),
            }
            if self.candidate_swap:
                rendered = {"b": rendered["c"], "c": rendered["b"]}
            if self.stream_order_shuffle:
                if bool(
                    torch.randint(
                        0, 2, (), generator=self._order_generator, device=self.device
                    )
                ):
                    rendered = {"c": rendered["c"], "b": rendered["b"]}
            streams.append(rendered)
        return ConflictSequence(streams=tuple(streams), targets=targets, roles=block_roles)

    def sample_random_reversal(
        self, batch_size: int, *, minimum_prefix: int = 1
    ) -> ConflictSequence:
        """Sample one hidden role reversal with a bounded block prefix/suffix."""
        if self.block_count < minimum_prefix + 2:
            raise ValueError("sequence is too short for a bounded random reversal")
        switches = torch.randint(
            minimum_prefix,
            self.block_count - minimum_prefix + 1,
            (batch_size, 1),
            generator=self._role_generator,
            device=self.device,
        )
        positions = torch.arange(
            self.block_count, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        roles = (positions >= switches).to(torch.long)
        return self.sample(batch_size, roles=roles)

    def sample_markov_roles(
        self, batch_size: int, *, switch_probability: float = 0.2
    ) -> ConflictSequence:
        """Sample a persistent but unpredictable hidden source schedule.

        The initial source is balanced and each subsequent block independently
        flips it with ``switch_probability``. This preserves useful temporal
        continuity while preventing a fixed clock position from revealing the
        current reliable source.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= switch_probability <= 1.0:
            raise ValueError("switch_probability must be in [0, 1]")
        roles = torch.empty(
            batch_size,
            self.block_count,
            dtype=torch.long,
            device=self.device,
        )
        roles[:, 0] = torch.randint(
            0,
            2,
            (batch_size,),
            generator=self._role_generator,
            device=self.device,
        )
        if self.block_count > 1:
            flips = torch.rand(
                batch_size,
                self.block_count - 1,
                generator=self._role_generator,
                device=self.device,
            ) < switch_probability
            for block in range(1, self.block_count):
                roles[:, block] = roles[:, block - 1] ^ flips[:, block - 1].to(
                    torch.long
                )
        return self.sample(batch_size, roles=roles)

    def _render(self, bit: torch.Tensor) -> torch.Tensor:
        sign = bit.to(torch.float32) * 2.0 - 1.0
        noise = 0.05 * torch.randn(
            bit.shape[0], 2, generator=self._generator, device=self.device
        )
        return torch.cat([sign[:, None], -sign[:, None], noise], dim=1)

    def outcome(self, actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the only verifier signal exposed to the learner."""
        if actions.shape != targets.shape:
            raise ValueError("actions and targets must have the same shape")
        return (actions.to(self.device) == targets).to(torch.float32)
