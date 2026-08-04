"""A verifier whose relevant candidate changes on every episode."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class ContextRelevanceVerifier:
    """Context stream plus two candidates with randomized relevance.

    The candidate whose tag agrees with the context is relevant.  The relevant
    candidate is randomized independently on every episode, so source identity
    and historical source reliability cannot solve the task.  The target and
    relevant index are private; the learner receives event streams and scalar
    verifier outcomes only.
    """

    action_count = 4
    raw_width = 6
    bit_count = 2
    stream_names = ("a", "b", "c")

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        force_relevant_index: int | None = None,
        stream_order_shuffle: bool = False,
        cross_episode_shuffle: bool = False,
        swap_candidates: bool = False,
    ) -> None:
        if force_relevant_index not in (None, 0, 1):
            raise ValueError("force_relevant_index must be None, 0, or 1")
        self.device = torch.device(device)
        self.force_relevant_index = force_relevant_index
        self.stream_order_shuffle = bool(stream_order_shuffle)
        self.cross_episode_shuffle = bool(cross_episode_shuffle)
        self.swap_candidates = bool(swap_candidates)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._target: torch.Tensor | None = None
        self._forced_relevant_pattern: torch.Tensor | None = None

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        high = torch.randint(
            0, 2, (batch_size,), generator=self._generator, device=self.device
        )
        context = torch.randint(
            0, 2, (batch_size,), generator=self._generator, device=self.device
        )
        candidate_bits = torch.randint(
            0, 2, (batch_size, 2), generator=self._generator, device=self.device
        )
        relevant = (
            self._forced_relevant_pattern.to(self.device).clone()
            if self._forced_relevant_pattern is not None
            else
            torch.full(
                (batch_size,),
                self.force_relevant_index,
                dtype=torch.long,
                device=self.device,
            )
            if self.force_relevant_index is not None
            else torch.randint(
                0, 2, (batch_size,), generator=self._generator, device=self.device
            )
        )
        self._target = high * 2 + candidate_bits[
            torch.arange(batch_size, device=self.device), relevant
        ]

        high_sign = high.to(torch.float32) * 2.0 - 1.0
        context_sign = context.to(torch.float32) * 2.0 - 1.0
        streams: dict[str, torch.Tensor] = {
            "a": torch.cat(
                [
                    high_sign[:, None],
                    -high_sign[:, None],
                    context_sign[:, None],
                    -context_sign[:, None],
                    torch.randn(
                        batch_size, 2, generator=self._generator, device=self.device
                    ),
                ],
                dim=1,
            )
        }
        for index, name in enumerate(("b", "c")):
            tag = torch.where(relevant == index, context, 1 - context)
            tag_sign = tag.to(torch.float32) * 2.0 - 1.0
            bit_sign = candidate_bits[:, index].to(torch.float32) * 2.0 - 1.0
            streams[name] = torch.cat(
                [
                    bit_sign[:, None],
                    -bit_sign[:, None],
                    tag_sign[:, None],
                    -tag_sign[:, None],
                    torch.randn(
                        batch_size, 2, generator=self._generator, device=self.device
                    ),
                ],
                dim=1,
            )

        if self.swap_candidates:
            streams = {"a": streams["a"], "b": streams["c"], "c": streams["b"]}
        if self.cross_episode_shuffle:
            streams["b"] = streams["b"][
                torch.randperm(batch_size, generator=self._generator, device=self.device)
            ]
            streams["c"] = streams["c"][
                torch.randperm(batch_size, generator=self._generator, device=self.device)
            ]
        if self.stream_order_shuffle:
            names = list(streams)
            order = torch.randperm(len(names), generator=self._generator).tolist()
            streams = {names[index]: streams[names[index]] for index in order}
        return streams

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        if self._target is None:
            raise RuntimeError("reset must be called before step")
        if actions.ndim != 1 or actions.shape != self._target.shape:
            raise ValueError("actions must have shape [batch]")
        if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actions must be integer protocol outputs")
        return (actions.to(self.device) == self._target).to(torch.float32)


class BalancedContextRelevanceCurriculum:
    """Expose both hidden relevance assignments equally during training."""

    action_count = ContextRelevanceVerifier.action_count
    raw_width = ContextRelevanceVerifier.raw_width
    bit_count = ContextRelevanceVerifier.bit_count

    def __init__(
        self,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self._verifier = ContextRelevanceVerifier(seed=seed, device=device)
        self._episode = 0

    @property
    def device(self) -> torch.device:
        return self._verifier.device

    def reset(self, batch_size: int) -> Mapping[str, torch.Tensor]:
        pattern = torch.arange(batch_size, device=self.device, dtype=torch.long) % 2
        order = torch.randperm(
            batch_size, generator=self._verifier._generator, device=self.device
        )
        self._verifier._forced_relevant_pattern = pattern[order]
        self._verifier.force_relevant_index = None
        self._episode += 1
        return self._verifier.reset(batch_size)

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        return self._verifier.step(actions)
