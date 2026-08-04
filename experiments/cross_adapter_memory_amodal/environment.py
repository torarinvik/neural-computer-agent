"""Verifier for retrieving opaque outcomes through a second adapter."""

from __future__ import annotations

import torch


class CrossAdapterRecallVerifier:
    """Hide bounded binary outcomes and score a later slot-specific recall.

    The writer and reader adapters receive ordinary opaque payloads. The
    target bits and selected query slot remain verifier state and are never
    passed to the controller or reader adapter.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seed: int,
        slot_count: int = 2,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if slot_count < 2:
            raise ValueError("slot count must be at least two")
        self.batch_size = batch_size
        self.slot_count = slot_count
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._bits = torch.zeros(batch_size, slot_count, device=self.device)
        self._query_slot = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._probe_rewards = torch.zeros(batch_size, slot_count, device=self.device)

    @property
    def query_slot(self) -> torch.Tensor:
        return self._query_slot

    def reset(self) -> None:
        self._bits = torch.randint(
            0,
            2,
            (self.batch_size, self.slot_count),
            generator=self._generator,
            device=self.device,
        ).to(torch.float32)
        self._query_slot = torch.randint(
            0,
            self.slot_count,
            (self.batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._probe_rewards.zero_()

    def duplicate_rows(self, repeats: int = 2) -> CrossAdapterRecallVerifier:
        """Duplicate one hidden verifier world for paired trainer arms."""
        if repeats < 2:
            raise ValueError("repeats must be at least two")
        duplicate = CrossAdapterRecallVerifier(
            batch_size=self.batch_size * repeats,
            seed=0,
            slot_count=self.slot_count,
            device=self.device,
        )
        duplicate._bits = self._bits.repeat_interleave(repeats, dim=0)
        duplicate._query_slot = self._query_slot.repeat_interleave(repeats, dim=0)
        duplicate._probe_rewards = self._probe_rewards.repeat_interleave(
            repeats, dim=0
        )
        return duplicate

    def score_probe(self, slot: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self._validate_slot(slot)
        self._validate_action(action)
        expected = self._bits.gather(1, slot[:, None]).squeeze(1).long()
        reward = (action.to(torch.long) == expected).to(torch.float32)
        self._probe_rewards.scatter_(1, slot[:, None], reward[:, None])
        return reward

    def score_recall(self, action: torch.Tensor) -> torch.Tensor:
        self._validate_action(action)
        expected = self._probe_rewards.gather(
            1, self._query_slot[:, None]
        ).squeeze(1)
        return (action.to(torch.long) == expected.to(torch.long)).to(torch.float32)

    def _validate_slot(self, slot: torch.Tensor) -> None:
        if slot.shape != (self.batch_size,) or slot.dtype != torch.long:
            raise ValueError("slot must be int64 with shape [batch]")
        if bool(torch.any((slot < 0) | (slot >= self.slot_count))):
            raise ValueError("slot is outside the verifier slot vocabulary")

    def _validate_action(self, action: torch.Tensor) -> None:
        if action.shape != (self.batch_size,) or action.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("action must be an integer tensor with shape [batch]")
