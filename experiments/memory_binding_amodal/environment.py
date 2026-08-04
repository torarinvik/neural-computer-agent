"""A verifier for binding two opaque memory keys to separate outcomes."""

from __future__ import annotations

import torch


class TwoSlotBindingVerifier:
    """Hide two binary probe outcomes and score one later recall.

    The target bits and selected query slot are private verifier state. The
    learner sees only opaque actions and scalar probe/recall outcomes.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least two for scope audits")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self._targets = torch.zeros(batch_size, 2, device=self.device)
        self._query_slot = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._probe_rewards = torch.zeros(batch_size, 2, device=self.device)

    @property
    def query_slot(self) -> torch.Tensor:
        return self._query_slot

    def reset(self) -> None:
        self._targets = torch.randint(
            0,
            2,
            (self.batch_size, 2),
            generator=self.generator,
            device=self.device,
        ).to(torch.float32)
        self._query_slot = torch.randint(
            0,
            2,
            (self.batch_size,),
            generator=self.generator,
            device=self.device,
        )
        self._probe_rewards.zero_()

    def score_probe(self, slot: int, action: torch.Tensor) -> torch.Tensor:
        if slot not in (0, 1):
            raise ValueError("slot must be zero or one")
        if action.shape != (self.batch_size,):
            raise ValueError("probe action must have one value per batch row")
        reward = (action.to(torch.long) == self._targets[:, slot].to(torch.long)).to(
            torch.float32
        )
        self._probe_rewards[:, slot] = reward
        return reward

    def score_recall(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (self.batch_size,):
            raise ValueError("recall action must have one value per batch row")
        expected = self._probe_rewards.gather(1, self._query_slot[:, None]).squeeze(1)
        return (action.to(torch.long) == expected.to(torch.long)).to(torch.float32)
