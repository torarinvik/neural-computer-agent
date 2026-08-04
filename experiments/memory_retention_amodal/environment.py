"""Verifier for cue-guided retention under bounded memory capacity."""

from __future__ import annotations

import torch


class OutcomeOnlyRetentionVerifier:
    """Hide binary outcomes and score recall of the cued one.

    The driver renders the opaque target cue and slot events. The learner only
    receives opaque actions and scalar verifier outcomes; the target bits,
    selected query slot, and presentation order remain verifier state.
    """

    action_count = 2

    def __init__(
        self,
        *,
        batch_size: int,
        seed: int,
        slot_count: int = 2,
        recall_probe_outcome: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least two")
        if slot_count < 2:
            raise ValueError("slot_count must be at least two")
        self.batch_size = batch_size
        self.slot_count = slot_count
        self.recall_probe_outcome = recall_probe_outcome
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._bits = torch.zeros(batch_size, slot_count, device=self.device)
        self._probe_rewards = torch.zeros(batch_size, slot_count, device=self.device)
        self._query_slot = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._balanced_position = (
            torch.arange(batch_size, device=self.device, dtype=torch.long) % slot_count
        )
        self._order = torch.zeros(
            batch_size, slot_count, dtype=torch.long, device=self.device
        )

    @property
    def query_slot(self) -> torch.Tensor:
        return self._query_slot

    @property
    def order(self) -> torch.Tensor:
        return self._order

    @property
    def balanced_position(self) -> torch.Tensor:
        """Trainer-only target position that survives counterfactual duplication."""
        return self._balanced_position

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
        self._order = torch.argsort(
            torch.rand(
                self.batch_size,
                self.slot_count,
                generator=self._generator,
                device=self.device,
            ),
            dim=1,
        )
        self._probe_rewards.zero_()

    def duplicate_rows(self, repeats: int = 2) -> OutcomeOnlyRetentionVerifier:
        """Duplicate one hidden verifier world for coupled training arms.

        This is a trainer-only common-random-number utility. The duplicated
        verifier state is never sent to the controller; each arm still sees
        only its own opaque actions and scalar outcomes.
        """
        if repeats < 2:
            raise ValueError("repeats must be at least two")
        duplicate = OutcomeOnlyRetentionVerifier(
            batch_size=self.batch_size * repeats,
            seed=0,
            slot_count=self.slot_count,
            recall_probe_outcome=self.recall_probe_outcome,
            device=self.device,
        )
        duplicate._bits = self._bits.repeat_interleave(repeats, dim=0)
        duplicate._probe_rewards = self._probe_rewards.repeat_interleave(
            repeats, dim=0
        )
        duplicate._query_slot = self._query_slot.repeat_interleave(repeats, dim=0)
        duplicate._balanced_position = self._balanced_position.repeat_interleave(
            repeats, dim=0
        )
        duplicate._order = self._order.repeat_interleave(repeats, dim=0)
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
        if self.recall_probe_outcome:
            expected = self._probe_rewards.gather(
                1, self._query_slot[:, None]
            ).squeeze(1).long()
        else:
            expected = self._bits.gather(
                1, self._query_slot[:, None]
            ).squeeze(1).long()
        return (action.to(torch.long) == expected).to(torch.float32)

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
