"""A verifier whose second decision requires remembering a scalar outcome."""

from __future__ import annotations

import torch


class OutcomeRecallVerifier:
    """Hide a probe target, then ask the agent to reproduce its scalar reward."""

    action_count = 2

    def __init__(
        self, *, seed: int = 0, device: torch.device | str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._target: torch.Tensor | None = None
        self._probe_reward: torch.Tensor | None = None

    def reset(self) -> None:
        self._target = torch.randint(
            0, self.action_count, (1,), generator=self._generator, device=self.device
        )
        self._probe_reward = None

    def score_probe(self, actions: torch.Tensor) -> torch.Tensor:
        if self._target is None:
            raise RuntimeError("reset must be called before score_probe")
        self._validate_action(actions)
        self._probe_reward = (actions.to(self.device) == self._target).to(torch.float32)
        return self._probe_reward

    def score_recall(self, actions: torch.Tensor) -> torch.Tensor:
        if self._probe_reward is None:
            raise RuntimeError("score_probe must be called before score_recall")
        self._validate_action(actions)
        return (actions.to(self.device) == self._probe_reward.long()).to(torch.float32)

    def _validate_action(self, actions: torch.Tensor) -> None:
        if actions.shape != (1,) or actions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("actions must be integer tensors with shape [1]")
