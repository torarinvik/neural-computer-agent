"""A small rendered-event Brain Workshop verifier for the canonical runtime.

The verifier owns the hidden n-back target and exposes only a raw symbol
observation followed by a deterministic scalar outcome.  The event encoder is
the caller-owned frontend: the amodal controller receives only its learned
event tensor, never the symbol index or the target relation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NBACK_ACTION_COUNT = 2
EVENT_ENCODER_SCHEMA = "neural-computer.brainworkshop-event-encoder.v1"


@dataclass(frozen=True)
class NBackVerifierStep:
    """Verifier outcome for one attempted response."""

    reward: torch.Tensor
    eligible: torch.Tensor


class NBackVerifier:
    """Generate symbol streams and score binary n-back responses.

    ``symbols`` and the target comparison remain verifier-private.  A caller
    can request the current observation and submit an integer action, but it
    cannot obtain the correct action through this interface.
    """

    action_count = NBACK_ACTION_COUNT

    def __init__(
        self,
        *,
        batch_size: int,
        n_back: int,
        steps: int | None = None,
        symbol_count: int = 4,
        seed: int = 0,
        time_shuffle: bool = False,
        cue_symbol: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if min(batch_size, n_back, symbol_count) < 1:
            raise ValueError("batch size, n-back, and symbol count must be positive")
        resolved_steps = n_back + 4 if steps is None else int(steps)
        if resolved_steps <= n_back:
            raise ValueError("steps must include at least one target-bearing trial")
        if cue_symbol is not None and cue_symbol < symbol_count:
            raise ValueError("cue symbol must be outside the symbol vocabulary")
        self.batch_size = int(batch_size)
        self.n_back = int(n_back)
        self.symbol_steps = resolved_steps
        self.steps = resolved_steps + (cue_symbol is not None)
        self.symbol_count = int(symbol_count)
        self.cue_symbol = None if cue_symbol is None else int(cue_symbol)
        self.time_shuffle = bool(time_shuffle)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._symbols = torch.empty(0, dtype=torch.long, device=self.device)
        self._target_bits = torch.empty(0, dtype=torch.bool, device=self.device)
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    @property
    def eligible_trials(self) -> int:
        return max(0, self.symbol_steps - self.n_back)

    @property
    def observation_symbol_count(self) -> int:
        """Return the frontend vocabulary needed for rendered observations."""

        if self.cue_symbol is None:
            return self.symbol_count
        return max(self.symbol_count, self.cue_symbol + 1)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        self._symbols = torch.empty(
            self.batch_size,
            self.symbol_steps,
            dtype=torch.long,
            device=self.device,
        )
        self._symbols[:, : self.n_back] = torch.randint(
            0,
            self.symbol_count,
            (self.batch_size, self.n_back),
            generator=self._generator,
            device=self.device,
        )
        eligible = self.eligible_trials
        balanced_targets = torch.arange(eligible, device=self.device) % 2
        self._target_bits = balanced_targets.bool().expand(self.batch_size, -1).clone()
        for row in range(self.batch_size):
            target_order = torch.randperm(
                eligible,
                generator=self._generator,
                device=self.device,
            )
            self._target_bits[row] = self._target_bits[row, target_order]
        for position in range(self.n_back, self.symbol_steps):
            reference = self._symbols[:, position - self.n_back]
            different = (
                reference
                + torch.randint(
                    1,
                    self.symbol_count,
                    (self.batch_size,),
                    generator=self._generator,
                    device=self.device,
                )
            ) % self.symbol_count
            target = self._target_bits[:, position - self.n_back]
            self._symbols[:, position] = torch.where(target, reference, different)
        if self.time_shuffle and eligible > 1:
            for row in range(self.batch_size):
                target_order = torch.randperm(
                    eligible,
                    generator=self._generator,
                    device=self.device,
                )
                self._target_bits[row] = self._target_bits[row, target_order]
        self._position = 0

    def observation(self) -> torch.Tensor:
        """Return the current raw symbol stream without its hidden target."""

        if self._symbols.numel() == 0:
            raise RuntimeError("reset must be called before observation")
        if self.done:
            raise RuntimeError("the verifier has no observations remaining")
        if self.cue_symbol is not None and self._position == 0:
            return torch.full(
                (self.batch_size,),
                self.cue_symbol,
                dtype=torch.long,
                device=self.device,
            )
        symbol_position = (
            self._position - 1 if self.cue_symbol is not None else self._position
        )
        return self._symbols[:, symbol_position].clone()

    def score(self, action: torch.Tensor) -> NBackVerifierStep:
        """Score one binary action and advance the hidden verifier state."""

        if self._symbols.numel() == 0:
            raise RuntimeError("reset must be called before score")
        if self.done:
            raise RuntimeError("the verifier episode is complete")
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError("action must have shape [batch] and dtype int64")
        if bool(torch.any((action < 0) | (action >= self.action_count))):
            raise ValueError("action is outside the keypress vocabulary")
        if self.cue_symbol is not None and self._position == 0:
            reward = torch.zeros(self.batch_size, device=self.device)
            eligible = torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            )
        else:
            symbol_position = (
                self._position - 1
                if self.cue_symbol is not None
                else self._position
            )
            if symbol_position < self.n_back:
                reward = torch.zeros(self.batch_size, device=self.device)
                eligible = torch.zeros(
                    self.batch_size, dtype=torch.bool, device=self.device
                )
            else:
                expected = self._target_bits[:, symbol_position - self.n_back]
                reward = (action == expected.to(torch.long)).to(torch.float32)
                eligible = torch.ones(
                    self.batch_size, dtype=torch.bool, device=self.device
                )
        self._position += 1
        return NBackVerifierStep(reward=reward, eligible=eligible)


class BrainWorkshopEventEncoder(nn.Module):
    """Map raw symbol observations to learned amodal event tensors."""

    schema = EVENT_ENCODER_SCHEMA

    def __init__(self, symbol_count: int, event_width: int) -> None:
        super().__init__()
        if min(symbol_count, event_width) < 1:
            raise ValueError("event encoder dimensions must be positive")
        self.symbol_count = int(symbol_count)
        self.event_width = int(event_width)
        self.embedding = nn.Embedding(self.symbol_count, self.event_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "symbol_count": self.symbol_count,
            "event_width": self.event_width,
        }

    def forward(self, symbol: torch.Tensor) -> torch.Tensor:
        if symbol.ndim != 1 or symbol.dtype != torch.long:
            raise ValueError("symbol observation must have shape [batch] and dtype int64")
        if bool(torch.any(symbol < 0)) or bool(torch.any(symbol >= self.symbol_count)):
            raise ValueError("symbol observation is outside the frontend vocabulary")
        return self.embedding(symbol)
