"""Live, batch-one Brain Workshop transport and online acquisition probe.

This module exercises the production cognitive tick boundary with one causal
experience at a time. It is intentionally a first-rung transport/acquisition
probe: the temporal capability consumes one learned event stream and emits an
opaque intention. It does not claim autonomous capability allocation,
multistream composition, pixel-level Brain Workshop operation, or promotion of
the full controller path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neural_computer import (
    AmodalEvent,
    AmodalEventCollection,
    CognitiveTickRuntime,
    IntentEvent,
    KeypressDecoder,
    LiveActionProposal,
    LiveActionReceipt,
    LiveInputBatch,
    LiveOutcomeEvent,
    LiveTickResult,
    ResolvedLiveOutcome,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier

LIVE_BRAINWORKSHOP_SCHEMA = "neural-computer.brainworkshop-live-session.v1"


class BrainWorkshopLiveDevice:
    """Batch-one live frontend/backend around the private n-back verifier.

    The learner-facing side receives only learned event tensors and scalar
    outcome events. Hidden symbols and target bits remain inside the device.
    Explicit ``present=False`` resolutions retire warm-up actions without
    pretending that an unobserved reward was zero.
    """

    def __init__(
        self,
        verifier: NBackVerifier,
        encoder: BrainWorkshopEventEncoder,
    ) -> None:
        if verifier.batch_size != 1:
            raise ValueError("the first live Brain Workshop device is batch-one")
        if encoder.symbol_count < verifier.observation_symbol_count:
            raise ValueError("event encoder vocabulary is too small for the verifier")
        self.verifier = verifier
        self.encoder = encoder
        self.batch_size = verifier.batch_size
        self.event_width = encoder.event_width
        self._observation_pending = True
        self._outcomes: list[LiveOutcomeEvent] = []
        self._emitted_actions = 0
        self.verifier.reset()

    @property
    def done(self) -> bool:
        return (
            self.verifier.done
            and not self._observation_pending
            and not self._outcomes
        )

    @property
    def emitted_actions(self) -> int:
        return self._emitted_actions

    def poll(self, now: float) -> LiveInputBatch:
        outcomes = tuple(self._outcomes)
        self._outcomes.clear()
        if self._observation_pending and not self.verifier.done:
            symbol = self.verifier.observation()
            with torch.no_grad():
                payload = self.encoder(symbol)
            event = AmodalEvent(
                payload=payload,
                timestamp=torch.full((self.batch_size,), now),
                confidence=torch.ones(self.batch_size),
            )
            events = AmodalEventCollection.from_events(
                (event,), width=self.event_width
            )
            self._observation_pending = False
        else:
            events = AmodalEventCollection.empty(
                self.batch_size,
                self.event_width,
            )
        return LiveInputBatch(events=events, outcomes=outcomes, observed_at=now)

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        if self._observation_pending or self.verifier.done:
            raise RuntimeError("Brain Workshop received an action without a stimulus")
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError("Brain Workshop keypress must be an int64 batch vector")
        scored = self.verifier.score(action)
        present = scored.eligible
        self._outcomes.append(
            LiveOutcomeEvent(
                receipt_id=receipt.receipt_id,
                reward=torch.where(
                    present,
                    scored.reward,
                    torch.zeros_like(scored.reward),
                ),
                present=present,
                observed_at=receipt.emitted_at,
                confidence=torch.ones(self.batch_size),
            )
        )
        self._emitted_actions += self.batch_size
        self._observation_pending = not self.verifier.done


@dataclass(frozen=True)
class _TemporalCreditState:
    current: torch.Tensor
    history: torch.Tensor
    history_present: torch.Tensor


class OnlineTemporalCapabilityMachine(nn.Module):
    """Small generic relative-history capability updated per outcome tick.

    No lag is assigned a semantic meaning. A learned distribution addresses a
    bounded relative history and a shared relation network emits an opaque
    intention. Decision credit stores only detached learner-visible events, so
    delayed outcomes never require retaining an old autograd graph.
    """

    def __init__(
        self,
        event_width: int,
        *,
        max_history: int,
        intention_width: int = 8,
        hidden: int = 32,
        learning_rate: float = 3e-3,
        sample: bool = True,
    ) -> None:
        super().__init__()
        if min(event_width, max_history, intention_width, hidden) < 1:
            raise ValueError("live temporal capability dimensions must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        self.event_width = int(event_width)
        self.max_history = int(max_history)
        self.intention_width = int(intention_width)
        self.output_key = "keypress"
        self.sample = bool(sample)
        self.learning_enabled = True
        self.relative_address_logits = nn.Parameter(torch.zeros(max_history))
        self.relation = nn.Sequential(
            nn.Linear(event_width * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, intention_width),
        )
        self.decoder = KeypressDecoder(intention_width, 2, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self._history: list[torch.Tensor] = []
        self.model_version = 0
        self.optimizer_updates = 0
        self.unique_outcome_bits = 0
        self.last_loss: float | None = None

    def reset_history(self) -> None:
        self._history.clear()

    def _history_tensors(
        self,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = current.shape[0]
        if not self._history:
            return (
                torch.zeros(
                    batch,
                    self.max_history,
                    self.event_width,
                    dtype=current.dtype,
                    device=current.device,
                ),
                torch.zeros(
                    batch,
                    self.max_history,
                    dtype=torch.bool,
                    device=current.device,
                ),
            )
        newest_first = list(reversed(self._history[-self.max_history :]))
        present_count = len(newest_first)
        history = torch.zeros(
            batch,
            self.max_history,
            self.event_width,
            dtype=current.dtype,
            device=current.device,
        )
        history[:, :present_count] = torch.stack(newest_first, dim=1)
        present = torch.zeros(
            batch,
            self.max_history,
            dtype=torch.bool,
            device=current.device,
        )
        present[:, :present_count] = True
        return history, present

    def _forward_credit(
        self,
        credit: _TemporalCreditState,
    ) -> tuple[IntentEvent, torch.Tensor]:
        masked_address = self.relative_address_logits.unsqueeze(0).expand(
            credit.current.shape[0], -1
        )
        masked_address = masked_address.masked_fill(
            ~credit.history_present, torch.finfo(masked_address.dtype).min
        )
        any_history = credit.history_present.any(dim=1)
        safe_address = torch.where(
            any_history[:, None],
            masked_address,
            torch.zeros_like(masked_address),
        )
        weights = safe_address.softmax(dim=1)
        weights = weights * credit.history_present.to(weights.dtype)
        retrieved = (weights.unsqueeze(-1) * credit.history).sum(dim=1)
        relation_input = torch.cat(
            (
                credit.current,
                retrieved,
                credit.current - retrieved,
                credit.current * retrieved,
            ),
            dim=-1,
        )
        intention = IntentEvent(self.relation(relation_input))
        return intention, self.decoder(intention)

    def _learn(self, outcomes: list[ResolvedLiveOutcome]) -> None:
        losses: list[torch.Tensor] = []
        observed_bits = 0
        for resolved in outcomes:
            present = resolved.event.present
            observed_bits += int(present.sum().item())
            if not bool(present.any()):
                continue
            credit = resolved.proposal.credit_state
            if not isinstance(credit, _TemporalCreditState):
                raise TypeError("live temporal proposal has incompatible credit state")
            _intention, logits = self._forward_credit(credit)
            probabilities = logits.softmax(dim=-1)
            selected = probabilities.gather(
                1, resolved.receipt.action.to(torch.long).unsqueeze(-1)
            ).squeeze(-1)
            selected = selected.clamp(min=1e-6, max=1.0 - 1e-6)
            reward = resolved.event.reward
            correctness_loss = -(
                reward * selected.log()
                + (1.0 - reward) * torch.log1p(-selected)
            )
            losses.append(correctness_loss[present].mean())
        self.unique_outcome_bits += observed_bits
        if not losses or not self.learning_enabled:
            return
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer_updates += 1
        self.model_version += 1
        self.last_loss = float(loss.detach())

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: tuple[ResolvedLiveOutcome, ...],
        *,
        now: float,
        elapsed: float,
    ) -> tuple[LiveActionProposal, ...]:
        del now, elapsed
        self._learn(list(outcomes))
        if events.payload.shape[1] == 0:
            return ()
        if events.payload.shape[1] != 1 or not bool(events.present.all()):
            raise ValueError(
                "the first live temporal probe requires one separately bound event"
            )
        current = events.payload[:, 0].detach()
        history, history_present = self._history_tensors(current)
        credit = _TemporalCreditState(
            current=current.clone(),
            history=history.detach().clone(),
            history_present=history_present.clone(),
        )
        with torch.no_grad():
            intention, logits = self._forward_credit(credit)
            probabilities = logits.softmax(dim=-1)
            action = (
                torch.multinomial(probabilities, 1).squeeze(-1)
                if self.sample
                else probabilities.argmax(dim=-1)
            )
            propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        self._history.append(current.clone())
        if len(self._history) > self.max_history:
            del self._history[0]
        return (
            LiveActionProposal(
                intention=IntentEvent(intention.payload.detach()),
                action=action,
                propensity=propensity,
                output_key=self.output_key,
                model_version=self.model_version,
                credit_state=credit,
            ),
        )


@dataclass(frozen=True)
class LiveBrainWorkshopLifetime:
    """Accounting for one never-replayed causal Brain Workshop lifetime."""

    actions: torch.Tensor
    rewards: torch.Tensor
    outcome_present: torch.Tensor
    eligible_accuracy: float
    input_events: int
    unique_verifier_bits: int
    optimizer_updates: int
    replayed_examples: int
    ticks: int
    deadline_misses: int
    machine_seconds_p50: float
    machine_seconds_p99: float
    total_seconds_p50: float
    total_seconds_p99: float
    schema: str = LIVE_BRAINWORKSHOP_SCHEMA


def run_live_lifetime(
    machine: OnlineTemporalCapabilityMachine,
    encoder: BrainWorkshopEventEncoder,
    *,
    n_back: int,
    steps: int,
    seed: int,
    tick_seconds: float = 0.01,
    learn: bool = True,
    sample: bool = True,
    max_machine_seconds: float | None = None,
    max_tick_seconds: float | None = None,
) -> LiveBrainWorkshopLifetime:
    """Consume one n-back lifetime online, updating at most once per tick."""

    if min(n_back, steps) < 1 or steps <= n_back:
        raise ValueError("live lifetime needs target-bearing positive dimensions")
    if tick_seconds <= 0.0:
        raise ValueError("tick duration must be positive")
    verifier = NBackVerifier(
        batch_size=1,
        n_back=n_back,
        steps=steps,
        symbol_count=encoder.symbol_count,
        seed=seed,
    )
    device = BrainWorkshopLiveDevice(verifier, encoder)
    machine.reset_history()
    machine.learning_enabled = learn
    machine.sample = sample
    runtime = CognitiveTickRuntime(
        device,
        machine,
        {"keypress": device},
        max_machine_seconds=max_machine_seconds,
        max_tick_seconds=max_tick_seconds,
    )
    action_rows: list[torch.Tensor] = []
    reward_rows: list[torch.Tensor] = []
    present_rows: list[torch.Tensor] = []
    input_events = 0
    deadline_misses = 0
    start_updates = machine.optimizer_updates
    now = 0.0
    max_ticks = steps + 2
    results: list[LiveTickResult] = []
    while not device.done or runtime.pending_receipts:
        result = runtime.tick(now)
        results.append(result)
        input_events += result.input_event_count
        deadline_misses += int(result.deadline_missed)
        action_rows.extend(receipt.action for receipt in result.emitted_receipts)
        for resolved in result.resolved_outcomes:
            reward_rows.append(resolved.event.reward)
            present_rows.append(resolved.event.present)
        if len(results) > max_ticks:
            raise RuntimeError("live Brain Workshop session failed to drain")
        now += tick_seconds
    actions = torch.stack(action_rows, dim=1)
    rewards = torch.stack(reward_rows, dim=1)
    present = torch.stack(present_rows, dim=1)
    eligible_count = present.sum().clamp_min(1)
    eligible_accuracy = float(
        (rewards * present.to(rewards.dtype)).sum() / eligible_count
    )
    machine_seconds = sorted(result.machine_seconds for result in results)
    total_seconds = sorted(result.total_seconds for result in results)

    def percentile(values: list[float], fraction: float) -> float:
        index = min(len(values) - 1, int(fraction * len(values)))
        return values[index]

    return LiveBrainWorkshopLifetime(
        actions=actions,
        rewards=rewards,
        outcome_present=present,
        eligible_accuracy=eligible_accuracy,
        input_events=input_events,
        unique_verifier_bits=int(present.sum().item()),
        optimizer_updates=machine.optimizer_updates - start_updates,
        replayed_examples=0,
        ticks=len(results),
        deadline_misses=deadline_misses,
        machine_seconds_p50=percentile(machine_seconds, 0.50),
        machine_seconds_p99=percentile(machine_seconds, 0.99),
        total_seconds_p50=percentile(total_seconds, 0.50),
        total_seconds_p99=percentile(total_seconds, 0.99),
    )
