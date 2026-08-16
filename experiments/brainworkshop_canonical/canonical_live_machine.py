"""Drive the canonical amodal agent through Neural Workshop's live boundary.

The historical live adapter uses ``SourcePreservingTemporalMachine``.  This
module provides the missing bridge for the canonical runner: rendered public
events enter the same ``CanonicalBrainWorkshopAgent`` that can later be used by
the maze adapter, and authenticated scalar outcomes update only its external
intention repertoire.  Controller parameters remain frozen.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import torch

from neural_computer import (
    CognitiveTickRuntime,
    ControllerFeedback,
    LiveActionProposal,
    ResolvedLiveOutcome,
)

from .neural_workshop_live import (
    NeuralWorkshopAudioEncoder,
    NeuralWorkshopEnvironment,
    NeuralWorkshopInstructionEncoder,
    NeuralWorkshopIntervention,
    NeuralWorkshopLiveConfig,
    NeuralWorkshopLiveDevice,
    NeuralWorkshopLiveReport,
    NeuralWorkshopRGBAEncoder,
)
from .runner import CanonicalBrainWorkshopAgent

CANONICAL_LIVE_MACHINE_SCHEMA = "neural-computer.canonical-live-machine.v1"


def _controller_digest(agent: CanonicalBrainWorkshopAgent) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(agent.controller.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class CanonicalBrainWorkshopLiveMachine:
    """Adapt one canonical agent to ``CognitiveTickRuntime``.

    The live transport owns receipts and device protocols.  This machine sees
    only learned event tensors and resolved scalar outcomes, then emits one
    opaque keypress proposal.  ``record_intention_memory`` is effectively
    always enabled: present outcomes update the external repertoire, while
    missing evidence remains explicitly absent.
    """

    schema = CANONICAL_LIVE_MACHINE_SCHEMA
    batch_size = 1

    def __init__(
        self,
        agent: CanonicalBrainWorkshopAgent,
        *,
        sample: bool = False,
        output_key: str = "keypress",
    ) -> None:
        if not isinstance(agent, CanonicalBrainWorkshopAgent):
            raise TypeError("canonical live machine needs a canonical agent")
        if not isinstance(sample, bool):
            raise TypeError("canonical live sampling flag must be boolean")
        if not isinstance(output_key, str) or not output_key:
            raise ValueError("canonical live output key must be non-empty")
        if output_key not in agent.runtime.output_bus.decoders:
            raise KeyError(f"canonical agent has no decoder named {output_key!r}")
        self.agent = agent
        self.sample = sample
        self.output_key = output_key
        self.event_width = agent.controller.width
        self._state = agent.initial_state(1, device="cpu")
        self._feedback = agent.initial_feedback(1, device="cpu")
        self._tick = 0

    def tick(
        self,
        events,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        now: float,
        elapsed: float,
    ) -> Sequence[LiveActionProposal]:
        del now
        if len(outcomes) > 1:
            raise RuntimeError("canonical Workshop machine received multiple outcomes")
        if outcomes:
            resolved = outcomes[0]
            action = resolved.receipt.action.to(dtype=torch.long)
            self._feedback = ControllerFeedback(
                action=self.agent.keypress_encoder(action),
                reward=resolved.event.reward,
                propensity=resolved.receipt.propensity,
                has_feedback=resolved.event.present.to(torch.float32),
            ).validate(
                batch=1,
                action_width=self.agent.controller.feedback_width,
            )
            self.agent.observe_intention(
                resolved.proposal.intention.payload.detach(),
                utility=resolved.event.reward.detach(),
                propensity=resolved.receipt.propensity.detach(),
                timestamp=self._tick,
                outcome_mask=resolved.event.present.detach(),
            )

        # The final authenticated outcome closes the last action, but the
        # device may have no next public frame.  Drain that outcome without
        # inventing another action after the environment has ended.
        if not bool(events.present.any()):
            return ()

        with torch.inference_mode():
            output, self._state = self.agent.runtime.step_events(
                events,
                self._state,
                self._feedback,
                elapsed=elapsed,
            )
            logits = output.decoded[self.output_key]
            decision = self.agent.keypress_decoder.decide_from_logits(
                logits,
                sample=self.sample,
            )
        self._tick += 1
        return (
            LiveActionProposal(
                intention=output.intention,
                action=decision.key_index,
                propensity=decision.propensity,
                output_key=self.output_key,
                model_version=0,
            ),
        )


def run_canonical_neural_workshop_live_lifetime(
    agent: CanonicalBrainWorkshopAgent,
    config: NeuralWorkshopLiveConfig,
    *,
    seed: int,
    environment: NeuralWorkshopEnvironment,
    verifier: Any,
    sample: bool = False,
    tick_seconds: float = 0.001,
    max_tick_seconds: float | None = None,
    intervention: NeuralWorkshopIntervention | None = None,
) -> NeuralWorkshopLiveReport:
    """Run one rendered Workshop lifetime on an existing canonical agent."""

    config.validate()
    if config.action_ports != 1:
        raise ValueError(
            "the canonical live bridge currently supports one Workshop action port"
        )
    if agent.controller.width != config.event_width:
        raise ValueError("canonical agent and live frontend event widths differ")
    if tick_seconds <= 0.0:
        raise ValueError("canonical live tick duration must be positive")

    encoder = NeuralWorkshopRGBAEncoder(config, seed=seed)
    instruction_encoder = NeuralWorkshopInstructionEncoder(config)
    device = NeuralWorkshopLiveDevice(
        environment,
        encoder,
        verifier,
        intervention=intervention,
        instruction_encoder=instruction_encoder,
    )
    machine = CanonicalBrainWorkshopLiveMachine(agent, sample=sample)
    runtime = CognitiveTickRuntime(
        device,
        machine,
        {"keypress": device},
        max_tick_seconds=max_tick_seconds,
    )
    controller_before = _controller_digest(agent)
    actions: list[int] = []
    propensities: list[float] = []
    results = []
    started = time.perf_counter()
    now = 0.0
    try:
        while not device.done or runtime.pending_receipts:
            result = runtime.tick(now)
            results.append(result)
            actions.extend(int(item.action.item()) for item in result.emitted_receipts)
            propensities.extend(
                float(item.propensity.item()) for item in result.emitted_receipts
            )
            if len(results) > config.trials + 2:
                raise RuntimeError("canonical live Workshop session failed to drain")
            now += tick_seconds
    finally:
        environment.close()
    wall_seconds = time.perf_counter() - started
    controller_after = _controller_digest(agent)
    authenticated = tuple(device.authenticated_outcomes)
    total_seconds = sorted(item.total_seconds for item in results)

    def percentile(fraction: float) -> float | None:
        if not total_seconds:
            return None
        index = min(len(total_seconds) - 1, int(fraction * len(total_seconds)))
        return total_seconds[index]

    accounting = getattr(environment, "accounting", None)
    snapshot = getattr(accounting, "snapshot", lambda: {})()
    logical_trials = int(snapshot.get("logical_trials", len(actions)))
    return NeuralWorkshopLiveReport(
        grid_size=config.grid_size,
        active_cells=config.active_cells,
        n_back=config.n_back,
        requested_trials=config.trials,
        logical_trials=logical_trials,
        input_events=sum(item.input_event_count for item in results),
        emitted_actions=len(actions),
        unique_verifier_bits=len(authenticated),
        learner_outcome_bits=sum(
            item.event.present.sum().item()
            for result in results
            for item in result.resolved_outcomes
        ),
        positive_verifier_bits=sum(
            item.verifier_reward >= 0.5 for item in authenticated
        ),
        optimizer_updates=0,
        program_file_updates=0,
        replayed_examples=0,
        controller_frozen=controller_before == controller_after,
        controller_digest_before=controller_before,
        controller_digest_after=controller_after,
        rewards=tuple(item.learner_reward for item in authenticated),
        verifier_rewards=tuple(item.verifier_reward for item in authenticated),
        signed_scalars=tuple(item.signed_scalar for item in authenticated),
        actions=tuple(actions),
        executed_actions=tuple(device.executed_actions),
        propensities=tuple(propensities),
        evidence_digests=tuple(item.evidence_digests for item in authenticated),
        event_payloads=tuple(
            tuple(float(value) for value in payload)
            for payload in encoder.emitted_payloads
        ),
        instruction_payloads=tuple(
            tuple(float(value) for value in payload)
            for payload in instruction_encoder.emitted_payloads
        ),
        audio_payloads=(),
        ticks=len(results),
        deadline_misses=sum(int(item.deadline_missed) for item in results),
        wall_seconds=wall_seconds,
        tick_seconds_p50=percentile(0.50),
        tick_seconds_p99=percentile(0.99),
        intervention=asdict((intervention or NeuralWorkshopIntervention()).validate()),
    )


__all__ = [
    "CANONICAL_LIVE_MACHINE_SCHEMA",
    "CanonicalBrainWorkshopLiveMachine",
    "run_canonical_neural_workshop_live_lifetime",
]
