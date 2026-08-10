"""Rendered replay-free acquisition of an external transition model.

This is a pressure test for the CPU/files boundary, not a claim that the
current random keypress decoder has mastered Brain Workshop.  The controller,
frontend, and decoder are frozen.  Fresh rendered verifier lifetimes produce
opaque planner-state transitions; an affine external transition bank consumes
each row once through sufficient statistics.  Recursive held-out error is
compared with a matched fresh bank, and the report keeps the claim narrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    ContentAddressedMemory,
    ControllerFeedback,
    ExternalModelBasedPlanner,
    ExternalOnlineTransitionContextResult,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
    ExternalTransitionRollout,
    PolicyFreeAmodalRuntime,
)

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent

REPLAY_FREE_TRANSITION_AUDIT_SCHEMA = (
    "neural-computer.brainworkshop-replay-free-transition-acquisition-audit.v1"
)


@dataclass(frozen=True)
class ReplayFreeTransitionAcquisitionReport:
    schema: str
    status: str
    controller_unchanged: bool
    replay_free_bank: bool
    model_improved_on_heldout_rollout: bool
    trained_heldout_error: float
    fresh_heldout_error: float
    training_lifetimes: int
    total_logical_lifetimes: int
    unique_verifier_bits: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int
    external_slot_count: int
    external_sample_count: int
    fresh_sample_count: int
    schema_version: str = REPLAY_FREE_TRANSITION_AUDIT_SCHEMA

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NonstationaryTransitionRetentionReport:
    """Report for two external transition families learned sequentially."""

    schema: str
    status: str
    controller_unchanged: bool
    replay_free_bank: bool
    source_slot_byte_stable: bool
    target_model_improved_on_heldout: bool
    source_heldout_error_before_target: float
    source_heldout_error_after_target: float
    trained_target_heldout_error: float
    fresh_target_heldout_error: float
    source_training_lifetimes: int
    target_training_lifetimes: int
    total_logical_lifetimes: int
    unique_verifier_bits: int
    transition_rows_consumed_once: int
    replayed_examples: int
    external_slot_count: int
    source_sample_count: int
    target_sample_count: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineTransitionDiscoveryReport:
    """Report for outcome-free online admission of a novel transition family."""

    schema: str
    status: str
    controller_unchanged: bool
    replay_free_bank: bool
    source_slot_byte_stable: bool
    target_context_discovered: bool
    target_route_recovered: bool
    target_model_improved_on_heldout: bool
    source_heldout_error_before_target: float
    source_heldout_error_after_target: float
    trained_target_heldout_error: float
    fresh_target_heldout_error: float
    source_training_lifetimes: int
    target_training_lifetimes: int
    total_logical_lifetimes: int
    unique_verifier_bits: int
    transition_rows_consumed_once: int
    replayed_examples: int
    external_slot_count: int
    source_sample_count: int
    target_sample_count: int
    target_promotion_accepted: bool
    target_promotion_reason: str
    target_discovery_status: str
    target_continuation_status: str
    target_heldout_status: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _controller_digest(agent: CanonicalBrainWorkshopAgent) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(agent.controller.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _opaque_context(
    agent: CanonicalBrainWorkshopAgent,
    cue_symbol: int,
) -> torch.Tensor:
    """Derive one route key from an encoded rendered cue only."""

    encoded = agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )
    return F.normalize(encoded.detach(), dim=-1)[0]


def _run_lifetime(
    agent: CanonicalBrainWorkshopAgent,
    policy_free: PolicyFreeAmodalRuntime,
    bank: ExternalTransitionModelBank,
    context: torch.Tensor,
    *,
    n_back: int,
    steps: int,
    seed: int,
    cue_symbol: int,
    candidate_intentions: torch.Tensor,
    learn: bool,
) -> tuple[ExternalTransitionRollout, int]:
    """Run one fresh rendered lifetime and optionally consume its rows once."""

    verifier = NBackVerifier(
        batch_size=1,
        n_back=n_back,
        steps=steps,
        symbol_count=4,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    verifier.reset()
    if isinstance(agent.runtime.memory, ContentAddressedMemory):
        agent.runtime.memory.clear()
    state = agent.initial_state(1, device=verifier.device)
    feedback = agent.initial_feedback(1, device=verifier.device)
    goal_state = torch.zeros(1, bank.state_width, device=verifier.device)
    previous = None
    outputs = []
    unique_verifier_bits = 0

    while not verifier.done:
        events = agent.runtime.encode_streams(
            {"stimulus": verifier.observation()}
        )
        output, next_state = policy_free.step_events(
            events,
            state,
            feedback,
            goal_state,
            candidate_intentions,
            horizon=1,
            beam_width=4,
            transition_context=context.unsqueeze(0),
        )
        if previous is not None and learn:
            observation = policy_free.transition_observation(previous, output)
            policy_free.learn_transition_once(observation, context)
        outputs.append(output)

        decision = agent.keypress_decoder.decide_from_logits(
            output.decoded["keypress"],
            sample=False,
        )
        scored = verifier.score(decision.key_index)
        unique_verifier_bits += int(scored.eligible.sum())
        feedback = ControllerFeedback(
            action=agent.keypress_encoder(decision.key_index),
            reward=scored.reward,
            propensity=decision.propensity,
            has_feedback=scored.eligible.to(scored.reward.dtype),
        )
        state = next_state
        previous = output

    if len(outputs) < 2:
        raise RuntimeError("transition audit lifetime needs at least two outputs")
    return (
        ExternalTransitionRollout(
            initial_state=outputs[0].state[0].detach().clone(),
            intentions=torch.cat(
                [output.intention.payload for output in outputs[:-1]], dim=0
            ).detach(),
            expected_states=torch.cat(
                [output.state for output in outputs[1:]], dim=0
            ).detach(),
        ).validate(
            state_width=bank.state_width,
            intention_width=bank.intention_width,
        ),
        unique_verifier_bits,
    )


def _rollout_observations(
    rollout: ExternalTransitionRollout,
) -> list[ExternalTransitionObservation]:
    """Expand an opaque rollout into one-pass row bundles for an external router."""

    state = rollout.initial_state.unsqueeze(0)
    observations: list[ExternalTransitionObservation] = []
    for index in range(rollout.horizon):
        observations.append(
            ExternalTransitionObservation(
                state=state.detach().clone(),
                intention=rollout.intentions[index : index + 1].detach().clone(),
                next_state=rollout.expected_states[index : index + 1]
                .detach()
                .clone(),
            ).validate(
                state_width=rollout.initial_state.shape[0],
                intention_width=rollout.intentions.shape[1],
            )
        )
        state = rollout.expected_states[index : index + 1]
    return observations


def _route_rollout(
    router: ExternalOnlineTransitionContextRouter,
    rollout: ExternalTransitionRollout,
    *,
    adapt: bool,
) -> ExternalOnlineTransitionContextResult:
    """Route a fresh rollout and optionally consume each admitted bundle once."""

    result = None
    for observation in _rollout_observations(rollout):
        result = router.observe(observation)
        if adapt and result.status in {
            "admitted",
            "reused",
            "matched",
            "continuation",
            "sparse_matched",
            "staged",
        }:
            router.adaptation_step(result, None, replay_evidence=False)
    if result is None:
        raise RuntimeError("online transition routing needs a non-empty rollout")
    return result


def _rollout_bundle(
    rollout: ExternalTransitionRollout,
) -> ExternalTransitionObservation:
    """Combine one held-out rollout into one opaque verification bundle."""

    rows = _rollout_observations(rollout)
    if not rows:
        raise RuntimeError("transition rollout bundle needs at least one row")
    return ExternalTransitionObservation(
        state=torch.cat([row.state for row in rows]),
        intention=torch.cat([row.intention for row in rows]),
        next_state=torch.cat([row.next_state for row in rows]),
    ).validate(
        state_width=rollout.initial_state.shape[0],
        intention_width=rollout.intentions.shape[1],
    )


def run_replay_free_transition_acquisition_audit(
    *,
    seed: int = 91,
    n_back: int = 2,
    steps: int = 6,
    training_lifetimes: int = 3,
    cue_symbol: int = 6,
) -> ReplayFreeTransitionAcquisitionReport:
    """Run the smallest rendered external-transition acquisition rung."""

    if min(n_back, steps, training_lifetimes) < 1:
        raise ValueError("transition audit dimensions must be positive")
    if cue_symbol < 4:
        raise ValueError("cue symbol must be outside the verifier vocabulary")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        event_width=4,
        intention_width=2,
        feedback_width=3,
        n_back=n_back,
        reader_kind="relation",
        seed=seed,
    )
    before = _controller_digest(agent)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)

    bank = ExternalTransitionModelBank(
        state_width=agent.controller.width * 3,
        intention_width=agent.controller.intention_width,
        context_width=agent.controller.width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    context = _opaque_context(agent, cue_symbol)
    bank.ensure_context(context)
    policy_free = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(bank, beam_width=4),
    )
    generator = torch.Generator().manual_seed(seed + 7000)
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=generator,
    )

    training_bits = 0
    training_rows = 0
    for lifetime in range(training_lifetimes):
        _, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            context,
            n_back=n_back,
            steps=steps,
            seed=seed + lifetime,
            cue_symbol=cue_symbol,
            candidate_intentions=candidate_intentions,
            learn=True,
        )
        training_bits += bits
        training_rows += steps

    heldout, heldout_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        context,
        n_back=n_back,
        steps=steps,
        seed=seed + 10000,
        cue_symbol=cue_symbol,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    trained_error = ExternalModelBasedPlanner(bank, beam_width=4).rollout_error(
        heldout,
        transition_context=context.unsqueeze(0),
    )

    fresh_bank = ExternalTransitionModelBank(
        state_width=bank.state_width,
        intention_width=bank.intention_width,
        context_width=bank.context_width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    fresh_bank.ensure_context(context)
    fresh_error = ExternalModelBasedPlanner(
        fresh_bank,
        beam_width=4,
    ).rollout_error(
        heldout,
        transition_context=context.unsqueeze(0),
    )
    unchanged = before == _controller_digest(agent)
    improved = trained_error < fresh_error
    return ReplayFreeTransitionAcquisitionReport(
        schema=REPLAY_FREE_TRANSITION_AUDIT_SCHEMA,
        status=(
            "rendered_replay_free_transition_boundary"
            if unchanged and improved
            else "rendered_replay_free_transition_boundary_failed"
        ),
        controller_unchanged=unchanged,
        replay_free_bank=bank.replay_free_updates,
        model_improved_on_heldout_rollout=improved,
        trained_heldout_error=trained_error,
        fresh_heldout_error=fresh_error,
        training_lifetimes=training_lifetimes,
        total_logical_lifetimes=training_lifetimes + 1,
        unique_verifier_bits=training_bits + heldout_bits,
        transition_rows_consumed_once=training_rows,
        optimizer_updates=0,
        replayed_examples=0,
        external_slot_count=bank.context_count,
        external_sample_count=int(bank.models[0].sample_count),
        fresh_sample_count=int(fresh_bank.models[0].sample_count),
    )


def run_nonstationary_transition_retention_audit(
    *,
    seed: int = 92,
    steps: int = 6,
    source_training_lifetimes: int = 2,
    target_training_lifetimes: int = 2,
) -> NonstationaryTransitionRetentionReport:
    """Learn two rendered families sequentially without replaying the source."""

    if min(steps, source_training_lifetimes, target_training_lifetimes) < 1:
        raise ValueError("nonstationary transition audit budgets must be positive")
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        event_width=4,
        intention_width=2,
        feedback_width=3,
        n_back=2,
        reader_kind="relation",
        seed=seed,
    )
    controller_before = _controller_digest(agent)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    bank = ExternalTransitionModelBank(
        state_width=agent.controller.width * 3,
        intention_width=agent.controller.intention_width,
        context_width=agent.controller.width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    source_context = _opaque_context(agent, 6)
    target_context = _opaque_context(agent, 7)
    source_index = bank.ensure_context(source_context)
    target_index = bank.ensure_context(target_context)
    policy_free = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(bank, beam_width=4),
    )
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )

    unique_bits = 0
    consumed_rows = 0
    for lifetime in range(source_training_lifetimes):
        _, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            source_context,
            n_back=2,
            steps=steps,
            seed=seed + lifetime,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=True,
        )
        unique_bits += bits
        consumed_rows += steps
    source_digest = bank.models[source_index].digest()
    source_heldout, source_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        source_context,
        n_back=2,
        steps=steps,
        seed=seed + 10000,
        cue_symbol=6,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_bits += source_bits
    planner = ExternalModelBasedPlanner(bank, beam_width=4)
    source_error_before = planner.rollout_error(
        source_heldout,
        transition_context=source_context.unsqueeze(0),
    )

    for lifetime in range(target_training_lifetimes):
        _, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            target_context,
            n_back=3,
            steps=steps,
            seed=seed + 2000 + lifetime,
            cue_symbol=7,
            candidate_intentions=candidate_intentions,
            learn=True,
        )
        unique_bits += bits
        consumed_rows += steps
    target_heldout, target_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        target_context,
        n_back=3,
        steps=steps,
        seed=seed + 12000,
        cue_symbol=7,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_bits += target_bits
    source_error_after = planner.rollout_error(
        source_heldout,
        transition_context=source_context.unsqueeze(0),
    )
    trained_target_error = planner.rollout_error(
        target_heldout,
        transition_context=target_context.unsqueeze(0),
    )

    fresh_bank = ExternalTransitionModelBank(
        state_width=bank.state_width,
        intention_width=bank.intention_width,
        context_width=bank.context_width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    fresh_bank.ensure_context(target_context)
    fresh_target_error = ExternalModelBasedPlanner(
        fresh_bank,
        beam_width=4,
    ).rollout_error(
        target_heldout,
        transition_context=target_context.unsqueeze(0),
    )
    unchanged = controller_before == _controller_digest(agent)
    source_stable = source_digest == bank.models[source_index].digest()
    target_improved = trained_target_error < fresh_target_error
    return NonstationaryTransitionRetentionReport(
        schema=(
            "neural-computer.brainworkshop-nonstationary-transition-retention-audit.v1"
        ),
        status=(
            "nonstationary_replay_free_transition_boundary"
            if unchanged and source_stable and target_improved
            else "nonstationary_replay_free_transition_boundary_failed"
        ),
        controller_unchanged=unchanged,
        replay_free_bank=bank.replay_free_updates,
        source_slot_byte_stable=source_stable,
        target_model_improved_on_heldout=target_improved,
        source_heldout_error_before_target=source_error_before,
        source_heldout_error_after_target=source_error_after,
        trained_target_heldout_error=trained_target_error,
        fresh_target_heldout_error=fresh_target_error,
        source_training_lifetimes=source_training_lifetimes,
        target_training_lifetimes=target_training_lifetimes,
        total_logical_lifetimes=(
            source_training_lifetimes
            + target_training_lifetimes
            + 2
        ),
        unique_verifier_bits=unique_bits,
        transition_rows_consumed_once=consumed_rows,
        replayed_examples=0,
        external_slot_count=bank.context_count,
        source_sample_count=int(bank.models[source_index].sample_count),
        target_sample_count=int(bank.models[target_index].sample_count),
    )


def run_online_transition_discovery_audit(
    *,
    seed: int = 93,
    steps: int = 6,
    source_training_lifetimes: int = 2,
    target_training_lifetimes: int = 2,
    affine_ridge: float = 1e-5,
) -> OnlineTransitionDiscoveryReport:
    """Discover and learn a novel rendered family without replay or a task label.

    The target lifetime initially plans through the already-known source slot.
    Only its opaque transition observations reach the online router.  The
    router stages a new external slot, consumes each training bundle once
    through replay-free affine statistics, and commits the discovered context
    only after held-out and source-retention probes. Cue symbols and n-back
    values remain verifier-private diagnostics and never enter the router.
    """

    if min(steps, source_training_lifetimes, target_training_lifetimes) < 1:
        raise ValueError("online transition audit budgets must be positive")
    if target_training_lifetimes < 2:
        raise ValueError("online transition discovery needs a continuation lifetime")
    if affine_ridge <= 0.0 or not math.isfinite(float(affine_ridge)):
        raise ValueError("online transition affine ridge must be finite and positive")
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        event_width=4,
        intention_width=2,
        feedback_width=3,
        n_back=2,
        reader_kind="relation",
        seed=seed,
    )
    controller_before = _controller_digest(agent)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    bank = ExternalTransitionModelBank(
        state_width=agent.controller.width * 3,
        intention_width=agent.controller.intention_width,
        context_width=agent.controller.width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        affine_ridge=affine_ridge,
    )
    source_context = _opaque_context(agent, 6)
    source_index = bank.ensure_context(source_context)
    policy_free = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(bank, beam_width=4),
    )
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )

    unique_bits = 0
    for lifetime in range(source_training_lifetimes):
        _, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            source_context,
            n_back=2,
            steps=steps,
            seed=seed + lifetime,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=True,
        )
        unique_bits += bits

    source_digest = bank.models[source_index].digest()
    source_heldout, source_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        source_context,
        n_back=2,
        steps=steps,
        seed=seed + 10000,
        cue_symbol=6,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_bits += source_bits
    planner = ExternalModelBasedPlanner(bank, beam_width=4)
    source_error_before = planner.rollout_error(
        source_heldout,
        transition_context=source_context.unsqueeze(0),
    )

    context_encoder = ExternalTransitionContextEncoder(
        bank.state_width,
        bank.intention_width,
        hidden_width=max(16, bank.state_width),
        context_width=bank.context_width,
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        context_encoder,
        admission_observations=steps,
        match_tolerance=0.05,
        match_margin=0.0,
        continuation_tolerance=0.05,
        defer_admission=True,
        max_contexts=2,
        provisional_evidence_policy="streaming_statistics",
        prior_selection_probe=(
            lambda transfer, fresh, observation: (
                float(transfer.loss(observation).detach()),
                float(fresh.loss(observation).detach()),
            )
        ),
    )

    # Acquire the target while still behaving through the known source slot.
    # The router sees only the resulting opaque transition rows and keeps the
    # candidate outside the committed bank until the held-out gate passes.
    discovery = None
    target_result = None
    for lifetime in range(target_training_lifetimes):
        target_rollout, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            source_context,
            n_back=3,
            steps=steps,
            seed=seed + 2000 + lifetime,
            cue_symbol=7,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        unique_bits += bits
        routed = _route_rollout(router, target_rollout, adapt=True)
        if discovery is None:
            discovery = routed
        target_result = routed
    source_error_after = planner.rollout_error(
        source_heldout,
        transition_context=source_context.unsqueeze(0),
    )
    discovery_status = "not_staged" if discovery is None else discovery.status
    if (
        discovery is None
        or target_result is None
        or discovery.context is None
        or discovery.status != "staged"
        or router.provisional_candidate_count != 1
    ):
        unchanged = controller_before == _controller_digest(agent)
        source_stable = source_digest == bank.models[source_index].digest()
        return OnlineTransitionDiscoveryReport(
            schema=(
                "neural-computer.brainworkshop-online-transition-discovery-audit.v3"
            ),
            status="online_replay_free_transition_discovery_boundary_failed",
            controller_unchanged=unchanged,
            replay_free_bank=bank.replay_free_updates,
            source_slot_byte_stable=source_stable,
            target_context_discovered=False,
            target_route_recovered=False,
            target_model_improved_on_heldout=False,
            source_heldout_error_before_target=source_error_before,
            source_heldout_error_after_target=source_error_after,
            trained_target_heldout_error=float("inf"),
            fresh_target_heldout_error=float("inf"),
            source_training_lifetimes=source_training_lifetimes,
            target_training_lifetimes=target_training_lifetimes,
            total_logical_lifetimes=(
                source_training_lifetimes + target_training_lifetimes + 1
            ),
            unique_verifier_bits=unique_bits,
            transition_rows_consumed_once=source_training_lifetimes * steps,
            replayed_examples=0,
            external_slot_count=bank.context_count,
            source_sample_count=int(bank.models[source_index].sample_count),
            target_sample_count=0,
            target_promotion_accepted=False,
            target_promotion_reason="no provisional target candidate was staged",
            target_discovery_status=discovery_status,
            target_continuation_status=(
                "not_run" if target_result is None else target_result.status
            ),
            target_heldout_status="not_run",
        )

    candidate_context = router.provisional_context_at(0)
    promotion_holdout, promotion_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        source_context,
        n_back=3,
        steps=steps,
        seed=seed + 12000,
        cue_symbol=7,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_bits += promotion_bits
    promotion_route = _route_rollout(router, promotion_holdout, adapt=False)
    promotion_observation = _rollout_bundle(promotion_holdout)

    def retention_probe(candidate_bank: ExternalTransitionModelBank) -> bool:
        if candidate_bank.models[source_index].digest() != source_digest:
            return False
        candidate_runtime = PolicyFreeAmodalRuntime(
            agent.runtime,
            ExternalModelBasedPlanner(candidate_bank, beam_width=4),
        )
        candidate_rollout, _ = _run_lifetime(
            agent,
            candidate_runtime,
            candidate_bank,
            candidate_context,
            n_back=3,
            steps=steps,
            seed=seed + 12000,
            cue_symbol=7,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        candidate_error = ExternalModelBasedPlanner(
            candidate_bank,
            beam_width=4,
        ).rollout_error(
            candidate_rollout,
            transition_context=candidate_context.unsqueeze(0),
        )
        fresh_probe_bank = ExternalTransitionModelBank(
            state_width=bank.state_width,
            intention_width=bank.intention_width,
            context_width=bank.context_width,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            affine_ridge=affine_ridge,
        )
        fresh_probe_bank.ensure_context(candidate_context)
        fresh_error = ExternalModelBasedPlanner(
            fresh_probe_bank,
            beam_width=4,
        ).rollout_error(
            candidate_rollout,
            transition_context=candidate_context.unsqueeze(0),
        )
        return candidate_error <= 0.2 and candidate_error < fresh_error

    promotion = router.promote_staged_candidate(
        promotion_observation,
        retention_probe,
        prediction_tolerance=0.2,
    )
    source_error_after = planner.rollout_error(
        source_heldout,
        transition_context=source_context.unsqueeze(0),
    )
    if not promotion.accepted or promotion.slot_index is None:
        unchanged = controller_before == _controller_digest(agent)
        source_stable = source_digest == bank.models[source_index].digest()
        return OnlineTransitionDiscoveryReport(
            schema=(
                "neural-computer.brainworkshop-online-transition-discovery-audit.v3"
            ),
            status="online_replay_free_transition_discovery_boundary_failed",
            controller_unchanged=unchanged,
            replay_free_bank=bank.replay_free_updates,
            source_slot_byte_stable=source_stable,
            target_context_discovered=False,
            target_route_recovered=False,
            target_model_improved_on_heldout=False,
            source_heldout_error_before_target=source_error_before,
            source_heldout_error_after_target=source_error_after,
            trained_target_heldout_error=float("inf"),
            fresh_target_heldout_error=float("inf"),
            source_training_lifetimes=source_training_lifetimes,
            target_training_lifetimes=target_training_lifetimes,
            total_logical_lifetimes=(
                source_training_lifetimes + target_training_lifetimes + 2
            ),
            unique_verifier_bits=unique_bits,
            transition_rows_consumed_once=(
                source_training_lifetimes * steps
                + target_training_lifetimes * steps
            ),
            replayed_examples=0,
            external_slot_count=bank.context_count,
            source_sample_count=int(bank.models[source_index].sample_count),
            target_sample_count=0,
            target_promotion_accepted=False,
            target_promotion_reason=promotion.reason,
            target_discovery_status=discovery.status,
            target_continuation_status=target_result.status,
            target_heldout_status=promotion_route.status,
        )

    target_context = bank.context_at(promotion.slot_index)
    target_recovery, recovery_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        target_context,
        n_back=3,
        steps=steps,
        seed=seed + 14000,
        cue_symbol=7,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_bits += recovery_bits
    target_recovery_result = _route_rollout(router, target_recovery, adapt=False)
    trained_target_error = ExternalModelBasedPlanner(
        bank,
        beam_width=4,
    ).rollout_error(
        target_recovery,
        transition_context=target_context.unsqueeze(0),
    )
    fresh_bank = ExternalTransitionModelBank(
        state_width=bank.state_width,
        intention_width=bank.intention_width,
        context_width=bank.context_width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        affine_ridge=affine_ridge,
    )
    fresh_bank.ensure_context(target_context)
    fresh_target_error = ExternalModelBasedPlanner(
        fresh_bank,
        beam_width=4,
    ).rollout_error(
        target_recovery,
        transition_context=target_context.unsqueeze(0),
    )
    unchanged = controller_before == _controller_digest(agent)
    source_stable = source_digest == bank.models[source_index].digest()
    discovered = promotion.accepted and promotion.slot_id == 1
    recovered = (
        target_recovery_result.stable_slot_id == promotion.slot_id
    )
    improved = trained_target_error < fresh_target_error
    passed = (
        unchanged
        and source_stable
        and discovered
        and recovered
        and improved
    )
    return OnlineTransitionDiscoveryReport(
        schema=(
            "neural-computer.brainworkshop-online-transition-discovery-audit.v3"
        ),
        status=(
            "online_replay_free_transition_discovery_boundary"
            if passed
            else "online_replay_free_transition_discovery_boundary_failed"
        ),
        controller_unchanged=unchanged,
        replay_free_bank=bank.replay_free_updates,
        source_slot_byte_stable=source_stable,
        target_context_discovered=discovered,
        target_route_recovered=recovered,
        target_model_improved_on_heldout=improved,
        source_heldout_error_before_target=source_error_before,
        source_heldout_error_after_target=source_error_after,
        trained_target_heldout_error=trained_target_error,
        fresh_target_heldout_error=fresh_target_error,
        source_training_lifetimes=source_training_lifetimes,
        target_training_lifetimes=target_training_lifetimes,
        total_logical_lifetimes=(
            source_training_lifetimes
            + target_training_lifetimes
            + 3
        ),
        unique_verifier_bits=unique_bits,
        transition_rows_consumed_once=(
            source_training_lifetimes * steps + target_training_lifetimes * steps
        ),
        replayed_examples=0,
        external_slot_count=bank.context_count,
        source_sample_count=int(bank.models[source_index].sample_count),
        target_sample_count=int(bank.models[promotion.slot_index].sample_count),
        target_promotion_accepted=promotion.accepted,
        target_promotion_reason=promotion.reason,
        target_discovery_status=discovery.status,
        target_continuation_status=target_result.status,
        target_heldout_status=target_recovery_result.status,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        choices=("single", "nonstationary", "online-discovery"),
        default="single",
    )
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--training-lifetimes", type=int, default=3)
    parser.add_argument("--source-training-lifetimes", type=int, default=2)
    parser.add_argument("--target-training-lifetimes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.audit == "nonstationary":
        report = run_nonstationary_transition_retention_audit(
            seed=args.seed,
            source_training_lifetimes=args.source_training_lifetimes,
            target_training_lifetimes=args.target_training_lifetimes,
            steps=args.steps,
        )
    elif args.audit == "online-discovery":
        report = run_online_transition_discovery_audit(
            seed=args.seed,
            source_training_lifetimes=args.source_training_lifetimes,
            target_training_lifetimes=args.target_training_lifetimes,
            steps=args.steps,
        )
    else:
        report = run_replay_free_transition_acquisition_audit(
            seed=args.seed,
            training_lifetimes=args.training_lifetimes,
            steps=args.steps,
        )
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report.payload(), indent=2) + "\n")
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    main()
