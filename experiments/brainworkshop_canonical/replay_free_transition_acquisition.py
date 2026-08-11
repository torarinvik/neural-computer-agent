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
    EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
    EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ContentAddressedMemory,
    ControllerFeedback,
    ExternalControllerEventWindowStateAdapter,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    ExternalOnlineTransitionContextResult,
    ExternalOnlineTransitionContextRouter,
    ExternalRoutedIntentionCostLedger,
    ExternalTransitionContextAddressAdapter,
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
    promotion_heldout_lifetimes: int
    state_adapter_schema: str
    state_width: int
    window_statistics: str
    window_gain: float
    recency_decay: float
    context_aggregation: str
    routing_match_tolerance: float = 0.01
    adaptive_address: bool = False
    random_feature_width: int = 128
    pretrain_context_encoder: bool = False
    context_encoder_optimizer_updates: int = 0
    context_encoder_rows_consumed_once: int = 0
    external_memory_update_mode: str = "sufficient_statistics"
    external_memory_optimizer_updates: int = 0
    admission_observations: int = 0
    discovery_probe_mode: str = "none"
    discovery_probe_rows: int = 0
    target_n_back: int = 3
    target_cue_symbol: int = 7
    goal_conditioned: bool = False
    target_goal_fragment_admitted: bool = False
    target_goal_fragment_used: bool = False
    target_goal_planner_improved_over_fresh: bool = False
    trained_target_goal_error: float = float("inf")
    fresh_target_goal_error: float = float("inf")
    target_goal_horizon: int = 0
    target_goal_missing_evidence_rejected: bool = False
    prior_selection_cost_aware: bool = False
    prior_selection_cost_ledger_used: bool = False
    prior_selection_cost_observed: bool = False

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
    time_shuffle: bool = False,
    intention_repertoire: ExternalIntentionRepertoire | None = None,
) -> tuple[ExternalTransitionRollout, int]:
    """Run one fresh rendered lifetime and optionally consume its rows once."""

    verifier = NBackVerifier(
        batch_size=1,
        n_back=n_back,
        steps=steps,
        symbol_count=4,
        cue_symbol=cue_symbol,
        seed=seed,
        time_shuffle=time_shuffle,
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
        if intention_repertoire is not None:
            intention_repertoire.observe(
                output.intention.payload.detach(),
                utility=scored.reward.detach(),
                propensity=decision.propensity.detach(),
                timestamp=verifier.position,
                outcome_mask=scored.eligible.detach(),
            )
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


def _run_active_discovery_lifetime(
    agent: CanonicalBrainWorkshopAgent,
    policy_free: PolicyFreeAmodalRuntime,
    probe_bank: ExternalTransitionModelBank,
    source_context: torch.Tensor,
    *,
    source_slot_id: int,
    target_slot_id: int,
    n_back: int,
    steps: int,
    seed: int,
    cue_symbol: int,
    candidate_intentions: torch.Tensor,
) -> tuple[ExternalTransitionRollout, int]:
    """Run one fresh lifetime with read-only disagreement-selected intentions."""

    if source_slot_id == target_slot_id:
        raise ValueError("active discovery probe requires distinct model slots")
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
    goal_state = torch.zeros(1, probe_bank.state_width, device=verifier.device)
    previous = None
    outputs = []
    unique_verifier_bits = 0
    probe_planner = ExternalModelBasedPlanner(probe_bank, beam_width=1)
    while not verifier.done:
        events = agent.runtime.encode_streams({"stimulus": verifier.observation()})
        current_candidates = candidate_intentions
        if previous is not None:
            probe = probe_planner.select_disambiguating_intention(
                probe_bank,
                previous.state,
                candidate_intentions,
                candidate_slot_ids=(source_slot_id, target_slot_id),
            )
            current_candidates = probe.selected_intention.unsqueeze(0)
        output, next_state = policy_free.step_events(
            events,
            state,
            feedback,
            goal_state,
            current_candidates,
            horizon=1,
            beam_width=1,
            transition_context=source_context.unsqueeze(0),
        )
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
        raise RuntimeError("active transition audit lifetime needs one transition")
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
            state_width=probe_bank.state_width,
            intention_width=probe_bank.intention_width,
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
    adapt_committed: bool = True,
    preferred_slot_id: int | None = None,
    preferred_continuation_tolerance: float | None = None,
    optimizer_update_counter: list[int] | None = None,
) -> ExternalOnlineTransitionContextResult:
    """Route a rollout with separately controlled provisional/committed writes.

    Discovery can consume staged candidate evidence while keeping committed
    slots read-only. This prevents a temporarily ambiguous novel stream from
    rewriting a mastered capability before the candidate passes promotion.
    """

    result = None
    for observation in _rollout_observations(rollout):
        observe_kwargs = {}
        if preferred_slot_id is not None:
            observe_kwargs["preferred_slot_id"] = preferred_slot_id
        if preferred_continuation_tolerance is not None:
            observe_kwargs["preferred_continuation_tolerance"] = (
                preferred_continuation_tolerance
            )
        result = router.observe(observation, **observe_kwargs)
        if adapt and (
            result.status == "staged"
            or (
                adapt_committed
                and result.status
                in {"admitted", "reused", "matched", "continuation", "sparse_matched"}
            )
        ):
            router.adaptation_step(result, None, replay_evidence=False)
            if (
                optimizer_update_counter is not None
                and result.status == "staged"
                and router.provisional_evidence_policy == "streaming_gradient"
            ):
                optimizer_update_counter[0] += 1
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


def _context_view(
    observation: ExternalTransitionObservation,
    scale: float = 0.01,
) -> ExternalTransitionObservation:
    """Create one deterministic front-end perturbation without storing rows."""

    return ExternalTransitionObservation(
        state=observation.state + scale * torch.tanh(observation.state),
        intention=observation.intention,
        next_state=observation.next_state + scale * torch.tanh(observation.next_state),
        confidence=observation.confidence,
    ).validate(
        state_width=observation.state.shape[1],
        intention_width=observation.intention.shape[1],
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
    window_gain: float = 0.15,
    window_statistics: str = "masked_mean_and_max_v1",
    recency_decay: float = 0.75,
    promotion_heldout_lifetimes: int = 3,
    context_aggregation: str = "last_token",
    goal_conditioned: bool = False,
    goal_horizon: int = 2,
    goal_verifier_threshold: float = 0.05,
    prior_selection_transfer_cost: float = 0.0,
    prior_selection_fresh_cost: float = 0.0,
    prior_selection_cost_weight: float = 0.0,
    learned_prior_selection_cost: bool = False,
    prior_selection_cost_learning_rate: float = 0.35,
    prior_selection_cost_initial: float = 0.25,
    prior_selection_cost_decision_weight: float = 1.0,
    routing_match_tolerance: float = 0.01,
    adaptive_address: bool = False,
    random_feature_width: int = 128,
    pretrain_context_encoder: bool = False,
    external_memory_update_mode: str = "sufficient_statistics",
    admission_observations: int | None = None,
    discovery_probe_mode: str = "none",
    target_n_back: int = 3,
    target_cue_symbol: int = 7,
) -> OnlineTransitionDiscoveryReport:
    """Discover and learn a novel rendered family without replay or a task label.

    The target lifetime initially plans through the already-known source slot.
    Only its opaque transition observations reach the online router.  The
    router stages a new external slot, consumes each training bundle once
    through either replay-free sufficient statistics or a caller-selected
    one-pass gradient memory, and commits the discovered context only after
    multiple independent held-out lifetimes, recursive prediction, and
    source-retention probes. Cue symbols and n-back values remain
    verifier-private diagnostics and never enter the router. The gradient arm
    is an explicit external-memory control; it is not counted as a
    sufficient-statistics or replay-free-bank result.

    With ``goal_conditioned=True``, the promoted target slot must additionally
    admit and use an opaque multi-step goal fragment against a matched fresh
    planner. The goal gate is opt-in so the historical transition-only report
    remains backward-compatible.
    """

    if min(steps, source_training_lifetimes, target_training_lifetimes) < 1:
        raise ValueError("online transition audit budgets must be positive")
    if target_training_lifetimes < 2:
        raise ValueError("online transition discovery needs a continuation lifetime")
    if promotion_heldout_lifetimes < 2:
        raise ValueError("online transition promotion needs multiple held-out lifetimes")
    if context_aggregation not in {"last_token", "mean_pool"}:
        raise ValueError("online transition context aggregation is unsupported")
    if window_statistics not in {
        "masked_mean_and_max_v1",
        "recency_weighted_and_latest_v1",
        "ordered_payload_and_presence_v1",
    }:
        raise ValueError("online transition window statistics are unsupported")
    if recency_decay <= 0.0 or not math.isfinite(float(recency_decay)):
        raise ValueError("online transition recency decay must be finite and positive")
    if affine_ridge <= 0.0 or not math.isfinite(float(affine_ridge)):
        raise ValueError("online transition affine ridge must be finite and positive")
    if routing_match_tolerance <= 0.0 or not math.isfinite(
        float(routing_match_tolerance)
    ):
        raise ValueError("online routing match tolerance must be finite and positive")
    if not isinstance(goal_conditioned, bool):
        raise TypeError("online goal-conditioned flag must be boolean")
    if not isinstance(learned_prior_selection_cost, bool):
        raise TypeError("learned prior-selection cost flag must be boolean")
    if not isinstance(adaptive_address, bool):
        raise TypeError("adaptive address flag must be boolean")
    if (
        not isinstance(random_feature_width, int)
        or isinstance(random_feature_width, bool)
        or random_feature_width < 1
    ):
        raise ValueError("random feature width must be a positive integer")
    if not isinstance(pretrain_context_encoder, bool):
        raise TypeError("context encoder pretraining flag must be boolean")
    if discovery_probe_mode not in {"none", "active", "passive"}:
        raise ValueError("unsupported discovery probe mode")
    if (
        not isinstance(target_n_back, int)
        or isinstance(target_n_back, bool)
        or target_n_back < 1
    ):
        raise ValueError("target n-back must be a positive integer")
    if (
        not isinstance(target_cue_symbol, int)
        or isinstance(target_cue_symbol, bool)
        or not 0 <= target_cue_symbol < 8
    ):
        raise ValueError("target cue symbol must lie in the agent symbol range")
    if external_memory_update_mode not in {
        "sufficient_statistics",
        "streaming_gradient",
    }:
        raise ValueError("unsupported external-memory update mode")
    selected_admission_observations = (
        steps if admission_observations is None else admission_observations
    )
    if (
        not isinstance(selected_admission_observations, int)
        or isinstance(selected_admission_observations, bool)
        or selected_admission_observations < 1
    ):
        raise ValueError("online admission observations must be positive")
    if goal_horizon < 1 or goal_horizon > steps - 1:
        raise ValueError("online goal horizon must fit held-out transitions")
    if goal_verifier_threshold <= 0.0 or not math.isfinite(
        float(goal_verifier_threshold)
    ):
        raise ValueError("online goal verifier threshold must be finite and positive")
    for name, value in (
        ("prior_selection_transfer_cost", prior_selection_transfer_cost),
        ("prior_selection_fresh_cost", prior_selection_fresh_cost),
        ("prior_selection_cost_weight", prior_selection_cost_weight),
    ):
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    for name, value in (
        ("prior_selection_cost_learning_rate", prior_selection_cost_learning_rate),
        ("prior_selection_cost_initial", prior_selection_cost_initial),
        ("prior_selection_cost_decision_weight", prior_selection_cost_decision_weight),
    ):
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if learned_prior_selection_cost and any(
        float(value) != 0.0
        for value in (prior_selection_transfer_cost, prior_selection_fresh_cost)
    ):
        raise ValueError("learned prior-selection cost cannot mix with static costs")
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
    ordered_window = window_statistics == "ordered_payload_and_presence_v1"
    state_adapter = ExternalControllerEventWindowStateAdapter(
        agent.controller.width,
        state_width=None if ordered_window else agent.controller.width * 3,
        window_gain=window_gain,
        window_statistics=window_statistics,
        recency_decay=recency_decay,
        window_capacity=(
            agent.controller.event_window_capacity if ordered_window else None
        ),
    )
    bank = ExternalTransitionModelBank(
        state_width=state_adapter.state_width,
        intention_width=agent.controller.intention_width,
        context_width=agent.controller.width,
        model_family=EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
        affine_ridge=affine_ridge,
        random_feature_width=random_feature_width,
    )
    if external_memory_update_mode == "streaming_gradient":
        candidate_model_families = (EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,)
        provisional_evidence_policy = "streaming_gradient"
    else:
        candidate_model_families = (
            EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        )
        provisional_evidence_policy = "streaming_statistics"
    source_context = _opaque_context(agent, 6)
    # The source is a known replay-free baseline. Keep it explicitly affine;
    # the target arm either selects among replay-free sufficient-statistics
    # families or exercises the isolated streaming-gradient control.
    source_index = bank.ensure_context(
        source_context,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    policy_free = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(bank, beam_width=4),
        state_adapter=state_adapter,
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
        aggregation=context_aggregation,
    )
    context_encoder_optimizer_updates = 0
    context_encoder_rows_consumed_once = 0
    if pretrain_context_encoder:
        source_context_rollout, source_context_bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            source_context,
            n_back=2,
            steps=steps,
            seed=seed + 13000,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        auxiliary_context_rollout, auxiliary_context_bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            source_context,
            n_back=1,
            steps=steps,
            seed=seed + 14000,
            cue_symbol=5,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        unique_bits += source_context_bits + auxiliary_context_bits
        left_views = [
            _rollout_bundle(source_context_rollout),
            _rollout_bundle(auxiliary_context_rollout),
        ]
        right_views = [_context_view(view) for view in left_views]
        context_encoder, _context_loss = context_encoder.copy_on_write_contrastive_step(
            left_views,
            right_views,
        )
        context_encoder_optimizer_updates = 1
        context_encoder_rows_consumed_once = sum(
            int(view.state.shape[0]) for view in left_views
        )
    prior_cost_ledger = (
        ExternalRoutedIntentionCostLedger.create(
            bank.context_width,
            learning_rate=prior_selection_cost_learning_rate,
            initial_cost=prior_selection_cost_initial,
            decision_weight=prior_selection_cost_decision_weight,
        )
        if learned_prior_selection_cost
        else None
    )
    address_adapter = (
        ExternalTransitionContextAddressAdapter(
            context_encoder,
            learning_rate=1e-3,
            adaptation_steps=8,
            anchor_cosine_ceiling=0.75,
        )
        if adaptive_address
        else None
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        context_encoder,
        admission_observations=selected_admission_observations,
        match_tolerance=routing_match_tolerance,
        match_margin=0.0,
        continuation_tolerance=routing_match_tolerance,
        provisional_continuation_tolerance=0.05,
        defer_admission=True,
        max_contexts=2,
        candidate_model_families=(
            *candidate_model_families,
        ),
        provisional_evidence_policy=provisional_evidence_policy,
        provisional_context_similarity_threshold=0.98,
        provisional_context_similarity_margin=0.005,
        provisional_context_error_tolerance=0.15,
        prior_selection_probe=(
            lambda transfer, fresh, observation: (
                float(transfer.loss(observation).detach()),
                float(fresh.loss(observation).detach()),
            )
        ),
        prior_selection_transfer_cost=prior_selection_transfer_cost,
        prior_selection_fresh_cost=prior_selection_fresh_cost,
        prior_selection_cost_weight=prior_selection_cost_weight,
        prior_selection_cost_ledger=prior_cost_ledger,
        address_adapter=address_adapter,
    )

    # Acquire the target while still behaving through the known source slot.
    # The router sees only the resulting opaque transition rows and keeps the
    # candidate outside the committed bank until the held-out gate passes.
    discovery = None
    target_result = None
    target_candidate_staged = False
    prior_selection_cost_aware = False
    external_memory_optimizer_updates = [0]
    active_probe_rows = 0
    for lifetime in range(target_training_lifetimes):
        target_rollout, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            source_context,
            n_back=target_n_back,
            steps=steps,
            seed=seed + 2000 + lifetime,
            cue_symbol=target_cue_symbol,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        unique_bits += bits
        routed = _route_rollout(
            router,
            target_rollout,
            adapt=True,
            adapt_committed=False,
            optimizer_update_counter=external_memory_optimizer_updates,
        )
        if discovery is None:
            discovery = routed
        target_candidate_staged = target_candidate_staged or routed.status == "staged"
        if router.provisional_candidate_count:
            prior_receipt = router.provisional_prior_selection_at(0)
            prior_selection_cost_aware = prior_selection_cost_aware or bool(
                prior_receipt is not None
                and prior_receipt.schema.endswith("prior-selection.v2")
            )
        target_result = routed
    if (
        discovery_probe_mode != "none"
        and target_candidate_staged
        and router.provisional_candidate_count
    ):
        candidate_context = router.provisional_context_at(0)
        active_probe_bank = ExternalTransitionModelBank.from_payload(bank.payload())
        active_probe_index = active_probe_bank.ensure_context(
            candidate_context,
            model_family=candidate_model_families[0],
        )
        active_probe_bank.models[active_probe_index].load_state_dict(
            router.provisional_model_at(0).state_dict()
        )
        if discovery_probe_mode == "active":
            active_probe_rollout, active_probe_bits = (
                _run_active_discovery_lifetime(
                    agent,
                    policy_free,
                    active_probe_bank,
                    source_context,
                    source_slot_id=active_probe_bank.slot_id_at(source_index),
                    target_slot_id=active_probe_bank.slot_id_at(active_probe_index),
                    n_back=target_n_back,
                    steps=steps,
                    seed=seed + 22000,
                    cue_symbol=target_cue_symbol,
                    candidate_intentions=candidate_intentions,
                )
            )
        else:
            active_probe_rollout, active_probe_bits = _run_lifetime(
                agent,
                policy_free,
                bank,
                source_context,
                n_back=target_n_back,
                steps=steps,
                seed=seed + 22000,
                cue_symbol=target_cue_symbol,
                candidate_intentions=candidate_intentions,
                learn=False,
            )
        unique_bits += active_probe_bits
        active_probe_rows = active_probe_rollout.horizon
        target_result = _route_rollout(
            router,
            active_probe_rollout,
            adapt=True,
            adapt_committed=False,
            optimizer_update_counter=external_memory_optimizer_updates,
        )
    source_error_after = planner.rollout_error(
        source_heldout,
        transition_context=source_context.unsqueeze(0),
    )
    discovery_status = (
        "staged"
        if target_candidate_staged
        else "not_staged"
        if discovery is None
        else discovery.status
    )
    if (
        discovery is None
        or target_result is None
        or not target_candidate_staged
        or router.provisional_candidate_count != 1
    ):
        unchanged = controller_before == _controller_digest(agent)
        source_stable = source_digest == bank.models[source_index].digest()
        return OnlineTransitionDiscoveryReport(
            schema=(
                "neural-computer.brainworkshop-online-transition-discovery-audit.v6"
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
                source_training_lifetimes
                + target_training_lifetimes
                + 1
                + (1 if active_probe_rows else 0)
            ),
            unique_verifier_bits=unique_bits,
            transition_rows_consumed_once=(
                source_training_lifetimes * steps
                + target_training_lifetimes * steps
                + active_probe_rows
            ),
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
            promotion_heldout_lifetimes=0,
            state_adapter_schema=state_adapter.schema,
            state_width=state_adapter.state_width,
            window_statistics=window_statistics,
            window_gain=window_gain,
            recency_decay=recency_decay,
            context_aggregation=context_aggregation,
            routing_match_tolerance=routing_match_tolerance,
            adaptive_address=adaptive_address,
            random_feature_width=random_feature_width,
            pretrain_context_encoder=pretrain_context_encoder,
            goal_conditioned=goal_conditioned,
            target_goal_horizon=goal_horizon if goal_conditioned else 0,
            prior_selection_cost_aware=prior_selection_cost_aware,
            prior_selection_cost_ledger_used=prior_cost_ledger is not None,
            context_encoder_optimizer_updates=context_encoder_optimizer_updates,
            context_encoder_rows_consumed_once=context_encoder_rows_consumed_once,
            external_memory_update_mode=external_memory_update_mode,
            external_memory_optimizer_updates=external_memory_optimizer_updates[0],
            admission_observations=selected_admission_observations,
            discovery_probe_mode=discovery_probe_mode,
            discovery_probe_rows=active_probe_rows,
            target_n_back=target_n_back,
            target_cue_symbol=target_cue_symbol,
        )

    candidate_context = router.provisional_context_at(0)
    # Build an isolated copy-on-write bank from the staged opaque model. The
    # promotion holdouts must exercise the candidate route itself; otherwise
    # a source-slot rollout can make a weak target candidate appear valid.
    shadow_bank = ExternalTransitionModelBank.from_payload(bank.payload())
    shadow_index = shadow_bank.ensure_context(
        candidate_context,
        model_family=candidate_model_families[0],
    )
    shadow_bank.models[shadow_index].load_state_dict(
        router.provisional_model_at(0).state_dict()
    )
    shadow_runtime = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(shadow_bank, beam_width=4),
        state_adapter=state_adapter,
    )
    promotion_holdouts: list[ExternalTransitionRollout] = []
    for holdout_index in range(promotion_heldout_lifetimes):
        holdout, holdout_bits = _run_lifetime(
            agent,
            shadow_runtime,
            shadow_bank,
            candidate_context,
            n_back=target_n_back,
            steps=steps,
            seed=seed + 12000 + holdout_index * 2000,
            cue_symbol=target_cue_symbol,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        promotion_holdouts.append(holdout)
        unique_bits += holdout_bits
    promotion_holdout = promotion_holdouts[0]
    promotion_route = _route_rollout(router, promotion_holdout, adapt=False)
    promotion_observations = [
        _rollout_bundle(holdout) for holdout in promotion_holdouts
    ]

    def retention_probe(candidate_bank: ExternalTransitionModelBank) -> bool:
        if candidate_bank.models[source_index].digest() != source_digest:
            return False
        candidate_planner = ExternalModelBasedPlanner(candidate_bank, beam_width=4)
        # A copy-on-write address adapter may refine the staged key while the
        # promotion holdout is routed. Resolve the candidate's final opaque
        # key from the candidate bank rather than querying a stale pre-holdout
        # snapshot.
        candidate_route_context = candidate_bank.context_at(
            candidate_bank.context_count - 1
        )
        fresh_probe_bank = ExternalTransitionModelBank(
            state_width=bank.state_width,
            intention_width=bank.intention_width,
            context_width=bank.context_width,
            model_family=EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
            affine_ridge=affine_ridge,
            random_feature_width=random_feature_width,
        )
        fresh_probe_bank.ensure_context(
            candidate_route_context,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        )
        fresh_planner = ExternalModelBasedPlanner(fresh_probe_bank, beam_width=4)
        candidate_errors: list[float] = []
        fresh_errors: list[float] = []
        for heldout in promotion_holdouts:
            candidate_error = candidate_planner.rollout_error(
                heldout,
                transition_context=candidate_route_context.unsqueeze(0),
            )
            fresh_error = fresh_planner.rollout_error(
                heldout,
                transition_context=candidate_route_context.unsqueeze(0),
            )
            if candidate_error > 0.2:
                return False
            candidate_errors.append(candidate_error)
            fresh_errors.append(fresh_error)
        if not candidate_errors:
            return False
        wins = sum(
            candidate_error < fresh_error
            for candidate_error, fresh_error in zip(
                candidate_errors,
                fresh_errors,
                strict=True,
            )
        )
        return (
            wins >= (len(candidate_errors) + 1) // 2
            and math.fsum(candidate_errors) / len(candidate_errors)
            < math.fsum(fresh_errors) / len(fresh_errors)
        )

    promotion = router.promote_staged_candidate(
        promotion_observations[0],
        retention_probe,
        prediction_tolerance=0.05,
        heldout_rollout=promotion_holdout,
        rollout_error_tolerance=0.2,
        additional_heldout_observations=tuple(promotion_observations[1:]),
        additional_heldout_rollouts=tuple(promotion_holdouts[1:]),
        prior_selection_observed_cost=(
            min(
                1.0,
                float(len(promotion_holdouts) * steps)
                / float(max(1, unique_bits)),
            )
            if prior_cost_ledger is not None
            else None
        ),
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
                "neural-computer.brainworkshop-online-transition-discovery-audit.v6"
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
                source_training_lifetimes
                + target_training_lifetimes
                + promotion_heldout_lifetimes
                + 1
                + (1 if active_probe_rows else 0)
            ),
            unique_verifier_bits=unique_bits,
            transition_rows_consumed_once=(
                source_training_lifetimes * steps
                + target_training_lifetimes * steps
                + active_probe_rows
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
            promotion_heldout_lifetimes=promotion_heldout_lifetimes,
            state_adapter_schema=state_adapter.schema,
            state_width=state_adapter.state_width,
            window_statistics=window_statistics,
            window_gain=window_gain,
            recency_decay=recency_decay,
            context_aggregation=context_aggregation,
            routing_match_tolerance=routing_match_tolerance,
            adaptive_address=adaptive_address,
            random_feature_width=random_feature_width,
            pretrain_context_encoder=pretrain_context_encoder,
            goal_conditioned=goal_conditioned,
            target_goal_horizon=goal_horizon if goal_conditioned else 0,
            prior_selection_cost_aware=prior_selection_cost_aware,
            prior_selection_cost_ledger_used=prior_cost_ledger is not None,
            context_encoder_optimizer_updates=context_encoder_optimizer_updates,
            context_encoder_rows_consumed_once=context_encoder_rows_consumed_once,
            external_memory_update_mode=external_memory_update_mode,
            external_memory_optimizer_updates=external_memory_optimizer_updates[0],
            admission_observations=selected_admission_observations,
            discovery_probe_mode=discovery_probe_mode,
            discovery_probe_rows=active_probe_rows,
            target_n_back=target_n_back,
            target_cue_symbol=target_cue_symbol,
        )

    target_context = bank.context_at(promotion.slot_index)
    target_recovery, recovery_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        target_context,
        n_back=target_n_back,
        steps=steps,
        seed=seed + 12000 + promotion_heldout_lifetimes * 2000 + 1000,
        cue_symbol=target_cue_symbol,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_bits += recovery_bits
    target_recovery_result = _route_rollout(
        router,
        target_recovery,
        adapt=False,
        preferred_slot_id=promotion.slot_id,
        preferred_continuation_tolerance=0.15,
    )
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
        model_family=EXTERNAL_TRANSITION_MIXED_MODEL_FAMILY,
        affine_ridge=affine_ridge,
        random_feature_width=random_feature_width,
    )
    fresh_bank.ensure_context(
        target_context,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    fresh_planner = ExternalModelBasedPlanner(
        fresh_bank,
        beam_width=4,
    )
    fresh_target_error = fresh_planner.rollout_error(
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
    goal_fragment_admitted = False
    goal_fragment_used = False
    goal_improved = False
    trained_goal_error = float("inf")
    fresh_goal_error = float("inf")
    goal_missing_rejected = False
    if goal_conditioned:
        # Import lazily: the standalone goal audit reuses this module's
        # lifetime helper, so a top-level import would create a cycle.
        from neural_computer import (
            ExternalGoalFragmentCandidate,
            ExternalGoalFragmentMemory,
            ExternalGoalFragmentStager,
        )

        from .goal_conditioned_planning import (
            _exact_goal_retention_probe,
            _plan_error,
        )

        candidate = ExternalGoalFragmentCandidate.from_state(
            target_recovery.expected_states[goal_horizon - 1].detach()
        )
        candidate_digest = candidate.digest(state_width=bank.state_width)
        probe_memory = ExternalGoalFragmentMemory(bank.state_width)
        values, masks = candidate.tensors(state_width=bank.state_width)
        probe_memory.append(values, masks)
        goal_probe_error, goal_used = _plan_error(
            planner,
            initial_state=target_recovery.initial_state,
            goal_memory=probe_memory,
            candidate_intentions=candidate_intentions,
            context=target_context,
            horizon=goal_horizon,
            fragment_id=0,
        )
        stager = ExternalGoalFragmentStager(
            bank.state_width,
            threshold=1.0,
            min_observations=1,
            min_stable_observations=1,
        )
        stager.observe(
            candidate,
            float(goal_probe_error <= goal_verifier_threshold),
        )
        goal_memory = ExternalGoalFragmentMemory(bank.state_width)
        admission = stager.admit_verified(
            goal_memory,
            candidate_digest,
            _exact_goal_retention_probe(candidate, bank.state_width),
        )
        goal_fragment_admitted = admission.accepted
        if admission.accepted and admission.fragment_id is not None:
            trained_goal_error, trained_goal_used = _plan_error(
                planner,
                initial_state=target_recovery.initial_state,
                goal_memory=goal_memory,
                candidate_intentions=candidate_intentions,
                context=target_context,
                horizon=goal_horizon,
                fragment_id=admission.fragment_id,
            )
            fresh_goal_error, fresh_goal_used = _plan_error(
                fresh_planner,
                initial_state=target_recovery.initial_state,
                goal_memory=goal_memory,
                candidate_intentions=candidate_intentions,
                context=target_context,
                horizon=goal_horizon,
                fragment_id=admission.fragment_id,
            )
            goal_fragment_used = bool(
                goal_used and trained_goal_used and fresh_goal_used
            )
            goal_improved = trained_goal_error < fresh_goal_error
        missing_stager = ExternalGoalFragmentStager(
            bank.state_width,
            threshold=1.0,
            min_observations=1,
            min_stable_observations=1,
        )
        missing_stager.observe(candidate, 0.0, eligible=False)
        missing_rejection = missing_stager.admit_verified(
            ExternalGoalFragmentMemory(bank.state_width),
            candidate_digest,
            _exact_goal_retention_probe(candidate, bank.state_width),
        )
        goal_missing_rejected = not missing_rejection.accepted
    passed = (
        unchanged
        and source_stable
        and discovered
        and recovered
        and improved
        and (
            not goal_conditioned
            or (
                goal_fragment_admitted
                and goal_fragment_used
                and goal_improved
                and goal_missing_rejected
            )
        )
        and (
            not any(
                float(value) != 0.0
                for value in (
                    prior_selection_transfer_cost,
                    prior_selection_fresh_cost,
                    prior_selection_cost_weight,
                )
            )
            or prior_selection_cost_aware
        )
        and (
            not learned_prior_selection_cost
            or promotion.prior_selection_cost_observation is not None
        )
    )
    return OnlineTransitionDiscoveryReport(
        schema=(
            "neural-computer.brainworkshop-online-transition-discovery-audit.v6"
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
            + promotion_heldout_lifetimes
            + 2
            + (1 if active_probe_rows else 0)
        ),
        unique_verifier_bits=unique_bits,
        transition_rows_consumed_once=(
            source_training_lifetimes * steps
            + target_training_lifetimes * steps
            + active_probe_rows
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
        promotion_heldout_lifetimes=promotion_heldout_lifetimes,
        state_adapter_schema=state_adapter.schema,
        state_width=state_adapter.state_width,
        window_statistics=window_statistics,
        window_gain=window_gain,
        recency_decay=recency_decay,
        context_aggregation=context_aggregation,
        routing_match_tolerance=routing_match_tolerance,
        adaptive_address=adaptive_address,
        random_feature_width=random_feature_width,
        pretrain_context_encoder=pretrain_context_encoder,
        goal_conditioned=goal_conditioned,
        target_goal_fragment_admitted=goal_fragment_admitted,
        target_goal_fragment_used=goal_fragment_used,
        target_goal_planner_improved_over_fresh=goal_improved,
        trained_target_goal_error=trained_goal_error,
        fresh_target_goal_error=fresh_goal_error,
        target_goal_horizon=goal_horizon if goal_conditioned else 0,
        target_goal_missing_evidence_rejected=goal_missing_rejected,
        prior_selection_cost_aware=prior_selection_cost_aware,
        prior_selection_cost_ledger_used=prior_cost_ledger is not None,
        prior_selection_cost_observed=(
            promotion.prior_selection_cost_observation is not None
        ),
        context_encoder_optimizer_updates=context_encoder_optimizer_updates,
        context_encoder_rows_consumed_once=context_encoder_rows_consumed_once,
        external_memory_update_mode=external_memory_update_mode,
        external_memory_optimizer_updates=external_memory_optimizer_updates[0],
        admission_observations=selected_admission_observations,
        discovery_probe_mode=discovery_probe_mode,
        discovery_probe_rows=active_probe_rows,
        target_n_back=target_n_back,
        target_cue_symbol=target_cue_symbol,
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
    parser.add_argument("--promotion-heldout-lifetimes", type=int, default=3)
    parser.add_argument(
        "--window-statistics",
        choices=(
            "masked_mean_and_max_v1",
            "recency_weighted_and_latest_v1",
            "ordered_payload_and_presence_v1",
        ),
        default="masked_mean_and_max_v1",
    )
    parser.add_argument("--window-gain", type=float, default=0.15)
    parser.add_argument("--recency-decay", type=float, default=0.75)
    parser.add_argument("--goal-conditioned", action="store_true")
    parser.add_argument("--goal-horizon", type=int, default=2)
    parser.add_argument("--goal-verifier-threshold", type=float, default=0.05)
    parser.add_argument("--prior-selection-transfer-cost", type=float, default=0.0)
    parser.add_argument("--prior-selection-fresh-cost", type=float, default=0.0)
    parser.add_argument("--prior-selection-cost-weight", type=float, default=0.0)
    parser.add_argument("--learned-prior-selection-cost", action="store_true")
    parser.add_argument("--prior-selection-cost-learning-rate", type=float, default=0.35)
    parser.add_argument("--prior-selection-cost-initial", type=float, default=0.25)
    parser.add_argument(
        "--prior-selection-cost-decision-weight", type=float, default=1.0
    )
    parser.add_argument("--adaptive-address", action="store_true")
    parser.add_argument("--routing-match-tolerance", type=float, default=0.01)
    parser.add_argument("--random-feature-width", type=int, default=128)
    parser.add_argument("--pretrain-context-encoder", action="store_true")
    parser.add_argument(
        "--external-memory-update-mode",
        choices=("sufficient_statistics", "streaming_gradient"),
        default="sufficient_statistics",
    )
    parser.add_argument("--admission-observations", type=int)
    parser.add_argument(
        "--discovery-probe-mode",
        choices=("none", "active", "passive"),
        default="none",
    )
    parser.add_argument("--target-n-back", type=int, default=3)
    parser.add_argument("--target-cue-symbol", type=int, default=7)
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
            promotion_heldout_lifetimes=args.promotion_heldout_lifetimes,
            window_statistics=args.window_statistics,
            window_gain=args.window_gain,
            recency_decay=args.recency_decay,
            goal_conditioned=args.goal_conditioned,
            goal_horizon=args.goal_horizon,
            goal_verifier_threshold=args.goal_verifier_threshold,
            prior_selection_transfer_cost=args.prior_selection_transfer_cost,
            prior_selection_fresh_cost=args.prior_selection_fresh_cost,
            prior_selection_cost_weight=args.prior_selection_cost_weight,
            learned_prior_selection_cost=args.learned_prior_selection_cost,
            prior_selection_cost_learning_rate=args.prior_selection_cost_learning_rate,
            prior_selection_cost_initial=args.prior_selection_cost_initial,
            prior_selection_cost_decision_weight=args.prior_selection_cost_decision_weight,
            adaptive_address=args.adaptive_address,
            routing_match_tolerance=args.routing_match_tolerance,
            random_feature_width=args.random_feature_width,
            pretrain_context_encoder=args.pretrain_context_encoder,
            external_memory_update_mode=args.external_memory_update_mode,
            admission_observations=args.admission_observations,
            discovery_probe_mode=args.discovery_probe_mode,
            target_n_back=args.target_n_back,
            target_cue_symbol=args.target_cue_symbol,
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
