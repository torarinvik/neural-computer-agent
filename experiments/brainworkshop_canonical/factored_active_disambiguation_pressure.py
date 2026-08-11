"""Fresh Brain Workshop pressure test for active factual disambiguation.

Two frozen-controller transition regimes are retained in external residual
memory.  A fresh verifier lifetime is deliberately stopped at an ambiguous
transition immediately before the first eligible n-back outcome.  The router
requests an opaque diagnostic intention, the caller decodes it through the
ordinary keypress output bus, and the resulting fresh successor is routed
without mutating the router.  A passive low-disagreement intention is measured
on a separate lifetime as the control.

This is a mechanistic boundary audit, not a claim that the frozen random
keypress decoder has learned n-back.  Its claim is narrower: learned factual
memory can request an opaque active probe, the caller can execute it through a
replaceable decoder, and fresh verifier evidence can resolve an ambiguity
without a controller or router write during the probe.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ContentAddressedMemory,
    ControllerFeedback,
    ExternalAffineTransitionStatistics,
    ExternalControllerEventWindowStateAdapter,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    ExternalTransitionContextEncoder,
    ExternalTransitionObservation,
    ExternalTransitionProbeUtilityMemory,
    ExternalTransitionRollout,
    ExternalTransitionSupportStatistics,
    PolicyFreeAmodalRuntime,
)

from .environment import NBackVerifier
from .replay_free_transition_acquisition import (
    _controller_digest,
    _rollout_bundle,
    _rollout_observations,
    _run_lifetime,
)
from .runner import CanonicalBrainWorkshopAgent

ACTIVE_DISAMBIGUATION_PRESSURE_SCHEMA = (
    "neural-computer.brainworkshop-factored-active-disambiguation-pressure.v1"
)


@dataclass(frozen=True)
class ActiveDisambiguationTrial:
    kind: str
    verifier_seed: int
    selected_intention_index: int
    maximum_disagreement: float
    selected_disagreement: float
    minimum_disagreement: float
    verifier_reward: float
    verifier_outcome_eligible: bool
    probe_model_errors: tuple[float, ...]
    strict_route_status: str
    strict_route_slot_id: int | None
    router_read_only: bool
    decoder_state_free: bool
    probe_horizon: int = 1
    selected_intention_indices: tuple[int, ...] = ()
    probe_model_errors_by_step: tuple[tuple[float, ...], ...] = ()
    selected_probe_leverage: tuple[float, ...] = ()
    selected_probe_support: tuple[float, ...] = ()
    selected_probe_utility: tuple[float, ...] = ()

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveDisambiguationPressureResult:
    seed: int
    status: str
    controller_unchanged: bool
    active_probe_recovered_target: bool
    passive_control_recovered_target: bool
    active_probe_read_only: bool
    active_decoder_state_free: bool
    source_slot_id: int | None
    target_slot_id: int | None
    ambiguous_transition_observed: bool
    active_trial: ActiveDisambiguationTrial | None
    passive_trial: ActiveDisambiguationTrial | None
    unique_verifier_bits: int
    logical_lifetimes: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int
    wall_time_seconds: float
    schema: str = ACTIVE_DISAMBIGUATION_PRESSURE_SCHEMA
    probe_horizon: int = 1
    candidate_intention_source: str = "seed_pool"
    support_calibration_rows: int = 0
    utility_calibration_observations: int = 0

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _policy(
    agent: CanonicalBrainWorkshopAgent,
    model: ExternalFactoredTransitionModel,
    state_adapter: ExternalControllerEventWindowStateAdapter,
) -> PolicyFreeAmodalRuntime:
    return PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(model, beam_width=4),
        state_adapter=state_adapter,
    )


def _feedback(
    agent: CanonicalBrainWorkshopAgent,
    scored: object,
    decision: object,
) -> ControllerFeedback:
    return ControllerFeedback(
        action=agent.keypress_encoder(decision.key_index),
        reward=scored.reward,
        propensity=decision.propensity,
        has_feedback=scored.eligible.to(scored.reward.dtype),
    )


def _diagnostic_probe_utility(trial: ActiveDisambiguationTrial) -> float:
    """Convert a route verifier result into a scalar resolution quality."""

    if trial.strict_route_status != "matched":
        return 0.0
    errors = sorted(float(error) for error in trial.probe_model_errors)
    if len(errors) < 2:
        return 1.0
    gap = max(0.0, errors[1] - errors[0])
    return gap / (gap + errors[0] + 1e-6)


@torch.no_grad()
def _observe_probe_support(
    router: ExternalFactoredTransitionRouter,
    rollout: ExternalTransitionRollout,
    support_statistics: ExternalTransitionSupportStatistics,
    *,
    candidate_slot_ids: tuple[int, ...],
    support_tolerance: float,
) -> int:
    """Calibrate leverage from one fresh rollout without retaining its rows."""

    if support_tolerance < 0.0:
        raise ValueError("probe support tolerance cannot be negative")
    if router.model.residual_bank is None:
        raise RuntimeError("probe support calibration requires residual statistics")
    contexts = torch.stack(
        [
            router._contexts[router._slot_ids.index(slot_id)]
            for slot_id in candidate_slot_ids
        ]
    )
    rows = 0
    for observation in _rollout_observations(rollout):
        leverage = router.model.residual_bank.predictive_leverage(
            observation.state,
            observation.intention,
            candidate_contexts=contexts,
        )
        errors = []
        for context in contexts:
            prediction = router.model.predict_with_context(
                observation.state,
                observation.intention,
                context.to(observation.state)
                .unsqueeze(0)
                .expand(observation.state.shape[0], -1),
            )
            errors.append((prediction - observation.next_state).square().mean(dim=-1))
        supported = (torch.stack(errors, dim=1) <= support_tolerance).to(
            leverage.dtype
        )
        support_statistics.observe(
            leverage,
            supported,
            slot_ids=candidate_slot_ids,
        )
        rows += observation.state.shape[0]
    return rows


def _execute_probe_trial(
    *,
    agent: CanonicalBrainWorkshopAgent,
    policy: PolicyFreeAmodalRuntime,
    router: ExternalFactoredTransitionRouter,
    verifier_seed: int,
    source_context: torch.Tensor,
    candidate_intentions: torch.Tensor,
    target_slot_id: int,
    probe_transition_index: int,
    active: bool,
    ambiguity_tolerance: float,
    probe_match_tolerance: float,
    probe_contradiction_tolerance: float,
    probe_horizon: int = 1,
    support_statistics: ExternalTransitionSupportStatistics | None = None,
    utility_memory: ExternalTransitionProbeUtilityMemory | None = None,
    forced_intention_index: int | None = None,
) -> tuple[ActiveDisambiguationTrial, bool, int, int]:
    """Run one fresh verifier until an opaque active/passive probe executes."""

    if not isinstance(probe_horizon, int) or isinstance(probe_horizon, bool) or probe_horizon < 1:
        raise ValueError("active probe horizon must be positive")

    verifier = NBackVerifier(
        batch_size=1,
        n_back=2,
        steps=9,
        symbol_count=4,
        cue_symbol=6,
        time_shuffle=True,
        seed=verifier_seed,
    )
    verifier.reset()
    if isinstance(agent.runtime.memory, ContentAddressedMemory):
        agent.runtime.memory.clear()
    state = agent.initial_state(1, device=verifier.device)
    feedback = agent.initial_feedback(1, device=verifier.device)
    goal_state = torch.zeros(1, router.model.state_width, device=verifier.device)
    previous = None
    unique_verifier_bits = 0

    while not verifier.done:
        events = agent.runtime.encode_streams({"stimulus": verifier.observation()})
        output, next_state = policy.step_events(
            events,
            state,
            feedback,
            goal_state,
            candidate_intentions,
            horizon=1,
            beam_width=4,
            transition_context=source_context.unsqueeze(0),
        )
        if previous is not None:
            transition = policy.transition_observation(previous, output)
            transition_index = verifier.position - 1
            if transition_index == probe_transition_index:
                route = router.route_partial_bundle(
                    (transition,),
                    match_tolerance=ambiguity_tolerance,
                    contradiction_tolerance=ambiguity_tolerance,
                    match_margin=0.05,
                )
                if route.status != "ambiguous":
                    raise AssertionError(
                        "active pressure setup did not produce an ambiguous transition"
                    )
                before_router = router.digest()
                before_controller = _controller_digest(agent)
                current_output = output
                current_controller_state = next_state
                current_observation = transition
                probe_observations: list[ExternalTransitionObservation] = []
                selected_indices: list[int] = []
                selection_disagreements: list[float] = []
                selected_probe_leverage: list[float] = []
                selected_probe_support: list[float] = []
                selected_probe_utility: list[float] = []
                decoder_state_free = True
                scored = None
                for probe_index in range(probe_horizon):
                    probe = router.request_disambiguation_probe(
                        current_observation,
                        candidate_intentions,
                        candidate_slot_ids=(0, target_slot_id),
                        probe_state=current_output.state,
                        support_statistics=support_statistics,
                        utility_memory=utility_memory,
                    )
                    disagreement = probe.disagreement_scores
                    if forced_intention_index is not None:
                        if not 0 <= forced_intention_index < candidate_intentions.shape[0]:
                            raise ValueError("forced probe intention index is invalid")
                        selected_index = forced_intention_index
                    else:
                        selected_index = (
                            probe.selected_intention_index
                            if active
                            else int(disagreement.argmin())
                        )
                    disagreement_value = float(disagreement[selected_index])
                    selected_intention = candidate_intentions[selected_index]
                    if router.model.residual_bank is None:
                        selected_probe_leverage.append(0.0)
                    else:
                        candidate_contexts = torch.stack(
                            [
                                router._contexts[router._slot_ids.index(slot_id)]
                                for slot_id in (0, target_slot_id)
                            ]
                        ).to(current_output.state)
                        leverage = router.model.residual_bank.predictive_leverage(
                            current_output.state,
                            candidate_intentions,
                            candidate_contexts=candidate_contexts,
                        ).mean(dim=1)
                    selected_probe_leverage.append(float(leverage[selected_index]))
                    selected_probe_support.append(
                        0.0
                        if probe.support_scores is None
                        else float(probe.support_scores[selected_index])
                    )
                    selected_probe_utility.append(
                        0.5
                        if probe.utility_scores is None
                        else float(probe.utility_scores[selected_index])
                    )
                    selected_indices.append(selected_index)
                    selection_disagreements.append(disagreement_value)
                    decoded = policy.decode_intention(selected_intention)
                    decision = agent.keypress_decoder.decide_from_logits(
                        decoded["keypress"],
                        sample=False,
                    )
                    decoder_state_free = (
                        decoder_state_free
                        and _controller_digest(agent) == before_controller
                    )
                    scored = verifier.score(decision.key_index)
                    unique_verifier_bits += int(scored.eligible.sum())
                    feedback = _feedback(agent, scored, decision)
                    next_events = agent.runtime.encode_streams(
                        {"stimulus": verifier.observation()}
                    )
                    successor, successor_controller_state = policy.step_events(
                        next_events,
                        current_controller_state,
                        feedback,
                        goal_state,
                        candidate_intentions,
                        horizon=1,
                        beam_width=4,
                        transition_context=source_context.unsqueeze(0),
                    )
                    probe_observation = ExternalTransitionObservation(
                        state=current_output.state.detach().clone(),
                        intention=selected_intention.unsqueeze(0).detach().clone(),
                        next_state=successor.state.detach().clone(),
                    ).validate(
                        state_width=router.model.state_width,
                        intention_width=router.model.intention_width,
                    )
                    probe_observations.append(probe_observation)
                    current_observation = probe_observation
                    current_output = successor
                    current_controller_state = successor_controller_state

                if scored is None:
                    raise RuntimeError("active probe produced no verifier outcome")
                if probe_horizon == 1:
                    strict_route = router.route_partial_bundle(
                        (probe_observations[0],),
                        match_tolerance=probe_match_tolerance,
                        contradiction_tolerance=probe_contradiction_tolerance,
                        match_margin=0.01,
                    )
                else:
                    sequence_match_tolerance = probe_match_tolerance * probe_horizon + (
                        0.05 * (probe_horizon - 1)
                    )
                    sequence_contradiction_tolerance = (
                        probe_contradiction_tolerance * probe_horizon
                    )
                    strict_route = router.route_partial_sequence(
                        tuple((item,) for item in probe_observations),
                        match_tolerance=sequence_match_tolerance,
                        contradiction_tolerance=sequence_contradiction_tolerance,
                        match_margin=0.01,
                        confirmation_bundles=probe_horizon,
                    )
                router_read_only = before_router == router.digest()
                final_observation = probe_observations[-1]
                evidence_state = torch.cat(
                    [item.state for item in probe_observations], dim=0
                )
                evidence_intention = torch.cat(
                    [item.intention for item in probe_observations], dim=0
                )
                evidence_next_state = torch.cat(
                    [item.next_state for item in probe_observations], dim=0
                )
                probe_model_errors = tuple(
                    float(
                        (
                            router.model.predict_with_context(
                                evidence_state,
                                evidence_intention,
                                context.to(final_observation.state)
                                .unsqueeze(0)
                                .expand(probe_horizon, -1),
                            )
                            - evidence_next_state
                        )
                        .square()
                        .mean()
                    )
                    for context in router._contexts[:2]
                )
                probe_model_errors_by_step = tuple(
                    tuple(
                        float(
                            (
                                router.model.predict_with_context(
                                    item.state,
                                    item.intention,
                                    context.to(item.state)
                                    .unsqueeze(0)
                                    .expand(1, -1),
                                )
                                - item.next_state
                            )
                            .square()
                            .mean()
                        )
                        for context in router._contexts[:2]
                    )
                    for item in probe_observations
                )
                trial = ActiveDisambiguationTrial(
                    kind="active" if active else "passive_low_disagreement",
                    verifier_seed=verifier_seed,
                    selected_intention_index=selected_indices[0],
                    maximum_disagreement=max(selection_disagreements),
                    selected_disagreement=sum(selection_disagreements)
                    / len(selection_disagreements),
                    minimum_disagreement=min(selection_disagreements),
                    verifier_reward=float(scored.reward.item()),
                    verifier_outcome_eligible=bool(scored.eligible.item()),
                    probe_model_errors=probe_model_errors,
                    strict_route_status=strict_route.status,
                    strict_route_slot_id=strict_route.slot_id,
                    router_read_only=router_read_only,
                    decoder_state_free=decoder_state_free,
                    probe_horizon=probe_horizon,
                    selected_intention_indices=tuple(selected_indices),
                    probe_model_errors_by_step=probe_model_errors_by_step,
                    selected_probe_leverage=tuple(selected_probe_leverage),
                    selected_probe_support=tuple(selected_probe_support),
                    selected_probe_utility=tuple(selected_probe_utility),
                )
                target_recovered = (
                    strict_route.status == "matched"
                    and strict_route.slot_id == target_slot_id
                )
                return trial, target_recovered, unique_verifier_bits, 1

        decision = agent.keypress_decoder.decide_from_logits(
            output.decoded["keypress"],
            sample=False,
        )
        scored = verifier.score(decision.key_index)
        unique_verifier_bits += int(scored.eligible.sum())
        feedback = _feedback(agent, scored, decision)
        state = next_state
        previous = output

    raise RuntimeError("active disambiguation lifetime ended before its probe")


def run_active_disambiguation_pressure(
    *,
    seed: int,
    training_lifetimes: int = 6,
    steps: int = 9,
    random_feature_width: int = 128,
    probe_horizon: int = 1,
) -> ActiveDisambiguationPressureResult:
    """Audit active disambiguation on fresh rendered verifier evidence."""

    if min(training_lifetimes, steps, random_feature_width, probe_horizon) < 1:
        raise ValueError("active disambiguation pressure budgets must be positive")
    started = time.perf_counter()
    torch.manual_seed(seed)
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=12,
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

    state_adapter = ExternalControllerEventWindowStateAdapter(
        agent.controller.width,
        state_width=agent.controller.width * 3,
    )
    base = ExternalAffineTransitionStatistics(
        state_adapter.state_width,
        agent.controller.intention_width,
        ridge=1e-3,
    )
    model = ExternalFactoredTransitionModel(
        state_adapter.state_width,
        agent.controller.intention_width,
        context_width=4,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=random_feature_width,
        residual_ridge=1e-3,
        base_model=base,
        residual_capacity=3,
    )
    source_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    if model.residual_bank is None:
        raise RuntimeError("active pressure requires learned residual memory")
    model.residual_bank.ensure_context(source_context)
    seed_candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )
    candidate_intentions = seed_candidate_intentions
    intention_repertoire = ExternalIntentionRepertoire(
        agent.controller.intention_width
    )

    unique_verifier_bits = 0
    transition_rows = 0
    for lifetime in range(training_lifetimes):
        for time_shuffle in (False, True):
            rollout, bits = _run_lifetime(
                agent,
                _policy(agent, model, state_adapter),
                model,
                source_context,
                n_back=2,
                steps=steps,
                seed=seed + 100 + lifetime,
                cue_symbol=6,
                candidate_intentions=candidate_intentions,
                learn=False,
                time_shuffle=time_shuffle,
                intention_repertoire=intention_repertoire,
            )
            for observation in _rollout_observations(rollout):
                base.observe(observation)
            unique_verifier_bits += bits
            transition_rows += rollout.horizon
    model.freeze_base()
    verified_proposal = intention_repertoire.propose(
        include_seed=False,
        max_candidates=32,
    )
    if verified_proposal.intentions.shape[1] >= 2:
        candidate_intentions = verified_proposal.intentions[0].detach().clone()
    else:
        candidate_intentions = seed_candidate_intentions

    encoder = ExternalTransitionContextEncoder(
        model.state_width,
        model.intention_width,
        hidden_width=16,
        context_width=model.context_width,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=1e-4,
        match_margin=1e-5,
        admission_observations=1,
        max_contexts=3,
        residual_adaptation_updates=1,
    )

    promoted_slots: list[int] = []
    for regime_index, time_shuffle in enumerate((False, True)):
        rollouts = []
        for lifetime in range(training_lifetimes):
            rollout, bits = _run_lifetime(
                agent,
                _policy(agent, model, state_adapter),
                model,
                source_context,
                n_back=2,
                steps=steps,
                seed=seed + 100 + lifetime,
                cue_symbol=6,
                candidate_intentions=candidate_intentions,
                learn=False,
                time_shuffle=time_shuffle,
            )
            rollouts.append(rollout)
            unique_verifier_bits += bits
            transition_rows += rollout.horizon
        route = router.route_bundle(_rollout_observations(rollouts[0]))
        if route.status != "staged":
            raise AssertionError(f"regime {regime_index} failed to stage")
        for rollout in rollouts[1:]:
            for observation in _rollout_observations(rollout):
                router.observe(observation)
        heldout, bits = _run_lifetime(
            agent,
            _policy(agent, router._candidate_model, state_adapter),
            router._candidate_model,
            router._candidate_context,
            n_back=2,
            steps=steps,
            seed=seed + 20000 + regime_index,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=False,
            time_shuffle=time_shuffle,
        )
        unique_verifier_bits += bits
        promotion = router.promote_staged_candidate(
            _rollout_bundle(heldout),
            lambda _candidate: True,
            prediction_tolerance=1e9,
        )
        if not promotion.accepted or promotion.slot_id is None:
            raise AssertionError(f"regime {regime_index} failed promotion")
        promoted_slots.append(promotion.slot_id)

    source_slot_id, target_slot_id = promoted_slots
    support_statistics = ExternalTransitionSupportStatistics(
        bin_count=8,
        leverage_scale=10.0,
    )
    support_calibration_rows = 0
    support_calibration_lifetimes = 0
    for calibration_index, time_shuffle in enumerate((False, True)):
        calibration_rollout, calibration_bits = _run_lifetime(
            agent,
            _policy(agent, router.model, state_adapter),
            router.model,
            source_context,
            n_back=2,
            steps=steps,
            seed=seed + 25000 + calibration_index,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=False,
            time_shuffle=time_shuffle,
        )
        unique_verifier_bits += calibration_bits
        transition_rows += calibration_rollout.horizon
        support_calibration_rows += _observe_probe_support(
            router,
            calibration_rollout,
            support_statistics,
            candidate_slot_ids=(source_slot_id, target_slot_id),
            support_tolerance=0.30,
        )
        support_calibration_lifetimes += 1

    utility_memory = ExternalTransitionProbeUtilityMemory(
        agent.controller.intention_width,
        merge_cosine=0.999,
        prior_strength=1.0,
    )
    utility_calibration_observations = 0
    utility_calibration_lifetimes = 0
    utility_calibration_repeats = 1
    for candidate_index in range(min(6, candidate_intentions.shape[0])):
        for repeat_index in range(utility_calibration_repeats):
            calibration_trial, _recovered, calibration_bits, calibration_lifetimes = (
                _execute_probe_trial(
                    agent=agent,
                    policy=_policy(agent, router.model, state_adapter),
                    router=router,
                    verifier_seed=(
                        seed
                        + 26000
                        + (candidate_index * utility_calibration_repeats)
                        + repeat_index
                    ),
                    source_context=source_context,
                    candidate_intentions=candidate_intentions,
                    target_slot_id=target_slot_id,
                    probe_transition_index=3,
                    active=True,
                    ambiguity_tolerance=0.70,
                    probe_match_tolerance=0.30,
                    probe_contradiction_tolerance=0.40,
                    probe_horizon=1,
                    support_statistics=support_statistics,
                    forced_intention_index=candidate_index,
                )
            )
            utility_memory.observe(
                candidate_intentions[candidate_index].unsqueeze(0),
                utility=_diagnostic_probe_utility(calibration_trial),
            )
            unique_verifier_bits += calibration_bits
            utility_calibration_observations += 1
            utility_calibration_lifetimes += calibration_lifetimes

    active_trial, active_recovered, active_bits, active_lifetimes = (
        _execute_probe_trial(
            agent=agent,
            policy=_policy(agent, router.model, state_adapter),
            router=router,
            verifier_seed=seed + 30000,
            source_context=source_context,
            candidate_intentions=candidate_intentions,
            target_slot_id=target_slot_id,
            probe_transition_index=3,
            active=True,
            ambiguity_tolerance=0.70,
            probe_match_tolerance=0.30,
            probe_contradiction_tolerance=0.40,
            probe_horizon=probe_horizon,
            support_statistics=support_statistics,
            utility_memory=utility_memory,
        )
    )
    passive_trial, passive_recovered, passive_bits, passive_lifetimes = (
        _execute_probe_trial(
            agent=agent,
            policy=_policy(agent, router.model, state_adapter),
            router=router,
            verifier_seed=seed + 30001,
            source_context=source_context,
            candidate_intentions=candidate_intentions,
            target_slot_id=target_slot_id,
            probe_transition_index=3,
            active=False,
            ambiguity_tolerance=0.70,
            probe_match_tolerance=0.30,
            probe_contradiction_tolerance=0.40,
            probe_horizon=probe_horizon,
            support_statistics=support_statistics,
            utility_memory=utility_memory,
        )
    )
    for trial in (active_trial, passive_trial):
        utility_memory.observe(
            candidate_intentions[trial.selected_intention_index].unsqueeze(0),
            utility=_diagnostic_probe_utility(trial),
        )
    return ActiveDisambiguationPressureResult(
        seed=seed,
        status=(
            "active_probe_resolved_fresh_target"
            if active_recovered
            else "active_probe_not_resolved"
        ),
        controller_unchanged=controller_before == _controller_digest(agent),
        active_probe_recovered_target=active_recovered,
        passive_control_recovered_target=passive_recovered,
        active_probe_read_only=active_trial.router_read_only,
        active_decoder_state_free=active_trial.decoder_state_free,
        source_slot_id=source_slot_id,
        target_slot_id=target_slot_id,
        ambiguous_transition_observed=True,
        active_trial=active_trial,
        passive_trial=passive_trial,
        unique_verifier_bits=unique_verifier_bits + active_bits + passive_bits,
        logical_lifetimes=(
            (training_lifetimes * 2 * 2)
            + 2
            + support_calibration_lifetimes
            + utility_calibration_lifetimes
            + active_lifetimes
            + passive_lifetimes
        ),
        transition_rows_consumed_once=(
            transition_rows
            + utility_calibration_lifetimes
            + active_lifetimes
            + passive_lifetimes
        ),
        optimizer_updates=0,
        replayed_examples=0,
        wall_time_seconds=time.perf_counter() - started,
        probe_horizon=probe_horizon,
        candidate_intention_source=(
            "verified_opaque_repertoire_intentions_first_32"
            if verified_proposal.intentions.shape[1] >= 2
            else "seed_pool"
        ),
        support_calibration_rows=support_calibration_rows,
        utility_calibration_observations=utility_calibration_observations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--training-lifetimes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--random-feature-width", type=int, default=128)
    parser.add_argument("--probe-horizon", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_active_disambiguation_pressure(
        seed=args.seed,
        training_lifetimes=args.training_lifetimes,
        steps=args.steps,
        random_feature_width=args.random_feature_width,
        probe_horizon=args.probe_horizon,
    )
    payload = result.payload()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
