"""Nonstationary source-retention and target-acquisition pressure test.

This is the next rung after the same-family goal audit.  A source rendered
family is learned first, then a structurally different target family is
learned in a separate opaque factual slot without replaying the source.  The
target also receives an opaque goal fragment and must beat a matched fresh
target bank while the source remains byte-stable.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    ExternalGoalFragmentCandidate,
    ExternalGoalFragmentMemory,
    ExternalGoalFragmentStager,
    ExternalModelBasedPlanner,
    ExternalTransitionModelBank,
    PolicyFreeAmodalRuntime,
)

from .goal_conditioned_planning import (
    _exact_goal_retention_probe,
    _plan_error,
)
from .replay_free_transition_acquisition import _opaque_context, _run_lifetime
from .runner import CanonicalBrainWorkshopAgent

NONSTATIONARY_GOAL_PLANNING_AUDIT_SCHEMA = (
    "neural-computer.brainworkshop-nonstationary-goal-planning-audit.v1"
)


@dataclass(frozen=True)
class NonstationaryGoalPlanningReport:
    """Accounting for source retention plus target goal-conditioned search."""

    schema: str
    status: str
    controller_unchanged: bool
    replay_free_bank: bool
    source_slot_byte_stable: bool
    target_goal_fragment_admitted: bool
    target_goal_fragment_used: bool
    target_planner_improved_over_fresh: bool
    source_error_before_target: float
    source_error_after_target: float
    trained_target_terminal_error: float
    fresh_target_terminal_error: float
    source_training_lifetimes: int
    target_training_lifetimes: int
    goal_horizon: int
    unique_verifier_bits: int
    total_logical_lifetimes: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int
    external_slot_count: int
    durable_fragment_count: int
    missing_evidence_rejected: bool

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


def run_nonstationary_goal_conditioned_planning_audit(
    *,
    seed: int = 93,
    steps: int = 6,
    source_training_lifetimes: int = 3,
    target_training_lifetimes: int = 3,
    goal_horizon: int = 2,
    goal_verifier_threshold: float = 0.05,
) -> NonstationaryGoalPlanningReport:
    """Learn B after A, use B's goal file, and verify A survives unchanged."""

    if min(
        steps,
        source_training_lifetimes,
        target_training_lifetimes,
        goal_horizon,
    ) < 1:
        raise ValueError("nonstationary goal audit budgets must be positive")
    if goal_horizon > steps - 1:
        raise ValueError("nonstationary goal horizon exceeds held-out transitions")
    if goal_verifier_threshold <= 0.0:
        raise ValueError("goal verifier threshold must be positive")

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
    bank.ensure_context(target_context)
    planner = ExternalModelBasedPlanner(bank, beam_width=8)
    policy_free = PolicyFreeAmodalRuntime(agent.runtime, planner)
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )

    unique_verifier_bits = 0
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
        unique_verifier_bits += bits
    source_holdout, source_holdout_bits = _run_lifetime(
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
    unique_verifier_bits += source_holdout_bits
    source_digest = bank.models[source_index].digest()
    source_error_before = planner.rollout_error(
        source_holdout,
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
        unique_verifier_bits += bits
    target_holdout, target_holdout_bits = _run_lifetime(
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
    unique_verifier_bits += target_holdout_bits
    source_error_after = planner.rollout_error(
        source_holdout,
        transition_context=source_context.unsqueeze(0),
    )

    candidate = ExternalGoalFragmentCandidate.from_state(
        target_holdout.expected_states[goal_horizon - 1].detach()
    )
    candidate_digest = candidate.digest(state_width=bank.state_width)
    probe_memory = ExternalGoalFragmentMemory(bank.state_width)
    values, masks = candidate.tensors(state_width=bank.state_width)
    probe_memory.append(values, masks)
    target_probe_error, target_goal_used = _plan_error(
        planner,
        initial_state=target_holdout.initial_state,
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
        float(target_probe_error <= goal_verifier_threshold),
    )
    memory = ExternalGoalFragmentMemory(bank.state_width)
    admission = stager.admit_verified(
        memory,
        candidate_digest,
        _exact_goal_retention_probe(candidate, bank.state_width),
    )
    if not admission.accepted or admission.fragment_id is None:
        raise RuntimeError("target goal fragment failed its verifier gate")

    trained_target_error, target_goal_used = _plan_error(
        planner,
        initial_state=target_holdout.initial_state,
        goal_memory=memory,
        candidate_intentions=candidate_intentions,
        context=target_context,
        horizon=goal_horizon,
        fragment_id=admission.fragment_id,
    )
    fresh_bank = ExternalTransitionModelBank(
        state_width=bank.state_width,
        intention_width=bank.intention_width,
        context_width=bank.context_width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    fresh_bank.ensure_context(target_context)
    fresh_target_error, fresh_goal_used = _plan_error(
        ExternalModelBasedPlanner(fresh_bank, beam_width=8),
        initial_state=target_holdout.initial_state,
        goal_memory=memory,
        candidate_intentions=candidate_intentions,
        context=target_context,
        horizon=goal_horizon,
        fragment_id=admission.fragment_id,
    )

    missing = ExternalGoalFragmentStager(
        bank.state_width,
        threshold=1.0,
        min_observations=1,
        min_stable_observations=1,
    )
    missing.observe(candidate, 0.0, eligible=False)
    missing_rejection = missing.admit_verified(
        ExternalGoalFragmentMemory(bank.state_width),
        candidate_digest,
        _exact_goal_retention_probe(candidate, bank.state_width),
    )
    unchanged = controller_before == _controller_digest(agent)
    source_stable = source_digest == bank.models[source_index].digest()
    improved = trained_target_error < fresh_target_error
    passed = (
        unchanged
        and bank.replay_free_updates
        and source_stable
        and admission.accepted
        and target_goal_used
        and fresh_goal_used
        and improved
        and source_error_after == source_error_before
        and not missing_rejection.accepted
    )
    return NonstationaryGoalPlanningReport(
        schema=NONSTATIONARY_GOAL_PLANNING_AUDIT_SCHEMA,
        status=(
            "nonstationary_goal_conditioned_external_planning_boundary"
            if passed
            else "nonstationary_goal_conditioned_external_planning_boundary_failed"
        ),
        controller_unchanged=unchanged,
        replay_free_bank=bank.replay_free_updates,
        source_slot_byte_stable=source_stable,
        target_goal_fragment_admitted=admission.accepted,
        target_goal_fragment_used=target_goal_used and fresh_goal_used,
        target_planner_improved_over_fresh=improved,
        source_error_before_target=source_error_before,
        source_error_after_target=source_error_after,
        trained_target_terminal_error=trained_target_error,
        fresh_target_terminal_error=fresh_target_error,
        source_training_lifetimes=source_training_lifetimes,
        target_training_lifetimes=target_training_lifetimes,
        goal_horizon=goal_horizon,
        unique_verifier_bits=unique_verifier_bits,
        total_logical_lifetimes=(
            source_training_lifetimes + target_training_lifetimes + 2
        ),
        transition_rows_consumed_once=(
            (source_training_lifetimes + target_training_lifetimes) * steps
        ),
        optimizer_updates=0,
        replayed_examples=0,
        external_slot_count=bank.context_count,
        durable_fragment_count=memory.fragment_count,
        missing_evidence_rejected=not missing_rejection.accepted,
    )


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            run_nonstationary_goal_conditioned_planning_audit().payload(),
            indent=2,
        )
    )
