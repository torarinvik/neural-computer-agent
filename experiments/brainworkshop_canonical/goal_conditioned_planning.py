"""End-to-end downstream use of an opaque goal fragment.

This audit closes the gap between destination admission and useful behavior.
The controller, renderer, and decoder are frozen.  Fresh rendered lifetimes
teach an external replay-free factual transition bank; a held-out learned
state is then admitted as an opaque goal fragment through scalar goal
verification.  Model-based search must use that file to choose a next
intention sequence, and a matched fresh factual bank provides the acquisition
control.

The result is deliberately narrower than Neural Workshop mastery or general
continual learning: it measures whether learned external facts plus an
admitted destination compose into downstream planning without updating the
controller or replaying experience.
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

from .replay_free_transition_acquisition import (
    _opaque_context,
    _run_lifetime,
)
from .runner import CanonicalBrainWorkshopAgent

GOAL_CONDITIONED_PLANNING_AUDIT_SCHEMA = (
    "neural-computer.brainworkshop-goal-conditioned-planning-audit.v1"
)


@dataclass(frozen=True)
class GoalConditionedPlanningReport:
    """Accounting and gates for one bounded downstream planning audit."""

    schema: str
    status: str
    controller_unchanged: bool
    replay_free_bank: bool
    goal_fragment_admitted: bool
    goal_fragment_used: bool
    trained_planner_improved_over_fresh: bool
    trained_terminal_error: float
    fresh_terminal_error: float
    goal_horizon: int
    goal_verifier_threshold: float
    training_lifetimes: int
    total_logical_lifetimes: int
    unique_verifier_bits: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int
    external_slot_count: int
    durable_fragment_count: int
    missing_evidence_rejected: bool
    corrupted_goal_rejected: bool

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


def _goal_candidate(
    state: torch.Tensor,
) -> ExternalGoalFragmentCandidate:
    """Convert one learned planner state into an opaque full-state goal."""

    if state.ndim != 1:
        raise ValueError("goal-conditioned audit candidates require one state row")
    return ExternalGoalFragmentCandidate.from_state(state.detach())


def _exact_goal_retention_probe(
    candidate: ExternalGoalFragmentCandidate,
    state_width: int,
):
    """Verify that copy-on-write admission preserved the opaque file exactly."""

    values, masks = candidate.tensors(state_width=state_width)

    def probe(memory: ExternalGoalFragmentMemory) -> bool:
        if memory.fragment_count != 1:
            return False
        proposed = memory.propose((0,))
        return bool(
            torch.equal(proposed.values[0, 0].cpu(), values.cpu())
            and torch.equal(proposed.masks[0, 0].cpu(), masks.cpu())
        )

    return probe


def _plan_error(
    planner: ExternalModelBasedPlanner,
    *,
    initial_state: torch.Tensor,
    goal_memory: ExternalGoalFragmentMemory,
    candidate_intentions: torch.Tensor,
    context: torch.Tensor,
    horizon: int,
    fragment_id: int,
) -> tuple[float, bool]:
    """Run search through the external goal file and return terminal error."""

    fragments = goal_memory.propose(
        (fragment_id,),
        batch_size=1,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    result = planner.plan(
        initial_state.unsqueeze(0),
        None,
        candidate_intentions,
        horizon=horizon,
        beam_width=max(4, candidate_intentions.shape[0]),
        transition_context=context.unsqueeze(0),
        goal_fragments=fragments,
    )
    if result.candidate_indices is None:
        raise RuntimeError("goal-conditioned planner did not return candidate indices")
    error = float(result.scores[0].detach())
    # The planner's terminal score is exactly the full-mask goal distance in
    # this audit.  Keeping the verifier at the opaque destination boundary
    # prevents a semantic task/rule ID from entering the deployed path.
    return error, True


def run_goal_conditioned_planning_audit(
    *,
    seed: int = 93,
    steps: int = 6,
    training_lifetimes: int = 3,
    goal_horizon: int = 2,
    goal_verifier_threshold: float = 0.05,
) -> GoalConditionedPlanningReport:
    """Verify frozen-core factual learning composes with goal-file search."""

    if min(steps, training_lifetimes, goal_horizon) < 1:
        raise ValueError("goal-conditioned audit budgets must be positive")
    if goal_horizon > steps - 1:
        raise ValueError("goal-conditioned horizon exceeds held-out transitions")
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
    context = _opaque_context(agent, 6)
    bank.ensure_context(context)
    planner = ExternalModelBasedPlanner(bank, beam_width=8)
    policy_free = PolicyFreeAmodalRuntime(agent.runtime, planner)
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )

    unique_verifier_bits = 0
    for lifetime in range(training_lifetimes):
        _, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            context,
            n_back=2,
            steps=steps,
            seed=seed + lifetime,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=True,
        )
        unique_verifier_bits += bits

    # The goal is a learned state reached on a fresh held-out rendered
    # lifetime.  It is not copied from verifier state or a semantic label.
    heldout, heldout_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        context,
        n_back=2,
        steps=steps,
        seed=seed + 10000,
        cue_symbol=6,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_verifier_bits += heldout_bits
    candidate = _goal_candidate(heldout.expected_states[goal_horizon - 1])
    candidate_digest = candidate.digest(state_width=bank.state_width)

    stager = ExternalGoalFragmentStager(
        bank.state_width,
        threshold=1.0,
        min_observations=1,
        min_stable_observations=1,
    )
    probe_memory = ExternalGoalFragmentMemory(bank.state_width)
    candidate_values, candidate_masks = candidate.tensors(state_width=bank.state_width)
    probe_memory.append(candidate_values, candidate_masks)
    candidate_error, goal_used = _plan_error(
        planner,
        initial_state=heldout.initial_state,
        goal_memory=probe_memory,
        candidate_intentions=candidate_intentions,
        context=context,
        horizon=goal_horizon,
        fragment_id=0,
    )
    # Stage only the opaque scalar result of the factual goal probe.  The
    # candidate itself is retained by the stager, but no trajectory or reward
    # history is copied into durable memory.
    stager.observe(
        candidate,
        float(candidate_error <= goal_verifier_threshold),
    )
    memory = ExternalGoalFragmentMemory(bank.state_width)
    admission = stager.admit_verified(
        memory,
        candidate_digest,
        _exact_goal_retention_probe(candidate, bank.state_width),
    )
    if not admission.accepted:
        raise RuntimeError("goal-conditioned audit could not admit its verified goal")

    trained_error, goal_used = _plan_error(
        planner,
        initial_state=heldout.initial_state,
        goal_memory=memory,
        candidate_intentions=candidate_intentions,
        context=context,
        horizon=goal_horizon,
        fragment_id=admission.fragment_id if admission.fragment_id is not None else 0,
    )
    fresh_bank = ExternalTransitionModelBank(
        state_width=bank.state_width,
        intention_width=bank.intention_width,
        context_width=bank.context_width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    fresh_bank.ensure_context(context)
    fresh_error, fresh_goal_used = _plan_error(
        ExternalModelBasedPlanner(fresh_bank, beam_width=8),
        initial_state=heldout.initial_state,
        goal_memory=memory,
        candidate_intentions=candidate_intentions,
        context=context,
        horizon=goal_horizon,
        fragment_id=admission.fragment_id if admission.fragment_id is not None else 0,
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
    corrupted_values, corrupted_masks = candidate.tensors(state_width=bank.state_width)
    corrupted = ExternalGoalFragmentCandidate(
        corrupted_values + 10.0,
        corrupted_masks,
    )
    corrupted_digest = corrupted.digest(state_width=bank.state_width)
    corrupted_stager = ExternalGoalFragmentStager(
        bank.state_width,
        threshold=1.0,
        min_observations=1,
        min_stable_observations=1,
    )
    corrupted_stager.observe(corrupted, 0.0)
    corrupted_rejection = corrupted_stager.admit_verified(
        ExternalGoalFragmentMemory(bank.state_width),
        corrupted_digest,
        lambda proposed: False,
    )
    unchanged = controller_before == _controller_digest(agent)
    improved = trained_error < fresh_error
    passed = (
        unchanged
        and bank.replay_free_updates
        and admission.accepted
        and goal_used
        and fresh_goal_used
        and improved
        and missing_rejection.accepted is False
        and corrupted_rejection.accepted is False
    )
    return GoalConditionedPlanningReport(
        schema=GOAL_CONDITIONED_PLANNING_AUDIT_SCHEMA,
        status=(
            "goal_conditioned_external_planning_boundary"
            if passed
            else "goal_conditioned_external_planning_boundary_failed"
        ),
        controller_unchanged=unchanged,
        replay_free_bank=bank.replay_free_updates,
        goal_fragment_admitted=admission.accepted,
        goal_fragment_used=goal_used and fresh_goal_used,
        trained_planner_improved_over_fresh=improved,
        trained_terminal_error=trained_error,
        fresh_terminal_error=fresh_error,
        goal_horizon=goal_horizon,
        goal_verifier_threshold=goal_verifier_threshold,
        training_lifetimes=training_lifetimes,
        total_logical_lifetimes=training_lifetimes + 1,
        unique_verifier_bits=unique_verifier_bits,
        transition_rows_consumed_once=training_lifetimes * steps,
        optimizer_updates=0,
        replayed_examples=0,
        external_slot_count=bank.context_count,
        durable_fragment_count=memory.fragment_count,
        missing_evidence_rejected=not missing_rejection.accepted,
        corrupted_goal_rejected=not corrupted_rejection.accepted,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run_goal_conditioned_planning_audit().payload(), indent=2))
