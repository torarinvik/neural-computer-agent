"""Pressure-test outcome-only goal-fragment staging on rendered Brain Workshop.

This audit deliberately stops at destination admission.  A cue is encoded into
an opaque event tensor, fresh verifier lifetimes provide only scalar episode
outcomes, and the frozen controller is never updated.  The resulting fragment
is not yet used to claim a new n-back capability; the audit only verifies that
the external destination boundary can learn, persist, and reject bad evidence
without replay.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import torch

from neural_computer import (
    ExternalGoalFragmentCandidate,
    ExternalGoalFragmentMemory,
    ExternalGoalFragmentStager,
)

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .trainer import train_reward_only

GOAL_FRAGMENT_STAGING_AUDIT_SCHEMA = (
    "neural-computer.brainworkshop-goal-fragment-staging-audit.v1"
)


@dataclass(frozen=True)
class GoalFragmentStagingReport:
    """Machine-readable accounting for one sub-minute staging audit."""

    schema: str
    status: str
    candidate_digest: str
    accepted: bool
    missing_evidence_accepted: bool
    inverted_outcome_accepted: bool
    controller_unchanged: bool
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    optimizer_updates: int
    replayed_examples: int
    staging_observations: int
    pending_candidates_after_admission: int
    durable_fragment_count: int

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


def _cue_candidate(
    agent: CanonicalBrainWorkshopAgent,
    cue_symbol: int,
) -> ExternalGoalFragmentCandidate:
    encoded = agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )
    return ExternalGoalFragmentCandidate.from_state(encoded[0].detach())


def _stable_probe(
    candidate: ExternalGoalFragmentCandidate,
    state_width: int,
):
    values, masks = candidate.tensors(state_width=state_width)

    def probe(memory: ExternalGoalFragmentMemory) -> bool:
        if memory.fragment_count != 1:
            return False
        proposed = memory.propose((0,))
        return bool(
            torch.equal(proposed.values[0, 0].cpu(), values)
            and torch.equal(proposed.masks[0, 0].cpu(), masks)
        )

    return probe


def run_goal_fragment_staging_audit(
    *,
    seed: int = 17,
    n_back: int = 2,
    cue_symbol: int = 4,
    updates: int = 32,
    batch_size: int = 16,
    steps: int = 6,
    staging_lifetimes: int = 4,
    learning_rate: float = 1e-2,
    threshold: float = 0.75,
) -> GoalFragmentStagingReport:
    """Run a bounded rendered-event destination-staging audit.

    The short reward-only pretraining phase shapes only external reader and
    decoder state.  Staging then consumes fresh episode scores as scalar
    evidence.  It is intentionally not a promotion of learned goal discovery:
    the retention probe checks persistence/integrity, not downstream mastery.
    """

    if min(n_back, updates, batch_size, steps, staging_lifetimes) < 1:
        raise ValueError("audit dimensions and budgets must be positive")
    if cue_symbol < 0:
        raise ValueError("cue symbol must be non-negative")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("staging threshold must lie in [0, 1]")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=max(4, cue_symbol + 1),
        n_back=n_back,
        event_width=16,
        intention_width=6,
        feedback_width=6,
        reader_kind="relation",
        seed=seed,
    )
    controller_before = _controller_digest(agent)
    history = train_reward_only(
        agent,
        n_back=n_back,
        updates=updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        learning_rate=learning_rate,
        context_route=True,
        cue_symbol=cue_symbol,
    )
    controller_after_training = _controller_digest(agent)

    candidate = _cue_candidate(agent, cue_symbol)
    stager = ExternalGoalFragmentStager(
        candidate.values.shape[-1],
        threshold=threshold,
        min_observations=3,
        min_stable_observations=2,
    )
    memory = ExternalGoalFragmentMemory(candidate.values.shape[-1])
    observed_lifetimes = 0
    staging_observations = 0
    unique_verifier_bits = 0
    for lifetime in range(staging_lifetimes):
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            symbol_count=4,
            cue_symbol=cue_symbol,
            seed=seed + 1000 + lifetime,
        )
        with torch.no_grad():
            rollout = agent.rollout(
                verifier,
                sample=False,
                record_retention=False,
                context_route=True,
            )
        for score in rollout.episode_scores:
            stager.observe(candidate, score)
            staging_observations += 1
        observed_lifetimes += batch_size
        unique_verifier_bits += batch_size * verifier.eligible_trials

    digest = candidate.digest(state_width=candidate.values.shape[-1])
    admission = stager.admit_verified(
        memory,
        digest,
        _stable_probe(candidate, candidate.values.shape[-1]),
    )

    missing = ExternalGoalFragmentStager(
        candidate.values.shape[-1],
        threshold=threshold,
        min_observations=3,
        min_stable_observations=2,
    )
    missing.observe(candidate, 0.0, eligible=False)
    missing_admission = missing.admit_verified(
        ExternalGoalFragmentMemory(candidate.values.shape[-1]),
        digest,
        _stable_probe(candidate, candidate.values.shape[-1]),
    )

    inverted = ExternalGoalFragmentStager(
        candidate.values.shape[-1],
        threshold=threshold,
        min_observations=3,
        min_stable_observations=2,
    )
    for lifetime in range(staging_lifetimes):
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            symbol_count=4,
            cue_symbol=cue_symbol,
            seed=seed + 1000 + lifetime,
        )
        with torch.no_grad():
            rollout = agent.rollout(
                verifier,
                sample=False,
                record_retention=False,
                context_route=True,
            )
        for score in rollout.episode_scores:
            inverted.observe(candidate, 1.0 - score)
    inverted_admission = inverted.admit_verified(
        ExternalGoalFragmentMemory(candidate.values.shape[-1]),
        digest,
        _stable_probe(candidate, candidate.values.shape[-1]),
    )

    return GoalFragmentStagingReport(
        schema=GOAL_FRAGMENT_STAGING_AUDIT_SCHEMA,
        status="staging_boundary_only",
        candidate_digest=digest,
        accepted=admission.accepted,
        missing_evidence_accepted=missing_admission.accepted,
        inverted_outcome_accepted=inverted_admission.accepted,
        controller_unchanged=(
            controller_before == controller_after_training == _controller_digest(agent)
        ),
        unique_verifier_bits=unique_verifier_bits,
        unique_logical_lifetimes=observed_lifetimes,
        optimizer_updates=len(history),
        replayed_examples=sum(row.replayed_examples for row in history),
        staging_observations=staging_observations,
        pending_candidates_after_admission=stager.pending_count,
        durable_fragment_count=memory.fragment_count,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run_goal_fragment_staging_audit().payload(), indent=2))
