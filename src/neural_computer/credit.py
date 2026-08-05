"""Trainer-only counterfactual credit assignment utilities.

The deployed controller never receives these interventions or their private
pairing metadata. A trainer may run common-random-number environment arms and
use only their scalar verifier outcomes to assign credit to an opaque policy
decision. Keeping the arithmetic here makes the protocol reusable for memory
writes, artifact routes, register updates, and other generic actions.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate_utilities(utilities: torch.Tensor) -> None:
    if utilities.ndim != 2 or utilities.shape[1] < 2:
        raise ValueError("utilities must have shape [pairs, at least two arms]")
    if utilities.shape[0] < 1:
        raise ValueError("utilities must contain at least one pair")
    if not bool(torch.isfinite(utilities).all()):
        raise ValueError("utilities must be finite")


def paired_counterfactual_advantage(
    utilities: torch.Tensor,
    *,
    positive_arm: int = 0,
    negative_arm: int = 1,
) -> torch.Tensor:
    """Return the detached scalar utility difference for each paired world."""
    _validate_utilities(utilities)
    arms = utilities.shape[1]
    if not 0 <= positive_arm < arms or not 0 <= negative_arm < arms:
        raise ValueError("counterfactual arm is out of range")
    if positive_arm == negative_arm:
        raise ValueError("counterfactual arms must be distinct")
    return (
        utilities[:, positive_arm] - utilities[:, negative_arm]
    ).detach()


def paired_counterfactual_policy_loss(
    decision_score: torch.Tensor,
    utilities: torch.Tensor,
    *,
    positive_arm: int = 0,
    negative_arm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train one opaque decision from a paired scalar outcome difference.

    ``decision_score`` is the policy score for the intervention being
    credited, usually a Bernoulli logit or log-probability. It must have one
    value per paired hidden world. The returned advantage is detached so the
    verifier cannot become part of the computation graph.
    """
    if decision_score.ndim != 1:
        raise ValueError("decision_score must have shape [pairs]")
    _validate_utilities(utilities)
    if decision_score.shape[0] != utilities.shape[0]:
        raise ValueError("decision scores and utility pairs must have equal length")
    if not bool(torch.isfinite(decision_score).all()):
        raise ValueError("decision_score must be finite")
    advantage = paired_counterfactual_advantage(
        utilities,
        positive_arm=positive_arm,
        negative_arm=negative_arm,
    )
    return -(advantage * decision_score).mean(), advantage


def factorized_counterfactual_policy_loss(
    decision_scores: torch.Tensor,
    utilities: torch.Tensor,
    *,
    positive_arm: int = 0,
    negative_arm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign paired utility credit independently to several decisions.

    ``decision_scores`` has shape ``[pairs, factors]`` and ``utilities`` has
    shape ``[pairs, factors, arms]``. Each factor must be evaluated in its
    own common-random intervention pair; reusing one global terminal reward
    for every factor would incorrectly claim causal attribution. The returned
    advantage tensor is detached and retains the factor axis for auditing.
    """
    if decision_scores.ndim != 2:
        raise ValueError("decision_scores must have shape [pairs, factors]")
    if utilities.ndim != 3:
        raise ValueError("utilities must have shape [pairs, factors, arms]")
    if decision_scores.shape[:2] != utilities.shape[:2]:
        raise ValueError("decision scores and utilities must share pairs and factors")
    if not bool(torch.isfinite(decision_scores).all()):
        raise ValueError("decision_scores must be finite")
    flat_advantage = paired_counterfactual_advantage(
        utilities.reshape(-1, utilities.shape[-1]),
        positive_arm=positive_arm,
        negative_arm=negative_arm,
    )
    advantage = flat_advantage.reshape_as(decision_scores)
    return -(advantage * decision_scores).mean(), advantage


def paired_counterfactual_ranking_loss(
    candidate_scores: torch.Tensor,
    attempted_arms: torch.Tensor,
    utilities: torch.Tensor,
    *,
    positive_arm: int = 0,
    negative_arm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank two attempted opaque candidates using paired verifier outcomes.

    ``candidate_scores`` contains the full memory/artifact score matrix. The
    trainer supplies only the two attempted row indices and their scalar
    outcomes; no correct-row label is needed. This is useful when a route and
    a downstream action must receive credit from the same verifier result.
    """
    if candidate_scores.ndim != 2 or candidate_scores.shape[1] < 2:
        raise ValueError("candidate_scores must have shape [pairs, rows]")
    if attempted_arms.ndim != 2 or attempted_arms.shape[1] < 2:
        raise ValueError("attempted_arms must have shape [pairs, at least two arms]")
    _validate_utilities(utilities)
    if candidate_scores.shape[0] != attempted_arms.shape[0]:
        raise ValueError("candidate scores and attempted arms must have equal length")
    if utilities.shape[0] != attempted_arms.shape[0]:
        raise ValueError("attempted arms and utilities must have equal length")
    if positive_arm >= utilities.shape[1] or negative_arm >= utilities.shape[1]:
        raise ValueError("counterfactual arm is out of range")
    if bool(torch.any(attempted_arms < 0)) or bool(
        torch.any(attempted_arms >= candidate_scores.shape[1])
    ):
        raise ValueError("attempted row is out of range")
    selected = candidate_scores.gather(1, attempted_arms.to(torch.long))
    decision_scores = selected[:, positive_arm] - selected[:, negative_arm]
    advantage = paired_counterfactual_advantage(
        utilities,
        positive_arm=positive_arm,
        negative_arm=negative_arm,
    )
    informative = advantage != 0.0
    if not bool(informative.any()):
        loss = decision_scores.sum() * 0.0
    else:
        loss = F.softplus(
            -advantage[informative] * decision_scores[informative]
        ).mean()
    return loss, advantage
