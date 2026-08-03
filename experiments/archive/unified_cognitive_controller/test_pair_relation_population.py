"""Contracts for retention-safe population selection."""
from __future__ import annotations

import pytest
import torch

from .train_pair_relation_population import (
    _score_rewards, _select_winner)


def _arm(
        learning_rate: float, dot_score: float, *,
        safe: bool = True, margin: float = 0.0,
        consolidation_steps: int = 40) -> dict[str, object]:
    return {
        "learning_rate": learning_rate,
        "dot_score": dot_score,
        "retention_safe": safe,
        "selection_eligible": safe and dot_score >= 0.95,
        "minimum_retention_margin": margin,
        "consolidation_steps": consolidation_steps,
    }


def test_population_prefers_dot_score_only_among_retention_safe_arms() -> None:
    winner = _select_winner([
        _arm(0.005, 0.96),
        _arm(0.006, 0.99, safe=False),
        _arm(0.007, 0.97),
    ])
    assert winner["learning_rate"] == 0.007


def test_population_uses_retention_margin_then_shorter_arm_as_ties() -> None:
    winner = _select_winner([
        _arm(0.007, 0.97, margin=-0.01, consolidation_steps=36),
        _arm(0.006, 0.97, margin=0.0, consolidation_steps=44),
        _arm(0.005, 0.97, margin=0.0, consolidation_steps=40),
    ])
    assert winner["learning_rate"] == 0.005


def test_population_rejects_when_no_arm_masters_and_retains() -> None:
    with pytest.raises(RuntimeError, match="mastery and retention"):
        _select_winner([
            _arm(0.005, 0.99, safe=False),
            _arm(0.006, 0.94, safe=True),
        ])


def test_population_scores_hidden_tasks_only_after_support() -> None:
    rewards = torch.tensor([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    assert _score_rewards(rewards, query_start=0) == pytest.approx(2 / 3)
    assert _score_rewards(rewards, query_start=1) == 1.0
