"""Contracts for recurrent-compute accounting on the relation repertoire."""
from __future__ import annotations

import torch

from .audit_pair_relation_compute import _rollout_with_extra_thought
from .environment import generate_lifetimes
from .model import UnifiedCognitiveController
from .train import rollout


def test_zero_extra_thought_is_bit_identical_to_standard_rollout() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    batch = generate_lifetimes(
        8, 6, seed=9901, task="pair_relation",
        appearance="dot_pairs", support_trials=1)
    standard = rollout(
        model, batch, sample_actions=False, feedback_trials=1)
    audited = _rollout_with_extra_thought(
        model, batch, extra_thought_steps=0, feedback_trials=1)
    for name in (
            "actions", "rewards", "logits", "final_workspace",
            "final_hidden"):
        assert torch.equal(standard[name], audited[name]), name


def test_extra_thought_adds_no_new_external_evidence() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    batch = generate_lifetimes(
        8, 6, seed=9902, task="pair_relation",
        appearance="diamonds", support_trials=1)
    result = _rollout_with_extra_thought(
        model, batch, extra_thought_steps=2, feedback_trials=1)
    assert result["actions"].shape == (8, 6)
    assert result["rewards"].shape == (8, 6)
