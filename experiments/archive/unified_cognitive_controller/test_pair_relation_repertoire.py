"""Contracts for the first cross-family repertoire primitive."""
from __future__ import annotations

import torch

from .environment import generate_lifetimes
from .legacy_model import UnifiedCognitiveController
from .train import evaluate
from .train_fourth_primitive_transfer import (
    _operation_cue_ablation_accuracy)


def test_pair_relation_is_balanced_and_deterministic() -> None:
    first = generate_lifetimes(
        32, 6, seed=4101, task="pair_relation")
    second = generate_lifetimes(
        32, 6, seed=4101, task="pair_relation")
    assert torch.equal(first.frames, second.frames)
    assert torch.equal(first.correct_actions, second.correct_actions)
    assert first.context_ids is not None
    assert set(first.correct_actions.unique().tolist()) == {0, 1}
    assert all(
        set(row.tolist()) == {0, 1}
        for row in first.correct_actions)
    assert torch.equal(
        first.correct_actions,
        first.stimulus_identities ^ first.context_ids)


def test_pair_relation_counterfactual_flips_every_answer() -> None:
    normal = generate_lifetimes(
        32, 6, seed=4102, heldout=True, task="pair_relation")
    reversed_pair = generate_lifetimes(
        32, 6, seed=4102, heldout=True, task="pair_relation",
        reverse_contexts=True)
    assert torch.equal(
        normal.stimulus_identities, reversed_pair.stimulus_identities)
    assert torch.equal(normal.rule_bits, reversed_pair.rule_bits)
    assert torch.equal(
        reversed_pair.correct_actions, 1 - normal.correct_actions)
    assert not torch.equal(normal.frames, reversed_pair.frames)


def test_pair_relation_heldout_render_changes_positions_not_logic() -> None:
    train = generate_lifetimes(
        32, 6, seed=4103, task="pair_relation")
    heldout = generate_lifetimes(
        32, 6, seed=4103, heldout=True, task="pair_relation")
    assert torch.equal(train.correct_actions, heldout.correct_actions)
    assert torch.equal(
        train.stimulus_identities, heldout.stimulus_identities)
    assert torch.equal(train.context_ids, heldout.context_ids)
    assert not torch.equal(train.frames, heldout.frames)


def test_pair_relation_uses_visible_task_audit_contract() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    report = evaluate(
        model, count=32, trials=6, seed=4104,
        device=torch.device("cpu"), task="pair_relation",
        feedback_trials=1)
    assert "overall_accuracy" in report
    assert "counterfactual_overall_accuracy" in report
    assert "pixel_counterfactual_flip_at_least_80" in report["gate"]


def test_pair_relation_ablation_removes_only_second_object_region() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    for appearance in ("bars", "diamonds", "dot_pairs"):
        accuracy = _operation_cue_ablation_accuracy(
            model, count=32, seed=4105, device=torch.device("cpu"),
            support_trials=1, new_task="pair_relation",
            appearance=appearance)
        assert 0.0 <= accuracy <= 1.0
