"""Contracts for the fourth-primitive rung.

These cover the properties that make the rung's numbers mean anything: only the
appended slot may move, retention gates must track what the parent actually
had, and the cue ablation must compare two renderings of the same events.
"""
from __future__ import annotations

import torch

from experiments.unified_cognitive_controller.environment import (
    generate_lifetimes)
from experiments.unified_cognitive_controller.model import (
    UnifiedCognitiveController)
from experiments.unified_cognitive_controller.train_fourth_primitive_transfer import (
    NEW_TASK, REPLAY_TASKS, _headline_accuracy, _new_skill_loss,
    _operation_cue_ablation_accuracy, _plastic_prefixes,
    _replay_appearance)


def test_plastic_prefixes_name_only_the_appended_slot() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16, 16, 16))
    names = list(model.state_dict())
    for slot in range(3):
        selected = [
            name for name in names
            if name.startswith(_plastic_prefixes(slot))]
        assert selected, slot
        # No other slot's parameters may be caught by this slot's prefixes.
        for other in range(3):
            if other == slot:
                continue
            assert not any(
                name.startswith(_plastic_prefixes(other))
                for name in selected), (slot, other)


def test_training_only_moves_the_appended_slot() -> None:
    parent = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        relation_adapter_width=16, relation_adapter_gated=True,
        action_adapter_width=16, action_adapter_gated=True)
    student = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        relation_adapter_width=16, relation_adapter_gated=True,
        action_adapter_width=16, action_adapter_gated=True,
        skill_adapter_widths=(16,))
    student.load_state_dict(parent.state_dict(), strict=False)
    prefixes = _plastic_prefixes(0)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    before = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=0.05)
    batch = generate_lifetimes(
        8, 6, seed=51, task=NEW_TASK, support_trials=2)
    loss = _new_skill_loss(
        student, batch, exploration=0.1, support_trials=2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Every frozen tensor must have no gradient at all, not merely a small one.
    for name, parameter in student.named_parameters():
        if name.startswith(prefixes):
            continue
        assert parameter.grad is None, name
    optimizer.step()
    moved = {
        name for name, value in student.state_dict().items()
        if not torch.equal(before[name], value)}
    assert moved
    assert all(name.startswith(prefixes) for name in moved), moved


def test_new_skill_loss_can_report_observed_training_accuracy() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16,))
    batch = generate_lifetimes(
        8, 6, seed=52, task=NEW_TASK, support_trials=2)
    loss, accuracy = _new_skill_loss(
        model, batch, exploration=0.1, support_trials=2,
        return_accuracy=True)
    assert loss.ndim == 0
    assert 0.0 <= accuracy <= 1.0


def test_cue_ablation_compares_the_same_events() -> None:
    """The ablation must change only the cue, never the task content."""
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    accuracy = _operation_cue_ablation_accuracy(
        model, count=32, seed=53, device=torch.device("cpu"),
        support_trials=2)
    assert 0.0 <= accuracy <= 1.0
    marked = generate_lifetimes(
        32, 6, seed=53, heldout=True, task=NEW_TASK, support_trials=2)
    unmarked = generate_lifetimes(
        32, 6, seed=53, heldout=True, task="visible_context",
        support_trials=2)
    assert torch.equal(
        marked.stimulus_identities, unmarked.stimulus_identities)
    assert torch.equal(marked.context_ids, unmarked.context_ids)
    assert torch.equal(marked.rule_bits, unmarked.rule_bits)
    assert not torch.equal(marked.frames, unmarked.frames)


def test_headline_accuracy_picks_the_metric_each_task_family_reports() -> None:
    visible = {"overall_accuracy": 0.93, "normal": {
        "post_feedback_accuracy": 0.11}}
    assert _headline_accuracy(visible) == 0.93
    hidden = {"normal": {"post_feedback_accuracy": 0.88}}
    assert _headline_accuracy(hidden) == 0.88


def test_replay_covers_every_earlier_primitive() -> None:
    assert NEW_TASK not in REPLAY_TASKS
    assert set(REPLAY_TASKS) == {
        "binary_mapping", "visible_context", "visible_context_xor"}


def test_relation_replay_can_cycle_its_full_appearance_repertoire() -> None:
    observed = [
        _replay_appearance("pair_relation", "cycle", update)
        for update in range(1, 7)]
    assert observed == [
        "bars", "diamonds", "dot_pairs",
        "bars", "diamonds", "dot_pairs"]
    assert _replay_appearance("pair_relation", "dot_pairs", 99) == "dot_pairs"
    assert _replay_appearance("binary_mapping", "cycle", 3) == "bars"
