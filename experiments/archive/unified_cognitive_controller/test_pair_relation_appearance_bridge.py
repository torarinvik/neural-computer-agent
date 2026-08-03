"""Contracts for matched cross-appearance relation transfer."""
from __future__ import annotations

import torch

from .legacy_model import UnifiedCognitiveController
from .train_pair_relation_appearance_bridge import (
    _concatenate, _pair_loss, _prioritized_replay_loss, _replay_specs,
    _reset_slot, _slot_prefixes, _unrelated_locality)
from .environment import generate_lifetimes


def test_reset_replaces_only_selected_slot() -> None:
    configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "skill_adapter_widths": (16,),
        "skill_adapter_gate_mode": "relu",
        "skill_adapter_gate_refiner_widths": (8,),
        "skill_adapter_gate_extension_widths": (8,),
    }
    model = UnifiedCognitiveController(**configuration)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()}
    # Make the learned slot observably different from its initializer.
    with torch.no_grad():
        model.skill_adapters[0][-1].bias.fill_(3.0)
        model.skill_adapter_gate_extensions[0][-1].bias.fill_(2.0)
    _reset_slot(model, configuration, slot=0)
    prefixes = _slot_prefixes(0)
    for name, value in model.state_dict().items():
        if name.startswith(prefixes):
            continue
        assert torch.equal(value, before[name]), name
    assert torch.equal(
        model.skill_adapters[0][-1].bias,
        torch.zeros_like(model.skill_adapters[0][-1].bias))
    assert torch.equal(
        model.skill_adapter_gate_extensions[0][-1].bias,
        torch.zeros_like(
            model.skill_adapter_gate_extensions[0][-1].bias))


def test_pair_loss_uses_every_event() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16,), skill_adapter_gate_mode="relu")
    batch = generate_lifetimes(
        8, 6, seed=9401, task="pair_relation",
        appearance="diamonds")
    loss, accuracy = _pair_loss(model, batch, exploration=0.1)
    assert loss.ndim == 0
    assert 0.0 <= accuracy <= 1.0


def test_appearance_blend_has_exact_endpoints() -> None:
    bars = generate_lifetimes(
        8, 6, seed=9402, task="pair_relation", appearance="bars")
    blend_zero = generate_lifetimes(
        8, 6, seed=9402, task="pair_relation",
        appearance="diamonds", appearance_blend=0.0)
    diamonds = generate_lifetimes(
        8, 6, seed=9402, task="pair_relation", appearance="diamonds")
    blend_one = generate_lifetimes(
        8, 6, seed=9402, task="pair_relation",
        appearance="bars", appearance_blend=1.0)
    assert torch.equal(bars.frames, blend_zero.frames)
    assert torch.equal(diamonds.frames, blend_one.frames)
    assert torch.equal(bars.correct_actions, diamonds.correct_actions)


def test_mixture_concatenation_preserves_every_field() -> None:
    bars = generate_lifetimes(
        4, 6, seed=9403, task="pair_relation", appearance="bars")
    diamonds = generate_lifetimes(
        6, 6, seed=9404, task="pair_relation", appearance="diamonds")
    combined = _concatenate([bars, diamonds])
    assert combined.batch_size == 10
    assert combined.context_ids is not None
    assert torch.equal(
        combined.correct_actions[:4], bars.correct_actions)
    assert torch.equal(
        combined.correct_actions[4:], diamonds.correct_actions)


def test_training_leak_does_not_change_checkpoint_architecture() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16,), skill_adapter_gate_mode="relu")
    keys = tuple(model.state_dict())
    model.skill_adapter_gate_leak = 0.25
    assert tuple(model.state_dict()) == keys
    model.skill_adapter_gate_leak = 0.0


def test_locality_excludes_mastered_relation_stream() -> None:
    bars = torch.tensor(100.0)
    diamonds = torch.tensor(200.0)
    unrelated = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    specs = _replay_specs("dot_pairs")
    assert torch.equal(
        _unrelated_locality(
            [bars, diamonds, *unrelated], specs), torch.tensor(2.0))


def test_dot_pair_bridge_rehearses_every_mastered_relation_form() -> None:
    assert _replay_specs("diamonds")[:1] == (
        ("pair_relation", "bars"),)
    assert _replay_specs("dot_pairs")[:2] == (
        ("pair_relation", "bars"),
        ("pair_relation", "diamonds"),
    )
    assert _replay_specs("dot_pairs", ("bars",))[:1] == (
        ("pair_relation", "bars"),)


def test_zero_output_gate_refiner_is_bit_identical() -> None:
    base_configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "skill_adapter_widths": (16,),
        "skill_adapter_gate_mode": "relu",
    }
    base = UnifiedCognitiveController(**base_configuration)
    expanded = UnifiedCognitiveController(
        **base_configuration,
        skill_adapter_gate_refiner_widths=(8,))
    missing, unexpected = expanded.load_state_dict(
        base.state_dict(), strict=False)
    assert missing
    assert all(
        name.startswith("skill_adapter_gate_refiners.0.")
        for name in missing)
    assert not unexpected
    batch = generate_lifetimes(
        8, 6, seed=9405, task="pair_relation")
    from .train import rollout
    base_result = rollout(
        base, batch, sample_actions=False, feedback_trials=1)
    expanded_result = rollout(
        expanded, batch, sample_actions=False, feedback_trials=1)
    assert torch.equal(base_result["logits"], expanded_result["logits"])


def test_zero_output_gate_extension_is_bit_identical() -> None:
    base_configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "skill_adapter_widths": (16,),
        "skill_adapter_gate_mode": "relu",
        "skill_adapter_gate_refiner_widths": (8,),
    }
    base = UnifiedCognitiveController(**base_configuration)
    expanded = UnifiedCognitiveController(
        **base_configuration,
        skill_adapter_gate_extension_widths=(8,))
    missing, unexpected = expanded.load_state_dict(
        base.state_dict(), strict=False)
    assert missing
    assert all(
        name.startswith("skill_adapter_gate_extensions.0.")
        for name in missing)
    assert not unexpected
    batch = generate_lifetimes(
        8, 6, seed=9406, task="pair_relation",
        appearance="dot_pairs")
    from .train import rollout
    base_result = rollout(
        base, batch, sample_actions=False, feedback_trials=1)
    expanded_result = rollout(
        expanded, batch, sample_actions=False, feedback_trials=1)
    assert torch.equal(base_result["logits"], expanded_result["logits"])


def test_replay_priority_is_uniform_at_zero_temperature() -> None:
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    loss, weights = _prioritized_replay_loss(losses, temperature=0.0)
    assert torch.equal(loss, torch.tensor(2.0))
    assert torch.allclose(weights, torch.full((3,), 1 / 3))


def test_replay_priority_concentrates_on_largest_detached_loss() -> None:
    losses = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(3.0, requires_grad=True),
    ]
    loss, weights = _prioritized_replay_loss(losses, temperature=0.1)
    assert weights[1] > 0.99
    loss.backward()
    assert losses[1].grad is not None
    assert float(losses[1].grad) > 0.99


def test_replay_base_weights_reallocate_without_changing_scale() -> None:
    losses = [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)]
    loss, weights = _prioritized_replay_loss(
        losses, temperature=0.0, base_weights=(1.0, 2.0, 1.0))
    assert torch.equal(loss, torch.tensor(1.0))
    assert torch.allclose(
        weights, torch.tensor((0.25, 0.5, 0.25)))
