"""Tests for the numerosity transfer learning path."""
from __future__ import annotations

import torch

from .audit_pair_numerosity_continuation import _parse_values
from .model import UnifiedCognitiveController
from .train_pair_numerosity_transfer import (
    _build_student, _retained_within_parent_floor)


def test_slot_specific_read_ablation_preserves_older_slots() -> None:
    torch.manual_seed(23400)
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(8, 8),
        skill_adapter_reads_prior=True,
        skill_adapter_reads_prior_from=1,
        skill_adapter_prior_read_limit=1)
    with torch.no_grad():
        for parameter in model.skill_adapters.parameters():
            parameter.normal_(0.0, 0.2)
        for gate in model.skill_adapter_gates:
            gate.weight.zero_()
            gate.bias.fill_(1.0)
    frames = torch.randn(4, 3, 32, 32)
    state = model.initial_state(4, device=torch.device("cpu"))
    actions = torch.full((4,), 2, dtype=torch.long)
    rewards = torch.zeros(4)
    feedback = torch.zeros(4, dtype=torch.bool)
    normal = model.step(
        frames, state, actions, rewards, feedback)[0].logits

    model.skill_adapter_ablate_prior_read_slot = 0
    irrelevant_ablation = model.step(
        frames, state, actions, rewards, feedback)[0].logits
    assert torch.equal(normal, irrelevant_ablation)

    model.skill_adapter_ablate_prior_read_slot = 1
    matched_ablation = model.step(
        frames, state, actions, rewards, feedback)[0].logits
    assert not torch.equal(normal, matched_ablation)


def test_continuation_reuses_the_existing_final_slot_bit_identically() -> None:
    parent = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(8, 8),
        skill_adapter_reads_prior=True,
        skill_adapter_reads_prior_from=1)
    payload = {
        "model_configuration": {
            "width": 32,
            "workspace_slots": 4,
            "intention_width": 8,
            "skill_adapter_widths": (8, 8),
            "skill_adapter_reads_prior": True,
            "skill_adapter_reads_prior_from": 1,
        },
        "state_dict": parent.state_dict(),
    }
    student, configuration, slot, prefixes = _build_student(
        payload, device=torch.device("cpu"), slot_width=8,
        continue_last_slot=True)
    assert configuration["skill_adapter_widths"] == (8, 8)
    assert slot == 1
    assert prefixes == (
        "skill_adapters.1.", "skill_adapter_gates.1.",
        "skill_adapter_gate_refiners.1.",
        "skill_adapter_gate_extensions.1.",
        "skill_adapter_read_projections.1.")
    for name, value in parent.state_dict().items():
        assert torch.equal(value, student.state_dict()[name])


def test_first_numerosity_rung_appends_exactly_one_zero_output_slot() -> None:
    parent = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(8,))
    payload = {
        "model_configuration": {
            "width": 32,
            "workspace_slots": 4,
            "intention_width": 8,
            "skill_adapter_widths": (8,),
        },
        "state_dict": parent.state_dict(),
    }
    student, configuration, slot, _ = _build_student(
        payload, device=torch.device("cpu"), slot_width=8,
        continue_last_slot=False)
    assert configuration["skill_adapter_widths"] == (8, 8)
    assert slot == 1
    assert torch.count_nonzero(
        student.skill_adapters[1][-1].weight) == 0
    assert torch.count_nonzero(
        student.skill_adapters[1][-1].bias) == 0


def test_retention_floor_is_matched_per_stream() -> None:
    parent = {
        "a": {"overall_accuracy": 0.91},
        "b": {"overall_accuracy": 0.96},
    }
    assert _retained_within_parent_floor(
        {
            "a": {"overall_accuracy": 0.90},
            "b": {"overall_accuracy": 0.941},
        },
        parent)
    assert not _retained_within_parent_floor(
        {
            "a": {"overall_accuracy": 0.90},
            "b": {"overall_accuracy": 0.939},
        },
        parent)


def test_continuation_audit_values_are_strictly_validated() -> None:
    assert _parse_values("0.224,0.23") == (0.224, 0.23)
    for invalid in ("", "0.2,0.2", "-0.1", "1.1"):
        try:
            _parse_values(invalid)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid audit values {invalid!r}")


def test_late_intention_read_preserves_older_slot_shapes() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16, 16),
        skill_adapter_gate_mode="relu",
        skill_adapter_reads_intention_from=1)
    assert model.skill_adapters[0][0].in_features == 64
    assert model.skill_adapters[1][0].in_features == 72
    assert model.skill_adapter_gates[0].in_features == 64
    assert model.skill_adapter_gates[1].in_features == 72


def test_late_multiplicative_intention_binding_is_behavior_preserving() -> None:
    base = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    bound = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16,),
        skill_adapter_gate_mode="relu",
        skill_adapter_reads_intention_from=0,
        skill_adapter_multiplies_intention_from=0)
    missing, unexpected = bound.load_state_dict(
        base.state_dict(), strict=False)
    assert missing
    assert not unexpected
    assert bound.skill_adapters[0][0].in_features == 80
    assert isinstance(
        bound.skill_adapter_intention_interactions[0], torch.nn.Linear)
    assert bound.skill_adapter_intention_interactions[0].in_features == 32
    assert bound.skill_adapter_intention_interactions[0].out_features == 8

    frames = torch.randn(32, 3, 32, 32)
    actions = torch.full((32,), 2, dtype=torch.long)
    zeros = torch.zeros(32)
    base_output = base.step(
        frames, base.initial_state(32, device="cpu"),
        actions, zeros, zeros)[0]
    bound_output = bound.step(
        frames, bound.initial_state(32, device="cpu"),
        actions, zeros, zeros)[0]
    assert torch.equal(bound_output.logits, base_output.logits)


def test_late_hidden_gate_preserves_older_gate_shapes() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16, 16),
        skill_adapter_gate_mode="relu",
        skill_adapter_gate_hidden=12,
        skill_adapter_gate_hidden_from=1)
    assert isinstance(model.skill_adapter_gates[0], torch.nn.Linear)
    assert isinstance(model.skill_adapter_gates[1], torch.nn.Sequential)
    assert model.skill_adapter_gates[1][0].out_features == 12


def test_action_adapter_canonicalization_preserves_emitted_logits() -> None:
    configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "action_adapter_width": 16,
        "action_adapter_gated": True,
    }
    legacy = UnifiedCognitiveController(**configuration)
    legacy.action_adapter[-1].bias.data.copy_(
        torch.tensor((0.35, -0.20)))
    canonical = UnifiedCognitiveController(
        **configuration, action_adapter_into_intention=True)
    canonical.load_state_dict(legacy.state_dict())
    frames = torch.randn(32, 3, 32, 32)
    actions = torch.full((32,), 2, dtype=torch.long)
    zeros = torch.zeros(32)

    legacy_output = legacy.step(
        frames, legacy.initial_state(32, device="cpu"),
        actions, zeros, zeros)[0]
    canonical_output = canonical.step(
        frames, canonical.initial_state(32, device="cpu"),
        actions, zeros, zeros)[0]

    assert torch.allclose(
        canonical_output.logits, legacy_output.logits,
        atol=2e-6, rtol=2e-6)
    assert torch.equal(
        canonical_output.logits.argmax(-1),
        legacy_output.logits.argmax(-1))
    assert not torch.equal(
        canonical_output.intention, legacy_output.intention)
