"""Tests for the numerosity transfer learning path."""
from __future__ import annotations

import torch

from .model import UnifiedCognitiveController


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
