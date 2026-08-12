from __future__ import annotations

import pytest
import torch

from neural_computer import (
    GatedResidualRegimeChangePolicy,
    GatedResidualRegimePolicyBank,
    OpaqueRegimeChangePolicy,
    RegimeChangePlan,
)


def _banks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    current = torch.randn(1, 6, 8, generator=generator)
    incoming = torch.randn(1, 6, 8, generator=generator)
    current_occupied = torch.ones(1, 6, dtype=torch.bool)
    incoming_occupied = torch.ones(1, 6, dtype=torch.bool)
    return current, current_occupied, incoming, incoming_occupied


def test_regime_policy_is_permutation_invariant_over_both_banks() -> None:
    policy = OpaqueRegimeChangePolicy(
        value_width=8,
        hidden=16,
        max_spectral_bins=4,
    ).eval()
    current, current_occupied, incoming, incoming_occupied = _banks()
    current_permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    incoming_permutation = torch.tensor([4, 1, 3, 0, 5, 2])

    original = policy(current, current_occupied, incoming, incoming_occupied)
    permuted = policy(
        current[:, current_permutation],
        current_occupied[:, current_permutation],
        incoming[:, incoming_permutation],
        incoming_occupied[:, incoming_permutation],
    )

    assert torch.allclose(original.logits, permuted.logits)
    assert policy.configuration()["proposal"] == "keep_or_replace_v1"


def test_regime_policy_ignores_unoccupied_padding_and_adapts() -> None:
    policy = OpaqueRegimeChangePolicy(
        value_width=8,
        hidden=16,
        max_spectral_bins=4,
    )
    current, current_occupied, incoming, incoming_occupied = _banks()
    padded_current = torch.cat((current, torch.full((1, 2, 8), 99.0)), dim=1)
    padded_incoming = torch.cat((incoming, torch.full((1, 2, 8), -99.0)), dim=1)
    padded_current_occupied = torch.tensor(
        [[True, True, True, True, True, True, False, False]]
    )
    padded_incoming_occupied = padded_current_occupied.clone()
    compact = policy(current, current_occupied, incoming, incoming_occupied)
    padded = policy(
        padded_current,
        padded_current_occupied,
        padded_incoming,
        padded_incoming_occupied,
    )
    assert torch.allclose(compact.logits, padded.logits)

    plan = policy.propose(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
        explore=True,
        generator=torch.Generator().manual_seed(32),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    loss = policy.adaptation_step(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
        plan,
        1.0,
        optimizer=optimizer,
    )

    assert isinstance(plan, RegimeChangePlan)
    assert torch.isfinite(torch.tensor(loss))


def test_gated_residual_keeps_base_frozen_and_starts_as_base_fallback() -> None:
    base = OpaqueRegimeChangePolicy(
        value_width=8,
        hidden=16,
        max_spectral_bins=4,
    ).eval()
    policy = GatedResidualRegimeChangePolicy(base)
    current, current_occupied, incoming, incoming_occupied = _banks()
    base_plan = base.propose(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
    )
    residual_plan = policy.propose(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
    )

    assert residual_plan.replace == base_plan.replace
    assert all(not parameter.requires_grad for parameter in base.parameters())
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=0.01)
    policy.adaptation_step(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
        residual_plan,
        1.0,
        optimizer=optimizer,
    )

    assert all(torch.equal(value, before[name]) for name, value in base.state_dict().items())
    assert any(
        bool(torch.any(parameter.detach() != 0.0))
        for parameter in policy.residual.parameters()
    )


def test_residual_policy_bank_routes_opaque_bindings_and_isolates_slots() -> None:
    base = OpaqueRegimeChangePolicy(
        value_width=8,
        hidden=16,
        max_spectral_bins=4,
    ).eval()
    bank = GatedResidualRegimePolicyBank(base, context_width=4, max_slots=2)
    key_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    key_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert bank.add_slot(key_a) == 0
    assert bank.add_slot(key_b) == 1
    contexts = torch.stack((key_a, key_b))
    assert torch.equal(bank.route_slot(contexts), torch.tensor([0, 1]))

    current, current_occupied, incoming, incoming_occupied = _banks()
    slot_one_before = {
        name: value.detach().clone()
        for name, value in bank.residual_slots[1].state_dict().items()
    }
    base_before = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    plan = bank.propose(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
        key_a.unsqueeze(0),
        explore=True,
        generator=torch.Generator().manual_seed(32),
    )
    optimizer = torch.optim.Adam(bank.trainable_parameters(0), lr=0.01)
    bank.adaptation_step(
        current,
        current_occupied,
        incoming,
        incoming_occupied,
        key_a.unsqueeze(0),
        0,
        plan,
        1.0,
        optimizer=optimizer,
    )

    assert all(
        torch.equal(value, bank.residual_slots[1].state_dict()[name])
        for name, value in slot_one_before.items()
    )
    assert all(
        torch.equal(value, base.state_dict()[name])
        for name, value in base_before.items()
    )
    bank.freeze_slot(0)
    bank.freeze_slot(1)
    with pytest.raises(RuntimeError, match="frozen"):
        bank.trainable_parameters(0)
    with pytest.raises(RuntimeError, match="capacity"):
        bank.add_slot(torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_residual_policy_bank_replacement_is_copy_on_write_and_verifier_gated() -> None:
    base = OpaqueRegimeChangePolicy(
        value_width=8,
        hidden=16,
        max_spectral_bins=4,
    ).eval()
    bank = GatedResidualRegimePolicyBank(base, context_width=4, max_slots=2)
    key_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    key_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    key_c = torch.tensor([0.0, 0.0, 1.0, 0.0])
    bank.add_slot(key_a)
    bank.add_slot(key_b)
    before = bank.slot_keys[1].detach().clone()
    candidate = bank.slot_replacement_candidate(1, key_c)

    assert torch.equal(bank.slot_keys[1], before)
    assert not bank.replace_slot_from_candidate(candidate, 1, retention_probe=lambda _: False)
    assert torch.equal(bank.slot_keys[1], before)
    assert bank.replace_slot_from_candidate(candidate, 1, retention_probe=lambda _: True)
    assert torch.allclose(bank.slot_keys[1], key_c)
    assert not bool(bank.slot_frozen[1])

    tampered = bank.slot_replacement_candidate(1, key_b)
    with torch.no_grad():
        tampered.base.scorer[0].weight[0, 0] += 1.0
    with pytest.raises(ValueError, match="frozen base"):
        bank.replace_slot_from_candidate(tampered, 1, retention_probe=lambda _: True)
