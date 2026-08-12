from __future__ import annotations

import torch

from neural_computer import (
    GatedResidualRegimeChangePolicy,
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
