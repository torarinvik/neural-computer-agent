from __future__ import annotations

import pytest
import torch

from neural_computer import (
    FactorizedOpaqueAddressRouter,
    OpaqueAddressRouter,
    attempted_outcome_loss,
)


def test_opaque_address_router_is_row_permutation_equivariant() -> None:
    router = OpaqueAddressRouter(width=4, hidden=8)
    query = torch.randn(3, 4)
    keys = torch.randn(2, 4)
    permutation = torch.tensor([1, 0])

    scores = router(query, keys)
    permuted_scores = router(query, keys[permutation])

    assert torch.allclose(permuted_scores, scores[:, permutation])


def test_attempted_outcome_loss_rejects_invalid_rows_and_nonbinary_outcomes() -> None:
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="out of range"):
        attempted_outcome_loss(logits, torch.tensor([0, 3]), torch.ones(2))
    with pytest.raises(ValueError, match="binary"):
        attempted_outcome_loss(logits, torch.tensor([0, 1]), torch.tensor([0.0, 0.5]))


def test_factorized_opaque_address_router_is_row_permutation_equivariant() -> None:
    router = FactorizedOpaqueAddressRouter(width=4, hidden=8)
    query = torch.randn(3, 4)
    keys = torch.randn(2, 4)
    permutation = torch.tensor([1, 0])

    scores = router(query, keys)
    permuted_scores = router(query, keys[permutation])

    assert torch.allclose(permuted_scores, scores[:, permutation])
