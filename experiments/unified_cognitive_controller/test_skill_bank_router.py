import pytest
import torch

from .audit_skill_bank_reward_router import _random_opaque_keys
from .skill_bank_router import SkillAddressSelector, attempted_outcome_loss


def test_skill_address_selector_broadcasts_and_is_permutation_equivariant():
    torch.manual_seed(93001)
    selector = SkillAddressSelector(width=5, hidden=8)
    with torch.no_grad():
        selector.net[-1].weight.normal_()
        selector.net[-1].bias.normal_()
    query = torch.randn(3, 5)
    keys = torch.randn(4, 5)
    permutation = torch.tensor([2, 0, 3, 1])
    scores = selector(query, keys)
    permuted = selector(query, keys[permutation])
    assert scores.shape == (3, 4)
    assert torch.allclose(permuted, scores[:, permutation])


def test_skill_address_selector_starts_neutral_but_has_live_gradient():
    selector = SkillAddressSelector(width=4, hidden=8)
    query = torch.randn(6, 4)
    keys = torch.randn(3, 4)
    logits = selector(query, keys)
    assert torch.equal(logits, torch.zeros_like(logits))
    loss = attempted_outcome_loss(
        logits, torch.tensor([0, 1, 2, 0, 1, 2]),
        torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    loss.backward()
    assert selector.net[0].weight.grad is not None
    assert selector.net[-1].weight.grad is not None


def test_attempted_outcome_loss_rejects_invalid_transitions():
    selector = SkillAddressSelector(width=3, hidden=4)
    logits = selector(torch.randn(2, 3), torch.randn(2, 2, 3))
    with pytest.raises(ValueError, match="out of range"):
        attempted_outcome_loss(logits, torch.tensor([0, 2]),
                               torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="binary"):
        attempted_outcome_loss(logits, torch.tensor([0, 1]),
                               torch.tensor([1.0, 0.5]))


def test_random_opaque_keys_are_deterministic_normalized_and_unaligned():
    first = _random_opaque_keys(3, 7, seed=93002, device=torch.device("cpu"))
    second = _random_opaque_keys(3, 7, seed=93002, device=torch.device("cpu"))
    other = _random_opaque_keys(3, 7, seed=93003, device=torch.device("cpu"))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.allclose(first.norm(dim=-1), torch.ones(3))
