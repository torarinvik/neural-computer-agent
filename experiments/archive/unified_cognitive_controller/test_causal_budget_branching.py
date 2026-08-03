import torch

from .causal_budget_branching import (
    BranchState,
    branch_state_digest,
    higher_budget_label,
    state_digest,
)
from .train_option_composition_race import OptionValueHead


def _outcome(bits: int | None, utility: float) -> dict[str, object]:
    return {"stable_target_bits": bits, "final_utility": utility}


def test_higher_budget_requires_causal_sample_saving_and_capability_guard() -> None:
    label, detail = higher_budget_label(
        _outcome(960, 0.89), _outcome(720, 0.888),
        capability_tolerance=0.003)
    assert label
    assert detail["saves_experience"]
    label, detail = higher_budget_label(
        _outcome(960, 0.89), _outcome(720, 0.885),
        capability_tolerance=0.003)
    assert not label
    assert not detail["keeps_capability"]
    label, _ = higher_budget_label(
        _outcome(960, 0.89), _outcome(960, 0.90),
        capability_tolerance=0.003)
    assert not label
    label, detail = higher_budget_label(
        _outcome(None, 0.50), _outcome(None, 0.50),
        capability_tolerance=0.003)
    assert label is None
    assert not detail["eligible_for_allocation"]


def test_branch_clone_is_independent_but_starts_bit_identical() -> None:
    router = OptionValueHead(7, 8)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(7)
    state = BranchState(
        router, optimizer,
        [(torch.randn(3, 7), torch.randn(3))], generator)
    clone = state.clone()
    assert state_digest(clone.router) == state_digest(state.router)
    assert branch_state_digest(clone) == branch_state_digest(state)
    with torch.no_grad():
        next(clone.router.parameters()).add_(1.0)
    assert state_digest(clone.router) != state_digest(state.router)
    assert branch_state_digest(clone) != branch_state_digest(state)
    assert clone.generator.initial_seed() == state.generator.initial_seed()
