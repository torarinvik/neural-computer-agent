import torch

from .train_identify_then_act import ActionHistoryCore, TEST_START
from .train_learning_mechanism_tournament import (
    BLIND_START,
    CANDIDATES,
    CachedBehaviorSystem,
)


def _candidate(name: str):
    return next(candidate for candidate in CANDIDATES
                if candidate.name == name)


def test_population_has_eight_unique_mechanisms() -> None:
    assert len(CANDIDATES) == 8
    assert len({candidate.name for candidate in CANDIDATES}) == 8


def test_zero_adapter_is_initially_exact_no_op() -> None:
    core = ActionHistoryCore(64)
    frozen = CachedBehaviorSystem(
        core, _candidate("frozen_readout"), initialization_seed=31)
    adapted = CachedBehaviorSystem(
        core, _candidate("residual_rank8"), initialization_seed=31)
    visual = torch.randn(5, 3, 64)
    previous = torch.randint(0, 3, (5, 3))
    assert torch.equal(
        frozen.latent_features(visual, previous),
        adapted.latent_features(visual, previous))
    assert torch.equal(
        frozen(visual, previous), adapted(visual, previous))


def test_frozen_candidate_only_trains_readout() -> None:
    core = ActionHistoryCore(64)
    model = CachedBehaviorSystem(
        core, _candidate("frozen_readout"), initialization_seed=31)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("readout.") for name in trainable)


def test_recurrent_candidate_has_precise_trainable_subset() -> None:
    core = ActionHistoryCore(64)
    model = CachedBehaviorSystem(
        core, _candidate("recurrent_predictor_tail_lr3e5"),
        initialization_seed=31)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(name.startswith("recurrent.") for name in trainable)
    assert not any(name.startswith("action_embedding.") for name in trainable)
    assert {
        name for name in trainable if name.startswith("predictor.")
    } == {"predictor.3.weight", "predictor.3.bias"}


def test_blind_range_is_disjoint_from_selection() -> None:
    assert BLIND_START >= TEST_START + 1_000_000
