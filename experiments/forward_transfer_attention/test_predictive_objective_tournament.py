import torch

from .train_identify_then_act import ActionHistoryCore
from .train_predictive_objective_tournament import (
    CANDIDATES,
    dynamics_features,
    refine_core,
)


def test_population_has_eight_unique_reward_free_objectives() -> None:
    assert len(CANDIDATES) == 8
    assert len({candidate.name for candidate in CANDIDATES}) == 8


def test_no_refinement_preserves_exact_core() -> None:
    core = ActionHistoryCore(64)
    visual = torch.randn(8, 4, 64)
    transitions = torch.randint(0, 3, (8, 3))
    previous = torch.randint(0, 3, (8, 3))
    candidate = CANDIDATES[0]
    refined, report = refine_core(
        core, candidate, visual, transitions, previous,
        steps=16, batch_size=8, learning_rate=1e-4, seed=19)
    assert report["optimizer_updates"] == 0
    for expected, actual in zip(
            core.state_dict().values(), refined.state_dict().values()):
        assert torch.equal(expected, actual)


def test_cached_dynamics_features_have_expected_shape_and_gradients() -> None:
    core = ActionHistoryCore(64)
    visual = torch.randn(5, 3, 64)
    previous = torch.randint(0, 3, (5, 3))
    features = dynamics_features(core, visual, previous)
    assert features.shape == (5, 192)
    features.sum().backward()
    assert core.recurrent.weight_ih_l0.grad is not None


def test_refinement_never_changes_frozen_vision() -> None:
    core = ActionHistoryCore(64)
    initial = {
        key: value.detach().clone()
        for key, value in core.vision.state_dict().items()}
    visual = torch.randn(8, 4, 64)
    transitions = torch.randint(0, 2, (8, 3))
    previous = torch.randint(0, 3, (8, 3))
    refined, report = refine_core(
        core, CANDIDATES[2], visual, transitions, previous,
        steps=2, batch_size=8, learning_rate=1e-4, seed=19)
    assert report["optimizer_updates"] == 2
    for key, value in refined.vision.state_dict().items():
        assert torch.equal(initial[key], value)
