import torch

from experiments.games_amodal.ewc_plasticity import (
    controller_named_parameters,
    estimate_diagonal_fisher,
    permuted_fisher_like,
)
from experiments.games_amodal.shared_controller import SharedControllerAgent


def _agent() -> SharedControllerAgent:
    torch.manual_seed(0)
    return SharedControllerAgent(
        event_width=32, intention_width=16, feedback_width=8, hidden=16
    )


def test_fisher_is_nonnegative_unit_mean_and_parameter_shaped() -> None:
    agent = _agent()
    fisher = estimate_diagonal_fisher(
        agent, "snake", batches=2, batch_size=4, steps=8, seed=1, gamma=0.9
    )
    named = dict(controller_named_parameters(agent))
    assert set(fisher) == set(named)
    for name, tensor in fisher.items():
        assert tensor.shape == named[name].shape
        assert bool((tensor >= 0).all())
    total = sum(t.sum() for t in fisher.values())
    count = sum(t.numel() for t in fisher.values())
    assert abs(float(total / count) - 1.0) < 1e-4


def test_fisher_estimation_leaves_parameters_and_grads_clean() -> None:
    agent = _agent()
    before = [p.detach().clone() for p in agent.controller.parameters()]
    estimate_diagonal_fisher(
        agent, "snake", batches=1, batch_size=4, steps=8, seed=2, gamma=0.9
    )
    for old, new in zip(before, agent.controller.parameters(), strict=True):
        assert torch.equal(old, new)
    assert all(p.grad is None for p in agent.controller.parameters())


def test_permuted_fisher_preserves_values_but_not_assignment() -> None:
    agent = _agent()
    fisher = estimate_diagonal_fisher(
        agent, "snake", batches=2, batch_size=4, steps=8, seed=3, gamma=0.9
    )
    permuted = permuted_fisher_like(fisher, seed=4)
    original = torch.cat([t.flatten() for t in fisher.values()])
    shuffled = torch.cat([t.flatten() for t in permuted.values()])
    assert torch.allclose(
        original.sort().values, shuffled.sort().values
    )
    assert not torch.equal(original, shuffled)
