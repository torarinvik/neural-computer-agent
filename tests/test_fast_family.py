"""The fast verifier must reproduce the reference semantics.

Each test mirrors a semantic unit test from test_game_family.py by
setting the state tensors directly, so the pinned behaviour is the
reference implementation's, not the fast module's own."""

import pytest
import torch

from experiments.games_amodal.fast_family import FastFamilyVerifier
from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier


def _mk(config, batch=1, seed=3):
    v = FastFamilyVerifier(config, batch_size=batch, seed=seed)
    v.reset(seed=seed)
    return v


def test_collect_rewards_and_respawns() -> None:
    v = _mk(FamilyConfig(collect=1))
    v.food[0, 0] = torch.tensor([4, 4])
    v.avatar[0] = torch.tensor([4, 3])
    out = v.step(torch.tensor([1]))
    assert float(out.reward[0]) == 1.0
    assert not torch.equal(v.food[0, 0], torch.tensor([4, 4]))


def test_hazard_contact_kills() -> None:
    v = _mk(FamilyConfig(avoid=1))
    v.hazards[0, 0] = torch.tensor([5, 3, 1])  # will move to (5, 4)
    v.avatar[0] = torch.tensor([4, 4])
    out = v.step(torch.tensor([2]))  # avatar steps down onto (5, 4)
    assert float(out.reward[0]) == -1.0
    assert not bool(out.alive[0])


def test_pursuer_moves_toward_avatar_and_kills() -> None:
    v = _mk(FamilyConfig(pursue=1))
    v.pursuers[0, 0] = torch.tensor([0, 0])
    v.avatar[0] = torch.tensor([4, 1])
    v.step(torch.tensor([2]))  # avatar to (5, 1); row gap larger
    assert torch.equal(v.pursuers[0, 0], torch.tensor([1, 0]))
    v.pursuers[0, 0] = torch.tensor([5, 3])
    v.avatar[0] = torch.tensor([5, 5])
    out = v.step(torch.tensor([3]))  # avatar to (5, 4); pursuer follows
    assert float(out.reward[0]) == -1.0
    assert not bool(out.alive[0])


def test_intercept_catch_and_miss() -> None:
    v = _mk(FamilyConfig(intercept=1))
    v.fallers[0, 0] = torch.tensor([7, 2])
    v.avatar[0] = torch.tensor([7, 1])
    out = v.step(torch.tensor([1]))  # under the landing faller
    assert float(out.reward[0]) == 1.0
    assert bool(out.alive[0])
    v.fallers[0, 0] = torch.tensor([7, 0])
    v.avatar[0] = torch.tensor([6, 6])
    out = v.step(torch.tensor([0]))
    assert float(out.reward[0]) == -1.0
    assert not bool(out.alive[0])


def test_delayed_switch_pays_exactly_k_steps_later() -> None:
    v = _mk(FamilyConfig(delayed=3))
    v.switch[0] = torch.tensor([4, 4])
    v.avatar[0] = torch.tensor([4, 3])
    rewards = [float(v.step(torch.tensor([1])).reward[0])]
    for _ in range(4):
        rewards.append(float(v.step(torch.tensor([0])).reward[0]))
    assert rewards[0] == 0.0
    assert rewards[3] == 1.0
    assert sum(rewards) == 1.0
    assert not torch.equal(v.switch[0], torch.tensor([4, 4]))


def test_resource_gates_food() -> None:
    v = _mk(FamilyConfig(collect=1, resource=1))
    v.food[0, 0] = torch.tensor([4, 4])
    v.resources[0, 0] = torch.tensor([0, 0])
    v.avatar[0] = torch.tensor([4, 3])
    out = v.step(torch.tensor([1]))
    assert float(out.reward[0]) == 0.0  # unfueled
    v.food[0, 0] = torch.tensor([4, 5])
    v.holding[0] = 1
    out = v.step(torch.tensor([1]))
    assert float(out.reward[0]) == 1.0
    assert int(v.holding[0]) == 0


def test_bait_pays_a_fifth_and_respawns_beside_a_hazard() -> None:
    v = _mk(FamilyConfig(avoid=1, deceptive=1))
    v.hazards[0, 0] = torch.tensor([0, 0, 1])
    v.bait[0, 0] = torch.tensor([4, 4])
    v.avatar[0] = torch.tensor([4, 3])
    out = v.step(torch.tensor([1]))
    assert float(out.reward[0]) == pytest.approx(0.2)
    fresh = v.bait[0, 0]
    hazard = v.hazards[0, 0, :2]
    assert int((fresh - hazard).abs().max()) <= 1


def test_unsupported_components_refuse_construction() -> None:
    for config in (FamilyConfig(navigate=True), FamilyConfig(forage=1),
                   FamilyConfig(avoid=1, blink=1),
                   FamilyConfig(oneway=1),
                   FamilyConfig(avoid=1, lever=1)):
        with pytest.raises(ValueError, match="fast verifier"):
            FastFamilyVerifier(config, batch_size=1)


def test_random_policy_distributions_match_reference() -> None:
    """Distributional cross-check: mean random-policy return within a
    tolerant band of the reference implementation's."""

    for config in (FamilyConfig(collect=2), FamilyConfig(avoid=2),
                   FamilyConfig(pursue=1),
                   FamilyConfig(collect=1, resource=1)):
        totals = {}
        for name, cls in (("ref", FamilyVerifier),
                          ("fast", FastFamilyVerifier)):
            v = cls(config, batch_size=256, seed=11)
            v.reset(seed=11)
            g = torch.Generator().manual_seed(99)
            total = torch.zeros(256)
            for _ in range(12):
                total += v.step(torch.randint(0, 4, (256,),
                                              generator=g)).reward
            totals[name] = float(total.mean())
        assert abs(totals["ref"] - totals["fast"]) < 0.25, (
            config, totals)
