import torch

from experiments.games_amodal.environments import PongVerifier
from experiments.games_amodal.game_routing import (
    COMMON_ACTIONS,
    COMMON_CHANNELS,
    PaddedVerifier,
    pad_observation,
    padded_factory,
    slot_candidate_key,
)
from experiments.games_amodal.snake_acquisition import SnakePolicy


def test_pad_observation_adds_zero_planes() -> None:
    observation = torch.rand(2, 2, 8, 8)
    padded = pad_observation(observation)
    assert padded.shape == (2, 3, 8, 8)
    assert torch.equal(padded[:, :2], observation)
    assert float(padded[:, 2].abs().sum()) == 0.0
    assert torch.equal(pad_observation(padded), padded)


def test_padded_verifier_clamps_actions_into_range() -> None:
    verifier = PaddedVerifier(PongVerifier(batch_size=2, seed=1))
    verifier.reset(seed=1)
    assert verifier.action_count == COMMON_ACTIONS
    assert verifier.observation().shape[1] == COMMON_CHANNELS
    outcome = verifier.step(torch.tensor([3, 0]))
    assert outcome.reward.shape == (2,)


def test_padded_factories_share_common_interface() -> None:
    for game in ("snake", "pong"):
        verifier = padded_factory(game)(batch_size=2, seed=4)
        verifier.reset(seed=4)
        assert verifier.observation().shape == (2, COMMON_CHANNELS, 8, 8)


def test_slot_candidate_keys_are_opaque_unit_vectors() -> None:
    torch.manual_seed(0)
    slot = SnakePolicy(
        height=8, width=8, event_width=16, intent_width=8, hidden=16,
        channels=COMMON_CHANNELS, action_count=COMMON_ACTIONS,
    )
    key = slot_candidate_key(
        slot, padded_factory("pong"), batch_size=4, steps=8, seed=5
    )
    assert key.shape == (16,)
    assert abs(float(key.norm()) - 1.0) < 1e-5
