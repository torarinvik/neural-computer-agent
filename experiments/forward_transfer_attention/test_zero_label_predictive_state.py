import inspect

import torch

from .train_zero_label_predictive_state import (
    PredictiveStateAgent, _future_targets, _standardized_prediction_loss,
    policy_sequences, predictive_sequences, reinforce_loss)


def test_predictive_sequences_expose_only_pixels():
    frames = predictive_sequences(100, 6)
    assert frames.shape[:2] == (6, 12)
    assert frames.shape[2:] == (3, 96, 160)
    assert frames.dtype == torch.float32
    assert 0.0 <= float(frames.min()) <= float(frames.max()) <= 1.0


def test_policy_private_rule_is_separate_from_sensory_tensor():
    frames, private_rules = policy_sequences(
        200, 6, heldout=False, palettes=((0, 1),))
    assert frames.shape == (6, 5, 3, 96, 160)
    assert private_rules.shape == (6,)
    assert set(private_rules.tolist()) <= {0, 1}
    assert int((private_rules == 0).sum()) == int((private_rules == 1).sum())


def test_reversal_is_valid_pixel_counterfactual_and_flips_private_rule():
    normal, normal_rules = policy_sequences(
        200, 6, heldout=True, palettes=((0, 1),))
    reversed_frames, reversed_rules = policy_sequences(
        200, 6, heldout=True, palettes=((0, 1),), reverse_events=True)
    assert torch.equal(normal[:, 0], reversed_frames[:, 1])
    assert torch.equal(normal[:, 1], reversed_frames[:, 0])
    assert torch.equal(normal[:, 2], reversed_frames[:, 2])
    assert torch.equal(normal[:, 3], reversed_frames[:, 4])
    assert torch.equal(normal[:, 4], reversed_frames[:, 3])
    assert torch.equal(1 - normal_rules, reversed_rules)


def test_support_and_query_reversals_isolate_causal_routes():
    normal, normal_rules = policy_sequences(
        205, 6, heldout=True, palettes=((0, 1),))
    support_reversed, support_rules = policy_sequences(
        205, 6, heldout=True, palettes=((0, 1),),
        reverse_support_only=True)
    query_reversed, query_rules = policy_sequences(
        205, 6, heldout=True, palettes=((0, 1),),
        reverse_query_only=True)

    assert torch.equal(normal[:, 0], support_reversed[:, 1])
    assert torch.equal(normal[:, 1], support_reversed[:, 0])
    assert torch.equal(normal[:, 2:], support_reversed[:, 2:])
    assert torch.equal(1 - normal_rules, support_rules)

    assert torch.equal(normal[:, :3], query_reversed[:, :3])
    assert torch.equal(normal[:, 3], query_reversed[:, 4])
    assert torch.equal(normal[:, 4], query_reversed[:, 3])
    assert torch.equal(normal_rules, query_rules)


def test_shuffled_future_changes_targets_not_online_frames():
    agent = PredictiveStateAgent(hidden=16)
    frames = predictive_sequences(300, 6)
    normal = _future_targets(agent.vision, frames, shuffled=False)
    shuffled = _future_targets(agent.vision, frames, shuffled=True)
    assert torch.equal(shuffled, normal.roll(1, dims=0))
    assert not torch.equal(normal, shuffled)


def test_delta_target_is_frame_to_frame_latent_change():
    agent = PredictiveStateAgent(hidden=16)
    frames = predictive_sequences(301, 6)
    with torch.no_grad():
        encoded = agent.vision(frames.flatten(0, 1)).reshape(6, 12, 16)
        expected = encoded[:, 1:] - encoded[:, :-1]
    actual = _future_targets(
        agent.vision, frames, shuffled=False, target_kind="delta")
    assert torch.allclose(actual, expected)


def test_reward_loss_has_no_correct_action_or_semantic_target_argument():
    parameters = inspect.signature(reinforce_loss).parameters
    assert set(parameters) == {
        "logits", "values", "sampled_actions", "verified_rewards",
        "entropy_weight"}
    logits = torch.zeros(4, 2, requires_grad=True)
    values = torch.zeros(4, requires_grad=True)
    sampled = torch.tensor([0, 1, 0, 1])
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = reinforce_loss(logits, values, sampled, rewards)
    loss.backward()
    assert logits.grad is not None and values.grad is not None


def test_standardized_prediction_distinguishes_correct_pairing():
    target = torch.randn(8, 3, 12)
    paired = target + 0.01 * torch.randn_like(target)
    shuffled = target.roll(1, dims=0)
    assert (
        _standardized_prediction_loss(paired, target) <
        _standardized_prediction_loss(paired, shuffled))
