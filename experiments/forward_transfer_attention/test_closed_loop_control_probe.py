import torch

from .probe_closed_loop_control import oracle_labels
from .train_actuator_transfer import SuccessSystem
from .train_closed_loop_intercept import (
    decision_features,
    execute_policy,
    trajectory_batch,
)
from .train_micro_intercept import InterceptPredictiveCore
from .train_micro_intercept import protocol_for_seed


def test_oracle_labels_are_deterministic_and_valid() -> None:
    protocol = protocol_for_seed(211)
    data = trajectory_batch(
        93_000_000, 6, heldout=False, protocol=protocol, horizon=4)
    first = oracle_labels(data["private_states"], protocol)
    second = oracle_labels(data["private_states"], protocol)
    assert torch.equal(first, second)
    assert first.shape == data["actions"].shape
    assert first.min() >= 0
    assert first.max() < 3


def test_oracle_uses_only_verifier_private_state() -> None:
    protocol = protocol_for_seed(211)
    data = trajectory_batch(
        93_000_000, 6, heldout=False, protocol=protocol, horizon=3)
    original = oracle_labels(data["private_states"], protocol)
    changed = data["private_states"].clone()
    changed[:, :, 1] *= -1
    reversed_labels = oracle_labels(changed, protocol)
    assert not torch.equal(original, reversed_labels)


def test_diagnostic_rollout_returns_aligned_states_and_frames() -> None:
    protocol = protocol_for_seed(211)
    core = InterceptPredictiveCore(hidden=8, action_width=4)
    model = SuccessSystem(32, 4, actions=3)
    rollout = execute_policy(
        core, model, start=97_000_000, count=6, protocol=protocol,
        horizon=3, passive=False, device=torch.device("cpu"),
        return_diagnostic_data=True)
    frames = rollout["diagnostic_frames"]
    private = rollout["diagnostic_private_states"]
    assert frames.shape[:2] == private.shape[:2] == (6, 3)
    features = decision_features(
        core, frames, passive=False, device=torch.device("cpu"))
    assert features.shape == (6, 3, 32)
