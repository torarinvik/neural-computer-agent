import torch

from .train_closed_loop_intercept import (
    _initial_private_state,
    protocol_for_seed,
    trajectory_batch,
)


def test_trajectory_is_deterministic_and_balanced() -> None:
    protocol = protocol_for_seed(211)
    first = trajectory_batch(
        91_000_000, 12, heldout=False, protocol=protocol)
    second = trajectory_batch(
        91_000_000, 12, heldout=False, protocol=protocol)
    assert torch.equal(first["frames"], second["frames"])
    assert torch.equal(first["actions"], second["actions"])
    for step in range(first["actions"].shape[1]):
        assert torch.bincount(
            first["actions"][:, step], minlength=3).tolist() == [4, 4, 4]


def test_actions_causally_change_later_pixels() -> None:
    protocol = protocol_for_seed(211)
    active = trajectory_batch(
        91_000_000, 12, heldout=False, protocol=protocol)
    no_effect = trajectory_batch(
        91_000_000, 12, heldout=False, protocol=protocol, no_effect=True)
    assert torch.equal(active["frames"][:, 0], no_effect["frames"][:, 0])
    assert not torch.equal(active["frames"][:, -1], no_effect["frames"][:, -1])


def test_missing_motion_removes_early_target_evidence() -> None:
    protocol = protocol_for_seed(211)
    visible = trajectory_batch(
        91_000_000, 6, heldout=False, protocol=protocol)
    missing = trajectory_batch(
        91_000_000, 6, heldout=False, protocol=protocol,
        missing_motion=True)
    assert not torch.equal(visible["frames"][:, 0], missing["frames"][:, 0])
    assert torch.equal(
        visible["actions"], missing["actions"])


def test_reversal_changes_private_motion_not_start_positions() -> None:
    normal = _initial_private_state(
        95_000_000, 12, True, reverse_motion=False)
    reverse = _initial_private_state(
        95_000_000, 12, True, reverse_motion=True)
    assert normal["target_x"] == reverse["target_x"]
    assert normal["cursor_x"] == reverse["cursor_x"]
    assert all(
        left == -right for left, right in zip(
            normal["target_v"], reverse["target_v"]))


def test_reversal_is_visibly_distinct_after_one_motion_step() -> None:
    protocol = protocol_for_seed(211)
    normal = trajectory_batch(
        95_000_000, 12, heldout=True, protocol=protocol)
    reverse = trajectory_batch(
        95_000_000, 12, heldout=True, protocol=protocol,
        reverse_motion=True)
    assert torch.equal(normal["frames"][:, 0], reverse["frames"][:, 0])
    assert not torch.equal(normal["frames"][:, 1], reverse["frames"][:, 1])


def test_reward_is_for_attempted_transition_only() -> None:
    protocol = protocol_for_seed(211)
    data = trajectory_batch(
        91_000_000, 12, heldout=False, protocol=protocol)
    assert data["rewards"].shape == data["actions"].shape
    assert set(data["rewards"].unique().tolist()).issubset({0.0, 1.0})
