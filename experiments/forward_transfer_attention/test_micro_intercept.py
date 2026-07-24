import torch

from .train_micro_intercept import (
    InterceptPredictiveCore,
    frozen_decision_features,
    intercept_sequences,
    protocol_for_seed,
    uniform_logged_buffer,
)


def test_intercept_generator_is_balanced_and_deterministic() -> None:
    protocol = protocol_for_seed(211)
    first = intercept_sequences(
        85_000_000, 60, heldout=True, protocol=protocol)
    second = intercept_sequences(
        85_000_000, 60, heldout=True, protocol=protocol)
    assert torch.equal(first["frames"], second["frames"])
    assert torch.bincount(
        first["correct_actions"], minlength=3).tolist() == [20, 20, 20]
    assert torch.bincount(first["actions"], minlength=3).tolist() == [20, 20, 20]


def test_velocity_reversal_preserves_middle_frame_and_changes_motion() -> None:
    protocol = protocol_for_seed(211)
    normal = intercept_sequences(
        85_000_000, 60, heldout=True, protocol=protocol)
    reversed_data = intercept_sequences(
        85_000_000, 60, heldout=True, protocol=protocol,
        reverse_velocity=True)
    assert torch.equal(normal["frames"][:, 1], reversed_data["frames"][:, 1])
    moving = normal["velocities"] != 0
    assert torch.all(
        normal["correct_actions"][moving] !=
        reversed_data["correct_actions"][moving])


def test_mirror_flips_target_motion_but_not_actuator_coordinates() -> None:
    protocol = protocol_for_seed(211)
    normal = intercept_sequences(
        85_000_000, 60, heldout=True, protocol=protocol)
    mirrored = intercept_sequences(
        85_000_000, 60, heldout=True, protocol=protocol, mirror=True)
    moving = normal["velocities"] != 0
    assert torch.all(
        normal["correct_actions"][moving] !=
        mirrored["correct_actions"][moving])
    assert torch.equal(
        normal["correct_actions"][~moving],
        mirrored["correct_actions"][~moving])
    # The logged opaque commands are held fixed, so any reward change is
    # caused by the counterfactual sensory motion rather than a policy change.
    assert torch.equal(normal["actions"], mirrored["actions"])


def test_missing_frames_preserve_remaining_velocity_evidence() -> None:
    protocol = protocol_for_seed(211)
    normal = intercept_sequences(
        85_000_000, 6, heldout=True, protocol=protocol)
    no_first = intercept_sequences(
        85_000_000, 6, heldout=True, protocol=protocol,
        omit_first=True)
    no_second = intercept_sequences(
        85_000_000, 6, heldout=True, protocol=protocol,
        omit_second=True)
    assert torch.equal(no_first["pre_frames"][:, 0],
                       normal["pre_frames"][:, 1])
    assert torch.equal(no_second["pre_frames"][:, 0],
                       normal["pre_frames"][:, 0])


def test_uniform_logging_does_not_consult_correct_action() -> None:
    states = torch.randn(12, 8)
    a = uniform_logged_buffer(
        states, torch.tensor([0, 1, 2] * 4), seed=17)
    b = uniform_logged_buffer(
        states, torch.tensor([2, 1, 0] * 4), seed=17)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[2], b[2])
    assert torch.bincount(a[2], minlength=3).tolist() == [4, 4, 4]
    assert torch.allclose(a[4], torch.full_like(a[4], 1 / 3))


def test_decision_features_include_candidate_action_consequences() -> None:
    protocol = protocol_for_seed(211)
    data = intercept_sequences(
        85_000_000, 6, heldout=True, protocol=protocol)
    core = InterceptPredictiveCore(hidden=8, action_width=4)
    active = frozen_decision_features(
        core, data["pre_frames"], 6, torch.device("cpu"), passive=False)
    passive = frozen_decision_features(
        core, data["pre_frames"], 6, torch.device("cpu"), passive=True)
    assert active.shape == passive.shape == (6, 32)
    # Passive predictions deliberately remove the action signal.
    assert torch.allclose(passive[:, 8:16], passive[:, 16:24])
    assert torch.allclose(passive[:, 16:24], passive[:, 24:32])
    # The action-conditioned path exposes distinct learned consequences.
    assert not torch.allclose(active[:, 8:16], active[:, 16:24])
