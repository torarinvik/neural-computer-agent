import torch

from .train_identify_then_act import (
    NULL_ACTION,
    ActionHistoryCore,
    DirectSuccessSystem,
    decision_features,
    identify_batch,
    make_readout,
)


def test_generator_is_deterministic_and_balanced() -> None:
    first = identify_batch(101_000_000, 16, heldout=False)
    second = identify_batch(101_000_000, 16, heldout=False)
    assert torch.equal(first["frames"], second["frames"])
    assert torch.bincount(
        first["probe_actions"], minlength=2).tolist() == [8, 8]
    assert torch.bincount(
        first["attempted_actions"], minlength=2).tolist() == [8, 8]
    assert torch.bincount(
        first["correct_actions"], minlength=2).tolist() == [8, 8]


def test_protocol_swap_is_a_valid_causal_rerender() -> None:
    normal = identify_batch(105_000_000, 16, heldout=True)
    swapped = identify_batch(
        105_000_000, 16, heldout=True, swap_protocol=True)
    assert torch.equal(normal["frames"][:, 0], swapped["frames"][:, 0])
    assert not torch.equal(normal["frames"][:, 1], swapped["frames"][:, 1])
    assert torch.equal(normal["frames"][:, 2], swapped["frames"][:, 2])
    assert torch.all(
        normal["correct_actions"] != swapped["correct_actions"])
    assert torch.equal(normal["probe_actions"], swapped["probe_actions"])


def test_target_reversal_preserves_probe_and_flips_answer() -> None:
    normal = identify_batch(105_000_000, 16, heldout=True)
    reversed_target = identify_batch(
        105_000_000, 16, heldout=True, reverse_target=True)
    assert torch.equal(
        normal["frames"][:, :2], reversed_target["frames"][:, :2])
    assert not torch.equal(
        normal["frames"][:, 2], reversed_target["frames"][:, 2])
    assert torch.all(
        normal["correct_actions"] !=
        reversed_target["correct_actions"])


def test_missing_consequence_removes_only_probe_effect() -> None:
    normal = identify_batch(105_000_000, 16, heldout=True)
    missing = identify_batch(
        105_000_000, 16, heldout=True, missing_consequence=True)
    assert torch.equal(normal["frames"][:, 0], missing["frames"][:, 0])
    assert not torch.equal(normal["frames"][:, 1], missing["frames"][:, 1])
    assert torch.equal(normal["frames"][:, 2], missing["frames"][:, 2])
    assert torch.equal(
        normal["correct_actions"], missing["correct_actions"])


def test_previous_action_alignment_and_feature_shape() -> None:
    data = identify_batch(105_000_000, 8, heldout=True)
    assert torch.all(data["previous_actions"][:, 0] == NULL_ACTION)
    assert torch.equal(
        data["previous_actions"][:, 1], data["probe_actions"])
    assert torch.all(data["previous_actions"][:, 2] == NULL_ACTION)
    core = ActionHistoryCore(hidden=8, action_width=4)
    features = decision_features(
        core, data, passive=False, device=torch.device("cpu"))
    assert features.shape == (8, 24)


def test_logging_choices_do_not_depend_on_correct_answer() -> None:
    normal = identify_batch(105_000_000, 16, heldout=True)
    reversed_target = identify_batch(
        105_000_000, 16, heldout=True, reverse_target=True)
    assert torch.equal(
        normal["attempted_actions"],
        reversed_target["attempted_actions"])
    assert torch.all(
        normal["correct_actions"] !=
        reversed_target["correct_actions"])


def test_curriculum_can_fix_probe_and_protocol_independently() -> None:
    direct = identify_batch(
        105_000_000, 16, heldout=True,
        fixed_protocol=0, fixed_probe_action=0)
    fixed_probe = identify_batch(
        105_000_000, 16, heldout=True, fixed_probe_action=0)
    assert torch.all(direct["probe_actions"] == 0)
    assert torch.all(fixed_probe["probe_actions"] == 0)
    assert torch.all(direct["private_protocol_ids"] == 0)
    assert torch.bincount(
        fixed_probe["private_protocol_ids"], minlength=2).tolist() == [8, 8]


def test_gradual_probe_curriculum_has_exact_train_mixture() -> None:
    eighth = identify_batch(
        105_000_000, 16, heldout=False,
        probe_action_one_fraction=0.125)
    quarter = identify_batch(
        105_000_000, 16, heldout=False,
        probe_action_one_fraction=0.25)
    assert torch.bincount(
        eighth["probe_actions"], minlength=2).tolist() == [14, 2]
    assert torch.bincount(
        quarter["probe_actions"], minlength=2).tolist() == [12, 4]


def test_fixed_target_rung_keeps_binding_variable() -> None:
    data = identify_batch(
        105_000_000, 16, heldout=True, fixed_target_direction=-1)
    assert torch.bincount(
        data["probe_actions"], minlength=2).tolist() == [8, 8]
    assert torch.bincount(
        data["private_protocol_ids"], minlength=2).tolist() == [8, 8]
    assert torch.equal(
        data["correct_actions"], data["private_protocol_ids"])


def test_direct_readout_preserves_two_opaque_outputs() -> None:
    model = make_readout("direct", hidden=192, intention_width=64)
    assert isinstance(model, DirectSuccessSystem)
    assert model(torch.zeros(3, 192)).shape == (3, 2)
