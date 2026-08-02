"""Small interface tests for the reward probe's amodal policy wrapper."""

import torch

from .train_brainworkshop_policy import (
    BrainWorkshopPolicy, _controller_action_index, _factorized_advantages,
    _history_features, _resolve_rehearsal_weights)
from .brainworkshop_gym import BrainWorkshopConfig


def test_brainworkshop_config_accepts_the_next_fifth_back_rung() -> None:
    BrainWorkshopConfig(n_back=5, trials=8).validate()


def test_dual_policy_uses_one_decoder_and_maps_opaque_masks() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("vision", "audio"),
        retrieved_memory_adapter_width=32,
        external_memory_adapter_width=32,
    )

    assert tuple(policy.encoders) == ("vision", "audio")
    assert policy.decoder.commands == 4
    assert [policy.action_mask(value) for value in range(4)] == [0, 1, 2, 3]


def test_policy_accepts_a_third_token_stream_without_resizing_controller() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("vision", "audio", "text"),
        factorized_output=True,
    )
    assert tuple(policy.encoders) == ("vision", "audio", "text")
    assert policy.controller.width == 32
    assert [policy.action_mask(value) for value in range(8)] == list(range(8))


def test_external_memory_adapter_is_an_exact_initial_noop() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("audio",),
        external_memory_adapter_width=32,
    )
    features = torch.randn(5, 96)
    assert torch.equal(
        policy.external_memory_adapter(features), torch.zeros(5, 32))


def test_deeper_history_appends_opaque_snapshot_relations() -> None:
    current = torch.randn(3, 4)
    previous = torch.randn(3, 4)
    older = torch.randn(3, 4)
    one = _history_features(current, [previous], depth=1)
    two = _history_features(current, [previous, older], depth=2)
    assert one.shape == (3, 12)
    assert two.shape == (3, 20)
    assert torch.equal(two[:, :12], one)
    assert torch.equal(two[:, 12:16], older)
    assert torch.equal(two[:, 16:], current * older)


def test_deeper_ram_and_intention_bridges_remain_zero_initialized() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("text",),
        external_memory_adapter_width=32,
        external_history_depth=2,
        per_stream_external_history=True,
        per_stream_intention_adapter_width=32,
    )
    features = torch.randn(5, 160)
    assert torch.equal(
        policy.external_memory_adapters["text"](features),
        torch.zeros(5, 32))
    assert torch.equal(
        policy.per_stream_intention_adapters["text"](features),
        torch.zeros(5, 16))


def test_per_stream_external_memory_adapters_are_independent_noops() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("vision", "audio"),
        external_memory_adapter_width=32,
        per_stream_external_history=True,
    )
    assert set(policy.external_memory_adapters) == {"vision", "audio"}
    features = torch.randn(5, 96)
    for adapter in policy.external_memory_adapters.values():
        assert torch.equal(adapter(features), torch.zeros(5, 32))


def test_slot_memory_composer_starts_as_the_legacy_mean() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("vision", "audio"),
        external_memory_adapter_width=32,
        per_stream_external_history=True,
        slot_memory_composer=True,
    )
    slots = torch.randn(5, 64)
    expected = slots.view(5, 2, 32).mean(dim=1)
    assert torch.allclose(policy.external_memory_composer(slots), expected)


def test_stream_intention_bridges_start_as_exact_noops() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("vision", "audio"),
        per_stream_intention_adapter_width=32,
    )
    assert set(policy.per_stream_intention_adapters) == {"vision", "audio"}
    features = torch.randn(5, 96)
    for adapter in policy.per_stream_intention_adapters.values():
        assert torch.equal(adapter(features), torch.zeros(5, 16))


def test_factorized_advantages_center_each_bit_independently() -> None:
    returns = torch.tensor([
        [[1.0, 10.0], [3.0, 14.0]],
        [[5.0, 18.0], [7.0, 22.0]],
    ])
    advantages = _factorized_advantages(returns)
    assert torch.allclose(advantages.mean(dim=(0, 1)), torch.zeros(2))
    assert torch.equal(advantages[:, :, 0], returns[:, :, 0] - 4.0)
    assert torch.equal(advantages[:, :, 1], returns[:, :, 1] - 16.0)


def test_legacy_controller_history_clamps_new_opaque_protocol_command() -> None:
    policy = BrainWorkshopPolicy(
        width=32, intention_width=16, modalities=("vision", "audio"))
    actions = torch.tensor([0, 1, 2, 3])
    assert torch.equal(
        _controller_action_index(policy, actions),
        torch.tensor([0, 1, 2, 2]))


def test_source_keys_are_zero_initialized_and_runtime_variable() -> None:
    policy = BrainWorkshopPolicy(
        width=32,
        intention_width=16,
        modalities=("vision", "audio"),
        learned_source_keys=True,
    )
    assert set(policy.source_keys) == {"vision", "audio"}
    assert torch.equal(policy.source_keys["vision"], torch.zeros(32))
    assert torch.equal(policy.source_keys["audio"], torch.zeros(32))


def test_rehearsal_weights_match_rungs_and_preserve_scalar_compatibility() -> None:
    assert _resolve_rehearsal_weights("", (1, 2, 3), 0.5) == (
        0.5, 0.5, 0.5)
    assert _resolve_rehearsal_weights("2,0.5,0.5", (1, 2, 3), 0.5) == (
        2.0, 0.5, 0.5)


def test_rehearsal_weights_reject_mismatched_or_negative_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="match rehearsal_n_backs"):
        _resolve_rehearsal_weights("1,2", (1, 2, 3), 0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        _resolve_rehearsal_weights("1,-1,1", (1, 2, 3), 0.5)
