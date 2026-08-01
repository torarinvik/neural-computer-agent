"""Small interface tests for the reward probe's amodal policy wrapper."""

import torch

from .train_brainworkshop_policy import BrainWorkshopPolicy


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
