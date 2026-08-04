from __future__ import annotations

import pytest
import torch

from experiments.contradiction_amodal.environment import SequentialConflictVerifier
from experiments.contradiction_amodal.train import build_runtime, train_steps


def test_conflicting_streams_are_always_opposite_and_roles_are_private() -> None:
    verifier = SequentialConflictVerifier(seed=3, sequence_length=8, block_length=4)
    sequence = verifier.sample(16, alternating_roles=True)
    assert sequence.targets.shape == (8, 16)
    assert sequence.roles.shape == (16, 2)
    assert set(sequence.streams[0]) == {"b", "c"}
    assert torch.allclose(sequence.streams[0]["b"][:, 0], -sequence.streams[0]["c"][:, 0])
    assert torch.equal(sequence.streams[0]["b"][:, 1], -sequence.streams[0]["c"][:, 1])


def test_stream_order_shuffle_preserves_named_payloads() -> None:
    plain = SequentialConflictVerifier(seed=7, sequence_length=8, block_length=4)
    shuffled = SequentialConflictVerifier(
        seed=7, sequence_length=8, block_length=4, stream_order_shuffle=True
    )
    plain_sequence = plain.sample(4, alternating_roles=True)
    shuffled_sequence = shuffled.sample(4, alternating_roles=True)
    for left, streams in zip(plain_sequence.streams, shuffled_sequence.streams):
        assert tuple(streams) in (("b", "c"), ("c", "b"))
        assert torch.equal(left["b"], streams["b"])
        assert torch.equal(left["c"], streams["c"])
        assert torch.allclose(streams["b"][:, 0], -streams["c"][:, 0])
        assert torch.allclose(streams["b"][:, 1], -streams["c"][:, 1])


def test_short_training_rung_records_sequence_accounting() -> None:
    runtime = build_runtime(seed=11)
    verifier = SequentialConflictVerifier(seed=12, sequence_length=16, block_length=1)
    history, accounting = train_steps(
        runtime, verifier, steps=2, batch_size=4, seed=13, eval_every=1
    )
    assert history
    assert accounting.unique_logical_lifetimes == 2 * 4 * 16
    assert accounting.unique_verifier_bits == accounting.unique_logical_lifetimes


def test_default_random_reversal_fits_the_default_four_block_sequence() -> None:
    verifier = SequentialConflictVerifier(seed=14, sequence_length=32, block_length=8)
    sequence = verifier.sample_random_reversal(8)
    assert sequence.roles.shape == (8, 4)
    assert torch.all(sequence.roles[:, 0] == 0)
    assert torch.all(sequence.roles[:, -1] == 1)


def test_markov_role_process_has_no_fixed_clock_schedule() -> None:
    verifier = SequentialConflictVerifier(seed=15, sequence_length=32, block_length=1)
    sequence = verifier.sample_markov_roles(512, switch_probability=0.2)
    assert sequence.roles.shape == (512, 32)
    assert sequence.roles[:, 0].float().mean().item() == pytest.approx(0.5, abs=0.08)
    assert sequence.roles[:, 8].float().mean().item() == pytest.approx(0.5, abs=0.08)
    assert torch.any(sequence.roles[:, 1:] != sequence.roles[:, :-1])
