from __future__ import annotations

import torch

from experiments.calibration_conflict_amodal.environment import (
    CalibrationConflictVerifier,
)
from experiments.calibration_conflict_amodal.train import build_runtime, train_steps


def test_calibration_streams_contradict_and_role_is_private() -> None:
    verifier = CalibrationConflictVerifier(seed=3, sequence_length=4)
    sequence = verifier.sample(16)
    assert sequence.targets.shape == (4, 16)
    assert torch.allclose(sequence.streams[0]["b"][:, 0], -sequence.streams[0]["c"][:, 0])
    assert torch.allclose(sequence.streams[0]["b"][:, 1], -sequence.streams[0]["c"][:, 1])


def test_short_calibration_rung_records_accounting() -> None:
    runtime = build_runtime(seed=7)
    verifier = CalibrationConflictVerifier(seed=8, sequence_length=4)
    history, accounting = train_steps(
        runtime, verifier, steps=2, batch_size=4, seed=9, eval_every=1
    )
    assert history
    assert accounting.unique_logical_lifetimes == 2 * 4 * 4
    assert accounting.unique_verifier_bits == accounting.unique_logical_lifetimes
