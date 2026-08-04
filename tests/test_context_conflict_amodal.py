from __future__ import annotations

import torch

from experiments.context_conflict_amodal.environment import (
    BalancedContextConflictCurriculum,
    ContextConflictVerifier,
)
from experiments.context_conflict_amodal.train import build_runtime, train_steps


def test_context_conflict_keeps_candidates_contradictory_and_context_private() -> None:
    verifier = ContextConflictVerifier(seed=3)
    streams = verifier.reset(32)
    assert set(streams) == {"a", "b", "c"}
    assert torch.allclose(streams["b"][:, 0], -streams["c"][:, 0])
    assert torch.allclose(streams["b"][:, 1], -streams["c"][:, 1])


def test_balanced_curriculum_contains_both_contexts() -> None:
    verifier = BalancedContextConflictCurriculum(seed=4)
    streams = verifier.reset(32)
    assert streams["a"].shape == (32, 4)


def test_short_context_conflict_rung_records_accounting() -> None:
    runtime = build_runtime(seed=7)
    verifier = BalancedContextConflictCurriculum(seed=8)
    history, accounting = train_steps(
        runtime, verifier, steps=2, batch_size=4, seed=9, eval_every=1
    )
    assert history
    assert accounting.unique_logical_lifetimes == 8
    assert accounting.unique_verifier_bits == 8
