from __future__ import annotations

import torch

from experiments.memory_binding_amodal.environment import TwoSlotBindingVerifier


def test_two_slot_verifier_exposes_only_scalar_outcomes() -> None:
    verifier = TwoSlotBindingVerifier(batch_size=2, seed=17)
    verifier.reset()
    first = verifier.score_probe(0, torch.zeros(2, dtype=torch.long))
    second = verifier.score_probe(1, torch.ones(2, dtype=torch.long))
    recalled = verifier.score_recall(verifier.query_slot * 0)

    assert first.shape == (2,)
    assert second.shape == (2,)
    assert recalled.shape == (2,)
    assert first.dtype == torch.float32
    assert second.dtype == torch.float32
