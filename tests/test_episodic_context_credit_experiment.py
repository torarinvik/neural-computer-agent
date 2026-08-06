from __future__ import annotations

import torch

from experiments.episodic_context_credit_amodal.train import _pattern_bank


def test_generated_pattern_bank_scales_without_duplicate_same_statistics_rows() -> None:
    patterns = _pattern_bank(9, episode_length=6)

    assert patterns.shape == (20, 6)
    assert torch.equal(patterns.sum(dim=1), torch.full((20,), 3, dtype=torch.long))
    assert torch.unique(patterns, dim=0).shape[0] == 20
