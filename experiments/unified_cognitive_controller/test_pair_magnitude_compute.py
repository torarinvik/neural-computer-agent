from __future__ import annotations

import torch

from .audit_pair_magnitude_compute import evaluate_budget
from .model import UnifiedCognitiveController


def test_magnitude_compute_audit_counts_only_optional_thought() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    result = evaluate_budget(
        model, count=8, trials=6, seed=23401,
        extra_thought_steps=2, device=torch.device("cpu"))
    assert result["controller_steps_per_event"] == 3
    assert result["controller_steps_per_lifetime"] == 18
    assert result["logical_lifetimes"] == 8
    assert result["verifier_bits"] == 96
    assert 0.0 <= result["normal_accuracy"] <= 1.0


def test_magnitude_compute_audit_rejects_negative_thought() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    try:
        evaluate_budget(
            model, count=8, trials=6, seed=23402,
            extra_thought_steps=-1, device=torch.device("cpu"))
    except ValueError as error:
        assert "must not be negative" in str(error)
    else:
        raise AssertionError("negative thought budget was accepted")
