from __future__ import annotations

import torch

from .audit_immutable_core_cross_primitive import (
    _logged_binary_feedback,
    _passes,
    _passes_behavior,
)


def test_logged_binary_feedback_is_exact_and_reproducible() -> None:
    rules = torch.tensor([0, 1] * 8)
    first = _logged_binary_feedback(rules, seed=19)
    second = _logged_binary_feedback(rules, seed=19)
    assert all(torch.equal(left, right)
               for left, right in zip(first, second))
    actions, rewards, order = first
    assert torch.equal(rewards, (actions == rules[order]).float())


def test_gate_requires_causality_missing_evidence_and_controls() -> None:
    passing = {
        "normal_accuracy": 0.90,
        "counterfactual_accuracy": 0.88,
        "counterfactual_flip_rate": 0.80,
        "missing": {"missing_first": 0.52, "missing_second": 0.50},
        "action_complement_accuracy": 0.10,
        "reward_complement_accuracy": 0.12,
    }
    assert _passes_behavior(passing)
    assert _passes(passing)
    for key in (
            "normal_accuracy", "counterfactual_accuracy",
            "counterfactual_flip_rate", "action_complement_accuracy",
            "reward_complement_accuracy"):
        broken = dict(passing)
        broken[key] = 0.0 if "complement" not in key else 0.9
        assert not _passes(broken)
    broken_missing = dict(passing)
    broken_missing["missing"] = {"missing_first": 0.9}
    assert not _passes(broken_missing)
