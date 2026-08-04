from __future__ import annotations

import torch

from experiments.memory_recall_amodal.environment import OutcomeRecallVerifier
from experiments.memory_recall_amodal.train import build_runtime, evaluate_condition


def test_memory_recall_verifier_exposes_only_scalar_outcomes() -> None:
    verifier = OutcomeRecallVerifier(seed=3)
    verifier.reset()
    probe_reward = verifier.score_probe(torch.tensor([0]))
    recall_reward = verifier.score_recall(torch.tensor([int(probe_reward.item())]))

    assert probe_reward.shape == (1,)
    assert recall_reward.tolist() == [1.0]
    assert not hasattr(verifier, "target")


def test_memory_recall_audit_conditions_are_callable() -> None:
    runtime = build_runtime(seed=5)
    for condition in ("intact", "clear", "corrupt", "fresh", "replacement", "random_action"):
        score = evaluate_condition(
            runtime,
            OutcomeRecallVerifier(seed=6),
            condition=condition,
            episodes=8,
        )
        assert 0.0 <= score <= 1.0
