from __future__ import annotations

import pytest
import torch

from experiments.memory_retention_amodal.environment import (
    OutcomeOnlyRetentionVerifier,
)
from experiments.memory_retention_amodal.train import (
    OutcomeValueCritic,
    OutcomeWriteCritic,
    _event,
    _probe_without_writing,
    _reset_optimizer_state,
    _retention_slots,
    _set_memory_write_policy_trainable,
    build_runtime,
    diagnose_write_policy,
    evaluate_condition,
    evaluate_persistent_reload,
    train_curriculum,
)


def test_outcome_write_critic_is_training_only_and_shape_checked() -> None:
    critic = OutcomeWriteCritic(width=16)
    values = critic(torch.zeros(3, 49))
    assert values.shape == (3,)
    assert torch.all((values >= 0.0) & (values <= 1.0))
    with pytest.raises(ValueError, match="critic features"):
        critic(torch.zeros(3, 48))


def test_outcome_value_critic_is_training_only_and_shape_checked() -> None:
    critic = OutcomeValueCritic(width=16)
    values = critic(torch.zeros(3, 16))
    assert values.shape == (3,)
    assert torch.all((values >= 0.0) & (values <= 1.0))
    with pytest.raises(ValueError, match="value critic features"):
        critic(torch.zeros(3, 15))


def test_retention_verifier_exposes_only_scalar_outcomes() -> None:
    verifier = OutcomeOnlyRetentionVerifier(batch_size=2, seed=17)
    verifier.reset()
    slots = torch.tensor([0, 1], dtype=torch.long)
    probe_reward = verifier.score_probe(slots, torch.zeros(2, dtype=torch.long))
    recall_reward = verifier.score_recall(torch.zeros(2, dtype=torch.long))

    assert probe_reward.shape == (2,)
    assert recall_reward.shape == (2,)
    assert probe_reward.dtype == torch.float32
    assert not hasattr(verifier, "bits")
    assert not hasattr(verifier, "target")


def test_retention_verifier_supports_bounded_multi_row_worlds() -> None:
    verifier = OutcomeOnlyRetentionVerifier(batch_size=4, seed=17, slot_count=3)
    verifier.reset()
    assert verifier.slot_count == 3
    assert verifier.order.shape == (4, 3)
    assert verifier.query_slot.max().item() < 3

    slots = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    rewards = verifier.score_probe(slots, torch.zeros(4, dtype=torch.long))
    assert rewards.shape == (4,)
    assert verifier.score_recall(torch.zeros(4, dtype=torch.long)).shape == (4,)


def test_retention_verifier_can_score_recall_of_logged_probe_outcomes() -> None:
    verifier = OutcomeOnlyRetentionVerifier(
        batch_size=2, seed=17, recall_probe_outcome=True
    )
    verifier.reset()
    slots = torch.tensor([0, 1], dtype=torch.long)
    probe_actions = torch.zeros(2, dtype=torch.long)
    probe_rewards = verifier.score_probe(slots, probe_actions)
    assert torch.equal(
        verifier.score_recall(probe_rewards.to(torch.long)),
        torch.ones(2),
    )


def test_multi_row_retention_orders_are_permutations_and_balanced() -> None:
    verifier = OutcomeOnlyRetentionVerifier(batch_size=6, seed=17, slot_count=3)
    verifier.reset()
    for order_name in ("random", "target_first", "target_last", "balanced"):
        slots = _retention_slots(verifier, order_name)
        assert slots.shape == (6, 3)
        assert torch.equal(torch.sort(slots, dim=1).values, torch.arange(3).expand(6, -1))
    balanced = _retention_slots(verifier, "balanced")
    target_positions = (balanced == verifier.query_slot[:, None]).nonzero()[:, 1]
    assert target_positions.tolist() == [0, 1, 2, 0, 1, 2]


def test_balanced_positions_survive_counterfactual_row_duplication() -> None:
    verifier = OutcomeOnlyRetentionVerifier(batch_size=3, seed=17, slot_count=3)
    verifier.reset()
    duplicate = verifier.duplicate_rows(2)
    assert duplicate.balanced_position.tolist() == [0, 0, 1, 1, 2, 2]


def test_retention_audit_controls_are_callable() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    tokens = torch.randn(2, runtime.event_width)
    for condition in (
        "intact",
        "clear",
        "corrupt",
        "reverse_order",
        "random_action",
        "missing_write_cue",
        "missing_query_cue",
        "target_first",
        "target_last",
    ):
        score = evaluate_condition(
            runtime,
            OutcomeOnlyRetentionVerifier(batch_size=2, seed=6),
            tokens,
            condition=condition,
            episodes=2,
        )
        assert 0.0 <= score <= 1.0


def test_write_policy_diagnostic_uses_private_labels_only_after_decision() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    tokens = torch.randn(2, runtime.event_width)
    diagnostics = diagnose_write_policy(
        runtime,
        OutcomeOnlyRetentionVerifier(batch_size=2, seed=6),
        tokens,
        retention_order="balanced",
        episodes=2,
    )

    assert set(diagnostics) == {
        "target_write_strength",
        "distractor_write_strength",
        "target_commit_rate",
        "distractor_commit_rate",
        "target_minus_distractor_strength",
        "target_minus_distractor_commit_rate",
    }
    assert all(
        0.0 <= diagnostics[name] <= 1.0
        for name in (
            "target_write_strength",
            "distractor_write_strength",
            "target_commit_rate",
            "distractor_commit_rate",
        )
    )
    assert -1.0 <= diagnostics["target_minus_distractor_strength"] <= 1.0
    assert -1.0 <= diagnostics["target_minus_distractor_commit_rate"] <= 1.0


def test_persistent_memory_audit_rejects_corruption_and_recovers() -> None:
    runtime = build_runtime(seed=5, batch_size=2, memory_scope_capacity=4)
    tokens = torch.randn(2, runtime.event_width)
    audit = evaluate_persistent_reload(
        runtime,
        OutcomeOnlyRetentionVerifier(batch_size=2, seed=6),
        tokens,
        episodes=2,
    )

    assert 0.0 <= audit["reload_intact_recall"] <= 1.0
    assert audit["corruption_rejected"] is True
    assert 0.0 <= audit["recovery_intact_recall"] <= 1.0


def test_forced_parent_action_keeps_propensity_contract_numeric() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    state = runtime.initial_state(2, device="cpu")
    token = torch.randn(2, runtime.event_width)
    action, propensity, log_probability, _ = _probe_without_writing(
        runtime,
        state,
        _event(token),
        forced_action=torch.tensor([0, 1]),
    )

    assert action.tolist() == [0, 1]
    assert torch.all(propensity > 0.0)
    assert torch.isfinite(log_probability).all()


def test_policy_reset_clears_stale_optimizer_moments() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    optimizer = torch.optim.Adam(runtime.parameters(), lr=1e-3)
    loss = sum(parameter.square().sum() for parameter in runtime.parameters())
    loss.backward()
    optimizer.step()
    parameters = tuple(runtime.controller.memory_write_policy.parameters())
    assert all(parameter in optimizer.state for parameter in parameters)

    _reset_optimizer_state(optimizer, parameters)

    assert all(parameter not in optimizer.state for parameter in parameters)


def test_parent_write_policy_protection_routes_gradients_by_phase() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    _set_memory_write_policy_trainable(runtime, False)
    assert all(
        not parameter.requires_grad
        for parameter in runtime.controller.memory_write_policy.parameters()
    )

    _set_memory_write_policy_trainable(runtime, True)
    assert all(
        parameter.requires_grad
        for parameter in runtime.controller.memory_write_policy.parameters()
    )


def test_retention_token_reuse_steps_must_be_positive() -> None:
    runtime = build_runtime(seed=5, batch_size=2)
    tokens = torch.randn(2, runtime.event_width)
    with pytest.raises(ValueError, match="token reuse steps"):
        train_curriculum(
            runtime,
            OutcomeOnlyRetentionVerifier(batch_size=2, seed=6),
            tokens,
            phase1_steps=1,
            phase2_steps=1,
            seed=7,
            reward_shuffle=False,
            retention_token_reuse_steps=0,
        )
