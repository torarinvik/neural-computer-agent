from pathlib import Path

import pytest
import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import attempted_success_loss, evaluate, rollout
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_redundancy_transfer import (
    _expanded_eight_feature_controller,
    _stable_crossing,
    build_transfer_arms,
    initialize_from_saved_strategy,
    redundancy_utility_batch,
)
from .train_contextual_full_residual import (
    adapter_scores,
    full_residual_context_key,
    residual_scores,
    retrieve_residual,
    select_verified_candidate,
)
from .train_persistent_memory import _grouped_read
from .probe_persistent_physical_stream import (
    _future_for_actions,
    _ranked_age,
)
from .compare_persistent_fresh_efficiency import _arm_metrics
from .strategy_memory import (
    LatentStrategyMemory,
    VerifierTrainedContextEncoder,
    physical_context_key,
)
from .train_persistent_physical_utility_adaptation import (
    _curriculum_phases,
    _restore_strategy_memory,
    _strategy_memory_payload,
    _value_diverse_admission,
)
from .dynamic_working_memory import (
    CapabilityLedger,
    DynamicWorkingMemory,
    LatencyTimer,
)


def test_lifetime_has_one_correct_action_and_balanced_private_rules() -> None:
    batch = generate_lifetimes(16, 5, seed=11)
    assert batch.frames.shape == (16, 5, 3, 32, 32)
    assert sorted(batch.rule_bits.tolist()) == [0] * 8 + [1] * 8
    assert torch.equal(
        batch.correct_actions,
        batch.stimulus_identities ^ batch.rule_bits.unsqueeze(1))
    assert sorted(batch.stimulus_identities[:, 0].tolist()) == (
        [0] * 8 + [1] * 8)
    supported = generate_lifetimes(
        16, 5, seed=11, support_trials=4)
    assert torch.equal(
        supported.stimulus_identities[:, 1],
        1 - supported.stimulus_identities[:, 0])


def test_rule_counterfactual_changes_answers_not_pixels() -> None:
    normal = generate_lifetimes(8, 4, seed=17)
    reversed_batch = generate_lifetimes(
        8, 4, seed=17, reverse_rules=True)
    assert torch.equal(normal.frames, reversed_batch.frames)
    assert torch.equal(
        normal.correct_actions, 1 - reversed_batch.correct_actions)

    constant = generate_lifetimes(
        8, 4, seed=17, task="constant_action")
    reversed_constant = generate_lifetimes(
        8, 4, seed=17, reverse_rules=True, task="constant_action")
    assert torch.equal(constant.frames, reversed_constant.frames)
    assert torch.equal(
        constant.correct_actions, 1 - reversed_constant.correct_actions)
    assert torch.equal(
        constant.correct_actions,
        constant.rule_bits.unsqueeze(1).expand(-1, 4))

    identity = generate_lifetimes(
        8, 4, seed=17, task="visible_identity")
    reversed_identity = generate_lifetimes(
        8, 4, seed=17, reverse_stimuli=True,
        task="visible_identity")
    assert not torch.equal(identity.frames, reversed_identity.frames)
    assert torch.equal(
        identity.correct_actions, 1 - reversed_identity.correct_actions)


def test_heldout_renderer_changes_public_surface() -> None:
    train = generate_lifetimes(8, 4, seed=19)
    heldout = generate_lifetimes(8, 4, seed=19, heldout=True)
    assert not torch.equal(train.frames, heldout.frames)
    assert torch.equal(train.correct_actions, heldout.correct_actions)


def test_novel_appearances_change_only_public_geometry() -> None:
    bars = generate_lifetimes(
        16, 6, seed=20, task="binary_mapping",
        appearance="bars", support_trials=1)
    for appearance in ("diamonds", "dot_pairs"):
        novel = generate_lifetimes(
            16, 6, seed=20, task="binary_mapping",
            appearance=appearance, support_trials=1)
        assert not torch.equal(bars.frames, novel.frames)
        assert torch.equal(
            bars.stimulus_identities, novel.stimulus_identities)
        assert torch.equal(bars.rule_bits, novel.rule_bits)
        assert torch.equal(bars.correct_actions, novel.correct_actions)
    diamonds = generate_lifetimes(
        16, 6, seed=20, task="binary_mapping",
        appearance="diamonds", support_trials=1)
    reversed_diamonds = generate_lifetimes(
        16, 6, seed=20, task="binary_mapping",
        appearance="diamonds", support_trials=1,
        reverse_rules=True)
    assert torch.equal(diamonds.frames, reversed_diamonds.frames)
    assert torch.equal(
        diamonds.correct_actions,
        1 - reversed_diamonds.correct_actions)


def test_four_rule_support_is_identifiable_and_balanced() -> None:
    batch = generate_lifetimes(
        16, 6, seed=21, task="four_rule", support_trials=2)
    assert sorted(batch.rule_bits.tolist()) == [0] * 4 + [1] * 4 + (
        [2] * 4 + [3] * 4)
    assert torch.equal(
        batch.stimulus_identities[:, 1],
        1 - batch.stimulus_identities[:, 0])
    rules = batch.rule_bits.unsqueeze(1).expand(-1, batch.trials)
    expected = torch.where(
        rules < 2, rules,
        batch.stimulus_identities ^ (rules - 2))
    assert torch.equal(batch.correct_actions, expected)
    reversed_batch = generate_lifetimes(
        16, 6, seed=21, task="four_rule", support_trials=2,
        reverse_rules=True)
    assert torch.equal(batch.frames, reversed_batch.frames)
    assert torch.equal(
        batch.correct_actions, 1 - reversed_batch.correct_actions)


def test_hidden_rule_gate_requires_real_vision(monkeypatch) -> None:
    """A feedback-only shortcut must never be admitted as composition."""
    model = UnifiedCognitiveController()

    def fake_rollout(_model, batch, **_kwargs):
        # Deliberately use verifier answers regardless of the input frame.
        actions = batch.correct_actions.clone()
        actions[:, 0] = 0
        return {
            "actions": actions,
            "rewards": (
                actions == batch.correct_actions).to(torch.float32),
            "logits": torch.zeros(batch.batch_size, batch.trials, 2),
            "final_workspace": torch.zeros(
                batch.batch_size, model.workspace_slots, model.width),
            "final_hidden": torch.zeros(batch.batch_size, model.width),
        }

    monkeypatch.setattr(
        "experiments.unified_cognitive_controller.train.rollout",
        fake_rollout)
    report = evaluate(
        model, count=8, trials=6, seed=42, device=torch.device("cpu"),
        task="four_rule", feedback_trials=2)
    assert not report["gate"]["vision_causally_used"]
    assert not report["gate"]["accepted"]


def test_unified_controller_rollout_and_workspace_shapes() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    batch = generate_lifetimes(4, 3, seed=23)
    result = rollout(model, batch, sample_actions=False)
    assert result["actions"].shape == (4, 3)
    assert result["logits"].shape == (4, 3, 2)
    assert result["final_workspace"].shape == (4, 4, 32)
    assert result["final_hidden"].shape == (4, 32)

    no_feedback = rollout(
        model, batch, sample_actions=False, feedback_trials=0)
    assert no_feedback["actions"].shape == (4, 3)


def test_attempted_loss_has_no_unattempted_target_argument() -> None:
    logits = torch.tensor([[0.2, -0.4], [0.3, 0.8]], requires_grad=True)
    actions = torch.tensor([0, 1])
    outcomes = torch.tensor([1.0, 0.0])
    loss = attempted_success_loss(logits, actions, outcomes)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 0] == 0


def test_disk_latent_memory_round_trip(tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=8, capacity=2)
    keys = torch.eye(8)[:2]
    values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    assert memory.commit(
        keys, values, torch.tensor([0.9, 0.1]), threshold=0.5) == 1
    path = tmp_path / "memory.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    assert restored.count == 1
    read, confidence = restored.retrieve(keys[:1], top_k=1)
    assert torch.allclose(read, values[:1])
    assert confidence.shape == (1,)


def test_disk_memory_records_persists_and_resets_access_counts(
        tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert memory.commit(
        keys, values, torch.ones(2), threshold=0.0) == 2
    memory.retrieve(
        keys[:1].repeat(3, 1), top_k=1,
        confidence_mode="cosine", record_access=True)
    memory.store.record_outcomes(
        keys[:1].repeat(3, 1), torch.tensor([1.0, 1.0, 0.0]))
    assert memory.store.access_count.tolist() == [3, 0]
    assert memory.store.success_count.tolist() == [2, 0]
    assert memory.store.failure_count.tolist() == [1, 0]
    path = tmp_path / "access-counts.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    assert restored.store.access_count.tolist() == [3, 0]
    assert restored.store.success_count.tolist() == [2, 0]
    assert restored.store.failure_count.tolist() == [1, 0]
    restored.replace(0, keys[1], values[1], 0.9)
    assert restored.store.access_count.tolist() == [0, 0]
    assert restored.store.success_count.tolist() == [0, 0]
    assert restored.store.failure_count.tolist() == [0, 0]


def test_persistent_stream_age_uses_current_insertion_rank() -> None:
    memory = DiskLatentMemory(width=4, capacity=3)
    memory.commit(
        torch.eye(4)[:3], torch.eye(4)[:3],
        torch.ones(3), threshold=0.0)
    memory.store.age[:3] = torch.tensor([17, 4, 11])
    assert torch.allclose(
        _ranked_age(memory),
        torch.tensor([1.0, 1 / 3, 2 / 3]))


def test_persistent_stream_future_replaces_only_selected_rows() -> None:
    batch = generate_lifetimes(4, 3, seed=330)
    candidate = generate_lifetimes(2, 3, seed=332)
    queries = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    candidate_queries = torch.full((2, 4), -1.0)
    updated, updated_queries = _future_for_actions(
        batch, queries, candidate, candidate_queries,
        torch.tensor([0, 2]), capacity=2)
    assert torch.equal(updated.seeds[:2], batch.seeds[:2])
    assert torch.equal(updated.seeds[2], batch.seeds[2])
    assert torch.equal(updated.seeds[3], candidate.seeds[1])
    assert torch.equal(updated_queries[0], queries[0])
    assert torch.equal(updated_queries[1, 0], queries[1, 0])
    assert torch.equal(updated_queries[1, 1], candidate_queries[1])


def test_efficiency_metric_normalizes_against_same_state_frozen_policy() -> None:
    report = {
        "total_seconds": 1.5,
        "trace": [
            {
                "phase": "reliability_dominant",
                "learned_reward": 0.8,
                "frozen_reward": 0.6,
                "learned_target_rate": 0.7,
                "frozen_target_rate": 0.4,
            },
            {
                "phase": "old_return",
                "learned_reward": 0.9,
                "frozen_reward": 0.85,
                "learned_target_rate": 0.8,
                "frozen_target_rate": 0.75,
            },
        ],
    }
    metrics = _arm_metrics(report)
    assert abs(metrics["verified_reward_advantage_auc"] - 0.25) < 1e-6
    assert abs(metrics["target_rate_advantage_auc"] - 0.35) < 1e-6
    assert abs(metrics["old_return_reward_advantage"] - 0.05) < 1e-6


def test_strategy_memory_is_bounded_retrievable_and_persistent(
        tmp_path: Path) -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=3, value_width=2)
    keys = torch.eye(3)
    memory.upsert(
        keys[0], torch.tensor([1.0, 0.0]),
        verified_improvement=0.2)
    memory.upsert(
        keys[1], torch.tensor([0.0, 1.0]),
        verified_improvement=-0.1)
    retrieved = memory.retrieve(
        keys[0], torch.zeros(2))
    assert retrieved.slot == 0
    assert torch.equal(retrieved.value, torch.tensor([1.0, 0.0]))
    memory.upsert(
        keys[2], torch.tensor([-1.0, -1.0]),
        verified_improvement=0.3)
    assert memory.count == 2
    path = tmp_path / "strategy-memory.pt"
    memory.save(path)
    restored = LatentStrategyMemory.load(path)
    assert restored.count == 2
    assert torch.equal(restored.keys, memory.keys)
    assert torch.equal(restored.values, memory.values)
    assert torch.equal(restored.usage, memory.usage)
    assert torch.equal(restored.success, memory.success)
    assert torch.equal(restored.failure, memory.failure)


def test_strategy_memory_can_replace_a_preferred_duplicate_slot() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=2, value_width=1)
    memory.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([1.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([2.0]),
        verified_improvement=1.0)
    slot = memory.upsert(
        torch.tensor([-1.0, 0.0]), torch.tensor([3.0]),
        verified_improvement=1.0, preferred_slot=0)
    assert slot == 0
    assert memory.count == 2
    assert torch.equal(memory.values[:, 0], torch.tensor([3.0, 2.0]))


def test_physical_context_key_ignores_skip_and_is_normalized() -> None:
    features = torch.zeros(2, 4, 7)
    features[:, 1:, 0] = 0.5
    features[:, 1:, 5] = -0.25
    features[:, 1:, 6] = 0.25
    first = physical_context_key(features)
    features[:, 0] = 999
    second = physical_context_key(features)
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(), torch.tensor(1.0))
    rewarded = physical_context_key(
        features, torch.tensor([0.4, 0.2, 0.1]))
    assert rewarded.shape == (13,)
    assert torch.allclose(rewarded.norm(), torch.tensor(1.0))


def test_context_encoder_learns_only_from_verified_improvement() -> None:
    encoder = VerifierTrainedContextEncoder(width=3)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5)
    keys = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    query = torch.tensor([1.0, 1.0, 0.0])
    before = encoder.log_scale.detach().clone()
    loss = encoder.reinforce(
        query, keys, selected_slot=0, verified_improvement=1.0,
        optimizer=optimizer)
    assert loss > 0
    assert not torch.equal(encoder.log_scale, before)
    learned = encoder.log_scale.detach().clone()
    encoder.reinforce(
        query, keys, selected_slot=0, verified_improvement=0.0,
        optimizer=optimizer)
    assert torch.equal(encoder.log_scale, learned)


def test_strategy_memory_can_retrieve_with_learned_context_metric() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=3, value_width=1)
    memory.upsert(
        torch.tensor([1.0, 0.1, 0.0]), torch.tensor([1.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.1, 1.0, 0.0]), torch.tensor([2.0]),
        verified_improvement=1.0)
    encoder = VerifierTrainedContextEncoder(width=3)
    encoder.log_scale.data.copy_(torch.tensor([3.0, -3.0, 0.0]))
    result = memory.retrieve(
        torch.tensor([0.2, 1.0, 0.0]), torch.zeros(1),
        encoder=encoder)
    assert result.slot == 0


def test_soft_strategy_retrieval_is_a_convex_mixture() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=2, value_width=2)
    memory.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 4.0]),
        verified_improvement=1.0)
    encoder = VerifierTrainedContextEncoder(width=2)
    result = memory.retrieve_soft(
        torch.tensor([1.0, 1.0]), torch.zeros(2),
        encoder=encoder, temperature=0.5)
    assert result.mixture_weights is not None
    assert torch.allclose(
        result.mixture_weights, torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.value, torch.tensor([1.0, 2.0]))


def test_context_encoder_spsa_step_follows_verified_difference() -> None:
    encoder = VerifierTrainedContextEncoder(width=3)
    direction = torch.tensor([1.0, -1.0, 1.0])
    advantage = encoder.spsa_step(
        direction, positive_reward=0.8, negative_reward=0.3,
        step_size=0.2)
    assert abs(advantage - 0.5) < 1e-6
    assert torch.allclose(
        encoder.log_scale,
        torch.tensor([0.1, -0.1, 0.1]))


def test_context_reliability_ramp_has_six_rounds_and_one_changing_axis(
        ) -> None:
    phases = _curriculum_phases(
        "context_reliability_ramp", rounds_per_phase=99)
    assert len(phases) == 6
    assert all(rounds == 1 for _, _, rounds in phases)
    reliability = [weights[2] for _, weights, _ in phases]
    assert reliability == [0.0, 0.1, 0.2, 0.3, 0.4, 0.0]
    for _, weights, _ in phases:
        assert abs(sum(weights) - 1.0) < 1e-9
        assert weights[0] == weights[1]


def test_interleaved_reliability_exposes_every_context_in_each_short_cycle(
        ) -> None:
    phases = _curriculum_phases(
        "interleaved_reliability", rounds_per_phase=4)
    assert len(phases) == 12
    assert all(rounds == 1 for _, _, rounds in phases)
    assert [phase for phase, _, _ in phases] == [
        "old_equal", "reliability_dominant", "old_return",
    ] * 4
    assert [weights for _, weights, _ in phases[:3]] == [
        (0.5, 0.5, 0.0),
        (0.3, 0.3, 0.4),
        (0.5, 0.5, 0.0),
    ]


def test_cyclic_reliability_blocks6_preserves_six_round_context_runs(
        ) -> None:
    phases = _curriculum_phases(
        "cyclic_reliability_blocks6", rounds_per_phase=18)
    assert [phase for phase, _, _ in phases] == [
        "old_equal", "reliability_dominant", "old_return",
    ] * 3
    assert [rounds for _, _, rounds in phases] == [6] * 9
    assert sum(rounds for _, _, rounds in phases) == 54


def test_cyclic_reliability_blocks6_rejects_incompatible_budget() -> None:
    with pytest.raises(ValueError, match="divisible by six"):
        _curriculum_phases("cyclic_reliability_blocks6", rounds_per_phase=5)


def test_value_diverse_admission_preserves_latent_extremes() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=2, value_width=2)
    memory.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]),
        verified_improvement=1.0)
    candidate, slot = _value_diverse_admission(
        memory,
        [
            torch.tensor([[0.1, 0.0]]),
            torch.tensor([[3.0, 0.0]]),
        ],
        [1.0, 0.5])
    assert candidate == 1
    assert slot == 1


def test_prefix_state_strategy_memory_round_trip_is_exact() -> None:
    source = LatentStrategyMemory(
        capacity=3, key_width=2, value_width=2)
    source.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.5]),
        verified_improvement=1.0)
    source.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([3.0, -0.5]),
        verified_improvement=-1.0)
    source.usage[:2] = torch.tensor([7, 4])
    restored = _restore_strategy_memory(
        _strategy_memory_payload(source), device=torch.device("cpu"))
    assert restored is not None
    assert restored.count == source.count
    for field in ("keys", "values", "usage", "success", "failure"):
        assert torch.equal(getattr(restored, field), getattr(source, field))


def test_dynamic_working_memory_mask_accounting_and_persistence(
        tmp_path: Path) -> None:
    memory = DynamicWorkingMemory(
        capacity=8, width=3, fixed_active_slots=4)
    assert memory.active_count == 4
    memory.write(1, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(memory.read(1), torch.tensor([1.0, 2.0, 3.0]))
    memory.write(4, torch.zeros(3))
    memory.write(4, torch.zeros(3))
    assert memory.stats.evictions == 1
    selected = memory.set_active_from_scores(
        torch.tensor([-1.0, -1.0, 0.9, 0.8, -1.0, -1.0, -1.0, -1.0]),
        minimum=2)
    assert selected.tolist() == [
        False, False, True, True, False, False, False, False]
    memory.record_step()
    assert memory.stats.mean_occupancy == 0.25
    path = tmp_path / "dynamic-memory.pt"
    memory.save(path)
    restored = DynamicWorkingMemory.load(path)
    assert torch.equal(restored.values, memory.values)
    assert torch.equal(restored.active, memory.active)
    assert torch.equal(restored.occupied, memory.occupied)
    assert torch.equal(restored.usage, memory.usage)
    assert torch.equal(restored.age, memory.age)
    assert restored.stats == memory.stats


def test_capability_ledger_records_memory_and_latency() -> None:
    memory = DynamicWorkingMemory(
        capacity=8, width=2, fixed_active_slots=2)
    memory.record_step()
    memory.active_values()
    ledger = CapabilityLedger(
        unique_verifier_bits=16, unique_logical_lifetimes=4)
    ledger.absorb_memory(memory.stats)
    with LatencyTimer(ledger):
        _ = sum(range(10))
    report = ledger.as_report()
    assert report["memory_reads"] == 2
    assert report["mean_active_fraction"] == 0.25
    assert report["latency_seconds"] >= 0.0


def test_old_disk_schema_loads_with_zero_access_counts(tmp_path: Path) -> None:
    path = tmp_path / "old-memory.pt"
    torch.save({
        "schema": "syllogimous-neural-computer-memory-v1",
        "keys": torch.zeros(2, 4),
        "values": torch.zeros(2, 4),
        "usage": torch.zeros(2),
        "age": torch.zeros(2, dtype=torch.long),
        "valid": torch.tensor([True, False]),
        "clock": 1,
        "growth_chunk": 2,
    }, path)
    restored = DiskLatentMemory.load(path)
    assert restored.count == 1
    assert restored.store.access_count.tolist() == [0, 0]
    assert restored.store.success_count.tolist() == [0, 0]
    assert restored.store.failure_count.tolist() == [0, 0]


def test_disk_memory_can_report_cosine_match_confidence() -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    strengths = torch.tensor([0.5, 0.9])
    assert memory.commit(
        keys, values, strengths, threshold=0.0) == 2
    _, ranked = memory.retrieve(keys[:1], top_k=1)
    read, cosine = memory.retrieve(
        keys[:1], top_k=1, confidence_mode="cosine")
    assert torch.allclose(read, values[:1])
    assert torch.allclose(cosine, torch.ones_like(cosine))
    assert ranked.item() < cosine.item()


def test_disk_memory_reports_adaptive_read_features() -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    strengths = torch.tensor([0.5, 0.9])
    assert memory.commit(
        keys, values, strengths, threshold=0.0) == 2
    read, features = memory.retrieve_with_features(keys[:1])
    assert torch.allclose(read, values[:1])
    assert features.shape == (1, 4)
    assert torch.allclose(features[:, 0], torch.ones(1))
    assert torch.allclose(features[:, 2], strengths[:1])
    assert torch.allclose(features[:, 3], torch.ones(1))


def test_adaptive_memory_read_is_optional_and_bounded() -> None:
    ordinary = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    assert ordinary.memory_read_gate is None
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True)
    probability = model.memory_read_probability(torch.zeros(3, 4))
    assert probability.shape == (3,)
    assert torch.all((probability > 0) & (probability < 1))
    nonlinear = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True, adaptive_memory_read_hidden=8)
    assert sum(
        parameter.numel()
        for parameter in nonlinear.memory_read_gate.parameters()) == 49


def test_adaptive_memory_replacement_is_optional_and_bounded() -> None:
    ordinary = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    assert ordinary.memory_replacement_gate is None
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8)
    scores = model.memory_replacement_scores(torch.zeros(3, 5, 5))
    assert scores.shape == (3, 5)
    assert sum(
        parameter.numel()
        for parameter in model.memory_replacement_gate.parameters()) == 57
    frequency_model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=6)
    frequency_scores = frequency_model.memory_replacement_scores(
        torch.zeros(3, 5, 6))
    assert frequency_scores.shape == (3, 5)
    assert sum(
        parameter.numel()
        for name, parameter in frequency_model.named_parameters()
        if name.startswith("memory_replacement_")) == 58
    missing, unexpected = frequency_model.load_state_dict(
        model.state_dict(), strict=False)
    assert missing == ["memory_replacement_extra_gate.weight"]
    assert not unexpected
    base_features = torch.randn(3, 5, 5)
    expanded_features = torch.cat((
        base_features, torch.zeros(3, 5, 1)), dim=-1)
    assert torch.equal(
        model.memory_replacement_scores(base_features),
        frequency_model.memory_replacement_scores(expanded_features))


def test_frequency_recency_utility_weights_are_validated_before_generation(
        ) -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    for recency_weight, frequency_weight in (
            (-0.1, 1.0), (1.0, -0.1), (0.0, 0.0)):
        try:
            frequency_recency_batch(
                model, banks=1, capacity=2, seed=1,
                device=torch.device("cpu"), write_threshold=0.5,
                recency_weight=recency_weight,
                frequency_weight=frequency_weight)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid utility weights were accepted")
    try:
        frequency_recency_batch(
            model, banks=1, capacity=2, seed=1,
            device=torch.device("cpu"), write_threshold=0.5,
            reliability_weight=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative reliability weight was accepted")


def test_eight_feature_expansion_preserves_old_replacement_scores() -> None:
    source = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=6)
    source.memory_replacement_extra_gate.weight.data.fill_(0.75)
    configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "adaptive_memory_read": True,
        "adaptive_memory_replace": True,
        "adaptive_memory_replace_hidden": 8,
        "adaptive_memory_replace_features": 6,
    }
    expanded = _expanded_eight_feature_controller(
        configuration, source.state_dict(), device=torch.device("cpu"))
    old_features = torch.randn(3, 5, 6)
    new_features = torch.cat((
        old_features, torch.zeros(3, 5, 2)), dim=-1)
    assert torch.equal(
        source.memory_replacement_scores(old_features),
        expanded.memory_replacement_scores(new_features))
    assert torch.equal(
        expanded.memory_replacement_extra_gate.weight,
        torch.tensor([[0.75, 0.0, 0.0]]))


def test_full_replacement_policy_reset_preserves_controller_backbone(
        ) -> None:
    source = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=6)
    configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "adaptive_memory_read": True,
        "adaptive_memory_replace": True,
        "adaptive_memory_replace_hidden": 8,
        "adaptive_memory_replace_features": 6,
    }
    arms = build_transfer_arms(
        {
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": source.state_dict(),
        },
        {
            "schema": "unified-controller-physical-prefix-state-v1",
            "model_state_dict": source.state_dict(),
        },
        device=torch.device("cpu"), fresh_seed=7)
    reset = arms["replacement_policy_reset"]
    assert all(
        int(torch.count_nonzero(parameter)) == 0
        for parameter in reset.memory_replacement_gate.parameters())
    assert int(torch.count_nonzero(
        reset.memory_replacement_extra_gate.weight)) == 0
    assert torch.equal(
        reset.vision.network[0].weight,
        arms["selected_experience"].vision.network[0].weight)


def test_redundancy_batch_is_seed_exact_and_exposes_only_row_statistic(
        ) -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    kwargs = {
        "banks": 2,
        "capacity": 3,
        "seed": 73,
        "device": torch.device("cpu"),
        "write_threshold": 0.0,
        "noise_scale": 0.01,
        "weights": (0.3, 0.3, 0.3, 0.1),
    }
    first = redundancy_utility_batch(model, **kwargs)
    second = redundancy_utility_batch(model, **kwargs)
    assert first["option_features"].shape == (2, 4, 8)
    assert torch.equal(
        first["option_features"], second["option_features"])
    assert torch.equal(first["target_action"], second["target_action"])
    assert torch.equal(
        first["option_features"][:, 1:, 7],
        first["row_novelty"] - 0.5)
    assert torch.all(first["target_action"] >= 1)
    assert torch.all(first["target_action"] <= 3)


def test_stable_crossing_rejects_transient_threshold_hits() -> None:
    values = [(0, 0.4), (10, 0.6), (20, 0.5), (30, 0.7)]
    assert _stable_crossing(values, 0.55) == 30
    assert _stable_crossing(values, 0.75) is None


def test_saved_strategy_initializes_old_dimensions_only() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    features = torch.zeros(2, 4, 8)
    features[:, 1:, 0] = 0.5
    selected = {
        "strategy_memory": {
            "count": 2,
            "keys": torch.stack((
                physical_context_key(
                    features, torch.tensor([0.8, 0.6, 0.4])),
                physical_context_key(
                    features, torch.tensor([0.4, 0.6, 0.8])),
            )),
            "values": torch.tensor([[2.0, -3.0], [-4.0, 5.0]]),
        },
        "context_encoder_state_dict": {
            "log_scale": torch.zeros(13),
        },
        "run_state": {
            "previous_reward_signature": torch.tensor([0.8, 0.6, 0.4]),
        },
    }
    report = initialize_from_saved_strategy(model, selected, features)
    assert report["slot"] == 0
    assert torch.equal(
        model.memory_replacement_extra_gate.weight,
        torch.tensor([[2.0, -3.0, 0.0]]))


def test_contextual_full_residual_is_exact_noop_until_retrieved() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    features = torch.randn(2, 4, 8)
    key = full_residual_context_key(features)
    assert key.shape == (18,)
    assert torch.allclose(key.norm(), torch.tensor(1.0))
    memory = LatentStrategyMemory(
        capacity=2, key_width=18, value_width=8)
    residual, accepted, slot, similarity = retrieve_residual(
        memory, key, threshold=0.982)
    assert not accepted and slot is None and similarity == 0.0
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(
        residual_scores(model, features, residual),
        model.memory_replacement_scores(features))


def test_contextual_full_residual_rejects_dissimilar_old_context() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=18, value_width=8)
    new_key = torch.nn.functional.normalize(
        torch.tensor([1.0] + [0.0] * 17), dim=0)
    old_key = torch.nn.functional.normalize(
        torch.tensor([0.0, 1.0] + [0.0] * 16), dim=0)
    value = torch.arange(8, dtype=torch.float32)
    memory.upsert(new_key, value, verified_improvement=1.0)
    accepted_value, accepted, slot, _ = retrieve_residual(
        memory, new_key, threshold=0.982)
    assert accepted and slot == 0
    assert torch.equal(accepted_value, value)
    rejected_value, accepted, slot, similarity = retrieve_residual(
        memory, old_key, threshold=0.982)
    assert not accepted and slot is None and similarity == 0.0
    assert torch.count_nonzero(rejected_value) == 0


def test_suppress_novelty_adapter_is_exact_noop_at_zero() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    features = torch.randn(2, 4, 8)
    assert torch.equal(
        adapter_scores(
            model, features, torch.zeros(2),
            mode="suppress_novelty"),
        model.memory_replacement_scores(features))
    adapter = torch.tensor([-6.0, -4.0])
    scores = adapter_scores(
        model, features, adapter, mode="suppress_novelty")
    assert scores.shape == (2, 4)
    assert not torch.equal(
        scores, model.memory_replacement_scores(features))


def test_verified_candidate_selection_prefers_center_on_ties() -> None:
    assert select_verified_candidate(
        torch.tensor([0.8, 0.8, 0.8])) == 1
    assert select_verified_candidate(
        torch.tensor([0.8000001, 0.8, 0.7])) == 1
    assert select_verified_candidate(
        torch.tensor([0.82, 0.8, 0.7])) == 0
    assert select_verified_candidate(
        torch.tensor([0.7, 0.8, 0.83])) == 2


def test_disk_memory_can_replace_without_growing(tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert memory.commit(
        keys, values, torch.ones(2), threshold=0.0) == 2
    replacement_key = torch.eye(4)[2]
    replacement_value = torch.full((4,), 7.0)
    memory.replace(0, replacement_key, replacement_value, 0.8)
    assert memory.count == 2
    path = tmp_path / "replaced.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    read, _ = restored.retrieve(
        replacement_key.unsqueeze(0), top_k=1,
        confidence_mode="cosine")
    assert torch.allclose(read, replacement_value.unsqueeze(0))
    assert restored.count == 2


def test_recurring_context_signature_is_stable_within_world() -> None:
    batch = generate_lifetimes(
        8, 3, seed=31, task="binary_mapping", support_trials=1)
    marked = _add_context_signatures(batch, seed=41)
    signatures = marked.frames[:, :, :, 2:5, 2:5]
    assert torch.equal(signatures[:, 0], signatures[:, 1])
    assert torch.equal(signatures[:, 1], signatures[:, 2])
    assert not torch.equal(signatures[0, 0], signatures[1, 0])


def test_grouped_memory_read_never_crosses_memory_banks() -> None:
    keys = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [0.0, 1.0],
    ])
    values = torch.tensor([
        [1.0, 10.0], [2.0, 20.0],
        [3.0, 30.0], [4.0, 40.0],
    ])
    read, selected = _grouped_read(
        keys, keys, values, capacity=2, mode="hard")
    assert torch.equal(read, values)
    assert torch.equal(selected, torch.tensor([0, 1, 0, 1]))
