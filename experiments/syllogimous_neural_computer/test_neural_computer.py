from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.syllogimous_bitter_lesson.parity_transfer import generate_parity_episode
from experiments.syllogimous_latent_agent.data import collate_episodes

from .memory import PersistentMemory
from .consolidation import (ConsolidationProposal, LearnedConsolidator, ReplayScore,
                            apply_proposal, transactional_consolidate,
                            transactional_consolidate_many)
from .model import NeuralComputerAgent, parameter_count
from .context_selection import ActiveContextSelector
from .lifetime import generate_sensory_lifetime
from .train_lifetime import run_lifetimes, straight_through_admission
from .training_memory import DifferentiableBatchMemory
from .train_continual import context_streams, run_streams
from .train_consolidation import consolidate_stream, split_queries
from . import train_consolidation as consolidation_training


def test_memory_grows_instead_of_evicting():
    memory = PersistentMemory.empty(2, 8, growth_chunk=2)
    keys = torch.eye(8)[:5]
    values = keys.roll(1, dims=1)
    assert memory.write(keys, values, torch.ones(5)) == 5
    assert memory.count == 5
    assert memory.capacity == 6
    assert torch.equal(memory.keys[:5], keys)


def test_persistent_memory_round_trip(tmp_path: Path):
    memory = PersistentMemory.empty(1, 4, growth_chunk=3)
    memory.write(torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                 torch.tensor([[0.0, 1.0, 0.0, 0.0]]), torch.ones(1))
    path = tmp_path / "lifetime-memory.pt"
    memory.save(path)
    restored = PersistentMemory.load(path)
    recalled, confidence = restored.read(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert restored.count == 1
    assert torch.allclose(recalled, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    assert confidence.item() > 0.99


def test_long_term_rows_transfer_into_compact_active_memory():
    memory = PersistentMemory.empty(4, 4)
    memory.write(torch.eye(4)[:3], torch.eye(4)[[1, 2, 3]],
                 torch.ones(3))
    active = memory.select([2])
    assert active.count == active.capacity == 1
    assert torch.equal(active.keys[0], memory.keys[2])
    assert torch.equal(active.values[0], memory.values[2])
    assert memory.count == 3
    empty = memory.select([])
    assert empty.count == 0 and empty.capacity == 1


def test_active_selector_sees_only_sensory_and_latent_memory():
    memory = PersistentMemory.empty(3, 4)
    memory.write(torch.eye(4)[:3], torch.eye(4)[[1, 2, 3]], torch.ones(3))
    selector = ActiveContextSelector(4, hidden=8)
    logits, indices = selector(torch.ones(4), memory)
    assert logits.shape == (4,)  # one null choice plus three latent rows
    assert indices.tolist() == [0, 1, 2]
    active, selected = selector.select(torch.ones(4), memory)
    assert active.count in (0, 1)
    assert selected is None or selected in indices.tolist()


def test_retrieval_summary_is_sensory_only_and_deterministic():
    episode = generate_sensory_lifetime(7, associations=1, delay=0).episodes[-1]
    batch = collate_episodes([episode])
    model = NeuralComputerAgent(hidden=40, workspace_slots=4, heads=4,
                                thought_steps=2, action_count=8)
    first = model.retrieval_summary(batch["frames"], batch["pcm"], batch["mask"])
    repeated = model.retrieval_summary(batch["frames"], batch["pcm"], batch["mask"])
    assert first.shape == (1, 40)
    assert torch.allclose(first, repeated)


def test_write_gate_can_leave_memory_unchanged():
    memory = PersistentMemory.empty(1, 4)
    committed = memory.write(torch.ones(2, 4), torch.ones(2, 4),
                             torch.tensor([0.1, 0.2]), threshold=0.5)
    assert committed == 0
    assert memory.count == 0


def test_agent_boundary_is_sensory_and_own_memory_only():
    batch = collate_episodes([generate_parity_episode(1, 2),
                              generate_parity_episode(2, 2)])
    model = NeuralComputerAgent(hidden=40, workspace_slots=4, heads=4,
                                thought_steps=3, action_count=8)
    memory = PersistentMemory.empty(2, 40, growth_chunk=2)
    before = memory.clone()
    output = model(batch["frames"], batch["pcm"], batch["mask"], memory)
    assert output.answer_logits.shape == (2, 3, 8)
    assert output.observation_logits.shape[:2] == batch["mask"].shape
    assert output.write_keys.shape == (2, 40)
    # Forward proposes writes but cannot mutate durable state implicitly.
    assert memory.count == before.count == 0
    assert parameter_count(model) > 0


def test_commit_is_explicit_and_controller_owned():
    batch = collate_episodes([generate_parity_episode(3, 2)])
    model = NeuralComputerAgent(hidden=40, workspace_slots=4, heads=4,
                                thought_steps=2, action_count=8)
    memory = PersistentMemory.empty(1, 40)
    output = model(batch["frames"], batch["pcm"], batch["mask"], memory)
    output.write_strengths.fill_(1.0)
    assert model.commit(memory, output) == 1
    assert memory.count == 1


def test_visual_lifetime_is_deterministic_and_requires_prior_study():
    first = generate_sensory_lifetime(17, associations=4, delay=12)
    repeated = generate_sensory_lifetime(17, associations=4, delay=12)
    assert len(first.episodes) == 20
    assert all(np.array_equal(a.frames, b.frames)
               for a, b in zip(first.episodes, repeated.episodes))
    assert [item.actions.item() for item in first.episodes] == \
           [item.actions.item() for item in repeated.episodes]
    assert all(item.actions.item() == int(1) for item in first.episodes[:4])
    assert all(item.actions.item() == int(0) for item in first.episodes[4:16])
    assert all(0 <= item.actions.item() < 8 for item in first.episodes[16:])


def test_lifetime_assignments_change_across_seeds():
    first = generate_sensory_lifetime(1, associations=4, delay=0)
    second = generate_sensory_lifetime(2, associations=4, delay=0)
    first_answers = [episode.actions.item() for episode in first.episodes[4:]]
    second_answers = [episode.actions.item() for episode in second.episodes[4:]]
    assert first_answers != second_answers


def test_audit_query_is_new_sensory_view_of_same_public_problem():
    lifetime = generate_sensory_lifetime(9, associations=2, delay=0,
                                         contextual=True, audit_variants=1)
    queries = lifetime.episodes[-2:]
    assert len(lifetime.audit_queries) == 2
    assert [item.actions.item() for item in queries] == [
        item.actions.item() for item in lifetime.audit_queries]
    assert all(not np.array_equal(original.frames, audit.frames)
               for original, audit in zip(queries, lifetime.audit_queries))
    assert all(not np.array_equal(original.pcm, audit.pcm)
               for original, audit in zip(queries, lifetime.audit_queries))


def test_differentiable_memory_carries_gradient_to_earlier_write():
    memory = DifferentiableBatchMemory(2, 4, device=torch.device("cpu"))
    keys = torch.randn(2, 4, requires_grad=True)
    values = torch.randn(2, 4, requires_grad=True)
    strengths = torch.full((2,), 0.8, requires_grad=True)
    memory = memory.append(keys, values, strengths)
    recalled, _ = memory.read(keys.detach())
    recalled.sum().backward()
    assert values.grad is not None
    assert strengths.grad is not None


def test_all_benchmark_conditions_execute_without_hidden_inputs():
    lifetimes = [generate_sensory_lifetime(seed, associations=1, delay=1)
                 for seed in (1, 2)]
    model = NeuralComputerAgent(hidden=40, workspace_slots=4, heads=4,
                                thought_steps=2, action_count=8)
    for condition in ("no_memory", "random_write", "learned_memory"):
        loss, metrics = run_lifetimes(model, lifetimes, condition,
                                      torch.device("cpu"), training=True)
        assert loss.isfinite()
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_memory_counterfactuals_preserve_shape_but_destroy_contents():
    memory = DifferentiableBatchMemory(3, 4, device=torch.device("cpu"))
    memory = memory.append(torch.arange(12, dtype=torch.float32).reshape(3, 4),
                           torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
                           torch.tensor([0.7, 0.8, 0.9]))
    shuffled = memory.counterfactual("shuffled")
    garbage = memory.counterfactual("garbage")
    empty = memory.counterfactual("empty")
    assert shuffled.keys.shape == garbage.keys.shape == memory.keys.shape
    assert torch.equal(shuffled.keys, memory.keys)
    assert not torch.equal(shuffled.values, memory.values)
    assert not torch.equal(garbage.keys, memory.keys)
    assert empty.count == 0


def test_discrete_admission_is_binary_forward_and_differentiable_backward():
    logits = torch.tensor([-2.0, 2.0], requires_grad=True)
    gates = straight_through_admission(logits, 1.0, stochastic=False, threshold=0.5)
    assert gates.tolist() == [0.0, 1.0]
    gates.sum().backward()
    assert logits.grad is not None
    assert torch.all(logits.grad > 0)


def test_context_stream_uses_shared_visible_marker_and_retention_query():
    streams = context_streams(10, 2, 3, 1, 8)
    assert len(streams) == 2 and len(streams[0]) == 3
    # The context marker is repeated in study and query frames of one context.
    study = streams[0][0].episodes[0].frames[0, 7:44, 5:36]
    query = streams[0][0].episodes[-1].frames[0, 7:44, 5:36]
    assert np.array_equal(study, query)
    other = streams[0][1].episodes[0].frames[0, 7:44, 5:36]
    assert not np.array_equal(study, other)


def test_continual_stream_runs_with_one_growing_memory_per_row():
    streams = context_streams(20, 2, 2, 0, 8)
    model = NeuralComputerAgent(hidden=40, workspace_slots=4, heads=4,
                                thought_steps=2, action_count=8)
    loss, metrics = run_streams(model, streams, torch.device("cpu"), threshold=0.05)
    assert loss.isfinite()
    assert len(metrics["retention_by_context"]) == 2
    assert metrics["stored_rows_per_stream"] == 4


def _drop_second_proposal(memory: PersistentMemory) -> ConsolidationProposal:
    return ConsolidationProposal(0, 1, memory.keys[0].clone(), memory.values[0].clone(),
                                 memory.usage[0].clone(), torch.tensor([0.0, 1.0, 0.0]))


def test_consolidation_proposal_uses_memory_latents_only():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))
    consolidator = LearnedConsolidator(4, hidden=8)
    proposal = consolidator.propose(memory)
    assert proposal is not None
    assert proposal.first != proposal.second
    assert proposal.key.shape == proposal.value.shape == (4,)
    assert proposal.operation_logits.shape == (3,)


def test_sampled_consolidation_receives_policy_gradient():
    memory = PersistentMemory.empty(3, 4)
    memory.write(torch.eye(4)[:3], torch.eye(4)[[1, 2, 3]], torch.ones(3))
    consolidator = LearnedConsolidator(4, hidden=8)
    proposal = consolidator.sample(memory)
    assert proposal is not None and proposal.log_probability is not None
    (-proposal.log_probability * 0.5).backward()
    assert any(parameter.grad is not None for parameter in consolidator.parameters())


def test_candidate_rewrite_does_not_mutate_transaction_base():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))
    before = memory.clone()
    candidate = apply_proposal(memory, _drop_second_proposal(memory))
    assert memory.count == before.count == 2
    assert torch.equal(memory.keys, before.keys)
    assert candidate.count == 1


def test_transaction_rolls_back_when_replay_accuracy_degrades():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))

    def verifier(candidate):
        return ReplayScore(candidate.count, 2)

    result = transactional_consolidate(memory, _drop_second_proposal(memory), verifier)
    assert not result.committed
    assert result.memory is memory
    assert result.memory.count == 2
    assert result.reward < 0
    assert result.provenance is not None


def test_transaction_commits_lossless_compression_and_rewards_storage():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))

    def invariant_verifier(candidate):
        return ReplayScore(8, 8)

    result = transactional_consolidate(memory, _drop_second_proposal(memory),
                                       invariant_verifier, invariant_verifier)
    assert result.committed
    assert result.memory.count == 1 and memory.count == 2
    assert result.rows_saved == 1
    assert result.reward > 0


def test_loss_guard_rejects_confidence_regression_at_equal_accuracy():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))

    def verifier(candidate):
        return ReplayScore(8, 8, loss=0.1 if candidate.count == 2 else 0.2)

    result = transactional_consolidate(memory, _drop_second_proposal(memory), verifier,
                                       loss_tolerance=0.0)
    assert not result.committed
    assert result.memory.count == 2


def test_every_rehearsal_group_must_preserve_accuracy():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))
    safe = lambda candidate: ReplayScore(1, 1)
    vulnerable = lambda candidate: ReplayScore(1 if candidate.count == 2 else 0, 1)
    result = transactional_consolidate_many(
        memory, _drop_second_proposal(memory), [safe, safe, vulnerable])
    assert not result.committed
    assert result.memory.count == 2


def test_autonomous_stop_preserves_memory_without_verifier_work():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))
    policy = LearnedConsolidator(4, hidden=8)
    for parameter in policy.rewrite_head.parameters():
        parameter.data.zero_()
    policy.rewrite_head[-1].bias.data.fill_(-20.0)
    model = NeuralComputerAgent(hidden=4, workspace_slots=2, heads=1,
                                thought_steps=1, action_count=8)
    episode = generate_sensory_lifetime(3, associations=1, delay=0).episodes[-1]
    result, accepted, _, queries, stopped = consolidate_stream(
        policy, None, memory, [episode], model, torch.device("cpu"), 4,
        training=False, rehearsal_groups=1, autonomous_stop=True)
    assert stopped and accepted == 0 and queries == 0
    assert result is memory and result.count == 2


def test_trajectory_stop_uses_state_value_without_verifier_work():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))
    policy = LearnedConsolidator(4, hidden=8)
    for parameter in policy.stop_head.parameters():
        parameter.data.zero_()
    policy.stop_head[-1].bias.data.fill_(20.0)
    model = NeuralComputerAgent(hidden=4, workspace_slots=2, heads=1,
                                thought_steps=1, action_count=8)
    episode = generate_sensory_lifetime(3, associations=1, delay=0).episodes[-1]
    result, accepted, _, queries, stopped = consolidate_stream(
        policy, None, memory, [episode], model, torch.device("cpu"), 4,
        training=False, rehearsal_groups=1, autonomous_stop=True,
        trajectory_stop=True)
    assert stopped and accepted == 0 and queries == 0
    assert result is memory and result.count == 2


def test_stop_threshold_above_one_is_exact_forced_continuation_control():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))
    policy = LearnedConsolidator(4, hidden=8)
    for parameter in policy.stop_head.parameters():
        parameter.data.zero_()
    policy.stop_head[-1].bias.data.fill_(20.0)
    model = NeuralComputerAgent(hidden=4, workspace_slots=2, heads=1,
                                thought_steps=1, action_count=8)
    episode = generate_sensory_lifetime(3, associations=1, delay=0).episodes[-1]
    _, _, _, queries, stopped = consolidate_stream(
        policy, None, memory, [episode], model, torch.device("cpu"), 1,
        training=False, rehearsal_groups=1, autonomous_stop=True,
        trajectory_stop=True, stop_threshold=1.01)
    assert not stopped
    assert queries == 2


def test_calibration_rejects_safe_cutoff_without_observed_savings(monkeypatch):
    def fake_evaluate(policy, model, device, args, *, streams, seed):
        return {"consolidated_accuracy": 0.75,
                "consolidated_audit_accuracy": 0.75,
                "consolidated_loss": 0.5,
                "consolidated_audit_loss": 0.5,
                "verifier_queries": 80.0}

    monkeypatch.setattr(consolidation_training, "evaluate", fake_evaluate)
    args = SimpleNamespace(autonomous_stop=True, stop_threshold=0.5)
    threshold, report = consolidation_training.calibrate_stop_threshold(
        None, None, None, args, streams=8, seed=1)
    assert threshold == 1.01
    assert len(report["trials"]) == 1
    assert not report["trials"][0]["useful"]


def test_state_stop_value_is_invariant_to_memory_row_order():
    memory = PersistentMemory.empty(3, 4)
    memory.write(torch.eye(4)[:3], torch.eye(4)[[1, 2, 3]],
                 torch.tensor([0.2, 0.7, 1.0]))
    permuted = memory.clone()
    order = torch.tensor([2, 0, 1])
    for field in ("keys", "values", "usage", "age", "valid"):
        source = getattr(memory, field)
        getattr(permuted, field).copy_(source[order])
    policy = LearnedConsolidator(4, hidden=8)
    assert torch.allclose(policy.stop_logit(memory), policy.stop_logit(permuted),
                          atol=1e-6)


def test_heldout_failure_prevents_replay_overfit_commit():
    memory = PersistentMemory.empty(2, 4)
    memory.write(torch.eye(4)[:2], torch.eye(4)[2:], torch.ones(2))

    replay = lambda candidate: ReplayScore(8, 8)
    heldout = lambda candidate: ReplayScore(8 if candidate.count == 2 else 6, 8)
    result = transactional_consolidate(memory, _drop_second_proposal(memory), replay, heldout)
    assert not result.committed
    assert result.memory.count == 2


def test_consolidation_audit_partition_is_never_used_for_commit():
    queries = list(range(8))
    replay, heldout, audit = split_queries(queries)
    assert replay == [0, 1, 2]
    assert heldout == [3, 4, 5]
    assert audit == [6, 7]
    assert not set(audit) & (set(replay) | set(heldout))
