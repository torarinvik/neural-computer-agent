import torch

from experiments.syllogimous_neural_computer.model import (
    ContentAddressedEventMemoryReader, EventIndexedMemoryReader,
    FactorizedEventAnswerRouter, NeuralComputerAgent)
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory

from .environment import generate_attention_lifetime
from .train import _control, _latest_row, run_batch


def test_memory_controls_preserve_shapes_and_break_correspondence():
    memory = DifferentiableBatchMemory(2, 4, device=torch.device("cpu"))
    for offset in range(3):
        keys = torch.arange(8, dtype=torch.float32).reshape(2, 4) + offset * 10
        values = keys + 100
        memory = memory.append(keys, values, torch.ones(2), torch.ones(2))
    shuffled = _control(memory, "shuffled")
    unrelated = _control(memory, "unrelated")
    garbage = _control(memory, "garbage")
    assert shuffled.count == unrelated.count == garbage.count == memory.count
    assert torch.equal(shuffled.keys, memory.keys)
    assert torch.equal(shuffled.values[0], memory.values[1])
    assert torch.equal(unrelated.keys[0], memory.keys[1])
    assert not torch.equal(garbage.keys, memory.keys)
    latest = _latest_row(memory)
    assert latest.count == 1
    assert torch.equal(latest.values[:, 0], memory.values[:, -1])


def test_tiny_forward_transfer_batch_has_all_metrics_and_gradients():
    torch.manual_seed(3)
    model = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                thought_steps=1, action_count=8)
    lifetimes = [generate_attention_lifetime(100 + index, query_count=1)
                 for index in range(2)]
    loss, metrics = run_batch(model, lifetimes, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.write_key.weight.grad is not None
    for key in ("accuracy_0_shot", "accuracy_1_shot", "accuracy_2_shot",
                "accuracy_4_shot", "few_shot_auc", "retention_accuracy"):
        assert key in metrics


def test_optional_order_route_is_backward_compatible_and_trainable():
    base = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                               thought_steps=1, action_count=8)
    routed = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                 thought_steps=1, action_count=8,
                                 order_routing=True)
    missing = routed.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing
    assert all(key.startswith("answer_route") for key in missing)
    assert not any(key.startswith("answer_route") for key in base.state_dict())
    base.eval()
    routed.eval()
    # Zero-initialized routing is exactly behavior-preserving before training.
    lifetime = generate_attention_lifetime(499, query_count=1)
    from experiments.syllogimous_latent_agent.data import collate_episodes
    batch = collate_episodes([lifetime.future_queries[0]])
    memory = DifferentiableBatchMemory(1, 20, device=torch.device("cpu"))
    with torch.no_grad():
        base_logits = base(batch["frames"], batch["pcm"], batch["mask"], memory).answer_logits
        routed_logits = routed(
            batch["frames"], batch["pcm"], batch["mask"], memory).answer_logits
    assert torch.equal(base_logits, routed_logits)
    lifetimes = [generate_attention_lifetime(500 + index, query_count=1)
                 for index in range(2)]
    loss, _ = run_batch(routed, lifetimes, torch.device("cpu"))
    loss.backward()
    assert routed.answer_route[0].weight.grad is not None
    assert routed.answer_route_gate.weight.grad is not None


def test_optional_write_binding_is_backward_compatible_and_trainable():
    base = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                               thought_steps=1, action_count=8)
    bound = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                thought_steps=1, action_count=8,
                                write_binding=True)
    missing = bound.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing
    assert all(key.startswith("write_binding") for key in missing)
    assert not any(key.startswith("write_binding") for key in base.state_dict())
    base.eval()
    bound.eval()
    lifetime = generate_attention_lifetime(599, query_count=1)
    from experiments.syllogimous_latent_agent.data import collate_episodes
    batch = collate_episodes([lifetime.supports[0]])
    memory = DifferentiableBatchMemory(1, 20, device=torch.device("cpu"))
    with torch.no_grad():
        base_output = base(batch["frames"], batch["pcm"], batch["mask"], memory)
        bound_output = bound(batch["frames"], batch["pcm"], batch["mask"], memory)
    assert torch.equal(base_output.write_keys, bound_output.write_keys)
    assert torch.equal(base_output.write_values, bound_output.write_values)
    assert torch.equal(base_output.write_logits, bound_output.write_logits)
    lifetimes = [generate_attention_lifetime(600 + index, query_count=1)
                 for index in range(2)]
    loss, _ = run_batch(bound, lifetimes, torch.device("cpu"))
    loss.backward()
    assert bound.write_binding_mlp[-1].weight.grad is not None
    assert bound.write_binding_attention.in_proj_weight.grad is not None


def test_factorized_answer_router_is_exact_noop_at_zero_strength():
    base = NeuralComputerAgent(
        hidden=20, workspace_slots=3, heads=2, thought_steps=1,
        action_count=8, latest_row_reader=True)
    routed = NeuralComputerAgent(
        hidden=20, workspace_slots=3, heads=2, thought_steps=1,
        action_count=8, latest_row_reader=True,
        latest_row_answer_factorized_router=True)
    missing = routed.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing
    assert all(key.startswith("latest_row_factorized") for key in missing)
    lifetime = generate_attention_lifetime(649, query_count=1)
    from experiments.syllogimous_latent_agent.data import collate_episodes
    batch = collate_episodes([lifetime.future_queries[0]])
    memory = DifferentiableBatchMemory(
        1, 20, device=torch.device("cpu"),
        keys=torch.randn(1, 1, 20), values=torch.randn(1, 1, 20),
        strengths=torch.ones(1, 1), admissions=torch.ones(1, 1))
    base.eval()
    routed.eval()
    captured = []
    handle = routed.latest_row_factorized_router.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone()))
    with torch.no_grad():
        base_logits = base(
            batch["frames"], batch["pcm"], batch["mask"], memory).answer_logits
        routed_logits = routed(
            batch["frames"], batch["pcm"], batch["mask"], memory).answer_logits
    handle.remove()
    assert torch.equal(base_logits, routed_logits)
    assert captured
    assert torch.equal(captured[-1], memory.keys[:, -1])


def test_factorized_router_normalization_is_checkpoint_portable():
    torch.manual_seed(17)
    normalized = FactorizedEventAnswerRouter(hidden=20)
    portable = FactorizedEventAnswerRouter(hidden=20)
    portable.load_state_dict(normalized.state_dict())
    raw = {
        "support": torch.randn(4, 20) * 2 + 3,
        "first": torch.randn(4, 20) * 4 - 1,
        "second": torch.randn(4, 20) * 3 + 2,
    }
    means = {key: value.mean(0, keepdim=True) for key, value in raw.items()}
    scales = {
        key: value.std(0, keepdim=True).clamp_min(1e-5)
        for key, value in raw.items()}
    for key in raw:
        getattr(portable, key + "_mean").copy_(means[key])
        getattr(portable, key + "_scale").copy_(scales[key])
    standardized = {
        key: (value - means[key]) / scales[key] for key, value in raw.items()}
    with torch.no_grad():
        expected = normalized(**standardized)["hard_action"]
        actual = portable(**raw)["hard_action"]
    assert torch.equal(expected, actual)


def test_factorized_router_candidate_override_is_exact_noop_at_zero():
    torch.manual_seed(19)
    router = FactorizedEventAnswerRouter(hidden=20)
    support = torch.randn(4, 20)
    first = torch.randn(4, 20)
    second = torch.randn(4, 20)
    first_override = torch.randn(4, 8)
    second_override = torch.randn(4, 8)
    with torch.no_grad():
        base = router(support, first, second)
        zero = router(
            support, first, second,
            first_action_override=first_override,
            second_action_override=second_override,
            override_strength=torch.tensor(0.0))
        full = router(
            support, first, second,
            first_action_override=first_override,
            second_action_override=second_override,
            override_strength=torch.tensor(1.0))
    assert torch.equal(base["hard_action"], zero["hard_action"])
    assert torch.allclose(full["first_action"], first_override, atol=1e-6)
    assert torch.allclose(full["second_action"], second_override, atol=1e-6)


def test_event_indexed_reader_is_order_invariant_and_normalization_portable():
    torch.manual_seed(23)
    raw_rows = torch.randn(5, 3, 20) * 3 + 2
    raw_query = torch.randn(5, 20) * 2 - 1
    normalized = EventIndexedMemoryReader(hidden=20, width=12)
    portable = EventIndexedMemoryReader(hidden=20, width=12)
    portable.load_state_dict(normalized.state_dict())
    rows_mean = raw_rows.mean((0, 1), keepdim=True)
    rows_scale = raw_rows.std((0, 1), keepdim=True).clamp_min(1e-5)
    query_mean = raw_query.mean(0, keepdim=True)
    query_scale = raw_query.std(0, keepdim=True).clamp_min(1e-5)
    portable.rows_mean.copy_(rows_mean)
    portable.rows_scale.copy_(rows_scale)
    portable.query_mean.copy_(query_mean)
    portable.query_scale.copy_(query_scale)
    with torch.no_grad():
        expected = normalized(
            (raw_rows - rows_mean) / rows_scale,
            (raw_query - query_mean) / query_scale)
        actual = portable(raw_rows, raw_query)
        permuted = portable(raw_rows[:, torch.tensor([2, 0, 1])], raw_query)
    assert torch.equal(expected, actual)
    assert torch.allclose(actual, permuted, atol=1e-6, rtol=1e-6)


def test_content_addressed_reader_is_order_invariant():
    torch.manual_seed(29)
    reader = ContentAddressedEventMemoryReader(hidden=20, width=12)
    rows = torch.randn(5, 3, 20)
    query = torch.randn(5, 20)
    with torch.no_grad():
        original = reader(rows, query)
        permuted = reader(rows[:, torch.tensor([2, 0, 1])], query)
    assert torch.allclose(original, permuted, atol=1e-6, rtol=1e-6)


def test_optional_event_indexed_reader_is_exact_noop_at_zero_strength():
    base = NeuralComputerAgent(
        hidden=20, workspace_slots=3, heads=2, thought_steps=1,
        action_count=8)
    indexed = NeuralComputerAgent(
        hidden=20, workspace_slots=3, heads=2, thought_steps=1,
        action_count=8, event_indexed_memory_reader=True,
        event_indexed_memory_reader_width=12)
    missing = indexed.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing
    assert all(key.startswith("event_indexed_memory_reader") for key in missing)
    lifetime = generate_attention_lifetime(679, query_count=1)
    from experiments.syllogimous_latent_agent.data import collate_episodes
    batch = collate_episodes([lifetime.future_queries[0]])
    memory = DifferentiableBatchMemory(
        1, 20, device=torch.device("cpu"),
        keys=torch.randn(1, 3, 20), values=torch.randn(1, 3, 20),
        strengths=torch.ones(1, 3), admissions=torch.ones(1, 3))
    base.eval()
    indexed.eval()
    with torch.no_grad():
        base_output = base(
            batch["frames"], batch["pcm"], batch["mask"], memory)
        indexed_output = indexed(
            batch["frames"], batch["pcm"], batch["mask"], memory)
    for field in ("observation_logits", "answer_logits", "halt_logits", "values",
                  "write_keys", "write_values", "write_logits",
                  "write_strengths", "read_confidence", "workspace",
                  "read_context", "write_source", "event_binding_residual"):
        assert torch.equal(
            getattr(base_output, field), getattr(indexed_output, field))


def test_optional_event_binding_is_exact_noop_and_trainable():
    base = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                               thought_steps=1, action_count=8)
    bound = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                thought_steps=1, action_count=8,
                                event_binding=True, event_binding_width=12)
    missing = bound.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing
    assert all(key.startswith("event_binding_module") for key in missing)
    base.eval()
    bound.eval()
    lifetime = generate_attention_lifetime(699, query_count=1)
    from experiments.syllogimous_latent_agent.data import collate_episodes
    batch = collate_episodes([lifetime.supports[0]])
    memory = DifferentiableBatchMemory(1, 20, device=torch.device("cpu"))
    with torch.no_grad():
        base_output = base(batch["frames"], batch["pcm"], batch["mask"], memory)
        bound_output = bound(batch["frames"], batch["pcm"], batch["mask"], memory)
    for field in ("observation_logits", "answer_logits", "halt_logits", "values",
                  "write_keys", "write_values", "write_logits", "write_strengths",
                  "read_confidence", "workspace", "write_source"):
        assert torch.equal(getattr(base_output, field), getattr(bound_output, field))
    assert torch.count_nonzero(bound_output.event_binding_residual) == 0

    bound.train()
    optimizer = torch.optim.SGD(bound.event_binding_module.parameters(), lr=0.1)
    lifetimes = [generate_attention_lifetime(700 + index, query_count=1)
                 for index in range(2)]
    first_loss, _ = run_batch(bound, lifetimes, torch.device("cpu"))
    first_loss.backward()
    assert bound.event_binding_module.relation[-1].weight.grad is not None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss, _ = run_batch(bound, lifetimes, torch.device("cpu"))
    second_loss.backward()
    assert bound.event_binding_module.project[1].weight.grad is not None


def test_event_binding_keeps_last_three_valid_states_in_order():
    from experiments.syllogimous_neural_computer.model import EventSnapshotWriteBinder

    states = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[True, True, True, True, True],
                         [True, True, False, False, False]])
    recent, valid = EventSnapshotWriteBinder.recent_events(states, mask)
    assert torch.equal(recent[0], states[0, 2:5])
    assert torch.equal(recent[1, :2], states[1, :2])
    assert valid.tolist() == [[True, True, True], [True, True, False]]


def test_counterfactual_margin_rewards_only_inherited_improvement():
    torch.manual_seed(7)
    model = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                thought_steps=1, action_count=8)
    lifetimes = [generate_attention_lifetime(300 + index, query_count=1)
                 for index in range(2)]
    _, baseline = run_batch(model, lifetimes, torch.device("cpu"), condition="empty")
    loss, metrics = run_batch(
        model, lifetimes, torch.device("cpu"),
        reference_future_loss=baseline["future_loss"],
        advantage_margin=0.5, advantage_weight=1.0)
    assert metrics["advantage_margin_penalty"] >= 0.0
    loss.backward()
    assert model.write_value.weight.grad is not None
