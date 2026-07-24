import torch

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory

from .consolidator import LatentConsolidator
from .environment import generate_attention_lifetime, generate_temporal_attention_lifetime
from .train_consolidator import run_compaction_batch


def test_consolidator_emits_one_differentiable_row():
    memory = DifferentiableBatchMemory(2, 20, device=torch.device("cpu"))
    for _ in range(3):
        memory = memory.append(torch.randn(2, 20), torch.randn(2, 20), torch.ones(2))
    module = LatentConsolidator(20, heads=2, layers=1)
    compact = module(memory)
    assert compact.count == 1
    assert compact.keys.shape == compact.values.shape == (2, 1, 20)
    compact.values.sum().backward()
    assert module.value_head.weight.grad is not None


def test_tiny_behavior_compaction_batch_backpropagates_only_to_consolidator():
    torch.manual_seed(9)
    model = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                thought_steps=1, action_count=8, read_top_k=1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    consolidator = LatentConsolidator(20, heads=2, layers=1)
    lifetimes = [generate_attention_lifetime(900 + index, query_count=1)
                 for index in range(2)]
    loss, metrics = run_compaction_batch(
        model, consolidator, lifetimes, torch.device("cpu"), train=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert consolidator.row_projection[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert metrics["rows_saved"] == 4.0


def test_write_rule_auxiliary_reaches_real_event_binder_through_raw_write():
    torch.manual_seed(10)
    model = NeuralComputerAgent(hidden=20, workspace_slots=3, heads=2,
                                thought_steps=1, action_count=8, read_top_k=1,
                                event_binding=True, event_binding_width=12)
    consolidator = LatentConsolidator(20, heads=2, layers=1)
    head = torch.nn.Linear(40, 2)
    lifetimes = [generate_temporal_attention_lifetime(
        950 + index, query_count=1, feedback_mode="color-button")
        for index in range(2)]
    targets = torch.tensor([item.rule for item in lifetimes])

    def auxiliary(output):
        row = torch.cat((output.write_keys, output.write_values), dim=-1)
        return torch.nn.functional.cross_entropy(head(row), targets)

    loss, metrics = run_compaction_batch(
        model, consolidator, lifetimes, torch.device("cpu"), train=True,
        train_model=True, write_auxiliary_loss=auxiliary,
        write_auxiliary_weight=1.0, write_residual_penalty_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert "write_auxiliary_loss" in metrics
    assert "write_residual_penalty" in metrics
    assert model.event_binding_module.relation[-1].weight.grad is not None
    assert head.weight.grad is not None
