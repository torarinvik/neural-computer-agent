import torch

from .recurrent_relational_gate import RecurrentRelationalGate


def test_relational_gate_starts_as_exact_noop() -> None:
    torch.manual_seed(7)
    gate = RecurrentRelationalGate(
        event_width=8, action_count=4, hidden_width=16,
        intention_width=6, max_history=4)
    event = torch.randn(3, 8)
    state = gate.initial_state(3, device=event.device)
    action = torch.tensor([0, 1, 3])
    reward = torch.tensor([0.0, 1.0, -1.0])
    feedback = torch.ones(3)
    residual, new_state, snapshot = gate(
        event, state, [], action, reward, feedback)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert new_state.shape == (3, 16)
    assert snapshot.shape == (3, 16)


def test_relational_gate_reads_only_retained_history() -> None:
    torch.manual_seed(8)
    gate = RecurrentRelationalGate(
        event_width=8, action_count=4, hidden_width=16,
        intention_width=6, max_history=2)
    with torch.no_grad():
        gate.output.weight.normal_(0.0, 0.1)
    event = torch.randn(3, 8)
    state = gate.initial_state(3, device=event.device)
    history = [torch.randn(3, 16) for _ in range(4)]
    args = (torch.zeros(3, dtype=torch.long), torch.ones(3), torch.ones(3))
    short, _, _ = gate(event, state, history[-2:], *args)
    long, _, _ = gate(event, state, history, *args)
    assert short.shape == long.shape == (3, 6)
    assert torch.isfinite(short).all() and torch.isfinite(long).all()

