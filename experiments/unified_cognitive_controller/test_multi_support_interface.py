from types import SimpleNamespace

import torch

from .audit_selective_disk import _support


class _TraceModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []

    def initial_state(self, count: int, *, device: torch.device) -> object:
        return object()

    def step(
            self, frame: torch.Tensor, state: object, action: torch.Tensor,
            outcome: torch.Tensor, feedback: torch.Tensor, *,
            retrieved_memory: torch.Tensor | None = None,
    ) -> tuple[SimpleNamespace, object]:
        marker = int(frame[0, 0].item())
        self.calls.append((marker, feedback.clone(), retrieved_memory))
        count = frame.shape[0]
        return SimpleNamespace(
            logits=torch.zeros(count, 2),
            memory_key=torch.full((count, 1), float(marker)),
            memory_value=torch.full((count, 1), float(marker + 10)),
            memory_write_strength=torch.full((count,), float(marker + 20)),
        ), state


def test_multi_support_consumes_each_feedback_before_query() -> None:
    model = _TraceModel()
    batch = SimpleNamespace(
        frames=torch.tensor([[[0.0], [1.0], [2.0], [3.0]],
                             [[0.0], [1.0], [2.0], [3.0]]]),
        correct_actions=torch.zeros(2, 4, dtype=torch.long),
        trials=4,
        batch_size=2,
    )
    retrieved = torch.ones(2, 1)
    key, value, strength = _support(
        model, batch, device=torch.device("cpu"), retrieved=retrieved,
        support_trials=2)

    assert [marker for marker, _, _ in model.calls] == [0, 1, 2]
    assert [float(feedback[0]) for _, feedback, _ in model.calls] == [0.0, 1.0, 1.0]
    assert model.calls[0][2] is retrieved
    assert model.calls[1][2] is None
    assert model.calls[2][2] is None
    assert torch.equal(key, torch.zeros(2, 1))
    assert torch.equal(value, torch.full((2, 1), 12.0))
    assert torch.equal(strength, torch.full((2,), 22.0))
