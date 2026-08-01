import torch

from .memory_intention_bridge import (
    MemoryActionComposer,
    MemoryCodeBridge,
    MemoryIntentionReader,
)


def test_intention_reader_is_an_exact_zero_initialized_noop() -> None:
    reader = MemoryIntentionReader(memory_width=12, intention_width=5)
    intention = torch.randn(7, 5)
    memory = torch.randn(7, 12)
    assert torch.equal(reader(intention, memory), torch.zeros_like(intention))


def test_memory_bridges_have_expected_shapes_and_gradients() -> None:
    memory = torch.randn(8, 12)
    intention = torch.randn(8, 5)
    code = MemoryCodeBridge(memory_width=12)(memory)
    logits = MemoryActionComposer(intention_width=5)(intention, code)
    residual = MemoryIntentionReader(12, 5)(intention, memory)
    assert code.shape == (8, 2)
    assert logits.shape == (8, 2)
    assert residual.shape == (8, 5)
    (logits.square().mean() + residual.square().mean()).backward()
