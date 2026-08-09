import pytest
import torch

from experiments.outcome_only_alignment_cell_stream.train import (
    ExternalAlignmentCellBank,
    OpaqueAlignmentRouter,
)
from neural_computer import AmodalEventBridge


def test_external_alignment_cell_bank_grows_and_freezes_cells() -> None:
    bank = ExternalAlignmentCellBank()
    bridge = AmodalEventBridge(4, 6, 4, hidden=8)
    bank.add("cell_0", bridge)
    bank.freeze("cell_0")

    assert bank.configuration()["schema"] == (
        "neural-computer.external-alignment-cell-bank.v1"
    )
    assert all(not parameter.requires_grad for parameter in bridge.parameters())
    with pytest.raises(ValueError, match="already exists"):
        bank.add("cell_0", AmodalEventBridge(4, 6, 4, hidden=8))
    with pytest.raises(ValueError, match="dot-free"):
        bank.add("cell.1", AmodalEventBridge(4, 6, 4, hidden=8))
    with pytest.raises(KeyError, match="unknown alignment cell"):
        bank.cell("missing")
    bank.remove("cell_0")
    assert bank.configuration()["logical_ids"] == ()


def test_alignment_cell_bank_state_is_independent() -> None:
    bank = ExternalAlignmentCellBank()
    first = AmodalEventBridge(4, 6, 4, hidden=8)
    second = AmodalEventBridge(4, 6, 4, hidden=8)
    bank.add("cell_0", first)
    bank.add("cell_1", second)
    before = second.residual[-1].bias.detach().clone()
    with torch.no_grad():
        first.residual[-1].bias.fill_(3.0)
    torch.testing.assert_close(second.residual[-1].bias, before)


def test_opaque_alignment_router_has_variable_cell_output() -> None:
    router = OpaqueAlignmentRouter(context_width=8, cell_count=3, hidden=8)
    logits = router(torch.zeros(5, 8))
    assert logits.shape == (5, 3)
