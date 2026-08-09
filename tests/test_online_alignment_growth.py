import pytest
import torch

from experiments.outcome_only_online_alignment_growth.train import (
    ExternalAlignmentKeyBank,
)


def test_external_alignment_key_bank_appends_and_selects() -> None:
    bank = ExternalAlignmentKeyBank(context_width=3)
    bank.add("cell_0", torch.tensor([0.0, 0.0, 0.0]))
    bank.add("cell_1", torch.tensor([1.0, 1.0, 1.0]))

    assert bank.select(torch.tensor([[0.1, 0.0, 0.0]])).item() == 0
    assert bank.select(torch.tensor([[0.9, 1.0, 1.0]])).item() == 1
    assert bank.configuration()["schema"] == (
        "neural-computer.external-alignment-key-bank.v1"
    )
    with pytest.raises(ValueError, match="unique"):
        bank.add("cell_0", torch.zeros(3))
    with pytest.raises(ValueError, match="wrong shape"):
        bank.add("cell_2", torch.zeros(2))
    removed = bank.remove("cell_0")
    torch.testing.assert_close(removed, torch.zeros(3))
    assert bank.configuration()["logical_ids"] == ("cell_1",)


def test_external_alignment_key_bank_rejects_empty_selection() -> None:
    bank = ExternalAlignmentKeyBank(context_width=2)
    with pytest.raises(ValueError, match="empty"):
        bank.select(torch.zeros(1, 2))
