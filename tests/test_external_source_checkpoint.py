import pytest
import torch

from experiments.external_register_composition_amodal.audit_interleaved_basis_acquisition import (
    _load_source_checkpoint,
    _positive_transfer,
    _save_source_checkpoint,
)
from experiments.external_register_composition_amodal.train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    _new_machine,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import OpaqueProtocolDecoder


def _source_bank():
    machine = _new_machine(
        3,
        operator_mode="factorized_protected_bounded_meta",
    )
    for _ in range(3):
        machine.add_basis_slot()
    decoders = [
        OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        for _ in range(3)
    ]
    return machine, decoders


def test_source_checkpoint_round_trips_verified_bank(tmp_path) -> None:
    parent = _runtime(seed=101, growth=False)
    machine, decoders = _source_bank()
    path = tmp_path / "source-bank.pt"
    scores = [0.8, 0.9375, 1.0]

    _save_source_checkpoint(
        path,
        parent=parent,
        machine=machine,
        source_decoders=decoders,
        source_operations=("reverse", "adjacent_xor", "complement"),
        source_scores=scores,
    )
    restored, restored_decoders = _source_bank()
    restored_parent = _runtime(seed=202, growth=False)
    loaded = _load_source_checkpoint(
        path,
        parent=restored_parent,
        machine=restored,
        source_operations=("reverse", "adjacent_xor", "complement"),
    )

    for name, value in machine.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    for name, value in parent.state_dict().items():
        assert torch.equal(value, restored_parent.state_dict()[name])
    for expected, actual in zip(decoders, loaded, strict=True):
        for name, value in expected.state_dict().items():
            assert torch.equal(value, actual.state_dict()[name])
    assert len(restored_decoders) == 3


def test_source_checkpoint_refuses_unmastered_bank(tmp_path) -> None:
    parent = _runtime(seed=303, growth=False)
    machine, decoders = _source_bank()

    with pytest.raises(ValueError, match="unmastered"):
        _save_source_checkpoint(
            tmp_path / "rejected.pt",
            parent=parent,
            machine=machine,
            source_decoders=decoders,
            source_operations=("reverse", "adjacent_xor", "complement"),
            source_scores=[0.8, 0.79, 1.0],
        )


def test_positive_transfer_requires_fewer_inherited_bits() -> None:
    assert _positive_transfer(8_192, 12_288)
    assert not _positive_transfer(12_288, 8_192)
    assert not _positive_transfer(None, 8_192)
