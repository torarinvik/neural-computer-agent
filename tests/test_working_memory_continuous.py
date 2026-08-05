from __future__ import annotations

from pathlib import Path

import torch

from experiments.working_memory_continuous.acquire_frozen_growth import (
    _build_successor,
)
from experiments.working_memory_continuous.audit_canonical_artifact_memory import (
    _load_canonical_bank,
)
from neural_computer import freeze_core


def test_canonical_bank_migrates_real_skill_state_without_metadata(tmp_path) -> None:
    source = Path("artifacts/memory/span_multi_skill_bank_seed49011")
    bank, payloads, source_names, keys = _load_canonical_bank(
        source, tmp_path / "bank", torch.device("cpu")
    )

    assert len(payloads) == 2
    assert source_names == ["span9.pt", "span10.pt"]
    assert bank.occupied == (0, 1)
    assert keys.shape == (2, 64)
    handle, artifact = bank.promote(keys[0])
    assert handle.index == 0
    assert all(isinstance(value, torch.Tensor) for value in artifact.values())
    assert "claim_boundary" not in artifact


def test_successor_builder_accepts_a_slot_free_frozen_parent() -> None:
    parent = torch.load(
        "artifacts/checkpoints/span8_addressed_parent_scale1_seed32001.pt",
        map_location="cpu",
        weights_only=False,
    )
    successor, _, slot, prefixes = _build_successor(
        parent, device=torch.device("cpu"), slot_width=16
    )
    freeze_core(successor, prefixes)

    assert slot == 0
    assert any(
        name.startswith(prefixes) and parameter.requires_grad
        for name, parameter in successor.named_parameters()
    )
    assert all(
        parameter.requires_grad == name.startswith(prefixes)
        for name, parameter in successor.named_parameters()
    )
