"""Small contracts for the repertoire audit entry point."""
from __future__ import annotations

import torch

from .audit_pair_relation_repertoire import _load
from .model import UnifiedCognitiveController


def test_pair_audit_loads_unified_checkpoint(tmp_path) -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    path = tmp_path / "controller.pt"
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": {
            "width": 32, "workspace_slots": 4, "intention_width": 8},
        "state_dict": model.state_dict(),
    }, path)
    restored = _load(path, torch.device("cpu"))
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(),
                               restored.state_dict().values()))
