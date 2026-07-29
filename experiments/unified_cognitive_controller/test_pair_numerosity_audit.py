"""Tests for the frozen numerosity transfer audit."""
from __future__ import annotations

import torch

from .audit_pair_numerosity_transfer import audit, parse_controls
from .model import UnifiedCognitiveController


def test_mass_control_parser_rejects_nonmonotonic_values() -> None:
    assert parse_controls("0,0.5,1") == (0.0, 0.5, 1.0)
    for value in ("", "0.5,0.5", "0.75,0.25", "-0.1,0.5"):
        try:
            parse_controls(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid controls: {value}")


def test_audit_trains_nothing_and_reports_every_control(tmp_path) -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": {
            "width": 32, "workspace_slots": 4, "intention_width": 8},
        "state_dict": model.state_dict(),
    }, checkpoint)
    before = {
        name: value.clone() for name, value in model.state_dict().items()}
    report = audit(
        checkpoint, count=32, seed=23302,
        mass_controls=(0.0, 1.0), device=torch.device("cpu"))
    assert report["schema"] == "pair-numerosity-transfer-audit-v1"
    assert set(report["curve"]) == {"0.0", "1.0"}
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert all(
        torch.equal(before[name], loaded["state_dict"][name])
        for name in before)
