from __future__ import annotations

from pathlib import Path

from experiments.brainworkshop_canonical.controller_pretraining import (
    load_temporal_controller_artifact,
)
from experiments.brainworkshop_canonical.execute_bank_slot import run_slot_execute

REPOSITORY = Path(__file__).resolve().parents[1]


def test_slot_one_executes_rendered_dual_without_rewriting_slot_zero() -> None:
    report = run_slot_execute(
        load_temporal_controller_artifact(
            REPOSITORY
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
        REPOSITORY / "artifacts/checkpoints/AgentBrain.bank",
        slot=1,
        rendered_steps=24,
        seed=113_117,
    )
    assert report["same_digest"] is True
    assert report["slot0_unchanged"] is True
    assert report["program_count"] >= 2
    rendered = report["rendered_dual_1back"]
    assert rendered["optimizer_updates"] == 0
    assert rendered["program_file_updates"] == 0
    assert rendered["unique_verifier_bits"] >= 8
    assert report["neural_workshop_dual_1back"] is None
    assert report["search"] is None


def test_search_selects_a_dual_file_without_a_hardcoded_slot() -> None:
    report = run_slot_execute(
        load_temporal_controller_artifact(
            REPOSITORY
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
        REPOSITORY / "artifacts/checkpoints/AgentBrain.bank",
        slot=0,
        rendered_steps=24,
        seed=113_117,
        search=True,
        search_n_back=1,
    )
    assert report["search"] is not None
    assert report["search"]["winner"] is not None
    assert report["search"]["winner"]["kind"] == "retrieve"
    assert report["search"]["winner"]["accuracy"] >= 0.8
    assert report["slot0_unchanged"] is True
    assert report["rendered_dual_1back"]["accuracy"] >= 0.8
    assert report["rendered_dual_1back"]["program_file_updates"] == 0
