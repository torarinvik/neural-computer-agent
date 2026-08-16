from __future__ import annotations

from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.integrated_navigation_v3 import (
    run_integrated_navigation_v3,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_persistent_identity_v3_is_composed_with_navigation_loop(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_integrated_navigation_v3(
        tmp_path,
        controller_path=CONTROLLER,
        bank_path=BANK,
        frontend_path=FRONTEND,
        tasks=1,
        steps=8,
        explore_episodes=4,
        starts=2,
    )

    assert report["claim_status"] == "development_composition_diagnostic_not_promoted"
    assert report["matched_arms"] == (
        "random",
        "episode_local",
        "persistent_v3",
        "stale_v3",
        "told_all",
    )
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
    assert "persistent_v3" in report["trained"]
    assert "persistent_v3_confident_wrong_rate" in report["trained"]
    assert (tmp_path / "integrated_navigation.json").is_file()
    assert (tmp_path / "persistent_identity_v3_navigation.json").is_file()
