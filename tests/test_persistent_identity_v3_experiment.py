from __future__ import annotations

from pathlib import Path

from experiments.brainworkshop_canonical.persistent_identity_v3 import (
    run_persistent_identity_v3,
)


def test_persistent_identity_v3_closed_loop_smoke(tmp_path: Path) -> None:
    report = run_persistent_identity_v3(tmp_path, steps=8, episodes=3)

    assert report["claim_status"] == "development_closed_loop_diagnostic_not_promoted"
    assert report["frontend_unchanged"]
    assert report["bank_unchanged"]
    assert report["arms"]["persistent_v3"]["confidently_wrong"] == 0
    assert (
        report["persistent_advantage_over_episode_local"] > 0.0
    )
    assert report["stale_persistent_arm"]["quarantine_count"] >= 1
    assert report["controls"]["action_shuffled_abstained"]
    assert report["controls"]["missing_evidence_abstained"]
    assert report["controls"]["exact_equivalence_abstained"]
    assert report["controls"]["partial_mimic_abstained"]
