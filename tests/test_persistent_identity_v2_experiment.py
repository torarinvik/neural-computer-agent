from experiments.brainworkshop_canonical.persistent_identity_v2 import (
    run_persistent_identity_v2,
)


def test_persistent_identity_v2_exercises_the_real_live_contract(tmp_path) -> None:
    report = run_persistent_identity_v2(tmp_path, steps=4, episodes=2)

    assert set(report["arms"]) == {
        "no_persistent",
        "episode_local",
        "persistent_v2",
        "oracle",
    }
    assert report["controls"]["action_shuffled_abstained"]
    assert report["controls"]["missing_evidence_abstained"]
    assert report["controls"]["exact_equivalence_abstained"]
    assert report["controls"]["crossing_track_order_abstained"]
    assert report["controls"]["birth_death_single_track_assigned"]
    assert report["frontend_unchanged"]
    assert report["bank_unchanged"]
    assert report["unique_verifier_bits"] > 0
    assert report["claim_status"] == "development_closed_loop_diagnostic_not_promoted"
