from experiments.brainworkshop_canonical.live_identity_assignment_pixel import (
    run_pixel_identity,
)


def test_rendered_pixels_reach_learned_events_and_abstention_stays_fail_closed(tmp_path) -> None:
    report = run_pixel_identity(tmp_path, steps=4)
    assert report["event_counts"] == [2, 2, 2, 2]
    assert report["assignment_emissions"] == 4
    assert report["no_assignment_emissions"] == 4
    assert report["ambiguous_abstentions"] == 4
    assert report["frontend_unchanged"]
    assert report["bank_unchanged"]
    assert report["unique_verifier_bits"] == 0
