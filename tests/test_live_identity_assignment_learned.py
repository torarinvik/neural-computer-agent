from experiments.brainworkshop_canonical.live_identity_assignment_learned import (
    run_learned_identity,
)


def test_rendered_event_histories_drive_external_identity_without_learning(tmp_path) -> None:
    report = run_learned_identity(tmp_path, steps=6)

    assert report["learned_emissions"] > 0
    assert report["learned_emissions"] < report["steps"]
    assert report["passive_abstentions"] == report["steps"]
    assert report["constant_action_abstentions"] == report["steps"]
    assert report["selected_slots"][-1] == 0
    assert report["constant_action_evidence"][-1] == [0.0, 0.0]
    assert report["frontend_unchanged"]
    assert report["bank_unchanged"]
    assert report["unique_verifier_bits"] == 0
