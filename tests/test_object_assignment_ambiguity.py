from experiments.brainworkshop_canonical.object_assignment_ambiguity import (
    STABLE_THRESHOLD,
    evaluate_episode,
    run_assignment,
    sample_episode,
)


def test_collision_keeps_two_separately_bound_events() -> None:
    episode = sample_episode(41, collision=True, response_probability=0.35)
    assert all(len(frame.detections) == 2 for frame in episode.frames)
    assert all({detection.appearance for detection in frame.detections} == {0} for frame in episode.frames)
    assert all(frame.detections[0].position != frame.detections[1].position for frame in episode.frames)


def test_action_conditioned_beam_beats_continuity_and_symbol_controls() -> None:
    episode = sample_episode(41, collision=True, response_probability=0.35)
    causal = evaluate_episode(episode, mode="causal_beam")
    nearest = evaluate_episode(episode, mode="nearest")
    appearance = evaluate_episode(episode, mode="appearance")
    assert causal["accuracy"] >= STABLE_THRESHOLD
    assert causal["accuracy"] > nearest["accuracy"]
    assert causal["accuracy"] > appearance["accuracy"]


def test_assignment_report_records_stable_prefix_and_controls(tmp_path) -> None:
    report = run_assignment(tmp_path, replicates=1)
    conditions = report["replicates"][0]["conditions"]
    assert conditions["approximate_collision"]["causal_beam"]["stable_bits_to_threshold"] == 144
    assert conditions["approximate_collision"]["nearest"]["stable_bits_to_threshold"] is None
    assert report["accounting"]["approximate_collision_causal_beam"]["optimizer_updates"] == 0
