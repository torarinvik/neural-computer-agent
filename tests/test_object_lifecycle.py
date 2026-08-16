from experiments.brainworkshop_canonical.object_lifecycle import (
    STABLE_THRESHOLD,
    evaluate_episode,
    run_lifecycle,
    sample_episode,
)


def test_lifecycle_stream_has_variable_object_count() -> None:
    episode = sample_episode(41, lifecycle=True, occlusion_rate=0.0)
    counts = {len(frame.detections) for frame in episode.frames}
    assert len(counts) > 1
    assert min(counts) < 5
    assert max(counts) <= 5


def test_persistent_tracker_survives_occlusion_better_than_zero_control() -> None:
    episode = sample_episode(41, lifecycle=True, occlusion_rate=0.25)
    persistent = evaluate_episode(episode, mode="persistent")
    zero_missing = evaluate_episode(episode, mode="zero_missing")
    reinitializing = evaluate_episode(episode, mode="reinitializing")
    assert persistent["accuracy"] >= STABLE_THRESHOLD
    assert persistent["accuracy"] > zero_missing["accuracy"]
    assert persistent["occlusion_recoveries"] > zero_missing["occlusion_recoveries"]
    assert reinitializing["coverage"] == 0.0


def test_lifecycle_report_records_stable_prefix_and_accounting(tmp_path) -> None:
    report = run_lifecycle(tmp_path, replicates=1)
    arms = report["replicates"][0]["conditions"]
    assert arms["stable"]["persistent"]["stable_bits_to_threshold"] == 248
    assert arms["lifecycle"]["persistent"]["stable_bits_to_threshold"] == 395
    assert arms["occluded"]["persistent"]["stable_bits_to_threshold"] == 408
    assert arms["occluded"]["zero_missing"]["stable_bits_to_threshold"] is None
    assert report["accounting"]["occluded_persistent"]["optimizer_updates"] == 0
