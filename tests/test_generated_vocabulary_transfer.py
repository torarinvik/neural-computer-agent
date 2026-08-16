from __future__ import annotations

from experiments.brainworkshop_canonical.generated_vocabulary_transfer import (
    ARTIFACT_SCHEMA,
    STREAM_STEPS,
    Predicate,
    admit_verified,
    discover,
    generate_candidates,
    run_transfer,
    sample_stream,
    score_candidate,
)


def test_candidate_generation_uses_bound_channels_and_temporal_operators() -> None:
    stream = sample_stream(41, rule="same_delta")
    candidates = generate_candidates(stream)
    operators = {candidate.op for candidate in candidates}
    assert {"equal", "persist", "change", "same_change", "same_delta"} <= operators
    assert any(candidate.op == "and" for candidate in candidates)
    assert any(candidate.op == "not" for candidate in candidates)
    assert all(candidate.digest for candidate in candidates)


def test_mdl_discovery_selects_the_hidden_temporal_relation_without_its_name() -> None:
    streams = tuple(
        sample_stream(100 + index, rule="same_delta", offset=index)
        for index in range(4)
    )
    predicate, summary = discover(streams)
    assert predicate.op == "same_delta"
    assert summary["training_errors"] == 0
    assert summary["candidate_count"] > 20


def test_quarantine_requires_fresh_verification() -> None:
    training = tuple(sample_stream(200 + index, rule="same_delta") for index in range(2))
    verification = (sample_stream(300, rule="same_delta"),)
    candidate = Predicate("same_delta", (0, 1)).validate()
    artifact = admit_verified(candidate, training, verification)
    assert artifact is not None
    assert artifact.schema == ARTIFACT_SCHEMA

    wrong = Predicate("same_change", (0, 1)).validate()
    assert admit_verified(wrong, training, verification) is None


def test_predicate_composition_is_not_allowed_to_skip_fresh_data() -> None:
    stream = sample_stream(400, rule="same_delta")
    candidate = Predicate(
        "and",
        children=(Predicate("same_delta", (0, 1)), Predicate("persist", (0,))),
    ).validate()
    score = score_candidate(candidate, stream)
    assert score["errors"] > 0
    assert score["total_bits"] > score["error_bits"]


def test_generated_vocabulary_transfer_beats_fresh_and_rejects_controls(tmp_path) -> None:
    report = run_transfer(tmp_path, replicates=1)
    row = report["replicates"][0]["arms"]
    assert report["generated_candidate_count"] > 20
    assert row["retained"]["stable_bits_to_threshold"] == STREAM_STEPS
    assert row["fresh"]["stable_bits_to_threshold"] > STREAM_STEPS
    assert row["irrelevant"]["stable_bits_to_threshold"] is None
    assert row["corrupted"]["stable_bits_to_threshold"] is None

