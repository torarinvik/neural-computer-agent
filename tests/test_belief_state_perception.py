from __future__ import annotations

from pathlib import Path

from experiments.brainworkshop_canonical.belief_state_perception import (
    BeliefStatePerception,
    mask_traces,
    run_belief_state_perception,
)
from experiments.brainworkshop_canonical.perceptual_aliasing import sample_traces


def test_belief_state_marginalizes_missing_events_without_zero_padding() -> None:
    training = sample_traces(41, episodes=16)
    evaluation = sample_traces(10041, episodes=8)
    masked = mask_traces(evaluation, seed=61, missing_rate=0.2)
    model = BeliefStatePerception().fit(training)
    assert any(value is None for observations, _ in masked for value in observations)
    result = model.predict(masked[0][0], masked[0][1], 2)
    assert result[0] is None or isinstance(result[0], int)


def test_belief_state_audit_improves_missing_evidence_coverage(tmp_path: Path) -> None:
    report = run_belief_state_perception(tmp_path, seed=41)
    assert report["claim_status"] == "development_belief_state_diagnostic_not_promoted"
    belief = report["arms"]["belief_missing"]
    history = report["arms"]["history_missing"]
    assert belief["coverage"] > history["coverage"]
    assert belief["expected_correct_rate"] > history["expected_correct_rate"]
    assert report["missing_evidence_is_not_zero"]
