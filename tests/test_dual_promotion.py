from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.dual_promotion import (
    CONTROLLER_SHA256,
    DEVELOPMENT_REPORT_SHA256,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    KNOWN_USED_SEEDS,
    assert_unused_holdout_seeds,
    build_promotion_evidence,
    development_manifest_digest,
    dual_acquisition_gate,
    dual_control_flags,
    extract_replicate_metrics,
    promotion_manifest_digest,
    require_controller,
)
from neural_computer.promotion import HoldoutLedger, evaluate_promotion, sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
DUAL_DIR = REPOSITORY / "session_records" / "brainworkshop_dual_acquisition_2026-08-15"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)


def _load_dual(name: str) -> dict:
    return json.loads((DUAL_DIR / name).read_text(encoding="utf-8"))


def _passing_wrong_depth() -> dict[str, float]:
    return {"packed_exact_accuracy": 0.1, "audio_events": 60, "vision_events": 60}


def test_holdout_seeds_are_disjoint_from_used_populations() -> None:
    assert_unused_holdout_seeds(HOLDOUT_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(KNOWN_USED_SEEDS)
    assert set(DEVELOPMENT_SEEDS).issubset(KNOWN_USED_SEEDS)
    assert {110_017, 111_017, 112_017}.issubset(KNOWN_USED_SEEDS)
    with pytest.raises(ValueError, match="collide"):
        assert_unused_holdout_seeds((113_017, 99_117, 115_017))
    with pytest.raises(ValueError, match="collide"):
        assert_unused_holdout_seeds((113_017, 110_017, 115_017))


def test_controller_and_development_digests_are_frozen() -> None:
    assert require_controller(CONTROLLER) == CONTROLLER_SHA256
    assert sha256_file(DUAL_DIR / "neural_workshop_seed99117.json") == (
        DEVELOPMENT_REPORT_SHA256[99_117]
    )
    assert sha256_file(DUAL_DIR / "neural_workshop_seed99217.json") == (
        DEVELOPMENT_REPORT_SHA256[99_217]
    )


def test_development_and_promotion_manifests_differ() -> None:
    assert development_manifest_digest() != promotion_manifest_digest()
    gate = dual_acquisition_gate()
    assert gate.development_population != gate.promotion_population
    assert gate.min_replicates == 3
    assert gate.capability == "dual-acquisition"
    assert "reward_shuffled" not in gate.required_controls
    assert "wrong_depth" in gate.required_controls


def test_development_seeds_pass_frozen_metrics() -> None:
    gate = dual_acquisition_gate()
    for name in (
        "neural_workshop_seed99117.json",
        "neural_workshop_seed99217.json",
    ):
        report = _load_dual(name)
        metrics = extract_replicate_metrics(report)
        for requirement in gate.metric_requirements:
            assert requirement.matches(metrics[requirement.name]), requirement.name
        assert metrics["warm_target_learning_bits"] == 0.0
        assert metrics["dual_1back_reached"] == 1.0


def test_controls_must_include_wrong_depth_and_combine() -> None:
    report = _load_dual("neural_workshop_seed99117.json")
    with pytest.raises(KeyError):
        dual_control_flags(report)

    passing = dual_control_flags(
        report,
        {**report["controls"], "wrong_depth": _passing_wrong_depth()},
    )
    failing = dual_control_flags(
        report,
        {
            **report["controls"],
            "wrong_depth": {"packed_exact_accuracy": 0.95},
        },
    )
    from experiments.brainworkshop_canonical.founding_promotion import (
        combine_control_flags,
    )

    combined = combine_control_flags((passing, failing))
    assert passing["wrong_depth"]
    assert passing["missing_history"]
    assert passing["action_reversed"]
    assert passing["fresh"]
    assert not failing["wrong_depth"]
    assert not combined["wrong_depth"]


def test_promotion_without_ledger_claim_is_ineligible(tmp_path: Path) -> None:
    report = _load_dual("neural_workshop_seed99217.json")
    metrics = extract_replicate_metrics(report)
    flags = dual_control_flags(
        report,
        {**report["controls"], "wrong_depth": _passing_wrong_depth()},
    )
    gate = dual_acquisition_gate()
    evidence = build_promotion_evidence(
        gate,
        replicate_metrics=(metrics, metrics, metrics),
        controls=flags,
        artifact_hashes={"controller": CONTROLLER_SHA256},
        git_commit="a" * 40,
        controller_sha256=CONTROLLER_SHA256,
    )
    unclaimed = evaluate_promotion(gate, evidence)
    assert not unclaimed.eligible
    assert any("holdout" in reason for reason in unclaimed.reasons)

    ledger = HoldoutLedger(tmp_path / "holdout-ledger.jsonl")
    ledger.claim(
        evidence.holdout_id,
        evidence.promotion_manifest_digest,
        evidence.holdout_attempt_id,
    )
    claimed = evaluate_promotion(gate, evidence, holdout_ledger=ledger)
    assert claimed.eligible
    with pytest.raises(ValueError, match="already been consumed"):
        ledger.claim(
            evidence.holdout_id,
            evidence.promotion_manifest_digest,
            "attempt-2",
        )


def test_consumed_holdout_record_rechecks() -> None:
    record_dir = (
        REPOSITORY / "session_records" / "brainworkshop_dual_holdout_2026-08-15"
    )
    from neural_computer.promotion import read_promotion_record

    ledger = HoldoutLedger(record_dir / "holdout-ledger.jsonl")
    gate, evidence, recorded = read_promotion_record(
        record_dir / "promotion_dual_acquisition.json"
    )
    decision = evaluate_promotion(gate, evidence, holdout_ledger=ledger)
    assert recorded.eligible
    assert decision.eligible
    assert evidence.holdout_uses == 1
    assert len(evidence.replicate_metrics) == 3
    with pytest.raises(ValueError, match="already been consumed"):
        ledger.claim(
            evidence.holdout_id,
            evidence.promotion_manifest_digest,
            "attempt-2",
        )
