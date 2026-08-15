from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.founding_promotion import (
    DEVELOPMENT_REPORT_SHA256,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    KNOWN_DEVELOPMENT_SEEDS,
    SOURCE_BANK_SHA256,
    assert_unused_holdout_seeds,
    combine_control_flags,
    development_manifest_digest,
    extract_replicate_metrics,
    first_time_depth_gate,
    founding_control_flags,
    header_transfer_gate,
    promotion_manifest_digest,
    require_source_bank,
)
from neural_computer.promotion import (
    HoldoutLedger,
    evaluate_promotion,
    read_promotion_record,
    sha256_file,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FOUNDING_DIR = (
    REPOSITORY
    / "session_records"
    / "brainworkshop_sealed_frontier_probation_2026-08-14"
)
SOURCE_BANK = (
    REPOSITORY
    / "artifacts/checkpoints/neural_workshop_instruction_route_seed81017.bank"
)


def _load_founding(name: str) -> dict:
    return json.loads((FOUNDING_DIR / name).read_text(encoding="utf-8"))


def _passing_controls() -> dict[str, dict[str, float]]:
    return {
        "wrong_depth": {"accuracy": 0.2},
        "missing_history": {"accuracy": 0.0},
        "action_reversed": {"accuracy": 0.3},
    }


def test_holdout_seeds_are_disjoint_from_development() -> None:
    assert_unused_holdout_seeds(HOLDOUT_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(KNOWN_DEVELOPMENT_SEEDS)
    assert set(DEVELOPMENT_SEEDS).issubset(KNOWN_DEVELOPMENT_SEEDS)
    with pytest.raises(ValueError, match="collide"):
        assert_unused_holdout_seeds((110_017, 94_017, 112_017))


def test_source_bank_digest_is_frozen() -> None:
    assert require_source_bank(SOURCE_BANK) == SOURCE_BANK_SHA256
    assert sha256_file(FOUNDING_DIR / "founding_report.json") == (
        DEVELOPMENT_REPORT_SHA256[94_017]
    )
    assert sha256_file(FOUNDING_DIR / "founding_report_seed97017.json") == (
        DEVELOPMENT_REPORT_SHA256[97_017]
    )


def test_development_and_promotion_manifests_differ() -> None:
    assert development_manifest_digest() != promotion_manifest_digest()
    header = header_transfer_gate()
    depth = first_time_depth_gate()
    assert header.development_population != header.promotion_population
    assert header.min_replicates == 3
    assert depth.min_replicates == 3
    assert header.capability != depth.capability
    assert header.digest() != depth.digest()
    assert "reward_shuffled" not in header.required_controls
    assert "reward_shuffled" not in depth.required_controls


def test_seed97017_would_pass_both_frozen_gates() -> None:
    report = _load_founding("founding_report_seed97017.json")
    metrics = extract_replicate_metrics(report)
    flags = founding_control_flags(report, _passing_controls())
    header = header_transfer_gate()
    depth = first_time_depth_gate()

    assert metrics["header_fresh_over_warm"] == pytest.approx(3.4642857142857144)
    assert metrics["depth_fresh_over_warm"] == pytest.approx(1.9538461538461538)
    assert metrics["header_retrieved"] == 1.0
    assert metrics["depth_composed"] == 1.0
    assert flags["fresh"]
    assert flags["controller_frozen"]
    for requirement in header.metric_requirements:
        assert requirement.matches(metrics[requirement.name])
    for requirement in depth.metric_requirements:
        assert requirement.matches(metrics[requirement.name])


def test_seed94017_passes_header_transfer_and_fails_first_time_depth() -> None:
    report = _load_founding("founding_report.json")
    metrics = extract_replicate_metrics(report)
    header = header_transfer_gate()
    depth = first_time_depth_gate()

    assert metrics["header_fresh_over_warm"] == pytest.approx(4.25)
    assert metrics["depth_fresh_over_warm"] == pytest.approx(1.017094017094017)
    assert metrics["header_retrieved"] == 1.0
    assert metrics["depth_composed"] == 1.0
    for requirement in header.metric_requirements:
        assert requirement.matches(metrics[requirement.name])
    failed = [
        requirement.name
        for requirement in depth.metric_requirements
        if not requirement.matches(metrics[requirement.name])
    ]
    assert failed == ["depth_fresh_over_warm"]


def test_controls_must_fail_closed_and_combine_across_seeds() -> None:
    report = _load_founding("founding_report_seed97017.json")
    passing = founding_control_flags(report, _passing_controls())
    failing = founding_control_flags(
        report,
        {
            "wrong_depth": {"accuracy": 0.95},
            "missing_history": {"accuracy": 0.0},
            "action_reversed": {"accuracy": 0.3},
        },
    )
    combined = combine_control_flags((passing, failing))

    assert passing["wrong_depth"]
    assert not failing["wrong_depth"]
    assert combined["fresh"]
    assert not combined["wrong_depth"]


def test_promotion_without_ledger_claim_is_ineligible(tmp_path: Path) -> None:
    report = _load_founding("founding_report_seed97017.json")
    metrics = extract_replicate_metrics(report)
    gate = header_transfer_gate()
    from experiments.brainworkshop_canonical.founding_promotion import (
        build_promotion_evidence,
    )

    evidence = build_promotion_evidence(
        gate,
        replicate_metrics=(metrics, metrics, metrics),
        controls=founding_control_flags(report, _passing_controls()),
        artifact_hashes={"source_bank": SOURCE_BANK_SHA256},
        git_commit="a" * 40,
        source_bank_sha256=SOURCE_BANK_SHA256,
        controller_sha256="b" * 64,
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


def test_consumed_holdout_records_recheck() -> None:
    record_dir = (
        REPOSITORY / "session_records" / "brainworkshop_founding_holdout_2026-08-15"
    )
    ledger = HoldoutLedger(record_dir / "holdout-ledger.jsonl")
    for name in (
        "promotion_header_transfer.json",
        "promotion_first_time_depth.json",
    ):
        gate, evidence, recorded = read_promotion_record(record_dir / name)
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
