from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from neural_computer.promotion import (
    HoldoutLedger,
    MetricRequirement,
    PromotionEvidence,
    PromotionGate,
    PromotionRejected,
    evaluate_promotion,
    read_promotion_record,
    require_promotion,
    sha256_files,
    write_promotion_record,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _gate() -> PromotionGate:
    return PromotionGate(
        experiment_id="demo-v1",
        capability="opaque-transfer",
        development_population="dev-2026-08",
        promotion_population="sealed-2026-08",
        metric_requirements=(MetricRequirement("stable_bits", maximum=1000),),
        required_controls=("fresh", "reward_shuffled", "reversal"),
        min_replicates=3,
    )


def _evidence(gate: PromotionGate, **changes: object) -> PromotionEvidence:
    values: dict[str, object] = {
        "gate_digest": gate.digest(),
        "holdout_id": "sealed-lease-1",
        "holdout_attempt_id": "attempt-1",
        "development_manifest_digest": _digest("development"),
        "promotion_manifest_digest": _digest("promotion"),
        "git_commit": "a" * 40,
        "configuration_digest": _digest("configuration"),
        "artifact_hashes": {"candidate": _digest("candidate")},
        "replicate_metrics": (
            {"stable_bits": 800},
            {"stable_bits": 900},
            {"stable_bits": 950},
        ),
        "controls": {"fresh": True, "reward_shuffled": True, "reversal": True},
        "search_attempts": 4,
        "workaround_count": 0,
    }
    values.update(changes)
    return PromotionEvidence(**values)  # type: ignore[arg-type]


def test_promotion_requires_every_replication_and_control() -> None:
    gate = _gate()
    evidence = _evidence(
        gate,
        replicate_metrics=({"stable_bits": 800}, {"stable_bits": 900}),
        controls={"fresh": True, "reward_shuffled": False, "reversal": True},
    )

    decision = evaluate_promotion(gate, evidence)

    assert not decision.eligible
    assert any("replications" in reason for reason in decision.reasons)
    assert "required control failed: reward_shuffled" in decision.reasons


def test_promotion_passes_only_when_gate_digest_and_all_metrics_match(tmp_path) -> None:
    gate = _gate()
    evidence = _evidence(gate)
    ledger = HoldoutLedger(tmp_path / "promotion.jsonl")
    ledger.claim(
        evidence.holdout_id,
        evidence.promotion_manifest_digest,
        evidence.holdout_attempt_id,
    )

    assert require_promotion(gate, evidence, holdout_ledger=ledger).eligible

    altered_gate = replace(gate, capability="different-capability")
    decision = evaluate_promotion(altered_gate, evidence, holdout_ledger=ledger)
    assert not decision.eligible
    assert "evidence was produced against a different gate" in decision.reasons


def test_require_promotion_fails_closed_on_workarounds(tmp_path) -> None:
    gate = _gate()
    evidence = _evidence(gate, workaround_count=1)
    ledger = HoldoutLedger(tmp_path / "promotion.jsonl")
    ledger.claim(
        evidence.holdout_id,
        evidence.promotion_manifest_digest,
        evidence.holdout_attempt_id,
    )

    with pytest.raises(PromotionRejected, match="workaround count"):
        require_promotion(gate, evidence, holdout_ledger=ledger)


def test_promotion_without_a_ledger_claim_is_ineligible() -> None:
    gate = _gate()
    evidence = _evidence(gate)

    decision = evaluate_promotion(gate, evidence)

    assert not decision.eligible
    assert "promotion holdout claim was not verified against a ledger" in decision.reasons


def test_holdout_ledger_rejects_reuse(tmp_path) -> None:
    ledger = HoldoutLedger(tmp_path / "holdouts.jsonl")
    manifest = _digest("promotion")
    ledger.claim("holdout-1", manifest, "attempt-1")
    assert ledger.verify_claim("holdout-1", manifest, "attempt-1")
    assert not ledger.verify_claim("holdout-1", manifest, "attempt-2")

    with pytest.raises(ValueError, match="already been consumed"):
        ledger.claim("holdout-1", manifest, "attempt-2")


def test_artifact_hashes_are_stable_and_labelled(tmp_path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    hashes = sha256_files({"z": second, "a": first})

    assert list(hashes) == ["a", "z"]
    assert hashes["a"] == hashlib.sha256(b"first").hexdigest()


def test_serialized_record_round_trips_and_rechecks(tmp_path) -> None:
    gate = _gate()
    evidence = _evidence(gate)
    ledger = HoldoutLedger(tmp_path / "holdouts.jsonl")
    ledger.claim(
        evidence.holdout_id,
        evidence.promotion_manifest_digest,
        evidence.holdout_attempt_id,
    )
    record_path = tmp_path / "promotion.json"
    decision = require_promotion(gate, evidence, holdout_ledger=ledger)
    write_promotion_record(
        record_path,
        gate,
        evidence,
        decision,
        holdout_ledger=ledger,
    )

    loaded_gate, loaded_evidence, recorded = read_promotion_record(record_path)

    assert loaded_gate.digest() == gate.digest()
    assert loaded_evidence == evidence
    assert recorded == decision
    assert evaluate_promotion(
        loaded_gate,
        loaded_evidence,
        holdout_ledger=ledger,
    ).eligible
