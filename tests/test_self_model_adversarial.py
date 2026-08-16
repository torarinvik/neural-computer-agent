from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.self_model_adversarial import (
    CONDITIONS,
    MIMIC_VARIANTS,
    run_self_model_adversarial,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
RECORD = (
    REPOSITORY
    / "session_records"
    / "brainworkshop_self_model_adversarial_2026-08-16"
)


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_adversarial_audit_is_accounted_and_fails_closed(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_self_model_adversarial(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        tasks=1,
        explore_episodes=4,
        steps=8,
    )

    assert set(report["conditions"]) == set(CONDITIONS)
    assert set(report["ablations"]) == {
        "guarded_honest",
        "guarded_poisoned",
        "likelihood_only_honest",
        "likelihood_only_poisoned",
    }
    assert set(report["baseline_arms"]) == {
        "episodic_identity",
        "remembered_likelihood_only",
        "remembered_likelihood_gated",
        "remembered_likelihood_controllable",
    }
    assert set(report["near_mimics"]) == set(MIMIC_VARIANTS)
    assert report["mechanism"] == {
        "applicability_margin": 0.25,
        "controllability_weight": 2.0,
        "frozen_for_holdout": True,
    }
    assert report["poisoned_recovery"][0]["confidently_wrong"] == 1.0
    assert report["poisoned_recovery"][-1]["confidently_wrong"] <= 1.0
    reversal = report["rows"][0]["reversal_stream"]
    assert reversal["detected"]
    assert reversal["recovered"]
    assert reversal["detection_events"]
    assert reversal["recovery_events"]
    # Exact observational equivalence has no evidence-supported identity.
    assert report["conditions"]["mimicked"]["abstained"] == 1.0
    assert report["conditions"]["mimicked"]["confidently_wrong"] == 0.0

    accounting = report["accounting"]
    assert accounting["unique_verifier_bits"] == 64
    assert accounting["unique_logical_lifetimes"] == 8
    assert accounting["optimizer_updates"] == 0
    assert accounting["replayed_examples"] > 0
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before


def test_rejected_record_has_matching_accounting_and_checksums() -> None:
    report = json.loads((RECORD / "self_model_adversarial.json").read_text())
    ledger = json.loads((RECORD / "sample_efficiency_ledger.json").read_text())

    assert ledger["status"].startswith("rejected_")
    assert ledger["unique_verifier_bits"]["total"] == report["accounting"][
        "unique_verifier_bits"
    ]
    assert ledger["unique_logical_lifetimes_all_arms"] == report["accounting"][
        "unique_logical_lifetimes"
    ]
    assert ledger["replayed_examples"] == report["accounting"][
        "replayed_examples"
    ]

    for line in (RECORD / "checksums.sha256").read_text().splitlines():
        digest, name = line.split()
        assert sha256_file(RECORD / name) == digest
