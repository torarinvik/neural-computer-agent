from __future__ import annotations

from pathlib import Path

from experiments.brainworkshop_canonical.learned_proposer import (
    run_learned_proposer_audit,
)


def test_learned_proposer_reduces_candidate_work_and_falls_back(tmp_path: Path) -> None:
    report = run_learned_proposer_audit(tmp_path, seed=41)
    assert report["claim_status"] == (
        "development_proposal_throughput_diagnostic_not_promoted"
    )
    assert report["learned_proposer"]["winner"]["combiner"] == "and"
    assert report["learned_proposer"]["hypotheses"] < report["exhaustive"]["hypotheses"]
    assert report["hypothesis_reduction"] > 0.8
    assert report["stranger_control"]["found"] is None
    assert report["stranger_control"]["fallback_used"]
    assert report["agent_bank_unchanged"]
