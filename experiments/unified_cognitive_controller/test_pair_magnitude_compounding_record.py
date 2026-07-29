"""Regression checks for the curated compounding milestone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECORD = (
    ROOT / "session_records"
    / "pair_magnitude_experience_compounding_2026-07-29")
REPORTS = RECORD / "reports"
CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints"
    / "unified_pair_magnitude_compounding_seed22022.pt")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_compounding_record_preserves_experience_and_gate_accounting() -> None:
    accepted = [
        _load(REPORTS / f"experienced96_e12_s{seed}.json")
        for seed in (22021, 22022, 22023)
    ]
    assert all(report["all_gates_passed"] for report in accepted)
    assert {
        report["accounting"]["new_unique_lifetimes"]
        for report in accepted
    } == {96}
    assert {
        report["accounting"]["new_verifier_bits"]
        for report in accepted
    } == {576}
    assert {
        report["accounting"]["optimizer_updates"]
        for report in accepted
    } == {12}
    assert not _load(REPORTS / "reset_s22023.json")["all_gates_passed"]
    assert not _load(REPORTS / "shuffled_s22023.json")[
        "all_gates_passed"]
    assert _load(REPORTS / "independent_audit_s62022_n32768.json")[
        "all_gates_passed"]


def test_compounding_population_and_checkpoint_are_exact() -> None:
    population = _load(RECORD / "paired_population_summary.json")
    assert population["gate"]["accepted"]
    summary = population["summary"]
    assert summary["parent_mastery_count"] == 0
    assert summary["child_mastery_count"] == 8
    assert summary["every_stream_improved"]
    assert summary["mean_paired_gain"] > 0.0045
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert digest == (
        "5aa030f0fb11d0765752f05cf6c6ecb6334ee31fa1b12a41eeef2603212fe1d4")
