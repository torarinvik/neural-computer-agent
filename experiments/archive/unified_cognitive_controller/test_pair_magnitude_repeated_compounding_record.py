"""Regression checks for the repeated-compounding magnitude milestone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECORD = (
    ROOT / "session_records"
    / "pair_magnitude_repeated_compounding_2026-07-29")
CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints"
    / "unified_pair_magnitude_repeated_compounding_seed23105.pt")
EXPECTED_SHA256 = (
    "c136841d60a5220bd09cd12029b6d59d903dc73d5deddb39248e7327ae48f2a2")


def _load(name: str) -> dict:
    return json.loads((RECORD / "reports" / name).read_text())


def test_first_stable_experience_threshold_is_44_lifetimes() -> None:
    for name in ("train32e12_s23101.json", "train40e12_s23102.json"):
        assert not _load(name)["all_gates_passed"]

    forty_two = [
        _load(f"train42e12_s{seed}.json")
        for seed in (23106, 23107, 23108)
    ]
    assert sum(report["all_gates_passed"] for report in forty_two) == 1

    forty_four = [
        _load(f"train44e12_s{seed}.json")
        for seed in (23103, 23104, 23105)
    ]
    assert all(report["all_gates_passed"] for report in forty_four)
    assert all(
        report["accounting"]["new_unique_lifetimes"] == 44
        for report in forty_four)
    assert all(
        report["accounting"]["new_verifier_bits"] == 264
        for report in forty_four)


def test_controls_population_and_checkpoint_are_causal() -> None:
    assert not _load("reset44e12_s23105.json")["all_gates_passed"]
    assert not _load("shuffled44e12_s23105.json")["all_gates_passed"]

    audit = _load("independent44_audit_s63105_n32768.json")
    assert audit["all_gates_passed"]
    assert audit["target_blend"]["overall_accuracy"] >= 0.90

    population = json.loads(
        (RECORD / "paired_population_summary.json").read_text())
    assert population["parent_mastery_streams"] == 2
    assert population["child_mastery_streams"] == 8
    assert population["all_streams_improved"]
    assert population["minimum_gain"] > 0

    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_SHA256
