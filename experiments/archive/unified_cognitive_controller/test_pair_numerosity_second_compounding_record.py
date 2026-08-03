"""Regression checks for the second numerosity compounding milestone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECORD = (
    ROOT / "session_records"
    / "pair_numerosity_second_compounding_2026-07-29")
CHECKPOINT = (
    ROOT / "artifacts" / "checkpoints"
    / "unified_pair_numerosity_second_compounding_seed24031.pt")
EXPECTED_SHA256 = (
    "1dcd9149089feaa1c54b3b5a24131716fe39c551cef6fd8ef3b9f67db94b0dc9")


def _load(name: str) -> dict:
    return json.loads((RECORD / "reports" / name).read_text())


def test_target_was_first_clean_failed_frontier() -> None:
    boundary = _load("frontier_boundary_n8192.json")
    assert boundary["selected_first_zero_of_three_blend"] == 0.248
    assert sum(boundary["blends"]["0.246"]["accepted_by_seed"]) == 1
    assert not any(boundary["blends"]["0.248"]["accepted_by_seed"])


def test_eight_lifetimes_beat_four_and_require_real_outcomes() -> None:
    ladder = _load("small_experiment_ladder.json")
    four_lifetime_real = [
        row for row in ladder["experiments"]
        if row["new_lifetimes"] == 4]
    assert sum(row["passed"] for row in four_lifetime_real) == 1

    real = [
        _load(f"real_n8e16_s{seed}.json")
        for seed in (24031, 24032)
    ]
    shuffled = [
        _load(f"shuffled_n8e16_s{seed}.json")
        for seed in (24031, 24032)
    ]
    assert all(
        report["accounting"]["new_unique_lifetimes"] == 8
        for report in real)
    assert all(
        report["accounting"]["new_verifier_bits"] == 48
        for report in real)
    assert not any(report["all_gates_passed"] for report in shuffled)

    independent = _load(
        "independent_target_audit_n32768_s100024800.json")
    assert not independent["parent"]["accepted"]
    assert independent["child_seed_24031"]["accepted"]
    assert independent["child_seed_24032"]["accepted"]


def test_selected_child_passes_full_causal_and_retention_audit() -> None:
    audit = _load("continuation_0248_audit_n8192_s24101.json")
    assert audit["all_gates_passed"]
    assert all(audit["gates"].values())
    assert audit["headline"]["target"] >= 0.90
    assert audit["headline"]["frozen_parent_target"] < 0.90
    assert (
        audit["headline"]["missing_second_object"]
        <= audit["headline"]["target"] - 0.15)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_SHA256
