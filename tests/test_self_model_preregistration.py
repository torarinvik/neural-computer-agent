from __future__ import annotations

from pathlib import Path

from experiments.brainworkshop_canonical.seed_ledger import (
    INTEGRATED_SESSIONS_PER_REPLICATE,
    assert_unused_block,
    block,
)
from experiments.brainworkshop_canonical.slot_alignment import (
    SELF_APPLICABILITY_MARGIN,
    SELF_CONTROLLABILITY_WEIGHT,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BLOCK_NAME = "integrated_self_model_holdout"


def test_integrated_self_model_holdout_is_reserved_and_unused() -> None:
    seeds = block(BLOCK_NAME)
    assert seeds == (9_000_017, 9_500_017, 10_000_017)
    assert_unused_block(
        BLOCK_NAME,
        seeds,
        sessions=INTEGRATED_SESSIONS_PER_REPLICATE,
    )


def test_preregistration_freezes_the_rejected_mechanism_boundary() -> None:
    document = (
        REPOSITORY
        / "docs/PREREGISTRATION_integrated_self_model_holdout_2026-08-16.md"
    ).read_text()
    assert "reserved, not consumed" in document
    assert "No tuning, seed selection, arm removal" in document
    assert f"applicability margin `{SELF_APPLICABILITY_MARGIN}`" in document
    assert "controllability" in document
    assert str(SELF_CONTROLLABILITY_WEIGHT) in document
    assert "exact mimic" in document
    assert "dynamics reversal" in document
