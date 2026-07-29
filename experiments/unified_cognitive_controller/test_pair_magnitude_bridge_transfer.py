from __future__ import annotations

import pytest

from .audit_pair_magnitude_bridge_transfer import _parse_blends


def test_transfer_curve_accepts_precise_increasing_frontier() -> None:
    assert _parse_blends(
        "0.203125,0.205078125,0.20703125"
    ) == (0.203125, 0.205078125, 0.20703125)


@pytest.mark.parametrize(
    "value",
    (
        "0.2",
        "0.2,0.2",
        "0.3,0.2",
        "-0.1,0.2",
        "0.2,1.1",
    ),
)
def test_transfer_curve_rejects_invalid_frontiers(value: str) -> None:
    with pytest.raises(ValueError, match="blends"):
        _parse_blends(value)
