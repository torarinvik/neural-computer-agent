from __future__ import annotations

from .audit_pair_magnitude_bridge_transfer import BLENDS


def test_bridge_transfer_frontier_is_strictly_untrained_and_ordered() -> None:
    assert BLENDS[0] == 0.15625
    assert BLENDS[1] == 0.171875
    assert all(
        left < right for left, right in zip(BLENDS, BLENDS[1:]))
