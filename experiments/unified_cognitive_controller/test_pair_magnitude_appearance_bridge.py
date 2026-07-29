from __future__ import annotations

from .train_pair_magnitude_appearance_bridge import REPLAY_SPECS


def test_magnitude_bridge_replays_every_inherited_capability() -> None:
    assert REPLAY_SPECS == (
        ("visible_pair_magnitude", "bars"),
        ("pair_relation", "bars"),
        ("pair_relation", "diamonds"),
        ("pair_relation", "dot_pairs"),
        ("binary_mapping", "bars"),
        ("visible_context", "bars"),
        ("visible_context_xor", "bars"),
    )
    assert len(set(REPLAY_SPECS)) == len(REPLAY_SPECS)
