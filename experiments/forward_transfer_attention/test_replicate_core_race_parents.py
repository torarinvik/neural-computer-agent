from .replicate_core_race_parents import (
    PARENT_SEEDS,
    PREFIXES,
    REPLICATION_AUDIT_START,
    REPLICATION_TRAIN_START,
)
from .train_feature_interface_tournament import BLIND_START


def test_replication_compares_both_scientific_parents() -> None:
    assert PARENT_SEEDS == (211, 263)


def test_replication_uses_gradual_prefixes() -> None:
    assert PREFIXES == (32, 48, 64)


def test_replication_streams_are_disjoint_from_selection() -> None:
    assert REPLICATION_TRAIN_START > BLIND_START
    assert REPLICATION_AUDIT_START > REPLICATION_TRAIN_START + max(PREFIXES)
