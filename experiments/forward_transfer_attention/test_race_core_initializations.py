from .race_core_initializations import (
    CORE_SEEDS,
    select_survivors,
)


def _records():
    return {
        "a": {"causal_floor": 0.9, "progress": 0.1,
              "mechanistic_score": 0.1},
        "b": {"causal_floor": 0.8, "progress": 0.2,
              "mechanistic_score": 0.2},
        "c": {"causal_floor": 0.7, "progress": 0.3,
              "mechanistic_score": 0.3},
        "d": {"causal_floor": 0.6, "progress": 0.4,
              "mechanistic_score": 0.9},
        "e": {"causal_floor": 0.5, "progress": 0.5,
              "mechanistic_score": 0.4},
    }


def test_population_contains_six_distinct_initializations() -> None:
    assert len(CORE_SEEDS) == 6
    assert len(set(CORE_SEEDS)) == 6


def test_32_bit_halving_reserves_mechanistic_late_slot() -> None:
    survivors = select_survivors(
        list(_records()), _records(), count=4, stage=32)
    assert survivors[:3] == ["a", "b", "c"]
    assert survivors[-1] == "d"


def test_48_bit_halving_keeps_leader_progress_and_mechanistic() -> None:
    survivors = select_survivors(
        list(_records()), _records(), count=3, stage=48)
    assert survivors == ["a", "e", "d"]
