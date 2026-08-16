from __future__ import annotations

from experiments.brainworkshop_canonical.dynamics_reversal import (
    reversed_world,
    run_reversal,
)
from experiments.brainworkshop_canonical.operator_world_transfer import (
    ACTION_COUNT,
    PLACE_COUNT,
    SOURCE_WORLD_SEED,
    sample_ring_world,
)


def test_reversal_changes_dynamics_but_not_the_observation_protocol() -> None:
    world = sample_ring_world(SOURCE_WORLD_SEED)
    changed = reversed_world(world)
    assert changed.transitions != world.transitions
    assert len(changed.transitions) == ACTION_COUNT
    assert all(len(row) == PLACE_COUNT for row in changed.transitions)
    assert changed.transitions[2] == world.transitions[2]


def test_rebuild_on_mismatch_recovers_before_mixed_model_control(tmp_path) -> None:
    report = run_reversal(tmp_path, replicates=1)
    assert report["recovery_stable_bits"] == [384]
    assert report["mixed_model_stable_bits"] == [512]
    assert report["accounting"]["recovery"]["optimizer_updates"] == 0
    assert report["accounting"]["mixed_model_control"]["replayed_examples"] == 0
