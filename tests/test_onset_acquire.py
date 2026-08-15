from __future__ import annotations

from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.current_symbol_acquire import (
    BOUND_FRONTEND_SEEDS,
    PREVIOUS_ACQUIRE_SEEDS,
    SEARCH_LEASE_SEEDS,
)
from experiments.brainworkshop_canonical.dual_promotion import KNOWN_USED_SEEDS
from experiments.brainworkshop_canonical.onset_acquire import (
    LONG_LEASE_ID,
    LONG_LEASE_SEEDS,
    LONG_STEPS,
    ONSET_LEASE_SEEDS,
    assert_unused_onset_seeds,
    run_onset_lease,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def test_onset_lease_seeds_are_disjoint_from_every_earlier_population() -> None:
    assert_unused_onset_seeds(ONSET_LEASE_SEEDS)
    assert_unused_onset_seeds(
        LONG_LEASE_SEEDS, additional_used=frozenset(ONSET_LEASE_SEEDS)
    )
    assert set(ONSET_LEASE_SEEDS).isdisjoint(KNOWN_USED_SEEDS)
    assert set(ONSET_LEASE_SEEDS).isdisjoint(PREVIOUS_ACQUIRE_SEEDS)
    assert set(ONSET_LEASE_SEEDS).isdisjoint(BOUND_FRONTEND_SEEDS)
    assert set(ONSET_LEASE_SEEDS).isdisjoint(SEARCH_LEASE_SEEDS)
    assert set(LONG_LEASE_SEEDS).isdisjoint(ONSET_LEASE_SEEDS)
    with pytest.raises(ValueError, match="collide"):
        assert_unused_onset_seeds((125_017, 122_017, 127_017))
    with pytest.raises(ValueError, match="collide"):
        assert_unused_onset_seeds(
            LONG_LEASE_SEEDS, additional_used=frozenset({129_017})
        )


def test_lease_lifetimes_may_not_overlap_an_earlier_lease_block() -> None:
    # 122_020 starts inside the search lease block 122_017..122_023.
    with pytest.raises(ValueError, match="reuses lifetimes"):
        assert_unused_onset_seeds((122_020, 126_017, 127_017))
    # A hold session must not reach back into an earlier block either.
    with pytest.raises(ValueError, match="collide"):
        assert_unused_onset_seeds((124_014, 126_017, 127_017))


def test_onset_lease_selects_and_and_does_not_write_the_bank(tmp_path: Path) -> None:
    before = sha256_file(BANK)
    campaign = run_onset_lease(
        CONTROLLER,
        BANK,
        tmp_path,
        seeds=LONG_LEASE_SEEDS,
        steps=LONG_STEPS,
        sessions=3,
        frontend_path=FRONTEND,
        experiment_id=LONG_LEASE_ID,
        additional_used=frozenset(ONSET_LEASE_SEEDS),
    )
    assert sha256_file(BANK) == before
    assert campaign["bank_unchanged"]
    assert campaign["admitted"] is False
    assert campaign["accepted"]
    assert campaign["status"] == "replicated_not_admitted"
    assert campaign["frontend_shared"]
    assert campaign["winner_kinds"] == ["and", "and", "and"]
    for row in campaign["replicates"]:
        winner = row["search"]["winner"]
        assert winner["kind"] == "and"
        assert winner["frontend_digest"] == campaign["frontend_digest"]
        assert row["stable_bits_to_threshold"] is not None
        assert row["frozen_holds"]
        # Onset needs two families: neither the delay file nor its inverse
        # clears threshold alone, and neither does the prototype alone.
        assert row["retrieve_slot0"]["accuracy"] < 0.8
        assert row["invert_slot0"]["accuracy"] < 0.8
        assert row["prototype_only"]["accuracy"] < 0.8
        assert row["zeros"]["accuracy"] < 0.8
        assert row["action_reversed"]["accuracy"] < 0.8
        assert row["reward_shuffled"]["accuracy"] < 0.8
        assert row["cross_encoder"]["accuracy"] < 0.8
        assert row["frontend_digest"] != row["cross_frontend_digest"]
        assert row["controller_digest"] == campaign["replicates"][0]["controller_digest"]
        executed = [
            item
            for item in row["search"]["attempts"]
            if item["executed"] and item["kind"] in {"retrieve", "invert"}
        ]
        assert executed
        assert all(item["accuracy"] < 0.8 for item in executed)
