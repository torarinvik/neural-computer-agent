from __future__ import annotations

from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.current_symbol_acquire import (
    BOUND_FRONTEND_SEEDS,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    PREVIOUS_ACQUIRE_SEEDS,
    SEARCH_LEASE_SEEDS,
    assert_unused_holdout_seeds,
    assert_unused_search_lease_seeds,
    require_controller,
    run_campaign,
    run_search_lease,
)
from experiments.brainworkshop_canonical.dual_promotion import (
    CONTROLLER_SHA256,
    KNOWN_USED_SEEDS,
)
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopEncoders,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"


def test_holdout_seeds_are_disjoint_from_used_populations() -> None:
    assert_unused_holdout_seeds(HOLDOUT_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(KNOWN_USED_SEEDS)
    assert set(HOLDOUT_SEEDS).isdisjoint(set(DEVELOPMENT_SEEDS))
    assert {113_017, 114_017, 115_017}.isdisjoint(set(HOLDOUT_SEEDS))
    assert PREVIOUS_ACQUIRE_SEEDS.isdisjoint(set(HOLDOUT_SEEDS))
    with pytest.raises(ValueError, match="collide"):
        assert_unused_holdout_seeds((119_017, 41, 121_017))
    with pytest.raises(ValueError, match="collide"):
        assert_unused_holdout_seeds((119_017, 116_017, 121_017))
    with pytest.raises(ValueError, match="Dual holdout"):
        assert_unused_holdout_seeds((119_017, 113_017, 121_017))


def test_search_lease_seeds_are_disjoint_from_bound_frontend_population() -> None:
    assert_unused_search_lease_seeds(SEARCH_LEASE_SEEDS)
    assert set(SEARCH_LEASE_SEEDS).isdisjoint(KNOWN_USED_SEEDS)
    assert set(SEARCH_LEASE_SEEDS).isdisjoint(PREVIOUS_ACQUIRE_SEEDS)
    assert set(SEARCH_LEASE_SEEDS).isdisjoint(BOUND_FRONTEND_SEEDS)
    with pytest.raises(ValueError, match="collide"):
        assert_unused_search_lease_seeds((122_017, 119_017, 124_017))


def test_controller_digest_is_frozen() -> None:
    assert require_controller(CONTROLLER) == CONTROLLER_SHA256


def test_unused_seed_campaign_does_not_write_the_bank(tmp_path: Path) -> None:
    before = sha256_file(BANK)
    campaign = run_campaign(
        CONTROLLER,
        BANK,
        tmp_path,
        seeds=HOLDOUT_SEEDS,
        steps=448,
    )
    assert sha256_file(BANK) == before
    assert campaign["bank_unchanged"]
    assert campaign["admitted"] is False
    assert campaign["accepted"]
    assert campaign["status"] == "replicated_not_admitted"
    assert campaign["frontend_shared"] is True
    assert len(campaign["replicates"]) == 3
    for row in campaign["replicates"]:
        assert row["frontend_digest"] == campaign["frontend_digest"]
        assert row["hold"]["accuracy"] >= 0.8
        assert row["zeros"]["accuracy"] < 0.8
        assert row["action_reversed"]["accuracy"] < 0.8
        assert row["delay_slot0"]["accuracy"] < 0.8
        assert row["cross_encoder"]["accuracy"] < 0.8
        assert row["frontend_digest"] != row["cross_frontend_digest"]
        assert row["controller_unchanged"]


def test_seeded_frontend_round_trips_and_is_reproducible(tmp_path: Path) -> None:
    first = RenderedBrainWorkshopEncoders.seeded(16, source_key_width=4, seed=1001)
    second = RenderedBrainWorkshopEncoders.seeded(16, source_key_width=4, seed=1001)
    other = RenderedBrainWorkshopEncoders.seeded(16, source_key_width=4, seed=1002)
    assert first.digest() == second.digest()
    assert first.digest() != other.digest()
    path = tmp_path / "rendered_frontend_seed1001.pt"
    first.save(path, seed=1001)
    restored = RenderedBrainWorkshopEncoders.load(path)
    assert restored.digest() == first.digest()


def test_one_bound_prototype_holds_on_another_seed_of_the_same_frontend() -> None:
    from experiments.brainworkshop_canonical.controller_pretraining import (
        build_pretrained_controller_program_machine,
        load_temporal_controller_artifact,
    )
    from experiments.brainworkshop_canonical.current_symbol_acquire import (
        current_symbol_config,
    )
    from experiments.brainworkshop_canonical.rendered_live import (
        run_rendered_live_lifetime,
    )
    from neural_computer.temporal_program import PROTOTYPE_MATCH_EXECUTION_SCHEMA

    machine = build_pretrained_controller_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        learning_rate=0.3,
        sample=True,
        inherit_program_prior=False,
    )
    machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
    encoders = RenderedBrainWorkshopEncoders.seeded(
        machine.event_width, source_key_width=machine.source_key_width, seed=1001
    )
    config = current_symbol_config(steps=24)
    train = run_rendered_live_lifetime(
        machine, encoders, config, seed=119_017, learn=True, sample=True
    )
    machine.learning_enabled = False
    machine.sample = False
    hold = run_rendered_live_lifetime(
        machine, encoders, config, seed=120_017, learn=False, sample=False
    )
    other = RenderedBrainWorkshopEncoders.seeded(
        machine.event_width, source_key_width=machine.source_key_width, seed=1002
    )
    crossed = run_rendered_live_lifetime(
        machine, other, config, seed=120_017, learn=False, sample=False
    )
    assert train.eligible_accuracy >= 0.8
    assert hold.eligible_accuracy >= 0.8
    assert crossed.eligible_accuracy < 0.8
    assert hold.program_file_updates == 0


def test_search_lease_binds_frontend_and_does_not_write_the_bank(
    tmp_path: Path,
) -> None:
    before = sha256_file(BANK)
    frontend = (
        REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
    )
    # 448 steps, not 24. Since `and` entered the grammar ahead of `invent`,
    # a short episode lets an AND scrape past 0.8 on the base rate before the
    # searcher ever reaches invent, and 448 is past the trial floor.
    campaign = run_search_lease(
        CONTROLLER,
        BANK,
        tmp_path,
        seeds=SEARCH_LEASE_SEEDS,
        steps=448,
        sessions=3,
        frontend_path=frontend,
    )
    assert sha256_file(BANK) == before
    assert campaign["bank_unchanged"]
    assert campaign["admitted"] is False
    assert campaign["accepted"]
    assert campaign["frontend_shared"]
    for row in campaign["replicates"]:
        assert row["search"]["winner"]["kind"] == "invent"
        assert row["search"]["winner"]["frontend_digest"] == campaign["frontend_digest"]
        assert row["stable_bits_to_threshold"] is not None
        assert row["zeros"]["accuracy"] < 0.8
        assert row["delay_slot0"]["accuracy"] < 0.8
        assert row["cross_encoder"]["accuracy"] < 0.8


def test_a_short_campaign_cannot_be_accepted(tmp_path: Path) -> None:
    import pytest as _pytest

    with _pytest.raises(ValueError, match="too short to discriminate"):
        run_campaign(CONTROLLER, BANK, tmp_path, seeds=HOLDOUT_SEEDS, steps=24)
    campaign = run_campaign(
        CONTROLLER,
        BANK,
        tmp_path,
        seeds=HOLDOUT_SEEDS,
        steps=24,
        enforce_discrimination=False,
    )
    assert campaign["discrimination"]["discriminating"] is False
    assert campaign["accepted"] is False
