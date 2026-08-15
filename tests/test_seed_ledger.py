from __future__ import annotations

import pytest

from experiments.brainworkshop_canonical.dual_promotion import KNOWN_USED_SEEDS
from experiments.brainworkshop_canonical.seed_ledger import (
    BLOCKS,
    LIFETIMES_PER_REPLICATE,
    assert_unused_block,
    block,
    consumed_seeds,
)


def test_every_recorded_block_is_disjoint_from_every_other() -> None:
    for name, seeds in BLOCKS.items():
        assert_unused_block(name, seeds)


def test_blocks_do_not_reuse_dual_or_founding_seeds() -> None:
    every = {
        seed + offset
        for seeds in BLOCKS.values()
        for seed in seeds
        for offset in range(LIFETIMES_PER_REPLICATE + 1)
    }
    assert every.isdisjoint(KNOWN_USED_SEEDS)


def test_a_block_excludes_itself_but_nothing_else() -> None:
    mine = consumed_seeds(exclude="onset_lease_48")
    assert 125_017 not in mine
    assert 128_017 in mine
    assert 122_017 in mine
    with pytest.raises(KeyError):
        consumed_seeds(exclude="not_a_block")
    with pytest.raises(KeyError):
        block("not_a_block")


def test_a_shifted_block_that_lands_in_an_earlier_lease_is_refused() -> None:
    # 128_020 starts inside 128_017..128_023.
    with pytest.raises(ValueError, match="collides"):
        assert_unused_block("candidate_block", (128_020, 141_017, 142_017))
    # A hold session must not reach back into an earlier block either.
    with pytest.raises(ValueError, match="collides"):
        assert_unused_block("candidate_block", (130_014, 141_017, 142_017))
    # A genuinely fresh candidate clears.
    assert_unused_block("candidate_block", (141_017, 142_017, 143_017))


def test_a_block_may_not_be_redefined_or_padded_silently() -> None:
    with pytest.raises(ValueError, match="recorded as"):
        assert_unused_block("onset_lease_48", (125_017, 126_017, 127_018))
    with pytest.raises(ValueError, match="at least three"):
        assert_unused_block("unnamed_block", (140_017, 141_017))
    with pytest.raises(ValueError, match="unique"):
        assert_unused_block("unnamed_block", (140_017, 140_017, 141_017))


def test_replicates_inside_one_block_may_not_overlap() -> None:
    with pytest.raises(ValueError, match="overlap each other"):
        assert_unused_block("unnamed_block", (140_017, 140_020, 141_017))
