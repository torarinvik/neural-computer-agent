from __future__ import annotations

import itertools

import pytest
import torch

from experiments.brainworkshop_canonical.composition_accumulation import (
    library_candidates,
)
from experiments.brainworkshop_canonical.compositional_rules import (
    COMBINERS,
    product_rule,
    sample_composite_population,
    sample_primitive_pool,
    shares_a_part,
)
from experiments.brainworkshop_canonical.rule_automata import (
    known_rule,
    positive_rate,
    sample_rule,
)


def _stream(count: int, symbol_count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, symbol_count, (count,), generator=generator).tolist()


def test_a_product_computes_the_combiner_of_its_parts() -> None:
    left = known_rule("onset", symbol_count=4)
    right = known_rule("n_back", symbol_count=4, n_back=1)
    symbols = _stream(512, 4, 3)
    for name, merge in COMBINERS.items():
        product = product_rule(left, right, name)
        expected = [
            merge(a, b)
            for a, b in zip(left.expected(symbols), right.expected(symbols))
        ]
        assert product.expected(symbols) == expected, name


def test_a_product_is_harder_than_either_part() -> None:
    left = sample_rule(symbol_count=4, state_count=3, seed=8000)
    right = sample_rule(symbol_count=4, state_count=3, seed=8100)
    assert left is not None and right is not None
    product = product_rule(left, right, "xor")
    # Minimised, so this is the honest complexity rather than the pair count.
    assert product.state_count > max(left.state_count, right.state_count)
    assert product.state_count <= left.state_count * right.state_count


def test_products_are_rejected_across_alphabets_and_unknown_combiners() -> None:
    left = sample_rule(symbol_count=4, state_count=2, seed=8000)
    narrow = sample_rule(symbol_count=3, state_count=2, seed=8000)
    assert left is not None and narrow is not None
    with pytest.raises(ValueError, match="share an alphabet"):
        product_rule(left, narrow, "and")
    with pytest.raises(ValueError, match="unknown combiner"):
        product_rule(left, left, "nand")


def test_the_composite_population_shares_parts_and_stays_measurable() -> None:
    pool = sample_primitive_pool(count=4)
    assert len({item.digest() for item in pool}) == 4
    composites = sample_composite_population(pool)
    assert composites
    assert len({item.digest() for item in composites}) == len(composites)
    for item in composites:
        # A task nobody can be separated from a constant policy on is useless.
        assert 0.15 <= positive_rate(item.automaton, seed=21) <= 0.85
        assert len(item.parts) == 2


def test_shared_parts_show_up_as_shared_behaviour() -> None:
    """The distribution must actually contain the structure it claims.

    Independent samples share nothing, which is why the original accumulation
    curve could not have found composition whatever the agent did.
    """

    pool = sample_primitive_pool(count=4)
    composites = sample_composite_population(pool)
    symbols = _stream(4096, 4, 5)
    predicted = [item.automaton.expected(symbols) for item in composites]
    related: list[float] = []
    unrelated: list[float] = []
    for first, second in itertools.combinations(range(len(composites)), 2):
        agreement = sum(
            1 for a, b in zip(predicted[first], predicted[second]) if a == b
        ) / len(symbols)
        target = related if shares_a_part(composites[first], composites[second]) else unrelated
        target.append(max(agreement, 1.0 - agreement))
    assert related and unrelated
    assert sum(related) / len(related) > sum(unrelated) / len(unrelated)


def test_library_candidates_offer_files_and_their_products() -> None:
    library = tuple(
        item
        for item in (
            sample_rule(symbol_count=4, state_count=2, seed=8000),
            sample_rule(symbol_count=4, state_count=3, seed=8100),
        )
        if item is not None
    )
    assert len(library) == 2
    candidates = library_candidates(library)
    labels = [label for label, _ in candidates]
    assert labels[:2] == ["retrieve:0", "retrieve:1"]
    assert {"and:0+1", "or:0+1", "xor:0+1"} <= set(labels)
    # Every candidate is a usable machine over the same alphabet.
    for _, machine in candidates:
        assert machine.symbol_count == 4
        machine.validate()


def test_an_empty_library_offers_nothing() -> None:
    assert library_candidates(()) == ()
