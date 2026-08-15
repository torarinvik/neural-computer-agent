from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.composition_accumulation import (
    fit_output_table,
    parts_of,
    table_accuracy,
    trivial_part,
)
from experiments.brainworkshop_canonical.compositional_rules import (
    sample_composite_population,
    sample_primitive_pool,
)
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.machine_factorization import (
    Partition,
    closed_partitions,
    factorize,
    reconstruct,
    smallest_closed_partition,
)
from experiments.brainworkshop_canonical.rule_automata import (
    known_rule,
    minimize,
    sample_rule,
)


def _stream(count: int, symbol_count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, symbol_count, (count,), generator=generator).tolist()


def _trace(rule, seed: int, length: int = 512) -> Trace:
    symbols = _stream(length, rule.symbol_count, seed)
    return Trace(
        symbols=tuple(symbols),
        outputs=tuple(rule.expected(symbols)),
        eligible=tuple([True] * length),
        symbol_count=rule.symbol_count,
    )


def test_a_closed_partition_really_is_closed() -> None:
    """States in one block must move to one block, for every symbol.

    This is the property the whole decomposition rests on, so it is checked
    directly rather than inferred from the algorithm terminating.
    """

    machine = minimize(known_rule("n_back", symbol_count=4, n_back=1))
    for partition in closed_partitions(machine):
        for symbol in range(machine.symbol_count):
            targets: dict[int, set[int]] = {}
            for state in range(machine.state_count):
                block = partition.blocks[state]
                moved = partition.blocks[int(machine.transitions[state][symbol])]
                targets.setdefault(block, set()).add(moved)
            assert all(len(seen) == 1 for seen in targets.values())


def test_the_smallest_closed_partition_identifies_what_it_was_asked_to() -> None:
    machine = minimize(known_rule("changed", symbol_count=4))
    partition = smallest_closed_partition(machine, 0, 1)
    assert partition.blocks[0] == partition.blocks[1]


def test_partition_algebra_behaves() -> None:
    left = Partition((0, 0, 1, 1))
    right = Partition((0, 1, 0, 1))
    assert left.meet(right).is_identity
    assert Partition((0, 0, 0, 0)).is_trivial
    assert Partition((0, 1, 2, 3)).is_identity
    assert left.meet(left).canonical().blocks == left.canonical().blocks
    with pytest.raises(ValueError, match="same state set"):
        left.meet(Partition((0, 1)))


def test_a_product_of_sampled_primitives_factors_back_apart() -> None:
    pool = sample_primitive_pool(count=4)
    composites = sample_composite_population(pool)
    symbols = _stream(2048, 4, 11)
    factored = 0
    for composite in composites:
        machine = minimize(composite.automaton)
        found = factorize(machine)
        if not found:
            continue
        factored += 1
        best = found[0]
        # The components multiply back to the same behaviour, both by
        # rebuilding the machine and by running the pair directly.
        assert reconstruct(best).expected(symbols) == machine.expected(symbols)
        assert best.predict(symbols) == machine.expected(symbols)
        assert min(best.part_sizes) > 1
        assert max(best.part_sizes) < machine.state_count
    # Most products retain the structure they were built with; some collapse
    # under minimisation and genuinely have none left.
    assert factored >= len(composites) // 2


def test_an_indecomposable_machine_reports_no_factors() -> None:
    # A one-state rule has nothing to split, and small machines are excluded.
    assert factorize(known_rule("current_symbol", symbol_count=4)) == ()
    assert parts_of(known_rule("current_symbol", symbol_count=4))[0].state_count == 1


def test_fitting_a_table_beats_enumerating_combiners(monkeypatch) -> None:
    """A fitted table expresses output functions no combiner list contains."""

    pool = sample_primitive_pool(count=2)
    left, right = pool[0], pool[1]
    symbols = _stream(1024, 4, 7)

    # An output rule that is not and, or, or xor: it depends on the symbol.
    def exotic(first: int, second: int, symbol: int) -> int:
        return int((first + second + symbol) % 3 == 0)

    outputs = []
    a = b = 0
    for symbol in symbols:
        outputs.append(exotic(a, b, symbol))
        a = int(left.transitions[a][symbol])
        b = int(right.transitions[b][symbol])
    trace = Trace(
        symbols=tuple(symbols),
        outputs=tuple(outputs),
        eligible=tuple([True] * len(symbols)),
        symbol_count=4,
    )
    table = fit_output_table(left, right, (trace,))
    assert table is not None
    assert table_accuracy(left, right, table, trace) == 1.0


def test_a_pair_that_cannot_explain_the_data_reports_a_conflict() -> None:
    rule = sample_rule(symbol_count=4, state_count=5, seed=6500)
    assert rule is not None
    trace = _trace(rule, 7)
    # Two trivial components cannot track five states, so some cell collides.
    assert fit_output_table(trivial_part(4), trivial_part(4), (trace,)) is None


def test_a_single_part_is_usable_through_the_trivial_component() -> None:
    rule = minimize(known_rule("onset", symbol_count=4))
    trace = _trace(rule, 7)
    table = fit_output_table(rule, trivial_part(4), (trace,))
    assert table is not None
    assert table_accuracy(rule, trivial_part(4), table, _trace(rule, 99)) == 1.0
