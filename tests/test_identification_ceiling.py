from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.identification_ceiling import (
    Trace,
    _rpni,
    held_out_accuracy,
    infer_machine,
)
from experiments.brainworkshop_canonical.rule_automata import known_rule, sample_rule


def _trace(rule, seed: int, length: int = 448) -> Trace:
    generator = torch.Generator().manual_seed(seed)
    symbols = torch.randint(
        0, rule.symbol_count, (length,), generator=generator
    ).tolist()
    return Trace(
        symbols=tuple(symbols),
        outputs=tuple(rule.expected(symbols)),
        eligible=tuple([True] * length),
        symbol_count=rule.symbol_count,
    )


@pytest.mark.parametrize(
    "name,kwargs,states,exact",
    [
        ("current_symbol", {}, 1, True),
        ("onset", {}, 2, True),
        ("changed", {}, 4, False),
        ("n_back", {"n_back": 1}, 4, False),
    ],
)
def test_one_episode_of_feedback_identifies_a_small_rule(
    name, kwargs, states, exact
) -> None:
    """One episode pins small rules down, and four-state ones only nearly.

    `exact` is measured rather than hoped for. At four states the search
    exhausts its node budget before it finds the minimal machine and settles
    for a five-state one that fits the probe exactly and misses roughly one
    held-out label in five hundred. An earlier version of this test asserted
    1.0 across the board and passed only because `minimize` was silently
    changing behaviour; see `test_rule_automata.py` for that regression.
    """

    rule = known_rule(name, symbol_count=4, **kwargs)
    assert rule.state_count == states
    probe = _trace(rule, 7)
    machine = infer_machine(probe)
    assert machine is not None
    # Whatever it returns must at least reproduce the evidence it was given.
    assert held_out_accuracy(machine, probe) == 1.0
    accuracy = held_out_accuracy(machine, _trace(rule, 99))
    if exact:
        assert accuracy == 1.0
        assert machine.state_count == states
    else:
        assert accuracy >= 0.99


def test_sampled_rules_up_to_four_states_are_identified() -> None:
    for state_count in (1, 2, 3, 4):
        rule = sample_rule(
            symbol_count=4, state_count=state_count, seed=6000 + 100 * state_count
        )
        assert rule is not None
        probe = _trace(rule, 7)
        machine = infer_machine(probe)
        assert machine is not None, state_count
        assert machine.state_count == state_count
        # Reproduces its evidence exactly; a single episode leaves a little
        # of the transition table unvisited, so held-out is near but not
        # always at 1.0.
        assert held_out_accuracy(machine, probe) == 1.0
        assert held_out_accuracy(machine, _trace(rule, 99)) >= 0.98


def test_the_inferred_machine_is_minimal_not_merely_consistent() -> None:
    rule = known_rule("onset", symbol_count=4)
    machine = infer_machine(_trace(rule, 7))
    assert machine is not None
    # State counts are searched in ascending order, so the first consistent
    # one is minimal by construction.
    assert machine.state_count == rule.state_count


def test_identification_stops_being_tractable_and_says_so() -> None:
    """Gold's NP-hardness, met in practice rather than in a footnote.

    A five-state rule does not finish within a budget an order of magnitude
    above what four states needs. Recording this is the point: it is why
    active querying, not more passive observation, is the way forward.
    """

    rule = sample_rule(symbol_count=4, state_count=5, seed=6500)
    assert rule is not None
    assert infer_machine(_trace(rule, 7), node_budget=500_000) is None
    # More passive episodes do not rescue it.
    episodes = tuple(_trace(rule, 7 + offset) for offset in range(4))
    assert infer_machine(episodes, node_budget=500_000) is None


def test_greedy_state_merging_fails_on_a_single_chain() -> None:
    """The documented negative result that motivates the exact search."""

    rule = known_rule("onset", symbol_count=4)
    merged = _rpni(_trace(rule, 7))
    assert merged is not None
    # Wildly over-states, because a chain gives the early merges no evidence.
    assert merged.state_count > 5 * rule.state_count


def test_a_trace_must_align() -> None:
    with pytest.raises(ValueError, match="align"):
        Trace(symbols=(0, 1), outputs=(1,), eligible=(True, True), symbol_count=2)


def test_a_degenerate_trace_yields_nothing() -> None:
    assert infer_machine(Trace((), (), (), 4)) is None
    assert infer_machine(Trace((0,), (1,), (True,), 4)) is None


def test_ineligible_steps_constrain_transitions_but_not_outputs() -> None:
    rule = known_rule("onset", symbol_count=4)
    trace = _trace(rule, 7)
    # Blind half the steps; the machine must still be recoverable.
    eligible = tuple(index % 2 == 0 for index in range(len(trace.symbols)))
    partial = Trace(
        symbols=trace.symbols,
        outputs=trace.outputs,
        eligible=eligible,
        symbol_count=trace.symbol_count,
    )
    machine = infer_machine(partial)
    assert machine is not None
    assert held_out_accuracy(machine, _trace(rule, 99)) == 1.0
