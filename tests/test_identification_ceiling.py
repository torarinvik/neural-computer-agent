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
    "name,kwargs,states",
    [
        ("current_symbol", {}, 1),
        ("onset", {}, 2),
        ("changed", {}, 4),
        ("n_back", {"n_back": 1}, 4),
    ],
)
def test_one_episode_of_feedback_identifies_a_small_rule(name, kwargs, states) -> None:
    rule = known_rule(name, symbol_count=4, **kwargs)
    assert rule.state_count == states
    machine = infer_machine(_trace(rule, 7))
    assert machine is not None
    # Scored by prediction, not by digest: cluster and state names are
    # arbitrary, and only behaviour is the claim.
    assert held_out_accuracy(machine, _trace(rule, 99)) == 1.0


def test_sampled_rules_up_to_four_states_are_identified() -> None:
    for state_count in (1, 2, 3, 4):
        rule = sample_rule(symbol_count=4, state_count=state_count, seed=6000 + 100 * state_count)
        assert rule is not None
        machine = infer_machine(_trace(rule, 7))
        assert machine is not None, state_count
        assert machine.state_count == state_count
        assert held_out_accuracy(machine, _trace(rule, 99)) == 1.0


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
