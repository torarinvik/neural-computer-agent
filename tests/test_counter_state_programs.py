from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.counter_state_programs import (
    PRESS_COUNTER,
    cluster_symbol_map,
    compile_rule,
    counter_layout,
    initial_counters,
    nearest_cluster,
    run_counter_program,
)
from experiments.brainworkshop_canonical.prototype_templates import (
    cluster_events,
    observe_events,
)
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from experiments.brainworkshop_canonical.rule_automata import known_rule, sample_rule

REPOSITORY = Path(__file__).resolve().parents[1]
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _setup(rule, *, steps: int = 128, seed: int = 41):
    encoders = RenderedBrainWorkshopEncoders.load(FRONTEND)
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        symbol_count=rule.symbol_count,
        match_rule="automaton",
        rule=rule,
    )
    clusters = cluster_events(observe_events(encoders, config, seed=seed))
    channels = cluster_symbol_map(encoders, clusters, symbol_count=rule.symbol_count)
    program = compile_rule(
        rule, channel_of_symbol=channels, cluster_count=int(clusters.shape[0])
    )
    start = initial_counters(
        program, cluster_count=int(clusters.shape[0]), states=rule.state_count
    )
    return encoders, config, clusters, program, start


def test_the_interface_reserves_press_and_inputs_then_leaves_state_free() -> None:
    layout = counter_layout(4, 6)
    assert layout["press"] == PRESS_COUNTER == 0
    assert layout["first_input"] == 1
    assert layout["first_working"] == 5
    assert layout["counter_count"] == 11
    with pytest.raises(ValueError, match="inputs and working state"):
        counter_layout(0, 4)


def test_a_compiled_rule_runs_at_one_through_the_real_executor() -> None:
    # A rule the temporal family cannot express: five states.
    rule = sample_rule(symbol_count=4, state_count=5, seed=6500)
    assert rule is not None
    encoders, config, clusters, program, start = _setup(rule)
    result = run_counter_program(
        program, encoders, config, clusters, seed=42, initial_counters=start
    )
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["statuses"] == "halted"


def test_every_hand_written_rule_compiles_including_two_back() -> None:
    for name, kwargs in (
        ("current_symbol", {}),
        ("onset", {}),
        ("changed", {}),
        ("n_back", {"n_back": 1}),
        ("n_back", {"n_back": 2}),
    ):
        rule = known_rule(name, symbol_count=4, **kwargs)
        encoders, config, clusters, program, start = _setup(rule)
        result = run_counter_program(
            program, encoders, config, clusters, seed=42, initial_counters=start
        )
        assert result["accuracy"] == pytest.approx(1.0), name


def test_state_persists_across_ticks_and_the_press_does_not() -> None:
    rule = known_rule("changed", symbol_count=4)
    encoders, config, clusters, program, start = _setup(rule)
    # `changed` is unsolvable without carrying the previous symbol, so a run
    # that wipes working state between ticks must fall below a perfect score.
    perfect = run_counter_program(
        program, encoders, config, clusters, seed=42, initial_counters=start
    )
    assert perfect["accuracy"] == pytest.approx(1.0)
    assert start[counter_layout(int(clusters.shape[0]), 2 * rule.state_count)["first_working"]] == 1


def test_the_bridge_refuses_a_program_that_does_not_match_the_layout() -> None:
    rule = sample_rule(symbol_count=4, state_count=2, seed=6200)
    assert rule is not None
    encoders, config, clusters, program, start = _setup(rule)
    with pytest.raises(ValueError, match="do not match the program"):
        run_counter_program(
            program,
            encoders,
            config,
            clusters,
            seed=42,
            initial_counters=start[:-1],
        )
    with pytest.raises(ValueError, match="does not cover the alphabet"):
        compile_rule(rule, channel_of_symbol=(0, 1), cluster_count=4)


def test_quantisation_sends_each_symbol_to_its_own_channel() -> None:
    rule = sample_rule(symbol_count=4, state_count=3, seed=6300)
    assert rule is not None
    encoders, config, clusters, _program, _start = _setup(rule)
    channels = cluster_symbol_map(encoders, clusters, symbol_count=4)
    # Four distinct stimuli must not collapse onto one another.
    assert len(set(channels)) == 4
    events = observe_events(encoders, config, seed=41)
    assigned = nearest_cluster(events, clusters)
    assert int(assigned.min()) >= 0
    assert int(assigned.max()) < clusters.shape[0]
    assert torch.equal(assigned, nearest_cluster(events, clusters))
