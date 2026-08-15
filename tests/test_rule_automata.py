from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopVerifier,
)
from experiments.brainworkshop_canonical.rule_automata import (
    RuleAutomaton,
    canonicalize,
    held_out_split,
    known_rule,
    minimize,
    positive_rate,
    sample_rule,
    sample_rule_population,
)


def _stream(count: int, symbol_count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, symbol_count, (count,), generator=generator).tolist()


def test_the_hand_written_rules_are_instances_of_the_sampled_class() -> None:
    symbols = _stream(200, 4, 7)

    def hand(name: str, index: int, symbol: int) -> int:
        if name == "current_symbol":
            return int(symbol == 0)
        if name == "changed":
            return int(symbol != symbols[index - 1])
        if name == "onset":
            return int(symbol == 0 and symbol != symbols[index - 1])
        if name == "n_back1":
            return int(symbol == symbols[index - 1])
        return int(symbol == symbols[index - 2])

    cases = (
        ("current_symbol", {}, "current_symbol", 0),
        ("changed", {}, "changed", 1),
        ("onset", {}, "onset", 1),
        ("n_back", {"n_back": 1}, "n_back1", 1),
        ("n_back", {"n_back": 2}, "n_back2", 2),
    )
    for name, kwargs, key, warmup in cases:
        rule = known_rule(name, symbol_count=4, **kwargs)
        produced = rule.expected(symbols)
        for index in range(warmup, len(symbols)):
            assert produced[index] == hand(key, index, symbols[index]), (name, index)


def test_minimal_state_counts_place_the_existing_records_on_the_axis() -> None:
    # These are the complexities of everything measured before this module.
    assert known_rule("current_symbol", symbol_count=4).state_count == 1
    assert known_rule("onset", symbol_count=4).state_count == 2
    assert known_rule("changed", symbol_count=4).state_count == 4
    assert known_rule("n_back", symbol_count=4, n_back=1).state_count == 4
    assert known_rule("n_back", symbol_count=4, n_back=2).state_count == 16


def test_minimisation_never_changes_behaviour() -> None:
    """The regression that mattered: `minimize` used to return a different rule.

    Merged blocks were numbered by signature order, which has no reason to put
    the block containing the start state first, while `expected` always starts
    at state 0. A symmetric two-state parity rule came back with its outputs
    swapped -- the exact inverse of the machine that went in -- and a sweep of
    random machines put the rate at 42.5%.
    """

    parity = RuleAutomaton(
        symbol_count=4,
        transitions=((1, 0, 0, 0), (0, 1, 1, 1)),
        outputs=((1, 0, 0, 0), (0, 1, 1, 1)),
    )
    symbols = _stream(256, 4, 7)
    assert minimize(parity).expected(symbols) == parity.expected(symbols)

    generator = torch.Generator().manual_seed(3)
    for states in (1, 2, 3, 4, 5):
        for _ in range(60):
            machine = RuleAutomaton(
                symbol_count=4,
                transitions=tuple(
                    tuple(
                        int(value)
                        for value in torch.randint(0, states, (4,), generator=generator)
                    )
                    for _ in range(states)
                ),
                outputs=tuple(
                    tuple(
                        int(value)
                        for value in torch.randint(0, 2, (4,), generator=generator)
                    )
                    for _ in range(states)
                ),
            )
            stream = [
                int(value)
                for value in torch.randint(0, 4, (128,), generator=generator)
            ]
            reduced = minimize(machine)
            assert reduced.expected(stream) == machine.expected(stream)
            # And it is a fixed point, so digests are stable identities.
            assert minimize(reduced).digest() == reduced.digest()


def test_minimisation_merges_duplicates_and_drops_unreachable_states() -> None:
    # States 0 and 1 behave identically; state 2 is unreachable.
    redundant = RuleAutomaton(
        symbol_count=2,
        transitions=((0, 1), (0, 1), (2, 2)),
        outputs=((1, 0), (1, 0), (0, 1)),
    )
    reduced = minimize(redundant)
    assert reduced.state_count == 1
    symbols = _stream(64, 2, 3)
    assert reduced.expected(symbols) == redundant.expected(symbols)


def test_identity_is_canonical_under_relabelling() -> None:
    rule = sample_rule(symbol_count=3, state_count=4, seed=515)
    assert rule is not None
    # Relabel every state by a permutation; behaviour and identity must hold.
    permutation = {0: 0, 1: 3, 2: 1, 3: 2}
    inverse = {new: old for old, new in permutation.items()}
    relabelled = RuleAutomaton(
        symbol_count=rule.symbol_count,
        transitions=tuple(
            tuple(permutation[rule.transitions[inverse[state]][symbol]] for symbol in range(rule.symbol_count))
            for state in range(rule.state_count)
        ),
        outputs=tuple(
            tuple(rule.outputs[inverse[state]]) for state in range(rule.state_count)
        ),
    )
    assert relabelled.digest() == rule.digest()
    symbols = _stream(128, 3, 5)
    assert relabelled.expected(symbols) == rule.expected(symbols)
    assert canonicalize(relabelled).payload() == rule.payload()


def test_sampling_yields_distinct_measurable_rules_at_the_asked_complexity() -> None:
    population = sample_rule_population(
        symbol_count=4, state_counts=(1, 2, 3, 4, 5), count=25, seed=9001
    )
    assert len(population) == 25
    assert len({rule.digest() for rule in population}) == 25
    assert {rule.state_count for rule in population} == {1, 2, 3, 4, 5}
    for rule in population:
        # A rule that almost never asks for a press cannot separate a learner
        # from a constant policy at any episode length.
        assert 0.15 <= positive_rate(rule, seed=11) <= 0.85


def test_the_holdout_is_unseen_rules_and_does_not_drift() -> None:
    population = sample_rule_population(
        symbol_count=4, state_counts=(2, 3, 4), count=30, seed=4242
    )
    train, holdout = held_out_split(population)
    assert train and holdout
    assert {rule.digest() for rule in train}.isdisjoint(
        {rule.digest() for rule in holdout}
    )
    # Growing the population must never move a rule across the boundary.
    grown = population + sample_rule_population(
        symbol_count=4,
        state_counts=(3, 5),
        count=10,
        seed=777,
        exclude_digests=frozenset(rule.digest() for rule in population),
    )
    grown_train, grown_holdout = held_out_split(grown)
    assert {rule.digest() for rule in holdout} <= {
        rule.digest() for rule in grown_holdout
    }
    assert {rule.digest() for rule in train} <= {rule.digest() for rule in grown_train}


def test_a_sampled_rule_runs_as_a_task_and_only_the_rule_solves_it() -> None:
    rule = sample_rule(symbol_count=4, state_count=5, seed=4242)
    assert rule is not None
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=64,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=rule,
    )
    verifier = RenderedBrainWorkshopVerifier(config, seed=11)
    state = 0
    hits = 0
    scored = 0
    while not verifier.done:
        symbol = int(verifier._symbols["vision"][verifier.position])
        press = int(rule.outputs[state][symbol])
        state = int(rule.transitions[state][symbol])
        step = verifier.score(torch.tensor([press], dtype=torch.long))
        hits += int(step.reward.item())
        scored += 1
    assert scored == 64
    assert hits == 64
    constant = RenderedBrainWorkshopVerifier(config, seed=11)
    constant_hits = sum(
        int(constant.score(torch.tensor([0], dtype=torch.long)).reward.item())
        for _ in range(64)
    )
    assert constant_hits < 64


def test_the_automaton_task_validates_its_rule() -> None:
    rule = sample_rule(symbol_count=4, state_count=3, seed=99)
    assert rule is not None
    with pytest.raises(ValueError, match="needs a sampled rule"):
        RenderedBrainWorkshopConfig(
            steps=32, streams=("vision",), match_rule="automaton"
        ).validate()
    with pytest.raises(ValueError, match="rule alphabet"):
        RenderedBrainWorkshopConfig(
            steps=32,
            streams=("vision",),
            symbol_count=8,
            match_rule="automaton",
            rule=rule,
        ).validate()
    with pytest.raises(ValueError, match="only the automaton task"):
        RenderedBrainWorkshopConfig(
            steps=32, streams=("vision",), symbol_count=4, rule=rule
        ).validate()


def test_observed_templates_come_only_from_what_the_learner_can_see() -> None:
    from pathlib import Path

    from experiments.brainworkshop_canonical.prototype_templates import (
        candidate_templates,
        cluster_events,
        observe_events,
        observed_templates,
    )
    from experiments.brainworkshop_canonical.rendered_environment import (
        RenderedBrainWorkshopEncoders,
    )

    repository = Path(__file__).resolve().parents[1]
    encoders = RenderedBrainWorkshopEncoders.load(
        repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
    )
    rule = sample_rule(symbol_count=4, state_count=3, seed=1234)
    assert rule is not None
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=64,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=rule,
    )
    events = observe_events(encoders, config, seed=5)
    assert events.shape == (64, encoders.event_width)
    # The alphabet is discovered from the stream, not supplied.
    clusters = cluster_events(events)
    assert clusters.shape[0] == 4
    templates = candidate_templates(clusters, maximum_subset=4)
    assert len(templates) == 15
    assert templates[0][0] == (0,)  # simplest hypotheses first
    assert all(
        template.shape == (encoders.event_width,) for _, template in templates
    )
    assert len(observed_templates(encoders, config, seed=5)) == 15


def test_templates_are_appended_so_an_earlier_winner_still_wins() -> None:
    from pathlib import Path

    from experiments.brainworkshop_canonical.program_search import propose_from_bank
    from neural_computer import ExternalTemporalProgramBank

    repository = Path(__file__).resolve().parents[1]
    bank = ExternalTemporalProgramBank.load_bank(
        repository / "artifacts/checkpoints/AgentBrain.bank"
    )
    plain = propose_from_bank(bank)
    templates = ((0,), torch.zeros(bank.context_width)), ((1,), torch.ones(bank.context_width))
    widened = propose_from_bank(bank, templates)
    assert len(widened) > len(plain)
    # Every original proposal keeps its position, so no recorded winner moves.
    assert [item.label() for item in widened[: len(plain)]] == [
        item.label() for item in plain
    ]
    tail = widened[len(plain) :]
    assert all(item.template is not None for item in tail)
    # Both polarities are offered.
    assert any(item.invert_intention for item in tail)
    assert any(not item.invert_intention for item in tail)
