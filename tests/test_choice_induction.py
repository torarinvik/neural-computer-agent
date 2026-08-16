from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.choice_induction import (
    ChoiceTrace,
    agreement,
    induce_from_choices,
)
from experiments.brainworkshop_canonical.rule_automata import (
    RuleAutomaton,
    best_constant_rate,
    minimize,
    sample_rule,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"


def _rule(states: int, actions: int, seed: int = 1200):
    rule = sample_rule(
        symbol_count=4,
        state_count=states,
        seed=seed + 10 * actions + states,
        action_count=actions,
        maximum_constant_rate=0.6,
    )
    assert rule is not None
    return rule


def _trace(rule, *, episodes: int, steps: int = 16, seed: int = 3, policy="uniform"):
    """Episodes under a chosen policy, scored exactly as the verifier would."""

    generator = torch.Generator().manual_seed(seed)
    produced = []
    for _ in range(episodes):
        symbols = torch.randint(0, 4, (steps,), generator=generator).tolist()
        targets = rule.expected(symbols)
        if policy == "uniform":
            actions = torch.randint(
                0, rule.action_count, (steps,), generator=generator
            ).tolist()
        else:
            actions = [0] * steps
        produced.append(
            ChoiceTrace(
                symbols=tuple(symbols),
                actions=tuple(int(a) for a in actions),
                rewards=tuple(int(a == t) for a, t in zip(actions, targets)),
                eligible=tuple([True] * steps),
                symbol_count=4,
                action_count=rule.action_count,
            )
        )
    return tuple(produced)


def _held_out(machine, rule, *, seed: int = 99, episodes: int = 20, steps: int = 48):
    generator = torch.Generator().manual_seed(seed)
    hits = trials = 0
    for _ in range(episodes):
        symbols = torch.randint(0, 4, (steps,), generator=generator).tolist()
        predicted = machine.expected(symbols)
        for want, got in zip(rule.expected(symbols), predicted):
            trials += 1
            hits += int(want == got)
    return hits / trials


# --- what changes when there are more than two actions ---------------------


def test_two_actions_hide_the_problem_and_three_expose_it() -> None:
    """The single most load-bearing assumption in this repository.

    With two actions a failure names the target as surely as a success does,
    so a policy that never varies still learns everything. With three it
    learns only where its one action is wrong.
    """

    binary = _rule(3, 2)
    fixed = induce_from_choices(_trace(binary, episodes=28, policy="fixed"))
    assert fixed is not None
    assert _held_out(fixed.machine, binary) == 1.0

    for actions in (3, 4):
        rule = _rule(3, actions)
        blind = induce_from_choices(_trace(rule, episodes=28, policy="fixed"))
        seeing = induce_from_choices(_trace(rule, episodes=28, policy="uniform"))
        assert blind is not None and seeing is not None
        assert _held_out(seeing.machine, rule) == 1.0
        assert _held_out(blind.machine, rule) < 0.9, actions


def test_a_rule_is_recovered_exactly_at_every_action_count() -> None:
    for actions in (2, 3, 4, 5):
        for states in (1, 2, 3):
            rule = _rule(states, actions)
            fit = induce_from_choices(_trace(rule, episodes=56))
            assert fit is not None, (actions, states)
            assert _held_out(fit.machine, rule) == 1.0, (actions, states)
            assert fit.machine.action_count == actions


def test_a_correct_machine_is_never_charged_for_a_failed_guess() -> None:
    """Why consistency rather than accuracy is the objective.

    Most steps under a uniform policy are failures. An objective that treated
    a correctly-predicted failure as a disagreement would score the truth
    worse than a machine that agreed with whatever was tried.
    """

    rule = _rule(3, 4)
    traces = _trace(rule, episodes=20)
    consistent, trials = agreement(rule, traces)
    assert consistent == trials
    fit = induce_from_choices(traces)
    assert fit is not None
    assert fit.disagreements == 0
    # And most of the evidence really is negative, which is the point.
    resolved = sum(trace.resolved for trace in traces)
    assert resolved < 0.4 * trials


def test_information_per_step_falls_as_the_action_set_grows() -> None:
    fractions = []
    for actions in (2, 4, 8):
        rule = _rule(2, actions) if actions < 8 else None
        if rule is None:
            rule = RuleAutomaton(
                symbol_count=4,
                transitions=((0, 0, 0, 0),),
                outputs=((0, 1, 2, 3),),
                action_count=8,
            ).validate()
        traces = _trace(rule, episodes=20)
        trials = sum(len(trace.symbols) for trace in traces)
        fractions.append(sum(trace.resolved for trace in traces) / trials)
    assert fractions[0] > fractions[1] > fractions[2]
    # Roughly one step in k is fully resolved under a uniform policy.
    assert 0.35 < fractions[0] < 0.65
    assert 0.05 < fractions[2] < 0.25


def test_shuffling_the_rewards_destroys_the_learning() -> None:
    rule = _rule(3, 4)
    traces = _trace(rule, episodes=56)
    generator = torch.Generator().manual_seed(7)
    scrambled = tuple(
        ChoiceTrace(
            symbols=trace.symbols,
            actions=trace.actions,
            rewards=tuple(
                trace.rewards[index]
                for index in torch.randperm(
                    len(trace.rewards), generator=generator
                ).tolist()
            ),
            eligible=trace.eligible,
            symbol_count=trace.symbol_count,
            action_count=trace.action_count,
        )
        for trace in traces
    )
    fit = induce_from_choices(scrambled)
    assert fit is not None
    assert _held_out(fit.machine, rule) < 0.6


def test_the_binary_case_is_unchanged_by_the_generalisation() -> None:
    """Every earlier result was collected on rules that must still be these."""

    for states in (1, 2, 3, 4, 5, 6):
        rule = sample_rule(symbol_count=4, state_count=states, seed=6000 + 100 * states)
        assert rule is not None
        assert rule.action_count == 2
        # The default action count is not written into the payload, so digests
        # recorded before actions were a choice still mean what they meant.
        assert "action_count" not in rule.payload()


def test_the_action_count_survives_minimisation() -> None:
    rule = _rule(3, 4)
    padded = RuleAutomaton(
        symbol_count=rule.symbol_count,
        transitions=rule.transitions + (rule.transitions[0],),
        outputs=rule.outputs + (rule.outputs[0],),
        action_count=4,
    )
    assert minimize(padded).action_count == 4


def test_the_best_constant_answer_is_the_baseline_to_beat() -> None:
    for actions in (2, 3, 5):
        rule = _rule(3, actions)
        rate = best_constant_rate(rule, seed=41)
        assert rate <= 0.6
        assert rate >= 1.0 / actions - 0.15


def test_degenerate_traces_are_refused() -> None:
    assert induce_from_choices(()) is None
    with pytest.raises(ValueError, match="square"):
        ChoiceTrace((0, 1), (0,), (1,), (True, True), 4, 3).validate()
    with pytest.raises(ValueError, match="outside the protocol"):
        ChoiceTrace((0,), (9,), (1,), (True,), 4, 3).validate()
    with pytest.raises(ValueError, match="not a scalar outcome"):
        ChoiceTrace((0,), (1,), (2,), (True,), 4, 3).validate()


# --- compiling, keeping and finding a k-action capability ------------------


def test_a_multi_action_machine_compiles_to_a_program_that_reproduces_it() -> None:
    """Without this the k-action result is a fitter result, not an agent one."""

    from experiments.brainworkshop_canonical.choice_programs import (
        choice_initial_counters,
        compile_choice_rule,
        predict_choice_symbols,
    )
    from neural_computer.induced_library import canonical_signature_stream

    stream = canonical_signature_stream(4)
    for actions in (2, 3, 4, 5):
        for states in (1, 2, 3):
            rule = _rule(states, actions)
            program = compile_choice_rule(rule, cluster_count=4)
            start = choice_initial_counters(rule, cluster_count=4)
            answers, statuses = predict_choice_symbols(
                program,
                stream,
                action_count=actions,
                cluster_count=4,
                initial_counters=start,
            )
            assert statuses == ("halted",), (actions, states)
            assert answers == tuple(rule.expected(list(stream))), (actions, states)


def test_a_k_action_program_stores_and_reloads(tmp_path) -> None:
    from experiments.brainworkshop_canonical.choice_accumulation import record_for
    from neural_computer.induced_library import InducedProgramLibrary

    rule = _rule(3, 4)
    library = InducedProgramLibrary(alphabet=4)
    library.append(record_for(rule, alphabet=4, provenance={"source": "induced"}))
    path = tmp_path / "choices.library"
    library.save(path)
    loaded = InducedProgramLibrary.load(path)
    assert loaded.digest() == library.digest()
    assert loaded.record(0).action_count == 4


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_widening_the_record_left_every_committed_library_untouched() -> None:
    """Six libraries were admitted before answers could be a choice."""

    from neural_computer.induced_library import InducedProgramLibrary

    stored = sorted((REPOSITORY / "artifacts/checkpoints").glob("*.library"))
    assert stored, "no committed libraries to check"
    for path in stored:
        library = InducedProgramLibrary.load(path)
        assert library.record_count > 0
        assert all(record.action_count == 2 for record in library.records())
        # The default is not written, so the bytes on disk are what they were.
        assert all("action_count" not in record.payload() for record in library.records())


def test_consistency_cannot_be_tested_directly_and_the_inversion_fixes_it() -> None:
    """A coin flip is consistent with 0.625 of outcomes at four actions.

    Testing that against a 0.8 gate is a different act at every action count,
    which is why the rate is inverted into an accuracy before it is tested.
    """

    from experiments.brainworkshop_canonical.choice_induction import implied_accuracy

    for actions in (2, 3, 4, 5):
        chance = 1.0 / actions
        floor = chance + (1 - chance) * (actions - 2) / actions
        estimate, effective = implied_accuracy(
            round(floor * 10_000), 10_000, actions
        )
        assert abs(estimate - chance) < 0.01, actions
        # Inverting costs precision, and the trial count says so.
        assert effective == max(1, int(10_000 / (actions / 2) ** 2))
        perfect, _ = implied_accuracy(10_000, 10_000, actions)
        assert perfect == 1.0
