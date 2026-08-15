from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.adversarial_probes import (
    count_parity,
    count_threshold,
    running_majority,
)
from experiments.brainworkshop_canonical.class_escalation import (
    Verdict,
    assess,
    escalate,
    induce_counter_machine,
)
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.rule_automata import sample_rule


def _traces(predicate, seed: int, count: int, length: int, symbols: int = 4):
    generator = torch.Generator().manual_seed(seed)
    produced = []
    for _ in range(count):
        stream = torch.randint(0, symbols, (length,), generator=generator).tolist()
        produced.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(predicate(stream)),
                eligible=tuple([True] * length),
                symbol_count=symbols,
            )
        )
    return tuple(produced)


def _error(machine, traces) -> float:
    wrong = 0
    trials = 0
    for trace in traces:
        predicted = machine.expected(list(trace.symbols))
        for position, flag in enumerate(trace.eligible):
            if not flag:
                continue
            trials += 1
            wrong += int(predicted[position] != trace.outputs[position])
    return wrong / trials if trials else 1.0


def test_a_counting_rule_finite_state_cannot_express_is_escalated_to() -> None:
    """The failure this module exists for.

    Running majority is not finite-state at any size. The Mealy inducer
    returns a twelve-state machine at chance accuracy on it; a one-state
    counter machine gets it exactly.
    """

    rule = running_majority(0, 1)
    fit = _traces(rule, 11, 112, 48)
    validation = _traces(rule, 99, 20, 48)
    report, machine = escalate(fit, validation)
    assert report.verdict is not Verdict.IDENTIFIED
    assert machine is not None
    assert machine.state_count == 1
    assert _error(machine, validation) == 0.0


def test_in_class_targets_are_identified_and_never_escalated() -> None:
    """Specificity. A wider class that gets used anyway is not a diagnosis."""

    for name, predicate in (
        ("parity", count_parity(0)),
        ("threshold-3", count_threshold(0, 3)),
    ):
        fit = _traces(predicate, 11, 112, 16)
        validation = _traces(predicate, 99, 20, 16)
        report, machine = escalate(fit, validation)
        assert report.verdict is Verdict.IDENTIFIED, name
        assert machine is None, name
        assert report.errors[-1] == 0.0, name


def test_sampled_mealy_rules_are_identified_without_escalation() -> None:
    for state_count in (1, 2, 3):
        rule = sample_rule(
            symbol_count=4, state_count=state_count, seed=7000 + 100 * state_count
        )
        assert rule is not None
        fit = _traces(rule.expected, 11, 112, 16)
        validation = _traces(rule.expected, 99, 20, 16)
        report, machine = escalate(fit, validation)
        assert report.verdict is Verdict.IDENTIFIED, state_count
        assert machine is None, state_count


def test_the_wider_class_does_not_fit_structureless_data() -> None:
    """The overfitting guard. A class that fits noise diagnoses nothing."""

    generator = torch.Generator().manual_seed(5)
    for count in (7, 28, 112):
        traces = []
        for _ in range(count):
            stream = torch.randint(0, 4, (16,), generator=generator).tolist()
            labels = torch.randint(0, 2, (16,), generator=generator).tolist()
            traces.append(
                Trace(
                    symbols=tuple(stream),
                    outputs=tuple(labels),
                    eligible=tuple([True] * 16),
                    symbol_count=4,
                )
            )
        assert induce_counter_machine(tuple(traces)) is None, count


def test_the_wider_class_does_not_claim_real_mealy_rules() -> None:
    for state_count in (3, 4, 5, 6):
        rule = sample_rule(
            symbol_count=4, state_count=state_count, seed=6000 + 100 * state_count
        )
        assert rule is not None
        assert induce_counter_machine(_traces(rule.expected, 10, 28, 16)) is None


def test_noise_produces_no_false_claim_in_either_class() -> None:
    """Under noise the stack learns nothing. It must still not lie."""

    generator = torch.Generator().manual_seed(9)

    def noisy(predicate, seed, count, length, rate):
        clean = _traces(predicate, seed, count, length)
        corrupted = []
        for trace in clean:
            flips = (torch.rand(length, generator=generator) < rate).tolist()
            corrupted.append(
                Trace(
                    symbols=trace.symbols,
                    outputs=tuple(
                        value ^ int(flip)
                        for value, flip in zip(trace.outputs, flips)
                    ),
                    eligible=trace.eligible,
                    symbol_count=trace.symbol_count,
                )
            )
        return tuple(corrupted)

    for predicate in (count_parity(0), running_majority(0, 1)):
        for rate in (0.01, 0.05):
            fit = noisy(predicate, 11, 112, 48, rate)
            validation = noisy(predicate, 99, 20, 48, rate)
            report, machine = escalate(fit, validation)
            assert report.verdict is not Verdict.IDENTIFIED
            assert machine is None


def test_a_verdict_of_identified_always_comes_with_zero_held_out_error() -> None:
    """The calibration claim, in the form a caller can rely on."""

    for predicate in (count_parity(0), count_threshold(0, 3)):
        fit = _traces(predicate, 11, 112, 16)
        validation = _traces(predicate, 99, 20, 16)
        report = assess(fit, validation)
        if report.verdict is Verdict.IDENTIFIED:
            assert report.errors[-1] == 0.0
            assert report.machine is not None
            assert _error(report.machine, validation) == 0.0


def test_a_counter_machine_prefers_ignoring_its_counter() -> None:
    """Increments are tried 0 first, so the wider class does not reach past
    finite state without cause."""

    rule = count_parity(0)
    machine = induce_counter_machine(_traces(rule, 3, 28, 16))
    assert machine is not None
    assert all(move == 0 for row in machine.increments for move in row)


def test_an_empty_or_degenerate_input_is_refused() -> None:
    assert induce_counter_machine(()) is None
    report = assess((), ())
    assert report.verdict is Verdict.NEED_MORE_DATA
    assert report.machine is None


@pytest.mark.parametrize("length", [16, 48])
def test_escalation_is_gated_on_held_out_evidence(length) -> None:
    """A wider hypothesis is accepted only if it predicts unseen episodes."""

    rule = running_majority(2, 3)
    fit = _traces(rule, 11, 112, length)
    validation = _traces(rule, 99, 20, length)
    _, machine = escalate(fit, validation)
    assert machine is not None
    assert _error(machine, validation) == 0.0
    # And on episodes drawn with a third seed, never seen by either step.
    assert _error(machine, _traces(rule, 1234, 20, length)) == 0.0
