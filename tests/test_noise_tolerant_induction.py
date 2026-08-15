from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.adversarial_probes import (
    count_parity,
    count_threshold,
    running_majority,
)
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.noise_tolerant_induction import (
    balanced_accuracy,
    induce_noise_tolerant,
    induce_validated,
)
from experiments.brainworkshop_canonical.prefix_denoising import (
    denoise,
    induce_denoised,
)
from experiments.brainworkshop_canonical.rule_automata import sample_rule


def _traces(predicate, seed, count, length, noise=0.0, symbols=4):
    generator = torch.Generator().manual_seed(seed)
    produced = []
    for _ in range(count):
        stream = torch.randint(0, symbols, (length,), generator=generator).tolist()
        labels = list(predicate(stream))
        if noise > 0:
            flips = (torch.rand(length, generator=generator) < noise).tolist()
            labels = [value ^ int(flip) for value, flip in zip(labels, flips)]
        produced.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(labels),
                eligible=tuple([True] * length),
                symbol_count=symbols,
            )
        )
    return tuple(produced)


def _clean_accuracy(machine, predicate, seed, symbols=4):
    hits = trials = 0
    for trace in _traces(predicate, seed, 20, 48, 0.0, symbols):
        predicted = machine.expected(list(trace.symbols))
        for index, flag in enumerate(trace.eligible):
            if flag:
                trials += 1
                hits += int(predicted[index] == trace.outputs[index])
    return hits / trials


@pytest.mark.parametrize("noise", [0.0, 0.05, 0.10, 0.20])
def test_a_rule_is_recovered_exactly_through_heavy_label_noise(noise) -> None:
    """The claim, at the noise levels it was calibrated at.

    Scored against the *clean* rule, so a hypothesis that merely reproduces
    the corrupted evidence does not pass.
    """

    for state_count in (2, 4):
        rule = sample_rule(
            symbol_count=4, state_count=state_count, seed=7000 + 100 * state_count
        )
        assert rule is not None
        fit = induce_noise_tolerant(
            _traces(rule.expected, 11, 112, 48, noise)
        )
        assert fit is not None
        assert _clean_accuracy(fit.machine, rule.expected, 99) == 1.0
        assert fit.machine.state_count == state_count


def test_the_reported_error_rate_estimates_the_noise() -> None:
    """The fit's own error is a usable self-report, not decoration."""

    rule = sample_rule(symbol_count=4, state_count=3, seed=7300)
    assert rule is not None
    for noise in (0.02, 0.05, 0.10):
        fit = induce_noise_tolerant(_traces(rule.expected, 11, 112, 48, noise))
        assert fit is not None
        assert abs(fit.error_rate - noise) < 0.02, noise


def test_structureless_data_is_not_dressed_up_as_a_rule() -> None:
    generator = torch.Generator().manual_seed(5)
    traces = []
    for _ in range(112):
        stream = torch.randint(0, 4, (48,), generator=generator).tolist()
        labels = torch.randint(0, 2, (48,), generator=generator).tolist()
        traces.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(labels),
                eligible=tuple([True] * 48),
                symbol_count=4,
            )
        )
    fit = induce_noise_tolerant(tuple(traces))
    assert fit is not None
    # It returns something, and says that something explains nothing.
    assert fit.error_rate > 0.4


def test_an_out_of_class_target_is_not_claimed_exactly() -> None:
    fit = induce_noise_tolerant(_traces(running_majority(0, 1), 11, 112, 48))
    assert fit is not None
    assert _clean_accuracy(fit.machine, running_majority(0, 1), 99) < 0.95


def test_the_search_does_not_depend_on_its_random_restarts() -> None:
    rule = sample_rule(symbol_count=4, state_count=4, seed=7400)
    assert rule is not None
    evidence = _traces(rule.expected, 11, 112, 48, 0.10)
    for seed in (0, 1, 2):
        fit = induce_noise_tolerant(evidence, seed=seed)
        assert fit is not None
        assert _clean_accuracy(fit.machine, rule.expected, 99) == 1.0


def test_more_evidence_never_hurts() -> None:
    rule = sample_rule(symbol_count=4, state_count=3, seed=7300)
    assert rule is not None
    scores = [
        _clean_accuracy(
            induce_noise_tolerant(_traces(rule.expected, 11, count, 48, 0.10)).machine,
            rule.expected,
            99,
        )
        for count in (28, 112)
    ]
    assert scores[-1] >= scores[0]
    assert scores[-1] == 1.0


def test_degenerate_input_is_refused() -> None:
    assert induce_noise_tolerant(()) is None
    assert induce_noise_tolerant((Trace((), (), (), 4),)) is None


def test_prefix_voting_is_kept_and_is_measurably_worse() -> None:
    """The approach that failed, retained so the comparison is reproducible.

    Voting per prefix cell drops the deep evidence identification needs. At
    zero noise it loses to the method that keeps it, which is why the unit had
    to be the state instead.
    """

    rule = count_parity(0)
    evidence = _traces(rule, 11, 112, 16)
    report = denoise(evidence)
    assert report.kept_fraction < 0.3
    voted, _ = induce_denoised(evidence)
    climbed = induce_noise_tolerant(evidence)
    assert climbed is not None
    assert _clean_accuracy(climbed.machine, rule, 99) == 1.0
    if voted is not None:
        assert _clean_accuracy(voted, rule, 99) < 1.0


def test_denoise_rejects_impossible_settings() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        denoise((), min_count=0)
    with pytest.raises(ValueError, match="room on both sides"):
        denoise((), margin=0.5)
    assert denoise(()).traces == ()


def _random_labels(p_one, seed, count=112, length=48, symbols=4):
    generator = torch.Generator().manual_seed(seed)
    produced = []
    for _ in range(count):
        stream = torch.randint(0, symbols, (length,), generator=generator).tolist()
        labels = (torch.rand(length, generator=generator) < p_one).long().tolist()
        produced.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(labels),
                eligible=tuple([True] * length),
                symbol_count=symbols,
            )
        )
    return tuple(produced)


def test_rare_positives_defeat_the_plain_objective() -> None:
    """The failure the noise-tolerance record named and did not fix.

    A threshold rule that fires late in a short episode presses on 2% of steps.
    Fewest-disagreements then prefers the machine that never presses: 0.963
    accuracy, and no capability whatsoever.
    """

    rule = count_threshold(0, 6)
    evidence = _traces(rule, 11, 112, 16)
    plain = induce_noise_tolerant(evidence, balanced=False)
    assert plain is not None
    assert plain.machine.state_count == 1
    # Chance, stated as chance rather than hidden behind an accuracy.
    assert balanced_accuracy(plain.machine, _traces(rule, 99, 20, 16)) == pytest.approx(
        0.5, abs=0.02
    )


def test_balancing_the_classes_recovers_them_and_costs_nothing_when_balanced() -> None:
    rule = count_threshold(0, 6)
    balanced = induce_noise_tolerant(_traces(rule, 11, 112, 16), balanced=True)
    assert balanced is not None
    assert balanced_accuracy(balanced.machine, _traces(rule, 99, 20, 16)) > 0.85

    # And on a balanced task the two objectives agree exactly, because equal
    # class frequencies give equal weights.
    for state_count in (3, 5):
        sampled = sample_rule(
            symbol_count=4, state_count=state_count, seed=7000 + 100 * state_count
        )
        assert sampled is not None
        evidence = _traces(sampled.expected, 11, 112, 48, 0.10)
        both = [
            induce_noise_tolerant(evidence, balanced=flag) for flag in (False, True)
        ]
        assert all(fit is not None for fit in both)
        assert both[0].machine.digest() == both[1].machine.digest()


def test_balancing_alone_buys_structure_in_skewed_noise() -> None:
    """Why balancing is not simply switched on."""

    evidence = _random_labels(0.05, 5)
    balanced = induce_noise_tolerant(evidence, balanced=True)
    plain = induce_noise_tolerant(evidence, balanced=False)
    assert plain is not None and balanced is not None
    assert plain.machine.state_count == 1
    assert balanced.machine.state_count > 1


def test_held_out_evidence_picks_the_right_objective_each_time() -> None:
    """Neither objective is a prior worth committing to, so nothing is."""

    rule = count_threshold(0, 6)
    chosen = induce_validated(
        _traces(rule, 11, 112, 16), _traces(rule, 55, 28, 16)
    )
    assert chosen is not None
    assert balanced_accuracy(chosen.machine, _traces(rule, 99, 20, 16)) > 0.85

    # Skewed noise: the balanced fit is refused and the honest one kept.
    kept = induce_validated(_random_labels(0.05, 5), _random_labels(0.05, 77))
    assert kept is not None
    assert kept.machine.state_count == 1

    # A balanced task is unaffected.
    sampled = sample_rule(symbol_count=4, state_count=4, seed=7400)
    assert sampled is not None
    same = induce_validated(
        _traces(sampled.expected, 11, 112, 48, 0.10),
        _traces(sampled.expected, 55, 28, 48, 0.10),
    )
    assert same is not None
    assert _clean_accuracy(same.machine, sampled.expected, 99) == 1.0


def test_choosing_needs_something_to_choose_on() -> None:
    with pytest.raises(ValueError, match="held-out evidence"):
        induce_validated(_traces(count_threshold(0, 3), 11, 8, 16), ())
