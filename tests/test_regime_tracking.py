from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.regime_tracking import (
    changed,
    episode_errors,
    track,
)
from experiments.brainworkshop_canonical.rule_automata import sample_rule


def _episodes(predicate, seed, count, length=48, noise=0.0, symbols=4):
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
    return produced


def _rule(state_count, offset=0):
    rule = sample_rule(
        symbol_count=4, state_count=state_count, seed=7000 + 100 * state_count + offset
    )
    assert rule is not None
    return rule


@pytest.mark.parametrize("noise", [0.0, 0.05, 0.10])
def test_a_stationary_stream_is_one_regime(noise) -> None:
    """The failure mode that would make this worse than useless."""

    report = track(_episodes(_rule(3).expected, 11, 48, noise=noise))
    assert report.change_points == []
    assert len(report.segments) == 1
    assert report.fits == 1


@pytest.mark.parametrize("noise", [0.0, 0.05, 0.10])
def test_a_change_is_detected_where_it_happened(noise) -> None:
    first, second = _rule(3), _rule(4)
    stream = _episodes(first.expected, 11, 24, noise=noise) + _episodes(
        second.expected, 23, 24, noise=noise
    )
    report = track(stream)
    assert 24 in report.change_points


@pytest.mark.parametrize("noise", [0.0, 0.05, 0.10])
def test_a_returning_regime_is_recognised_rather_than_relearned(noise) -> None:
    """Where the library stops being decoration.

    Three regimes, two of them the same. An agent without recognition pays
    three fits; this pays two.
    """

    first, second = _rule(3), _rule(4)
    stream = (
        _episodes(first.expected, 11, 20, noise=noise)
        + _episodes(second.expected, 23, 20, noise=noise)
        + _episodes(first.expected, 37, 20, noise=noise)
    )
    report = track(stream)
    assert report.reuses >= 1
    assert report.fits <= 2
    assert any(segment.reused for segment in report.segments)


def test_a_structureless_stream_is_not_shredded_into_regimes() -> None:
    generator = torch.Generator().manual_seed(5)
    traces = []
    for _ in range(48):
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
    report = track(traces)
    assert report.change_points == []


def test_detection_needs_more_than_a_single_surprising_window() -> None:
    """The floor, which matters more than the threshold.

    A machine fitted at zero disagreements must not read one flipped label as
    proof the world changed.
    """

    assert not changed(1, 200, 0.0)
    assert not changed(0, 200, 0.0)
    # A window that is wrong about a third of the time is not noise.
    assert changed(70, 200, 0.02)


def test_detection_scales_with_the_noise_it_was_fitted_at() -> None:
    # Ten percent errors are unremarkable for a fit that already sees ten.
    assert not changed(20, 200, 0.10)
    # The same count is decisive for a fit that sees almost none.
    assert changed(20, 200, 0.0)


def test_episode_errors_counts_only_scored_steps() -> None:
    rule = _rule(2)
    trace = _episodes(rule.expected, 3, 1)[0]
    wrong, trials = episode_errors(rule, trace)
    assert wrong == 0
    assert trials == len(trace.symbols)
    blinded = Trace(
        symbols=trace.symbols,
        outputs=trace.outputs,
        eligible=tuple([False] * len(trace.symbols)),
        symbol_count=4,
    )
    assert episode_errors(rule, blinded) == (0, 0)


def test_an_empty_stream_tracks_nothing() -> None:
    report = track(())
    assert report.segments == []
    assert report.fits == 0
    assert report.change_points == []


def test_a_supplied_library_is_used_before_fitting() -> None:
    rule = _rule(3)
    report = track(
        _episodes(rule.expected, 11, 12), library=[rule]
    )
    assert report.fits == 0
    assert report.reuses >= 1
    assert report.segments[0].reused
