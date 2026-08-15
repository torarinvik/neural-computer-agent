"""Notice the world changed, and recognise it when it changes back.

Every learner in this session assumes a fixed target. The switching probe --
one rule for the first half of each episode and another for the second --
defeats all of them, and it is the last of the adversarial probes still
standing. It is also the one that matters most outside a benchmark: a world
that stops behaving the way it did is the normal case, not the exception.

Two capabilities are needed and they are different.

**Detection.** Knowing that the current hypothesis has stopped working, and
distinguishing that from the noise it was already tolerating. The noise-tolerant
fitter makes this possible for the first time, because it reports a *calibrated*
disagreement rate rather than demanding perfection. A regime change is a recent
disagreement rate too high to be the noise the fit already accounted for, and
that is a binomial tail test on evidence already paid for -- the same test the
leases use to refuse a near-miss, pointed at a different question.

**Recognition.** Not relearning a regime that has been seen before. This is
where the accumulation work pays off outside its own record: a library of
machines already fitted is checked before any new fit is attempted, so a world
that oscillates between two regimes costs two fits rather than one per switch.
Without it, an agent in a changing world pays full price forever and never
accumulates anything.

The honest risk in both is the same and runs in opposite directions. A detector
that fires too readily shreds a stationary stream into imaginary regimes; one
that fires too late is indistinguishable from not having it. So the measurement
that matters is not "did it detect the change" but the pair -- false alarms on a
stream that never changes, against latency on one that does -- and both are in
`test_regime_tracking.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lease_discrimination import binomial_upper_tail
from .noise_tolerant_induction import induce_noise_tolerant
from .rule_automata import RuleAutomaton

REGIME_SCHEMA = "neural-computer.regime-tracking.v1"
# Swept at 1e-3, 1e-5 and 1e-8 against 216 stationary streams and 72 changing
# ones. Tightening is free here: false alarms fall from 0.25 to 0.06 per
# 48-episode stream while detection stays at 24/24 and localisation stays
# exact, because a real regime change is not a marginal deviation -- the
# machine that fitted the old world is simply wrong about the new one.
DETECTION_ALPHA = 1e-8
MIN_WINDOW = 4
FLOOR_RATE = 0.02


@dataclass
class Segment:
    """One stretch of episodes explained by one machine."""

    start: int
    end: int
    machine: RuleAutomaton | None
    error_rate: float
    reused: bool

    def payload(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "states": None if self.machine is None else self.machine.state_count,
            "error_rate": self.error_rate,
            "reused": self.reused,
        }


@dataclass
class TrackReport:
    """What the tracker did, and what it spent doing it."""

    segments: list[Segment] = field(default_factory=list)
    fits: int = 0
    reuses: int = 0
    change_points: list[int] = field(default_factory=list)

    def payload(self) -> dict[str, object]:
        return {
            "schema": REGIME_SCHEMA,
            "segments": [item.payload() for item in self.segments],
            "fits": self.fits,
            "reuses": self.reuses,
            "change_points": list(self.change_points),
        }


def episode_errors(machine: RuleAutomaton, trace) -> tuple[int, int]:
    """Disagreements and scored steps for one episode."""

    predicted = machine.expected(list(trace.symbols))
    wrong = 0
    trials = 0
    for position, flag in enumerate(trace.eligible):
        if not flag:
            continue
        trials += 1
        wrong += int(predicted[position] != trace.outputs[position])
    return wrong, trials


def changed(
    wrong: int,
    trials: int,
    fitted_rate: float,
    *,
    alpha: float = DETECTION_ALPHA,
    floor: float = FLOOR_RATE,
) -> bool:
    """Is this window's disagreement too high to be the noise we fitted?

    The floor matters more than the alpha. A machine fitted at zero
    disagreements would otherwise treat a single flipped label as proof the
    world changed, so the rate the test compares against never drops below
    something a real verifier could produce on its own.
    """

    if trials <= 0:
        return False
    observed = wrong / trials
    reference = max(fitted_rate, floor)
    if observed <= reference:
        return False
    return binomial_upper_tail(trials, reference, observed) <= alpha


def track(
    traces,
    *,
    window: int = MIN_WINDOW,
    alpha: float = DETECTION_ALPHA,
    library: list[RuleAutomaton] | None = None,
    reuse_slack: float = 1.5,
    **fit_kwargs,
) -> TrackReport:
    """Walk a stream of episodes, refitting only when the world stops fitting.

    A refit is the expensive thing, so it is the last resort: when the current
    machine fails, the library is asked first, and only a genuinely new regime
    costs a fit.

    Two thresholds here are *relative to the noise*, and both were fixed
    constants first, and both were wrong.

    A library machine is accepted when its error is within `reuse_slack` of
    the noise already being seen. With a fixed 0.05 it could never accept a
    correct machine on a stream carrying 10% label noise, and reuse fell to
    zero on exactly the streams that needed it most -- measured at 0/8.

    And the rate a window is compared against is the running error over the
    *whole current segment*, not the error on the short window the machine was
    fitted to. Fitting on four episodes underestimates the noise, which made
    later windows look anomalous and produced up to 0.81 spurious change points
    per stationary stream.
    """

    episodes = tuple(traces)
    report = TrackReport()
    held: list[RuleAutomaton] = list(library) if library else []
    if not episodes:
        return report

    index = 0
    noise_estimate = FLOOR_RATE
    while index < len(episodes):
        # Fit on a small window, then run until it stops working.
        head = episodes[index : index + window]
        machine = None
        reused = False
        allowance = max(FLOOR_RATE, noise_estimate * reuse_slack)
        for candidate in held:
            wrong = trials = 0
            for trace in head:
                bad, seen = episode_errors(candidate, trace)
                wrong += bad
                trials += seen
            if trials and wrong / trials <= allowance:
                machine, reused = candidate, True
                break
        if machine is None:
            fit = induce_noise_tolerant(head, **fit_kwargs)
            report.fits += 1
            if fit is None:
                report.segments.append(Segment(index, len(episodes), None, 1.0, False))
                break
            machine = fit.machine
            if all(item.digest() != machine.digest() for item in held):
                held.append(machine)
        else:
            report.reuses += 1

        wrong = trials = 0
        for trace in head:
            bad, seen = episode_errors(machine, trace)
            wrong += bad
            trials += seen
        fitted_rate = wrong / trials if trials else 0.0

        cursor = index + len(head)
        running_wrong, running_seen = wrong, trials
        while cursor < len(episodes):
            bad = seen = 0
            for trace in episodes[cursor : cursor + window]:
                one, two = episode_errors(machine, trace)
                bad += one
                seen += two
            # Compare against everything this machine has explained so far,
            # not against the four episodes it was fitted on.
            reference = running_wrong / running_seen if running_seen else fitted_rate
            if changed(bad, seen, reference, alpha=alpha):
                report.change_points.append(cursor)
                break
            running_wrong += bad
            running_seen += seen
            cursor += window
        cursor = min(cursor, len(episodes))
        total_wrong = total_seen = 0
        for trace in episodes[index:cursor]:
            one, two = episode_errors(machine, trace)
            total_wrong += one
            total_seen += two
        segment_rate = total_wrong / total_seen if total_seen else 1.0
        noise_estimate = max(FLOOR_RATE, segment_rate)
        report.segments.append(
            Segment(
                start=index,
                end=cursor,
                machine=machine,
                error_rate=segment_rate,
                reused=reused,
            )
        )
        if cursor <= index:
            break
        index = cursor
    return report
