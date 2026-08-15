"""How many eligible trials a fixed accuracy gate needs to mean anything.

A lease gates on `accuracy >= threshold`. On these tasks the strongest
wrong answer is not chance: a single-family policy sits at the rule's base
rate, a little below the gate. Over few eligible trials that policy crosses
the gate often, so a pass carries almost no evidence. This module states the
floor as arithmetic instead of taste.

Read `required_eligible_trials(0.8, 0.75, 0.01) == 411` as: from 411 eligible
trials on, a policy whose true accuracy is 0.75 always has at most a 1% chance
of reaching 0.8 by luck. At 47 trials that chance is 22.8%, which is why the
48-step onset lease was rejected on a control rather than on the winner.
"""

from __future__ import annotations

from functools import cache
from math import ceil, exp, fsum, lgamma, log

# A pass must be at most this likely to come from a policy whose true
# accuracy is `threshold - NEAR_MISS_MARGIN`.
DISCRIMINATION_ALPHA = 0.01
NEAR_MISS_MARGIN = 0.05


def binomial_upper_tail(trials: int, rate: float, threshold: float) -> float:
    """P(a `rate` policy scores at least `threshold` over `trials` trials)."""

    if trials < 1:
        raise ValueError("a discrimination floor needs at least one trial")
    if not 0.0 <= rate <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("rate and threshold are probabilities")
    needed = ceil(threshold * trials)
    if needed > trials:
        return 0.0
    if rate <= 0.0:
        return 1.0 if needed <= 0 else 0.0
    if rate >= 1.0:
        return 1.0 if needed <= trials else 0.0
    # In log space: the binomial coefficient alone overflows a float long
    # before the trial counts a tight floor asks for.
    log_rate = log(rate)
    log_miss = log(1.0 - rate)
    log_trials = lgamma(trials + 1)
    return fsum(
        exp(
            log_trials
            - lgamma(hits + 1)
            - lgamma(trials - hits + 1)
            + hits * log_rate
            + (trials - hits) * log_miss
        )
        for hits in range(needed, trials + 1)
    )


# The pass count is `ceil(threshold * trials)`, so the tail does not fall
# monotonically: 379 trials clear a 1% floor while 380 through 383 do not.
# The floor is therefore the first length past which it stays cleared, and a
# campaign must also clear it at its own exact length.
STABILITY_WINDOW = 200


@cache
def required_eligible_trials(
    threshold: float,
    near_miss_rate: float | None = None,
    alpha: float = DISCRIMINATION_ALPHA,
) -> int:
    """First trial count past which a near-miss policy always rarely passes."""

    rate = threshold - NEAR_MISS_MARGIN if near_miss_rate is None else near_miss_rate
    if rate >= threshold:
        raise ValueError("the near-miss rate must sit below the gate")
    trials = 1
    while any(
        binomial_upper_tail(length, rate, threshold) > alpha
        for length in range(trials, trials + STABILITY_WINDOW)
    ):
        trials += 1
    return trials


def discrimination_report(
    eligible_trials: int,
    *,
    threshold: float,
    near_miss_rate: float | None = None,
    alpha: float = DISCRIMINATION_ALPHA,
) -> dict[str, float | int | bool]:
    """Record what one lifetime's trial count buys against a near-miss."""

    rate = threshold - NEAR_MISS_MARGIN if near_miss_rate is None else near_miss_rate
    required = required_eligible_trials(threshold, rate, alpha)
    tail = binomial_upper_tail(eligible_trials, rate, threshold)
    return {
        "schema": "neural-computer.lease-discrimination.v1",
        "eligible_trials": int(eligible_trials),
        "threshold": float(threshold),
        "near_miss_rate": float(rate),
        "alpha": float(alpha),
        "required_eligible_trials": required,
        "near_miss_pass_probability": tail,
        "discriminating": bool(eligible_trials >= required and tail <= alpha),
    }


def assert_discriminating(
    eligible_trials: int,
    *,
    threshold: float,
    near_miss_rate: float | None = None,
    alpha: float = DISCRIMINATION_ALPHA,
) -> dict[str, float | int | bool]:
    """Fail closed before a campaign spends an unused seed block too cheaply."""

    report = discrimination_report(
        eligible_trials,
        threshold=threshold,
        near_miss_rate=near_miss_rate,
        alpha=alpha,
    )
    if not report["discriminating"]:
        raise ValueError(
            "episode is too short to discriminate: "
            f"{eligible_trials} eligible trials, "
            f"{report['required_eligible_trials']} required; a "
            f"{report['near_miss_rate']} policy passes "
            f"{float(report['near_miss_pass_probability']):.3f} of the time"
        )
    return report


def binomial_lower_tail(trials: int, rate: float, observed_rate: float) -> float:
    """P(a `rate` policy scores at most `observed_rate` over `trials` trials)."""

    if trials < 1:
        raise ValueError("a discrimination floor needs at least one trial")
    if not 0.0 <= rate <= 1.0 or not 0.0 <= observed_rate <= 1.0:
        raise ValueError("rate and observed rate are probabilities")
    hits = round(observed_rate * trials)
    if hits >= trials:
        return 1.0
    if rate <= 0.0:
        return 1.0
    if rate >= 1.0:
        return 0.0
    log_rate = log(rate)
    log_miss = log(1.0 - rate)
    log_trials = lgamma(trials + 1)
    return fsum(
        exp(
            log_trials
            - lgamma(hits_seen + 1)
            - lgamma(trials - hits_seen + 1)
            + hits_seen * log_rate
            + (trials - hits_seen) * log_miss
        )
        for hits_seen in range(hits + 1)
    )


def control_below_threshold_report(
    control_accuracy: float,
    eligible_trials: int,
    *,
    threshold: float,
    control_label: str = "",
    alpha: float = DISCRIMINATION_ALPHA,
) -> dict[str, float | int | bool | str]:
    """Can this arm's true rate be ruled out as sitting at or above the gate?

    Observing a control under the gate is not the same as showing it belongs
    under the gate. An arm seen at 0.779 over 447 trials is exactly what a
    true 0.8 arm produces 14% of the time, so a claim of the form "no single
    family suffices" is weak there even though the observed value passed.
    """

    tail = binomial_lower_tail(eligible_trials, threshold, control_accuracy)
    return {
        "schema": "neural-computer.control-below-threshold.v1",
        "control_label": control_label,
        "control_accuracy": float(control_accuracy),
        "threshold": float(threshold),
        "eligible_trials": int(eligible_trials),
        "alpha": float(alpha),
        "at_least_threshold_probability": tail,
        "ruled_out_at_threshold": bool(tail <= alpha),
    }


def separation_report(
    winner_accuracy: float,
    control_accuracy: float,
    eligible_trials: int,
    *,
    control_label: str = "",
    alpha: float = DISCRIMINATION_ALPHA,
) -> dict[str, float | int | bool | str]:
    """Could the best rejected arm have produced the winner's run by luck?

    This is the gate that matters. A fixed threshold asks whether two arms
    land on opposite sides of a constant, which says little when one of them
    sits just under it; this asks directly how often a policy at the best
    control's own observed rate would score what the winner scored.
    """

    tail = binomial_upper_tail(eligible_trials, control_accuracy, winner_accuracy)
    return {
        "schema": "neural-computer.lease-separation.v1",
        "winner_accuracy": float(winner_accuracy),
        "control_accuracy": float(control_accuracy),
        "control_label": control_label,
        "eligible_trials": int(eligible_trials),
        "margin": float(winner_accuracy) - float(control_accuracy),
        "alpha": float(alpha),
        "control_reproduces_winner_probability": tail,
        "separated": bool(tail <= alpha),
    }


def best_control(controls: dict[str, float]) -> tuple[str, float]:
    """The strongest rejected arm, which is the one worth ruling out."""

    if not controls:
        raise ValueError("separation needs at least one control arm")
    label = max(controls, key=lambda name: controls[name])
    return label, float(controls[label])


def assert_separated(
    winner_accuracy: float,
    controls: dict[str, float],
    eligible_trials: int,
    *,
    alpha: float = DISCRIMINATION_ALPHA,
) -> dict[str, float | int | bool | str]:
    """Fail closed when the winner is not distinguishable from its best rival."""

    label, rate = best_control(controls)
    report = separation_report(
        winner_accuracy,
        rate,
        eligible_trials,
        control_label=label,
        alpha=alpha,
    )
    if not report["separated"]:
        raise ValueError(
            "winner is not separated from its best control: "
            f"{winner_accuracy:.3f} against {label} at {rate:.3f} over "
            f"{eligible_trials} trials reproduces with probability "
            f"{float(report['control_reproduces_winner_probability']):.3f}"
        )
    return report
