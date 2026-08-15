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
