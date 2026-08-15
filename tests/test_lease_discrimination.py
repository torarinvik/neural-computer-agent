from __future__ import annotations

import pytest

from experiments.brainworkshop_canonical.lease_discrimination import (
    assert_discriminating,
    binomial_upper_tail,
    discrimination_report,
    required_eligible_trials,
)


def test_upper_tail_matches_the_hand_computed_lease_lengths() -> None:
    # The 48-step onset lease scored 47 eligible trials per session, and one
    # of its three seeds did produce a spurious pass.
    assert binomial_upper_tail(47, 0.75, 0.8) == pytest.approx(0.228, abs=0.005)
    assert binomial_upper_tail(191, 0.75, 0.8) == pytest.approx(0.059, abs=0.005)
    assert binomial_upper_tail(383, 0.75, 0.8) == pytest.approx(0.010, abs=0.002)
    # A policy already at the gate is not a near miss.
    assert binomial_upper_tail(100, 0.8, 0.8) > 0.5
    # Certainty and impossibility.
    assert binomial_upper_tail(10, 1.0, 0.8) == pytest.approx(1.0)
    assert binomial_upper_tail(10, 0.0, 0.8) == pytest.approx(0.0)


def test_more_trials_never_help_a_near_miss() -> None:
    tails = [binomial_upper_tail(n, 0.75, 0.8) for n in (47, 95, 191, 383, 575)]
    assert tails == sorted(tails, reverse=True)


def test_required_trials_are_the_floor_the_leases_use() -> None:
    # The floor is the first length past which the tail stays under alpha.
    # 379 clears 1% on its own but 380 through 383 do not, so 379 is not it.
    assert required_eligible_trials(0.8) == 411
    assert binomial_upper_tail(379, 0.75, 0.8) <= 0.01
    assert binomial_upper_tail(383, 0.75, 0.8) > 0.01
    # A tighter alpha is never cheaper.
    assert required_eligible_trials(0.8, 0.75, 0.001) >= required_eligible_trials(0.8)
    with pytest.raises(ValueError, match="below the gate"):
        required_eligible_trials(0.8, 0.85)


def test_report_flags_the_recorded_lease_lengths() -> None:
    short = discrimination_report(47, threshold=0.8)
    long = discrimination_report(191, threshold=0.8)
    standing = discrimination_report(447, threshold=0.8)
    assert short["discriminating"] is False
    assert long["discriminating"] is False
    assert standing["discriminating"] is True
    assert standing["required_eligible_trials"] == 411
    assert float(standing["near_miss_pass_probability"]) <= 0.01
    # Clearing alpha at one length is not enough on its own.
    assert discrimination_report(379, threshold=0.8)["discriminating"] is False


def test_assert_refuses_a_campaign_that_cannot_discriminate() -> None:
    with pytest.raises(ValueError, match="too short to discriminate"):
        assert_discriminating(47, threshold=0.8)
    with pytest.raises(ValueError, match="too short to discriminate"):
        assert_discriminating(383, threshold=0.8)
    report = assert_discriminating(447, threshold=0.8)
    assert report["discriminating"] is True


def test_the_tail_matches_exact_integer_arithmetic_and_survives_big_counts() -> None:
    from math import ceil, comb

    for trials, rate, threshold in (
        (47, 0.75, 0.8),
        (191, 0.75, 0.8),
        (120, 0.3, 0.8),
        (447, 0.78, 0.8),
        (500, 0.5, 0.5),
    ):
        exact = sum(
            comb(trials, hits) * rate**hits * (1.0 - rate) ** (trials - hits)
            for hits in range(ceil(threshold * trials), trials + 1)
        )
        assert binomial_upper_tail(trials, rate, threshold) == pytest.approx(
            exact, abs=1e-12
        )
    # The binomial coefficient overflows a float well before the trial count
    # a floor against a 0.78 near miss would ask for.
    assert 0.0 < binomial_upper_tail(3000, 0.78, 0.8) < 1.0


def test_a_closer_near_miss_costs_far_more_trials() -> None:
    assert required_eligible_trials(0.8, 0.75, 0.01) == 411
    assert required_eligible_trials(0.8, 0.78, 0.01) == 2326


def test_separation_kills_the_near_miss_a_fixed_gate_let_through() -> None:
    from experiments.brainworkshop_canonical.lease_discrimination import (
        best_control,
        separation_report,
    )

    # The spurious 48-step AND: it cleared 0.8, but a 0.75 arm reproduces it
    # one time in five, so the winner was never distinguishable from a rival.
    weak = separation_report(0.812, 0.75, 48, control_label="invert_slot0")
    assert weak["separated"] is False
    # The standing onset lease: a 0.779 arm does not score 447 of 447.
    strong = separation_report(1.0, 0.779, 447, control_label="prototype_only")
    assert strong["separated"] is True
    assert float(strong["control_reproduces_winner_probability"]) < 1e-40
    assert best_control({"a": 0.5, "b": 0.78})[0] == "b"


def test_a_control_seen_under_the_gate_need_not_belong_under_it() -> None:
    from experiments.brainworkshop_canonical.lease_discrimination import (
        binomial_lower_tail,
        control_below_threshold_report,
    )

    # 0.779 over 447 trials is what a true 0.8 arm gives 14% of the time.
    weak = control_below_threshold_report(0.779, 447, threshold=0.8)
    assert weak["ruled_out_at_threshold"] is False
    assert float(weak["at_least_threshold_probability"]) == pytest.approx(
        0.141, abs=0.005
    )
    clear = control_below_threshold_report(0.729, 447, threshold=0.8)
    assert clear["ruled_out_at_threshold"] is True
    # The tails are complementary around the observed count.
    assert binomial_lower_tail(100, 0.5, 1.0) == pytest.approx(1.0)
    assert binomial_lower_tail(100, 1.0, 0.5) == pytest.approx(0.0)
