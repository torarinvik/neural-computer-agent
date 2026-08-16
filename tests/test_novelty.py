from __future__ import annotations

import pytest
import torch

from experiments.brainworkshop_canonical.novelty import (
    LIFELONG_CLIP,
    NoveltyCounts,
    SlidingWindowUCB,
)

# --- novelty as a weight vector --------------------------------------------


def test_visiting_a_place_makes_it_less_wanted() -> None:
    counts = NoveltyCounts(alphabet=4)
    before = counts.weights()
    assert float(before.min()) == pytest.approx(float(before.max()))
    for _ in range(5):
        counts.observe(2, (2, 3))
    after = counts.weights()
    assert float(after[2]) < float(after[0])
    # The task changed; nothing was recomputed but a vector.
    assert after.shape == before.shape


def test_the_two_timescales_do_different_jobs() -> None:
    """Episodic resets, lifelong does not, and that is the whole point."""

    counts = NoveltyCounts(alphabet=4)
    for _ in range(6):
        counts.observe(1, (1,))
    counts.start_episode()
    # Within this episode place one is untouched again...
    assert counts.episodic == {}
    # ...but a whole run of visits is not forgotten, so it stays the least
    # attractive place on the board.
    weights = counts.weights()
    assert float(weights[1]) < float(weights[0])


def test_the_lifelong_term_modulates_and_never_dominates() -> None:
    counts = NoveltyCounts(alphabet=3)
    for _ in range(500):
        counts.observe(0, (0,))
    weights = counts.weights()
    # An unvisited place is more attractive, but by a bounded factor.
    assert float(weights[1]) > float(weights[0])
    assert float(weights[1]) <= LIFELONG_CLIP


def _paced(distractor: bool, *, steps: int = 40) -> NoveltyCounts:
    """The agent paces between three places; optionally something else moves."""

    counts = NoveltyCounts(alphabet=8)
    for step in range(steps):
        place = step % 3
        reading = (place, 7, (step * 5) % 8) if distractor else (place, 7)
        counts.observe(place, reading)
    return counts


def test_the_ungated_view_is_degraded_by_something_else_moving() -> None:
    """The noisy television, measured rather than asserted.

    Counting novelty over whole readings works while the reading is basically
    the agent's own place. Put a distractor in the frame and no reading ever
    repeats, so a place the agent has stood on forty times still reads as
    half-new. The signal is not destroyed -- somewhere never visited has no
    readings at all and still stands out -- but the contrast that tells two
    *visited* places apart is what erodes, and that is the part exploration
    needs late.
    """

    quiet = _paced(distractor=False).weights(gated=False)
    noisy = _paced(distractor=True).weights(gated=False)
    quiet_spread = float(quiet.max()) - float(quiet.min())
    noisy_spread = float(noisy.max()) - float(noisy.min())
    assert noisy_spread < 0.7 * quiet_spread

    # Gating is what makes the two conditions indistinguishable, because the
    # distractor is not part of what the agent controls.
    gated_quiet = _paced(distractor=False).weights(gated=True)
    gated_noisy = _paced(distractor=True).weights(gated=True)
    assert torch.allclose(gated_quiet, gated_noisy)


def test_readings_are_unordered() -> None:
    counts = NoveltyCounts(alphabet=8)
    counts.observe(1, (1, 4))
    assert counts.reading_novelty((4, 1)) < 1.0
    assert counts.reading_novelty((2, 5)) == pytest.approx(1.0)


# --- the meta-controller ----------------------------------------------------


def test_the_bandit_tries_everything_before_it_prefers_anything() -> None:
    bandit = SlidingWindowUCB(("a", "b", "c"), seed=0)
    for expected in range(3):
        assert bandit.select() == expected
        bandit.update(expected, 0.0)
    assert bandit.counts() == (1, 1, 1)


def test_the_bandit_converges_on_the_arm_that_pays() -> None:
    bandit = SlidingWindowUCB((0.5, 0.8, 0.95, 0.99), window=16, epsilon=0.0, seed=1)
    for _ in range(60):
        arm = bandit.select()
        bandit.update(arm, 1.0 if arm == 2 else 0.0)
    assert bandit.counts()[2] == max(bandit.counts())
    assert bandit.means()[2] > 0.9


def test_the_window_lets_a_regime_change_be_noticed() -> None:
    """An ordinary bandit averages over a regime that has already ended."""

    bandit = SlidingWindowUCB(("early", "late"), window=8, epsilon=0.0, seed=2)
    for _ in range(30):
        arm = bandit.select()
        bandit.update(arm, 1.0 if arm == 0 else 0.0)
    assert bandit.counts()[0] > bandit.counts()[1]
    settled = bandit.counts()
    for _ in range(40):
        arm = bandit.select()
        bandit.update(arm, 0.0 if arm == 0 else 1.0)
    gained = tuple(a - b for a, b in zip(bandit.counts(), settled))
    assert gained[1] > gained[0]


def test_a_bandit_needs_arms_and_valid_updates() -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        SlidingWindowUCB(())
    bandit = SlidingWindowUCB(("a",))
    with pytest.raises(ValueError, match="out of range"):
        bandit.update(3, 1.0)


def test_weights_are_the_shape_generalised_policy_improvement_wants() -> None:
    counts = NoveltyCounts(alphabet=6)
    counts.observe(0, (0,))
    weights = counts.weights()
    assert weights.dtype == torch.float64
    assert weights.shape == (6,)
