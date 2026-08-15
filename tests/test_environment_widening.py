from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.environment_widening import (
    config_factory,
    run_widening,
    widened_pool,
)
from experiments.brainworkshop_canonical.prototype_templates import (
    cluster_events,
    estimated_tolerance,
)
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopVerifier,
)
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _events(spread: float, *, groups: int = 4, per_group: int = 40, width: int = 8):
    """Well-separated groups with a controllable within-group spread."""

    generator = torch.Generator().manual_seed(3)
    centres = torch.eye(groups, width) * 10.0
    return torch.cat(
        [
            centres[index] + spread * torch.randn(per_group, width, generator=generator)
            for index in range(groups)
        ]
    )


# --- the estimator ---------------------------------------------------------


def test_the_tolerance_is_measured_rather_than_assumed() -> None:
    """A fixed 0.5 is right only in a room where symbols render identically."""

    tight = estimated_tolerance(_events(0.01))
    loose = estimated_tolerance(_events(0.5))
    assert tight is not None and loose is not None
    assert loose > tight
    for spread, events in ((0.01, _events(0.01)), (0.5, _events(0.5))):
        tolerance = estimated_tolerance(events)
        assert (
            int(cluster_events(events, tolerance=tolerance, maximum_clusters=32).shape[0])
            == 4
        ), spread


def test_a_fixed_tolerance_shatters_the_alphabet_the_estimator_recovers() -> None:
    """The bug, kept as a comparison rather than described."""

    events = _events(0.5)
    shattered = cluster_events(events, tolerance=0.5, maximum_clusters=32)
    recovered = cluster_events(
        events, tolerance=estimated_tolerance(events), maximum_clusters=32
    )
    assert shattered.shape[0] > 4
    assert recovered.shape[0] == 4


def test_overlapping_stimuli_are_refused_rather_than_named() -> None:
    """The failure that must never be silent.

    Naming the wrong alphabet corrupts everything downstream and nothing later
    can detect it, so the estimator has to be able to say it cannot tell.
    """

    generator = torch.Generator().manual_seed(5)
    formless = torch.randn(160, 8, generator=generator)
    assert estimated_tolerance(formless) is None
    # And a spread wide enough that the groups actually touch is refused too:
    # at 1.0 the largest within-group distance is 7.97 and the smallest
    # between-group distance is 9.84, so there is no honest boundary left.
    assert estimated_tolerance(_events(1.0)) is None


def test_the_estimator_needs_something_to_estimate_from() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        estimated_tolerance(torch.zeros(1, 8))


# --- the environment -------------------------------------------------------


def test_frame_noise_is_drawn_per_observation_not_per_symbol() -> None:
    """Otherwise the same symbol would still render identically every time."""

    rule = sample_rule(symbol_count=4, state_count=2, seed=1)
    assert rule is not None

    def frames(noise: float):
        config = RenderedBrainWorkshopConfig(
            n_back=1,
            steps=24,
            streams=("vision",),
            symbol_count=4,
            match_rule="automaton",
            rule=rule,
            frame_noise=noise,
        ).validate()
        verifier = RenderedBrainWorkshopVerifier(config, seed=5)
        seen = []
        while not verifier.done:
            seen.append(verifier.observation().vision)
            verifier.score(torch.zeros(1, dtype=torch.long))
        return seen

    clean, noisy = frames(0.0), frames(0.12)
    repeats = [
        (a, b)
        for index, a in enumerate(clean)
        for b in clean[index + 1 :]
        if torch.equal(a, b)
    ]
    assert repeats, "the clean stream should repeat symbols exactly"
    assert not any(
        torch.equal(a, b)
        for index, a in enumerate(noisy)
        for b in noisy[index + 1 :]
    )


def test_turning_on_noise_does_not_change_which_symbols_appear() -> None:
    """Otherwise a noisy run and a clean run would not be comparable."""

    rule = sample_rule(symbol_count=4, state_count=3, seed=2)
    assert rule is not None
    build = config_factory(0.0), config_factory(0.1)
    streams = []
    for factory in build:
        verifier = RenderedBrainWorkshopVerifier(factory(rule, 32), seed=9)
        streams.append(verifier._symbols["vision"].clone())
        del verifier
    assert torch.equal(streams[0], streams[1])


def test_a_widened_pool_spans_the_complexity_axis() -> None:
    pool = widened_pool(6, pool_size=4)
    assert all(rule.symbol_count == 6 for rule in pool)
    assert len({rule.state_count for rule in pool}) == 4


# --- end to end ------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_agent_survives_a_wider_alphabet_and_says_when_it_cannot(
    tmp_path,
) -> None:
    before = sha256_file(BANK)
    report = run_widening(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        stream_length=6,
        pool_size=3,
        alphabets=(8,),
        noise_levels=(0.0, 0.10, 0.30),
    )
    rows = {row["noise"]: row for row in report["rows"]}

    # A wider alphabet is not what breaks it.
    assert rows[0.0]["discovered_alphabet"] == 8
    assert rows[0.0]["solved"] == rows[0.0]["tasks"]
    assert rows[0.0]["acquisition_ratio"] < 1.0

    # Nor is noise it can still read through.
    assert rows[0.10]["discovered_alphabet"] == 8
    assert rows[0.10]["solved"] == rows[0.10]["tasks"]

    # And where it genuinely cannot, it refuses instead of inventing letters.
    assert rows[0.30]["discovered_alphabet"] is None
    assert rows[0.30]["failure"]

    assert report["false_recognitions"] == 0
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
