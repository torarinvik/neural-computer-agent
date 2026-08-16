from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.learned_decomposition import (
    _canonical_order,
    _isolated,
    candidate_cuts,
    measure_cut,
    run_learned_decomposition,
    wander,
)
from experiments.brainworkshop_canonical.navigation_environment import (
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.object_scene import render_scene
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _encoders():
    from experiments.brainworkshop_canonical.controller_pretraining import (
        load_temporal_controller_artifact,
    )
    from experiments.brainworkshop_canonical.current_symbol_acquire import (
        FRONTEND_SEED,
        _machine,
        curated_frontend,
    )

    payload = load_temporal_controller_artifact(CONTROLLER)
    return curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )


# --- the candidates ---------------------------------------------------------


def test_every_candidate_hands_the_encoder_the_same_shaped_frame() -> None:
    """No cut may win by presenting a differently sized input."""

    frame = render_scene(1, 6)
    for cut in candidate_cuts():
        parts = cut(frame)
        assert parts
        for _, isolated in parts:
            assert isolated.shape == frame.shape


def test_the_candidate_family_spans_too_coarse_and_too_fine() -> None:
    names = {cut.name for cut in candidate_cuts()}
    assert {"whole", "components", "cells", "scatter"} <= names
    counts = {cut.name: len(cut(render_scene(0, 5))) for cut in candidate_cuts()}
    assert counts["whole"] == 1
    assert counts["components"] == 2
    assert counts["cells"] == 9


def test_isolating_a_region_keeps_the_background_identical() -> None:
    frame = render_scene(0, 8 - 1)
    size = int(frame.shape[-1])
    nothing = torch.zeros(size, size, dtype=torch.bool)
    empty = _isolated(frame, nothing)
    everything = _isolated(frame, ~nothing)
    assert torch.equal(everything, frame)
    # An empty region is the bare grid, not a black square.
    assert float(empty.max()) == pytest.approx(0.12)


# --- canonicalisation -------------------------------------------------------


def test_part_order_is_canonicalised_by_motion() -> None:
    """Slot order is positional, so index means nothing across episodes.

    Pooling tables without this mixes the agent and the goal under one key and
    charges the difference as error -- measured, that was the whole of the
    component cut's error term.
    """

    traces = [[7, 1], [7, 4], [7, 2]]
    assert _canonical_order(traces) == [[1, 7], [4, 7], [2, 7]]
    # Already in motion order, so it is left alone.
    assert _canonical_order([[1, 7], [4, 7]]) == [[1, 7], [4, 7]]
    # A single part has no ordering question.
    assert _canonical_order([[3], [5]]) == [[3], [5]]


# --- the measurement --------------------------------------------------------


def _scored(encoders, episodes):
    return {
        row["cut"]: row
        for row in (measure_cut(cut, encoders, episodes) for cut in candidate_cuts())
        if row["status"] == "scored"
    }


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_cheapest_cut_is_the_one_nobody_specified() -> None:
    """Components is not assumed, it is selected -- against a coarser cut, two
    finer ones and a partition that respects nothing."""

    encoders = _encoders()
    task = sample_navigation_task(seed=9000)
    assert task is not None
    episodes = [
        wander(task, start=start, goal=(start * 3 + 1) % 8, steps=30, seed=41 + start)
        for start in range(3)
    ]
    scored = _scored(encoders, episodes)
    assert "components" in scored and "whole" in scored
    best = min(scored.values(), key=lambda row: row["total_bits"])
    assert best["cut"] == "components"

    # Each failure direction is punished by a different term.
    assert scored["components"]["alphabet"] < scored["whole"]["alphabet"]
    assert scored["cells"]["error_bits"] > scored["components"]["error_bits"]
    # Reading the scene whole cannot even predict itself, because two markers
    # of one colour make "agent at a, goal at g" the same picture as its swap.
    assert scored["whole"]["error_bits"] > 0.0
    # A cut with the right number of pieces and no structure is not enough.
    assert scored["scatter"]["total_bits"] > scored["components"]["total_bits"]


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_decomposition_is_refused_when_only_one_thing_varies() -> None:
    """The criterion is not a preference for more parts.

    Hold the goal still across every episode and one of the two markers never
    changes, so there is nothing to decompose: the second part costs bits and
    earns none, and reading the scene whole is correctly the cheaper answer.
    This is the same fact the object-navigation record measured from the other
    side, where the scene agent did fine on the goals it trained on.
    """

    encoders = _encoders()
    task = sample_navigation_task(seed=9000)
    assert task is not None
    episodes = [
        wander(task, start=start, goal=5, steps=30, seed=41 + start)
        for start in range(3)
    ]
    scored = _scored(encoders, episodes)
    assert scored["whole"]["total_bits"] < scored["components"]["total_bits"]


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_run_selects_components_and_leaves_the_bank_alone(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_learned_decomposition(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=2,
        episodes=3,
    )
    assert report["components_chosen"] == report["tasks"]
    assert report["cuts"]["components"]["total_bits"] < (
        report["cuts"]["whole"]["total_bits"]
    )
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
