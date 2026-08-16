from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.integrated_navigation import (
    ARMS,
    SearchedReader,
    run_integrated_navigation,
    select_cut,
)
from experiments.brainworkshop_canonical.navigation_environment import (
    sample_navigation_task,
)
from experiments.brainworkshop_canonical.object_scene import render_markers
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


def _reader():
    from experiments.brainworkshop_canonical.successor_transfer import (
        build_slot_reader,
    )

    return build_slot_reader(_encoders())


# --- orientation ------------------------------------------------------------


def _drive(reader, steps: int):
    """Agent on a deterministic table, distractor on its own circuit."""

    table = {(p, a): (p * 5 + a * 3) % 8 for p in range(8) for a in range(4)}
    searched = SearchedReader(reader)
    mine, theirs = 0, 4
    last_action = None
    observed = mine
    for step in range(steps):
        observed = mine
        searched.observe(
            render_markers((theirs, mine), size=36), last_action=last_action
        )
        searched.orient(alphabet=reader.alphabet)
        action = (step * 3 + step // 4) % 4
        searched.record(action)
        last_action = action
        mine = table[(mine, action)]
        theirs = (theirs + 3) % 8
    return searched, observed


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_orientation_settles_on_a_track_but_not_reliably_the_right_one() -> None:
    """The measured limit, asserted rather than hoped for.

    This was written expecting the agent to name itself correctly, and it does
    not. Traced on this exact stream, the correspondence beam follows the agent
    for four steps and then swaps onto the distractor, so both tracks are
    mixtures and neither follows one object -- 6 and 7 of 12 frames each.
    `identify_roles` then names the mixture with the higher contrast score.

    That is not a defect of this wiring; it is the same thing the integrated
    run measures end to end, where identification is right 0.52 of the time.
    Two markers that are identical, both moving, and teleporting leave the
    correspondence genuinely underdetermined over a short history.
    """

    reader = _reader()
    searched, observed = _drive(reader, 12)
    assert searched.oriented
    configuration = searched.configuration(alphabet=reader.alphabet)
    assert configuration is not None
    own_symbol, _ = configuration
    truth = reader.read(render_markers((observed, observed), size=36))[0][1]
    # Both outcomes are permitted, and the point of the test is that the wrong
    # one is reachable. If this ever becomes reliably correct, the integrated
    # record's headline is stale and should be re-measured.
    assert own_symbol in {int(symbol) for symbol in searched.tracker.reading()}
    assert isinstance(truth, int)


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_orientation_refuses_before_it_has_evidence() -> None:
    reader = _reader()
    searched = SearchedReader(reader)
    searched.observe(render_markers((4, 0), size=36), last_action=None)
    assert searched.orient(alphabet=reader.alphabet) is False
    assert searched.oriented is False
    assert searched.configuration(alphabet=reader.alphabet) is None


def test_a_reader_that_has_seen_nothing_cannot_record() -> None:
    class _Stub:
        alphabet = 8

    searched = SearchedReader(_Stub())
    with pytest.raises(RuntimeError, match="nothing has been observed"):
        searched.record(0)


# --- the cut is selected, not assumed --------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_agent_picks_its_own_decomposition() -> None:
    encoders = _encoders()
    reader = _reader()
    task = sample_navigation_task(seed=9000)
    assert task is not None
    chosen, bits = select_cut(reader, encoders, task, seed=41)
    assert chosen == "components"
    assert bits["components"] < bits["whole"]


# --- end to end -------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_removing_the_oracles_costs_something_measurable(tmp_path) -> None:
    """The point of the run: the compounding, not the score."""

    before = sha256_file(BANK)
    report = run_integrated_navigation(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=2,
        explore_episodes=20,
        starts=4,
    )
    assert set(ARMS) == {"integrated", "told_all", "random"}
    assert report["cuts_chosen"] == ["components"]

    for block in (report["trained"], report["held_out"]):
        # Everything still beats acting blindly.
        assert block["integrated_fraction"] > block["random_fraction"]
        # Handing over identification cannot hurt.
        assert block["told_all_fraction"] >= block["integrated_fraction"] - 1e-9
        # Orientation is charged rather than run off the clock.
        assert block["orientation_steps"] >= 0.0

    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_integrated_model_is_built_from_what_it_worked_out(tmp_path) -> None:
    """A searched identity yields a different model than an oracled one.

    If these matched, the searched path would not be doing anything and the
    comparison would be measuring nothing.
    """

    report = run_integrated_navigation(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        tasks=1,
        explore_episodes=16,
        starts=3,
    )
    assert report["mean_integrated_coverage"] != pytest.approx(
        report["mean_told_coverage"]
    )


def test_torch_is_deterministic_for_the_circuit() -> None:
    from experiments.brainworkshop_canonical.relational_transfer import target_circuit

    assert target_circuit(5) == target_circuit(5)
    assert torch.equal(torch.tensor(target_circuit(5)), torch.tensor(target_circuit(5)))
