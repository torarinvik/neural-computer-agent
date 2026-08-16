from __future__ import annotations

from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.compositional_transfer import (
    build_stream,
    in_cluster_symbols,
    run_transfer,
)
from experiments.brainworkshop_canonical.hierarchical_transfer import (
    build_layers,
    run_hierarchy,
)
from experiments.brainworkshop_canonical.integrated_agent import noisy_feedback
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def test_the_stream_puts_the_parts_before_the_wholes() -> None:
    stream, learned, chosen = build_stream(seed=41, composites=4)
    kinds = [row[3] for row in stream]
    assert set(kinds) == {"primitive", "composite"}
    assert kinds.index("composite") > max(
        index for index, kind in enumerate(kinds) if kind == "primitive"
    )
    # A composite the agent has never met is not one of the primitives.
    primitives = {rule.digest() for rule in learned}
    assert all(rule.automaton.digest() not in primitives for rule in chosen)


def test_the_disjoint_stream_shares_no_parts_with_what_is_learned() -> None:
    """The control that separates composition from coincidence."""

    _, learned, _ = build_stream(seed=41, composites=4)
    _, apart_learned, apart_chosen = build_stream(
        seed=41, composites=4, disjoint=True
    )
    # The agent still learns the same primitives...
    assert {rule.digest() for rule in learned} == {
        rule.digest() for rule in apart_learned
    }
    # ...and the composites are built from a pool it never sees.
    known = {rule.digest() for rule in learned}
    assert all(
        not (set(rule.part_digests()) & known) for rule in apart_chosen
    )


def test_no_task_in_the_stream_is_clearable_by_a_constant_policy() -> None:
    """Otherwise composition would be credited with doing nothing."""

    from experiments.brainworkshop_canonical.rule_automata import positive_rate

    _, _, chosen = build_stream(seed=41, composites=6)
    for rule in chosen:
        rate = positive_rate(rule.automaton, seed=41)
        assert max(rate, 1.0 - rate) < 0.8


def test_rewriting_a_rule_in_cluster_symbols_preserves_its_behaviour() -> None:
    """The scoring-side check that a first version of this record got wrong.

    An induced machine speaks cluster indices and a sampled rule speaks the
    verifier's, so comparing their canonical digests directly reports a
    mismatch every time -- which is what happened, while the slots being
    doubted were correct.
    """

    rule = sample_rule(symbol_count=4, state_count=3, seed=6300)
    assert rule is not None
    permutation = (2, 0, 3, 1)
    rewritten = in_cluster_symbols(rule, permutation)
    stream = [0, 1, 2, 3, 3, 2, 1, 0, 2, 2, 0, 3]
    assert rewritten.expected([permutation[s] for s in stream]) == rule.expected(stream)
    assert rewritten.digest() != rule.digest()
    with pytest.raises(ValueError, match="permutation"):
        in_cluster_symbols(rule, (0, 0, 1, 2))


def test_noisy_feedback_flips_labels_without_touching_the_stimuli() -> None:
    from experiments.brainworkshop_canonical.identification_ceiling import Trace

    trace = Trace((0, 1, 2, 3) * 8, (0, 1, 1, 0) * 8, tuple([True] * 32), 4)
    corrupted = noisy_feedback(0.5, 11)(trace, 0)
    assert corrupted.symbols == trace.symbols
    assert corrupted.outputs != trace.outputs
    unchanged = noisy_feedback(0.0, 11)(trace, 0)
    assert unchanged.outputs == trace.outputs
    with pytest.raises(ValueError, match="fraction below one"):
        noisy_feedback(1.0, 1)


def test_a_triple_is_built_from_a_pair_that_is_in_the_stream() -> None:
    with_pairs, without_pairs, _, pairs, triples = build_layers(seed=41)
    assert [row[3] for row in with_pairs].count("pair") == len(pairs)
    assert "pair" not in {row[3] for row in without_pairs}
    known = {pair.automaton.digest() for pair in pairs}
    # Every triple has a depth-2 route: one of its parts is a stream pair.
    assert all(set(rule.part_digests()) & known for rule in triples)


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_novel_composites_are_cheaper_and_only_when_the_parts_are_there(
    tmp_path,
) -> None:
    """The claim, with the control that can refute it.

    Small enough to run in a test; the record's numbers come from replicates.
    What must hold at any size is the direction and the disjoint control.
    """

    before = sha256_file(BANK)
    report = run_transfer(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        composites=4,
    )
    arms = report["arms"]
    composing = arms["composing"]["composites"]

    # It solves them by building them.
    assert composing["composed"] >= 3
    assert composing["solved_nontrivial"] == composing["tasks"] - composing["trivial"]
    # And it uses the operator the task was actually built with.
    assert composing["combiner_recovered"] >= 3

    # Cheaper than a library that can only retrieve, and than no library.
    assert report["composition_ratio_against_recognition"] < 0.75
    assert report["composition_ratio_against_control"] < 0.75

    # But not when the parts are missing: the same mechanism, nothing to build
    # from, and no advantage over its own control.
    assert report["disjoint_ratio_against_its_control"] > 0.75

    # Destroying the feedback destroys it entirely.
    assert arms["shuffled"]["composites"]["solved_nontrivial"] == 0

    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_a_composite_the_agent_built_becomes_a_part(tmp_path) -> None:
    """Hierarchy: depth appears because admission makes it available."""

    before = sha256_file(BANK)
    report = run_hierarchy(
        CONTROLLER, BANK, tmp_path, frontend_path=FRONTEND, seed=41
    )
    triples = report["triples"]
    assert triples["with_pairs"]["solved"] == triples["with_pairs"]["tasks"]
    assert triples["with_pairs"]["composed"] >= 3
    # Cheaper than a fresh agent, and cheaper than the same agent that never
    # saw the intermediate layer.
    assert (
        triples["with_pairs"]["acquisition_steps"]
        < triples["control"]["acquisition_steps"]
    )
    assert report["depth_ratio"] < 1.0
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
