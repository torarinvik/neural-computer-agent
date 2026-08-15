from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.behaviour_signature import (
    NO_ACTION,
    behaviour_signature,
    observe_stream,
    partition_by_behaviour,
)
from experiments.brainworkshop_canonical.controller_pretraining import (
    load_temporal_controller_artifact,
)
from experiments.brainworkshop_canonical.current_symbol_acquire import (
    FRONTEND_SEED,
    _machine,
    curated_frontend,
)
from experiments.brainworkshop_canonical.program_search import (
    install_proposal,
    propose_from_bank,
)
from experiments.brainworkshop_canonical.prototype_templates import observed_templates
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
)
from experiments.brainworkshop_canonical.rendered_live import run_rendered_live_lifetime
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
STEPS = 96
PROBE_SEED = 41


@pytest.fixture(scope="module")
def fixtures():
    payload = load_temporal_controller_artifact(CONTROLLER)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    rule = sample_rule(symbol_count=4, state_count=5, seed=6500)
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=STEPS,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=rule,
    )
    machine = _machine(payload, learn=False)
    templates = observed_templates(encoders, config, seed=PROBE_SEED)
    proposals = propose_from_bank(bank, templates)
    return payload, encoders, bank, config, machine, proposals


def test_an_offline_signature_equals_the_actions_taken_online(fixtures) -> None:
    """The whole filter rests on this: replay must match the real lifetime."""

    _, encoders, bank, config, machine, proposals = fixtures
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    install_proposal(machine, bank, proposals[0])
    lifetime = run_rendered_live_lifetime(
        machine, encoders, config, seed=PROBE_SEED, learn=False, sample=False
    )
    signature = behaviour_signature(machine, stream)
    online = [int(value) for value in lifetime.actions.reshape(-1)]
    assert list(signature) == online
    assert NO_ACTION not in signature


def test_collapsing_a_class_cannot_change_a_score(fixtures) -> None:
    """Losslessness, checked against real scored runs rather than argued."""

    _, encoders, bank, config, machine, proposals = fixtures
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    classes = partition_by_behaviour(
        proposals, machine, bank, stream, install=install_proposal
    )
    assert classes.distinct < len(proposals)
    largest = max(classes.members, key=len)
    assert len(largest) > 1
    scores = []
    for index in largest[:4]:
        install_proposal(machine, bank, proposals[index])
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=PROBE_SEED, learn=False, sample=False
        )
        scores.append(round(float(report.eligible_accuracy), 9))
    assert len(set(scores)) == 1


def test_distinct_classes_really_do_press_differently(fixtures) -> None:
    _, encoders, bank, config, machine, proposals = fixtures
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    classes = partition_by_behaviour(
        proposals, machine, bank, stream, install=install_proposal
    )
    signed = [item for item in classes.signatures if item is not None]
    assert len(set(signed)) == len(signed)
    # Every proposal is accounted for exactly once.
    covered = sum(len(group) for group in classes.members) + len(classes.unsignable)
    assert covered == len(proposals)
    # Proposals that train before scoring are singletons, never collapsed.
    for index in classes.trained:
        assert classes.members[classes.representatives.index(index)] == (index,)


def test_the_representative_is_the_earliest_member_so_no_winner_moves(
    fixtures,
) -> None:
    _, encoders, bank, config, machine, proposals = fixtures
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    classes = partition_by_behaviour(
        proposals, machine, bank, stream, install=install_proposal
    )
    for leader, group in zip(classes.representatives, classes.members):
        assert leader == min(group)
        assert classes.representative_of(leader) == leader
    assert classes.representative_of(10**6) is None


def test_a_signature_depends_on_the_program_and_not_on_what_ran_before(
    fixtures,
) -> None:
    _, encoders, bank, config, machine, proposals = fixtures
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    install_proposal(machine, bank, proposals[0])
    first = behaviour_signature(machine, stream)
    for index in range(1, min(6, len(proposals))):
        try:
            install_proposal(machine, bank, proposals[index])
        except ValueError:
            continue
        behaviour_signature(machine, stream)
    install_proposal(machine, bank, proposals[0])
    assert behaviour_signature(machine, stream) == first


def test_an_observation_pass_reads_no_reward_and_leaves_the_bank_alone(
    fixtures,
) -> None:
    _, encoders, _, config, _, _ = fixtures
    before = sha256_file(BANK)
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    assert len(stream) == STEPS
    assert sha256_file(BANK) == before
    # The stream carries stimuli only; nothing in it is a score.
    assert all(hasattr(events, "payload") for events in stream)


def test_an_empty_stream_is_refused(fixtures) -> None:
    _, _, _, _, machine, _ = fixtures
    with pytest.raises(ValueError, match="recorded stream"):
        behaviour_signature(machine, ())


def test_two_rules_share_a_stimulus_stream_so_context_cannot_route(
    fixtures,
) -> None:
    """The tasks differ in reward, not in what the learner sees.

    This is why routing on observed context cannot order proposals here, and
    why selection has to come from feedback instead.
    """

    _, encoders, _, config, _, _ = fixtures
    other = sample_rule(symbol_count=4, state_count=3, seed=6301)
    assert other is not None
    elsewhere = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=STEPS,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=other,
    )
    first = observe_stream(encoders, config, seed=PROBE_SEED)
    second = observe_stream(encoders, elsewhere, seed=PROBE_SEED)
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert torch.equal(left.payload, right.payload)
