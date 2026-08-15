from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.behaviour_signature import observe_stream
from experiments.brainworkshop_canonical.controller_pretraining import (
    load_temporal_controller_artifact,
)
from experiments.brainworkshop_canonical.current_symbol_acquire import (
    FRONTEND_SEED,
    _machine,
    curated_frontend,
)
from experiments.brainworkshop_canonical.feedback_proposer import (
    FeedbackProbe,
    probe_target,
    rank_by_agreement,
    signatures_for,
)
from experiments.brainworkshop_canonical.program_search import (
    install_proposal,
    propose_from_bank,
)
from experiments.brainworkshop_canonical.prototype_templates import observed_templates
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopVerifier,
)
from experiments.brainworkshop_canonical.rendered_live import run_rendered_live_lifetime
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer import ExternalTemporalProgramBank

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
    proposals = propose_from_bank(
        bank, observed_templates(encoders, config, seed=PROBE_SEED)
    )
    return encoders, bank, config, machine, proposals, rule


def test_one_episode_of_feedback_recovers_the_target_exactly(fixtures) -> None:
    """The inversion is checked against the generating rule.

    The agent never reads the rule; this test does, because a test is allowed
    to know what the agent had to infer.
    """

    encoders, bank, config, machine, proposals, rule = fixtures
    install_proposal(machine, bank, proposals[0])
    lifetime = run_rendered_live_lifetime(
        machine, encoders, config, seed=PROBE_SEED, learn=False, sample=False
    )
    probe = probe_target(lifetime, probe_label=proposals[0].label())
    verifier = RenderedBrainWorkshopVerifier(config, seed=PROBE_SEED)
    symbols = [int(value) for value in verifier._symbols["vision"]]
    expected = rule.expected(symbols)
    for index, eligible in enumerate(probe.eligible):
        if eligible:
            assert probe.target[index] == expected[index], index


def test_agreement_predicts_the_accuracy_an_episode_would_have_scored(
    fixtures,
) -> None:
    encoders, bank, config, machine, proposals, _ = fixtures
    stream = observe_stream(encoders, config, seed=PROBE_SEED)
    install_proposal(machine, bank, proposals[0])
    probe = probe_target(
        run_rendered_live_lifetime(
            machine, encoders, config, seed=PROBE_SEED, learn=False, sample=False
        )
    )
    signatures = signatures_for(
        proposals, machine, bank, stream, install=install_proposal
    )
    assert signatures
    for index, signature in list(signatures.items())[:6]:
        install_proposal(machine, bank, proposals[index])
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=PROBE_SEED, learn=False, sample=False
        )
        assert probe.agreement(signature) == pytest.approx(
            float(report.eligible_accuracy), abs=1e-9
        )


def test_ranking_puts_the_best_agreement_first_and_breaks_ties_by_order() -> None:
    probe = FeedbackProbe(
        target=(1, 0, 1, 0),
        eligible=(True, True, True, True),
        probe_label="x",
        probe_accuracy=0.5,
    )
    ranked = rank_by_agreement(
        {
            5: (1, 0, 1, 0),   # perfect
            2: (1, 0, 1, 1),   # 0.75
            9: (1, 0, 1, 1),   # 0.75, later index
            7: (0, 1, 0, 1),   # 0.0
        },
        probe,
    )
    assert [index for index, _ in ranked] == [5, 2, 9, 7]
    assert [score for _, score in ranked] == [1.0, 0.75, 0.75, 0.0]


def test_a_probe_needs_aligned_and_informative_evidence() -> None:
    with pytest.raises(ValueError, match="must align"):
        FeedbackProbe(
            target=(1, 0), eligible=(True,), probe_label="", probe_accuracy=0.0
        )
    with pytest.raises(ValueError, match="no eligible step"):
        FeedbackProbe(
            target=(1, 0),
            eligible=(False, False),
            probe_label="",
            probe_accuracy=0.0,
        )
    probe = FeedbackProbe(
        target=(1, 0, 1),
        eligible=(True, True, False),
        probe_label="",
        probe_accuracy=0.5,
    )
    assert probe.trials == 2
    with pytest.raises(ValueError, match="different episodes"):
        probe.agreement((1, 0))


def test_a_probe_reports_what_it_cost_and_what_it_saw() -> None:
    probe = FeedbackProbe(
        target=(1, 1, 0, 0),
        eligible=(True, True, True, False),
        probe_label="retrieve:0",
        probe_accuracy=0.25,
    )
    payload = probe.payload()
    assert payload["trials"] == 3
    assert payload["probe_label"] == "retrieve:0"
    assert payload["target_press_rate"] == pytest.approx(2 / 3)


def test_probe_inversion_is_arithmetic_not_a_lookup() -> None:
    class _Lifetime:
        actions = torch.tensor([[1, 0, 1, 0]])
        rewards = torch.tensor([[1, 1, 0, 0]])
        outcome_present = torch.tensor([[1, 1, 1, 1]])
        eligible_accuracy = 0.5

    probe = probe_target(_Lifetime())
    # Rewarded steps keep the action; punished steps take its complement.
    assert probe.target == (1, 0, 0, 1)
