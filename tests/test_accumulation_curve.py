from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.accumulation_curve import (
    ACCUMULATION_SCHEMA,
    _copy_bank,
    _curve,
    _find_proposal,
    _library_prefix_length,
    _winner_artifact,
    curriculum_rules,
    run_accumulation_curve,
)
from experiments.brainworkshop_canonical.program_search import (
    ProgramProposal,
    propose_from_bank,
)
from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file
from neural_computer.temporal_program import (
    INTENTION_AND_EXECUTION_SCHEMA,
    INTENTION_INVERT_EXECUTION_SCHEMA,
    PROTOTYPE_MATCH_EXECUTION_SCHEMA,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


class _Machine:
    """Only the attribute `_winner_artifact` reads off a live machine."""

    def __init__(self, width: int) -> None:
        self.prototype = torch.arange(float(width))


def test_the_curriculum_is_the_population_the_ceilings_were_measured_on() -> None:
    rules = curriculum_rules()
    assert len(rules) == 18
    assert len({rule.digest() for rule in rules}) == 18
    # Ascending complexity, so library size and rule difficulty do not confound.
    assert [rule.state_count for rule in rules] == sorted(
        rule.state_count for rule in rules
    )
    assert {rule.state_count for rule in rules} == {1, 2, 3, 4, 5, 6}


def test_a_winning_polarity_survives_the_trip_into_the_bank() -> None:
    """An inverted template must not be stored as a plain prototype.

    `install_proposal` flips the intention on the machine, not in the
    artifact, so storing the artifact alone would silently admit a program
    that behaves differently the next time it is retrieved.
    """

    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = _Machine(bank.context_width)
    template = torch.ones(bank.context_width)
    plain = ProgramProposal(
        "invent", (), None, template=template, template_label=(0,), invert_intention=False
    )
    flipped = ProgramProposal(
        "invent", (), None, template=template, template_label=(0,), invert_intention=True
    )
    plain_artifact, _, _ = _winner_artifact(
        plain, machine, bank, frontend_digest="a" * 64
    )
    flipped_artifact, _, _ = _winner_artifact(
        flipped, machine, bank, frontend_digest="a" * 64
    )
    assert plain_artifact is not None and flipped_artifact is not None
    assert plain_artifact.execution_schema == PROTOTYPE_MATCH_EXECUTION_SCHEMA
    assert flipped_artifact.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA
    assert plain_artifact.digest() != flipped_artifact.digest()


def test_an_and_winner_names_the_delay_parent_it_needs() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = _Machine(bank.context_width)
    proposal = ProgramProposal("and", (1,), None)
    artifact, delay_slot, reason = _winner_artifact(
        proposal, machine, bank, frontend_digest="b" * 64
    )
    assert artifact is not None
    assert artifact.execution_schema == INTENTION_AND_EXECUTION_SCHEMA
    # Without the parent slot the child cannot be installed again later.
    assert delay_slot == 1
    assert "and" in reason


def test_a_retrieve_winner_adds_nothing_to_the_library() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    artifact, delay_slot, reason = _winner_artifact(
        ProgramProposal("retrieve", (0,), bank.artifact(0)),
        _Machine(bank.context_width),
        bank,
        frontend_digest="c" * 64,
    )
    assert artifact is None and delay_slot is None
    assert "already in the library" in reason


def test_the_library_prefix_is_what_a_rule_must_walk_before_inventing() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    proposals = propose_from_bank(bank)
    prefix = _library_prefix_length(proposals)
    assert 0 < prefix < len(proposals)
    assert all(item.kind != "invent" for item in proposals[:prefix])
    assert proposals[prefix].kind == "invent"


def test_proposals_are_recoverable_by_the_label_the_search_records() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    templates = (((0,), torch.ones(bank.context_width)),)
    proposals = propose_from_bank(bank, templates)
    for proposal in proposals:
        assert _find_proposal(proposals, proposal.label()) is not None
    assert _find_proposal(proposals, "no-such-proposal") is None


def test_the_curve_refuses_arms_that_disagree_about_the_order() -> None:
    growing = {"rules": [{"rule_digest": "a", "state_count": 1, "library_size": 3,
                          "programs_executed": 4, "solved": True,
                          "reproduces": True, "reused_learned_slot": False}]}
    control = {"rules": [{"rule_digest": "b", "state_count": 1, "library_size": 3,
                          "programs_executed": 8, "solved": True,
                          "reproduces": True, "reused_learned_slot": False}]}
    with pytest.raises(RuntimeError, match="curriculum order"):
        _curve(growing, control)
    control["rules"][0]["rule_digest"] = "a"
    point = _curve(growing, control)[0]
    assert point["cost_ratio"] == 0.5


def test_copying_a_bank_brings_the_checksum_it_refuses_to_load_without(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "scratch.bank"
    _copy_bank(BANK, destination)
    copied = ExternalTemporalProgramBank.load_bank(destination)
    assert copied.program_count == ExternalTemporalProgramBank.load_bank(BANK).program_count
    # Without the sidecar the loader must fail closed rather than trust bytes.
    destination.with_suffix(".bank.sha256").unlink()
    with pytest.raises(ValueError, match="checksum is missing"):
        ExternalTemporalProgramBank.load_bank(destination)


def test_both_arms_run_and_the_curated_bank_is_never_written(tmp_path: Path) -> None:
    before = sha256_file(BANK)
    report = run_accumulation_curve(
        CONTROLLER,
        BANK,
        tmp_path / "record",
        scratch_directory=tmp_path / "scratch",
        frontend_path=FRONTEND,
        state_counts=(1, 2),
        rules_per_state_count=1,
        steps=64,
    )
    assert sha256_file(BANK) == before
    assert report["schema"] == ACCUMULATION_SCHEMA
    assert report["status"] == "diagnostic"
    assert report["bank_unchanged"] is True
    assert len(report["curve"]) == 2
    # The control arm is a fresh agent per rule: every rule must see exactly
    # the founding library, whatever the rules before it learned.
    founding = report["control"]["founding_program_count"]
    assert report["control"]["final_program_count"] == founding
    assert all(row["library_size"] == founding for row in report["control"]["rules"])
    assert all(row["library_grown_by"] == 0 for row in report["control"]["rules"])
    # The growing arm's library may only ever move forward.
    sizes = [row["library_size"] for row in report["growing"]["rules"]]
    assert sizes == sorted(sizes)
    assert report["growing"]["final_program_count"] >= founding
    recorded = json.loads((tmp_path / "record" / "accumulation.json").read_text())
    assert recorded["experiment_id"] == report["experiment_id"]
    assert (tmp_path / "record" / "checksums.sha256").is_file()
