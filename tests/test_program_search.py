from __future__ import annotations

from pathlib import Path

import torch

from experiments.brainworkshop_canonical.controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from experiments.brainworkshop_canonical.program_search import (
    propose_from_bank,
    search_temporal_programs,
)
from experiments.brainworkshop_canonical.rendered_dual_transfer_pilot import _encoders
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
)
from experiments.brainworkshop_canonical.rendered_live import run_rendered_live_lifetime
from neural_computer import ExternalTemporalProgramBank

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)


def _machine():
    return build_recursive_temporal_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        sample=False,
        max_sources=2,
        pack_source_actions=True,
    )


def test_proposer_tries_retrieve_before_compose_and_marks_unequal_pairs() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    proposals = propose_from_bank(bank)
    kinds = [item.kind for item in proposals]
    assert kinds[0] == "retrieve"
    assert kinds[1] == "retrieve"
    assert "compose" in kinds
    assert "illegal_compose" in kinds
    assert "invert" in kinds
    assert "and" in kinds
    assert kinds[-1] == "invent"
    assert any(item.kind == "invert" and item.slots == (0,) for item in proposals)
    illegal = [item for item in proposals if item.kind == "illegal_compose"]
    assert any(item.slots == (0, 1) for item in illegal)
    assert any(item.slots == (1, 0) for item in illegal)
    legal = [item for item in proposals if item.kind == "compose"]
    assert (1, 1) in {item.slots for item in legal}
    assert all(item.artifact is not None for item in legal)


def test_search_does_not_execute_illegal_pairs_and_can_select_compose() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = _machine()
    seen: list[str] = []

    def evaluate(proposal):
        seen.append(proposal.label())
        if proposal.kind == "compose" and proposal.slots == (1, 1):
            return {"accuracy": 1.0, "unique_verifier_bits": 16}
        return {"accuracy": 0.1, "unique_verifier_bits": 16}

    report = search_temporal_programs(bank, machine, evaluate, threshold=0.8)
    assert report["winner"]["label"] == "compose:1+1"
    assert "illegal_compose:0+1" not in seen
    assert "illegal_compose:1+0" not in seen
    assert report["illegal_compose"] >= 2
    assert all(
        not row["executed"]
        for row in report["attempts"]
        if row["kind"] == "illegal_compose"
    )


def test_search_finds_dual_2back_by_composing_slot_one_not_a_task_id() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = _machine()
    encoders = _encoders(machine)

    def evaluate(proposal):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=2, steps=24, streams=("vision", "audio")
            ),
            seed=77 + sum(proposal.slots),
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    report = search_temporal_programs(
        bank, machine, evaluate, threshold=0.8, minimum_bits=4
    )
    assert report["winner"] is not None
    assert report["winner"]["accuracy"] >= 0.8
    if report["winner"]["kind"] == "retrieve":
        child = bank.artifact(int(report["winner"]["slots"][0]))
        assert child.program_length >= 2
    else:
        assert report["winner"]["kind"] == "compose"
        assert report["winner"]["slots"] in ([0, 0], [1, 1])
        retrieves = [
            row
            for row in report["attempts"]
            if row["kind"] == "retrieve" and row["executed"]
        ]
        assert retrieves
        assert all(row["accuracy"] < 0.8 for row in retrieves)
    assert all(
        not row["executed"]
        for row in report["attempts"]
        if row["kind"] == "illegal_compose"
    )


def test_search_solves_changed_by_inverting_a_delay_file() -> None:
    from experiments.brainworkshop_canonical.controller_pretraining import (
        build_pretrained_controller_program_machine,
    )

    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = build_pretrained_controller_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        sample=False,
        inherit_program_prior=False,
    )
    encoders = _encoders(machine)

    def evaluate(proposal):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=1,
                steps=24,
                streams=("vision",),
                match_rule="changed",
            ),
            seed=59 + sum(proposal.slots),
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    report = search_temporal_programs(
        bank, machine, evaluate, threshold=0.8, minimum_bits=8
    )
    assert report["winner"] is not None
    assert report["winner"]["kind"] == "invert"
    assert report["winner"]["accuracy"] >= 0.8
    retrieves = [
        row
        for row in report["attempts"]
        if row["kind"] == "retrieve" and row["executed"]
    ]
    assert retrieves
    assert all(row["accuracy"] < 0.8 for row in retrieves)


def test_search_retrieves_an_admitted_invert_child_for_changed(
    tmp_path: Path,
) -> None:
    import shutil

    from experiments.brainworkshop_canonical.bank_program import admit_invert_child
    from experiments.brainworkshop_canonical.controller_pretraining import (
        build_pretrained_controller_program_machine,
    )

    path = tmp_path / "AgentBrain.bank"
    shutil.copy2(BANK, path)
    shutil.copy2(BANK.with_suffix(".bank.sha256"), path.with_suffix(".bank.sha256"))
    machine = build_pretrained_controller_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        sample=False,
        inherit_program_prior=False,
    )
    receipt = admit_invert_child(
        path,
        0,
        torch.nn.functional.normalize(torch.randn(16), dim=0),
        (1.0,),
        machine=machine,
        min_observations=1,
    )
    assert receipt.accepted
    bank = ExternalTemporalProgramBank.load_bank(path)
    encoders = _encoders(machine)

    def evaluate(proposal):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=1,
                steps=24,
                streams=("vision",),
                match_rule="changed",
            ),
            seed=61 + sum(proposal.slots),
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    report = search_temporal_programs(
        bank, machine, evaluate, threshold=0.8, minimum_bits=8
    )
    assert report["winner"] is not None
    assert report["winner"]["kind"] == "retrieve"
    assert report["winner"]["slots"] == [int(receipt.slot)]
    assert report["winner"]["accuracy"] >= 0.8
    assert ExternalTemporalProgramBank.load_bank(BANK).program_count == 3


def test_search_solves_onset_by_and_acquire() -> None:
    from experiments.brainworkshop_canonical.controller_pretraining import (
        build_pretrained_controller_program_machine,
    )
    from experiments.brainworkshop_canonical.rendered_environment import (
        RenderedBrainWorkshopEncoders,
    )

    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = build_pretrained_controller_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        learning_rate=0.3,
        sample=True,
        inherit_program_prior=False,
    )
    encoders = RenderedBrainWorkshopEncoders.load(
        REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
    )
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=48,
        streams=("vision",),
        match_rule="onset",
        target_symbol=0,
    )

    def acquire(proposal):
        del proposal
        machine.learning_enabled = True
        machine.sample = False
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=71, learn=True, sample=False
        )
        machine.learning_enabled = False
        machine.sample = False
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    def evaluate(proposal):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=73 + sum(proposal.slots),
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    report = search_temporal_programs(
        bank,
        machine,
        evaluate,
        threshold=0.8,
        minimum_bits=8,
        acquire=acquire,
        encoders=encoders,
    )
    assert report["winner"] is not None
    assert report["winner"]["kind"] == "and"
    assert report["winner"]["accuracy"] >= 0.8
    earlier = [
        row
        for row in report["attempts"]
        if row["executed"] and row["kind"] in {"retrieve", "invert"}
    ]
    assert earlier
    assert all(row["accuracy"] < 0.8 for row in earlier)


def test_and_of_invert_and_a_prototype_solves_onset() -> None:
    from experiments.brainworkshop_canonical.bank_program import (
        install_temporal_artifact,
        invert_artifact,
    )
    from experiments.brainworkshop_canonical.controller_pretraining import (
        build_pretrained_controller_program_machine,
    )
    from experiments.brainworkshop_canonical.rendered_environment import (
        RenderedBrainWorkshopEncoders,
    )
    from neural_computer.temporal_program import PROTOTYPE_MATCH_EXECUTION_SCHEMA

    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = build_pretrained_controller_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        learning_rate=0.3,
        sample=True,
        inherit_program_prior=False,
    )
    frontend = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
    encoders = RenderedBrainWorkshopEncoders.load(frontend)
    onset = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=48,
        streams=("vision",),
        match_rule="onset",
        target_symbol=0,
    )
    current = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=48,
        streams=("vision",),
        match_rule="current_symbol",
        target_symbol=0,
    )
    install_temporal_artifact(machine, bank, bank.artifact(0))
    retrieve = run_rendered_live_lifetime(
        machine, encoders, onset, seed=73, learn=False, sample=False
    )
    install_temporal_artifact(machine, bank, invert_artifact(bank.artifact(0)))
    machine._invert_intention = True
    inverted = run_rendered_live_lifetime(
        machine, encoders, onset, seed=73, learn=False, sample=False
    )
    machine._invert_intention = False
    machine._combine_and = False
    machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
    run_rendered_live_lifetime(
        machine, encoders, current, seed=71, learn=True, sample=True
    )
    machine.learning_enabled = False
    machine.sample = False
    install_temporal_artifact(machine, bank, invert_artifact(bank.artifact(0)))
    machine._invert_intention = True
    machine._combine_and = True
    combined = run_rendered_live_lifetime(
        machine, encoders, onset, seed=73, learn=False, sample=False
    )
    assert retrieve.eligible_accuracy < 0.8
    assert inverted.eligible_accuracy < 0.8
    assert combined.eligible_accuracy >= 0.8
    assert combined.program_file_updates == 0
    assert any(item.kind == "and" for item in propose_from_bank(bank))
