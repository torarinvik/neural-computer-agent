from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.bank_program import (
    admit_temporal_program,
    compose_admitted_temporal,
    prototype_match_artifact,
    require_prototype_frontend,
)
from experiments.brainworkshop_canonical.controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from experiments.brainworkshop_canonical.program_search import propose_from_bank
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopVerifier,
)
from experiments.brainworkshop_canonical.rendered_live import run_rendered_live_lifetime
from neural_computer import ExternalTemporalProgramBank
from neural_computer.temporal_program import PROTOTYPE_MATCH_EXECUTION_SCHEMA

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)


def _machine(*, learn: bool = False):
    machine = build_pretrained_controller_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        learning_rate=0.3,
        sample=learn,
        inherit_program_prior=False,
    )
    machine.learning_enabled = learn
    machine.sample = learn
    return machine


def _symbol_config(*, steps: int = 24, seed_offset: int = 0) -> RenderedBrainWorkshopConfig:
    del seed_offset
    return RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        match_rule="current_symbol",
        target_symbol=0,
    )


def test_delay_files_fail_current_symbol_and_invent_is_proposed() -> None:
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = _machine()
    machine.load_admitted_program_artifact(
        bank.artifact(0), controller_digest=bank.controller_digest
    )
    delay = run_rendered_live_lifetime(
        machine,
        _vision_encoders(machine),
        _symbol_config(),
        seed=17,
        learn=False,
        sample=False,
    )
    assert delay.eligible_accuracy < 0.8
    kinds = [item.kind for item in propose_from_bank(bank)]
    assert kinds[-1] == "invent"


def _vision_encoders(machine):
    from experiments.brainworkshop_canonical.rendered_environment import (
        RenderedBrainWorkshopEncoders,
    )

    encoders = RenderedBrainWorkshopEncoders(
        machine.event_width, source_key_width=machine.source_key_width
    )
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)
    return encoders


def test_prototype_learn_updates_the_template_not_the_controller() -> None:
    machine = _machine(learn=True)
    machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
    encoders = _vision_encoders(machine)
    digest = machine.controller_digest()
    before = machine.prototype.detach().clone()
    report = run_rendered_live_lifetime(
        machine,
        encoders,
        _symbol_config(steps=24),
        seed=41,
        learn=True,
        sample=True,
    )
    assert report.program_file_updates >= 1
    assert report.eligible_accuracy >= 0.8
    assert machine.controller_digest() == digest
    assert not torch.equal(machine.prototype.detach().cpu(), before.cpu())
    assert float(machine.prototype.detach().norm()) > 0.0
    machine.learning_enabled = False
    machine.sample = False
    hold = run_rendered_live_lifetime(
        machine,
        encoders,
        _symbol_config(steps=24),
        seed=43,
        learn=False,
        sample=False,
    )
    learned = machine.prototype.detach().clone()
    machine.prototype.data.zero_()
    zeros = run_rendered_live_lifetime(
        machine,
        encoders,
        _symbol_config(steps=24),
        seed=43,
        learn=False,
        sample=False,
    )
    machine.prototype.data.copy_(learned)
    assert hold.eligible_accuracy >= 0.8
    assert zeros.eligible_accuracy < 0.8


def test_prototype_file_round_trips_without_changing_controller_digest(
    tmp_path: Path,
) -> None:
    machine = _machine(learn=True)
    machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
    encoders = _vision_encoders(machine)
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    assert machine.controller_digest() == bank.controller_digest
    report = run_rendered_live_lifetime(
        machine,
        encoders,
        _symbol_config(steps=24),
        seed=41,
        learn=True,
        sample=True,
    )
    assert report.program_file_updates >= 1
    artifact = prototype_match_artifact(
        machine.event_width,
        prototype=machine.prototype.detach(),
        frontend_digest=encoders.digest(),
    )
    path = tmp_path / "AgentBrain.bank"
    shutil.copy2(BANK, path)
    shutil.copy2(BANK.with_suffix(".bank.sha256"), path.with_suffix(".bank.sha256"))
    slot0 = ExternalTemporalProgramBank.load_bank(path).artifact(0).digest()
    receipt = admit_temporal_program(
        path,
        artifact,
        torch.nn.functional.normalize(torch.randn(16), dim=0),
        (0.9, 0.9),
        machine=machine,
        min_observations=2,
        min_stable_observations=1,
    )
    restored = ExternalTemporalProgramBank.load_bank(path)
    assert receipt.accepted
    assert restored.artifact(0).digest() == slot0
    assert restored.artifact(int(receipt.slot)).execution_schema == (
        PROTOTYPE_MATCH_EXECUTION_SCHEMA
    )
    assert restored.artifact(int(receipt.slot)).frontend_digest == encoders.digest()
    frozen = _machine(learn=False)
    frozen.load_prototype_artifact(
        restored.artifact(int(receipt.slot)),
        controller_digest=restored.controller_digest,
    )
    assert frozen._execution_schema == PROTOTYPE_MATCH_EXECUTION_SCHEMA
    assert frozen.controller_digest() == bank.controller_digest
    with pytest.raises(ValueError, match="only temporal-address primitives"):
        compose_admitted_temporal(restored, (int(receipt.slot), int(receipt.slot)))


def test_unbound_prototype_cannot_be_admitted(tmp_path: Path) -> None:
    path = tmp_path / "AgentBrain.bank"
    shutil.copy2(BANK, path)
    shutil.copy2(BANK.with_suffix(".bank.sha256"), path.with_suffix(".bank.sha256"))
    unbound = prototype_match_artifact(16)
    with pytest.raises(ValueError, match="frontend digest"):
        admit_temporal_program(
            path,
            unbound,
            torch.nn.functional.normalize(torch.randn(16), dim=0),
            (0.9, 0.9),
            machine=_machine(),
            min_observations=2,
            min_stable_observations=1,
        )
    assert ExternalTemporalProgramBank.load_bank(path).program_count == (
        ExternalTemporalProgramBank.load_bank(BANK).program_count
    )


def test_learned_prototype_fails_on_a_different_encoder() -> None:
    machine = _machine(learn=True)
    machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
    encoders = _vision_encoders(machine)
    run_rendered_live_lifetime(
        machine,
        encoders,
        _symbol_config(steps=24),
        seed=41,
        learn=True,
        sample=True,
    )
    machine.learning_enabled = False
    machine.sample = False
    same = run_rendered_live_lifetime(
        machine,
        encoders,
        _symbol_config(steps=24),
        seed=43,
        learn=False,
        sample=False,
    )
    other = _vision_encoders(machine)
    assert encoders.digest() != other.digest()
    crossed = run_rendered_live_lifetime(
        machine,
        other,
        _symbol_config(steps=24),
        seed=43,
        learn=False,
        sample=False,
    )
    bound = prototype_match_artifact(
        machine.event_width,
        prototype=machine.prototype.detach(),
        frontend_digest=encoders.digest(),
    )
    with pytest.raises(ValueError, match="frontend digest does not match"):
        require_prototype_frontend(bound, other)
    require_prototype_frontend(bound, encoders)
    assert same.eligible_accuracy >= 0.8
    assert crossed.eligible_accuracy < 0.8


def test_search_executes_invent_after_delay_files_fail_current_symbol() -> None:
    from experiments.brainworkshop_canonical.program_search import (
        search_temporal_programs,
    )

    bank = ExternalTemporalProgramBank.load_bank(BANK)
    machine = _machine()
    encoders = _vision_encoders(machine)
    verifier = RenderedBrainWorkshopVerifier(_symbol_config(), seed=17)
    assert verifier.eligible_trials == 24

    def evaluate(proposal):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            _symbol_config(),
            seed=17 + sum(proposal.slots),
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
    executed = [row for row in report["attempts"] if row["executed"]]
    kinds = [row["kind"] for row in executed]
    assert "retrieve" in kinds
    assert "invent" in kinds
    assert kinds[-1] == "invent"
    assert all(row["accuracy"] < 0.8 for row in executed)
    assert report["winner"] is None
    assert not any(
        row["executed"]
        for row in report["attempts"]
        if row["kind"] == "illegal_compose"
    )


def test_search_can_acquire_invent_without_admitting() -> None:
    from experiments.brainworkshop_canonical.program_search import (
        search_temporal_programs,
    )

    bank = ExternalTemporalProgramBank.load_bank(BANK)
    before = bank.program_count
    slot0 = bank.artifact(0).digest()
    machine = _machine()
    encoders = _vision_encoders(machine)

    def acquire(proposal):
        del proposal
        machine.learning_enabled = True
        machine.sample = True
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            _symbol_config(),
            seed=41,
            learn=True,
            sample=True,
        )
        machine.learning_enabled = False
        machine.sample = False
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    def evaluate(proposal):
        del proposal
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            _symbol_config(),
            seed=43,
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
    assert report["winner"]["kind"] == "invent"
    assert report["winner"]["accuracy"] >= 0.8
    assert report["winner"]["frontend_digest"] == encoders.digest()
    assert machine._frontend_digest == encoders.digest()
    restored = ExternalTemporalProgramBank.load_bank(BANK)
    assert restored.program_count == before
    assert restored.artifact(0).digest() == slot0
