from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.bank_program import (
    admit_and_child,
    admit_composed_child,
    admit_invert_child,
    admit_temporal_program,
    compose_admitted_temporal,
    composition_lineage,
    install_temporal_artifact,
    invert_artifact,
    retrieve_temporal_program,
    temporal_address_artifact,
)
from experiments.brainworkshop_canonical.controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from neural_computer import ExternalTemporalProgramBank
from neural_computer.program import ExternalProgramArtifact
from neural_computer.temporal_program import (
    INTENTION_INVERT_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)


def _copy_bank(tmp_path: Path) -> Path:
    destination = tmp_path / "AgentBrain.bank"
    shutil.copy2(SOURCE_BANK, destination)
    shutil.copy2(SOURCE_BANK.with_suffix(".bank.sha256"), destination.with_suffix(".bank.sha256"))
    return destination


def _dual_machine(*, learn: bool = False):
    return build_recursive_temporal_program_machine(
        load_temporal_controller_artifact(CONTROLLER),
        sample=learn,
        max_sources=2,
        pack_source_actions=True,
    )


def _context(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(16, generator=generator), dim=0)


def test_compose_requires_equal_one_row_primitives(tmp_path: Path) -> None:
    bank = ExternalTemporalProgramBank.load_bank(_copy_bank(tmp_path))
    composed = compose_admitted_temporal(bank, (0, 0))
    if bank.program_count >= 2:
        dual = compose_admitted_temporal(bank, (1, 1))
        assert dual.program_length == 2
        assert torch.equal(dual.codes[0], bank.artifact(1).codes[0])
        if not torch.equal(bank.artifact(0).codes[0], bank.artifact(1).codes[0]):
            with pytest.raises(ValueError, match="unequal temporal primitives"):
                compose_admitted_temporal(bank, (0, 1))
    assert composed.program_length == 2
    assert torch.equal(composed.codes[0], bank.artifact(0).codes[0])
    assert torch.equal(composed.codes[1], bank.artifact(0).codes[0])

    other = ExternalProgramArtifact(
        codes=bank.artifact(0).codes.clone(),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    other.codes[0, 0] += 1.0
    receipt = bank.admit(
        other,
        _context(3),
        (1.0, 1.0),
        min_observations=2,
        min_stable_observations=1,
    )
    assert receipt.accepted
    with pytest.raises(ValueError, match="unequal temporal primitives"):
        compose_admitted_temporal(bank, (0, int(receipt.slot)))


def test_admit_attaches_duplicate_and_keeps_slot_zero(tmp_path: Path) -> None:
    path = _copy_bank(tmp_path)
    bank = ExternalTemporalProgramBank.load_bank(path)
    slot0 = bank.artifact(0).digest()
    machine = _dual_machine()
    machine.load_admitted_program_artifact(
        bank.artifact(0), controller_digest=bank.controller_digest
    )
    artifact = temporal_address_artifact(machine)
    before_count = bank.program_count
    receipt = admit_temporal_program(
        path,
        artifact,
        _context(11),
        (0.9, 1.0, 1.0),
        machine=machine,
    )
    restored = ExternalTemporalProgramBank.load_bank(path)
    assert receipt.accepted
    assert receipt.slot == 0
    assert restored.program_count == before_count
    assert restored.artifact(0).digest() == slot0


def test_admit_appends_a_new_slot_without_rewriting_slot_zero(tmp_path: Path) -> None:
    path = _copy_bank(tmp_path)
    bank = ExternalTemporalProgramBank.load_bank(path)
    slot0 = bank.artifact(0).digest()
    before_count = bank.program_count
    novel = ExternalProgramArtifact(
        codes=torch.tensor([[-12.0, 12.0, -12.0, -12.0]]),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    machine = _dual_machine()
    receipt = admit_temporal_program(
        path,
        novel,
        _context(17),
        (0.85, 1.0),
        machine=machine,
    )
    restored = ExternalTemporalProgramBank.load_bank(path)
    assert receipt.accepted
    assert receipt.slot == before_count
    assert restored.program_count == before_count + 1
    assert restored.artifact(0).digest() == slot0
    assert restored.artifact(int(receipt.slot)).digest() == novel.digest()


def test_dual_machine_retrieves_an_admitted_file(tmp_path: Path) -> None:
    path = _copy_bank(tmp_path)
    bank = ExternalTemporalProgramBank.load_bank(path)
    machine = _dual_machine()
    events = torch.stack([_context(21), _context(22)])
    selection = retrieve_temporal_program(machine, bank, events)
    assert selection.slot in {0, 1}
    assert machine.controller_digest() == bank.controller_digest
    assert torch.equal(
        machine.relative_address_logits.detach().cpu(),
        bank.artifact(selection.slot).codes[0].cpu(),
    )


def test_composed_child_admits_without_rewriting_slot_zero(tmp_path: Path) -> None:
    from experiments.brainworkshop_canonical.program_search import (
        search_temporal_programs,
    )
    from experiments.brainworkshop_canonical.rendered_dual_transfer_pilot import (
        _encoders,
    )
    from experiments.brainworkshop_canonical.rendered_environment import (
        RenderedBrainWorkshopConfig,
    )
    from experiments.brainworkshop_canonical.rendered_live import (
        run_rendered_live_lifetime,
    )

    path = _copy_bank(tmp_path)
    before = ExternalTemporalProgramBank.load_bank(path)
    slot0 = before.artifact(0).digest()
    machine = _dual_machine()
    encoders = _encoders(machine)
    child = compose_admitted_temporal(before, (0, 0))
    install_temporal_artifact(machine, before, child)
    scored = run_rendered_live_lifetime(
        machine,
        encoders,
        RenderedBrainWorkshopConfig(n_back=2, steps=24, streams=("vision", "audio")),
        seed=91,
        learn=False,
        sample=False,
    )
    assert scored.eligible_accuracy >= 0.8
    receipt = admit_composed_child(
        path,
        (0, 0),
        _context(31),
        (float(scored.eligible_accuracy),),
        machine=machine,
        min_observations=1,
    )
    restored = ExternalTemporalProgramBank.load_bank(path)
    assert receipt.accepted
    assert restored.artifact(0).digest() == slot0
    assert restored.artifact(int(receipt.slot)).program_length == 2
    if receipt.reason and "identical" in receipt.reason:
        assert restored.program_count == before.program_count
    else:
        assert receipt.slot == before.program_count
        assert restored.program_count == before.program_count + 1

    def evaluate(proposal):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            RenderedBrainWorkshopConfig(
                n_back=2, steps=24, streams=("vision", "audio")
            ),
            seed=101 + sum(proposal.slots),
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    search = search_temporal_programs(
        restored, machine, evaluate, threshold=0.8, minimum_bits=4
    )
    assert search["winner"] is not None
    assert search["winner"]["kind"] == "retrieve"
    assert search["winner"]["slots"] == [int(receipt.slot)]
    lineage = composition_lineage(restored, int(receipt.slot))
    assert lineage is not None
    assert lineage["inferred"] is False
    assert lineage["parent_slots"] == [0, 0]
    assert lineage["parent_digests"] == [slot0, slot0]
    assert lineage["child_digest"] == restored.artifact(int(receipt.slot)).digest()
    inferred = composition_lineage(
        ExternalTemporalProgramBank.load_bank(SOURCE_BANK), 2
    )
    assert inferred is not None
    assert inferred["inferred"] is True
    assert inferred["parent_slots"] == [0, 0]
    assert inferred["depth"] == 2


def test_invert_child_admits_without_rewriting_slot_zero(tmp_path: Path) -> None:
    path = _copy_bank(tmp_path)
    before = ExternalTemporalProgramBank.load_bank(path)
    slot0 = before.artifact(0).digest()
    before_count = before.program_count
    with pytest.raises(ValueError, match="cannot invert again"):
        invert_artifact(invert_artifact(before.artifact(0)))
    machine = _dual_machine()
    receipt = admit_invert_child(
        path,
        0,
        _context(41),
        (1.0,),
        machine=machine,
        min_observations=1,
    )
    restored = ExternalTemporalProgramBank.load_bank(path)
    assert receipt.accepted
    assert restored.artifact(0).digest() == slot0
    assert restored.program_count == before_count + 1
    child = restored.artifact(int(receipt.slot))
    assert child.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA
    assert torch.equal(child.codes, before.artifact(0).codes)
    lineage = composition_lineage(restored, int(receipt.slot))
    assert lineage is not None
    assert lineage["parent_slots"] == [0]
    assert lineage["parent_digests"] == [slot0]
    install_temporal_artifact(machine, restored, child)
    assert machine._invert_intention is True


def test_and_child_admits_and_reloads_without_rewriting_slot_zero(
    tmp_path: Path,
) -> None:
    from experiments.brainworkshop_canonical.rendered_environment import (
        RenderedBrainWorkshopEncoders,
    )
    from neural_computer.temporal_program import INTENTION_AND_EXECUTION_SCHEMA

    path = _copy_bank(tmp_path)
    before = ExternalTemporalProgramBank.load_bank(path)
    slot0 = before.artifact(0).digest()
    machine = _dual_machine()
    encoders = RenderedBrainWorkshopEncoders.seeded(
        16, source_key_width=4, seed=1001
    )
    proto = torch.nn.functional.normalize(torch.randn(16), dim=0)
    receipt = admit_and_child(
        path,
        0,
        proto,
        _context(43),
        (1.0,),
        frontend_digest=encoders.digest(),
        machine=machine,
        min_observations=1,
    )
    restored = ExternalTemporalProgramBank.load_bank(path)
    assert receipt.accepted
    assert restored.artifact(0).digest() == slot0
    child = restored.artifact(int(receipt.slot))
    assert child.execution_schema == INTENTION_AND_EXECUTION_SCHEMA
    lineage = composition_lineage(restored, int(receipt.slot))
    assert lineage is not None
    assert lineage["parent_slots"] == [0]
    install_temporal_artifact(machine, restored, child)
    assert machine._combine_and is True
    assert machine._invert_intention is True
