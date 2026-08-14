from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.physical_program_bank import (
    admit_physical_training_program,
    learned_event_context,
    retrieve_instruction_program,
    retrieve_physical_program,
)
from experiments.brainworkshop_canonical.physical_train import (
    PhysicalTrainingCampaign,
    PhysicalTrainingSession,
)
from experiments.brainworkshop_canonical.rendered_live import (
    PretrainedControllerProgramMachine,
    RecursiveTemporalProgramMachine,
    SourcePreservingTemporalMachine,
)
from neural_computer import (
    DEFAULT_AGENT_BANK_FILENAME,
    RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
    compose_recursive_temporal_program,
    recursive_temporal_primitive,
)


def _digest(seed: int = 0) -> str:
    return f"{seed:064x}"


def _artifact(values: tuple[float, ...]) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.tensor([values], dtype=torch.float32),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def test_rejected_provisional_program_leaves_live_bank_unchanged() -> None:
    bank = ExternalTemporalProgramBank(
        4, 3, controller_digest=_digest(), min_mastery_observations=3
    )
    before = bank.digest()

    receipt = bank.admit(
        _artifact((4.0, -2.0, -2.0)),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        [0.2, 1.0, 0.1],
        min_observations=3,
        min_stable_observations=2,
    )

    assert not receipt.accepted
    assert receipt.slot is None
    assert bank.program_count == 0
    assert bank.digest() == before


def test_verified_previous_primitive_composes_and_round_trips(tmp_path: Path) -> None:
    legacy = _artifact((8.0, -8.0, -8.0, -8.0))
    primitive = recursive_temporal_primitive(legacy)
    composed = compose_recursive_temporal_program(primitive, 2)
    assert primitive.program_length == 1
    assert composed.program_length == 2
    assert torch.equal(composed.codes[0], composed.codes[1])

    bank = ExternalTemporalProgramBank(
        4,
        4,
        controller_digest=_digest(19),
        min_mastery_observations=3,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    receipt = bank.admit(
        composed,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    path = tmp_path / DEFAULT_AGENT_BANK_FILENAME
    bank.save_bank(path)
    restored = ExternalTemporalProgramBank.load_bank(path)

    assert receipt.accepted
    assert restored.configuration()["execution_schema"] == (
        RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
    )
    assert restored.artifact(0).digest() == composed.digest()


def test_learned_event_context_accepts_live_tensor_rows() -> None:
    rows = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

    context = learned_event_context(rows, width=2)

    assert torch.allclose(context, torch.tensor([2**-0.5, 2**-0.5]))


def test_verified_programs_are_selected_from_opaque_reward_evidence() -> None:
    bank = ExternalTemporalProgramBank(
        4, 3, controller_digest=_digest(), min_mastery_observations=3
    )
    first_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second_context = torch.tensor([0.0, 1.0, 0.0, 0.0])
    first = bank.admit(
        _artifact((5.0, -3.0, -3.0)),
        first_context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    second = bank.admit(
        _artifact((-3.0, 5.0, -3.0)),
        second_context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )

    first_selection = bank.select(first_context)
    second_selection = bank.select(second_context)
    unknown = bank.select(torch.tensor([0.0, 0.0, 1.0, 0.0]))

    assert first.accepted and first.slot == 0
    assert second.accepted and second.slot == 1
    assert first_selection.slot == 0
    assert second_selection.slot == 1
    assert first_selection.propensity == 1.0
    assert second_selection.propensity == 1.0
    assert unknown.propensity == pytest.approx(0.5)
    assert not hasattr(bank, "task_id")


def test_context_shuffled_evidence_does_not_claim_a_learned_route() -> None:
    bank = ExternalTemporalProgramBank(
        4, 3, controller_digest=_digest(), min_mastery_observations=3
    )
    first_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second_context = torch.tensor([0.0, 1.0, 0.0, 0.0])
    bank.admit(
        _artifact((5.0, -3.0, -3.0)),
        first_context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    bank.admit(
        _artifact((-3.0, 5.0, -3.0)),
        first_context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )

    selection = bank.select(second_context)

    assert selection.propensity == pytest.approx(0.5)


def test_temporal_program_bank_round_trip_and_corruption_rejection(
    tmp_path: Path,
) -> None:
    bank = ExternalTemporalProgramBank(
        4, 3, controller_digest=_digest(7), min_mastery_observations=3
    )
    context = torch.tensor([1.0, 2.0, 3.0, 4.0])
    bank.admit(
        _artifact((5.0, -3.0, -3.0)),
        context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    path = tmp_path / "program-bank.pt"
    bank.save(path)

    restored = ExternalTemporalProgramBank.load(path)

    assert restored.digest() == bank.digest()
    assert restored.select(context).artifact.digest() == bank.artifact(0).digest()
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        ExternalTemporalProgramBank.load(path)


def test_agent_bank_loads_and_adds_a_skill_without_changing_the_first(
    tmp_path: Path,
) -> None:
    path = tmp_path / DEFAULT_AGENT_BANK_FILENAME
    first_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second_context = torch.tensor([0.0, 1.0, 0.0, 0.0])
    first_artifact = _artifact((5.0, -3.0, -3.0))
    bank = ExternalTemporalProgramBank(
        4, 3, controller_digest=_digest(7), min_mastery_observations=3
    )
    bank.admit(
        first_artifact,
        first_context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    bank.save_bank(path)

    resumed = ExternalTemporalProgramBank.load_bank(path)
    first_digest = resumed.artifact(0).digest()
    receipt = resumed.admit(
        _artifact((-3.0, 5.0, -3.0)),
        second_context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    resumed.save_bank(path)
    restored = ExternalTemporalProgramBank.load_bank(path)

    assert receipt.accepted and receipt.slot == 1
    assert restored.program_count == 2
    assert restored.artifact(0).digest() == first_digest
    assert restored.select(first_context).slot == 0
    assert restored.select(second_context).slot == 1


def test_canonical_agent_bank_rejects_non_bank_extension(tmp_path: Path) -> None:
    bank = ExternalTemporalProgramBank(4, 3, controller_digest=_digest(7))

    with pytest.raises(ValueError, match=r"must use the \.bank extension"):
        bank.save_bank(tmp_path / "AgentBrain.pt")
    with pytest.raises(ValueError, match=r"must use the \.bank extension"):
        ExternalTemporalProgramBank.load_bank(tmp_path / "AgentBrain.pt")


def test_retrieved_program_executes_with_controller_and_program_frozen() -> None:
    torch.manual_seed(9)
    source = SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=3,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
    )
    controller_state = {
        name: value.detach().clone()
        for name, value in source.named_parameters()
        if name != "relative_address_logits"
    }
    machine = PretrainedControllerProgramMachine(
        8,
        source_key_width=3,
        max_history=3,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        controller_state=controller_state,
        program_prior=torch.zeros(3),
    )
    controller_before = machine.controller_digest()
    bank = ExternalTemporalProgramBank(
        4,
        3,
        controller_digest=controller_before,
        min_mastery_observations=3,
    )
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    bank.admit(
        _artifact((-4.0, 6.0, -4.0)),
        context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )

    selection = bank.select(context)
    machine.load_admitted_program_artifact(
        selection.artifact,
        controller_digest=bank.controller_digest,
    )

    assert machine.controller_digest() == controller_before
    assert torch.equal(
        machine.relative_address_logits.detach().cpu(),
        torch.tensor([-4.0, 6.0, -4.0]),
    )
    assert not machine.learning_enabled
    assert not machine.sample
    assert machine.program_file_updates == 0


def test_physical_campaign_handoff_admits_and_retrieves_from_saved_events(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    source = SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=3,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
    )
    machine = PretrainedControllerProgramMachine(
        8,
        source_key_width=3,
        max_history=3,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        controller_state={
            name: value.detach().clone()
            for name, value in source.named_parameters()
            if name != "relative_address_logits"
        },
        program_prior=torch.zeros(3),
    )
    with torch.no_grad():
        machine.relative_address_logits.copy_(torch.tensor([7.0, -4.0, -4.0]))
    sessions = tuple(
        PhysicalTrainingSession(
            session=index,
            unique_public_outcomes=2,
            optimizer_updates=0,
            program_file_updates=2,
            emitted_actions=3,
            deadline_misses=0,
            accuracy=1.0,
            rolling_accuracy=1.0,
            cumulative_accuracy=1.0,
            cumulative_public_outcomes=index * 2,
            cumulative_optimizer_updates=0,
            elapsed_seconds=1.0,
        )
        for index in range(1, 4)
    )
    campaign = PhysicalTrainingCampaign(
        sessions=sessions,
        rewards=(1.0,) * 6,
        requested_sessions=3,
        completed_sessions=3,
        unique_public_outcomes=6,
        optimizer_updates=0,
        program_file_updates=6,
        replayed_examples=0,
        wall_seconds=3.0,
        rolling_window=3,
        learning_target=machine.learning_target,
        controller_digest_before=machine.controller_digest(),
        controller_digest_after=machine.controller_digest(),
        controller_frozen=True,
        program_digest_before="0" * 64,
        program_digest_after=machine.program_digest(),
    )
    event_rows = [
        [1.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.9, 0.4, 0.2, 0.2, 0.1, 0.4, 0.3, 0.6],
    ]
    for index in range(1, 4):
        (tmp_path / f"session-{index:03d}.json").write_text(
            json.dumps({"event_payloads": event_rows})
        )
    bank_path = tmp_path / DEFAULT_AGENT_BANK_FILENAME

    receipt = admit_physical_training_program(
        machine,
        campaign,
        tmp_path,
        bank_path,
        min_lifetimes=3,
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    selection = retrieve_physical_program(machine, bank, event_rows)

    assert receipt.accepted
    assert selection.slot == 0
    assert torch.equal(
        machine.relative_address_logits.detach(), torch.tensor([7.0, -4.0, -4.0])
    )
    assert not machine.learning_enabled
    assert not machine.sample


def _recursive_artifact(values: tuple[float, ...], length: int) -> ExternalProgramArtifact:
    row = torch.tensor([values], dtype=torch.float32)
    return ExternalProgramArtifact(
        codes=row.repeat(length, 1),
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def test_instruction_context_retrieves_composed_depth_without_task_id() -> None:
    torch.manual_seed(13)
    source = SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
    )
    controller_state = {
        name: value.detach().clone()
        for name, value in source.named_parameters()
        if name != "relative_address_logits"
    }
    machine = RecursiveTemporalProgramMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        controller_state=controller_state,
        program_prior=torch.zeros(4),
        sample=False,
    )
    one_back = torch.nn.functional.normalize(
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dim=0
    )
    two_back = torch.nn.functional.normalize(
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dim=0
    )
    shuffled = torch.nn.functional.normalize(
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dim=0
    )
    bank = ExternalTemporalProgramBank(
        8,
        4,
        controller_digest=machine.controller_digest(),
        generalization_tolerance=0.15,
        min_mastery_observations=3,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    first = bank.admit(
        _recursive_artifact((8.0, -8.0, -8.0, -8.0), 1),
        one_back,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    second = bank.admit(
        _recursive_artifact((8.0, -8.0, -8.0, -8.0), 2),
        two_back,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )

    selected_one = retrieve_instruction_program(machine, bank, one_back)
    depth_one = machine.composition_depth
    selected_two = retrieve_instruction_program(machine, bank, two_back)
    depth_two = machine.composition_depth
    selected_shuffled = retrieve_instruction_program(machine, bank, shuffled)

    assert first.accepted and first.slot == 0
    assert second.accepted and second.slot == 1
    assert selected_one.slot == 0 and depth_one == 1
    assert selected_two.slot == 1 and depth_two == 2
    assert selected_one.propensity == 1.0
    assert selected_two.propensity == 1.0
    assert not bank.router.has_context(shuffled)
    assert selected_shuffled.propensity == pytest.approx(0.5)
    assert not hasattr(bank, "n_back")
    assert not hasattr(bank, "task_id")


def test_same_program_binds_a_new_instruction_context_without_a_new_file() -> None:
    bank = ExternalTemporalProgramBank(
        4,
        4,
        controller_digest=_digest(23),
        generalization_tolerance=0.0,
        min_mastery_observations=3,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    )
    two_cell_one = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
    two_cell_two = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=0)
    three_cell_two = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=0)
    unseen_three_back = torch.nn.functional.normalize(
        torch.tensor([0.0, 0.0, 0.0, 1.0]), dim=0
    )
    primitive = _recursive_artifact((8.0, -8.0, -8.0, -8.0), 1)
    composed = _recursive_artifact((8.0, -8.0, -8.0, -8.0), 2)
    bank.admit(
        primitive,
        two_cell_one,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    first = bank.admit(
        composed,
        two_cell_two,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    digest_before = bank.digest()
    unknown = bank.select(three_cell_two)
    assert not bank.router.has_context(three_cell_two)
    rebound = bank.admit(
        composed,
        three_cell_two,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    heldout = bank.select(three_cell_two)
    retained = bank.select(two_cell_two)
    unseen = bank.select(unseen_three_back)

    assert first.accepted and first.slot == 1
    assert unknown.propensity == pytest.approx(0.5)
    assert rebound.accepted and rebound.slot == 1
    assert rebound.reason.startswith("verified experience attached")
    assert bank.program_count == 2
    assert heldout.slot == 1 and heldout.propensity == 1.0
    assert retained.slot == 1 and retained.propensity == 1.0
    assert not bank.router.has_context(unseen_three_back)
    assert unseen.propensity == pytest.approx(0.5)
    assert bank.digest() != digest_before
