from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from neural_computer import (
    AGENT_BRAIN_EXECUTIVE_KIND,
    AGENT_BRAIN_TEMPORAL_KIND,
    EXTERNAL_AGENT_BRAIN_BANK_SCHEMA,
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    ExecutiveCompositionEvaluation,
    ExecutiveInstruction,
    ExternalAgentBrainBank,
    ExternalExecutiveCompositionSearch,
    ExternalExecutiveOperatorSpec,
    ExternalExecutiveProgram,
    ExternalExecutiveProgramArtifact,
    ExternalExecutiveProgramBank,
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
    build_temporal_equality_executive_artifact,
    compose_executive_artifacts,
)


def _digest(seed: int = 7) -> str:
    return f"{seed:064x}"


def test_promoted_v1_composition_bank_remains_loadable() -> None:
    path = (
        Path(__file__).parents[1]
        / "session_records"
        / "brainworkshop_executive_compositional_transfer_promoted_2026-08-14"
        / "AgentBrain.bank"
    )

    restored = ExternalAgentBrainBank.load_bank(path)

    assert restored.program_count == 4
    assert restored.composition_provenance[0]["schema"].endswith(".v1")


def _temporal_artifact(values: tuple[float, ...]) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=torch.tensor([values], dtype=torch.float32),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def _finite_executive_artifact(variant: int = 0) -> ExternalExecutiveProgramArtifact:
    wait_target = None if variant == 0 else 2 if variant == 1 else 1
    wait = ExecutiveInstruction("wait", next_target=wait_target)
    return ExternalExecutiveProgramArtifact(
        program=ExternalExecutiveProgram(
            1,
            (
                ExecutiveInstruction("receive", destination=0),
                wait,
                ExecutiveInstruction("halt"),
            ),
        ),
        operator_specs=(),
        intention_width=1,
    ).validate()


def test_executive_composition_rebases_slots_and_terminal_handoff() -> None:
    first = _finite_executive_artifact()
    second = _finite_executive_artifact(1)

    composed = compose_executive_artifacts((first, second))

    assert composed.program.slot_count == 2
    assert len(composed.program.instructions) == 5
    assert composed.program.instructions[2].destination == 1
    assert composed.program.instructions[-1].op == "halt"
    assert composed.program.instructions[-1].source is None


def test_executive_composition_rejects_unreachable_later_components() -> None:
    looping = _finite_executive_artifact(2)
    finite = _finite_executive_artifact()

    with pytest.raises(ValueError, match="terminal handoff"):
        compose_executive_artifacts((looping, finite))


def test_executive_composition_can_share_compatible_operator_state_explicitly() -> None:
    def finite_with_value_operator(handle: int) -> ExternalExecutiveProgramArtifact:
        return ExternalExecutiveProgramArtifact(
            program=ExternalExecutiveProgram(
                2,
                (
                    ExecutiveInstruction("receive", destination=0),
                    ExecutiveInstruction(
                        "call", destination=1, operator_handle=handle, arguments=(0,)
                    ),
                    ExecutiveInstruction("wait"),
                    ExecutiveInstruction("halt"),
                ),
            ),
            operator_specs=(
                ExternalExecutiveOperatorSpec(
                    handle, "singleton_event_value", width=3
                ),
            ),
            intention_width=1,
        ).validate()

    first = finite_with_value_operator(4)
    second = finite_with_value_operator(9)
    isolated = compose_executive_artifacts((first, second))
    shared = compose_executive_artifacts(
        (first, second), share_compatible_operators=True
    )

    assert len(isolated.operator_specs) == 2
    assert len(shared.operator_specs) == 1
    assert isolated.digest() != shared.digest()
    assert shared.program.instructions[1].operator_handle == 0
    assert shared.program.instructions[4].operator_handle == 0


def test_bank_composes_admitted_slots_with_provenance_and_reload(
    tmp_path: Path,
) -> None:
    controller_digest = _digest(33)
    bank = ExternalAgentBrainBank(controller_digest=controller_digest, capacity=4)
    first = _finite_executive_artifact()
    second = _finite_executive_artifact(1)
    bank.admit_executive(first, [1.0])
    bank.admit_executive(second, [1.0])

    receipt = bank.compose_executive((0, 1), [1.0])
    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    restored = ExternalAgentBrainBank.load_bank(path)

    assert receipt.accepted and receipt.slot == 2
    assert bank.program_count == 3
    assert len(bank.composition_provenance) == 1
    provenance = bank.composition_provenance[0]
    assert provenance["parent_slots"] == [0, 1]
    assert provenance["child_digest"] == bank.artifact(AGENT_BRAIN_EXECUTIVE_KIND, 2).digest()
    assert restored.digest() == bank.digest()
    assert restored.composition_provenance == bank.composition_provenance
    restored.executable(2, controller_digest=controller_digest)


def test_rejected_executive_composition_does_not_change_the_bank() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(34), capacity=4)
    bank.admit_executive(_finite_executive_artifact(), [1.0])
    bank.admit_executive(_finite_executive_artifact(1), [1.0])
    before = bank.digest()

    rejected = bank.compose_executive((0, 1), [0.0], threshold=0.9)

    assert not rejected.accepted
    assert bank.program_count == 2
    assert bank.digest() == before
    assert not bank.composition_provenance


def test_composition_provenance_cannot_rebind_parent_digests() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(35), capacity=4)
    bank.admit_executive(_finite_executive_artifact(), [1.0])
    bank.admit_executive(_finite_executive_artifact(1), [1.0])
    bank.compose_executive((0, 1), [1.0])
    payload = bank.payload()
    payload["composition_provenance"][0]["parent_digests"][0] = bank.artifact(
        AGENT_BRAIN_EXECUTIVE_KIND, 1
    ).digest()

    with pytest.raises(ValueError, match="parent digest binding"):
        ExternalAgentBrainBank.from_payload(payload)


def test_composition_provenance_binds_child_derivation_and_admission() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(351), capacity=4)
    bank.admit_executive(_finite_executive_artifact(), [1.0])
    bank.admit_executive(_finite_executive_artifact(1), [1.0])
    bank.compose_executive((0, 1), [1.0])

    wrong_child = bank.payload()
    wrong_child["composition_provenance"][0]["child_digest"] = bank.artifact(
        AGENT_BRAIN_EXECUTIVE_KIND, 0
    ).digest()
    content = {key: value for key, value in wrong_child.items() if key != "sha256"}
    wrong_child["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="child derivation"):
        ExternalAgentBrainBank.from_payload(wrong_child)

    wrong_admission = bank.payload()
    wrong_admission["composition_provenance"][0]["admission"]["slot"] = 0
    content = {key: value for key, value in wrong_admission.items() if key != "sha256"}
    wrong_admission["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="admission binding"):
        ExternalAgentBrainBank.from_payload(wrong_admission)


def test_composition_provenance_and_payload_are_isolated_snapshots() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(352), capacity=4)
    bank.admit_executive(_finite_executive_artifact(), [1.0])
    bank.admit_executive(_finite_executive_artifact(1), [1.0])
    bank.compose_executive((0, 1), [1.0])
    before = bank.digest()

    provenance = bank.composition_provenance
    provenance[0]["parent_slots"][0] = 1
    payload = bank.payload()
    payload["composition_provenance"][0]["admission"]["slot"] = 0

    assert bank.digest() == before
    assert bank.composition_provenance[0]["parent_slots"] == [0, 1]
    assert bank.composition_provenance[0]["admission"]["slot"] == 2


def test_opaque_composition_search_appends_first_stable_child() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(36), capacity=6)
    for variant in range(3):
        bank.admit_executive(_finite_executive_artifact(variant), [1.0])
    search = ExternalExecutiveCompositionSearch(bank, seed=19)
    candidates = search.candidate_parent_slots()
    target = candidates[1]
    evaluated: list[tuple[int, int]] = []

    def evaluate(
        parent_slots: tuple[int, int],
        child: ExternalExecutiveProgramArtifact,
        stage: int,
    ):
        assert child.digest()
        if stage == 0:
            evaluated.append(parent_slots)
        return ExecutiveCompositionEvaluation(
            outcomes=(1.0,) if parent_slots == target else (0.0,),
            unique_verifier_bits=10,
            unique_logical_lifetimes=1,
        )

    result = search.search(
        evaluate,
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
        screening_threshold=0.9,
    )

    assert result.accepted
    assert result.parent_slots == target
    assert result.receipt is not None and result.receipt.accepted
    assert result.candidate_count == 4
    assert tuple(evaluated) == (candidates[0], candidates[1])
    assert result.unique_verifier_bits == 30
    assert result.unique_logical_lifetimes == 3
    assert result.replayed_examples == 0
    assert result.stable_bits_to_threshold == 30
    assert result.bank_digest_after == bank.digest()
    assert bank.program_count == 4
    assert result.payload()["attempted_parent_slots"] == [list(candidates[0]), list(candidates[1])]
    assert result.payload()["attempted_evaluation_stages"] == [1, 2]


def test_rejected_composition_search_is_memory_side_noop() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(37), capacity=6)
    for variant in range(3):
        bank.admit_executive(_finite_executive_artifact(variant), [1.0])
    before = bank.digest()
    search = ExternalExecutiveCompositionSearch(bank, seed=23)

    result = search.search(
        lambda parent_slots, child, stage: ExecutiveCompositionEvaluation(
            outcomes=(0.0,),
            unique_verifier_bits=10,
            unique_logical_lifetimes=1,
        ),
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
        screening_threshold=0.9,
    )

    assert not result.accepted
    assert result.parent_slots is None and result.receipt is None
    assert result.candidate_count == 4
    assert result.unique_verifier_bits == 40
    assert result.bank_digest_before == before == result.bank_digest_after
    assert bank.program_count == 3

    with pytest.raises(TypeError, match="explicit evaluation accounting"):
        search.search(lambda parent_slots, child, stage: [[1.0, 1.0]])
    with pytest.raises(ValueError, match="screening threshold"):
        search.search(
            lambda parent_slots, child, stage: ExecutiveCompositionEvaluation(
                outcomes=(1.0,),
                unique_verifier_bits=1,
                unique_logical_lifetimes=1,
            ),
            threshold=0.8,
            screening_threshold=0.9,
        )
    assert bank.digest() == before


def test_composition_search_order_is_invariant_to_bank_slot_order() -> None:
    artifacts = tuple(_finite_executive_artifact(variant) for variant in range(3))
    first = ExternalAgentBrainBank(controller_digest=_digest(38), capacity=6)
    second = ExternalAgentBrainBank(controller_digest=_digest(38), capacity=6)
    for artifact in artifacts:
        first.admit_executive(artifact, [1.0])
    for artifact in reversed(artifacts):
        second.admit_executive(artifact, [1.0])

    def ordered_parent_digests(bank: ExternalAgentBrainBank):
        return tuple(
            tuple(
                bank.artifact(AGENT_BRAIN_EXECUTIVE_KIND, slot).digest()
                for slot in pair
            )
            for pair in ExternalExecutiveCompositionSearch(
                bank, seed=29
            ).candidate_parent_slots()
        )

    assert ordered_parent_digests(first) == ordered_parent_digests(second)


def test_mixed_agent_brain_round_trip_preserves_both_skill_families(
    tmp_path: Path,
) -> None:
    controller_digest = _digest()
    bank = ExternalAgentBrainBank(controller_digest=controller_digest, capacity=4)
    executive = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    executive_receipt = bank.admit_executive(
        executive,
        [1.0, 1.0],
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
    )

    legacy = ExternalTemporalProgramBank(
        4,
        3,
        controller_digest=controller_digest,
        min_mastery_observations=3,
    )
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    temporal_receipt = legacy.admit(
        _temporal_artifact((5.0, -3.0, -3.0)),
        context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    bank.import_temporal_bank(legacy)

    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    payload = json.loads(path.read_text())
    restored = ExternalAgentBrainBank.load_bank(path)

    assert payload["schema"] == EXTERNAL_AGENT_BRAIN_BANK_SCHEMA
    assert payload["temporal_bank"]["artifacts"][0]["codes"]["data_b64"]
    assert executive_receipt.accepted and executive_receipt.slot == 0
    assert temporal_receipt.accepted and temporal_receipt.slot == 0
    assert bank.program_count == 2
    assert bank.executive_program_count == 1
    assert bank.temporal_program_count == 1
    assert restored.digest() == bank.digest()
    assert restored.manifest() == bank.manifest()
    assert restored.artifact(AGENT_BRAIN_EXECUTIVE_KIND, 0).digest() == executive.digest()
    assert restored.artifact(AGENT_BRAIN_TEMPORAL_KIND, 0).digest() == legacy.artifact(0).digest()
    assert restored.temporal_bank is not None
    assert restored.temporal_bank.select(context).slot == 0
    restored.executable(0, controller_digest=controller_digest)


def test_executive_route_evidence_round_trip_and_context_specific_preferences(
    tmp_path: Path,
) -> None:
    controller_digest = _digest(44)
    bank = ExternalAgentBrainBank(controller_digest=controller_digest, capacity=4)
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=1),
        [1.0],
    )
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=2),
        [1.0],
    )
    context_a = torch.tensor([1.0, 0.0, 0.0])
    context_b = torch.tensor([0.0, 1.0, 0.0])
    route = bank.executive_route_evidence(
        3,
        min_mastery_observations=2,
    )
    for _ in range(2):
        bank.observe_executive_route(context_a, 1, 1.0)
        bank.observe_executive_route(context_b, 0, 1.0)

    assert route.preferred_slots(torch.stack((context_a, context_b))).tolist() == [1, 0]
    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    restored = ExternalAgentBrainBank.load_bank(path)

    assert restored.executive_route is not None
    assert restored.executive_route.preferred_slots(
        torch.stack((context_a, context_b))
    ).tolist() == [1, 0]
    assert restored.digest() == bank.digest()


def test_legacy_temporal_bank_requires_explicit_migration(tmp_path: Path) -> None:
    controller_digest = _digest(9)
    legacy = ExternalTemporalProgramBank(4, 3, controller_digest=controller_digest)
    legacy.admit(
        _temporal_artifact((5.0, -3.0, -3.0)),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        [1.0] * 8,
    )
    source = tmp_path / "legacy.bank"
    destination = tmp_path / "AgentBrain.bank"
    legacy.save_bank(source)

    with pytest.raises(ValueError, match="not loaded implicitly"):
        ExternalAgentBrainBank.load_bank(source)

    migrated = ExternalAgentBrainBank.migrate_legacy_temporal_bank(source, destination)
    restored = ExternalAgentBrainBank.load_bank(destination)
    assert migrated.digest() == restored.digest()
    assert restored.temporal_program_count == 1
    assert restored.executive_program_count == 0


def test_old_executive_json_bank_is_wrapped_without_losing_artifacts(tmp_path: Path) -> None:
    old = ExternalExecutiveProgramBank(controller_digest=_digest(), capacity=2)
    artifact = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    old.admit(artifact, [1.0])
    path = tmp_path / "old-executive.bank"
    old.save_bank(path)

    wrapped = ExternalAgentBrainBank.load_bank(path)

    assert wrapped.controller_digest == old.controller_digest
    assert wrapped.executive_program_count == 1
    assert wrapped.artifact(AGENT_BRAIN_EXECUTIVE_KIND, 0).digest() == artifact.digest()


def test_mixed_bank_capacity_and_controller_binding_are_fail_closed() -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest(), capacity=1)
    artifact = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    accepted = bank.admit_executive(artifact, [1.0])
    rejected = bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=2),
        [1.0],
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert bank.program_count == 1
    with pytest.raises(ValueError, match="controller digest is incompatible"):
        bank.executable(0, controller_digest=_digest(99))


def test_unified_bank_checksum_rejects_tampering(tmp_path: Path) -> None:
    bank = ExternalAgentBrainBank(controller_digest=_digest())
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=1), [1.0]
    )
    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    path.write_text(path.read_text() + "tamper")

    with pytest.raises(ValueError, match="file checksum mismatch"):
        ExternalAgentBrainBank.load_bank(path)
