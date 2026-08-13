from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from experiments.compute_candidate_screen_amodal.train import (
    _candidate_key_diagnostics,
)
from neural_computer import (
    AdaptiveOnlineEpisodicRelationReader,
    AppendOnlyLearnedComputeCandidateScreen,
    EpisodicBindingArchive,
    EpisodicBindingArtifactIndex,
    EpisodicBindingRouter,
    EpisodicContextEncoder,
    EpisodicCreditHead,
    EpisodicIntentAdapter,
    ExternalCapabilityComposition,
    ExternalCapabilityPipeline,
    ExternalCapabilityProgram,
    ExternalCapabilityResidualComputeBank,
    ExternalCapabilityReusableComputeLibrary,
    ExternalCapabilitySharedResidualBank,
    ExternalComputeCandidateScreen,
    ExternalFastWeightCapabilityProgram,
    ExternalProgramFastCell,
    ExternalWorkingMemoryCell,
    IntentEvent,
    LearnedComputeCandidateScreen,
    LearnedOpaqueCandidateKeyMemory,
    OnlineEpisodicRelationReader,
    OpaqueCandidateIdentityView,
    OpaqueCandidateSignatureNormalizer,
    PageLocalLearnedComputeCandidateScreen,
    episodic_context_contrastive_loss,
    paired_event_credit_loss,
    select_reusable_binding,
    select_reusable_compute_slot,
    select_reusable_compute_slot_by_efficiency,
)


def test_episodic_binding_archive_retains_evicted_records_and_round_trips() -> None:
    archive = EpisodicBindingArchive(
        context_width=4,
        signature_width=5,
        active_slots=2,
        matching_threshold=0.9,
        min_mastery_observations=2,
    )
    key_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    key_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    key_c = torch.tensor([0.0, 0.0, 1.0, 0.0])
    signature_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    signature_b = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    signature_c = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    binding_a = archive.register(key_a, signature_a)
    binding_b = archive.register(key_b, signature_b)
    archive.activate(binding_a, 0)
    archive.activate(binding_b, 1)
    archive.observe(binding_a, 1.0, step=0)
    archive.observe(binding_a, 1.0, step=1)
    assert archive.is_protected(binding_a)

    binding_c = archive.register(key_c, signature_c)
    archive.activate(binding_c, 1)
    lookup_b = archive.lookup(signature_b)
    assert lookup_b.binding_id == binding_b
    assert lookup_b.active_slot is None
    assert archive.active_binding_ids == (binding_a, binding_c)
    assert archive.record_count == 3

    restored = EpisodicBindingArchive.from_payload(archive.payload())
    assert restored.active_binding_ids == (binding_a, binding_c)
    assert restored.lookup(signature_b).binding_id == binding_b
    assert restored.lookup(signature_b).active_slot is None
    assert restored.is_protected(binding_a)
    assert restored.configuration()["schema"] == (
        "neural-computer.episodic-binding-archive.v2"
    )


def test_episodic_binding_archive_batches_lookup_and_rejects_corruption() -> None:
    archive = EpisodicBindingArchive(
        context_width=3,
        signature_width=4,
        active_slots=2,
        matching_threshold=0.95,
        min_mastery_observations=2,
        reversal_threshold=0.5,
        reversal_patience=2,
    )
    keys = [
        torch.eye(3)[0],
        torch.eye(3)[1],
        torch.eye(3)[2],
    ]
    signatures = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    ]
    records = [archive.register(key, signature) for key, signature in zip(keys, signatures)]
    archive.activate(records[0], 0)
    archive.activate(records[1], 1)
    archive.observe(records[0], 1.0, step=0)
    archive.observe(records[0], 1.0, step=1)
    assert archive.is_protected(records[0])
    archive.observe(records[0], 0.0, step=2)
    archive.observe(records[0], 0.0, step=3)
    status = archive.status()
    assert status.reversal_count[records[0]] == 1
    assert not archive.is_protected(records[0])

    batch = archive.lookup_many(torch.stack((signatures[0], signatures[1], signatures[2])))
    assert tuple(result.binding_id for result in batch) == (
        records[0],
        records[1],
        records[2],
    )
    corrupted = archive.payload()
    corrupted["signature_keys"][0][0] += 0.125
    with pytest.raises(ValueError, match="checksum"):
        EpisodicBindingArchive.from_payload(corrupted)


def test_episodic_binding_archive_compact_snapshot_round_trip_and_corruption() -> None:
    archive = EpisodicBindingArchive(
        context_width=3,
        signature_width=4,
        active_slots=2,
        min_mastery_observations=2,
    )
    record_a = archive.register(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    record_b = archive.register(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    archive.activate(record_a, 0)
    archive.activate(record_b, 1)
    archive.observe(record_a, 1.0, step=0)
    archive.observe(record_a, 1.0, step=1)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archive.pt"
        archive.snapshot(path)
        restored = EpisodicBindingArchive(
            context_width=3,
            signature_width=4,
            active_slots=2,
            min_mastery_observations=2,
        )
        restored.load_snapshot(path)
        assert restored.payload() == archive.payload()
        assert restored.active_binding_ids == (record_a, record_b)
        corrupted = torch.load(path, weights_only=False)
        corrupted["state_dict"]["attempts"][0] += 1
        torch.save(corrupted, path)
        with pytest.raises(ValueError, match="checksum"):
            restored.load_snapshot(path)


def test_episodic_binding_archive_json_round_trip_preserves_learned_keys() -> None:
    archive = EpisodicBindingArchive(
        context_width=5,
        signature_width=5,
        active_slots=2,
    )
    key = torch.tensor([0.17, -0.31, 0.44, 0.52, -0.63])
    archive.register(key, key)

    restored = EpisodicBindingArchive.from_payload(archive.payload())

    assert restored.payload() == archive.payload()


def test_episodic_binding_artifact_index_reactivates_opaque_file_handles() -> None:
    index = EpisodicBindingArtifactIndex.create(
        context_width=3,
        signature_width=4,
        active_slots=2,
        matching_threshold=0.9,
    )
    contexts = torch.eye(3)
    signatures = torch.eye(4)[:3]
    handles = ("sha256:artifact-a", "logical-file-b", "opaque-file-c")
    binding_ids = [
        index.register(context, signature, handle)
        for context, signature, handle in zip(contexts, signatures, handles, strict=True)
    ]
    index.activate(binding_ids[0], 0)
    index.activate(binding_ids[1], 1)
    lookup_c = index.lookup(signatures[2])
    assert lookup_c.binding_id == binding_ids[2]
    assert lookup_c.artifact_handle == handles[2]
    assert lookup_c.active_slot is None
    request = index.activate(binding_ids[2], 1)
    assert request.artifact_handle == handles[2]
    assert request.active_slot == 1
    assert index.lookup(signatures[1]).active_slot is None

    restored = EpisodicBindingArtifactIndex.from_payload(index.payload())
    assert restored.lookup(signatures[2]) == index.lookup(signatures[2])
    assert restored.active_binding_ids == index.active_binding_ids
    corrupted = index.payload()
    corrupted["artifact_handles"] = list(corrupted["artifact_handles"])
    corrupted["artifact_handles"][0] = "sha256:tampered"
    with pytest.raises(ValueError, match="checksum"):
        EpisodicBindingArtifactIndex.from_payload(corrupted)


def test_episodic_artifact_reactivation_is_retention_gated_and_copy_on_write() -> None:
    index = EpisodicBindingArtifactIndex.create(
        context_width=3,
        signature_width=4,
        active_slots=2,
        min_mastery_observations=1,
    )
    keys = torch.eye(3)
    signatures = torch.eye(4)[:3]
    handles = ("file-a", "file-b", "file-c")
    binding_ids = [
        index.register(context, signature, handle)
        for context, signature, handle in zip(keys, signatures, handles, strict=True)
    ]
    index.activate(binding_ids[0], 0)
    index.activate(binding_ids[1], 1)
    index.archive.observe(binding_ids[0], 1.0, step=0)
    before = index.payload()

    protected = index.reactivate_verified(binding_ids[2], 0, lambda _: True)
    assert not protected.accepted
    assert "protected" in protected.reason
    assert index.payload() == before

    failed = index.reactivate_verified(binding_ids[2], 1, lambda _: False)
    assert not failed.accepted
    assert failed.reason == "held-out retention probe failed"
    assert index.payload() == before

    mutated = index.reactivate_verified(
        binding_ids[2],
        1,
        lambda candidate: (
            candidate.archive.observe(binding_ids[2], 1.0, step=1) or True
        ),
    )
    assert not mutated.accepted
    assert "mutated" in mutated.reason
    assert index.payload() == before

    accepted = index.reactivate_verified(
        binding_ids[2],
        1,
        lambda candidate: (
            candidate.lookup(signatures[2]).artifact_handle == handles[2]
        ),
    )
    assert accepted.accepted
    assert accepted.destination_version > accepted.source_version
    assert index.lookup(signatures[2]).active_slot == 1
    assert index.lookup(signatures[1]).active_slot is None
    assert index.lookup(signatures[0]).active_slot == 0


def test_episodic_context_encoder_masks_padding_and_normalizes_context() -> None:
    encoder = EpisodicContextEncoder(4, 3, hidden=8, context_width=6)
    events = torch.randn(2, 4, 4)
    actions = torch.randn(2, 4, 3)
    outcomes = torch.randn(2, 4)
    present = torch.tensor([[True, True, False, False], [True, True, True, False]])

    output = encoder(events, actions, outcomes, present)

    assert output.context.shape == (2, 6)
    assert torch.allclose(output.context.norm(dim=-1), torch.ones(2))
    assert torch.equal(output.credit_weights[~present], torch.zeros(3))
    assert torch.allclose(
        (output.credit_weights * present).sum(dim=-1),
        torch.ones(2),
    )


def test_episodic_binding_router_keeps_opaque_slots_permutation_equivariant() -> None:
    router = EpisodicBindingRouter(
        event_width=4,
        action_width=2,
        hidden=8,
        context_width=6,
        max_slots=2,
    )
    first = torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0, 0, 0]), dim=0)
    second = torch.nn.functional.normalize(torch.tensor([0.0, 1.0, 0, 0, 0, 0]), dim=0)
    assert router.add_slot(first) == 0
    assert router.add_slot(second) == 1

    context = torch.stack((first, second))
    route = router.route(context)
    permuted = router.route(context, slot_order=torch.tensor([1, 0]))

    assert torch.equal(route.selected_slot, torch.tensor([0, 1]))
    assert torch.equal(permuted.selected_slot, torch.tensor([1, 0]))
    assert torch.equal(route.known, torch.tensor([True, True]))
    assert router.configuration()["schema"] == (
        "neural-computer.episodic-binding-router.v3"
    )


def test_episodic_binding_router_adapts_from_attempted_scalar_utility() -> None:
    router = EpisodicBindingRouter(
        event_width=3,
        action_width=2,
        hidden=8,
        context_width=5,
    )
    events = torch.randn(1, 2, 3)
    actions = torch.zeros(1, 2, 2)
    outcomes = torch.zeros(1, 2)
    with torch.no_grad():
        first = router.encode(events, actions, outcomes)[0]
        second = torch.roll(first, shifts=1, dims=0)
    router.add_slot(first)
    router.add_slot(second)
    optimizer = torch.optim.SGD(router.trainable_parameters(), lr=0.01)
    context = router.encode(events, actions, outcomes)
    loss = router.adaptation_step(
        context,
        selected_slot=0,
        verifier_utility=1.0,
        optimizer=optimizer,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert any(parameter.grad is not None for parameter in router.encoder.parameters())


def test_episodic_binding_router_detects_unknown_and_replaces_copy_on_write() -> None:
    router = EpisodicBindingRouter(
        event_width=4,
        action_width=2,
        hidden=8,
        context_width=4,
        max_slots=2,
        route_threshold=0.75,
    )
    key_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    key_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    key_c = torch.tensor([0.0, 0.0, 1.0, 0.0])
    key_d = torch.tensor([0.0, 0.0, 0.0, 1.0])
    router.add_slot(key_a)
    router.add_slot(key_b)

    unknown = router.route(key_c.unsqueeze(0))
    assert torch.equal(unknown.known, torch.tensor([False]))
    with pytest.raises(RuntimeError, match="capacity"):
        router.add_slot(key_c)

    before = router.slot_keys[1].detach().clone()
    candidate = router.slot_replacement_candidate(1, key_c)
    assert torch.equal(router.slot_keys[1], before)
    assert not router.replace_slot_from_candidate(
        candidate,
        1,
        retention_probe=lambda _: False,
    )
    assert torch.equal(router.slot_keys[1], before)
    assert router.replace_slot_from_candidate(
        candidate,
        1,
        retention_probe=lambda proposal: bool(
            proposal.route(key_a.unsqueeze(0)).known.item()
            and proposal.route(key_c.unsqueeze(0)).known.item()
        ),
    )
    assert torch.equal(router.slot_keys[0], key_a)
    assert torch.equal(router.slot_keys[1], key_c)

    tampered = router.slot_replacement_candidate(1, key_d)
    with torch.no_grad():
        next(tampered.encoder.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="encoder"):
        router.replace_slot_from_candidate(tampered, 1, retention_probe=lambda _: True)


def test_episodic_binding_signature_preserves_novelty_and_permutation() -> None:
    router = EpisodicBindingRouter(
        event_width=2,
        action_width=1,
        hidden=8,
        context_width=4,
        max_slots=2,
        route_threshold=0.75,
        signature_weight=1.0,
    )
    events = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[-1.0, 0.0], [-1.0, 0.0]],
        ]
    )
    actions = torch.zeros(3, 2, 1)
    outcomes = torch.zeros(3, 2)
    encoded = router.encode_binding(events, actions, outcomes)
    assert encoded.signature.shape == (3, 2 * 2 + 1 + 2)
    router.add_slot(encoded.context[0], encoded.signature[0])
    router.add_slot(encoded.context[1], encoded.signature[1])

    known = router.route(
        encoded.context[:2],
        signature=encoded.signature[:2],
    )
    novel = router.route(
        encoded.context[2:3],
        signature=encoded.signature[2:3],
    )
    permuted = router.route(
        encoded.context[:2],
        signature=encoded.signature[:2],
        slot_order=torch.tensor([1, 0]),
    )

    assert torch.equal(known.selected_slot, torch.tensor([0, 1]))
    assert torch.equal(known.known, torch.tensor([True, True]))
    assert torch.equal(novel.known, torch.tensor([False]))
    assert torch.equal(permuted.selected_slot, torch.tensor([1, 0]))


def test_episodic_context_contrastive_loss_has_gradient() -> None:
    left = torch.randn(4, 8, requires_grad=True)
    right = torch.randn(4, 8)

    loss = episodic_context_contrastive_loss(left, right)
    loss.backward()

    assert loss.ndim == 0
    assert left.grad is not None
    assert torch.isfinite(left.grad).all()


def test_external_credit_head_can_be_trained_without_changing_context_encoder() -> None:
    encoder = EpisodicContextEncoder(4, 2, hidden=8, context_width=6)
    head = EpisodicCreditHead(hidden=8, context_width=6)
    events = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 2)
    outcomes = torch.zeros(2, 3)
    present = torch.ones(2, 3, dtype=torch.bool)
    output = encoder(events, actions, outcomes, present)
    before = output.context.detach().clone()

    logits = head(output.sequence.detach(), output.context.detach(), outcomes, present)
    loss, _ = paired_event_credit_loss(
        logits,
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        present=present,
    )
    loss.backward()

    assert torch.equal(before, output.context)
    assert head.network[-1].weight.grad is not None


def test_online_context_step_matches_full_episode_prefix() -> None:
    encoder = EpisodicContextEncoder(4, 2, hidden=8, context_width=6)
    events = torch.randn(2, 3, 4)
    actions = torch.randn(2, 3, 2)
    outcomes = torch.randn(2, 3)
    full = encoder(events, actions, outcomes)
    state = encoder.initial_state(2, device="cpu")
    online = None
    for index in range(3):
        online, state = encoder.step(
            events[:, index],
            actions[:, index],
            outcomes[:, index],
            state,
        )
    assert online is not None
    assert torch.allclose(online.context, full.context)


def test_episodic_intent_adapter_is_behavior_preserving_at_initialization() -> None:
    adapter = EpisodicIntentAdapter(context_width=6, intention_width=4, hidden=8)
    intention = IntentEvent(torch.randn(2, 4))
    adapted = adapter(intention, torch.randn(2, 6))
    assert torch.allclose(adapted.payload, intention.payload)
    assert adapter.configuration()["schema"] == (
        "neural-computer.episodic-intent-adapter.v1"
    )


def test_external_capability_program_keeps_state_outside_controller() -> None:
    program = ExternalCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        context_hidden=8,
        context_width=5,
        adapter_hidden=7,
    )
    intention = IntentEvent(torch.randn(3, 6))
    state = program.initial_state(3, device="cpu")
    adapted, next_state = program.step(
        event=torch.randn(3, 4),
        action=torch.zeros(3, 2),
        outcome=torch.zeros(3),
        intention=intention,
        state=state,
    )

    assert torch.equal(adapted.payload, intention.payload)
    assert next_state.context.shape == (3, 8)
    assert program.configuration()["schema"] == (
        "neural-computer.external-capability.v1"
    )


def test_fast_weight_capability_binds_external_state_to_intention_bus() -> None:
    program = ExternalFastWeightCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        key_width=8,
        query_hidden=10,
        fast_weight_hidden=8,
    )
    event = torch.randn(2, 4)
    intention = IntentEvent(torch.zeros(2, 6))
    state = program.initial_state(2, device="cpu")
    action = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    _, written = program.step(
        event=event,
        action=action,
        outcome=torch.ones(2),
        intention=intention,
        state=state,
    )
    _, missing = program.step(
        event=event,
        action=-action,
        outcome=torch.ones(2),
        intention=intention,
        state=written,
        present=torch.zeros(2, dtype=torch.bool),
    )

    assert bool(torch.any(written.weights != 0.0))
    assert torch.equal(missing.weights, written.weights)
    assert program.configuration()["schema"] == (
        "neural-computer.external-capability-fast-weight.v1"
    )


def test_external_program_fast_cell_isolated_outcome_only_and_persistent() -> None:
    torch.manual_seed(915)
    cell = ExternalProgramFastCell(
        event_width=4,
        action_width=3,
        intention_width=6,
        register_width=8,
        key_width=5,
        query_hidden=8,
        adapter_hidden=7,
        fast_weight_hidden=6,
    )
    event = torch.randn(2, 4)
    action = torch.randn(2, 3)
    intention = IntentEvent(torch.zeros(2, 6))
    state = cell.initial_state(2, device="cpu")
    initial_context = cell.read(state, event, intention)
    context, written = cell.step(
        event=event,
        action=action,
        outcome=torch.ones(2),
        intention=intention,
        state=state,
    )
    failed = cell.fast_weight.update(
        written,
        cell._query(event, intention),
        cell.value_encoder(-action),
        torch.zeros(2),
    )
    missing = cell.fast_weight.update(
        written,
        cell._query(event, intention),
        cell.value_encoder(-action),
        torch.ones(2),
        present=torch.zeros(2, dtype=torch.bool),
    )
    restored = cell.state_from_payload(cell.state_payload(written))

    assert torch.equal(initial_context, torch.zeros_like(initial_context))
    assert torch.equal(context, initial_context)
    assert bool(torch.any(written.weights != 0.0))
    assert torch.equal(failed.weights, written.weights)
    assert torch.equal(missing.weights, written.weights)
    assert torch.equal(restored.weights, written.weights)
    assert torch.equal(restored.updates, written.updates)
    assert cell.configuration()["schema"] == (
        "neural-computer.external-program-fast-cell.v1"
    )


def test_shared_residual_bank_grows_without_changing_old_slot_or_state() -> None:
    bank = ExternalCapabilitySharedResidualBank(
        event_width=4,
        action_width=2,
        intention_width=6,
        slot_count=1,
        context_hidden=8,
        context_width=5,
        adapter_hidden=7,
    )
    old_shared = {
        name: value.detach().clone()
        for name, value in bank.shared_context_encoder.state_dict().items()
    }
    old_residual = {
        name: value.detach().clone()
        for name, value in bank.residual_slots[0].state_dict().items()
    }

    assert bank.add_slot() == 1
    assert bank.slot_count == 2
    assert all(
        torch.equal(value, bank.shared_context_encoder.state_dict()[name])
        for name, value in old_shared.items()
    )
    assert all(
        torch.equal(value, bank.residual_slots[0].state_dict()[name])
        for name, value in old_residual.items()
    )

    state = bank.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }
    slot_adapted, slot_next = bank.step_slot(
        0,
        kwargs["event"],
        kwargs["action"],
        kwargs["outcome"],
        intention=kwargs["intention"],
        state=state.programs[0],
    )
    adapted, next_state = bank.step(slot_index=0, **kwargs)

    assert torch.equal(slot_adapted.payload, adapted.payload)
    assert torch.equal(slot_next.context, next_state.programs[0].context)
    assert torch.equal(adapted.payload, kwargs["intention"].payload)
    assert not torch.equal(next_state.programs[0].context, state.programs[0].context)
    assert torch.equal(next_state.programs[1].context, state.programs[1].context)
    assert bank.configuration()["schema"] == (
        "neural-computer.external-capability-shared-residual.v1"
    )
    bank.freeze_shared_base()
    assert all(
        not parameter.requires_grad
        for parameter in bank.shared_context_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in bank.residual_slots[1].parameters())
    bank.freeze_slot(0)
    assert all(
        not parameter.requires_grad for parameter in bank.residual_slots[0].parameters()
    )


def test_residual_compute_bank_adds_local_recurrent_capacity() -> None:
    bank = ExternalCapabilityResidualComputeBank(
        event_width=4,
        action_width=2,
        intention_width=6,
        slot_count=1,
        shared_context_hidden=8,
        shared_context_width=5,
        residual_context_hidden=3,
        residual_context_width=2,
        adapter_hidden=7,
    )
    old_shared = {
        name: value.detach().clone()
        for name, value in bank.shared_context_encoder.state_dict().items()
    }
    old_slot = {
        name: value.detach().clone()
        for name, value in bank.residual_slots[0].state_dict().items()
    }
    assert bank.add_slot() == 1
    assert all(
        torch.equal(value, bank.shared_context_encoder.state_dict()[name])
        for name, value in old_shared.items()
    )
    assert all(
        torch.equal(value, bank.residual_slots[0].state_dict()[name])
        for name, value in old_slot.items()
    )
    state = bank.initial_state(2, device="cpu")
    event = torch.randn(2, 4)
    action = torch.zeros(2, 2)
    outcome = torch.zeros(2)
    intention = IntentEvent(torch.randn(2, 6))
    adapted, next_state = bank.step_slot(
        slot_index=0,
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state.programs[0],
    )
    assert adapted.payload.shape == (2, 6)
    assert next_state.context.shape == (2, 11)
    assert not torch.equal(next_state.context, state.programs[0].context)
    bank.freeze_shared_base()
    bank.freeze_slot(0)
    assert all(
        not parameter.requires_grad
        for parameter in bank.shared_context_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in bank.residual_slots[0].parameters()
    )
    assert bank.configuration()["schema"] == (
        "neural-computer.external-capability-residual-compute.v1"
    )


def test_reusable_compute_library_shares_physical_compute_with_isolated_bindings() -> None:
    library = ExternalCapabilityReusableComputeLibrary(
        event_width=4,
        action_width=2,
        intention_width=6,
        compute_slot_count=1,
        binding_compute_slots=(0,),
        shared_context_hidden=8,
        shared_context_width=5,
        residual_context_hidden=3,
        residual_context_width=2,
        adapter_hidden=7,
    )
    old_compute = {
        name: value.detach().clone()
        for name, value in library.compute_slots[0].state_dict().items()
    }
    assert library.add_binding(0) == 1
    assert library.slot_count == 2
    assert library.compute_slot_count == 1
    assert library.binding_compute_slots == (0, 0)
    assert all(
        torch.equal(value, library.compute_slots[0].state_dict()[name])
        for name, value in old_compute.items()
    )
    state = library.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
    }
    first, first_state = library.step_binding(
        binding_index=0,
        state=state.programs[0],
        **kwargs,
    )
    second, second_state = library.step_binding(
        binding_index=1,
        state=state.programs[1],
        **kwargs,
    )
    assert torch.equal(first.payload, second.payload)
    assert not torch.equal(first_state.context, state.programs[0].context)
    assert not torch.equal(second_state.context, state.programs[1].context)
    library.freeze_shared_base()
    library.freeze_compute_slot(0)
    library.freeze_binding(0)
    assert all(
        not parameter.requires_grad
        for parameter in library.shared_context_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in library.compute_slots[0].parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in library.binding_adapters[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in library.binding_adapters[1].parameters()
    )
    assert library.configuration()["schema"] == (
        "neural-computer.external-capability-reusable-compute.v1"
    )
    library.remove_binding(1)
    assert library.slot_count == 1


def test_reusable_compute_library_can_share_adapter_with_isolated_bindings() -> None:
    library = ExternalCapabilityReusableComputeLibrary(
        event_width=4,
        action_width=2,
        intention_width=6,
        compute_slot_count=1,
        binding_compute_slots=(0, 0),
        binding_adapter_slots=(0, 0),
        shared_context_hidden=8,
        shared_context_width=5,
        residual_context_hidden=3,
        residual_context_width=2,
        adapter_hidden=7,
    )
    assert library.slot_count == 2
    assert library.compute_slot_count == 1
    assert library.adapter_slot_count == 1
    assert library.binding_compute_slots == (0, 0)
    assert library.binding_adapter_slots == (0, 0)
    assert library.binding_modules(0)[1] is library.binding_modules(1)[1]
    assert library.add_binding(0, adapter_slot_index=0) == 2
    assert library.binding_adapter_slots == (0, 0, 0)
    library.remove_binding(2)
    trial_compute = library.add_compute_slot()
    trial_binding = library.add_binding(trial_compute, adapter_slot_index=0)
    library.remove_binding(trial_binding)
    library.remove_compute_slot(trial_compute)
    trial_adapter_binding = library.add_binding(0)
    trial_adapter = library.binding_adapter_slots[trial_adapter_binding]
    library.remove_binding(trial_adapter_binding)
    library.remove_adapter_slot(trial_adapter)
    assert library.compute_slot_count == 1
    assert library.binding_compute_slots == (0, 0)
    library.freeze_binding(1)
    assert all(
        not parameter.requires_grad
        for parameter in library.binding_adapters[0].parameters()
    )
    assert library.compute_slot_count == 1


def test_reusable_compute_admission_requires_every_fresh_probe() -> None:
    reuse = select_reusable_compute_slot(
        {3: (0.82, 0.91), 1: (0.84, 0.84)}, threshold=0.8
    )
    assert reuse.action == "reuse"
    assert reuse.compute_slot_index == 1
    reject = select_reusable_compute_slot({0: (0.8, 0.74)}, threshold=0.8)
    assert reject.action == "grow"
    assert reject.compute_slot_index is None
    empty = select_reusable_compute_slot({}, threshold=0.8)
    assert empty.action == "grow"
    assert empty.reason == "no_compute_candidates"


def test_efficiency_aware_compute_admission_rejects_slower_or_unstable_reuse() -> None:
    reuse = select_reusable_compute_slot_by_efficiency(
        {0: (0.91, 0.88), 1: (0.92, 0.9)},
        {0: 8_192, 1: 16_384},
        fresh_stable_bits=12_288,
        threshold=0.8,
    )
    slower = select_reusable_compute_slot_by_efficiency(
        {0: (0.91, 0.88)},
        {0: 16_384},
        fresh_stable_bits=12_288,
        threshold=0.8,
    )
    incomplete = select_reusable_compute_slot_by_efficiency(
        {0: (0.91, 0.88)},
        {0: 8_192},
        fresh_stable_bits=None,
        threshold=0.8,
    )

    assert reuse.action == "reuse"
    assert reuse.compute_slot_index == 0
    assert slower.action == "grow"
    assert incomplete.action == "grow"


def test_reusable_binding_admission_scores_compute_and_adapter_pairs() -> None:
    reuse = select_reusable_binding(
        {(2, 7): (0.82, 0.91), (1, 4): (0.84, 0.84)}, threshold=0.8
    )
    assert reuse.action == "reuse"
    assert reuse.compute_slot_index == 1
    assert reuse.adapter_slot_index == 4
    reject = select_reusable_binding({(0, 0): (0.8, 0.74)}, threshold=0.8)
    assert reject.action == "grow"
    assert reject.compute_slot_index is None
    assert reject.adapter_slot_index is None
    empty = select_reusable_binding({}, threshold=0.8)
    assert empty.action == "grow"
    assert empty.reason == "no_binding_candidates"


def test_compute_candidate_screen_orders_from_learned_event_outcomes_only() -> None:
    screen = ExternalComputeCandidateScreen(width=4)
    assert screen.add_candidate() == 0
    assert screen.add_candidate() == 1
    assert screen.add_candidate() == 2
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert screen.order(query) == (0, 1, 2)
    screen.observe(query, 2, 1.0)

    assert screen.order(query) == (2, 1, 0)
    unseen_query = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert screen.order(unseen_query) == (2, 1, 0)
    screen.observe(unseen_query, 1, 1.0)
    assert screen.order(query) == (2, 1, 0)
    assert screen.order(unseen_query) == (1, 2, 0)
    assert screen.configuration()["role"] == "order_only_fresh_admission_required"


def test_compute_candidate_screen_reloads_without_semantic_metadata() -> None:
    screen = ExternalComputeCandidateScreen(width=3, matching_tolerance=1e-3)
    screen.add_candidate()
    screen.add_candidate()
    query = torch.tensor([0.0, 1.0, 0.0])
    screen.observe(query, 1, 1.0)

    restored = ExternalComputeCandidateScreen.from_payload(screen.payload())

    assert restored.payload() == screen.payload()
    assert restored.order(query) == (1, 0)
    assert "task" not in restored.payload()
    assert "label" not in restored.payload()


def test_learned_compute_screen_is_neutral_and_permutation_equivariant() -> None:
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    query = torch.randn(2, 4)
    keys = torch.randn(3, 3)
    permutation = torch.tensor([2, 0, 1])

    scores = screen(query, keys)
    permuted_scores = screen(query, keys[permutation])

    assert torch.equal(scores, torch.zeros_like(scores))
    assert torch.allclose(permuted_scores, scores[:, permutation])
    assert screen.order(query[0], keys) == (0, 1, 2)
    assert screen.configuration()["role"] == "order_only_fresh_admission_required"


def test_candidate_key_diagnostics_detects_collapsed_signatures() -> None:
    separated = _candidate_key_diagnostics(torch.eye(3))
    collapsed = _candidate_key_diagnostics(torch.ones(3, 3))

    assert separated["max_off_diagonal_cosine"] == 0.0
    assert separated["effective_rank"] == pytest.approx(3.0)
    assert collapsed["max_off_diagonal_cosine"] == pytest.approx(1.0)
    assert collapsed["effective_rank"] == pytest.approx(1.0)


def test_learned_candidate_key_memory_appends_and_freezes_base() -> None:
    memory = LearnedOpaqueCandidateKeyMemory(3, torch.eye(3))
    memory.append_extension(torch.ones(2, 3))

    assert memory.candidate_count == 5
    assert memory().shape == (5, 3)
    memory.freeze_base()
    memory.freeze_extension(0)

    assert not memory.base_keys.requires_grad
    assert not memory.extensions[0].requires_grad
    assert memory.configuration()["role"] == (
        "opaque_address_memory_not_controller_reasoning"
    )


def test_signature_normalizer_is_fitted_once_and_permutation_invariant() -> None:
    normalizer = OpaqueCandidateSignatureNormalizer(3)
    keys = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 2.0]]
    )
    normalizer.fit(keys)
    original = normalizer(keys)
    permuted = normalizer(keys[torch.tensor([2, 0, 1])])

    assert torch.allclose(permuted[torch.tensor([1, 2, 0])], original)
    assert normalizer.configuration()["append_policy"] == (
        "never_refit_existing_address_space_v1"
    )
    with pytest.raises(RuntimeError, match="already fitted"):
        normalizer.fit(keys)


def test_page_local_screen_uses_local_rank_after_verifier_failure() -> None:
    torch.manual_seed(25)
    base_view = OpaqueCandidateSignatureNormalizer(4)
    base_keys_for_fit = torch.randn(4, 4)
    base_view.fit(base_keys_for_fit)
    screen = PageLocalLearnedComputeCandidateScreen(
        query_width=4,
        key_width=4,
        latent_width=5,
        hidden=8,
        base_query_view=base_view,
        base_key_view=base_view,
        activation_margin=1.0,
    )
    assert screen.append_extension(2, query_view=OpaqueCandidateIdentityView(4)) == 0
    screen.enable_base()
    screen.enable_extension(0)
    query = torch.randn(3, 4)
    base_keys = torch.randn(3, 4)
    extension_keys = torch.randn(2, 4)
    base_scores = screen.base_screen(base_view(query), base_view(base_keys))

    cold = screen(query, base_keys, extension_keys, failed_extensions=False)
    assert torch.equal(cold[:, :3], base_scores)
    assert torch.equal(cold.argmax(dim=-1), base_scores.argmax(dim=-1))

    failed = screen(
        query,
        base_keys,
        extension_keys,
        failed_extensions=torch.ones(3, 1, dtype=torch.bool),
    )
    local_scores = screen.extensions[0](query, extension_keys)
    assert torch.equal(failed[:, 3:].argmax(dim=-1), local_scores.argmax(dim=-1))
    assert bool((failed[:, 3:].max(dim=-1).values > base_scores.max(dim=-1).values).all())
    assert screen.configuration()["activation"] == "page_local_rank_margin_v1"


def test_page_local_screen_state_round_trips_with_independent_views() -> None:
    torch.manual_seed(26)
    normalizer = OpaqueCandidateSignatureNormalizer(4)
    normalizer.fit(torch.randn(4, 4))
    screen = PageLocalLearnedComputeCandidateScreen(
        query_width=4,
        key_width=4,
        latent_width=5,
        hidden=8,
        base_query_view=normalizer,
        base_key_view=normalizer,
    )
    screen.append_extension(1)
    screen.enable_base()
    screen.enable_extension(0)
    restored_normalizer = OpaqueCandidateSignatureNormalizer(4)
    restored_normalizer.fit(torch.randn(4, 4))
    restored = PageLocalLearnedComputeCandidateScreen(
        query_width=4,
        key_width=4,
        latent_width=5,
        hidden=8,
        base_query_view=restored_normalizer,
        base_key_view=restored_normalizer,
    )
    restored.append_extension(1)
    restored.load_state_dict(screen.state_dict(), strict=True)
    assert screen.configuration() == restored.configuration()
    query = torch.randn(2, 4)
    base_keys = torch.randn(2, 4)
    extension_keys = torch.randn(1, 4)
    failures = torch.ones(2, 1, dtype=torch.bool)
    assert torch.equal(
        screen(query, base_keys, extension_keys, failures),
        restored(query, base_keys, extension_keys, failures),
    )


def test_learned_compute_screen_ranking_loss_uses_only_scalar_outcomes() -> None:
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    query = torch.randn(2, 4)
    keys = torch.randn(3, 3)
    outcomes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    screen.enable()
    loss, informative = screen.outcome_ranking_loss(query, keys, outcomes)
    loss.backward()

    assert loss.ndim == 0
    assert informative == 4
    assert any(parameter.grad is not None for parameter in screen.parameters())


def test_learned_compute_screen_calibrates_a_single_attempted_candidate() -> None:
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    query = torch.randn(4, 4)
    keys = torch.randn(1, 3)
    attempted = torch.zeros(4, dtype=torch.long)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0])

    screen.enable()
    loss, informative = screen.outcome_calibration_loss(
        query,
        keys,
        attempted,
        outcomes,
    )
    loss.backward()

    assert loss.ndim == 0
    assert informative == 4
    assert any(parameter.grad is not None for parameter in screen.parameters())


def test_learned_compute_screen_state_round_trips() -> None:
    torch.manual_seed(11)
    screen = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    restored = LearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    restored.load_state_dict(screen.state_dict(), strict=True)
    query = torch.randn(2, 4)
    keys = torch.randn(3, 3)

    assert torch.equal(screen(query, keys), restored(query, keys))
    assert screen.configuration() == restored.configuration()


def test_append_only_learned_screen_preserves_base_until_failure() -> None:
    torch.manual_seed(19)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
    )
    screen.enable_base()
    base_before = {
        name: value.detach().clone()
        for name, value in screen.base_screen.state_dict().items()
    }
    assert screen.append_extension(2) == 0
    screen.enable_extension(0)
    query = torch.randn(3, 4)
    base_keys = torch.randn(3, 3)
    extension_keys = torch.randn(2, 3)
    base_scores = screen.base_screen(query, base_keys)

    cold = screen(query, base_keys, extension_keys)
    assert torch.equal(cold[:, :3], base_scores)
    assert torch.equal(cold.argmax(dim=-1), base_scores.argmax(dim=-1))

    failed = screen(
        query,
        base_keys,
        extension_keys,
        failed_extensions=torch.ones(3, 1, dtype=torch.bool),
    )
    assert torch.equal(failed[:, :3], base_scores)
    expected_extension = base_scores.max(dim=-1).values.unsqueeze(1) + screen.extensions[
        0
    ](query, extension_keys)
    assert torch.allclose(failed[:, 3:], expected_extension)
    assert all(
        torch.equal(value, screen.base_screen.state_dict()[name])
        for name, value in base_before.items()
    )


def test_append_only_learned_screen_can_copy_base_as_independent_prior() -> None:
    torch.manual_seed(21)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1,),
    )
    screen.enable_base()
    base_before = {
        name: value.detach().clone()
        for name, value in screen.base_screen.state_dict().items()
    }

    screen.initialize_extension_from_base(0)

    assert not bool(screen.extensions[0].enabled.item())
    assert all(
        torch.equal(value, screen.extensions[0].state_dict()[name])
        for name, value in base_before.items()
        if name != "enabled"
    )
    with torch.no_grad():
        screen.extensions[0].query_projection[0].bias.add_(1.0)
    assert all(
        torch.equal(value, screen.base_screen.state_dict()[name])
        for name, value in base_before.items()
    )


def test_append_only_learned_screen_can_copy_only_query_prior() -> None:
    torch.manual_seed(22)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1,),
    )
    screen.enable_base()
    base_state = {
        name: value.detach().clone()
        for name, value in screen.base_screen.state_dict().items()
    }
    fresh_extension_state = {
        name: value.detach().clone()
        for name, value in screen.extensions[0].state_dict().items()
    }

    screen.initialize_extension_from_base(0, mode="query_path")
    extension_state = screen.extensions[0].state_dict()

    for prefix in ("query_projection.", "router.query_encoder."):
        assert all(
            torch.equal(extension_state[name], base_state[name])
            for name in extension_state
            if name.startswith(prefix)
        )
    assert any(
        not torch.equal(extension_state[name], fresh_extension_state[name])
        for name in extension_state
        if name.startswith(("query_projection.", "router.query_encoder."))
    )
    assert all(
        torch.equal(extension_state[name], fresh_extension_state[name])
        for name in extension_state
        if name.startswith(("key_projection.", "router.key_encoder."))
    )
    assert not bool(screen.extensions[0].enabled.item())


def test_append_only_learned_screen_can_blend_query_prior() -> None:
    torch.manual_seed(23)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1,),
    )
    base_state = {
        name: value.detach().clone()
        for name, value in screen.base_screen.state_dict().items()
    }
    fresh_state = {
        name: value.detach().clone()
        for name, value in screen.extensions[0].state_dict().items()
    }

    screen.initialize_extension_from_base(
        0,
        mode="query_path",
        prior_strength=0.5,
    )
    extension_state = screen.extensions[0].state_dict()
    for prefix in ("query_projection.", "router.query_encoder."):
        for name in extension_state:
            if name.startswith(prefix):
                expected = torch.lerp(fresh_state[name], base_state[name], 0.5)
                assert torch.equal(extension_state[name], expected)
    assert all(
        torch.equal(extension_state[name], fresh_state[name])
        for name in extension_state
        if name.startswith(("key_projection.", "router.key_encoder."))
    )


def test_append_only_learned_screen_state_round_trips() -> None:
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 2),
    )
    screen.enable_base()
    screen.enable_extension(1)
    restored = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 2),
    )
    restored.load_state_dict(screen.state_dict(), strict=True)
    assert screen.configuration() == restored.configuration()
    query = torch.randn(2, 4)
    base_keys = torch.randn(2, 3)
    extension_keys = torch.randn(3, 3)
    failures = torch.tensor([[False, True], [True, False]])
    assert torch.equal(
        screen(query, base_keys, extension_keys, failures),
        restored(query, base_keys, extension_keys, failures),
    )


def test_append_only_screen_verified_consolidation_compacts_consecutive_stages() -> None:
    torch.manual_seed(24)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 1, 2),
    )
    screen.enable_base()
    screen.enable_extension(0)
    screen.enable_extension(1)
    screen.enable_extension(2)
    source_state = {
        name: value.detach().clone() for name, value in screen.state_dict().items()
    }
    replacement = LearnedComputeCandidateScreen(4, 3, latent_width=5, hidden=8)
    replacement.enable()

    compacted, receipt = screen.consolidate_verified(
        (0, 1),
        replacement,
        verifier=lambda candidate: candidate.extension_sizes == [2, 2],
    )

    assert compacted is not None
    assert receipt.accepted
    assert receipt.extensions_saved == 1
    assert compacted.extension_sizes == [2, 2]
    assert bool(compacted.extensions[0].enabled.item())
    assert all(
        torch.equal(value, screen.state_dict()[name])
        for name, value in source_state.items()
    )


def test_append_only_screen_rejected_consolidation_does_not_mutate_source() -> None:
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 1),
    )
    source_state = {
        name: value.detach().clone() for name, value in screen.state_dict().items()
    }
    replacement = LearnedComputeCandidateScreen(4, 3, latent_width=5, hidden=8)

    compacted, receipt = screen.consolidate_verified(
        (0, 1),
        replacement,
        verifier=lambda _candidate: False,
    )

    assert compacted is None
    assert not receipt.accepted
    assert receipt.extensions_saved == 0
    assert all(
        torch.equal(value, screen.state_dict()[name])
        for name, value in source_state.items()
    )


def test_append_only_learned_screen_cannot_skip_an_unfailed_prior_stage() -> None:
    torch.manual_seed(23)
    screen = AppendOnlyLearnedComputeCandidateScreen(
        query_width=4,
        key_width=3,
        latent_width=5,
        hidden=8,
        extension_sizes=(1, 1),
    )
    screen.enable_base()
    screen.enable_extension(0)
    screen.enable_extension(1)
    with torch.no_grad():
        screen.extensions[0].router.query_encoder[-1].bias.zero_()
        screen.extensions[0].router.key_encoder[-1].bias.zero_()
        screen.extensions[1].router.query_encoder[-1].bias.fill_(3.0)
        screen.extensions[1].router.key_encoder[-1].bias.fill_(3.0)
    query = torch.randn(2, 4)
    base_keys = torch.randn(2, 3)
    extension_keys = torch.randn(2, 3)

    skipped = screen(
        query,
        base_keys,
        extension_keys,
        failed_extensions=torch.tensor([[False, True], [False, True]]),
    )

    assert bool((skipped.argmax(dim=-1) < 2).all())


def test_external_capability_pipeline_keeps_program_states_independent() -> None:
    first = ExternalCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        context_hidden=8,
        context_width=5,
        adapter_hidden=7,
    )
    second = ExternalCapabilityProgram(
        event_width=4,
        action_width=2,
        intention_width=6,
        context_hidden=9,
        context_width=5,
        adapter_hidden=7,
    )
    pipeline = ExternalCapabilityPipeline((first, second))
    intention = IntentEvent(torch.randn(3, 6))
    event = torch.randn(3, 4)
    action = torch.zeros(3, 2)
    outcome = torch.zeros(3)
    state = pipeline.initial_state(3, device="cpu")

    adapted, next_state = pipeline.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state,
    )
    first_intention, first_state = first.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=state.programs[0],
    )
    expected, second_state = second.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=first_intention,
        state=state.programs[1],
    )

    assert torch.equal(adapted.payload, expected.payload)
    assert torch.equal(next_state.programs[0].context, first_state.context)
    assert torch.equal(next_state.programs[1].context, second_state.context)
    assert pipeline.configuration()["program_count"] == 2
    assert pipeline.configuration()["program_schemas"] == (
        "neural-computer.external-capability.v1",
        "neural-computer.external-capability.v1",
    )


def test_external_capability_composition_routes_external_slots_and_keeps_identity() -> None:
    programs = tuple(
        ExternalCapabilityProgram(
            event_width=4,
            action_width=2,
            intention_width=6,
            context_hidden=8,
            context_width=5,
            adapter_hidden=7,
        )
        for _ in range(2)
    )
    composition = ExternalCapabilityComposition(
        programs,
        composition_steps=2,
        router_hidden=9,
    )
    intention = IntentEvent(torch.randn(3, 6))
    state = composition.initial_state(3, device="cpu")
    adapted, next_state = composition.step(
        event=torch.randn(3, 4),
        action=torch.zeros(3, 2),
        outcome=torch.zeros(3),
        intention=intention,
        state=state,
    )

    assert torch.allclose(adapted.payload, intention.payload)
    assert len(next_state.programs) == 2
    assert all(item.context.shape == (3, 8) for item in next_state.programs)
    assert composition.configuration()["composition_steps"] == 2
    assert composition.configuration()["routing"] == (
        "learned_event_conditioned_soft_slot_binding_v1"
    )


def test_external_capability_composition_accepts_opaque_slot_binding() -> None:
    programs = tuple(
        ExternalCapabilityProgram(
            event_width=4,
            action_width=2,
            intention_width=6,
            context_hidden=8,
            context_width=5,
            adapter_hidden=7,
        )
        for _ in range(2)
    )
    composition = ExternalCapabilityComposition(programs, composition_steps=1)
    state = composition.initial_state(2, device="cpu")
    kwargs = {
        "event": torch.randn(2, 4),
        "action": torch.zeros(2, 2),
        "outcome": torch.zeros(2),
        "intention": IntentEvent(torch.randn(2, 6)),
        "state": state,
    }

    adapted, next_state = composition.step(
        **kwargs,
        slot_mask=torch.tensor([[True, False], [True, False]]),
    )

    assert adapted.payload.shape == (2, 6)
    assert torch.equal(next_state.programs[1].context, state.programs[1].context)
    assert composition.configuration()["binding"] == (
        "optional_opaque_external_slot_mask_v1"
    )
    assert composition.configuration()["execution"] == "masked_sparse_active_slots_v1"
    mixed_state = composition.initial_state(2, device="cpu")
    _, mixed_next_state = composition.step(
        **{**kwargs, "state": mixed_state},
        slot_mask=torch.tensor([[True, False], [False, True]]),
    )
    assert torch.equal(
        mixed_next_state.programs[0].context[1], mixed_state.programs[0].context[1]
    )
    assert torch.equal(
        mixed_next_state.programs[1].context[0], mixed_state.programs[1].context[0]
    )
    with pytest.raises(ValueError, match="at least one slot"):
        composition.step(**kwargs, slot_mask=torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        composition.step(**kwargs, slot_mask=torch.ones(2, 2))


def test_empty_external_capability_pipeline_is_identity() -> None:
    pipeline = ExternalCapabilityPipeline(
        event_width=4,
        action_width=2,
        intention_width=6,
    )
    intention = IntentEvent(torch.randn(3, 6))
    adapted, next_state = pipeline.step(
        event=torch.randn(3, 4),
        action=torch.zeros(3, 2),
        outcome=torch.zeros(3),
        intention=intention,
        state=pipeline.initial_state(3, device="cpu"),
    )

    assert torch.equal(adapted.payload, intention.payload)
    assert next_state.programs == ()
    assert pipeline.configuration()["program_count"] == 0


def test_external_capability_pipeline_can_hide_events_after_first_program() -> None:
    def make_programs() -> tuple[ExternalCapabilityProgram, ...]:
        return tuple(
            ExternalCapabilityProgram(
                event_width=4,
                action_width=2,
                intention_width=6,
                context_hidden=8,
                context_width=5,
                adapter_hidden=7,
            )
            for _ in range(2)
        )

    visible = ExternalCapabilityPipeline(make_programs())
    hidden = ExternalCapabilityPipeline(
        make_programs(),
        hide_downstream_events=True,
    )
    intention = IntentEvent(torch.randn(3, 6))
    event = torch.randn(3, 4)
    action = torch.zeros(3, 2)
    outcome = torch.zeros(3)
    visible_state = visible.initial_state(3, device="cpu")
    hidden_state = hidden.initial_state(3, device="cpu")
    _, visible_next = visible.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=visible_state,
    )
    _, hidden_next = hidden.step(
        event=event,
        action=action,
        outcome=outcome,
        intention=intention,
        state=hidden_state,
    )

    assert not torch.equal(
        visible_next.programs[1].context,
        hidden_next.programs[1].context,
    )
    assert hidden.configuration()["event_visibility"] == "head_only"


def test_online_relation_reader_returns_external_content_age_context() -> None:
    reader = OnlineEpisodicRelationReader(
        event_width=4,
        action_width=2,
        memory_capacity=3,
        context_width=6,
        hidden=8,
    )
    state = reader.initial_state(2, device="cpu")
    context, state = reader.step(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.zeros(2),
        state,
    )
    assert context.shape == (2, 6)
    assert state.events.shape == (2, 3, 4)
    assert bool(state.present[:, -1].all())
    assert reader.configuration()["schema"] == (
        "neural-computer.online-episodic-relation-reader.v1"
    )


def test_adaptive_relation_reader_scores_each_external_row() -> None:
    reader = AdaptiveOnlineEpisodicRelationReader(
        event_width=4,
        action_width=2,
        memory_capacity=5,
        context_width=6,
        hidden=8,
    )
    state = reader.initial_state(2, device="cpu")
    context, state = reader.step(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.zeros(2),
        state,
    )
    assert context.shape == (2, 6)
    assert state.events.shape == (2, 5, 4)
    assert reader.configuration()["schema"] == (
        "neural-computer.adaptive-online-episodic-relation-reader.v1"
    )


def test_adaptive_relation_reader_can_expand_without_losing_shared_weights() -> None:
    reader = AdaptiveOnlineEpisodicRelationReader(
        event_width=4,
        action_width=2,
        memory_capacity=5,
        context_width=6,
        hidden=8,
    )
    before = {
        name: value.detach().clone()
        for name, value in reader.named_parameters()
        if name != "age_embedding"
    }

    expanded = reader.expand_capacity(7)

    assert expanded.memory_capacity == 7
    assert expanded.age_embedding.shape == (7, 8)
    for name, value in before.items():
        assert torch.equal(value, dict(expanded.named_parameters())[name])


def test_external_working_memory_cell_reads_before_write_and_persists() -> None:
    cell = ExternalWorkingMemoryCell(
        event_width=4,
        action_width=2,
        memory_capacity=3,
        context_width=6,
        hidden=8,
    )
    state = cell.initial_state(1, device="cpu")
    first_event = torch.randn(1, 4)
    first_action = torch.tensor([[1.0, 0.0]])
    first_outcome = torch.ones(1)

    before = state.events.clone()
    before_read = cell.read(first_event, state)
    context, next_state = cell.step(
        first_event,
        first_action,
        first_outcome,
        state,
    )

    assert context.shape == (1, 6)
    assert torch.equal(state.events, before)
    assert torch.allclose(context, before_read)
    assert torch.equal(next_state.events[:, -1], first_event)
    assert bool(next_state.present[:, -1].all())

    payload = cell.state_payload(next_state)
    restored = cell.state_from_payload(payload)
    assert all(
        torch.equal(getattr(restored, name), getattr(next_state, name))
        for name in ("events", "actions", "outcomes", "present")
    )


def test_external_working_memory_cell_grows_from_the_newest_rows() -> None:
    cell = ExternalWorkingMemoryCell(
        event_width=3,
        action_width=2,
        memory_capacity=2,
        context_width=5,
        hidden=7,
    )
    state = cell.initial_state(1, device="cpu")
    rows = []
    for index in range(2):
        event = torch.full((1, 3), float(index + 1))
        rows.append(event)
        _, state = cell.step(
            event,
            torch.zeros(1, 2),
            torch.ones(1),
            state,
        )

    grown = cell.grow(4)
    grown_state = cell.grow_state(state, 4)
    assert grown.memory_capacity == 4
    assert torch.equal(grown_state.events[:, -2], rows[0])
    assert torch.equal(grown_state.events[:, -1], rows[1])
    assert not bool(grown_state.present[:, :2].any())


def test_paired_event_credit_loss_returns_detached_counterfactual_advantage() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    utilities = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        ]
    )

    loss, advantage = paired_event_credit_loss(logits, utilities)
    loss.backward()

    assert advantage.tolist() == [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]
    assert not advantage.requires_grad
    assert logits.grad is not None


def test_episodic_context_rejects_mismatched_trajectory_shapes() -> None:
    encoder = EpisodicContextEncoder(4, 2)
    with pytest.raises(ValueError, match="share batch and time"):
        encoder(torch.randn(2, 3, 4), torch.randn(2, 2, 2), torch.randn(2, 3))
