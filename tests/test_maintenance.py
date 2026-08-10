from __future__ import annotations

import torch

from experiments.external_memory_maintenance_policy.real_train import (
    run as run_real_maintenance,
)
from experiments.external_memory_maintenance_policy.train import run
from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMemoryMaintenancePolicy,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


def _router() -> ExternalLearnedMultiStreamTransitionContextRouter:
    torch.manual_seed(6101)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=8,
        context_width=4,
    )
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        hidden_width=8,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=4,
    )
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=4,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    return ExternalLearnedMultiStreamTransitionContextRouter(
        ExternalOnlineStreamBindingMemory(
            encoder,
            window_capacity=2,
            max_streams=4,
            provisional_capacity=2,
        ),
        ExternalMultiStreamTransitionContextRouter(single, stream_key_width=4),
    )


def test_external_maintenance_policy_is_masked_and_persistent() -> None:
    torch.manual_seed(6102)
    policy = ExternalMemoryMaintenancePolicy(hidden_width=12, learning_rate=0.03)
    features = torch.linspace(0.0, 1.0, 12)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.03)
    before = policy.digest()
    mask = torch.ones(4, dtype=torch.bool)
    proposal = policy.propose(
        features,
        mask,
        sample=True,
        generator=torch.Generator().manual_seed(6106),
    )
    policy.adaptation_step(proposal, 1.0, optimizer=optimizer)
    assert policy.digest() != before
    restored = ExternalMemoryMaintenancePolicy.from_payload(policy.state_payload())
    assert restored.digest() == policy.digest()
    assert restored.configuration() == policy.configuration()


def test_external_maintenance_policy_consumes_one_scalar_without_replay() -> None:
    torch.manual_seed(6103)
    policy = ExternalMemoryMaintenancePolicy(hidden_width=10, learning_rate=0.02)
    features = torch.zeros(12)
    mask = torch.ones(4, dtype=torch.bool)
    generator = torch.Generator().manual_seed(6104)
    proposal = policy.propose(features, mask, sample=True, generator=generator)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.02)
    policy.adaptation_step(proposal, torch.tensor(1.0), optimizer=optimizer)
    try:
        policy.adaptation_step(proposal, torch.tensor([1.0, 0.0]), optimizer=optimizer)
    except ValueError as error:
        assert "scalar" in str(error)
    else:
        raise AssertionError("non-scalar maintenance outcome was accepted")


def test_learned_router_exposes_generic_maintenance_and_forced_legal_actions() -> None:
    router = _router()
    policy = ExternalMemoryMaintenancePolicy(hidden_width=8)
    features = router.maintenance_features(
        redundancy_pressure=0.75,
        compression_opportunity=0.5,
    )
    assert features.shape == (12,)
    assert bool(torch.isfinite(features).all())
    for index, (action, kwargs) in enumerate(
        (
            ("grow", {"grow_available": True}),
            ("share", {"share_available": True}),
            ("compress", {"compression_available": True}),
            ("defer", {}),
        )
    ):
        with torch.no_grad():
            policy.network[-1].bias.fill_(-10.0)
            policy.network[-1].bias[index] = 10.0
        proposal = router.propose_maintenance(policy, **kwargs)
        assert proposal.action == action
        assert proposal.available_actions[proposal.action_index]
    before = router.digest()
    deferred = router.propose_maintenance(policy)
    assert deferred.action == "defer"
    assert router.apply_maintenance_proposal(deferred) is None
    assert router.digest() == before


def test_compression_commit_is_copy_on_write_and_probe_gated() -> None:
    torch.manual_seed(6105)
    bank = ExternalTransitionModelBank(
        2,
        1,
        2,
        hidden_width=8,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=2,
    )
    bank.ensure_context(torch.tensor([1.0, 0.0]))
    bank.ensure_context(torch.tensor([0.0, 1.0]))
    before = bank.content_digest()
    rejected = bank.compress_and_commit_verified(
        dtype=torch.float16,
        retention_probe=lambda _candidate: False,
    )
    assert not rejected.accepted
    assert bank.content_digest() == before
    accepted = bank.compress_and_commit_verified(
        dtype=torch.float16,
        retention_probe=lambda candidate: candidate.context_count == 2,
    )
    assert accepted.accepted
    assert bank.digest() == accepted.candidate_digest
    assert bank.context_count == 2


def test_bank_growth_and_consolidation_reject_mutating_probes_atomically() -> None:
    torch.manual_seed(6106)
    bank = ExternalTransitionModelBank(
        2,
        1,
        2,
        hidden_width=8,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=2,
    )
    first = bank.ensure_context(torch.tensor([1.0, 0.0]))
    second = bank.ensure_context(
        torch.tensor([0.0, 1.0]),
        initialize_from=first,
    )
    before = bank.digest()

    def mutating_growth_probe(candidate) -> bool:
        next(iter(candidate.models[0].state_dict().values())).add_(1.0)
        return False

    growth = bank.grow_verified(
        3,
        mutating_growth_probe,
    )
    assert not growth.accepted
    assert bank.capacity == 2
    assert bank.digest() == before
    heldout = ExternalTransitionObservation(
        state=torch.tensor([[0.2, -0.4]]),
        intention=torch.tensor([[0.7]]),
        next_state=torch.tensor([[0.1, 0.2]]),
    )

    def mutating_probe(candidate) -> bool:
        next(iter(candidate.models[0].state_dict().values())).add_(1.0)
        return True

    consolidation = bank.consolidate_verified(
        first,
        second,
        [heldout],
        retention_probe=mutating_probe,
    )
    assert not consolidation.accepted
    assert bank.physical_model_count == 2
    assert bank.digest() == before


def test_learned_maintenance_pressure_test_passes(tmp_path) -> None:
    report = run(6107, tmp_path / "maintenance.json")
    assert report["promoted"] is True
    assert report["gates"]["trained_beats_fresh"] is True
    assert report["gates"]["trained_beats_shuffled_verifier"] is True
    assert report["accounting"]["replayed_examples"] == 0


def test_real_maintenance_pressure_test_passes(tmp_path) -> None:
    report = run_real_maintenance(6110, tmp_path / "real-maintenance.json")
    assert report["promoted"] is True
    assert report["gates"]["real_transaction_observed"] is True
    assert report["gates"]["compression_bytes_observed"] is True
    assert report["gates"]["unsafe_probe_atomic"] is True
