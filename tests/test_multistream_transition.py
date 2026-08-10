import pytest
import torch

from experiments.external_transition_model_multistream.train import _run
from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


def _observation(next_state: tuple[float, float]) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=torch.tensor([[0.2, -0.4]]),
        intention=torch.tensor([[0.7]]),
        next_state=torch.tensor([list(next_state)]),
        confidence=torch.ones(1),
    )


def test_multistream_router_keeps_interleaved_pending_windows_separate() -> None:
    torch.manual_seed(1401)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=4,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=3,
        max_contexts=4,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    router = ExternalMultiStreamTransitionContextRouter(single, stream_key_width=3)
    stream_a = torch.tensor([1.0, 0.0, 0.0])
    stream_b = torch.tensor([0.0, 1.0, 0.0])

    router.observe(_observation((0.1, 0.2)), stream_a)
    router.observe(_observation((0.1, 0.2)), stream_b)
    router.observe(_observation((0.1, 0.2)), stream_a)
    router.observe(_observation((0.1, 0.2)), stream_b)

    assert router.stream_count == 2
    assert router.pending_observations(stream_a) == 2
    assert router.pending_observations(stream_b) == 2
    assert router.bank.context_count == 0

    result_a = router.observe(_observation((0.1, 0.2)), stream_a)
    result_b = router.observe(_observation((0.1, 0.2)), stream_b)

    assert result_a.result.status == "staged"
    assert result_b.result.status == "staged"
    assert router.provisional_candidate_count == 2
    assert router.provisional_evidence_count(stream_a) == 3
    assert router.provisional_evidence_count(stream_b) == 3
    assert router.bank is single.bank


def test_multistream_router_persists_stream_local_candidates_over_shared_bank() -> None:
    torch.manual_seed(1402)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=4,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=4,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    router = ExternalMultiStreamTransitionContextRouter(single, stream_key_width=3)
    stream_a = torch.tensor([1.0, 0.0, 0.0])
    stream_b = torch.tensor([0.0, 1.0, 0.0])
    for stream in (stream_a, stream_b):
        router.observe(_observation((0.1, 0.2)), stream)
        router.observe(_observation((0.1, 0.2)), stream)

    payload = router.state_payload()
    restored = ExternalMultiStreamTransitionContextRouter.from_payload(payload)

    assert restored.stream_count == 2
    assert restored.provisional_candidate_count == 2
    assert restored.bank.context_count == 0
    assert restored.pending_observations(stream_a) == 0
    assert restored.provisional_evidence_count(stream_a) == 2
    assert restored.provisional_evidence_count(stream_b) == 2
    assert restored.bank is restored.router.bank

    corrupted = router.state_payload()
    corrupted["streams"][0]["stream_key"][0] += 0.01
    with pytest.raises(ValueError, match="checksum"):
        ExternalMultiStreamTransitionContextRouter.from_payload(corrupted)


def test_multistream_pressure_test_binds_each_promoted_stream() -> None:
    report = _run(1901)

    assert report["all_promoted"] is True
    assert report["untouched_nonselected_candidates"] is True
    assert report["route_slot_ids"] == [0, 1, 2, 0, 1, 2]
    assert report["restored_route_slot_ids"] == [0, 1, 2]
    assert report["persistence_exact"] is True
    assert report["checksum_rejected"] is True
