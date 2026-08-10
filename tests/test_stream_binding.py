from __future__ import annotations

import torch

from experiments.external_learned_stream_binding.train import run as run_pressure_test
from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


def _fixture(seed: int = 501) -> tuple[ExternalTransitionContextEncoder, list[ExternalTransitionObservation]]:
    torch.manual_seed(seed)
    encoder = ExternalTransitionContextEncoder(
        2,
        1,
        hidden_width=16,
        context_width=4,
    )
    observations: list[ExternalTransitionObservation] = []
    for stream in range(3):
        state = torch.randn(6, 2)
        state[:, 1] += stream * 5.0
        intention = torch.randn(6, 1)
        next_state = state + intention * torch.tensor([0.2 + stream, 1.0])
        observations.append(
            ExternalTransitionObservation(
                state,
                intention,
                next_state,
                torch.ones(6),
            )
        )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    for update in range(160):
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        for stream, observation in enumerate(observations):
            left_index = (update + stream) % 6
            right_index = (update * 3 + stream + 1) % 6
            left.append(_row(observation, left_index, encoder))
            right.append(_row(observation, right_index, encoder))
        loss = encoder.contrastive_loss(torch.stack(left), torch.stack(right))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    encoder.eval()
    return encoder, observations


def _row(
    observation: ExternalTransitionObservation,
    index: int,
    encoder: ExternalTransitionContextEncoder | None = None,
) -> ExternalTransitionObservation | torch.Tensor:
    row = ExternalTransitionObservation(
        observation.state[index : index + 1],
        observation.intention[index : index + 1],
        observation.next_state[index : index + 1],
        observation.confidence[index : index + 1]
        if observation.confidence is not None
        else None,
    )
    return row if encoder is None else encoder.encode_observation(row)


def test_external_binding_learns_anonymous_tracks_and_persists() -> None:
    encoder, observations = _fixture()
    memory = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=3,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        match_margin=0.05,
    )
    assert all(not parameter.requires_grad for parameter in memory.encoder.parameters())
    assignments: dict[int, int] = {}
    results = []
    for step in range(6):
        for stream in range(3):
            result = memory.observe(
                _row(observations[stream], step),
                timestamp=step + stream * 0.1,
            )
            assert result.status in {"new", "matched"}
            assert result.track_id is not None
            previous = assignments.setdefault(stream, result.track_id)
            assert result.track_id == previous
            results.append(result)

    assert memory.stream_count == 3
    assert len(set(assignments.values())) == 3
    assert all(memory.track_state(track_id)["mean_delay"] is not None for track_id in memory.track_ids)

    memory.observe_verifier_outcome(results[-1], 1.0)
    memory.observe_verifier_outcome(results[-1], 0.0)
    assert memory.track_state(results[-1].track_id)["reliability"] == 0.5

    restored = ExternalOnlineStreamBindingMemory.from_payload(memory.state_payload())
    assert restored.digest() == memory.digest()
    assert restored.track_ids == memory.track_ids


def test_learned_binding_routes_without_caller_stream_keys() -> None:
    encoder, observations = _fixture(502)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=3,
    )
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=3,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    router = ExternalLearnedMultiStreamTransitionContextRouter(
        ExternalOnlineStreamBindingMemory(
            encoder,
            window_capacity=4,
            max_streams=3,
            match_tolerance=0.55,
            new_track_tolerance=0.7,
            match_margin=0.05,
        ),
        ExternalMultiStreamTransitionContextRouter(single, stream_key_width=4),
    )

    route_results = []
    for row_index in range(2):
        for stream in range(3):
            route_results.append(
                router.observe(
                    _row(observations[stream], row_index),
                    timestamp=row_index + stream * 0.1,
                )
            )

    assert all(result.routing is not None for result in route_results)
    assert router.stream_count == 3
    assert router.router.stream_count == 3
    assert all(result.binding.stream_key is not None for result in route_results)
    assert len({result.binding.track_id for result in route_results}) == 3

    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert restored.digest() == router.digest()

    corrupted = router.state_payload()
    corrupted["binding"]["tracks"][0]["stream_key"][0] += 0.01
    try:
        ExternalLearnedMultiStreamTransitionContextRouter.from_payload(corrupted)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("corrupted learned binding state was accepted")


def test_learned_binding_pressure_test_passes() -> None:
    report = run_pressure_test(2301)
    assert report["promoted"] is True
    assert report["gates"]["learned_assignment_consistent"] is True
    assert report["gates"]["learned_beats_fresh"] is True
    assert report["gates"]["binding_encoder_frozen"] is True
