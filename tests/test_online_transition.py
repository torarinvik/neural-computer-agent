from __future__ import annotations

import torch

from neural_computer import (
    ExternalAffineTransitionStatistics,
    ExternalTransitionObservation,
)


def test_affine_transition_statistics_learns_and_persists_one_pass() -> None:
    torch.manual_seed(1301)
    model = ExternalAffineTransitionStatistics(2, 1, ridge=1e-7)
    state = torch.randn(8, 2)
    intention = torch.randn(8, 1)
    features = torch.cat((state, intention, torch.ones(8, 1)), dim=-1)
    true_weights = torch.tensor(
        [
            [1.0, 0.2],
            [-0.3, 0.8],
            [0.7, -1.1],
            [0.4, -0.6],
        ]
    )
    next_state = features @ true_weights
    observation = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(8),
    )

    for row in range(state.shape[0]):
        model.observe(
            ExternalTransitionObservation(
                state=observation.state[row : row + 1],
                intention=observation.intention[row : row + 1],
                next_state=observation.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
    )

    assert int(model.sample_count) == 8
    assert float(model.loss(observation)) < 1e-7
    heldout_state = torch.randn(4, 2)
    heldout_intention = torch.randn(4, 1)
    heldout_features = torch.cat(
        (heldout_state, heldout_intention, torch.ones(4, 1)), dim=-1
    )
    heldout = ExternalTransitionObservation(
        state=heldout_state,
        intention=heldout_intention,
        next_state=heldout_features @ true_weights,
    )
    assert float(model.loss(heldout)) < 1e-6
    restored = ExternalAffineTransitionStatistics.from_payload(model.state_payload())
    assert restored.digest() == model.digest()
    assert torch.allclose(
        restored(state, intention),
        model(state, intention),
        atol=1e-7,
        rtol=0.0,
    )


def test_affine_transition_statistics_rejects_checksum_corruption() -> None:
    model = ExternalAffineTransitionStatistics(1, 1)
    model.observe(
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[2.0]]),
            next_state=torch.tensor([[3.0]]),
        )
    )
    payload = model.state_payload()
    payload["state"]["normal_matrix"][0, 0] += 1.0

    try:
        ExternalAffineTransitionStatistics.from_payload(payload)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("expected affine statistics checksum rejection")
