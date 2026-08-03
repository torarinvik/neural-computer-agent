import torch

from .legacy_runtime import AmodalEventWindowStatus
from .amodal_wait_policy import (
    AmodalArrivalPredictor,
    AmodalWaitDecisionPolicy,
    arrival_features,
)


def test_arrival_features_use_only_generic_transport_metadata() -> None:
    status = AmodalEventWindowStatus(
        timestamp=4.0,
        age=1.0,
        present=(True, False, True),
        complete=False,
    )
    features = arrival_features(status, [True, False, True], deadline=2.0)
    assert features.shape == (5,)
    assert torch.allclose(
        features,
        torch.tensor([2 / 3, 0.5, 2 / 3, 0.25, 0.0]),
    )


def test_arrival_predictor_has_generic_bounded_probability_output() -> None:
    predictor = AmodalArrivalPredictor(hidden=8)
    probabilities = predictor(torch.zeros(4, 5))
    assert probabilities.shape == (4,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_wait_decision_policy_returns_wait_probability() -> None:
    policy = AmodalWaitDecisionPolicy(hidden=8)
    probabilities = policy(torch.zeros(4, 5))
    assert probabilities.shape == (4,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
