import json

import torch

from .probe_adaptive_consolidation import (
    FEATURES, PrefixObservation, calibrate_threshold, load_observations,
    select_budgets, train_probe)


def _report(path, *, seed: int, budget: int, passed: bool) -> None:
    row = {
        "optimizer_update": budget,
        "new_batch_accuracy": 0.5 + 0.02 * budget,
        "skill_loss": 1.0 / budget,
        "retention_loss": 0.5 / budget,
        "locality": 0.1 / budget,
        "total_loss": 1.5 / budget,
    }
    path.write_text(json.dumps({
        "schema": "pair-magnitude-appearance-bridge-v1",
        "configuration": {
            "seed": seed,
            "epochs_per_batch": budget,
            # Private metadata must not enter the stream key or features.
            "blend_end": 0.2 + seed / 1_000_000,
        },
        "history": [row],
        "all_gates_passed": passed,
    }))


def test_report_loader_uses_only_learner_visible_features(tmp_path) -> None:
    _report(tmp_path / "a.json", seed=1, budget=2, passed=False)
    _report(tmp_path / "b.json", seed=1, budget=4, passed=True)
    observations = load_observations(sorted(tmp_path.glob("*.json")))
    assert [row.stream for row in observations] == ["1", "1"]
    assert [row.budget for row in observations] == [2, 4]
    assert len(observations[0].features) == len(FEATURES)
    assert observations[0].features[-3:] == (0.0, 0.0, 0.0)
    assert observations[1].features[-1] > 0


def test_policy_selects_first_safe_prefix_and_falls_back() -> None:
    rows = [
        PrefixObservation("a", 2, (0.0,) * len(FEATURES), False),
        PrefixObservation("a", 4, (0.0,) * len(FEATURES), True),
        PrefixObservation("a", 8, (0.0,) * len(FEATURES), True),
        PrefixObservation("b", 2, (0.0,) * len(FEATURES), False),
        PrefixObservation("b", 8, (0.0,) * len(FEATURES), True),
    ]
    selected = select_budgets(
        rows, [0.1, 0.8, 0.9, 0.1, 0.2], threshold=0.5)
    assert selected["a"].budget == 4
    assert selected["b"].budget == 8


def test_calibration_requires_training_safety() -> None:
    rows = [
        PrefixObservation("a", 2, (0.0,) * len(FEATURES), False),
        PrefixObservation("a", 8, (0.0,) * len(FEATURES), True),
    ]
    threshold = calibrate_threshold(rows, [0.8, 0.9])
    assert threshold > 0.8
    selected = select_budgets(rows, [0.8, 0.9], threshold=threshold)
    assert selected["a"].passed


def test_probe_round_trip_shape() -> None:
    rows = [
        PrefixObservation(
            str(index // 3), 2 ** (index % 3),
            tuple(float(index + offset) for offset in range(len(FEATURES))),
            bool(index % 3 == 2))
        for index in range(12)
    ]
    model, mean, scale = train_probe(rows, seed=3, updates=2)
    assert model(torch.zeros(4, len(FEATURES))).shape == (4,)
    assert mean.shape == scale.shape == (len(FEATURES),)
    assert torch.all(scale > 0)
