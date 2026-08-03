import json

import torch

from .probe_familiarity_stop import (
    FamiliarityObservation, calibrate_threshold, fit_projection,
    load_observations)


def _write_stream(root, *, tag: str, seed: int, stable: tuple[bool, bool]):
    (root / f"{tag}_train.json").write_text(json.dumps({
        "schema": "pair-magnitude-appearance-bridge-v1",
        "configuration": {"seed": seed, "blend_end": 0.2},
        "history": [{
            "new_batch_accuracy": 0.7,
            "skill_loss": 0.3,
            "retention_loss": 0.01,
            "locality": 0.2,
            "total_loss": 0.4,
            "sensory_summary": [float(seed), float(seed + 1)],
        }],
    }))
    for suffix, mastery in zip(
            ("target16k", "target16k_b"), stable, strict=True):
        (root / f"{tag}_{suffix}.json").write_text(json.dumps({
            "schema": "pair-magnitude-compute-audit-v1",
            "results": {"0": {"mastery": mastery}},
        }))


def test_stable_label_requires_both_independent_audits(tmp_path) -> None:
    for index in range(6):
        _write_stream(
            tmp_path, tag=str(index), seed=index,
            stable=(True, index % 2 == 0))
    rows = load_observations(
        tmp_path, audit_suffixes=("target16k", "target16k_b"))
    assert len(rows) == 6
    assert sum(row.stable_after_one for row in rows) == 3
    assert all(len(row.sensory_features) == 2 for row in rows)


def test_projection_is_train_fitted_and_finite() -> None:
    values = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 4.0],
        [3.0, 2.0, 5.0],
    ])
    projection = fit_projection(values, rank=2)
    transformed = projection.apply(values)
    assert transformed.shape == (3, 2)
    assert torch.isfinite(transformed).all()


def test_threshold_forbids_false_stops() -> None:
    rows = [
        FamiliarityObservation(
            "a", (0.0,) * 5, (0.0,), False, 0.2),
        FamiliarityObservation(
            "b", (0.0,) * 5, (0.0,), True, 0.2),
    ]
    threshold = calibrate_threshold(rows, [0.75, 0.90])
    assert threshold > 0.75
