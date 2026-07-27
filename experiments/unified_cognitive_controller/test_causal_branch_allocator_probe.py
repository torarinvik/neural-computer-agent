import json

import pytest
import torch

from .causal_branch_allocator_probe import allocator_metrics, load_branch_report


def _payload(*, integrity: bool = True) -> dict[str, object]:
    return {
        "schema": "causal-budget-branching-v1",
        "configuration": {"seed": 42},
        "integrity": {
            "all_branches_started_from_identical_state": integrity,
            "all_examples_have_future_episodes": True,
            "labels_are_verifier_only": True,
        },
        "examples": [
            {
                "state_features": [1.0, 2.0], "choose_higher_budget": True,
                "outcome": {"eligible_for_allocation": True},
            },
            {
                "state_features": [3.0, 4.0], "choose_higher_budget": False,
                "outcome": {"eligible_for_allocation": True},
            },
        ],
    }


def test_loader_exposes_only_pre_branch_features(tmp_path) -> None:
    path = tmp_path / "branch.json"
    payload = _payload()
    payload["examples"][0]["future_utility"] = -99.0  # type: ignore[index]
    path.write_text(json.dumps(payload))
    features, labels, seed = load_branch_report(path)
    assert features.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert labels.flatten().tolist() == [1.0, 0.0]
    assert seed == 42


def test_loader_rejects_failed_branch_integrity(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_payload(integrity=False)))
    with pytest.raises(ValueError, match="integrity"):
        load_branch_report(path)


def test_loader_rejects_pre_guard_unsolved_labels(tmp_path) -> None:
    path = tmp_path / "old.json"
    payload = _payload()
    del payload["examples"][0]["outcome"]  # type: ignore[index]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="predates"):
        load_branch_report(path)


def test_allocator_metrics_penalize_never_spending_extra_compute() -> None:
    metrics = allocator_metrics(
        torch.tensor([[0.1], [0.2], [0.3]]),
        torch.tensor([[1.0], [0.0], [0.0]]))
    assert metrics["higher_precision"] == 0.0
    assert metrics["higher_recall"] == 0.0
