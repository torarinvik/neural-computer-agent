from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from .train_complement_population import (
    _audit_summary, _select_winner, _train_command)


def _audit(
        *, complement: float = 0.65, zeroed: float = 0.50,
        span9_parent: float = 0.86, span9_candidate: float = 0.85,
        span10_parent: float = 0.83, span10_candidate: float = 0.82,
        reset: float = 0.50) -> dict[str, object]:
    return {
        "causal_gain_points": (complement - zeroed) * 100.0,
        "parent_audit": {
            "complement": {"accuracy": zeroed},
            "span9": {"accuracy": span9_parent},
            "span10": {"accuracy": span10_parent},
        },
        "candidate_audit": {
            "complement": {
                "accuracy": complement,
                "all_memory_reset_accuracy": reset,
            },
            "span9": {"accuracy": span9_candidate},
            "span10": {"accuracy": span10_candidate},
        },
    }


def test_population_summary_requires_causal_and_retention_gates() -> None:
    summary = _audit_summary(
        _audit(), minimum_causal_gain=5.0,
        minimum_complement_accuracy=0.60,
        maximum_retention_drop_points=2.0)
    assert summary["selection_eligible"]
    rejected = _audit_summary(
        _audit(span9_candidate=0.83), minimum_causal_gain=5.0,
        minimum_complement_accuracy=0.60,
        maximum_retention_drop_points=2.0)
    assert not rejected["selection_eligible"]
    private = _audit_summary(
        _audit(reset=0.40), minimum_causal_gain=5.0,
        minimum_complement_accuracy=0.60,
        maximum_retention_drop_points=2.0, require_reset=False)
    assert private["selection_eligible"]


def test_population_selects_best_eligible_arm_only() -> None:
    safe = {
        "seed": 1,
        "private": _audit_summary(
            _audit(complement=0.62), minimum_causal_gain=5.0,
            minimum_complement_accuracy=0.60,
            maximum_retention_drop_points=2.0),
    }
    better = {
        "seed": 2,
        "private": _audit_summary(
            _audit(complement=0.66), minimum_causal_gain=5.0,
            minimum_complement_accuracy=0.60,
            maximum_retention_drop_points=2.0),
    }
    unsafe = {
        "seed": 3,
        "private": _audit_summary(
            _audit(complement=0.90, span9_candidate=0.80),
            minimum_causal_gain=5.0, minimum_complement_accuracy=0.60,
            maximum_retention_drop_points=2.0),
    }
    assert _select_winner([safe, better, unsafe]) is better
    assert _select_winner([unsafe]) is None


def test_population_training_command_separates_data_seed() -> None:
    args = Namespace(
        parent=Path("parent.pt"), data_seed=11, train_lifetimes=1024,
        span=10, distractors=2, epochs=128, batch_size=512,
        learning_rate=0.001, skill_adapter_width=256,
        skill_adapter_gate_hidden=64, replay_residual_penalty=0.01,
        replay_gate_penalty=0.01, replay_logit_penalty=0.01,
        binary_margin=0.25, replay_buffer_in=Path("replay.pt"),
        private_count=256, device="cpu")
    command = _train_command(
        args=args, seed=99, report=Path("report.json"),
        checkpoint=Path("candidate.pt"))
    assert command[command.index("--seed") + 1] == "99"
    assert command[command.index("--data-seed") + 1] == "11"
