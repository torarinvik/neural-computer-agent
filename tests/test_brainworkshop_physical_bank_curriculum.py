from __future__ import annotations

import torch

from experiments.brainworkshop_canonical.physical_bank_curriculum_pilot import (
    _stable_bits_to_threshold,
    summarize_curriculum,
)
from neural_computer import (
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
)


def _bank() -> ExternalTemporalProgramBank:
    bank = ExternalTemporalProgramBank(
        4,
        3,
        controller_digest="0" * 64,
        min_mastery_observations=2,
    )
    bank.admit(
        ExternalProgramArtifact(
            codes=torch.tensor([[5.0, -3.0, -3.0]]),
            interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        ),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        [1.0, 1.0],
        min_observations=2,
        min_stable_observations=2,
    )
    return bank


def test_stable_bits_requires_a_persistent_later_prefix() -> None:
    assert (
        _stable_bits_to_threshold(
            [0.0, 1.0, 1.0, 1.0, 1.0],
            threshold=0.8,
            min_stable_observations=2,
        )
        is None
    )
    assert (
        _stable_bits_to_threshold(
            [1.0, 1.0, 1.0], threshold=0.8, min_stable_observations=2
        )
        == 1
    )
    assert (
        _stable_bits_to_threshold(
            [1.0, 1.0, 0.0], threshold=0.8, min_stable_observations=2
        )
        is None
    )


def test_curriculum_summary_accounts_for_read_only_live_experience() -> None:
    reports = [
        {
            "rewards": [1.0, 1.0],
            "tick_seconds_p50": 0.04,
            "tick_seconds_p99": 0.11,
            "controller_digest": "0" * 64,
            "program_artifact_digest": "1" * 64,
            "input_events": 3,
            "emitted_actions": 2,
            "router_observations": 2,
            "elapsed_seconds": 1.0,
            "deadline_misses": 0,
        },
        {
            "rewards": [1.0, 1.0],
            "tick_seconds_p50": 0.05,
            "tick_seconds_p99": 0.12,
            "controller_digest": "0" * 64,
            "program_artifact_digest": "1" * 64,
            "input_events": 4,
            "emitted_actions": 3,
            "router_observations": 2,
            "elapsed_seconds": 1.1,
            "deadline_misses": 1,
        },
    ]

    summary = summarize_curriculum(
        reports,
        requested_sessions=2,
        threshold=0.8,
        min_stable_observations=2,
        source_bank_sha256="2" * 64,
        final_bank=_bank(),
        wall_seconds=2.2,
    )

    assert summary["retention_gate_passed"]
    assert summary["unique_logical_lifetimes"] == 2
    assert summary["unique_verifier_bits"] == 4
    assert summary["stable_bits_to_threshold"] == 1
    assert summary["controller_optimizer_updates"] == 0
    assert summary["program_optimizer_updates"] == 0
    assert summary["replayed_examples"] == 0
    assert summary["tick_seconds_p99_max"] == 0.12
