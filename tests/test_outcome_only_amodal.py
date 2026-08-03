from __future__ import annotations

import torch

from experiments.outcome_only_amodal.environment import OutcomeOnlyComplementVerifier
from experiments.outcome_only_amodal.train import build_runtime, evaluate_condition


def test_verifier_exposes_raw_streams_and_scalar_reward_only() -> None:
    verifier = OutcomeOnlyComplementVerifier(seed=3)
    streams = verifier.reset(8)
    assert set(streams) == {"a", "b"}
    assert all(stream.shape == (8, verifier.raw_width) for stream in streams.values())
    reward = verifier.step(torch.zeros(8, dtype=torch.long))
    assert reward.shape == (8,)
    assert reward.dtype == torch.float32


def test_clean_runtime_keeps_two_stream_tokens_separate() -> None:
    runtime = build_runtime(seed=5)
    verifier = OutcomeOnlyComplementVerifier(seed=6)
    streams = verifier.reset(4)
    state = runtime.initial_state(4, device="cpu")
    feedback = torch.zeros(4, runtime.controller.feedback_width)
    from neural_computer import ControllerFeedback

    output, next_state = runtime.step_streams(
        streams,
        state,
        ControllerFeedback(
            action=feedback,
            reward=torch.zeros(4),
            propensity=torch.ones(4),
            has_feedback=torch.zeros(4),
        ),
    )
    assert output.intention.payload.shape == (4, runtime.intention_width)
    assert next_state.event_window.present.sum().item() == 8
    assert output.controller.event_attention.shape == (4, runtime.controller.event_window_capacity)


def test_audit_interventions_run_through_the_intention_bus() -> None:
    runtime = build_runtime(seed=9)
    verifier = OutcomeOnlyComplementVerifier(seed=10)
    for condition in ("fused", "a_only", "shuffled_partner", "intention_shuffled", "intention_zero"):
        score = evaluate_condition(
            runtime,
            verifier,
            condition=condition,
            batches=1,
            batch_size=16,
        )
        assert 0.0 <= score <= 1.0
