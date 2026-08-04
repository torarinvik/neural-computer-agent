from __future__ import annotations

import torch

from experiments.relevance_amodal.environment import (
    BalancedContextRelevanceCurriculum,
    ContextRelevanceVerifier,
)
from experiments.relevance_amodal.train import build_runtime, train_steps


def test_relevance_assignment_is_private_and_candidate_order_is_transport_only() -> None:
    verifier = ContextRelevanceVerifier(seed=3)
    streams = verifier.reset(4)
    assert set(streams) == {"a", "b", "c"}
    assert not hasattr(verifier, "relevant_label")
    assert not hasattr(verifier, "correct_action")

    forced = ContextRelevanceVerifier(seed=4, force_relevant_index=1)
    assert set(forced.reset(4)) == {"a", "b", "c"}
    shuffled = ContextRelevanceVerifier(seed=5, stream_order_shuffle=True)
    assert set(shuffled.reset(4)) == {"a", "b", "c"}


def test_relevance_runtime_keeps_three_event_tokens() -> None:
    runtime = build_runtime(seed=6)
    verifier = ContextRelevanceVerifier(seed=7)
    from neural_computer import ControllerFeedback

    streams = verifier.reset(4)
    output, state = runtime.step_streams(
        streams,
        runtime.initial_state(4, device="cpu"),
        ControllerFeedback(
            action=torch.zeros(4, runtime.controller.feedback_width),
            reward=torch.zeros(4),
            propensity=torch.ones(4),
            has_feedback=torch.zeros(4),
        ),
    )
    assert state.event_window.present.sum().item() == 12
    assert output.controller.event_attention.shape == (
        4,
        runtime.controller.event_window_capacity,
    )


def test_relevance_curriculum_alternates_hidden_assignments() -> None:
    curriculum = BalancedContextRelevanceCurriculum(seed=8)
    first = curriculum.reset(2)
    curriculum.step(torch.zeros(2, dtype=torch.long))
    second = curriculum.reset(2)
    assert set(first) == set(second) == {"a", "b", "c"}


def test_relevance_rung_records_outcome_only_accounting() -> None:
    runtime = build_runtime(seed=8)
    verifier = ContextRelevanceVerifier(seed=9)
    history, accounting = train_steps(
        runtime, verifier, steps=2, batch_size=4, seed=9, eval_every=2
    )
    assert history
    assert accounting.unique_logical_lifetimes == 8
    assert accounting.unique_verifier_bits == 16
    assert accounting.optimizer_updates == 2
    assert accounting.replayed_examples == 0
