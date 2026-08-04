from __future__ import annotations

import torch

from experiments.reliability_amodal.environment import RedundantComplementVerifier
from experiments.reliability_amodal.train import build_runtime, train_steps


def test_corruption_is_unmarked_and_missing_stream_is_transport_only() -> None:
    verifier = RedundantComplementVerifier(
        seed=3, corruption_probability=1.0, missing_probability=0.0
    )
    streams = verifier.reset(4)
    assert set(streams) == {"a", "b", "c", "d"}
    assert not hasattr(verifier, "correct_action")
    assert not hasattr(verifier, "corruption_label")

    missing = RedundantComplementVerifier(seed=4, missing_probability=1.0)
    assert set(missing.reset(4)) in (
        {"a", "b", "c"},
        {"a", "b", "d"},
        {"a", "c", "d"},
    )

    conflict = RedundantComplementVerifier(
        seed=5, force_flip_mask=(False, True, True)
    )
    assert set(conflict.reset(4)) == {"a", "b", "c", "d"}


def test_runtime_keeps_redundant_tokens_separate_and_permutation_invariant() -> None:
    runtime = build_runtime(seed=5)
    verifier = RedundantComplementVerifier(seed=6, stream_order_shuffle=True)
    streams = verifier.reset(4)
    state = runtime.initial_state(4, device="cpu")
    from neural_computer import ControllerFeedback

    feedback = ControllerFeedback(
        action=torch.zeros(4, runtime.controller.feedback_width),
        reward=torch.zeros(4),
        propensity=torch.ones(4),
        has_feedback=torch.zeros(4),
    )
    output, next_state = runtime.step_streams(streams, state, feedback)
    assert next_state.event_window.present.sum().item() == 16
    assert output.controller.event_attention.shape == (
        4,
        runtime.controller.event_window_capacity,
    )


def test_reliability_rung_records_outcome_only_accounting() -> None:
    runtime = build_runtime(seed=7)
    verifier = RedundantComplementVerifier(
        seed=8, corruption_probability=0.33, missing_probability=0.33
    )
    history, accounting = train_steps(
        runtime, verifier, steps=2, batch_size=4, seed=8, eval_every=2
    )
    assert history
    assert accounting.unique_logical_lifetimes == 8
    assert accounting.unique_verifier_bits == 16
    assert accounting.optimizer_updates == 2
    assert accounting.replayed_examples == 0
