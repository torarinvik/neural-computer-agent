from __future__ import annotations

import torch

from experiments.async_memory_amodal.environment import DelayedComplementVerifier
from experiments.async_memory_amodal.train import build_runtime, evaluate_condition
from neural_computer import ControllerFeedback, MemoryQuery


def test_delayed_verifier_exposes_only_streams_and_scalar_reward() -> None:
    verifier = DelayedComplementVerifier(seed=12)
    first = verifier.start(8)
    second = verifier.next_arrival()
    assert set(first) == {"a"}
    assert set(second) == {"b"}
    reward = verifier.score(torch.zeros(8, dtype=torch.long))
    assert reward.shape == (8,)
    assert reward.dtype == torch.float32


def test_async_runtime_accumulates_separately_arriving_events() -> None:
    runtime = build_runtime(seed=13)
    verifier = DelayedComplementVerifier(seed=14)
    from experiments.async_memory_amodal.train import _episode

    _, _, _, reward = _episode(runtime, verifier, batch_size=8, device=torch.device("cpu"))
    assert reward.shape == (8,)


def test_delayed_audit_conditions_are_callable() -> None:
    runtime = build_runtime(seed=15)
    for condition in ("fused", "missing_second", "contradictory", "shuffled_partner"):
        score = evaluate_condition(
            runtime,
            DelayedComplementVerifier(seed=16),
            condition=condition,
            batches=1,
            batch_size=16,
        )
        assert 0.0 <= score <= 1.0


def test_memory_round_trip_and_corruption_are_visible_at_the_controller_boundary() -> None:
    runtime = build_runtime(seed=17, with_memory=True)
    assert runtime.memory is not None
    verifier = DelayedComplementVerifier(seed=18)
    streams = verifier.start(4)
    from experiments.async_memory_amodal.train import _timed_events, zero_feedback

    feedback = zero_feedback(4, runtime.controller.feedback_width, torch.device("cpu"))
    output, _ = runtime.step_events(
        _timed_events(runtime, streams, timestamp=0.0),
        runtime.initial_state(4, device="cpu"),
        feedback,
    )
    query_key = output.controller.memory_query_key.detach().clone()
    runtime.memory.clear()
    runtime.memory.write(query_key, torch.ones_like(query_key), torch.ones(4))
    query_output, _ = runtime.step_events(
        _timed_events(runtime, streams, timestamp=0.0),
        runtime.initial_state(4, device="cpu"),
        ControllerFeedback(
            action=torch.zeros(4, runtime.controller.feedback_width),
            reward=torch.zeros(4),
            propensity=torch.ones(4),
            has_feedback=torch.zeros(4),
        ),
    )
    assert query_output.controller.memory_read is not None
    assert query_output.controller.memory_read.hit.all()
    runtime.memory.clear()
    assert not runtime.memory.read(MemoryQuery(query_key)).hit.any()
