from __future__ import annotations

from experiments.deliberation_amodal.async_audit import buffered_rollout
from experiments.deliberation_amodal.environment import VariableDeliberationVerifier
from experiments.deliberation_amodal.train import (
    build_runtime,
    evaluate_metrics,
    train_steps,
)


def test_deliberation_verifier_releases_partner_without_target_access() -> None:
    verifier = VariableDeliberationVerifier(seed=3, easy_probability=0.0)
    streams = verifier.reset(1)

    assert set(streams) == {"a"}
    delayed = verifier.release_delayed()
    assert set(delayed) == {"b"}
    assert not hasattr(verifier, "correct_action")
    assert not hasattr(verifier, "target_for_learner")


def test_think_required_event_is_released_only_after_internal_tick() -> None:
    verifier = VariableDeliberationVerifier(seed=4, easy_probability=0.0, think_probability=1.0)
    streams = verifier.reset(1)

    assert set(streams) == {"a"}
    assert streams["a"][0, -1].item() == 0.25
    assert verifier.release_delayed() == {}
    assert set(verifier.release_delayed(after_think=True)) == {"b"}


def test_missing_partner_stays_absent_after_think_and_timeout() -> None:
    verifier = VariableDeliberationVerifier(
        seed=5,
        easy_probability=0.0,
        think_probability=0.0,
        missing_probability=1.0,
    )
    streams = verifier.reset(1)

    assert set(streams) == {"a"}
    assert verifier.release_delayed() == {}
    assert verifier.release_delayed(after_think=True) == {}


def test_deliberation_audits_use_opaque_runtime_controls() -> None:
    runtime = build_runtime(seed=5)
    verifier = VariableDeliberationVerifier(seed=6)

    for condition in (
        "adaptive",
        "commit_immediate",
        "wait_fixed",
        "think_fixed",
        "missing_delayed",
        "random_action",
    ):
        metrics = evaluate_metrics(runtime, verifier, episodes=4, condition=condition)
        assert 0.0 <= metrics["reward"] <= 1.0
        assert 0.0 <= metrics["wait_fraction"] <= 1.0
        assert 0.0 <= metrics["think_fraction"] <= 1.0
        assert 0.0 <= metrics["commit_fraction"] <= 1.0


def test_deliberation_rung_records_required_accounting() -> None:
    runtime = build_runtime(seed=7)
    verifier = VariableDeliberationVerifier(seed=8)
    history, accounting = train_steps(runtime, verifier, steps=4, seed=8, warmup_steps=0)

    assert history
    assert accounting.unique_logical_lifetimes == 4
    assert accounting.unique_verifier_bits == 8
    assert accounting.optimizer_updates == 4
    assert accounting.replayed_examples == 0


def test_timestamped_buffer_replay_preserves_bounded_execution_paths() -> None:
    runtime = build_runtime(seed=9)
    cases = (
        (1.0, 0.0, "commit", True, False),
        (0.0, 0.0, "wait", False, False),
        (0.0, 1.0, "think", False, False),
        (0.0, 0.0, "wait", False, True),
    )
    for index, (easy, think, mode, out_of_order, force_missing) in enumerate(cases):
        verifier = VariableDeliberationVerifier(
            seed=10 + index,
            easy_probability=easy,
            think_probability=think,
        )
        decision, reward = buffered_rollout(
            runtime,
            verifier,
            mode_override=mode,
            out_of_order=out_of_order,
            timestamp_jitter=0.1,
            force_missing=force_missing,
        )
        assert decision == mode
        assert reward in {0.0, 1.0}


def test_complete_buffer_wait_does_not_create_timeout_decision() -> None:
    runtime = build_runtime(seed=11)
    verifier = VariableDeliberationVerifier(seed=12, easy_probability=1.0)

    decision, timeout, reward = buffered_rollout(
        runtime,
        verifier,
        mode_override="wait",
        include_timeout=True,
    )

    assert decision == "wait"
    assert timeout is None
    assert reward in {0.0, 1.0}
