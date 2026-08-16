from __future__ import annotations

from experiments.brainworkshop_canonical.perceptual_aliasing import (
    STABLE_THRESHOLD,
    WORLD,
    evaluate_model,
    fit_model,
    run_aliasing,
    sample_traces,
)


def test_two_latent_places_share_one_observation_symbol() -> None:
    assert len(WORLD.observation_symbols) == 6
    assert len(set(WORLD.observation_symbols)) == 5
    assert WORLD.observation_symbols[0] == WORLD.observation_symbols[3]


def test_history_context_resolves_aliasing_better_than_current_symbol() -> None:
    training = sample_traces(41, episodes=16)
    evaluation = sample_traces(10016, episodes=40)
    merged = evaluate_model(fit_model(training, mode="merged"), evaluation, mode="merged")
    history = evaluate_model(fit_model(training, mode="history"), evaluation, mode="history")
    assert merged["accuracy"] < STABLE_THRESHOLD
    assert history["accuracy"] >= STABLE_THRESHOLD


def test_corrupted_history_does_not_pass_the_stable_gate() -> None:
    training = sample_traces(41, episodes=16)
    evaluation = sample_traces(10016, episodes=40)
    corrupted = evaluate_model(
        fit_model(training, mode="corrupted_history"),
        evaluation,
        mode="corrupted_history",
    )
    assert corrupted["accuracy"] < STABLE_THRESHOLD


def test_aliasing_report_records_stable_prefix_and_accounting(tmp_path) -> None:
    report = run_aliasing(tmp_path, replicates=1)
    assert report["replicates"][0]["arms"]["history"]["stable_bits_to_threshold"] == 920
    assert report["replicates"][0]["arms"]["merged"]["stable_bits_to_threshold"] is None
    assert report["accounting"]["history"]["optimizer_updates"] == 0
