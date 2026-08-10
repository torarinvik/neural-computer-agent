from pathlib import Path

from experiments.policy_free_factual_residual_growth.train import run


def test_factual_residual_growth_promotes_and_controls_are_honest(tmp_path: Path) -> None:
    for seed in (101, 102):
        report = run(seed, tmp_path / f"factual-residual-{seed}.json")

        assert report["promoted"]
        assert all(report["gates"].values())
        assert report["accounting"]["residual_replayed_examples"] == 0
        assert report["metrics"]["full_model_copy_control"][
            "source_retained_at_target_stability"
        ] is False
        assert report["metrics"]["fresh_control"]["optimizer_updates"] > 0
