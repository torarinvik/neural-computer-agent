from pathlib import Path

from experiments.policy_free_factual_residual_capacity.train import run


def test_factual_residual_capacity_and_reliability_promote(
    tmp_path: Path,
) -> None:
    for seed in (101, 102):
        report = run(seed, tmp_path / f"factual-residual-capacity-{seed}.json")

        assert report["promoted"]
        assert all(report["gates"].values())
        assert report["accounting"]["logical_lifetimes"] == 10
        assert report["accounting"]["residual_replayed_examples"] == 0
        assert report["accounting"]["reliability_replayed_examples"] == 0
        assert report["metrics"]["compression"]["selected_codec"] == "torch.float16"
