from pathlib import Path

from experiments.policy_free_factual_residual_stream.train import run


def test_factual_residual_stream_promotes_complete_prefix_and_controls(
    tmp_path: Path,
) -> None:
    for seed in (101, 102):
        report = run(seed, tmp_path / f"factual-residual-stream-{seed}.json")

        assert report["promoted"]
        assert all(report["gates"].values())
        assert report["accounting"]["logical_lifetimes"] == 7
        assert report["accounting"]["residual_replayed_examples"] == 0
        assert report["accounting"]["fresh_replayed_examples"] > 0
        assert report["metrics"]["compression"]["selected_codec"] == "torch.float16"
