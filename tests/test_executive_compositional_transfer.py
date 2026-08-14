import argparse

from experiments.brainworkshop_canonical.executive_compositional_transfer import run


def test_inherited_executive_fragment_reduces_heldout_search_bits() -> None:
    report = run(
        argparse.Namespace(
            report_out=None,
            seed=31,
            seeds=3,
            target_n_back=2,
            batch_size=8,
            steps=8,
            event_width=6,
        )
    )

    assert report["promoted"]
    assert report["aggregate"]["transfer_ratio_fresh_over_warm"] > 1.0
    assert not report["controls"]["irrelevant_bank_admitted"]
    assert not report["controls"]["destroyed_reward_admitted"]
