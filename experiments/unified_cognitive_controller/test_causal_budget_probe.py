import torch

from .causal_budget_probe import FEATURES, state_features, target_label


def _payload(*, loss: float = 0.2) -> dict:
    return {
        "replay_trace": [
            {
                "arm": "composition",
                "experience_step": step,
                "replay_index": 0,
                "loss_before": loss + step * 0.01,
                "previous_loss_reduction": 0.01 * step,
                "previous_gradient_norm": 0.2 * step,
                "observed_examples": 60 * step,
                "loss_after": 99.0,
                "loss_reduction": -99.0,
            }
            for step in range(1, 4)
        ],
    }


def _outcome(bits: int, utility: float) -> dict:
    return {
        "stable_target_bits": {"option_composition": bits},
        "histories": {"option_composition": [
            {"verified_utility": utility},
        ]},
    }


def test_state_features_exclude_post_update_outcomes() -> None:
    payload = _payload()
    original = state_features(payload, 3)
    for row in payload["replay_trace"]:
        row["loss_after"] = -123.0
        row["loss_reduction"] = 456.0
    assert torch.equal(original, state_features(payload, 3))
    assert original.shape == (len(FEATURES),)


def test_causal_label_requires_sample_gain_and_capability_guard() -> None:
    label, details = target_label(
        _outcome(3_000, 0.90), _outcome(2_400, 0.898),
        capability_tolerance=0.003)
    assert label == 1.0
    assert details["saves_experience"]
    assert details["keeps_capability"]

    label, _ = target_label(
        _outcome(3_000, 0.90), _outcome(2_400, 0.895),
        capability_tolerance=0.003)
    assert label == 0.0
