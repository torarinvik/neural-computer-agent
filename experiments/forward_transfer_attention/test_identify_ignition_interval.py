from .map_identify_ignition_interval import PREFIXES, _passes


def _passing_audit():
    return {
        "normal_accuracy": 0.8,
        "normal_entropy": 0.2,
        "protocol_swap_accuracy": 0.8,
        "protocol_swap_prediction_flip": 0.8,
        "target_reverse_accuracy": 0.8,
        "target_reverse_prediction_flip": 0.8,
        "missing_consequence_accuracy": 0.5,
        "missing_consequence_entropy": 0.8,
        "no_probe_effect_accuracy": 0.5,
    }


def _passing_controls():
    return {
        name: {"verified_accuracy": 0.5}
        for name in (
            "action_shuffled", "reward_shuffled", "fully_fresh_core")
    }


def test_prefixes_fill_the_missing_interval_in_eight_bit_steps() -> None:
    assert PREFIXES == (32, 40, 48, 56, 64)


def test_gate_requires_every_causal_and_control_condition() -> None:
    audit = _passing_audit()
    controls = _passing_controls()
    assert _passes(audit, controls)
    for key in (
            "normal_accuracy", "protocol_swap_accuracy",
            "protocol_swap_prediction_flip", "target_reverse_accuracy",
            "target_reverse_prediction_flip"):
        broken = audit.copy()
        broken[key] = 0.7
        assert not _passes(broken, controls)
    broken_controls = _passing_controls()
    broken_controls["reward_shuffled"]["verified_accuracy"] = 0.7
    assert not _passes(audit, broken_controls)
