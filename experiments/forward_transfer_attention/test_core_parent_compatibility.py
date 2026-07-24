from .audit_core_parent_compatibility import TASKS, task_gate


def _audit():
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


def _controls():
    return {
        name: {"verified_accuracy": 0.5}
        for name in (
            "action_shuffled", "reward_shuffled", "fully_fresh_core")
    }


def test_compatibility_suite_contains_both_prior_rungs() -> None:
    assert set(TASKS) == {"fixed_probe", "fixed_target"}
    assert TASKS["fixed_probe"]["established_threshold"] == 16
    assert TASKS["fixed_target"]["established_threshold"] == 64


def test_fixed_target_does_not_require_untrained_target_reversal() -> None:
    audit = _audit()
    audit["target_reverse_accuracy"] = 0.0
    audit["target_reverse_prediction_flip"] = 0.0
    assert task_gate(
        audit, _controls(), target_reversal_required=False)
    assert not task_gate(
        audit, _controls(), target_reversal_required=True)
