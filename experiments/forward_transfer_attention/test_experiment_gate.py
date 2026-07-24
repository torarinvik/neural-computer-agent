from .experiment_gate import assess_report, assess_reports


def test_short_healthy_valley_does_not_earn_escalation():
    result = assess_report({"gradient_norm": 0.02, "residual_rms": 0.1,
                            "aux_loss_history": [0.71, 0.70]},
                           budget_seconds=30)
    assert result["status"] == "HEALTHY_VALLEY"


def test_heldout_signal_earns_escalation():
    result = assess_report({"gradient_norm": 0.02,
                            "aux_loss_history": [0.71, 0.64],
                            "heldout_accuracy": 0.61,
                            "shuffled_label_accuracy": 0.50},
                           budget_seconds=180)
    assert result["status"] == "PROMISING_CANDIDATE"


def test_two_clean_seeds_are_required_for_escalation():
    report = {"gradient_norm": 0.02,
              "aux_loss_history": [0.71, 0.64],
              "heldout_accuracy": 0.61,
              "shuffled_label_accuracy": 0.50}
    result = assess_reports([report, report], budget_seconds=180)
    assert result["status"] == "PROMISING"


def test_bad_control_is_a_red_flag_even_with_high_accuracy():
    result = assess_report({"heldout_accuracy": 0.90,
                            "shuffled_label_accuracy": 0.84},
                           budget_seconds=30)
    assert result["status"] == "RED_FLAG"
    assert "shuffled-label control is not at chance" in result["issues"]


def test_flat_report_is_not_promoted():
    result = assess_report({"heldout_accuracy": 0.51}, budget_seconds=30)
    assert result["status"] == "NO_SIGNAL"
