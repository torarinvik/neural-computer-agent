from .outcome_replay_controller import (
    OutcomeCalibratedReplayController,
    ReplayBudget,
    estimate_outcome_delta,
)


def test_paired_outcome_estimate_is_conservative() -> None:
    estimate = estimate_outcome_delta(
        (0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0))
    assert estimate.count == 4
    assert estimate.mean_delta == 0.25
    assert estimate.lower < estimate.mean_delta < estimate.upper


def test_diagnostic_lifetimes_are_charged_to_replay_budget() -> None:
    budget = ReplayBudget(maximum_lifetimes=10)
    budget.consume_replay(4)
    budget.consume_diagnostics(3)
    assert budget.total_lifetimes == 7
    assert budget.remaining_lifetimes == 3


def test_controller_stops_only_after_acquisition_and_all_retention_gates() -> None:
    controller = OutcomeCalibratedReplayController(
        maximum_lifetimes=12, minimum_diagnostic_lifetimes=4)
    controller.observe_acquisition((0.0, 0.0, 0.0, 0.0),
                                   (1.0, 1.0, 1.0, 1.0))
    controller.observe_protected("old-a", (1.0, 1.0, 1.0, 1.0),
                                 (1.0, 1.0, 1.0, 1.0))
    controller.observe_protected("old-b", (1.0, 1.0, 1.0, 1.0),
                                 (1.0, 1.0, 1.0, 1.0))
    decision = controller.decide()
    assert decision.action == "stop"
    assert decision.acquisition_ready
    assert decision.retention_ready
    assert controller.budget.diagnostic_lifetimes == 12


def test_controller_continues_when_one_protected_stream_regresses() -> None:
    controller = OutcomeCalibratedReplayController(
        maximum_lifetimes=12, minimum_diagnostic_lifetimes=4)
    controller.observe_acquisition((0.0,) * 4, (1.0,) * 4)
    controller.observe_protected("old", (1.0,) * 4, (0.0,) * 4)
    decision = controller.decide()
    assert decision.action == "continue"
    assert decision.acquisition_ready
    assert not decision.retention_ready


def test_budget_exhaustion_is_not_reported_as_success() -> None:
    controller = OutcomeCalibratedReplayController(
        maximum_lifetimes=4, minimum_diagnostic_lifetimes=4)
    controller.observe_acquisition((0.0,) * 4, (0.0,) * 4)
    decision = controller.decide()
    assert decision.action == "budget_exhausted"
    assert not decision.retention_ready
