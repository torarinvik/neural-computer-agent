import pytest
import torch

from neural_computer import (
    ExternalRoutedIntentionCostLedger,
    ExternalRoutedIntentionCostModel,
    ExternalRoutedIntentionCostModelState,
)


def test_cost_ledger_updates_selected_branch_and_round_trips() -> None:
    ledger = ExternalRoutedIntentionCostLedger.create(
        2,
        initial_cost=0.5,
        decision_weight=0.7,
    )
    context = torch.tensor([[0.4, -0.1]])
    receipt = ledger.observe(
        context,
        selected_initialization="transfer",
        observed_cost=0.0,
        cell_count=3,
    )

    assert receipt.selected_initialization == "transfer"
    assert receipt.predicted_cost == pytest.approx(0.5)
    assert receipt.transfer_cost_after < receipt.predicted_cost
    assert receipt.fresh_cost_after == pytest.approx(0.5)
    assert ledger.configuration()["decision_weight"] == pytest.approx(0.7)

    restored = ExternalRoutedIntentionCostLedger.from_payload(ledger.state_payload())
    assert restored.configuration() == ledger.configuration()
    assert restored.estimate(context, cell_count=3) == ledger.estimate(
        context,
        cell_count=3,
    )


def test_cost_model_is_opaque_replay_free_and_branch_local() -> None:
    model = ExternalRoutedIntentionCostModel(3, learning_rate=0.4, initial_cost=0.5)
    state = model.initial_state()
    context = torch.tensor([[0.3, -0.2, 0.8]])
    mask = torch.tensor([[True, False, True]])
    before = model.estimate(
        state,
        context,
        context_mask=mask,
        source_coverage=0.75,
        cell_count=4,
    )
    updated = model.observe(
        state,
        context,
        context_mask=mask,
        source_coverage=0.75,
        cell_count=4,
        selected_initialization="transfer",
        observed_cost=0.0,
    )
    after = model.estimate(
        updated,
        context,
        context_mask=mask,
        source_coverage=0.75,
        cell_count=4,
    )

    assert before.transfer_cost == pytest.approx(0.5)
    assert after.transfer_cost < before.transfer_cost
    assert after.fresh_cost == pytest.approx(before.fresh_cost)
    assert int(updated.transfer_observations.item()) == 1
    assert int(updated.fresh_observations.item()) == 0
    assert torch.equal(state.fresh_weights, updated.fresh_weights)


def test_cost_model_payload_round_trip_is_exact() -> None:
    model = ExternalRoutedIntentionCostModel(2, initial_cost=0.2)
    state = model.initial_state()
    state = model.observe(
        state,
        torch.tensor([[1.0, 0.0]]),
        selected_initialization="fresh",
        observed_cost=0.8,
        cell_count=2,
    )
    restored = model.state_from_payload(model.state_payload(state))

    for name, value in state._tensors().items():
        assert torch.equal(value, restored._tensors()[name])
    assert model.estimate(restored, torch.tensor([[1.0, 0.0]]), cell_count=2) == model.estimate(
        state,
        torch.tensor([[1.0, 0.0]]),
        cell_count=2,
    )


def test_cost_model_rejects_invalid_observations() -> None:
    model = ExternalRoutedIntentionCostModel(2)
    state = model.initial_state()
    with pytest.raises(ValueError, match="mask"):
        model.estimate(state, torch.zeros(1, 2), context_mask=torch.ones(1, 2))
    with pytest.raises(ValueError, match="observed cost"):
        model.observe(
            state,
            torch.zeros(1, 2),
            selected_initialization="transfer",
            observed_cost=1.1,
        )
    with pytest.raises(ValueError, match="branch"):
        model.observe(
            state,
            torch.zeros(1, 2),
            selected_initialization="unknown",
            observed_cost=0.2,
        )


def test_cost_model_state_validation_rejects_wrong_shape() -> None:
    model = ExternalRoutedIntentionCostModel(2)
    state = model.initial_state()
    invalid = ExternalRoutedIntentionCostModelState(
        transfer_weights=torch.zeros(3),
        fresh_weights=state.fresh_weights,
        transfer_bias=state.transfer_bias,
        fresh_bias=state.fresh_bias,
        transfer_observations=state.transfer_observations,
        fresh_observations=state.fresh_observations,
        transfer_absolute_error=state.transfer_absolute_error,
        fresh_absolute_error=state.fresh_absolute_error,
    )
    with pytest.raises(ValueError, match="wrong shape"):
        invalid.validate(feature_width=model.feature_width)
