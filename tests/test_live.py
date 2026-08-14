from __future__ import annotations

import pytest
import torch

from neural_computer import (
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    CognitiveTickRuntime,
    ExternalProgramArtifact,
    ExternalTemporalProgramBank,
    IntentEvent,
    LiveActionProposal,
    LiveActionReceipt,
    LiveInputBatch,
    LiveInputInstruction,
    LiveOutcomeEvent,
    QueuedOutcomeInputDevice,
    ResolvedLiveOutcome,
    TemporalProgramOutcomeObserver,
)


class _LoopbackDevice:
    batch_size = 1
    event_width = 4

    def __init__(self, *, present: bool = True) -> None:
        self._first = True
        self._outcomes: list[LiveOutcomeEvent] = []
        self.present = present
        self.receipts: list[LiveActionReceipt] = []

    def poll(self, now: float) -> LiveInputBatch:
        if self._first:
            events = AmodalEventCollection.from_events(
                (AmodalEvent(torch.ones(1, self.event_width)),),
                width=self.event_width,
            )
            self._first = False
        else:
            events = AmodalEventCollection.empty(1, self.event_width)
        outcomes = tuple(self._outcomes)
        self._outcomes.clear()
        return LiveInputBatch(events, outcomes, now)

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        self.receipts.append(receipt)
        present = torch.tensor([self.present])
        self._outcomes.append(
            LiveOutcomeEvent(
                receipt_id=receipt.receipt_id,
                reward=torch.where(present, torch.ones(1), torch.zeros(1)),
                present=present,
                observed_at=receipt.emitted_at,
            )
        )


class _OneShotMachine:
    def __init__(self) -> None:
        self.outcomes: list[ResolvedLiveOutcome] = []

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: tuple[ResolvedLiveOutcome, ...],
        *,
        now: float,
        elapsed: float,
    ) -> tuple[LiveActionProposal, ...]:
        del now, elapsed
        self.outcomes.extend(outcomes)
        if events.payload.shape[1] == 0:
            return ()
        return (
            LiveActionProposal(
                intention=IntentEvent(torch.zeros(1, 3)),
                action=torch.tensor([1]),
                propensity=torch.tensor([0.25]),
                output_key="keyboard",
                model_version=7,
                credit_state={"opaque": 1},
            ),
        )


class _EventInputPort:
    batch_size = 1
    event_width = 4

    def __init__(self, value: float, source: tuple[float, float]) -> None:
        self.value = value
        self.source = source
        self.pending = True

    def poll(self, now: float) -> LiveInputBatch:
        if self.pending:
            events = AmodalEventCollection.from_events(
                (
                    AmodalEvent(
                        torch.full((1, self.event_width), self.value),
                        source_key=torch.tensor([self.source]),
                        timestamp=torch.tensor([now]),
                    ),
                ),
                width=self.event_width,
            )
            self.pending = False
        else:
            events = AmodalEventCollection.empty(1, self.event_width)
        return LiveInputBatch(events, (), now)


class _ReceiptSink:
    def __init__(self) -> None:
        self.receipts: list[LiveActionReceipt] = []

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        del action
        self.receipts.append(receipt)


def test_live_tick_binds_outcome_to_exact_detached_action_receipt() -> None:
    device = _LoopbackDevice()
    machine = _OneShotMachine()
    runtime = CognitiveTickRuntime(device, machine, {"keyboard": device})

    first = runtime.tick(1.0)
    assert first.input_event_count == 1
    assert first.pending_receipt_count == 1
    assert len(first.emitted_receipts) == 1
    receipt = first.emitted_receipts[0]
    assert receipt.receipt_id == 1
    assert receipt.model_version == 7
    assert receipt.action.tolist() == [1]
    assert receipt.action.requires_grad is False

    second = runtime.tick(1.1)
    assert second.input_event_count == 0
    assert second.outcome_bit_count == 1
    assert second.pending_receipt_count == 0
    assert len(machine.outcomes) == 1
    resolved = machine.outcomes[0]
    assert resolved.receipt == receipt
    assert resolved.proposal.credit_state == {"opaque": 1}


def test_live_tick_distinguishes_explicit_missing_evidence_from_zero_reward() -> None:
    device = _LoopbackDevice(present=False)
    machine = _OneShotMachine()
    runtime = CognitiveTickRuntime(device, machine, {"keyboard": device})

    runtime.tick(0.0)
    result = runtime.tick(0.5)

    assert result.outcome_bit_count == 0
    assert len(result.resolved_outcomes) == 1
    assert result.resolved_outcomes[0].event.present.tolist() == [False]
    assert result.resolved_outcomes[0].event.reward.tolist() == [0.0]
    assert result.pending_receipt_count == 0


def test_live_tick_rejects_nonmonotonic_time() -> None:
    device = _LoopbackDevice()
    runtime = CognitiveTickRuntime(device, _OneShotMachine(), {"keyboard": device})
    runtime.tick(2.0)

    try:
        runtime.tick(1.0)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("nonmonotonic live time was accepted")


def test_live_tick_rejects_unknown_outcome_receipts() -> None:
    device = _LoopbackDevice()
    device._first = False
    device._outcomes.append(
        LiveOutcomeEvent(
            receipt_id=99,
            reward=torch.ones(1),
            present=torch.ones(1, dtype=torch.bool),
            observed_at=0.0,
        )
    )
    runtime = CognitiveTickRuntime(device, _OneShotMachine(), {"keyboard": device})

    try:
        runtime.tick(0.0)
    except ValueError as error:
        assert "unknown or resolved receipt" in str(error)
    else:
        raise AssertionError("unknown causal receipt was accepted")


def test_input_instruction_polls_sensory_and_reward_ports_together() -> None:
    sensory = _EventInputPort(1.0, (1.0, 0.0))
    reward = QueuedOutcomeInputDevice(1, 4)
    instruction = LiveInputInstruction({"sensory": sensory, "verifier": reward})
    sink = _ReceiptSink()
    machine = _OneShotMachine()
    runtime = CognitiveTickRuntime(instruction, machine, {"keyboard": sink})

    first = runtime.tick(0.0)
    reward.submit(first.emitted_receipts[0], 1.0, observed_at=0.1)
    second = runtime.tick(0.1)

    assert instruction.port_count == 2
    assert first.input_event_count == 1
    assert second.outcome_bit_count == 1
    assert len(machine.outcomes) == 1
    assert machine.outcomes[0].event.reward.tolist() == [1.0]
    assert reward.pending_count == 0


def test_input_instruction_attaches_streams_without_resizing_event_abi() -> None:
    first = _EventInputPort(1.0, (1.0, 0.0))
    instruction = LiveInputInstruction({"first": first})
    instruction.attach("second", _EventInputPort(2.0, (0.0, 1.0)))

    batch = instruction.poll(0.0)

    assert instruction.event_width == 4
    assert batch.events.payload.shape == (1, 2, 4)
    assert batch.events.source_key.shape == (1, 2, 2)
    assert batch.events.payload[0, :, 0].tolist() == [1.0, 2.0]
    assert instruction.detach("second").event_width == 4
    with pytest.raises(ValueError, match="final port"):
        instruction.detach("first")


def test_input_instruction_rejects_duplicate_reward_receipts_across_ports() -> None:
    sensory = _EventInputPort(1.0, (1.0, 0.0))
    first_reward = QueuedOutcomeInputDevice(1, 4)
    second_reward = QueuedOutcomeInputDevice(1, 4)
    instruction = LiveInputInstruction(
        {"sensory": sensory, "reward-a": first_reward, "reward-b": second_reward}
    )
    sink = _ReceiptSink()
    runtime = CognitiveTickRuntime(instruction, _OneShotMachine(), {"keyboard": sink})
    receipt = runtime.tick(0.0).emitted_receipts[0]
    first_reward.submit(receipt, 1.0, observed_at=0.1)
    second_reward.submit(receipt, 0.0, observed_at=0.1)

    with pytest.raises(ValueError, match="resolve a receipt twice"):
        runtime.tick(0.1)


def test_reward_input_updates_only_bound_external_program_route() -> None:
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    artifact = ExternalProgramArtifact(
        codes=torch.tensor([[5.0, -3.0, -3.0]]),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    bank = ExternalTemporalProgramBank(
        4,
        3,
        controller_digest="0" * 64,
        min_mastery_observations=3,
    )
    bank.admit(
        artifact,
        context,
        [1.0, 1.0, 1.0],
        min_observations=3,
        min_stable_observations=3,
    )
    selection = bank.select(context)
    observer = TemporalProgramOutcomeObserver(bank, selection)
    reward = QueuedOutcomeInputDevice(1, 4)
    instruction = LiveInputInstruction(
        {"sensory": _EventInputPort(1.0, (1.0, 0.0)), "reward": reward}
    )
    sink = _ReceiptSink()
    runtime = CognitiveTickRuntime(
        instruction,
        _OneShotMachine(),
        {"keyboard": sink},
        outcome_observers=(observer,),
    )
    artifact_before = bank.artifact(0).digest()
    bank_before = bank.digest()
    receipt = runtime.tick(0.0).emitted_receipts[0]
    reward.submit(receipt, 1.0, observed_at=0.1)

    result = runtime.tick(0.1)

    assert result.outcome_bit_count == 1
    assert observer.unique_outcome_bits == 1
    assert bank.digest() != bank_before
    assert bank.artifact(0).digest() == artifact_before
