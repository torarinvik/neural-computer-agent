"""Event-driven live interaction transport for the amodal runtime.

The classes in this module own timing, causal receipts, and device dispatch.
They deliberately do not own perception, reasoning, device protocols, or a
reward-specific cognitive operation. Frontends poll raw devices and emit
learned :class:`AmodalEventCollection` objects. Cognitive machines emit opaque
intentions plus already-decoded actions. Trusted scalar outcomes return later
through the input side and are resolved against the exact action proposal that
created them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch

from .interface import AmodalEventCollection, IntentEvent

LIVE_ACTION_RECEIPT_SCHEMA = "neural-computer.live-action-receipt.v1"
LIVE_INPUT_INSTRUCTION_SCHEMA = "neural-computer.live-input-instruction.v1"
LIVE_OUTCOME_EVENT_SCHEMA = "neural-computer.live-outcome-event.v1"
LIVE_TICK_SCHEMA = "neural-computer.live-tick.v1"
QUEUED_OUTCOME_INPUT_SCHEMA = "neural-computer.queued-outcome-input.v1"


def _validate_time(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_action(action: torch.Tensor, *, batch_size: int) -> None:
    if action.ndim < 1 or action.shape[0] != batch_size:
        raise ValueError("opaque action must have the live batch on its first axis")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("opaque action must be finite")


def _validate_propensity(propensity: torch.Tensor, *, batch_size: int) -> None:
    if propensity.shape != (batch_size,):
        raise ValueError("logging propensity must have shape [batch]")
    if not bool(torch.isfinite(propensity).all()):
        raise ValueError("logging propensity must be finite")
    if bool(torch.any((propensity <= 0.0) | (propensity > 1.0))):
        raise ValueError("logging propensity must lie in (0, 1]")


@dataclass(frozen=True)
class LiveActionProposal:
    """One cognitive intention and its opaque, externally decoded action.

    ``credit_state`` is private trainer state retained until causal evidence
    arrives. Delayed-credit implementations should prefer detached sufficient
    statistics over a long-lived autograd graph. The public
    :class:`LiveActionReceipt` is always detached.
    """

    intention: IntentEvent
    action: torch.Tensor
    propensity: torch.Tensor
    output_key: str
    model_version: int = 0
    credit_state: object | None = None

    def validate(self, *, batch_size: int) -> LiveActionProposal:
        self.intention.validate()
        if self.intention.payload.shape[0] != batch_size:
            raise ValueError("intention batch does not match the live runtime")
        _validate_action(self.action, batch_size=batch_size)
        _validate_propensity(self.propensity, batch_size=batch_size)
        if not self.output_key:
            raise ValueError("output key must be non-empty")
        if self.model_version < 0:
            raise ValueError("model version must be non-negative")
        return self


@dataclass(frozen=True)
class LiveActionReceipt:
    """Authenticated transport record for one emitted opaque action."""

    receipt_id: int
    action: torch.Tensor
    propensity: torch.Tensor
    output_key: str
    emitted_at: float
    model_version: int
    schema: str = LIVE_ACTION_RECEIPT_SCHEMA

    def validate(self, *, batch_size: int) -> LiveActionReceipt:
        if self.schema != LIVE_ACTION_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported action receipt schema: {self.schema}")
        if self.receipt_id < 1:
            raise ValueError("receipt id must be positive")
        _validate_action(self.action, batch_size=batch_size)
        _validate_propensity(self.propensity, batch_size=batch_size)
        if not self.output_key:
            raise ValueError("output key must be non-empty")
        _validate_time(self.emitted_at, "emission time")
        if self.model_version < 0:
            raise ValueError("model version must be non-negative")
        return self


@dataclass(frozen=True)
class LiveOutcomeEvent:
    """Trusted scalar verifier evidence linked to an opaque action receipt.

    ``present=False`` closes an action without fabricating a zero reward. It is
    useful for warm-up or explicitly unscored actions. An action for which no
    event has arrived remains pending and is observably different from this
    explicit no-evidence resolution.
    """

    receipt_id: int
    reward: torch.Tensor
    present: torch.Tensor
    observed_at: float
    confidence: torch.Tensor | None = None
    schema: str = LIVE_OUTCOME_EVENT_SCHEMA

    def validate(self, *, batch_size: int) -> LiveOutcomeEvent:
        if self.schema != LIVE_OUTCOME_EVENT_SCHEMA:
            raise ValueError(f"unsupported outcome schema: {self.schema}")
        if self.receipt_id < 1:
            raise ValueError("outcome receipt id must be positive")
        if self.reward.shape != (batch_size,) or not bool(
            torch.isfinite(self.reward).all()
        ):
            raise ValueError("outcome reward must be finite with shape [batch]")
        if self.present.shape != (batch_size,) or self.present.dtype != torch.bool:
            raise ValueError("outcome presence must be boolean with shape [batch]")
        if bool(torch.any(self.reward[~self.present] != 0.0)):
            raise ValueError("an absent outcome cannot carry a reward value")
        if self.confidence is not None:
            if self.confidence.shape != (batch_size,) or not bool(
                torch.isfinite(self.confidence).all()
            ):
                raise ValueError(
                    "outcome confidence must be finite with shape [batch]"
                )
            if bool(torch.any(self.confidence < 0.0)):
                raise ValueError("outcome confidence cannot be negative")
        _validate_time(self.observed_at, "outcome observation time")
        return self


@dataclass(frozen=True)
class LiveInputBatch:
    """Learned sensory events and trusted outcomes returned by ``RECEIVE``."""

    events: AmodalEventCollection
    outcomes: tuple[LiveOutcomeEvent, ...]
    observed_at: float

    def validate(self, *, batch_size: int, event_width: int) -> LiveInputBatch:
        self.events.validate(width=event_width)
        if self.events.payload.shape[0] != batch_size:
            raise ValueError("input event batch does not match the live runtime")
        _validate_time(self.observed_at, "input observation time")
        receipt_ids: set[int] = set()
        for outcome in self.outcomes:
            outcome.validate(batch_size=batch_size)
            if outcome.receipt_id in receipt_ids:
                raise ValueError("one input poll cannot resolve a receipt twice")
            receipt_ids.add(outcome.receipt_id)
        return self


@dataclass(frozen=True)
class ResolvedLiveOutcome:
    """An outcome paired with the exact proposal and public receipt."""

    event: LiveOutcomeEvent
    receipt: LiveActionReceipt
    proposal: LiveActionProposal


@dataclass(frozen=True)
class LiveTickResult:
    """Auditable result of one cognitive tick."""

    tick: int
    observed_at: float
    elapsed: float
    input_event_count: int
    outcome_bit_count: int
    resolved_outcomes: tuple[ResolvedLiveOutcome, ...]
    emitted_receipts: tuple[LiveActionReceipt, ...]
    pending_receipt_count: int
    input_seconds: float
    machine_seconds: float
    output_seconds: float
    total_seconds: float
    deadline_missed: bool
    schema: str = LIVE_TICK_SCHEMA


class LiveInputDevice(Protocol):
    """Replaceable frontend that polls raw devices and emits learned events."""

    batch_size: int
    event_width: int

    def poll(self, now: float) -> LiveInputBatch: ...


def _merge_event_collections(
    collections: Sequence[AmodalEventCollection],
    *,
    batch_size: int,
    event_width: int,
) -> AmodalEventCollection:
    """Concatenate variable input ports without reducing simultaneous streams."""

    if not collections:
        return AmodalEventCollection.empty(batch_size, event_width)
    validated = [collection.validate(width=event_width) for collection in collections]
    reference = validated[0].payload
    for collection in validated:
        if collection.payload.shape[0] != batch_size:
            raise ValueError("input port event batches do not match")
        if (
            collection.payload.device != reference.device
            or collection.payload.dtype != reference.dtype
        ):
            raise ValueError("input port event tensors must share device and dtype")
    nonempty = [collection for collection in validated if collection.payload.shape[1]]
    source_key = None
    if nonempty:
        source_presence = [collection.source_key is not None for collection in nonempty]
        if any(source_presence) and not all(source_presence):
            raise ValueError("input ports must either all bind source keys or none")
        if all(source_presence):
            key_widths = {
                int(collection.source_key.shape[2])
                for collection in nonempty
                if collection.source_key is not None
            }
            if len(key_widths) != 1:
                raise ValueError("input port source-key widths do not match")
            source_key = torch.cat(
                [
                    collection.source_key
                    for collection in nonempty
                    if collection.source_key is not None
                ],
                dim=1,
            )

    def merge_optional(
        field: str, present_field: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not any(getattr(collection, field) is not None for collection in nonempty):
            return None, None
        values: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for collection in nonempty:
            count = collection.payload.shape[1]
            value = getattr(collection, field)
            present = getattr(collection, present_field)
            if value is None:
                values.append(
                    torch.zeros(
                        batch_size,
                        count,
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                )
                masks.append(
                    torch.zeros(
                        batch_size,
                        count,
                        device=reference.device,
                        dtype=torch.bool,
                    )
                )
            else:
                if present is None:
                    raise ValueError(f"input port {field} lacks a presence mask")
                values.append(value)
                masks.append(present)
        return torch.cat(values, dim=1), torch.cat(masks, dim=1)

    timestamp, timestamp_present = merge_optional("timestamp", "timestamp_present")
    duration, duration_present = merge_optional("duration", "duration_present")
    return AmodalEventCollection(
        payload=torch.cat([collection.payload for collection in validated], dim=1),
        present=torch.cat([collection.present for collection in validated], dim=1),
        confidence=torch.cat(
            [collection.confidence for collection in validated], dim=1
        ),
        source_key=source_key,
        timestamp=timestamp,
        timestamp_present=timestamp_present,
        duration=duration,
        duration_present=duration_present,
    ).validate(width=event_width)


class LiveInputInstruction:
    """Poll a runtime-variable set of sensory and verifier input ports.

    Ports all implement :class:`LiveInputDevice`; the instruction concatenates
    learned events and causal scalar outcomes without assigning semantic roles
    to either. Port handles remain external transport metadata and are never
    included in controller tensors.
    """

    schema = LIVE_INPUT_INSTRUCTION_SCHEMA

    def __init__(self, ports: Mapping[str, LiveInputDevice]) -> None:
        if not ports or any(not name for name in ports):
            raise ValueError("INPUT requires at least one named port")
        first = next(iter(ports.values()))
        if min(first.batch_size, first.event_width) < 1:
            raise ValueError("INPUT port dimensions must be positive")
        self.batch_size = int(first.batch_size)
        self.event_width = int(first.event_width)
        self._ports: dict[str, LiveInputDevice] = {}
        for name, port in ports.items():
            self.attach(name, port)

    @property
    def port_count(self) -> int:
        return len(self._ports)

    @property
    def port_names(self) -> tuple[str, ...]:
        """Return external handles for diagnostics, never controller input."""

        return tuple(self._ports)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "batch_size": self.batch_size,
            "event_width": self.event_width,
            "port_count": self.port_count,
            "merge": "source_preserving_event_concat_plus_causal_outcomes_v1",
        }

    def attach(self, name: str, port: LiveInputDevice) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("INPUT port name must be nonempty")
        if name in self._ports:
            raise ValueError("INPUT port name is already attached")
        if (
            int(port.batch_size) != self.batch_size
            or int(port.event_width) != self.event_width
        ):
            raise ValueError("INPUT port dimensions must match the fixed event ABI")
        self._ports[name] = port

    def detach(self, name: str) -> LiveInputDevice:
        if name not in self._ports:
            raise KeyError("INPUT port is not attached")
        if len(self._ports) == 1:
            raise ValueError("INPUT cannot detach its final port")
        return self._ports.pop(name)

    def poll(self, now: float) -> LiveInputBatch:
        batches = [
            port.poll(now).validate(
                batch_size=self.batch_size, event_width=self.event_width
            )
            for port in self._ports.values()
        ]
        events = _merge_event_collections(
            [batch.events for batch in batches],
            batch_size=self.batch_size,
            event_width=self.event_width,
        )
        outcomes = tuple(
            outcome for batch in batches for outcome in batch.outcomes
        )
        return LiveInputBatch(events, outcomes, now).validate(
            batch_size=self.batch_size, event_width=self.event_width
        )


class QueuedOutcomeInputDevice:
    """Generic reward/verifier input port with exact action attribution.

    Environments submit only outcomes for receipts the agent actually emitted.
    Polling drains each outcome exactly once. The port contains no task names,
    correct-action labels, or program identities.
    """

    schema = QUEUED_OUTCOME_INPUT_SCHEMA

    def __init__(self, batch_size: int, event_width: int) -> None:
        if min(batch_size, event_width) < 1:
            raise ValueError("outcome input dimensions must be positive")
        self.batch_size = int(batch_size)
        self.event_width = int(event_width)
        self._queue: list[LiveOutcomeEvent] = []

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "batch_size": self.batch_size,
            "event_width": self.event_width,
            "queue": "exact_once_causal_receipt_outcomes_v1",
        }

    def submit(
        self,
        receipt: LiveActionReceipt,
        reward: torch.Tensor | float,
        *,
        observed_at: float,
        present: torch.Tensor | bool = True,
        confidence: torch.Tensor | float | None = None,
    ) -> None:
        receipt.validate(batch_size=self.batch_size)

        def batch_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=dtype)
            if tensor.ndim == 0 and self.batch_size == 1:
                tensor = tensor.reshape(1)
            return tensor

        reward_tensor = batch_tensor(reward, dtype=torch.float32)
        present_tensor = batch_tensor(present, dtype=torch.bool)
        confidence_tensor = (
            None
            if confidence is None
            else batch_tensor(confidence, dtype=torch.float32)
        )
        outcome = LiveOutcomeEvent(
            receipt_id=receipt.receipt_id,
            reward=reward_tensor,
            present=present_tensor,
            observed_at=observed_at,
            confidence=confidence_tensor,
        ).validate(batch_size=self.batch_size)
        self._queue.append(outcome)

    def poll(self, now: float) -> LiveInputBatch:
        outcomes = tuple(self._queue)
        self._queue.clear()
        return LiveInputBatch(
            AmodalEventCollection.empty(self.batch_size, self.event_width),
            outcomes,
            now,
        )


class LiveOutcomeObserver(Protocol):
    """External learner that consumes attributed outcome input events."""

    def observe(self, outcome: ResolvedLiveOutcome) -> None: ...


class LiveOutputDevice(Protocol):
    """Replaceable backend for one externally decoded action protocol."""

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None: ...


class LiveCognitiveMachine(Protocol):
    """One fixed-width cognitive machine driven by an event collection."""

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        now: float,
        elapsed: float,
    ) -> Sequence[LiveActionProposal]: ...


class CognitiveTickRuntime:
    """Execute ``RECEIVE -> learn/think -> EMIT`` with causal accounting."""

    def __init__(
        self,
        input_device: LiveInputDevice,
        machine: LiveCognitiveMachine,
        output_devices: Mapping[str, LiveOutputDevice],
        *,
        outcome_observers: Sequence[LiveOutcomeObserver] = (),
        max_machine_seconds: float | None = None,
        max_tick_seconds: float | None = None,
    ) -> None:
        if input_device.batch_size < 1 or input_device.event_width < 1:
            raise ValueError("live input dimensions must be positive")
        if not output_devices or any(not key for key in output_devices):
            raise ValueError("at least one named output device is required")
        if max_machine_seconds is not None and max_machine_seconds <= 0.0:
            raise ValueError("machine deadline must be positive")
        if max_tick_seconds is not None and max_tick_seconds <= 0.0:
            raise ValueError("tick deadline must be positive")
        self.input_device = input_device
        self.machine = machine
        self.output_devices = dict(output_devices)
        self.outcome_observers = tuple(outcome_observers)
        if any(not callable(getattr(observer, "observe", None)) for observer in self.outcome_observers):
            raise TypeError("live outcome observers must implement observe")
        self.max_machine_seconds = max_machine_seconds
        self.max_tick_seconds = max_tick_seconds
        self.batch_size = int(input_device.batch_size)
        self.event_width = int(input_device.event_width)
        self._tick = 0
        self._next_receipt_id = 1
        self._last_now: float | None = None
        self._pending: dict[
            int, tuple[LiveActionReceipt, LiveActionProposal]
        ] = {}

    @property
    def pending_receipts(self) -> tuple[LiveActionReceipt, ...]:
        return tuple(receipt for receipt, _proposal in self._pending.values())

    def tick(self, now: float) -> LiveTickResult:
        """Advance one monotonic cognitive tick.

        Outcomes are delivered exactly once. Device emission is transactional
        with respect to the pending-receipt table: a backend exception cannot
        leave a receipt that was never physically emitted.
        """

        tick_started = perf_counter()
        _validate_time(now, "tick time")
        if self._last_now is not None and now < self._last_now:
            raise ValueError("live tick time must be monotonic")
        elapsed = 0.0 if self._last_now is None else now - self._last_now
        input_started = perf_counter()
        input_batch = self.input_device.poll(now).validate(
            batch_size=self.batch_size,
            event_width=self.event_width,
        )
        input_seconds = perf_counter() - input_started
        resolved: list[ResolvedLiveOutcome] = []
        for outcome in input_batch.outcomes:
            pending = self._pending.get(outcome.receipt_id)
            if pending is None:
                raise ValueError("outcome references an unknown or resolved receipt")
            receipt, proposal = pending
            if outcome.observed_at < receipt.emitted_at:
                raise ValueError("outcome cannot precede its action emission")
            resolved.append(ResolvedLiveOutcome(outcome, receipt, proposal))

        started = perf_counter()
        for item in resolved:
            for observer in self.outcome_observers:
                observer.observe(item)
        proposals = tuple(
            self.machine.tick(
                input_batch.events,
                tuple(resolved),
                now=now,
                elapsed=elapsed,
            )
        )
        machine_seconds = perf_counter() - started

        for item in resolved:
            del self._pending[item.receipt.receipt_id]

        for proposal in proposals:
            proposal.validate(batch_size=self.batch_size)
            if proposal.output_key not in self.output_devices:
                raise KeyError(
                    f"no output device is registered as {proposal.output_key!r}"
                )

        emitted: list[LiveActionReceipt] = []
        output_started = perf_counter()
        for proposal in proposals:
            device = self.output_devices[proposal.output_key]
            receipt = LiveActionReceipt(
                receipt_id=self._next_receipt_id,
                action=proposal.action.detach().clone(),
                propensity=proposal.propensity.detach().clone(),
                output_key=proposal.output_key,
                emitted_at=now,
                model_version=proposal.model_version,
            ).validate(batch_size=self.batch_size)
            self._next_receipt_id += 1
            self._pending[receipt.receipt_id] = (receipt, proposal)
            try:
                device.emit(receipt.action, receipt)
            except Exception:
                del self._pending[receipt.receipt_id]
                raise
            emitted.append(receipt)
        output_seconds = perf_counter() - output_started

        self._tick += 1
        self._last_now = now
        event_count = int(input_batch.events.present.sum().item())
        outcome_bits = sum(
            int(item.event.present.sum().item()) for item in resolved
        )
        total_seconds = perf_counter() - tick_started
        return LiveTickResult(
            tick=self._tick,
            observed_at=input_batch.observed_at,
            elapsed=elapsed,
            input_event_count=event_count,
            outcome_bit_count=outcome_bits,
            resolved_outcomes=tuple(resolved),
            emitted_receipts=tuple(emitted),
            pending_receipt_count=len(self._pending),
            input_seconds=input_seconds,
            machine_seconds=machine_seconds,
            output_seconds=output_seconds,
            total_seconds=total_seconds,
            deadline_missed=(
                (
                    self.max_machine_seconds is not None
                    and machine_seconds > self.max_machine_seconds
                )
                or (
                    self.max_tick_seconds is not None
                    and total_seconds > self.max_tick_seconds
                )
            ),
        )
