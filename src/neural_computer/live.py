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
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

import torch

from .agent_brain_bank import ExternalAgentBrainBank
from .executive import ExternalAmodalExecutive, TrustedExternalExecutiveState
from .executive_route import (
    ExternalExecutiveSkillRouter,
    ExternalExecutiveSkillSelection,
)
from .interface import AmodalEventCollection, IntentEvent
from .program import ExternalProgramAdmissionReceipt

LIVE_ACTION_RECEIPT_SCHEMA = "neural-computer.live-action-receipt.v1"
LIVE_INPUT_INSTRUCTION_SCHEMA = "neural-computer.live-input-instruction.v1"
LIVE_OUTCOME_EVENT_SCHEMA = "neural-computer.live-outcome-event.v1"
LIVE_TICK_SCHEMA = "neural-computer.live-tick.v1"
QUEUED_OUTCOME_INPUT_SCHEMA = "neural-computer.queued-outcome-input.v1"
LIVE_EXECUTIVE_MACHINE_SCHEMA = "neural-computer.live-executive-machine.v1"
LIVE_EXECUTIVE_ROUTER_MACHINE_SCHEMA = (
    "neural-computer.live-executive-router-machine.v1"
)
LIVE_EXECUTIVE_ADMISSION_MACHINE_SCHEMA = (
    "neural-computer.live-executive-admission-machine.v1"
)


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


class LiveIntentionDecoder(Protocol):
    """Replaceable output adapter for an external executive intention."""

    intention_width: int

    def decide(
        self, intention: IntentEvent, *, sample: bool = True
    ) -> LiveDecoderDecision: ...


class LiveDecoderDecision(Protocol):
    """Minimal action/propensity result shared by all output decoders."""

    action: torch.Tensor
    propensity: torch.Tensor


@dataclass(frozen=True)
class ExternalExecutiveLiveCredit:
    """Opaque receipt-local identity for routing delayed outcomes."""

    program_digest: str
    operator_registry_digest: str
    executive_tick: int
    executed_instructions: int
    schema: str = LIVE_EXECUTIVE_MACHINE_SCHEMA

    def validate(self) -> ExternalExecutiveLiveCredit:
        if self.schema != LIVE_EXECUTIVE_MACHINE_SCHEMA:
            raise ValueError("unsupported live executive credit schema")
        for name, value in (
            ("program_digest", self.program_digest),
            ("operator_registry_digest", self.operator_registry_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"live executive {name} must be a SHA-256 digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(
                    f"live executive {name} must be a SHA-256 digest"
                ) from error
        if (
            not isinstance(self.executive_tick, int)
            or isinstance(self.executive_tick, bool)
            or not isinstance(self.executed_instructions, int)
            or isinstance(self.executed_instructions, bool)
            or self.executive_tick < 1
            or self.executed_instructions < 0
        ):
            raise ValueError("live executive credit counters are invalid")
        return self


class ExternalExecutiveLiveMachine:
    """Run one admitted external skill inside the live causal tick boundary.

    The executive and its state are deliberately separate from the decoder.
    The executive is frozen program logic; the decoder is the replaceable
    intention-to-device adapter. Resolved outcomes are passed through the
    normal runtime to observers, while this machine never trains or exposes
    them to the controller.
    """

    schema = LIVE_EXECUTIVE_MACHINE_SCHEMA

    def __init__(
        self,
        executive: ExternalAmodalExecutive,
        decoder: LiveIntentionDecoder,
        *,
        batch_size: int,
        output_key: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sample: bool = True,
        model_version: int = 0,
        skill_digest: str | None = None,
        freeze_decoder: bool = True,
    ) -> None:
        if not isinstance(executive, ExternalAmodalExecutive):
            raise TypeError("live executive machine needs an external executive")
        if not callable(getattr(decoder, "decide", None)):
            raise TypeError("live executive machine decoder must implement decide")
        decoder_width = getattr(decoder, "intention_width", None)
        if decoder_width != executive.intention_width:
            raise ValueError("live executive decoder intention width is incompatible")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
            or not isinstance(output_key, str)
            or not output_key
        ):
            raise ValueError("live executive machine dimensions and output are invalid")
        if (
            not isinstance(model_version, int)
            or isinstance(model_version, bool)
            or model_version < 0
        ):
            raise ValueError("live executive model version cannot be negative")
        resolved_skill_digest = (
            executive.program.digest() if skill_digest is None else skill_digest
        )
        if (
            not isinstance(resolved_skill_digest, str)
            or len(resolved_skill_digest) != 64
        ):
            raise ValueError("live executive skill digest must be a SHA-256 digest")
        try:
            int(resolved_skill_digest, 16)
        except ValueError as error:
            raise ValueError(
                "live executive skill digest must be a SHA-256 digest"
            ) from error
        self.executive = executive
        self.decoder = decoder
        self.batch_size = int(batch_size)
        self.output_key = output_key
        self.sample = bool(sample)
        self.model_version = int(model_version)
        self.freeze_decoder = bool(freeze_decoder)
        if self.freeze_decoder:
            eval_method = getattr(decoder, "eval", None)
            if callable(eval_method):
                eval_method()
            parameters_method = getattr(decoder, "parameters", None)
            if callable(parameters_method):
                for parameter in parameters_method():
                    parameter.requires_grad_(False)
        self._device = torch.device(device)
        self._dtype = dtype
        self._skill_digest = resolved_skill_digest
        self._state: TrustedExternalExecutiveState = executive.initial_sealed_state(
            self.batch_size, device=self._device, dtype=self._dtype
        )
        self._executive_ticks = 0

    @classmethod
    def from_artifact(
        cls,
        artifact,
        decoder: LiveIntentionDecoder,
        *,
        batch_size: int,
        output_key: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sample: bool = True,
        model_version: int = 0,
        max_instructions_per_tick: int = 128,
        skill_digest: str | None = None,
        freeze_decoder: bool = True,
    ) -> ExternalExecutiveLiveMachine:
        """Instantiate a live machine from a validated bank artifact."""

        instantiate = getattr(artifact, "instantiate", None)
        digest = getattr(artifact, "digest", None)
        validate = getattr(artifact, "validate", None)
        if not callable(instantiate) or not callable(digest) or not callable(validate):
            raise TypeError("live executive artifact must be executable")
        validate()
        executive = instantiate(
            max_instructions_per_tick=max_instructions_per_tick
        )
        return cls(
            executive,
            decoder,
            batch_size=batch_size,
            output_key=output_key,
            device=device,
            dtype=dtype,
            sample=sample,
            model_version=model_version,
            skill_digest=(digest() if skill_digest is None else skill_digest),
            freeze_decoder=freeze_decoder,
        )

    @property
    def executive_state(self) -> TrustedExternalExecutiveState:
        """Return the live-only state lease; it cannot be serialized."""

        return self._state

    @property
    def executive_ticks(self) -> int:
        return self._executive_ticks

    def reset(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Start a clean skill lifetime without changing program or decoder weights."""

        self._state = self.executive.initial_sealed_state(
            self.batch_size,
            device=self._device if device is None else device,
            dtype=self._dtype if dtype is None else dtype,
        )
        self._executive_ticks = 0

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        now: float,
        elapsed: float,
    ) -> tuple[LiveActionProposal, ...]:
        del outcomes, now, elapsed
        output, self._state = self.executive.tick_fast(events, self._state)
        self._executive_ticks += 1
        if output.intention is None:
            return ()
        with torch.no_grad():
            decision = self.decoder.decide(
                output.intention,
                sample=self.sample,
            )
        action = getattr(decision, "action", None)
        propensity = getattr(decision, "propensity", None)
        if not isinstance(action, torch.Tensor) or not isinstance(propensity, torch.Tensor):
            raise TypeError(
                "live executive decoder must return action and propensity tensors"
            )
        credit = ExternalExecutiveLiveCredit(
            program_digest=self._skill_digest,
            operator_registry_digest=output.operator_registry_digest,
            executive_tick=self._executive_ticks,
            executed_instructions=output.executed_instructions,
        ).validate()
        return (
            LiveActionProposal(
                intention=output.intention,
                action=action.detach(),
                propensity=propensity.detach(),
                output_key=self.output_key,
                model_version=self.model_version,
                credit_state=credit,
            ).validate(batch_size=self.batch_size),
        )


class LiveContextEncoder(Protocol):
    """Replaceable learned-event to opaque route-context adapter."""

    context_width: int

    def encode(self, events: AmodalEventCollection) -> torch.Tensor: ...


@dataclass(frozen=True)
class ExternalExecutiveRouteCredit:
    """Receipt-local route choice used for delayed memory-side feedback."""

    selection: ExternalExecutiveSkillSelection
    executive_credit: ExternalExecutiveLiveCredit
    schema: str = LIVE_EXECUTIVE_ROUTER_MACHINE_SCHEMA

    def validate(self) -> ExternalExecutiveRouteCredit:
        if self.schema != LIVE_EXECUTIVE_ROUTER_MACHINE_SCHEMA:
            raise ValueError("unsupported live executive route credit schema")
        context = self.selection.context
        if not isinstance(context, torch.Tensor) or context.ndim != 1:
            raise TypeError("live executive route credit context is invalid")
        self.selection.validate(context_width=int(context.shape[0]))
        self.executive_credit.validate()
        if self.executive_credit.program_digest != self.selection.artifact.digest():
            raise ValueError("live executive route credit artifact digest is incompatible")
        return self


class ExternalExecutiveRouterLiveMachine:
    """Select one verified executive skill per live episode and learn its route."""

    schema = LIVE_EXECUTIVE_ROUTER_MACHINE_SCHEMA

    def __init__(
        self,
        router: ExternalExecutiveSkillRouter,
        decoder: LiveIntentionDecoder,
        context_encoder: LiveContextEncoder,
        *,
        batch_size: int,
        output_key: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sample: bool = True,
        exploration: float = 0.0,
        sample_route: bool = False,
        route_feedback_mode: str = "episode_mean",
        model_version: int = 0,
        max_instructions_per_tick: int = 128,
    ) -> None:
        if not isinstance(router, ExternalExecutiveSkillRouter):
            raise TypeError("live executive router machine needs a skill router")
        if not callable(getattr(context_encoder, "encode", None)):
            raise TypeError("live executive router needs a context encoder")
        context_width = getattr(context_encoder, "context_width", None)
        if context_width != router.context_width:
            raise ValueError("live executive route context width is incompatible")
        if batch_size != 1:
            raise ValueError("live executive route machine currently requires batch one")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("live executive route exploration must lie in [0, 1]")
        if route_feedback_mode not in {"episode_mean", "per_outcome"}:
            raise ValueError(
                "live executive route feedback mode must be episode_mean or per_outcome"
            )
        self.router = router
        self.decoder = decoder
        self.context_encoder = context_encoder
        self.batch_size = int(batch_size)
        self.output_key = output_key
        self.device = torch.device(device)
        self.dtype = dtype
        self.sample = bool(sample)
        self.exploration = float(exploration)
        self.sample_route = bool(sample_route)
        self.route_feedback_mode = route_feedback_mode
        self.model_version = int(model_version)
        self.max_instructions_per_tick = int(max_instructions_per_tick)
        self._machines: dict[int, ExternalExecutiveLiveMachine] = {}
        self._active_selection: ExternalExecutiveSkillSelection | None = None
        self._active_machine: ExternalExecutiveLiveMachine | None = None
        self._episode_selection: ExternalExecutiveSkillSelection | None = None
        self._episode_rewards: list[float] = []
        self.route_updates = 0

    @property
    def selected_slot(self) -> int | None:
        return None if self._active_selection is None else self._active_selection.slot

    @property
    def executive_ticks(self) -> int:
        return sum(machine.executive_ticks for machine in self._machines.values())

    def _context(self, events: AmodalEventCollection) -> torch.Tensor:
        encoded = self.context_encoder.encode(events)
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("live executive context encoder must return a tensor")
        if encoded.ndim == 2:
            if encoded.shape[0] != self.batch_size:
                raise ValueError("live executive context batch is incompatible")
            encoded = encoded[0]
        if encoded.ndim != 1 or encoded.shape[0] != self.router.context_width:
            raise ValueError("live executive context encoder returned the wrong shape")
        return encoded

    def _machine_for(
        self, selection: ExternalExecutiveSkillSelection
    ) -> ExternalExecutiveLiveMachine:
        machine = self._machines.get(selection.slot)
        if machine is None:
            machine = ExternalExecutiveLiveMachine.from_artifact(
                selection.artifact,
                self.decoder,
                batch_size=self.batch_size,
                output_key=self.output_key,
                device=self.device,
                dtype=self.dtype,
                sample=self.sample,
                model_version=self.model_version,
                max_instructions_per_tick=self.max_instructions_per_tick,
                skill_digest=selection.artifact.digest(),
            )
            self._machines[selection.slot] = machine
        return machine

    def _observe_outcomes(
        self, outcomes: Sequence[ResolvedLiveOutcome]
    ) -> None:
        for resolved in outcomes:
            credit = resolved.proposal.credit_state
            if not isinstance(credit, ExternalExecutiveRouteCredit):
                continue
            if resolved.event.present.shape != (self.batch_size,):
                raise ValueError("live executive route outcomes have the wrong batch")
            if not bool(resolved.event.present.item()):
                continue
            reward = float(resolved.event.reward.reshape(()).item())
            if self.route_feedback_mode == "per_outcome":
                self.router.observe(credit.selection, reward)
                self.route_updates += 1
            else:
                if self._episode_selection is None:
                    self._episode_selection = credit.selection
                elif (
                    self._episode_selection.slot != credit.selection.slot
                    or self._episode_selection.artifact.digest()
                    != credit.selection.artifact.digest()
                ):
                    raise RuntimeError("live executive route changed inside an episode")
                self._episode_rewards.append(reward)

    def finish_episode(self) -> float | None:
        """Commit one aggregate selector outcome at an episode boundary.

        Action receipts and verifier outcomes remain individual live events.
        Only the memory-side route ledger uses this aggregate by default, which
        prevents a lucky streak inside a failed lifetime from promoting a skill.
        """

        if self.route_feedback_mode == "per_outcome":
            return None
        if self._episode_selection is None or not self._episode_rewards:
            self._episode_selection = None
            self._episode_rewards.clear()
            return None
        outcome = sum(self._episode_rewards) / len(self._episode_rewards)
        self.router.observe(self._episode_selection, outcome)
        self.route_updates += 1
        self._episode_selection = None
        self._episode_rewards.clear()
        return outcome

    def reset(self) -> None:
        """Start a new episode and clear all per-skill executive states."""

        self.finish_episode()
        self._active_selection = None
        self._active_machine = None
        for machine in self._machines.values():
            machine.reset(device=self.device, dtype=self.dtype)

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        now: float,
        elapsed: float,
    ) -> tuple[LiveActionProposal, ...]:
        self._observe_outcomes(outcomes)
        if self._active_machine is None:
            if events.payload.shape[1] == 0:
                return ()
            selection = self.router.select(
                self._context(events),
                exploration=self.exploration,
                sample=self.sample_route,
            )
            self._active_selection = selection
            self._active_machine = self._machine_for(selection)
        assert self._active_selection is not None
        assert self._active_machine is not None
        proposals = self._active_machine.tick(
            events,
            (),
            now=now,
            elapsed=elapsed,
        )
        if proposals:
            credit = proposals[0].credit_state
            if not isinstance(credit, ExternalExecutiveLiveCredit):
                raise TypeError("live executive proposal lost its executive credit")
            route_credit = ExternalExecutiveRouteCredit(
                selection=self._active_selection,
                executive_credit=credit,
            ).validate()
            return tuple(
                replace(proposal, credit_state=route_credit)
                for proposal in proposals
            )
        if self._active_machine.executive_state.state.status == "halted":
            self._active_selection = None
            self._active_machine = None
        return ()


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


class ExternalExecutiveCandidateLiveMachine:
    """Evaluate one frozen executive candidate through the live tick boundary.

    The candidate is supplied by an outer search or proposal mechanism and is
    *not* visible to the controller.  This wrapper only accumulates the scalar
    verifier outcomes attached to this candidate's own action receipts.  At
    episode boundaries it applies the common stable-prefix admission gate and
    appends the immutable artifact to ``AgentBrain.bank`` only after the gate
    accepts it.  A rejected candidate leaves the bank byte-for-byte unchanged.

    This is deliberately a verifier-gated staging primitive, not autonomous
    program synthesis: candidate proposal remains an explicit upstream
    concern, while admission is causal, replay-free, and reusable by any live
    frontend that emits the normal receipt/outcome protocol.
    """

    schema = LIVE_EXECUTIVE_ADMISSION_MACHINE_SCHEMA

    def __init__(
        self,
        artifact,
        bank: ExternalAgentBrainBank,
        decoder: LiveIntentionDecoder,
        *,
        parent_slots: Sequence[int] | None = None,
        share_compatible_operators: bool = False,
        final_emit_only: bool = False,
        batch_size: int,
        output_key: str,
        threshold: float = 0.8,
        min_observations: int = 3,
        min_stable_observations: int = 2,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sample: bool = True,
        model_version: int = 0,
        max_instructions_per_tick: int = 128,
    ) -> None:
        if not isinstance(bank, ExternalAgentBrainBank):
            raise TypeError("live executive admission needs an AgentBrain bank")
        if not isinstance(share_compatible_operators, bool):
            raise TypeError("live executive operator sharing must be boolean")
        if not isinstance(final_emit_only, bool):
            raise TypeError("live executive final emit policy must be boolean")
        for method_name in ("digest", "validate", "instantiate"):
            if not callable(getattr(artifact, method_name, None)):
                raise TypeError(
                    "live executive admission candidate must be executable"
                )
        artifact.validate()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("live executive admission threshold must lie in [0, 1]")
        for name, value in (
            ("min_observations", min_observations),
            ("min_stable_observations", min_stable_observations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"live executive admission {name} must be positive")
        self.candidate_digest = artifact.digest()
        self._artifact = artifact
        self._share_compatible_operators = share_compatible_operators
        self._final_emit_only = final_emit_only
        if parent_slots is None:
            self._parent_slots: tuple[int, ...] | None = None
        else:
            normalized_parent_slots = tuple(parent_slots)
            if len(normalized_parent_slots) < 2 or any(
                not isinstance(slot, int)
                or isinstance(slot, bool)
                or not 0 <= slot < bank.executive_program_count
                for slot in normalized_parent_slots
            ):
                raise ValueError("live executive composition parent slots are invalid")
            derived = bank.composed_executive_artifact(
                normalized_parent_slots,
                share_compatible_operators=share_compatible_operators,
                final_emit_only=final_emit_only,
            )
            if derived.digest() != self.candidate_digest:
                raise ValueError(
                    "live executive composition candidate does not match its parents"
                )
            self._parent_slots = normalized_parent_slots
        self.bank = bank
        self.threshold = float(threshold)
        self.min_observations = int(min_observations)
        self.min_stable_observations = int(min_stable_observations)
        self._machine = ExternalExecutiveLiveMachine.from_artifact(
            artifact,
            decoder,
            batch_size=batch_size,
            output_key=output_key,
            device=device,
            dtype=dtype,
            sample=sample,
            model_version=model_version,
            max_instructions_per_tick=max_instructions_per_tick,
            skill_digest=self.candidate_digest,
        )
        self._episode_rewards: list[float] = []
        self._lifetime_outcomes: list[float] = []
        self._unique_verifier_bits = 0
        self._unique_logical_lifetimes = 0
        self._admission_receipt: ExternalProgramAdmissionReceipt | None = None
        self._bank_digest_before = bank.digest()
        self._bank_digest_after = self._bank_digest_before

    @classmethod
    def from_parent_slots(
        cls,
        bank: ExternalAgentBrainBank,
        parent_slots: Sequence[int],
        decoder: LiveIntentionDecoder,
        **kwargs,
    ) -> ExternalExecutiveCandidateLiveMachine:
        """Stage the deterministic child derived from existing bank slots.

        The parent slots and child digest are checked before any live input is
        consumed.  This is the bank-backed proposal path: no candidate program
        bytes or task labels are supplied by the controller.
        """

        share_compatible_operators = kwargs.get(
            "share_compatible_operators", False
        )
        final_emit_only = kwargs.get("final_emit_only", False)
        artifact = bank.composed_executive_artifact(
            parent_slots,
            share_compatible_operators=share_compatible_operators,
            final_emit_only=final_emit_only,
        )
        return cls(
            artifact,
            bank,
            decoder,
            parent_slots=parent_slots,
            **kwargs,
        )

    @property
    def executive(self) -> ExternalAmodalExecutive:
        return self._machine.executive

    @property
    def decoder(self) -> LiveIntentionDecoder:
        return self._machine.decoder

    @property
    def executive_state(self) -> TrustedExternalExecutiveState:
        return self._machine.executive_state

    @property
    def executive_ticks(self) -> int:
        return self._machine.executive_ticks

    @property
    def lifetime_outcomes(self) -> tuple[float, ...]:
        return tuple(self._lifetime_outcomes)

    @property
    def unique_verifier_bits(self) -> int:
        return self._unique_verifier_bits

    @property
    def unique_logical_lifetimes(self) -> int:
        return self._unique_logical_lifetimes

    @property
    def replayed_examples(self) -> int:
        return 0

    @property
    def admission_receipt(self) -> ExternalProgramAdmissionReceipt | None:
        return self._admission_receipt

    @property
    def admitted(self) -> bool:
        return bool(
            self._admission_receipt is not None
            and self._admission_receipt.accepted
        )

    @property
    def bank_digest_before(self) -> str:
        return self._bank_digest_before

    @property
    def bank_digest_after(self) -> str:
        return self._bank_digest_after

    @property
    def parent_slots(self) -> tuple[int, ...] | None:
        return self._parent_slots

    @property
    def share_compatible_operators(self) -> bool:
        return self._share_compatible_operators

    @property
    def final_emit_only(self) -> bool:
        return self._final_emit_only

    def _observe_outcomes(self, outcomes: Sequence[ResolvedLiveOutcome]) -> None:
        for resolved in outcomes:
            credit = resolved.proposal.credit_state
            if not isinstance(credit, ExternalExecutiveLiveCredit):
                continue
            if credit.program_digest != self.candidate_digest:
                continue
            event = resolved.event
            if event.present.shape != (self._machine.batch_size,):
                raise ValueError("live executive candidate outcome batch is invalid")
            rewards = event.reward.detach().reshape(-1)
            present = event.present.detach().reshape(-1)
            for reward in rewards[present].tolist():
                self._episode_rewards.append(float(reward))
                self._unique_verifier_bits += 1

    def finish_episode(self) -> float | None:
        """Record one lifetime and try the stable verifier admission gate."""

        if not self._episode_rewards:
            return None
        outcome = sum(self._episode_rewards) / len(self._episode_rewards)
        self._episode_rewards.clear()
        self._lifetime_outcomes.append(outcome)
        self._unique_logical_lifetimes += 1
        if not self.admitted:
            if self._parent_slots is None:
                self._admission_receipt = self.bank.admit_executive(
                    self._artifact,
                    self._lifetime_outcomes,
                    threshold=self.threshold,
                    min_observations=self.min_observations,
                    min_stable_observations=self.min_stable_observations,
                )
            else:
                self._admission_receipt = self.bank.compose_executive(
                    self._parent_slots,
                    self._lifetime_outcomes,
                    threshold=self.threshold,
                    min_observations=self.min_observations,
                    min_stable_observations=self.min_stable_observations,
                    share_compatible_operators=self._share_compatible_operators,
                    final_emit_only=self._final_emit_only,
                )
            self._bank_digest_after = self.bank.digest()
        return outcome

    def reset(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Close any active lifetime, then reset only the frozen execution state."""

        self.finish_episode()
        self._machine.reset(device=device, dtype=dtype)

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        now: float,
        elapsed: float,
    ) -> tuple[LiveActionProposal, ...]:
        self._observe_outcomes(outcomes)
        return self._machine.tick(events, (), now=now, elapsed=elapsed)
