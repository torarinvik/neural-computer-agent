"""Typed, persistent executive for external amodal programs.

The neural controller may select or construct these programs, but the
interpreter and its workspace live outside controller weights.  Programs see
only standardized event collections, typed workspace values, registered
operator handles, and the intention bus.  They never see modality formats,
device protocols, task labels, or verifier-private state.

This first ABI deliberately uses one shared instruction pointer for a batch.
Live deployment is batch one; batched diagnostic execution must take the same
branch in every row and fails closed on divergent control flow.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .interface import AmodalEventCollection, IntentEvent

EXECUTIVE_PROGRAM_SCHEMA = "neural-computer.external-executive-program.v1"
EXECUTIVE_STATE_SCHEMA = "neural-computer.external-executive-state.v1"
EXECUTIVE_OPERATOR_SCHEMA = "neural-computer.external-executive-operator.v1"
EXECUTIVE_OPERATOR_STATE_SCHEMA = (
    "neural-computer.external-executive-operator-state.v1"
)

WorkspaceKind = Literal["empty", "events", "value", "evidence", "intention"]
ExecutiveOp = Literal[
    "receive",
    "read",
    "write",
    "copy",
    "call",
    "branch",
    "wait",
    "emit",
    "halt",
]
ExecutiveStatus = Literal[
    "ready",
    "waiting",
    "emitted",
    "halted",
    "step_budget_exhausted",
]


def _clone_collection(events: AmodalEventCollection) -> AmodalEventCollection:
    return AmodalEventCollection(
        payload=events.payload.detach().clone(),
        present=events.present.detach().clone(),
        confidence=events.confidence.detach().clone(),
        source_key=(
            None if events.source_key is None else events.source_key.detach().clone()
        ),
        timestamp=(
            None if events.timestamp is None else events.timestamp.detach().clone()
        ),
        timestamp_present=(
            None
            if events.timestamp_present is None
            else events.timestamp_present.detach().clone()
        ),
        duration=(
            None if events.duration is None else events.duration.detach().clone()
        ),
        duration_present=(
            None
            if events.duration_present is None
            else events.duration_present.detach().clone()
        ),
    )


@dataclass(frozen=True)
class TypedWorkspaceValue:
    """One typed, batch-aligned external workspace value."""

    kind: WorkspaceKind
    payload: torch.Tensor | AmodalEventCollection | None
    present: torch.Tensor
    confidence: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> TypedWorkspaceValue:
        if batch_size < 1:
            raise ValueError("typed workspace needs a positive batch size")
        return cls(
            "empty",
            None,
            torch.zeros(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, dtype=dtype, device=device),
        )

    @classmethod
    def from_events(cls, events: AmodalEventCollection) -> TypedWorkspaceValue:
        events.validate()
        present = events.present.any(dim=1)
        confidence = (
            torch.zeros(
                events.payload.shape[0],
                device=events.payload.device,
                dtype=events.payload.dtype,
            )
            if events.payload.shape[1] == 0
            else torch.where(
                events.present,
                events.confidence,
                torch.zeros_like(events.confidence),
            ).amax(dim=1)
        )
        return cls(
            "events",
            _clone_collection(events),
            present.detach().clone(),
            confidence.detach().clone(),
        ).validate(batch_size=events.payload.shape[0])

    @classmethod
    def from_tensor(
        cls,
        kind: Literal["value", "evidence", "intention"],
        payload: torch.Tensor,
        *,
        present: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> TypedWorkspaceValue:
        if payload.ndim != 2:
            raise ValueError("typed tensor payload must have shape [batch, width]")
        batch = payload.shape[0]
        if present is None:
            present = torch.ones(batch, dtype=torch.bool, device=payload.device)
        if confidence is None:
            confidence = torch.ones(batch, dtype=payload.dtype, device=payload.device)
        return cls(
            kind,
            payload.detach().clone(),
            present.detach().clone(),
            confidence.detach().clone(),
        ).validate(batch_size=batch)

    def validate(self, *, batch_size: int) -> TypedWorkspaceValue:
        if self.kind not in {"empty", "events", "value", "evidence", "intention"}:
            raise ValueError("typed workspace value has an unsupported kind")
        if self.present.shape != (batch_size,) or self.present.dtype != torch.bool:
            raise ValueError("typed workspace presence must be boolean [batch]")
        if self.confidence.shape != (batch_size,):
            raise ValueError("typed workspace confidence must have shape [batch]")
        if not self.confidence.is_floating_point() or not bool(
            torch.isfinite(self.confidence).all()
        ):
            raise ValueError("typed workspace confidence must be finite floating point")
        if bool(torch.any(self.confidence < 0.0)):
            raise ValueError("typed workspace confidence cannot be negative")
        if self.kind == "empty":
            if self.payload is not None or bool(self.present.any()):
                raise ValueError(
                    "empty workspace values cannot carry payload or presence"
                )
            return self
        if self.kind == "events":
            if not isinstance(self.payload, AmodalEventCollection):
                raise TypeError("events workspace value needs an event collection")
            self.payload.validate()
            if self.payload.payload.shape[0] != batch_size:
                raise ValueError("events workspace batch does not match state")
            expected_present = self.payload.present.any(dim=1)
            if not torch.equal(self.present, expected_present):
                raise ValueError("events workspace presence does not match collection")
            return self
        if not isinstance(self.payload, torch.Tensor) or self.payload.ndim != 2:
            raise TypeError("typed workspace tensor value needs [batch, width] payload")
        if self.payload.shape[0] != batch_size:
            raise ValueError("typed workspace tensor batch does not match state")
        if not self.payload.is_floating_point() or not bool(
            torch.isfinite(self.payload).all()
        ):
            raise ValueError(
                "typed workspace tensor payload must be finite floating point"
            )
        if (
            self.payload.device != self.present.device
            or self.payload.device != self.confidence.device
        ):
            raise ValueError("typed workspace tensors must share a device")
        if self.kind == "evidence" and self.payload.shape[1] != 1:
            raise ValueError("evidence workspace payload must have width one")
        return self

    def detached_clone(self) -> TypedWorkspaceValue:
        if isinstance(self.payload, AmodalEventCollection):
            payload: torch.Tensor | AmodalEventCollection | None = _clone_collection(
                self.payload
            )
        elif isinstance(self.payload, torch.Tensor):
            payload = self.payload.detach().clone()
        else:
            payload = None
        return TypedWorkspaceValue(
            self.kind,
            payload,
            self.present.detach().clone(),
            self.confidence.detach().clone(),
        )


@dataclass(frozen=True)
class ExternalExecutiveOperatorState:
    """Version-bound tensor state owned by one external operator handle.

    Operators receive a detached clone and must return a complete replacement.
    The registry validates that replacement before the executive commits it, so
    failed calls cannot partially update the durable executive state.
    """

    interface_version: str
    tensors: tuple[tuple[str, torch.Tensor], ...] = ()
    schema: str = EXECUTIVE_OPERATOR_STATE_SCHEMA

    @classmethod
    def from_mapping(
        cls,
        interface_version: str,
        tensors: Mapping[str, torch.Tensor],
    ) -> ExternalExecutiveOperatorState:
        return cls(
            interface_version,
            tuple(
                (name, value.detach().clone())
                for name, value in sorted(tensors.items())
            ),
        ).validate()

    def validate(self) -> ExternalExecutiveOperatorState:
        if self.schema != EXECUTIVE_OPERATOR_STATE_SCHEMA:
            raise ValueError("unsupported external executive operator state schema")
        if not isinstance(self.interface_version, str) or not self.interface_version:
            raise ValueError("external operator state needs an interface version")
        names: set[str] = set()
        for name, value in self.tensors:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("external operator state tensor names must be unique")
            if not isinstance(value, torch.Tensor):
                raise TypeError("external operator state values must be tensors")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("external operator state tensors must be finite")
            names.add(name)
        return self

    def tensor(self, name: str) -> torch.Tensor:
        for candidate, value in self.tensors:
            if candidate == name:
                return value
        raise KeyError(f"external operator state has no tensor named {name!r}")

    def detached_clone(self) -> ExternalExecutiveOperatorState:
        return ExternalExecutiveOperatorState(
            self.interface_version,
            tuple((name, value.detach().clone()) for name, value in self.tensors),
            self.schema,
        ).validate()


@dataclass(frozen=True)
class ExecutiveInstruction:
    """One generic instruction with opaque slots and operator handles."""

    op: ExecutiveOp
    source: int | None = None
    destination: int | None = None
    operator_handle: int | None = None
    arguments: tuple[int, ...] = ()
    true_target: int | None = None
    false_target: int | None = None
    unknown_target: int | None = None
    next_target: int | None = None

    def validate(self, *, slot_count: int, program_length: int) -> None:
        if self.op not in {
            "receive",
            "read",
            "write",
            "copy",
            "call",
            "branch",
            "wait",
            "emit",
            "halt",
        }:
            raise ValueError("external executive instruction has an unknown operation")
        for name, slot in (
            ("source", self.source),
            ("destination", self.destination),
            *(("argument", value) for value in self.arguments),
        ):
            if slot is not None and (
                not isinstance(slot, int)
                or isinstance(slot, bool)
                or not 0 <= slot < slot_count
            ):
                raise ValueError(f"external executive {name} slot is invalid")
        if self.op == "receive" and self.destination is None:
            raise ValueError("RECEIVE needs a destination slot")
        elif self.op == "read" and self.source is None:
            raise ValueError("READ needs a source slot")
        elif self.op == "write" and self.destination is None:
            raise ValueError("WRITE needs a destination slot")
        elif self.op == "copy" and (self.source is None or self.destination is None):
            raise ValueError("COPY needs source and destination slots")
        elif self.op == "call":
            if (
                self.operator_handle is None
                or not isinstance(self.operator_handle, int)
                or isinstance(self.operator_handle, bool)
                or self.operator_handle < 0
                or self.destination is None
            ):
                raise ValueError("CALL needs an opaque handle and destination")
        elif self.op == "branch":
            if self.source is None:
                raise ValueError("BRANCH needs an evidence source")
            targets = (self.true_target, self.false_target, self.unknown_target)
            if any(
                target is None
                or not isinstance(target, int)
                or isinstance(target, bool)
                or not 0 <= target < program_length
                for target in targets
            ):
                raise ValueError("BRANCH needs valid true, false, and unknown targets")
        elif self.op == "emit" and self.source is None:
            raise ValueError("EMIT needs an intention source")
        if self.next_target is not None:
            if self.op not in {"wait", "emit"}:
                raise ValueError("only WAIT and EMIT may set a next target")
            if (
                not isinstance(self.next_target, int)
                or isinstance(self.next_target, bool)
                or not 0 <= self.next_target < program_length
            ):
                raise ValueError("external executive next target is invalid")


@dataclass(frozen=True)
class ExternalExecutiveProgram:
    slot_count: int
    instructions: tuple[ExecutiveInstruction, ...]
    schema: str = EXECUTIVE_PROGRAM_SCHEMA

    def validate(self) -> ExternalExecutiveProgram:
        if self.schema != EXECUTIVE_PROGRAM_SCHEMA:
            raise ValueError("unsupported external executive program schema")
        if self.slot_count < 1 or not self.instructions:
            raise ValueError("external executive program dimensions must be positive")
        for instruction in self.instructions:
            instruction.validate(
                slot_count=self.slot_count,
                program_length=len(self.instructions),
            )
        if self.instructions[-1].op != "halt":
            raise ValueError("external executive programs must end with HALT")
        return self

    def digest(self) -> str:
        payload = {
            "schema": self.schema,
            "slot_count": self.slot_count,
            "instructions": [instruction.__dict__ for instruction in self.instructions],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class ExternalExecutiveOperator(ABC):
    """Independently versioned implementation behind one opaque CALL handle."""

    schema = EXECUTIVE_OPERATOR_SCHEMA

    def __init__(
        self,
        handle: int,
        input_kinds: Sequence[WorkspaceKind],
        output_kind: WorkspaceKind,
        *,
        interface_version: str,
    ) -> None:
        if not isinstance(handle, int) or isinstance(handle, bool) or handle < 0:
            raise ValueError("external executive operator handle must be non-negative")
        if not input_kinds or any(kind == "empty" for kind in input_kinds):
            raise ValueError("external executive operator inputs must be typed")
        if output_kind == "empty":
            raise ValueError("external executive operator output must be typed")
        if not isinstance(interface_version, str) or not interface_version:
            raise ValueError(
                "external executive operator interface version is required"
            )
        self.handle = handle
        self.input_kinds = tuple(input_kinds)
        self.output_kind = output_kind
        self.interface_version = interface_version

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "handle": self.handle,
            "input_kinds": self.input_kinds,
            "output_kind": self.output_kind,
            "interface_version": self.interface_version,
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ExternalExecutiveOperatorState:
        """Create explicit per-executive state; pure operators use an empty bundle."""
        del batch_size, device, dtype
        return ExternalExecutiveOperatorState(self.interface_version).validate()

    def validate_state(
        self,
        state: ExternalExecutiveOperatorState,
        *,
        batch_size: int,
    ) -> ExternalExecutiveOperatorState:
        del batch_size
        state.validate()
        if state.interface_version != self.interface_version:
            raise ValueError("external operator state interface version is incompatible")
        if state.tensors:
            raise ValueError("pure external operator state must be empty")
        return state

    @abstractmethod
    def execute(
        self,
        arguments: tuple[TypedWorkspaceValue, ...],
    ) -> TypedWorkspaceValue:
        """Return one detached typed result without mutating arguments."""

    def execute_with_state(
        self,
        arguments: tuple[TypedWorkspaceValue, ...],
        state: ExternalExecutiveOperatorState,
    ) -> tuple[TypedWorkspaceValue, ExternalExecutiveOperatorState]:
        """Execute transactionally; stateful operators override this method."""
        return self.execute(arguments), state.detached_clone()


class ExternalExecutiveOperatorRegistry:
    def __init__(self, operators: Sequence[ExternalExecutiveOperator]) -> None:
        self._operators: dict[int, ExternalExecutiveOperator] = {}
        for operator in operators:
            if not isinstance(operator, ExternalExecutiveOperator):
                raise TypeError("external executive registry received a non-operator")
            if operator.handle in self._operators:
                raise ValueError("external executive operator handles must be unique")
            self._operators[operator.handle] = operator

    def operator(self, handle: int) -> ExternalExecutiveOperator:
        try:
            return self._operators[handle]
        except KeyError as error:
            raise LookupError("unknown external executive operator handle") from error

    def initial_states(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[tuple[int, ExternalExecutiveOperatorState], ...]:
        states = []
        for handle, operator in sorted(self._operators.items()):
            state = operator.initial_state(
                batch_size, device=device, dtype=dtype
            ).detached_clone()
            operator.validate_state(state, batch_size=batch_size)
            states.append((handle, state))
        return tuple(states)

    def validate_states(
        self,
        states: tuple[tuple[int, ExternalExecutiveOperatorState], ...],
        *,
        batch_size: int,
    ) -> None:
        if tuple(handle for handle, _ in states) != tuple(sorted(self._operators)):
            raise ValueError("external executive operator state handles are incompatible")
        for handle, state in states:
            self.operator(handle).validate_state(state, batch_size=batch_size)

    def call(
        self,
        handle: int,
        arguments: tuple[TypedWorkspaceValue, ...],
        state: ExternalExecutiveOperatorState,
        *,
        batch_size: int,
    ) -> tuple[TypedWorkspaceValue, ExternalExecutiveOperatorState]:
        operator = self.operator(handle)
        if len(arguments) != len(operator.input_kinds):
            raise ValueError("external executive CALL argument count is incompatible")
        for argument, expected_kind in zip(
            arguments, operator.input_kinds, strict=True
        ):
            argument.validate(batch_size=batch_size)
            if argument.kind != expected_kind:
                raise TypeError("external executive CALL argument kind is incompatible")
        operator.validate_state(state, batch_size=batch_size)
        result, next_state = operator.execute_with_state(
            tuple(value.detached_clone() for value in arguments),
            state.detached_clone(),
        )
        if not isinstance(result, TypedWorkspaceValue):
            raise TypeError("external executive operator returned an untyped value")
        result.validate(batch_size=batch_size)
        if result.kind != operator.output_kind:
            raise TypeError("external executive operator returned an incompatible kind")
        if not isinstance(next_state, ExternalExecutiveOperatorState):
            raise TypeError("external executive operator returned untyped state")
        operator.validate_state(next_state, batch_size=batch_size)
        return result.detached_clone(), next_state.detached_clone()

    def digest(self) -> str:
        payload = [
            self._operators[handle].configuration()
            for handle in sorted(self._operators)
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ExternalExecutiveState:
    workspace: tuple[TypedWorkspaceValue, ...]
    accumulator: TypedWorkspaceValue
    instruction_pointer: int
    status: ExecutiveStatus
    batch_size: int
    operator_states: tuple[tuple[int, ExternalExecutiveOperatorState], ...] = ()
    operator_registry_digest: str = ""
    ticks: int = 0
    executed_instructions: int = 0
    schema: str = EXECUTIVE_STATE_SCHEMA

    def validate(self, *, slot_count: int) -> ExternalExecutiveState:
        if self.schema != EXECUTIVE_STATE_SCHEMA:
            raise ValueError("unsupported external executive state schema")
        if self.batch_size < 1 or len(self.workspace) != slot_count:
            raise ValueError("external executive state dimensions are invalid")
        if (
            self.instruction_pointer < 0
            or self.ticks < 0
            or self.executed_instructions < 0
        ):
            raise ValueError("external executive state counters cannot be negative")
        if self.status not in {
            "ready",
            "waiting",
            "emitted",
            "halted",
            "step_budget_exhausted",
        }:
            raise ValueError("external executive state status is invalid")
        self.accumulator.validate(batch_size=self.batch_size)
        for value in self.workspace:
            value.validate(batch_size=self.batch_size)
        handles = tuple(handle for handle, _ in self.operator_states)
        if handles != tuple(sorted(handles)) or len(handles) != len(set(handles)):
            raise ValueError("external executive operator states must have unique handles")
        for handle, operator_state in self.operator_states:
            if not isinstance(handle, int) or isinstance(handle, bool) or handle < 0:
                raise ValueError("external executive operator state handle is invalid")
            operator_state.validate()
        if not isinstance(self.operator_registry_digest, str):
            raise TypeError("external executive operator registry digest must be text")
        return self


@dataclass(frozen=True)
class ExternalExecutiveTick:
    intention: IntentEvent | None
    status: ExecutiveStatus
    executed_instructions: int
    program_digest: str
    operator_registry_digest: str


class ExternalAmodalExecutive:
    """Execute a persistent typed program until it yields or halts."""

    def __init__(
        self,
        program: ExternalExecutiveProgram,
        operators: ExternalExecutiveOperatorRegistry,
        *,
        intention_width: int,
        max_instructions_per_tick: int = 128,
    ) -> None:
        program.validate()
        if not isinstance(operators, ExternalExecutiveOperatorRegistry):
            raise TypeError("external executive needs an operator registry")
        if intention_width < 1 or max_instructions_per_tick < 1:
            raise ValueError("external executive bounds must be positive")
        self.program = program
        self.operators = operators
        self.intention_width = int(intention_width)
        self.max_instructions_per_tick = int(max_instructions_per_tick)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalExecutiveState:
        empty = TypedWorkspaceValue.empty(batch_size, device=device, dtype=dtype)
        return ExternalExecutiveState(
            workspace=tuple(
                TypedWorkspaceValue.empty(batch_size, device=device, dtype=dtype)
                for _ in range(self.program.slot_count)
            ),
            accumulator=empty,
            instruction_pointer=0,
            status="ready",
            batch_size=batch_size,
            operator_states=self.operators.initial_states(
                batch_size, device=device, dtype=dtype
            ),
            operator_registry_digest=self.operators.digest(),
        ).validate(slot_count=self.program.slot_count)

    @staticmethod
    def _write_slot(
        workspace: tuple[TypedWorkspaceValue, ...],
        destination: int,
        value: TypedWorkspaceValue,
    ) -> tuple[TypedWorkspaceValue, ...]:
        mutable = list(workspace)
        mutable[destination] = value.detached_clone()
        return tuple(mutable)

    def tick(
        self,
        events: AmodalEventCollection,
        state: ExternalExecutiveState,
    ) -> tuple[ExternalExecutiveTick, ExternalExecutiveState]:
        events.validate()
        state.validate(slot_count=self.program.slot_count)
        if state.operator_registry_digest != self.operators.digest():
            raise ValueError("external executive state operator registry is incompatible")
        self.operators.validate_states(
            state.operator_states, batch_size=state.batch_size
        )
        if events.payload.shape[0] != state.batch_size:
            raise ValueError("external executive input batch does not match state")
        if state.status == "halted":
            output = ExternalExecutiveTick(
                None, "halted", 0, self.program.digest(), self.operators.digest()
            )
            return output, state
        if state.status == "step_budget_exhausted":
            raise RuntimeError("external executive cannot resume an exhausted program")

        workspace = state.workspace
        accumulator = state.accumulator
        operator_states = dict(state.operator_states)
        pointer = state.instruction_pointer
        executed = 0
        status: ExecutiveStatus = "ready"
        intention: IntentEvent | None = None

        while executed < self.max_instructions_per_tick:
            if not 0 <= pointer < len(self.program.instructions):
                raise RuntimeError(
                    "external executive instruction pointer escaped program"
                )
            instruction = self.program.instructions[pointer]
            executed += 1

            if instruction.op == "receive":
                assert instruction.destination is not None
                workspace = self._write_slot(
                    workspace,
                    instruction.destination,
                    TypedWorkspaceValue.from_events(events),
                )
                pointer += 1
            elif instruction.op == "read":
                assert instruction.source is not None
                value = workspace[instruction.source]
                if value.kind == "empty":
                    raise RuntimeError(
                        "external executive READ encountered an empty slot"
                    )
                accumulator = value.detached_clone()
                pointer += 1
            elif instruction.op == "write":
                assert instruction.destination is not None
                if accumulator.kind == "empty":
                    raise RuntimeError(
                        "external executive WRITE has an empty accumulator"
                    )
                workspace = self._write_slot(
                    workspace, instruction.destination, accumulator
                )
                pointer += 1
            elif instruction.op == "copy":
                assert (
                    instruction.source is not None
                    and instruction.destination is not None
                )
                value = workspace[instruction.source]
                if value.kind == "empty":
                    raise RuntimeError(
                        "external executive COPY encountered an empty slot"
                    )
                workspace = self._write_slot(workspace, instruction.destination, value)
                pointer += 1
            elif instruction.op == "call":
                assert instruction.operator_handle is not None
                assert instruction.destination is not None
                # Resolve first so an unknown program handle fails with the
                # stable registry error rather than leaking an internal KeyError.
                self.operators.operator(instruction.operator_handle)
                result, next_operator_state = self.operators.call(
                    instruction.operator_handle,
                    tuple(workspace[index] for index in instruction.arguments),
                    operator_states[instruction.operator_handle],
                    batch_size=state.batch_size,
                )
                operator_states[instruction.operator_handle] = next_operator_state
                workspace = self._write_slot(workspace, instruction.destination, result)
                accumulator = result.detached_clone()
                pointer += 1
            elif instruction.op == "branch":
                assert instruction.source is not None
                evidence = workspace[instruction.source]
                if evidence.kind != "evidence" or not isinstance(
                    evidence.payload, torch.Tensor
                ):
                    raise TypeError("external executive BRANCH needs evidence")
                targets = torch.full(
                    (state.batch_size,),
                    int(instruction.unknown_target),
                    dtype=torch.long,
                    device=evidence.present.device,
                )
                targets = torch.where(
                    evidence.present & (evidence.payload[:, 0] > 0.0),
                    torch.full_like(targets, int(instruction.true_target)),
                    targets,
                )
                targets = torch.where(
                    evidence.present & (evidence.payload[:, 0] <= 0.0),
                    torch.full_like(targets, int(instruction.false_target)),
                    targets,
                )
                if not bool(torch.all(targets == targets[0])):
                    raise RuntimeError(
                        "external executive batch control flow diverged; use batch one"
                    )
                pointer = int(targets[0])
            elif instruction.op == "wait":
                pointer = (
                    pointer + 1
                    if instruction.next_target is None
                    else instruction.next_target
                )
                status = "waiting"
                break
            elif instruction.op == "emit":
                assert instruction.source is not None
                value = workspace[instruction.source]
                if value.kind != "intention" or not isinstance(
                    value.payload, torch.Tensor
                ):
                    raise TypeError("external executive EMIT needs an intention value")
                if value.payload.shape[1] != self.intention_width:
                    raise ValueError(
                        "external executive intention width is incompatible"
                    )
                if not bool(value.present.all()):
                    raise RuntimeError(
                        "external executive cannot EMIT absent intention rows"
                    )
                intention = IntentEvent(
                    payload=value.payload.detach().clone(),
                    confidence=value.confidence.detach().clone(),
                ).validate(width=self.intention_width)
                pointer = (
                    pointer + 1
                    if instruction.next_target is None
                    else instruction.next_target
                )
                status = "emitted"
                break
            else:
                status = "halted"
                break

        if executed >= self.max_instructions_per_tick and status == "ready":
            status = "step_budget_exhausted"
        next_state = ExternalExecutiveState(
            workspace=workspace,
            accumulator=accumulator,
            instruction_pointer=pointer,
            status=status,
            batch_size=state.batch_size,
            operator_states=tuple(sorted(operator_states.items())),
            operator_registry_digest=state.operator_registry_digest,
            ticks=state.ticks + 1,
            executed_instructions=state.executed_instructions + executed,
        ).validate(slot_count=self.program.slot_count)
        output = ExternalExecutiveTick(
            intention,
            status,
            executed,
            self.program.digest(),
            self.operators.digest(),
        )
        return output, next_state


__all__ = [
    "EXECUTIVE_OPERATOR_SCHEMA",
    "EXECUTIVE_OPERATOR_STATE_SCHEMA",
    "EXECUTIVE_PROGRAM_SCHEMA",
    "EXECUTIVE_STATE_SCHEMA",
    "ExecutiveInstruction",
    "ExternalAmodalExecutive",
    "ExternalExecutiveOperator",
    "ExternalExecutiveOperatorRegistry",
    "ExternalExecutiveOperatorState",
    "ExternalExecutiveProgram",
    "ExternalExecutiveState",
    "ExternalExecutiveTick",
    "TypedWorkspaceValue",
]
