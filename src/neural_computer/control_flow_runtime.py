"""Canonical amodal bridge for typed external control-flow programs.

The controller emits an opaque learned intention.  This module keeps the
typed counter-machine ABI outside that controller: a replaceable adapter
encodes the intention into bounded integer counters, an external program
executes with explicit resource limits, and another adapter returns an opaque
intention to the canonical output bus.

No counter position is assigned a semantic meaning here.  The adapter is an
independently versioned boundary component and may be replaced or trained
without changing controller parameters.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .control_flow import (
    ControlFlowExecution,
    ControlFlowProgram,
    ControlFlowProgramMemory,
)
from .controller import ControllerOutput, ControllerState
from .interface import (
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    IntentEvent,
)
from .runtime import AmodalControllerRuntime

CONTROL_FLOW_INTENTION_ADAPTER_SCHEMA = (
    "neural-computer.control-flow-intention-adapter.v1"
)
CONTROL_FLOW_RUNTIME_SCHEMA = "neural-computer.control-flow-runtime.v1"
CONTROL_FLOW_RUNTIME_STATE_SCHEMA = "neural-computer.control-flow-runtime-state.v1"


def _digest_payload(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


class ControlFlowIntentionAdapter(nn.Module, ABC):
    """Replaceable opaque-intention/counter codec.

    ``encode`` receives the controller's opaque intention and the previous
    external counter state.  It may reset or continue that state, but it must
    return only non-negative integer counters.  ``decode`` receives the
    bounded execution result and the controller intention as a metadata
    template, and must return another opaque ``IntentEvent`` of the same
    runtime width.
    """

    schema = CONTROL_FLOW_INTENTION_ADAPTER_SCHEMA

    def __init__(self, intention_width: int, counter_count: int) -> None:
        super().__init__()
        if intention_width < 1 or counter_count < 2:
            raise ValueError("control-flow adapter dimensions are invalid")
        self.intention_width = int(intention_width)
        self.counter_count = int(counter_count)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "intention_width": self.intention_width,
            "counter_count": self.counter_count,
            "input": "opaque_intent_event_v2",
            "output": "opaque_intent_event_v2",
            "state": "external_nonnegative_integer_counters_v1",
        }

    @abstractmethod
    def encode(
        self,
        intention: IntentEvent,
        previous_counters: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one batch of opaque intentions into integer counters."""

    @abstractmethod
    def decode(
        self,
        counters: torch.Tensor,
        template: IntentEvent,
    ) -> IntentEvent:
        """Decode counters into an opaque intention for the output bus."""

    def validate_counters(
        self,
        counters: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if counters.ndim != 2 or counters.shape != (batch, self.counter_count):
            raise ValueError(
                f"{name} must have shape [{batch}, {self.counter_count}]"
            )
        if counters.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use an integer dtype")
        if counters.device != device:
            raise ValueError(f"{name} must be on the intention device")
        if bool(torch.any(counters < 0)):
            raise ValueError(f"{name} cannot contain negative counters")
        return counters


@dataclass(frozen=True)
class ControlFlowRuntimeState:
    """Pause/resume state for one controller and one external program file."""

    controller: ControllerState
    counters: torch.Tensor
    counter_count: int
    schema: str = CONTROL_FLOW_RUNTIME_STATE_SCHEMA

    def validate(self) -> ControlFlowRuntimeState:
        if self.schema != CONTROL_FLOW_RUNTIME_STATE_SCHEMA:
            raise ValueError("unsupported control-flow runtime-state schema")
        if self.counter_count < 2:
            raise ValueError("control-flow runtime counter width is invalid")
        if self.counters.ndim != 2 or self.counters.shape[1] != self.counter_count:
            raise ValueError("control-flow runtime counters have the wrong shape")
        if self.counters.dtype not in (torch.int32, torch.int64):
            raise TypeError("control-flow runtime counters must be integer")
        if bool(torch.any(self.counters < 0)):
            raise ValueError("control-flow runtime counters cannot be negative")
        if self.controller.hidden.ndim != 2:
            raise ValueError("control-flow runtime controller state is malformed")
        if self.controller.hidden.shape[0] != self.counters.shape[0]:
            raise ValueError("control-flow runtime state batch sizes differ")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        body: dict[str, object] = {
            "schema": self.schema,
            "controller": self.controller.payload(),
            "counters": self.counters.detach().cpu().clone(),
            "counter_count": self.counter_count,
        }
        return {**body, "sha256": _digest_payload(body)}

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowRuntimeState:
        if not isinstance(payload, dict):
            raise TypeError("control-flow runtime-state payload must be a mapping")
        expected = payload.get("sha256")
        unsigned = {key: value for key, value in payload.items() if key != "sha256"}
        if payload.get("schema") != CONTROL_FLOW_RUNTIME_STATE_SCHEMA:
            raise ValueError("unsupported control-flow runtime-state schema")
        if not isinstance(expected, str) or expected != _digest_payload(unsigned):
            raise ValueError("control-flow runtime-state checksum mismatch")
        controller = unsigned.get("controller")
        counters = unsigned.get("counters")
        if not isinstance(controller, dict) or not isinstance(counters, torch.Tensor):
            raise TypeError("control-flow runtime-state payload is incomplete")
        return cls(
            controller=ControllerState.from_payload(controller),
            counters=counters,
            counter_count=int(unsigned.get("counter_count", -1)),
            schema=unsigned.get("schema"),
        ).validate()

    def digest(self) -> str:
        return str(self.payload()["sha256"])


@dataclass(frozen=True)
class ControlFlowRuntimeOutput:
    """One canonical INPUT -> PROCESS -> external computation -> OUTPUT cycle."""

    controller: ControllerOutput
    executions: tuple[ControlFlowExecution, ...]
    intention: IntentEvent
    decoded: dict[str, torch.Tensor]
    program_digest: str
    program_slot: int | None
    schema: str = CONTROL_FLOW_RUNTIME_SCHEMA


class ControlFlowProgramAmodalRuntime(nn.Module):
    """Execute a typed external program behind the canonical intention bus.

    The program is a file-like external capability.  The controller does not
    receive counters, instruction pointers, program IDs, or protocol fields;
    it only emits and receives opaque learned tensors through the adapter.
    A memory-backed runtime reads the selected file on every step, so external
    file replacement/growth does not resize or mutate the controller.
    """

    schema = CONTROL_FLOW_RUNTIME_SCHEMA

    def __init__(
        self,
        runtime: AmodalControllerRuntime,
        adapter: ControlFlowIntentionAdapter,
        *,
        program: ControlFlowProgram | None = None,
        program_memory: ControlFlowProgramMemory | None = None,
        program_slot: int = 0,
        max_steps: int = 128,
        max_counter: int = 1_000_000,
        trace_limit: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(runtime, AmodalControllerRuntime):
            raise TypeError("control-flow runtime requires an amodal runtime")
        if not isinstance(adapter, ControlFlowIntentionAdapter):
            raise TypeError("control-flow runtime requires an intention adapter")
        if (program is None) == (program_memory is None):
            raise ValueError("control-flow runtime requires one program source")
        if not isinstance(program_slot, int) or isinstance(program_slot, bool):
            raise TypeError("control-flow runtime program slot must be an integer")
        if max_steps < 1 or max_counter < 1 or trace_limit < 0:
            raise ValueError("control-flow runtime bounds are invalid")
        if adapter.intention_width != runtime.intention_width:
            raise ValueError("control-flow adapter width does not match runtime")
        if program is not None:
            program.validate()
            if program.counter_count != adapter.counter_count:
                raise ValueError("control-flow program width does not match adapter")
        else:
            if not isinstance(program_memory, ControlFlowProgramMemory):
                raise TypeError("control-flow program memory has the wrong type")
            if program_memory.file_count < 1:
                raise ValueError("control-flow program memory cannot be empty")
            if not 0 <= program_slot < program_memory.file_count:
                raise ValueError("control-flow program slot is out of range")
            if program_memory.counter_count != adapter.counter_count:
                raise ValueError("control-flow memory width does not match adapter")
        self.runtime = runtime
        self.adapter = adapter
        self.program = program
        self.program_memory = program_memory
        self.program_slot = None if program is not None else int(program_slot)
        self.max_steps = int(max_steps)
        self.max_counter = int(max_counter)
        self.trace_limit = int(trace_limit)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "boundary": "opaque_intention_to_external_control_flow_to_intention_bus_v1",
            "runtime": self.runtime.configuration(),
            "adapter": self.adapter.configuration(),
            "program_source": (
                "portable_control_flow_program_v1"
                if self.program is not None
                else "checksummed_control_flow_program_memory_v1"
            ),
            "program": None if self.program is None else self.program.payload(),
            "program_memory": (
                None
                if self.program_memory is None
                else self.program_memory.payload()
            ),
            "program_slot": self.program_slot,
            "max_steps": self.max_steps,
            "max_counter": self.max_counter,
            "trace_limit": self.trace_limit,
            "controller_input": "opaque_event_collection_v2",
            "controller_output": "opaque_intention_event_v2",
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControlFlowRuntimeState:
        if batch_size < 1:
            raise ValueError("control-flow runtime batch size must be positive")
        return ControlFlowRuntimeState(
            controller=self.runtime.controller.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            ),
            counters=torch.zeros(
                batch_size,
                self.adapter.counter_count,
                device=device,
                dtype=torch.int64,
            ),
            counter_count=self.adapter.counter_count,
        ).validate()

    def state_from_payload(self, payload: object) -> ControlFlowRuntimeState:
        state = ControlFlowRuntimeState.from_payload(payload)
        if state.counter_count != self.adapter.counter_count:
            raise ValueError("restored control-flow state width does not match adapter")
        return state

    def _active_program(self) -> tuple[ControlFlowProgram, int | None]:
        if self.program is not None:
            program = self.program
            slot = None
        else:
            if self.program_memory is None or self.program_slot is None:
                raise RuntimeError("control-flow program memory is not configured")
            program = self.program_memory.program(self.program_slot)
            slot = self.program_slot
        return program.validate(), slot

    def step_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ControlFlowRuntimeState,
        feedback: ControllerFeedback,
        *,
        persistent_events: AmodalEventCollection | Sequence[AmodalEvent] | None = None,
        elapsed: torch.Tensor | float = 1.0,
        disable_workspace: bool = False,
        memory_scope: torch.Tensor | None = None,
        sample_memory_writes: bool = False,
        memory_write_override: torch.Tensor | None = None,
        memory_write_uniform: torch.Tensor | None = None,
        memory_write_gradient: bool = True,
    ) -> tuple[ControlFlowRuntimeOutput, ControlFlowRuntimeState]:
        if not isinstance(state, ControlFlowRuntimeState):
            raise TypeError("control-flow runtime state has the wrong type")
        state.validate()
        if state.counter_count != self.adapter.counter_count:
            raise ValueError("control-flow runtime state width does not match adapter")
        collection = self.runtime.input_bus(events)
        controller_output, next_controller = self.runtime.controller.step(
            collection,
            state.controller,
            feedback,
            self.runtime.memory,
            persistent_events=persistent_events,
            elapsed=elapsed,
            disable_workspace=disable_workspace,
            memory_scope=memory_scope,
            sample_memory_writes=sample_memory_writes,
            memory_write_override=memory_write_override,
            memory_write_uniform=memory_write_uniform,
            memory_write_gradient=memory_write_gradient,
        )
        batch = controller_output.intention.payload.shape[0]
        device = controller_output.intention.payload.device
        if state.counters.shape[0] != batch or state.counters.device != device:
            raise ValueError("control-flow state does not match controller output")
        encoded = self.adapter.encode(controller_output.intention, state.counters)
        encoded = self.adapter.validate_counters(
            encoded,
            batch=batch,
            device=device,
            name="encoded control-flow counters",
        )
        program, program_slot = self._active_program()
        executions = tuple(
            program.execute(
                tuple(int(value) for value in row),
                max_steps=self.max_steps,
                max_counter=self.max_counter,
                trace_limit=self.trace_limit,
            )
            for row in encoded.detach().cpu().tolist()
        )
        result_counters = torch.tensor(
            [execution.counters for execution in executions],
            device=device,
            dtype=torch.int64,
        )
        result_counters = self.adapter.validate_counters(
            result_counters,
            batch=batch,
            device=device,
            name="executed control-flow counters",
        )
        intention = self.adapter.decode(result_counters, controller_output.intention)
        if not isinstance(intention, IntentEvent):
            raise TypeError("control-flow adapter decode must return an IntentEvent")
        intention.validate(width=self.runtime.intention_width)
        if intention.payload.device != device:
            raise ValueError("decoded intention must remain on the controller device")
        next_state = ControlFlowRuntimeState(
            controller=next_controller,
            counters=result_counters.detach(),
            counter_count=self.adapter.counter_count,
        ).validate()
        output = ControlFlowRuntimeOutput(
            controller=controller_output,
            executions=executions,
            intention=intention,
            decoded=self.runtime.output_bus(intention),
            program_digest=program.digest(),
            program_slot=program_slot,
        )
        return output, next_state


__all__ = [
    "CONTROL_FLOW_INTENTION_ADAPTER_SCHEMA",
    "CONTROL_FLOW_RUNTIME_SCHEMA",
    "CONTROL_FLOW_RUNTIME_STATE_SCHEMA",
    "ControlFlowIntentionAdapter",
    "ControlFlowProgramAmodalRuntime",
    "ControlFlowRuntimeOutput",
    "ControlFlowRuntimeState",
]
