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
from dataclasses import dataclass, field

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
from .plasticity import (
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
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
    """Pause/resume state for one controller and external program files."""

    controller: ControllerState
    counters: torch.Tensor
    counter_count: int
    program_counters: Mapping[int, torch.Tensor] = field(default_factory=dict)
    program_router: ExternalOutcomeProgramRouterState | None = None
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
        if not isinstance(self.program_counters, Mapping):
            raise TypeError("control-flow per-program counters must be a mapping")
        for slot, counters in self.program_counters.items():
            if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
                raise ValueError("control-flow program counter slot is invalid")
            if not isinstance(counters, torch.Tensor):
                raise TypeError("control-flow program counters must be tensors")
            if counters.shape != self.counters.shape:
                raise ValueError("control-flow program counters have the wrong shape")
            if counters.dtype not in (torch.int32, torch.int64):
                raise TypeError("control-flow program counters must be integer")
            if counters.device != self.counters.device:
                raise ValueError("control-flow program counters use the wrong device")
            if bool(torch.any(counters < 0)):
                raise ValueError("control-flow program counters cannot be negative")
        if self.program_router is not None:
            if not isinstance(self.program_router, ExternalOutcomeProgramRouterState):
                raise TypeError("control-flow program router state has the wrong type")
            self.program_router.validate(
                feature_width=self.program_router.credit.policy.shape[1],
                program_capacity=self.program_router.credit.policy.shape[2],
            )
        return self

    def payload(
        self,
        *,
        program_router: ExternalOutcomeProgramRouter | None = None,
    ) -> dict[str, object]:
        self.validate()
        if self.program_router is not None and program_router is None:
            raise ValueError(
                "serializing routed control-flow state requires its router ABI"
            )
        body: dict[str, object] = {
            "schema": self.schema,
            "controller": self.controller.payload(),
            "counters": self.counters.detach().cpu().clone(),
            "counter_count": self.counter_count,
            "program_counters": tuple(
                {
                    "slot": int(slot),
                    "counters": counters.detach().cpu().clone(),
                }
                for slot, counters in sorted(self.program_counters.items())
            ),
            "program_router": (
                None
                if self.program_router is None
                else program_router.state_payload(self.program_router)
            ),
        }
        return {**body, "sha256": _digest_payload(body)}

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        program_router: ExternalOutcomeProgramRouter | None = None,
    ) -> ControlFlowRuntimeState:
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
        raw_program_counters = unsigned.get("program_counters", ())
        if not isinstance(raw_program_counters, (tuple, list)):
            raise TypeError("control-flow program counters payload must be a sequence")
        program_counters: dict[int, torch.Tensor] = {}
        for record in raw_program_counters:
            if not isinstance(record, Mapping):
                raise TypeError("control-flow program counter record is malformed")
            slot = record.get("slot")
            value = record.get("counters")
            if (
                not isinstance(slot, int)
                or isinstance(slot, bool)
                or slot < 0
                or not isinstance(value, torch.Tensor)
            ):
                raise TypeError("control-flow program counter record is invalid")
            if slot in program_counters:
                raise ValueError("control-flow program counter slots must be unique")
            program_counters[slot] = value
        raw_router = unsigned.get("program_router")
        if raw_router is None:
            restored_router = None
        elif program_router is None:
            raise ValueError(
                "restoring routed control-flow state requires its router ABI"
            )
        elif not isinstance(raw_router, Mapping):
            raise TypeError("control-flow program router payload is malformed")
        else:
            restored_router = program_router.state_from_payload(raw_router)
        return cls(
            controller=ControllerState.from_payload(controller),
            counters=counters,
            counter_count=int(unsigned.get("counter_count", -1)),
            program_counters=program_counters,
            program_router=restored_router,
            schema=unsigned.get("schema"),
        ).validate()

    def digest(
        self,
        *,
        program_router: ExternalOutcomeProgramRouter | None = None,
    ) -> str:
        return str(self.payload(program_router=program_router)["sha256"])


@dataclass(frozen=True)
class ControlFlowRuntimeOutput:
    """One canonical INPUT -> PROCESS -> external computation -> OUTPUT cycle."""

    controller: ControllerOutput
    executions: tuple[ControlFlowExecution, ...]
    intention: IntentEvent
    decoded: dict[str, torch.Tensor]
    program_digest: str | None
    program_slot: int | None
    selected_program_slots: torch.Tensor | None = None
    program_digests: tuple[str, ...] = ()
    program_route_query: torch.Tensor | None = None
    program_route_probabilities: torch.Tensor | None = None
    program_route_propensities: torch.Tensor | None = None
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
        program_router: ExternalOutcomeProgramRouter | None = None,
        program_route_query_adapter: nn.Module | None = None,
        program_route_exploration: float = 0.0,
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
        if not 0.0 <= float(program_route_exploration) <= 1.0:
            raise ValueError("control-flow program route exploration is invalid")
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
        if program_router is not None:
            if program_memory is None or program is not None:
                raise ValueError("control-flow program routing requires program memory")
            if program_router.initial_programs != program_memory.file_count:
                raise ValueError(
                    "control-flow program router and memory file counts differ"
                )
        if program_route_query_adapter is not None:
            if program_router is None:
                raise ValueError(
                    "a route query adapter requires a program router"
                )
            if not isinstance(program_route_query_adapter, nn.Module):
                raise TypeError("control-flow route query adapter must be a module")
            query_width = getattr(program_route_query_adapter, "query_width", None)
            if query_width is not None and query_width != program_router.feature_width:
                raise ValueError(
                    "control-flow route query adapter width does not match router"
                )
        elif program_router is not None and program_router.feature_width != runtime.intention_width:
            raise ValueError(
                "control-flow program router must consume opaque intentions"
            )
        self.runtime = runtime
        self.adapter = adapter
        self.program = program
        self.program_memory = program_memory
        self.program_slot = None if program is not None else int(program_slot)
        self.program_router = program_router
        self.program_route_query_adapter = program_route_query_adapter
        self.program_route_exploration = float(program_route_exploration)
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
            "program_router": (
                None
                if self.program_router is None
                else self.program_router.configuration()
            ),
            "route_feedback": "optional_external_router_only_v1",
            "program_route_query_adapter": (
                None
                if self.program_route_query_adapter is None
                else self.program_route_query_adapter.configuration()
            ),
            "program_route_exploration": self.program_route_exploration,
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
        counters = torch.zeros(
            batch_size,
            self.adapter.counter_count,
            device=device,
            dtype=torch.int64,
        )
        program_counters = (
            {
                slot: counters.clone()
                for slot in range(self.program_memory.file_count)
            }
            if self.program_router is not None and self.program_memory is not None
            else {}
        )
        router_state = (
            None
            if self.program_router is None
            else self.program_router.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            )
        )
        return ControlFlowRuntimeState(
            controller=self.runtime.controller.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            ),
            counters=counters,
            counter_count=self.adapter.counter_count,
            program_counters=program_counters,
            program_router=router_state,
        ).validate()

    def state_from_payload(self, payload: object) -> ControlFlowRuntimeState:
        state = ControlFlowRuntimeState.from_payload(
            payload,
            program_router=self.program_router,
        )
        if state.counter_count != self.adapter.counter_count:
            raise ValueError("restored control-flow state width does not match adapter")
        if (state.program_router is None) != (self.program_router is None):
            raise ValueError("restored control-flow routing state does not match runtime")
        return state

    def _program_for_slot(self, slot: int | None) -> ControlFlowProgram:
        if self.program is not None:
            if slot is not None:
                raise RuntimeError("standalone control-flow program has no slot")
            return self.program.validate()
        if self.program_memory is None or slot is None:
            raise RuntimeError("control-flow program memory is not configured")
        return self.program_memory.program(slot).validate()

    def _select_programs(
        self,
        controller_output: ControllerOutput,
        controller_state: ControllerState,
        router_state: ExternalOutcomeProgramRouterState | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        ExternalOutcomeProgramRouterState | None,
    ]:
        intention = controller_output.intention
        batch = intention.payload.shape[0]
        device = intention.payload.device
        if self.program is not None:
            return (
                torch.full((batch,), -1, dtype=torch.long, device=device),
                None,
                None,
                None,
                router_state,
            )
        if self.program_memory is None or self.program_slot is None:
            raise RuntimeError("control-flow program memory is not configured")
        if self.program_router is None:
            return (
                torch.full(
                    (batch,), self.program_slot, dtype=torch.long, device=device
                ),
                None,
                None,
                None,
                router_state,
            )
        if router_state is None:
            raise RuntimeError("control-flow program router state is missing")
        if router_state.active_programs != self.program_memory.file_count:
            raise RuntimeError("control-flow program memory and router are out of sync")
        if self.program_route_query_adapter is None:
            features = intention.payload.detach()
        else:
            features = self.program_route_query_adapter(
                controller_output,
                controller_state,
            )
            if (
                not isinstance(features, torch.Tensor)
                or features.ndim != 2
                or features.shape[0] != batch
                or features.shape[1] != self.program_router.feature_width
            ):
                raise ValueError(
                    "control-flow route query adapter returned the wrong shape"
                )
            if not bool(torch.isfinite(features).all()):
                raise ValueError(
                    "control-flow route query adapter returned non-finite features"
                )
            features = features.detach()
        behavior = self.program_router.behavior_probabilities(
            router_state,
            features,
            exploration=self.program_route_exploration,
        )
        if self.program_route_exploration:
            selected = torch.multinomial(behavior, 1).squeeze(-1)
            propensity = behavior.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        else:
            selected = behavior.argmax(dim=-1)
            propensity = torch.ones(
                selected.shape,
                device=device,
                dtype=behavior.dtype,
            )
        next_router_state = self.program_router.record_decision(
            router_state,
            features,
            selected,
            propensity,
        )
        return selected, behavior, propensity, features, next_router_state

    def step_events(
        self,
        events: AmodalEventCollection | Sequence[AmodalEvent],
        state: ControlFlowRuntimeState,
        feedback: ControllerFeedback,
        *,
        route_feedback: ControllerFeedback | None = None,
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
        if route_feedback is None:
            route_feedback = feedback
        else:
            if self.program_router is None:
                raise ValueError(
                    "separate route feedback requires a program router"
                )
            route_feedback.validate(
                batch=batch,
                action_width=self.runtime.controller.feedback_width,
            )
        router_state = state.program_router
        if self.program_router is not None:
            if router_state is None:
                raise RuntimeError("control-flow program router state is missing")
            feedback_present = route_feedback.has_feedback.reshape(-1).to(torch.bool)
            feedback_reward = route_feedback.reward.reshape(-1)
            if bool(
                ((feedback_reward < 0.0) | (feedback_reward > 1.0))
                .logical_and(feedback_present)
                .any()
            ):
                raise ValueError("control-flow route feedback must lie in [0, 1]")
            router_state = self.program_router.apply_feedback(
                router_state,
                torch.where(
                    feedback_present,
                    feedback_reward,
                    torch.zeros_like(feedback_reward),
                ),
                present=feedback_present,
                terminal=feedback_present,
            )
        (
            selected_slots,
            route_probabilities,
            route_propensities,
            route_query,
            router_state,
        ) = (
            self._select_programs(controller_output, next_controller, router_state)
        )
        encoded = torch.zeros(
            batch,
            self.adapter.counter_count,
            device=device,
            dtype=torch.int64,
        )
        next_program_counters = dict(state.program_counters)
        executions_by_row: list[ControlFlowExecution | None] = [None] * batch
        program_digests: list[str] = [""] * batch
        for selected in torch.unique(selected_slots, sorted=True).tolist():
            selected = int(selected)
            mask = selected_slots == selected
            previous = (
                state.counters
                if selected < 0
                else next_program_counters.get(
                    selected,
                    torch.zeros_like(state.counters),
                )
            )
            candidate = self.adapter.encode(controller_output.intention, previous)
            candidate = self.adapter.validate_counters(
                candidate,
                batch=batch,
                device=device,
                name="encoded control-flow counters",
            )
            encoded = torch.where(mask.unsqueeze(-1), candidate, encoded)
            program = self._program_for_slot(None if selected < 0 else selected)
            digest = program.digest()
            for row in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
                execution = program.execute(
                    tuple(int(value) for value in candidate[row].tolist()),
                    max_steps=self.max_steps,
                    max_counter=self.max_counter,
                    trace_limit=self.trace_limit,
                )
                executions_by_row[row] = execution
                program_digests[row] = digest
        result_counters = torch.tensor(
            [
                execution.counters
                for execution in executions_by_row
                if execution is not None
            ],
            device=device,
            dtype=torch.int64,
        )
        if len(executions_by_row) != int(result_counters.shape[0]):
            raise RuntimeError("control-flow execution did not cover every batch row")
        executions = tuple(
            execution
            for execution in executions_by_row
            if execution is not None
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
        if self.program_router is not None:
            for selected in torch.unique(selected_slots, sorted=True).tolist():
                selected = int(selected)
                mask = selected_slots == selected
                previous = next_program_counters.get(
                    selected,
                    torch.zeros_like(state.counters),
                )
                next_program_counters[selected] = torch.where(
                    mask.unsqueeze(-1),
                    result_counters,
                    previous,
                ).detach()
        uniform_selection = bool(torch.all(selected_slots == selected_slots[0]))
        uniform_digest = (
            program_digests[0]
            if uniform_selection and program_digests
            else None
        )
        uniform_slot = (
            None
            if not uniform_selection or int(selected_slots[0]) < 0
            else int(selected_slots[0])
        )
        next_state = ControlFlowRuntimeState(
            controller=next_controller,
            counters=result_counters.detach(),
            counter_count=self.adapter.counter_count,
            program_counters=next_program_counters,
            program_router=router_state,
        ).validate()
        output = ControlFlowRuntimeOutput(
            controller=controller_output,
            executions=executions,
            intention=intention,
            decoded=self.runtime.output_bus(intention),
            program_digest=uniform_digest,
            program_slot=uniform_slot,
            selected_program_slots=selected_slots.detach().clone(),
            program_digests=tuple(program_digests),
            program_route_query=(
                None if route_query is None else route_query.detach().clone()
            ),
            program_route_probabilities=(
                None
                if route_probabilities is None
                else route_probabilities.detach().clone()
            ),
            program_route_propensities=(
                None
                if route_propensities is None
                else route_propensities.detach().clone()
            ),
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
