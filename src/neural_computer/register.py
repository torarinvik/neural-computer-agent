"""A shared learned interpreter for external capability instructions.

The controller is the processor boundary and this module is its replaceable
external program store. Instructions are opaque learned vectors; the shared
interpreter applies them to an external working register. No instruction
receives raw modality data, a task identifier, or a protocol action ID.

This is deliberately a small execution contract, not a claim of arbitrary
program synthesis. Its purpose is to make that claim testable: a capability
should eventually be stored as instruction data and compose through one
interpreter, rather than adding another whole neural reasoning branch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .interface import IntentEvent

EXTERNAL_REGISTER_SCHEMA = "neural-computer.external-register.v2"
EXTERNAL_REGISTER_INSTRUCTION_SCHEMA = (
    "neural-computer.external-register-instruction.v1"
)
EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA = (
    "neural-computer.external-register-read-execute.v1"
)


@dataclass(frozen=True)
class ExternalRegisterState:
    """External working state owned by the register interpreter."""

    register: torch.Tensor
    context: torch.Tensor
    initialized: torch.Tensor

    def validate(
        self,
        *,
        batch_size: int,
        register_width: int,
        context_width: int,
    ) -> ExternalRegisterState:
        if self.register.ndim != 2 or self.register.shape != (
            batch_size,
            register_width,
        ):
            raise ValueError("external register has the wrong shape")
        if self.context.ndim != 2 or self.context.shape != (
            batch_size,
            context_width,
        ):
            raise ValueError("external register context has the wrong shape")
        if self.initialized.shape != (batch_size,):
            raise ValueError("external register initialization mask has the wrong shape")
        if self.initialized.dtype is not torch.bool:
            raise ValueError("external register initialization mask must be boolean")
        if self.initialized.device != self.register.device:
            raise ValueError("external register state must share a device")
        if self.context.device != self.register.device:
            raise ValueError("external register context must share a device")
        if not bool(torch.isfinite(self.register).all()) or not bool(
            torch.isfinite(self.context).all()
        ):
            raise ValueError("external register must contain only finite values")
        return self


class ExternalRegisterInstruction(nn.Module):
    """One opaque, independently persisted instruction vector.

    The vector has no assigned coordinate meaning. Its semantics are learned
    by the shared interpreter from verifier outcomes, so adding an instruction
    adds external program data rather than a modality-specific module branch.
    """

    def __init__(self, instruction_width: int) -> None:
        super().__init__()
        if instruction_width < 1:
            raise ValueError("instruction width must be positive")
        self.instruction_width = int(instruction_width)
        self.code = nn.Parameter(torch.randn(1, self.instruction_width) * 0.02)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_REGISTER_INSTRUCTION_SCHEMA,
            "instruction_width": self.instruction_width,
            "storage": "one_opaque_learned_vector_v1",
        }

    def expanded(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("instruction batch size must be positive")
        return self.code.to(device=device, dtype=dtype).expand(batch_size, -1)


class ExternalCapabilityRegisterMachine(nn.Module):
    """Execute variable-length external instruction data on one register.

    The recurrent context encoder reads every standardized event,
    feedback record, and opaque controller intention. It writes that context
    into the external working register once per active tick. Instructions
    then operate only on the register and their opaque code vectors, so a
    downstream instruction cannot bypass composition by rereading a raw event.
    """

    def __init__(
        self,
        event_width: int,
        action_width: int,
        intention_width: int,
        register_width: int,
        instruction_width: int,
        *,
        interpreter_hidden: int = 64,
        context_width: int | None = None,
        operator_mode: str = "factorized_low_rank",
        operator_rank: int = 8,
        instructions: Iterable[ExternalRegisterInstruction] = (),
    ) -> None:
        super().__init__()
        if min(
            event_width,
            action_width,
            intention_width,
            register_width,
            instruction_width,
            interpreter_hidden,
            operator_rank,
        ) < 1:
            raise ValueError("external register dimensions must be positive")
        if operator_mode not in (
            "factorized_low_rank",
            "factorized_film",
            "factorized_hybrid",
            "unconstrained_mlp",
        ):
            raise ValueError("unsupported external register operator mode")
        if context_width is not None and context_width < 1:
            raise ValueError("context width must be positive")
        members = tuple(instructions)
        if any(
            instruction.instruction_width != instruction_width
            for instruction in members
        ):
            raise ValueError("instructions must share the machine code width")
        self.event_width = int(event_width)
        self.action_width = int(action_width)
        self.intention_width = int(intention_width)
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.interpreter_hidden = int(interpreter_hidden)
        self.context_width = int(
            interpreter_hidden if context_width is None else context_width
        )
        self.operator_mode = operator_mode
        self.operator_rank = int(operator_rank)
        seed_width = self.event_width + self.action_width + 2 + self.intention_width
        self.input_encoder = nn.Sequential(
            nn.Linear(seed_width, interpreter_hidden),
            nn.GELU(),
        )
        self.context_recurrent = nn.GRUCell(interpreter_hidden, self.context_width)
        write_width = self.context_width + register_width
        self.register_writer = nn.Sequential(
            nn.Linear(write_width, interpreter_hidden),
            nn.GELU(),
            nn.Linear(interpreter_hidden, register_width),
        )
        self.register_write_gate = nn.Linear(write_width, 1)
        if operator_mode in (
            "factorized_low_rank",
            "factorized_hybrid",
        ):
            self.operator_left = nn.Linear(
                instruction_width,
                register_width * operator_rank,
            )
            self.operator_right = nn.Linear(
                instruction_width,
                operator_rank * register_width,
            )
            self.operator_bias = nn.Linear(instruction_width, register_width)
            for module in (
                self.operator_left,
                self.operator_right,
                self.operator_bias,
            ):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
        if operator_mode in ("factorized_film", "factorized_hybrid"):
            self.operator_feature = nn.Linear(register_width, interpreter_hidden)
            self.operator_modulation = nn.Linear(
                instruction_width, interpreter_hidden
            )
            self.operator_output = nn.Linear(interpreter_hidden, register_width)
            self.operator_gate = nn.Linear(interpreter_hidden * 2, 1)
            if operator_mode == "factorized_film":
                self.operator_bias = nn.Linear(instruction_width, register_width)
                nn.init.zeros_(self.operator_bias.bias)
            else:
                self.operator_film_bias = nn.Linear(
                    instruction_width, register_width
                )
                nn.init.zeros_(self.operator_output.weight)
                nn.init.zeros_(self.operator_output.bias)
                nn.init.zeros_(self.operator_film_bias.weight)
                nn.init.zeros_(self.operator_film_bias.bias)
        if operator_mode == "unconstrained_mlp":
            transition_width = register_width + instruction_width
            self.transition = nn.Sequential(
                nn.Linear(transition_width, interpreter_hidden),
                nn.GELU(),
                nn.Linear(interpreter_hidden, register_width),
            )
            self.update_gate = nn.Sequential(
                nn.Linear(transition_width, interpreter_hidden),
                nn.GELU(),
                nn.Linear(interpreter_hidden, 1),
            )
        self.output_adapter = nn.Linear(register_width, intention_width)
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)
        self.instructions = nn.ModuleList(members)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": EXTERNAL_REGISTER_SCHEMA,
            "event_width": self.event_width,
            "action_width": self.action_width,
            "intention_width": self.intention_width,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "interpreter_hidden": self.interpreter_hidden,
            "context_width": self.context_width,
            "operator_mode": self.operator_mode,
            "operator_rank": self.operator_rank,
            "instruction_count": len(self.instructions),
            "state": "external_working_register_with_recurrent_context_v2",
            "execution": "shared_interpreter_serial_instruction_chain_v1",
            "read_execute": EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
            "downstream_input": "preceding_register_only_v1",
        }

    def add_instruction(self, instruction: ExternalRegisterInstruction) -> int:
        """Append one learned program datum without resizing the machine."""

        if instruction.instruction_width != self.instruction_width:
            raise ValueError("instruction width does not match the machine")
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ExternalRegisterState:
        if batch_size < 1:
            raise ValueError("external register batch size must be positive")
        return ExternalRegisterState(
            register=torch.zeros(
                batch_size,
                self.register_width,
                device=device,
                dtype=dtype,
            ),
            context=torch.zeros(
                batch_size,
                self.context_width,
                device=device,
                dtype=dtype,
            ),
            initialized=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    def _validate_step_inputs(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor,
    ) -> None:
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("event has the wrong shape for external register")
        if action.ndim != 2 or action.shape != (
            event.shape[0],
            self.action_width,
        ):
            raise ValueError("action has the wrong shape for external register")
        if outcome.ndim != 1 or outcome.shape[0] != event.shape[0]:
            raise ValueError("outcome has the wrong shape for external register")
        if present.shape != outcome.shape or present.dtype is not torch.bool:
            raise ValueError("present must have shape [batch] and boolean dtype")
        if present.device != event.device:
            raise ValueError("present must share the event device")
        intention.validate(width=self.intention_width)
        if intention.payload.shape[0] != event.shape[0]:
            raise ValueError("intention batch does not match external register")
        state.validate(
            batch_size=event.shape[0],
            register_width=self.register_width,
            context_width=self.context_width,
        )
        for name, value in (
            ("event", event),
            ("action", action),
            ("outcome", outcome),
            ("intention", intention.payload),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

    def _advance_context(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        present: torch.Tensor,
        state: ExternalRegisterState,
    ) -> torch.Tensor:
        token = torch.cat(
            (
                event,
                action,
                outcome.unsqueeze(-1),
                present.to(dtype=event.dtype).unsqueeze(-1),
                intention.payload,
            ),
            dim=-1,
        )
        encoded = self.input_encoder(token)
        context = self.context_recurrent(encoded, state.context)
        return torch.where(present.unsqueeze(-1), context, state.context)

    def _write_context_to_register(
        self,
        *,
        register: torch.Tensor,
        context: torch.Tensor,
        present: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((context, register), dim=-1)
        proposal = self.register_writer(features)
        gate = torch.sigmoid(self.register_write_gate(features))
        return torch.where(
            present.unsqueeze(-1),
            register + gate * proposal,
            register,
        )

    def execute(
        self,
        register: torch.Tensor,
        instruction: ExternalRegisterInstruction,
    ) -> torch.Tensor:
        """Apply one instruction without access to raw events or feedback."""

        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for execution")
        if instruction.instruction_width != self.instruction_width:
            raise ValueError("instruction width does not match the machine")
        code = instruction.expanded(
            register.shape[0],
            device=register.device,
            dtype=register.dtype,
        )
        if self.operator_mode in (
            "factorized_low_rank",
            "factorized_hybrid",
        ):
            left = torch.tanh(self.operator_left(code)).reshape(
                register.shape[0],
                self.register_width,
                self.operator_rank,
            )
            right = torch.tanh(self.operator_right(code)).reshape(
                register.shape[0],
                self.operator_rank,
                self.register_width,
            )
            projected = torch.einsum("br,bkr->bk", register, right)
            base_proposal = torch.einsum("bk,brk->br", projected, left)
            if self.operator_mode == "factorized_low_rank":
                return register + base_proposal + self.operator_bias(code)
        if self.operator_mode in ("factorized_film", "factorized_hybrid"):
            features = torch.tanh(self.operator_feature(register))
            modulation = torch.tanh(self.operator_modulation(code))
            hidden = features * (1.0 + modulation)
            bias = (
                self.operator_bias
                if self.operator_mode == "factorized_film"
                else self.operator_film_bias
            )
            proposal = self.operator_output(hidden) + bias(code)
            gate = torch.sigmoid(
                self.operator_gate(torch.cat((features, modulation), dim=-1))
            )
            if self.operator_mode == "factorized_hybrid":
                return register + base_proposal + gate * proposal
            return register + gate * proposal
        features = torch.cat((register, code), dim=-1)
        proposal = self.transition(features)
        gate = torch.sigmoid(self.update_gate(features))
        return register + gate * proposal

    def execute_chain(
        self,
        register: torch.Tensor,
        instructions: Iterable[ExternalRegisterInstruction],
    ) -> torch.Tensor:
        """Apply a memory-selected instruction chain to a working register."""

        for instruction in instructions:
            register = self.execute(register, instruction)
        return register

    def observe_register(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState]:
        """Ingest one event into durable external working state.

        This phase never executes program data.  The returned register is the
        persistent observation store and the returned state is safe to carry
        across later execution snapshots.
        """

        if present is None:
            present = torch.ones(
                event.shape[0], dtype=torch.bool, device=event.device
            )
        self._validate_step_inputs(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
        )
        context = self._advance_context(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            present=present,
            state=state,
        )
        register = torch.where(
            state.initialized.unsqueeze(-1),
            self._write_context_to_register(
                register=state.register,
                context=context,
                present=present,
            ),
            torch.where(
                present.unsqueeze(-1),
                self.register_writer(
                    torch.cat((context, state.register), dim=-1)
                ),
                state.register,
            ),
        )
        next_state = ExternalRegisterState(
            register=register,
            context=context,
            initialized=state.initialized | present,
        )
        return register, next_state

    def read_execute_register(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor | None = None,
        instructions: Iterable[ExternalRegisterInstruction] | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState]:
        """Observe once, then execute a transient program snapshot.

        Program execution starts from the newly observed persistent register,
        but the execution result is not written back into ``next_state``.
        Therefore a later instruction or later tick cannot accidentally see
        an earlier execution result as if it were new sensory evidence.
        """

        if present is None:
            present = torch.ones(
                event.shape[0], dtype=torch.bool, device=event.device
            )
        register, next_state = self.observe_register(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
        )
        selected = self.instructions if instructions is None else tuple(instructions)
        executed = self.execute_chain(register, selected)
        return torch.where(
            present.unsqueeze(-1), executed, register
        ), next_state

    def to_intention(self, register: torch.Tensor) -> IntentEvent:
        """Project a register to the opaque intention transport boundary."""

        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for intention projection")
        return IntentEvent(self.output_adapter(register))

    def step_register(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor | None = None,
        instructions: Iterable[ExternalRegisterInstruction] | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState]:
        """Run the compatibility in-place execution path.

        ``instructions`` is memory-side program data. It is intentionally a
        sequence of module objects rather than a task or protocol identifier.
        The default uses the machine's complete registered chain. Unlike
        :meth:`read_execute_register`, its execution result is written back
        into the returned external state.
        """

        if present is None:
            present = torch.ones(
                event.shape[0], dtype=torch.bool, device=event.device
            )
        register, observed_state = self.observe_register(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            present=present,
            state=state,
        )
        selected = self.instructions if instructions is None else tuple(instructions)
        executed = self.execute_chain(register, selected)
        register = torch.where(
            present.unsqueeze(-1), executed, observed_state.register
        )
        return register, ExternalRegisterState(
            register=register,
            context=observed_state.context,
            initialized=observed_state.initialized,
        )

    def step(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor | None = None,
        instructions: Iterable[ExternalRegisterInstruction] | None = None,
    ) -> tuple[IntentEvent, ExternalRegisterState]:
        """Read context, write the register, execute, and return an intention."""

        register, next_state = self.read_execute_register(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
            instructions=instructions,
        )
        return self.to_intention(register), next_state
