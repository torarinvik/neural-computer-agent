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

EXTERNAL_REGISTER_SCHEMA = "neural-computer.external-register.v1"
EXTERNAL_REGISTER_INSTRUCTION_SCHEMA = (
    "neural-computer.external-register-instruction.v1"
)


@dataclass(frozen=True)
class ExternalRegisterState:
    """External working state owned by the register interpreter."""

    register: torch.Tensor
    initialized: torch.Tensor

    def validate(
        self,
        *,
        batch_size: int,
        register_width: int,
    ) -> ExternalRegisterState:
        if self.register.ndim != 2 or self.register.shape != (
            batch_size,
            register_width,
        ):
            raise ValueError("external register has the wrong shape")
        if self.initialized.shape != (batch_size,):
            raise ValueError("external register initialization mask has the wrong shape")
        if self.initialized.dtype is not torch.bool:
            raise ValueError("external register initialization mask must be boolean")
        if self.initialized.device != self.register.device:
            raise ValueError("external register state must share a device")
        if not bool(torch.isfinite(self.register).all()):
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

    The input encoder is the machine's learned read boundary from the opaque
    controller intention and standardized event/feedback record. Once a
    register has been initialized, instructions operate only on the register
    and their opaque code vectors. Consequently a downstream instruction
    cannot bypass composition by rereading a raw event.
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
        if operator_mode not in ("factorized_low_rank", "unconstrained_mlp"):
            raise ValueError("unsupported external register operator mode")
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
        self.operator_mode = operator_mode
        self.operator_rank = int(operator_rank)
        seed_width = self.event_width + self.action_width + 2 + self.intention_width
        self.input_encoder = nn.Sequential(
            nn.Linear(seed_width, interpreter_hidden),
            nn.GELU(),
            nn.Linear(interpreter_hidden, register_width),
        )
        if operator_mode == "factorized_low_rank":
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
        else:
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
            "operator_mode": self.operator_mode,
            "operator_rank": self.operator_rank,
            "instruction_count": len(self.instructions),
            "state": "external_working_register_v1",
            "execution": "shared_interpreter_serial_instruction_chain_v1",
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
        )
        for name, value in (
            ("event", event),
            ("action", action),
            ("outcome", outcome),
            ("intention", intention.payload),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

    def _seed_register(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        present: torch.Tensor,
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
        return self.input_encoder(token)

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
        if self.operator_mode == "factorized_low_rank":
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
            proposal = torch.einsum("bk,brk->br", projected, left)
            return register + proposal + self.operator_bias(code)
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
        """Advance state and return the register for an external decoder.

        ``instructions`` is memory-side program data. It is intentionally a
        sequence of module objects rather than a task or protocol identifier.
        The default uses the machine's complete registered chain.
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
        active = present.unsqueeze(-1)
        seed = self._seed_register(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            present=present,
        )
        register = torch.where(
            state.initialized.unsqueeze(-1),
            state.register,
            torch.where(active, seed, state.register),
        )
        selected = self.instructions if instructions is None else tuple(instructions)
        for instruction in selected:
            updated = self.execute(register, instruction)
            register = torch.where(active, updated, register)
        return register, ExternalRegisterState(
            register=register,
            initialized=state.initialized | present,
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
        """Seed, execute memory-side instructions, and return an intention."""

        register, next_state = self.step_register(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
            instructions=instructions,
        )
        return self.to_intention(register), next_state
