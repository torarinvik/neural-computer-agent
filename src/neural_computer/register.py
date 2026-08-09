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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .interface import IntentEvent

EXTERNAL_REGISTER_SCHEMA = "neural-computer.external-register.v4"
EXTERNAL_REGISTER_INSTRUCTION_SCHEMA = (
    "neural-computer.external-register-instruction.v1"
)
EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA = (
    "neural-computer.external-register-read-execute.v1"
)
EXTERNAL_REGISTER_EXECUTION_TRACE_SCHEMA = (
    "neural-computer.external-register-execution-trace.v1"
)
EXTERNAL_REGISTER_BASIS_SCHEMA = "neural-computer.external-register-compute-basis.v1"
EXTERNAL_REGISTER_COMPATIBILITY_SCHEMA = (
    "neural-computer.external-register-compatibility-prior.v1"
)
EXTERNAL_REGISTER_READOUT_SCHEMA = (
    "neural-computer.external-register-canonical-readout.v1"
)
EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE = "factorized_shared_interpreter"
EXTERNAL_REGISTER_SHARED_BOUNDED_MODE = "factorized_shared_bounded"
EXTERNAL_REGISTER_SHARED_BANKED_MODE = "factorized_shared_banked"
EXTERNAL_REGISTER_SHARED_CANONICAL_MODE = "factorized_shared_canonical"
EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE = "factorized_shared_role_bound"
EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE = "factorized_shared_relational"
EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE = (
    "factorized_shared_stable_relational"
)
EXTERNAL_SEQUENCE_MEMORY_SCHEMA = "neural-computer.external-sequence-memory.v1"
EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA = (
    "neural-computer.external-sequence-program-memory.v1"
)


EXTERNAL_REGISTER_ROLE_BINDING_SCHEMA = (
    "neural-computer.external-register-learned-role-binding.v1"
)


class LearnedRegisterRoleBinding(nn.Module):
    """Bind one register state into shared, learned latent role slots.

    Slots have no hand-assigned semantics.  They are persistent parameters
    shared by every instruction, while the opaque instruction conditions the
    query used to read a projected state-token bank.  The output is a compact
    ``[batch, role, role_width]`` representation that preserves separately
    bindable content for a downstream decoder.
    """

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        *,
        role_count: int = 4,
        instruction_conditioned: bool = True,
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, role_count) < 1:
            raise ValueError("role-binding dimensions must be positive")
        if register_width % role_count:
            raise ValueError("register width must divide evenly into role slots")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.role_count = int(role_count)
        self.instruction_conditioned = bool(instruction_conditioned)
        self.role_width = self.register_width // self.role_count
        self.role_seed = nn.Parameter(torch.randn(role_count, self.role_width) * 0.02)
        self.state_tokens = nn.Linear(
            register_width, role_count * self.role_width
        )
        self.key = nn.Linear(self.role_width, self.role_width)
        self.value = nn.Linear(self.role_width, self.role_width)
        self.query = nn.Linear(instruction_width, self.role_width)
        self.gate = nn.Linear(instruction_width, role_count)
        self.contract = nn.LayerNorm(self.role_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_REGISTER_ROLE_BINDING_SCHEMA,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "role_count": self.role_count,
            "role_width": self.role_width,
            "binding": "shared_instruction_conditioned_slot_attention_v1",
            "instruction_conditioned": self.instruction_conditioned,
            "semantics": "learned_from_verifier_outcomes_no_assigned_roles_v1",
        }

    def forward(
        self,
        register: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for role binding")
        if code.shape != (register.shape[0], self.instruction_width):
            raise ValueError("instruction code has the wrong shape for role binding")
        tokens = self.state_tokens(register).view(
            register.shape[0], self.role_count, self.role_width
        )
        queries = self.role_seed.unsqueeze(0)
        if self.instruction_conditioned:
            queries = queries + self.query(code).unsqueeze(1)
        scores = torch.einsum(
            "brd,btd->brt", queries, self.key(tokens)
        ).div(self.role_width**0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("brt,btd->brd", weights, self.value(tokens))
        roles = self.contract(attended + queries)
        if self.instruction_conditioned:
            roles = roles * torch.sigmoid(self.gate(code)).unsqueeze(-1)
        return roles


EXTERNAL_REGISTER_RELATIONAL_TRANSITION_SCHEMA = (
    "neural-computer.external-register-relational-transition.v1"
)


class InstructionConditionedRelationalTransition(nn.Module):
    """Use learned role relations inside, rather than after, a transition.

    The role slots are an internal coordinate system shared by all opaque
    instructions.  Each instruction conditions both the role binding and a
    cross-role attention pass; the mixed slots are then projected back into a
    bounded register proposal.  No slot is assigned a semantic label.
    """

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        *,
        role_count: int = 4,
        instruction_conditioned_binding: bool = True,
    ) -> None:
        super().__init__()
        self.binding = LearnedRegisterRoleBinding(
            register_width,
            instruction_width,
            role_count=role_count,
            instruction_conditioned=instruction_conditioned_binding,
        )
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.role_count = int(role_count)
        self.role_width = self.binding.role_width
        self.instruction_conditioned_binding = bool(instruction_conditioned_binding)
        self.query = nn.Linear(instruction_width, self.role_width)
        self.key = nn.Linear(self.role_width, self.role_width)
        self.value = nn.Linear(self.role_width, self.role_width)
        self.output = nn.Linear(register_width, register_width)
        self.gate = nn.Linear(instruction_width, register_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_REGISTER_RELATIONAL_TRANSITION_SCHEMA,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "role_count": self.role_count,
            "role_width": self.role_width,
            "transition": "instruction_conditioned_cross_role_attention_v1",
            "instruction_conditioned_binding": self.instruction_conditioned_binding,
            "semantics": "learned_from_verifier_outcomes_no_assigned_roles_v1",
        }

    def forward(
        self,
        register: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        roles = self.binding(register, code)
        queries = roles + self.query(code).unsqueeze(1)
        scores = torch.einsum(
            "brd,btd->brt", queries, self.key(roles)
        ).div(self.role_width**0.5)
        weights = torch.softmax(scores, dim=-1)
        mixed = roles + torch.einsum(
            "brt,btd->brd", weights, self.value(roles)
        )
        proposal = self.output(mixed.flatten(1))
        return proposal * torch.sigmoid(self.gate(code))


class CanonicalRegisterReadout(nn.Module):
    """Learned, protocol-independent boundary from register state to output state.

    The readout is deliberately separate from the register interpreter.  It is
    identity-initialized, so inserting it does not change an existing runtime,
    while its residual path can learn a shared coordinate/scale convention
    across mastered external programs.  It consumes only the executed
    register; no operation, modality, or verifier metadata is available here.
    """

    def __init__(self, register_width: int, output_width: int, *, hidden: int = 64) -> None:
        super().__init__()
        if min(register_width, output_width, hidden) < 1:
            raise ValueError("canonical readout dimensions must be positive")
        self.register_width = int(register_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.base = nn.Linear(register_width, output_width)
        if register_width == output_width:
            nn.init.eye_(self.base.weight)
        else:
            nn.init.xavier_uniform_(self.base.weight)
        nn.init.zeros_(self.base.bias)
        self.residual = nn.Sequential(
            nn.Linear(register_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_width),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_REGISTER_READOUT_SCHEMA,
            "register_width": self.register_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "initialization": "identity_plus_zero_residual_v1",
            "metadata": "register_only_v1",
        }

    def forward(self, register: torch.Tensor) -> torch.Tensor:
        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for canonical readout")
        if not bool(torch.isfinite(register).all()):
            raise ValueError("register must contain only finite values")
        return self.base(register) + self.residual(register)


@dataclass(frozen=True)
class ExternalRegisterState:
    """External working state owned by the register interpreter."""

    register: torch.Tensor
    context: torch.Tensor
    initialized: torch.Tensor
    event_window: torch.Tensor | None = None
    event_window_mask: torch.Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int,
        register_width: int,
        context_width: int,
        event_width: int | None = None,
        event_window_size: int = 0,
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
        if event_window_size < 0:
            raise ValueError("event window size must be non-negative")
        if self.event_window is not None or self.event_window_mask is not None:
            if self.event_window is None or self.event_window_mask is None:
                raise ValueError("event window and mask must be provided together")
            if event_width is None:
                raise ValueError("event width is required for an event window")
            if self.event_window.shape != (
                batch_size, event_window_size, event_width
            ):
                raise ValueError("external event window has the wrong shape")
            if self.event_window_mask.shape != (batch_size, event_window_size):
                raise ValueError("external event window mask has the wrong shape")
            if self.event_window_mask.dtype is not torch.bool:
                raise ValueError("external event window mask must be boolean")
            if self.event_window.device != self.register.device:
                raise ValueError("external event window must share a device")
            if not bool(torch.isfinite(self.event_window).all()):
                raise ValueError("external event window must contain finite values")
        return self


class ExternalSequenceMemory(nn.Module):
    """Append-only opaque state slots for frozen-controller adaptation.

    Slots are external learned state, not controller computation.  A slot is
    read as a register-width context and can therefore specialize a shared
    operator without adding a modality- or procedure-specific controller
    branch.  The slot count grows independently of the controller width.
    """

    def __init__(self, value_width: int) -> None:
        super().__init__()
        if value_width < 1:
            raise ValueError("sequence memory value width must be positive")
        self.value_width = int(value_width)
        self.slots = nn.ParameterList()

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_SEQUENCE_MEMORY_SCHEMA,
            "value_width": self.value_width,
            "slot_count": len(self.slots),
            "growth": "append_only_external_state_v1",
        }

    def add_slot(self, value: torch.Tensor | None = None) -> int:
        if value is None:
            value = torch.zeros(self.value_width)
        if value.shape != (self.value_width,):
            raise ValueError("sequence memory slot has the wrong shape")
        self.slots.append(nn.Parameter(value.detach().clone()))
        return len(self.slots) - 1

    def read(
        self,
        slot: int,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not 0 <= slot < len(self.slots):
            raise ValueError("sequence memory slot index is out of range")
        return self.slots[slot].to(device=device, dtype=dtype).expand(
            batch_size, -1
        )


class ExternalSequenceProgramMemory(nn.Module):
    """Append-only opaque program data executed by one shared interpreter."""

    schema = EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA

    def __init__(
        self,
        instruction_width: int,
        *,
        router_hidden: int = 32,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(instruction_width, router_hidden) < 1:
            raise ValueError("sequence program memory dimensions must be positive")
        if router_temperature <= 0.0:
            raise ValueError("sequence program router temperature must be positive")
        self.instruction_width = int(instruction_width)
        self.router_hidden = int(router_hidden)
        self.router_temperature = float(router_temperature)
        self.programs = nn.ParameterList()
        self.slot_keys = nn.ParameterList()
        self.query_encoder = nn.GRU(
            self.instruction_width, self.router_hidden, batch_first=True
        )
        self.program_query = nn.Linear(
            self.router_hidden, self.instruction_width
        )
        self.route_query_encoder = nn.Sequential(
            nn.Linear(self.instruction_width, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.router_hidden),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(self.instruction_width, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.router_hidden),
        )

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "instruction_width": self.instruction_width,
            "router_hidden": self.router_hidden,
            "router_temperature": self.router_temperature,
            "slot_count": len(self.programs),
            "program_lengths": [int(program.shape[0]) for program in self.programs],
            "storage": "append_only_opaque_instruction_sequences_v1",
            "computation": "shared_register_interpreter_v1",
        }

    def add_program(self, codes: torch.Tensor) -> int:
        if (
            codes.ndim != 2
            or codes.shape[0] < 1
            or codes.shape[1] != self.instruction_width
        ):
            raise ValueError(
                "program codes must have shape [steps, instruction_width]"
            )
        if not bool(torch.isfinite(codes).all()):
            raise ValueError("program codes must be finite")
        self.programs.append(nn.Parameter(codes.detach().clone()))
        key = nn.Parameter(torch.empty(self.instruction_width))
        nn.init.normal_(key, mean=0.0, std=0.02)
        self.slot_keys.append(key)
        return len(self.programs) - 1

    def program_codes(
        self,
        slot: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not 0 <= slot < len(self.programs):
            raise ValueError("sequence program memory slot index is out of range")
        return self.programs[slot].to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def encode_program(self, codes: torch.Tensor) -> torch.Tensor:
        if (
            codes.ndim != 3
            or codes.shape[1] < 1
            or codes.shape[2] != self.instruction_width
        ):
            raise ValueError("program codes must have shape [batch, steps, width]")
        _, hidden = self.query_encoder(codes)
        return self.program_query(hidden[-1])

    def route_weights(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or query.shape[1] != self.instruction_width:
            raise ValueError("sequence program route query has the wrong shape")
        if not len(self.programs):
            raise ValueError("cannot route an empty sequence program memory")
        keys = torch.stack(tuple(self.slot_keys), dim=0)
        query_latent = self.route_query_encoder(query)
        key_latent = self.key_encoder(keys)
        logits = torch.einsum("bh,sh->bs", query_latent, key_latent)
        return torch.softmax(
            logits / (self.router_temperature * (self.router_hidden**0.5)), dim=-1
        )


class _ExternalSequenceOperatorSlot(nn.Module):
    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        operator_rank: int,
    ) -> None:
        super().__init__()
        self.register_width = register_width
        self.instruction_width = instruction_width
        self.operator_rank = operator_rank
        self.left = nn.Linear(
            instruction_width, register_width * operator_rank
        )
        self.right = nn.Linear(
            instruction_width, operator_rank * register_width
        )
        self.bias = nn.Linear(instruction_width, register_width)
        self.gate = nn.Linear(instruction_width, register_width)
        for module in (self.left, self.right, self.bias):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def residual(
        self, register: torch.Tensor, code: torch.Tensor
    ) -> torch.Tensor:
        left = torch.tanh(self.left(code)).reshape(
            register.shape[0], self.register_width, self.operator_rank
        )
        right = torch.tanh(self.right(code)).reshape(
            register.shape[0], self.operator_rank, self.register_width
        )
        projected = torch.einsum("br,bkr->bk", register, right)
        proposal = torch.einsum("bk,brk->br", projected, left) + self.bias(code)
        return 0.5 * torch.sigmoid(self.gate(code)) * torch.tanh(proposal)


class ExternalSequenceOperatorMemory(nn.Module):
    """Append-only instruction-conditioned transition slots.

    Unlike a value slot, each entry stores a small learned operator.  Slots
    are external state and grow independently from the controller and shared
    register interpreter.
    """

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        *,
        operator_rank: int = 8,
        router_hidden: int = 32,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, operator_rank, router_hidden) < 1:
            raise ValueError("sequence operator memory dimensions must be positive")
        if router_temperature <= 0.0:
            raise ValueError("sequence operator router temperature must be positive")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.operator_rank = int(operator_rank)
        self.router_hidden = int(router_hidden)
        self.router_temperature = float(router_temperature)
        self.slots = nn.ModuleList()
        self.slot_keys = nn.ParameterList()
        self.query_encoder = nn.Sequential(
            nn.Linear(self.instruction_width, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.router_hidden),
        )
        self.key_encoder = nn.Sequential(
            nn.Linear(self.instruction_width, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.router_hidden),
        )
        self.program_encoder = nn.GRU(
            self.instruction_width,
            self.router_hidden,
            batch_first=True,
        )
        self.program_query = nn.Linear(
            self.router_hidden, self.instruction_width
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": "neural-computer.external-sequence-operator-memory.v1",
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "operator_rank": self.operator_rank,
            "router_hidden": self.router_hidden,
            "router_temperature": self.router_temperature,
            "route_encoder": "gru_order_sensitive_v1",
            "slot_count": len(self.slots),
            "growth": "append_only_external_operator_state_v1",
        }

    def add_slot(self) -> int:
        self.slots.append(
            _ExternalSequenceOperatorSlot(
                self.register_width,
                self.instruction_width,
                self.operator_rank,
            )
        )
        key = nn.Parameter(torch.empty(self.instruction_width))
        nn.init.normal_(key, mean=0.0, std=0.02)
        self.slot_keys.append(key)
        return len(self.slots) - 1

    def encode_program(self, codes: torch.Tensor) -> torch.Tensor:
        """Encode an ordered opaque instruction chain into a route query."""

        if (
            codes.ndim != 3
            or codes.shape[2] != self.instruction_width
            or codes.shape[1] < 1
        ):
            raise ValueError(
                "sequence operator program codes must have shape [batch, steps, width]"
            )
        _, hidden = self.program_encoder(codes)
        return self.program_query(hidden[-1])

    def route_weights(self, query: torch.Tensor) -> torch.Tensor:
        """Return learned soft address weights for an opaque program query."""

        if query.ndim != 2 or query.shape[1] != self.instruction_width:
            raise ValueError("sequence operator route query has the wrong shape")
        if not len(self.slots):
            raise ValueError("cannot route an empty sequence operator memory")
        keys = torch.stack(tuple(self.slot_keys), dim=0)
        query_latent = self.query_encoder(query)
        key_latent = self.key_encoder(keys)
        logits = torch.einsum("bh,sh->bs", query_latent, key_latent)
        return torch.softmax(
            logits / (self.router_temperature * (self.router_hidden**0.5)), dim=-1
        )

    def residual(
        self,
        slot: int,
        register: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        if not 0 <= slot < len(self.slots):
            raise ValueError("sequence operator memory slot index is out of range")
        return self.slots[slot].residual(register, code)

    def routed_residual(
        self,
        query: torch.Tensor,
        register: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a learned soft mixture of all opaque operator slots."""

        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for routed memory")
        if query.shape[0] != register.shape[0] or code.shape[0] != register.shape[0]:
            raise ValueError("routed memory batch dimensions do not match")
        weights = self.route_weights(query)
        residuals = torch.stack(
            tuple(slot.residual(register, code) for slot in self.slots), dim=1
        )
        return torch.einsum("bs,bsr->br", weights, residuals)


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


class ExternalRegisterComputeBasis(nn.Module):
    """One append-only external computation slot.

    A slot is fresh computation capacity, not a semantic label. It sees only
    the current register and an opaque instruction vector and returns a bounded
    register update. New slots can be trained without changing the controller
    or parameters of previously mastered slots.
    """

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        *,
        hidden: int = 64,
        event_width: int = 0,
        event_window_size: int = 0,
        microsteps: int = 1,
        event_read_mode: str = "flattened_window",
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, hidden) < 1:
            raise ValueError("compute basis dimensions must be positive")
        if min(event_width, event_window_size) < 0:
            raise ValueError("event window dimensions must be non-negative")
        if event_window_size and event_width < 1:
            raise ValueError("event width must be positive for an event window")
        if microsteps < 1:
            raise ValueError("compute basis microsteps must be positive")
        if event_read_mode not in ("flattened_window", "attention_pool"):
            raise ValueError("unsupported compute basis event read mode")
        if event_read_mode == "attention_pool" and not event_window_size:
            raise ValueError("attention event reading requires an event window")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.hidden = int(hidden)
        self.event_width = int(event_width)
        self.event_window_size = int(event_window_size)
        self.microsteps = int(microsteps)
        self.event_read_mode = event_read_mode
        self.event_window_width = self.event_width * self.event_window_size
        event_feature_width = (
            self.event_width
            if event_read_mode == "attention_pool"
            else self.event_window_width
        )
        width = self.register_width + self.instruction_width + event_feature_width
        if event_read_mode == "attention_pool":
            query_width = self.register_width + self.instruction_width
            self.event_query = nn.Linear(query_width, self.hidden)
            self.event_key = nn.Linear(self.event_width, self.hidden)
            self.event_value = nn.Linear(self.event_width, self.event_width)
        self.network = nn.Sequential(
            nn.Linear(width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.register_width),
        )
        self.gate = nn.Sequential(
            nn.Linear(width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.signature = nn.Parameter(torch.randn(self.instruction_width) * 0.02)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_REGISTER_BASIS_SCHEMA,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "hidden": self.hidden,
            "event_width": self.event_width,
            "event_window_size": self.event_window_size,
            "microsteps": self.microsteps,
            "event_read_mode": self.event_read_mode,
            "storage": "append_only_external_compute_slot_v1",
            "signature": "one_opaque_learned_slot_key_v1",
        }

    def forward(
        self,
        register: torch.Tensor,
        code: torch.Tensor,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for compute basis")
        if code.shape != (register.shape[0], self.instruction_width):
            raise ValueError("instruction code has the wrong shape for compute basis")
        if self.event_window_size:
            if event_window is None or event_window_mask is None:
                raise ValueError("event window is required for this compute basis")
            if event_window.shape != (
                register.shape[0], self.event_window_size, self.event_width
            ):
                raise ValueError("event window has the wrong shape for compute basis")
            if event_window_mask.shape != (
                register.shape[0], self.event_window_size
            ) or event_window_mask.dtype is not torch.bool:
                raise ValueError("event window mask has the wrong shape")
            window = event_window * event_window_mask.unsqueeze(-1).to(event_window.dtype)
        else:
            if event_window is not None or event_window_mask is not None:
                raise ValueError("event window is unsupported by this compute basis")
            window = None
        for _ in range(self.microsteps):
            if self.event_read_mode == "attention_pool":
                query = self.event_query(torch.cat((register, code), dim=-1))
                keys = self.event_key(window)
                values = self.event_value(window)
                scores = torch.einsum("bd,btd->bt", query, keys)
                scores = scores / (self.hidden**0.5)
                valid = event_window_mask
                scores = scores.masked_fill(~valid, -1e9)
                weights = torch.softmax(scores, dim=-1)
                event_features = (
                    weights * valid.to(weights.dtype)
                ).unsqueeze(-1) * values
                event_features = event_features.sum(dim=1)
            else:
                event_features = (
                    torch.zeros(
                        register.shape[0], self.event_window_width,
                        device=register.device, dtype=register.dtype,
                    )
                    if window is None
                    else window.flatten(1)
                )
            features = torch.cat(
                (register, code, event_features)
                if self.event_window_size
                else (register, code),
                dim=-1,
            )
            register = register + torch.sigmoid(self.gate(features)) * torch.tanh(
                self.network(features)
            )
        return register


class ExternalRegisterBasisCompatibilityPrior(nn.Module):
    """Learn an opaque ordering over external basis slots from outcomes.

    This is memory-side screening only. It sees an opaque instruction query
    and opaque learned slot signatures; it can order candidates using scalar
    attempted outcomes, but it cannot admit a slot. Fresh stable-prefix
    verifier evidence remains the authority for reuse versus growth.
    """

    def __init__(
        self,
        instruction_width: int,
        *,
        latent_width: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(instruction_width, latent_width, hidden) < 1:
            raise ValueError("compatibility prior dimensions must be positive")
        from .capability import LearnedComputeCandidateScreen

        self.instruction_width = int(instruction_width)
        self.latent_width = int(latent_width)
        self.hidden = int(hidden)
        self.screen = LearnedComputeCandidateScreen(
            instruction_width,
            instruction_width,
            latent_width=latent_width,
            hidden=hidden,
        )

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": EXTERNAL_REGISTER_COMPATIBILITY_SCHEMA,
            "instruction_width": self.instruction_width,
            "latent_width": self.latent_width,
            "hidden": self.hidden,
            "query": "opaque_instruction_vector_v1",
            "candidate_key": "opaque_basis_signature_v1",
            "training": "attempted_scalar_outcome_pairwise_ranking_v1",
            "role": "screening_only_fresh_admission_required",
            "enabled": bool(self.screen.enabled.item()),
        }

    def enable(self) -> None:
        self.screen.enable()

    def forward(
        self,
        instruction_query: torch.Tensor,
        basis_keys: torch.Tensor,
    ) -> torch.Tensor:
        return self.screen(instruction_query, basis_keys)

    def outcome_ranking_loss(
        self,
        instruction_query: torch.Tensor,
        basis_keys: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        return self.screen.outcome_ranking_loss(
            instruction_query,
            basis_keys,
            outcomes,
        )

    @torch.no_grad()
    def order(
        self,
        instruction_query: torch.Tensor,
        basis_keys: torch.Tensor,
    ) -> tuple[int, ...]:
        """Order opaque basis candidates for staged trial scheduling."""

        return self.screen.order(instruction_query, basis_keys)

    @staticmethod
    def basis_keys(
        basis_slots: Iterable[ExternalRegisterComputeBasis],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        slots = tuple(basis_slots)
        if not slots:
            raise ValueError("compatibility prior needs at least one basis slot")
        keys = torch.stack(tuple(slot.signature for slot in slots))
        if device is not None or dtype is not None:
            keys = keys.to(device=device, dtype=dtype)
        return keys


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
        basis_slots: Iterable[ExternalRegisterComputeBasis] = (),
        basis_hidden: int = 64,
        basis_microsteps: int = 1,
        basis_event_read_mode: str = "flattened_window",
        event_input_mode: str = "frontend",
        event_window_size: int = 0,
        role_count: int = 4,
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
            basis_hidden,
            basis_microsteps,
        ) < 1:
            raise ValueError("external register dimensions must be positive")
        if basis_event_read_mode not in ("flattened_window", "attention_pool"):
            raise ValueError("unsupported basis event read mode")
        if basis_event_read_mode == "attention_pool" and not event_window_size:
            raise ValueError("attention basis reading requires an event window")
        if event_input_mode not in (
            "frontend",
            "append_controller_state",
            "controller_state",
        ):
            raise ValueError("unsupported external event input mode")
        if event_window_size < 0:
            raise ValueError("event window size must be non-negative")
        if role_count < 1:
            raise ValueError("role count must be positive")
        if operator_mode not in (
            "factorized_low_rank",
            "factorized_film",
            "factorized_hybrid",
            "factorized_bounded_residual",
            "factorized_protected_meta",
            "factorized_protected_bounded_meta",
            EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
            EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
            EXTERNAL_REGISTER_SHARED_BANKED_MODE,
            EXTERNAL_REGISTER_SHARED_CANONICAL_MODE,
            EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
            EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
            EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
            "unconstrained_mlp",
        ):
            raise ValueError("unsupported external register operator mode")
        if (
            operator_mode == EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE
            and register_width % role_count
        ):
            raise ValueError("role-bound mode requires divisible register width")
        if (
            operator_mode == EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE
            and register_width % role_count
        ):
            raise ValueError("relational mode requires divisible register width")
        if (
            operator_mode == EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE
            and register_width % role_count
        ):
            raise ValueError("stable relational mode requires divisible register width")
        if context_width is not None and context_width < 1:
            raise ValueError("context width must be positive")
        members = tuple(instructions)
        basis_members = tuple(basis_slots)
        if any(
            instruction.instruction_width != instruction_width
            for instruction in members
        ):
            raise ValueError("instructions must share the machine code width")
        if any(
            basis.register_width != register_width
            or basis.instruction_width != instruction_width
            or basis.event_window_size != event_window_size
            or basis.microsteps != basis_microsteps
            or basis.event_read_mode != basis_event_read_mode
            or (event_window_size and basis.event_width != event_width)
            for basis in basis_members
        ):
            raise ValueError("basis slots must share machine register and code widths")
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
        self.basis_hidden = int(basis_hidden)
        self.basis_microsteps = int(basis_microsteps)
        self.basis_event_read_mode = basis_event_read_mode
        self.event_input_mode = event_input_mode
        self.event_window_size = int(event_window_size)
        self.role_count = int(role_count)
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
            "factorized_bounded_residual",
            "factorized_protected_meta",
            "factorized_protected_bounded_meta",
            EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
            EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
            EXTERNAL_REGISTER_SHARED_BANKED_MODE,
            EXTERNAL_REGISTER_SHARED_CANONICAL_MODE,
            EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
            EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
            EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
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
        if operator_mode in (
            "factorized_bounded_residual",
            EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
            EXTERNAL_REGISTER_SHARED_BANKED_MODE,
            EXTERNAL_REGISTER_SHARED_CANONICAL_MODE,
            EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
            EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
            EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
            "factorized_protected_bounded_meta",
        ):
            # Serial instruction chains are sensitive to unbounded additive
            # drift.  Normalize the read state, bound the learned proposal,
            # and let the opaque instruction choose a feature-wise residual
            # gate.  The register remains the only downstream input.
            self.operator_normalizer = nn.LayerNorm(register_width)
            self.operator_composition_gate = nn.Linear(
                instruction_width, register_width
            )
            nn.init.zeros_(self.operator_composition_gate.bias)
        if operator_mode == EXTERNAL_REGISTER_SHARED_BANKED_MODE:
            self.bank_query = nn.Linear(instruction_width, instruction_width)
            self.bank_key = nn.Linear(register_width, instruction_width)
            self.bank_value = nn.Linear(register_width, register_width)
            self.bank_gate = nn.Linear(instruction_width, 1)
            nn.init.zeros_(self.bank_gate.bias)
        if operator_mode == EXTERNAL_REGISTER_SHARED_CANONICAL_MODE:
            self.state_contract = nn.LayerNorm(register_width)
        if operator_mode == EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE:
            self.role_binding = LearnedRegisterRoleBinding(
                register_width,
                instruction_width,
                role_count=role_count,
            )
        if operator_mode == EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE:
            self.relational_transition = InstructionConditionedRelationalTransition(
                register_width,
                instruction_width,
                role_count=role_count,
            )
        if operator_mode == EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE:
            self.relational_transition = InstructionConditionedRelationalTransition(
                register_width,
                instruction_width,
                role_count=role_count,
                instruction_conditioned_binding=False,
            )
        if operator_mode in (
            "factorized_protected_meta",
            "factorized_protected_bounded_meta",
        ):
            # This branch is an isolated, initially inert operator-family
            # prior.  The base low-rank operator can be protected while this
            # residual family learns across later fresh procedures.
            self.operator_meta_left = nn.Linear(
                instruction_width, register_width * operator_rank
            )
            self.operator_meta_right = nn.Linear(
                instruction_width, operator_rank * register_width
            )
            self.operator_meta_bias = nn.Linear(
                instruction_width, register_width
            )
            self.operator_meta_gate = nn.Linear(
                instruction_width, register_width
            )
            for module in (
                self.operator_meta_left,
                self.operator_meta_right,
                self.operator_meta_bias,
                self.operator_meta_gate,
            ):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
            nn.init.constant_(self.operator_meta_gate.bias, -4.0)
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
        self.basis_slots = nn.ModuleList(basis_members)

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
            "basis_slot_count": len(self.basis_slots),
            "basis_hidden": self.basis_hidden,
            "basis_microsteps": self.basis_microsteps,
            "basis_event_read_mode": self.basis_event_read_mode,
            "event_input_mode": self.event_input_mode,
            "event_window_size": self.event_window_size,
            "role_count": self.role_count,
            "state": "external_working_register_with_recurrent_context_v2",
            "execution": "shared_interpreter_serial_instruction_chain_v1",
            "compute_basis": (
                "neural-computer.external-register-shared-interpreter.v1"
                if self.operator_mode in (
                    EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
                    EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
                    EXTERNAL_REGISTER_SHARED_BANKED_MODE,
                    EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
                    EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
                    EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
                    "factorized_protected_bounded_meta",
                )
                else EXTERNAL_REGISTER_BASIS_SCHEMA
            ),
            "basis_binding": (
                "instruction_vector_selects_shared_interpreter_v1"
                if self.operator_mode in (
                    EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
                    EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
                    EXTERNAL_REGISTER_SHARED_BANKED_MODE,
                    EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
                    EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
                    EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
                    "factorized_protected_bounded_meta",
                )
                else "opaque_memory_side_slot_index_v1"
            ),
            "read_execute": EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
            "execution_trace": EXTERNAL_REGISTER_EXECUTION_TRACE_SCHEMA,
            "downstream_input": "preceding_register_plus_bounded_event_window_v1",
        }

    def add_instruction(self, instruction: ExternalRegisterInstruction) -> int:
        """Append one learned program datum without resizing the machine."""

        if instruction.instruction_width != self.instruction_width:
            raise ValueError("instruction width does not match the machine")
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def add_basis_slot(
        self, basis: ExternalRegisterComputeBasis | None = None
    ) -> int:
        """Append fresh external computation capacity and return its address."""

        if basis is None:
            basis = ExternalRegisterComputeBasis(
                self.register_width,
                self.instruction_width,
                hidden=self.basis_hidden,
                microsteps=self.basis_microsteps,
                event_read_mode=self.basis_event_read_mode,
                event_width=self.event_width if self.event_window_size else 0,
                event_window_size=self.event_window_size,
            )
        if (
            basis.register_width != self.register_width
            or basis.instruction_width != self.instruction_width
            or basis.event_window_size != self.event_window_size
            or basis.microsteps != self.basis_microsteps
            or basis.event_read_mode != self.basis_event_read_mode
            or (
                self.event_window_size
                and basis.event_width != self.event_width
            )
        ):
            raise ValueError("basis slot dimensions do not match the machine")
        self.basis_slots.append(basis)
        return len(self.basis_slots) - 1

    def select_basis_slot(
        self,
        candidate_outcomes: Mapping[int, Sequence[float]],
        *,
        threshold: float,
    ):
        """Apply the verifier-gated memory policy to existing basis slots.

        This is deliberately pure: it never edits weights or chooses based on
        a semantic/task identifier. The caller supplies fresh verifier probes;
        the shared admission policy returns either an opaque existing slot to
        reuse or an instruction to grow capacity with :meth:`add_basis_slot`.
        """

        from .capability import select_reusable_compute_slot

        if any(index < 0 or index >= len(self.basis_slots) for index in candidate_outcomes):
            raise ValueError("basis candidate index is out of range")
        return select_reusable_compute_slot(
            candidate_outcomes,
            threshold=threshold,
        )

    def select_basis_slot_by_efficiency(
        self,
        candidate_outcomes: Mapping[int, Sequence[float]],
        candidate_stable_bits: Mapping[int, int | None],
        *,
        fresh_stable_bits: int | None,
        threshold: float,
    ):
        """Require both fresh mastery and no worse stable cost than fresh."""

        from .capability import select_reusable_compute_slot_by_efficiency

        candidate_indices = set(candidate_outcomes) | set(candidate_stable_bits)
        if any(
            index < 0 or index >= len(self.basis_slots)
            for index in candidate_indices
        ):
            raise ValueError("basis candidate index is out of range")
        return select_reusable_compute_slot_by_efficiency(
            candidate_outcomes,
            candidate_stable_bits,
            fresh_stable_bits=fresh_stable_bits,
            threshold=threshold,
        )

    def order_basis_candidates(
        self,
        prior: ExternalRegisterBasisCompatibilityPrior,
        instruction_query: torch.Tensor,
        candidate_indices: Iterable[int] | None = None,
    ) -> tuple[int, ...]:
        """Order memory-selected basis candidates without admitting one."""

        if not self.basis_slots:
            raise ValueError("cannot order an empty basis slot set")
        selected = (
            tuple(range(len(self.basis_slots)))
            if candidate_indices is None
            else tuple(candidate_indices)
        )
        if not selected:
            raise ValueError("basis candidate set cannot be empty")
        if any(index < 0 or index >= len(self.basis_slots) for index in selected):
            raise ValueError("basis candidate index is out of range")
        keys = prior.basis_keys(self.basis_slots)
        local_order = prior.order(instruction_query, keys[list(selected)])
        return tuple(selected[index] for index in local_order)

    def freeze_basis_slot(self, basis_slot: int) -> None:
        """Protect one mastered external computation slot from later updates."""

        if not 0 <= basis_slot < len(self.basis_slots):
            raise IndexError("basis slot index is out of range")
        for parameter in self.basis_slots[basis_slot].parameters():
            parameter.requires_grad_(False)

    def remove_basis_slot(self, basis_slot: int) -> None:
        """Discard only the newest unpromoted external computation slot."""

        if basis_slot != len(self.basis_slots) - 1:
            raise ValueError("only the newest basis slot can be discarded")
        if not self.basis_slots:
            raise ValueError("no basis slots are registered")
        del self.basis_slots[basis_slot]

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
            event_window=torch.zeros(
                batch_size,
                self.event_window_size,
                self.event_width,
                device=device,
                dtype=dtype,
            ),
            event_window_mask=torch.zeros(
                batch_size, self.event_window_size, device=device, dtype=torch.bool
            ),
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
            event_width=self.event_width,
            event_window_size=self.event_window_size,
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

    def _advance_event_window(
        self,
        *,
        event: torch.Tensor,
        present: torch.Tensor,
        state: ExternalRegisterState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.event_window is None or state.event_window_mask is None:
            return (
                torch.zeros(
                    event.shape[0], self.event_window_size, self.event_width,
                    device=event.device, dtype=event.dtype,
                ),
                torch.zeros(
                    event.shape[0], self.event_window_size,
                    device=event.device, dtype=torch.bool,
                ),
            )
        if not self.event_window_size:
            return state.event_window, state.event_window_mask
        shifted = torch.cat(
            (state.event_window[:, 1:], event.unsqueeze(1)), dim=1
        )
        shifted_mask = torch.cat(
            (state.event_window_mask[:, 1:], present.unsqueeze(1)), dim=1
        )
        active = present.unsqueeze(-1)
        return (
            torch.where(active.unsqueeze(-1), shifted, state.event_window),
            torch.where(active, shifted_mask, state.event_window_mask),
        )

    def _read_state_bank(
        self,
        code: torch.Tensor,
        state_bank: torch.Tensor | None,
    ) -> torch.Tensor:
        if state_bank is None:
            return torch.zeros(
                code.shape[0],
                self.register_width,
                device=code.device,
                dtype=code.dtype,
            )
        if state_bank.ndim != 3 or state_bank.shape[0] != code.shape[0]:
            raise ValueError("state bank must have shape [batch, slots, width]")
        if state_bank.shape[2] != self.register_width or state_bank.shape[1] < 1:
            raise ValueError("state bank has incompatible register dimensions")
        if not bool(torch.isfinite(state_bank).all()):
            raise ValueError("state bank must contain only finite values")
        query = self.bank_query(code)
        keys = self.bank_key(state_bank)
        scores = torch.einsum("bd,btd->bt", query, keys)
        scores = scores / (self.instruction_width**0.5)
        weights = torch.softmax(scores, dim=-1)
        values = self.bank_value(state_bank)
        pooled = torch.einsum("bt,btd->bd", weights, values)
        return torch.sigmoid(self.bank_gate(code)) * pooled

    def bind_roles(
        self,
        register: torch.Tensor,
        instruction: ExternalRegisterInstruction,
    ) -> torch.Tensor:
        """Return the learned role-slot view of one executed register state."""

        if self.operator_mode != EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE:
            raise ValueError("role binding requires shared role-bound mode")
        code = instruction.expanded(
            register.shape[0], device=register.device, dtype=register.dtype
        )
        return self.role_binding(register, code)

    def execute(
        self,
        register: torch.Tensor,
        instruction: ExternalRegisterInstruction | None = None,
        *,
        instruction_code: torch.Tensor | None = None,
        basis_slot: int | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        state_bank: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one instruction without access to raw events or feedback."""

        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for execution")
        if (instruction is None) == (instruction_code is None):
            raise ValueError("execution requires exactly one instruction or code")
        if instruction is not None:
            if instruction.instruction_width != self.instruction_width:
                raise ValueError("instruction width does not match the machine")
        else:
            if (
                instruction_code.ndim != 2
                or instruction_code.shape != (
                    register.shape[0], self.instruction_width
                )
                or not bool(torch.isfinite(instruction_code).all())
            ):
                raise ValueError("instruction code has the wrong shape or values")
        if meta_context is not None:
            if meta_context.shape != register.shape:
                raise ValueError("meta context has the wrong shape")
            if not bool(torch.isfinite(meta_context).all()):
                raise ValueError("meta context must contain finite values")
            if self.operator_mode not in (
                "factorized_protected_meta",
                "factorized_protected_bounded_meta",
            ):
                raise ValueError("meta context requires a protected-meta mode")
        if sequence_operator_memory is not None:
            if (sequence_operator_slot is None) == (sequence_operator_route_query is None):
                raise ValueError(
                    "sequence operator memory requires exactly one slot or route query"
                )
            if self.operator_mode not in (
                "factorized_protected_meta",
                "factorized_protected_bounded_meta",
            ):
                raise ValueError("sequence operator memory requires protected-meta mode")
        elif sequence_operator_slot is not None:
            raise ValueError("sequence operator slot requires a memory object")
        elif sequence_operator_route_query is not None:
            raise ValueError("sequence operator route query requires a memory object")
        if sequence_operator_route_query is not None:
            if (
                sequence_operator_route_query.ndim != 2
                or sequence_operator_route_query.shape
                != (register.shape[0], self.instruction_width)
            ):
                raise ValueError("sequence operator route query has the wrong shape")
            if not bool(torch.isfinite(sequence_operator_route_query).all()):
                raise ValueError("sequence operator route query must be finite")
        code = (
            instruction.expanded(
                register.shape[0],
                device=register.device,
                dtype=register.dtype,
            )
            if instruction is not None
            else instruction_code.to(device=register.device, dtype=register.dtype)
        )
        if self.operator_mode == EXTERNAL_REGISTER_SHARED_BANKED_MODE:
            bank_read = self._read_state_bank(code, state_bank)
        elif state_bank is not None:
            raise ValueError("state bank addressing requires shared banked mode")
        if basis_slot is not None and self.operator_mode not in (
            "factorized_protected_meta",
            "factorized_protected_bounded_meta",
            EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
            EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
            EXTERNAL_REGISTER_SHARED_BANKED_MODE,
            EXTERNAL_REGISTER_SHARED_CANONICAL_MODE,
            EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
            EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
            EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
        ):
            if not 0 <= basis_slot < len(self.basis_slots):
                raise ValueError("basis slot index is out of range")
            if self.event_window_size:
                return self.basis_slots[basis_slot](
                    register, code, event_window, event_window_mask
                )
            return self.basis_slots[basis_slot](register, code)
        if self.operator_mode in (
            "factorized_low_rank",
            "factorized_hybrid",
            "factorized_bounded_residual",
            "factorized_protected_meta",
            "factorized_protected_bounded_meta",
            EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
            EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
            EXTERNAL_REGISTER_SHARED_BANKED_MODE,
            EXTERNAL_REGISTER_SHARED_CANONICAL_MODE,
            EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
            EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
            EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
        ):
            operator_register = (
                self.operator_normalizer(register)
                if self.operator_mode in (
                    "factorized_bounded_residual",
                    EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
                    EXTERNAL_REGISTER_SHARED_BANKED_MODE,
                    EXTERNAL_REGISTER_SHARED_CANONICAL_MODE,
                    EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
                    EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
                    "factorized_protected_bounded_meta",
                )
                else register
            )
            if self.operator_mode == EXTERNAL_REGISTER_SHARED_BANKED_MODE:
                operator_register = operator_register + bank_read
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
            projected = torch.einsum("br,bkr->bk", operator_register, right)
            base_proposal = torch.einsum("bk,brk->br", projected, left)
            operator_memory_residual = (
                sequence_operator_memory.routed_residual(
                    sequence_operator_route_query, register, code
                )
                if sequence_operator_route_query is not None
                else sequence_operator_memory.residual(
                    sequence_operator_slot, register, code
                )
                if sequence_operator_memory is not None
                else 0.0
            )
            if self.operator_mode in (
                "factorized_low_rank",
                EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
            ):
                return register + base_proposal + self.operator_bias(code)
            if self.operator_mode == "factorized_bounded_residual":
                proposal = torch.tanh(base_proposal + self.operator_bias(code))
                gate = torch.sigmoid(self.operator_composition_gate(code))
                return register + gate * proposal
            if self.operator_mode == EXTERNAL_REGISTER_SHARED_CANONICAL_MODE:
                proposal = torch.tanh(base_proposal + self.operator_bias(code))
                gate = torch.sigmoid(self.operator_composition_gate(code))
                return self.state_contract(register + gate * proposal)
            if self.operator_mode in (
                EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
                EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
            ):
                proposal = torch.tanh(base_proposal + self.operator_bias(code))
                gate = torch.sigmoid(self.operator_composition_gate(code))
                return register + gate * proposal
            if self.operator_mode in (
                EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
                EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
            ):
                relational_proposal = self.relational_transition(
                    operator_register, code
                )
                proposal = torch.tanh(
                    base_proposal + self.operator_bias(code) + relational_proposal
                )
                gate = torch.sigmoid(self.operator_composition_gate(code))
                return register + gate * proposal
            if self.operator_mode == EXTERNAL_REGISTER_SHARED_BANKED_MODE:
                proposal = torch.tanh(base_proposal + self.operator_bias(code))
                gate = torch.sigmoid(self.operator_composition_gate(code))
                return register + gate * proposal
            if self.operator_mode in (
                "factorized_protected_meta",
                "factorized_protected_bounded_meta",
            ):
                meta_left = torch.tanh(self.operator_meta_left(code)).reshape(
                    register.shape[0], self.register_width, self.operator_rank
                )
                meta_right = torch.tanh(self.operator_meta_right(code)).reshape(
                    register.shape[0], self.operator_rank, self.register_width
                )
                meta_register = (
                    self.operator_normalizer(register)
                    if self.operator_mode == "factorized_protected_bounded_meta"
                    else register
                )
                meta_projected = torch.einsum(
                    "br,bkr->bk", meta_register, meta_right
                )
                meta_proposal = torch.einsum(
                    "bk,brk->br", meta_projected, meta_left
                ) + self.operator_meta_bias(code)
                meta_gate = torch.sigmoid(self.operator_meta_gate(code))
                if self.operator_mode == "factorized_protected_bounded_meta":
                    base = torch.tanh(base_proposal + self.operator_bias(code))
                    base_gate = torch.sigmoid(self.operator_composition_gate(code))
                    meta_residual = 0.5 * meta_gate * torch.tanh(meta_proposal)
                    memory_residual = (
                        0.5 * torch.tanh(meta_context)
                        if meta_context is not None
                        else 0.0
                    )
                    return register + base_gate * (
                        base + meta_residual + memory_residual + operator_memory_residual
                    )
                residual = 0.5 * meta_gate * torch.tanh(meta_proposal)
                if meta_context is not None:
                    residual = residual + 0.5 * torch.tanh(meta_context)
                return (
                    register + base_proposal + self.operator_bias(code) + residual
                    + operator_memory_residual
                )
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

    def execute_code_chain(
        self,
        register: torch.Tensor,
        program_codes: torch.Tensor,
        *,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute opaque program codes through the shared interpreter."""

        if (
            program_codes.ndim != 3
            or program_codes.shape[0] != register.shape[0]
            or program_codes.shape[1] < 1
            or program_codes.shape[2] != self.instruction_width
        ):
            raise ValueError("program codes have the wrong shape for execution")
        for code in program_codes.transpose(0, 1):
            register = self.execute(
                register,
                instruction_code=code,
                event_window=event_window,
                event_window_mask=event_window_mask,
                meta_context=meta_context,
                sequence_operator_memory=sequence_operator_memory,
                sequence_operator_slot=sequence_operator_slot,
                sequence_operator_route_query=sequence_operator_route_query,
            )
        return register

    def execute_chain(
        self,
        register: torch.Tensor,
        instructions: Iterable[ExternalRegisterInstruction],
        *,
        basis_slots: Iterable[int | None] | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a memory-selected instruction chain to a working register."""
        register, _ = self.execute_chain_trace(
            register,
            instructions,
            basis_slots=basis_slots,
            event_window=event_window,
            event_window_mask=event_window_mask,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        return register

    def execute_chain_trace(
        self,
        register: torch.Tensor,
        instructions: Iterable[ExternalRegisterInstruction],
        *,
        basis_slots: Iterable[int | None] | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Execute a chain while preserving each intermediate register state.

        The trace is an opaque positional state bank.  It preserves binding
        between an instruction's output and the later decoder without
        assigning semantic meaning to register coordinates.
        """
        selected = tuple(instructions)
        selected_basis = (
            (None,) * len(selected)
            if basis_slots is None
            else tuple(basis_slots)
        )
        if len(selected_basis) != len(selected):
            raise ValueError("basis slot bindings must match instruction count")
        states: list[torch.Tensor] = []
        for instruction, basis_slot in zip(selected, selected_basis):
            register = self.execute(
                register,
                instruction,
                basis_slot=basis_slot,
                event_window=event_window,
                event_window_mask=event_window_mask,
                meta_context=meta_context,
                sequence_operator_memory=sequence_operator_memory,
                sequence_operator_slot=sequence_operator_slot,
                sequence_operator_route_query=sequence_operator_route_query,
                state_bank=(
                    torch.stack(states, dim=1)
                    if states and self.operator_mode == EXTERNAL_REGISTER_SHARED_BANKED_MODE
                    else None
                ),
            )
            states.append(register)
        return register, tuple(states)

    def execute_chain_role_trace(
        self,
        register: torch.Tensor,
        instructions: Iterable[ExternalRegisterInstruction],
        *,
        basis_slots: Iterable[int | None] | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Execute a chain and return one learned role bank per instruction."""

        if self.operator_mode != EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE:
            raise ValueError("role traces require shared role-bound mode")
        selected = tuple(instructions)
        final, states = self.execute_chain_trace(
            register,
            selected,
            basis_slots=basis_slots,
            event_window=event_window,
            event_window_mask=event_window_mask,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        role_trace = tuple(
            self.bind_roles(state, instruction)
            for state, instruction in zip(states, selected, strict=True)
        )
        return final, role_trace

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
        event_window, event_window_mask = self._advance_event_window(
            event=event, present=present, state=state
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
            event_window=event_window,
            event_window_mask=event_window_mask,
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
        basis_slots: Iterable[int | None] | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
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
        executed = self.execute_chain(
            register,
            selected,
            basis_slots=basis_slots,
            event_window=next_state.event_window,
            event_window_mask=next_state.event_window_mask,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        return torch.where(
            present.unsqueeze(-1), executed, register
        ), next_state

    def read_execute_register_trace(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor | None = None,
        instructions: Iterable[ExternalRegisterInstruction] | None = None,
        basis_slots: Iterable[int | None] | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState, tuple[torch.Tensor, ...]]:
        """Read context, execute, and return an opaque intermediate-state bank."""
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
        executed, trace = self.execute_chain_trace(
            register,
            selected,
            basis_slots=basis_slots,
            event_window=next_state.event_window,
            event_window_mask=next_state.event_window_mask,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        trace = tuple(
            torch.where(present.unsqueeze(-1), value, register)
            for value in trace
        )
        return torch.where(
            present.unsqueeze(-1), executed, register
        ), next_state, trace

    def read_execute_register_role_trace(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        present: torch.Tensor | None = None,
        instructions: Iterable[ExternalRegisterInstruction] | None = None,
        basis_slots: Iterable[int | None] | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState, tuple[torch.Tensor, ...]]:
        """Observe, execute, and expose the learned role bank per step."""

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
        executed, role_trace = self.execute_chain_role_trace(
            register,
            selected,
            basis_slots=basis_slots,
            event_window=next_state.event_window,
            event_window_mask=next_state.event_window_mask,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        role_trace = tuple(
            torch.where(
                present.unsqueeze(-1).unsqueeze(-1),
                value,
                self.bind_roles(register, instruction),
            )
            for value, instruction in zip(role_trace, selected, strict=True)
        )
        return torch.where(
            present.unsqueeze(-1), executed, register
        ), next_state, role_trace

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
        basis_slots: Iterable[int | None] | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
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
        executed = self.execute_chain(
            register,
            selected,
            basis_slots=basis_slots,
            event_window=observed_state.event_window,
            event_window_mask=observed_state.event_window_mask,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        register = torch.where(
            present.unsqueeze(-1), executed, observed_state.register
        )
        return register, ExternalRegisterState(
            register=register,
            context=observed_state.context,
            initialized=observed_state.initialized,
            event_window=observed_state.event_window,
            event_window_mask=observed_state.event_window_mask,
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
        basis_slots: Iterable[int | None] | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
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
            basis_slots=basis_slots,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        return self.to_intention(register), next_state
