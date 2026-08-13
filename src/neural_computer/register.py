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

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .interface import IntentEvent
from .program import (
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
    ExternalProgramMemoryTransactionReceipt,
    evaluate_external_program_admission,
)

if TYPE_CHECKING:
    from .maintenance import (
        ExternalMemoryMaintenancePolicy,
        ExternalMemoryMaintenanceProposal,
    )

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
EXTERNAL_REGISTER_STATE_SCHEMA = "neural-computer.external-register-state.v1"
EXTERNAL_REGISTER_EXECUTION_SNAPSHOT_SCHEMA = (
    "neural-computer.external-register-execution-snapshot.v1"
)
EXTERNAL_REGISTER_BASIS_SCHEMA = "neural-computer.external-register-compute-basis.v1"
EXTERNAL_REGISTER_BASIS_ARTIFACT_SCHEMA = (
    "neural-computer.external-register-compute-basis-artifact.v1"
)
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
EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE = "factorized_shared_stable_relational"
EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE = "factorized_shared_operator_basis"
EXTERNAL_SEQUENCE_MEMORY_SCHEMA = "neural-computer.external-sequence-memory.v1"
LEGACY_EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA = (
    "neural-computer.external-sequence-program-memory.v1"
)
EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA = (
    "neural-computer.external-sequence-program-memory.v2"
)
EXTERNAL_SEQUENCE_PROGRAM_MEMORY_COMPRESSED_SCHEMA = (
    "neural-computer.external-sequence-program-memory-compressed.v1"
)
EXTERNAL_SEQUENCE_OPERATOR_BINDING_SCHEMA = (
    "neural-computer.external-sequence-operator-binding.v1"
)
LEGACY_EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA = (
    "neural-computer.external-sequence-operator-memory.v1"
)
EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA = (
    "neural-computer.external-sequence-operator-memory.v2"
)


def _digest_mapping(value: object) -> str:
    """Checksum nested tensor metadata without retaining verifier rows."""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


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
        self.state_tokens = nn.Linear(register_width, role_count * self.role_width)
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
        scores = torch.einsum("brd,btd->brt", queries, self.key(tokens)).div(
            self.role_width**0.5
        )
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
        scores = torch.einsum("brd,btd->brt", queries, self.key(roles)).div(
            self.role_width**0.5
        )
        weights = torch.softmax(scores, dim=-1)
        mixed = roles + torch.einsum("brt,btd->brd", weights, self.value(roles))
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

    def __init__(
        self, register_width: int, output_width: int, *, hidden: int = 64
    ) -> None:
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
            raise ValueError(
                "external register initialization mask has the wrong shape"
            )
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
            if self.event_window.shape != (batch_size, event_window_size, event_width):
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

    def payload(self) -> dict[str, torch.Tensor | str | None]:
        """Return a detached tensor-only payload for durable external state."""

        return {
            "schema": EXTERNAL_REGISTER_STATE_SCHEMA,
            "register": self.register.detach().cpu().clone(),
            "context": self.context.detach().cpu().clone(),
            "initialized": self.initialized.detach().cpu().clone(),
            "event_window": (
                self.event_window.detach().cpu().clone()
                if self.event_window is not None
                else None
            ),
            "event_window_mask": (
                self.event_window_mask.detach().cpu().clone()
                if self.event_window_mask is not None
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ExternalRegisterState:
        """Restore one state payload; dimensional validation remains runtime-owned."""

        if not isinstance(payload, dict):
            raise TypeError("external register state payload must be a dictionary")
        if payload.get("schema") != EXTERNAL_REGISTER_STATE_SCHEMA:
            raise ValueError("unsupported external register state schema")
        required = ("register", "context", "initialized")
        if any(not isinstance(payload.get(name), torch.Tensor) for name in required):
            raise TypeError("external register state payload is missing tensors")
        event_window = payload.get("event_window")
        event_window_mask = payload.get("event_window_mask")
        if (event_window is None) != (event_window_mask is None):
            raise ValueError("external register event window payload is incomplete")
        if event_window is not None and not isinstance(event_window, torch.Tensor):
            raise TypeError("external register event window must be a tensor or null")
        if event_window_mask is not None and not isinstance(
            event_window_mask, torch.Tensor
        ):
            raise TypeError(
                "external register event window mask must be a tensor or null"
            )
        return cls(
            register=payload["register"],
            context=payload["context"],
            initialized=payload["initialized"],
            event_window=event_window,
            event_window_mask=event_window_mask,
        )


@dataclass(frozen=True)
class ExternalExecutionSnapshot:
    """Typed boundary between durable observation state and transient execution."""

    observed: ExternalRegisterState
    executed: torch.Tensor
    trace: tuple[torch.Tensor, ...] = ()
    program_digest: str | None = None

    def validate(
        self,
        *,
        batch_size: int,
        register_width: int,
        context_width: int,
        event_width: int | None = None,
        event_window_size: int = 0,
        program_length: int | None = None,
    ) -> ExternalExecutionSnapshot:
        self.observed.validate(
            batch_size=batch_size,
            register_width=register_width,
            context_width=context_width,
            event_width=event_width,
            event_window_size=event_window_size,
        )
        if self.executed.shape != (batch_size, register_width):
            raise ValueError("execution snapshot output has the wrong shape")
        if not bool(torch.isfinite(self.executed).all()):
            raise ValueError("execution snapshot output must be finite")
        if program_length is not None and len(self.trace) != program_length:
            raise ValueError("execution snapshot trace length does not match program")
        for state in self.trace:
            if state.shape != (batch_size, register_width):
                raise ValueError("execution snapshot trace has the wrong shape")
            if not bool(torch.isfinite(state).all()):
                raise ValueError("execution snapshot trace must be finite")
        if self.program_digest is not None:
            if len(self.program_digest) != 64:
                raise ValueError("execution snapshot program digest is malformed")
            try:
                int(self.program_digest, 16)
            except ValueError as error:
                raise ValueError(
                    "execution snapshot program digest is malformed"
                ) from error
        return self

    def payload(self) -> dict[str, object]:
        """Return an opaque payload suitable for transient checkpointing."""

        return {
            "schema": EXTERNAL_REGISTER_EXECUTION_SNAPSHOT_SCHEMA,
            "observed": self.observed.payload(),
            "executed": self.executed.detach().cpu().clone(),
            "trace": tuple(value.detach().cpu().clone() for value in self.trace),
            "program_digest": self.program_digest,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ExternalExecutionSnapshot:
        """Restore a snapshot without interpreting its learned state."""

        if not isinstance(payload, dict):
            raise TypeError("execution snapshot payload must be a dictionary")
        if payload.get("schema") != EXTERNAL_REGISTER_EXECUTION_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported external execution snapshot schema")
        observed = payload.get("observed")
        executed = payload.get("executed")
        trace = payload.get("trace", ())
        if not isinstance(observed, dict) or not isinstance(executed, torch.Tensor):
            raise TypeError("execution snapshot payload is missing tensors")
        if not isinstance(trace, (tuple, list)) or not all(
            isinstance(value, torch.Tensor) for value in trace
        ):
            raise TypeError("execution snapshot trace must be a tensor sequence")
        return cls(
            observed=ExternalRegisterState.from_payload(observed),
            executed=executed,
            trace=tuple(trace),
            program_digest=payload.get("program_digest"),
        )


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
        return self.slots[slot].to(device=device, dtype=dtype).expand(batch_size, -1)


class ExternalSequenceProgramMemory(nn.Module):
    """Versioned opaque program files executed by one shared interpreter.

    Admission grows the bank, while copy-on-write lifecycle transactions can
    safely retire equivalent or unneeded files and compress durable storage.
    The controller sees only the selected executable tensor; logical file IDs
    remain entirely on the external-memory side of the boundary.
    """

    schema = EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA

    def __init__(
        self,
        instruction_width: int,
        *,
        router_hidden: int = 32,
        router_temperature: float = 1.0,
        hard_routing: bool = False,
        content_addressing: bool = False,
    ) -> None:
        super().__init__()
        if min(instruction_width, router_hidden) < 1:
            raise ValueError("sequence program memory dimensions must be positive")
        if router_temperature <= 0.0:
            raise ValueError("sequence program router temperature must be positive")
        self.instruction_width = int(instruction_width)
        self.router_hidden = int(router_hidden)
        self.router_temperature = float(router_temperature)
        self.hard_routing = bool(hard_routing)
        self.content_addressing = bool(content_addressing)
        self.programs = nn.ParameterList()
        self.address_programs = nn.ParameterList()
        self.slot_keys = nn.ParameterList()
        self._protected_slots: list[bool] = []
        self._output_schemas: list[str | None] = []
        self._logical_slot_ids: list[int] = []
        self._next_logical_slot_id = 0
        self.query_encoder = nn.GRU(
            self.instruction_width, self.router_hidden, batch_first=True
        )
        self.program_query = nn.Linear(self.router_hidden, self.instruction_width)
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
        if self.content_addressing:
            # Content addressing is a data-structure operation, not a second
            # learned reasoning branch. Keep its shared code embedding fixed;
            # only the stored executable program data may adapt.
            for module in (
                self.query_encoder,
                self.program_query,
                self.route_query_encoder,
                self.key_encoder,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "instruction_width": self.instruction_width,
            "router_hidden": self.router_hidden,
            "router_temperature": self.router_temperature,
            "hard_routing": self.hard_routing,
            "content_addressing": self.content_addressing,
            "slot_count": len(self.programs),
            "program_lengths": [int(program.shape[0]) for program in self.programs],
            "output_schemas": list(self._output_schemas),
            "protected_slots": list(self._protected_slots),
            "logical_slot_ids": list(self._logical_slot_ids),
            "next_logical_slot_id": self._next_logical_slot_id,
            "addressing": (
                "immutable_opaque_program_content_v1"
                if self.content_addressing
                else "learned_slot_keys_v1"
            ),
            "storage": "copy_on_write_opaque_instruction_files_v2",
            "computation": "shared_register_interpreter_v1",
            "artifact_schema": "neural-computer.external-program-artifact.v1",
        }

    def add_artifact(self, artifact: ExternalProgramArtifact) -> int:
        """Admit one portable opaque program after validating its ABI."""

        if not isinstance(artifact, ExternalProgramArtifact):
            raise TypeError(
                "sequence program memory requires an external program artifact"
            )
        artifact.validate_for(
            instruction_width=self.instruction_width,
            interpreter_schema="neural-computer.external-register.v4",
            execution_schema="neural-computer.external-register-read-execute.v1",
        )
        slot = self.add_program(artifact.codes)
        self._output_schemas[slot] = artifact.output_schema
        return slot

    def artifact(
        self,
        slot: int,
        *,
        output_schema: str | None = None,
    ) -> ExternalProgramArtifact:
        """Snapshot one stored program as a portable external artifact."""

        if not 0 <= slot < len(self.programs):
            raise ValueError("sequence program memory slot index is out of range")
        resolved_output_schema = (
            self._output_schemas[slot] if output_schema is None else output_schema
        )
        return ExternalProgramArtifact(
            codes=self.programs[slot].detach(),
            interpreter_schema="neural-computer.external-register.v4",
            execution_schema="neural-computer.external-register-read-execute.v1",
            output_schema=resolved_output_schema,
        )

    def add_program(self, codes: torch.Tensor) -> int:
        if (
            codes.ndim != 2
            or codes.shape[0] < 1
            or codes.shape[1] != self.instruction_width
        ):
            raise ValueError("program codes must have shape [steps, instruction_width]")
        if not bool(torch.isfinite(codes).all()):
            raise ValueError("program codes must be finite")
        detached_codes = codes.detach().clone()
        self.programs.append(nn.Parameter(detached_codes.clone()))
        self.address_programs.append(nn.Parameter(detached_codes, requires_grad=False))
        key = nn.Parameter(torch.empty(self.instruction_width))
        nn.init.normal_(key, mean=0.0, std=0.02)
        self.slot_keys.append(key)
        self._protected_slots.append(False)
        self._output_schemas.append(None)
        self._logical_slot_ids.append(self._next_logical_slot_id)
        self._next_logical_slot_id += 1
        return len(self.programs) - 1

    @property
    def file_count(self) -> int:
        """Return the number of executable files in the external bank."""

        return len(self.programs)

    @property
    def logical_slot_ids(self) -> tuple[int, ...]:
        """Return stable opaque IDs, independent of physical file positions."""

        return tuple(self._logical_slot_ids)

    def logical_slot_id(self, slot: int) -> int:
        """Resolve one current physical position to its stable logical ID."""

        if not 0 <= slot < self.file_count:
            raise ValueError("sequence program memory file index is out of range")
        return self._logical_slot_ids[slot]

    def physical_index_for_logical_id(self, slot_id: int) -> int:
        """Resolve a stable file ID without exposing it to the controller."""

        if slot_id < 0:
            raise ValueError("sequence program memory logical ID cannot be negative")
        try:
            return self._logical_slot_ids.index(int(slot_id))
        except ValueError as error:
            raise KeyError(
                "sequence program memory logical ID is not retained"
            ) from error

    def protection_mask(self) -> torch.Tensor:
        """Return memory-side protection without exposing file semantics."""

        return torch.tensor(tuple(self._protected_slots), dtype=torch.bool)

    def protect_file(self, slot: int) -> None:
        """Make one admitted file ineligible for replacement or eviction."""

        if not 0 <= slot < self.file_count:
            raise ValueError("sequence program memory file index is out of range")
        self._protected_slots[slot] = True

    def is_file_protected(self, slot: int) -> bool:
        """Return whether a file has passed the external retention gate."""

        if not 0 <= slot < self.file_count:
            raise ValueError("sequence program memory file index is out of range")
        return self._protected_slots[slot]

    def admit_verified_artifact(
        self,
        artifact: ExternalProgramArtifact,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        protect: bool = False,
    ) -> ExternalProgramAdmissionReceipt:
        """Stage and atomically admit one file after scalar verification.

        ``artifact`` is never visible to the controller.  A failed candidate
        leaves the module parameters and the file count untouched.  The
        verifier outcomes are sufficient statistics for admission; raw rows
        are not replayed or retained here.
        """

        artifact.validate_for(
            instruction_width=self.instruction_width,
            interpreter_schema="neural-computer.external-register.v4",
            execution_schema="neural-computer.external-register-read-execute.v1",
        )
        receipt = evaluate_external_program_admission(
            artifact,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if not receipt.accepted:
            return receipt
        slot = self.add_artifact(artifact)
        if protect:
            self.protect_file(slot)
        return ExternalProgramAdmissionReceipt(
            accepted=True,
            candidate_digest=receipt.candidate_digest,
            slot=slot,
            observations=receipt.observations,
            stable_bits_to_threshold=receipt.stable_bits_to_threshold,
            stable_prefix_minimum=receipt.stable_prefix_minimum,
            reason="candidate verified and committed as an external file",
        ).validate()

    @staticmethod
    def _maintenance_scalar(value: float, *, name: str) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError(f"executable-memory maintenance {name} must lie in [0, 1]")
        return resolved

    def maintenance_features(
        self,
        *,
        capacity_limit: int | None = None,
        mean_usage: float = 0.0,
        mean_age: float = 0.0,
        mean_prediction_error: float = 0.0,
        max_prediction_error: float = 0.0,
        binding_pressure: float = 0.0,
        provisional_pressure: float = 0.0,
        redundancy_pressure: float = 0.0,
        compression_opportunity: float = 0.0,
    ) -> torch.Tensor:
        """Return generic storage telemetry for the maintenance policy.

        The executable bank owns file count and protection, while callers may
        provide normalized usage, age, verifier-error, binding, redundancy,
        and compression telemetry from their external ledger.  No task,
        modality, protocol, or file meaning is assigned to a feature.
        """

        if capacity_limit is not None and capacity_limit < 1:
            raise ValueError("executable-memory maintenance capacity must be positive")
        denominator = max(
            1, capacity_limit if capacity_limit is not None else self.file_count
        )
        logical_fraction = min(self.file_count / float(denominator), 1.0)
        physical_fraction = logical_fraction
        capacity_pressure = float(
            capacity_limit is not None and self.file_count >= capacity_limit
        )
        return torch.tensor(
            [
                capacity_pressure,
                logical_fraction,
                physical_fraction,
                0.0,
                self._maintenance_scalar(mean_usage, name="mean_usage"),
                self._maintenance_scalar(mean_age, name="mean_age"),
                self._maintenance_scalar(
                    mean_prediction_error,
                    name="mean_prediction_error",
                ),
                self._maintenance_scalar(
                    max_prediction_error,
                    name="max_prediction_error",
                ),
                self._maintenance_scalar(binding_pressure, name="binding_pressure"),
                self._maintenance_scalar(
                    provisional_pressure,
                    name="provisional_pressure",
                ),
                self._maintenance_scalar(
                    redundancy_pressure,
                    name="redundancy_pressure",
                ),
                self._maintenance_scalar(
                    compression_opportunity,
                    name="compression_opportunity",
                ),
            ],
            dtype=torch.float32,
        )

    def propose_maintenance(
        self,
        policy: ExternalMemoryMaintenancePolicy,
        *,
        capacity_limit: int | None = None,
        growth_available: bool = False,
        share_available: bool = False,
        compression_available: bool = False,
        evict_available: bool = False,
        mean_usage: float = 0.0,
        mean_age: float = 0.0,
        mean_prediction_error: float = 0.0,
        max_prediction_error: float = 0.0,
        binding_pressure: float = 0.0,
        provisional_pressure: float = 0.0,
        redundancy_pressure: float = 0.0,
        compression_opportunity: float = 0.0,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> ExternalMemoryMaintenanceProposal:
        """Select one legal executable-memory operation without mutating files."""

        from .maintenance import ExternalMemoryMaintenancePolicy

        if not isinstance(policy, ExternalMemoryMaintenancePolicy):
            raise TypeError("executable-memory maintenance policy is invalid")
        availability = (
            growth_available,
            share_available,
            compression_available,
            evict_available,
            True,
        )
        if not all(isinstance(value, bool) for value in availability):
            raise TypeError("executable-memory maintenance availability must be bool")
        features = self.maintenance_features(
            capacity_limit=capacity_limit,
            mean_usage=mean_usage,
            mean_age=mean_age,
            mean_prediction_error=mean_prediction_error,
            max_prediction_error=max_prediction_error,
            binding_pressure=binding_pressure,
            provisional_pressure=provisional_pressure,
            redundancy_pressure=redundancy_pressure,
            compression_opportunity=compression_opportunity,
        )
        return policy.propose(
            features,
            torch.tensor(availability, dtype=torch.bool),
            sample=sample,
            generator=generator,
        )

    def apply_maintenance_proposal(
        self,
        proposal: ExternalMemoryMaintenanceProposal,
        *,
        retention_probe: Callable[[ExternalSequenceProgramMemory], bool] | None = None,
        share_pair: tuple[int, int] | None = None,
        equivalence_probe: Callable[
            [ExternalProgramArtifact, ExternalProgramArtifact], bool
        ]
        | None = None,
        evict_slot_id: int | None = None,
        growth_artifact: ExternalProgramArtifact | None = None,
        growth_outcomes: torch.Tensor | Sequence[float] | None = None,
        growth_threshold: float = 0.8,
        growth_min_observations: int = 1,
        growth_min_stable_observations: int = 1,
        compression_dtype: torch.dtype | str = torch.float16,
        protect_growth: bool = False,
    ) -> (
        ExternalProgramAdmissionReceipt | ExternalProgramMemoryTransactionReceipt | None
    ):
        """Execute one proposal through verifier-gated file transactions.

        The policy chooses only the generic operation.  Candidate artifacts,
        opaque logical IDs, equivalence probes, and retention probes stay on
        the external-memory side and remain authoritative for commitment.
        ``defer`` is an explicit no-op; ``grow`` uses the normal admission
        transaction and the other actions use the lifecycle transactions.
        """

        from .maintenance import ExternalMemoryMaintenanceProposal

        if not isinstance(proposal, ExternalMemoryMaintenanceProposal):
            raise TypeError("executable-memory maintenance proposal is invalid")
        proposal.validate()
        if proposal.action == "defer":
            return None
        if proposal.action == "grow":
            if growth_artifact is None or growth_outcomes is None:
                raise ValueError(
                    "executable-memory growth needs an artifact and outcomes"
                )
            return self.admit_verified_artifact(
                growth_artifact,
                growth_outcomes,
                threshold=growth_threshold,
                min_observations=growth_min_observations,
                min_stable_observations=growth_min_stable_observations,
                protect=protect_growth,
            )
        if retention_probe is None:
            raise ValueError("executable-memory maintenance needs a retention probe")
        if proposal.action == "share":
            if share_pair is None or equivalence_probe is None:
                raise ValueError(
                    "executable-memory sharing needs an equivalent pair and probe"
                )
            return self.consolidate_verified(
                share_pair[0],
                share_pair[1],
                equivalence_probe,
                retention_probe,
            )
        if proposal.action == "compress":
            return self.compress_verified(
                dtype=compression_dtype,
                retention_probe=retention_probe,
            )
        if proposal.action == "evict":
            if evict_slot_id is None:
                raise ValueError("executable-memory eviction needs a logical slot ID")
            return self.evict_verified(evict_slot_id, retention_probe)
        raise ValueError(
            f"unsupported executable-memory maintenance action: {proposal.action}"
        )

    def _state_storage_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size() for value in self.state_dict().values()
        )

    @staticmethod
    def _tensor_mapping_storage_bytes(mapping: Mapping[str, torch.Tensor]) -> int:
        return sum(value.numel() * value.element_size() for value in mapping.values())

    @staticmethod
    def _copy_shared_module_state(
        source: ExternalSequenceProgramMemory,
        target: ExternalSequenceProgramMemory,
    ) -> None:
        for name in (
            "query_encoder",
            "program_query",
            "route_query_encoder",
            "key_encoder",
        ):
            source_module = getattr(source, name)
            target_module = getattr(target, name)
            target_module.load_state_dict(source_module.state_dict(), strict=True)
            source_parameters = dict(source_module.named_parameters())
            for parameter_name, target_parameter in target_module.named_parameters():
                target_parameter.requires_grad_(
                    source_parameters[parameter_name].requires_grad
                )

    def _copy_selected_files(
        self,
        slots: Sequence[int],
    ) -> ExternalSequenceProgramMemory:
        selected = tuple(int(slot) for slot in slots)
        if not selected:
            raise ValueError("external program memory candidate cannot be empty")
        if any(not 0 <= slot < self.file_count for slot in selected):
            raise IndexError("external program memory candidate slot is out of range")
        if len(set(selected)) != len(selected):
            raise ValueError("external program memory candidate slots must be unique")
        candidate = ExternalSequenceProgramMemory(
            self.instruction_width,
            router_hidden=self.router_hidden,
            router_temperature=self.router_temperature,
            hard_routing=self.hard_routing,
            content_addressing=self.content_addressing,
        )
        for slot in selected:
            new_slot = candidate.add_program(self.programs[slot].detach())
            candidate._output_schemas[new_slot] = self._output_schemas[slot]
            candidate._protected_slots[new_slot] = self._protected_slots[slot]
            candidate._logical_slot_ids[new_slot] = self._logical_slot_ids[slot]
            with torch.no_grad():
                candidate.programs[new_slot].copy_(self.programs[slot])
                candidate.address_programs[new_slot].copy_(self.address_programs[slot])
                candidate.slot_keys[new_slot].copy_(self.slot_keys[slot])
            candidate.programs[new_slot].requires_grad_(
                self.programs[slot].requires_grad
            )
            candidate.slot_keys[new_slot].requires_grad_(
                self.slot_keys[slot].requires_grad
            )
        candidate._next_logical_slot_id = self._next_logical_slot_id
        self._copy_shared_module_state(self, candidate)
        return candidate

    def _commit_candidate(self, candidate: ExternalSequenceProgramMemory) -> None:
        if not isinstance(candidate, ExternalSequenceProgramMemory):
            raise TypeError("external program memory candidate has the wrong type")
        if candidate.instruction_width != self.instruction_width:
            raise ValueError("external program memory candidate width is incompatible")
        if candidate.file_count < 1:
            raise ValueError("external program memory candidate cannot be empty")
        self.programs = candidate.programs
        self.address_programs = candidate.address_programs
        self.slot_keys = candidate.slot_keys
        self._protected_slots = list(candidate._protected_slots)
        self._output_schemas = list(candidate._output_schemas)
        self._logical_slot_ids = list(candidate._logical_slot_ids)
        self._next_logical_slot_id = candidate._next_logical_slot_id
        self._copy_shared_module_state(candidate, self)

    @staticmethod
    def _transaction_receipt(
        *,
        accepted: bool,
        operation: str,
        affected_slot_id: int | None,
        source_file_count: int,
        destination_file_count: int,
        source_digest: str,
        candidate_digest: str,
        source_storage_bytes: int,
        candidate_storage_bytes: int,
        reason: str,
    ) -> ExternalProgramMemoryTransactionReceipt:
        return ExternalProgramMemoryTransactionReceipt(
            accepted=accepted,
            operation=operation,
            affected_slot_id=affected_slot_id,
            source_file_count=source_file_count,
            destination_file_count=destination_file_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            source_storage_bytes=source_storage_bytes,
            candidate_storage_bytes=candidate_storage_bytes,
            reason=reason,
        ).validate()

    def _run_lifecycle_probe(
        self,
        candidate: ExternalSequenceProgramMemory,
        retention_probe: Callable[[ExternalSequenceProgramMemory], bool],
    ) -> tuple[bool, bool]:
        if not callable(retention_probe):
            raise TypeError("external program memory retention probe is invalid")
        before = candidate.digest()
        accepted = bool(retention_probe(candidate))
        return accepted, candidate.digest() == before

    def evict_verified(
        self,
        slot_id: int,
        retention_probe: Callable[[ExternalSequenceProgramMemory], bool],
    ) -> ExternalProgramMemoryTransactionReceipt:
        """Remove one unprotected logical file after a held-out proof.

        Physical indices may change, but surviving logical IDs remain stable.
        The live bank is changed only after the copy-on-write candidate passes
        the caller-owned retention probe and the probe itself is non-mutating.
        """

        index = self.physical_index_for_logical_id(slot_id)
        source_digest = self.digest()
        source_count = self.file_count
        source_bytes = self._state_storage_bytes()
        if self._protected_slots[index]:
            return self._transaction_receipt(
                accepted=False,
                operation="evict",
                affected_slot_id=slot_id,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason="protected executable file cannot be evicted",
            )
        if source_count <= 1:
            return self._transaction_receipt(
                accepted=False,
                operation="evict",
                affected_slot_id=slot_id,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason="external program memory must retain one file",
            )
        candidate = self._copy_selected_files(
            [slot for slot in range(source_count) if slot != index]
        )
        accepted, probe_unchanged = self._run_lifecycle_probe(
            candidate,
            retention_probe,
        )
        candidate_digest = candidate.digest()
        if not accepted or not probe_unchanged:
            return self._transaction_receipt(
                accepted=False,
                operation="evict",
                affected_slot_id=slot_id,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason=(
                    "retention probe mutated the candidate"
                    if not probe_unchanged
                    else "post-eviction retention probe failed"
                ),
            )
        candidate_bytes = candidate._state_storage_bytes()
        self._commit_candidate(candidate)
        return self._transaction_receipt(
            accepted=True,
            operation="evict",
            affected_slot_id=slot_id,
            source_file_count=source_count,
            destination_file_count=self.file_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            source_storage_bytes=source_bytes,
            candidate_storage_bytes=candidate_bytes,
            reason="retention-verified logical executable file eviction committed",
        )

    def consolidate_verified(
        self,
        survivor_slot_id: int,
        duplicate_slot_id: int,
        equivalence_probe: Callable[
            [ExternalProgramArtifact, ExternalProgramArtifact], bool
        ],
        retention_probe: Callable[[ExternalSequenceProgramMemory], bool],
    ) -> ExternalProgramMemoryTransactionReceipt:
        """Drop an equivalent unprotected file without changing its survivor.

        Equivalence is verified by a caller-owned held-out execution probe;
        this store never interprets the program or the verifier. The survivor
        keeps its logical ID and physical parameters, while the duplicate's
        logical ID is retired only after retention passes.
        """

        survivor_index = self.physical_index_for_logical_id(survivor_slot_id)
        duplicate_index = self.physical_index_for_logical_id(duplicate_slot_id)
        if survivor_index == duplicate_index:
            raise ValueError("executable consolidation needs two distinct files")
        source_digest = self.digest()
        source_count = self.file_count
        source_bytes = self._state_storage_bytes()
        if self._protected_slots[duplicate_index]:
            return self._transaction_receipt(
                accepted=False,
                operation="consolidate",
                affected_slot_id=duplicate_slot_id,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason="protected duplicate executable file cannot be consolidated",
            )
        if not callable(equivalence_probe):
            raise TypeError("executable equivalence probe is invalid")
        if not bool(
            equivalence_probe(
                self.artifact(survivor_index),
                self.artifact(duplicate_index),
            )
        ):
            return self._transaction_receipt(
                accepted=False,
                operation="consolidate",
                affected_slot_id=duplicate_slot_id,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason="held-out executable functions are not equivalent",
            )
        candidate = self._copy_selected_files(
            [slot for slot in range(source_count) if slot != duplicate_index]
        )
        accepted, probe_unchanged = self._run_lifecycle_probe(
            candidate,
            retention_probe,
        )
        candidate_digest = candidate.digest()
        if not accepted or not probe_unchanged:
            return self._transaction_receipt(
                accepted=False,
                operation="consolidate",
                affected_slot_id=duplicate_slot_id,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason=(
                    "retention probe mutated the candidate"
                    if not probe_unchanged
                    else "post-consolidation retention probe failed"
                ),
            )
        candidate_bytes = candidate._state_storage_bytes()
        self._commit_candidate(candidate)
        return self._transaction_receipt(
            accepted=True,
            operation="consolidate",
            affected_slot_id=duplicate_slot_id,
            source_file_count=source_count,
            destination_file_count=self.file_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            source_storage_bytes=source_bytes,
            candidate_storage_bytes=candidate_bytes,
            reason="held-out-equivalent executable files were consolidated",
        )

    def compressed_payload(
        self,
        *,
        dtype: torch.dtype | str = torch.float16,
    ) -> dict[str, object]:
        """Create a smaller durable representation of the complete file bank."""

        from .growth import compress_growth_artifact

        source = self.payload()
        state = source["state"]
        if not isinstance(state, dict):
            raise TypeError("external program memory state is not a tensor mapping")
        compressed_state = compress_growth_artifact(state, dtype=dtype)
        payload: dict[str, object] = {
            "schema": EXTERNAL_SEQUENCE_PROGRAM_MEMORY_COMPRESSED_SCHEMA,
            "codec": str(dtype),
            "source_schema": self.schema,
            "configuration": source["configuration"],
            "artifacts": source["artifacts"],
            "output_schemas": source["output_schemas"],
            "protected_slots": source["protected_slots"],
            "state": compressed_state,
            "source_sha256": source["sha256"],
        }
        payload["sha256"] = _digest_mapping(payload)
        return payload

    @classmethod
    def from_compressed_payload(
        cls,
        payload: dict[str, object],
    ) -> ExternalSequenceProgramMemory:
        """Restore a compressed file-bank payload through the normal ABI."""

        from .growth import decompress_growth_artifact

        if not isinstance(payload, dict):
            raise TypeError(
                "compressed external program memory payload must be a dictionary"
            )
        if payload.get("schema") != EXTERNAL_SEQUENCE_PROGRAM_MEMORY_COMPRESSED_SCHEMA:
            raise ValueError("unsupported compressed external program memory schema")
        expected = payload.get("sha256")
        unsigned = {key: value for key, value in payload.items() if key != "sha256"}
        if not isinstance(expected, str) or expected != _digest_mapping(unsigned):
            raise ValueError("compressed external program memory checksum mismatch")
        source_schema = payload.get("source_schema")
        configuration = payload.get("configuration")
        artifacts = payload.get("artifacts")
        output_schemas = payload.get("output_schemas")
        protected_slots = payload.get("protected_slots")
        state = payload.get("state")
        source_sha256 = payload.get("source_sha256")
        if source_schema != cls.schema:
            raise ValueError(
                "compressed external program memory source schema is incompatible"
            )
        if not isinstance(configuration, dict) or not isinstance(artifacts, list):
            raise TypeError("compressed external program memory metadata is incomplete")
        if not isinstance(output_schemas, list) or not isinstance(
            protected_slots, list
        ):
            raise TypeError(
                "compressed external program memory slot metadata is invalid"
            )
        if not isinstance(state, dict) or not isinstance(source_sha256, str):
            raise TypeError("compressed external program memory state is invalid")
        if len(source_sha256) != 64:
            raise ValueError(
                "compressed external program memory source digest is malformed"
            )
        try:
            int(source_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "compressed external program memory source digest is malformed"
            ) from error
        decompressed = decompress_growth_artifact(state)
        restored = cls.from_payload(
            {
                "schema": cls.schema,
                "configuration": configuration,
                "artifacts": artifacts,
                "output_schemas": output_schemas,
                "protected_slots": protected_slots,
                "state": decompressed,
                "sha256": source_sha256,
            },
            verify_checksum=False,
        )
        return restored

    def compress_verified(
        self,
        *,
        dtype: torch.dtype | str = torch.float16,
        retention_probe: Callable[[ExternalSequenceProgramMemory], bool],
    ) -> ExternalProgramMemoryTransactionReceipt:
        """Commit storage compression only after a non-mutating behavior probe."""

        source_digest = self.digest()
        source_count = self.file_count
        source_bytes = self._state_storage_bytes()
        compressed = self.compressed_payload(dtype=dtype)
        candidate = self.from_compressed_payload(compressed)
        accepted, probe_unchanged = self._run_lifecycle_probe(
            candidate,
            retention_probe,
        )
        candidate_digest = candidate.digest()
        compressed_state = compressed.get("state")
        if not isinstance(compressed_state, dict):
            raise TypeError("compressed external program memory state is invalid")
        candidate_bytes = self._tensor_mapping_storage_bytes(compressed_state)
        if candidate_bytes >= source_bytes:
            return self._transaction_receipt(
                accepted=False,
                operation="compress",
                affected_slot_id=None,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason="compressed representation is not smaller than source storage",
            )
        if not accepted or not probe_unchanged:
            return self._transaction_receipt(
                accepted=False,
                operation="compress",
                affected_slot_id=None,
                source_file_count=source_count,
                destination_file_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                source_storage_bytes=source_bytes,
                candidate_storage_bytes=source_bytes,
                reason=(
                    "retention probe mutated the compressed candidate"
                    if not probe_unchanged
                    else "compressed candidate failed held-out retention"
                ),
            )
        self._commit_candidate(candidate)
        return self._transaction_receipt(
            accepted=True,
            operation="compress",
            affected_slot_id=None,
            source_file_count=source_count,
            destination_file_count=self.file_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            source_storage_bytes=source_bytes,
            candidate_storage_bytes=candidate_bytes,
            reason="storage-compressed executable memory committed after retention",
        )

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
        return (
            self.programs[slot]
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

    def encode_program(self, codes: torch.Tensor) -> torch.Tensor:
        if (
            codes.ndim != 3
            or codes.shape[1] < 1
            or codes.shape[2] != self.instruction_width
        ):
            raise ValueError("program codes must have shape [batch, steps, width]")
        if self.content_addressing:
            positions = torch.arange(
                1,
                codes.shape[1] + 1,
                device=codes.device,
                dtype=codes.dtype,
            ).view(1, -1, 1)
            weighted = (codes * positions).sum(dim=1)
            # Instruction vectors are deliberately small. A fixed gain keeps
            # distinct ordered programs separable in the address metric while
            # preserving the opaque learned code values.
            return weighted * 32.0
        _, hidden = self.query_encoder(codes)
        return self.program_query(hidden[-1])

    def route_probabilities(self, query: torch.Tensor) -> torch.Tensor:
        """Return soft route probabilities for external credit accounting."""

        if query.ndim != 2 or query.shape[1] != self.instruction_width:
            raise ValueError("sequence program route query has the wrong shape")
        if not len(self.programs):
            raise ValueError("cannot route an empty sequence program memory")
        if self.content_addressing:
            stored = torch.stack(
                tuple(
                    self.encode_program(program.unsqueeze(0)).squeeze(0)
                    for program in self.address_programs
                ),
                dim=0,
            )
            logits = -(query.unsqueeze(1) - stored.unsqueeze(0)).square().mean(dim=-1)
        else:
            keys = torch.stack(tuple(self.slot_keys), dim=0)
            query_latent = self.route_query_encoder(query)
            key_latent = self.key_encoder(keys)
            logits = torch.einsum("bh,sh->bs", query_latent, key_latent)
        soft_weights = torch.softmax(
            logits / (self.router_temperature * (self.router_hidden**0.5)), dim=-1
        )
        return soft_weights

    def route_weights(self, query: torch.Tensor) -> torch.Tensor:
        """Return the configured hard or soft route weights."""

        soft_weights = self.route_probabilities(query)
        if not self.hard_routing:
            return soft_weights
        hard_weights = F.one_hot(
            soft_weights.argmax(dim=-1), num_classes=soft_weights.shape[-1]
        ).to(dtype=soft_weights.dtype)
        return hard_weights + soft_weights - soft_weights.detach()

    def lookup_program_codes(self, query: torch.Tensor) -> torch.Tensor:
        """Return one executable opaque program selected by content.

        Lookup is intentionally singular: a controller asks for one program
        for one opaque query, and the memory returns executable data rather
        than exposing a slot identity to the controller.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("program lookup requires one query vector")
        slot = int(self.route_weights(query).argmax(dim=-1).item())
        return self.program_codes(
            slot,
            batch_size=1,
            device=query.device,
            dtype=query.dtype,
        ).squeeze(0)

    def digest(self) -> str:
        """Return a checksum over the executable bank and file protection state."""

        return self._digest_components(
            self.schema, self.configuration(), self.state_dict()
        )

    @staticmethod
    def _digest_components(
        schema: str,
        configuration: Mapping[str, object],
        state: Mapping[str, torch.Tensor],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(schema.encode("utf-8"))
        digest.update(repr(sorted(configuration.items())).encode("utf-8"))
        for name, value in sorted(state.items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()

    def payload(self) -> dict[str, object]:
        """Serialize the external file bank independently of the controller."""

        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "artifacts": [
                self.artifact(index).payload() for index in range(self.file_count)
            ],
            "output_schemas": list(self._output_schemas),
            "protected_slots": list(self._protected_slots),
            "state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        *,
        verify_checksum: bool = True,
    ) -> ExternalSequenceProgramMemory:
        """Restore one independently versioned external bank.

        ``verify_checksum=False`` is private plumbing for a compressed payload:
        the compressed envelope is verified before its dequantized state is
        loaded, and quantization necessarily changes the uncompressed digest.
        """

        if not isinstance(payload, dict):
            raise TypeError("sequence program memory payload must be a dictionary")
        payload_schema = payload.get("schema")
        if payload_schema not in {
            EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA,
            LEGACY_EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA,
        }:
            raise ValueError("unsupported sequence program memory schema")
        configuration = payload.get("configuration")
        artifacts = payload.get("artifacts")
        output_schemas = payload.get("output_schemas")
        protected_slots = payload.get("protected_slots")
        state = payload.get("state")
        if not isinstance(configuration, dict):
            raise TypeError(
                "sequence program memory configuration must be a dictionary"
            )
        if not isinstance(artifacts, list):
            raise TypeError("sequence program memory artifacts must be a list")
        if not isinstance(output_schemas, list):
            raise TypeError("sequence program memory output schemas must be a list")
        if not isinstance(protected_slots, list):
            raise TypeError("sequence program memory protection must be a list")
        if not isinstance(state, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        ):
            raise TypeError("sequence program memory state must be a tensor mapping")
        memory = cls(
            int(configuration.get("instruction_width", -1)),
            router_hidden=int(configuration.get("router_hidden", -1)),
            router_temperature=float(configuration.get("router_temperature", -1.0)),
            hard_routing=bool(configuration.get("hard_routing", False)),
            content_addressing=bool(configuration.get("content_addressing", False)),
        )
        for artifact_payload in artifacts:
            if not isinstance(artifact_payload, dict):
                raise TypeError("sequence program memory artifact is not a dictionary")
            memory.add_artifact(ExternalProgramArtifact.from_payload(artifact_payload))
        if len(protected_slots) != memory.file_count or any(
            not isinstance(value, bool) for value in protected_slots
        ):
            raise ValueError("sequence program memory protection has the wrong shape")
        if len(output_schemas) != memory.file_count or any(
            value is not None and not isinstance(value, str) for value in output_schemas
        ):
            raise ValueError(
                "sequence program memory output schemas have the wrong shape"
            )
        memory._protected_slots = list(protected_slots)
        memory._output_schemas = list(output_schemas)
        logical_slot_ids = configuration.get("logical_slot_ids")
        if logical_slot_ids is None:
            logical_slot_ids = list(range(memory.file_count))
        if (
            not isinstance(logical_slot_ids, list)
            or len(logical_slot_ids) != memory.file_count
            or any(
                not isinstance(value, int) or value < 0 for value in logical_slot_ids
            )
            or len(set(logical_slot_ids)) != len(logical_slot_ids)
        ):
            raise ValueError("sequence program memory logical IDs have the wrong shape")
        memory._logical_slot_ids = list(logical_slot_ids)
        next_logical_slot_id = configuration.get(
            "next_logical_slot_id",
            max(memory._logical_slot_ids, default=-1) + 1,
        )
        if not isinstance(next_logical_slot_id, int) or next_logical_slot_id <= max(
            memory._logical_slot_ids, default=-1
        ):
            raise ValueError("sequence program memory next logical ID is invalid")
        memory._next_logical_slot_id = next_logical_slot_id
        memory.load_state_dict(state, strict=True)
        if int(configuration.get("slot_count", -1)) != memory.file_count:
            raise ValueError("sequence program memory slot metadata mismatch")
        if verify_checksum:
            expected = payload.get("sha256")
            current_digest = (
                memory.digest()
                if payload_schema == EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA
                else memory._digest_components(
                    LEGACY_EXTERNAL_SEQUENCE_PROGRAM_MEMORY_SCHEMA,
                    configuration,
                    state,
                )
            )
            if not isinstance(expected, str) or expected != current_digest:
                raise ValueError("sequence program memory checksum mismatch")
        return memory


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
        self.left = nn.Linear(instruction_width, register_width * operator_rank)
        self.right = nn.Linear(instruction_width, operator_rank * register_width)
        self.bias = nn.Linear(instruction_width, register_width)
        self.gate = nn.Linear(instruction_width, register_width)
        for module in (self.left, self.right, self.bias):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def residual(self, register: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
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
        self.slot_values = nn.ParameterList()
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
        self.program_query = nn.Linear(self.router_hidden, self.instruction_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "operator_rank": self.operator_rank,
            "router_hidden": self.router_hidden,
            "router_temperature": self.router_temperature,
            "route_encoder": "gru_order_sensitive_v1",
            "slot_count": len(self.slots),
            "growth": "append_only_external_operator_state_v1",
            "file_token": "learned_slot_values_v1",
        }

    def digest(self) -> str:
        """Return an integrity digest over the complete external operator bank."""

        return _digest_mapping(
            {
                "schema": EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
                "configuration": self.configuration(),
                "state": self.state_dict(),
            }
        )

    def payload(self) -> dict[str, object]:
        """Serialize the operator bank independently of the controller."""

        return {
            "schema": EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
            "configuration": self.configuration(),
            "state": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        verify_checksum: bool = True,
    ) -> ExternalSequenceOperatorMemory:
        """Restore one independently versioned operator-memory file."""

        if not isinstance(payload, Mapping):
            raise TypeError("sequence operator memory payload must be a mapping")
        payload_schema = payload.get("schema")
        if payload_schema not in (
            EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
            LEGACY_EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
        ):
            raise ValueError("unsupported sequence operator memory schema")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping):
            raise TypeError("sequence operator memory configuration is invalid")
        if not isinstance(state, Mapping) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        ):
            raise TypeError("sequence operator memory state must be a tensor mapping")
        slot_count = configuration.get("slot_count")
        if not isinstance(slot_count, int) or slot_count < 0:
            raise ValueError("sequence operator memory slot count is invalid")
        memory = cls(
            int(configuration.get("register_width", -1)),
            int(configuration.get("instruction_width", -1)),
            operator_rank=int(configuration.get("operator_rank", -1)),
            router_hidden=int(configuration.get("router_hidden", -1)),
            router_temperature=float(configuration.get("router_temperature", -1.0)),
        )
        for _ in range(slot_count):
            memory.add_slot()
        memory.load_state_dict(
            state,
            strict=payload_schema == EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
        )
        if payload_schema == EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA and (
            memory.configuration() != dict(configuration)
        ):
            raise ValueError("sequence operator memory configuration mismatch")
        if payload_schema == LEGACY_EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA:
            current_configuration = memory.configuration()
            if any(
                current_configuration.get(name) != value
                for name, value in configuration.items()
                if name != "schema"
            ):
                raise ValueError("legacy sequence operator memory configuration mismatch")
        if verify_checksum:
            expected = payload.get("sha256")
            current_digest = (
                memory.digest()
                if payload_schema == EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA
                else _digest_mapping(
                    {
                        "schema": LEGACY_EXTERNAL_SEQUENCE_OPERATOR_MEMORY_SCHEMA,
                        "configuration": configuration,
                        "state": state,
                    }
                )
            )
            if not isinstance(expected, str) or expected != current_digest:
                raise ValueError("sequence operator memory checksum mismatch")
        return memory

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
        self.slot_values.append(nn.Parameter(torch.zeros(self.instruction_width)))
        return len(self.slots) - 1

    def bind(self, query: torch.Tensor) -> BoundExternalSequenceOperatorMemory:
        """Materialize one route for a fixed execution rollout.

        Routing is context lookup, not the recurrent computation itself.  A
        caller that will execute several instructions for the same context
        should bind once and pass the returned handle through the chain.  The
        returned handle is ephemeral: if the external bank grows, it must be
        rebound so the new slot set is addressed explicitly.
        """

        return BoundExternalSequenceOperatorMemory(self, self.route_weights(query))

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

    def read_token(self, route_weights: torch.Tensor) -> torch.Tensor:
        """Read one opaque file token from an already materialized route.

        The token is a learned, route-weighted view of external file values.  It
        is deliberately separate from :meth:`residual`: a caller can expose
        the file read to a replaceable memory-side adapter while the shared
        interpreter continues to execute the same bound route.  No route
        encoder is run here, so repeated recurrent steps cannot silently turn
        a file read into repeated contextual lookup.
        """

        if route_weights.ndim != 2 or route_weights.shape[1] != len(self.slots):
            raise ValueError("sequence operator route weights have the wrong shape")
        if not len(self.slots):
            raise ValueError("cannot read an empty sequence operator memory")
        if not bool(torch.isfinite(route_weights).all()):
            raise ValueError("sequence operator route weights must be finite")
        values = torch.stack(tuple(self.slot_values), dim=0)
        return torch.einsum("bs,sw->bw", route_weights, values)

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
        return self.residual_from_weights(weights, register, code)

    def residual_from_weights(
        self,
        weights: torch.Tensor,
        register: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a previously materialized route without re-encoding it."""

        if weights.ndim != 2 or weights.shape[1] != len(self.slots):
            raise ValueError("sequence operator route weights have the wrong shape")
        if weights.shape[0] != register.shape[0] or code.shape[0] != register.shape[0]:
            raise ValueError("sequence operator batch dimensions do not match")
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("sequence operator route weights must be finite")
        if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(dim=-1),
            torch.ones(weights.shape[0], device=weights.device, dtype=weights.dtype),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("sequence operator route weights must be a distribution")
        residuals = torch.stack(
            tuple(slot.residual(register, code) for slot in self.slots), dim=1
        )
        return torch.einsum("bs,bsr->br", weights, residuals)


class BoundExternalSequenceOperatorMemory:
    """An ephemeral, versioned bind-once view over operator memory.

    The underlying bank remains external state and owns its parameters.  This
    object stores only the route distribution selected for one batch/rollout;
    the shared interpreter can therefore iterate without repeatedly running
    the route encoder.  It is intentionally not an ``nn.Module`` and must not
    be registered as controller state.
    """

    def __init__(
        self,
        memory: ExternalSequenceOperatorMemory,
        route_weights: torch.Tensor,
    ) -> None:
        if not isinstance(memory, ExternalSequenceOperatorMemory):
            raise TypeError("bound operator memory requires an operator memory")
        if route_weights.ndim != 2 or route_weights.shape[1] != len(memory.slots):
            raise ValueError("bound operator route weights have the wrong shape")
        if not len(memory.slots):
            raise ValueError("cannot bind an empty sequence operator memory")
        if not bool(torch.isfinite(route_weights).all()):
            raise ValueError("bound operator route weights must be finite")
        self.memory = memory
        self._route_weights = route_weights
        self._slot_count = len(memory.slots)

    @property
    def slots(self) -> nn.ModuleList:
        """Expose slots for diagnostics without copying or owning them."""

        return self.memory.slots

    @property
    def route_weights(self) -> torch.Tensor:
        """Return the route selected at bind time."""

        return self._route_weights

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": EXTERNAL_SEQUENCE_OPERATOR_BINDING_SCHEMA,
            "memory_schema": self.memory.configuration()["schema"],
            "register_width": self.memory.register_width,
            "instruction_width": self.memory.instruction_width,
            "slot_count": self._slot_count,
            "batch_size": int(self._route_weights.shape[0]),
            "routing": "materialized_once_per_rollout_v1",
            "persistence": "ephemeral_external_binding_v1",
        }

    def _validate_memory_snapshot(self) -> None:
        if len(self.memory.slots) != self._slot_count:
            raise RuntimeError(
                "sequence operator memory changed after binding; rebind the rollout"
            )

    def residual(self, register: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Apply the bound route to one recurrent instruction."""

        self._validate_memory_snapshot()
        return self.memory.residual_from_weights(
            self._route_weights,
            register,
            code,
        )

    def read_token(self) -> torch.Tensor:
        """Read the bound external file once without rerunning its router."""

        self._validate_memory_snapshot()
        return self.memory.read_token(self._route_weights)

    def routed_residual(
        self,
        query: torch.Tensor,
        register: torch.Tensor,
        code: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility-shaped call that deliberately ignores ``query``.

        The query was consumed by :meth:`ExternalSequenceOperatorMemory.bind`.
        Keeping this method makes the binding safe to pass through generic
        execution helpers while preserving the no-rerouting guarantee.
        """

        del query
        return self.residual(register, code)


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


@dataclass(frozen=True)
class ExternalRegisterComputeBasisArtifact:
    """Portable, checksummed file for one learned external compute slot.

    Instruction vectors and compute-slot weights are separate kinds of
    external state.  This artifact makes the latter independently durable,
    so a new computation can be moved between interpreters without copying
    the shared controller or accidentally admitting a partial checkpoint.
    """

    configuration: Mapping[str, int | str]
    state: Mapping[str, torch.Tensor]
    schema: str = EXTERNAL_REGISTER_BASIS_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXTERNAL_REGISTER_BASIS_ARTIFACT_SCHEMA:
            raise ValueError("unsupported compute basis artifact schema")
        if not isinstance(self.configuration, Mapping) or not self.configuration:
            raise TypeError("compute basis artifact configuration must be a mapping")
        if self.configuration.get("schema") != EXTERNAL_REGISTER_BASIS_SCHEMA:
            raise ValueError("compute basis artifact ABI schema is invalid")
        if not isinstance(self.state, Mapping) or not self.state:
            raise ValueError("compute basis artifact state must be nonempty")
        for name, value in self.state.items():
            if not isinstance(name, str) or not name:
                raise ValueError("compute basis artifact state names must be nonempty")
            if not isinstance(value, torch.Tensor):
                raise TypeError("compute basis artifact state values must be tensors")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"compute basis artifact state entry {name!r} is non-finite"
                )

    def _digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(repr(sorted(dict(self.configuration).items())).encode("utf-8"))
        for name, value in sorted(self.state.items()):
            detached = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(repr(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
        return digest.hexdigest()

    def digest(self) -> str:
        """Return an integrity digest over the ABI and all learned tensors."""

        return self._digest()

    def payload(self) -> dict[str, object]:
        """Return a tensor-only payload suitable for a memory-side file."""

        return {
            "schema": self.schema,
            "configuration": dict(self.configuration),
            "state": {
                name: value.detach().cpu().clone() for name, value in self.state.items()
            },
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> ExternalRegisterComputeBasisArtifact:
        """Restore and integrity-check one external compute file."""

        if not isinstance(payload, Mapping):
            raise TypeError("compute basis artifact payload must be a mapping")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping):
            raise TypeError("compute basis artifact configuration is invalid")
        if not isinstance(state, Mapping):
            raise TypeError("compute basis artifact state is invalid")
        artifact = cls(
            configuration=dict(configuration),
            state=dict(state),
            schema=str(payload.get("schema", "")),
        )
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != artifact.digest():
            raise ValueError("compute basis artifact checksum mismatch")
        return artifact


class ExternalRegisterComputeBasis(nn.Module):
    """One append-only external computation slot.

    A slot is fresh computation capacity, not a semantic label. It sees only
    the current register and an opaque instruction vector and returns a bounded
    register update. New slots can be trained without changing the controller
    or parameters of previously mastered slots.
    """

    HISTORY_HEAD_COUNT = 4
    HISTORY_AGE_SLOT_COUNT = 8
    HISTORY_RELATION_WIDTH = 8

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
        register_input_mode: str = "full",
        history_query_mode: str = "instruction_only",
        history_age_slot_count: int | None = None,
        history_relation_width: int | None = None,
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
        if event_read_mode not in (
            "flattened_window",
            "attention_pool",
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ):
            raise ValueError("unsupported compute basis event read mode")
        if event_read_mode == "attention_pool" and not event_window_size:
            raise ValueError("attention event reading requires an event window")
        if event_read_mode in (
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ) and event_window_size:
            raise ValueError(
                "variable history readers use external history, not an event window"
            )
        if event_read_mode in (
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ) and event_width < 1:
            raise ValueError("variable history readers require a positive event width")
        if register_input_mode not in ("full", "event_window_only"):
            raise ValueError("unsupported compute basis register input mode")
        if history_query_mode not in (
            "instruction_only",
            "instruction_current_event",
        ):
            raise ValueError("unsupported history query mode")
        if history_query_mode != "instruction_only" and event_read_mode not in (
            "history_attention",
            "history_indexed",
        ):
            raise ValueError(
                "current-event history queries require history attention"
            )
        if event_read_mode in ("history_indexed", "history_relation_indexed"):
            effective_history_age_slot_count = (
                self.HISTORY_AGE_SLOT_COUNT
                if history_age_slot_count is None
                else int(history_age_slot_count)
            )
            if effective_history_age_slot_count < 1:
                raise ValueError("indexed history age slot count must be positive")
        elif history_age_slot_count is not None:
            raise ValueError(
                "history age slot count is only valid for indexed history"
            )
        else:
            effective_history_age_slot_count = 0
        if event_read_mode == "history_relation_indexed":
            effective_history_relation_width = (
                self.HISTORY_RELATION_WIDTH
                if history_relation_width is None
                else int(history_relation_width)
            )
            if effective_history_relation_width < 1:
                raise ValueError("history relation width must be positive")
        elif history_relation_width is not None:
            raise ValueError(
                "history relation width is only valid for relation-indexed history"
            )
        else:
            effective_history_relation_width = 0
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.hidden = int(hidden)
        self.event_width = int(event_width)
        self.event_window_size = int(event_window_size)
        self.microsteps = int(microsteps)
        self.event_read_mode = event_read_mode
        self.register_input_mode = register_input_mode
        self.history_query_mode = history_query_mode
        self.event_window_width = self.event_width * self.event_window_size
        self.history_head_count = (
            self.HISTORY_HEAD_COUNT if event_read_mode == "history_attention" else 1
        )
        self.history_age_slot_count = (
            effective_history_age_slot_count
            if event_read_mode in ("history_indexed", "history_relation_indexed")
            else 0
        )
        self.history_relation_width = (
            effective_history_relation_width
            if event_read_mode == "history_relation_indexed"
            else 0
        )
        event_feature_width = (
            self.event_width
            * self.history_age_slot_count
            if event_read_mode == "history_indexed"
            else self.history_relation_width * self.history_age_slot_count
            if event_read_mode == "history_relation_indexed"
            else self.event_width * self.history_head_count
            if event_read_mode == "history_attention"
            else self.event_width
            if event_read_mode == "attention_pool"
            else self.event_window_width
        )
        current_event_width = (
            self.event_width
            if event_read_mode in (
                "history_attention",
                "history_indexed",
                "history_relation_indexed",
            )
            else 0
        )
        width = (
            (self.register_width if register_input_mode == "full" else 0)
            + self.instruction_width
            + event_feature_width
            + current_event_width
        )
        if event_read_mode in ("attention_pool", "history_attention"):
            query_width = (
                self.register_width if register_input_mode == "full" else 0
            ) + self.instruction_width + (
                self.event_width
                if event_read_mode == "history_attention"
                and history_query_mode == "instruction_current_event"
                else 0
            )
            self.event_query = nn.Linear(
                query_width,
                self.hidden * self.history_head_count,
            )
            if event_read_mode == "history_attention":
                self.history_recurrent = nn.GRU(
                    self.event_width,
                    self.hidden,
                    batch_first=True,
                )
                self.history_key = nn.Linear(self.hidden, self.hidden)
                self.history_value = nn.Linear(self.hidden, self.event_width)
                # Relative age is memory addressing information, not a task
                # or modality feature.  Fourier features keep the ABI
                # unbounded while making nearby offsets distinguishable to a
                # fresh external file.
                self.history_age_frequency_count = 8
                self.history_age_key = nn.Sequential(
                    nn.Linear(17, self.hidden),
                    nn.GELU(),
                    nn.Linear(self.hidden, self.hidden),
                )
                self.history_age_value = nn.Sequential(
                    nn.Linear(17, self.hidden),
                    nn.GELU(),
                    nn.Linear(self.hidden, self.event_width),
                )
                self.history_summary = nn.Linear(self.hidden, self.event_width)
                self.history_token_encoder = nn.Sequential(
                    nn.Linear(self.event_width + 17, self.hidden),
                    nn.GELU(),
                    nn.Linear(self.hidden, self.hidden),
                )
                self.history_set_summary = nn.Sequential(
                    nn.Linear(self.hidden, self.hidden),
                    nn.GELU(),
                    nn.Linear(self.hidden, self.event_width),
                )
            else:
                self.event_key = nn.Linear(self.event_width, self.hidden)
                self.event_value = nn.Linear(self.event_width, self.event_width)
        if event_read_mode == "history_relation_indexed":
            self.history_relation_encoder = nn.Sequential(
                nn.Linear(self.event_width * 4 + 1, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, self.history_relation_width),
            )
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
        configuration: dict[str, int | str] = {
            "schema": EXTERNAL_REGISTER_BASIS_SCHEMA,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "hidden": self.hidden,
            "event_width": self.event_width,
            "event_window_size": self.event_window_size,
            "microsteps": self.microsteps,
            "event_read_mode": self.event_read_mode,
            "register_input_mode": self.register_input_mode,
            "storage": "append_only_external_compute_slot_v1",
            "signature": "one_opaque_learned_slot_key_v1",
        }
        if self.event_read_mode in (
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ):
            configuration["history_query_mode"] = self.history_query_mode
            if self.event_read_mode == "history_attention":
                configuration["history_contract"] = (
                    "variable_external_history_attention_v3"
                    if self.history_query_mode == "instruction_current_event"
                    else "variable_external_history_attention_v2"
                )
                configuration["history_head_count"] = self.history_head_count
            elif self.event_read_mode == "history_indexed":
                configuration["history_contract"] = (
                    "variable_external_history_indexed_v1"
                )
                configuration["history_age_slot_count"] = (
                    self.history_age_slot_count
                )
            else:
                configuration["history_contract"] = (
                    "variable_external_history_relation_indexed_v2"
                )
                configuration["history_age_slot_count"] = (
                    self.history_age_slot_count
                )
                configuration["history_relation_width"] = (
                    self.history_relation_width
                )
        return configuration

    def artifact(self) -> ExternalRegisterComputeBasisArtifact:
        """Snapshot this learned slot as an independently portable artifact.

        The slot is external computation, so its learned weights must be
        reloadable without serializing or mutating the shared interpreter.
        The returned artifact contains only the slot ABI and tensor state; it
        carries no task, modality, or verifier-private metadata.
        """

        return ExternalRegisterComputeBasisArtifact(
            configuration=self.configuration(),
            state={
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: ExternalRegisterComputeBasisArtifact,
        *,
        expected_configuration: Mapping[str, int | str] | None = None,
    ) -> ExternalRegisterComputeBasis:
        """Rehydrate one slot without constructing a shared interpreter."""

        if not isinstance(artifact, ExternalRegisterComputeBasisArtifact):
            raise TypeError("compute basis restoration requires a basis artifact")
        configuration = dict(artifact.configuration)
        # v1 artifacts predate the explicit register-input isolation mode;
        # their behavior is the original full-register path.
        configuration.setdefault("register_input_mode", "full")
        if configuration.get("event_read_mode") in (
            "history_attention",
            "history_indexed",
        ):
            configuration.setdefault("history_query_mode", "instruction_only")
        if configuration.get("event_read_mode") == "history_indexed":
            configuration.setdefault(
                "history_age_slot_count", cls.HISTORY_AGE_SLOT_COUNT
            )
        if configuration.get("event_read_mode") == "history_relation_indexed":
            configuration.setdefault(
                "history_age_slot_count", cls.HISTORY_AGE_SLOT_COUNT
            )
            configuration.setdefault(
                "history_relation_width", cls.HISTORY_RELATION_WIDTH
            )
        if expected_configuration is not None:
            expected = dict(expected_configuration)
            expected.setdefault("register_input_mode", "full")
            if configuration != expected:
                raise ValueError("compute basis artifact configuration is incompatible")
        basis = cls(
            int(configuration["register_width"]),
            int(configuration["instruction_width"]),
            hidden=int(configuration["hidden"]),
            event_width=int(configuration["event_width"]),
            event_window_size=int(configuration["event_window_size"]),
            microsteps=int(configuration["microsteps"]),
            event_read_mode=str(configuration["event_read_mode"]),
            register_input_mode=str(
                configuration.get("register_input_mode", "full")
            ),
            history_query_mode=str(
                configuration.get("history_query_mode", "instruction_only")
            ),
            history_age_slot_count=(
                int(configuration["history_age_slot_count"])
                if configuration.get("event_read_mode")
                in ("history_indexed", "history_relation_indexed")
                else None
            ),
            history_relation_width=(
                int(configuration["history_relation_width"])
                if configuration.get("event_read_mode")
                == "history_relation_indexed"
                else None
            ),
        )
        current = basis.state_dict()
        if set(current) != set(artifact.state):
            raise ValueError("compute basis artifact state entries are incompatible")
        for name, value in artifact.state.items():
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(
                    f"compute basis artifact state entry {name!r} is incompatible"
                )
        basis.load_state_dict(
            {
                name: value.detach().clone().to(device=current[name].device)
                for name, value in artifact.state.items()
            },
            strict=True,
        )
        return basis

    def forward(
        self,
        register: torch.Tensor,
        code: torch.Tensor,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for compute basis")
        if self.register_input_mode == "event_window_only":
            register = torch.zeros_like(register)
        if code.shape != (register.shape[0], self.instruction_width):
            raise ValueError("instruction code has the wrong shape for compute basis")
        if self.event_read_mode in (
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ):
            if event_window is not None or event_window_mask is not None:
                raise ValueError("history attention does not accept a fixed event window")
            if (
                event_history is None
                or event_history_mask is None
                or event_history_age is None
                or current_event is None
            ):
                raise ValueError(
                    "history attention requires values, mask, relative age, and current event"
                )
            if current_event.shape != (register.shape[0], self.event_width):
                raise ValueError("current event has the wrong shape")
            if not bool(torch.isfinite(current_event).all()):
                raise ValueError("current event must contain finite values")
            if event_history.ndim != 3 or event_history.shape[0] != register.shape[0]:
                raise ValueError("external event history has the wrong shape")
            if event_history.shape[2] != self.event_width:
                raise ValueError("external event history has the wrong width")
            if (
                event_history_mask.shape != event_history.shape[:2]
                or event_history_mask.dtype is not torch.bool
            ):
                raise ValueError("external event history mask has the wrong shape")
            if (
                event_history_age.shape != event_history.shape[:2]
                or not torch.is_floating_point(event_history_age)
            ):
                raise ValueError("external event history age has the wrong shape or dtype")
            if not bool(torch.isfinite(event_history_age).all()):
                raise ValueError("external event history age must be finite")
            if bool(torch.any(event_history_age < 0)):
                raise ValueError("external event history age cannot be negative")
            if self.event_read_mode in (
                "history_indexed",
                "history_relation_indexed",
            ):
                if not bool(
                    torch.all(event_history_age == event_history_age.round())
                ):
                    raise ValueError(
                        "indexed history ages must be integral relative offsets"
                    )
                if bool(
                    torch.any(event_history_age >= self.history_age_slot_count)
                ):
                    raise ValueError(
                        "indexed history age exceeds the bounded slot ABI"
                    )
            if not bool(torch.isfinite(event_history).all()):
                raise ValueError("external event history must contain finite values")
            window = event_history * event_history_mask.unsqueeze(-1).to(
                event_history.dtype
            )
            read_mask = event_history_mask
        elif self.event_window_size:
            if (
                event_history is not None
                or event_history_mask is not None
                or event_history_age is not None
                or current_event is not None
            ):
                raise ValueError("fixed event-window basis does not accept external history")
            if event_window is None or event_window_mask is None:
                raise ValueError("event window is required for this compute basis")
            if event_window.shape != (
                register.shape[0],
                self.event_window_size,
                self.event_width,
            ):
                raise ValueError("event window has the wrong shape for compute basis")
            if (
                event_window_mask.shape != (register.shape[0], self.event_window_size)
                or event_window_mask.dtype is not torch.bool
            ):
                raise ValueError("event window mask has the wrong shape")
            window = event_window * event_window_mask.unsqueeze(-1).to(
                event_window.dtype
            )
            read_mask = event_window_mask
        else:
            if any(
                value is not None
                for value in (
                    event_window,
                    event_window_mask,
                    event_history,
                    event_history_mask,
                    event_history_age,
                    current_event,
                )
            ):
                raise ValueError("event history is unsupported by this compute basis")
            window = None
            read_mask = None
        for _ in range(self.microsteps):
            basis_register = (
                register
                if self.register_input_mode == "full"
                else torch.zeros_like(register)
            )
            if self.event_read_mode in ("attention_pool", "history_attention"):
                query_inputs = (
                    (
                        basis_register,
                        code,
                        current_event,
                    )
                    if self.history_query_mode == "instruction_current_event"
                    and self.register_input_mode == "full"
                    else (basis_register, code)
                    if self.register_input_mode == "full"
                    else (
                        (code, current_event)
                        if self.history_query_mode == "instruction_current_event"
                        else (code,)
                    )
                )
                query = self.event_query(
                    torch.cat(query_inputs, dim=-1)
                )
                if self.event_read_mode == "history_attention":
                    # The memory read is addressed by relative age, but the
                    # sequence reducer consumes valid records oldest to
                    # newest, followed by the current event.  Keeping these
                    # concerns separate preserves opaque addressing while
                    # making the causal order explicit to the reducer.  The
                    # external read may be left-padded or sparsely masked, so
                    # compact valid history before the recurrent pass.  This
                    # prevents padding from changing the hidden state and
                    # keeps the current event at the true sequence boundary.
                    history_length = read_mask.sum(dim=1)
                    history_rank = read_mask.to(torch.long).cumsum(dim=1) - 1
                    compact_history = torch.zeros_like(window)
                    compact_history.scatter_add_(
                        1,
                        history_rank.clamp_min(0).unsqueeze(-1).expand_as(window),
                        window * read_mask.unsqueeze(-1).to(window.dtype),
                    )
                    compact_sequence = torch.zeros(
                        register.shape[0],
                        window.shape[1] + 1,
                        self.event_width,
                        device=register.device,
                        dtype=register.dtype,
                    )
                    compact_sequence[:, :-1] = compact_history
                    compact_sequence.scatter_add_(
                        1,
                        history_length[:, None, None].expand(
                            -1, 1, self.event_width
                        ),
                        current_event.unsqueeze(1),
                    )
                    sequence_mask = (
                        torch.arange(
                            compact_sequence.shape[1],
                            device=register.device,
                        )
                        .unsqueeze(0)
                        <= history_length[:, None]
                    )
                    sequence_states, _ = self.history_recurrent(compact_sequence)
                    compact_history_states = sequence_states[:, :-1]
                    ranked_history_states = compact_history_states.gather(
                        1,
                        history_rank.clamp_min(0).unsqueeze(-1).expand_as(
                            compact_history_states
                        ),
                    )
                    history_states = ranked_history_states * read_mask.unsqueeze(
                        -1
                    ).to(compact_history_states.dtype)
                    keys = self.history_key(history_states)
                    values = self.history_value(history_states)
                    last_index = history_length
                    summary = sequence_states.gather(
                        1,
                        last_index[:, None, None].expand(
                            -1, -1, sequence_states.shape[-1]
                        ),
                    ).squeeze(1)
                    summary = self.history_summary(summary)
                    summary = summary * read_mask.any(dim=1, keepdim=True).to(
                        summary.dtype
                    )
                    age = event_history_age.to(
                        device=window.device,
                        dtype=window.dtype,
                    )
                    frequencies = torch.arange(
                        self.history_age_frequency_count,
                        device=window.device,
                        dtype=window.dtype,
                    )
                    periods = torch.pow(
                        torch.tensor(2.0, device=window.device, dtype=window.dtype),
                        frequencies,
                    )
                    phase = age.unsqueeze(-1) * math.pi / periods
                    age_features = torch.cat(
                        (
                            torch.log1p(age).unsqueeze(-1),
                            phase.sin(),
                            phase.cos(),
                        ),
                        dim=-1,
                    )
                    token_features = self.history_token_encoder(
                        torch.cat((window, age_features), dim=-1)
                    )
                    set_summary = self.history_set_summary(
                        (
                            token_features
                            * read_mask.unsqueeze(-1).to(token_features.dtype)
                        ).sum(dim=1)
                    )
                    set_summary = set_summary * read_mask.any(
                        dim=1, keepdim=True
                    ).to(set_summary.dtype)
                    summary = summary + set_summary
                    keys = keys + self.history_age_key(age_features)
                    values = values + self.history_age_value(age_features)
                else:
                    keys = self.event_key(window)
                    values = self.event_value(window)
                if self.event_read_mode == "history_attention":
                    query = query.reshape(
                        register.shape[0], self.history_head_count, self.hidden
                    )
                    scores = torch.einsum("bhd,btd->bht", query, keys)
                    scores = scores / (self.hidden**0.5)
                    valid = read_mask.unsqueeze(1)
                    scores = scores.masked_fill(~valid, -1e9)
                    weights = torch.softmax(scores, dim=-1)
                    event_features = torch.einsum(
                        "bht,bte->bhe", weights * valid.to(weights.dtype), values
                    )
                    event_features = event_features + summary.unsqueeze(1)
                    event_features = event_features.flatten(1)
                else:
                    scores = torch.einsum("bd,btd->bt", query, keys)
                    scores = scores / (self.hidden**0.5)
                    valid = read_mask
                    scores = scores.masked_fill(~valid, -1e9)
                    weights = torch.softmax(scores, dim=-1)
                    event_features = (weights * valid.to(weights.dtype)).unsqueeze(
                        -1
                    ) * values
                    event_features = event_features.sum(dim=1)
            elif self.event_read_mode in (
                "history_indexed",
                "history_relation_indexed",
            ):
                age_slot_indices = event_history_age.to(torch.long).clamp(
                    min=0, max=self.history_age_slot_count - 1
                )
                age_slots = torch.zeros(
                    register.shape[0],
                    self.history_age_slot_count,
                    self.event_width,
                    device=window.device,
                    dtype=window.dtype,
                )
                age_slots.scatter_add_(
                    1,
                    age_slot_indices.unsqueeze(-1).expand_as(window),
                    window * read_mask.unsqueeze(-1).to(window.dtype),
                )
                age_slot_counts = torch.zeros(
                    register.shape[0],
                    self.history_age_slot_count,
                    device=window.device,
                    dtype=window.dtype,
                )
                age_slot_counts.scatter_add_(
                    1,
                    age_slot_indices,
                    read_mask.to(window.dtype),
                )
                age_slots = age_slots / age_slot_counts.clamp_min(1).unsqueeze(-1)
                if self.event_read_mode == "history_relation_indexed":
                    present_slots = (age_slot_counts > 0).to(window.dtype)
                    repeated_current = current_event.unsqueeze(1).expand_as(age_slots)
                    relation_input = torch.cat(
                        (
                            age_slots,
                            repeated_current,
                            age_slots - repeated_current,
                            age_slots * repeated_current,
                            present_slots.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                    event_features = self.history_relation_encoder(
                        relation_input
                    ) * present_slots.unsqueeze(-1)
                    event_features = event_features.flatten(1)
                else:
                    event_features = age_slots.flatten(1)
            else:
                event_features = (
                    torch.zeros(
                        register.shape[0],
                        self.event_window_width,
                        device=register.device,
                        dtype=register.dtype,
                    )
                    if window is None
                    else window.flatten(1)
                )
            features = torch.cat(
                (basis_register, code, current_event, event_features)
                if self.event_read_mode in (
                    "history_attention",
                    "history_indexed",
                    "history_relation_indexed",
                )
                and self.register_input_mode == "full"
                else (code, current_event, event_features)
                if self.event_read_mode in (
                    "history_attention",
                    "history_indexed",
                    "history_relation_indexed",
                )
                else (basis_register, code, event_features)
                if self.event_window_size and self.register_input_mode == "full"
                else (code, event_features)
                if self.event_window_size
                else (basis_register, code)
                if self.register_input_mode == "full"
                else (code,),
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
        basis_register_input_mode: str = "full",
        basis_history_query_mode: str = "instruction_only",
        basis_history_age_slot_count: int | None = None,
        basis_history_relation_width: int | None = None,
        event_input_mode: str = "frontend",
        event_window_size: int = 0,
        role_count: int = 4,
        operator_basis_count: int = 4,
    ) -> None:
        super().__init__()
        if (
            min(
                event_width,
                action_width,
                intention_width,
                register_width,
                instruction_width,
                interpreter_hidden,
                operator_rank,
                basis_hidden,
                basis_microsteps,
            )
            < 1
        ):
            raise ValueError("external register dimensions must be positive")
        if basis_event_read_mode not in (
            "flattened_window",
            "attention_pool",
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ):
            raise ValueError("unsupported basis event read mode")
        if basis_event_read_mode == "attention_pool" and not event_window_size:
            raise ValueError("attention basis reading requires an event window")
        if basis_event_read_mode in (
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ) and event_window_size:
            raise ValueError(
                "variable history readers use external history, not an event window"
            )
        if basis_register_input_mode not in ("full", "event_window_only"):
            raise ValueError("unsupported basis register input mode")
        if basis_history_query_mode not in (
            "instruction_only",
            "instruction_current_event",
        ):
            raise ValueError("unsupported basis history query mode")
        if (
            basis_history_query_mode != "instruction_only"
            and basis_event_read_mode
            not in ("history_attention", "history_indexed")
        ):
            raise ValueError(
                "current-event history queries require history attention"
            )
        if basis_event_read_mode in ("history_indexed", "history_relation_indexed"):
            effective_basis_history_age_slot_count = (
                ExternalRegisterComputeBasis.HISTORY_AGE_SLOT_COUNT
                if basis_history_age_slot_count is None
                else int(basis_history_age_slot_count)
            )
            if effective_basis_history_age_slot_count < 1:
                raise ValueError("indexed history age slot count must be positive")
        elif basis_history_age_slot_count is not None:
            raise ValueError(
                "basis history age slot count is only valid for indexed history"
            )
        else:
            effective_basis_history_age_slot_count = 0
        if basis_event_read_mode == "history_relation_indexed":
            effective_basis_history_relation_width = (
                ExternalRegisterComputeBasis.HISTORY_RELATION_WIDTH
                if basis_history_relation_width is None
                else int(basis_history_relation_width)
            )
            if effective_basis_history_relation_width < 1:
                raise ValueError("basis history relation width must be positive")
        elif basis_history_relation_width is not None:
            raise ValueError(
                "basis history relation width is only valid for relation-indexed history"
            )
        else:
            effective_basis_history_relation_width = 0
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
        if operator_basis_count < 1:
            raise ValueError("operator basis count must be positive")
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
            EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE,
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
            or basis.register_input_mode != basis_register_input_mode
            or basis.history_query_mode != basis_history_query_mode
            or basis.history_age_slot_count
            != effective_basis_history_age_slot_count
            or basis.history_relation_width
            != effective_basis_history_relation_width
            or (
                (
                    event_window_size
                    or basis_event_read_mode
                    in (
                        "history_attention",
                        "history_indexed",
                        "history_relation_indexed",
                    )
                )
                and basis.event_width != event_width
            )
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
        self.basis_register_input_mode = basis_register_input_mode
        self.basis_history_query_mode = basis_history_query_mode
        self.basis_history_age_slot_count = effective_basis_history_age_slot_count
        self.basis_history_relation_width = effective_basis_history_relation_width
        self.event_input_mode = event_input_mode
        self.event_window_size = int(event_window_size)
        self.role_count = int(role_count)
        self.operator_basis_count = int(operator_basis_count)
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
        if operator_mode == EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE:
            # Every opaque instruction is a mixture of the same learned
            # transition factors. This creates a common state algebra while
            # keeping operation identity latent and controller-independent.
            self.operator_basis_left = nn.Parameter(
                torch.empty(
                    self.operator_basis_count,
                    register_width,
                    operator_rank,
                )
            )
            self.operator_basis_right = nn.Parameter(
                torch.empty(
                    self.operator_basis_count,
                    operator_rank,
                    register_width,
                )
            )
            self.operator_basis_bias = nn.Parameter(
                torch.zeros(self.operator_basis_count, register_width)
            )
            self.operator_basis_selector = nn.Linear(
                instruction_width, self.operator_basis_count
            )
            nn.init.normal_(self.operator_basis_left, mean=0.0, std=0.02)
            nn.init.normal_(self.operator_basis_right, mean=0.0, std=0.02)
            nn.init.normal_(self.operator_basis_selector.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.operator_basis_selector.bias)
            self.operator_normalizer = nn.LayerNorm(register_width)
            self.operator_composition_gate = nn.Linear(
                instruction_width, register_width
            )
            nn.init.zeros_(self.operator_composition_gate.bias)
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
            self.operator_meta_bias = nn.Linear(instruction_width, register_width)
            self.operator_meta_gate = nn.Linear(instruction_width, register_width)
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
            self.operator_modulation = nn.Linear(instruction_width, interpreter_hidden)
            self.operator_output = nn.Linear(interpreter_hidden, register_width)
            self.operator_gate = nn.Linear(interpreter_hidden * 2, 1)
            if operator_mode == "factorized_film":
                self.operator_bias = nn.Linear(instruction_width, register_width)
                nn.init.zeros_(self.operator_bias.bias)
            else:
                self.operator_film_bias = nn.Linear(instruction_width, register_width)
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
            "basis_register_input_mode": self.basis_register_input_mode,
            "basis_history_query_mode": self.basis_history_query_mode,
            "basis_history_age_slot_count": self.basis_history_age_slot_count,
            "basis_history_relation_width": self.basis_history_relation_width,
            "event_input_mode": self.event_input_mode,
            "event_window_size": self.event_window_size,
            "role_count": self.role_count,
            "operator_basis_count": self.operator_basis_count,
            "state": "external_working_register_with_recurrent_context_v2",
            "execution": "shared_interpreter_serial_instruction_chain_v1",
            "compute_basis": (
                "neural-computer.external-register-shared-interpreter.v1"
                if self.operator_mode
                in (
                    EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
                    EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
                    EXTERNAL_REGISTER_SHARED_BANKED_MODE,
                    EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
                    EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
                    EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
                    EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE,
                    "factorized_protected_bounded_meta",
                )
                else EXTERNAL_REGISTER_BASIS_SCHEMA
            ),
            "basis_binding": (
                "instruction_vector_selects_shared_interpreter_v1"
                if self.operator_mode
                in (
                    EXTERNAL_REGISTER_SHARED_INTERPRETER_MODE,
                    EXTERNAL_REGISTER_SHARED_BOUNDED_MODE,
                    EXTERNAL_REGISTER_SHARED_BANKED_MODE,
                    EXTERNAL_REGISTER_SHARED_ROLE_BOUND_MODE,
                    EXTERNAL_REGISTER_SHARED_RELATIONAL_MODE,
                    EXTERNAL_REGISTER_SHARED_STABLE_RELATIONAL_MODE,
                    EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE,
                    "factorized_protected_bounded_meta",
                )
                else "opaque_memory_side_slot_index_v1"
            ),
            "read_execute": EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
            "execution_trace": EXTERNAL_REGISTER_EXECUTION_TRACE_SCHEMA,
            "downstream_input": (
                "preceding_register_plus_variable_external_history_v1"
                if self.basis_event_read_mode in (
                    "history_attention",
                    "history_indexed",
                    "history_relation_indexed",
                )
                else "preceding_register_plus_bounded_event_window_v1"
            ),
        }

    def add_instruction(self, instruction: ExternalRegisterInstruction) -> int:
        """Append one learned program datum without resizing the machine."""

        if instruction.instruction_width != self.instruction_width:
            raise ValueError("instruction width does not match the machine")
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def add_basis_slot(self, basis: ExternalRegisterComputeBasis | None = None) -> int:
        """Append fresh external computation capacity and return its address."""

        if basis is None:
            basis = ExternalRegisterComputeBasis(
                self.register_width,
                self.instruction_width,
                hidden=self.basis_hidden,
                microsteps=self.basis_microsteps,
                event_read_mode=self.basis_event_read_mode,
                register_input_mode=self.basis_register_input_mode,
                history_query_mode=self.basis_history_query_mode,
                history_age_slot_count=(
                    self.basis_history_age_slot_count
                    if self.basis_event_read_mode
                    in ("history_indexed", "history_relation_indexed")
                    else None
                ),
                history_relation_width=(
                    self.basis_history_relation_width
                    if self.basis_event_read_mode == "history_relation_indexed"
                    else None
                ),
                event_width=(
                    self.event_width
                    if self.event_window_size
                or self.basis_event_read_mode
                    in (
                        "history_attention",
                        "history_indexed",
                        "history_relation_indexed",
                    )
                    else 0
                ),
                event_window_size=self.event_window_size,
            )
        if (
            basis.register_width != self.register_width
            or basis.instruction_width != self.instruction_width
            or basis.event_window_size != self.event_window_size
            or basis.microsteps != self.basis_microsteps
            or basis.event_read_mode != self.basis_event_read_mode
            or basis.register_input_mode != self.basis_register_input_mode
            or basis.history_query_mode != self.basis_history_query_mode
            or basis.history_age_slot_count != self.basis_history_age_slot_count
            or basis.history_relation_width != self.basis_history_relation_width
            or (
                (
                    self.event_window_size
                    or self.basis_event_read_mode
                    in (
                        "history_attention",
                        "history_indexed",
                        "history_relation_indexed",
                    )
                )
                and basis.event_width != self.event_width
            )
        ):
            raise ValueError("basis slot dimensions do not match the machine")
        self.basis_slots.append(basis)
        return len(self.basis_slots) - 1

    def _basis_artifact_configuration(self) -> dict[str, int | str]:
        """Return the ABI expected by every slot in this interpreter."""

        configuration: dict[str, int | str] = {
            "schema": EXTERNAL_REGISTER_BASIS_SCHEMA,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "hidden": self.basis_hidden,
            "event_width": (
                self.event_width
                if self.event_window_size
                or self.basis_event_read_mode
                in (
                    "history_attention",
                    "history_indexed",
                    "history_relation_indexed",
                )
                else 0
            ),
            "event_window_size": self.event_window_size,
            "microsteps": self.basis_microsteps,
            "event_read_mode": self.basis_event_read_mode,
            "register_input_mode": self.basis_register_input_mode,
            "storage": "append_only_external_compute_slot_v1",
            "signature": "one_opaque_learned_slot_key_v1",
        }
        if self.basis_event_read_mode in (
            "history_attention",
            "history_indexed",
            "history_relation_indexed",
        ):
            configuration["history_query_mode"] = self.basis_history_query_mode
            if self.basis_event_read_mode == "history_attention":
                configuration["history_contract"] = (
                    "variable_external_history_attention_v3"
                    if self.basis_history_query_mode == "instruction_current_event"
                    else "variable_external_history_attention_v2"
                )
                configuration["history_head_count"] = (
                    self.basis_slots[0].history_head_count
                    if self.basis_slots
                    else ExternalRegisterComputeBasis.HISTORY_HEAD_COUNT
                )
            elif self.basis_event_read_mode == "history_indexed":
                configuration["history_contract"] = (
                    "variable_external_history_indexed_v1"
                )
                configuration["history_age_slot_count"] = (
                    self.basis_slots[0].history_age_slot_count
                    if self.basis_slots
                    else self.basis_history_age_slot_count
                )
            else:
                configuration["history_contract"] = (
                    "variable_external_history_relation_indexed_v1"
                )
                configuration["history_age_slot_count"] = (
                    self.basis_slots[0].history_age_slot_count
                    if self.basis_slots
                    else self.basis_history_age_slot_count
                )
                configuration["history_relation_width"] = (
                    self.basis_slots[0].history_relation_width
                    if self.basis_slots
                    else self.basis_history_relation_width
                )
        return configuration

    def basis_artifact(self, basis_slot: int) -> ExternalRegisterComputeBasisArtifact:
        """Export one slot without exposing or copying shared interpreter state."""

        if not 0 <= basis_slot < len(self.basis_slots):
            raise IndexError("basis slot index is out of range")
        return self.basis_slots[basis_slot].artifact()

    def add_basis_artifact(self, artifact: ExternalRegisterComputeBasisArtifact) -> int:
        """Append a verified external slot restored from a portable file."""

        basis = ExternalRegisterComputeBasis.from_artifact(
            artifact,
            expected_configuration=self._basis_artifact_configuration(),
        )
        return self.add_basis_slot(basis)

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

        if any(
            index < 0 or index >= len(self.basis_slots) for index in candidate_outcomes
        ):
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
            index < 0 or index >= len(self.basis_slots) for index in candidate_indices
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
                    event.shape[0],
                    self.event_window_size,
                    self.event_width,
                    device=event.device,
                    dtype=event.dtype,
                ),
                torch.zeros(
                    event.shape[0],
                    self.event_window_size,
                    device=event.device,
                    dtype=torch.bool,
                ),
            )
        if not self.event_window_size:
            return state.event_window, state.event_window_mask
        shifted = torch.cat((state.event_window[:, 1:], event.unsqueeze(1)), dim=1)
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
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        state_bank: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
                or instruction_code.shape != (register.shape[0], self.instruction_width)
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
            if isinstance(
                sequence_operator_memory, BoundExternalSequenceOperatorMemory
            ):
                if (
                    sequence_operator_slot is not None
                    or sequence_operator_route_query is not None
                ):
                    raise ValueError(
                        "bound sequence operator memory already contains its route"
                    )
            elif (sequence_operator_slot is None) == (
                sequence_operator_route_query is None
            ):
                raise ValueError(
                    "sequence operator memory requires exactly one slot or route query"
                )
            if self.operator_mode not in (
                "factorized_protected_meta",
                "factorized_protected_bounded_meta",
            ):
                raise ValueError(
                    "sequence operator memory requires protected-meta mode"
                )
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
        if self.operator_mode == EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE:
            normalized = self.operator_normalizer(register)
            projected = torch.einsum(
                "bd,kid->bki", normalized, self.operator_basis_right
            )
            proposals = torch.einsum(
                "bki,kri->bkr", projected, self.operator_basis_left
            )
            proposals = proposals + self.operator_basis_bias.unsqueeze(0)
            mixture = torch.softmax(self.operator_basis_selector(code), dim=-1)
            proposal = torch.einsum("bk,bkr->br", mixture, proposals)
            gate = torch.sigmoid(self.operator_composition_gate(code))
            return register + gate * torch.tanh(proposal)
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
            EXTERNAL_REGISTER_SHARED_OPERATOR_BASIS_MODE,
        ):
            if not 0 <= basis_slot < len(self.basis_slots):
                raise ValueError("basis slot index is out of range")
            return self.basis_slots[basis_slot](
                register,
                code,
                event_window=event_window,
                event_window_mask=event_window_mask,
                event_history=event_history,
                event_history_mask=event_history_mask,
                event_history_age=event_history_age,
                current_event=current_event,
            )
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
                if self.operator_mode
                in (
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
            if isinstance(
                sequence_operator_memory, BoundExternalSequenceOperatorMemory
            ):
                operator_memory_residual = sequence_operator_memory.residual(
                    register, code
                )
            elif sequence_operator_route_query is not None:
                operator_memory_residual = sequence_operator_memory.routed_residual(
                    sequence_operator_route_query, register, code
                )
            elif sequence_operator_memory is not None:
                operator_memory_residual = sequence_operator_memory.residual(
                    sequence_operator_slot, register, code
                )
            else:
                operator_memory_residual = 0.0
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
                meta_projected = torch.einsum("br,bkr->bk", meta_register, meta_right)
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
                        base
                        + meta_residual
                        + memory_residual
                        + operator_memory_residual
                    )
                residual = 0.5 * meta_gate * torch.tanh(meta_proposal)
                if meta_context is not None:
                    residual = residual + 0.5 * torch.tanh(meta_context)
                return (
                    register
                    + base_proposal
                    + self.operator_bias(code)
                    + residual
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
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute opaque program codes through the shared interpreter."""

        executed, _trace = self.execute_code_chain_trace(
            register,
            program_codes,
            event_window=event_window,
            event_window_mask=event_window_mask,
            event_history=event_history,
            event_history_mask=event_history_mask,
            event_history_age=event_history_age,
            current_event=current_event,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        return executed

    def execute_fragment_composition(
        self,
        register: torch.Tensor,
        composition: object,
        *,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute a fragment-bank chain while ignoring only padding.

        ``composition`` is intentionally accepted at the shared execution
        seam rather than importing a bank into the interpreter.  The bank
        owns addressing and growth; this machine owns execution.  Each row
        can contain a different number of instructions, so padding is removed
        before the opaque chain reaches the interpreter.
        """

        return self.execute_fragment_composition_trace(
            register,
            composition,
            event_window=event_window,
            event_window_mask=event_window_mask,
            event_history=event_history,
            event_history_mask=event_history_mask,
            event_history_age=event_history_age,
            current_event=current_event,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        ).final_state

    def execute_fragment_composition_trace(
        self,
        register: torch.Tensor,
        composition: object,
        *,
        include_codes: bool = False,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> object:
        """Execute a fragment chain and retain an opaque execution trace.

        ``include_codes`` opt-in adds materialized learned instruction tokens
        and transition deltas to the trace.  It never adds fragment indices,
        raw events, or verifier metadata to the learner-facing contract.
        """

        from .fragments import (
            ExternalSkillFragmentComposition,
            ExternalSkillFragmentExecutionTrace,
        )

        if not isinstance(composition, ExternalSkillFragmentComposition):
            raise TypeError(
                "fragment composition must use the versioned composition contract"
            )
        composition.validate(
            batch_size=register.shape[0],
            instruction_width=self.instruction_width,
            fragment_count=composition.bank_fragment_count,
        )
        if register.ndim != 2 or register.shape[1] != self.register_width:
            raise ValueError("register has the wrong shape for fragment composition")
        # Group rows by executable length so variable-length files retain their
        # exact semantics without forcing the interpreter through one Python
        # call per batch item.  The grouping is transport-only; no group ID or
        # row identity is exposed in the returned learner-facing trace.
        row_lengths = composition.mask.sum(dim=1).to(dtype=torch.int64)
        if bool((row_lengths < 1).any()):
            raise ValueError("fragment composition cannot be empty")
        rows_by_length: dict[int, list[int]] = {}
        for row, length in enumerate(row_lengths.detach().cpu().tolist()):
            rows_by_length.setdefault(int(length), []).append(row)
        traces: list[torch.Tensor | None] = [None] * register.shape[0]
        deltas: list[torch.Tensor | None] = [None] * register.shape[0]
        code_rows: list[torch.Tensor | None] = [None] * register.shape[0]
        for length, rows in rows_by_length.items():
            row_ids = torch.tensor(rows, dtype=torch.int64, device=register.device)
            codes = torch.stack(
                [composition.codes[row][composition.mask[row]] for row in rows]
            )
            if codes.shape[1] != length:
                raise ValueError("fragment composition mask must select valid codes")
            _executed, trace = self.execute_code_chain_trace(
                register.index_select(0, row_ids),
                codes,
                event_window=(
                    event_window.index_select(0, row_ids)
                    if event_window is not None
                    else None
                ),
                event_window_mask=(
                    event_window_mask.index_select(0, row_ids)
                    if event_window_mask is not None
                    else None
                ),
                event_history=(
                    event_history.index_select(0, row_ids)
                    if event_history is not None
                    else None
                ),
                event_history_mask=(
                    event_history_mask.index_select(0, row_ids)
                    if event_history_mask is not None
                    else None
                ),
                event_history_age=(
                    event_history_age.index_select(0, row_ids)
                    if event_history_age is not None
                    else None
                ),
                current_event=(
                    current_event.index_select(0, row_ids)
                    if current_event is not None
                    else None
                ),
                meta_context=(
                    meta_context.index_select(0, row_ids)
                    if meta_context is not None
                    else None
                ),
                sequence_operator_memory=sequence_operator_memory,
                sequence_operator_slot=sequence_operator_slot,
                sequence_operator_route_query=(
                    sequence_operator_route_query.index_select(0, row_ids)
                    if sequence_operator_route_query is not None
                    else None
                ),
            )
            grouped_states = torch.stack(trace, dim=1)
            for local, row in enumerate(rows):
                row_states = grouped_states[local]
                traces[row] = row_states
                if include_codes:
                    previous = torch.cat(
                        (register[row : row + 1], row_states[:-1]), dim=0
                    )
                    deltas[row] = row_states - previous
                    code_rows[row] = codes[local]
        if any(value is None for value in traces):
            raise RuntimeError("fragment execution did not produce every trace row")
        resolved_traces = [value for value in traces if value is not None]
        if include_codes and (
            any(value is None for value in deltas)
            or any(value is None for value in code_rows)
        ):
            raise RuntimeError("rich fragment execution did not produce every row")
        resolved_deltas = [value for value in deltas if value is not None]
        resolved_code_rows = [value for value in code_rows if value is not None]
        states = torch.nn.utils.rnn.pad_sequence(resolved_traces, batch_first=True)
        mask = torch.nn.utils.rnn.pad_sequence(
            [
                torch.ones(trace.shape[0], dtype=torch.bool, device=trace.device)
                for trace in resolved_traces
            ],
            batch_first=True,
            padding_value=False,
        )
        instruction_codes = (
            torch.nn.utils.rnn.pad_sequence(resolved_code_rows, batch_first=True)
            if include_codes
            else None
        )
        transition_deltas = (
            torch.nn.utils.rnn.pad_sequence(resolved_deltas, batch_first=True)
            if include_codes
            else None
        )
        if include_codes and composition.fragment_step_counts is None:
            raise ValueError("rich fragment traces require fragment segment counts")
        from .fragments import (
            EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA,
            EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA,
        )

        return ExternalSkillFragmentExecutionTrace(
            states=states,
            mask=mask,
            fragment_indices=composition.fragment_indices,
            route_scores=composition.route_scores,
            bank_fragment_count=composition.bank_fragment_count,
            schema=(
                EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA
                if include_codes
                else EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA
            ),
            instruction_codes=instruction_codes,
            transition_deltas=transition_deltas,
            fragment_step_counts=(
                composition.fragment_step_counts if include_codes else None
            ),
        ).validate(
            batch_size=register.shape[0],
            register_width=self.register_width,
        )

    def execute_code_chain_trace(
        self,
        register: torch.Tensor,
        program_codes: torch.Tensor,
        *,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Execute opaque program codes and retain an opaque state trace.

        The trace is positional execution evidence only.  It is useful for
        verifier probes and debugging, but it is never fed back into the
        controller as privileged program semantics.
        """

        if (
            program_codes.ndim != 3
            or program_codes.shape[0] != register.shape[0]
            or program_codes.shape[1] < 1
            or program_codes.shape[2] != self.instruction_width
        ):
            raise ValueError("program codes have the wrong shape for execution")
        trace: list[torch.Tensor] = []
        for code in program_codes.transpose(0, 1):
            register = self.execute(
                register,
                instruction_code=code,
                event_window=event_window,
                event_window_mask=event_window_mask,
                event_history=event_history,
                event_history_mask=event_history_mask,
                event_history_age=event_history_age,
                current_event=current_event,
                meta_context=meta_context,
                sequence_operator_memory=sequence_operator_memory,
                sequence_operator_slot=sequence_operator_slot,
                sequence_operator_route_query=sequence_operator_route_query,
            )
            trace.append(register)
        return register, tuple(trace)

    def execute_artifact(
        self,
        register: torch.Tensor,
        artifact: ExternalProgramArtifact,
        *,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one ABI-validated portable program artifact."""

        if not isinstance(artifact, ExternalProgramArtifact):
            raise TypeError("register execution requires an external program artifact")
        artifact.validate_for(
            instruction_width=self.instruction_width,
            interpreter_schema=EXTERNAL_REGISTER_SCHEMA,
            execution_schema=EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
        )
        return self.execute_code_chain(
            register,
            artifact.codes.unsqueeze(0).expand(register.shape[0], -1, -1),
            event_window=event_window,
            event_window_mask=event_window_mask,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )

    def execute_artifact_trace(
        self,
        register: torch.Tensor,
        artifact: ExternalProgramArtifact,
        *,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Execute a portable artifact while retaining its state trace."""

        if not isinstance(artifact, ExternalProgramArtifact):
            raise TypeError("register execution requires an external program artifact")
        artifact.validate_for(
            instruction_width=self.instruction_width,
            interpreter_schema=EXTERNAL_REGISTER_SCHEMA,
            execution_schema=EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
        )
        return self.execute_code_chain_trace(
            register,
            artifact.codes.unsqueeze(0).expand(register.shape[0], -1, -1),
            event_window=event_window,
            event_window_mask=event_window_mask,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )

    def read_execute_artifact_snapshot(
        self,
        *,
        event: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor,
        intention: IntentEvent,
        state: ExternalRegisterState,
        artifact: ExternalProgramArtifact,
        present: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> ExternalExecutionSnapshot:
        """Observe once and execute one external file copy-on-write.

        The observed register is durable external state.  The executed
        register and trace are a transient candidate result, so a failed
        verifier can discard them without mutating the durable state.
        """

        if present is None:
            present = torch.ones(event.shape[0], dtype=torch.bool, device=event.device)
        register, observed = self.observe_register(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
        )
        executed, trace = self.execute_artifact_trace(
            register,
            artifact,
            event_window=observed.event_window,
            event_window_mask=observed.event_window_mask,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        trace = tuple(
            torch.where(present.unsqueeze(-1), value, register) for value in trace
        )
        snapshot = ExternalExecutionSnapshot(
            observed=observed,
            executed=torch.where(present.unsqueeze(-1), executed, register),
            trace=trace,
            program_digest=artifact.digest(),
        )
        return snapshot.validate(
            batch_size=event.shape[0],
            register_width=self.register_width,
            context_width=self.context_width,
            event_width=self.event_width,
            event_window_size=self.event_window_size,
            program_length=artifact.program_length,
        )

    def execute_chain(
        self,
        register: torch.Tensor,
        instructions: Iterable[ExternalRegisterInstruction],
        *,
        basis_slots: Iterable[int | None] | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
            event_history=event_history,
            event_history_mask=event_history_mask,
            event_history_age=event_history_age,
            current_event=current_event,
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
        event_history: torch.Tensor | None = None,
        event_history_mask: torch.Tensor | None = None,
        event_history_age: torch.Tensor | None = None,
        current_event: torch.Tensor | None = None,
        meta_context: torch.Tensor | None = None,
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
            (None,) * len(selected) if basis_slots is None else tuple(basis_slots)
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
                event_history=event_history,
                event_history_mask=event_history_mask,
                event_history_age=event_history_age,
                current_event=current_event,
                meta_context=meta_context,
                sequence_operator_memory=sequence_operator_memory,
                sequence_operator_slot=sequence_operator_slot,
                sequence_operator_route_query=sequence_operator_route_query,
                state_bank=(
                    torch.stack(states, dim=1)
                    if states
                    and self.operator_mode == EXTERNAL_REGISTER_SHARED_BANKED_MODE
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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
            present = torch.ones(event.shape[0], dtype=torch.bool, device=event.device)
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
                self.register_writer(torch.cat((context, state.register), dim=-1)),
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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
            present = torch.ones(event.shape[0], dtype=torch.bool, device=event.device)
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
        return torch.where(present.unsqueeze(-1), executed, register), next_state

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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState, tuple[torch.Tensor, ...]]:
        """Read context, execute, and return an opaque intermediate-state bank."""
        if present is None:
            present = torch.ones(event.shape[0], dtype=torch.bool, device=event.device)
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
            torch.where(present.unsqueeze(-1), value, register) for value in trace
        )
        return torch.where(present.unsqueeze(-1), executed, register), next_state, trace

    def read_execute_register_snapshot(
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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
        program_digest: str | None = None,
    ) -> ExternalExecutionSnapshot:
        """Return one typed observe/execute snapshot without mutating memory."""

        selected = self.instructions if instructions is None else tuple(instructions)
        executed, observed, trace = self.read_execute_register_trace(
            event=event,
            action=action,
            outcome=outcome,
            intention=intention,
            state=state,
            present=present,
            instructions=selected,
            basis_slots=basis_slots,
            meta_context=meta_context,
            sequence_operator_memory=sequence_operator_memory,
            sequence_operator_slot=sequence_operator_slot,
            sequence_operator_route_query=sequence_operator_route_query,
        )
        snapshot = ExternalExecutionSnapshot(
            observed=observed,
            executed=executed,
            trace=trace,
            program_digest=program_digest,
        )
        return snapshot.validate(
            batch_size=event.shape[0],
            register_width=self.register_width,
            context_width=self.context_width,
            event_width=self.event_width,
            event_window_size=self.event_window_size,
            program_length=len(selected),
        )

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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
        sequence_operator_slot: int | None = None,
        sequence_operator_route_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExternalRegisterState, tuple[torch.Tensor, ...]]:
        """Observe, execute, and expose the learned role bank per step."""

        if present is None:
            present = torch.ones(event.shape[0], dtype=torch.bool, device=event.device)
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
        return (
            torch.where(present.unsqueeze(-1), executed, register),
            next_state,
            role_trace,
        )

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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
            present = torch.ones(event.shape[0], dtype=torch.bool, device=event.device)
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
        register = torch.where(present.unsqueeze(-1), executed, observed_state.register)
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
        sequence_operator_memory: ExternalSequenceOperatorMemory
        | BoundExternalSequenceOperatorMemory
        | None = None,
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
