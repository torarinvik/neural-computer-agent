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
EXTERNAL_REGISTER_BASIS_SCHEMA = "neural-computer.external-register-compute-basis.v1"
EXTERNAL_REGISTER_COMPATIBILITY_SCHEMA = (
    "neural-computer.external-register-compatibility-prior.v1"
)


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
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, hidden) < 1:
            raise ValueError("compute basis dimensions must be positive")
        if min(event_width, event_window_size) < 0:
            raise ValueError("event window dimensions must be non-negative")
        if event_window_size and event_width < 1:
            raise ValueError("event width must be positive for an event window")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.hidden = int(hidden)
        self.event_width = int(event_width)
        self.event_window_size = int(event_window_size)
        self.event_window_width = self.event_width * self.event_window_size
        width = self.register_width + self.instruction_width + self.event_window_width
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
            features = torch.cat((register, code, window.flatten(1)), dim=-1)
        else:
            if event_window is not None or event_window_mask is not None:
                raise ValueError("event window is unsupported by this compute basis")
            features = torch.cat((register, code), dim=-1)
        return register + torch.sigmoid(self.gate(features)) * torch.tanh(
            self.network(features)
        )


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
        event_window_size: int = 0,
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
        ) < 1:
            raise ValueError("external register dimensions must be positive")
        if event_window_size < 0:
            raise ValueError("event window size must be non-negative")
        if operator_mode not in (
            "factorized_low_rank",
            "factorized_film",
            "factorized_hybrid",
            "factorized_bounded_residual",
            "factorized_protected_meta",
            "unconstrained_mlp",
        ):
            raise ValueError("unsupported external register operator mode")
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
        self.event_window_size = int(event_window_size)
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
        if operator_mode == "factorized_bounded_residual":
            # Serial instruction chains are sensitive to unbounded additive
            # drift.  Normalize the read state, bound the learned proposal,
            # and let the opaque instruction choose a feature-wise residual
            # gate.  The register remains the only downstream input.
            self.operator_normalizer = nn.LayerNorm(register_width)
            self.operator_composition_gate = nn.Linear(
                instruction_width, register_width
            )
            nn.init.zeros_(self.operator_composition_gate.bias)
        if operator_mode == "factorized_protected_meta":
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
            "event_window_size": self.event_window_size,
            "state": "external_working_register_with_recurrent_context_v2",
            "execution": "shared_interpreter_serial_instruction_chain_v1",
            "compute_basis": EXTERNAL_REGISTER_BASIS_SCHEMA,
            "basis_binding": "opaque_memory_side_slot_index_v1",
            "read_execute": EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
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
                event_width=self.event_width if self.event_window_size else 0,
                event_window_size=self.event_window_size,
            )
        if (
            basis.register_width != self.register_width
            or basis.instruction_width != self.instruction_width
            or basis.event_window_size != self.event_window_size
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
        return (
            torch.where(shifted_mask.unsqueeze(-1), shifted, state.event_window),
            shifted_mask,
        )

    def execute(
        self,
        register: torch.Tensor,
        instruction: ExternalRegisterInstruction,
        *,
        basis_slot: int | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
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
        if basis_slot is not None:
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
        ):
            operator_register = (
                self.operator_normalizer(register)
                if self.operator_mode == "factorized_bounded_residual"
                else register
            )
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
            if self.operator_mode == "factorized_low_rank":
                return register + base_proposal + self.operator_bias(code)
            if self.operator_mode == "factorized_bounded_residual":
                proposal = torch.tanh(base_proposal + self.operator_bias(code))
                gate = torch.sigmoid(self.operator_composition_gate(code))
                return register + gate * proposal
            if self.operator_mode == "factorized_protected_meta":
                meta_left = torch.tanh(self.operator_meta_left(code)).reshape(
                    register.shape[0], self.register_width, self.operator_rank
                )
                meta_right = torch.tanh(self.operator_meta_right(code)).reshape(
                    register.shape[0], self.operator_rank, self.register_width
                )
                meta_projected = torch.einsum(
                    "br,bkr->bk", register, meta_right
                )
                meta_proposal = torch.einsum(
                    "bk,brk->br", meta_projected, meta_left
                ) + self.operator_meta_bias(code)
                meta_gate = torch.sigmoid(self.operator_meta_gate(code))
                return register + base_proposal + self.operator_bias(code) + (
                    0.5 * meta_gate * torch.tanh(meta_proposal)
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

    def execute_chain(
        self,
        register: torch.Tensor,
        instructions: Iterable[ExternalRegisterInstruction],
        *,
        basis_slots: Iterable[int | None] | None = None,
        event_window: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a memory-selected instruction chain to a working register."""

        selected = tuple(instructions)
        selected_basis = (
            (None,) * len(selected)
            if basis_slots is None
            else tuple(basis_slots)
        )
        if len(selected_basis) != len(selected):
            raise ValueError("basis slot bindings must match instruction count")
        for instruction, basis_slot in zip(selected, selected_basis):
            register = self.execute(
                register,
                instruction,
                basis_slot=basis_slot,
                event_window=event_window,
                event_window_mask=event_window_mask,
            )
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
        )
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
        basis_slots: Iterable[int | None] | None = None,
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
        )
        return self.to_intention(register), next_state
