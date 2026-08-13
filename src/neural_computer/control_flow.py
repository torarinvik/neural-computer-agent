"""Generic external control-flow programs with a fail-closed executor.

The recipe boundary previously stored only straight-line instruction sequences.
This module adds the smallest generic control-flow substrate needed for
loop-like reusable computation: non-negative counters, increment/decrement,
unconditional and zero-tested jumps, and halt.  It is a two-counter-machine
style ABI, so the *potential* computation class is unbounded even though every
runtime execution is bounded by an explicit step budget and counter limit.

The executor is an external-memory component.  It does not inspect raw
modalities, task names, verifier answers, or device protocols.  Programs are
opaque durable files; scalar verifier outcomes are accepted only by the
separate admission boundary and are never persisted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

CONTROL_FLOW_INSTRUCTION_SCHEMA = (
    "neural-computer.external-control-flow-instruction.v1"
)
CONTROL_FLOW_PROGRAM_SCHEMA = "neural-computer.external-control-flow-program.v1"
CONTROL_FLOW_EXECUTION_SCHEMA = (
    "neural-computer.external-control-flow-execution.v1"
)
CONTROL_FLOW_MEMORY_SCHEMA = "neural-computer.external-control-flow-memory.v1"

ControlFlowOp = Literal[
    "inc",
    "dec",
    "jump",
    "jump_if_zero",
    "jump_if_nonzero",
    "halt",
]
ExecutionStatus = Literal[
    "halted",
    "step_budget_exhausted",
    "counter_limit",
    "counter_underflow",
]


def _digest_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ControlFlowInstruction:
    """One generic instruction in the external counter-machine ABI."""

    op: ControlFlowOp
    counter: int | None = None
    target: int | None = None

    def validate(self, *, counter_count: int, program_length: int) -> None:
        if counter_count < 2:
            raise ValueError("control-flow programs need at least two counters")
        if program_length < 1:
            raise ValueError("control-flow programs need at least one instruction")
        if self.op in ("inc", "dec"):
            if self.counter is None or not 0 <= self.counter < counter_count:
                raise ValueError("counter instruction has an invalid counter")
            if self.target is not None:
                raise ValueError("counter instruction cannot carry a jump target")
            return
        if self.op == "halt":
            if self.counter is not None or self.target is not None:
                raise ValueError("halt cannot carry operands")
            return
        if self.op == "jump":
            if self.counter is not None:
                raise ValueError("unconditional jump cannot carry a counter")
        elif self.op in ("jump_if_zero", "jump_if_nonzero"):
            if self.counter is None or not 0 <= self.counter < counter_count:
                raise ValueError("conditional jump has an invalid counter")
        else:
            raise ValueError(f"unsupported control-flow operation: {self.op!r}")
        if self.target is None or not 0 <= self.target < program_length:
            raise ValueError("jump target is outside the program")

    def payload(self) -> dict[str, object]:
        return {
            "schema": CONTROL_FLOW_INSTRUCTION_SCHEMA,
            "op": self.op,
            "counter": self.counter,
            "target": self.target,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowInstruction:
        if not isinstance(payload, dict):
            raise TypeError("control-flow instruction payload must be a mapping")
        if payload.get("schema") != CONTROL_FLOW_INSTRUCTION_SCHEMA:
            raise ValueError("unsupported control-flow instruction schema")
        return cls(
            op=payload.get("op"),
            counter=(None if payload.get("counter") is None else int(payload["counter"])),
            target=(None if payload.get("target") is None else int(payload["target"])),
        )


@dataclass(frozen=True)
class ControlFlowExecution:
    """Result of one bounded external execution."""

    counters: tuple[int, ...]
    instruction_pointer: int
    steps: int
    status: ExecutionStatus
    trace: tuple[tuple[int, tuple[int, ...]], ...] = ()
    schema: str = CONTROL_FLOW_EXECUTION_SCHEMA

    def validate(self, *, counter_count: int) -> ControlFlowExecution:
        if self.schema != CONTROL_FLOW_EXECUTION_SCHEMA:
            raise ValueError("unsupported control-flow execution schema")
        if len(self.counters) != counter_count or any(
            not isinstance(value, int) or value < 0 for value in self.counters
        ):
            raise ValueError("control-flow execution counters are invalid")
        if self.instruction_pointer < 0 or self.steps < 0:
            raise ValueError("control-flow execution positions are invalid")
        if self.status not in (
            "halted",
            "step_budget_exhausted",
            "counter_limit",
            "counter_underflow",
        ):
            raise ValueError("control-flow execution status is invalid")
        for pointer, state in self.trace:
            if pointer < 0 or len(state) != counter_count:
                raise ValueError("control-flow execution trace is invalid")
        return self


@dataclass(frozen=True)
class ControlFlowProgram:
    """A portable counter-machine program stored outside the controller."""

    counter_count: int
    instructions: tuple[ControlFlowInstruction, ...]
    schema: str = CONTROL_FLOW_PROGRAM_SCHEMA

    def validate(self) -> ControlFlowProgram:
        if self.schema != CONTROL_FLOW_PROGRAM_SCHEMA:
            raise ValueError("unsupported control-flow program schema")
        if self.counter_count < 2:
            raise ValueError("control-flow programs need at least two counters")
        if not self.instructions:
            raise ValueError("control-flow programs need at least one instruction")
        for instruction in self.instructions:
            instruction.validate(
                counter_count=self.counter_count,
                program_length=len(self.instructions),
            )
        if self.instructions[-1].op != "halt":
            raise ValueError("control-flow programs must end with halt")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "counter_count": self.counter_count,
            "instructions": [instruction.payload() for instruction in self.instructions],
        }

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowProgram:
        if not isinstance(payload, dict):
            raise TypeError("control-flow program payload must be a mapping")
        instructions = payload.get("instructions")
        if not isinstance(instructions, list):
            raise TypeError("control-flow program instructions must be a list")
        return cls(
            counter_count=int(payload.get("counter_count", -1)),
            instructions=tuple(
                ControlFlowInstruction.from_payload(instruction)
                for instruction in instructions
            ),
            schema=payload.get("schema"),
        ).validate()

    def digest(self) -> str:
        return _digest_payload(self.payload())

    def execute(
        self,
        counters: Sequence[int],
        *,
        max_steps: int,
        max_counter: int = 1_000_000,
        trace_limit: int = 0,
    ) -> ControlFlowExecution:
        """Execute with explicit resource bounds and no implicit wraparound."""

        self.validate()
        if len(counters) != self.counter_count or any(
            not isinstance(value, int) or value < 0 for value in counters
        ):
            raise ValueError("initial counters are invalid")
        if max_steps < 1 or max_counter < 1:
            raise ValueError("execution bounds must be positive")
        state = list(counters)
        pointer = 0
        trace: list[tuple[int, tuple[int, ...]]] = []
        for steps in range(max_steps):
            instruction = self.instructions[pointer]
            if len(trace) < trace_limit:
                trace.append((pointer, tuple(state)))
            if instruction.op == "halt":
                return ControlFlowExecution(
                    tuple(state), pointer, steps + 1, "halted", tuple(trace)
                ).validate(counter_count=self.counter_count)
            if instruction.op == "inc":
                assert instruction.counter is not None
                state[instruction.counter] += 1
                if state[instruction.counter] > max_counter:
                    return ControlFlowExecution(
                        tuple(state), pointer, steps + 1, "counter_limit", tuple(trace)
                    ).validate(counter_count=self.counter_count)
                pointer += 1
            elif instruction.op == "dec":
                assert instruction.counter is not None
                if state[instruction.counter] == 0:
                    return ControlFlowExecution(
                        tuple(state), pointer, steps + 1, "counter_underflow", tuple(trace)
                    ).validate(counter_count=self.counter_count)
                state[instruction.counter] -= 1
                pointer += 1
            elif instruction.op == "jump":
                assert instruction.target is not None
                pointer = instruction.target
            else:
                assert instruction.counter is not None and instruction.target is not None
                condition = state[instruction.counter] == 0
                if instruction.op == "jump_if_nonzero":
                    condition = not condition
                pointer = instruction.target if condition else pointer + 1
            if pointer >= len(self.instructions):
                raise RuntimeError("validated control-flow program fell off its end")
        return ControlFlowExecution(
            tuple(state), pointer, max_steps, "step_budget_exhausted", tuple(trace)
        ).validate(counter_count=self.counter_count)


def insert_control_flow_instruction(
    program: ControlFlowProgram,
    position: int,
    instruction: ControlFlowInstruction,
) -> ControlFlowProgram:
    """Insert one non-terminal instruction while relocating jump targets.

    Instruction pointers are part of the external program ABI.  A structural
    insertion therefore shifts every existing target at or after the edit;
    silently leaving targets attached to their old numeric positions would
    corrupt loop control flow while still producing a syntactically valid file.
    """

    program.validate()
    if not 0 <= position < len(program.instructions):
        raise ValueError("control-flow insertion position is invalid")
    if instruction.op == "halt":
        raise ValueError("control-flow insertion cannot add a terminal halt")
    new_length = len(program.instructions) + 1
    instruction.validate(
        counter_count=program.counter_count,
        program_length=new_length,
    )

    def relocate(existing: ControlFlowInstruction) -> ControlFlowInstruction:
        if existing.op not in {"jump", "jump_if_zero", "jump_if_nonzero"}:
            return existing
        assert existing.target is not None
        target = existing.target + int(existing.target >= position)
        return ControlFlowInstruction(
            existing.op,
            counter=existing.counter,
            target=target,
        )

    instructions = tuple(relocate(existing) for existing in program.instructions)
    return ControlFlowProgram(
        program.counter_count,
        (*instructions[:position], instruction, *instructions[position:]),
    ).validate()


def delete_control_flow_instruction(
    program: ControlFlowProgram,
    position: int,
) -> ControlFlowProgram:
    """Delete one non-terminal instruction and relocate surviving targets.

    Deletion fails closed when another instruction targets the removed position;
    inventing a successor for that edge would be a semantic edit rather than a
    structural deletion and could silently change a learned program.
    """

    program.validate()
    if not 0 <= position < len(program.instructions) - 1:
        raise ValueError("control-flow deletion position is invalid")

    def relocate(existing: ControlFlowInstruction) -> ControlFlowInstruction:
        if existing.op not in {"jump", "jump_if_zero", "jump_if_nonzero"}:
            return existing
        assert existing.target is not None
        if existing.target == position:
            raise ValueError("control-flow deletion would remove a jump target")
        target = existing.target - int(existing.target > position)
        return ControlFlowInstruction(
            existing.op,
            counter=existing.counter,
            target=target,
        )

    instructions = tuple(
        relocate(existing)
        for index, existing in enumerate(program.instructions)
        if index != position
    )
    return ControlFlowProgram(program.counter_count, instructions).validate()


def compose_control_flow_programs(
    programs: Sequence[ControlFlowProgram],
) -> ControlFlowProgram:
    """Materialize a sequential composition with relocated jump targets.

    Each component is an external file with one terminal ``halt``. The
    terminal halt transfers to the next component, while the final terminal
    halt becomes the composition's single halt. Internal halts are rejected
    because they would make the declared sequence silently skip later files.
    The returned program is ordinary control-flow ABI data; no task-specific
    composition operator enters the controller.
    """

    if not programs:
        raise ValueError("control-flow composition needs at least one program")
    validated = tuple(program.validate() for program in programs)
    counter_count = validated[0].counter_count
    if any(program.counter_count != counter_count for program in validated):
        raise ValueError("composed control-flow programs need a common counter width")
    if any(
        instruction.op == "halt"
        for program in validated
        for instruction in program.instructions[:-1]
    ):
        raise ValueError("composed control-flow programs cannot contain internal halt")

    body_lengths = tuple(len(program.instructions) - 1 for program in validated)
    offsets: list[int] = []
    offset = 0
    for length in body_lengths:
        offsets.append(offset)
        offset += length
    total_body_length = offset
    instructions: list[ControlFlowInstruction] = []
    for program_index, program in enumerate(validated):
        component_offset = offsets[program_index]
        next_target = (
            offsets[program_index + 1]
            if program_index + 1 < len(validated)
            else total_body_length
        )
        terminal_index = len(program.instructions) - 1
        for instruction in program.instructions[:-1]:
            if instruction.op not in {"jump", "jump_if_zero", "jump_if_nonzero"}:
                instructions.append(instruction)
                continue
            assert instruction.target is not None
            target = (
                next_target
                if instruction.target == terminal_index
                else component_offset + instruction.target
            )
            instructions.append(
                ControlFlowInstruction(
                    instruction.op,
                    counter=instruction.counter,
                    target=target,
                )
            )
    instructions.append(ControlFlowInstruction("halt"))
    return ControlFlowProgram(counter_count, tuple(instructions)).validate()


def splice_control_flow_program(
    program: ControlFlowProgram,
    position: int,
    fragment: ControlFlowProgram,
) -> ControlFlowProgram:
    """Insert a verified multi-instruction fragment into a program.

    Both files use the same counter ABI and terminate with ``HALT``.  The
    fragment's terminal exits to the original instruction at ``position``;
    its internal jump targets are rebased, while every parent edge after the
    insertion boundary shifts by the fragment body length. Edges targeting the
    boundary itself enter the inserted fragment, preserving the usual meaning
    of inserting a block before an instruction.
    This is a structural transformation only: no task-specific macro or
    controller branch is introduced.
    """

    program.validate()
    fragment.validate()
    if program.counter_count != fragment.counter_count:
        raise ValueError("spliced control-flow programs need a common counter width")
    if not 0 <= position <= len(program.instructions) - 1:
        raise ValueError("control-flow splice position is invalid")
    fragment_body = fragment.instructions[:-1]
    if not fragment_body:
        raise ValueError("control-flow splice fragment needs a non-terminal body")
    shift = len(fragment_body)
    fragment_terminal = len(fragment.instructions) - 1
    continuation = position + shift

    def shift_parent_target(instruction: ControlFlowInstruction) -> ControlFlowInstruction:
        if instruction.op not in {"jump", "jump_if_zero", "jump_if_nonzero"}:
            return instruction
        assert instruction.target is not None
        target = instruction.target + (shift if instruction.target > position else 0)
        return ControlFlowInstruction(
            instruction.op,
            counter=instruction.counter,
            target=target,
        )

    def rebase_fragment_target(
        instruction: ControlFlowInstruction,
    ) -> ControlFlowInstruction:
        if instruction.op not in {"jump", "jump_if_zero", "jump_if_nonzero"}:
            return instruction
        assert instruction.target is not None
        target = (
            continuation
            if instruction.target == fragment_terminal
            else position + instruction.target
        )
        return ControlFlowInstruction(
            instruction.op,
            counter=instruction.counter,
            target=target,
        )

    rebased_fragment = tuple(rebase_fragment_target(item) for item in fragment_body)
    rebased_parent = tuple(shift_parent_target(item) for item in program.instructions)
    return ControlFlowProgram(
        program.counter_count,
        (
            *rebased_parent[:position],
            *rebased_fragment,
            *rebased_parent[position:],
        ),
    ).validate()


@dataclass(frozen=True)
class ControlFlowAdmissionReceipt:
    accepted: bool
    slot: int | None
    stable_bits_to_threshold: int | None
    reason: str


def evaluate_control_flow_admission(
    program: ControlFlowProgram,
    outcomes: Sequence[float],
    *,
    threshold: float = 1.0,
    min_observations: int = 1,
    min_stable_observations: int = 1,
) -> ControlFlowAdmissionReceipt:
    """Evaluate scalar admission without mutating external program memory."""

    program.validate()
    values = tuple(float(value) for value in outcomes)
    if not values or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise ValueError("control-flow outcomes must be finite probabilities")
    if (
        not 0.0 < threshold <= 1.0
        or min_observations < 1
        or min_stable_observations < 1
    ):
        raise ValueError("control-flow admission thresholds are invalid")
    stable = None
    for index in range(len(values)):
        if (
            len(values) - index >= min_stable_observations
            and min(values[index:]) >= threshold
        ):
            stable = index + 1
            break
    accepted = len(values) >= min_observations and stable is not None
    return ControlFlowAdmissionReceipt(
        accepted,
        None,
        stable,
        (
            "control-flow verifier prefix did not remain above threshold"
            if not accepted
            else "control-flow program passed a stable verifier prefix"
        ),
    )


class ControlFlowProgramMemory:
    """Checksummed external files with protected slots and scalar admission."""

    schema = CONTROL_FLOW_MEMORY_SCHEMA

    def __init__(self, counter_count: int) -> None:
        if counter_count < 2:
            raise ValueError("control-flow memory needs at least two counters")
        self.counter_count = int(counter_count)
        self._programs: list[ControlFlowProgram] = []
        self._protected: list[bool] = []

    @property
    def file_count(self) -> int:
        return len(self._programs)

    def add_program(self, program: ControlFlowProgram, *, protect: bool = False) -> int:
        program.validate()
        if program.counter_count != self.counter_count:
            raise ValueError("control-flow program counter width is incompatible")
        self._programs.append(program)
        self._protected.append(bool(protect))
        return len(self._programs) - 1

    def program(self, slot: int) -> ControlFlowProgram:
        if not 0 <= slot < self.file_count:
            raise IndexError("control-flow memory slot is out of range")
        return self._programs[slot]

    def protect_file(self, slot: int) -> None:
        if not 0 <= slot < self.file_count:
            raise IndexError("control-flow memory slot is out of range")
        self._protected[slot] = True

    def is_file_protected(self, slot: int) -> bool:
        if not 0 <= slot < self.file_count:
            raise IndexError("control-flow memory slot is out of range")
        return self._protected[slot]

    def compose(self, slots: Sequence[int]) -> ControlFlowProgram:
        """Materialize a sequential composition from existing opaque files."""

        if not slots:
            raise ValueError("control-flow composition needs at least one slot")
        normalized: list[int] = []
        for slot in slots:
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise TypeError("control-flow composition slots must be integers")
            if not 0 <= slot < self.file_count:
                raise IndexError("control-flow composition slot is out of range")
            normalized.append(slot)
        return compose_control_flow_programs(
            tuple(self.program(slot) for slot in normalized)
        )

    def splice(
        self,
        parent_slot: int,
        position: int,
        fragment_slot: int,
    ) -> ControlFlowProgram:
        """Insert one existing opaque file into another at ``position``."""

        for name, slot in (
            ("parent", parent_slot),
            ("fragment", fragment_slot),
        ):
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise TypeError(f"control-flow {name} slot must be an integer")
            if not 0 <= slot < self.file_count:
                raise IndexError(f"control-flow {name} slot is out of range")
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError("control-flow splice position must be an integer")
        return splice_control_flow_program(
            self.program(parent_slot),
            position,
            self.program(fragment_slot),
        )

    def compose_verified(
        self,
        slots: Sequence[int],
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        protect: bool = False,
    ) -> ControlFlowAdmissionReceipt:
        """Compose existing files and admit the result through scalar evidence."""

        return self.admit_verified(
            self.compose(slots),
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
            protect=protect,
        )

    def splice_verified(
        self,
        parent_slot: int,
        position: int,
        fragment_slot: int,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        protect: bool = False,
    ) -> ControlFlowAdmissionReceipt:
        """Splice existing files and admit the result through scalar evidence."""

        return self.admit_verified(
            self.splice(parent_slot, position, fragment_slot),
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
            protect=protect,
        )

    def admit_verified(
        self,
        program: ControlFlowProgram,
        outcomes: Sequence[float],
        *,
        threshold: float = 1.0,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        protect: bool = False,
    ) -> ControlFlowAdmissionReceipt:
        program.validate()
        if program.counter_count != self.counter_count:
            raise ValueError("control-flow program counter width is incompatible")
        receipt = evaluate_control_flow_admission(
            program,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if not receipt.accepted:
            return receipt
        slot = self.add_program(program, protect=protect)
        return ControlFlowAdmissionReceipt(
            True,
            slot,
            receipt.stable_bits_to_threshold,
            "control-flow program admitted after stable verifier prefix",
        )

    def payload(self) -> dict[str, object]:
        body = {
            "schema": self.schema,
            "counter_count": self.counter_count,
            "programs": [program.payload() for program in self._programs],
            "protected": list(self._protected),
        }
        return {**body, "sha256": _digest_payload(body)}

    @classmethod
    def from_payload(cls, payload: object) -> ControlFlowProgramMemory:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported control-flow memory payload")
        unsigned = {key: value for key, value in payload.items() if key != "sha256"}
        if payload.get("sha256") != _digest_payload(unsigned):
            raise ValueError("control-flow memory checksum mismatch")
        programs = payload.get("programs")
        protected = payload.get("protected")
        if not isinstance(programs, list) or not isinstance(protected, list):
            raise TypeError("control-flow memory payload is incomplete")
        if len(programs) != len(protected):
            raise ValueError("control-flow memory metadata lengths differ")
        memory = cls(int(payload.get("counter_count", -1)))
        for raw, is_protected in zip(programs, protected, strict=True):
            if not isinstance(is_protected, bool):
                raise TypeError("control-flow protected metadata must be boolean")
            memory.add_program(ControlFlowProgram.from_payload(raw), protect=is_protected)
        return memory

    def digest(self) -> str:
        return str(self.payload()["sha256"])


__all__ = [
    "CONTROL_FLOW_EXECUTION_SCHEMA",
    "CONTROL_FLOW_INSTRUCTION_SCHEMA",
    "CONTROL_FLOW_MEMORY_SCHEMA",
    "CONTROL_FLOW_PROGRAM_SCHEMA",
    "ControlFlowAdmissionReceipt",
    "ControlFlowExecution",
    "ControlFlowInstruction",
    "ControlFlowProgram",
    "ControlFlowProgramMemory",
    "compose_control_flow_programs",
    "delete_control_flow_instruction",
    "evaluate_control_flow_admission",
    "insert_control_flow_instruction",
    "splice_control_flow_program",
]
