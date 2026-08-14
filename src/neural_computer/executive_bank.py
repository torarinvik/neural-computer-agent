"""Durable, checksummed banks for self-contained external-executive programs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Literal

import torch

from .executive import (
    EXECUTIVE_PROGRAM_SCHEMA,
    ExecutiveInstruction,
    ExternalAmodalExecutive,
    ExternalExecutiveOperatorRegistry,
    ExternalExecutiveProgram,
)
from .executive_memory import ExternalValueDelayOperator
from .executive_operators import (
    ExternalEvidenceBinaryIntentionOperator,
    ExternalSingletonEventValueOperator,
    ExternalValueEqualityEvidenceOperator,
)
from .program import (
    ExternalProgramAdmissionReceipt,
    evaluate_program_digest_admission,
)
from .temporal_program import AGENT_BANK_EXTENSION

EXECUTIVE_OPERATOR_SPEC_SCHEMA = "neural-computer.executive-operator-spec.v1"
EXECUTIVE_PROGRAM_ARTIFACT_SCHEMA = "neural-computer.executive-program-artifact.v1"
EXECUTIVE_COMPOSITION_SCHEMA = "neural-computer.executive-composition.v1"
EXTERNAL_EXECUTIVE_PROGRAM_BANK_SCHEMA = (
    "neural-computer.external-executive-program-bank.v1"
)

OperatorKind = Literal[
    "singleton_event_value",
    "value_delay",
    "value_equality_evidence",
    "evidence_binary_intention",
]


def _validate_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ExternalExecutiveOperatorSpec:
    """One allow-listed operator constructor with no arbitrary import path."""

    handle: int
    kind: OperatorKind
    width: int | None = None
    delay: int | None = None
    schema: str = EXECUTIVE_OPERATOR_SPEC_SCHEMA

    def validate(self) -> ExternalExecutiveOperatorSpec:
        if self.schema != EXECUTIVE_OPERATOR_SPEC_SCHEMA:
            raise ValueError("unsupported executive operator spec schema")
        if not isinstance(self.handle, int) or isinstance(self.handle, bool) or self.handle < 0:
            raise ValueError("executive operator spec handle is invalid")
        if self.kind not in {
            "singleton_event_value",
            "value_delay",
            "value_equality_evidence",
            "evidence_binary_intention",
        }:
            raise ValueError("executive operator spec kind is not allow-listed")
        if self.kind in {"singleton_event_value", "value_delay"}:
            if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width < 1:
                raise ValueError("executive operator spec width is invalid")
        elif self.width is not None:
            raise ValueError("this executive operator spec cannot set width")
        if self.kind == "value_delay":
            if not isinstance(self.delay, int) or isinstance(self.delay, bool) or self.delay < 1:
                raise ValueError("executive value delay is invalid")
        elif self.delay is not None:
            raise ValueError("this executive operator spec cannot set delay")
        return self

    def build(self):
        self.validate()
        if self.kind == "singleton_event_value":
            assert self.width is not None
            return ExternalSingletonEventValueOperator(self.handle, width=self.width)
        if self.kind == "value_delay":
            assert self.width is not None and self.delay is not None
            return ExternalValueDelayOperator(
                self.handle, width=self.width, delay=self.delay
            )
        if self.kind == "value_equality_evidence":
            return ExternalValueEqualityEvidenceOperator(self.handle)
        return ExternalEvidenceBinaryIntentionOperator(self.handle)

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "handle": self.handle,
            "kind": self.kind,
            "width": self.width,
            "delay": self.delay,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalExecutiveOperatorSpec:
        if not isinstance(payload, dict):
            raise TypeError("executive operator spec payload must be a dictionary")
        return cls(
            handle=payload.get("handle"),
            kind=payload.get("kind"),
            width=payload.get("width"),
            delay=payload.get("delay"),
            schema=payload.get("schema"),
        ).validate()


def _instruction_payload(instruction: ExecutiveInstruction) -> dict[str, object]:
    return {
        "op": instruction.op,
        "source": instruction.source,
        "destination": instruction.destination,
        "operator_handle": instruction.operator_handle,
        "arguments": list(instruction.arguments),
        "true_target": instruction.true_target,
        "false_target": instruction.false_target,
        "unknown_target": instruction.unknown_target,
        "next_target": instruction.next_target,
    }


@dataclass(frozen=True)
class ExternalExecutiveProgramArtifact:
    """Portable executable program plus a safe, complete operator manifest."""

    program: ExternalExecutiveProgram
    operator_specs: tuple[ExternalExecutiveOperatorSpec, ...]
    intention_width: int
    schema: str = EXECUTIVE_PROGRAM_ARTIFACT_SCHEMA

    def validate(self) -> ExternalExecutiveProgramArtifact:
        if self.schema != EXECUTIVE_PROGRAM_ARTIFACT_SCHEMA:
            raise ValueError("unsupported executive program artifact schema")
        self.program.validate()
        if self.program.schema != EXECUTIVE_PROGRAM_SCHEMA or self.intention_width < 1:
            raise ValueError("executive program artifact dimensions are invalid")
        handles = tuple(spec.handle for spec in self.operator_specs)
        if handles != tuple(sorted(handles)) or len(handles) != len(set(handles)):
            raise ValueError("executive artifact operator handles must be sorted and unique")
        registry = ExternalExecutiveOperatorRegistry(
            tuple(spec.validate().build() for spec in self.operator_specs)
        )
        for instruction in self.program.instructions:
            if instruction.op == "call":
                assert instruction.operator_handle is not None
                registry.operator(instruction.operator_handle)
        return self

    def registry(self) -> ExternalExecutiveOperatorRegistry:
        self.validate()
        return ExternalExecutiveOperatorRegistry(
            tuple(spec.build() for spec in self.operator_specs)
        )

    def instantiate(
        self, *, max_instructions_per_tick: int = 128
    ) -> ExternalAmodalExecutive:
        self.validate()
        return ExternalAmodalExecutive(
            self.program,
            self.registry(),
            intention_width=self.intention_width,
            max_instructions_per_tick=max_instructions_per_tick,
        )

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "program": {
                "schema": self.program.schema,
                "slot_count": self.program.slot_count,
                "instructions": [
                    _instruction_payload(instruction)
                    for instruction in self.program.instructions
                ],
            },
            "operator_specs": [spec.payload() for spec in self.operator_specs],
            "intention_width": self.intention_width,
        }

    @cached_property
    def _cached_digest(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                self._content_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def digest(self) -> str:
        return self._cached_digest

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalExecutiveProgramArtifact:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported executive program artifact payload")
        program_payload = payload.get("program")
        specs_payload = payload.get("operator_specs")
        if not isinstance(program_payload, dict) or not isinstance(specs_payload, list):
            raise TypeError("executive program artifact payload is malformed")
        instructions_payload = program_payload.get("instructions")
        if not isinstance(instructions_payload, list) or not all(
            isinstance(item, dict) for item in instructions_payload
        ):
            raise TypeError("executive program instructions are malformed")
        instructions = tuple(
            ExecutiveInstruction(
                op=item.get("op"),
                source=item.get("source"),
                destination=item.get("destination"),
                operator_handle=item.get("operator_handle"),
                arguments=tuple(item.get("arguments", ())),
                true_target=item.get("true_target"),
                false_target=item.get("false_target"),
                unknown_target=item.get("unknown_target"),
                next_target=item.get("next_target"),
            )
            for item in instructions_payload
        )
        artifact = cls(
            program=ExternalExecutiveProgram(
                slot_count=program_payload.get("slot_count"),
                instructions=instructions,
                schema=program_payload.get("schema"),
            ),
            operator_specs=tuple(
                ExternalExecutiveOperatorSpec.from_payload(item)
                for item in specs_payload
            ),
            intention_width=payload.get("intention_width"),
            schema=payload.get("schema"),
        ).validate()
        if payload.get("sha256") != artifact.digest():
            raise ValueError("executive program artifact checksum mismatch")
        return artifact


def compose_executive_artifacts(
    artifacts: Sequence[ExternalExecutiveProgramArtifact],
    *,
    share_compatible_operators: bool = False,
    final_emit_only: bool = False,
) -> ExternalExecutiveProgramArtifact:
    """Compose existing executable files into one sequential artifact.

    Each component keeps its own workspace namespace. By default it also keeps
    its own operator namespace; ``share_compatible_operators=True`` instead
    gives components with identical allow-listed operator specifications one
    explicit state namespace, which lets a temporal prelude hand history to a
    later component.
    The component terminal ``HALT`` is removed for every non-final component,
    and all branch/WAIT/EMIT targets that pointed at that terminal transfer to
    the next component.  This permits a finite prelude to hand off to a
    persistent game-loop skill while preserving stateful operator isolation.
    With ``final_emit_only=True``, an intermediate EMIT becomes an internal
    HANDOFF, so a prelude can process the same learned event without producing
    an extra external action.
    No task identity or verifier-private label enters the resulting artifact.
    """

    if not artifacts:
        raise ValueError("executive composition needs at least one artifact")
    if not isinstance(share_compatible_operators, bool):
        raise TypeError("executive composition operator sharing must be boolean")
    if not isinstance(final_emit_only, bool):
        raise TypeError("executive composition final emit policy must be boolean")
    validated = tuple(artifact.validate() for artifact in artifacts)
    if any(artifact.schema != EXECUTIVE_PROGRAM_ARTIFACT_SCHEMA for artifact in validated):
        raise ValueError("executive composition artifacts use incompatible schemas")
    intention_width = validated[0].intention_width
    if any(artifact.intention_width != intention_width for artifact in validated):
        raise ValueError("executive composition needs a common intention width")
    if any(
        not executive_artifact_can_handoff(artifact) for artifact in validated[:-1]
    ):
        raise ValueError(
            "every non-final executive composition component must reach its terminal handoff"
        )
    body_lengths = tuple(len(artifact.program.instructions) - 1 for artifact in validated)
    if any(length < 1 for length in body_lengths):
        raise ValueError("executive composition components need a non-terminal body")

    program_offsets: list[int] = []
    slot_offsets: list[int] = []
    handle_offsets: list[int] = []
    handle_maps: list[dict[int, int]] = []
    body_offset = 0
    slot_offset = 0
    handle_offset = 0
    shared_handles: dict[tuple[object, ...], int] = {}
    for artifact, body_length in zip(validated, body_lengths, strict=True):
        program_offsets.append(body_offset)
        slot_offsets.append(slot_offset)
        handle_offsets.append(handle_offset)
        body_offset += body_length
        slot_offset += artifact.program.slot_count
        current_map: dict[int, int] = {}
        for spec in artifact.operator_specs:
            if share_compatible_operators:
                key = (spec.kind, spec.width, spec.delay, spec.schema)
                global_handle = shared_handles.get(key)
                if global_handle is None:
                    global_handle = handle_offset
                    shared_handles[key] = global_handle
                    handle_offset += 1
            else:
                global_handle = handle_offset + spec.handle
            current_map[spec.handle] = global_handle
        handle_maps.append(current_map)
        if not share_compatible_operators:
            maximum_handle = max(
                (spec.handle for spec in artifact.operator_specs), default=-1
            )
            handle_offset += maximum_handle + 1

    instructions: list[ExecutiveInstruction] = []
    composed_specs: list[ExternalExecutiveOperatorSpec] = []
    emitted_shared_keys: set[tuple[object, ...]] = set()
    for index, (artifact, program_offset, slots, _handles) in enumerate(
        zip(validated, program_offsets, slot_offsets, handle_offsets, strict=True)
    ):
        terminal = len(artifact.program.instructions) - 1
        next_component = (
            program_offsets[index + 1]
            if index + 1 < len(validated)
            else body_offset
        )

        def relocate_target(
            target: int,
            *,
            terminal: int = terminal,
            next_component: int = next_component,
            program_offset: int = program_offset,
            is_non_final: bool = index + 1 < len(validated),
        ) -> int:
            if target == terminal and is_non_final:
                return next_component
            return program_offset + target

        for instruction in artifact.program.instructions[:-1]:
            intermediate_handoff = (
                final_emit_only
                and index + 1 < len(validated)
                and instruction.op == "emit"
            )
            operation = "handoff" if intermediate_handoff else instruction.op
            instructions.append(
                ExecutiveInstruction(
                    op=operation,
                    source=(
                        None
                        if intermediate_handoff or instruction.source is None
                        else instruction.source + slots
                    ),
                    destination=(
                        None
                        if intermediate_handoff or instruction.destination is None
                        else instruction.destination + slots
                    ),
                    operator_handle=(
                        None
                        if intermediate_handoff or instruction.operator_handle is None
                        else handle_maps[index][instruction.operator_handle]
                    ),
                    arguments=(
                        ()
                        if intermediate_handoff
                        else tuple(argument + slots for argument in instruction.arguments)
                    ),
                    true_target=(
                        None
                        if intermediate_handoff or instruction.true_target is None
                        else relocate_target(instruction.true_target)
                    ),
                    false_target=(
                        None
                        if intermediate_handoff or instruction.false_target is None
                        else relocate_target(instruction.false_target)
                    ),
                    unknown_target=(
                        None
                        if intermediate_handoff or instruction.unknown_target is None
                        else relocate_target(instruction.unknown_target)
                    ),
                    next_target=(
                        None
                        if instruction.next_target is None
                        else relocate_target(instruction.next_target)
                    ),
                )
            )
        for spec in artifact.operator_specs:
            if share_compatible_operators:
                key = (spec.kind, spec.width, spec.delay, spec.schema)
                if key in emitted_shared_keys:
                    continue
                emitted_shared_keys.add(key)
            composed_specs.append(replace(spec, handle=handle_maps[index][spec.handle]))
    instructions.append(ExecutiveInstruction("halt"))
    return ExternalExecutiveProgramArtifact(
        program=ExternalExecutiveProgram(
            slot_count=slot_offset,
            instructions=tuple(instructions),
            schema=EXECUTIVE_PROGRAM_SCHEMA,
        ),
        operator_specs=tuple(sorted(composed_specs, key=lambda spec: spec.handle)),
        intention_width=intention_width,
    ).validate()


@lru_cache(maxsize=4096)
def executive_artifact_can_handoff(
    artifact: ExternalExecutiveProgramArtifact,
) -> bool:
    """Return whether some control-flow path reaches the terminal HALT."""

    program = artifact.validate().program
    terminal = len(program.instructions) - 1
    pending = [0]
    reachable: set[int] = set()
    while pending:
        pointer = pending.pop()
        if pointer in reachable:
            continue
        reachable.add(pointer)
        if pointer == terminal:
            return True
        instruction = program.instructions[pointer]
        if instruction.op == "branch":
            pending.extend(
                (
                    int(instruction.true_target),
                    int(instruction.false_target),
                    int(instruction.unknown_target),
                )
            )
        elif instruction.op in {"wait", "emit"}:
            pending.append(
                pointer + 1
                if instruction.next_target is None
                else instruction.next_target
            )
        elif instruction.op != "halt":
            pending.append(pointer + 1)
    return False


class ExternalExecutiveProgramBank:
    """Append-only verified executive artifacts in a resumable `.bank` file."""

    schema = EXTERNAL_EXECUTIVE_PROGRAM_BANK_SCHEMA

    def __init__(
        self,
        *,
        controller_digest: str,
        capacity: int = 16,
    ) -> None:
        if capacity < 1:
            raise ValueError("executive program bank capacity must be positive")
        self.controller_digest = _validate_digest(
            controller_digest, name="controller_digest"
        )
        self.capacity = int(capacity)
        self._artifacts: list[ExternalExecutiveProgramArtifact] = []
        self._admissions: list[dict[str, Any]] = []
        self._version = 0

    @property
    def program_count(self) -> int:
        return len(self._artifacts)

    @property
    def version(self) -> int:
        return self._version

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "controller_digest": self.controller_digest,
            "capacity": self.capacity,
            "storage": "append_only_verified_self_contained_executive_artifacts_v1",
        }

    def artifact(self, slot: int) -> ExternalExecutiveProgramArtifact:
        if not isinstance(slot, int) or not 0 <= slot < self.program_count:
            raise IndexError("executive program bank slot is outside the bank")
        return ExternalExecutiveProgramArtifact.from_payload(
            self._artifacts[slot].payload()
        )

    def admit(
        self,
        artifact: ExternalExecutiveProgramArtifact,
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ExternalProgramAdmissionReceipt:
        artifact.validate()
        receipt = evaluate_program_digest_admission(
            artifact.digest(),
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if not receipt.accepted:
            return receipt
        duplicate = next(
            (
                index
                for index, current in enumerate(self._artifacts)
                if current.digest() == artifact.digest()
            ),
            None,
        )
        if duplicate is None:
            if self.program_count >= self.capacity:
                return replace(
                    receipt,
                    accepted=False,
                    reason="verified executive artifact could not enter a full bank",
                ).validate()
            slot = self.program_count
            self._artifacts.append(
                ExternalExecutiveProgramArtifact.from_payload(artifact.payload())
            )
        else:
            slot = duplicate
        committed = replace(
            receipt,
            slot=slot,
            reason=(
                "verified executive artifact committed"
                if duplicate is None
                else "verified experience attached to an identical executive artifact"
            ),
        ).validate()
        self._admissions.append(committed.payload())
        self._version += 1
        return committed

    def executable(
        self,
        slot: int,
        *,
        controller_digest: str,
        max_instructions_per_tick: int = 128,
    ) -> ExternalAmodalExecutive:
        if _validate_digest(controller_digest, name="controller_digest") != self.controller_digest:
            raise ValueError("executive program bank controller digest is incompatible")
        return self.artifact(slot).instantiate(
            max_instructions_per_tick=max_instructions_per_tick
        )

    def digest(self) -> str:
        content = {
            "configuration": self.configuration(),
            "version": self.version,
            "artifact_digests": [artifact.digest() for artifact in self._artifacts],
            "admissions": self._admissions,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "version": self.version,
            "artifacts": [artifact.payload() for artifact in self._artifacts],
            "admissions": list(self._admissions),
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalExecutiveProgramBank:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported executive program bank schema")
        configuration = payload.get("configuration")
        artifacts = payload.get("artifacts")
        admissions = payload.get("admissions")
        version = payload.get("version")
        if (
            not isinstance(configuration, dict)
            or not isinstance(artifacts, list)
            or not isinstance(admissions, list)
            or not isinstance(version, int)
            or version < 0
        ):
            raise TypeError("executive program bank payload is malformed")
        bank = cls(
            controller_digest=configuration.get("controller_digest"),
            capacity=configuration.get("capacity"),
        )
        if bank.configuration() != configuration:
            raise ValueError("executive program bank configuration mismatch")
        bank._artifacts = [
            ExternalExecutiveProgramArtifact.from_payload(item) for item in artifacts
        ]
        if len(bank._artifacts) > bank.capacity:
            raise ValueError("executive program bank exceeds capacity")
        if not all(isinstance(item, dict) for item in admissions):
            raise TypeError("executive program bank admissions are malformed")
        bank._admissions = [dict(item) for item in admissions]
        bank._version = version
        if payload.get("sha256") != bank.digest():
            raise ValueError("executive program bank checksum mismatch")
        return bank

    def save_bank(self, path: Path) -> None:
        path = Path(path)
        if path.suffix != AGENT_BANK_EXTENSION:
            raise ValueError(
                f"canonical agent banks must use the {AGENT_BANK_EXTENSION} extension"
            )
        _atomic_text_write(
            path,
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")) + "\n",
        )
        _atomic_text_write(
            path.with_suffix(path.suffix + ".sha256"),
            _sha256_file(path) + "\n",
        )

    @classmethod
    def load_bank(cls, path: Path) -> ExternalExecutiveProgramBank:
        path = Path(path)
        if path.suffix != AGENT_BANK_EXTENSION:
            raise ValueError(
                f"canonical agent banks must use the {AGENT_BANK_EXTENSION} extension"
            )
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.is_file():
            raise ValueError("executive program bank checksum is missing")
        if checksum_path.read_text().strip() != _sha256_file(path):
            raise ValueError("executive program bank file checksum mismatch")
        return cls.from_payload(json.loads(path.read_text()))


def build_temporal_equality_executive_artifact(
    *,
    event_width: int,
    delay: int,
) -> ExternalExecutiveProgramArtifact:
    """Compose a reusable temporal relation from generic allow-listed operators.

    The positive displacement is external learned binding state. The artifact
    contains no n-back label, modality, symbol vocabulary, or device action.
    """

    specs = (
        ExternalExecutiveOperatorSpec(1, "singleton_event_value", width=event_width),
        ExternalExecutiveOperatorSpec(
            2, "value_delay", width=event_width, delay=delay
        ),
        ExternalExecutiveOperatorSpec(3, "value_equality_evidence"),
        ExternalExecutiveOperatorSpec(4, "evidence_binary_intention"),
    )
    program = ExternalExecutiveProgram(
        5,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction(
                "call", destination=1, operator_handle=1, arguments=(0,)
            ),
            ExecutiveInstruction(
                "call", destination=2, operator_handle=2, arguments=(1,)
            ),
            ExecutiveInstruction(
                "call", destination=3, operator_handle=3, arguments=(1, 2)
            ),
            ExecutiveInstruction(
                "branch",
                source=3,
                true_target=5,
                false_target=5,
                unknown_target=7,
            ),
            ExecutiveInstruction(
                "call", destination=4, operator_handle=4, arguments=(3,)
            ),
            ExecutiveInstruction("emit", source=4, next_target=0),
            ExecutiveInstruction("wait", next_target=0),
            ExecutiveInstruction("halt"),
        ),
    )
    return ExternalExecutiveProgramArtifact(program, specs, 2).validate()


__all__ = [
    "EXECUTIVE_COMPOSITION_SCHEMA",
    "EXECUTIVE_OPERATOR_SPEC_SCHEMA",
    "EXECUTIVE_PROGRAM_ARTIFACT_SCHEMA",
    "EXTERNAL_EXECUTIVE_PROGRAM_BANK_SCHEMA",
    "ExternalExecutiveOperatorSpec",
    "ExternalExecutiveProgramArtifact",
    "ExternalExecutiveProgramBank",
    "build_temporal_equality_executive_artifact",
    "compose_executive_artifacts",
    "executive_artifact_can_handoff",
]
