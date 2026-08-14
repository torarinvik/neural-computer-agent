"""Persistent, outcome-routed external temporal programs.

The bank stores immutable learned instruction tensors beside a frozen
controller.  It receives only opaque learned context vectors and scalar
verifier outcomes: no task names, rule IDs, correct unattempted actions, or
modality fields are part of its interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from .addressing import PersistentOpaqueContextRouteEvidence
from .program import (
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
    evaluate_external_program_admission,
)

if TYPE_CHECKING:
    from .live import ResolvedLiveOutcome

EXTERNAL_TEMPORAL_PROGRAM_BANK_SCHEMA = (
    "neural-computer.external-temporal-program-bank.v1"
)
AGENT_BANK_EXTENSION = ".bank"
DEFAULT_AGENT_BANK_FILENAME = "AgentBrain.bank"
TEMPORAL_ADDRESS_INTERPRETER_SCHEMA = (
    "neural-computer.temporal-address-controller.v1"
)
TEMPORAL_ADDRESS_EXECUTION_SCHEMA = (
    "neural-computer.relative-history-select.v1"
)
TEMPORAL_ADDRESS_OUTPUT_SCHEMA = "neural-computer.amodal-intention.v1"
RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA = (
    "neural-computer.recursive-temporal-controller.v1"
)
RECURSIVE_TEMPORAL_EXECUTION_SCHEMA = (
    "neural-computer.relative-history-compose.v1"
)


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
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TemporalProgramSelection:
    """One logged memory-side program choice and its exact propensity."""

    slot: int
    propensity: float
    context: torch.Tensor
    artifact: ExternalProgramArtifact
    bank_version: int


def recursive_temporal_primitive(
    artifact: ExternalProgramArtifact,
) -> ExternalProgramArtifact:
    """Lift one legacy relative-address row into the recursive interpreter.

    At depth one the new interpreter is behaviorally identical to the legacy
    relative-history selector. The caller must still verify the resulting
    candidate before bank admission; this function assigns no semantic label.
    """

    artifact.validate_for(
        instruction_width=artifact.instruction_width,
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    if artifact.program_length != 1:
        raise ValueError("a recursive temporal primitive needs one legacy row")
    return ExternalProgramArtifact(
        codes=artifact.snapshot(),
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def compose_recursive_temporal_program(
    primitive: ExternalProgramArtifact, repetitions: int
) -> ExternalProgramArtifact:
    """Compose one verified relative-step primitive with itself in sequence."""

    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("recursive temporal repetitions must be positive")
    primitive.validate_for(
        instruction_width=primitive.instruction_width,
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    if primitive.program_length != 1:
        raise ValueError("recursive composition requires one primitive row")
    if repetitions > primitive.instruction_width:
        raise ValueError("recursive temporal depth exceeds fixed history capacity")
    return ExternalProgramArtifact(
        codes=primitive.snapshot().repeat(repetitions, 1),
        interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
        execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


class TemporalProgramOutcomeObserver:
    """Route attributed reward input to one currently selected program.

    Binding is external memory state. The observer ignores explicit
    missing-evidence closures and never receives a task identity or an
    unattempted program label.
    """

    def __init__(
        self,
        bank: ExternalTemporalProgramBank,
        selection: TemporalProgramSelection | None = None,
    ) -> None:
        self.bank = bank
        self.selection = selection
        self.unique_outcome_bits = 0

    def bind(self, selection: TemporalProgramSelection) -> None:
        if not isinstance(selection, TemporalProgramSelection):
            raise TypeError("temporal outcome observer needs a bank selection")
        if selection.artifact.digest() != self.bank.artifact(selection.slot).digest():
            raise ValueError("temporal outcome observer selection is not in its bank")
        self.selection = selection

    def clear(self) -> None:
        self.selection = None

    def observe(self, outcome: ResolvedLiveOutcome) -> None:
        selection = self.selection
        if selection is None:
            return
        present = outcome.event.present
        if present.numel() != 1:
            raise ValueError("temporal program route feedback currently requires batch one")
        if not bool(present.item()):
            return
        self.bank.observe(selection, outcome.event.reward.reshape(()))
        self.unique_outcome_bits += 1


class ExternalTemporalProgramBank:
    """Append-only verified programs with learned opaque-context routing.

    Candidate learning happens before :meth:`admit`.  Only a candidate whose
    ordered public verifier scores clear the stable-prefix gate enters this
    live bank.  Admission also teaches the external router that the attempted
    candidate worked for the supplied learned context.  Later selections can
    be reinforced or reversed through :meth:`observe` without changing the
    controller or any admitted instruction tensor.
    """

    schema = EXTERNAL_TEMPORAL_PROGRAM_BANK_SCHEMA

    def __init__(
        self,
        context_width: int,
        instruction_width: int,
        *,
        controller_digest: str,
        capacity: int = 16,
        matching_tolerance: float = 1e-4,
        generalization_tolerance: float = 0.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
        interpreter_schema: str = TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema: str = TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema: str = TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    ) -> None:
        if min(context_width, instruction_width, capacity) < 1:
            raise ValueError("temporal program bank dimensions must be positive")
        self.context_width = int(context_width)
        self.instruction_width = int(instruction_width)
        self.capacity = int(capacity)
        self.controller_digest = _validate_digest(
            controller_digest, name="controller_digest"
        )
        if not all(
            isinstance(value, str) and value
            for value in (interpreter_schema, execution_schema, output_schema)
        ):
            raise ValueError("temporal program interfaces must be non-empty")
        self.interpreter_schema = interpreter_schema
        self.execution_schema = execution_schema
        self.output_schema = output_schema
        self.router = PersistentOpaqueContextRouteEvidence(
            self.context_width,
            matching_tolerance=matching_tolerance,
            generalization_tolerance=generalization_tolerance,
            mastery_threshold=mastery_threshold,
            min_mastery_observations=min_mastery_observations,
        )
        self._artifacts: list[ExternalProgramArtifact] = []
        self._admissions: list[dict[str, Any]] = []
        self._version = 0

    @property
    def program_count(self) -> int:
        return len(self._artifacts)

    @property
    def version(self) -> int:
        return self._version

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "instruction_width": self.instruction_width,
            "capacity": self.capacity,
            "controller_digest": self.controller_digest,
            "interpreter_schema": self.interpreter_schema,
            "execution_schema": self.execution_schema,
            "output_schema": self.output_schema,
            "routing": "opaque_context_plus_attempted_scalar_outcome_v1",
            "storage": "append_only_verified_instruction_tensors_v1",
        }

    def _validate_context(self, context: torch.Tensor) -> torch.Tensor:
        if not isinstance(context, torch.Tensor):
            raise TypeError("temporal program context must be a learned event tensor")
        if context.shape != (self.context_width,):
            raise ValueError(
                f"temporal program context must have shape [{self.context_width}]"
            )
        if not bool(torch.isfinite(context).all()):
            raise ValueError("temporal program context must contain finite values")
        value = context.detach().to(device="cpu", dtype=torch.float32)
        if float(torch.linalg.vector_norm(value)) <= 1e-8:
            raise ValueError("temporal program context cannot be zero")
        return F.normalize(value, dim=0).contiguous()

    def _validate_artifact(self, artifact: ExternalProgramArtifact) -> None:
        artifact.validate_for(
            instruction_width=self.instruction_width,
            interpreter_schema=self.interpreter_schema,
            execution_schema=self.execution_schema,
            output_schema=self.output_schema,
        )
        if (
            self.execution_schema == TEMPORAL_ADDRESS_EXECUTION_SCHEMA
            and artifact.program_length != 1
        ):
            raise ValueError("temporal address programs must contain one address row")
        if (
            self.execution_schema == RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
            and artifact.program_length > self.instruction_width
        ):
            raise ValueError("recursive temporal program exceeds history capacity")

    def artifact(self, slot: int) -> ExternalProgramArtifact:
        if not isinstance(slot, int) or not 0 <= slot < self.program_count:
            raise IndexError("temporal program slot is outside the bank")
        source = self._artifacts[slot]
        return ExternalProgramArtifact(
            codes=source.snapshot(),
            interpreter_schema=source.interpreter_schema,
            execution_schema=source.execution_schema,
            output_schema=source.output_schema,
        )

    def admit(
        self,
        artifact: ExternalProgramArtifact,
        context: torch.Tensor,
        outcomes: torch.Tensor | Sequence[float],
        *,
        threshold: float = 0.8,
        min_observations: int = 8,
        min_stable_observations: int = 8,
    ) -> ExternalProgramAdmissionReceipt:
        """Verify a provisional file, then commit it and its route evidence."""

        self._validate_artifact(artifact)
        key = self._validate_context(context)
        receipt = evaluate_external_program_admission(
            artifact,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if not receipt.accepted:
            return receipt

        values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
        digest = artifact.digest()
        duplicate = next(
            (
                index
                for index, current in enumerate(self._artifacts)
                if current.digest() == digest
            ),
            None,
        )
        if duplicate is None:
            if self.program_count >= self.capacity:
                return replace(
                    receipt,
                    accepted=False,
                    reason="verified candidate could not enter a full append-only bank",
                ).validate()
            slot = self.router.append_slot()
            if slot != self.program_count:
                raise RuntimeError("temporal program and route slots diverged")
            self._artifacts.append(
                ExternalProgramArtifact(
                    codes=artifact.snapshot(),
                    interpreter_schema=artifact.interpreter_schema,
                    execution_schema=artifact.execution_schema,
                    output_schema=artifact.output_schema,
                )
            )
        else:
            slot = duplicate

        for outcome in values:
            self.router.observe(key, slot, outcome)
        committed = replace(
            receipt,
            slot=slot,
            reason=(
                "candidate verified and committed as an immutable external file"
                if duplicate is None
                else "verified experience attached to an identical external file"
            ),
        ).validate()
        self._admissions.append(committed.payload())
        self._version += 1
        return committed

    def select(
        self,
        context: torch.Tensor,
        *,
        exploration: float = 0.0,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> TemporalProgramSelection:
        """Retrieve one checksum-validated file from learned event evidence."""

        if self.program_count < 1:
            raise LookupError("temporal program bank is empty")
        key = self._validate_context(context)
        probabilities = self.router.behavior_probabilities(
            key.unsqueeze(0),
            exploration=exploration,
            strategy="balanced",
        )[0]
        slot = (
            int(torch.multinomial(probabilities, 1, generator=generator).item())
            if sample
            else int(probabilities.argmax().item())
        )
        artifact = self.artifact(slot)
        self._validate_artifact(artifact)
        return TemporalProgramSelection(
            slot=slot,
            propensity=float(probabilities[slot].item()),
            context=key,
            artifact=artifact,
            bank_version=self.version,
        )

    def observe(
        self,
        selection: TemporalProgramSelection,
        outcome: float | torch.Tensor,
    ) -> None:
        """Attach one public reward to the exact attempted external file."""

        if not isinstance(selection, TemporalProgramSelection):
            raise TypeError("temporal routing feedback needs a logged selection")
        if not 0 <= selection.slot < self.program_count:
            raise IndexError("logged temporal program slot is no longer present")
        if selection.artifact.digest() != self._artifacts[selection.slot].digest():
            raise ValueError("logged temporal program does not match the live bank")
        self.router.observe(selection.context, selection.slot, outcome)
        self._version += 1

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode())
        digest.update(
            json.dumps(self.configuration(), sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(str(self._version).encode())
        for artifact in self._artifacts:
            digest.update(artifact.digest().encode())
        digest.update(self.router.digest().encode())
        digest.update(
            json.dumps(self._admissions, sort_keys=True, separators=(",", ":")).encode()
        )
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "version": self.version,
            "artifacts": [artifact.payload() for artifact in self._artifacts],
            "router": self.router.payload(),
            "admissions": list(self._admissions),
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalTemporalProgramBank:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported external temporal program bank schema")
        configuration = payload.get("configuration")
        artifacts = payload.get("artifacts")
        router_payload = payload.get("router")
        admissions = payload.get("admissions")
        version = payload.get("version")
        if (
            not isinstance(configuration, dict)
            or not isinstance(artifacts, list)
            or not isinstance(router_payload, dict)
            or not isinstance(admissions, list)
            or not isinstance(version, int)
            or version < 0
        ):
            raise TypeError("external temporal program bank payload is malformed")
        bank = cls(
            int(configuration["context_width"]),
            int(configuration["instruction_width"]),
            controller_digest=str(configuration["controller_digest"]),
            capacity=int(configuration["capacity"]),
            matching_tolerance=float(router_payload["matching_tolerance"]),
            generalization_tolerance=float(router_payload["generalization_tolerance"]),
            mastery_threshold=float(router_payload["mastery_threshold"]),
            min_mastery_observations=int(router_payload["min_mastery_observations"]),
            interpreter_schema=str(configuration["interpreter_schema"]),
            execution_schema=str(configuration["execution_schema"]),
            output_schema=str(configuration["output_schema"]),
        )
        if bank.configuration() != configuration:
            raise ValueError("external temporal program bank configuration mismatch")
        bank._artifacts = [
            ExternalProgramArtifact.from_payload(item) for item in artifacts
        ]
        if len(bank._artifacts) > bank.capacity:
            raise ValueError("external temporal program bank exceeds capacity")
        for artifact in bank._artifacts:
            bank._validate_artifact(artifact)
        bank.router = PersistentOpaqueContextRouteEvidence.from_payload(router_payload)
        if bank.router.slot_count != len(bank._artifacts):
            raise ValueError("external temporal program routes do not align with files")
        if not all(isinstance(item, dict) for item in admissions):
            raise TypeError("external temporal program admissions must be records")
        bank._admissions = [dict(item) for item in admissions]
        bank._version = version
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != bank.digest():
            raise ValueError("external temporal program bank checksum mismatch")
        return bank

    def save(self, path: Path) -> None:
        """Atomically persist the bank with an independent file checksum."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(self.payload(), temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        _atomic_text_write(
            path.with_suffix(path.suffix + ".sha256"),
            _sha256_file(path) + "\n",
        )

    def save_bank(self, path: Path) -> None:
        """Persist one canonical, resumable ``.bank`` brain artifact."""

        path = Path(path)
        if path.suffix != AGENT_BANK_EXTENSION:
            raise ValueError(
                f"canonical agent banks must use the {AGENT_BANK_EXTENSION} extension"
            )
        self.save(path)

    @classmethod
    def load(cls, path: Path) -> ExternalTemporalProgramBank:
        path = Path(path)
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.is_file():
            raise ValueError("external temporal program bank checksum is missing")
        if checksum_path.read_text().strip() != _sha256_file(path):
            raise ValueError("external temporal program bank file checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_payload(payload)

    @classmethod
    def load_bank(cls, path: Path) -> ExternalTemporalProgramBank:
        """Load and validate one canonical, resumable ``.bank`` brain artifact."""

        path = Path(path)
        if path.suffix != AGENT_BANK_EXTENSION:
            raise ValueError(
                f"canonical agent banks must use the {AGENT_BANK_EXTENSION} extension"
            )
        return cls.load(path)


__all__ = [
    "AGENT_BANK_EXTENSION",
    "DEFAULT_AGENT_BANK_FILENAME",
    "EXTERNAL_TEMPORAL_PROGRAM_BANK_SCHEMA",
    "RECURSIVE_TEMPORAL_EXECUTION_SCHEMA",
    "RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA",
    "TEMPORAL_ADDRESS_EXECUTION_SCHEMA",
    "TEMPORAL_ADDRESS_INTERPRETER_SCHEMA",
    "TEMPORAL_ADDRESS_OUTPUT_SCHEMA",
    "ExternalTemporalProgramBank",
    "TemporalProgramOutcomeObserver",
    "TemporalProgramSelection",
    "compose_recursive_temporal_program",
    "recursive_temporal_primitive",
]
