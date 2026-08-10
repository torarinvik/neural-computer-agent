"""Versioned opaque artifacts for replaceable external programs.

The controller does not own these files and the artifact does not contain
task names, modality labels, protocol actions, or verifier-private answers.
It carries only learned executable instruction data and the independently
versioned interfaces required to validate a compatible runtime.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

EXTERNAL_PROGRAM_ARTIFACT_SCHEMA = "neural-computer.external-program-artifact.v1"
EXTERNAL_PROGRAM_ADMISSION_SCHEMA = (
    "neural-computer.external-program-admission.v1"
)
EXTERNAL_PROGRAM_MEMORY_TRANSACTION_SCHEMA = (
    "neural-computer.external-program-memory-transaction.v1"
)


@dataclass(frozen=True)
class ExternalProgramArtifact:
    """Portable learned instruction data for one generic external executor."""

    codes: torch.Tensor
    interpreter_schema: str
    execution_schema: str
    output_schema: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.codes, torch.Tensor):
            raise TypeError("program artifact codes must be a tensor")
        if (
            self.codes.ndim != 2
            or self.codes.shape[0] < 1
            or self.codes.shape[1] < 1
        ):
            raise ValueError("program artifact codes must have shape [steps, width]")
        if not bool(torch.isfinite(self.codes).all()):
            raise ValueError("program artifact codes must be finite")
        for name, value in (
            ("interpreter_schema", self.interpreter_schema),
            ("execution_schema", self.execution_schema),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty schema string")
        if self.output_schema is not None and (
            not isinstance(self.output_schema, str) or not self.output_schema
        ):
            raise ValueError("output_schema must be a non-empty schema string or null")

    @property
    def instruction_width(self) -> int:
        return int(self.codes.shape[1])

    @property
    def program_length(self) -> int:
        return int(self.codes.shape[0])

    def configuration(self) -> dict[str, Any]:
        """Return metadata without exposing the learned tensor contents."""

        return {
            "schema": EXTERNAL_PROGRAM_ARTIFACT_SCHEMA,
            "instruction_width": self.instruction_width,
            "program_length": self.program_length,
            "interpreter_schema": self.interpreter_schema,
            "execution_schema": self.execution_schema,
            "output_schema": self.output_schema,
            "storage": "opaque_learned_instruction_tensor_v1",
        }

    def snapshot(self) -> torch.Tensor:
        """Return detached CPU data suitable for external storage."""

        return self.codes.detach().to(device="cpu").clone()

    def digest(self) -> str:
        """Return a stable content/interface digest for integrity checks."""

        digest = hashlib.sha256()
        digest.update(EXTERNAL_PROGRAM_ARTIFACT_SCHEMA.encode("utf-8"))
        for value in (
            self.interpreter_schema,
            self.execution_schema,
            self.output_schema or "",
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        codes = self.snapshot().contiguous()
        digest.update(str(codes.dtype).encode("utf-8"))
        digest.update(repr(tuple(codes.shape)).encode("utf-8"))
        digest.update(codes.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def validate_for(
        self,
        *,
        instruction_width: int,
        interpreter_schema: str,
        execution_schema: str,
        output_schema: str | None = None,
    ) -> None:
        """Reject a program whose learned ABI cannot run in this runtime."""

        if instruction_width < 1:
            raise ValueError("runtime instruction width must be positive")
        if self.instruction_width != instruction_width:
            raise ValueError("program artifact instruction width is incompatible")
        if self.interpreter_schema != interpreter_schema:
            raise ValueError("program artifact interpreter schema is incompatible")
        if self.execution_schema != execution_schema:
            raise ValueError("program artifact execution schema is incompatible")
        if output_schema is not None and self.output_schema != output_schema:
            raise ValueError("program artifact output schema is incompatible")

    def payload(self) -> dict[str, Any]:
        """Create a torch-safe, versioned payload for a memory backend."""

        return {
            "schema": EXTERNAL_PROGRAM_ARTIFACT_SCHEMA,
            "configuration": self.configuration(),
            "codes": self.snapshot(),
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalProgramArtifact:
        """Restore and integrity-check one external program artifact."""

        if not isinstance(payload, dict):
            raise TypeError("program artifact payload must be a dictionary")
        if payload.get("schema") != EXTERNAL_PROGRAM_ARTIFACT_SCHEMA:
            raise ValueError("unsupported program artifact schema")
        configuration = payload.get("configuration")
        codes = payload.get("codes")
        if not isinstance(configuration, dict):
            raise TypeError("program artifact configuration must be a dictionary")
        if configuration.get("schema") != EXTERNAL_PROGRAM_ARTIFACT_SCHEMA:
            raise ValueError("program artifact configuration schema mismatch")
        if not isinstance(codes, torch.Tensor):
            raise TypeError("program artifact payload codes must be a tensor")
        artifact = cls(
            codes=codes,
            interpreter_schema=configuration.get("interpreter_schema"),
            execution_schema=configuration.get("execution_schema"),
            output_schema=configuration.get("output_schema"),
        )
        expected_digest = payload.get("sha256")
        if not isinstance(expected_digest, str) or expected_digest != artifact.digest():
            raise ValueError("program artifact checksum mismatch")
        if int(configuration.get("instruction_width", -1)) != artifact.instruction_width:
            raise ValueError("program artifact instruction width metadata mismatch")
        if int(configuration.get("program_length", -1)) != artifact.program_length:
            raise ValueError("program artifact length metadata mismatch")
        return artifact


@dataclass(frozen=True)
class ExternalProgramAdmissionReceipt:
    """Verifier-only admission result for one staged program file.

    The receipt contains no task or protocol identity.  It records only the
    candidate checksum and scalar verifier accounting needed to decide whether
    a file may enter durable external memory.  A rejected candidate must not
    change the executable bank.
    """

    accepted: bool
    candidate_digest: str
    slot: int | None
    observations: int
    stable_bits_to_threshold: int | None
    stable_prefix_minimum: float | None
    reason: str
    schema: str = EXTERNAL_PROGRAM_ADMISSION_SCHEMA

    def validate(self) -> ExternalProgramAdmissionReceipt:
        if self.schema != EXTERNAL_PROGRAM_ADMISSION_SCHEMA:
            raise ValueError("unsupported external program admission schema")
        if len(self.candidate_digest) != 64:
            raise ValueError("program admission candidate digest is malformed")
        try:
            int(self.candidate_digest, 16)
        except ValueError as error:
            raise ValueError("program admission candidate digest is malformed") from error
        if self.observations < 0:
            raise ValueError("program admission observations cannot be negative")
        if self.slot is not None and self.slot < 0:
            raise ValueError("program admission slot cannot be negative")
        if self.stable_bits_to_threshold is not None and (
            self.stable_bits_to_threshold < 1
            or self.stable_bits_to_threshold > self.observations
        ):
            raise ValueError("program admission stable prefix is invalid")
        if self.stable_prefix_minimum is not None and not (
            0.0 <= self.stable_prefix_minimum <= 1.0
        ):
            raise ValueError("program admission stable prefix must lie in [0, 1]")
        if not self.reason:
            raise ValueError("program admission reason must be nonempty")
        return self

    def payload(self) -> dict[str, Any]:
        """Return a portable audit record without candidate tensor contents."""

        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "candidate_digest": self.candidate_digest,
            "slot": self.slot,
            "observations": self.observations,
            "stable_bits_to_threshold": self.stable_bits_to_threshold,
            "stable_prefix_minimum": self.stable_prefix_minimum,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExternalProgramMemoryTransactionReceipt:
    """Auditable result of a copy-on-write executable-memory maintenance step.

    The receipt deliberately carries logical file identity and storage
    accounting, but no task, modality, protocol, or verifier-row contents.
    ``candidate_digest`` is the digest that would have been committed; a
    rejected transaction must leave the live source digest unchanged.
    """

    accepted: bool
    operation: str
    affected_slot_id: int | None
    source_file_count: int
    destination_file_count: int
    source_digest: str
    candidate_digest: str
    source_storage_bytes: int
    candidate_storage_bytes: int
    reason: str
    schema: str = EXTERNAL_PROGRAM_MEMORY_TRANSACTION_SCHEMA

    def validate(self) -> ExternalProgramMemoryTransactionReceipt:
        if self.schema != EXTERNAL_PROGRAM_MEMORY_TRANSACTION_SCHEMA:
            raise ValueError("unsupported external program memory transaction schema")
        if self.operation not in {"evict", "consolidate", "compress"}:
            raise ValueError("external program memory transaction operation is unknown")
        for name, digest in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
        ):
            if len(digest) != 64:
                raise ValueError(f"{name} is malformed")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(f"{name} is malformed") from error
        if self.affected_slot_id is not None and self.affected_slot_id < 0:
            raise ValueError("affected logical slot ID cannot be negative")
        if self.source_file_count < 1 or self.destination_file_count < 1:
            raise ValueError("external program memory must retain one file")
        if self.source_storage_bytes < 1 or self.candidate_storage_bytes < 1:
            raise ValueError("external program memory storage accounting is invalid")
        if not self.reason:
            raise ValueError("external program memory transaction reason is required")
        if not self.accepted and self.source_digest != self.candidate_digest:
            raise ValueError("rejected transaction cannot expose a changed live digest")
        return self

    def payload(self) -> dict[str, object]:
        """Return a metadata-only receipt without tensors or verifier rows."""

        self.validate()
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "operation": self.operation,
            "affected_slot_id": self.affected_slot_id,
            "source_file_count": self.source_file_count,
            "destination_file_count": self.destination_file_count,
            "source_digest": self.source_digest,
            "candidate_digest": self.candidate_digest,
            "source_storage_bytes": self.source_storage_bytes,
            "candidate_storage_bytes": self.candidate_storage_bytes,
            "reason": self.reason,
        }


def evaluate_external_program_admission(
    artifact: ExternalProgramArtifact,
    outcomes: torch.Tensor | list[float] | tuple[float, ...],
    *,
    threshold: float = 0.8,
    min_observations: int = 1,
    min_stable_observations: int = 1,
) -> ExternalProgramAdmissionReceipt:
    """Evaluate a staged file using only an ordered scalar outcome stream.

    The first prefix whose remaining outcomes all clear ``threshold`` is the
    stable promotion prefix.  This is intentionally a sufficient accounting
    rule: the raw examples need not be replayed or retained by the file store.
    """

    if not isinstance(artifact, ExternalProgramArtifact):
        raise TypeError("program admission requires an external program artifact")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("program admission threshold must lie in [0, 1]")
    if min_observations < 1:
        raise ValueError("program admission needs positive minimum observations")
    if min_stable_observations < 1:
        raise ValueError(
            "program admission needs positive minimum stable observations"
        )
    values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        return ExternalProgramAdmissionReceipt(
            accepted=False,
            candidate_digest=artifact.digest(),
            slot=None,
            observations=0,
            stable_bits_to_threshold=None,
            stable_prefix_minimum=None,
            reason="no verifier outcomes were supplied",
        ).validate()
    if not bool(torch.isfinite(values).all()) or bool(
        torch.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("program admission outcomes must be finite values in [0, 1]")
    if values.numel() < min_observations:
        return ExternalProgramAdmissionReceipt(
            accepted=False,
            candidate_digest=artifact.digest(),
            slot=None,
            observations=int(values.numel()),
            stable_bits_to_threshold=None,
            stable_prefix_minimum=float(values.min().item()),
            reason="candidate has not reached the minimum verifier observations",
        ).validate()
    stable_prefix: int | None = None
    for index in range(values.numel()):
        if (
            values.numel() - index >= min_stable_observations
            and bool(torch.all(values[index:] >= threshold))
        ):
            stable_prefix = index + 1
            break
    if stable_prefix is None:
        return ExternalProgramAdmissionReceipt(
            accepted=False,
            candidate_digest=artifact.digest(),
            slot=None,
            observations=int(values.numel()),
            stable_bits_to_threshold=None,
            stable_prefix_minimum=float(values.min().item()),
            reason="candidate did not clear a stable verifier prefix",
        ).validate()
    return ExternalProgramAdmissionReceipt(
        accepted=True,
        candidate_digest=artifact.digest(),
        slot=None,
        observations=int(values.numel()),
        stable_bits_to_threshold=stable_prefix,
        stable_prefix_minimum=float(values[stable_prefix - 1 :].min().item()),
        reason="candidate cleared the stable verifier prefix",
    ).validate()


__all__ = [
    "EXTERNAL_PROGRAM_ADMISSION_SCHEMA",
    "EXTERNAL_PROGRAM_ARTIFACT_SCHEMA",
    "EXTERNAL_PROGRAM_MEMORY_TRANSACTION_SCHEMA",
    "ExternalProgramAdmissionReceipt",
    "ExternalProgramArtifact",
    "ExternalProgramMemoryTransactionReceipt",
    "evaluate_external_program_admission",
]
