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


__all__ = ["EXTERNAL_PROGRAM_ARTIFACT_SCHEMA", "ExternalProgramArtifact"]
