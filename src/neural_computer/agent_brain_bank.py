"""One durable heterogeneous ``AgentBrain.bank`` container.

The repository historically had two unrelated files claiming the same
``.bank`` extension: a torch-serialized temporal routing bank and a JSON
executive-program bank.  This module is the canonical container.  It keeps
the legacy temporal family and the newer self-contained executive family in
one checksummed, versioned payload while preserving each family's own ABI.

Legacy temporal files are imported explicitly through
:meth:`ExternalAgentBrainBank.migrate_legacy_temporal_bank`.  The normal
loader never executes an arbitrary pickle merely because a file has the
``.bank`` suffix.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch

from .executive_bank import (
    EXECUTIVE_COMPOSITION_SCHEMA,
    ExternalExecutiveProgramArtifact,
    ExternalExecutiveProgramBank,
    compose_executive_artifacts,
)
from .program import (
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
)
from .temporal_program import (
    AGENT_BANK_EXTENSION,
    ExternalTemporalProgramBank,
)

EXTERNAL_AGENT_BRAIN_BANK_SCHEMA = "neural-computer.external-agent-brain-bank.v1"
AGENT_BRAIN_BANK_ENTRY_SCHEMA = "neural-computer.agent-brain-bank-entry.v1"
AGENT_BRAIN_EXECUTIVE_KIND = "executive_program"
AGENT_BRAIN_TEMPORAL_KIND = "temporal_program_bank"

BrainEntryKind = Literal["executive_program", "temporal_program_bank"]

_DTYPES: dict[str, torch.dtype] = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
        "complex128",
    )
    if hasattr(torch, name)
}


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


def _validate_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return value


def _encode_tensor(tensor: torch.Tensor) -> dict[str, object]:
    """Encode opaque tensor bytes without invoking pickle or changing values."""

    value = tensor.detach().to(device="cpu").contiguous()
    dtype_name = str(value.dtype).removeprefix("torch.")
    if dtype_name not in _DTYPES:
        raise TypeError(f"unsupported tensor dtype for AgentBrain.bank: {value.dtype}")
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": dtype_name,
        "shape": list(value.shape),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def _decode_tensor(payload: object) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise TypeError("encoded AgentBrain tensor must be a dictionary")
    dtype_name = payload.get("dtype")
    shape = payload.get("shape")
    encoded = payload.get("data_b64")
    if (
        not isinstance(dtype_name, str)
        or dtype_name not in _DTYPES
        or not isinstance(shape, list)
        or not all(isinstance(item, int) and item >= 0 for item in shape)
        or not isinstance(encoded, str)
    ):
        raise ValueError("encoded AgentBrain tensor metadata is malformed")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise ValueError("encoded AgentBrain tensor bytes are malformed") from error
    dtype = _DTYPES[dtype_name]
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    if len(raw) != element_count * element_size:
        raise ValueError("encoded AgentBrain tensor byte count does not match shape")
    byte_tensor = torch.tensor(list(raw), dtype=torch.uint8)
    return byte_tensor.view(dtype).reshape(tuple(shape)).clone()


def _portable_temporal_payload(bank: ExternalTemporalProgramBank) -> dict[str, object]:
    payload = dict(bank.payload())
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("temporal bank artifacts are malformed")
    portable_artifacts: list[dict[str, object]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise TypeError("temporal bank artifact is malformed")
        artifact = dict(item)
        artifact["codes"] = _encode_tensor(item.get("codes"))
        portable_artifacts.append(artifact)
    payload["artifacts"] = portable_artifacts
    return payload


def _restore_temporal_payload(payload: dict[str, object]) -> ExternalTemporalProgramBank:
    restored = dict(payload)
    artifacts = restored.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("temporal bank artifacts are malformed")
    restored_artifacts: list[dict[str, object]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise TypeError("temporal bank artifact is malformed")
        artifact = dict(item)
        artifact["codes"] = _decode_tensor(item.get("codes"))
        restored_artifacts.append(artifact)
    restored["artifacts"] = restored_artifacts
    return ExternalTemporalProgramBank.from_payload(restored)


def _capacity_rejection(
    candidate_digest: str,
    outcomes: torch.Tensor | list[float] | tuple[float, ...],
    *,
    reason: str,
) -> ExternalProgramAdmissionReceipt:
    values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
    return ExternalProgramAdmissionReceipt(
        accepted=False,
        candidate_digest=candidate_digest,
        slot=None,
        observations=int(values.numel()),
        stable_bits_to_threshold=None,
        stable_prefix_minimum=None,
        reason=reason,
    ).validate()


class ExternalAgentBrainBank:
    """Canonical append-only bank for heterogeneous reusable skills.

    Executive artifacts are individually addressable and executable.  Legacy
    temporal programs retain their learned opaque-context router as one family
    inside the same container, so migration does not flatten or discard route
    evidence.  Both families remain bound to the same frozen controller
    digest, and the top-level capacity counts all executable skill slots.
    """

    schema = EXTERNAL_AGENT_BRAIN_BANK_SCHEMA

    def __init__(self, *, controller_digest: str, capacity: int = 32) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("AgentBrain bank capacity must be positive")
        self.controller_digest = _validate_digest(
            controller_digest, name="controller_digest"
        )
        self.capacity = capacity
        self._temporal_bank: ExternalTemporalProgramBank | None = None
        self._executive_bank = ExternalExecutiveProgramBank(
            controller_digest=self.controller_digest, capacity=capacity
        )
        self._composition_provenance: list[dict[str, object]] = []
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def temporal_bank(self) -> ExternalTemporalProgramBank | None:
        return self._temporal_bank

    @property
    def executive_bank(self) -> ExternalExecutiveProgramBank:
        return self._executive_bank

    @property
    def temporal_program_count(self) -> int:
        return 0 if self._temporal_bank is None else self._temporal_bank.program_count

    @property
    def executive_program_count(self) -> int:
        return self._executive_bank.program_count

    @property
    def program_count(self) -> int:
        return self.temporal_program_count + self.executive_program_count

    @property
    def skill_count(self) -> int:
        return self.program_count

    @property
    def composition_provenance(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._composition_provenance)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "controller_digest": self.controller_digest,
            "capacity": self.capacity,
            "families": [AGENT_BRAIN_TEMPORAL_KIND, AGENT_BRAIN_EXECUTIVE_KIND],
            "storage": "checksummed_heterogeneous_agent_brain_json_v1",
        }

    def _has_executive_digest(self, digest: str) -> bool:
        return any(
            self._executive_bank.artifact(slot).digest() == digest
            for slot in range(self.executive_program_count)
        )

    def _has_temporal_digest(self, digest: str) -> bool:
        return self._temporal_bank is not None and any(
            self._temporal_bank.artifact(slot).digest() == digest
            for slot in range(self.temporal_program_count)
        )

    def manifest(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        if self._temporal_bank is not None:
            entries.append(
                {
                    "schema": AGENT_BRAIN_BANK_ENTRY_SCHEMA,
                    "kind": AGENT_BRAIN_TEMPORAL_KIND,
                    "slot": None,
                    "program_count": self.temporal_program_count,
                    "digest": self._temporal_bank.digest(),
                }
            )
        for slot in range(self.executive_program_count):
            entries.append(
                {
                    "schema": AGENT_BRAIN_BANK_ENTRY_SCHEMA,
                    "kind": AGENT_BRAIN_EXECUTIVE_KIND,
                    "slot": slot,
                    "program_count": 1,
                    "digest": self._executive_bank.artifact(slot).digest(),
                }
            )
        return entries

    def admit_executive(
        self,
        artifact: ExternalExecutiveProgramArtifact,
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ExternalProgramAdmissionReceipt:
        if not hasattr(artifact, "digest") or not hasattr(artifact, "validate"):
            raise TypeError("AgentBrain executive admission needs an executable artifact")
        digest = artifact.digest()
        if self.skill_count >= self.capacity and not self._has_executive_digest(digest):
            return _capacity_rejection(
                digest,
                outcomes,
                reason="verified executive artifact could not enter a full AgentBrain bank",
            )
        receipt = self._executive_bank.admit(
            artifact,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if receipt.accepted:
            self._version += 1
        return receipt

    def admit(
        self,
        artifact: ExternalExecutiveProgramArtifact,
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        **kwargs: Any,
    ) -> ExternalProgramAdmissionReceipt:
        """Alias for executive admission when the bank has one active family."""

        return self.admit_executive(artifact, outcomes, **kwargs)

    def admit_temporal(
        self,
        artifact: ExternalProgramArtifact,
        context: torch.Tensor,
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        *,
        threshold: float = 0.8,
        min_observations: int = 8,
        min_stable_observations: int = 8,
    ) -> ExternalProgramAdmissionReceipt:
        if not isinstance(context, torch.Tensor):
            raise TypeError("AgentBrain temporal context must be a learned event tensor")
        digest = artifact.digest()
        if self.skill_count >= self.capacity and not self._has_temporal_digest(digest):
            return _capacity_rejection(
                digest,
                outcomes,
                reason="verified temporal artifact could not enter a full AgentBrain bank",
            )
        if self._temporal_bank is None:
            self._temporal_bank = ExternalTemporalProgramBank(
                int(context.numel()),
                artifact.instruction_width,
                controller_digest=self.controller_digest,
                capacity=self.capacity,
            )
        receipt = self._temporal_bank.admit(
            artifact,
            context,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if receipt.accepted:
            self._version += 1
            self._temporal_bank.capacity = self.capacity
        elif self._temporal_bank.program_count == 0:
            self._temporal_bank = None
        return receipt

    def compose_executive(
        self,
        parent_slots: Sequence[int],
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
    ) -> ExternalProgramAdmissionReceipt:
        """Compose admitted executive slots, then gate the child by outcomes."""

        normalized = tuple(parent_slots)
        if len(normalized) < 2:
            raise ValueError("executive composition needs at least two parent slots")
        if any(
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot < self.executive_program_count
            for slot in normalized
        ):
            raise IndexError("executive composition parent slot is outside the bank")
        parents = tuple(self._executive_bank.artifact(slot) for slot in normalized)
        child = compose_executive_artifacts(parents)
        receipt = self.admit_executive(
            child,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if receipt.accepted:
            self._composition_provenance.append(
                {
                    "schema": EXECUTIVE_COMPOSITION_SCHEMA,
                    "child_digest": child.digest(),
                    "parent_slots": list(normalized),
                    "parent_digests": [parent.digest() for parent in parents],
                    "admission": receipt.payload(),
                }
            )
            self._version += 1
        return receipt

    def import_temporal_bank(self, bank: ExternalTemporalProgramBank) -> None:
        """Copy a validated legacy temporal family into this heterogeneous bank."""

        if not isinstance(bank, ExternalTemporalProgramBank):
            raise TypeError("AgentBrain temporal import needs an external temporal bank")
        if bank.controller_digest != self.controller_digest:
            raise ValueError("temporal bank controller digest is incompatible")
        if self._temporal_bank is not None and self._temporal_bank.digest() != bank.digest():
            raise ValueError("AgentBrain already contains a different temporal family")
        if self._temporal_bank is None and self.executive_program_count + bank.program_count > self.capacity:
            raise ValueError("temporal family exceeds remaining AgentBrain capacity")
        self._temporal_bank = ExternalTemporalProgramBank.from_payload(bank.payload())
        self._temporal_bank.capacity = self.capacity
        self._version += 1

    def artifact(self, kind: BrainEntryKind, slot: int) -> Any:
        if kind == AGENT_BRAIN_EXECUTIVE_KIND:
            return self._executive_bank.artifact(slot)
        if kind == AGENT_BRAIN_TEMPORAL_KIND:
            if self._temporal_bank is None:
                raise IndexError("AgentBrain temporal family is empty")
            return self._temporal_bank.artifact(slot)
        raise ValueError("unknown AgentBrain entry kind")

    def executable(
        self,
        slot: int,
        *,
        controller_digest: str,
        max_instructions_per_tick: int = 128,
    ) -> Any:
        """Instantiate an allow-listed executive skill after ABI binding."""

        return self._executive_bank.executable(
            slot,
            controller_digest=controller_digest,
            max_instructions_per_tick=max_instructions_per_tick,
        )

    def _content_payload(self) -> dict[str, object]:
        content: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "version": self.version,
            "temporal_bank": (
                None
                if self._temporal_bank is None
                else _portable_temporal_payload(self._temporal_bank)
            ),
            "executive_bank": self._executive_bank.payload(),
            "manifest": self.manifest(),
        }
        if self._composition_provenance:
            content["composition_provenance"] = [
                dict(record) for record in self._composition_provenance
            ]
        return content

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self._content_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def payload(self) -> dict[str, object]:
        return {**self._content_payload(), "sha256": self.digest()}

    @classmethod
    def from_temporal_bank(
        cls,
        bank: ExternalTemporalProgramBank,
        *,
        capacity: int | None = None,
    ) -> ExternalAgentBrainBank:
        if not isinstance(bank, ExternalTemporalProgramBank):
            raise TypeError("temporal migration needs an external temporal bank")
        target_capacity = max(bank.program_count, bank.capacity) if capacity is None else capacity
        result = cls(controller_digest=bank.controller_digest, capacity=target_capacity)
        result.import_temporal_bank(bank)
        return result

    @classmethod
    def from_executive_bank(
        cls,
        bank: ExternalExecutiveProgramBank,
        *,
        capacity: int | None = None,
    ) -> ExternalAgentBrainBank:
        if not isinstance(bank, ExternalExecutiveProgramBank):
            raise TypeError("executive migration needs an external executive bank")
        target_capacity = max(bank.program_count, bank.capacity) if capacity is None else capacity
        if bank.program_count > target_capacity:
            raise ValueError("executive bank exceeds target AgentBrain capacity")
        result = cls(controller_digest=bank.controller_digest, capacity=target_capacity)
        result._executive_bank = ExternalExecutiveProgramBank.from_payload(bank.payload())
        result._executive_bank.capacity = target_capacity
        result._version = 1 if bank.program_count else 0
        return result

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExternalAgentBrainBank:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported AgentBrain bank schema")
        configuration = payload.get("configuration")
        temporal_payload = payload.get("temporal_bank")
        executive_payload = payload.get("executive_bank")
        version = payload.get("version")
        manifest = payload.get("manifest")
        provenance = payload.get("composition_provenance", [])
        if (
            not isinstance(configuration, dict)
            or not isinstance(executive_payload, dict)
            or not isinstance(manifest, list)
            or not isinstance(version, int)
            or version < 0
            or not isinstance(provenance, list)
        ):
            raise TypeError("AgentBrain bank payload is malformed")
        result = cls(
            controller_digest=configuration.get("controller_digest"),
            capacity=configuration.get("capacity"),
        )
        if result.configuration() != configuration:
            raise ValueError("AgentBrain bank configuration mismatch")
        if temporal_payload is not None:
            if not isinstance(temporal_payload, dict):
                raise TypeError("AgentBrain temporal payload is malformed")
            result._temporal_bank = _restore_temporal_payload(temporal_payload)
            if result._temporal_bank.controller_digest != result.controller_digest:
                raise ValueError("AgentBrain temporal controller digest is incompatible")
        result._executive_bank = ExternalExecutiveProgramBank.from_payload(executive_payload)
        if result._executive_bank.controller_digest != result.controller_digest:
            raise ValueError("AgentBrain executive controller digest is incompatible")
        if result.program_count > result.capacity:
            raise ValueError("AgentBrain bank exceeds capacity")
        result._version = version
        if not all(isinstance(record, dict) for record in provenance):
            raise TypeError("AgentBrain composition provenance is malformed")
        result._composition_provenance = [dict(record) for record in provenance]
        result._validate_composition_provenance()
        if manifest != result.manifest():
            raise ValueError("AgentBrain bank manifest mismatch")
        if payload.get("sha256") != result.digest():
            raise ValueError("AgentBrain bank checksum mismatch")
        return result

    def _validate_composition_provenance(self) -> None:
        executable_digests = {
            self._executive_bank.artifact(slot).digest()
            for slot in range(self.executive_program_count)
        }
        for record in self._composition_provenance:
            if record.get("schema") != EXECUTIVE_COMPOSITION_SCHEMA:
                raise ValueError("unsupported AgentBrain composition provenance schema")
            child_digest = record.get("child_digest")
            parent_slots = record.get("parent_slots")
            parent_digests = record.get("parent_digests")
            admission = record.get("admission")
            if (
                not isinstance(child_digest, str)
                or child_digest not in executable_digests
                or not isinstance(parent_slots, list)
                or len(parent_slots) < 2
                or not all(
                    isinstance(slot, int)
                    and not isinstance(slot, bool)
                    and 0 <= slot < self.executive_program_count
                    for slot in parent_slots
                )
                or not isinstance(parent_digests, list)
                or len(parent_digests) != len(parent_slots)
                or not all(
                    isinstance(digest, str) and digest in executable_digests
                    for digest in parent_digests
                )
                or not isinstance(admission, dict)
            ):
                raise ValueError("AgentBrain composition provenance is invalid")
            expected_parent_digests = [
                self._executive_bank.artifact(slot).digest() for slot in parent_slots
            ]
            if parent_digests != expected_parent_digests:
                raise ValueError("AgentBrain composition parent digest binding is invalid")

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
    def load_bank(cls, path: Path) -> ExternalAgentBrainBank:
        path = Path(path)
        if path.suffix != AGENT_BANK_EXTENSION:
            raise ValueError(
                f"canonical agent banks must use the {AGENT_BANK_EXTENSION} extension"
            )
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.is_file():
            raise ValueError("AgentBrain bank checksum is missing")
        if checksum_path.read_text().strip() != _sha256_file(path):
            raise ValueError("AgentBrain bank file checksum mismatch")
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                "legacy temporal .bank is not loaded implicitly; use "
                "migrate_legacy_temporal_bank"
            ) from error
        if isinstance(payload, dict) and payload.get("schema") == ExternalExecutiveProgramBank.schema:
            return cls.from_executive_bank(
                ExternalExecutiveProgramBank.from_payload(payload)
            )
        return cls.from_payload(payload)

    @classmethod
    def migrate_legacy_temporal_bank(
        cls,
        source: Path,
        destination: Path,
        *,
        capacity: int | None = None,
    ) -> ExternalAgentBrainBank:
        """Validate a trusted old torch bank, then rewrite it canonically as JSON."""

        legacy = ExternalTemporalProgramBank.load_bank(Path(source))
        result = cls.from_temporal_bank(legacy, capacity=capacity)
        result.save_bank(Path(destination))
        return result


__all__ = [
    "AGENT_BRAIN_BANK_ENTRY_SCHEMA",
    "AGENT_BRAIN_EXECUTIVE_KIND",
    "AGENT_BRAIN_TEMPORAL_KIND",
    "EXTERNAL_AGENT_BRAIN_BANK_SCHEMA",
    "ExternalAgentBrainBank",
]
