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
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .addressing import PersistentOpaqueContextRouteEvidence
from .executive_bank import (
    EXECUTIVE_COMPOSITION_SCHEMA,
    ExternalExecutiveProgramArtifact,
    ExternalExecutiveProgramBank,
    compose_executive_artifacts,
    executive_artifact_can_handoff,
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
EXECUTIVE_COMPOSITION_SEARCH_SCHEMA = (
    "neural-computer.executive-composition-search.v3"
)

BrainEntryKind = Literal["executive_program", "temporal_program_bank"]


@dataclass(frozen=True)
class ExecutiveCompositionEvaluation:
    """Fresh verifier evidence and explicit accounting for one candidate."""

    outcomes: tuple[float, ...]
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    replayed_examples: int = 0

    def validate(self) -> ExecutiveCompositionEvaluation:
        values = torch.as_tensor(self.outcomes, dtype=torch.float64)
        if values.ndim != 1 or values.numel() < 1 or not bool(torch.isfinite(values).all()):
            raise ValueError("composition evaluation outcomes must be finite scalars")
        if not bool(((0.0 <= values) & (values <= 1.0)).all()):
            raise ValueError("composition evaluation outcomes must lie in [0, 1]")
        for name, value in (
            ("unique_verifier_bits", self.unique_verifier_bits),
            ("unique_logical_lifetimes", self.unique_logical_lifetimes),
            ("replayed_examples", self.replayed_examples),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"composition evaluation {name} must be nonnegative")
        if self.unique_verifier_bits < len(self.outcomes):
            raise ValueError("composition evaluation cannot contain more outcomes than bits")
        return self


@dataclass(frozen=True)
class ExecutiveCompositionSearchResult:
    """Opaque accounting for one verifier-gated parent-pair search."""

    accepted: bool
    parent_slots: tuple[int, int] | None
    receipt: ExternalProgramAdmissionReceipt | None
    candidate_count: int
    attempted_parent_slots: tuple[tuple[int, int], ...]
    attempted_child_digests: tuple[str, ...]
    attempted_evaluation_stages: tuple[int, ...]
    unique_verifier_bits: int
    unique_logical_lifetimes: int
    replayed_examples: int
    stable_bits_to_threshold: int | None
    admission_threshold: float
    screening_threshold: float | None
    min_observations: int
    min_stable_observations: int
    bank_digest_before: str
    bank_digest_after: str
    reason: str
    schema: str = EXECUTIVE_COMPOSITION_SEARCH_SCHEMA

    def validate(self) -> ExecutiveCompositionSearchResult:
        if self.schema != EXECUTIVE_COMPOSITION_SEARCH_SCHEMA:
            raise ValueError("unsupported executive composition search schema")
        if self.candidate_count < 0 or len(self.attempted_parent_slots) > self.candidate_count:
            raise ValueError("executive composition search candidate accounting is invalid")
        if len(self.attempted_parent_slots) != len(self.attempted_child_digests):
            raise ValueError("executive composition search attempt records are misaligned")
        if (
            len(self.attempted_parent_slots) != len(self.attempted_evaluation_stages)
            or any(stages < 1 for stages in self.attempted_evaluation_stages)
        ):
            raise ValueError("executive composition search stage records are invalid")
        if (
            self.unique_verifier_bits < 0
            or self.unique_logical_lifetimes < 0
            or self.replayed_examples < 0
        ):
            raise ValueError("executive composition search accounting cannot be negative")
        if self.stable_bits_to_threshold is not None and not (
            1 <= self.stable_bits_to_threshold <= self.unique_verifier_bits
        ):
            raise ValueError("executive composition search stable prefix is invalid")
        if not 0.0 <= self.admission_threshold <= 1.0:
            raise ValueError("executive composition admission threshold is invalid")
        if self.screening_threshold is not None and not (
            0.0 <= self.screening_threshold <= self.admission_threshold
        ):
            raise ValueError("executive composition screening threshold is invalid")
        if self.min_observations < 1 or self.min_stable_observations < 1:
            raise ValueError("executive composition observation gates are invalid")
        for digest in (self.bank_digest_before, self.bank_digest_after):
            _validate_digest(digest, name="executive composition search bank digest")
        if self.accepted:
            if self.parent_slots is None or self.receipt is None or not self.receipt.accepted:
                raise ValueError("accepted executive composition search needs a receipt")
        elif self.parent_slots is not None or self.receipt is not None:
            raise ValueError("rejected executive composition search cannot have a receipt")
        if not self.reason:
            raise ValueError("executive composition search reason is required")
        return self

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "parent_slots": None if self.parent_slots is None else list(self.parent_slots),
            "receipt": None if self.receipt is None else self.receipt.payload(),
            "candidate_count": self.candidate_count,
            "attempted_parent_slots": [list(pair) for pair in self.attempted_parent_slots],
            "attempted_child_digests": list(self.attempted_child_digests),
            "attempted_evaluation_stages": list(self.attempted_evaluation_stages),
            "unique_verifier_bits": self.unique_verifier_bits,
            "unique_logical_lifetimes": self.unique_logical_lifetimes,
            "replayed_examples": self.replayed_examples,
            "stable_bits_to_threshold": self.stable_bits_to_threshold,
            "admission_threshold": self.admission_threshold,
            "screening_threshold": self.screening_threshold,
            "min_observations": self.min_observations,
            "min_stable_observations": self.min_stable_observations,
            "bank_digest_before": self.bank_digest_before,
            "bank_digest_after": self.bank_digest_after,
            "reason": self.reason,
        }

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
        self._executive_route: PersistentOpaqueContextRouteEvidence | None = None
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
    def executive_route(self) -> PersistentOpaqueContextRouteEvidence | None:
        """Return opaque executive-slot evidence, when routing is enabled."""

        return self._executive_route

    def executive_route_evidence(
        self,
        context_width: int,
        *,
        matching_tolerance: float = 1e-4,
        generalization_tolerance: float = 0.0,
        prior_strength: float = 1.0,
        mastery_threshold: float = 0.8,
        min_mastery_observations: int = 8,
        reversal_threshold: float = 0.5,
        reversal_patience: int = 4,
    ) -> PersistentOpaqueContextRouteEvidence:
        """Create or retrieve persistent opaque route evidence for executive slots."""

        if self._executive_route is None:
            self._executive_route = PersistentOpaqueContextRouteEvidence(
                context_width,
                matching_tolerance=matching_tolerance,
                generalization_tolerance=generalization_tolerance,
                prior_strength=prior_strength,
                mastery_threshold=mastery_threshold,
                min_mastery_observations=min_mastery_observations,
                reversal_threshold=reversal_threshold,
                reversal_patience=reversal_patience,
            )
            self._version += 1
        elif self._executive_route.width != context_width:
            raise ValueError("executive route context width is incompatible")
        while self._executive_route.slot_count < self.executive_program_count:
            self._executive_route.append_slot()
        if self._executive_route.slot_count > self.executive_program_count:
            raise ValueError("executive route has more slots than the executive bank")
        return self._executive_route

    def observe_executive_route(
        self,
        context: torch.Tensor,
        slot: int,
        outcome: float | torch.Tensor,
    ) -> None:
        """Record one attempted executive skill against its learned context."""

        route = self._executive_route
        if route is None:
            raise RuntimeError("executive route evidence has not been initialized")
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise TypeError("executive route slot must be an integer")
        if not 0 <= slot < self.executive_program_count:
            raise IndexError("executive route slot is outside the executive bank")
        route.observe(context, slot, outcome)
        self._version += 1

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
        return tuple(copy.deepcopy(record) for record in self._composition_provenance)

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
            if self._executive_route is not None:
                while self._executive_route.slot_count < self.executive_program_count:
                    self._executive_route.append_slot()
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

    def composed_executive_artifact(
        self,
        parent_slots: Sequence[int],
        *,
        share_compatible_operators: bool = False,
        final_emit_only: bool = False,
    ) -> ExternalExecutiveProgramArtifact:
        """Materialize a child from existing slots without changing the bank."""

        if not isinstance(share_compatible_operators, bool):
            raise TypeError("executive composition operator sharing must be boolean")
        if not isinstance(final_emit_only, bool):
            raise TypeError("executive composition final emit policy must be boolean")

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
        return compose_executive_artifacts(
            tuple(self._executive_bank.artifact(slot) for slot in normalized),
            share_compatible_operators=share_compatible_operators,
            final_emit_only=final_emit_only,
        )

    def compose_executive(
        self,
        parent_slots: Sequence[int],
        outcomes: torch.Tensor | list[float] | tuple[float, ...],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        share_compatible_operators: bool = False,
        final_emit_only: bool = False,
    ) -> ExternalProgramAdmissionReceipt:
        """Compose admitted executive slots, then gate the child by outcomes."""

        if not isinstance(share_compatible_operators, bool):
            raise TypeError("executive composition operator sharing must be boolean")
        if not isinstance(final_emit_only, bool):
            raise TypeError("executive composition final emit policy must be boolean")

        normalized = tuple(parent_slots)
        child = self.composed_executive_artifact(
            normalized,
            share_compatible_operators=share_compatible_operators,
            final_emit_only=final_emit_only,
        )
        parents = tuple(self._executive_bank.artifact(slot) for slot in normalized)
        receipt = self.admit_executive(
            child,
            outcomes,
            threshold=threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
        )
        if receipt.accepted:
            provenance = {
                "schema": EXECUTIVE_COMPOSITION_SCHEMA,
                "child_digest": child.digest(),
                "parent_slots": list(normalized),
                "parent_digests": [parent.digest() for parent in parents],
                "admission": receipt.payload(),
            }
            if share_compatible_operators:
                provenance["share_compatible_operators"] = True
            if final_emit_only:
                provenance["final_emit_only"] = True
            self._composition_provenance.append(provenance)
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
        if self._executive_route is not None:
            content["executive_route"] = self._executive_route.payload()
        if self._composition_provenance:
            content["composition_provenance"] = copy.deepcopy(
                self._composition_provenance
            )
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
        executive_route_payload = payload.get("executive_route")
        if (
            not isinstance(configuration, dict)
            or not isinstance(executive_payload, dict)
            or not isinstance(manifest, list)
            or not isinstance(version, int)
            or version < 0
            or not isinstance(provenance, list)
            or (
                executive_route_payload is not None
                and not isinstance(executive_route_payload, dict)
            )
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
        if executive_route_payload is not None:
            result._executive_route = PersistentOpaqueContextRouteEvidence.from_payload(
                executive_route_payload
            )
            if result._executive_route.slot_count != result.executive_program_count:
                raise ValueError("AgentBrain executive route slot count is incompatible")
        if result.program_count > result.capacity:
            raise ValueError("AgentBrain bank exceeds capacity")
        result._version = version
        if not all(isinstance(record, dict) for record in provenance):
            raise TypeError("AgentBrain composition provenance is malformed")
        result._composition_provenance = copy.deepcopy(provenance)
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
            share_compatible_operators = record.get(
                "share_compatible_operators", False
            )
            final_emit_only = record.get("final_emit_only", False)
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
                or not isinstance(share_compatible_operators, bool)
                or not isinstance(final_emit_only, bool)
            ):
                raise ValueError("AgentBrain composition provenance is invalid")
            expected_parent_digests = [
                self._executive_bank.artifact(slot).digest() for slot in parent_slots
            ]
            if parent_digests != expected_parent_digests:
                raise ValueError("AgentBrain composition parent digest binding is invalid")
            expected_child = self.composed_executive_artifact(
                parent_slots,
                share_compatible_operators=share_compatible_operators,
                final_emit_only=final_emit_only,
            )
            if child_digest != expected_child.digest():
                raise ValueError("AgentBrain composition child derivation is invalid")
            try:
                admission_receipt = ExternalProgramAdmissionReceipt(
                    accepted=admission.get("accepted"),
                    candidate_digest=admission.get("candidate_digest"),
                    slot=admission.get("slot"),
                    observations=admission.get("observations"),
                    stable_bits_to_threshold=admission.get("stable_bits_to_threshold"),
                    stable_prefix_minimum=admission.get("stable_prefix_minimum"),
                    reason=admission.get("reason"),
                    schema=admission.get("schema"),
                ).validate()
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "AgentBrain composition admission receipt is invalid"
                ) from error
            if (
                admission_receipt.accepted is not True
                or admission_receipt.candidate_digest != child_digest
                or not isinstance(admission_receipt.slot, int)
                or isinstance(admission_receipt.slot, bool)
                or not 0 <= admission_receipt.slot < self.executive_program_count
                or self._executive_bank.artifact(admission_receipt.slot).digest()
                != child_digest
            ):
                raise ValueError("AgentBrain composition admission binding is invalid")

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


class ExternalExecutiveCompositionSearch:
    """Search opaque ordered parent pairs and append the first verified child."""

    schema = EXECUTIVE_COMPOSITION_SEARCH_SCHEMA

    def __init__(self, bank: ExternalAgentBrainBank, *, seed: int = 0) -> None:
        if not isinstance(bank, ExternalAgentBrainBank):
            raise TypeError("executive composition search needs an AgentBrain bank")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("executive composition search seed must be an integer")
        self.bank = bank
        self.seed = seed

    def candidate_parent_slots(self, *, max_candidates: int | None = None) -> tuple[tuple[int, int], ...]:
        """Return deterministic ordered pairs using only content-addressed order."""

        if max_candidates is not None and (
            not isinstance(max_candidates, int)
            or isinstance(max_candidates, bool)
            or max_candidates < 1
        ):
            raise ValueError("composition search max_candidates must be positive")
        pairs: list[tuple[str, tuple[int, int]]] = []
        for first in range(self.bank.executive_program_count):
            first_artifact = self.bank.executive_bank.artifact(first)
            if not executive_artifact_can_handoff(first_artifact):
                continue
            for second in range(self.bank.executive_program_count):
                if first == second:
                    continue
                parent_digests = (
                    first_artifact.digest(),
                    self.bank.executive_bank.artifact(second).digest(),
                )
                child = self.bank.composed_executive_artifact((first, second))
                key = hashlib.sha256(
                    f"{self.seed}:{parent_digests[0]}:{parent_digests[1]}:{child.digest()}".encode()
                ).hexdigest()
                pairs.append((key, (first, second)))
        pairs.sort(key=lambda item: item[0])
        ordered = tuple(pair for _, pair in pairs)
        return ordered if max_candidates is None else ordered[:max_candidates]

    def search(
        self,
        evaluate: Callable[
            [tuple[int, int], ExternalExecutiveProgramArtifact, int],
            ExecutiveCompositionEvaluation,
        ],
        *,
        threshold: float = 0.8,
        min_observations: int = 1,
        min_stable_observations: int = 1,
        screening_threshold: float | None = None,
        max_candidates: int | None = None,
    ) -> ExecutiveCompositionSearchResult:
        """Evaluate candidates through scalar outcomes and append one stable child."""

        if not callable(evaluate):
            raise TypeError("executive composition search evaluator must be callable")
        if screening_threshold is not None and not (
            0.0 <= screening_threshold <= threshold
        ):
            raise ValueError(
                "composition screening threshold must lie in [0, admission threshold]"
            )
        before = self.bank.digest()
        candidates = self.candidate_parent_slots(max_candidates=max_candidates)
        attempted_slots: list[tuple[int, int]] = []
        attempted_digests: list[str] = []
        attempted_stages: list[int] = []
        unique_bits = 0
        unique_lifetimes = 0
        replayed_examples = 0
        for parent_slots in candidates:
            child = self.bank.composed_executive_artifact(parent_slots)
            outcomes: list[float] = []
            stages = 0
            required_outcomes = max(min_observations, min_stable_observations)
            screened_out = False
            while len(outcomes) < required_outcomes:
                if stages >= required_outcomes:
                    break
                evaluation = evaluate(parent_slots, child, stages)
                if not isinstance(evaluation, ExecutiveCompositionEvaluation):
                    raise TypeError(
                        "composition search evaluator must return explicit evaluation accounting"
                    )
                evaluation.validate()
                stage_values = torch.as_tensor(
                    evaluation.outcomes, dtype=torch.float64
                )
                outcomes.extend(float(value) for value in stage_values.tolist())
                unique_bits += evaluation.unique_verifier_bits
                unique_lifetimes += evaluation.unique_logical_lifetimes
                replayed_examples += evaluation.replayed_examples
                stages += 1
                if (
                    stages == 1
                    and screening_threshold is not None
                    and bool(torch.all(stage_values < screening_threshold))
                ):
                    screened_out = True
                    break
            attempted_slots.append(parent_slots)
            attempted_digests.append(child.digest())
            attempted_stages.append(stages)
            if screened_out:
                continue
            values = torch.as_tensor(outcomes, dtype=torch.float64)
            receipt = self.bank.compose_executive(
                parent_slots,
                values,
                threshold=threshold,
                min_observations=min_observations,
                min_stable_observations=min_stable_observations,
            )
            if receipt.accepted:
                return ExecutiveCompositionSearchResult(
                    accepted=True,
                    parent_slots=parent_slots,
                    receipt=receipt,
                    candidate_count=len(candidates),
                    attempted_parent_slots=tuple(attempted_slots),
                    attempted_child_digests=tuple(attempted_digests),
                    attempted_evaluation_stages=tuple(attempted_stages),
                    unique_verifier_bits=unique_bits,
                    unique_logical_lifetimes=unique_lifetimes,
                    replayed_examples=replayed_examples,
                    stable_bits_to_threshold=unique_bits,
                    admission_threshold=threshold,
                    screening_threshold=screening_threshold,
                    min_observations=min_observations,
                    min_stable_observations=min_stable_observations,
                    bank_digest_before=before,
                    bank_digest_after=self.bank.digest(),
                    reason="first stable composition admitted",
                ).validate()
        return ExecutiveCompositionSearchResult(
            accepted=False,
            parent_slots=None,
            receipt=None,
            candidate_count=len(candidates),
            attempted_parent_slots=tuple(attempted_slots),
            attempted_child_digests=tuple(attempted_digests),
            attempted_evaluation_stages=tuple(attempted_stages),
            unique_verifier_bits=unique_bits,
            unique_logical_lifetimes=unique_lifetimes,
            replayed_examples=replayed_examples,
            stable_bits_to_threshold=None,
            admission_threshold=threshold,
            screening_threshold=screening_threshold,
            min_observations=min_observations,
            min_stable_observations=min_stable_observations,
            bank_digest_before=before,
            bank_digest_after=self.bank.digest(),
            reason="no composition candidate cleared the stable-prefix verifier gate",
        ).validate()


__all__ = [
    "AGENT_BRAIN_BANK_ENTRY_SCHEMA",
    "AGENT_BRAIN_EXECUTIVE_KIND",
    "AGENT_BRAIN_TEMPORAL_KIND",
    "EXECUTIVE_COMPOSITION_SEARCH_SCHEMA",
    "EXTERNAL_AGENT_BRAIN_BANK_SCHEMA",
    "ExecutiveCompositionEvaluation",
    "ExecutiveCompositionSearchResult",
    "ExternalAgentBrainBank",
    "ExternalExecutiveCompositionSearch",
]
