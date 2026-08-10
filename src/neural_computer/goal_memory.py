"""External opaque goal fragments for factual model-based search.

Goals are memory content, not controller semantics.  A fragment is an opaque
state target plus a learned/verified mask describing which coordinates it
constrains.  Fragments can be composed as a union (satisfy any fragment) or an
intersection (satisfy every fragment).  The controller never receives the
fragment address or the composition metadata; the factual planner consumes
only the resulting goal set.

This is intentionally a storage and composition boundary.  A caller-owned
verifier decides when a new fragment is safe to admit and when old fragments
may be retired.  No task label or hand-assigned coordinate meaning is stored.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from .representation import DEFAULT_STATE_SPACE_ID, validate_representation_space_id

EXTERNAL_GOAL_FRAGMENT_SCHEMA = "neural-computer.external-goal-fragment.v1"
EXTERNAL_GOAL_FRAGMENT_SET_SCHEMA = "neural-computer.external-goal-fragment-set.v1"
EXTERNAL_GOAL_FRAGMENT_MEMORY_SCHEMA = (
    "neural-computer.external-goal-fragment-memory.v1"
)
EXTERNAL_GOAL_FRAGMENT_ADMISSION_SCHEMA = (
    "neural-computer.external-goal-fragment-admission.v1"
)


def _validate_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    width: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if width is not None and value.shape[-1] != width:
        raise ValueError(f"{name} has the wrong width")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ExternalGoalFragmentSet:
    """Runtime-sized opaque goal fragments consumed by the planner.

    ``values`` and ``masks`` have shape ``[batch, fragments, state_width]``.
    A fragment is satisfied by matching its masked coordinates.  ``union``
    selects the nearest satisfied fragment; ``intersection`` requires every
    fragment and therefore scores the worst fragment.  These are structural
    composition operators, not semantic fields in the learned state.
    """

    values: torch.Tensor
    masks: torch.Tensor
    composition: str = "union"
    fragment_ids: tuple[int, ...] = ()
    schema: str = EXTERNAL_GOAL_FRAGMENT_SET_SCHEMA

    def validate(
        self,
        *,
        state_width: int,
        batch: int | None = None,
    ) -> ExternalGoalFragmentSet:
        if self.schema != EXTERNAL_GOAL_FRAGMENT_SET_SCHEMA:
            raise ValueError("unsupported goal-fragment set schema")
        if self.composition not in {"union", "intersection"}:
            raise ValueError("goal-fragment composition must be union or intersection")
        if self.values.ndim != 3 or self.values.shape[-1] != state_width:
            raise ValueError(
                "goal-fragment values must be [batch,fragments,state_width]"
            )
        if self.masks.shape != self.values.shape or self.masks.dtype is not torch.bool:
            raise ValueError("goal-fragment masks must match values and be boolean")
        if self.values.shape[1] < 1:
            raise ValueError("goal-fragment set cannot be empty")
        if batch is not None and self.values.shape[0] != batch:
            raise ValueError("goal-fragment batch does not match state batch")
        if self.fragment_ids and (
            len(self.fragment_ids) != self.values.shape[1]
            or tuple(sorted(set(self.fragment_ids))) != self.fragment_ids
            or any(index < 0 for index in self.fragment_ids)
        ):
            raise ValueError("goal-fragment IDs are invalid")
        _validate_tensor(
            self.values, name="goal-fragment values", ndim=3, width=state_width
        )
        return self


@dataclass(frozen=True)
class ExternalGoalFragmentAdmissionReceipt:
    """Copy-on-write admission result for one opaque destination fragment."""

    accepted: bool
    fragment_id: int | None
    source_count: int
    destination_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_FRAGMENT_ADMISSION_SCHEMA

    def validate(self) -> ExternalGoalFragmentAdmissionReceipt:
        if self.schema != EXTERNAL_GOAL_FRAGMENT_ADMISSION_SCHEMA:
            raise ValueError("unsupported goal-fragment admission schema")
        if min(self.source_count, self.destination_count) < 0:
            raise ValueError("goal-fragment admission counts are invalid")
        if self.accepted:
            if self.fragment_id != self.source_count:
                raise ValueError("accepted goal-fragment ID is invalid")
            if self.destination_count != self.source_count + 1:
                raise ValueError("accepted goal-fragment count is invalid")
        elif self.fragment_id is not None:
            raise ValueError("rejected goal-fragment admission has an ID")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"goal-fragment admission {name} is missing")
        return self


class ExternalGoalFragmentMemory:
    """Append-only, independently persisted memory for opaque destinations."""

    schema = EXTERNAL_GOAL_FRAGMENT_MEMORY_SCHEMA

    def __init__(
        self,
        state_width: int,
        *,
        state_space_id: str = DEFAULT_STATE_SPACE_ID,
    ) -> None:
        if (
            not isinstance(state_width, int)
            or isinstance(state_width, bool)
            or state_width < 1
        ):
            raise ValueError("goal-fragment state width must be positive")
        self.state_width = int(state_width)
        self.state_space_id = validate_representation_space_id(
            state_space_id,
            name="goal_fragment_state_space_id",
        )
        self._values: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []
        self._version = 0

    @property
    def fragment_count(self) -> int:
        return len(self._values)

    @property
    def version(self) -> int:
        return self._version

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "state_space_id": self.state_space_id,
            "storage": "append_only_opaque_goal_fragments_v1",
            "composition": "runtime_union_or_intersection_v1",
            "controller": "not_serialized_or_mutated_v1",
        }

    def _validate_fragment(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)
        _validate_tensor(
            values, name="goal-fragment values", ndim=2, width=self.state_width
        )
        if masks.shape != values.shape or masks.dtype is not torch.bool:
            raise ValueError("goal-fragment masks must match values and be boolean")
        if values.shape[0] != 1:
            raise ValueError("one goal-fragment admission accepts one fragment")
        if not bool(masks.any()):
            raise ValueError("goal-fragment mask cannot be empty")
        return values.detach().cpu().clone(), masks.detach().cpu().clone()

    def append(self, values: torch.Tensor, masks: torch.Tensor) -> int:
        """Append a fragment; callers should prefer verified admission."""

        value, mask = self._validate_fragment(values, masks)
        self._values.append(value.squeeze(0))
        self._masks.append(mask.squeeze(0))
        self._version += 1
        return len(self._values) - 1

    def _clone(self) -> ExternalGoalFragmentMemory:
        clone = ExternalGoalFragmentMemory(
            self.state_width,
            state_space_id=self.state_space_id,
        )
        clone._values = [value.clone() for value in self._values]
        clone._masks = [mask.clone() for mask in self._masks]
        clone._version = self._version
        return clone

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(str(self.state_width).encode("utf-8"))
        digest.update(self.state_space_id.encode("utf-8"))
        for value, mask in zip(self._values, self._masks):
            for tensor in (value, mask):
                detached = tensor.detach().cpu().contiguous()
                digest.update(str(detached.dtype).encode("utf-8"))
                digest.update(repr(tuple(detached.shape)).encode("utf-8"))
                digest.update(detached.numpy().tobytes())
        return digest.hexdigest()

    def admit_verified(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        retention_probe: Callable[[ExternalGoalFragmentMemory], bool],
        *,
        reason: str = "caller_owned_heldout_goal_retention_probe",
    ) -> ExternalGoalFragmentAdmissionReceipt:
        """Admit one destination only if an independent probe accepts the copy."""

        if not callable(retention_probe):
            raise TypeError("goal-fragment retention probe must be callable")
        candidate_values, candidate_masks = self._validate_fragment(values, masks)
        source_count = self.fragment_count
        source_digest = self.digest()
        candidate = self._clone()
        candidate.append(candidate_values, candidate_masks)
        candidate_digest = candidate.digest()
        accepted = bool(retention_probe(candidate))
        if accepted:
            self._values = candidate._values
            self._masks = candidate._masks
            self._version = candidate._version
            destination_digest = self.digest()
            fragment_id: int | None = source_count
        else:
            destination_digest = source_digest
            fragment_id = None
        return ExternalGoalFragmentAdmissionReceipt(
            accepted=accepted,
            fragment_id=fragment_id,
            source_count=source_count,
            destination_count=self.fragment_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=destination_digest,
            reason=reason
            if accepted
            else "goal-fragment heldout retention probe rejected",
        ).validate()

    def propose(
        self,
        indices: Sequence[int] | torch.Tensor | None = None,
        *,
        batch_size: int = 1,
        composition: str = "union",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalGoalFragmentSet:
        """Read opaque destination fragments without changing memory."""

        if self.fragment_count < 1:
            raise ValueError("goal-fragment memory is empty")
        if batch_size < 1:
            raise ValueError("goal-fragment batch size must be positive")
        if indices is None:
            selected = tuple(range(self.fragment_count))
        elif isinstance(indices, torch.Tensor):
            if indices.ndim != 1:
                raise ValueError("goal-fragment indices must be one-dimensional")
            selected = tuple(int(index) for index in indices.detach().cpu().tolist())
        else:
            selected = tuple(int(index) for index in indices)
        if not selected or tuple(sorted(set(selected))) != selected:
            raise ValueError("goal-fragment indices must be ordered and unique")
        if selected[0] < 0 or selected[-1] >= self.fragment_count:
            raise IndexError("goal-fragment index is out of range")
        values = torch.stack([self._values[index] for index in selected]).to(
            device=device, dtype=dtype
        )
        masks = torch.stack([self._masks[index] for index in selected]).to(
            device=device
        )
        result = ExternalGoalFragmentSet(
            values=values.unsqueeze(0).expand(batch_size, -1, -1),
            masks=masks.unsqueeze(0).expand(batch_size, -1, -1),
            composition=composition,
            fragment_ids=selected,
        )
        return result.validate(state_width=self.state_width, batch=batch_size)

    def state_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "version": self.version,
            "values": [value.clone() for value in self._values],
            "masks": [mask.clone() for mask in self._masks],
        }
        payload["sha256"] = self._payload_digest(payload)
        return payload

    @staticmethod
    def _payload_digest(payload: Mapping[str, object]) -> str:
        digest = hashlib.sha256()
        for key in ("schema", "configuration", "version", "values", "masks"):
            value = payload[key]
            digest.update(str(key).encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            elif isinstance(value, list):
                for item in value:
                    tensor = item.detach().cpu().contiguous()
                    digest.update(str(tensor.dtype).encode("utf-8"))
                    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                    digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExternalGoalFragmentMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported goal-fragment memory payload")
        configuration = payload.get("configuration")
        values = payload.get("values")
        masks = payload.get("masks")
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(values, list)
            or not isinstance(masks, list)
        ):
            raise TypeError("goal-fragment memory payload is incomplete")
        if len(values) != len(masks):
            raise ValueError("goal-fragment value/mask counts differ")
        memory = cls(
            int(configuration["state_width"]),
            state_space_id=str(configuration["state_space_id"]),
        )
        for value, mask in zip(values, masks):
            if not isinstance(value, torch.Tensor) or not isinstance(
                mask, torch.Tensor
            ):
                raise TypeError("goal-fragment payload tensors are invalid")
            memory.append(value, mask)
        if payload.get("version") != memory.version:
            raise ValueError("goal-fragment memory version is inconsistent")
        if payload.get("sha256") != memory._payload_digest(
            {
                "schema": payload["schema"],
                "configuration": configuration,
                "version": payload["version"],
                "values": values,
                "masks": masks,
            }
        ):
            raise ValueError("goal-fragment memory checksum mismatch")
        return memory


__all__ = [
    "EXTERNAL_GOAL_FRAGMENT_ADMISSION_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_MEMORY_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_SET_SCHEMA",
    "ExternalGoalFragmentAdmissionReceipt",
    "ExternalGoalFragmentMemory",
    "ExternalGoalFragmentSet",
]
