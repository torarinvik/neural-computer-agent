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
import math
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
EXTERNAL_GOAL_FRAGMENT_CANDIDATE_SCHEMA = (
    "neural-computer.external-goal-fragment-candidate.v1"
)
EXTERNAL_GOAL_FRAGMENT_STAGER_SCHEMA = (
    "neural-computer.external-goal-fragment-stager.v1"
)
EXTERNAL_GOAL_FRAGMENT_OBSERVATION_SCHEMA = (
    "neural-computer.external-goal-fragment-observation.v1"
)
EXTERNAL_GOAL_FRAGMENT_STAGING_ADMISSION_SCHEMA = (
    "neural-computer.external-goal-fragment-staging-admission.v1"
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
class ExternalGoalFragmentCandidate:
    """One opaque destination proposed from a learned state observation.

    Candidates are deliberately separate from durable memory.  A caller may
    derive one from a terminal controller/model state, but the candidate has
    no task name, coordinate semantics, or verifier answer attached to it.
    """

    values: torch.Tensor
    masks: torch.Tensor
    schema: str = EXTERNAL_GOAL_FRAGMENT_CANDIDATE_SCHEMA

    @classmethod
    def from_state(
        cls,
        state: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> ExternalGoalFragmentCandidate:
        """Create a candidate from one learned terminal state.

        The state is already inside the learned representation boundary.  The
        optional mask is an opaque learned relevance mask; when omitted, the
        candidate constrains every coordinate without assigning any meaning
        to those coordinates.
        """

        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        if state.ndim != 1:
            raise ValueError("goal-fragment candidate state must contain one row")
        if mask is None:
            mask = torch.ones(state.shape, dtype=torch.bool, device=state.device)
        elif mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        return cls(state, mask)

    def validate(self, *, state_width: int) -> ExternalGoalFragmentCandidate:
        if self.schema != EXTERNAL_GOAL_FRAGMENT_CANDIDATE_SCHEMA:
            raise ValueError("unsupported goal-fragment candidate schema")
        if self.values.ndim == 2 and self.values.shape[0] == 1:
            values = self.values.squeeze(0)
        else:
            values = self.values
        if self.masks.ndim == 2 and self.masks.shape[0] == 1:
            masks = self.masks.squeeze(0)
        else:
            masks = self.masks
        _validate_tensor(
            values, name="goal-fragment candidate values", ndim=1, width=state_width
        )
        if masks.shape != values.shape or masks.dtype is not torch.bool:
            raise ValueError(
                "goal-fragment candidate masks must match values and be boolean"
            )
        if not bool(masks.any()):
            raise ValueError("goal-fragment candidate mask cannot be empty")
        return self

    def tensors(self, *, state_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate(state_width=state_width)
        values = self.values.squeeze(0) if self.values.ndim == 2 else self.values
        masks = self.masks.squeeze(0) if self.masks.ndim == 2 else self.masks
        return values.detach().cpu().clone(), masks.detach().cpu().clone()

    def digest(self, *, state_width: int) -> str:
        values, masks = self.tensors(state_width=state_width)
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for tensor in (values, masks):
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.contiguous().numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class ExternalGoalFragmentObservationReceipt:
    """Aggregate state after one fresh scalar observation.

    The stager retains counts and sufficient statistics only.  It never keeps
    the outcome stream, event rows, or a replayable trajectory.
    """

    candidate_digest: str
    eligible: bool
    observations: int
    outcome_sum: float
    prefix_mean: float | None
    minimum_stable_prefix_mean: float | None
    stable_observations: int
    ready: bool
    schema: str = EXTERNAL_GOAL_FRAGMENT_OBSERVATION_SCHEMA

    def validate(self) -> ExternalGoalFragmentObservationReceipt:
        if self.schema != EXTERNAL_GOAL_FRAGMENT_OBSERVATION_SCHEMA:
            raise ValueError("unsupported goal-fragment observation schema")
        if len(self.candidate_digest) != 64:
            raise ValueError("goal-fragment candidate digest is malformed")
        try:
            int(self.candidate_digest, 16)
        except ValueError as error:
            raise ValueError("goal-fragment candidate digest is malformed") from error
        if self.observations < 0 or self.stable_observations < 0:
            raise ValueError("goal-fragment observation counts are invalid")
        if not math.isfinite(self.outcome_sum) or self.outcome_sum < 0.0:
            raise ValueError("goal-fragment outcome sum is invalid")
        for name, value in (
            ("prefix_mean", self.prefix_mean),
            ("minimum_stable_prefix_mean", self.minimum_stable_prefix_mean),
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"goal-fragment {name} is invalid")
        if self.observations == 0 and (
            self.prefix_mean is not None
            or self.minimum_stable_prefix_mean is not None
            or self.stable_observations != 0
        ):
            raise ValueError("empty goal-fragment observations have statistics")
        if self.stable_observations > self.observations:
            raise ValueError("goal-fragment stable observations exceed observations")
        return self


@dataclass(frozen=True)
class ExternalGoalFragmentStagingAdmissionReceipt:
    """Result of promoting one staged candidate into durable goal memory."""

    accepted: bool
    candidate_digest: str
    observations: int
    stable_observations: int
    fragment_id: int | None
    source_count: int
    destination_count: int
    source_digest: str
    candidate_memory_digest: str
    destination_digest: str
    reason: str
    schema: str = EXTERNAL_GOAL_FRAGMENT_STAGING_ADMISSION_SCHEMA

    def validate(self) -> ExternalGoalFragmentStagingAdmissionReceipt:
        if self.schema != EXTERNAL_GOAL_FRAGMENT_STAGING_ADMISSION_SCHEMA:
            raise ValueError("unsupported goal-fragment staging admission schema")
        for name, digest in (
            ("candidate_digest", self.candidate_digest),
            ("source_digest", self.source_digest),
            ("candidate_memory_digest", self.candidate_memory_digest),
            ("destination_digest", self.destination_digest),
        ):
            if len(digest) != 64:
                raise ValueError(f"goal-fragment {name} is malformed")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(f"goal-fragment {name} is malformed") from error
        if (
            min(
                self.observations,
                self.stable_observations,
                self.source_count,
                self.destination_count,
            )
            < 0
        ):
            raise ValueError("goal-fragment staging admission counts are invalid")
        if self.stable_observations > self.observations:
            raise ValueError("goal-fragment stable observations exceed observations")
        if self.accepted and self.fragment_id != self.source_count:
            raise ValueError("accepted goal-fragment staging ID is invalid")
        if not self.accepted and self.fragment_id is not None:
            raise ValueError("rejected goal-fragment staging has an ID")
        if not self.reason:
            raise ValueError("goal-fragment staging admission reason is missing")
        if not self.accepted and self.source_digest != self.destination_digest:
            raise ValueError("rejected goal-fragment staging changed live memory")
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

    def propose_per_batch(
        self,
        indices: Sequence[Sequence[int]] | torch.Tensor,
        *,
        batch_size: int,
        composition: str = "union",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ExternalGoalFragmentSet:
        """Read independently selected opaque fragments for each batch row.

        Unlike :meth:`propose`, this method does not broadcast one shared
        address set across the batch.  Each row may select a different
        ordered fragment tuple, which keeps binding information intact when
        an external route learner has acquired context-conditioned evidence.
        Fragment IDs are intentionally omitted from the controller-facing set:
        the address is memory metadata, not a learned semantic feature.
        """

        if self.fragment_count < 1:
            raise ValueError("goal-fragment memory is empty")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("goal-fragment batch size must be positive")
        if isinstance(indices, torch.Tensor):
            if indices.ndim == 1:
                if indices.shape[0] != batch_size:
                    raise ValueError(
                        "per-batch goal-fragment indices must match batch size"
                    )
                rows = [[int(index)] for index in indices.detach().cpu().tolist()]
            elif indices.ndim == 2:
                if indices.shape[0] != batch_size:
                    raise ValueError(
                        "per-batch goal-fragment indices must match batch size"
                    )
                rows = [
                    [int(index) for index in row]
                    for row in indices.detach().cpu().tolist()
                ]
            else:
                raise ValueError(
                    "per-batch goal-fragment indices must be one- or two-dimensional"
                )
        else:
            rows = [[int(index) for index in row] for row in indices]
            if len(rows) != batch_size:
                raise ValueError("per-batch goal-fragment indices must match batch size")

        if any(
            not row or tuple(sorted(set(row))) != tuple(row)
            for row in rows
        ):
            raise ValueError(
                "per-batch goal-fragment indices must be ordered and unique"
            )
        if any(
            index < 0 or index >= self.fragment_count
            for row in rows
            for index in row
        ):
            raise IndexError("goal-fragment index is out of range")

        values = torch.stack(
            [
                torch.stack([self._values[index] for index in row])
                for row in rows
            ]
        ).to(device=device, dtype=dtype)
        masks = torch.stack(
            [
                torch.stack([self._masks[index] for index in row])
                for row in rows
            ]
        ).to(device=device)
        result = ExternalGoalFragmentSet(
            values=values,
            masks=masks,
            composition=composition,
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


@dataclass
class _GoalFragmentStagingRecord:
    candidate: ExternalGoalFragmentCandidate
    observations: int = 0
    outcome_sum: float = 0.0
    minimum_stable_prefix_mean: float | None = None


class ExternalGoalFragmentStager:
    """Outcome-only staging for destinations proposed from learned states.

    The stager is a memory-side learner, not part of the controller.  It
    retains one opaque candidate tensor pair plus scalar sufficient statistics
    per pending candidate.  It never stores verifier rows or a replay buffer.
    A candidate becomes eligible for admission only after its prefix mean has
    remained above the threshold for the configured stable prefix.
    """

    schema = EXTERNAL_GOAL_FRAGMENT_STAGER_SCHEMA

    def __init__(
        self,
        state_width: int,
        *,
        threshold: float = 0.8,
        min_observations: int = 4,
        min_stable_observations: int = 2,
        max_pending_candidates: int = 128,
        state_space_id: str = DEFAULT_STATE_SPACE_ID,
    ) -> None:
        if (
            not isinstance(state_width, int)
            or isinstance(state_width, bool)
            or state_width < 1
        ):
            raise ValueError("goal-fragment stager state width must be positive")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("goal-fragment stager threshold must lie in [0, 1]")
        if min_observations < 1 or min_stable_observations < 1:
            raise ValueError("goal-fragment stager observation counts must be positive")
        if max_pending_candidates < 1:
            raise ValueError("goal-fragment stager capacity must be positive")
        self.state_width = int(state_width)
        self.threshold = float(threshold)
        self.min_observations = int(min_observations)
        self.min_stable_observations = int(min_stable_observations)
        self.max_pending_candidates = int(max_pending_candidates)
        self.state_space_id = validate_representation_space_id(
            state_space_id,
            name="goal_fragment_stager_state_space_id",
        )
        self._records: dict[str, _GoalFragmentStagingRecord] = {}

    @property
    def pending_count(self) -> int:
        return len(self._records)

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "state_space_id": self.state_space_id,
            "threshold": self.threshold,
            "min_observations": self.min_observations,
            "min_stable_observations": self.min_stable_observations,
            "max_pending_candidates": self.max_pending_candidates,
            "storage": "opaque_candidate_plus_scalar_sufficient_statistics_v1",
            "replay": "zero_verifier_rows_retained_v1",
            "controller": "not_serialized_or_mutated_v1",
        }

    def _record(
        self,
        candidate: ExternalGoalFragmentCandidate,
    ) -> tuple[str, _GoalFragmentStagingRecord]:
        values, masks = candidate.tensors(state_width=self.state_width)
        normalized = ExternalGoalFragmentCandidate(values, masks)
        digest = normalized.digest(state_width=self.state_width)
        record = self._records.get(digest)
        if record is None:
            if self.pending_count >= self.max_pending_candidates:
                raise MemoryError("goal-fragment staging capacity is full")
            record = _GoalFragmentStagingRecord(normalized)
            self._records[digest] = record
        return digest, record

    @staticmethod
    def _scalar_outcome(outcome: torch.Tensor | float) -> float:
        value = float(torch.as_tensor(outcome, dtype=torch.float64).item())
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("goal-fragment outcomes must be finite values in [0, 1]")
        return value

    def observe(
        self,
        candidate: ExternalGoalFragmentCandidate,
        outcome: torch.Tensor | float,
        *,
        eligible: bool = True,
    ) -> ExternalGoalFragmentObservationReceipt:
        """Record one fresh scalar outcome without retaining the experience."""

        candidate.validate(state_width=self.state_width)
        if not isinstance(eligible, bool):
            raise TypeError("goal-fragment eligibility must be boolean")
        digest, record = self._record(candidate)
        if eligible:
            value = self._scalar_outcome(outcome)
            record.observations += 1
            record.outcome_sum += value
            if record.observations >= self.min_observations:
                prefix_mean = record.outcome_sum / record.observations
                if record.minimum_stable_prefix_mean is None:
                    record.minimum_stable_prefix_mean = prefix_mean
                else:
                    record.minimum_stable_prefix_mean = min(
                        record.minimum_stable_prefix_mean,
                        prefix_mean,
                    )
        prefix_mean = (
            None
            if record.observations == 0
            else record.outcome_sum / record.observations
        )
        stable_observations = max(
            0,
            record.observations - self.min_observations + 1,
        )
        ready = bool(
            record.observations >= self.min_observations
            and stable_observations >= self.min_stable_observations
            and record.minimum_stable_prefix_mean is not None
            and record.minimum_stable_prefix_mean >= self.threshold
        )
        return ExternalGoalFragmentObservationReceipt(
            candidate_digest=digest,
            eligible=eligible,
            observations=record.observations,
            outcome_sum=record.outcome_sum,
            prefix_mean=prefix_mean,
            minimum_stable_prefix_mean=record.minimum_stable_prefix_mean,
            stable_observations=stable_observations,
            ready=ready,
        ).validate()

    def observe_state(
        self,
        state: torch.Tensor,
        outcome: torch.Tensor | float,
        *,
        mask: torch.Tensor | None = None,
        eligible: bool = True,
    ) -> ExternalGoalFragmentObservationReceipt:
        """Stage one learned state directly from a fresh scalar outcome."""

        candidate = ExternalGoalFragmentCandidate.from_state(state, mask=mask)
        return self.observe(candidate, outcome, eligible=eligible)

    def candidate(self, digest: str) -> ExternalGoalFragmentCandidate:
        """Return a detached pending candidate by opaque content digest."""

        record = self._records.get(digest)
        if record is None:
            raise KeyError("unknown goal-fragment candidate digest")
        values, masks = record.candidate.tensors(state_width=self.state_width)
        return ExternalGoalFragmentCandidate(values, masks)

    def observation(self, digest: str) -> ExternalGoalFragmentObservationReceipt:
        """Read aggregate evidence for one pending candidate."""

        record = self._records.get(digest)
        if record is None:
            raise KeyError("unknown goal-fragment candidate digest")
        prefix_mean = (
            None
            if record.observations == 0
            else record.outcome_sum / record.observations
        )
        stable_observations = max(
            0,
            record.observations - self.min_observations + 1,
        )
        ready = bool(
            record.observations >= self.min_observations
            and stable_observations >= self.min_stable_observations
            and record.minimum_stable_prefix_mean is not None
            and record.minimum_stable_prefix_mean >= self.threshold
        )
        return ExternalGoalFragmentObservationReceipt(
            candidate_digest=digest,
            eligible=True,
            observations=record.observations,
            outcome_sum=record.outcome_sum,
            prefix_mean=prefix_mean,
            minimum_stable_prefix_mean=record.minimum_stable_prefix_mean,
            stable_observations=stable_observations,
            ready=ready,
        ).validate()

    def admit_verified(
        self,
        memory: ExternalGoalFragmentMemory,
        digest: str,
        retention_probe: Callable[[ExternalGoalFragmentMemory], bool],
        *,
        reason: str = "caller_owned_heldout_goal_fragment_probe",
    ) -> ExternalGoalFragmentStagingAdmissionReceipt:
        """Promote a ready candidate through memory's copy-on-write gate."""

        if not isinstance(memory, ExternalGoalFragmentMemory):
            raise TypeError("goal-fragment staging needs external goal memory")
        record = self._records.get(digest)
        if record is None:
            raise KeyError("unknown goal-fragment candidate digest")
        evidence = self.observation(digest)
        source_digest = memory.digest()
        if not evidence.ready:
            return ExternalGoalFragmentStagingAdmissionReceipt(
                accepted=False,
                candidate_digest=digest,
                observations=evidence.observations,
                stable_observations=evidence.stable_observations,
                fragment_id=None,
                source_count=memory.fragment_count,
                destination_count=memory.fragment_count,
                source_digest=source_digest,
                candidate_memory_digest=source_digest,
                destination_digest=source_digest,
                reason="goal-fragment candidate has not cleared a stable prefix",
            ).validate()
        values, masks = record.candidate.tensors(state_width=self.state_width)
        receipt = memory.admit_verified(
            values,
            masks,
            retention_probe,
            reason=reason,
        )
        if receipt.accepted:
            self._records.pop(digest)
        return ExternalGoalFragmentStagingAdmissionReceipt(
            accepted=receipt.accepted,
            candidate_digest=digest,
            observations=evidence.observations,
            stable_observations=evidence.stable_observations,
            fragment_id=receipt.fragment_id,
            source_count=receipt.source_count,
            destination_count=receipt.destination_count,
            source_digest=receipt.source_digest,
            candidate_memory_digest=receipt.candidate_digest,
            destination_digest=receipt.destination_digest,
            reason=receipt.reason,
        ).validate()

    def state_payload(self) -> dict[str, object]:
        records: list[dict[str, object]] = []
        for digest, record in sorted(self._records.items()):
            values, masks = record.candidate.tensors(state_width=self.state_width)
            records.append(
                {
                    "digest": digest,
                    "values": values,
                    "masks": masks,
                    "observations": record.observations,
                    "outcome_sum": record.outcome_sum,
                    "minimum_stable_prefix_mean": record.minimum_stable_prefix_mean,
                }
            )
        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "records": records,
        }
        payload["sha256"] = self._payload_digest(payload)
        return payload

    @staticmethod
    def _payload_digest(payload: Mapping[str, object]) -> str:
        digest = hashlib.sha256()
        digest.update(str(payload["schema"]).encode("utf-8"))
        digest.update(repr(payload["configuration"]).encode("utf-8"))
        records = payload["records"]
        if not isinstance(records, list):
            raise TypeError("goal-fragment stager records must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("goal-fragment stager record must be a mapping")
            for key in (
                "digest",
                "values",
                "masks",
                "observations",
                "outcome_sum",
                "minimum_stable_prefix_mean",
            ):
                value = record[key]
                digest.update(key.encode("utf-8"))
                if isinstance(value, torch.Tensor):
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str(tensor.dtype).encode("utf-8"))
                    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                    digest.update(tensor.numpy().tobytes())
                else:
                    digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> ExternalGoalFragmentStager:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported goal-fragment stager payload")
        configuration = payload.get("configuration")
        records = payload.get("records")
        if not isinstance(configuration, Mapping) or not isinstance(records, list):
            raise TypeError("goal-fragment stager payload is incomplete")
        stager = cls(
            int(configuration["state_width"]),
            threshold=float(configuration["threshold"]),
            min_observations=int(configuration["min_observations"]),
            min_stable_observations=int(configuration["min_stable_observations"]),
            max_pending_candidates=int(configuration["max_pending_candidates"]),
            state_space_id=str(configuration["state_space_id"]),
        )
        for item in records:
            if not isinstance(item, Mapping):
                raise TypeError("goal-fragment stager record is invalid")
            values = item.get("values")
            masks = item.get("masks")
            if not isinstance(values, torch.Tensor) or not isinstance(
                masks, torch.Tensor
            ):
                raise TypeError("goal-fragment stager record tensors are invalid")
            candidate = ExternalGoalFragmentCandidate(values, masks)
            digest = candidate.digest(state_width=stager.state_width)
            if digest != item.get("digest"):
                raise ValueError("goal-fragment stager candidate checksum mismatch")
            _, record = stager._record(candidate)
            record.observations = int(item["observations"])
            record.outcome_sum = float(item["outcome_sum"])
            minimum = item["minimum_stable_prefix_mean"]
            record.minimum_stable_prefix_mean = (
                None if minimum is None else float(minimum)
            )
            stager.observation(digest)
        expected = payload.get("sha256")
        if expected != stager._payload_digest(
            {
                "schema": payload["schema"],
                "configuration": configuration,
                "records": records,
            }
        ):
            raise ValueError("goal-fragment stager checksum mismatch")
        return stager


__all__ = [
    "EXTERNAL_GOAL_FRAGMENT_ADMISSION_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_CANDIDATE_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_MEMORY_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_OBSERVATION_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_SET_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_STAGER_SCHEMA",
    "EXTERNAL_GOAL_FRAGMENT_STAGING_ADMISSION_SCHEMA",
    "ExternalGoalFragmentAdmissionReceipt",
    "ExternalGoalFragmentCandidate",
    "ExternalGoalFragmentMemory",
    "ExternalGoalFragmentObservationReceipt",
    "ExternalGoalFragmentSet",
    "ExternalGoalFragmentStager",
    "ExternalGoalFragmentStagingAdmissionReceipt",
]
