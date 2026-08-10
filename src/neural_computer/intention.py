"""External memory for opaque intention discovery and reuse.

The controller emits intentions, but it should not have to remember every
intention it has ever tried.  This module is an append-only, protocol-agnostic
repertoire for those learned output vectors.  It stores only opaque vectors
and verifier statistics; model-based planning remains the authority that
chooses behavior for a goal.

The repertoire deliberately does not rank intentions by reward.  A reward
ranking would quietly become another policy.  It exposes the available
experience to factual search, while an ephemeral controller seed keeps novel
output content discoverable before it has been written to external memory.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

EXTERNAL_INTENTION_REPERTOIRE_SCHEMA = (
    "neural-computer.external-intention-repertoire.v1"
)
EXTERNAL_INTENTION_PROPOSAL_SCHEMA = "neural-computer.external-intention-proposal.v1"
EXTERNAL_INTENTION_OBSERVATION_SCHEMA = (
    "neural-computer.external-intention-observation.v1"
)
EXTERNAL_INTENTION_ADMISSION_SCHEMA = (
    "neural-computer.external-intention-admission.v1"
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
class ExternalIntentionProposal:
    """Runtime-sized opaque candidates available to factual model search."""

    intentions: torch.Tensor
    source_indices: tuple[int, ...]
    propensities: torch.Tensor
    exploration_mask: torch.Tensor
    version: int
    schema: str = EXTERNAL_INTENTION_PROPOSAL_SCHEMA

    def validate(self, *, width: int, batch: int | None = None) -> ExternalIntentionProposal:
        if self.schema != EXTERNAL_INTENTION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported intention-proposal schema")
        _validate_tensor(
            self.intentions,
            name="intention proposal",
            ndim=3,
            width=width,
        )
        candidate_count = self.intentions.shape[1]
        if candidate_count < 1:
            raise ValueError("intention proposal requires one candidate")
        if batch is not None and self.intentions.shape[0] != batch:
            raise ValueError("intention proposal batch differs")
        if len(self.source_indices) != candidate_count:
            raise ValueError("intention proposal source indices are misaligned")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < -1
            for index in self.source_indices
        ):
            raise ValueError("intention proposal source index is invalid")
        expected_shape = (self.intentions.shape[0], candidate_count)
        if self.propensities.shape != expected_shape:
            raise ValueError("intention proposal propensities have the wrong shape")
        if self.exploration_mask.shape != expected_shape or (
            self.exploration_mask.dtype != torch.bool
        ):
            raise ValueError("intention proposal exploration mask has the wrong shape")
        source_exploration = torch.tensor(
            [index == -1 for index in self.source_indices],
            dtype=torch.bool,
            device=self.exploration_mask.device,
        ).unsqueeze(0)
        if not bool(
            torch.equal(
                self.exploration_mask,
                source_exploration.expand_as(self.exploration_mask),
            )
        ):
            raise ValueError("intention proposal exploration flags are inconsistent")
        if not bool(torch.isfinite(self.propensities).all()) or bool(
            torch.any(self.propensities <= 0.0) or torch.any(self.propensities > 1.0)
        ):
            raise ValueError("intention proposal propensities must lie in (0, 1]")
        if not isinstance(self.version, int) or self.version < 0:
            raise ValueError("intention proposal version is invalid")
        return self


@dataclass(frozen=True)
class ExternalIntentionObservationReceipt:
    """Auditable external-memory write for one observed intention batch."""

    entry_indices: tuple[int, ...]
    added: tuple[bool, ...]
    outcome_observed: bool
    version: int
    record_count: int
    content_digest: str
    schema: str = EXTERNAL_INTENTION_OBSERVATION_SCHEMA

    def validate(self) -> ExternalIntentionObservationReceipt:
        if self.schema != EXTERNAL_INTENTION_OBSERVATION_SCHEMA:
            raise ValueError("unsupported intention-observation schema")
        if len(self.entry_indices) != len(self.added) or not self.entry_indices:
            raise ValueError("intention-observation receipt is empty or misaligned")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in self.entry_indices
        ):
            raise ValueError("intention-observation entry index is invalid")
        if not all(isinstance(value, bool) for value in self.added):
            raise TypeError("intention-observation added flags must be boolean")
        if not isinstance(self.outcome_observed, bool):
            raise TypeError("intention-observation outcome flag must be boolean")
        if min(self.version, self.record_count) < 1:
            raise ValueError("intention-observation receipt version is invalid")
        if not isinstance(self.content_digest, str) or not self.content_digest:
            raise ValueError("intention-observation digest is missing")
        return self


@dataclass(frozen=True)
class ExternalIntentionAdmissionReceipt:
    """Copy-on-write admission result for one novel opaque intention."""

    accepted: bool
    entry_index: int | None
    source_record_count: int
    destination_record_count: int
    source_digest: str
    candidate_digest: str
    destination_digest: str
    reason: str
    schema: str = EXTERNAL_INTENTION_ADMISSION_SCHEMA

    def validate(self) -> ExternalIntentionAdmissionReceipt:
        if self.schema != EXTERNAL_INTENTION_ADMISSION_SCHEMA:
            raise ValueError("unsupported intention-admission schema")
        if min(self.source_record_count, self.destination_record_count) < 0:
            raise ValueError("intention-admission record counts cannot be negative")
        if self.accepted:
            if self.entry_index is None or self.entry_index < 0:
                raise ValueError("accepted intention admission has no entry index")
            if self.destination_record_count != self.source_record_count + 1:
                raise ValueError("accepted intention admission has wrong growth")
        elif self.entry_index is not None:
            raise ValueError("rejected intention admission has an entry index")
        for name, value in (
            ("source_digest", self.source_digest),
            ("candidate_digest", self.candidate_digest),
            ("destination_digest", self.destination_digest),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"intention-admission {name} is missing")
        return self


class ExternalIntentionRepertoire:
    """Append-only memory of opaque intention vectors.

    Entries are identified only by their position in this external store.  A
    cosine merge threshold prevents duplicate writes, while the original
    vector is retained so intention magnitude remains part of the learned
    representation.  ``observe`` accepts a scalar verifier outcome and its
    exact logging propensity, accumulating sufficient statistics without
    replaying old evidence or changing controller parameters.
    """

    schema = EXTERNAL_INTENTION_REPERTOIRE_SCHEMA

    def __init__(self, width: int, *, merge_cosine: float = 0.999) -> None:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("intention repertoire width must be positive")
        if not -1.0 <= merge_cosine <= 1.0 or not math.isfinite(merge_cosine):
            raise ValueError("intention repertoire merge cosine is invalid")
        self.width = int(width)
        self.merge_cosine = float(merge_cosine)
        self._intentions: list[torch.Tensor] = []
        self._attempts: list[int] = []
        self._outcome_counts: list[int] = []
        self._utility_sums: list[float] = []
        self._utility_square_sums: list[float] = []
        self._propensity_sums: list[float] = []
        self._inverse_propensity_utility_sums: list[float] = []
        self._last_propensities: list[float] = []
        self._last_seen: list[int] = []
        self._version = 0

    @property
    def record_count(self) -> int:
        return len(self._intentions)

    @property
    def version(self) -> int:
        return self._version

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "storage": "append_only_opaque_intention_experience_v1",
            "proposal": (
                "verified_retrieval_default_plus_explicit_ephemeral_controller_seed_v1"
            ),
            "learning": "outcome_sufficient_statistics_without_replay_v1",
        }

    def _validate_batch(
        self,
        intentions: torch.Tensor,
        utility: torch.Tensor | float | None,
        propensity: torch.Tensor | float | None,
        timestamp: torch.Tensor | int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if intentions.ndim == 1:
            intentions = intentions.unsqueeze(0)
        _validate_tensor(
            intentions,
            name="observed intention",
            ndim=2,
            width=self.width,
        )
        batch = intentions.shape[0]
        if batch < 1:
            raise ValueError("intention observation batch cannot be empty")

        def scalar_batch(
            value: torch.Tensor | float | None,
            *,
            name: str,
            default: float,
        ) -> torch.Tensor:
            if value is None:
                result = torch.full((batch,), default, dtype=torch.float64)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    result = value.reshape(1).expand(batch)
                elif value.shape == (batch,) or value.shape == (batch, 1):
                    result = value.reshape(batch)
                else:
                    raise ValueError(f"{name} must contain one value per intention")
                result = result.detach().to(device="cpu", dtype=torch.float64)
            else:
                result = torch.full((batch,), float(value), dtype=torch.float64)
            if not bool(torch.isfinite(result).all()):
                raise ValueError(f"{name} must be finite")
            return result

        utility_values = None if utility is None else scalar_batch(
            utility, name="intention utility", default=0.0
        )
        propensity_values = scalar_batch(
            propensity, name="intention logging propensity", default=1.0
        )
        if bool(torch.any(propensity_values <= 0.0)) or bool(
            torch.any(propensity_values > 1.0)
        ):
            raise ValueError("intention logging propensities must lie in (0, 1]")

        if timestamp is None:
            timestamp_values = torch.full(
                (batch,), self._version + 1, dtype=torch.int64
            )
        elif isinstance(timestamp, torch.Tensor):
            if timestamp.ndim == 0:
                timestamp_values = timestamp.reshape(1).expand(batch)
            elif timestamp.shape == (batch,) or timestamp.shape == (batch, 1):
                timestamp_values = timestamp.reshape(batch)
            else:
                raise ValueError("intention timestamps must contain one value per intention")
            if timestamp_values.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("intention timestamps must be integer tensors")
            timestamp_values = timestamp_values.detach().to(device="cpu", dtype=torch.int64)
        else:
            timestamp_values = torch.full((batch,), int(timestamp), dtype=torch.int64)
        if bool(torch.any(timestamp_values < 0)):
            raise ValueError("intention timestamps cannot be negative")
        return (
            intentions.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            utility_values,
            propensity_values,
            timestamp_values,
        )

    def _entry_similarity(self, left: torch.Tensor, right: torch.Tensor) -> float:
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        if float(left_norm) <= 1e-12 or float(right_norm) <= 1e-12:
            return 1.0 if torch.equal(left, right) else -1.0
        return float(torch.dot(left, right) / (left_norm * right_norm))

    def _find_entry(self, intention: torch.Tensor) -> int | None:
        for index, stored in enumerate(self._intentions):
            if self._entry_similarity(stored, intention) >= self.merge_cosine:
                return index
        return None

    def _prefix_digest(self, count: int) -> str:
        if not isinstance(count, int) or count < 0 or count > self.record_count:
            raise ValueError("intention prefix count is invalid")
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(str(self.width).encode("utf-8"))
        digest.update(str(count).encode("utf-8"))
        tensors = self.statistics()
        for name in sorted(tensors):
            value = tensors[name][:count].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def admit_verified(
        self,
        intention: torch.Tensor,
        verifier: Callable[[ExternalIntentionRepertoire], bool],
        *,
        reason: str = "caller_owned_heldout_verifier",
    ) -> ExternalIntentionAdmissionReceipt:
        """Admit one novel vector only through an isolated verifier transaction.

        ``verifier`` receives a copy containing the staged vector. It may run
        an independent held-out factual probe and may record the new outcome
        on that copy. Existing entries must remain byte-equivalent and the
        verifier may not add a second entry. Rejection, including a mutating
        verifier, leaves the live repertoire unchanged.
        """

        if not callable(verifier):
            raise TypeError("intention admission verifier must be callable")
        if not isinstance(reason, str) or not reason:
            raise ValueError("intention admission reason is missing")
        normalized, _utility, _propensity, _timestamp = self._validate_batch(
            intention,
            None,
            None,
            None,
        )
        if normalized.shape[0] != 1:
            raise ValueError("intention admission accepts one vector")
        candidate_intention = normalized[0]
        source_count = self.record_count
        source_digest = self.content_digest()
        if self._find_entry(candidate_intention) is not None:
            return ExternalIntentionAdmissionReceipt(
                accepted=False,
                entry_index=None,
                source_record_count=source_count,
                destination_record_count=source_count,
                source_digest=source_digest,
                candidate_digest=source_digest,
                destination_digest=source_digest,
                reason="intention already exists in verified repertoire",
            ).validate()

        candidate = ExternalIntentionRepertoire.from_payload(self.payload())
        candidate.observe(candidate_intention)
        candidate_digest = candidate.content_digest()
        accepted = bool(verifier(candidate))
        prefix_unchanged = candidate._prefix_digest(source_count) == self._prefix_digest(
            source_count
        )
        shape_unchanged = candidate.record_count == source_count + 1
        staged_vector_unchanged = shape_unchanged and torch.equal(
            candidate.statistics()["intentions"][source_count], candidate_intention
        )
        accepted = accepted and prefix_unchanged and shape_unchanged and staged_vector_unchanged
        if accepted:
            self._copy_from(candidate)
            destination_digest = self.content_digest()
            return ExternalIntentionAdmissionReceipt(
                accepted=True,
                entry_index=source_count,
                source_record_count=source_count,
                destination_record_count=self.record_count,
                source_digest=source_digest,
                candidate_digest=candidate_digest,
                destination_digest=destination_digest,
                reason=reason,
            ).validate()
        return ExternalIntentionAdmissionReceipt(
            accepted=False,
            entry_index=None,
            source_record_count=source_count,
            destination_record_count=source_count,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            destination_digest=source_digest,
            reason=(
                "heldout verifier rejected or mutated retained intention state"
            ),
        ).validate()

    def _copy_from(self, other: ExternalIntentionRepertoire) -> None:
        if not isinstance(other, ExternalIntentionRepertoire):
            raise TypeError("intention repertoire replacement must use same type")
        if self.width != other.width or self.merge_cosine != other.merge_cosine:
            raise ValueError("intention repertoire replacement configuration differs")
        other.validate_state()
        self._intentions = [row.clone() for row in other._intentions]
        self._attempts = list(other._attempts)
        self._outcome_counts = list(other._outcome_counts)
        self._utility_sums = list(other._utility_sums)
        self._utility_square_sums = list(other._utility_square_sums)
        self._propensity_sums = list(other._propensity_sums)
        self._inverse_propensity_utility_sums = list(
            other._inverse_propensity_utility_sums
        )
        self._last_propensities = list(other._last_propensities)
        self._last_seen = list(other._last_seen)
        self._version = other._version

    def observe(
        self,
        intentions: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
    ) -> ExternalIntentionObservationReceipt:
        """Record opaque output experience without touching controller weights."""

        (
            normalized_intentions,
            utility_values,
            propensity_values,
            timestamp_values,
        ) = self._validate_batch(intentions, utility, propensity, timestamp)
        located: list[int] = []
        added: list[bool] = []
        for row in normalized_intentions:
            index = self._find_entry(row)
            if index is None:
                self._intentions.append(row.clone())
                self._attempts.append(0)
                self._outcome_counts.append(0)
                self._utility_sums.append(0.0)
                self._utility_square_sums.append(0.0)
                self._propensity_sums.append(0.0)
                self._inverse_propensity_utility_sums.append(0.0)
                self._last_propensities.append(0.0)
                self._last_seen.append(0)
                index = len(self._intentions) - 1
                added.append(True)
            else:
                added.append(False)
            located.append(index)

        self._version += 1
        for row_index, entry_index in enumerate(located):
            self._attempts[entry_index] += 1
            propensity_value = float(propensity_values[row_index])
            self._propensity_sums[entry_index] += propensity_value
            self._last_propensities[entry_index] = propensity_value
            self._last_seen[entry_index] = int(timestamp_values[row_index])
            if utility_values is not None:
                utility_value = float(utility_values[row_index])
                self._outcome_counts[entry_index] += 1
                self._utility_sums[entry_index] += utility_value
                self._utility_square_sums[entry_index] += utility_value * utility_value
                self._inverse_propensity_utility_sums[entry_index] += (
                    utility_value / propensity_value
                )
        self.validate_state()
        return ExternalIntentionObservationReceipt(
            entry_indices=tuple(located),
            added=tuple(added),
            outcome_observed=utility_values is not None,
            version=self._version,
            record_count=self.record_count,
            content_digest=self.content_digest(),
        ).validate()

    def propose(
        self,
        seed_intention: torch.Tensor | None = None,
        *,
        max_candidates: int | None = None,
        include_seed: bool = True,
    ) -> ExternalIntentionProposal:
        """Expose stored experience plus an ephemeral controller seed.

        The proposal is a candidate set, not a decision.  It contains no
        reward-ranked ordering and does not mutate the repertoire.  A novel
        controller seed is marked as exploration until a later observation
        commits it to external memory.  Callers that already have verified
        candidates should set ``include_seed=False`` so an unverified vector
        cannot contaminate factual search; the policy-free runtime does this
        by default and falls back to the seed only for an empty repertoire.
        """

        if not isinstance(include_seed, bool):
            raise TypeError("intention proposal include_seed must be boolean")

        if seed_intention is None:
            seed_batch = None
            output_device = torch.device("cpu")
            output_dtype = torch.float32
        else:
            if seed_intention.ndim == 1:
                seed_batch = seed_intention.unsqueeze(0)
            elif seed_intention.ndim == 2:
                seed_batch = seed_intention
            else:
                raise ValueError("seed intention must be [width] or [batch,width]")
            _validate_tensor(
                seed_batch,
                name="seed intention",
                ndim=2,
                width=self.width,
            )
            output_device = seed_intention.device
            output_dtype = (
                seed_intention.dtype
                if seed_intention.is_floating_point()
                else torch.float32
            )

        stored = [row.clone() for row in self._intentions]
        rows: list[torch.Tensor] = []
        source_indices: list[int] = []
        exploration_flags: list[bool] = []
        if seed_batch is not None and include_seed:
            for seed in seed_batch.detach().to(device="cpu", dtype=torch.float32):
                matching_index = self._find_entry(seed)
                if matching_index is None:
                    if not any(
                        self._entry_similarity(existing, seed) >= self.merge_cosine
                        for existing in rows
                    ):
                        rows.append(seed.clone())
                        source_indices.append(-1)
                        exploration_flags.append(True)
                elif not any(index == matching_index for index in source_indices):
                    rows.append(self._intentions[matching_index].clone())
                    source_indices.append(matching_index)
                    exploration_flags.append(False)
        for index, stored_row in enumerate(stored):
            if index not in source_indices:
                rows.append(stored_row)
                source_indices.append(index)
                exploration_flags.append(False)
        if not rows:
            raise ValueError("intention repertoire cannot propose an empty set")
        if max_candidates is not None:
            if not isinstance(max_candidates, int) or max_candidates < 1:
                raise ValueError("maximum intention candidate count must be positive")
            if max_candidates < sum(exploration_flags):
                raise ValueError("maximum candidate count would discard an exploration seed")
            if len(rows) > max_candidates:
                keep = list(range(max_candidates))
                rows = [rows[index] for index in keep]
                source_indices = [source_indices[index] for index in keep]
                exploration_flags = [exploration_flags[index] for index in keep]
        candidates = torch.stack(rows).to(device=output_device, dtype=output_dtype)
        batch = 1 if seed_batch is None else seed_batch.shape[0]
        intentions = candidates.unsqueeze(0).expand(batch, -1, -1).clone()
        exploration_mask = torch.tensor(
            exploration_flags,
            dtype=torch.bool,
            device=output_device,
        ).unsqueeze(0).expand(batch, -1).clone()
        propensities = torch.full(
            (batch, candidates.shape[0]),
            1.0 / candidates.shape[0],
            dtype=output_dtype,
            device=output_device,
        )
        return ExternalIntentionProposal(
            intentions=intentions,
            source_indices=tuple(source_indices),
            propensities=propensities,
            exploration_mask=exploration_mask,
            version=self._version,
        ).validate(width=self.width, batch=batch)

    def statistics(self) -> dict[str, torch.Tensor]:
        """Return detached sufficient statistics for external diagnostics."""

        self.validate_state()
        return {
            "intentions": self._stack_intentions(),
            "attempts": torch.tensor(self._attempts, dtype=torch.long),
            "outcome_counts": torch.tensor(self._outcome_counts, dtype=torch.long),
            "utility_sums": torch.tensor(self._utility_sums, dtype=torch.float64),
            "utility_square_sums": torch.tensor(
                self._utility_square_sums, dtype=torch.float64
            ),
            "propensity_sums": torch.tensor(
                self._propensity_sums, dtype=torch.float64
            ),
            "inverse_propensity_utility_sums": torch.tensor(
                self._inverse_propensity_utility_sums, dtype=torch.float64
            ),
            "last_propensities": torch.tensor(
                self._last_propensities, dtype=torch.float64
            ),
            "last_seen": torch.tensor(self._last_seen, dtype=torch.long),
        }

    def _stack_intentions(self) -> torch.Tensor:
        if not self._intentions:
            return torch.empty((0, self.width), dtype=torch.float32)
        return torch.stack(self._intentions).detach().clone()

    def validate_state(self) -> None:
        count = self.record_count
        if not isinstance(self._version, int) or self._version < 0:
            raise ValueError("intention repertoire version is invalid")
        lengths = (
            len(self._attempts),
            len(self._outcome_counts),
            len(self._utility_sums),
            len(self._utility_square_sums),
            len(self._propensity_sums),
            len(self._inverse_propensity_utility_sums),
            len(self._last_propensities),
            len(self._last_seen),
        )
        if any(length != count for length in lengths):
            raise ValueError("intention repertoire statistics are misaligned")
        for row in self._intentions:
            _validate_tensor(row, name="stored intention", ndim=1, width=self.width)
        for name, values in (
            ("attempts", self._attempts),
            ("outcome counts", self._outcome_counts),
            ("last seen", self._last_seen),
        ):
            if any(not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"intention repertoire {name} are invalid")
        for name, values in (
            ("utility sums", self._utility_sums),
            ("utility square sums", self._utility_square_sums),
            ("propensity sums", self._propensity_sums),
            ("inverse-propensity utility sums", self._inverse_propensity_utility_sums),
            ("last propensities", self._last_propensities),
        ):
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"intention repertoire {name} are not finite")
        if any(
            outcome_count > attempt
            for outcome_count, attempt in zip(
                self._outcome_counts, self._attempts, strict=True
            )
        ):
            raise ValueError("intention outcome counts exceed attempts")

    @staticmethod
    def _digest_payload(payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(payload):
            if name == "sha256":
                continue
            value = payload[name]
            digest.update(name.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        self.validate_state()
        payload: dict[str, Any] = {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "version": self._version,
            **self.statistics(),
        }
        payload["sha256"] = self._digest_payload(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExternalIntentionRepertoire:
        if payload.get("schema") != EXTERNAL_INTENTION_REPERTOIRE_SCHEMA:
            raise ValueError("unsupported intention repertoire payload")
        width = payload.get("width")
        merge_cosine = payload.get("merge_cosine")
        version = payload.get("version")
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError("intention repertoire payload width is invalid")
        if not isinstance(merge_cosine, (int, float)) or not math.isfinite(
            float(merge_cosine)
        ):
            raise ValueError("intention repertoire payload merge cosine is invalid")
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("intention repertoire payload version is invalid")
        expected_digest = payload.get("sha256")
        if expected_digest != cls._digest_payload(payload):
            raise ValueError("intention repertoire payload checksum mismatch")
        required = (
            "intentions",
            "attempts",
            "outcome_counts",
            "utility_sums",
            "utility_square_sums",
            "propensity_sums",
            "inverse_propensity_utility_sums",
            "last_propensities",
            "last_seen",
        )
        if any(name not in payload for name in required):
            raise ValueError("intention repertoire payload is incomplete")
        repertoire = cls(width, merge_cosine=float(merge_cosine))
        intentions = payload["intentions"]
        if not isinstance(intentions, torch.Tensor) or intentions.ndim != 2:
            raise ValueError("intention repertoire payload vectors are invalid")
        if intentions.shape[1] != width:
            raise ValueError("intention repertoire payload vector width differs")
        count = intentions.shape[0]
        tensors = {
            name: payload[name]
            for name in required
            if name != "intentions"
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor) or value.shape[0] != count:
                raise ValueError(f"intention repertoire payload {name} is misaligned")
        repertoire._intentions = [
            row.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for row in intentions
        ]
        repertoire._attempts = [int(value) for value in tensors["attempts"].tolist()]
        repertoire._outcome_counts = [
            int(value) for value in tensors["outcome_counts"].tolist()
        ]
        repertoire._utility_sums = [
            float(value) for value in tensors["utility_sums"].tolist()
        ]
        repertoire._utility_square_sums = [
            float(value) for value in tensors["utility_square_sums"].tolist()
        ]
        repertoire._propensity_sums = [
            float(value) for value in tensors["propensity_sums"].tolist()
        ]
        repertoire._inverse_propensity_utility_sums = [
            float(value)
            for value in tensors["inverse_propensity_utility_sums"].tolist()
        ]
        repertoire._last_propensities = [
            float(value) for value in tensors["last_propensities"].tolist()
        ]
        repertoire._last_seen = [int(value) for value in tensors["last_seen"].tolist()]
        repertoire._version = version
        repertoire.validate_state()
        return repertoire

    def content_digest(self) -> str:
        return self._digest_payload(self.payload_without_digest())

    def payload_without_digest(self) -> dict[str, Any]:
        self.validate_state()
        return {
            "schema": self.schema,
            "width": self.width,
            "merge_cosine": self.merge_cosine,
            "version": self._version,
            **self.statistics(),
        }


__all__ = [
    "EXTERNAL_INTENTION_ADMISSION_SCHEMA",
    "EXTERNAL_INTENTION_OBSERVATION_SCHEMA",
    "EXTERNAL_INTENTION_PROPOSAL_SCHEMA",
    "EXTERNAL_INTENTION_REPERTOIRE_SCHEMA",
    "ExternalIntentionAdmissionReceipt",
    "ExternalIntentionObservationReceipt",
    "ExternalIntentionProposal",
    "ExternalIntentionRepertoire",
]
