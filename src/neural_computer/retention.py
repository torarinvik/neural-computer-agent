"""Retention-safe state for continual learning without controller replay.

The controller is not updated by this module.  It tracks opaque capability
addresses and scalar verifier outcomes at the external-memory boundary, then
protects capabilities that have demonstrated stable mastery.  This gives a
frozen processor a growing, replaceable memory-side state while keeping
reversal handling explicit and conservative.

No task names, semantic labels, correct actions, or protocol fields are
stored.  A caller supplies only a learned opaque key and a deterministic
outcome in ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

RETENTION_LEDGER_SCHEMA = "neural-computer.capability-retention-ledger.v1"


@dataclass(frozen=True)
class RetentionPolicyConfig:
    """Conservative mastery and reversal thresholds for opaque capabilities."""

    mastery_threshold: float = 0.8
    min_mastery_observations: int = 8
    reversal_threshold: float = 0.5
    reversal_patience: int = 4
    recent_window: int = 8

    def validate(self) -> RetentionPolicyConfig:
        if not 0.0 <= self.mastery_threshold <= 1.0:
            raise ValueError("mastery threshold must lie in [0, 1]")
        if self.min_mastery_observations < 1:
            raise ValueError("minimum mastery observations must be positive")
        if not 0.0 <= self.reversal_threshold <= 1.0:
            raise ValueError("reversal threshold must lie in [0, 1]")
        if self.reversal_patience < 1:
            raise ValueError("reversal patience must be positive")
        if self.recent_window < 1:
            raise ValueError("recent window must be positive")
        return self

    def as_dict(self) -> dict[str, float | int]:
        self.validate()
        return {
            "mastery_threshold": self.mastery_threshold,
            "min_mastery_observations": self.min_mastery_observations,
            "reversal_threshold": self.reversal_threshold,
            "reversal_patience": self.reversal_patience,
            "recent_window": self.recent_window,
        }


@dataclass(frozen=True)
class CapabilityRetentionStatus:
    """Observable state for one opaque capability address."""

    key_digest: str
    observations: int
    lifetime_observations: int
    mean_outcome: float
    stable_prefix_minimum: float
    recent_mean: float
    protected: bool
    reversal_streak: int
    reversal_count: int
    last_step: int


@dataclass(frozen=True)
class RetentionGateDecision:
    """Auditable result of a candidate promotion/retention check."""

    accepted: bool
    candidate_stable: bool
    retained: bool
    candidate_prefix_minimum: float
    retained_minimum: float | None
    reason: str


@dataclass(frozen=True)
class CapabilityRetentionProbe:
    """Fresh outcomes for one opaque candidate capability address."""

    key: torch.Tensor
    outcomes: Sequence[float] | torch.Tensor


@dataclass
class _CapabilityRecord:
    key: tuple[float, ...]
    era_success: float = 0.0
    era_observations: int = 0
    lifetime_observations: int = 0
    stable_prefix_minimum: float = 1.0
    recent: deque[float] | None = None
    protected: bool = False
    reversal_streak: int = 0
    reversal_count: int = 0
    last_step: int = 0


def _validate_key(key: torch.Tensor, width: int) -> torch.Tensor:
    if not isinstance(key, torch.Tensor):
        raise TypeError("capability key must be a tensor")
    if key.ndim != 1 or key.shape[0] != width:
        raise ValueError(f"capability key must have shape [{width}]")
    if not bool(torch.isfinite(key).all()):
        raise ValueError("capability key must contain only finite values")
    return key.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _key_digest(key: torch.Tensor, width: int) -> str:
    normalized = _validate_key(key, width)
    return hashlib.sha256(normalized.numpy().tobytes()).hexdigest()


def stable_prefix_minimum(
    outcomes: Sequence[float] | torch.Tensor,
    *,
    min_observations: int,
) -> float:
    """Return the lowest cumulative mean after the first measured prefix.

    The first qualifying threshold must remain satisfied at every later
    measured prefix.  This intentionally prevents a late spike from turning
    a capability with an earlier retention failure into a false mastery.
    """

    if min_observations < 1:
        raise ValueError("minimum observations must be positive")
    values = torch.as_tensor(outcomes, dtype=torch.float64).reshape(-1)
    if values.numel() < min_observations:
        return float("-inf")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("outcomes must be finite")
    if bool(torch.any((values < 0.0) | (values > 1.0))):
        raise ValueError("outcomes must lie in [0, 1]")
    cumulative = values.cumsum(0) / torch.arange(
        1, values.numel() + 1, dtype=values.dtype
    )
    return float(cumulative[min_observations - 1 :].min())


def evaluate_retention_gate(
    candidate_outcomes: Sequence[float] | torch.Tensor,
    retained_scores: Sequence[float] | torch.Tensor,
    *,
    candidate_threshold: float,
    retention_floor: float,
    min_candidate_observations: int,
) -> RetentionGateDecision:
    """Require stable new mastery and a full retained-capability floor.

    ``retained_scores`` are current verifier scores for already-promoted
    capabilities.  The gate is intentionally transactional: callers may
    adopt a new artifact or consolidation only when both conditions pass.
    """

    if not 0.0 <= candidate_threshold <= 1.0:
        raise ValueError("candidate threshold must lie in [0, 1]")
    if not 0.0 <= retention_floor <= 1.0:
        raise ValueError("retention floor must lie in [0, 1]")
    candidate_values = torch.as_tensor(candidate_outcomes, dtype=torch.float64).reshape(-1)
    retained_values = torch.as_tensor(retained_scores, dtype=torch.float64).reshape(-1)
    candidate_minimum = stable_prefix_minimum(
        candidate_values, min_observations=min_candidate_observations
    )
    candidate_stable = bool(
        candidate_values.numel() >= min_candidate_observations
        and candidate_minimum >= candidate_threshold
    )
    if retained_values.numel() == 0:
        retained_minimum = None
        retained = True
    else:
        if not bool(torch.isfinite(retained_values).all()):
            raise ValueError("retained scores must be finite")
        if bool(torch.any((retained_values < 0.0) | (retained_values > 1.0))):
            raise ValueError("retained scores must lie in [0, 1]")
        retained_minimum = float(retained_values.min())
        retained = retained_minimum >= retention_floor
    accepted = candidate_stable and retained
    if not candidate_stable:
        reason = "candidate mastery is not stable across measured prefixes"
    elif not retained:
        reason = "a retained capability fell below its retention floor"
    else:
        reason = "candidate mastery and retained-capability floor passed"
    return RetentionGateDecision(
        accepted=accepted,
        candidate_stable=candidate_stable,
        retained=retained,
        candidate_prefix_minimum=candidate_minimum,
        retained_minimum=retained_minimum,
        reason=reason,
    )


class CapabilityRetentionLedger:
    """Persistent opaque mastery state for a replaceable memory boundary.

    A protected row is never selected for eviction.  If every occupied row is
    protected, ``choose_eviction_index`` returns ``None``: the caller must
    grow the bank or perform a verified consolidation instead of silently
    forgetting a mastered capability.  Sustained low outcomes can trigger a
    reversal, but hysteresis prevents a single noisy failure from releasing a
    protected capability.
    """

    schema = RETENTION_LEDGER_SCHEMA

    def __init__(
        self,
        width: int,
        *,
        config: RetentionPolicyConfig | None = None,
    ) -> None:
        if width < 1:
            raise ValueError("retention ledger width must be positive")
        self.width = int(width)
        self.config = (config or RetentionPolicyConfig()).validate()
        self._records: dict[str, _CapabilityRecord] = {}
        self._step = 0

    @property
    def version(self) -> int:
        return self._step

    def _record(self, key: torch.Tensor) -> tuple[str, _CapabilityRecord]:
        normalized = _validate_key(key, self.width)
        digest = _key_digest(normalized, self.width)
        record = self._records.get(digest)
        if record is None:
            record = _CapabilityRecord(
                key=tuple(float(value) for value in normalized.tolist()),
                recent=deque(maxlen=self.config.recent_window),
            )
            self._records[digest] = record
        return digest, record

    def observe(self, key: torch.Tensor, outcome: float | torch.Tensor) -> CapabilityRetentionStatus:
        """Record one scalar verifier outcome without replaying old examples."""

        if isinstance(outcome, torch.Tensor):
            if outcome.numel() != 1:
                raise ValueError("retention outcome must be scalar")
            value = float(outcome.detach().cpu().item())
        else:
            value = float(outcome)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("retention outcome must be finite and lie in [0, 1]")
        _digest, record = self._record(key)
        self._step += 1
        record.lifetime_observations += 1
        record.era_observations += 1
        record.era_success += value
        if record.recent is None:
            record.recent = deque(maxlen=self.config.recent_window)
        record.recent.append(value)
        record.last_step = self._step
        if record.era_observations >= self.config.min_mastery_observations:
            era_mean = record.era_success / record.era_observations
            record.stable_prefix_minimum = min(
                record.stable_prefix_minimum, era_mean
            )

        if record.protected:
            if value <= self.config.reversal_threshold:
                record.reversal_streak += 1
            else:
                record.reversal_streak = 0
            if record.reversal_streak >= self.config.reversal_patience:
                record.protected = False
                record.reversal_count += 1
                record.reversal_streak = 0
                # Start a fresh era after the reversal evidence.  The
                # failures that established the reversal cannot be reused as
                # positive evidence for a future capability.
                record.era_success = 0.0
                record.era_observations = 0
                record.stable_prefix_minimum = 1.0
                record.recent.clear()
        elif (
            record.era_observations >= self.config.min_mastery_observations
            and record.stable_prefix_minimum >= self.config.mastery_threshold
        ):
            record.protected = True
            record.reversal_streak = 0
        return self.status(key)

    def status(self, key: torch.Tensor) -> CapabilityRetentionStatus:
        digest, record = self._record(key)
        recent = tuple(record.recent or ())
        mean = record.era_success / max(1, record.era_observations)
        return CapabilityRetentionStatus(
            key_digest=digest,
            observations=record.era_observations,
            lifetime_observations=record.lifetime_observations,
            mean_outcome=mean,
            stable_prefix_minimum=record.stable_prefix_minimum,
            recent_mean=sum(recent) / len(recent) if recent else 0.0,
            protected=record.protected,
            reversal_streak=record.reversal_streak,
            reversal_count=record.reversal_count,
            last_step=record.last_step,
        )

    def is_protected(self, key: torch.Tensor) -> bool:
        digest = _key_digest(key, self.width)
        record = self._records.get(digest)
        return bool(record is not None and record.protected)

    def contains(self, key: torch.Tensor) -> bool:
        """Return whether opaque evidence exists without creating a record."""

        normalized = _validate_key(key, self.width)
        return _key_digest(normalized, self.width) in self._records

    def mask_eviction_scores(
        self,
        keys: torch.Tensor,
        learned_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask protected rows from a scorer whose larger value means evict."""

        if keys.ndim != 2 or keys.shape[1] != self.width:
            raise ValueError(f"keys must have shape [rows, {self.width}]")
        scores = learned_scores.reshape(-1)
        if scores.shape[0] != keys.shape[0]:
            raise ValueError("learned scores must align with keys")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("learned eviction scores must be finite")
        protected = torch.tensor(
            [self.is_protected(key) for key in keys],
            dtype=torch.bool,
            device=scores.device,
        )
        masked = scores.masked_fill(protected, -torch.inf)
        return masked, protected

    def choose_eviction_index(
        self,
        keys: torch.Tensor,
        learned_scores: torch.Tensor,
    ) -> int | None:
        """Return an unprotected candidate, or ``None`` when growth is needed."""

        masked, protected = self.mask_eviction_scores(keys, learned_scores)
        if bool(protected.all()):
            return None
        return int(masked.argmax().item())

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "width": self.width,
            "policy": self.config.as_dict(),
        }

    def subset(self, keys: Sequence[torch.Tensor]) -> CapabilityRetentionLedger:
        """Copy retention state for a selected opaque row subset."""

        selected = CapabilityRetentionLedger(self.width, config=self.config)
        selected._step = self._step
        for key in keys:
            digest = _key_digest(key, self.width)
            record = self._records.get(digest)
            if record is not None:
                selected._records[digest] = deepcopy(record)
        return selected

    def adopt(
        self,
        source: CapabilityRetentionLedger,
        key: torch.Tensor,
    ) -> CapabilityRetentionStatus:
        """Adopt one externally accumulated opaque evidence record.

        This transfers verifier accounting across an external-memory boundary
        without replaying the observations that produced it.  It is used when
        a staged candidate has earned admission: the candidate's evidence is
        moved into the executable bank's ledger as state, not reconstructed by
        fabricating new outcomes.
        """

        if not isinstance(source, CapabilityRetentionLedger):
            raise TypeError("retention evidence source must be a capability ledger")
        if source.width != self.width:
            raise ValueError("retention evidence ledgers must have matching widths")
        if source.config.as_dict() != self.config.as_dict():
            raise ValueError("retention evidence ledgers must have matching policies")
        normalized = _validate_key(key, self.width)
        digest = _key_digest(normalized, self.width)
        record = source._records.get(digest)
        if record is None:
            raise KeyError("retention evidence source has no record for key")
        if digest in self._records:
            raise ValueError("retention evidence key already exists in destination")
        self._records[digest] = deepcopy(record)
        self._step = max(self._step, source._step) + 1
        self.validate()
        return self.status(normalized)

    def validate(self) -> None:
        """Validate persisted opaque state before it governs eviction."""

        self.config.validate()
        if self._step < 0:
            raise ValueError("retention ledger step cannot be negative")
        for digest, record in self._records.items():
            key = torch.tensor(record.key, dtype=torch.float32)
            if _key_digest(key, self.width) != digest:
                raise ValueError("retention ledger key digest mismatch")
            if min(
                record.era_observations,
                record.lifetime_observations,
                record.reversal_streak,
                record.reversal_count,
                record.last_step,
            ) < 0:
                raise ValueError("retention ledger counts cannot be negative")
            if not 0.0 <= record.era_success <= record.era_observations:
                raise ValueError("retention ledger outcome sum is invalid")
            if not 0.0 <= record.stable_prefix_minimum <= 1.0:
                raise ValueError("retention ledger prefix minimum is invalid")
            if record.recent is None or len(record.recent) > self.config.recent_window:
                raise ValueError("retention ledger recent window is invalid")
            if any(not 0.0 <= value <= 1.0 for value in record.recent):
                raise ValueError("retention ledger recent outcomes are invalid")

    def _payload(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for digest, record in sorted(self._records.items()):
            records.append(
                {
                    "key_digest": digest,
                    "key": list(record.key),
                    "era_success": record.era_success,
                    "era_observations": record.era_observations,
                    "lifetime_observations": record.lifetime_observations,
                    "stable_prefix_minimum": record.stable_prefix_minimum,
                    "recent": list(record.recent or ()),
                    "protected": record.protected,
                    "reversal_streak": record.reversal_streak,
                    "reversal_count": record.reversal_count,
                    "last_step": record.last_step,
                }
            )
        return {
            **self.configuration(),
            "step": self._step,
            "records": records,
        }

    def save(self, path: Path) -> None:
        """Persist only opaque keys and scalar retention state."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n")

    def payload(self) -> dict[str, Any]:
        """Return a validated serializable payload for runtime checkpoints."""

        self.validate()
        return self._payload()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CapabilityRetentionLedger:
        if payload.get("schema") != RETENTION_LEDGER_SCHEMA:
            raise ValueError("unsupported retention ledger schema")
        policy = payload.get("policy")
        if not isinstance(policy, dict):
            raise TypeError("retention ledger policy is invalid")
        ledger = cls(
            int(payload.get("width", 0)),
            config=RetentionPolicyConfig(**policy),
        )
        step = int(payload.get("step", 0))
        if step < 0:
            raise ValueError("retention ledger step cannot be negative")
        records = payload.get("records")
        if not isinstance(records, list):
            raise TypeError("retention ledger records are invalid")
        for item in records:
            if not isinstance(item, dict):
                raise TypeError("retention ledger record is invalid")
            key = torch.tensor(item.get("key", ()), dtype=torch.float32)
            digest = _key_digest(key, ledger.width)
            if digest != item.get("key_digest"):
                raise ValueError("retention ledger key digest mismatch")
            recent = item.get("recent", [])
            if not isinstance(recent, list):
                raise TypeError("retention ledger recent outcomes are invalid")
            record = _CapabilityRecord(
                key=tuple(float(value) for value in key.tolist()),
                era_success=float(item.get("era_success", 0.0)),
                era_observations=int(item.get("era_observations", 0)),
                lifetime_observations=int(item.get("lifetime_observations", 0)),
                stable_prefix_minimum=float(item.get("stable_prefix_minimum", 1.0)),
                recent=deque(
                    (float(value) for value in recent),
                    maxlen=ledger.config.recent_window,
                ),
                protected=bool(item.get("protected", False)),
                reversal_streak=int(item.get("reversal_streak", 0)),
                reversal_count=int(item.get("reversal_count", 0)),
                last_step=int(item.get("last_step", 0)),
            )
            if min(
                record.era_observations,
                record.lifetime_observations,
                record.reversal_streak,
                record.reversal_count,
                record.last_step,
            ) < 0:
                raise ValueError("retention ledger counts cannot be negative")
            ledger._records[digest] = record
        ledger._step = step
        ledger.validate()
        return ledger

    @classmethod
    def load(cls, path: Path) -> CapabilityRetentionLedger:
        return cls.from_payload(json.loads(Path(path).read_text()))
