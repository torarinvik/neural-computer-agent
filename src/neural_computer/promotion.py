"""Machine-checkable promotion evidence for capability experiments.

This module is deliberately separate from the controller.  It does not decide
what a capability means; it makes the evidence required for a promotion claim
explicit and reproducible.

The promotion population itself stays outside the repository.  Only its
content digest and a one-use holdout claim are recorded here.  A digest is
provenance, not secrecy: the process that owns the holdout must keep the
underlying examples and the ledger protected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROMOTION_SCHEMA = "neural-computer.promotion-evidence.v1"
HOLDOUT_LEDGER_SCHEMA = "neural-computer.holdout-ledger.v1"
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: Mapping[str, str | Path]) -> dict[str, str]:
    """Return stable, label-preserving hashes for a set of artifacts."""
    return {label: sha256_file(path) for label, path in sorted(paths.items())}


def _validate_digest(value: str, name: str) -> None:
    if not _HEX_DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class MetricRequirement:
    """A gate constraint applied independently to every replication."""

    name: str
    minimum: float | None = None
    maximum: float | None = None

    def validate(self) -> MetricRequirement:
        if not self.name:
            raise ValueError("metric requirement name cannot be empty")
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"metric {self.name!r} has no bound")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"metric {self.name!r} has inverted bounds")
        return self

    def matches(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        return self.maximum is None or value <= self.maximum


@dataclass(frozen=True)
class PromotionGate:
    """Versioned, experiment-specific criteria for a promotion claim."""

    experiment_id: str
    capability: str
    development_population: str
    promotion_population: str
    metric_requirements: tuple[MetricRequirement, ...]
    required_controls: tuple[str, ...]
    min_replicates: int = 3
    max_workarounds: int = 0
    schema: str = PROMOTION_SCHEMA

    def validate(self) -> PromotionGate:
        if self.schema != PROMOTION_SCHEMA:
            raise ValueError(f"unsupported promotion schema: {self.schema}")
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("capability", self.capability),
            ("development_population", self.development_population),
            ("promotion_population", self.promotion_population),
        ):
            if not value:
                raise ValueError(f"{name} cannot be empty")
        if self.development_population == self.promotion_population:
            raise ValueError("development and promotion populations must differ")
        if not self.metric_requirements:
            raise ValueError("a promotion gate needs at least one metric")
        if len({requirement.name for requirement in self.metric_requirements}) != len(
            self.metric_requirements
        ):
            raise ValueError("metric requirement names must be unique")
        for requirement in self.metric_requirements:
            requirement.validate()
        if len(set(self.required_controls)) != len(self.required_controls):
            raise ValueError("required control names must be unique")
        if any(not control for control in self.required_controls):
            raise ValueError("required control names cannot be empty")
        if self.min_replicates < 1:
            raise ValueError("min_replicates must be positive")
        if self.max_workarounds < 0:
            raise ValueError("max_workarounds cannot be negative")
        return self

    def canonical(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.canonical()))


@dataclass(frozen=True)
class PromotionEvidence:
    """The immutable evidence bundle evaluated against a :class:`PromotionGate`."""

    gate_digest: str
    holdout_id: str
    holdout_attempt_id: str
    development_manifest_digest: str
    promotion_manifest_digest: str
    git_commit: str
    configuration_digest: str
    artifact_hashes: Mapping[str, str]
    replicate_metrics: tuple[Mapping[str, float], ...]
    controls: Mapping[str, bool]
    search_attempts: int
    workaround_count: int
    holdout_uses: int = 1
    schema: str = PROMOTION_SCHEMA

    def validate(self) -> PromotionEvidence:
        if self.schema != PROMOTION_SCHEMA:
            raise ValueError(f"unsupported promotion schema: {self.schema}")
        if not self.holdout_id or not self.holdout_attempt_id:
            raise ValueError("holdout_id and holdout_attempt_id cannot be empty")
        for name, value in (
            ("gate_digest", self.gate_digest),
            ("development_manifest_digest", self.development_manifest_digest),
            ("promotion_manifest_digest", self.promotion_manifest_digest),
            ("configuration_digest", self.configuration_digest),
        ):
            _validate_digest(value, name)
        if self.development_manifest_digest == self.promotion_manifest_digest:
            raise ValueError("development and promotion manifests must differ")
        if not _GIT_COMMIT.fullmatch(self.git_commit):
            raise ValueError("git_commit must be a hexadecimal commit identifier")
        if not self.artifact_hashes:
            raise ValueError("at least one candidate artifact hash is required")
        for label, digest in self.artifact_hashes.items():
            if not label:
                raise ValueError("artifact labels cannot be empty")
            _validate_digest(digest, f"artifact hash for {label!r}")
        if not self.replicate_metrics:
            raise ValueError("replicate_metrics cannot be empty")
        for index, metrics in enumerate(self.replicate_metrics):
            if not metrics:
                raise ValueError(f"replicate {index} has no metrics")
            for name, value in metrics.items():
                if (
                    not name
                    or not isinstance(value, (float, int))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("replicate metrics must be named finite numbers")
        if any(not isinstance(value, bool) for value in self.controls.values()):
            raise ValueError("control results must be booleans")
        if self.search_attempts < 1:
            raise ValueError("search_attempts must be positive")
        if self.workaround_count < 0:
            raise ValueError("workaround_count cannot be negative")
        if self.holdout_uses != 1:
            raise ValueError("a promotion holdout must be consumed exactly once")
        return self

    def canonical(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]

    def canonical(self) -> dict[str, object]:
        return {"eligible": self.eligible, "reasons": list(self.reasons)}


def evaluate_promotion(
    gate: PromotionGate,
    evidence: PromotionEvidence,
    *,
    holdout_ledger: HoldoutLedger | None = None,
) -> PromotionDecision:
    """Evaluate evidence without silently filling missing fields.

    Missing controls and missing per-replication metrics are failures, not
    unknowns that can be inferred from a summary score.
    """
    reasons: list[str] = []
    try:
        gate.validate()
    except ValueError as error:
        reasons.append(f"invalid gate: {error}")
    try:
        evidence.validate()
    except ValueError as error:
        reasons.append(f"invalid evidence: {error}")

    if reasons:
        return PromotionDecision(False, tuple(reasons))
    if evidence.gate_digest != gate.digest():
        reasons.append("evidence was produced against a different gate")
    if holdout_ledger is None:
        reasons.append("promotion holdout claim was not verified against a ledger")
    elif not holdout_ledger.verify_claim(
        evidence.holdout_id,
        evidence.promotion_manifest_digest,
        evidence.holdout_attempt_id,
    ):
        reasons.append("promotion holdout claim is absent or does not match evidence")
    if len(evidence.replicate_metrics) < gate.min_replicates:
        reasons.append(
            f"only {len(evidence.replicate_metrics)} replications; "
            f"{gate.min_replicates} required"
        )
    for control in gate.required_controls:
        if control not in evidence.controls:
            reasons.append(f"required control is missing: {control}")
        elif not evidence.controls[control]:
            reasons.append(f"required control failed: {control}")
    for index, metrics in enumerate(evidence.replicate_metrics):
        for requirement in gate.metric_requirements:
            if requirement.name not in metrics:
                reasons.append(f"replicate {index} is missing metric: {requirement.name}")
            elif not requirement.matches(float(metrics[requirement.name])):
                reasons.append(
                    f"replicate {index} failed metric: {requirement.name}="
                    f"{metrics[requirement.name]}"
                )
    if evidence.workaround_count > gate.max_workarounds:
        reasons.append(
            f"workaround count {evidence.workaround_count} exceeds "
            f"maximum {gate.max_workarounds}"
        )
    return PromotionDecision(not reasons, tuple(reasons))


class PromotionRejected(RuntimeError):
    """Raised when code attempts to claim an ineligible promotion."""

    def __init__(self, decision: PromotionDecision) -> None:
        self.decision = decision
        super().__init__("promotion rejected: " + "; ".join(decision.reasons))


def require_promotion(
    gate: PromotionGate,
    evidence: PromotionEvidence,
    *,
    holdout_ledger: HoldoutLedger,
) -> PromotionDecision:
    """Return a passing decision or fail closed with all rejection reasons."""
    decision = evaluate_promotion(gate, evidence, holdout_ledger=holdout_ledger)
    if not decision.eligible:
        raise PromotionRejected(decision)
    return decision


def write_promotion_record(
    path: str | Path,
    gate: PromotionGate,
    evidence: PromotionEvidence,
    decision: PromotionDecision | None = None,
    *,
    holdout_ledger: HoldoutLedger | None = None,
) -> None:
    """Write a deterministic audit record for either a pass or a rejection."""
    selected = decision or evaluate_promotion(
        gate, evidence, holdout_ledger=holdout_ledger
    )
    record = {
        "schema": PROMOTION_SCHEMA,
        "gate": gate.canonical(),
        "gate_digest": gate.digest(),
        "evidence": evidence.canonical(),
        "decision": selected.canonical(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_json(record) + b"\n")


def read_promotion_record(
    path: str | Path,
) -> tuple[PromotionGate, PromotionEvidence, PromotionDecision]:
    """Load and type-check a serialized promotion record."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != PROMOTION_SCHEMA:
        raise ValueError("unsupported promotion record schema")
    gate_payload = dict(payload["gate"])
    gate_payload["metric_requirements"] = tuple(
        MetricRequirement(**requirement)
        for requirement in gate_payload["metric_requirements"]
    )
    gate_payload["required_controls"] = tuple(gate_payload["required_controls"])
    gate = PromotionGate(**gate_payload)
    evidence_payload = dict(payload["evidence"])
    evidence_payload["replicate_metrics"] = tuple(
        dict(metrics) for metrics in evidence_payload["replicate_metrics"]
    )
    evidence_payload["controls"] = dict(evidence_payload["controls"])
    evidence_payload["artifact_hashes"] = dict(evidence_payload["artifact_hashes"])
    evidence = PromotionEvidence(**evidence_payload)
    decision_payload = payload["decision"]
    decision = PromotionDecision(
        eligible=bool(decision_payload["eligible"]),
        reasons=tuple(decision_payload["reasons"]),
    )
    gate.validate()
    evidence.validate()
    if payload.get("gate_digest") != gate.digest():
        raise ValueError("promotion record gate digest does not match gate")
    return gate, evidence, decision


class HoldoutLedger:
    """Append-only local ledger that rejects reuse of a promotion holdout.

    The ledger stores identifiers and hashes only.  It is intentionally not a
    substitute for a protected holdout service; it prevents accidental reuse
    in ordinary local campaigns and makes reuse visible in review.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _claims(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        claims: list[dict[str, str]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema") != HOLDOUT_LEDGER_SCHEMA:
                raise ValueError("holdout ledger contains an unsupported record")
            claims.append(payload)
        return claims

    def claim(self, holdout_id: str, manifest_digest: str, attempt_id: str) -> None:
        if not holdout_id or not attempt_id:
            raise ValueError("holdout_id and attempt_id cannot be empty")
        _validate_digest(manifest_digest, "manifest_digest")
        if any(claim["holdout_id"] == holdout_id for claim in self._claims()):
            raise ValueError(f"holdout has already been consumed: {holdout_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": HOLDOUT_LEDGER_SCHEMA,
            "holdout_id": holdout_id,
            "manifest_digest": manifest_digest,
            "attempt_id": attempt_id,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    def verify_claim(
        self, holdout_id: str, manifest_digest: str, attempt_id: str
    ) -> bool:
        """Check the exact lease used to produce a promotion evidence bundle."""
        _validate_digest(manifest_digest, "manifest_digest")
        return any(
            claim == {
                "schema": HOLDOUT_LEDGER_SCHEMA,
                "holdout_id": holdout_id,
                "manifest_digest": manifest_digest,
                "attempt_id": attempt_id,
            }
            for claim in self._claims()
        )
