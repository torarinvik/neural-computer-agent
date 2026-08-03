"""Outcome-calibrated replay stopping for a disposable verifier probe.

The policy receives only paired scalar outcomes from a frozen parent and the
candidate currently being considered.  It does not receive correct actions,
task names, logits, or ablation results.  The verifier may keep protected
streams separate so that one regressing stream cannot be hidden by another.

This module is deliberately a scheduler primitive rather than a deployed
controller component.  Its diagnostic lifetimes are charged to the same
budget as replay lifetimes; a calibration probe therefore cannot be free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from statistics import NormalDist
from typing import Iterable, Mapping


def _values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one outcome")
    if any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"{name} must contain scalar outcomes in [0, 1]")
    return result


@dataclass(frozen=True)
class OutcomeEstimate:
    """A conservative paired child-minus-parent scalar-outcome estimate."""

    count: int
    mean_delta: float
    standard_error: float
    lower: float
    upper: float


def estimate_outcome_delta(
        parent_outcomes: Iterable[float], child_outcomes: Iterable[float], *,
        confidence: float = 0.95) -> OutcomeEstimate:
    """Estimate paired outcome change without exposing action labels.

    Inputs should be one scalar per logical lifetime.  Pairing keeps the
    estimate tied to the same rerendered experience while the controller sees
    only the resulting scalar outcomes.
    """
    parent = _values(parent_outcomes, "parent_outcomes")
    child = _values(child_outcomes, "child_outcomes")
    if len(parent) != len(child):
        raise ValueError("parent and child outcome counts must match")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    deltas = tuple(child_value - parent_value
                   for parent_value, child_value in zip(parent, child))
    count = len(deltas)
    mean = sum(deltas) / count
    if count == 1:
        standard_error = 0.0
    else:
        variance = sum((delta - mean) ** 2 for delta in deltas) / (count - 1)
        standard_error = sqrt(variance / count)
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    margin = z * standard_error
    return OutcomeEstimate(
        count=count, mean_delta=mean, standard_error=standard_error,
        lower=mean - margin, upper=mean + margin)


@dataclass
class ReplayBudget:
    """Unique logical lifetimes spent by replay and diagnostics."""

    maximum_lifetimes: int
    replay_lifetimes: int = 0
    diagnostic_lifetimes: int = 0

    def __post_init__(self) -> None:
        if self.maximum_lifetimes < 1:
            raise ValueError("maximum_lifetimes must be positive")
        if self.replay_lifetimes < 0 or self.diagnostic_lifetimes < 0:
            raise ValueError("lifetime counts must be non-negative")
        if self.total_lifetimes > self.maximum_lifetimes:
            raise ValueError("initial lifetime counts exceed maximum budget")

    @property
    def total_lifetimes(self) -> int:
        return self.replay_lifetimes + self.diagnostic_lifetimes

    @property
    def remaining_lifetimes(self) -> int:
        return self.maximum_lifetimes - self.total_lifetimes

    def consume_replay(self, count: int) -> None:
        self._consume(count, "replay_lifetimes")
        self.replay_lifetimes += count

    def consume_diagnostics(self, count: int) -> None:
        self._consume(count, "diagnostic_lifetimes")
        self.diagnostic_lifetimes += count

    def _consume(self, count: int, name: str) -> None:
        if count < 0:
            raise ValueError(f"{name} cannot be negative")
        if self.total_lifetimes + count > self.maximum_lifetimes:
            raise ValueError(
                f"{name} would exceed the replay budget: "
                f"requested={count}, remaining={self.remaining_lifetimes}")


@dataclass(frozen=True)
class ReplayDecision:
    action: str
    reason: str
    acquisition_ready: bool
    retention_ready: bool
    budget_exhausted: bool
    acquisition: OutcomeEstimate | None
    protected_streams: Mapping[str, OutcomeEstimate]


@dataclass
class OutcomeCalibratedReplayController:
    """Stop only after outcome evidence supports acquisition and retention."""

    maximum_lifetimes: int
    retention_tolerance: float = 0.02
    minimum_acquisition_gain: float = 0.0
    confidence: float = 0.95
    minimum_diagnostic_lifetimes: int = 64
    budget: ReplayBudget = field(init=False)
    _acquisition_parent: list[float] = field(default_factory=list, init=False)
    _acquisition_child: list[float] = field(default_factory=list, init=False)
    _protected: dict[str, tuple[list[float], list[float]]] = field(
        default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.retention_tolerance < 0.0:
            raise ValueError("retention_tolerance must be non-negative")
        if self.minimum_acquisition_gain < 0.0:
            raise ValueError("minimum_acquisition_gain must be non-negative")
        if self.minimum_diagnostic_lifetimes < 1:
            raise ValueError("minimum_diagnostic_lifetimes must be positive")
        # Validate confidence through the shared estimator before any data is
        # consumed, making configuration errors deterministic.
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1)")
        self.budget = ReplayBudget(self.maximum_lifetimes)

    def consume_replay(self, count: int) -> None:
        self.budget.consume_replay(count)

    def observe_acquisition(
            self, parent_outcomes: Iterable[float],
            child_outcomes: Iterable[float]) -> None:
        parent = list(_values(parent_outcomes, "parent_outcomes"))
        child = list(_values(child_outcomes, "child_outcomes"))
        self._check_pair(parent, child)
        self.budget.consume_diagnostics(len(parent))
        self._acquisition_parent.extend(parent)
        self._acquisition_child.extend(child)

    def observe_protected(
            self, stream: str, parent_outcomes: Iterable[float],
            child_outcomes: Iterable[float]) -> None:
        if not stream:
            raise ValueError("protected stream name must not be empty")
        parent = list(_values(parent_outcomes, "parent_outcomes"))
        child = list(_values(child_outcomes, "child_outcomes"))
        self._check_pair(parent, child)
        self.budget.consume_diagnostics(len(parent))
        previous = self._protected.setdefault(stream, ([], []))
        previous[0].extend(parent)
        previous[1].extend(child)

    @staticmethod
    def _check_pair(parent: list[float], child: list[float]) -> None:
        if len(parent) != len(child):
            raise ValueError("parent and child outcome counts must match")

    def _acquisition_estimate(self) -> OutcomeEstimate | None:
        if not self._acquisition_parent:
            return None
        return estimate_outcome_delta(
            self._acquisition_parent, self._acquisition_child,
            confidence=self.confidence)

    def _protected_estimates(self) -> dict[str, OutcomeEstimate]:
        return {
            stream: estimate_outcome_delta(parent, child,
                                           confidence=self.confidence)
            for stream, (parent, child) in self._protected.items()}

    def decide(self) -> ReplayDecision:
        acquisition = self._acquisition_estimate()
        protected = self._protected_estimates()
        enough_data = (
            self.budget.diagnostic_lifetimes
            >= self.minimum_diagnostic_lifetimes)
        acquisition_ready = bool(
            acquisition is not None
            and acquisition.count >= self.minimum_diagnostic_lifetimes
            and acquisition.lower >= self.minimum_acquisition_gain)
        retention_ready = bool(
            protected
            and all(
                estimate.count >= self.minimum_diagnostic_lifetimes
                and estimate.lower >= -self.retention_tolerance
                for estimate in protected.values()))
        if enough_data and acquisition_ready and retention_ready:
            action = "stop"
            reason = "scalar outcome acquisition and retention gates passed"
        elif self.budget.remaining_lifetimes == 0:
            action = "budget_exhausted"
            reason = "budget exhausted before both scalar outcome gates passed"
        else:
            action = "continue"
            reason = "scalar outcome evidence is insufficient for a safe stop"
        return ReplayDecision(
            action=action, reason=reason,
            acquisition_ready=acquisition_ready,
            retention_ready=retention_ready,
            budget_exhausted=self.budget.remaining_lifetimes == 0,
            acquisition=acquisition, protected_streams=protected)

    def report(self) -> dict[str, object]:
        decision = self.decide()
        return {
            "schema": "outcome-calibrated-replay-controller-v1",
            "configuration": {
                "maximum_lifetimes": self.maximum_lifetimes,
                "retention_tolerance": self.retention_tolerance,
                "minimum_acquisition_gain": self.minimum_acquisition_gain,
                "confidence": self.confidence,
                "minimum_diagnostic_lifetimes": (
                    self.minimum_diagnostic_lifetimes),
            },
            "budget": asdict(self.budget),
            "decision": asdict(decision),
            "claim_boundary": (
                "This is a verifier-side retrospective calibration probe. "
                "The policy consumes paired scalar outcomes only; correct "
                "actions, task identities, logits, and causal ablations "
                "remain outside the policy."),
        }
