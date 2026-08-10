"""Learned, memory-side binding for asynchronous transition streams.

The controller should not be handed a caller-owned stream label.  This module
keeps that transport concern outside the controller and outside the factual
transition models: a frozen context encoder proposes an opaque identity from
the evidence prefix, while a small external store maintains anonymous tracks,
bounded evidence, inter-arrival statistics, and verifier-calibrated trust.

The binding memory is deliberately non-authoritative.  Ambiguous evidence is
returned as ``ambiguous`` without mutating a track or the shared model bank.
The downstream factual router still verifies every transition before it can
match or promote a model slot.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .multistream_transition import (
    ExternalMultiStreamTransitionContextResult,
    ExternalMultiStreamTransitionContextRouter,
)
from .world_model import (
    ExternalTransitionContextEncoder,
    ExternalTransitionObservation,
)

EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA = (
    "neural-computer.external-stream-binding-memory.v2"
)
EXTERNAL_STREAM_BINDING_PROMOTION_SCHEMA = (
    "neural-computer.external-stream-binding-promotion.v1"
)
EXTERNAL_STREAM_BINDING_RETIREMENT_SCHEMA = (
    "neural-computer.external-stream-binding-retirement.v1"
)
EXTERNAL_STREAM_BINDING_REPLACEMENT_SCHEMA = (
    "neural-computer.external-stream-binding-replacement.v1"
)
EXTERNAL_STREAM_BINDING_FACTUAL_REPLACEMENT_SCHEMA = (
    "neural-computer.external-stream-binding-factual-replacement.v1"
)
EXTERNAL_STREAM_BINDING_LIFECYCLE_POLICY_SCHEMA = (
    "neural-computer.external-stream-binding-lifecycle-policy.v1"
)
EXTERNAL_STREAM_BINDING_LIFECYCLE_PROPOSAL_SCHEMA = (
    "neural-computer.external-stream-binding-lifecycle-proposal.v1"
)
EXTERNAL_LEARNED_MULTI_STREAM_ROUTER_SCHEMA = (
    "neural-computer.external-learned-multi-stream-router.v2"
)


def _digest_value(digest: hashlib._Hash, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            _digest_value(digest, str(key))
            _digest_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        digest.update(repr(len(value)).encode("utf-8"))
        for item in value:
            _digest_value(digest, item)
        return
    if value is None:
        digest.update(b"none")
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def _payload_digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        _digest_value(digest, value)
    return digest.hexdigest()


def _retained_factual_digest(
    bank: Any,
    *,
    excluded_slot_ids: set[int],
) -> str:
    """Digest retained factual slots without including the replacement slot."""

    digest = hashlib.sha256()
    for slot_id in bank.slot_ids:
        if slot_id in excluded_slot_ids:
            continue
        index = bank.physical_index_for_slot_id(slot_id)
        digest.update(str(slot_id).encode("utf-8"))
        _digest_value(digest, bank.context_at(index))
        digest.update(bank.model_family_at(index).encode("utf-8"))
        digest.update(bank.models[index].digest().encode("utf-8"))
    return digest.hexdigest()


def _normalize_key(key: torch.Tensor, *, width: int) -> torch.Tensor:
    if not isinstance(key, torch.Tensor):
        raise TypeError("stream identity key must be a tensor")
    if key.ndim != 1 or key.shape[0] != width:
        raise ValueError("stream identity key has the wrong shape")
    if not bool(torch.isfinite(key).all()):
        raise ValueError("stream identity key must be finite")
    if float(torch.linalg.vector_norm(key)) <= 1e-12:
        raise ValueError("stream identity key must be non-zero")
    return torch.nn.functional.normalize(
        key.detach().to(device="cpu", dtype=torch.float32), dim=0
    )


def _copy_valid_key(key: torch.Tensor, *, width: int) -> torch.Tensor:
    """Validate a persisted normalized key without changing its bytes."""

    if not isinstance(key, torch.Tensor):
        raise TypeError("persisted stream identity key must be a tensor")
    value = key.detach().to(device="cpu", dtype=torch.float32).clone()
    if value.ndim != 1 or value.shape[0] != width:
        raise ValueError("persisted stream identity key has the wrong shape")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("persisted stream identity key must be finite")
    if float(torch.linalg.vector_norm(value)) <= 1e-12:
        raise ValueError("persisted stream identity key must be non-zero")
    return value


def _timestamp_value(timestamp: torch.Tensor | float | None) -> float | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, torch.Tensor):
        values = timestamp.detach().reshape(-1).to(dtype=torch.float32)
        if values.numel() != 1:
            raise ValueError("stream-binding timestamps must contain one value")
        value = float(values[0])
    else:
        value = float(timestamp)
    if not math.isfinite(value):
        raise ValueError("stream-binding timestamps must be finite")
    return value


@dataclass(frozen=True)
class ExternalStreamBindingResult:
    """A non-authoritative anonymous-track proposal."""

    stream_key: torch.Tensor | None
    track_id: int | None
    status: str
    similarity: float | None
    margin: float | None
    reliability: float
    estimated_delay: float | None
    observation_count: int
    provisional_id: int | None = None
    schema: str = EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA

    def validate(
        self, *, stream_key_width: int
    ) -> ExternalStreamBindingResult:
        if self.schema != EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA:
            raise ValueError("unsupported stream-binding result schema")
        if self.status not in {"new", "matched", "ambiguous", "capacity", "provisional"}:
            raise ValueError("unsupported stream-binding result status")
        if self.stream_key is None:
            if self.status in {"new", "matched"}:
                raise ValueError("bound stream results require a stream key")
        else:
            _normalize_key(self.stream_key, width=stream_key_width)
        if self.status in {"ambiguous", "capacity", "provisional"} and self.stream_key is not None:
            raise ValueError("unresolved stream results cannot carry a live key")
        if self.track_id is not None and (
            not isinstance(self.track_id, int)
            or isinstance(self.track_id, bool)
            or self.track_id < 0
        ):
            raise ValueError("stream-binding track ID is invalid")
        if self.provisional_id is not None and (
            not isinstance(self.provisional_id, int)
            or isinstance(self.provisional_id, bool)
            or self.provisional_id < 0
        ):
            raise ValueError("stream-binding provisional ID is invalid")
        if self.status in {"new", "matched"} and self.provisional_id is not None:
            raise ValueError("live stream results cannot carry a provisional ID")
        if self.status in {"new", "matched"} and self.track_id is None:
            raise ValueError("live stream results require a track ID")
        if self.status in {"ambiguous", "capacity"} and (
            self.track_id is not None or self.provisional_id is not None
        ):
            raise ValueError("unresolved stream results cannot carry an ID")
        if self.status == "provisional" and self.provisional_id is None:
            raise ValueError("provisional results require a provisional ID")
        if self.status == "provisional" and self.track_id is not None:
            raise ValueError("provisional results cannot carry a live track ID")
        for name, value in (
            ("similarity", self.similarity),
            ("margin", self.margin),
            ("reliability", self.reliability),
            ("estimated_delay", self.estimated_delay),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"stream-binding {name} must be finite")
        if not 0.0 <= float(self.reliability) <= 1.0:
            raise ValueError("stream-binding reliability must lie in [0, 1]")
        if self.estimated_delay is not None and self.estimated_delay < 0:
            raise ValueError("stream-binding delay cannot be negative")
        if self.observation_count < 0:
            raise ValueError("stream-binding observation count cannot be negative")
        return self


@dataclass(frozen=True)
class ExternalStreamBindingPromotionReceipt:
    """Transactional admission/retirement result for an external track."""

    accepted: bool
    provisional_id: int
    track_id: int | None
    reason: str
    observation_count: int
    schema: str = EXTERNAL_STREAM_BINDING_PROMOTION_SCHEMA

    def validate(self) -> ExternalStreamBindingPromotionReceipt:
        if self.schema != EXTERNAL_STREAM_BINDING_PROMOTION_SCHEMA:
            raise ValueError("unsupported stream-binding promotion schema")
        if self.provisional_id < 0:
            raise ValueError("stream-binding provisional ID cannot be negative")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("stream-binding track ID cannot be negative")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("stream-binding promotion reason is missing")
        if self.observation_count < 0:
            raise ValueError("stream-binding observation count cannot be negative")
        if self.accepted and self.track_id is None:
            raise ValueError("accepted stream-binding promotion needs a track ID")
        return self


@dataclass(frozen=True)
class ExternalStreamBindingRetirementReceipt:
    """Transactional, retention-gated removal of one live binding track."""

    accepted: bool
    track_id: int
    reason: str
    schema: str = EXTERNAL_STREAM_BINDING_RETIREMENT_SCHEMA

    def validate(self) -> ExternalStreamBindingRetirementReceipt:
        if self.schema != EXTERNAL_STREAM_BINDING_RETIREMENT_SCHEMA:
            raise ValueError("unsupported stream-binding retirement schema")
        if self.track_id < 0:
            raise ValueError("stream-binding track ID cannot be negative")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("stream-binding retirement reason is missing")
        return self


@dataclass(frozen=True)
class ExternalStreamBindingReplacementReceipt:
    """Atomic replacement of one live track by one provisional track."""

    accepted: bool
    provisional_id: int
    retired_track_id: int
    track_id: int | None
    reason: str
    observation_count: int
    schema: str = EXTERNAL_STREAM_BINDING_REPLACEMENT_SCHEMA

    def validate(self) -> ExternalStreamBindingReplacementReceipt:
        if self.schema != EXTERNAL_STREAM_BINDING_REPLACEMENT_SCHEMA:
            raise ValueError("unsupported stream-binding replacement schema")
        for name, value in (
            ("provisional ID", self.provisional_id),
            ("retired track ID", self.retired_track_id),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"stream-binding {name} is invalid")
        if self.track_id is not None and (
            not isinstance(self.track_id, int)
            or isinstance(self.track_id, bool)
            or self.track_id < 0
        ):
            raise ValueError("stream-binding replacement track ID is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("stream-binding replacement reason is missing")
        if self.observation_count < 0:
            raise ValueError("stream-binding replacement observation count is invalid")
        if self.accepted and self.track_id is None:
            raise ValueError("accepted stream-binding replacement needs a track ID")
        return self


@dataclass(frozen=True)
class ExternalStreamBindingFactualReplacementReceipt:
    """Atomic binding replacement plus held-out factual-model promotion."""

    accepted: bool
    provisional_id: int
    retired_track_id: int
    track_id: int | None
    retired_slot_id: int | None
    slot_id: int | None
    heldout_error: float
    retention_outcome: float
    binding_digest_before: str
    binding_digest_after: str
    router_digest_before: str
    router_digest_after: str
    reason: str
    schema: str = EXTERNAL_STREAM_BINDING_FACTUAL_REPLACEMENT_SCHEMA

    def validate(self) -> ExternalStreamBindingFactualReplacementReceipt:
        if self.schema != EXTERNAL_STREAM_BINDING_FACTUAL_REPLACEMENT_SCHEMA:
            raise ValueError("unsupported stream-binding factual replacement schema")
        for name, value in (
            ("provisional ID", self.provisional_id),
            ("retired track ID", self.retired_track_id),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"stream-binding factual replacement {name} is invalid")
        for name, value in (
            ("track ID", self.track_id),
            ("retired slot ID", self.retired_slot_id),
            ("slot ID", self.slot_id),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"stream-binding factual replacement {name} is invalid")
        if (
            (not math.isfinite(self.heldout_error) and self.heldout_error != float("inf"))
            or self.heldout_error < 0.0
        ):
            raise ValueError("stream-binding factual replacement held-out error is invalid")
        if not math.isfinite(self.retention_outcome) or not 0.0 <= self.retention_outcome <= 1.0:
            raise ValueError("stream-binding factual replacement outcome is invalid")
        for name, value in (
            ("binding digest before", self.binding_digest_before),
            ("binding digest after", self.binding_digest_after),
            ("router digest before", self.router_digest_before),
            ("router digest after", self.router_digest_after),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"stream-binding factual replacement {name} is missing")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("stream-binding factual replacement reason is missing")
        if self.accepted and any(
            value is None
            for value in (self.track_id, self.retired_slot_id, self.slot_id)
        ):
            raise ValueError("accepted stream-binding factual replacement is incomplete")
        return self


@dataclass(frozen=True)
class ExternalStreamBindingLifecycleProposal:
    """A policy-only proposal over legal anonymous replacement candidates."""

    selected_provisional_id: int | None
    selected_track_id: int | None
    scores: torch.Tensor
    hold_score: float
    eligible_pairs: tuple[tuple[int, int], ...]
    features: torch.Tensor
    selected_probability: float
    selected_propensity: float
    selection_mode: str
    reason: str
    schema: str = EXTERNAL_STREAM_BINDING_LIFECYCLE_PROPOSAL_SCHEMA

    def validate(self, *, feature_width: int) -> ExternalStreamBindingLifecycleProposal:
        if self.schema != EXTERNAL_STREAM_BINDING_LIFECYCLE_PROPOSAL_SCHEMA:
            raise ValueError("unsupported stream-binding lifecycle proposal schema")
        if self.scores.ndim != 1 or self.scores.shape[0] != len(self.eligible_pairs):
            raise ValueError("stream-binding lifecycle scores are misaligned")
        if (
            self.features.ndim != 2
            or self.features.shape[0] != len(self.eligible_pairs)
            or self.features.shape[1] != feature_width
        ):
            raise ValueError("stream-binding lifecycle features are misaligned")
        if not bool(torch.isfinite(self.scores).all()) or not bool(
            torch.isfinite(self.features).all()
        ):
            raise ValueError("stream-binding lifecycle proposal is non-finite")
        if not math.isfinite(self.hold_score):
            raise ValueError("stream-binding lifecycle hold score is invalid")
        if len(set(self.eligible_pairs)) != len(self.eligible_pairs):
            raise ValueError("stream-binding lifecycle candidates are duplicated")
        if any(
            not isinstance(provisional_id, int)
            or isinstance(provisional_id, bool)
            or provisional_id < 0
            or not isinstance(track_id, int)
            or isinstance(track_id, bool)
            or track_id < 0
            for provisional_id, track_id in self.eligible_pairs
        ):
            raise ValueError("stream-binding lifecycle candidate ID is invalid")
        selected = (self.selected_provisional_id, self.selected_track_id)
        if (self.selected_provisional_id is None) != (
            self.selected_track_id is None
        ):
            raise ValueError("stream-binding lifecycle selection is incomplete")
        if self.selected_provisional_id is not None and selected not in self.eligible_pairs:
            raise ValueError("stream-binding lifecycle selection is ineligible")
        if not math.isfinite(self.selected_probability) or not 0.0 <= self.selected_probability <= 1.0:
            raise ValueError("stream-binding lifecycle probability is invalid")
        if not math.isfinite(self.selected_propensity) or not 0.0 < self.selected_propensity <= 1.0:
            raise ValueError("stream-binding lifecycle propensity is invalid")
        if self.selection_mode not in {"greedy", "sampled", "hold"}:
            raise ValueError("stream-binding lifecycle selection mode is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("stream-binding lifecycle proposal reason is missing")
        return self


@dataclass(frozen=True)
class ExternalLearnedMultiStreamTransitionResult:
    """Binding plus factual routing for one unlabelled transition arrival."""

    binding: ExternalStreamBindingResult
    routing: ExternalMultiStreamTransitionContextResult | None
    schema: str = EXTERNAL_LEARNED_MULTI_STREAM_ROUTER_SCHEMA

    def validate(
        self,
        *,
        stream_key_width: int,
        state_width: int,
        intention_width: int,
        context_width: int,
    ) -> ExternalLearnedMultiStreamTransitionResult:
        if self.schema != EXTERNAL_LEARNED_MULTI_STREAM_ROUTER_SCHEMA:
            raise ValueError("unsupported learned multi-stream result schema")
        self.binding.validate(stream_key_width=stream_key_width)
        if self.routing is not None:
            self.routing.validate(
                stream_key_width=stream_key_width,
                state_width=state_width,
                intention_width=intention_width,
                context_width=context_width,
            )
        return self


class ExternalStreamBindingLifecyclePolicy(torch.nn.Module):
    """Learn which anonymous replacement proposal is worth verifying.

    The policy sees only opaque prototype vectors and generic lifecycle
    telemetry. It proposes a legal pair; a scalar verifier outcome remains the
    authority that can commit the copy-on-write replacement. The policy is
    therefore replaceable external memory infrastructure, not a new reasoning
    branch in the controller.
    """

    schema = EXTERNAL_STREAM_BINDING_LIFECYCLE_POLICY_SCHEMA
    feature_width_extra = 11

    def __init__(
        self,
        context_width: int,
        *,
        hidden_width: int = 32,
        learning_rate: float = 1e-2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if context_width < 1 or hidden_width < 1:
            raise ValueError("stream-binding lifecycle policy widths must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("stream-binding lifecycle policy learning rate is invalid")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("stream-binding lifecycle policy temperature is invalid")
        self.context_width = int(context_width)
        self.hidden_width = int(hidden_width)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.feature_width = 2 * self.context_width + self.feature_width_extra
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.feature_width, self.hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_width, 1),
        )
        self.hold_bias = torch.nn.Parameter(torch.tensor(-1.0))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "hidden_width": self.hidden_width,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "inputs": "opaque_prototypes_reliability_delay_age_similarity_v1",
            "output": "replacement_candidate_acceptance_score_v1",
            "updates": "single_scalar_verifier_outcome_no_replay_v1",
        }

    @torch.no_grad()
    def propose(
        self,
        memory: ExternalOnlineStreamBindingMemory,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> ExternalStreamBindingLifecycleProposal:
        pairs, features = memory.lifecycle_candidate_features()
        if not pairs:
            return ExternalStreamBindingLifecycleProposal(
                selected_provisional_id=None,
                selected_track_id=None,
                scores=features.new_empty((0,)),
                hold_score=float(self.hold_bias),
                eligible_pairs=(),
                features=features,
                selected_probability=1.0,
                selected_propensity=1.0,
                selection_mode="hold",
                reason="no legal provisional replacement candidates",
            ).validate(feature_width=self.feature_width)
        scores = self.network(features).squeeze(-1)
        hold_score = self.hold_bias.to(scores)
        all_scores = torch.cat((scores, hold_score.reshape(1)))
        probabilities = torch.softmax(all_scores / self.temperature, dim=0)
        if sample:
            selected_index = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )[0]
            )
            selection_mode = "sampled"
            propensity = float(probabilities[selected_index])
        else:
            selected_index = int(torch.argmax(all_scores))
            selection_mode = "greedy"
            propensity = 1.0
        selected_provisional_id: int | None
        selected_track_id: int | None
        if selected_index == len(pairs):
            selected_provisional_id = None
            selected_track_id = None
            selection_mode = "hold"
        else:
            selected_provisional_id, selected_track_id = pairs[selected_index]
        return ExternalStreamBindingLifecycleProposal(
            selected_provisional_id=selected_provisional_id,
            selected_track_id=selected_track_id,
            scores=scores.detach().clone(),
            hold_score=float(hold_score),
            eligible_pairs=tuple(pairs),
            features=features.detach().clone(),
            selected_probability=float(probabilities[selected_index]),
            selected_propensity=propensity,
            selection_mode=selection_mode,
            reason="learned lifecycle score selected a replacement candidate",
        ).validate(feature_width=self.feature_width)

    def adaptation_step(
        self,
        proposal: ExternalStreamBindingLifecycleProposal,
        verifier_outcome: torch.Tensor | float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Consume one scalar outcome without retaining the source evidence."""

        proposal.validate(feature_width=self.feature_width)
        if isinstance(verifier_outcome, torch.Tensor):
            values = verifier_outcome.detach().reshape(-1)
            if values.numel() != 1:
                raise ValueError("stream-binding lifecycle outcomes must be scalar")
            outcome = float(values[0])
        else:
            outcome = float(verifier_outcome)
        if not math.isfinite(outcome) or not 0.0 <= outcome <= 1.0:
            raise ValueError("stream-binding lifecycle outcomes must lie in [0, 1]")
        if proposal.selected_provisional_id is None:
            logits = self.hold_bias.reshape(1)
            selected_index = None
        else:
            selected_index = proposal.eligible_pairs.index(
                (proposal.selected_provisional_id, proposal.selected_track_id)
            )
            logits = self.network(proposal.features)[selected_index : selected_index + 1, 0]
        target = torch.tensor(
            [outcome],
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
        )
        # An exploratory outcome is weighted by its exact logging propensity;
        # greedy deployment has propensity one by construction.
        loss = loss / max(proposal.selected_propensity, 1e-3)
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())

    def state_payload(self) -> dict[str, object]:
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }
        digest = hashlib.sha256()
        digest.update(repr(self.configuration()).encode("utf-8"))
        for name, value in state.items():
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.contiguous().numpy().tobytes())
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": digest.hexdigest(),
        }

    def digest(self) -> str:
        return str(self.state_payload()["sha256"])

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ExternalStreamBindingLifecyclePolicy:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported stream-binding lifecycle policy payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("stream-binding lifecycle policy payload is incomplete")
        policy = cls(
            int(configuration["context_width"]),
            hidden_width=int(configuration["hidden_width"]),
            learning_rate=float(configuration["learning_rate"]),
            temperature=float(configuration["temperature"]),
        )
        current = policy.state_dict()
        if tuple(state) != tuple(current):
            raise ValueError("stream-binding lifecycle policy state names differ")
        normalized: dict[str, torch.Tensor] = {}
        for name, expected in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("stream-binding lifecycle policy state is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("stream-binding lifecycle policy state is incompatible")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("stream-binding lifecycle policy state is not finite")
            normalized[name] = value.detach().clone()
        policy.load_state_dict(normalized, strict=True)
        if payload.get("sha256") != policy.digest():
            raise ValueError("stream-binding lifecycle policy checksum mismatch")
        return policy


class ExternalOnlineStreamBindingMemory:
    """Bind asynchronous transition evidence without caller-owned keys.

    The neural encoder is trained outside deployment from paired views and is
    frozen while this object grows.  Deployment updates only external state:
    each track has a stable opaque key, a bounded evidence window, a moving
    prefix prototype, inter-arrival estimates, and positive/negative verifier
    counts.  A new arrival is admitted only when its best candidate clears both
    a similarity threshold and a separation margin.  This makes uncertainty a
    first-class outcome instead of silently assigning contradictory evidence.
    """

    schema = EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA

    def __init__(
        self,
        encoder: ExternalTransitionContextEncoder,
        *,
        window_capacity: int = 4,
        max_streams: int = 32,
        provisional_capacity: int = 8,
        match_tolerance: float = 0.75,
        new_track_tolerance: float | None = None,
        provisional_tolerance: float | None = None,
        match_margin: float = 0.05,
        prototype_decay: float = 0.25,
        delay_decay: float = 0.25,
        reliability_prior: float = 1.0,
        reliability_warmup: int = 2,
    ) -> None:
        if not isinstance(encoder, ExternalTransitionContextEncoder):
            raise TypeError("stream binding requires a transition context encoder")
        if window_capacity < 1 or max_streams < 1 or provisional_capacity < 1:
            raise ValueError("stream binding capacities must be positive")
        if not 0.0 < match_tolerance <= 1.0:
            raise ValueError("stream binding match tolerance must be in (0, 1]")
        if new_track_tolerance is None:
            new_track_tolerance = match_tolerance
        if not 0.0 < new_track_tolerance <= 1.0:
            raise ValueError("stream binding new-track tolerance must be in (0, 1]")
        if provisional_tolerance is None:
            provisional_tolerance = match_tolerance
        if not 0.0 < provisional_tolerance <= 1.0:
            raise ValueError("stream binding provisional tolerance must be in (0, 1]")
        if match_margin < 0.0 or match_margin > 1.0:
            raise ValueError("stream binding match margin must lie in [0, 1]")
        if not 0.0 < prototype_decay <= 1.0:
            raise ValueError("stream binding prototype decay must be in (0, 1]")
        if not 0.0 < delay_decay <= 1.0:
            raise ValueError("stream binding delay decay must be in (0, 1]")
        if reliability_prior <= 0.0 or not math.isfinite(reliability_prior):
            raise ValueError("stream binding reliability prior must be positive")
        if reliability_warmup < 0:
            raise ValueError("stream binding reliability warmup cannot be negative")
        self.encoder = encoder
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.state_width = encoder.state_width
        self.intention_width = encoder.intention_width
        self.stream_key_width = encoder.context_width
        self.window_capacity = int(window_capacity)
        self.max_streams = int(max_streams)
        self.provisional_capacity = int(provisional_capacity)
        self.match_tolerance = float(match_tolerance)
        self.new_track_tolerance = float(new_track_tolerance)
        self.provisional_tolerance = float(provisional_tolerance)
        self.match_margin = float(match_margin)
        self.prototype_decay = float(prototype_decay)
        self.delay_decay = float(delay_decay)
        self.reliability_prior = float(reliability_prior)
        self.reliability_warmup = int(reliability_warmup)
        self._next_track_id = 0
        self._next_provisional_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}
        self._provisional: dict[int, dict[str, Any]] = {}

    @property
    def stream_count(self) -> int:
        return len(self._tracks)

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    @property
    def provisional_count(self) -> int:
        return len(self._provisional)

    @property
    def provisional_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._provisional))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "stream_key_width": self.stream_key_width,
            "window_capacity": self.window_capacity,
            "max_streams": self.max_streams,
            "provisional_capacity": self.provisional_capacity,
            "match_tolerance": self.match_tolerance,
            "new_track_tolerance": self.new_track_tolerance,
            "provisional_tolerance": self.provisional_tolerance,
            "match_margin": self.match_margin,
            "prototype_decay": self.prototype_decay,
            "delay_decay": self.delay_decay,
            "reliability_prior": self.reliability_prior,
            "reliability_warmup": self.reliability_warmup,
            "identity": "frozen_event_encoder_external_tracks_v1",
            "updates": "prototype_delay_reliability_sufficient_state_v1",
            "ambiguity": "no_mutation_on_unresolved_margin_v1",
        }

    def _validate_observation(
        self,
        observation: ExternalTransitionObservation,
        *,
        single_arrival: bool = True,
    ) -> ExternalTransitionObservation:
        observation.validate(
            state_width=self.state_width,
            intention_width=self.intention_width,
        )
        if single_arrival and observation.state.shape[0] != 1:
            raise ValueError("stream binding consumes one transition arrival at a time")
        return observation

    @staticmethod
    def _append(
        prior: ExternalTransitionObservation | None,
        current: ExternalTransitionObservation,
    ) -> ExternalTransitionObservation:
        if prior is None:
            return current
        confidence: torch.Tensor | None
        if prior.confidence is None and current.confidence is None:
            confidence = None
        elif prior.confidence is not None and current.confidence is not None:
            confidence = torch.cat((prior.confidence.reshape(-1), current.confidence.reshape(-1)))
        else:
            prior_confidence = (
                torch.ones(prior.state.shape[0], dtype=prior.state.dtype, device=prior.state.device)
                if prior.confidence is None
                else prior.confidence.reshape(-1)
            )
            current_confidence = (
                torch.ones(current.state.shape[0], dtype=current.state.dtype, device=current.state.device)
                if current.confidence is None
                else current.confidence.reshape(-1)
            )
            confidence = torch.cat((prior_confidence, current_confidence))
        return ExternalTransitionObservation(
            state=torch.cat((prior.state, current.state), dim=0),
            intention=torch.cat((prior.intention, current.intention), dim=0),
            next_state=torch.cat((prior.next_state, current.next_state), dim=0),
            confidence=confidence,
        )

    def _window_with(
        self,
        track: Mapping[str, Any],
        observation: ExternalTransitionObservation | None = None,
    ) -> ExternalTransitionObservation:
        prior = track.get("observations")
        if observation is not None:
            prior = self._append(prior, observation)
        if prior is None:
            raise ValueError("stream-binding track has no evidence")
        start = max(0, prior.state.shape[0] - self.window_capacity)
        confidence = (
            None
            if prior.confidence is None
            else prior.confidence.reshape(-1)[start:]
        )
        return ExternalTransitionObservation(
            state=prior.state[start:],
            intention=prior.intention[start:],
            next_state=prior.next_state[start:],
            confidence=confidence,
        )

    def _reliability(self, track: Mapping[str, Any]) -> float:
        positive = float(track["positive_count"])
        negative = float(track["negative_count"])
        return (positive + self.reliability_prior) / (
            positive + negative + 2.0 * self.reliability_prior
        )

    def _new_entry(
        self,
        observation: ExternalTransitionObservation,
        timestamp: float | None,
    ) -> dict[str, Any]:
        with torch.no_grad():
            key = self.encoder.encode_observation(observation).detach().cpu()
        return {
            "stream_key": _normalize_key(key, width=self.stream_key_width),
            "prototype": key.detach().cpu(),
            "observations": observation,
            "last_timestamp": timestamp,
            "mean_delay": None,
            "delay_count": 0,
            "positive_count": 0.0,
            "negative_count": 0.0,
        }

    def _stage_provisional(
        self,
        observation: ExternalTransitionObservation,
        timestamp: float | None,
        *,
        similarity: float | None,
        margin: float | None,
    ) -> ExternalStreamBindingResult:
        with torch.no_grad():
            candidate = self.encoder.encode_observation(observation).detach()
        ranked: list[tuple[float, int]] = []
        for provisional_id, entry in self._provisional.items():
            prototype = _normalize_key(
                entry["prototype"], width=self.stream_key_width
            ).to(candidate)
            ranked.append((float(torch.dot(candidate, prototype)), provisional_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if ranked and ranked[0][0] >= self.provisional_tolerance:
            provisional_id = ranked[0][1]
            entry = self._provisional[provisional_id]
            previous_timestamp = entry["last_timestamp"]
            if timestamp is not None and previous_timestamp is not None:
                delta = timestamp - float(previous_timestamp)
                if delta >= 0.0:
                    if entry["mean_delay"] is None:
                        entry["mean_delay"] = delta
                    else:
                        decay = self.delay_decay
                        entry["mean_delay"] = (
                            (1.0 - decay) * float(entry["mean_delay"])
                            + decay * delta
                        )
                    entry["delay_count"] += 1
            entry["last_timestamp"] = timestamp
            entry["observations"] = self._window_with(entry, observation)
            entry["prototype"] = torch.nn.functional.normalize(
                (1.0 - self.prototype_decay) * entry["prototype"]
                + self.prototype_decay * candidate.detach().cpu(),
                dim=0,
            )
            return ExternalStreamBindingResult(
                stream_key=None,
                track_id=None,
                provisional_id=provisional_id,
                status="provisional",
                similarity=ranked[0][0],
                margin=margin,
                reliability=self._reliability(entry),
                estimated_delay=entry["mean_delay"],
                observation_count=int(entry["observations"].state.shape[0]),
            ).validate(stream_key_width=self.stream_key_width)
        if self.provisional_count >= self.provisional_capacity:
            return ExternalStreamBindingResult(
                stream_key=None,
                track_id=None,
                provisional_id=None,
                status="capacity",
                similarity=similarity,
                margin=margin,
                reliability=0.5,
                estimated_delay=None,
                observation_count=0,
            ).validate(stream_key_width=self.stream_key_width)
        provisional_id = self._next_provisional_id
        self._next_provisional_id += 1
        self._provisional[provisional_id] = self._new_entry(observation, timestamp)
        return ExternalStreamBindingResult(
            stream_key=None,
            track_id=None,
            provisional_id=provisional_id,
            status="provisional",
            similarity=similarity,
            margin=margin,
            reliability=0.5,
            estimated_delay=None,
            observation_count=1,
        ).validate(stream_key_width=self.stream_key_width)

    def _temporal_score(
        self, track: Mapping[str, Any], timestamp: float | None
    ) -> float:
        if timestamp is None or track["last_timestamp"] is None:
            return 1.0
        delta = timestamp - float(track["last_timestamp"])
        if delta < 0.0:
            return 0.0
        expected = track["mean_delay"]
        if expected is None:
            return 1.0
        scale = max(float(expected), 1e-3)
        return math.exp(-abs(delta - float(expected)) / scale)

    def _rank(
        self,
        observation: ExternalTransitionObservation,
        timestamp: float | None,
    ) -> list[tuple[float, float, int]]:
        with torch.no_grad():
            scores: list[tuple[float, float, int]] = []
            for track_id, track in self._tracks.items():
                # Identity is matched from the current learned event.  The
                # bounded prefix remains external evidence and is used to
                # update the prototype after a verified assignment, but
                # making the match depend on a candidate-specific prefix
                # would let a wrong first assignment contaminate its own
                # score before ambiguity can be reported.
                candidate = self.encoder.encode_observation(observation).detach()
                prototype = _normalize_key(
                    track["prototype"], width=self.stream_key_width
                ).to(candidate)
                similarity = float(torch.dot(candidate, prototype))
                temporal = self._temporal_score(track, timestamp)
                reliability = self._reliability(track)
                if track["positive_count"] + track["negative_count"] >= self.reliability_warmup:
                    reliability_factor = 0.5 + 0.5 * reliability
                else:
                    reliability_factor = 1.0
                score = similarity * (0.75 + 0.25 * temporal) * reliability_factor
                scores.append((score, similarity, track_id))
        return sorted(scores, key=lambda item: (-item[0], item[2]))

    def observe(
        self,
        observation: ExternalTransitionObservation,
        *,
        timestamp: torch.Tensor | float | None = None,
    ) -> ExternalStreamBindingResult:
        observation = self._validate_observation(observation)
        current_timestamp = _timestamp_value(timestamp)
        ranked = self._rank(observation, current_timestamp)
        if not ranked:
            if self.stream_count >= self.max_streams:
                return self._stage_provisional(
                    observation,
                    current_timestamp,
                    similarity=None,
                    margin=None,
                )
            with torch.no_grad():
                key = self.encoder.encode_observation(observation).detach().cpu()
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = {
                "stream_key": _normalize_key(key, width=self.stream_key_width),
                "prototype": key.detach().cpu(),
                "observations": observation,
                "last_timestamp": current_timestamp,
                "mean_delay": None,
                "delay_count": 0,
                "positive_count": 0.0,
                "negative_count": 0.0,
            }
            return ExternalStreamBindingResult(
                stream_key=self._tracks[track_id]["stream_key"].clone(),
                track_id=track_id,
                status="new",
                similarity=1.0,
                margin=None,
                reliability=0.5,
                estimated_delay=None,
                observation_count=1,
            ).validate(stream_key_width=self.stream_key_width)

        best_score, best_similarity, best_id = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None
        margin = None if second_score is None else best_score - second_score
        threshold = (
            self.new_track_tolerance
            if self.stream_count < self.max_streams
            else self.match_tolerance
        )
        if best_score < threshold:
            if self.stream_count >= self.max_streams:
                return self._stage_provisional(
                    observation,
                    current_timestamp,
                    similarity=best_similarity,
                    margin=margin,
                )
            with torch.no_grad():
                key = self.encoder.encode_observation(observation).detach().cpu()
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = {
                "stream_key": _normalize_key(key, width=self.stream_key_width),
                "prototype": key.detach().cpu(),
                "observations": observation,
                "last_timestamp": current_timestamp,
                "mean_delay": None,
                "delay_count": 0,
                "positive_count": 0.0,
                "negative_count": 0.0,
            }
            return ExternalStreamBindingResult(
                stream_key=self._tracks[track_id]["stream_key"].clone(),
                track_id=track_id,
                status="new",
                similarity=best_similarity,
                margin=margin,
                reliability=0.5,
                estimated_delay=None,
                observation_count=1,
            ).validate(stream_key_width=self.stream_key_width)
        if margin is not None and margin < self.match_margin:
            return ExternalStreamBindingResult(
                stream_key=None,
                track_id=None,
                status="ambiguous",
                similarity=best_similarity,
                margin=margin,
                reliability=self._reliability(self._tracks[best_id]),
                estimated_delay=self._tracks[best_id]["mean_delay"],
                observation_count=0,
            ).validate(stream_key_width=self.stream_key_width)

        track = self._tracks[best_id]
        previous_timestamp = track["last_timestamp"]
        if current_timestamp is not None and previous_timestamp is not None:
            delta = current_timestamp - float(previous_timestamp)
            if delta >= 0.0:
                if track["mean_delay"] is None:
                    track["mean_delay"] = delta
                else:
                    decay = self.delay_decay
                    track["mean_delay"] = (1.0 - decay) * float(track["mean_delay"]) + decay * delta
                track["delay_count"] += 1
        track["last_timestamp"] = current_timestamp
        track["observations"] = self._window_with(track, observation)
        with torch.no_grad():
            prototype = self.encoder.encode_observation(observation)
        decay = self.prototype_decay
        track["prototype"] = torch.nn.functional.normalize(
            (1.0 - decay) * track["prototype"] + decay * prototype.detach().cpu(),
            dim=0,
        )
        return ExternalStreamBindingResult(
            stream_key=track["stream_key"].clone(),
            track_id=best_id,
            status="matched",
            similarity=best_similarity,
            margin=margin,
            reliability=self._reliability(track),
            estimated_delay=track["mean_delay"],
            observation_count=int(track["observations"].state.shape[0]),
        ).validate(stream_key_width=self.stream_key_width)

    def observe_verifier_outcome(
        self,
        result: ExternalStreamBindingResult,
        outcome: torch.Tensor | float,
    ) -> None:
        """Consume one scalar same-track verifier outcome without replay."""

        result.validate(stream_key_width=self.stream_key_width)
        value = float(outcome.detach().reshape(-1)[0]) if isinstance(outcome, torch.Tensor) else float(outcome)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("binding verifier outcomes must lie in [0, 1]")
        if result.track_id is not None:
            if result.track_id not in self._tracks:
                raise ValueError("verifier outcome refers to an unknown live track")
            track = self._tracks[result.track_id]
        elif result.provisional_id is not None:
            if result.provisional_id not in self._provisional:
                raise ValueError("verifier outcome refers to an unknown provisional")
            track = self._provisional[result.provisional_id]
        else:
            raise ValueError("verifier outcomes require a live or provisional track")
        track["positive_count"] += value
        track["negative_count"] += 1.0 - value

    def track_state(self, track_id: int) -> dict[str, object]:
        if track_id not in self._tracks:
            raise KeyError(track_id)
        track = self._tracks[track_id]
        return {
            "track_id": track_id,
            "stream_key": track["stream_key"].clone(),
            "prototype": track["prototype"].clone(),
            "observation_count": int(track["observations"].state.shape[0]),
            "last_timestamp": track["last_timestamp"],
            "mean_delay": track["mean_delay"],
            "delay_count": int(track["delay_count"]),
            "positive_count": float(track["positive_count"]),
            "negative_count": float(track["negative_count"]),
            "reliability": self._reliability(track),
        }

    def provisional_state(self, provisional_id: int) -> dict[str, object]:
        if provisional_id not in self._provisional:
            raise KeyError(provisional_id)
        entry = self._provisional[provisional_id]
        return {
            "provisional_id": provisional_id,
            "stream_key": entry["stream_key"].clone(),
            "prototype": entry["prototype"].clone(),
            "observation_count": int(entry["observations"].state.shape[0]),
            "last_timestamp": entry["last_timestamp"],
            "mean_delay": entry["mean_delay"],
            "delay_count": int(entry["delay_count"]),
            "positive_count": float(entry["positive_count"]),
            "negative_count": float(entry["negative_count"]),
            "reliability": self._reliability(entry),
        }

    def provisional_observation(self, provisional_id: int) -> ExternalTransitionObservation:
        """Return a detached bounded evidence window for external factual routing."""

        if provisional_id not in self._provisional:
            raise KeyError(provisional_id)
        observation = self._window_with(self._provisional[provisional_id])
        return ExternalTransitionObservation(
            state=observation.state.detach().cpu().clone(),
            intention=observation.intention.detach().cpu().clone(),
            next_state=observation.next_state.detach().cpu().clone(),
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence.detach().cpu().clone()
            ),
        )

    def lifecycle_candidate_features(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], torch.Tensor]:
        """Return opaque, permutation-stable features for legal replacements."""

        if self.stream_count < self.max_streams or not self._provisional:
            return (), torch.empty((0, 2 * self.stream_key_width + 11))
        pairs: list[tuple[int, int]] = []
        features: list[torch.Tensor] = []
        for provisional_id in sorted(self._provisional):
            provisional = self._provisional[provisional_id]
            provisional_delay = provisional["mean_delay"]
            provisional_observations = int(
                provisional["observations"].state.shape[0]
            )
            provisional_scalars = (
                math.log1p(provisional_observations),
                self._reliability(provisional),
                0.0
                if provisional_delay is None
                else math.log1p(max(float(provisional_delay), 0.0)),
                0.0 if provisional_delay is None else 1.0,
                math.log1p(float(provisional["delay_count"])),
            )
            for track_id in sorted(self._tracks):
                track = self._tracks[track_id]
                track_delay = track["mean_delay"]
                track_observations = int(track["observations"].state.shape[0])
                provisional_key = _normalize_key(
                    provisional["prototype"], width=self.stream_key_width
                )
                track_key = _normalize_key(
                    track["prototype"], width=self.stream_key_width
                )
                similarity = float(torch.dot(provisional_key, track_key))
                track_scalars = (
                    math.log1p(track_observations),
                    self._reliability(track),
                    0.0
                    if track_delay is None
                    else math.log1p(max(float(track_delay), 0.0)),
                    0.0 if track_delay is None else 1.0,
                    math.log1p(float(track["delay_count"])),
                )
                features.append(
                    torch.cat(
                        (
                            provisional_key,
                            track_key,
                            torch.tensor(
                                provisional_scalars
                                + track_scalars
                                + (similarity,),
                                dtype=torch.float32,
                            ),
                        )
                    )
                )
                pairs.append((provisional_id, track_id))
        return tuple(pairs), torch.stack(features)

    def replace_verified_track_with_provisional(
        self,
        provisional_id: int,
        track_id: int,
        retention_probe: Callable[[ExternalOnlineStreamBindingMemory], bool],
    ) -> ExternalStreamBindingReplacementReceipt:
        """Atomically replace one live track with one provisional track."""

        if provisional_id not in self._provisional:
            return ExternalStreamBindingReplacementReceipt(
                accepted=False,
                provisional_id=provisional_id,
                retired_track_id=track_id,
                track_id=None,
                reason="unknown_provisional",
                observation_count=0,
            ).validate()
        if track_id not in self._tracks:
            return ExternalStreamBindingReplacementReceipt(
                accepted=False,
                provisional_id=provisional_id,
                retired_track_id=track_id,
                track_id=None,
                reason="unknown_track",
                observation_count=0,
            ).validate()
        candidate = type(self).from_payload(self.state_payload())
        entry = candidate._provisional.pop(provisional_id)
        candidate._tracks.pop(track_id)
        new_track_id = candidate._next_track_id
        candidate._next_track_id += 1
        candidate._tracks[new_track_id] = entry
        observation_count = int(entry["observations"].state.shape[0])
        if not bool(retention_probe(candidate)):
            return ExternalStreamBindingReplacementReceipt(
                accepted=False,
                provisional_id=provisional_id,
                retired_track_id=track_id,
                track_id=None,
                reason="retention_probe_rejected",
                observation_count=observation_count,
            ).validate()
        self._tracks = candidate._tracks
        self._provisional = candidate._provisional
        self._next_track_id = candidate._next_track_id
        self._next_provisional_id = candidate._next_provisional_id
        return ExternalStreamBindingReplacementReceipt(
            accepted=True,
            provisional_id=provisional_id,
            retired_track_id=track_id,
            track_id=new_track_id,
            reason="verified_atomic_replacement",
            observation_count=observation_count,
        ).validate()

    def replace_on_verifier_outcome(
        self,
        provisional_id: int,
        track_id: int,
        verifier_outcome: torch.Tensor | float,
        *,
        acceptance_threshold: float = 1.0,
    ) -> ExternalStreamBindingReplacementReceipt:
        """Commit an evaluated proposal from one scalar verifier outcome.

        The caller supplies only the deterministic verifier result. Candidate
        selection and transaction construction remain inside the external
        memory boundary; no caller-owned structural retention probe is needed
        on this path.
        """

        if isinstance(verifier_outcome, torch.Tensor):
            values = verifier_outcome.detach().reshape(-1)
            if values.numel() != 1:
                raise ValueError("stream-binding replacement outcomes must be scalar")
            outcome = float(values[0])
        else:
            outcome = float(verifier_outcome)
        if not math.isfinite(outcome) or not 0.0 <= outcome <= 1.0:
            raise ValueError("stream-binding replacement outcomes must lie in [0, 1]")
        if (
            not math.isfinite(acceptance_threshold)
            or not 0.0 <= acceptance_threshold <= 1.0
        ):
            raise ValueError("stream-binding replacement acceptance threshold is invalid")
        if provisional_id not in self._provisional:
            return ExternalStreamBindingReplacementReceipt(
                accepted=False,
                provisional_id=provisional_id,
                retired_track_id=track_id,
                track_id=None,
                reason="unknown_provisional",
                observation_count=0,
            ).validate()
        if track_id not in self._tracks:
            return ExternalStreamBindingReplacementReceipt(
                accepted=False,
                provisional_id=provisional_id,
                retired_track_id=track_id,
                track_id=None,
                reason="unknown_track",
                observation_count=0,
            ).validate()
        candidate = type(self).from_payload(self.state_payload())
        entry = candidate._provisional.pop(provisional_id)
        candidate._tracks.pop(track_id)
        new_track_id = candidate._next_track_id
        candidate._next_track_id += 1
        candidate._tracks[new_track_id] = entry
        observation_count = int(entry["observations"].state.shape[0])
        if outcome < acceptance_threshold:
            return ExternalStreamBindingReplacementReceipt(
                accepted=False,
                provisional_id=provisional_id,
                retired_track_id=track_id,
                track_id=None,
                reason="verifier_outcome_rejected",
                observation_count=observation_count,
            ).validate()
        self._tracks = candidate._tracks
        self._provisional = candidate._provisional
        self._next_track_id = candidate._next_track_id
        self._next_provisional_id = candidate._next_provisional_id
        return ExternalStreamBindingReplacementReceipt(
            accepted=True,
            provisional_id=provisional_id,
            retired_track_id=track_id,
            track_id=new_track_id,
            reason="verifier_outcome_accepted",
            observation_count=observation_count,
        ).validate()

    def promote_provisional_track(
        self,
        provisional_id: int,
        retention_probe: Callable[[ExternalOnlineStreamBindingMemory], bool],
    ) -> ExternalStreamBindingPromotionReceipt:
        """Admit one quarantined stream only through a retention gate."""

        if provisional_id not in self._provisional:
            return ExternalStreamBindingPromotionReceipt(
                accepted=False,
                provisional_id=provisional_id,
                track_id=None,
                reason="unknown_provisional",
                observation_count=0,
            ).validate()
        if self.stream_count >= self.max_streams:
            return ExternalStreamBindingPromotionReceipt(
                accepted=False,
                provisional_id=provisional_id,
                track_id=None,
                reason="live_capacity_full",
                observation_count=int(
                    self._provisional[provisional_id]["observations"].state.shape[0]
                ),
            ).validate()
        candidate = type(self).from_payload(self.state_payload())
        entry = candidate._provisional.pop(provisional_id)
        track_id = candidate._next_track_id
        candidate._next_track_id += 1
        candidate._tracks[track_id] = entry
        observation_count = int(entry["observations"].state.shape[0])
        if not bool(retention_probe(candidate)):
            return ExternalStreamBindingPromotionReceipt(
                accepted=False,
                provisional_id=provisional_id,
                track_id=None,
                reason="retention_probe_rejected",
                observation_count=observation_count,
            ).validate()
        self._tracks = candidate._tracks
        self._provisional = candidate._provisional
        self._next_track_id = candidate._next_track_id
        self._next_provisional_id = candidate._next_provisional_id
        return ExternalStreamBindingPromotionReceipt(
            accepted=True,
            provisional_id=provisional_id,
            track_id=track_id,
            reason="verified_new_stream",
            observation_count=observation_count,
        ).validate()

    def retire_verified_track(
        self,
        track_id: int,
        retention_probe: Callable[[ExternalOnlineStreamBindingMemory], bool],
    ) -> ExternalStreamBindingRetirementReceipt:
        """Remove one live track only after a complete-retention probe."""

        if track_id not in self._tracks:
            return ExternalStreamBindingRetirementReceipt(
                accepted=False,
                track_id=track_id,
                reason="unknown_track",
            ).validate()
        candidate = type(self).from_payload(self.state_payload())
        candidate._tracks.pop(track_id)
        if not bool(retention_probe(candidate)):
            return ExternalStreamBindingRetirementReceipt(
                accepted=False,
                track_id=track_id,
                reason="retention_probe_rejected",
            ).validate()
        self._tracks = candidate._tracks
        self._provisional = candidate._provisional
        self._next_track_id = candidate._next_track_id
        self._next_provisional_id = candidate._next_provisional_id
        return ExternalStreamBindingRetirementReceipt(
            accepted=True,
            track_id=track_id,
            reason="verified_retirement",
        ).validate()

    @staticmethod
    def _entry_payload(
        identifier: int,
        entry: Mapping[str, Any],
        *,
        identifier_name: str,
    ) -> dict[str, object]:
        observation = entry["observations"]
        return {
            identifier_name: identifier,
            "stream_key": entry["stream_key"].clone(),
            "prototype": entry["prototype"].clone(),
            "observation": {
                "state": observation.state.detach().cpu().clone(),
                "intention": observation.intention.detach().cpu().clone(),
                "next_state": observation.next_state.detach().cpu().clone(),
                "confidence": None
                if observation.confidence is None
                else observation.confidence.detach().cpu().clone(),
            },
            "last_timestamp": entry["last_timestamp"],
            "mean_delay": entry["mean_delay"],
            "delay_count": entry["delay_count"],
            "positive_count": entry["positive_count"],
            "negative_count": entry["negative_count"],
        }

    def configuration_payload(self) -> dict[str, object]:
        return {
            "configuration": self.configuration(),
            "encoder": self.encoder.state_payload(),
        }

    def state_payload(self) -> dict[str, object]:
        tracks = [
            self._entry_payload(
                track_id,
                self._tracks[track_id],
                identifier_name="track_id",
            )
            for track_id in sorted(self._tracks)
        ]
        provisional = [
            self._entry_payload(
                provisional_id,
                self._provisional[provisional_id],
                identifier_name="provisional_id",
            )
            for provisional_id in sorted(self._provisional)
        ]
        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "encoder": self.encoder.state_payload(),
            "next_track_id": self._next_track_id,
            "next_provisional_id": self._next_provisional_id,
            "tracks": tracks,
            "provisional": provisional,
        }
        payload["sha256"] = _payload_digest(
            payload["schema"],
            payload["configuration"],
            payload["encoder"],
            payload["next_track_id"],
            payload["next_provisional_id"],
            payload["tracks"],
            payload["provisional"],
        )
        return payload

    def digest(self) -> str:
        return str(self.state_payload()["sha256"])

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExternalOnlineStreamBindingMemory:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported stream-binding memory payload")
        configuration = payload.get("configuration")
        encoder_payload = payload.get("encoder")
        tracks = payload.get("tracks")
        provisional = payload.get("provisional")
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(encoder_payload, Mapping)
            or not isinstance(tracks, list)
            or not isinstance(provisional, list)
        ):
            raise TypeError("stream-binding memory payload is incomplete")
        encoder = ExternalTransitionContextEncoder.from_payload(encoder_payload)
        expected = cls(
            encoder,
            window_capacity=int(configuration["window_capacity"]),
            max_streams=int(configuration["max_streams"]),
            provisional_capacity=int(configuration["provisional_capacity"]),
            match_tolerance=float(configuration["match_tolerance"]),
            new_track_tolerance=float(configuration["new_track_tolerance"]),
            provisional_tolerance=float(configuration["provisional_tolerance"]),
            match_margin=float(configuration["match_margin"]),
            prototype_decay=float(configuration["prototype_decay"]),
            delay_decay=float(configuration["delay_decay"]),
            reliability_prior=float(configuration["reliability_prior"]),
            reliability_warmup=int(configuration["reliability_warmup"]),
        )
        if dict(configuration) != expected.configuration():
            raise ValueError("stream-binding memory configuration mismatch")
        expected._next_track_id = int(payload.get("next_track_id", 0))
        if expected._next_track_id < 0:
            raise ValueError("stream-binding next track ID is invalid")

        def restore_entries(
            entries: list[object],
            *,
            identifier_name: str,
            destination: dict[int, dict[str, Any]],
        ) -> set[int]:
            seen: set[int] = set()
            for item in entries:
                if not isinstance(item, Mapping):
                    raise TypeError("stream-binding entry is invalid")
                identifier = item.get(identifier_name)
                if (
                    not isinstance(identifier, int)
                    or isinstance(identifier, bool)
                    or identifier < 0
                    or identifier in seen
                ):
                    raise ValueError("stream-binding entry ID is invalid or duplicated")
                observation_payload = item.get("observation")
                if not isinstance(observation_payload, Mapping):
                    raise TypeError("stream-binding observation is invalid")
                observation = ExternalTransitionObservation(
                    state=observation_payload["state"],
                    intention=observation_payload["intention"],
                    next_state=observation_payload["next_state"],
                    confidence=observation_payload.get("confidence"),
                )
                expected._validate_observation(observation, single_arrival=False)
                if observation.state.shape[0] > expected.window_capacity:
                    raise ValueError("stream-binding observation window exceeds capacity")
                stream_key = _copy_valid_key(
                    item["stream_key"], width=expected.stream_key_width
                )
                prototype = _copy_valid_key(
                    item["prototype"], width=expected.stream_key_width
                )
                last_timestamp = item.get("last_timestamp")
                mean_delay = item.get("mean_delay")
                for name, value in (
                    ("last_timestamp", last_timestamp),
                    ("mean_delay", mean_delay),
                ):
                    if value is not None and (
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(f"stream-binding {name} is invalid")
                if mean_delay is not None and float(mean_delay) < 0:
                    raise ValueError("stream-binding mean delay cannot be negative")
                delay_count = int(item.get("delay_count", 0))
                positive_count = float(item.get("positive_count", 0.0))
                negative_count = float(item.get("negative_count", 0.0))
                if delay_count < 0 or positive_count < 0 or negative_count < 0:
                    raise ValueError("stream-binding sufficient statistics are invalid")
                destination[identifier] = {
                    "stream_key": stream_key,
                    "prototype": prototype,
                    "observations": observation,
                    "last_timestamp": (
                        None if last_timestamp is None else float(last_timestamp)
                    ),
                    "mean_delay": None if mean_delay is None else float(mean_delay),
                    "delay_count": delay_count,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                }
                seen.add(identifier)
            return seen

        seen = restore_entries(
            tracks,
            identifier_name="track_id",
            destination=expected._tracks,
        )
        provisional_seen = restore_entries(
            provisional,
            identifier_name="provisional_id",
            destination=expected._provisional,
        )
        expected._next_provisional_id = int(payload.get("next_provisional_id", 0))
        if expected._next_provisional_id < 0:
            raise ValueError("stream-binding next provisional ID is invalid")
        if expected._next_provisional_id <= max(provisional_seen, default=-1):
            raise ValueError("stream-binding next provisional ID must exceed entries")
        if expected._next_track_id <= max(seen, default=-1):
            raise ValueError("stream-binding next track ID must exceed live tracks")
        expected_payload = expected.state_payload()
        if payload.get("sha256") != expected_payload["sha256"]:
            raise ValueError("stream-binding memory checksum mismatch")
        return expected


class ExternalLearnedMultiStreamTransitionContextRouter:
    """Run learned binding before the shared factual multi-stream router."""

    schema = EXTERNAL_LEARNED_MULTI_STREAM_ROUTER_SCHEMA

    def __init__(
        self,
        binding: ExternalOnlineStreamBindingMemory,
        router: ExternalMultiStreamTransitionContextRouter,
    ) -> None:
        if not isinstance(binding, ExternalOnlineStreamBindingMemory):
            raise TypeError("learned multi-stream router requires binding memory")
        if not isinstance(router, ExternalMultiStreamTransitionContextRouter):
            raise TypeError("learned multi-stream router requires multi-stream router")
        if binding.stream_key_width != router.stream_key_width:
            raise ValueError("binding and router stream-key widths differ")
        self.binding = binding
        self.router = router

    @property
    def bank(self):
        return self.router.bank

    @property
    def stream_count(self) -> int:
        return self.binding.stream_count

    @property
    def stream_keys(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            self.binding.track_state(track_id)["stream_key"]
            for track_id in self.binding.track_ids
        )

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "binding": self.binding.configuration(),
            "router": self.router.configuration(),
            "ownership": "binding_and_factual_router_external_to_frozen_controller_v1",
        }

    def observe(
        self,
        observation: ExternalTransitionObservation,
        *,
        timestamp: torch.Tensor | float | None = None,
    ) -> ExternalLearnedMultiStreamTransitionResult:
        binding = self.binding.observe(observation, timestamp=timestamp)
        routing = None
        if binding.stream_key is not None:
            routing = self.router.observe(observation, binding.stream_key)
        return ExternalLearnedMultiStreamTransitionResult(binding, routing).validate(
            stream_key_width=self.router.stream_key_width,
            state_width=self.router.bank.state_width,
            intention_width=self.router.bank.intention_width,
            context_width=self.router.bank.context_width,
        )

    def promote_provisional_track(
        self,
        result: ExternalLearnedMultiStreamTransitionResult,
        retention_probe: Callable[[ExternalOnlineStreamBindingMemory], bool],
    ) -> ExternalStreamBindingPromotionReceipt:
        """Admit a quarantined identity before factual routing consumes it."""

        result.validate(
            stream_key_width=self.router.stream_key_width,
            state_width=self.router.bank.state_width,
            intention_width=self.router.bank.intention_width,
            context_width=self.router.bank.context_width,
        )
        if result.binding.provisional_id is None:
            raise ValueError("result does not contain a provisional binding")
        return self.binding.promote_provisional_track(
            result.binding.provisional_id,
            retention_probe,
        )

    def retire_verified_track(
        self,
        track_id: int,
        retention_probe: Callable[[ExternalOnlineStreamBindingMemory], bool],
    ) -> ExternalStreamBindingRetirementReceipt:
        """Retire a binding only after its complete retention probe passes."""

        return self.binding.retire_verified_track(track_id, retention_probe)

    def apply_lifecycle_proposal(
        self,
        proposal: ExternalStreamBindingLifecycleProposal,
        verifier_outcome: torch.Tensor | float,
        *,
        acceptance_threshold: float = 1.0,
    ) -> ExternalStreamBindingReplacementReceipt | None:
        """Apply one policy-selected replacement from a scalar verifier result."""

        proposal.validate(feature_width=self.binding.stream_key_width * 2 + 11)
        if proposal.selected_provisional_id is None:
            return None
        if proposal.selected_track_id is None:
            raise ValueError("lifecycle proposal is missing its selected track")
        return self.binding.replace_on_verifier_outcome(
            proposal.selected_provisional_id,
            proposal.selected_track_id,
            verifier_outcome,
            acceptance_threshold=acceptance_threshold,
        )

    def replace_with_factual_candidate(
        self,
        proposal: ExternalStreamBindingLifecycleProposal,
        heldout_observation: ExternalTransitionObservation,
        verifier_outcome: torch.Tensor | float,
        *,
        prediction_tolerance: float = 0.05,
        acceptance_threshold: float = 1.0,
    ) -> ExternalStreamBindingFactualReplacementReceipt:
        """Jointly replace a binding and its factual model on isolated copies.

        The policy chooses only the anonymous binding pair. The external
        router replays no old controller state: it consumes the provisional
        evidence once, checks an independent held-out observation, and commits
        the binding plus factual-bank replacement together only when the
        scalar retention outcome authorizes the transaction.
        """

        proposal.validate(feature_width=self.binding.stream_key_width * 2 + 11)
        heldout_observation.validate(
            state_width=self.router.bank.state_width,
            intention_width=self.router.bank.intention_width,
        )
        if prediction_tolerance < 0.0 or not math.isfinite(prediction_tolerance):
            raise ValueError("joint factual replacement prediction tolerance is invalid")
        if isinstance(verifier_outcome, torch.Tensor):
            values = verifier_outcome.detach().reshape(-1)
            if values.numel() != 1:
                raise ValueError("joint factual replacement outcomes must be scalar")
            outcome = float(values[0])
        else:
            outcome = float(verifier_outcome)
        if not math.isfinite(outcome) or not 0.0 <= outcome <= 1.0:
            raise ValueError("joint factual replacement outcomes must lie in [0, 1]")
        if (
            not math.isfinite(acceptance_threshold)
            or not 0.0 <= acceptance_threshold <= 1.0
        ):
            raise ValueError("joint factual replacement acceptance threshold is invalid")
        binding_before = self.binding.digest()
        router_before = self.router.digest()
        selected_provisional_id = proposal.selected_provisional_id
        selected_track_id = proposal.selected_track_id
        if selected_provisional_id is None or selected_track_id is None:
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=0,
                retired_track_id=0,
                track_id=None,
                retired_slot_id=None,
                slot_id=None,
                heldout_error=float("inf"),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="policy_hold",
            ).validate()

        if outcome < acceptance_threshold:
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=None,
                slot_id=None,
                heldout_error=float("inf"),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="verifier_outcome_rejected",
            ).validate()

        retired_state = self.binding.track_state(selected_track_id)
        retired_slot_id = self.router.bound_slot_id(retired_state["stream_key"])
        if retired_slot_id is None:
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=None,
                slot_id=None,
                heldout_error=float("inf"),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="retired_track_has_no_factual_slot",
            ).validate()
        retained_factual_digest = _retained_factual_digest(
            self.router.bank,
            excluded_slot_ids={retired_slot_id},
        )

        provisional_observation = self.binding.provisional_observation(
            selected_provisional_id
        )
        candidate_binding = ExternalOnlineStreamBindingMemory.from_payload(
            self.binding.state_payload()
        )
        binding_receipt = candidate_binding.replace_on_verifier_outcome(
            selected_provisional_id,
            selected_track_id,
            outcome,
            acceptance_threshold=acceptance_threshold,
        )
        if not binding_receipt.accepted or binding_receipt.track_id is None:
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=retired_slot_id,
                slot_id=None,
                heldout_error=float("inf"),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="binding_replacement_failed",
            ).validate()

        new_key = candidate_binding.track_state(binding_receipt.track_id)["stream_key"]
        candidate_router = ExternalMultiStreamTransitionContextRouter.from_payload(
            self.router.state_payload()
        )
        eviction = candidate_router.evict_verified_id(
            retired_slot_id,
            lambda _candidate: True,
        )
        if not bool(getattr(eviction, "accepted", False)):
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=retired_slot_id,
                slot_id=None,
                heldout_error=float("inf"),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="factual_slot_eviction_failed",
            ).validate()

        for row_index in range(provisional_observation.state.shape[0]):
            confidence = (
                None
                if provisional_observation.confidence is None
                else provisional_observation.confidence[row_index : row_index + 1]
            )
            row = ExternalTransitionObservation(
                state=provisional_observation.state[row_index : row_index + 1],
                intention=provisional_observation.intention[row_index : row_index + 1],
                next_state=provisional_observation.next_state[row_index : row_index + 1],
                confidence=confidence,
            )
            routed = candidate_router.observe(row, new_key)
            if routed.result.status == "staged":
                candidate_router.adaptation_step(
                    routed,
                    None,
                    replay_evidence=False,
                )
        if candidate_router.provisional_candidate_count == 0:
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=retired_slot_id,
                slot_id=None,
                heldout_error=float("inf"),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="provisional_factual_candidate_not_staged",
            ).validate()
        factual_receipt = candidate_router.promote_staged_candidate(
            new_key,
            heldout_observation,
            lambda _candidate: True,
            prediction_tolerance=prediction_tolerance,
        )
        if not bool(getattr(factual_receipt, "accepted", False)):
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=retired_slot_id,
                slot_id=None,
                heldout_error=float(getattr(factual_receipt, "heldout_error", float("inf"))),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="heldout_factual_candidate_rejected",
            ).validate()

        replacement_slot_id = getattr(factual_receipt, "slot_id", None)
        if replacement_slot_id is None or _retained_factual_digest(
            candidate_router.bank,
            excluded_slot_ids={replacement_slot_id},
        ) != retained_factual_digest:
            return ExternalStreamBindingFactualReplacementReceipt(
                accepted=False,
                provisional_id=selected_provisional_id,
                retired_track_id=selected_track_id,
                track_id=None,
                retired_slot_id=retired_slot_id,
                slot_id=replacement_slot_id,
                heldout_error=float(getattr(factual_receipt, "heldout_error", float("inf"))),
                retention_outcome=outcome,
                binding_digest_before=binding_before,
                binding_digest_after=binding_before,
                router_digest_before=router_before,
                router_digest_after=router_before,
                reason="retained_factual_state_changed",
            ).validate()

        self.binding = candidate_binding
        self.router = candidate_router
        return ExternalStreamBindingFactualReplacementReceipt(
            accepted=True,
            provisional_id=selected_provisional_id,
            retired_track_id=selected_track_id,
            track_id=binding_receipt.track_id,
            retired_slot_id=retired_slot_id,
            slot_id=getattr(factual_receipt, "slot_id", None),
            heldout_error=float(getattr(factual_receipt, "heldout_error", float("inf"))),
            retention_outcome=outcome,
            binding_digest_before=binding_before,
            binding_digest_after=self.binding.digest(),
            router_digest_before=router_before,
            router_digest_after=self.router.digest(),
            reason="joint_binding_and_factual_replacement_committed",
        ).validate()

    def adaptation_step(
        self,
        result: ExternalLearnedMultiStreamTransitionResult,
        optimizer: torch.optim.Optimizer | Mapping[str, torch.optim.Optimizer] | None,
        *,
        replay_evidence: bool = True,
    ) -> float:
        result.validate(
            stream_key_width=self.router.stream_key_width,
            state_width=self.router.bank.state_width,
            intention_width=self.router.bank.intention_width,
            context_width=self.router.bank.context_width,
        )
        if result.routing is None:
            raise ValueError("ambiguous binding results cannot adapt a route")
        return self.router.adaptation_step(
            result.routing,
            optimizer,
            replay_evidence=replay_evidence,
        )

    def promote_staged_candidate(
        self,
        result: ExternalLearnedMultiStreamTransitionResult,
        heldout_observation: ExternalTransitionObservation,
        retention_probe: Any,
        **kwargs: Any,
    ) -> Any:
        if result.binding.stream_key is None:
            raise ValueError("only bound streams can promote a factual candidate")
        return self.router.promote_staged_candidate(
            result.binding.stream_key,
            heldout_observation,
            retention_probe,
            **kwargs,
        )

    def observe_binding_outcome(
        self,
        result: ExternalLearnedMultiStreamTransitionResult,
        outcome: torch.Tensor | float,
    ) -> None:
        result.validate(
            stream_key_width=self.router.stream_key_width,
            state_width=self.router.bank.state_width,
            intention_width=self.router.bank.intention_width,
            context_width=self.router.bank.context_width,
        )
        self.binding.observe_verifier_outcome(result.binding, outcome)

    def state_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "binding": self.binding.state_payload(),
            "router": self.router.state_payload(),
        }
        payload["sha256"] = _payload_digest(
            payload["schema"], payload["configuration"], payload["binding"], payload["router"]
        )
        return payload

    def digest(self) -> str:
        return str(self.state_payload()["sha256"])

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        evidence_evaluator: torch.nn.Module | None = None,
        prior_selection_probe: Any | None = None,
    ) -> ExternalLearnedMultiStreamTransitionContextRouter:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported learned multi-stream router payload")
        configuration = payload.get("configuration")
        binding_payload = payload.get("binding")
        router_payload = payload.get("router")
        if not isinstance(configuration, Mapping) or not isinstance(binding_payload, Mapping) or not isinstance(router_payload, Mapping):
            raise TypeError("learned multi-stream router payload is incomplete")
        binding = ExternalOnlineStreamBindingMemory.from_payload(binding_payload)
        router = ExternalMultiStreamTransitionContextRouter.from_payload(
            router_payload,
            evidence_evaluator=evidence_evaluator,
            prior_selection_probe=prior_selection_probe,
        )
        restored = cls(binding, router)
        if dict(configuration) != restored.configuration():
            raise ValueError("learned multi-stream router configuration mismatch")
        if payload.get("sha256") != restored.state_payload()["sha256"]:
            raise ValueError("learned multi-stream router checksum mismatch")
        return restored


__all__ = [
    "EXTERNAL_LEARNED_MULTI_STREAM_ROUTER_SCHEMA",
    "EXTERNAL_STREAM_BINDING_FACTUAL_REPLACEMENT_SCHEMA",
    "EXTERNAL_STREAM_BINDING_LIFECYCLE_POLICY_SCHEMA",
    "EXTERNAL_STREAM_BINDING_LIFECYCLE_PROPOSAL_SCHEMA",
    "EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA",
    "EXTERNAL_STREAM_BINDING_PROMOTION_SCHEMA",
    "EXTERNAL_STREAM_BINDING_REPLACEMENT_SCHEMA",
    "EXTERNAL_STREAM_BINDING_RETIREMENT_SCHEMA",
    "ExternalLearnedMultiStreamTransitionContextRouter",
    "ExternalLearnedMultiStreamTransitionResult",
    "ExternalOnlineStreamBindingMemory",
    "ExternalStreamBindingFactualReplacementReceipt",
    "ExternalStreamBindingLifecyclePolicy",
    "ExternalStreamBindingLifecycleProposal",
    "ExternalStreamBindingPromotionReceipt",
    "ExternalStreamBindingReplacementReceipt",
    "ExternalStreamBindingResult",
    "ExternalStreamBindingRetirementReceipt",
]
