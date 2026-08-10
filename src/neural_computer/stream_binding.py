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
from collections.abc import Mapping
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
    "neural-computer.external-stream-binding-memory.v1"
)
EXTERNAL_LEARNED_MULTI_STREAM_ROUTER_SCHEMA = (
    "neural-computer.external-learned-multi-stream-router.v1"
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
    schema: str = EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA

    def validate(
        self, *, stream_key_width: int
    ) -> ExternalStreamBindingResult:
        if self.schema != EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA:
            raise ValueError("unsupported stream-binding result schema")
        if self.status not in {"new", "matched", "ambiguous", "capacity"}:
            raise ValueError("unsupported stream-binding result status")
        if self.stream_key is None:
            if self.status in {"new", "matched"}:
                raise ValueError("bound stream results require a stream key")
        else:
            _normalize_key(self.stream_key, width=stream_key_width)
        if self.track_id is not None and (
            not isinstance(self.track_id, int)
            or isinstance(self.track_id, bool)
            or self.track_id < 0
        ):
            raise ValueError("stream-binding track ID is invalid")
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
        match_tolerance: float = 0.75,
        new_track_tolerance: float | None = None,
        match_margin: float = 0.05,
        prototype_decay: float = 0.25,
        delay_decay: float = 0.25,
        reliability_prior: float = 1.0,
        reliability_warmup: int = 2,
    ) -> None:
        if not isinstance(encoder, ExternalTransitionContextEncoder):
            raise TypeError("stream binding requires a transition context encoder")
        if window_capacity < 1 or max_streams < 1:
            raise ValueError("stream binding capacities must be positive")
        if not 0.0 < match_tolerance <= 1.0:
            raise ValueError("stream binding match tolerance must be in (0, 1]")
        if new_track_tolerance is None:
            new_track_tolerance = match_tolerance
        if not 0.0 < new_track_tolerance <= 1.0:
            raise ValueError("stream binding new-track tolerance must be in (0, 1]")
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
        self.match_tolerance = float(match_tolerance)
        self.new_track_tolerance = float(new_track_tolerance)
        self.match_margin = float(match_margin)
        self.prototype_decay = float(prototype_decay)
        self.delay_decay = float(delay_decay)
        self.reliability_prior = float(reliability_prior)
        self.reliability_warmup = int(reliability_warmup)
        self._next_track_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    @property
    def stream_count(self) -> int:
        return len(self._tracks)

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "state_width": self.state_width,
            "intention_width": self.intention_width,
            "stream_key_width": self.stream_key_width,
            "window_capacity": self.window_capacity,
            "max_streams": self.max_streams,
            "match_tolerance": self.match_tolerance,
            "new_track_tolerance": self.new_track_tolerance,
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
                return ExternalStreamBindingResult(
                    stream_key=None,
                    track_id=None,
                    status="capacity",
                    similarity=None,
                    margin=None,
                    reliability=0.5,
                    estimated_delay=None,
                    observation_count=0,
                ).validate(stream_key_width=self.stream_key_width)
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
                return ExternalStreamBindingResult(
                    stream_key=None,
                    track_id=None,
                    status="capacity",
                    similarity=best_similarity,
                    margin=margin,
                    reliability=self._reliability(self._tracks[best_id]),
                    estimated_delay=self._tracks[best_id]["mean_delay"],
                    observation_count=0,
                ).validate(stream_key_width=self.stream_key_width)
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
        if result.track_id is None or result.track_id not in self._tracks:
            raise ValueError("verifier outcomes require a live binding track")
        value = float(outcome.detach().reshape(-1)[0]) if isinstance(outcome, torch.Tensor) else float(outcome)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("binding verifier outcomes must lie in [0, 1]")
        track = self._tracks[result.track_id]
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

    def configuration_payload(self) -> dict[str, object]:
        return {
            "configuration": self.configuration(),
            "encoder": self.encoder.state_payload(),
        }

    def state_payload(self) -> dict[str, object]:
        tracks: list[dict[str, object]] = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            observation = track["observations"]
            tracks.append(
                {
                    "track_id": track_id,
                    "stream_key": track["stream_key"].clone(),
                    "prototype": track["prototype"].clone(),
                    "observation": {
                        "state": observation.state.detach().cpu().clone(),
                        "intention": observation.intention.detach().cpu().clone(),
                        "next_state": observation.next_state.detach().cpu().clone(),
                        "confidence": None
                        if observation.confidence is None
                        else observation.confidence.detach().cpu().clone(),
                    },
                    "last_timestamp": track["last_timestamp"],
                    "mean_delay": track["mean_delay"],
                    "delay_count": track["delay_count"],
                    "positive_count": track["positive_count"],
                    "negative_count": track["negative_count"],
                }
            )
        payload: dict[str, object] = {
            "schema": self.schema,
            "configuration": self.configuration(),
            "encoder": self.encoder.state_payload(),
            "next_track_id": self._next_track_id,
            "tracks": tracks,
        }
        payload["sha256"] = _payload_digest(
            payload["schema"], payload["configuration"], payload["encoder"], payload["next_track_id"], payload["tracks"]
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
        if not isinstance(configuration, Mapping) or not isinstance(encoder_payload, Mapping) or not isinstance(tracks, list):
            raise TypeError("stream-binding memory payload is incomplete")
        encoder = ExternalTransitionContextEncoder.from_payload(encoder_payload)
        expected = cls(
            encoder,
            window_capacity=int(configuration["window_capacity"]),
            max_streams=int(configuration["max_streams"]),
            match_tolerance=float(configuration["match_tolerance"]),
            new_track_tolerance=float(configuration["new_track_tolerance"]),
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
        seen: set[int] = set()
        for item in tracks:
            if not isinstance(item, Mapping):
                raise TypeError("stream-binding track is invalid")
            track_id = item.get("track_id")
            if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0 or track_id in seen:
                raise ValueError("stream-binding track ID is invalid or duplicated")
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
            stream_key = _copy_valid_key(item["stream_key"], width=expected.stream_key_width)
            prototype = _copy_valid_key(item["prototype"], width=expected.stream_key_width)
            last_timestamp = item.get("last_timestamp")
            mean_delay = item.get("mean_delay")
            for name, value in (("last_timestamp", last_timestamp), ("mean_delay", mean_delay)):
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                    raise ValueError(f"stream-binding {name} is invalid")
            if mean_delay is not None and float(mean_delay) < 0:
                raise ValueError("stream-binding mean delay cannot be negative")
            delay_count = int(item.get("delay_count", 0))
            positive_count = float(item.get("positive_count", 0.0))
            negative_count = float(item.get("negative_count", 0.0))
            if delay_count < 0 or positive_count < 0 or negative_count < 0:
                raise ValueError("stream-binding sufficient statistics are invalid")
            expected._tracks[track_id] = {
                "stream_key": stream_key,
                "prototype": prototype,
                "observations": observation,
                "last_timestamp": None if last_timestamp is None else float(last_timestamp),
                "mean_delay": None if mean_delay is None else float(mean_delay),
                "delay_count": delay_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
            }
            seen.add(track_id)
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
    "EXTERNAL_STREAM_BINDING_MEMORY_SCHEMA",
    "ExternalLearnedMultiStreamTransitionContextRouter",
    "ExternalLearnedMultiStreamTransitionResult",
    "ExternalOnlineStreamBindingMemory",
    "ExternalStreamBindingResult",
]
