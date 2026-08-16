"""Discover and verify a relational vocabulary from opaque event streams.

The decomposition and relational navigation records select from fixed
candidate lists.  This audit removes the semantic names from that boundary.
It generates temporal predicates over separately bound event channels,
composes the generated predicates, and ranks them by description bits plus
verifier-error bits.  A candidate selected on one world is quarantined until
it reproduces a fresh stream before it can become an external artifact.

The hidden rule names in this file are scoring-side annotations only.  The
learner sees integer event channels and scalar verifier outcomes, never the
rule name or a semantic coordinate map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_ID = "brainworkshop-generated-vocabulary-transfer-2026-08-16"
VOCABULARY_SCHEMA = "neural-computer.generated-vocabulary-transfer.v1"
PREDICATE_SCHEMA = "neural-computer.generated-temporal-predicate.v1"
ARTIFACT_SCHEMA = "neural-computer.verified-predicate-artifact.v1"
DEVELOPMENT_SEED = 41
ALPHABET = 5
CHANNELS = 2
STREAM_STEPS = 64
SOURCE_STREAMS = 4
SOURCE_VERIFICATION_STREAMS = 2
TARGET_DISCOVERY_STREAMS = 3
TARGET_EVALUATION_STREAMS = 4
CHECKPOINT_THRESHOLD = 0.95
RULES = ("equal", "same_change", "same_delta")


@dataclass(frozen=True)
class EventStream:
    """Opaque bound event channels and scalar verifier outcomes."""

    channels: tuple[tuple[int, ...], ...]
    outcomes: tuple[int, ...]
    alphabet: int
    world_seed: int

    def validate(self) -> EventStream:
        if self.alphabet < 2 or len(self.channels) < 2:
            raise ValueError("an event stream needs two bounded channels")
        length = len(self.outcomes)
        if length < 2 or any(len(channel) != length for channel in self.channels):
            raise ValueError("event channels and outcomes must have equal length")
        if any(value not in (0, 1) for value in self.outcomes):
            raise ValueError("verifier outcomes must be binary")
        if any(
            value < 0 or value >= self.alphabet
            for channel in self.channels
            for value in channel
        ):
            raise ValueError("event symbols leave the discovered alphabet")
        return self

    @property
    def digest(self) -> str:
        self.validate()
        payload = json.dumps(
            {"channels": self.channels, "outcomes": self.outcomes},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Predicate:
    """A generated expression, not a hand-written semantic field."""

    op: str
    channels: tuple[int, ...] = ()
    children: tuple[Predicate, ...] = ()

    def validate(self) -> Predicate:
        primitives = {
            "equal",
            "persist",
            "change",
            "same_change",
            "same_delta",
        }
        if self.op in primitives and self.children:
            raise ValueError("primitive predicate cannot have children")
        if self.op in {"equal", "same_change", "same_delta"} and len(self.channels) != 2:
            raise ValueError("binary predicate needs two channels")
        if self.op in {"persist", "change"} and len(self.channels) != 1:
            raise ValueError("unary temporal predicate needs one channel")
        if self.op in {"and", "or", "xor"} and len(self.children) != 2:
            raise ValueError("composition needs two children")
        if self.op not in primitives | {"and", "or", "xor", "not"}:
            raise ValueError("unknown generated predicate operator")
        if self.op == "not" and len(self.children) != 1:
            raise ValueError("negation needs one child")
        for child in self.children:
            child.validate()
        return self

    @property
    def complexity(self) -> int:
        return 1 + sum(child.complexity for child in self.children)

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": PREDICATE_SCHEMA,
            "op": self.op,
            "channels": list(self.channels),
            "children": [child.payload() for child in self.children],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Predicate:
        if payload.get("schema") != PREDICATE_SCHEMA:
            raise ValueError("unsupported predicate schema")
        predicate = cls(
            op=str(payload["op"]),
            channels=tuple(int(value) for value in payload.get("channels", ())),
            children=tuple(cls.from_payload(child) for child in payload.get("children", ())),
        )
        return predicate.validate()

    @property
    def digest(self) -> str:
        payload = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def evaluate(self, stream: EventStream) -> tuple[int, ...]:
        stream.validate()
        values = []
        for index in range(len(stream.outcomes)):
            values.append(self._at(stream, index))
        return tuple(values)

    def _at(self, stream: EventStream, index: int) -> int:
        if self.op == "equal":
            return int(stream.channels[self.channels[0]][index] == stream.channels[self.channels[1]][index])
        if self.op in {"persist", "change"}:
            if index == 0:
                return 0
            equal = stream.channels[self.channels[0]][index] == stream.channels[self.channels[0]][index - 1]
            return int(equal if self.op == "persist" else not equal)
        if self.op in {"same_change", "same_delta"}:
            if index == 0:
                return 0
            left = stream.channels[self.channels[0]]
            right = stream.channels[self.channels[1]]
            left_delta = (left[index] - left[index - 1]) % stream.alphabet
            right_delta = (right[index] - right[index - 1]) % stream.alphabet
            if self.op == "same_delta":
                return int(left_delta == right_delta)
            return int((left_delta == 0) == (right_delta == 0))
        child_values = [child._at(stream, index) for child in self.children]
        if self.op == "not":
            return int(not child_values[0])
        if self.op == "and":
            return int(child_values[0] and child_values[1])
        if self.op == "or":
            return int(child_values[0] or child_values[1])
        if self.op == "xor":
            return int(child_values[0] != child_values[1])
        raise AssertionError("validated predicate reached an unknown operator")


def sample_stream(
    seed: int,
    *,
    rule: str,
    steps: int = STREAM_STEPS,
    alphabet: int = ALPHABET,
    offset: int = 0,
) -> EventStream:
    """Generate a stream with a shared but world-shifted event alphabet."""

    if rule not in RULES:
        raise ValueError(f"unknown hidden rule: {rule}")
    generator = torch.Generator().manual_seed(int(seed))
    channels = [[
        int(torch.randint(alphabet, (1,), generator=generator).item())
    ] for _ in range(CHANNELS)]
    for _ in range(1, int(steps)):
        for channel in channels:
            delta = int(torch.randint(alphabet, (1,), generator=generator).item())
            channel.append((channel[-1] + delta + int(offset)) % alphabet)
    shifted = tuple(
        tuple((value + int(offset)) % alphabet for value in channel)
        for channel in channels
    )
    stream = EventStream(
        channels=shifted,
        outcomes=tuple(
            _hidden_outcome(rule, shifted, index, alphabet)
            for index in range(int(steps))
        ),
        alphabet=alphabet,
        world_seed=int(seed),
    )
    return stream.validate()


def _hidden_outcome(
    rule: str, channels: tuple[tuple[int, ...], ...], index: int, alphabet: int
) -> int:
    if index == 0:
        return 0
    if rule == "equal":
        return int(channels[0][index] == channels[1][index])
    left_delta = (channels[0][index] - channels[0][index - 1]) % alphabet
    right_delta = (channels[1][index] - channels[1][index - 1]) % alphabet
    if rule == "same_change":
        return int((left_delta == 0) == (right_delta == 0))
    if rule == "same_delta":
        return int(left_delta == right_delta)
    raise ValueError(f"unknown hidden rule: {rule}")


def generate_candidates(stream: EventStream) -> tuple[Predicate, ...]:
    """Generate temporal/equality candidates from the stream's channel count."""

    stream.validate()
    base: list[Predicate] = []
    for channel in range(len(stream.channels)):
        base.extend((Predicate("persist", (channel,)), Predicate("change", (channel,))))
    for left, right in combinations(range(len(stream.channels)), 2):
        base.extend(
            (
                Predicate("equal", (left, right)),
                Predicate("same_change", (left, right)),
                Predicate("same_delta", (left, right)),
            )
        )
    candidates = list(base)
    for left, right in combinations(base, 2):
        candidates.extend(
            Predicate(op, children=(left, right)) for op in ("and", "or", "xor")
        )
    candidates.extend(Predicate("not", children=(candidate,)) for candidate in base)
    unique = {candidate.digest: candidate.validate() for candidate in candidates}
    return tuple(unique.values())


def score_candidate(candidate: Predicate, stream: EventStream) -> dict[str, float | int]:
    prediction = candidate.evaluate(stream)
    errors = sum(int(left != right) for left, right in zip(prediction, stream.outcomes))
    description_bits = candidate.complexity * max(1.0, torch.log2(torch.tensor(stream.alphabet)).item())
    return {
        "description_bits": description_bits,
        "error_bits": errors,
        "total_bits": description_bits + errors,
        "errors": errors,
    }


def discover(streams: tuple[EventStream, ...]) -> tuple[Predicate, dict[str, float | int]]:
    if not streams:
        raise ValueError("candidate discovery needs at least one stream")
    first = streams[0].validate()
    if any(
        stream.validate().alphabet != first.alphabet
        or len(stream.channels) != len(first.channels)
        for stream in streams[1:]
    ):
        raise ValueError("candidate discovery streams have incompatible schemas")
    candidates = generate_candidates(first)
    rows = []
    for candidate in candidates:
        scores = [score_candidate(candidate, stream) for stream in streams]
        total = sum(float(score["total_bits"]) for score in scores)
        errors = sum(int(score["errors"]) for score in scores)
        rows.append((total, errors, candidate.complexity, candidate.digest, candidate))
    _, _, _, _, selected = min(rows)
    selected_scores = [score_candidate(selected, stream) for stream in streams]
    return selected, {
        "candidate_count": len(candidates),
        "training_errors": sum(int(score["errors"]) for score in selected_scores),
        "training_total_bits": sum(float(score["total_bits"]) for score in selected_scores),
        "selected_complexity": selected.complexity,
    }


@dataclass(frozen=True)
class PredicateArtifact:
    schema: str
    predicate: Predicate
    source_world_digest: str
    source_training_streams: int
    source_verification_streams: int

    def validate(self) -> PredicateArtifact:
        if self.schema != ARTIFACT_SCHEMA:
            raise ValueError("unsupported predicate artifact")
        self.predicate.validate()
        if self.source_training_streams < 1 or self.source_verification_streams < 1:
            raise ValueError("an artifact needs discovery and verification evidence")
        return self

    @property
    def digest(self) -> str:
        self.validate()
        payload = json.dumps(
            {
                "schema": self.schema,
                "predicate": self.predicate.payload(),
                "source_world_digest": self.source_world_digest,
                "source_training_streams": self.source_training_streams,
                "source_verification_streams": self.source_verification_streams,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def admit_verified(
    candidate: Predicate,
    training_streams: tuple[EventStream, ...],
    verification_streams: tuple[EventStream, ...],
) -> PredicateArtifact | None:
    """Quarantine until both discovery and fresh evidence are exact."""

    if not training_streams or not verification_streams:
        raise ValueError("candidate admission needs discovery and verification streams")
    if any(score_candidate(candidate, stream)["errors"] for stream in training_streams):
        return None
    if any(score_candidate(candidate, stream)["errors"] for stream in verification_streams):
        return None
    return PredicateArtifact(
        schema=ARTIFACT_SCHEMA,
        predicate=candidate,
        source_world_digest=training_streams[0].digest,
        source_training_streams=len(training_streams),
        source_verification_streams=len(verification_streams),
    ).validate()


def _stable_bits(curve: list[dict[str, float | int]], *, offset_bits: int) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["accuracy"]) >= CHECKPOINT_THRESHOLD for later in curve[index:]):
            return offset_bits + int(row["unique_verifier_bits"])
    return None


def _evaluate_artifact(
    artifact: PredicateArtifact,
    streams: tuple[EventStream, ...],
    *,
    offset_bits: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    curve = []
    bits = 0
    for index, stream in enumerate(streams, start=1):
        scores = score_candidate(artifact.predicate, stream)
        bits += len(stream.outcomes)
        curve.append(
            {
                "stream": index,
                "unique_verifier_bits": bits,
                "accuracy": 1.0 - float(scores["errors"]) / len(stream.outcomes),
                "errors": scores["errors"],
            }
        )
    elapsed = time.perf_counter() - started
    return {
        "artifact_digest": artifact.digest,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve, offset_bits=offset_bits),
        "unique_verifier_bits": offset_bits + bits,
        "unique_logical_lifetimes": len(streams),
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": elapsed,
        "latency_ms_per_event": elapsed * 1_000.0 / max(1, bits),
        "retention_on_mastered_primitive": "not_claimed",
    }


def _discover_artifact(*, seed: int, rule: str) -> tuple[PredicateArtifact, dict[str, Any]]:
    training = tuple(
        sample_stream(seed + index, rule=rule, offset=index % ALPHABET)
        for index in range(SOURCE_STREAMS)
    )
    verification = tuple(
        sample_stream(seed + 100 + index, rule=rule, offset=(index + 2) % ALPHABET)
        for index in range(SOURCE_VERIFICATION_STREAMS)
    )
    predicate, discovery = discover(training)
    artifact = admit_verified(predicate, training, verification)
    if artifact is None:
        raise RuntimeError("generated vocabulary candidate failed source verification")
    return artifact, {
        "predicate": predicate.payload(),
        "predicate_digest": predicate.digest,
        "discovery": discovery,
        "source_verification_errors": 0,
    }


def run_transfer(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 3,
    target_rule: str = "same_delta",
) -> dict[str, Any]:
    """Compare retained vocabulary with fresh and corrupted candidate controls."""

    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        source, source_discovery = _discover_artifact(
            seed=seed + 10_000 * replicate, rule=target_rule
        )
        irrelevant, _ = _discover_artifact(
            seed=seed + 10_000 * replicate + 500, rule="same_change"
        )
        target_streams = tuple(
            sample_stream(
                seed + 200_000 + 1_000 * replicate + index,
                rule=target_rule,
                offset=(index + 1) % ALPHABET,
            )
            for index in range(TARGET_EVALUATION_STREAMS)
        )
        target_discovery = tuple(
            sample_stream(
                seed + 300_000 + 1_000 * replicate + index,
                rule=target_rule,
                offset=(index + 3) % ALPHABET,
            )
            for index in range(TARGET_DISCOVERY_STREAMS)
        )
        fresh_predicate, fresh_discovery = discover(target_discovery)
        fresh_artifact = admit_verified(fresh_predicate, target_discovery, target_streams)
        if fresh_artifact is None:
            raise RuntimeError("fresh candidate failed target verification")
        corrupted_predicate = Predicate("not", children=(source.predicate,)).validate()
        corrupted = PredicateArtifact(
            schema=ARTIFACT_SCHEMA,
            predicate=corrupted_predicate,
            source_world_digest=source.source_world_digest,
            source_training_streams=source.source_training_streams,
            source_verification_streams=source.source_verification_streams,
        ).validate()
        arms = {
            "retained": _evaluate_artifact(source, target_streams, offset_bits=0),
            "fresh": _evaluate_artifact(
                fresh_artifact,
                target_streams,
                offset_bits=len(target_discovery) * STREAM_STEPS,
            ),
            "irrelevant": _evaluate_artifact(irrelevant, target_streams, offset_bits=0),
            "corrupted": _evaluate_artifact(corrupted, target_streams, offset_bits=0),
        }
        rows.append(
            {
                "replicate": replicate,
                "target_rule": target_rule,
                "source_artifact_digest": source.digest,
                "source_discovery": source_discovery,
                "fresh_discovery": fresh_discovery,
                "fresh_predicate_digest": fresh_artifact.predicate.digest,
                "arms": arms,
            }
        )
    retained_bits = [row["arms"]["retained"]["stable_bits_to_threshold"] for row in rows]
    fresh_bits = [row["arms"]["fresh"]["stable_bits_to_threshold"] for row in rows]
    ratios = [
        float(retained) / float(fresh)
        for retained, fresh in zip(retained_bits, fresh_bits)
        if retained is not None and fresh is not None and fresh > 0
    ]
    accounting = {
        arm: {
            "unique_verifier_bits": sum(int(row["arms"][arm]["unique_verifier_bits"]) for row in rows),
            "unique_logical_lifetimes": sum(int(row["arms"][arm]["unique_logical_lifetimes"]) for row in rows),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(
                float(row["arms"][arm]["wall_seconds"]) for row in rows
            ),
            "latency_ms_per_event": sum(
                float(row["arms"][arm]["latency_ms_per_event"]) for row in rows
            )
            / len(rows),
            "stable_bits_to_threshold": [row["arms"][arm]["stable_bits_to_threshold"] for row in rows],
            "retention_on_mastered_primitive": "not_claimed",
        }
        for arm in ("retained", "fresh", "irrelevant", "corrupted")
    }
    report = {
        "schema": VOCABULARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "generated_candidate_count": len(generate_candidates(target_streams[0])),
        "target_rule_scoring_annotation": target_rule,
        "transfer_ratio_against_fresh_learner": sum(ratios) / len(ratios) if ratios else None,
        "accounting": accounting,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "generated_vocabulary_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_generated_vocabulary_transfer_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--target-rule", choices=RULES, default="same_delta")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_transfer(
                arguments.output,
                seed=arguments.seed,
                replicates=arguments.replicates,
                target_rule=arguments.target_rule,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
