"""Test tracking when objects appear, disappear, and go temporarily unseen.

This audit changes one environment axis after perceptual aliasing: the latent
dynamics stay fixed, while the number of visible objects varies through birth,
death, and occlusion.  Frames contain only opaque appearance symbols and
positions.  A persistent tracker may carry a track through missing evidence;
the controls either reinitialize every frame or turn missingness into a fake
zero-valued observation.

Latent object identifiers are retained only by the verifier to score whether a
prediction belongs to the right lifetime.  They are never present in a frame or
in a tracker state.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_ID = "brainworkshop-object-lifecycle-2026-08-16"
OBJECT_LIFECYCLE_SCHEMA = "neural-computer.object-lifecycle.v1"
DEVELOPMENT_SEED = 41
RING_SIZE = 12
MAX_OBJECTS = 5
EPISODE_STEPS = 32
EPISODES = 16
EVALUATION_EPISODES = 12
STABLE_THRESHOLD = 0.75
MAX_OCCLUSION_GAP = 3


@dataclass(frozen=True)
class Detection:
    """The only per-object information exposed to a tracker."""

    appearance: int
    position: int


@dataclass(frozen=True)
class ObjectFrame:
    detections: tuple[Detection, ...]
    # Verifier-side truth; not passed to tracker implementations.
    next_position_by_lifetime: tuple[tuple[int, int | None], ...]


@dataclass(frozen=True)
class Episode:
    frames: tuple[ObjectFrame, ...]


def sample_episode(
    seed: int,
    *,
    lifecycle: bool,
    occlusion_rate: float,
) -> Episode:
    if not 0.0 <= occlusion_rate < 1.0:
        raise ValueError("occlusion rate must be in [0, 1)")
    generator = torch.Generator().manual_seed(int(seed))
    objects = []
    count = 2 if not lifecycle else MAX_OBJECTS
    for lifetime in range(count):
        birth = 0 if not lifecycle else int(torch.randint(0, 8, (1,), generator=generator).item())
        death = EPISODE_STEPS if not lifecycle else int(
            torch.randint(20, EPISODE_STEPS + 1, (1,), generator=generator).item()
        )
        position = int(torch.randint(RING_SIZE, (1,), generator=generator).item())
        velocity = int(
            torch.tensor((-2, -1, 1, 2))[int(torch.randint(4, (1,), generator=generator).item())]
        )
        appearance = lifetime + 1
        objects.append((appearance, birth, death, position, velocity))

    frames = []
    for step in range(EPISODE_STEPS):
        detections = []
        next_truth = []
        for appearance, birth, death, position, velocity in objects:
            active = birth <= step < death
            next_active = birth <= step + 1 < death
            if active and torch.rand((), generator=generator).item() >= occlusion_rate:
                detections.append(Detection(appearance, position))
            if active:
                next_truth.append(
                    (
                        appearance,
                        (position + velocity) % RING_SIZE if next_active else None,
                    )
                )
            if active:
                objects[objects.index((appearance, birth, death, position, velocity))] = (
                    appearance,
                    birth,
                    death,
                    (position + velocity) % RING_SIZE,
                    velocity,
                )
        frames.append(
            ObjectFrame(
                detections=tuple(detections),
                next_position_by_lifetime=tuple(next_truth),
            )
        )
    return Episode(tuple(frames))


@dataclass
class Track:
    last_position: int
    velocity: int | None
    last_seen: int


class PersistentTracker:
    """Carry opaque appearance-bound tracks through bounded missingness."""

    def __init__(self, *, max_gap: int = MAX_OCCLUSION_GAP) -> None:
        self.max_gap = int(max_gap)
        self.tracks: dict[int, Track] = {}

    def observe(self, frame: ObjectFrame, step: int) -> dict[int, int | None]:
        predictions = {}
        for appearance, track in self.tracks.items():
            gap = step - track.last_seen
            if gap <= self.max_gap and track.velocity is not None:
                predictions[appearance] = (
                    track.last_position + gap * track.velocity
                ) % RING_SIZE
            elif gap <= self.max_gap:
                predictions[appearance] = None
        for detection in frame.detections:
            prior = self.tracks.get(detection.appearance)
            if prior is not None and step > prior.last_seen:
                gap = step - prior.last_seen
                velocity = prior.velocity
                for candidate in (-2, -1, 1, 2):
                    if (prior.last_position + gap * candidate) % RING_SIZE == detection.position:
                        velocity = candidate
                        break
                self.tracks[detection.appearance] = Track(
                    detection.position,
                    velocity,
                    step,
                )
            else:
                self.tracks[detection.appearance] = Track(detection.position, None, step)
        self.tracks = {
            appearance: track
            for appearance, track in self.tracks.items()
            if step - track.last_seen <= self.max_gap
        }
        return predictions


class ReinitializingTracker:
    """No cross-frame state; a new object is always an abstention."""

    def observe(self, frame: ObjectFrame, step: int) -> dict[int, int | None]:
        del step
        return {detection.appearance: None for detection in frame.detections}


class ZeroMissingTracker(PersistentTracker):
    """Corrupted control that turns an absent track into position zero."""

    def observe(self, frame: ObjectFrame, step: int) -> dict[int, int | None]:
        predictions = super().observe(frame, step)
        for appearance in self.tracks:
            if appearance not in {detection.appearance for detection in frame.detections}:
                predictions[appearance] = 0
        return predictions


def evaluate_episode(episode: Episode, *, mode: str) -> dict[str, float | int]:
    if mode == "persistent":
        tracker = PersistentTracker()
    elif mode == "reinitializing":
        tracker = ReinitializingTracker()
    elif mode == "zero_missing":
        tracker = ZeroMissingTracker()
    else:
        raise ValueError(f"unknown lifecycle tracker: {mode}")
    correct = eligible = predicted = 0
    births = recoveries = false_deaths = 0
    for step, frame in enumerate(episode.frames):
        predictions = tracker.observe(frame, step)
        visible = {detection.appearance for detection in frame.detections}
        if step == 0:
            births += len(visible)
            continue

        # Predictions are emitted before the current frame updates each track,
        # so they target the current latent state.  The prior frame's verifier
        # transition supplies truth even when the current object is occluded.
        truth = dict(episode.frames[step - 1].next_position_by_lifetime)
        for appearance, expected in truth.items():
            if expected is None:
                if appearance in predictions:
                    false_deaths += 1
                continue
            eligible += 1
            prediction = predictions.get(appearance)
            if prediction is None:
                if appearance in visible:
                    births += 1
                continue
            predicted += 1
            correct += int(prediction == expected)
            if appearance not in visible:
                recoveries += int(prediction == expected)
    return {
        "accuracy": correct / eligible if eligible else 0.0,
        "coverage": predicted / eligible if eligible else 0.0,
        "correct": correct,
        "eligible": eligible,
        "predicted": predicted,
        "birth_or_new_track_abstentions": births,
        "occlusion_recoveries": recoveries,
        "false_death_predictions": false_deaths,
    }


def _stable_bits(curve: list[dict[str, float | int]]) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["score"]) >= STABLE_THRESHOLD for later in curve[index:]):
            return int(row["unique_verifier_bits"])
    return None


def run_arm(
    *,
    mode: str,
    seed: int,
    lifecycle: bool,
    occlusion_rate: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    curve = []
    verifier_bits = 0
    episodes_seen = 0
    for prefix in (4, 8, 12, 16):
        metrics = []
        for index in range(prefix):
            episode = sample_episode(
                # Each measured prefix gets a disjoint seed block so the
                # verifier-bit ledger does not count the same lifetime again.
                seed + prefix * 10_000 + index,
                lifecycle=lifecycle,
                occlusion_rate=occlusion_rate,
            )
            metrics.append(evaluate_episode(episode, mode=mode))
        eligible = sum(int(row["eligible"]) for row in metrics)
        correct = sum(int(row["correct"]) for row in metrics)
        predicted = sum(int(row["predicted"]) for row in metrics)
        verifier_bits += eligible
        episodes_seen += prefix
        curve.append(
            {
                "training_episodes": prefix,
                "score": correct / eligible if eligible else 0.0,
                "coverage": predicted / eligible if eligible else 0.0,
                "unique_verifier_bits": verifier_bits,
                "occlusion_recoveries": sum(int(row["occlusion_recoveries"]) for row in metrics),
                "false_death_predictions": sum(int(row["false_death_predictions"]) for row in metrics),
            }
        )
    return {
        "arm": mode,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": episodes_seen,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "latency_ms_per_prediction": 0.0,
        "retention_on_mastered_primitive": "not_claimed",
    }


def run_lifecycle(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        arms = {
            condition: {
                mode: run_arm(
                    mode=mode,
                    seed=seed + 1_000 * replicate + 100 * condition_index,
                    lifecycle=condition != "stable",
                    occlusion_rate=0.0 if condition != "occluded" else 0.25,
                )
                for mode in ("persistent", "reinitializing", "zero_missing")
            }
            for condition_index, condition in enumerate(("stable", "lifecycle", "occluded"))
        }
        rows.append({"replicate": replicate, "conditions": arms})
    accounting = {}
    for condition in ("stable", "lifecycle", "occluded"):
        for mode in ("persistent", "reinitializing", "zero_missing"):
            key = f"{condition}_{mode}"
            values = [row["conditions"][condition][mode] for row in rows]
            accounting[key] = {
                "unique_verifier_bits": sum(int(value["unique_verifier_bits"]) for value in values),
                "unique_logical_lifetimes": sum(int(value["unique_logical_lifetimes"]) for value in values),
                "optimizer_updates": 0,
                "replayed_examples": 0,
                "wall_seconds": sum(float(value["wall_seconds"]) for value in values),
                "latency_ms_per_prediction": sum(
                    float(value["latency_ms_per_prediction"]) for value in values
                )
                / len(values),
                "stable_bits_to_threshold": [value["stable_bits_to_threshold"] for value in values],
                "retention_on_mastered_primitive": "not_claimed",
            }
    report = {
        "schema": OBJECT_LIFECYCLE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "stable_threshold": STABLE_THRESHOLD,
        "accounting": accounting,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "object_lifecycle.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_object_lifecycle_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=3)
    arguments = parser.parse_args()
    print(json.dumps(run_lifecycle(arguments.output, seed=arguments.seed, replicates=arguments.replicates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
