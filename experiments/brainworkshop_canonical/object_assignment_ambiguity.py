"""Track an action-responsive object when appearances collide.

The lifecycle diagnostic deliberately gave every lifetime a unique opaque
appearance symbol.  This audit removes that shortcut: the controlled object
and an independently moving distractor can emit the same symbol, cross on the
ring, and remain separately visible.  A frame therefore contains only an
unordered pair of ``(appearance, position)`` events; latent lifetime labels
exist only in the verifier.

The causal beam keeps several assignments alive and scores them by whether the
candidate controlled track responds to the observed action.  Nearest-neighbor
tracking is the continuity-only control, while the first-event tracker is a
symbol/correspondence control.  The distractor responds to the action on a
fixed fraction of steps, so action response is useful evidence but not a
perfect label.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_ID = "brainworkshop-object-assignment-ambiguity-2026-08-16"
OBJECT_ASSIGNMENT_SCHEMA = "neural-computer.object-assignment-ambiguity.v1"
DEVELOPMENT_SEED = 41
RING_SIZE = 16
EPISODE_STEPS = 36
STABLE_THRESHOLD = 0.75
TRAINING_PREFIXES = (4, 8, 12, 16)
BEAM_WIDTH = 8
ACTION_DELTAS = (-1, 1)


@dataclass(frozen=True)
class Detection:
    """The only object event exposed to a tracker."""

    appearance: int
    position: int


@dataclass(frozen=True)
class Frame:
    detections: tuple[Detection, ...]
    # Action that transformed the previous frame into this frame.  ``None``
    # marks the first observation and is part of the ordinary action log.
    action: int | None


@dataclass(frozen=True)
class Episode:
    frames: tuple[Frame, ...]
    # Verifier-side truth; never passed to a tracker.
    controlled_positions: tuple[int, ...]


def _signed_delta(previous: int, current: int) -> int:
    delta = (int(current) - int(previous)) % RING_SIZE
    return delta if delta <= RING_SIZE // 2 else delta - RING_SIZE


def sample_episode(
    seed: int,
    *,
    collision: bool,
    response_probability: float,
) -> Episode:
    if not 0.0 <= response_probability <= 1.0:
        raise ValueError("response probability must be in [0, 1]")
    generator = torch.Generator().manual_seed(int(seed))
    controlled = int(torch.randint(RING_SIZE, (1,), generator=generator).item())
    # An odd separation stays odd under +/-1 moves, so the two visible objects
    # can cross by swapping positions without merging into one rendered part.
    distractor = (controlled + 5) % RING_SIZE
    distractor_velocity = ACTION_DELTAS[
        int(torch.randint(2, (1,), generator=generator).item())
    ]
    frames: list[Frame] = []
    truth: list[int] = []
    action_from_previous: int | None = None
    for _ in range(EPISODE_STEPS):
        detections = [
            Detection(0 if collision else 1, controlled),
            Detection(0 if collision else 2, distractor),
        ]
        detections.sort(key=lambda detection: detection.position)
        frames.append(Frame(tuple(detections), action_from_previous))
        truth.append(controlled)
        action = int(torch.randint(2, (1,), generator=generator).item())
        delta = ACTION_DELTAS[action]
        controlled = (controlled + delta) % RING_SIZE
        if float(torch.rand((), generator=generator).item()) < response_probability:
            distractor_delta = delta
        else:
            distractor_delta = distractor_velocity
        distractor = (distractor + distractor_delta) % RING_SIZE
        action_from_previous = action
    return Episode(tuple(frames), tuple(truth))


class AppearanceTracker:
    """A control that keeps the first visible event as the controlled object."""

    def __init__(self) -> None:
        self.position: int | None = None

    def observe(self, frame: Frame) -> int | None:
        if not frame.detections:
            return None
        if self.position is None:
            self.position = frame.detections[0].position
        else:
            # The symbol is deliberately not interpreted as a role.  When the
            # two objects collide in appearance this reduces to first-event
            # selection; it cannot recover from a crossing.
            self.position = frame.detections[0].position
        return self.position


class NearestTracker:
    """Continuity-only tracking without action-conditioned identity evidence."""

    def __init__(self) -> None:
        self.position: int | None = None
        self.velocity: int | None = None

    def observe(self, frame: Frame) -> int | None:
        if not frame.detections:
            return None
        if self.position is None:
            self.position = frame.detections[0].position
            return self.position
        previous = self.position
        expected = previous if self.velocity is None else (previous + self.velocity) % RING_SIZE
        chosen = min(
            (detection.position for detection in frame.detections),
            key=lambda position: abs(_signed_delta(expected, position)),
        )
        self.velocity = _signed_delta(previous, chosen)
        self.position = chosen
        return self.position


@dataclass(frozen=True)
class Hypothesis:
    controlled: int
    distractor: int
    cost: float


class CausalBeamTracker:
    """Keep assignments whose controlled track responds to the action."""

    def __init__(self, *, beam_width: int = BEAM_WIDTH) -> None:
        self.beam_width = int(beam_width)
        self.hypotheses: tuple[Hypothesis, ...] = ()

    def observe(self, frame: Frame) -> int | None:
        positions = tuple(detection.position for detection in frame.detections)
        if len(positions) != 2:
            return None
        if not self.hypotheses:
            self.hypotheses = (
                Hypothesis(positions[0], positions[1], 0.0),
                Hypothesis(positions[1], positions[0], 0.0),
            )
            return self.hypotheses[0].controlled
        candidates = []
        for hypothesis in self.hypotheses:
            for controlled, distractor in (
                (positions[0], positions[1]),
                (positions[1], positions[0]),
            ):
                mismatch = 0.0
                if frame.action is not None:
                    expected = ACTION_DELTAS[int(frame.action)]
                    mismatch = float(
                        _signed_delta(hypothesis.controlled, controlled) != expected
                    )
                candidates.append(
                    Hypothesis(controlled, distractor, hypothesis.cost + mismatch)
                )
        candidates.sort(key=lambda hypothesis: (hypothesis.cost, hypothesis.controlled, hypothesis.distractor))
        self.hypotheses = tuple(candidates[: self.beam_width])
        return self.hypotheses[0].controlled


def evaluate_episode(episode: Episode, *, mode: str) -> dict[str, float | int]:
    if mode == "appearance":
        tracker = AppearanceTracker()
    elif mode == "nearest":
        tracker = NearestTracker()
    elif mode == "causal_beam":
        tracker = CausalBeamTracker()
    else:
        raise ValueError(f"unknown assignment tracker: {mode}")
    correct = 0
    for frame, expected in zip(episode.frames, episode.controlled_positions):
        prediction = tracker.observe(frame)
        correct += int(prediction == expected)
    return {
        "accuracy": correct / len(episode.frames),
        "correct": correct,
        "eligible": len(episode.frames),
    }


def _stable_bits(curve: list[dict[str, float | int]]) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["score"]) >= STABLE_THRESHOLD for later in curve[index:]):
            return int(row["unique_verifier_bits"])
    return None


def run_arm(*, mode: str, seed: int, collision: bool, response_probability: float) -> dict[str, Any]:
    started = time.perf_counter()
    curve = []
    verifier_bits = 0
    lifetimes = 0
    for prefix in TRAINING_PREFIXES:
        metrics = [
            evaluate_episode(
                sample_episode(
                    seed + prefix * 10_000 + index,
                    collision=collision,
                    response_probability=response_probability,
                ),
                mode=mode,
            )
            for index in range(prefix)
        ]
        correct = sum(int(row["correct"]) for row in metrics)
        eligible = sum(int(row["eligible"]) for row in metrics)
        verifier_bits += eligible
        lifetimes += prefix
        curve.append(
            {
                "training_episodes": prefix,
                "score": correct / eligible,
                "unique_verifier_bits": verifier_bits,
            }
        )
    return {
        "arm": mode,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": lifetimes,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "latency_ms_per_prediction": 0.0,
        "retention_on_mastered_primitive": "not_claimed",
    }


def run_assignment(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        conditions = {
            condition: {
                mode: run_arm(
                    mode=mode,
                    seed=seed + 1_000 * replicate + 100 * condition_index,
                    collision=True,
                    response_probability=response_probability,
                )
                for mode in ("causal_beam", "nearest", "appearance")
            }
            for condition_index, (condition, response_probability) in enumerate(
                (("independent_collision", 0.0), ("approximate_collision", 0.35))
            )
        }
        rows.append({"replicate": replicate, "conditions": conditions})
    accounting = {}
    for condition in ("independent_collision", "approximate_collision"):
        for mode in ("causal_beam", "nearest", "appearance"):
            values = [row["conditions"][condition][mode] for row in rows]
            accounting[f"{condition}_{mode}"] = {
                "unique_verifier_bits": sum(int(value["unique_verifier_bits"]) for value in values),
                "unique_logical_lifetimes": sum(int(value["unique_logical_lifetimes"]) for value in values),
                "optimizer_updates": 0,
                "replayed_examples": 0,
                "wall_seconds": sum(float(value["wall_seconds"]) for value in values),
                "latency_ms_per_prediction": 0.0,
                "stable_bits_to_threshold": [value["stable_bits_to_threshold"] for value in values],
                "retention_on_mastered_primitive": "not_claimed",
            }
    report = {
        "schema": OBJECT_ASSIGNMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "ring_size": RING_SIZE,
        "stable_threshold": STABLE_THRESHOLD,
        "accounting": accounting,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "object_assignment_ambiguity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_object_assignment_ambiguity_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=3)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_assignment(arguments.output, seed=arguments.seed, replicates=arguments.replicates),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
