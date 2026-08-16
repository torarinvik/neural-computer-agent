"""Test whether short history resolves perceptual aliasing.

Only one difficulty axis changes here.  The latent world remains a deterministic
ring, but two latent places render to the same learned event symbol.  A model
that keys only on the current symbol must merge incompatible transitions.  A
history-conditioned model can keep the separately bound recent event/action
context and recover a belief-state transition table.

The verifier-side latent place is used only to generate the next observed
symbol and score a prediction.  The fitted models receive observations and
opaque actions, never latent coordinates or the alias map.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_ID = "brainworkshop-perceptual-aliasing-2026-08-16"
PERCEPTUAL_ALIASING_SCHEMA = "neural-computer.perceptual-aliasing.v1"
DEVELOPMENT_SEED = 41
LATENT_PLACES = 6
ACTION_COUNT = 3
EPISODE_STEPS = 24
TRAINING_PREFIXES = (8, 16, 32, 64, 96)
EVALUATION_EPISODES = 40
STABLE_THRESHOLD = 0.95


@dataclass(frozen=True)
class AliasedWorld:
    transitions: tuple[tuple[int, ...], ...]
    observation_symbols: tuple[int, ...]

    def observe(self, latent_place: int) -> int:
        return int(self.observation_symbols[int(latent_place)])


WORLD = AliasedWorld(
    transitions=(
        tuple((place + 1) % LATENT_PLACES for place in range(LATENT_PLACES)),
        tuple((place - 1) % LATENT_PLACES for place in range(LATENT_PLACES)),
        tuple(range(LATENT_PLACES)),
    ),
    # Latent places 0 and 3 are observationally identical.
    observation_symbols=(0, 1, 2, 0, 4, 5),
)


Trace = tuple[tuple[int, ...], tuple[int, ...]]
ModelKey = tuple[int, ...]


def sample_traces(seed: int, *, episodes: int) -> tuple[Trace, ...]:
    generator = torch.Generator().manual_seed(int(seed))
    traces: list[Trace] = []
    for _ in range(int(episodes)):
        place = int(torch.randint(LATENT_PLACES, (1,), generator=generator).item())
        observations = [WORLD.observe(place)]
        actions = []
        for _ in range(EPISODE_STEPS):
            action = int(torch.randint(ACTION_COUNT, (1,), generator=generator).item())
            actions.append(action)
            place = int(WORLD.transitions[action][place])
            observations.append(WORLD.observe(place))
        traces.append((tuple(observations), tuple(actions)))
    return tuple(traces)


def _key(
    observations: tuple[int, ...],
    actions: tuple[int, ...],
    index: int,
    *,
    mode: str,
) -> ModelKey | None:
    if mode == "merged":
        return (observations[index], actions[index])
    if mode == "history":
        if index == 0:
            return None
        return (
            observations[index - 1],
            actions[index - 1],
            observations[index],
            actions[index],
        )
    if mode == "corrupted_history":
        if index == 0:
            return None
        # The memory channel is present but its previous event is corrupted.
        return (
            -1,
            actions[index - 1],
            observations[index],
            actions[index],
        )
    raise ValueError(f"unknown aliasing model: {mode}")


def fit_model(traces: tuple[Trace, ...], *, mode: str):
    table: dict[ModelKey, Counter[int]] = defaultdict(Counter)
    for observations, actions in traces:
        for index in range(len(actions)):
            key = _key(observations, actions, index, mode=mode)
            if key is not None:
                table[key][observations[index + 1]] += 1
    return table


def evaluate_model(model, traces: tuple[Trace, ...], *, mode: str) -> dict[str, float | int]:
    correct = 0
    total = 0
    predicted = 0
    for observations, actions in traces:
        for index in range(len(actions)):
            key = _key(observations, actions, index, mode=mode)
            if key is None:
                continue
            total += 1
            if key not in model:
                continue
            predicted += 1
            correct += int(model[key].most_common(1)[0][0] == observations[index + 1])
    return {
        "accuracy": correct / total if total else 0.0,
        "coverage": predicted / total if total else 0.0,
        "correct": correct,
        "eligible": total,
        "known": predicted,
    }


def _stable_bits(curve: list[dict[str, float | int]]) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["accuracy"]) >= STABLE_THRESHOLD for later in curve[index:]):
            return int(row["unique_verifier_bits"])
    return None


def run_arm(
    *,
    mode: str,
    seed: int,
    training_prefixes: tuple[int, ...] = TRAINING_PREFIXES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    started = time.perf_counter()
    curve = []
    latency = []
    verifier_bits = 0
    for prefix in training_prefixes:
        training = sample_traces(seed + prefix, episodes=prefix)
        model = fit_model(training, mode=mode)
        evaluation = sample_traces(seed + 100_000 + prefix, episodes=evaluation_episodes)
        decision_started = time.perf_counter_ns()
        score = evaluate_model(model, evaluation, mode=mode)
        elapsed = time.perf_counter_ns() - decision_started
        verifier_bits += int(score["eligible"])
        latency.append(elapsed / 1_000_000.0 / max(1, int(score["eligible"])))
        curve.append(
            {
                "training_episodes": prefix,
                "accuracy": score["accuracy"],
                "coverage": score["coverage"],
                "unique_verifier_bits": verifier_bits,
                "known_contexts": len(model),
            }
        )
    return {
        "arm": mode,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": sum(training_prefixes) + len(training_prefixes) * evaluation_episodes,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "latency_ms_per_prediction": sum(latency) / len(latency),
        "retention_on_mastered_primitive": "not_claimed",
    }


def run_aliasing(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        arms = {
            mode: run_arm(mode=mode, seed=seed + 1_000 * replicate)
            for mode in ("merged", "history", "corrupted_history")
        }
        rows.append({"replicate": replicate, "arms": arms})
    accounting = {
        mode: {
            "unique_verifier_bits": sum(
                int(row["arms"][mode]["unique_verifier_bits"]) for row in rows
            ),
            "unique_logical_lifetimes": sum(
                int(row["arms"][mode]["unique_logical_lifetimes"]) for row in rows
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(float(row["arms"][mode]["wall_seconds"]) for row in rows),
            "latency_ms_per_prediction": sum(
                float(row["arms"][mode]["latency_ms_per_prediction"]) for row in rows
            )
            / len(rows),
            "stable_bits_to_threshold": [row["arms"][mode]["stable_bits_to_threshold"] for row in rows],
            "retention_on_mastered_primitive": "not_claimed",
        }
        for mode in ("merged", "history", "corrupted_history")
    }
    report = {
        "schema": PERCEPTUAL_ALIASING_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "latent_places": LATENT_PLACES,
        "observation_symbol_count": len(set(WORLD.observation_symbols)),
        "aliased_latent_places": [0, 3],
        "stable_threshold": STABLE_THRESHOLD,
        "accounting": accounting,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "perceptual_aliasing.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_perceptual_aliasing_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=3)
    arguments = parser.parse_args()
    print(json.dumps(run_aliasing(arguments.output, seed=arguments.seed, replicates=arguments.replicates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
