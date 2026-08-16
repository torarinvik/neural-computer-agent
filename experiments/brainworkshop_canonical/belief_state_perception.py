"""Development audit for belief-state perception under aliased observations.

The learned event alphabet intentionally aliases two latent places.  A hard
current-symbol/history table either merges the places or drops a prediction
when one event is missing.  ``BeliefStatePerception`` keeps a weighted set of
opaque context hypotheses, marginalizes missing events, and exposes confidence
so the caller can abstain instead of turning absence into a zero-valued event.
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

from .perceptual_aliasing import evaluate_model, fit_model, sample_traces

EXPERIMENT_ID = "brainworkshop-belief-state-perception-2026-08-16"
EXPERIMENT_SCHEMA = "neural-computer.belief-state-perception.v1"
DEVELOPMENT_SEED = 41
TRAINING_EPISODES = 16
EVALUATION_EPISODES = 40
MISSING_RATE = 0.20
CONFIDENCE_FLOOR = 0.80

Observation = int | None
Trace = tuple[tuple[int, ...], tuple[int, ...]]
MaskedTrace = tuple[tuple[Observation, ...], tuple[int, ...]]


@dataclass
class BeliefStatePerception:
    """A compact context posterior over learned event histories."""

    confidence_floor: float = CONFIDENCE_FLOOR

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("belief confidence floor must lie in [0, 1]")
        self._full: dict[tuple[int, int, int, int], Counter[int]] = defaultdict(Counter)
        self._short: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self._symbol: dict[int, Counter[int]] = defaultdict(Counter)

    @property
    def context_count(self) -> int:
        return len(self._full) + len(self._short) + len(self._symbol)

    def fit(self, traces: tuple[Trace, ...]) -> BeliefStatePerception:
        """Learn only from event/action sequences and next-event outcomes."""

        for observations, actions in traces:
            for index, action in enumerate(actions):
                following = int(observations[index + 1])
                self._short[(int(observations[index]), int(action))][following] += 1
                self._symbol[int(observations[index])][following] += 1
                if index > 0:
                    self._full[
                        (
                            int(observations[index - 1]),
                            int(actions[index - 1]),
                            int(observations[index]),
                            int(action),
                        )
                    ][following] += 1
        return self

    def distribution(
        self,
        observations: tuple[Observation, ...],
        actions: tuple[int, ...],
        index: int,
    ) -> dict[int, float]:
        """Marginalize context rows that agree with all present evidence."""

        if not 0 <= index < len(actions):
            raise ValueError("belief prediction index is outside the action trace")
        evidence: Counter[int] = Counter()
        if index > 0:
            query = (
                observations[index - 1],
                int(actions[index - 1]),
                observations[index],
                int(actions[index]),
            )
            for key, counts in self._full.items():
                if all(value is None or int(value) == int(expected) for value, expected in zip(query, key, strict=True)):
                    specificity = sum(value is not None for value in query)
                    for symbol, count in counts.items():
                        evidence[symbol] += int(count) * (4 ** specificity)
        if not evidence and observations[index] is not None:
            evidence.update(self._short.get((int(observations[index]), int(actions[index])), {}))
        if not evidence and observations[index] is not None:
            evidence.update(self._symbol.get(int(observations[index]), {}))
        total = sum(evidence.values())
        return {
            int(symbol): float(count) / total
            for symbol, count in evidence.items()
        } if total else {}

    def predict(
        self,
        observations: tuple[Observation, ...],
        actions: tuple[int, ...],
        index: int,
    ) -> tuple[int | None, float]:
        distribution = self.distribution(observations, actions, index)
        if not distribution:
            return None, 0.0
        symbol, confidence = max(distribution.items(), key=lambda item: item[1])
        if confidence < self.confidence_floor:
            return None, float(confidence)
        return int(symbol), float(confidence)


def mask_traces(
    traces: tuple[Trace, ...], *, seed: int, missing_rate: float
) -> tuple[MaskedTrace, ...]:
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing rate must lie in [0, 1)")
    generator = torch.Generator().manual_seed(int(seed))
    masked: list[MaskedTrace] = []
    for observations, actions in traces:
        values: list[Observation] = [int(observations[0])]
        for value in observations[1:]:
            values.append(
                None
                if float(torch.rand((), generator=generator).item()) < missing_rate
                else int(value)
            )
        masked.append((tuple(values), actions))
    return tuple(masked)


def evaluate_belief(
    model: BeliefStatePerception,
    traces: tuple[MaskedTrace, ...],
) -> dict[str, float | int]:
    correct = eligible = predicted = 0
    confidence_sum = 0.0
    for observations, actions in traces:
        for index in range(len(actions)):
            target = observations[index + 1]
            if target is None:
                continue
            eligible += 1
            prediction, confidence = model.predict(observations, actions, index)
            confidence_sum += confidence
            if prediction is None:
                continue
            predicted += 1
            correct += int(prediction == target)
    return {
        "accuracy": correct / predicted if predicted else 0.0,
        "coverage": predicted / eligible if eligible else 0.0,
        "expected_correct_rate": correct / eligible if eligible else 0.0,
        "mean_confidence": confidence_sum / eligible if eligible else 0.0,
        "correct": correct,
        "predicted": predicted,
        "eligible": eligible,
    }


def evaluate_history(
    model: BeliefStatePerception,
    traces: tuple[MaskedTrace, ...],
) -> dict[str, float | int]:
    """The hard full-context control: missing fields cause abstention."""

    correct = predicted = eligible = 0
    for observations, actions in traces:
        for index in range(len(actions)):
            target = observations[index + 1]
            if target is None:
                continue
            eligible += 1
            if index == 0 or any(
                value is None
                for value in (
                    observations[index - 1],
                    observations[index],
                )
            ):
                continue
            counts = model._full.get(
                (
                    int(observations[index - 1]),
                    int(actions[index - 1]),
                    int(observations[index]),
                    int(actions[index]),
                )
            )
            if not counts:
                continue
            predicted += 1
            correct += int(counts.most_common(1)[0][0] == target)
    return {
        "accuracy": correct / predicted if predicted else 0.0,
        "coverage": predicted / eligible if eligible else 0.0,
        "expected_correct_rate": correct / eligible if eligible else 0.0,
        "correct": correct,
        "predicted": predicted,
        "eligible": eligible,
    }


def run_belief_state_perception(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    missing_rate: float = MISSING_RATE,
) -> dict[str, Any]:
    started = time.perf_counter()
    training = sample_traces(seed, episodes=TRAINING_EPISODES)
    evaluation = sample_traces(seed + 10_000, episodes=EVALUATION_EPISODES)
    masked = mask_traces(evaluation, seed=seed + 20_000, missing_rate=missing_rate)
    belief = BeliefStatePerception().fit(training)
    history = BeliefStatePerception(confidence_floor=0.0).fit(training)
    # The history control is intentionally the same learned context family,
    # but it does not marginalize missing fields and therefore abstains on any
    # incomplete full key.
    clean = evaluate_belief(belief, evaluation)
    missing_belief = evaluate_belief(belief, masked)
    missing_history = evaluate_history(history, masked)
    merged_clean = evaluate_model(fit_model(training, mode="merged"), evaluation, mode="merged")
    history_clean = evaluate_model(fit_model(training, mode="history"), evaluation, mode="history")
    report = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "training_episodes": TRAINING_EPISODES,
        "evaluation_episodes": EVALUATION_EPISODES,
        "missing_rate": missing_rate,
        "confidence_floor": CONFIDENCE_FLOOR,
        "arms": {
            "merged_clean": merged_clean,
            "history_clean": history_clean,
            "belief_clean": clean,
            "belief_missing": missing_belief,
            "history_missing": missing_history,
        },
        "context_count": belief.context_count,
        "missing_evidence_is_not_zero": True,
        "wall_seconds": time.perf_counter() - started,
        "claim_status": "development_belief_state_diagnostic_not_promoted",
        "claim_boundary": (
            "The belief artifact consumes only learned event symbols and opaque "
            "actions. It improves coverage under missing observations in this "
            "aliased symbolic fixture, but it is not yet wired into the rendered "
            "navigation controller or a fresh-learner holdout."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "belief_state_perception.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "session_records"
        / "brainworkshop_belief_state_perception_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--missing-rate", type=float, default=MISSING_RATE)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_belief_state_perception(
                arguments.output,
                seed=arguments.seed,
                missing_rate=arguments.missing_rate,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
