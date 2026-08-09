"""Two-seed replay-free nonlinear feature-memory audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ExternalRandomFeatureTransitionStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
TRAIN_ROWS = 64
HELDOUT_ROWS = 64
FEATURE_WIDTH = 128


def _make_observation(seed: int) -> tuple[ExternalTransitionObservation, ExternalTransitionObservation]:
    torch.manual_seed(seed)
    state = torch.rand(TRAIN_ROWS + HELDOUT_ROWS, STATE_WIDTH) * 2.0 - 1.0
    intention = torch.rand(TRAIN_ROWS + HELDOUT_ROWS, INTENTION_WIDTH) * 2.0 - 1.0
    next_state = torch.cat(
        (
            torch.sin(2.0 * state[:, 0:1] + intention),
            state[:, 0:1] * state[:, 1:2] + intention.square(),
        ),
        dim=-1,
    )
    return (
        ExternalTransitionObservation(
            state=state[:TRAIN_ROWS],
            intention=intention[:TRAIN_ROWS],
            next_state=next_state[:TRAIN_ROWS],
        ),
        ExternalTransitionObservation(
            state=state[TRAIN_ROWS:],
            intention=intention[TRAIN_ROWS:],
            next_state=next_state[TRAIN_ROWS:],
        ),
    )


def _digest(model: ExternalRandomFeatureTransitionStatistics) -> str:
    return hashlib.sha256(model.state_payload()["sha256"].encode()).hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    train, heldout = _make_observation(seed)
    model = ExternalRandomFeatureTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        feature_width=FEATURE_WIDTH,
        ridge=1e-4,
        seed=17,
    )
    model.observe(train)
    payload = model.state_payload()
    restored = ExternalRandomFeatureTransitionStatistics.from_payload(payload)
    train_error = float(model.loss(train))
    heldout_error = float(model.loss(heldout))
    report = {
        "schema": "neural-computer.external-random-feature-one-pass-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "train_rows": TRAIN_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "feature_width": FEATURE_WIDTH,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "representation": "opaque_frozen_random_features_v1",
        },
        "gates": {
            "train_prediction_passes": train_error < 0.02,
            "heldout_prediction_passes": heldout_error < 0.02,
            "one_pass_sample_count": int(model.sample_count) == TRAIN_ROWS,
            "raw_evidence_not_stored": "observations" not in payload,
            "exact_persistence": restored.digest() == model.digest(),
        },
        "promoted": train_error < 0.02 and heldout_error < 0.02,
        "metrics": {
            "train_error": train_error,
            "heldout_error": heldout_error,
            "sample_count": int(model.sample_count),
            "state_digest": _digest(model),
        },
        "accounting": {
            "unique_verifier_bits": HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": TRAIN_ROWS + HELDOUT_ROWS,
            "optimizer_updates": 0,
            "streaming_statistics_updates": 1,
            "replayed_examples": 0,
            "old_evidence_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "bounded replay-free nonlinear feature memory; not general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
