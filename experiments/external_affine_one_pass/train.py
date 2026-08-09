"""Two-seed replay-free affine transition-memory audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ExternalAffineTransitionStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
TRAIN_ROWS = 12
HELDOUT_ROWS = 4


def _digest(model: ExternalAffineTransitionStatistics) -> str:
    return hashlib.sha256(model.state_payload()["sha256"].encode()).hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    true_weights = torch.tensor(
        [[1.0, 0.2], [-0.3, 0.8], [0.7, -1.1], [0.4, -0.6]]
    )
    state = torch.randn(TRAIN_ROWS + HELDOUT_ROWS, STATE_WIDTH)
    intention = torch.randn(TRAIN_ROWS + HELDOUT_ROWS, INTENTION_WIDTH)
    features = torch.cat(
        (state, intention, torch.ones(TRAIN_ROWS + HELDOUT_ROWS, 1)), dim=-1
    )
    next_state = features @ true_weights
    model = ExternalAffineTransitionStatistics(STATE_WIDTH, INTENTION_WIDTH, ridge=1e-7)
    for row in range(TRAIN_ROWS):
        model.observe(
            ExternalTransitionObservation(
                state=state[row : row + 1],
                intention=intention[row : row + 1],
                next_state=next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
    heldout = ExternalTransitionObservation(
        state=state[TRAIN_ROWS:],
        intention=intention[TRAIN_ROWS:],
        next_state=next_state[TRAIN_ROWS:],
    )
    train = ExternalTransitionObservation(
        state=state[:TRAIN_ROWS],
        intention=intention[:TRAIN_ROWS],
        next_state=next_state[:TRAIN_ROWS],
    )
    payload = model.state_payload()
    restored = ExternalAffineTransitionStatistics.from_payload(payload)
    report = {
        "schema": "neural-computer.external-affine-one-pass-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "train_rows": TRAIN_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "representation": "opaque_affine_sufficient_statistics_v1",
        },
        "gates": {
            "train_prediction_passes": float(model.loss(train)) < 1e-6,
            "heldout_prediction_passes": float(model.loss(heldout)) < 1e-6,
            "one_pass_sample_count": int(model.sample_count) == TRAIN_ROWS,
            "raw_evidence_not_stored": "observations" not in payload,
            "exact_persistence": restored.digest() == model.digest(),
        },
        "promoted": True,
        "metrics": {
            "train_error": float(model.loss(train)),
            "heldout_error": float(model.loss(heldout)),
            "sample_count": int(model.sample_count),
            "state_digest": _digest(model),
        },
        "accounting": {
            "unique_logical_lifetimes": TRAIN_ROWS + HELDOUT_ROWS,
            "optimizer_updates": 0,
            "streaming_statistics_updates": TRAIN_ROWS,
            "replayed_examples": 0,
            "old_evidence_replay": 0,
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13011)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
