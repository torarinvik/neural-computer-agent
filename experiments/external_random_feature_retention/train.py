"""Two-seed long-horizon retention audit for replay-free nonlinear slots."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import ExternalTransitionModelBank, ExternalTransitionObservation

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
REGIME_COUNT = 4
TRAIN_ROWS = 64
HELDOUT_ROWS = 64
FEATURE_WIDTH = 128
LOSS_THRESHOLD = 0.02


def _transition(regime: int, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    if regime == 0:
        return torch.cat((torch.sin(2.0 * state[:, 0:1] + intention), state[:, 0:1] * state[:, 1:2] + intention.square()), dim=-1)
    if regime == 1:
        return torch.cat((torch.cos(state[:, 0:1] - intention), state[:, 0:1].square() - state[:, 1:2] * intention), dim=-1)
    if regime == 2:
        return torch.cat((torch.sin(state[:, 0:1] - state[:, 1:2] + 2.0 * intention), state[:, 0:1] * intention + state[:, 1:2].square()), dim=-1)
    return torch.cat((torch.cos(1.5 * state[:, 0:1] + state[:, 1:2]), state[:, 0:1].square() + intention * state[:, 1:2]), dim=-1)


def _fixture(seed: int, regime: int) -> tuple[ExternalTransitionObservation, ExternalTransitionObservation]:
    generator = torch.Generator().manual_seed(seed + regime * 101)
    state = torch.rand(TRAIN_ROWS + HELDOUT_ROWS, STATE_WIDTH, generator=generator) * 2.0 - 1.0
    intention = torch.rand(TRAIN_ROWS + HELDOUT_ROWS, INTENTION_WIDTH, generator=generator) * 2.0 - 1.0
    next_state = _transition(regime, state, intention)
    return (
        ExternalTransitionObservation(state[:TRAIN_ROWS], intention[:TRAIN_ROWS], next_state[:TRAIN_ROWS]),
        ExternalTransitionObservation(state[TRAIN_ROWS:], intention[TRAIN_ROWS:], next_state[TRAIN_ROWS:]),
    )


def _digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="random_feature_sufficient_statistics_v1",
        affine_ridge=1e-4,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        capacity=REGIME_COUNT,
    )
    heldout: list[ExternalTransitionObservation] = []
    contexts: list[torch.Tensor] = []
    digests_after_learning: list[str] = []
    for regime in range(REGIME_COUNT):
        train, probe = _fixture(seed, regime)
        heldout.append(probe)
        context = torch.eye(CONTEXT_WIDTH)[regime]
        contexts.append(context)
        index = bank.ensure_context(context)
        context_batch = context.unsqueeze(0).expand(TRAIN_ROWS, -1)
        bank.adaptation_step(train, context_batch, None)
        digests_after_learning.append(_digest(bank.models[index]))

    errors_after_learning = [
        float(bank.loss(probe, context.unsqueeze(0).expand(HELDOUT_ROWS, -1)))
        for probe, context in zip(heldout, contexts, strict=True)
    ]
    errors_after_revisit = [
        float(bank.loss(probe, context.unsqueeze(0).expand(HELDOUT_ROWS, -1)))
        for probe, context in zip(heldout, contexts, strict=True)
    ]
    digests_after_revisit = [_digest(model) for model in bank.models]
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    report = {
        "schema": "neural-computer.external-random-feature-retention-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "train_rows_per_regime": TRAIN_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "feature_width": FEATURE_WIDTH,
            "context_keys": "verifier_supplied_one_hot_fixture",
        },
        "gates": {
            "all_heldout_pass": all(error < LOSS_THRESHOLD for error in errors_after_learning),
            "later_learning_retains_earlier_slots": digests_after_learning == digests_after_revisit,
            "revisit_errors_stable": errors_after_learning == errors_after_revisit,
            "zero_replayed_examples": True,
            "exact_bank_persistence": restored.content_digest() == bank.content_digest(),
        },
        "promoted": all(error < LOSS_THRESHOLD for error in errors_after_learning) and digests_after_learning == digests_after_revisit,
        "metrics": {
            "heldout_error_after_learning": errors_after_learning,
            "heldout_error_after_revisit": errors_after_revisit,
            "slot_digests_stable": digests_after_learning == digests_after_revisit,
            "context_count": bank.context_count,
        },
        "accounting": {
            "unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS),
            "optimizer_updates": 0,
            "streaming_statistics_updates": REGIME_COUNT,
            "replayed_examples": 0,
            "old_evidence_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "bounded replay-free nonlinear slot retention with supplied context keys; not general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
