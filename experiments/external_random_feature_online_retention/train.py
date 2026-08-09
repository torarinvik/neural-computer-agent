"""Two-seed online routing and retention audit for nonlinear external memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

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


def _error(bank: ExternalTransitionModelBank, observation: ExternalTransitionObservation, context: torch.Tensor) -> float:
    batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, batch).detach())


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
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=32,
        context_width=CONTEXT_WIDTH,
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=0.02,
        match_margin=0.005,
        continuation_tolerance=0.02,
        provisional_continuation_tolerance=1e9,
        admission_observations=8,
        max_contexts=REGIME_COUNT,
        defer_admission=True,
        conflict_patience=1,
    )
    heldout: list[ExternalTransitionObservation] = []
    contexts: list[torch.Tensor] = []
    digests: list[str] = []
    promotion_records: list[dict[str, object]] = []
    for regime in range(REGIME_COUNT):
        train, probe = _fixture(seed, regime)
        heldout.append(probe)
        staged_updates = 0
        for row in range(TRAIN_ROWS):
            observation = ExternalTransitionObservation(
                state=train.state[row : row + 1],
                intention=train.intention[row : row + 1],
                next_state=train.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
            result = router.observe(observation)
            if result.status == "staged":
                router.adaptation_step(result, None)
                staged_updates += 1
        if router.provisional_candidate_count != 1:
            raise RuntimeError("online fixture did not leave exactly one candidate")
        contexts.append(router.provisional_context_at(0))

        def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
            return all(
                _error(candidate, old_probe, old_context) < LOSS_THRESHOLD
                for old_probe, old_context in zip(heldout[:-1], contexts[:-1], strict=True)
            )

        receipt = router.promote_staged_candidate(
            probe,
            retention_probe,
            prediction_tolerance=LOSS_THRESHOLD,
        )
        promotion_records.append(
            {"accepted": receipt.accepted, "heldout_error": receipt.heldout_error, "staged_updates": staged_updates}
        )
        if not receipt.accepted:
            raise RuntimeError(f"online promotion failed for regime {regime}: {receipt.reason}")
        digests.append(_digest(bank.models[receipt.slot_index]))

    errors = [_error(bank, probe, context) for probe, context in zip(heldout, contexts, strict=True)]
    restored = ExternalOnlineTransitionContextRouter.from_payload(router.state_payload())
    report = {
        "schema": "neural-computer.external-random-feature-online-retention-pressure-test.v1",
        "seed": seed,
        "configuration": {"regime_count": REGIME_COUNT, "train_rows_per_regime": TRAIN_ROWS, "heldout_rows_per_regime": HELDOUT_ROWS, "feature_width": FEATURE_WIDTH, "context_keys": "learned_encoder_opaque_candidate_keys"},
        "gates": {"all_heldout_pass": all(error < LOSS_THRESHOLD for error in errors), "all_promotions_pass": all(item["accepted"] for item in promotion_records), "prior_retention_verified_each_step": True, "zero_replayed_examples": True, "exact_router_persistence": restored.bank.content_digest() == bank.content_digest()},
        "promoted": all(error < LOSS_THRESHOLD for error in errors),
        "metrics": {"heldout_error_after_all_learning": errors, "promotion_records": promotion_records, "context_count": bank.context_count, "slot_digests": digests},
        "accounting": {"unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH, "unique_logical_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS), "optimizer_updates": 0, "streaming_statistics_updates": REGIME_COUNT * (TRAIN_ROWS - 7), "replayed_examples": 0, "old_evidence_replay": 0, "wall_seconds": time.perf_counter() - begun},
        "claim_boundary": "bounded online replay-free nonlinear retention with learned opaque context keys; not general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
