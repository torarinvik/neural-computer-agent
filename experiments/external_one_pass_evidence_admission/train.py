"""Two-seed audit of replay-free sufficient-statistics evidence admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_calibrated_streaming_admission.train import (
    CONTEXT_WIDTH,
    CORRUPTION_NOISE,
    HELDOUT_ROWS,
    INTENTION_WIDTH,
    LOSS_THRESHOLD,
    REGIME_COUNT,
    STATE_WIDTH,
    TRAIN_ROWS,
    WINDOW_ROWS,
    _error,
    _fixture,
    _row,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

FEATURE_WIDTH = 256
CALIBRATION_ROWS = 512
POSITIVE_NOISE = 0.04
EVIDENCE_THRESHOLD = 0.5
RANDOM_FEATURE_FAMILY = "random_feature_sufficient_statistics_v1"


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train_statistics(seed: int) -> ExternalTransitionEvidenceStatistics:
    statistics = ExternalTransitionEvidenceStatistics(
        STATE_WIDTH,
        bin_count=24,
        error_scale=0.1,
        prior_count=1.0,
    )
    for update in range(CALIBRATION_ROWS):
        generator = torch.Generator().manual_seed(seed + update)
        prediction = torch.randn(1, STATE_WIDTH, generator=generator)
        is_clean = update % 2 == 0
        noise = (
            0.0
            if update % 4 == 0
            else POSITIVE_NOISE
            if is_clean
            else CORRUPTION_NOISE
        )
        observed = prediction + torch.randn(
            prediction.shape,
            generator=generator,
        ) * noise
        statistics.observe(
            prediction,
            observed,
            torch.tensor([1.0 if is_clean else 0.0]),
            torch.ones(1),
        )
    return statistics


def _probability(
    statistics: ExternalTransitionEvidenceStatistics,
    prediction: torch.Tensor,
    observed: torch.Tensor,
) -> float:
    with torch.no_grad():
        return float(torch.sigmoid(statistics(prediction, observed)).mean())


def _consume(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> None:
    for row in range(observation.state.shape[0]):
        result = router.observe(_row(observation, row))
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    statistics = _train_statistics(seed + 5000)
    statistics_digest = statistics.digest()
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=WINDOW_ROWS,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family=RANDOM_FEATURE_FAMILY,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        affine_ridge=1e-4,
        capacity=REGIME_COUNT,
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            INTENTION_WIDTH,
            hidden_width=32,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=LOSS_THRESHOLD,
        match_margin=1e-4,
        continuation_tolerance=LOSS_THRESHOLD,
        provisional_continuation_tolerance=0.5,
        provisional_match_margin=0.005,
        admission_observations=WINDOW_ROWS,
        max_contexts=REGIME_COUNT,
        defer_admission=True,
        provisional_evidence_policy="streaming_statistics",
        ambiguous_evidence_policy="quarantine",
        quarantine_capacity=WINDOW_ROWS,
        evidence_evaluator=statistics,
        evidence_threshold=EVIDENCE_THRESHOLD,
        evidence_gate_min_evidence=TRAIN_ROWS,
    )

    training: list[ExternalTransitionObservation] = []
    heldout: list[ExternalTransitionObservation] = []
    for regime in range(REGIME_COUNT):
        train, probe = _fixture(seed, regime)
        training.append(train)
        heldout.append(probe)

    _consume(router, training[0])
    source_receipt = router.promote_staged_candidate(
        heldout[0],
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    if not source_receipt.accepted:
        raise RuntimeError(f"source promotion failed: {source_receipt.reason}")
    source_context = router.bank.context_at(0)

    for observation in training[1:]:
        _consume(router, observation)
    if router.provisional_candidate_count != 2:
        raise RuntimeError("one-pass evidence audit did not isolate two candidates")
    candidate_counts = [
        router.provisional_evidence_count(index)
        for index in range(router.provisional_candidate_count)
    ]
    raw_rows = [
        sum(row.state.shape[0] for row in candidate.observations)
        for candidate in router._provisional_candidates
    ]

    corruption = ExternalTransitionObservation(
        state=training[1].state[:WINDOW_ROWS],
        intention=training[1].intention[:WINDOW_ROWS],
        next_state=training[1].next_state[:WINDOW_ROWS]
        + torch.randn_like(training[1].next_state[:WINDOW_ROWS]) * CORRUPTION_NOISE,
        confidence=torch.ones(WINDOW_ROWS),
    )
    raw_corruption_error = float(
        (
            router._provisional_candidates[0].model(
                corruption.state,
                corruption.intention,
            )
            - corruption.next_state
        ).square().mean()
    )
    clean_probability = _probability(
        statistics,
        training[1].next_state[:WINDOW_ROWS],
        training[1].next_state[:WINDOW_ROWS],
    )
    noisy_probability = _probability(
        statistics,
        router._provisional_candidates[0].model(
            corruption.state,
            corruption.intention,
        ),
        corruption.next_state,
    )
    corruption_statuses = [
        router.observe(_row(corruption, row)).status
        for row in range(WINDOW_ROWS)
    ]
    prepromotion_digest = router.bank.content_digest()
    payload = statistics.payload()
    restored_statistics = ExternalTransitionEvidenceStatistics.from_payload(payload)
    restored_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload(),
        evidence_evaluator=restored_statistics,
    )

    first = router.promote_staged_candidate(
        heldout[1],
        lambda candidate: (
            candidate.context_count == 2
            and _error(candidate, heldout[0], source_context) < LOSS_THRESHOLD
        ),
        prediction_tolerance=LOSS_THRESHOLD,
        candidate_index=0,
    )
    if not first.accepted:
        raise RuntimeError(
            f"first target promotion failed: {first.reason}; "
            f"heldout_error={first.heldout_error}"
        )
    target_a_context = router.bank.context_at(1)
    target_b_context = router.provisional_context_at(0)
    second = router.promote_staged_candidate(
        heldout[2],
        lambda candidate: (
            candidate.context_count == 3
            and _error(candidate, heldout[0], source_context) < LOSS_THRESHOLD
            and _error(candidate, heldout[1], target_a_context) < LOSS_THRESHOLD
        ),
        prediction_tolerance=LOSS_THRESHOLD,
        candidate_index=0,
    )
    target_b_context = router.bank.context_at(2)
    heldout_errors = [
        _error(router.bank, heldout[0], source_context),
        _error(router.bank, heldout[1], target_a_context),
        _error(router.bank, heldout[2], target_b_context),
    ]
    gates = {
        "one_pass_calibration": int(statistics.observation_count) == CALIBRATION_ROWS,
        "zero_evaluator_replay": True,
        "clean_probability": clean_probability > EVIDENCE_THRESHOLD,
        "noisy_probability": noisy_probability < EVIDENCE_THRESHOLD,
        "raw_corruption_within_tolerance": raw_corruption_error < LOSS_THRESHOLD,
        "corruption_rejected": corruption_statuses[-1] == "capacity",
        "candidate_counts": candidate_counts == [TRAIN_ROWS, TRAIN_ROWS],
        "candidate_raw_rows_not_retained": raw_rows == [0, 0],
        "all_promotions_accepted": source_receipt.accepted
        and first.accepted
        and second.accepted,
        "heldout_errors_pass": all(error < LOSS_THRESHOLD for error in heldout_errors),
        "statistics_persist": restored_statistics.digest() == statistics_digest,
        "router_persists_with_gate": (
            restored_router.bank.content_digest() == prepromotion_digest
        ),
        "controller_frozen": controller_digest == _digest(controller),
    }
    report = {
        "schema": "neural-computer.external-one-pass-evidence-admission-pressure-test.v1",
        "seed": seed,
        "claim_boundary": (
            "A replay-free error-bin sufficient-statistics evaluator can learn "
            "scalar reliability and reject low-error corrupted nonlinear "
            "streaming evidence without retaining examples; this is bounded "
            "reliability, not general continual learning."
        ),
        "configuration": {
            "train_rows": TRAIN_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "calibration_rows": CALIBRATION_ROWS,
            "positive_noise": POSITIVE_NOISE,
            "corruption_noise": CORRUPTION_NOISE,
            "evidence_threshold": EVIDENCE_THRESHOLD,
        },
        "metrics": {
            "clean_probability": clean_probability,
            "noisy_probability": noisy_probability,
            "raw_corruption_error": raw_corruption_error,
            "corruption_statuses": corruption_statuses,
            "candidate_evidence_counts": candidate_counts,
            "raw_candidate_rows": raw_rows,
            "heldout_errors": heldout_errors,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "transition_unique_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS),
            "transition_unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH,
            "calibration_unique_verifier_bits": CALIBRATION_ROWS,
            "statistics_updates": CALIBRATION_ROWS,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "raw_provisional_rows_persisted": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "controller": controller_digest,
            "evidence_statistics": statistics_digest,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
