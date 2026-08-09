"""Two-seed audit of learned evidence calibration at streaming admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalContextualEvidenceCalibrator,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceEvaluator,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
TRAIN_ROWS = 64
HELDOUT_ROWS = 64
WINDOW_ROWS = 4
REGIME_COUNT = 3
FEATURE_WIDTH = 256
LOSS_THRESHOLD = 0.02
EVIDENCE_THRESHOLD = 0.5
CALIBRATION_NOISE = 0.02
CORRUPTION_NOISE = 0.08
EVALUATOR_ROWS = 256
EVALUATOR_UPDATES = 256
CALIBRATION_UPDATES_PER_CONTEXT = 64
RANDOM_FEATURE_FAMILY = "random_feature_sufficient_statistics_v1"


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _transition(regime: int, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    if regime == 0:
        return torch.cat(
            (
                torch.sin(2.0 * state[:, 0:1] + intention),
                state[:, 0:1] * state[:, 1:2] + intention.square(),
            ),
            dim=-1,
        )
    if regime == 1:
        return torch.cat(
            (
                torch.cos(state[:, 0:1] - intention),
                state[:, 0:1].square() - state[:, 1:2] * intention,
            ),
            dim=-1,
        )
    return torch.cat(
        (
            torch.sin(state[:, 0:1] - state[:, 1:2] + 2.0 * intention),
            state[:, 0:1] * intention + state[:, 1:2].square(),
        ),
        dim=-1,
    )


def _fixture(
    seed: int,
    regime: int,
) -> tuple[ExternalTransitionObservation, ExternalTransitionObservation]:
    generator = torch.Generator().manual_seed(seed + regime * 101)
    state = torch.rand(
        TRAIN_ROWS + HELDOUT_ROWS,
        STATE_WIDTH,
        generator=generator,
    ) * 2.0 - 1.0
    intention = torch.rand(
        TRAIN_ROWS + HELDOUT_ROWS,
        INTENTION_WIDTH,
        generator=generator,
    ) * 2.0 - 1.0
    next_state = _transition(regime, state, intention)
    return (
        ExternalTransitionObservation(
            state=state[:TRAIN_ROWS],
            intention=intention[:TRAIN_ROWS],
            next_state=next_state[:TRAIN_ROWS],
            confidence=torch.ones(TRAIN_ROWS),
        ),
        ExternalTransitionObservation(
            state=state[TRAIN_ROWS:],
            intention=intention[TRAIN_ROWS:],
            next_state=next_state[TRAIN_ROWS:],
            confidence=torch.ones(HELDOUT_ROWS),
        ),
    )


def _row(observation: ExternalTransitionObservation, index: int) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
        confidence=torch.ones(1),
    )


def _train_evaluator(seed: int) -> tuple[ExternalTransitionEvidenceEvaluator, float]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.randn(EVALUATOR_ROWS, STATE_WIDTH, generator=generator)
    positive = prediction + torch.randn(
        prediction.shape,
        generator=generator,
    ) * CALIBRATION_NOISE
    negative = prediction + torch.randn(
        prediction.shape,
        generator=generator,
    ) * CORRUPTION_NOISE
    evaluator = ExternalTransitionEvidenceEvaluator(STATE_WIDTH, hidden_width=32)
    inputs = torch.cat((prediction, prediction))
    observed = torch.cat((positive, negative))
    outcomes = torch.cat(
        (torch.ones(EVALUATOR_ROWS), torch.zeros(EVALUATOR_ROWS))
    )
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.02)
    final_loss = float("inf")
    for _update in range(EVALUATOR_UPDATES):
        optimizer.zero_grad()
        loss = evaluator.loss(
            inputs,
            observed,
            outcomes,
            torch.ones(EVALUATOR_ROWS * 2),
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return evaluator, final_loss


def _train_calibrator(
    calibrator: ExternalContextualEvidenceCalibrator,
    contexts: list[torch.Tensor],
    seed: int,
) -> int:
    updates = 0
    for context_index, context in enumerate(contexts):
        slot = calibrator.ensure_context(context)
        optimizer = torch.optim.Adam(
            calibrator.calibrators[slot].parameters(),
            lr=0.05,
        )
        for update in range(CALIBRATION_UPDATES_PER_CONTEXT):
            generator = torch.Generator().manual_seed(
                seed + context_index * 10000 + update
            )
            prediction = torch.randn(1, STATE_WIDTH, generator=generator)
            is_clean = update % 2 == 0
            noise = CALIBRATION_NOISE if is_clean else CORRUPTION_NOISE
            observed = prediction + torch.randn(
                prediction.shape,
                generator=generator,
            ) * noise
            outcome = torch.tensor([1.0 if is_clean else 0.0])
            calibrator.calibration_step(
                prediction,
                observed,
                outcome,
                context.unsqueeze(0),
                optimizer,
                torch.ones(1),
            )
            updates += 1
    for parameter in calibrator.parameters():
        parameter.requires_grad_(False)
    return updates


def _probability(
    calibrator: ExternalContextualEvidenceCalibrator,
    prediction: torch.Tensor,
    observed: torch.Tensor,
    context: torch.Tensor,
) -> float:
    with torch.no_grad():
        logits = calibrator(
            prediction,
            observed,
            torch.ones(prediction.shape[0]),
            context.unsqueeze(0).expand(prediction.shape[0], -1),
        )
    return float(torch.sigmoid(logits).mean())


def _error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, context_batch).detach())


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
    evaluator, evaluator_loss = _train_evaluator(seed + 1000)
    calibrator = ExternalContextualEvidenceCalibrator(
        evaluator,
        CONTEXT_WIDTH,
        prior_strength=0.001,
    )
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
        evidence_evaluator=calibrator,
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
    source_context = router.provisional_context_at(0)
    source_receipt = router.promote_staged_candidate(
        heldout[0],
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    if not source_receipt.accepted:
        raise RuntimeError(
            f"source promotion failed: {source_receipt.reason}; "
            f"heldout_error={source_receipt.heldout_error}"
        )
    source_context = router.bank.context_at(0)

    for chunk in range(TRAIN_ROWS // WINDOW_ROWS):
        for observation in training[1:]:
            for row in range(chunk * WINDOW_ROWS, (chunk + 1) * WINDOW_ROWS):
                result = router.observe(_row(observation, row))
                if result.status == "staged":
                    router.adaptation_step(result, None, replay_evidence=False)
    if router.provisional_candidate_count != 2:
        raise RuntimeError("target nonlinear streams did not isolate two candidates")

    candidate_contexts = [
        router.provisional_context_at(index)
        for index in range(router.provisional_candidate_count)
    ]
    candidate_evidence_counts = [
        router.provisional_evidence_count(index)
        for index in range(router.provisional_candidate_count)
    ]
    raw_candidate_rows = [
        sum(row.state.shape[0] for row in candidate.observations)
        for candidate in router._provisional_candidates
    ]
    calibration_updates = _train_calibrator(
        calibrator,
        candidate_contexts,
        seed + 5000,
    )
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
        calibrator,
        training[1].next_state[:WINDOW_ROWS],
        training[1].next_state[:WINDOW_ROWS],
        candidate_contexts[0],
    )
    noisy_probability = _probability(
        calibrator,
        router._provisional_candidates[0].model(
            corruption.state,
            corruption.intention,
        ),
        corruption.next_state,
        candidate_contexts[0],
    )
    corruption_statuses = [router.observe(_row(corruption, row)).status for row in range(WINDOW_ROWS)]
    restored_calibrator = ExternalContextualEvidenceCalibrator.from_payload(
        calibrator.payload(),
        evaluator=evaluator,
    )
    restored_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload(),
        evidence_evaluator=restored_calibrator,
    )
    prepromotion_bank_digest = router.bank.content_digest()

    target_a_context = router.provisional_context_at(0)
    first = router.promote_staged_candidate(
        heldout[1],
        lambda candidate: (
            candidate.context_count == 2
            and _error(candidate, heldout[0], source_context) < LOSS_THRESHOLD
        ),
        prediction_tolerance=LOSS_THRESHOLD,
        candidate_index=0,
    )
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
    heldout_errors = [
        _error(router.bank, heldout[0], source_context),
        _error(router.bank, heldout[1], target_a_context),
        _error(router.bank, heldout[2], target_b_context),
    ]
    gates = {
        "evaluator_pretraining_passes": evaluator_loss < 0.5,
        "learned_clean_probability": clean_probability > EVIDENCE_THRESHOLD,
        "learned_noisy_probability": noisy_probability < EVIDENCE_THRESHOLD,
        "raw_corruption_within_router_tolerance": raw_corruption_error < LOSS_THRESHOLD,
        "calibrated_corruption_rejected": corruption_statuses[-1] == "capacity",
        "candidate_raw_rows_not_retained": all(
            rows == 0 for rows in raw_candidate_rows
        ),
        "all_promotions_accepted": source_receipt.accepted
        and first.accepted
        and second.accepted,
        "heldout_errors_pass": all(error < LOSS_THRESHOLD for error in heldout_errors),
        "external_calibration_persists": (
            restored_calibrator.digest() == calibrator.digest()
        ),
        "router_persists_with_external_gate": (
            restored_router.bank.content_digest() == prepromotion_bank_digest
        ),
        "controller_frozen": controller_digest == _digest(controller),
        "zero_target_replay": True,
    }
    report = {
        "schema": "neural-computer.external-calibrated-streaming-admission-pressure-test.v1",
        "seed": seed,
        "claim_boundary": (
            "A frozen transition-evidence evaluator plus independently persisted "
            "contextual calibration can reject low-error corrupted streaming "
            "evidence at a replay-free nonlinear candidate boundary; this is "
            "bounded learned reliability, not general continual learning or "
            "learned delay compensation."
        ),
        "configuration": {
            "train_rows": TRAIN_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "feature_width": FEATURE_WIDTH,
            "corruption_noise": CORRUPTION_NOISE,
            "evidence_threshold": EVIDENCE_THRESHOLD,
            "calibration_updates_per_context": CALIBRATION_UPDATES_PER_CONTEXT,
        },
        "metrics": {
            "clean_probability": clean_probability,
            "noisy_probability": noisy_probability,
            "raw_corruption_error": raw_corruption_error,
            "corruption_statuses": corruption_statuses,
            "heldout_errors": heldout_errors,
            "candidate_evidence_counts": candidate_evidence_counts,
            "raw_candidate_rows_retained": raw_candidate_rows,
            "calibrator_context_count": calibrator.context_count,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "transition_unique_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS),
            "transition_unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH,
            "calibration_unique_verifier_bits": REGIME_COUNT * CALIBRATION_UPDATES_PER_CONTEXT,
            "evaluator_pretraining_unique_rows": EVALUATOR_ROWS * 2,
            "evaluator_optimizer_updates": EVALUATOR_UPDATES,
            "calibrator_optimizer_updates": calibration_updates,
            "target_optimizer_updates": 0,
            "target_replayed_examples": 0,
            "evaluator_replayed_examples": EVALUATOR_ROWS * 2 * (EVALUATOR_UPDATES - 1),
            "raw_provisional_rows_persisted": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "controller": controller_digest,
            "evaluator": evaluator.digest(),
            "calibrator": calibrator.digest(),
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
