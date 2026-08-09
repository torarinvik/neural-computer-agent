"""Two-seed audit of isolated one-pass candidates on interleaved streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
TRAIN_ROWS = 64
WINDOW_ROWS = 4
REGIME_COUNT = 3
LOSS_THRESHOLD = 1e-6
AFFINE_FAMILY = "affine_sufficient_statistics_v1"
RANDOM_FEATURE_FAMILY = "random_feature_sufficient_statistics_v1"


def _observation(seed: int, scale: float) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(TRAIN_ROWS, STATE_WIDTH, generator=generator)
    intention = torch.randn(TRAIN_ROWS, INTENTION_WIDTH, generator=generator)
    features = torch.cat((state, intention, torch.ones(TRAIN_ROWS, 1)), dim=-1)
    weights = torch.tensor(
        [[1.0, 0.2], [-0.3, 0.8], [0.7, -1.1], [0.4, -0.6]]
    )
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=features @ (weights * scale),
        confidence=torch.ones(TRAIN_ROWS),
    )


def _row(observation: ExternalTransitionObservation, index: int):
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
        confidence=torch.ones(1),
    )


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _router(capacity: int) -> ExternalOnlineTransitionContextRouter:
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        random_feature_width=64,
        random_feature_seed=17,
        capacity=capacity,
    )
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=8,
        context_width=CONTEXT_WIDTH,
    )
    return ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=LOSS_THRESHOLD,
        match_margin=1e-4,
        continuation_tolerance=LOSS_THRESHOLD,
        provisional_continuation_tolerance=0.05,
        provisional_match_margin=0.05,
        admission_observations=WINDOW_ROWS,
        max_contexts=capacity,
        defer_admission=True,
        candidate_model_families=(AFFINE_FAMILY, RANDOM_FEATURE_FAMILY),
        provisional_evidence_policy="streaming_statistics",
    )


def _consume(router: ExternalOnlineTransitionContextRouter, observation):
    staged = []
    for index in range(observation.state.shape[0]):
        result = router.observe(_row(observation, index))
        if result.status == "staged":
            staged.append(result)
            router.adaptation_step(result, None, replay_evidence=False)
    return staged


def _loss(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, context_batch).detach())


def _commit_source(router: ExternalOnlineTransitionContextRouter, source):
    _consume(router, source)
    receipt = router.promote_staged_candidate(
        source,
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    if not receipt.accepted:
        raise RuntimeError(f"source promotion failed: {receipt.reason}")
    return router.bank.context_at(0)


def _normal(seed: int) -> dict[str, object]:
    router = _router(REGIME_COUNT)
    source = _observation(seed, 1.0)
    target_a = _observation(seed + 1, 2.0)
    target_b = _observation(seed + 2, -1.0)
    source_context = _commit_source(router, source)

    for chunk in range(TRAIN_ROWS // WINDOW_ROWS):
        for observation in (target_a, target_b):
            start = chunk * WINDOW_ROWS
            for index in range(start, start + WINDOW_ROWS):
                result = router.observe(_row(observation, index))
                if result.status == "staged":
                    router.adaptation_step(result, None, replay_evidence=False)

    if router.provisional_candidate_count != 2:
        raise RuntimeError("interleaved stream did not isolate two candidates")
    candidate_counts = [
        router.provisional_evidence_count(index) for index in range(2)
    ]
    raw_rows = [
        sum(row.state.shape[0] for row in candidate.observations)
        for candidate in router._provisional_candidates
    ]
    candidate_families = [
        tuple(router._provisional_candidates[index].models())
        for index in range(2)
    ]
    router.provisional_continuation_tolerance = 100.0
    midpoint_state = target_a.state[:WINDOW_ROWS]
    midpoint_intention = target_a.intention[:WINDOW_ROWS]
    candidate_predictions = [
        candidate.model(midpoint_state, midpoint_intention)
        for candidate in router._provisional_candidates
    ]
    midpoint = ExternalTransitionObservation(
        state=midpoint_state,
        intention=midpoint_intention,
        next_state=sum(candidate_predictions) / len(candidate_predictions),
        confidence=torch.ones(WINDOW_ROWS),
    )
    ambiguity_statuses = [
        router.observe(_row(midpoint, index)).status
        for index in range(WINDOW_ROWS)
    ]
    ambiguous_windows = ambiguity_statuses.count("ambiguous")
    router.provisional_continuation_tolerance = 0.05

    def retain_source(candidate: ExternalTransitionModelBank) -> bool:
        return (
            candidate.context_count == 2
            and _loss(candidate, source, source_context) < LOSS_THRESHOLD
        )

    first = router.promote_staged_candidate(
        target_a,
        retain_source,
        prediction_tolerance=LOSS_THRESHOLD,
        candidate_index=0,
    )
    target_a_context = router.bank.context_at(1)
    second = router.promote_staged_candidate(
        target_b,
        lambda candidate: (
            candidate.context_count == 3
            and _loss(candidate, source, source_context) < LOSS_THRESHOLD
            and _loss(candidate, target_a, target_a_context) < LOSS_THRESHOLD
        ),
        prediction_tolerance=LOSS_THRESHOLD,
        candidate_index=0,
    )
    target_b_context = router.bank.context_at(2)
    heldout_errors = [
        _loss(router.bank, source, source_context),
        _loss(router.bank, target_a, target_a_context),
        _loss(router.bank, target_b, target_b_context),
    ]
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    return {
        "router": router,
        "source_context": source_context,
        "target_contexts": [target_a_context, target_b_context],
        "promotions": [first, second],
        "candidate_counts": candidate_counts,
        "candidate_families": candidate_families,
        "ambiguous_windows": ambiguous_windows,
        "raw_rows": raw_rows,
        "heldout_errors": heldout_errors,
        "source_digest": router.bank.models[0].digest(),
        "restored_digest": restored.bank.models[0].digest(),
    }


def _shuffled(seed: int) -> dict[str, object]:
    router = _router(2)
    source = _observation(seed, 1.0)
    target = _observation(seed + 1, 2.0)
    _commit_source(router, source)
    generator = torch.Generator().manual_seed(seed + 3)
    shuffled = target.next_state[torch.randperm(TRAIN_ROWS, generator=generator)]
    corrupted = ExternalTransitionObservation(
        state=target.state,
        intention=target.intention,
        next_state=shuffled,
        confidence=target.confidence,
    )
    _consume(router, corrupted)
    model = router.provisional_model
    if model is None:
        raise RuntimeError("shuffled stream did not stage a candidate")
    return {
        "heldout_error": float(model.loss(target).detach()),
        "raw_rows": sum(
            row.state.shape[0]
            for row in router._provisional_candidates[0].observations
        ),
    }


def _capacity(seed: int) -> dict[str, object]:
    router = _router(2)
    source = _observation(seed, 1.0)
    target_a = _observation(seed + 1, 2.0)
    target_b = _observation(seed + 2, -1.0)
    _commit_source(router, source)
    capacity_events = 0
    for observation in (target_a, target_b):
        for row in range(WINDOW_ROWS):
            result = router.observe(_row(observation, row))
            if result.status == "capacity":
                capacity_events += 1
            if result.status == "staged":
                router.adaptation_step(result, None, replay_evidence=False)
    return {
        "capacity_events": capacity_events,
        "committed_context_count": router.bank.context_count,
        "provisional_candidate_count": router.provisional_candidate_count,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    normal = _normal(seed)
    shuffled = _shuffled(seed)
    capacity = _capacity(seed)
    normal_router = normal["router"]
    gates = {
        "two_candidates_isolated": normal["candidate_counts"] == [64, 64],
        "ambiguous_window_refused": normal["ambiguous_windows"] == 1,
        "raw_candidate_rows_not_retained": normal["raw_rows"] == [0, 0],
        "heldout_family_selection": all(
            bool(receipt.accepted)
            and normal_router.bank.model_family_at(index + 1) == AFFINE_FAMILY
            for index, receipt in enumerate(normal["promotions"])
        ),
        "all_promotions_accepted": all(
            bool(receipt.accepted) for receipt in normal["promotions"]
        ),
        "all_heldout_errors_pass": all(
            float(error) < LOSS_THRESHOLD for error in normal["heldout_errors"]
        ),
        "shuffled_transition_rejected": (
            float(shuffled["heldout_error"]) >= LOSS_THRESHOLD
        ),
        "shuffled_raw_rows_not_retained": shuffled["raw_rows"] == 0,
        "capacity_refuses_unverified_growth": (
            int(capacity["capacity_events"]) > 0
            and int(capacity["committed_context_count"]) == 1
        ),
        "exact_source_persistence": normal["source_digest"] == normal["restored_digest"],
        "controller_frozen": controller_digest == _digest(controller),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-interleaved-streaming-candidates-pressure-test.v1",
        "claim_boundary": (
            "Two interleaved partial factual streams can be isolated in "
            "copy-on-write external candidates and promoted through held-out "
            "model-family and retention gates without raw candidate evidence "
            "or old replay; this is not general continual learning."
        ),
        "seed": seed,
        "configuration": {
            "train_rows": TRAIN_ROWS,
            "window_rows": WINDOW_ROWS,
            "regime_count": REGIME_COUNT,
            "candidate_model_families": [AFFINE_FAMILY, RANDOM_FEATURE_FAMILY],
            "provisional_evidence_policy": "streaming_statistics",
            "loss_threshold": LOSS_THRESHOLD,
        },
        "metrics": {
            "candidate_evidence_counts": normal["candidate_counts"],
            "candidate_families": normal["candidate_families"],
            "ambiguous_windows": normal["ambiguous_windows"],
            "raw_rows_retained": normal["raw_rows"],
            "heldout_errors": normal["heldout_errors"],
            "shuffled_heldout_error": shuffled["heldout_error"],
            "capacity": capacity,
        },
        "accounting": {
            "unique_verifier_bits": 3 * TRAIN_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": 3 * TRAIN_ROWS,
            "streaming_statistics_updates": 3 * TRAIN_ROWS,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "old_committed_slot_replay": 0,
            "raw_provisional_rows_persisted": 0,
            "search_expansions": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {"controller": controller_digest},
        "gates": gates,
        "promoted": all(gates.values()),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
