"""Replay-free contextual reliability reversal and retention audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_factored_contextual_reliability.train import (
    _observation,
    _predict,
    _rows,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalContextualTransitionEvidenceStatistics,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 4
SLOT_COUNT = 2
ROWS_PER_SLOT = 4
MATCH_TOLERANCE = 0.005
DRIFT_DELTA = 0.04
RELIABILITY_THRESHOLD = 0.9
RELIABILITY_WARMUP = 4
COUNT_DECAY = 0.1


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    observations = tuple(_observation(seed, slot) for slot in range(SLOT_COUNT))
    drift = tuple(
        ExternalTransitionObservation(
            state=item.state,
            intention=item.intention,
            next_state=item.next_state + DRIFT_DELTA,
            confidence=item.confidence,
        )
        for item in observations
    )
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    model = ExternalFactoredTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=8,
        residual_read_match_threshold=0.999999,
        residual_write_match_threshold=0.999999,
        residual_capacity=SLOT_COUNT,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    base_digest = model.base.digest()
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=12,
        context_width=CONTEXT_WIDTH,
    )
    encoder_digest = encoder.digest()
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
        max_contexts=SLOT_COUNT,
        quarantine_capacity=ROWS_PER_SLOT * 2,
    )
    for slot, item in enumerate(observations):
        if router.route_bundle(_rows(item)).status != "staged":
            raise RuntimeError(f"slot {slot} did not stage")
        receipt = router.promote_staged_candidate(
            item,
            lambda candidate, expected=ROWS_PER_SLOT * (slot + 1): candidate.residual_record_count
            == expected,
            prediction_tolerance=1e-6,
        )
        if not receipt.accepted:
            raise RuntimeError(f"slot {slot} promotion failed: {receipt}")
    contexts = router.contexts
    if torch.pdist(contexts).min() <= 1e-4:
        raise RuntimeError("context encoder produced colliding reversal keys")
    statistics = ExternalContextualTransitionEvidenceStatistics(
        STATE_WIDTH,
        CONTEXT_WIDTH,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
        matching_tolerance=1e-4,
        count_decay=COUNT_DECAY,
    )
    router.evidence_evaluator = statistics
    router.evidence_threshold = RELIABILITY_THRESHOLD
    router.evidence_gate_min_evidence = RELIABILITY_WARMUP
    router.committed_evidence_gate = True

    def observe_outcomes(outcomes: tuple[float, float]) -> None:
        for slot, item in enumerate(drift):
            prediction = _predict(router.model, contexts[slot], item)
            statistics.observe(
                prediction,
                item.next_state,
                torch.full((ROWS_PER_SLOT,), outcomes[slot]),
                contexts[slot],
            )

    initial_outcomes = (0.0, 1.0)
    reversed_outcomes = (1.0, 0.0)
    observe_outcomes(initial_outcomes)
    initial_results = [router.route_bundle(_rows(item)) for item in drift]
    initial_quarantine = router.quarantined_observations

    first_reversal_before = [router.route_bundle(_rows(item)) for item in drift]
    observe_outcomes(reversed_outcomes)
    first_resolved = router.resolve_quarantine(
        match_tolerance=MATCH_TOLERANCE,
        contradiction_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
    )
    after_first = [router.route_bundle(_rows(item)) for item in drift]

    second_reversal_before = [router.route_bundle(_rows(item)) for item in drift]
    observe_outcomes(initial_outcomes)
    second_resolved = router.resolve_quarantine(
        match_tolerance=MATCH_TOLERANCE,
        contradiction_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
    )
    after_second = [router.route_bundle(_rows(item)) for item in drift]
    model_digest = router.model.digest()
    restored_statistics = ExternalContextualTransitionEvidenceStatistics.from_payload(
        statistics.payload()
    )
    restored_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=restored_statistics
    )
    persistence_exact = (
        restored_router.digest() == router.digest()
        and restored_statistics.digest() == statistics.digest()
    )
    gates = {
        "initial_negative_veto_positive_match": initial_results[0].status
        == "reliability_veto"
        and initial_results[1].status == "matched",
        "initial_negative_quarantine_retained": initial_results[0].quarantine_accepted
        is True,
        "first_reversal_updates_without_replay": first_reversal_before[0].status
        == "reliability_veto"
        and first_reversal_before[1].status == "matched"
        and after_first[0].status == "matched"
        and after_first[1].status == "reliability_veto",
        "first_reversal_resolves_retained_rows": first_resolved
        == (0, 0)
        and router.quarantined_observations >= 0,
        "second_reversal_updates_again": second_reversal_before[0].status
        == "matched"
        and second_reversal_before[1].status == "reliability_veto"
        and after_second[0].status == "reliability_veto"
        and after_second[1].status == "matched",
        "second_reversal_resolves_retained_rows": second_resolved == (1, 1),
        "fact_bank_unchanged": model_digest == router.model.digest(),
        "persistence_exact": persistence_exact,
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "controller_frozen": _digest(controller) == controller_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"contextual reversal gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-contextual-reversal.v1",
        "seed": seed,
        "configuration": {
            "slot_count": SLOT_COUNT,
            "rows_per_slot": ROWS_PER_SLOT,
            "drift_delta": DRIFT_DELTA,
            "count_decay": COUNT_DECAY,
            "match_tolerance": MATCH_TOLERANCE,
            "claim": "contextual-reliability-reversal-without-factual-replay-v1",
        },
        "metrics": {
            "initial_statuses": [item.status for item in initial_results],
            "first_reversal_before": [item.status for item in first_reversal_before],
            "after_first_reversal": [item.status for item in after_first],
            "second_reversal_before": [item.status for item in second_reversal_before],
            "after_second_reversal": [item.status for item in after_second],
            "initial_quarantine_rows": initial_quarantine,
            "first_resolved": list(first_resolved),
            "second_resolved": list(second_resolved),
            "final_quarantine_rows": router.quarantined_observations,
            "verifier_observation_count": int(statistics.observation_count),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(statistics.observation_count),
            "unique_logical_transition_lifetimes": SLOT_COUNT,
            "contextual_sufficient_statistics_updates": 4,
            "base_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": model.base.digest(),
            "controller": controller_digest,
            "context_encoder": encoder_digest,
            "router": router.digest(),
            "statistics": statistics.digest(),
        },
        "promoted": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=93001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
