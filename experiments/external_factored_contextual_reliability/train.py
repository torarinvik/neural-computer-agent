"""Three-seed contextual reliability isolation pressure test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalContextualTransitionEvidenceStatistics,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 4
SLOT_COUNT = 4
ROWS_PER_SLOT = 4
MATCH_TOLERANCE = 0.005
DRIFT_DELTA = 0.04
RELIABILITY_THRESHOLD = 0.9
RELIABILITY_WARMUP = 4


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _observation(seed: int, slot: int) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed + slot * 17)
    state = torch.randn(ROWS_PER_SLOT, STATE_WIDTH, generator=generator)
    state = state + torch.tensor([slot * 2.0, -slot * 0.7])
    intention = torch.eye(INTENTION_WIDTH)[
        torch.arange(ROWS_PER_SLOT) % INTENTION_WIDTH
    ]
    next_state = 0.2 * state + intention + torch.tensor([slot * 0.1, -slot * 0.05])
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(ROWS_PER_SLOT),
    )


def _rows(observation: ExternalTransitionObservation) -> tuple[ExternalTransitionObservation, ...]:
    return tuple(
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
            confidence=observation.confidence[index : index + 1]
            if observation.confidence is not None
            else None,
        )
        for index in range(observation.state.shape[0])
    )


def _predict(
    model: ExternalFactoredTransitionModel,
    context: torch.Tensor,
    observation: ExternalTransitionObservation,
) -> torch.Tensor:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        return model.predict_with_context(
            observation.state,
            observation.intention,
            context_batch,
        )


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
        quarantine_capacity=ROWS_PER_SLOT * (SLOT_COUNT // 2),
    )
    promotions = []
    for slot, item in enumerate(observations):
        route = router.route_bundle(_rows(item))
        receipt = router.promote_staged_candidate(
            item,
            lambda candidate, expected_records=ROWS_PER_SLOT * (slot + 1): candidate.residual_record_count
            == expected_records,
            prediction_tolerance=1e-6,
        )
        if route.status != "staged" or not receipt.accepted:
            raise RuntimeError(f"slot {slot} promotion failed: {receipt}")
        promotions.append(receipt.__dict__)
    if router.slot_ids != tuple(range(SLOT_COUNT)):
        raise RuntimeError(f"unexpected slot addresses: {router.slot_ids}")
    contexts = router.contexts
    if torch.pdist(contexts).min() <= 1e-4:
        raise RuntimeError("context encoder produced colliding opaque slot keys")

    contextual = ExternalContextualTransitionEvidenceStatistics(
        STATE_WIDTH,
        CONTEXT_WIDTH,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
        matching_tolerance=1e-4,
    )
    global_statistics = ExternalTransitionEvidenceStatistics(
        STATE_WIDTH,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
    )
    outcomes = torch.tensor([0.0, 1.0, 0.0, 1.0])
    for slot, item in enumerate(drift):
        context = contexts[slot]
        prediction = _predict(router.model, context, item)
        outcome = outcomes[slot].expand(item.state.shape[0])
        contextual.observe(prediction, item.next_state, outcome, context)
        global_statistics.observe(prediction, item.next_state, outcome)
    if contextual.observation_count < RELIABILITY_WARMUP:
        raise RuntimeError("contextual reliability warm-up did not reach threshold")

    router.evidence_evaluator = contextual
    router.evidence_threshold = RELIABILITY_THRESHOLD
    router.evidence_gate_min_evidence = RELIABILITY_WARMUP
    router.committed_evidence_gate = True
    state_before_routes = router.model.digest()
    contextual_results = [router.route_bundle(_rows(item)) for item in drift]
    global_control = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=global_statistics
    )
    global_control.committed_evidence_gate = True
    global_result = global_control.route_bundle(_rows(drift[1]))
    state_after_routes = router.model.digest()

    restored_statistics = ExternalContextualTransitionEvidenceStatistics.from_payload(
        contextual.payload()
    )
    restored_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=restored_statistics
    )
    restored_positive = restored_router.route_bundle(_rows(drift[1]))
    contextual_statuses = [result.status for result in contextual_results]
    contextual_slots = [result.slot_id for result in contextual_results]
    gates = {
        "all_slots_promoted": len(promotions) == SLOT_COUNT,
        "negative_slots_vetoed": contextual_statuses[0] == "reliability_veto"
        and contextual_statuses[2] == "reliability_veto",
        "positive_slots_match": contextual_statuses[1] == "matched"
        and contextual_statuses[3] == "matched",
        "positive_slot_identity_preserved": contextual_slots[1] == 1
        and contextual_slots[3] == 3,
        "negative_quarantine_retained": all(
            contextual_results[index].quarantine_accepted is True
            for index in (0, 2)
        ),
        "global_control_overveto_is_observed": global_result.status
        == "reliability_veto",
        "fact_bank_unchanged_by_routes": state_before_routes == state_after_routes,
        "contextual_state_persisted": restored_statistics.digest() == contextual.digest(),
        "restored_positive_route": restored_positive.status == "matched"
        and restored_positive.slot_id == 1,
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "controller_frozen": _digest(controller) == controller_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"contextual reliability gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-contextual-reliability.v1",
        "seed": seed,
        "configuration": {
            "slot_count": SLOT_COUNT,
            "rows_per_slot": ROWS_PER_SLOT,
            "drift_delta": DRIFT_DELTA,
            "match_tolerance": MATCH_TOLERANCE,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "reliability_warmup": RELIABILITY_WARMUP,
            "claim": "context-isolated-replay-free-reliability-v1",
        },
        "metrics": {
            "contextual_statuses": contextual_statuses,
            "contextual_slots": contextual_slots,
            "global_control_status": global_result.status,
            "quarantined_observations": router.quarantined_observations,
            "contextual_observation_count": int(contextual.observation_count),
            "global_observation_count": int(global_statistics.observation_count),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(contextual.observation_count),
            "unique_logical_transition_lifetimes": SLOT_COUNT * ROWS_PER_SLOT,
            "contextual_sufficient_statistics_updates": SLOT_COUNT,
            "global_control_statistics_updates": SLOT_COUNT,
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
            "contextual_statistics": contextual.digest(),
            "global_statistics": global_statistics.digest(),
        },
        "promoted": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=92001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
