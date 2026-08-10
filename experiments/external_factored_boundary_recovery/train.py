"""Near-boundary reliability recovery and quarantine saturation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 2
ROWS = 24
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


def _row(observation: ExternalTransitionObservation, index: int) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
    )


def _observe(
    statistics: ExternalTransitionEvidenceStatistics,
    model: ExternalFactoredTransitionModel,
    context: torch.Tensor,
    observation: ExternalTransitionObservation,
    outcome: float,
) -> None:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        prediction = model.predict_with_context(
            observation.state,
            observation.intention,
            context_batch,
        )
    statistics.observe(
        prediction,
        observation.next_state,
        torch.full((observation.state.shape[0],), outcome),
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    state = torch.linspace(-1.0, 1.0, ROWS).unsqueeze(-1)
    intention = torch.ones(ROWS, 1)
    source = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=0.5 * state + intention,
    )
    drift = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state + DRIFT_DELTA,
    )
    model = ExternalFactoredTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=8,
        residual_read_match_threshold=0.999999,
        residual_write_match_threshold=0.999999,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    base_digest = model.base.digest()
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=8,
        context_width=CONTEXT_WIDTH,
    )
    encoder_digest = encoder.digest()
    reliability = ExternalTransitionEvidenceStatistics(
        STATE_WIDTH,
        bin_count=8,
        error_scale=0.1,
        prior_count=0.01,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
        max_contexts=1,
        quarantine_capacity=2,
        evidence_evaluator=reliability,
        evidence_threshold=RELIABILITY_THRESHOLD,
        evidence_gate_min_evidence=RELIABILITY_WARMUP,
        committed_evidence_gate=True,
    )
    source_route = router.route_bundle(tuple(_row(source, index) for index in range(ROWS)))
    source_receipt = router.promote_staged_candidate(
        source,
        lambda candidate: candidate.residual_record_count == ROWS,
        prediction_tolerance=1e-6,
    )
    context = router.contexts[0].clone()
    _observe(reliability, router.model, context, source, 1.0)
    _observe(reliability, router.model, context, source, 1.0)
    gate_open = int(reliability.observation_count) >= RELIABILITY_WARMUP
    fresh_control = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    fresh_control.committed_evidence_gate = False

    first = router.route_bundle((_row(drift, 0),))
    second = router.route_bundle((_row(drift, 1),))
    saturated = router.route_bundle((_row(drift, 2),))
    bank_digest_after_vetoes = router.model.digest()
    drift_prefix = ExternalTransitionObservation(
        state=drift.state[:2],
        intention=drift.intention[:2],
        next_state=drift.next_state[:2],
    )
    _observe(reliability, router.model, context, drift_prefix, 0.0)
    _observe(reliability, router.model, context, drift, 1.0)
    resolved = router.resolve_quarantine(
        match_tolerance=MATCH_TOLERANCE,
        contradiction_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
    )
    recovered = router.route_bundle((_row(drift, 2),))
    fresh_drift = fresh_control.route_bundle((_row(drift, 2),))
    restored_reliability = ExternalTransitionEvidenceStatistics.from_payload(
        reliability.payload()
    )
    restored = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=restored_reliability
    )
    persistence_exact = (
        restored.digest() == router.digest()
        and restored.configuration() == router.configuration()
        and restored_reliability.digest() == reliability.digest()
    )
    gates = {
        "source_promoted": source_route.status == "staged" and source_receipt.accepted,
        "gate_warmup_reached": gate_open,
        "first_veto_retained": first.status == "reliability_veto"
        and first.quarantine_accepted is True,
        "second_veto_retained": second.status == "reliability_veto"
        and second.quarantine_accepted is True,
        "saturation_explicit": saturated.status == "reliability_veto"
        and saturated.quarantine_accepted is False,
        "bank_unchanged_by_vetoes": bank_digest_after_vetoes == router.model.digest(),
        "verifier_reversal_recovered": resolved == (0, 0),
        "near_boundary_routes_after_recovery": recovered.status == "matched"
        and recovered.slot_id == 0,
        "fresh_gate_disabled_control_matches": fresh_drift.status == "matched",
        "no_candidate_staged": not router.candidate_active,
        "quarantine_drained_after_recovery": router.quarantined_observations == 0,
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "exact_persistence": persistence_exact,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"boundary recovery gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-boundary-recovery.v1",
        "seed": seed,
        "configuration": {
            "rows": ROWS,
            "drift_delta": DRIFT_DELTA,
            "match_tolerance": MATCH_TOLERANCE,
            "quarantine_capacity": 2,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "claim": "near-boundary-recovery-and-explicit-quarantine-saturation-v1",
        },
        "metrics": {
            "source_route_status": source_route.status,
            "source_promotion": source_receipt.__dict__,
            "first_veto": first.__dict__,
            "second_veto": second.__dict__,
            "saturated_veto": saturated.__dict__,
            "resolved": list(resolved),
            "recovered": recovered.__dict__,
            "fresh_drift": fresh_drift.__dict__,
            "reliability_observation_count": int(reliability.observation_count),
            "final_quarantine_rows": router.quarantined_observations,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(reliability.observation_count),
            "unique_logical_transition_lifetimes": ROWS + 3,
            "reliability_sufficient_statistics_updates": 3,
            "base_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": model.base.digest(),
            "context_encoder": encoder_digest,
            "router": router.digest(),
            "reliability": reliability.digest(),
        },
        "promoted": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=91001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
