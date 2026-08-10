"""Three-seed audit of separated replay-free reliability and delay state.

The factual model and controller stay fixed.  A sufficient-statistics
reliability component learns from scalar verifier outcomes and may veto a
committed route, but it cannot mutate a factual model or its opaque identity.
An independent wait-statistics component learns whether incomplete evidence is
worth waiting for.  The two state machines are persisted separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    EventWaitPolicy,
    EventWaitStatistics,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
CALIBRATION_ROWS = 128
ROUTE_ROWS = 8
MATCH_TOLERANCE = 0.02
CORRUPTION_DELTA = 0.12
RELIABILITY_THRESHOLD = 0.9
WAIT_TRAINING_ROWS = 64


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _source_observation(seed: int) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(ROUTE_ROWS, STATE_WIDTH, generator=generator)
    intention = torch.randn(ROUTE_ROWS, INTENTION_WIDTH, generator=generator)
    features = torch.cat((state, intention, torch.ones(ROUTE_ROWS, 1)), dim=-1)
    weights = torch.tensor(
        [[1.0, 0.2], [-0.3, 0.8], [0.7, -1.1], [0.4, -0.6]]
    )
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=features @ weights,
        confidence=torch.ones(ROUTE_ROWS),
    )


def _row(observation: ExternalTransitionObservation, index: int) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
        confidence=(
            None
            if observation.confidence is None
            else observation.confidence[index : index + 1]
        ),
    )


def _train_reliability(seed: int) -> ExternalTransitionEvidenceStatistics:
    statistics = ExternalTransitionEvidenceStatistics(
        STATE_WIDTH,
        bin_count=16,
        error_scale=0.2,
        prior_count=0.01,
    )
    for update in range(CALIBRATION_ROWS):
        generator = torch.Generator().manual_seed(seed + 1000 + update)
        prediction = torch.randn(1, STATE_WIDTH, generator=generator)
        clean = update % 2 == 0
        observed = prediction if clean else prediction + CORRUPTION_DELTA
        outcome = torch.tensor([1.0 if clean else 0.0])
        statistics.observe(prediction, observed, outcome)
    return statistics


def _train_wait_statistics(seed: int) -> EventWaitStatistics:
    del seed
    statistics = EventWaitStatistics(
        bin_count=4,
        ridge=1e-3,
        outcome_scale=8.0,
        minimum_context_observations=1,
    )
    delayed = EventWaitPolicy.features(
        age=torch.full((WAIT_TRAINING_ROWS, ), 3.0),
        present_fraction=torch.full((WAIT_TRAINING_ROWS, ), 0.5),
        complete=torch.zeros(WAIT_TRAINING_ROWS),
        arrival_count=torch.full((WAIT_TRAINING_ROWS, ), 8.0),
        arrival_delta=torch.full((WAIT_TRAINING_ROWS, ), 3.0),
    )
    immediate = EventWaitPolicy.features(
        age=torch.full((WAIT_TRAINING_ROWS, ), 0.1),
        present_fraction=torch.full((WAIT_TRAINING_ROWS, ), 0.5),
        complete=torch.zeros(WAIT_TRAINING_ROWS),
        arrival_count=torch.ones(WAIT_TRAINING_ROWS),
        arrival_delta=torch.full((WAIT_TRAINING_ROWS, ), 0.1),
    )
    statistics.observe(delayed, torch.ones(WAIT_TRAINING_ROWS))
    statistics.observe(immediate, torch.zeros(WAIT_TRAINING_ROWS))
    return statistics


def _route(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> list[str]:
    statuses: list[str] = []
    for index in range(observation.state.shape[0]):
        statuses.append(router.observe(_row(observation, index)).status)
    return statuses


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

    reliability = _train_reliability(seed)
    reliability_digest = reliability.digest()
    wait_statistics = _train_wait_statistics(seed)
    wait_digest = wait_statistics.digest()

    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    slot = bank.ensure_context(context)
    source = _source_observation(seed)
    bank.adaptation_step(
        source,
        context.unsqueeze(0).expand(source.state.shape[0], -1),
        None,
    )
    bank_digest_before = bank.digest()

    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            INTENTION_WIDTH,
            hidden_width=16,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
        continuation_tolerance=MATCH_TOLERANCE,
        admission_observations=4,
        max_contexts=1,
        evidence_evaluator=reliability,
        evidence_threshold=RELIABILITY_THRESHOLD,
        evidence_gate_min_evidence=CALIBRATION_ROWS,
        committed_evidence_gate=True,
    )

    clean_statuses = _route(router, source)
    corrupted = ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state + CORRUPTION_DELTA,
        confidence=source.confidence,
    )
    corrupted_statuses = _route(router, corrupted)
    reversal_statuses = _route(router, source)

    fresh_router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            INTENTION_WIDTH,
            hidden_width=16,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
        admission_observations=4,
        max_contexts=1,
        committed_evidence_gate=False,
    )
    fresh_corrupted_statuses = _route(fresh_router, corrupted)

    delayed_features = EventWaitPolicy.features(
        age=torch.tensor([3.0]),
        present_fraction=torch.tensor([0.5]),
        complete=torch.tensor([0.0]),
        arrival_count=torch.tensor([8.0]),
        arrival_delta=torch.tensor([3.0]),
    )
    immediate_features = EventWaitPolicy.features(
        age=torch.tensor([0.1]),
        present_fraction=torch.tensor([0.5]),
        complete=torch.tensor([0.0]),
        arrival_count=torch.tensor([1.0]),
        arrival_delta=torch.tensor([0.1]),
    )
    with torch.no_grad():
        delayed_probability = float(wait_statistics(delayed_features)[0])
        immediate_probability = float(wait_statistics(immediate_features)[0])

    restored_reliability = ExternalTransitionEvidenceStatistics.from_payload(
        reliability.payload()
    )
    restored_wait = EventWaitStatistics.from_payload(wait_statistics.payload())
    restored_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload(),
        evidence_evaluator=restored_reliability,
    )

    gates = {
        "clean_reuses_committed_slot": clean_statuses[-1] in {"matched", "continuation"},
        "learned_gate_vetoes_low_error_corruption": corrupted_statuses[-1]
        in {"capacity", "conflict"},
        "reversal_reuses_original_slot": reversal_statuses[-1]
        in {"matched", "continuation"},
        "fresh_gate_control_matches_corruption": fresh_corrupted_statuses[-1]
        == "matched",
        "historical_bank_unchanged": bank.digest() == bank_digest_before,
        "controller_frozen": _digest(controller) == controller_digest,
        "reliability_persistence": restored_reliability.digest() == reliability_digest,
        "wait_persistence": restored_wait.digest() == wait_digest,
        "router_gate_persistence": restored_router.configuration()
        == router.configuration(),
        "learned_delay_waits": delayed_probability > 0.75,
        "learned_absence_releases": immediate_probability < 0.25,
    }
    if not all(gates.values()):
        raise RuntimeError(f"reliability/delay gates failed: {gates}")

    report: dict[str, object] = {
        "schema": "neural-computer.external-reliability-delay.v1",
        "seed": seed,
        "configuration": {
            "calibration_rows": CALIBRATION_ROWS,
            "route_rows": ROUTE_ROWS,
            "match_tolerance": MATCH_TOLERANCE,
            "corruption_delta": CORRUPTION_DELTA,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "wait_training_rows": WAIT_TRAINING_ROWS,
            "committed_evidence_gate": True,
            "controller_updates": 0,
            "old_evidence_replay": 0,
        },
        "metrics": {
            "slot": slot,
            "clean_statuses": clean_statuses,
            "corrupted_statuses": corrupted_statuses,
            "reversal_statuses": reversal_statuses,
            "fresh_corrupted_statuses": fresh_corrupted_statuses,
            "delayed_wait_probability": delayed_probability,
            "immediate_wait_probability": immediate_probability,
            "bank_digest_before": bank_digest_before,
            "bank_digest_after": bank.digest(),
            "reliability_digest": reliability_digest,
            "wait_digest": wait_digest,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": CALIBRATION_ROWS + 2 * WAIT_TRAINING_ROWS,
            "unique_logical_transition_lifetimes": ROUTE_ROWS,
            "replayed_examples": 0,
            "controller_optimizer_updates": 0,
            "reliability_optimizer_updates": 0,
            "wait_optimizer_updates": 0,
            "external_sufficient_statistics_updates": CALIBRATION_ROWS
            + 2 * WAIT_TRAINING_ROWS,
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed, args.report_out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
