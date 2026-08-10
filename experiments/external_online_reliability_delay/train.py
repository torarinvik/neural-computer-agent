"""Online interleaved nonlinear-stream audit for reliability and delay state."""

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
ROWS_PER_STREAM = 8
ROUTE_GROUP_SIZE = 2
MATCH_TOLERANCE = 0.02
CORRUPTION_DELTA = 0.12
RELIABILITY_THRESHOLD = 0.9
GATE_WARMUP = 4
WAIT_ROWS = 32


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _stream(seed: int, weights: torch.Tensor) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(ROWS_PER_STREAM, STATE_WIDTH, generator=generator)
    intention = torch.randn(ROWS_PER_STREAM, INTENTION_WIDTH, generator=generator)
    features = torch.cat((state, intention, torch.ones(ROWS_PER_STREAM, 1)), dim=-1)
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=features @ weights,
        confidence=torch.ones(ROWS_PER_STREAM),
    )


def _row(
    observation: ExternalTransitionObservation,
    index: int,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
        confidence=observation.confidence[index : index + 1]
        if observation.confidence is not None
        else None,
    )


def _slice(
    observation: ExternalTransitionObservation,
    start: int,
    stop: int,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[start:stop],
        intention=observation.intention[start:stop],
        next_state=observation.next_state[start:stop],
        confidence=observation.confidence[start:stop]
        if observation.confidence is not None
        else None,
    )


def _corrupted(observation: ExternalTransitionObservation) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state,
        intention=observation.intention,
        next_state=observation.next_state + CORRUPTION_DELTA,
        confidence=observation.confidence,
    )


def _route(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> list[str]:
    statuses: list[str] = []
    for index in range(observation.state.shape[0]):
        statuses.append(router.observe(_row(observation, index)).status)
    return statuses


def _observe_verifier(
    statistics: ExternalTransitionEvidenceStatistics,
    bank: ExternalTransitionModelBank,
    slot: int,
    observation: ExternalTransitionObservation,
    outcome: float,
) -> None:
    context = bank.context_at(slot).to(observation.state)
    prediction = bank(
        observation.state,
        observation.intention,
        context.unsqueeze(0).expand(observation.state.shape[0], -1),
    )
    statistics.observe(
        prediction,
        observation.next_state,
        torch.full((observation.state.shape[0],), outcome),
    )


def _train_wait_online() -> EventWaitStatistics:
    statistics = EventWaitStatistics(
        bin_count=4,
        ridge=1e-3,
        outcome_scale=8.0,
        minimum_context_observations=1,
    )
    delayed = EventWaitPolicy.features(
        age=torch.full((WAIT_ROWS,), 3.0),
        present_fraction=torch.full((WAIT_ROWS,), 0.5),
        complete=torch.zeros(WAIT_ROWS),
        arrival_count=torch.full((WAIT_ROWS,), 8.0),
        arrival_delta=torch.full((WAIT_ROWS,), 3.0),
    )
    immediate = EventWaitPolicy.features(
        age=torch.full((WAIT_ROWS,), 0.1),
        present_fraction=torch.full((WAIT_ROWS,), 0.5),
        complete=torch.zeros(WAIT_ROWS),
        arrival_count=torch.ones(WAIT_ROWS),
        arrival_delta=torch.full((WAIT_ROWS,), 0.1),
    )
    # These are newly arriving scalar outcomes, not replayed event rows.
    statistics.observe(delayed, torch.ones(WAIT_ROWS))
    statistics.observe(immediate, torch.zeros(WAIT_ROWS))
    return statistics


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

    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=2,
    )
    contexts = (
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    weights = (
        torch.tensor([[1.0, 0.2], [-0.3, 0.8], [0.7, -1.1], [0.4, -0.6]]),
        torch.tensor([[0.6, -0.1], [0.2, 0.9], [-0.4, 0.7], [0.3, 0.2]]),
    )
    streams = tuple(
        _stream(seed + 10 * index, weights[index]) for index in range(2)
    )
    slots = tuple(bank.ensure_context(context) for context in contexts)
    for slot, stream, context in zip(slots, streams, contexts, strict=True):
        bank.adaptation_step(
            stream,
            context.unsqueeze(0).expand(stream.state.shape[0], -1),
            None,
        )
    bank_digest_before = bank.digest()

    reliability = ExternalTransitionEvidenceStatistics(
        STATE_WIDTH,
        bin_count=16,
        error_scale=0.2,
        prior_count=0.01,
    )
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
        admission_observations=ROUTE_GROUP_SIZE,
        max_contexts=2,
        evidence_evaluator=reliability,
        evidence_threshold=RELIABILITY_THRESHOLD,
        evidence_gate_min_evidence=GATE_WARMUP,
        committed_evidence_gate=True,
    )

    a_clean = _slice(streams[0], 0, ROUTE_GROUP_SIZE)
    b_clean = _slice(streams[1], 0, ROUTE_GROUP_SIZE)
    b_corrupt = _corrupted(
        _slice(streams[1], ROUTE_GROUP_SIZE, 2 * ROUTE_GROUP_SIZE)
    )
    a_reversal = _slice(streams[0], ROUTE_GROUP_SIZE, 2 * ROUTE_GROUP_SIZE)
    b_reversal = _slice(streams[1], 2 * ROUTE_GROUP_SIZE, 3 * ROUTE_GROUP_SIZE)

    a_clean_statuses = _route(router, a_clean)
    _observe_verifier(reliability, bank, slots[0], a_clean, 1.0)
    b_clean_statuses = _route(router, b_clean)
    _observe_verifier(reliability, bank, slots[1], b_clean, 1.0)
    gate_open_before_corruption = int(reliability.observation_count) >= GATE_WARMUP
    b_corrupt_statuses = _route(router, b_corrupt)
    _observe_verifier(reliability, bank, slots[1], b_corrupt, 0.0)
    a_reversal_statuses = _route(router, a_reversal)
    _observe_verifier(reliability, bank, slots[0], a_reversal, 1.0)
    b_reversal_statuses = _route(router, b_reversal)
    _observe_verifier(reliability, bank, slots[1], b_reversal, 1.0)

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
        admission_observations=ROUTE_GROUP_SIZE,
        max_contexts=2,
        committed_evidence_gate=False,
    )
    fresh_corrupt_statuses = _route(fresh_router, b_corrupt)

    wait_statistics = _train_wait_online()
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
        "gate_was_open_before_corruption": gate_open_before_corruption,
        "stream_a_clean_routes": a_clean_statuses[-1] == "matched",
        "stream_b_clean_routes": b_clean_statuses[-1] == "matched",
        "online_gate_vetoes_corrupted_stream_b": b_corrupt_statuses[-1]
        in {"capacity", "conflict", "reliability_veto", "ambiguous"},
        "corruption_does_not_stage_candidate": router.provisional_candidate_count == 0,
        "stream_a_reversal_routes": a_reversal_statuses[-1] == "matched",
        "stream_b_reversal_routes": b_reversal_statuses[-1] == "matched",
        "fresh_gate_disabled_control_matches_corruption": fresh_corrupt_statuses[-1]
        == "matched",
        "historical_bank_unchanged": bank.digest() == bank_digest_before,
        "controller_frozen": _digest(controller) == controller_digest,
        "reliability_persistence": restored_reliability.digest() == reliability.digest(),
        "wait_persistence": restored_wait.digest() == wait_statistics.digest(),
        "router_persistence": restored_router.configuration() == router.configuration(),
        "learned_delay_waits": delayed_probability > 0.75,
        "learned_absence_releases": immediate_probability < 0.25,
    }
    if not all(gates.values()):
        raise RuntimeError(f"online reliability/delay gates failed: {gates}")

    report: dict[str, object] = {
        "schema": "neural-computer.external-online-reliability-delay.v1",
        "seed": seed,
        "configuration": {
            "interleaved_streams": 2,
            "route_group_size": ROUTE_GROUP_SIZE,
            "match_tolerance": MATCH_TOLERANCE,
            "corruption_delta": CORRUPTION_DELTA,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "gate_warmup": GATE_WARMUP,
            "controller_optimizer_updates": 0,
            "old_evidence_replay": 0,
        },
        "metrics": {
            "a_clean_statuses": a_clean_statuses,
            "b_clean_statuses": b_clean_statuses,
            "b_corrupt_statuses": b_corrupt_statuses,
            "a_reversal_statuses": a_reversal_statuses,
            "b_reversal_statuses": b_reversal_statuses,
            "fresh_corrupt_statuses": fresh_corrupt_statuses,
            "reliability_observation_count": int(reliability.observation_count),
            "delayed_wait_probability": delayed_probability,
            "immediate_wait_probability": immediate_probability,
            "bank_digest_before": bank_digest_before,
            "bank_digest_after": bank.digest(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": 2 * ROUTE_GROUP_SIZE * 5 + 2 * WAIT_ROWS,
            "unique_logical_transition_lifetimes": 2 * ROUTE_GROUP_SIZE * 5,
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "reliability_optimizer_updates": 0,
            "wait_optimizer_updates": 0,
            "reliability_sufficient_statistics_updates": 10,
            "wait_sufficient_statistics_updates": 2 * WAIT_ROWS,
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
    print(json.dumps(run(args.seed, args.report_out), indent=2))


if __name__ == "__main__":
    main()
