"""Replay-free partial-evidence and gradual-drift factual-memory audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from functools import partial
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 8
CAPACITY = 4
ROW_COUNT = 12
PRESENTED_ROWS = 8
ADMISSION_ROWS = 4
DRIFT_SLOPES = (1.0, 1.5, 2.0)
HELDOUT_STATES = (-0.75, 0.25, 1.25, 2.25)
MATCH_TOLERANCE = 1e-5
PREDICTION_TOLERANCE = 1e-2


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _observation(slope: float, *, heldout: bool = False) -> ExternalTransitionObservation:
    if heldout:
        states = torch.tensor(HELDOUT_STATES, dtype=torch.float32).unsqueeze(1)
        intentions = torch.tensor((-1.0, 1.0, -1.0, 1.0)).unsqueeze(1)
    else:
        states = torch.tensor(
            (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
            dtype=torch.float32,
        ).unsqueeze(1)
        intentions = torch.tensor(
            (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            dtype=torch.float32,
        ).unsqueeze(1)
    return ExternalTransitionObservation(
        state=states,
        intention=intentions,
        next_state=states + slope * intentions,
        confidence=torch.ones(states.shape[0]),
    )


def _row(observation: ExternalTransitionObservation, index: int) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
        confidence=torch.ones(1),
    )


def _consume(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
    *,
    indices: tuple[int, ...],
) -> tuple[list[str], int]:
    statuses: list[str] = []
    staged = 0
    for index in indices:
        result = router.observe(_row(observation, index))
        statuses.append(result.status)
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
            staged += 1
    return statuses, staged


def _planner_mastery(
    bank: ExternalTransitionModelBank,
    context: torch.Tensor,
    slope: float,
) -> float:
    planner = ExternalModelBasedPlanner(bank, beam_width=2)
    candidate_intentions = torch.tensor([[-1.0], [1.0]])
    targets = ((0.0, 2.0 * slope), (1.0, 1.0 + 2.0 * slope), (-1.0, -1.0 + 2.0 * slope))
    successes: list[bool] = []
    for start, goal in targets:
        result = planner.plan(
            torch.tensor([[start]]),
            torch.tensor([[goal]]),
            candidate_intentions,
            horizon=2,
            transition_context=context.unsqueeze(0),
        )
        actual = start
        for intention in result.intentions[0]:
            actual += slope * float(intention[0])
        successes.append(abs(actual - goal) <= 1e-5)
    return sum(successes) / len(successes)


def _retention_probe(
    candidate: ExternalTransitionModelBank,
    *,
    expected_context_count: int,
    retained_digests: dict[int, str],
) -> bool:
    if candidate.context_count != expected_context_count:
        return False
    return all(
        candidate.models[index].digest() == digest
        for index, digest in retained_digests.items()
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=ADMISSION_ROWS,
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
        capacity=CAPACITY,
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
        match_margin=MATCH_TOLERANCE,
        continuation_tolerance=MATCH_TOLERANCE,
        conflict_patience=1,
        # The current provisional candidate is allowed to accumulate the
        # current stream; promotion remains held-out and copy-on-write.
        provisional_continuation_tolerance=1e9,
        provisional_match_margin=0.0,
        admission_observations=ADMISSION_ROWS,
        max_contexts=CAPACITY,
        defer_admission=True,
        provisional_evidence_policy="streaming_statistics",
    )
    presented_indices = tuple(range(PRESENTED_ROWS))
    source_digest: str | None = None
    phase_records: list[dict[str, object]] = []
    route_counts: Counter[str] = Counter()
    slot_ids: list[int] = []
    prior_digests: dict[int, str] = {}

    for phase_index, slope in enumerate(DRIFT_SLOPES):
        training = _observation(slope)
        heldout = _observation(slope, heldout=True)
        statuses, staged = _consume(router, training, indices=presented_indices)
        route_counts.update(statuses)
        if router.provisional_candidate_count != 1:
            raise RuntimeError(
                f"phase {phase_index} left {router.provisional_candidate_count} "
                "provisional candidates instead of one drift candidate"
            )
        candidate_context = router.provisional_context_at(0)
        source_before = None if not bank.context_count else bank.models[0].digest()
        expected_context_count = phase_index + 1
        retained_digests = prior_digests.copy()
        retention_probe = partial(
            _retention_probe,
            expected_context_count=expected_context_count,
            retained_digests=retained_digests,
        )
        receipt = router.promote_staged_candidate(
            heldout,
            retention_probe,
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        if not receipt.accepted or receipt.slot_index is None:
            raise RuntimeError(
                f"phase {phase_index} promotion failed: {receipt.reason}; "
                f"heldout_error={receipt.heldout_error}"
            )
        slot_index = receipt.slot_index
        slot_ids.append(bank.slot_id_at(slot_index))
        if source_before is None:
            source_digest = bank.models[slot_index].digest()
        prior_digests[slot_index] = bank.models[slot_index].digest()
        phase_records.append(
            {
                "phase": phase_index,
                "slope": slope,
                "statuses": dict(Counter(statuses)),
                "staged_windows": staged,
                "presented_rows": len(presented_indices),
                "available_rows": ROW_COUNT,
                "candidate_evidence_count": PRESENTED_ROWS,
                "slot_id": bank.slot_id_at(slot_index),
                "heldout_error": receipt.heldout_error,
                "planner_mastery": _planner_mastery(
                    bank,
                    candidate_context,
                    slope,
                ),
            }
        )

    source_training = _observation(DRIFT_SLOPES[0])
    return_statuses, _return_staged = _consume(
        router,
        source_training,
        indices=presented_indices,
    )
    route_counts.update(return_statuses)
    source_return_matched = return_statuses.count("matched") == 2
    source_context = bank.context_at(0)
    source_return_mastery = _planner_mastery(
        bank,
        source_context,
        DRIFT_SLOPES[0],
    )
    source_retained = (
        bank.models[0].digest() == source_digest
        and source_return_mastery == 1.0
    )

    corrupted = _observation(DRIFT_SLOPES[-1] + 5.0)
    corrupted_statuses, _corrupted_staged = _consume(
        router,
        corrupted,
        indices=presented_indices,
    )
    bank_before_rejection = bank.content_digest()
    rejection = router.promote_staged_candidate(
        _observation(DRIFT_SLOPES[-1] + 5.0, heldout=True),
        lambda _candidate: False,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    bank_after_rejection = bank.content_digest()
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    all_planner_mastery = all(
        float(record["planner_mastery"]) == 1.0 for record in phase_records
    )
    gates = {
        "replay_free_bank_family": bank.replay_free_updates,
        "partial_rows_used": all(
            int(record["presented_rows"]) < int(record["available_rows"])
            for record in phase_records
        ),
        "all_drift_versions_promoted": len(slot_ids) == len(DRIFT_SLOPES),
        "all_drift_planner_mastery": all_planner_mastery,
        "source_return_routed_to_old_slot": source_return_matched,
        "source_slot_retained": source_retained,
        "corruption_rejected_without_bank_write": (
            not rejection.accepted
            and bank_before_rejection == bank_after_rejection
        ),
        "no_raw_candidate_rows_retained": all(
            not candidate.observations for candidate in router._provisional_candidates
        ),
        "controller_unchanged": controller_digest == _digest(controller),
        "exact_router_persistence": (
            restored.bank.digest() == bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-partial-drift-streaming-model.v1",
        "seed": seed,
        "configuration": {
            "drift_slopes": list(DRIFT_SLOPES),
            "available_rows_per_phase": ROW_COUNT,
            "presented_rows_per_phase": PRESENTED_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "model_family": "affine_sufficient_statistics_v1",
            "policy": "none_replay_free_versioned_factual_drift_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "phases": phase_records,
        "return_to_source": {
            "statuses": dict(Counter(return_statuses)),
            "source_return_matched": source_return_matched,
            "source_return_mastery": source_return_mastery,
        },
        "corruption_control": {
            "statuses": dict(Counter(corrupted_statuses)),
            "promotion_accepted": rejection.accepted,
            "reason": rejection.reason,
        },
        "routing": {
            "slot_ids": slot_ids,
            "route_counts": dict(route_counts),
            "committed_context_count": bank.context_count,
        },
        "accounting": {
            "unique_transition_rows": len(DRIFT_SLOPES) * PRESENTED_ROWS,
            "unique_heldout_rows": len(DRIFT_SLOPES) * len(HELDOUT_STATES),
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "streaming_statistics_updates": len(DRIFT_SLOPES) * PRESENTED_ROWS,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": (
            "bounded replay-free partial-evidence factual drift versions with "
            "planner verification; not learned multimodal context formation, "
            "unrestricted memory growth, or general continual learning"
        ),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=81001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
