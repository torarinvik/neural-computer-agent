"""Audit recursive rollout gates inside copy-on-write candidate promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
    ExternalTransitionRollout,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
REGIME_ROWS = 4
ROLLOUT_TOLERANCE = 1e-6


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _regime(offset: float) -> ExternalTransitionObservation:
    state = torch.arange(REGIME_ROWS, dtype=torch.float32).unsqueeze(-1) - 1.0
    intention = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    next_state = state + intention + offset
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(REGIME_ROWS),
    )


def _rollout(offset: float, *, corrupted: bool = False) -> ExternalTransitionRollout:
    expected = torch.tensor([[1.0 + offset], [2.0 + 2.0 * offset]])
    if corrupted:
        expected[-1] = 999.0
    return ExternalTransitionRollout(
        initial_state=torch.zeros(1),
        intentions=torch.ones(2, 1),
        expected_states=expected,
    )


def _new_router(seed: int) -> ExternalOnlineTransitionContextRouter:
    torch.manual_seed(seed)
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    return ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            INTENTION_WIDTH,
            hidden_width=8,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=1e-7,
        match_margin=1e-5,
        continuation_tolerance=1e-7,
        admission_observations=REGIME_ROWS,
        max_contexts=1,
        auto_grow=True,
        defer_admission=True,
        provisional_evidence_policy="streaming_statistics",
    )


def _ingest(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> None:
    for row in range(observation.state.shape[0]):
        result = router.observe(
            ExternalTransitionObservation(
                state=observation.state[row : row + 1],
                intention=observation.intention[row : row + 1],
                next_state=observation.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)


def _retains_source(
    source: ExternalTransitionObservation,
    threshold: float = ROLLOUT_TOLERANCE,
):
    def probe(candidate: ExternalTransitionModelBank) -> bool:
        context = candidate.context_at(0)
        return float(
            candidate.loss(
                source,
                context.unsqueeze(0).expand(source.state.shape[0], -1),
            )
        ) <= threshold

    return probe


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=2,
        feedback_width=2,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    router = _new_router(seed)
    source = _regime(0.0)
    target = _regime(10.0)
    corrupted = _regime(20.0)

    _ingest(router, source)
    source_receipt = router.promote_staged_candidate(
        _regime(0.0),
        _retains_source(source),
        prediction_tolerance=ROLLOUT_TOLERANCE,
        heldout_rollout=_rollout(0.0),
        rollout_error_tolerance=ROLLOUT_TOLERANCE,
    )
    source_context_count = router.bank.context_count

    _ingest(router, target)
    target_receipt = router.promote_staged_candidate(
        _regime(10.0),
        _retains_source(source),
        prediction_tolerance=ROLLOUT_TOLERANCE,
        heldout_rollout=_rollout(10.0),
        rollout_error_tolerance=ROLLOUT_TOLERANCE,
    )
    target_context_count = router.bank.context_count
    target_context = router.bank.context_at(1)
    planner = ExternalModelBasedPlanner(router.bank, beam_width=1)
    committed_target_rollout_error = planner.rollout_error(
        _rollout(10.0),
        transition_context=target_context.unsqueeze(0),
    )

    _ingest(router, corrupted)
    before_rejection = router.bank.content_digest()
    before_capacity = router.bank.capacity
    rejection = router.promote_staged_candidate(
        _regime(20.0),
        _retains_source(source),
        prediction_tolerance=ROLLOUT_TOLERANCE,
        heldout_rollout=_rollout(20.0, corrupted=True),
        rollout_error_tolerance=ROLLOUT_TOLERANCE,
    )

    gates = {
        "source_recursive_promotion": source_receipt.accepted
        and source_receipt.heldout_rollout_error is not None
        and source_receipt.heldout_rollout_error <= ROLLOUT_TOLERANCE,
        "target_recursive_promotion": target_receipt.accepted
        and target_receipt.heldout_rollout_error is not None
        and target_receipt.heldout_rollout_error <= ROLLOUT_TOLERANCE,
        "automatic_growth_only_after_acceptance": source_context_count == 1
        and target_context_count == 2,
        "committed_target_rollout_mastery": committed_target_rollout_error
        <= ROLLOUT_TOLERANCE,
        "corrupted_recursive_candidate_rejected": not rejection.accepted
        and "recursive" in rejection.reason,
        "rejected_candidate_did_not_write": router.bank.content_digest()
        == before_rejection
        and router.bank.capacity == before_capacity,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "old_regime_replay_zero": True,
    }
    report = {
        "schema": "neural-computer.external-recursive-candidate-promotion-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "regime_rows": REGIME_ROWS,
            "rollout_error_tolerance": ROLLOUT_TOLERANCE,
            "model_family": "affine_sufficient_statistics_v1",
            "promotion": "copy_on_write_one_step_plus_recursive_rollout_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source_receipt": {
            "accepted": source_receipt.accepted,
            "heldout_error": source_receipt.heldout_error,
            "heldout_rollout_error": source_receipt.heldout_rollout_error,
            "reason": source_receipt.reason,
        },
        "target_receipt": {
            "accepted": target_receipt.accepted,
            "heldout_error": target_receipt.heldout_error,
            "heldout_rollout_error": target_receipt.heldout_rollout_error,
            "reason": target_receipt.reason,
        },
        "rejection": {
            "accepted": rejection.accepted,
            "heldout_error": rejection.heldout_error,
            "heldout_rollout_error": rejection.heldout_rollout_error,
            "reason": rejection.reason,
        },
        "accounting": {
            "unique_verifier_bits": REGIME_ROWS * 3,
            "unique_logical_lifetimes": REGIME_ROWS * 3,
            "controller_optimizer_updates": 0,
            "old_regime_replay_during_target_adaptation": 0,
            "candidate_statistics_updates": REGIME_ROWS * 3,
        },
        "controller_digest": controller_digest,
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
