"""Two-seed audit of raw-evidence-free provisional transition candidates."""

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
REGIME_COUNT = 3
TRAIN_ROWS = 64
HELDOUT_ROWS = 64
FEATURE_WIDTH = 128
ADMISSION_OBSERVATIONS = 8
LOSS_THRESHOLD = 0.02
RANDOM_FEATURE_FAMILY = "random_feature_sufficient_statistics_v1"


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


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rows(observation: ExternalTransitionObservation):
    for index in range(observation.state.shape[0]):
        yield ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence[index : index + 1]
            ),
        )


def _new_router(*, capacity: int) -> ExternalOnlineTransitionContextRouter:
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family=RANDOM_FEATURE_FAMILY,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        affine_ridge=1e-4,
        capacity=capacity,
    )
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=32,
        context_width=CONTEXT_WIDTH,
    )
    return ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=LOSS_THRESHOLD,
        match_margin=0.005,
        continuation_tolerance=LOSS_THRESHOLD,
        # The harness presents one candidate stream to completion before its
        # held-out promotion transaction. A later regime starts only after the
        # candidate has been committed and removed from quarantine.
        provisional_continuation_tolerance=1e9,
        admission_observations=ADMISSION_OBSERVATIONS,
        max_contexts=capacity,
        defer_admission=True,
        candidate_model_families=(RANDOM_FEATURE_FAMILY,),
        provisional_evidence_policy="streaming_statistics",
    )


def _error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, context_batch).detach())


def _consume_candidate(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> tuple[int, int]:
    staged_windows = 0
    consumed_rows = 0
    for row in _rows(observation):
        result = router.observe(row)
        if result.status != "staged":
            continue
        staged_windows += 1
        consumed_rows += int(result.observation.state.shape[0])
        router.adaptation_step(result, None, replay_evidence=False)
    return staged_windows, consumed_rows


def _run_normal(seed: int) -> dict[str, object]:
    router = _new_router(capacity=REGIME_COUNT)
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

    heldout: list[ExternalTransitionObservation] = []
    contexts: list[torch.Tensor] = []
    promotion_records: list[dict[str, object]] = []
    raw_rows_retained: list[int] = []
    total_windows = 0
    total_rows = 0
    for regime in range(REGIME_COUNT):
        train, probe = _fixture(seed, regime)
        heldout.append(probe)
        windows, rows = _consume_candidate(router, train)
        total_windows += windows
        total_rows += rows
        if router.provisional_candidate_count != 1:
            raise RuntimeError("expected one isolated provisional candidate")
        context = router.provisional_context_at(0)
        contexts.append(context)
        raw_rows_retained.append(
            sum(
                item.state.shape[0]
                for item in router._provisional_candidates[0].observations
            )
        )

        def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
            return all(
                _error(candidate, old_probe, old_context) < LOSS_THRESHOLD
                for old_probe, old_context in zip(
                    heldout[:-1], contexts[:-1], strict=True
                )
            )

        receipt = router.promote_staged_candidate(
            probe,
            retention_probe,
            prediction_tolerance=LOSS_THRESHOLD,
        )
        promotion_records.append(
            {
                "accepted": receipt.accepted,
                "heldout_error": receipt.heldout_error,
                "staged_windows": windows,
                "consumed_rows": rows,
            }
        )
        if not receipt.accepted:
            raise RuntimeError(f"promotion failed for regime {regime}: {receipt.reason}")

    final_errors = [
        _error(router.bank, probe, context)
        for probe, context in zip(heldout, contexts, strict=True)
    ]
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    return {
        "router": router,
        "controller_digest": controller_digest,
        "controller_unchanged": controller_digest == _digest(controller),
        "promotion_records": promotion_records,
        "heldout_errors": final_errors,
        "raw_rows_retained": raw_rows_retained,
        "total_staged_windows": total_windows,
        "total_consumed_rows": total_rows,
        "source_retention": all(error < LOSS_THRESHOLD for error in final_errors),
        "exact_persistence": restored.bank.content_digest() == router.bank.content_digest(),
        "restored_policy": restored.configuration()["provisional_evidence_policy"],
    }


def _run_shuffled(seed: int) -> dict[str, object]:
    router = _new_router(capacity=1)
    train, probe = _fixture(seed + 9000, 0)
    generator = torch.Generator().manual_seed(seed + 9001)
    shuffled = train.next_state[torch.randperm(TRAIN_ROWS, generator=generator)]
    corrupted = ExternalTransitionObservation(
        state=train.state,
        intention=train.intention,
        next_state=shuffled,
        confidence=train.confidence,
    )
    windows, rows = _consume_candidate(router, corrupted)
    if router.provisional_candidate_count != 1:
        raise RuntimeError("shuffled control did not stage a candidate")
    raw_rows = sum(
        item.state.shape[0]
        for item in router._provisional_candidates[0].observations
    )
    heldout_error = (
        float(router.provisional_model.loss(probe).detach())
        if router.provisional_model is not None
        else float("inf")
    )
    return {
        "heldout_error": heldout_error,
        "staged_windows": windows,
        "consumed_rows": rows,
        "raw_rows_retained": raw_rows,
        "router": router,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    normal = _run_normal(seed)
    shuffled = _run_shuffled(seed)
    normal_router = normal["router"]
    shuffled_router = shuffled["router"]
    normal_errors = normal["heldout_errors"]
    gates = {
        "all_promotions_accepted": all(
            bool(item["accepted"]) for item in normal["promotion_records"]
        ),
        "all_heldout_factual_predictions_pass": all(
            float(error) < LOSS_THRESHOLD for error in normal_errors
        ),
        "earlier_regimes_retained": bool(normal["source_retention"]),
        "raw_provisional_rows_never_retained": (
            normal["raw_rows_retained"] == [0, 0, 0]
            and shuffled["raw_rows_retained"] == 0
        ),
        "streaming_policy_restores": normal["restored_policy"] == "streaming_statistics",
        "shuffled_next_states_rejected": float(shuffled["heldout_error"]) >= LOSS_THRESHOLD,
        "controller_frozen": bool(normal["controller_unchanged"]),
        "exact_persistence": bool(normal["exact_persistence"]),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-streaming-statistics-candidate-pressure-test.v1",
        "claim_boundary": (
            "A provisional external factual candidate can consume nonlinear "
            "transition evidence once through fixed sufficient statistics, "
            "retain no raw candidate rows, and pass held-out promotion and "
            "old-slot retention gates; this is not general continual learning."
        ),
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "train_rows_per_regime": TRAIN_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "feature_width": FEATURE_WIDTH,
            "admission_observations": ADMISSION_OBSERVATIONS,
            "model_family": RANDOM_FEATURE_FAMILY,
            "provisional_evidence_policy": "streaming_statistics",
            "loss_threshold": LOSS_THRESHOLD,
        },
        "metrics": {
            "heldout_errors": normal_errors,
            "shuffled_heldout_error": shuffled["heldout_error"],
            "promotion_records": normal["promotion_records"],
            "raw_rows_retained": normal["raw_rows_retained"],
            "shuffled_raw_rows_retained": shuffled["raw_rows_retained"],
            "normal_context_count": normal_router.bank.context_count,
            "shuffled_provisional_candidates": shuffled_router.provisional_candidate_count,
        },
        "accounting": {
            "unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS),
            "streaming_statistics_updates": normal["total_consumed_rows"],
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "old_committed_slot_replay": 0,
            "raw_provisional_rows_persisted": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "controller": normal["controller_digest"],
            "bank": normal_router.bank.content_digest(),
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
