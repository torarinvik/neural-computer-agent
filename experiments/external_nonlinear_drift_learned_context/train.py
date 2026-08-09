"""Pressure test learned addresses with replay-free nonlinear drift memory.

The controller is frozen.  A context encoder is trained only on source
transition bundles, then frozen before two nonlinear target regimes arrive as
one-row streams.  The target model slots consume each staged evidence bundle
once through random-feature sufficient statistics.  The verifier keeps the
held-out promotion and retention decisions outside the deployed learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
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
CONTEXT_WIDTH = 12
REGIME_NAMES = ("source_a", "source_b", "target_c", "target_d")
SOURCE_REGIMES = 2
TARGET_REGIMES = (2, 3)
TRAIN_ROWS = 64
HELDOUT_ROWS = 64
PRESENTED_ROWS = 32
ADMISSION_ROWS = 8
CONTEXT_HIDDEN_WIDTH = 40
CONTEXT_UPDATES = 400
FEATURE_WIDTH = 128
LOSS_THRESHOLD = 0.02
RANDOM_FEATURE_FAMILY = "random_feature_sufficient_statistics_v1"


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _transition(
    regime: int,
    state: torch.Tensor,
    intention: torch.Tensor,
) -> torch.Tensor:
    phase = 0.35 * regime
    scale = 1.0 + 0.12 * regime
    return torch.cat(
        (
            torch.sin(scale * state[:, 0:1] + intention + phase),
            torch.cos(state[:, 1:2] - 0.4 * intention + phase)
            + 0.15 * state[:, 0:1] * intention,
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
        confidence=observation.confidence[index : index + 1]
        if observation.confidence is not None
        else None,
    )


def _rows(
    observation: ExternalTransitionObservation,
    count: int | None = None,
) -> list[ExternalTransitionObservation]:
    limit = observation.state.shape[0] if count is None else count
    return [_row(observation, index) for index in range(limit)]


def _noisy_view(
    observation: ExternalTransitionObservation,
    *,
    seed: int,
    noise: float,
) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(observation.state.shape[0], generator=generator)
    state = observation.state.index_select(0, permutation)
    next_state = observation.next_state.index_select(0, permutation)
    return ExternalTransitionObservation(
        state=state + noise * torch.randn(state.shape, generator=generator),
        intention=observation.intention.index_select(0, permutation),
        next_state=next_state
        + noise * torch.randn(next_state.shape, generator=generator),
        confidence=observation.confidence.index_select(0, permutation)
        if observation.confidence is not None
        else None,
    )


def _train_context_encoder(
    encoder: ExternalTransitionContextEncoder,
    observations: dict[str, ExternalTransitionObservation],
    *,
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
    final_loss = float("inf")
    source_names = REGIME_NAMES[:SOURCE_REGIMES]
    for update in range(1, CONTEXT_UPDATES + 1):
        left = [
            encoder.encode_observation(
                _noisy_view(
                    observations[name],
                    seed=seed + update * 31 + index,
                    noise=0.005,
                )
            )
            for index, name in enumerate(source_names)
        ]
        right = [
            encoder.encode_observation(
                _noisy_view(
                    observations[name],
                    seed=seed + update * 47 + index,
                    noise=0.01,
                )
            )
            for index, name in enumerate(source_names)
        ]
        loss = encoder.contrastive_loss(
            torch.stack(left),
            torch.stack(right),
            temperature=0.1,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss, CONTEXT_UPDATES


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family=RANDOM_FEATURE_FAMILY,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        affine_ridge=1e-4,
        capacity=len(REGIME_NAMES),
    )


def _error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, context_batch).detach())


def _consume_target(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> tuple[Counter[str], int, int]:
    statuses: Counter[str] = Counter()
    staged_windows = 0
    consumed_rows = 0
    for row in _rows(observation, PRESENTED_ROWS):
        result = router.observe(row)
        statuses[result.status] += 1
        if result.status == "staged":
            staged_windows += 1
            consumed_rows += int(result.observation.state.shape[0])
            router.adaptation_step(result, None, replay_evidence=False)
    return statuses, staged_windows, consumed_rows


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    fixtures = {
        name: _fixture(seed, regime)
        for regime, name in enumerate(REGIME_NAMES)
    }
    observations = {name: pair[0] for name, pair in fixtures.items()}
    heldout = {name: pair[1] for name, pair in fixtures.items()}

    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
        aggregation="mean_pool",
    )
    context_loss, context_updates = _train_context_encoder(
        encoder,
        observations,
        seed=seed,
    )
    encoder.eval()
    with torch.no_grad():
        contexts = {
            name: encoder.encode_observation(observation)
            for name, observation in observations.items()
        }

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

    bank = _new_bank()
    source_records: list[dict[str, object]] = []
    prior_digests: dict[int, str] = {}
    source_contexts: list[torch.Tensor] = []
    for name in REGIME_NAMES[:SOURCE_REGIMES]:
        context = contexts[name]
        index = bank.ensure_context(context)
        source_contexts.append(context)
        context_batch = context.unsqueeze(0).expand(observations[name].state.shape[0], -1)
        loss = bank.adaptation_step(observations[name], context_batch, None)
        prior_digests[index] = bank.models[index].digest()
        source_records.append(
            {
                "name": name,
                "slot_index": index,
                "loss": loss,
                "heldout_error": _error(bank, heldout[name], context),
            }
        )

    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=LOSS_THRESHOLD,
        match_margin=0.005,
        continuation_tolerance=LOSS_THRESHOLD,
        provisional_continuation_tolerance=1e9,
        admission_observations=ADMISSION_ROWS,
        max_contexts=len(REGIME_NAMES),
        defer_admission=True,
        candidate_model_families=(RANDOM_FEATURE_FAMILY,),
        provisional_evidence_policy="streaming_statistics",
    )
    target_records: list[dict[str, object]] = []
    target_contexts: dict[str, torch.Tensor] = {}
    all_promoted = True
    all_target_errors_pass = True
    all_prior_retained = True

    for regime in TARGET_REGIMES:
        name = REGIME_NAMES[regime]
        statuses, staged_windows, consumed_rows = _consume_target(
            router,
            observations[name],
        )
        if router.provisional_candidate_count != 1:
            raise RuntimeError(
                f"{name} left {router.provisional_candidate_count} candidates"
            )
        candidate_context = router.provisional_context_at(0)
        target_contexts[name] = candidate_context

        def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
            return all(
                _error(candidate, heldout[old_name], old_context) < LOSS_THRESHOLD
                for old_name, old_context in zip(
                    REGIME_NAMES[:SOURCE_REGIMES], source_contexts, strict=True
                )
            )

        receipt = router.promote_staged_candidate(
            heldout[name],
            retention_probe,
            prediction_tolerance=LOSS_THRESHOLD,
        )
        all_promoted = all_promoted and receipt.accepted
        if not receipt.accepted or receipt.slot_index is None:
            raise RuntimeError(
                f"{name} promotion failed: {receipt.reason}; "
                f"heldout_error={receipt.heldout_error}"
            )
        target_error = _error(bank, heldout[name], candidate_context)
        all_target_errors_pass = all_target_errors_pass and target_error < LOSS_THRESHOLD
        all_prior_retained = all_prior_retained and all(
            bank.models[index].digest() == digest
            for index, digest in prior_digests.items()
        )
        prior_digests[receipt.slot_index] = bank.models[receipt.slot_index].digest()
        target_records.append(
            {
                "name": name,
                "statuses": dict(statuses),
                "staged_windows": staged_windows,
                "presented_rows": PRESENTED_ROWS,
                "consumed_rows": consumed_rows,
                "available_rows": TRAIN_ROWS,
                "slot_index": receipt.slot_index,
                "slot_id": bank.slot_id_at(receipt.slot_index),
                "heldout_error": target_error,
                "promotion_reason": receipt.reason,
            }
        )

    source_return_statuses, _source_return_windows, _source_return_rows = _consume_target(
        router,
        observations[REGIME_NAMES[0]],
    )
    source_return_context = bank.context_at(0)
    source_return_error = _error(
        bank,
        heldout[REGIME_NAMES[0]],
        source_return_context,
    )
    corruption_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    # Give the isolated control one extra destination slot. Otherwise a full
    # production-capacity bank would reject corruption for capacity before its
    # held-out factual rejection gate could run.
    corruption_router.bank.capacity = len(REGIME_NAMES) + 1
    corruption_router.max_contexts = len(REGIME_NAMES) + 1
    bank_before_corruption = corruption_router.bank.content_digest()
    corrupted = ExternalTransitionObservation(
        state=observations[REGIME_NAMES[-1]].state[:PRESENTED_ROWS],
        intention=observations[REGIME_NAMES[-1]].intention[:PRESENTED_ROWS],
        next_state=observations[REGIME_NAMES[-1]].next_state[:PRESENTED_ROWS].roll(1, 0),
        confidence=torch.ones(PRESENTED_ROWS),
    )
    corrupted_statuses, _corrupted_windows, _corrupted_rows = _consume_target(
        corruption_router,
        corrupted,
    )
    rejection = corruption_router.promote_staged_candidate(
        heldout[REGIME_NAMES[-1]],
        lambda _candidate: False,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    corruption_did_not_write = (
        corrupted_statuses["staged"] > 0
        and not rejection.accepted
        and corruption_router.bank.content_digest() == bank_before_corruption
    )
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    gates = {
        "context_encoder_converged": context_loss < 0.05,
        "mean_pool_configuration_persisted": (
            encoder.configuration()["aggregation"] == "mean_pool"
        ),
        "replay_free_model_family": bank.replay_free_updates,
        "partial_evidence_used": all(
            int(row["presented_rows"]) < int(row["available_rows"])
            for row in target_records
        ),
        "all_targets_promoted": all_promoted,
        "all_target_heldout_errors_pass": all_target_errors_pass,
        "all_prior_slots_retained": all_prior_retained,
        "source_return_routes_to_original_slot": source_return_statuses["matched"] >= 1,
        "source_return_error_passes": source_return_error < LOSS_THRESHOLD,
        "corruption_rejected_without_bank_write": corruption_did_not_write,
        "no_raw_candidate_rows_retained": all(
            not candidate.observations for candidate in router._provisional_candidates
        ),
        "controller_unchanged": controller_digest == _digest(controller),
        "exact_router_persistence": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-nonlinear-drift-learned-context-pressure-test.v1",
        "seed": seed,
        "claim_boundary": (
            "bounded replay-free nonlinear drift retention with a frozen, "
            "source-trained permutation-invariant context encoder; not "
            "unrestricted memory growth or general continual learning"
        ),
        "configuration": {
            "regimes": list(REGIME_NAMES),
            "source_regimes": list(REGIME_NAMES[:SOURCE_REGIMES]),
            "target_regimes": list(REGIME_NAMES[SOURCE_REGIMES:]),
            "train_rows": TRAIN_ROWS,
            "presented_rows": PRESENTED_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "context_aggregation": "mean_pool",
            "model_family": RANDOM_FEATURE_FAMILY,
            "policy": "none_external_factual_model_search_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": context_updates,
            "loss": context_loss,
            "digest": encoder.digest(),
        },
        "source": source_records,
        "targets": target_records,
        "return_to_source": {
            "statuses": dict(source_return_statuses),
            "heldout_error": source_return_error,
        },
        "corruption_control": {
            "statuses": dict(corrupted_statuses),
            "promotion_accepted": rejection.accepted,
            "reason": rejection.reason,
        },
        "accounting": {
            "unique_verifier_bits": len(REGIME_NAMES) * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": len(REGIME_NAMES) * (TRAIN_ROWS + HELDOUT_ROWS),
            "context_encoder_optimizer_updates": context_updates,
            "model_statistics_updates": len(REGIME_NAMES[:SOURCE_REGIMES]) + PRESENTED_ROWS // ADMISSION_ROWS * len(TARGET_REGIMES),
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "controller": controller_digest,
            "bank": bank.digest(),
            "router": router.context_encoder.digest(),
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
