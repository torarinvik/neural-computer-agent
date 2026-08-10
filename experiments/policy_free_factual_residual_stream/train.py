"""Audit replay-free growth of an external factual residual bank.

The shared transition model is trained once and then frozen.  Each novel
regime is admitted into an opaque external residual slot after independent
held-out one-step, recursive-rollout, and complete-prefix retention probes.
The experiment also tests reversal isolation, missing/corrupted evidence,
exact persistence, and verifier-selected storage compression.

This is a bounded factual-memory pressure test.  It is not a claim of general
continual learning, unlimited memory growth, or policy learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.policy_free_factual_residual_growth import train as base
from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModel,
    ExternalTransitionObservation,
    ExternalTransitionRollout,
)

STATE_WIDTH = base.STATE_WIDTH
INTENTION_WIDTH = base.INTENTION_WIDTH
CONTEXT_WIDTH = base.CONTEXT_WIDTH
BASE_HIDDEN_WIDTH = base.BASE_HIDDEN_WIDTH
RESIDUAL_HIDDEN_WIDTH = 32
RESIDUAL_FEATURE_WIDTH = 64
BASE_UPDATES = base.BASE_UPDATES
CONTROL_UPDATES = 400
REGIME_COUNT = 6
TOTAL_REGIMES = REGIME_COUNT + 1
TARGET_ERROR_FLOOR = 0.04
SOURCE_RETENTION_FLOOR = base.SOURCE_RETENTION_FLOOR
ROUTER_MATCH_TOLERANCE = 0.005
ROUTER_MATCH_MARGIN = 0.001
COMPRESSION_FLOOR = 0.04


def _regime_transition(
    regime: int,
    state: torch.Tensor,
    intention: torch.Tensor,
) -> torch.Tensor:
    """Create distinct, non-linear factual regimes with a shared base."""

    return base._source_transition(state, intention) + (
        0.35 + 0.04 * regime
    ) * torch.sin((1.2 + 0.18 * regime) * state) + (
        0.2 + 0.02 * (regime % 2)
    ) * intention * state + (regime - 2.5) * 0.22


def _reversal_transition(
    state: torch.Tensor,
    intention: torch.Tensor,
) -> torch.Tensor:
    """A held-out reversal with a deliberately different factual rule."""

    return base._source_transition(state, intention) - (
        0.42 * torch.sin(1.25 * state) + 0.2 * intention * state + 0.50
    )


def _observation(
    state: torch.Tensor,
    intention: torch.Tensor,
    transition,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=state.detach(),
        intention=intention.detach(),
        next_state=transition(state, intention).detach(),
    )


def _rows(
    observation: ExternalTransitionObservation,
) -> list[ExternalTransitionObservation]:
    return [
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
        )
        for index in range(observation.state.shape[0])
    ]


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(repr(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _module_storage_bytes(module: torch.nn.Module) -> int:
    return sum(
        value.detach().numel() * value.element_size()
        for value in module.state_dict().values()
    )


def _regime_data(
    regime: int,
) -> tuple[ExternalTransitionObservation, ExternalTransitionObservation]:
    train_state = torch.linspace(-1.0, 1.0, 16).unsqueeze(-1).repeat_interleave(2, 0)
    train_intention = torch.tensor([[-1.0], [1.0]]).repeat(16, 1)
    heldout_state = torch.tensor(
        [[-0.85], [-0.55], [-0.25], [0.05], [0.35], [0.65], [0.90]]
    ).repeat_interleave(2, 0)
    heldout_intention = torch.tensor([[-0.5], [0.5]]).repeat(7, 1)
    return (
        _observation(
            train_state,
            train_intention,
            lambda state, intention: _regime_transition(
                regime, state, intention
            ),
        ),
        _observation(
            heldout_state,
            heldout_intention,
            lambda state, intention: _regime_transition(
                regime, state, intention
            ),
        ),
    )


def _reversal_data() -> tuple[
    ExternalTransitionObservation,
    ExternalTransitionObservation,
]:
    train_state = torch.linspace(-0.95, 0.95, 16).unsqueeze(-1).repeat_interleave(2, 0)
    train_intention = torch.tensor([[-1.0], [1.0]]).repeat(16, 1)
    heldout_state = torch.tensor(
        [[-0.80], [-0.50], [-0.20], [0.10], [0.40], [0.70], [0.88]]
    ).repeat_interleave(2, 0)
    heldout_intention = torch.tensor([[-0.5], [0.5]]).repeat(7, 1)
    return (
        _observation(train_state, train_intention, _reversal_transition),
        _observation(heldout_state, heldout_intention, _reversal_transition),
    )


def _fit_base(
    seed: int,
    source_train: ExternalTransitionObservation,
) -> tuple[ExternalTransitionModel, int]:
    torch.manual_seed(seed)
    model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _update in range(BASE_UPDATES):
        optimizer.zero_grad()
        loss = model.loss(source_train)
        loss.backward()
        optimizer.step()
    return model, BASE_UPDATES


def _new_router(
    source_base: ExternalTransitionModel,
    seed: int,
) -> ExternalFactoredTransitionRouter:
    model = ExternalFactoredTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_hidden_width=RESIDUAL_HIDDEN_WIDTH,
        residual_random_feature_width=RESIDUAL_FEATURE_WIDTH,
        residual_random_feature_seed=seed,
        residual_ridge=0.1,
        residual_capacity=1,
    )
    model.base.load_state_dict(source_base.state_dict(), strict=True)
    model.freeze_base()
    return ExternalFactoredTransitionRouter(
        model,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            INTENTION_WIDTH,
            hidden_width=16,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=ROUTER_MATCH_TOLERANCE,
        match_margin=ROUTER_MATCH_MARGIN,
        residual_adaptation_updates=1,
        max_contexts=1,
        auto_grow=True,
    )


def _rollout(
    regime: int,
    *,
    reversal: bool = False,
) -> ExternalTransitionRollout:
    initial = torch.tensor([0.05])
    intentions = torch.tensor([[-0.5], [0.5]])
    transition = _reversal_transition if reversal else lambda state, action: _regime_transition(
        regime, state, action
    )
    states: list[torch.Tensor] = []
    current = initial.unsqueeze(0)
    for index in range(intentions.shape[0]):
        current = transition(current, intentions[index : index + 1])
        states.append(current.squeeze(0))
    return ExternalTransitionRollout(
        initial_state=initial,
        intentions=intentions,
        expected_states=torch.stack(states),
    )


def _residual_observation(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
) -> ExternalTransitionObservation:
    with torch.no_grad():
        next_state = observation.next_state - model.base(
            observation.state, observation.intention
        )
    return ExternalTransitionObservation(
        state=observation.state,
        intention=observation.intention,
        next_state=next_state,
    )


def _train_fresh_control(
    *,
    seed: int,
    source_heldout: ExternalTransitionObservation,
    target_train: ExternalTransitionObservation,
    target_heldout: ExternalTransitionObservation,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _update in range(CONTROL_UPDATES):
        optimizer.zero_grad()
        loss = model.loss(target_train)
        loss.backward()
        optimizer.step()
    return {
        "optimizer_updates": CONTROL_UPDATES,
        "replayed_examples": CONTROL_UPDATES * int(target_train.state.shape[0]),
        "target_error": float(model.loss(target_heldout).detach()),
        "source_error": float(model.loss(source_heldout).detach()),
        "parameter_bytes": _module_storage_bytes(model),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    source_train, source_heldout, _, _ = base._base_data()
    source_base, base_updates = _fit_base(seed, source_train)
    base_digest = _digest_module(source_base)
    source_error_before = float(source_base.loss(source_heldout).detach())
    router = _new_router(source_base, seed + 10_000)

    records: list[dict[str, object]] = []
    retention_probe_errors: list[dict[str, object]] = []
    fresh_controls: list[dict[str, object]] = []

    for regime in range(REGIME_COUNT):
        train_observation, heldout_observation = _regime_data(regime)
        staging = router.route_bundle(_rows(train_observation))
        if staging.status != "staged":
            raise AssertionError(f"regime {regime} did not stage: {staging}")
        prior_records = list(records)

        def retention_probe(
            candidate: ExternalFactoredTransitionModel,
            *,
            prior_records: list[dict[str, object]] = prior_records,
            regime: int = regime,
        ) -> bool:
            if float(candidate.base.loss(source_heldout).detach()) > SOURCE_RETENTION_FLOOR:
                return False
            for prior in prior_records:
                context = prior["context"]
                heldout = prior["heldout"]
                assert isinstance(context, torch.Tensor)
                assert isinstance(heldout, ExternalTransitionObservation)
                context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
                error = float(candidate.loss(heldout, context=context_batch).detach())
                retention_probe_errors.append(
                    {"regime": regime, "retained_regime": prior["regime"], "error": error}
                )
                if error > TARGET_ERROR_FLOOR:
                    return False
            return True

        receipt = router.promote_staged_candidate(
            heldout_observation,
            retention_probe,
            prediction_tolerance=TARGET_ERROR_FLOOR,
            heldout_rollout=_rollout(regime),
            rollout_error_tolerance=TARGET_ERROR_FLOOR,
        )
        if not receipt.accepted or receipt.slot_id is None:
            raise AssertionError(f"regime {regime} was rejected: {receipt}")
        records.append(
            {
                "regime": regime,
                "slot_id": receipt.slot_id,
                "context": staging.context,
                "heldout": heldout_observation,
                "train": train_observation,
                "train_rows": int(train_observation.state.shape[0]),
                "heldout_rows": int(heldout_observation.state.shape[0]),
                "heldout_error": receipt.heldout_error,
                "rollout_error": receipt.heldout_rollout_error,
            }
        )
        fresh_controls.append(
            _train_fresh_control(
                seed=seed + 40_000 + regime,
                source_heldout=source_heldout,
                target_train=train_observation,
                target_heldout=heldout_observation,
            )
        )

    reversal_train, reversal_heldout = _reversal_data()
    shuffled_router = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    permutation = torch.randperm(
        reversal_train.next_state.shape[0],
        generator=torch.Generator().manual_seed(seed + 30_000),
    )
    shuffled_train = ExternalTransitionObservation(
        state=reversal_train.state,
        intention=reversal_train.intention,
        next_state=reversal_train.next_state[permutation],
    )
    shuffled_before = shuffled_router.digest()
    shuffled_staging = shuffled_router.route_bundle(_rows(shuffled_train))
    shuffled_receipt = None
    if shuffled_staging.status == "staged":
        shuffled_receipt = shuffled_router.promote_staged_candidate(
            reversal_heldout,
            lambda candidate: float(candidate.base.loss(source_heldout).detach())
            <= SOURCE_RETENTION_FLOOR,
            prediction_tolerance=TARGET_ERROR_FLOOR,
        )
    shuffled_after = shuffled_router.digest()

    live_staging = router.route_bundle(_rows(reversal_train))
    if live_staging.status != "staged":
        raise AssertionError(f"reversal did not stage: {live_staging}")

    def reversal_retention_probe(candidate: ExternalFactoredTransitionModel) -> bool:
        if float(candidate.base.loss(source_heldout).detach()) > SOURCE_RETENTION_FLOOR:
            return False
        for prior in records:
            context = prior["context"]
            heldout = prior["heldout"]
            assert isinstance(context, torch.Tensor)
            assert isinstance(heldout, ExternalTransitionObservation)
            context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
            if float(candidate.loss(heldout, context=context_batch).detach()) > TARGET_ERROR_FLOOR:
                return False
        return True

    reversal_receipt = router.promote_staged_candidate(
        reversal_heldout,
        reversal_retention_probe,
        prediction_tolerance=TARGET_ERROR_FLOOR,
        heldout_rollout=_rollout(0, reversal=True),
        rollout_error_tolerance=TARGET_ERROR_FLOOR,
    )
    if not reversal_receipt.accepted or reversal_receipt.slot_id is None:
        raise AssertionError(f"reversal was rejected: {reversal_receipt}")
    records.append(
        {
            "regime": "reversal",
            "slot_id": reversal_receipt.slot_id,
            "context": live_staging.context,
            "heldout": reversal_heldout,
            "train": reversal_train,
            "train_rows": int(reversal_train.state.shape[0]),
            "heldout_rows": int(reversal_heldout.state.shape[0]),
            "heldout_error": reversal_receipt.heldout_error,
            "rollout_error": reversal_receipt.heldout_rollout_error,
        }
    )

    route_roundtrip_slots: list[int | None] = []
    for record in records:
        train_observation = record["train"]
        assert isinstance(train_observation, ExternalTransitionObservation)
        route_result = router.route_bundle(_rows(train_observation))
        route_roundtrip_slots.append(route_result.slot_id)

    prefix_errors: list[float] = []
    for record in records:
        context = record["context"]
        heldout = record["heldout"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(heldout, ExternalTransitionObservation)
        context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
        prefix_errors.append(
            float(router.model.loss(heldout, context=context_batch).detach())
        )

    before_missing = router.digest()
    missing_result = router.route_partial_bundle([])
    after_missing = router.digest()

    corrupt_record = records[2]
    corrupt_heldout = corrupt_record["heldout"]
    assert isinstance(corrupt_heldout, ExternalTransitionObservation)
    corrupted = ExternalTransitionObservation(
        state=corrupt_heldout.state[:1],
        intention=corrupt_heldout.intention[:1],
        next_state=corrupt_heldout.next_state[:1] + 2.0,
    )
    before_corruption = router.digest()
    corruption_result = router.route_partial_bundle(
        [corrupted], match_tolerance=ROUTER_MATCH_TOLERANCE
    )
    after_corruption = router.digest()

    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    restored_prefix_errors: list[float] = []
    for record in records:
        context = record["context"]
        heldout = record["heldout"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(heldout, ExternalTransitionObservation)
        context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
        restored_prefix_errors.append(
            float(restored.model.loss(heldout, context=context_batch).detach())
        )

    residual_bank = router.model.residual_bank
    if residual_bank is None:
        raise AssertionError("residual bank was not created")

    def compression_probe(candidate_bank) -> bool:
        for record in records:
            context = record["context"]
            heldout = record["heldout"]
            assert isinstance(context, torch.Tensor)
            assert isinstance(heldout, ExternalTransitionObservation)
            residual = _residual_observation(router.model, heldout)
            context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
            if float(candidate_bank.loss(residual, context_batch).detach()) > COMPRESSION_FLOOR:
                return False
        return True

    compression = router.select_compression_verified(
        (torch.float16, "int4"), retention_probe=compression_probe
    )
    digest_before_controls = router.digest()

    gates = {
        "all_regimes_promoted": len(records) == TOTAL_REGIMES,
        "all_one_step_passed": all(
            float(record["heldout_error"]) <= TARGET_ERROR_FLOOR for record in records
        ),
        "all_recursive_rollouts_passed": all(
            record["rollout_error"] is not None
            and float(record["rollout_error"]) <= TARGET_ERROR_FLOOR
            for record in records
        ),
        "opaque_route_roundtrip_passed": route_roundtrip_slots == [
            int(record["slot_id"]) for record in records
        ],
        "complete_prefix_retention_passed": max(prefix_errors) <= TARGET_ERROR_FLOOR,
        "source_retention_passed": float(router.model.base.loss(source_heldout).detach())
        <= SOURCE_RETENTION_FLOOR,
        "base_frozen": router.model.base_frozen,
        "base_byte_stable": _digest_module(router.model.base) == base_digest,
        "shuffled_reversal_not_promoted": (
            shuffled_receipt is None or not shuffled_receipt.accepted
        )
        and shuffled_before == shuffled_after,
        "missing_evidence_is_noop": (
            missing_result.status == "ambiguous" and before_missing == after_missing
        ),
        "corruption_is_quarantined_or_rejected": (
            corruption_result.status in {"ambiguous", "reliability_veto"}
            and before_corruption == after_corruption
        ),
        "exact_persistence": (
            restored.digest() == router.digest()
            and max(
                abs(left - right)
                for left, right in zip(prefix_errors, restored_prefix_errors, strict=True)
            )
            <= 1e-9
        ),
        "one_pass_residual_accounting": all(
            int(residual_bank.models[residual_bank.physical_index_for_slot_id(int(record["slot_id"]))].sample_count.item())
            == int(record["train_rows"])
            for record in records
        ),
        "compression_verified": compression.accepted,
        "compression_reduces_storage": any(
            receipt.accepted and receipt.compressed_bytes < receipt.source_bytes
            for receipt in compression.receipts
        ),
        "fresh_controls_accounted": all(
            control["optimizer_updates"] == CONTROL_UPDATES
            and control["replayed_examples"] > 0
            for control in fresh_controls
        ),
        "committed_state_unchanged_by_probes": digest_before_controls == router.digest(),
    }
    report = {
        "schema": "neural-computer.policy-free-factual-residual-stream.v1",
        "claim_boundary": (
            "one frozen shared transition model plus a bounded seven-slot opaque "
            "one-pass factual residual bank with held-out prefix retention and "
            "verified compression; not general continual learning or unrestricted growth"
        ),
        "seed": seed,
        "configuration": {
            "base_updates": BASE_UPDATES,
            "control_updates": CONTROL_UPDATES,
            "regime_count": REGIME_COUNT,
            "total_regimes_including_reversal": TOTAL_REGIMES,
            "target_error_floor": TARGET_ERROR_FLOOR,
            "source_retention_floor": SOURCE_RETENTION_FLOOR,
            "compression_floor": COMPRESSION_FLOOR,
            "residual_feature_width": RESIDUAL_FEATURE_WIDTH,
            "residual_ridge": 0.1,
            "residual": router.model.configuration(),
            "routing": router.configuration(),
            "admission": "heldout_one_step_plus_recursive_rollout_plus_complete_prefix_retention_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "base_source_error_before": source_error_before,
            "base_source_error_after": float(router.model.base.loss(source_heldout).detach()),
            "prefix_errors": prefix_errors,
            "restored_prefix_errors": restored_prefix_errors,
            "retention_probe_errors": retention_probe_errors,
            "records": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"context", "heldout", "train"}
                }
                for record in records
            ],
            "route_roundtrip_slots": route_roundtrip_slots,
            "route_comparisons_for_novel_bundles": sum(range(TOTAL_REGIMES)),
            "shuffled_staging_status": shuffled_staging.status,
            "shuffled_receipt": (
                None
                if shuffled_receipt is None
                else {
                    "accepted": shuffled_receipt.accepted,
                    "heldout_error": shuffled_receipt.heldout_error,
                    "reason": shuffled_receipt.reason,
                }
            ),
            "missing_result_status": missing_result.status,
            "corruption_result_status": corruption_result.status,
            "compression": {
                "accepted": compression.accepted,
                "selected_codec": compression.selected_codec,
                "reason": compression.reason,
                "receipts": [
                    {
                        "codec": receipt.codec,
                        "accepted": receipt.accepted,
                        "source_bytes": receipt.source_bytes,
                        "compressed_bytes": receipt.compressed_bytes,
                        "reason": receipt.reason,
                    }
                    for receipt in compression.receipts
                ],
            },
            "residual_sample_counts": [
                int(
                    residual_bank.models[
                        residual_bank.physical_index_for_slot_id(int(record["slot_id"]))
                    ].sample_count.item()
                )
                for record in records
            ],
            "residual_bank_storage_bytes": _module_storage_bytes(residual_bank),
            "shared_base_storage_bytes": _module_storage_bytes(source_base),
            "fresh_controls": fresh_controls,
        },
        "accounting": {
            "base_optimizer_updates": base_updates,
            "residual_optimizer_updates": 0,
            "residual_unique_transition_rows": sum(int(record["train_rows"]) for record in records),
            "residual_heldout_transition_rows": sum(int(record["heldout_rows"]) for record in records),
            "residual_rollout_transition_rows": sum(
                _rollout(0, reversal=record["regime"] == "reversal").horizon
                for record in records
            ),
            "logical_lifetimes": len(records),
            "residual_replayed_examples": 0,
            "fresh_optimizer_updates": sum(int(control["optimizer_updates"]) for control in fresh_controls),
            "fresh_replayed_examples": sum(int(control["replayed_examples"]) for control in fresh_controls),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
