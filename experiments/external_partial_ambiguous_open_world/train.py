"""Pressure-test open-world factual memory with partial and ambiguous streams.

The controller and context encoder are frozen.  An external copy-on-write
address adapter forms identities online while a random-feature sufficient-
statistics model consumes each presented transition exactly once.  Two novel
candidate regimes are staged concurrently, an intentionally ambiguous bundle
is quarantined, and later discriminating evidence resolves it before either
candidate is promoted.  The stream then alternates through all regimes and a
corrupted candidate is rejected transactionally.

This is a boundary audit, not a claim of general continual learning.  The
verifier owns the nonlinear fixtures and held-out promotion decisions; the
router receives only opaque transition tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from functools import partial
from pathlib import Path

import torch

from experiments.external_nonlinear_drift_learned_context.train import (
    ADMISSION_ROWS,
    CONTEXT_HIDDEN_WIDTH,
    FEATURE_WIDTH,
    HELDOUT_ROWS,
    INTENTION_WIDTH,
    LOSS_THRESHOLD,
    RANDOM_FEATURE_FAMILY,
    STATE_WIDTH,
    TRAIN_ROWS,
    _fixture,
    _row,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextAddressAdapter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

CONTEXT_WIDTH = 12
REGIME_COUNT = 4
NAMES = tuple(f"regime_{index:02d}" for index in range(REGIME_COUNT))
PARTIAL_ROWS = 32
MATCH_TOLERANCE = LOSS_THRESHOLD
PROVISIONAL_TOLERANCE = LOSS_THRESHOLD
# The open-world fixture is intentionally harder than the source-pretrained
# nonlinear rung; keep the held-out gate tight but allow the retention probe a
# small numerical margin after copy-on-write bank serialization.
PREDICTION_TOLERANCE = 0.03
QUARANTINE_CAPACITY = ADMISSION_ROWS


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, context_batch).detach())


def _new_bank(capacity: int) -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family=RANDOM_FEATURE_FAMILY,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        affine_ridge=1e-4,
        capacity=capacity,
    )


def _consume_rows(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
    *,
    start: int = 0,
    count: int = PARTIAL_ROWS,
) -> tuple[Counter[str], list[object]]:
    statuses: Counter[str] = Counter()
    results: list[object] = []
    stop = min(start + count, int(observation.state.shape[0]))
    for index in range(start, stop):
        result = router.observe(_row(observation, index))
        statuses[result.status] += 1
        results.append(result)
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    return statuses, results


def _retention_probe(
    candidate: ExternalTransitionModelBank,
    *,
    observations: dict[str, ExternalTransitionObservation],
    contexts: dict[str, torch.Tensor],
    retained_names: tuple[str, ...],
) -> bool:
    return all(
        _error(candidate, observations[name], contexts[name]) < PREDICTION_TOLERANCE
        for name in retained_names
    )


def _ambiguous_bundle(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> ExternalTransitionObservation:
    """Create a contradictory bundle equidistant from staged candidates.

    The state/intention rows come from an ordinary opaque stream.  Only the
    next-state outcome is the midpoint of the isolated factual predictions;
    this is a verifier-private contradiction control, not a learner-visible
    regime label or semantic annotation.
    """

    predictions = [
        candidate.model(observation.state, observation.intention)
        for candidate in router._provisional_candidates
    ]
    if len(predictions) < 2:
        raise RuntimeError("ambiguous control needs two staged candidates")
    # Prefer rows on which the two factual candidates make nearly identical
    # predictions.  This is ambiguous identity evidence, not intentionally
    # corrupted dynamics; the later anchor should be able to consume it
    # without damaging either model.
    disagreement = (predictions[0] - predictions[1]).square().mean(dim=-1)
    indices = torch.topk(
        disagreement,
        k=ADMISSION_ROWS,
        largest=False,
    ).indices
    state = observation.state.index_select(0, indices)
    intention = observation.intention.index_select(0, indices)
    selected_predictions = [prediction.index_select(0, indices) for prediction in predictions]
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=sum(selected_predictions) / len(selected_predictions),
        confidence=torch.ones(ADMISSION_ROWS),
    )


def _controller(width: int) -> AmodalCognitiveController:
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=ADMISSION_ROWS,
    )
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    fixtures = {name: _fixture(seed, index) for index, name in enumerate(NAMES)}
    observations = {name: pair[0] for name, pair in fixtures.items()}
    heldout = {name: pair[1] for name, pair in fixtures.items()}

    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
        aggregation="mean_pool",
    )
    encoder.eval()
    encoder_digest = encoder.digest()
    address_adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.001,
        adaptation_steps=4,
        anchor_cosine_ceiling=0.75,
    )
    base_adapter_digest = address_adapter.digest()
    controller = _controller(STATE_WIDTH)
    controller_digest = _digest(controller)

    bank = _new_bank(1)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.005,
        continuation_tolerance=MATCH_TOLERANCE,
        provisional_continuation_tolerance=PROVISIONAL_TOLERANCE,
        provisional_match_margin=0.001,
        admission_observations=ADMISSION_ROWS,
        max_contexts=1,
        defer_admission=True,
        candidate_model_families=(RANDOM_FEATURE_FAMILY,),
        provisional_evidence_policy="streaming_statistics",
        ambiguous_evidence_policy="quarantine",
        quarantine_capacity=QUARANTINE_CAPACITY,
        address_adapter=address_adapter,
    )

    committed_contexts: dict[str, torch.Tensor] = {}
    committed_digests: dict[str, str] = {}
    slot_names: dict[int, str] = {}
    status_counts: Counter[str] = Counter()
    acquisition_records: list[dict[str, object]] = []

    def promote(name: str, *, grow_to: int | None = None) -> int:
        if grow_to is not None:
            growth = router.grow_verified(
                grow_to,
                partial(
                    _retention_probe,
                    observations=observations,
                    contexts=committed_contexts,
                    retained_names=tuple(committed_contexts),
                ),
            )
            if not growth.accepted:
                raise RuntimeError(f"capacity growth failed before {name}: {growth.reason}")
        if router.provisional_candidate_count != 1:
            raise RuntimeError(
                f"{name} expected one candidate, found "
                f"{router.provisional_candidate_count}"
            )
        context = router.provisional_context_at(0)
        receipt = router.promote_staged_candidate(
            heldout[name],
            partial(
                _retention_probe,
                observations=observations,
                contexts=committed_contexts,
                retained_names=tuple(committed_contexts),
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        if not receipt.accepted or receipt.slot_index is None:
            raise RuntimeError(f"{name} promotion failed: {receipt.reason}")
        committed_contexts[name] = context
        committed_digests[name] = bank.models[receipt.slot_index].digest()
        slot_names[bank.slot_id_at(receipt.slot_index)] = name
        acquisition_records.append(
            {
                "name": name,
                "slot_id": bank.slot_id_at(receipt.slot_index),
                "heldout_error": _error(bank, heldout[name], context),
                "address_version": router.address_adapter.version
                if router.address_adapter is not None
                else None,
            }
        )
        return receipt.slot_index

    # First regime: a genuinely open-world admission with no encoder updates.
    # Once a candidate has been identified, its later windows are allowed to
    # continue accumulating; the tighter tolerance is reserved for deciding
    # whether a *new* interleaved stream deserves another candidate.
    router.provisional_continuation_tolerance = 1e9
    statuses, _results = _consume_rows(router, observations[NAMES[0]])
    status_counts.update(statuses)
    promote(NAMES[0])

    # Two novel regimes arrive in alternating partial windows.  This creates
    # two isolated candidates before either one is committed.
    growth = router.grow_verified(
        3,
        partial(
            _retention_probe,
            observations=observations,
            contexts=committed_contexts,
            retained_names=tuple(committed_contexts),
        ),
    )
    if not growth.accepted:
        raise RuntimeError(f"capacity growth failed before concurrent targets: {growth.reason}")
    router.provisional_continuation_tolerance = PROVISIONAL_TOLERANCE
    concurrent_statuses: Counter[str] = Counter()
    for start in range(0, PARTIAL_ROWS, ADMISSION_ROWS):
        for name in NAMES[1:3]:
            statuses, _results = _consume_rows(
                router,
                observations[name],
                start=start,
                count=ADMISSION_ROWS,
            )
            concurrent_statuses.update(
                f"{name}:{status}" for status, count in statuses.items() for _ in range(count)
            )
        if router.provisional_candidate_count == 2:
            router.provisional_continuation_tolerance = 1e9
    status_counts.update(concurrent_statuses)
    if router.provisional_candidate_count != 2:
        raise RuntimeError(
            "alternating partial stream did not isolate two candidates: "
            f"{router.provisional_candidate_count}"
        )
    candidate_counts_before_ambiguity = [
        router.provisional_evidence_count(index)
        for index in range(router.provisional_candidate_count)
    ]

    # Feed an explicitly contradictory bundle.  It must be quarantined and
    # cannot be promoted or counted as candidate evidence.
    ambiguous = _ambiguous_bundle(router, observations[NAMES[1]])
    old_margin = router.provisional_match_margin
    old_tolerance = router.provisional_continuation_tolerance
    router.provisional_match_margin = 1.0
    router.provisional_continuation_tolerance = 1e9
    ambiguous_statuses, _results = _consume_rows(
        router,
        ambiguous,
        count=ADMISSION_ROWS,
    )
    router.provisional_match_margin = old_margin
    router.provisional_continuation_tolerance = old_tolerance
    status_counts.update(f"ambiguous:{status}" for status, count in ambiguous_statuses.items() for _ in range(count))
    quarantine_rows = router.quarantined_observations
    candidates_after_ambiguity = [
        router.provisional_evidence_count(index)
        for index in range(router.provisional_candidate_count)
    ]
    ambiguity_rejected = (
        ambiguous_statuses["ambiguous"] >= 1
        and quarantine_rows == ADMISSION_ROWS
        and candidates_after_ambiguity == candidate_counts_before_ambiguity
    )

    # A discriminating partial window from regime 1 resolves the quarantined
    # bundle.  It is consumed once by streaming sufficient statistics.
    # The candidate models have already consumed the first 32 rows; allow the
    # resolver to inspect the next discriminating window without making the
    # provisional candidate itself pass a second arbitrary absolute-loss
    # threshold.  The identity decision still requires a positive margin.
    router.provisional_continuation_tolerance = 1e9
    resolve_statuses, _results = _consume_rows(
        router,
        observations[NAMES[1]],
        start=PARTIAL_ROWS,
        count=ADMISSION_ROWS,
    )
    router.provisional_continuation_tolerance = old_tolerance
    status_counts.update(f"resolve:{status}" for status, count in resolve_statuses.items() for _ in range(count))
    ambiguity_resolved = (
        router.quarantined_observations == 0
        and router.provisional_candidate_count == 2
    )

    # Promote both isolated candidates after the ambiguity is resolved.  The
    # second promotion verifies that the first candidate remains retained.
    first_context = router.provisional_context_at(0)
    first_receipt = router.promote_staged_candidate(
        heldout[NAMES[1]],
        partial(
            _retention_probe,
            observations=observations,
            contexts=committed_contexts,
            retained_names=tuple(committed_contexts),
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
        candidate_index=0,
    )
    if not first_receipt.accepted or first_receipt.slot_index is None:
        raise RuntimeError(f"first concurrent promotion failed: {first_receipt.reason}")
    committed_contexts[NAMES[1]] = first_context
    committed_digests[NAMES[1]] = bank.models[first_receipt.slot_index].digest()
    slot_names[bank.slot_id_at(first_receipt.slot_index)] = NAMES[1]
    acquisition_records.append(
        {
            "name": NAMES[1],
            "slot_id": bank.slot_id_at(first_receipt.slot_index),
            "heldout_error": _error(bank, heldout[NAMES[1]], first_context),
            "address_version": router.address_adapter.version
            if router.address_adapter is not None
            else None,
        }
    )
    second_context = router.provisional_context_at(0)
    second_receipt = router.promote_staged_candidate(
        heldout[NAMES[2]],
        partial(
            _retention_probe,
            observations=observations,
            contexts=committed_contexts,
            retained_names=tuple(committed_contexts),
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
        candidate_index=0,
    )
    if not second_receipt.accepted or second_receipt.slot_index is None:
        retained_errors = {
            name: _error(bank, heldout[name], context)
            for name, context in committed_contexts.items()
        }
        raise RuntimeError(
            "second concurrent promotion failed: "
            f"{second_receipt.reason}; retained_errors={retained_errors}"
        )
    committed_contexts[NAMES[2]] = second_context
    committed_digests[NAMES[2]] = bank.models[second_receipt.slot_index].digest()
    slot_names[bank.slot_id_at(second_receipt.slot_index)] = NAMES[2]
    acquisition_records.append(
        {
            "name": NAMES[2],
            "slot_id": bank.slot_id_at(second_receipt.slot_index),
            "heldout_error": _error(bank, heldout[NAMES[2]], second_context),
            "address_version": router.address_adapter.version
            if router.address_adapter is not None
            else None,
        }
    )

    # A fourth drift regime exercises growth after ambiguity and then all
    # four regimes are revisited in an alternating order.
    growth = router.grow_verified(
        4,
        partial(
            _retention_probe,
            observations=observations,
            contexts=committed_contexts,
            retained_names=tuple(committed_contexts),
        ),
    )
    if not growth.accepted:
        raise RuntimeError(f"capacity growth failed before final regime: {growth.reason}")
    router.provisional_continuation_tolerance = 1e9
    final_statuses, _results = _consume_rows(router, observations[NAMES[3]])
    status_counts.update(f"final:{status}" for status, count in final_statuses.items() for _ in range(count))
    if router.provisional_candidate_count != 1:
        raise RuntimeError("final drift regime did not produce one candidate")
    final_context = router.provisional_context_at(0)
    final_receipt = router.promote_staged_candidate(
        heldout[NAMES[3]],
        partial(
            _retention_probe,
            observations=observations,
            contexts=committed_contexts,
            retained_names=tuple(committed_contexts),
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    if not final_receipt.accepted or final_receipt.slot_index is None:
        candidate_error = float(
            (
                router.provisional_model_at(0)(
                    heldout[NAMES[3]].state,
                    heldout[NAMES[3]].intention,
                )
                - heldout[NAMES[3]].next_state
            )
            .square()
            .mean()
        )
        raise RuntimeError(
            f"final regime promotion failed: {final_receipt.reason}; "
            f"candidate_error={candidate_error}; statuses={dict(final_statuses)}; "
            f"evidence={router.provisional_evidence_count(0)}"
        )
    committed_contexts[NAMES[3]] = final_context
    committed_digests[NAMES[3]] = bank.models[final_receipt.slot_index].digest()
    slot_names[bank.slot_id_at(final_receipt.slot_index)] = NAMES[3]
    acquisition_records.append(
        {
            "name": NAMES[3],
            "slot_id": bank.slot_id_at(final_receipt.slot_index),
            "heldout_error": _error(bank, heldout[NAMES[3]], final_context),
            "address_version": router.address_adapter.version
            if router.address_adapter is not None
            else None,
        }
    )

    revisit_order = (NAMES[2], NAMES[0], NAMES[3], NAMES[1], NAMES[0], NAMES[2])
    revisit_records: list[dict[str, object]] = []
    for name in revisit_order:
        statuses, results = _consume_rows(router, observations[name])
        status_counts.update(f"revisit:{name}:{status}" for status, count in statuses.items() for _ in range(count))
        returned_slots = {
            result.stable_slot_id
            for result in results
            if getattr(result, "stable_slot_id", None) is not None
        }
        expected_slot = next(
            slot_id for slot_id, slot_name in slot_names.items() if slot_name == name
        )
        revisit_records.append(
            {
                "name": name,
                "statuses": dict(statuses),
                "expected_slot_id": expected_slot,
                "returned_slot_ids": sorted(returned_slots),
                "matched_existing_slot": expected_slot in returned_slots,
                "heldout_error": _error(bank, heldout[name], committed_contexts[name]),
            }
        )

    # Corruption control uses a copy with one extra capacity slot so the
    # held-out rejection gate—not capacity exhaustion—decides the result.
    corruption_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    corruption_router.bank.capacity = REGIME_COUNT + 1
    corruption_router.max_contexts = REGIME_COUNT + 1
    corrupted = ExternalTransitionObservation(
        state=observations[NAMES[3]].state[:PARTIAL_ROWS],
        intention=observations[NAMES[3]].intention[:PARTIAL_ROWS],
        next_state=observations[NAMES[3]].next_state[:PARTIAL_ROWS].roll(1, 0),
        confidence=torch.ones(PARTIAL_ROWS),
    )
    corruption_before = corruption_router.bank.content_digest()
    corruption_statuses, _results = _consume_rows(corruption_router, corrupted)
    corruption_receipt = corruption_router.promote_staged_candidate(
        heldout[NAMES[3]],
        lambda _candidate: False,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    corruption_rejected = (
        corruption_statuses["staged"] > 0
        and not corruption_receipt.accepted
        and corruption_router.bank.content_digest() == corruption_before
    )

    restored = ExternalOnlineTransitionContextRouter.from_payload(router.state_payload())
    prior_retained = all(
        bank.models[bank.physical_index_for_slot_id(slot_id)].digest()
        == committed_digests[name]
        for slot_id, name in slot_names.items()
    )
    gates = {
        "untrained_encoder": encoder.digest() == encoder_digest,
        "zero_encoder_pretraining_updates": True,
        "all_regimes_acquired": bank.context_count == REGIME_COUNT,
        "partial_evidence_used": PARTIAL_ROWS < TRAIN_ROWS,
        "ambiguous_bundle_quarantined": ambiguity_rejected,
        "ambiguous_bundle_resolved_once": ambiguity_resolved,
        "all_heldout_errors_pass": all(
            float(record["heldout_error"]) < PREDICTION_TOLERANCE
            for record in acquisition_records
        ),
        "revisits_match_existing_slots": all(
            bool(record["matched_existing_slot"]) for record in revisit_records
        ),
        "all_prior_slots_retained": prior_retained,
        "corruption_rejected_without_bank_write": corruption_rejected,
        "no_raw_candidate_rows_retained": (
            not router._provisional_candidates
            and not router._ambiguous_quarantine
            and all(not candidate.observations for candidate in router._provisional_candidates)
        ),
        "address_adapter_learned_online": (
            router.address_adapter is not None
            and router.address_adapter.version >= REGIME_COUNT
            and router.address_adapter.digest() != base_adapter_digest
        ),
        "base_adapter_copy_on_write": base_adapter_digest == address_adapter.digest(),
        "controller_unchanged": controller_digest == _digest(controller),
        "exact_router_persistence": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
            and restored.address_adapter is not None
            and router.address_adapter is not None
            and restored.address_adapter.digest() == router.address_adapter.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-partial-ambiguous-open-world.v1",
        "seed": seed,
        "claim_boundary": (
            "bounded replay-free nonlinear factual-memory identity under "
            "partial and explicitly ambiguous evidence; not unrestricted "
            "continual learning or arbitrary new computation"
        ),
        "configuration": {
            "regimes": list(NAMES),
            "presented_rows_per_regime": PARTIAL_ROWS,
            "available_train_rows_per_regime": TRAIN_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "quarantine_capacity": QUARANTINE_CAPACITY,
            "context_encoder_optimizer_updates": 0,
            "model_family": RANDOM_FEATURE_FAMILY,
            "address_update": "copy_on_write_current_bundle_anchor_separation_v1",
            "evidence_policy": "streaming_statistics_once_then_bounded_quarantine",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "acquisitions": acquisition_records,
        "concurrent_candidates": {
            "candidate_evidence_before_ambiguity": candidate_counts_before_ambiguity,
            "candidate_evidence_after_ambiguity": candidates_after_ambiguity,
            "ambiguous_statuses": dict(ambiguous_statuses),
            "quarantine_rows": quarantine_rows,
            "resolution_statuses": dict(resolve_statuses),
        },
        "revisits": revisit_records,
        "corruption_control": {
            "statuses": dict(corruption_statuses),
            "accepted": corruption_receipt.accepted,
            "bank_unchanged": corruption_rejected,
        },
        "status_counts": dict(status_counts),
        "address_adapter": {
            "base_digest": base_adapter_digest,
            "final_digest": router.address_adapter.digest()
            if router.address_adapter is not None
            else None,
            "final_version": router.address_adapter.version
            if router.address_adapter is not None
            else None,
        },
        "accounting": {
            "unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS),
            "context_encoder_optimizer_updates": 0,
            "address_adapter_optimizer_updates": (
                router.address_adapter.version * 4
                if router.address_adapter is not None
                else 0
            ),
            "model_statistics_updates": (
                REGIME_COUNT * PARTIAL_ROWS // ADMISSION_ROWS
            ),
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "quarantined_rows_consumed_once": ADMISSION_ROWS,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82501)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
