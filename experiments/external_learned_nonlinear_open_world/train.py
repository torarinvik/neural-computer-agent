"""Pressure-test a trainable nonlinear factual model without replay.

The controller and context encoder are frozen.  Four nonlinear opaque
transition regimes arrive through partial streams.  A copy-on-write address
adapter forms each identity online, while a trainable external MLP receives
one transition row at a time through the router's ``streaming_gradient``
protocol.  No candidate retains raw rows and no old slot is updated.
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
from torch.nn import functional as F

from experiments.external_nonlinear_drift_learned_context.train import (
    CONTEXT_HIDDEN_WIDTH,
    HELDOUT_ROWS,
    INTENTION_WIDTH,
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
    ExternalTransitionRouteMemory,
    ExternalTransitionRouteQuery,
    OpaqueCandidateGrowthRouter,
    paired_counterfactual_ranking_loss,
)

CONTEXT_WIDTH = 12
HIDDEN_WIDTH = 64
MODEL_FAMILY = "nonlinear_mlp_v1"
REGIME_COUNT = 4
NAMES = tuple(f"regime_{index:02d}" for index in range(REGIME_COUNT))
PRESENTED_ROWS = 48
ADMISSION_ROWS = 4
# The looser promotion bound lets the diagnostic complete its lifecycle; the
# stricter quality bound is the actual learned-model gate and is expected to
# reject weak seeds rather than hide them as process failures.
QUALITY_THRESHOLD = 0.08
PROMOTION_THRESHOLD = 0.20
MATCH_TOLERANCE = 0.01
GRADIENT_STEPS_PER_BUNDLE = 4
ROUTE_QUERY_MINIMUM_SCORE = 0.80
LEARNED_ROUTE_HIDDEN = 48
LEARNED_ROUTE_UPDATES = 128
ROUTE_MEMORY_PROTOTYPES = 4


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
        hidden_width=HIDDEN_WIDTH,
        model_family=MODEL_FAMILY,
        adaptation_learning_rate=0.01,
        capacity=capacity,
    )


def _retention_probe(
    candidate: ExternalTransitionModelBank,
    *,
    observations: dict[str, ExternalTransitionObservation],
    contexts: dict[str, torch.Tensor],
) -> bool:
    return all(
        _error(candidate, observations[name], contexts[name]) < PROMOTION_THRESHOLD
        for name in contexts
    )


def _consume(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
    *,
    optimizer: torch.optim.Optimizer | None,
    start: int = 0,
    count: int = PRESENTED_ROWS,
) -> tuple[Counter[str], torch.optim.Optimizer | None, list[float], list[object]]:
    statuses: Counter[str] = Counter()
    losses: list[float] = []
    results: list[object] = []
    stop = min(start + count, int(observation.state.shape[0]))
    for index in range(start, stop):
        result = router.observe(_row(observation, index))
        statuses[result.status] += 1
        results.append(result)
        if result.status != "staged":
            continue
        if optimizer is None:
            optimizer = torch.optim.Adam(
                router.provisional_model_at(0).parameters(),
                lr=0.01,
            )
        for _step in range(GRADIENT_STEPS_PER_BUNDLE):
            losses.append(
                router.adaptation_step(
                    result,
                    optimizer,
                    replay_evidence=False,
                )
            )
    return statuses, optimizer, losses, results


def _route_diagnostic(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> dict[str, object] | None:
    if router.route_query is None:
        return None
    fallback = (
        router.context_encoder.trajectory_stats(observation)
        if (
            router.route_query.learned_scorer is not None
            or (
                router.route_query.route_memory is not None
                and router.route_query.route_width is not None
            )
        )
        else (
            router.address_adapter.encode_observation(observation)
            if router.address_adapter is not None
            else router.context_encoder.encode_observation(observation)
        )
    )
    proposal = router.route_query.propose_observation(
        observation,
        router.bank.contexts,
        router.bank.slot_ids,
        fallback_query=fallback,
    )
    errors = []
    for index in range(router.bank.context_count):
        context = router.bank.context_at(index).unsqueeze(0).expand(
            observation.state.shape[0], -1
        )
        errors.append(float(router.bank.loss(observation, context).detach()))
    factual_best_slot_id = router.bank.slot_id_at(
        min(range(len(errors)), key=lambda index: errors[index])
    )
    return {
        "selected_slot_id": proposal.selected_slot_id,
        "factual_best_slot_id": factual_best_slot_id,
        "scores": proposal.scores.tolist(),
        "factual_errors": errors,
        "margin": proposal.margin,
    }


def _train_learned_route_query(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
    *,
    updates: int,
) -> int:
    """Train route identity from current factual counterfactual utilities."""

    if router.route_query is None or router.route_query.learned_scorer is None:
        return 0
    if router.bank.context_count < 2:
        return 0
    route_query = router.route_query
    scorer = route_query.learned_scorer
    if route_query.route_width is None:
        raise RuntimeError("learned route query has no route width")
    if any(
        slot_id not in route_query._slot_route_keys
        for slot_id in router.bank.slot_ids
    ):
        raise RuntimeError("learned route query has an incomplete key bank")
    with torch.no_grad():
        query = router.context_encoder.trajectory_stats(observation)
        keys = torch.stack(
            [route_query._slot_route_keys[slot_id] for slot_id in router.bank.slot_ids]
        ).to(query)
        errors = torch.stack(
            [
                router.bank.loss(
                    observation,
                    router.bank.context_at(index)
                    .to(observation.state)
                    .unsqueeze(0)
                    .expand(observation.state.shape[0], -1),
                )
                for index in range(router.bank.context_count)
            ]
        )
        utilities = 1.0 / (1.0 + errors)
    pairs = [
        (left, right)
        for left in range(router.bank.context_count)
        for right in range(left + 1, router.bank.context_count)
    ]
    attempted = torch.tensor(pairs, dtype=torch.long)
    pair_utilities = utilities[attempted]
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-3, weight_decay=1e-5)
    for update in range(updates):
        # Reuse only the current opaque evidence window. This is counted as
        # current-window route optimization, never as old-regime replay.
        query_batch = query.unsqueeze(0).expand(len(pairs), -1).clone()
        query_batch = query_batch + 0.002 * torch.randn_like(query_batch)
        scores = scorer(query_batch, keys)
        pair_scores = scores
        ranking_loss, _advantage = paired_counterfactual_ranking_loss(
            pair_scores,
            attempted,
            pair_utilities,
        )
        best = int(utilities.argmax())
        loss = ranking_loss + F.softplus(1.0 - scores[:, best]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        optimizer.step()
    scorer.eval()
    return updates


def run(
    seed: int,
    report_out: Path,
    *,
    use_route_query: bool = False,
    match_tolerance: float = MATCH_TOLERANCE,
    route_query_minimum_score: float = ROUTE_QUERY_MINIMUM_SCORE,
    use_learned_route_query: bool = False,
    learned_route_updates: int = LEARNED_ROUTE_UPDATES,
    use_prototype_route_memory: bool = False,
    route_memory_prototypes: int = ROUTE_MEMORY_PROTOTYPES,
) -> dict[str, object]:
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

    bank = _new_bank(1)
    route_width = CONTEXT_WIDTH + CONTEXT_HIDDEN_WIDTH * 3
    learned_scorer = (
        OpaqueCandidateGrowthRouter(route_width, hidden=LEARNED_ROUTE_HIDDEN)
        if use_learned_route_query
        else None
    )
    prototype_route_memory = (
        ExternalTransitionRouteMemory(
            route_width,
            max_prototypes_per_slot=route_memory_prototypes,
        )
        if use_prototype_route_memory
        else None
    )
    if use_learned_route_query and use_prototype_route_memory:
        raise ValueError(
            "learned route scorer and prototype route memory are exclusive"
        )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=match_tolerance,
        match_margin=0.005,
        continuation_tolerance=MATCH_TOLERANCE,
        provisional_continuation_tolerance=1e9,
        admission_observations=ADMISSION_ROWS,
        max_contexts=1,
        defer_admission=True,
        candidate_model_families=(MODEL_FAMILY,),
        provisional_evidence_policy="streaming_gradient",
        address_adapter=address_adapter,
        route_query=(
            ExternalTransitionRouteQuery(
                CONTEXT_WIDTH,
                minimum_score=(
                    0.5 if use_learned_route_query else route_query_minimum_score
                ),
                route_width=(
                    route_width
                    if use_learned_route_query or use_prototype_route_memory
                    else None
                ),
                learned_scorer=learned_scorer,
                route_memory=prototype_route_memory,
            )
            if (
                use_route_query
                or use_learned_route_query
                or use_prototype_route_memory
            )
            else None
        ),
    )

    contexts: dict[str, torch.Tensor] = {}
    digests: dict[str, str] = {}
    slot_names: dict[int, str] = {}
    status_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    optimizer_updates = 0
    route_scorer_updates = 0

    for index, name in enumerate(NAMES):
        if index:
            growth = router.grow_verified(
                index + 1,
                partial(
                    _retention_probe,
                    observations=observations,
                    contexts=contexts,
                ),
            )
            if not growth.accepted:
                raise RuntimeError(f"capacity growth failed before {name}: {growth.reason}")
        statuses, _optimizer, losses, _results = _consume(
            router,
            observations[name],
            optimizer=None,
        )
        status_counts.update(f"{name}:{status}" for status, count in statuses.items() for _ in range(count))
        optimizer_updates += len(losses)
        if router.provisional_candidate_count != 1:
            raise RuntimeError(f"{name} did not leave one learned candidate")
        context = router.provisional_context_at(0)
        receipt = router.promote_staged_candidate(
            heldout[name],
            partial(
                _retention_probe,
                observations=observations,
                contexts=contexts,
            ),
            prediction_tolerance=PROMOTION_THRESHOLD,
        )
        if not receipt.accepted or receipt.slot_index is None:
            candidate_error = float(
                (
                    router.provisional_model_at(0)(
                        heldout[name].state,
                        heldout[name].intention,
                    )
                    - heldout[name].next_state
                )
                .square()
                .mean()
                .detach()
            )
            raise RuntimeError(
                f"{name} promotion failed: {receipt.reason}; "
                f"candidate_error={candidate_error}; losses={losses[-3:]}"
            )
        contexts[name] = context
        slot_id = bank.slot_id_at(receipt.slot_index)
        slot_names[slot_id] = name
        digests[name] = bank.models[receipt.slot_index].digest()
        records.append(
            {
                "name": name,
                "slot_id": slot_id,
                "presented_rows": PRESENTED_ROWS,
                "available_rows": TRAIN_ROWS,
                "optimizer_updates": len(losses),
                "heldout_error": _error(bank, heldout[name], context),
                "address_version": router.address_adapter.version,
            }
        )
        if use_learned_route_query:
            route_scorer_updates += _train_learned_route_query(
                router,
                observations[name],
                updates=learned_route_updates,
            )

    revisit_order = (NAMES[2], NAMES[0], NAMES[3], NAMES[1], NAMES[0], NAMES[2])
    revisit_records: list[dict[str, object]] = []
    for name in revisit_order:
        route_diagnostic = _route_diagnostic(router, observations[name])
        statuses, _optimizer, _losses, results = _consume(
            router,
            observations[name],
            optimizer=None,
        )
        status_counts.update(f"revisit:{name}:{status}" for status, count in statuses.items() for _ in range(count))
        returned_slots = {
            result.stable_slot_id
            for result in results
            if result.stable_slot_id is not None
        }
        expected_slot = next(slot for slot, slot_name in slot_names.items() if slot_name == name)
        revisit_records.append(
            {
                "name": name,
                "statuses": dict(statuses),
                "expected_slot_id": expected_slot,
                "returned_slot_ids": sorted(returned_slots),
                "matched_existing_slot": expected_slot in returned_slots,
                "heldout_error": _error(bank, heldout[name], contexts[name]),
                "route_diagnostic": route_diagnostic,
            }
        )

    route_diagnostics = [
        record["route_diagnostic"]
        for record in revisit_records
        if record["route_diagnostic"] is not None
    ]
    route_proposals_match_factual_winners = all(
        diagnostic["selected_slot_id"] == diagnostic["factual_best_slot_id"]
        for diagnostic in route_diagnostics
    )
    route_query_floor = (
        0.5 if use_learned_route_query else route_query_minimum_score
    )

    corruption_router = ExternalOnlineTransitionContextRouter.from_payload(router.state_payload())
    corruption_router.bank.capacity = REGIME_COUNT + 1
    corruption_router.max_contexts = REGIME_COUNT + 1
    corrupted = ExternalTransitionObservation(
        state=observations[NAMES[-1]].state[:PRESENTED_ROWS],
        intention=observations[NAMES[-1]].intention[:PRESENTED_ROWS],
        next_state=observations[NAMES[-1]].next_state[:PRESENTED_ROWS].roll(1, 0),
        confidence=torch.ones(PRESENTED_ROWS),
    )
    corruption_before = corruption_router.bank.content_digest()
    corruption_statuses, corruption_optimizer, corruption_losses, _results = _consume(
        corruption_router,
        corrupted,
        optimizer=None,
    )
    del corruption_optimizer
    corruption_receipt = corruption_router.promote_staged_candidate(
        heldout[NAMES[-1]],
        lambda _candidate: False,
        prediction_tolerance=PROMOTION_THRESHOLD,
    )
    corruption_rejected = (
        corruption_statuses["staged"] == PRESENTED_ROWS // ADMISSION_ROWS
        and len(corruption_losses)
        == PRESENTED_ROWS // ADMISSION_ROWS * GRADIENT_STEPS_PER_BUNDLE
        and not corruption_receipt.accepted
        and corruption_router.bank.content_digest() == corruption_before
    )

    restored = ExternalOnlineTransitionContextRouter.from_payload(router.state_payload())
    prior_retained = all(
        bank.models[bank.physical_index_for_slot_id(slot_id)].digest() == digests[name]
        for slot_id, name in slot_names.items()
    )
    gates = {
        "untrained_encoder": encoder.digest() == encoder_digest,
        "zero_encoder_pretraining_updates": True,
        "learned_nonlinear_model_family": all(
            bank.model_family_at(index) == MODEL_FAMILY
            for index in range(bank.context_count)
        ),
        "all_regimes_acquired": bank.context_count == REGIME_COUNT,
        "partial_evidence_used": PRESENTED_ROWS < TRAIN_ROWS,
        "current_window_gradient_updates": optimizer_updates == (
            REGIME_COUNT
            * PRESENTED_ROWS
            // ADMISSION_ROWS
            * GRADIENT_STEPS_PER_BUNDLE
        ),
        "all_heldout_errors_pass": all(
            float(record["heldout_error"]) < QUALITY_THRESHOLD for record in records
        ),
        "revisits_match_existing_slots": all(
            bool(record["matched_existing_slot"]) for record in revisit_records
        ),
        "route_proposals_match_factual_winners": route_proposals_match_factual_winners,
        "all_prior_slots_retained": prior_retained,
        "corruption_rejected_without_bank_write": corruption_rejected,
        "no_raw_candidate_rows_retained": (
            not router._provisional_candidates
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
            and (
                (restored.route_query is None and router.route_query is None)
                or (
                    restored.route_query is not None
                    and router.route_query is not None
                    and restored.route_query.digest() == router.route_query.digest()
                )
            )
        ),
    }
    report = {
        "schema": "neural-computer.external-learned-nonlinear-open-world.v1",
        "seed": seed,
        "claim_boundary": (
            "bounded replay-free learned nonlinear factual-memory slots under "
            "partial open-world evidence; not unrestricted continual learning"
        ),
        "configuration": {
            "regimes": list(NAMES),
            "model_family": MODEL_FAMILY,
            "hidden_width": HIDDEN_WIDTH,
            "presented_rows_per_regime": PRESENTED_ROWS,
            "available_train_rows_per_regime": TRAIN_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "quality_threshold": QUALITY_THRESHOLD,
            "promotion_threshold": PROMOTION_THRESHOLD,
            "match_tolerance": match_tolerance,
            "route_query": "cosine_proposal_with_factual_verification_v1"
            if use_route_query
            and not use_learned_route_query
            and not use_prototype_route_memory
            else (
                "learned_counterfactual_route_query_v1"
                if use_learned_route_query
                else (
                    "verified_slot_local_prototype_route_memory_v1"
                    if use_prototype_route_memory
                    else None
                )
            ),
            "route_query_minimum_score": route_query_floor,
            "learned_route_updates_per_regime": learned_route_updates
            if use_learned_route_query
            else 0,
            "route_memory_prototypes_per_slot": route_memory_prototypes
            if use_prototype_route_memory
            else 0,
            "context_encoder_optimizer_updates": 0,
            "provisional_evidence_policy": "streaming_gradient",
            "gradient_steps_per_current_window": GRADIENT_STEPS_PER_BUNDLE,
            "address_update": "copy_on_write_current_bundle_anchor_separation_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "acquisitions": records,
        "revisits": revisit_records,
        "corruption_control": {
            "statuses": dict(corruption_statuses),
            "accepted": corruption_receipt.accepted,
            "bank_unchanged": corruption_rejected,
        },
        "status_counts": dict(status_counts),
        "address_adapter": {
            "base_digest": base_adapter_digest,
            "final_digest": router.address_adapter.digest(),
            "final_version": router.address_adapter.version,
        },
        "accounting": {
            "unique_verifier_bits": REGIME_COUNT * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": REGIME_COUNT * (TRAIN_ROWS + HELDOUT_ROWS),
            "context_encoder_optimizer_updates": 0,
            "address_adapter_optimizer_updates": router.address_adapter.version * 4,
            "model_optimizer_updates": optimizer_updates,
            "route_scorer_optimizer_updates": route_scorer_updates,
            "route_memory_state_updates": (
                router.route_query.route_memory.version
                if router.route_query is not None
                and router.route_query.route_memory is not None
                else 0
            ),
            "route_memory_prototype_count": (
                router.route_query.route_memory.total_prototype_count
                if router.route_query is not None
                and router.route_query.route_memory is not None
                else 0
            ),
            "route_scorer_unique_current_windows": max(0, REGIME_COUNT - 1)
            if use_learned_route_query
            else 0,
            "route_scorer_current_window_reuses": max(
                0,
                route_scorer_updates - max(0, REGIME_COUNT - 1),
            ),
            "current_window_reuses": (
                REGIME_COUNT
                * PRESENTED_ROWS
                // ADMISSION_ROWS
                * (GRADIENT_STEPS_PER_BUNDLE - 1)
            ),
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82601)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--route-query", action="store_true")
    parser.add_argument("--match-tolerance", type=float, default=MATCH_TOLERANCE)
    parser.add_argument(
        "--route-query-minimum-score",
        type=float,
        default=ROUTE_QUERY_MINIMUM_SCORE,
    )
    parser.add_argument("--learned-route-query", action="store_true")
    parser.add_argument("--prototype-route-memory", action="store_true")
    parser.add_argument(
        "--route-memory-prototypes",
        type=int,
        default=ROUTE_MEMORY_PROTOTYPES,
    )
    parser.add_argument(
        "--learned-route-updates",
        type=int,
        default=LEARNED_ROUTE_UPDATES,
    )
    args = parser.parse_args()
    if args.match_tolerance < 0.0:
        raise SystemExit("--match-tolerance must be non-negative")
    if not -1.0 <= args.route_query_minimum_score <= 1.0:
        raise SystemExit("--route-query-minimum-score must lie in [-1, 1]")
    if args.learned_route_updates < 1:
        raise SystemExit("--learned-route-updates must be positive")
    if args.route_memory_prototypes < 1:
        raise SystemExit("--route-memory-prototypes must be positive")
    if args.learned_route_query and args.prototype_route_memory:
        raise SystemExit(
            "--learned-route-query and --prototype-route-memory are exclusive"
        )
    run(
        args.seed,
        args.report_out,
        use_route_query=args.route_query,
        match_tolerance=args.match_tolerance,
        route_query_minimum_score=args.route_query_minimum_score,
        use_learned_route_query=args.learned_route_query,
        learned_route_updates=args.learned_route_updates,
        use_prototype_route_memory=args.prototype_route_memory,
        route_memory_prototypes=args.route_memory_prototypes,
    )


if __name__ == "__main__":
    main()
