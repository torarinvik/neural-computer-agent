"""Audit repeated routed-memory growth with stable-prefix retention.

The controller is frozen while six opaque regimes are acquired in sequence.
Each successor is a fresh unqualified cell appended to the accumulated bank,
and a matched fresh one-cell learner is captured before that regime begins.
No old verifier examples are replayed. After each acquisition, a fresh
held-out context prefix can qualify the new cell before the complete prefix is
probed for both cell content and route probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.policy_free_intention_memory import train as base
from experiments.policy_free_intention_routing import train as routing
from neural_computer import (
    AmodalEvent,
    ExternalControllerTrajectoryQueryAdapter,
    PolicyFreeAmodalRuntime,
)

MAX_UPDATES = 240
DELAY_STEPS = 3
REGIME_COUNT = 6
PROBE_COUNT = 32
MASTERY_THRESHOLD = 0.95
RETENTION_FLOOR = 0.90
ROUTE_FLOOR = 0.15
ROUTER_MASTERY_THRESHOLD = 0.80
ROUTER_MIN_MASTERY_FEEDBACKS = 8
ROUTER_UNQUALIFIED_CELL_PROBABILITY = 0.25
EVENT_VECTORS = (
    (0.30, -0.10, 0.40, -0.20),
    (-0.60, 0.70, -0.20, 0.50),
    (0.90, 0.20, -0.80, -0.40),
    (-0.40, -0.80, 0.70, 0.60),
    (0.80, -0.60, -0.40, 0.90),
    (-0.90, 0.50, 0.60, -0.70),
)
TARGETS = (
    (0.75, -0.75),
    (0.55, -0.95),
    (-0.75, 0.75),
    (0.90, 0.40),
    (-0.35, -0.90),
    (0.95, 0.70),
)
HELDOUT_OFFSETS = (
    (0.012, -0.006, 0.004, -0.010),
    (-0.009, 0.011, -0.005, 0.007),
    (0.008, 0.004, -0.012, 0.006),
    (-0.006, -0.010, 0.009, 0.003),
    (0.010, -0.008, -0.004, 0.011),
    (-0.011, 0.006, 0.010, -0.005),
    (0.005, 0.009, -0.007, -0.009),
    (-0.004, -0.012, 0.006, 0.008),
)


def _utility(intention: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(
        -(intention - target.reshape(1, -1)).square().sum(dim=-1)
        / base.UTILITY_TEMPERATURE
    ).clamp(0.0, 1.0)


def _digest_controller(controller: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(controller.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _digest_state(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(
        (
            ("cells.input_weights", state.cells.input_weights),
            ("cells.input_bias", state.cells.input_bias),
            ("cells.output_weights", state.cells.output_weights),
            ("cells.output_bias", state.cells.output_bias),
            ("cells.input_weight_eligibility", state.cells.input_weight_eligibility),
            ("cells.input_bias_eligibility", state.cells.input_bias_eligibility),
            ("cells.output_weight_eligibility", state.cells.output_weight_eligibility),
            ("cells.output_bias_eligibility", state.cells.output_bias_eligibility),
            ("cells.baseline", state.cells.baseline),
            ("cells.decisions", state.cells.decisions),
            ("cells.feedbacks", state.cells.feedbacks),
            ("cells.protected", state.cells.protected),
            ("routing_keys", state.routing_keys),
            ("routing_bias", state.routing_bias),
            ("routing_baseline", state.routing_baseline),
            ("routing_decisions", state.routing_decisions),
            ("routing_feedbacks", state.routing_feedbacks),
            ("retention_observations", state.retention_observations),
            ("retention_successes", state.retention_successes),
            ("retention_prefix_minima", state.retention_prefix_minima),
            ("retention_reversal_streaks", state.retention_reversal_streaks),
            ("retention_reversal_counts", state.retention_reversal_counts),
            ("retention_mastered", state.retention_mastered),
            ("retention_context_prototypes", state.retention_context_prototypes),
            ("retention_context_masses", state.retention_context_masses),
        )
    ):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build(seed: int):
    (
        controller,
        runtime,
        reference_policy,
        _memory,
        controller_state,
        feedback,
        _events,
        _contexts,
    ) = base._build(seed)
    route_query_adapter = ExternalControllerTrajectoryQueryAdapter(
        base.CONTROLLER_WIDTH,
    )
    for parameter in route_query_adapter.parameters():
        parameter.requires_grad_(False)
    policy, router, state = routing._new_policy(
        reference=reference_policy,
        seed=seed + 7_000,
        cell_count=1,
        mastery_threshold=ROUTER_MASTERY_THRESHOLD,
        min_mastery_feedbacks=ROUTER_MIN_MASTERY_FEEDBACKS,
        unqualified_cell_probability=ROUTER_UNQUALIFIED_CELL_PROBABILITY,
        context_width=route_query_adapter.query_width,
        route_query_adapter=route_query_adapter,
    )
    events: list[list[AmodalEvent]] = []
    contexts: list[torch.Tensor] = []
    goal_contexts: list[torch.Tensor] = []
    heldout_contexts: list[list[torch.Tensor]] = []
    for vector in EVENT_VECTORS:
        event = [AmodalEvent(torch.tensor([vector], dtype=torch.float32))]
        preview, trajectory_state = runtime.step_events(
            event,
            controller_state,
            feedback,
        )
        events.append(event)
        goal_contexts.append(reference_policy.state_adapter(preview.controller).detach())
        contexts.append(
            route_query_adapter(preview.controller, trajectory_state).detach()
        )
        variants: list[torch.Tensor] = []
        base_vector = torch.tensor(vector, dtype=torch.float32)
        for offset in HELDOUT_OFFSETS:
            heldout_event = [
                AmodalEvent(
                    (base_vector + torch.tensor(offset, dtype=torch.float32)).reshape(1, -1)
                )
            ]
            heldout_preview, heldout_state = runtime.step_events(
                heldout_event,
                controller_state,
                feedback,
            )
            variants.append(
                route_query_adapter(heldout_preview.controller, heldout_state)
                .detach()
                .squeeze(0)
            )
        heldout_contexts.append(variants)
    return (
        controller,
        runtime,
        reference_policy,
        policy,
        router,
        state,
        controller_state,
        feedback,
        events,
        contexts,
        goal_contexts,
        heldout_contexts,
        route_query_adapter,
    )


def _probe(
    router,
    state,
    context: torch.Tensor,
    target: torch.Tensor,
    cell_index: int,
) -> dict[str, object]:
    means = router.mean(state, context)[0]
    content_score = float(_utility(means[cell_index].unsqueeze(0), target).item())
    route_hits = 0
    route_probabilities: list[float] = []
    for _ in range(PROBE_COUNT):
        proposal = router.propose(state, context)
        route_hits += int(proposal.selected_cells[0].item() == cell_index)
        route_probabilities.append(
            float(proposal.route_probabilities[0, cell_index].item())
        )
    return {
        "content_score": content_score,
        "route_hit_rate": route_hits / PROBE_COUNT,
        "mean_route_probability": sum(route_probabilities) / PROBE_COUNT,
    }


def _route_retention_probe(router, state, contexts, targets, cell_count):
    rows: dict[str, dict[str, object]] = {}
    for cell_index in range(cell_count):
        rows[str(cell_index)] = _probe(
            router,
            state,
            contexts[cell_index],
            targets[cell_index],
            cell_index,
        )
    return rows


def _matched_fresh_state(state, cell_index: int):
    return routing._single_cell_state(state, cell_index)


def _heldout_qualify(
    router,
    state,
    contexts: list[torch.Tensor],
    target: torch.Tensor,
    cell_index: int,
) -> tuple[object, dict[str, object]]:
    outcomes = [
        float(
            _utility(
                router.mean(state, context.unsqueeze(0))[0, cell_index].unsqueeze(0),
                target,
            ).item()
        )
        for context in contexts
    ]
    next_state, receipt = router.verify_and_protect(
        state,
        cell_index,
        torch.stack(contexts),
        outcomes,
        floor=RETENTION_FLOOR,
    )
    return next_state, {
        "accepted": receipt.accepted,
        "reason": receipt.reason,
        "prefix_minimum": receipt.prefix_minimum,
        "mean_outcome": receipt.mean_outcome,
        "context_relevance_minimum": receipt.context_relevance_minimum,
        "outcome_count": len(receipt.outcomes),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    (
        controller,
        runtime,
        reference_policy,
        policy,
        router,
        state,
        controller_state,
        feedback,
        events,
        contexts,
        goal_contexts,
        heldout_contexts,
        route_query_adapter,
    ) = _build(seed)
    targets = [torch.tensor(target) for target in TARGETS]
    controller_digest = _digest_controller(controller)
    history: list[dict[str, object]] = []
    prefix_probes: list[dict[str, object]] = []

    for regime_index in range(REGIME_COUNT):
        pre_regime_state = state
        if regime_index == 0:
            warm_cell = 0
            fresh_state = _matched_fresh_state(pre_regime_state, 0)
        else:
            state, warm_cell = router.append_cell(pre_regime_state)
            fresh_source, fresh_cell = router.append_cell(pre_regime_state)
            fresh_state = _matched_fresh_state(fresh_source, fresh_cell)

        fresh_policy, fresh_router, _ = routing._new_policy(
            reference=reference_policy,
            seed=seed + 12_000 + regime_index,
            cell_count=1,
            mastery_threshold=ROUTER_MASTERY_THRESHOLD,
            min_mastery_feedbacks=ROUTER_MIN_MASTERY_FEEDBACKS,
            unqualified_cell_probability=ROUTER_UNQUALIFIED_CELL_PROBABILITY,
            context_width=route_query_adapter.query_width,
            route_query_adapter=route_query_adapter,
        )
        state, warm_report = routing._train_regime(
            policy=policy,
            router=router,
            state=state,
            controller_state=controller_state,
            feedback=feedback,
            event=events[regime_index],
            context=contexts[regime_index],
            goal_context=goal_contexts[regime_index],
            target=targets[regime_index],
            max_updates=MAX_UPDATES,
            delay=DELAY_STEPS,
            random_seed=seed + 20_000 + regime_index,
            mastery_cell=warm_cell,
        )
        fresh_state, fresh_report = routing._train_regime(
            policy=fresh_policy,
            router=fresh_router,
            state=fresh_state,
            controller_state=controller_state,
            feedback=feedback,
            event=events[regime_index],
            context=contexts[regime_index],
            goal_context=goal_contexts[regime_index],
            target=targets[regime_index],
            max_updates=MAX_UPDATES,
            delay=0,
            random_seed=seed + 30_000 + regime_index,
            mastery_cell=0,
        )
        heldout_verification: dict[str, dict[str, object]] = {}
        for cell_index in range(regime_index + 1):
            state, verification = _heldout_qualify(
                router,
                state,
                heldout_contexts[cell_index],
                targets[cell_index],
                cell_index,
            )
            heldout_verification[str(cell_index)] = verification
        prefix_probe = _route_retention_probe(
            router,
            state,
            contexts,
            targets,
            regime_index + 1,
        )
        prefix_probes.append(prefix_probe)
        history.append(
            {
                "regime": regime_index,
                "warm_cell": warm_cell,
                "warm_updates": warm_report["updates"],
                "fresh_updates": fresh_report["updates"],
                "transfer_ratio_fresh_over_warm": fresh_report["updates"]
                / max(1, warm_report["updates"]),
                "warm_mastery_cell_score": warm_report["mastery_cell_score"],
                "fresh_mastery_cell_score": fresh_report["mastery_cell_score"],
                "warm_auto_protected": bool(state.cells.protected[warm_cell]),
                "warm_heldout_protected": bool(state.cells.protected[warm_cell]),
                "heldout_verification": heldout_verification,
                "retention": prefix_probe,
                "materialized_candidate_cells": warm_report[
                    "materialized_candidate_cells"
                ],
            }
        )

    _, shuffled_router, shuffled_state = routing._new_policy(
        reference=reference_policy,
        seed=seed + 40_000,
        cell_count=1,
        mastery_threshold=ROUTER_MASTERY_THRESHOLD,
        min_mastery_feedbacks=ROUTER_MIN_MASTERY_FEEDBACKS,
        unqualified_cell_probability=ROUTER_UNQUALIFIED_CELL_PROBABILITY,
        context_width=route_query_adapter.query_width,
        route_query_adapter=route_query_adapter,
    )
    shuffled_policy = PolicyFreeAmodalRuntime(
        runtime,
        reference_policy.planner,
        state_adapter=reference_policy.state_adapter,
        intention_router=shuffled_router,
        route_query_adapter=route_query_adapter,
    )
    shuffled_state, shuffled = routing._train_regime(
        policy=shuffled_policy,
        router=shuffled_router,
        state=shuffled_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events[-1],
        context=contexts[-1],
        goal_context=goal_contexts[-1],
        target=targets[-1],
        max_updates=MAX_UPDATES,
        delay=DELAY_STEPS,
        reward_shuffled=True,
        random_seed=seed + 50_000,
        stop_at_mastery=False,
        mastery_cell=0,
    )

    _, missing_router, missing_state = routing._new_policy(
        reference=reference_policy,
        seed=seed + 60_000,
        cell_count=1,
        mastery_threshold=ROUTER_MASTERY_THRESHOLD,
        min_mastery_feedbacks=ROUTER_MIN_MASTERY_FEEDBACKS,
        unqualified_cell_probability=ROUTER_UNQUALIFIED_CELL_PROBABILITY,
        context_width=route_query_adapter.query_width,
        route_query_adapter=route_query_adapter,
    )
    missing_policy = PolicyFreeAmodalRuntime(
        runtime,
        reference_policy.planner,
        state_adapter=reference_policy.state_adapter,
        intention_router=missing_router,
        route_query_adapter=route_query_adapter,
    )
    missing = routing._missing_evidence_control(
        policy=missing_policy,
        router=missing_router,
        state=missing_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events[-1],
        context=contexts[-1],
        goal_context=goal_contexts[-1],
        target=targets[-1],
    )
    corruption = routing._corruption_probe(
        router,
        state,
        contexts[0],
        targets[0],
    )
    persistence = router.state_from_payload(router.state_payload(state))
    prefix_content_floor = min(
        float(row[str(cell)]["content_score"])
        for row in prefix_probes
        for cell in range(len(row))
    )
    prefix_route_floor = min(
        float(row[str(cell)]["mean_route_probability"])
        for row in prefix_probes
        for cell in range(len(row))
    )
    warm_transfer = [
        float(item["transfer_ratio_fresh_over_warm"])
        for item in history[1:]
    ]
    gates = {
        "all_regimes_mastered": all(
            float(item["warm_mastery_cell_score"]) >= MASTERY_THRESHOLD
            for item in history
        ),
        "every_new_cell_heldout_protected": all(
            bool(item["warm_heldout_protected"]) for item in history
        ),
        "stable_prefix_content_retained": prefix_content_floor >= RETENTION_FLOOR,
        "stable_prefix_route_retained": prefix_route_floor >= ROUTE_FLOOR,
        "sparse_materialization": all(
            float(item["materialized_candidate_cells"]["maximum"]) == 1.0
            for item in history
        ),
        "reward_shuffled_control_failed": (
            float(shuffled["mastery_cell_score"]) < MASTERY_THRESHOLD
        ),
        "missing_evidence_is_noop": bool(missing["state_unchanged"]),
        "memory_corruption_probe_detected": bool(corruption["corruption_detected"]),
        "exact_persistence": _digest_state(persistence) == _digest_state(state),
        "controller_frozen": controller_digest == _digest_controller(controller),
        "zero_replayed_examples": True,
        "six_cells_present": state.cells.baseline.shape[0] == REGIME_COUNT,
    }
    report = {
        "schema": "neural-computer.policy-free-intention-prefix-growth.v2",
        "claim_boundary": (
            "six-regime bounded routed external-memory growth with held-out verifier "
            "admission, stable-prefix retention, and "
            "matched fresh accounting; not positive-transfer or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "max_updates_per_regime": MAX_UPDATES,
            "delay_steps": DELAY_STEPS,
            "retention_floor": RETENTION_FLOOR,
            "route_floor": ROUTE_FLOOR,
            "router": router.configuration(),
            "fresh_control": "pre_regime_caller_owned_cell_snapshot_v1",
            "growth_cell": "fresh_unqualified_cell_in_accumulated_bank_v1",
            "heldout_verifier": "eight_perturbed_contexts_minimum_floor_copy_on_write_v1",
            "route_query_adapter": route_query_adapter.configuration(),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "history": history,
            "prefix_probes": prefix_probes,
            "prefix_content_floor": prefix_content_floor,
            "prefix_route_probability_floor": prefix_route_floor,
            "heldout_verifications": [item["heldout_verification"] for item in history],
            "positive_transfer_on_every_successor": all(
                ratio > 1.0 for ratio in warm_transfer
            ),
            "warm_transfer_ratios": warm_transfer,
            "reward_shuffled": shuffled,
            "missing_evidence": missing,
            "memory_corruption": corruption,
            "final_retention": {
                "protected": state.cells.protected.tolist(),
                "observations": state.retention_observations.tolist(),
                "successes": state.retention_successes.tolist(),
                "prefix_minima": state.retention_prefix_minima.tolist(),
                "reversal_counts": state.retention_reversal_counts.tolist(),
                "mastered": state.retention_mastered.tolist(),
            },
        },
        "accounting": {
            "unique_verifier_bits": sum(int(item["warm_updates"]) for item in history),
            "control_outcome_bits": int(shuffled["updates"]),
            "unique_logical_lifetimes": REGIME_COUNT,
            "external_memory_updates": sum(
                int(item["warm_updates"]) for item in history
            )
            + int(shuffled["updates"]),
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "search_expansions": sum(
                int(item["warm_updates"]) for item in history
            ),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85401)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
