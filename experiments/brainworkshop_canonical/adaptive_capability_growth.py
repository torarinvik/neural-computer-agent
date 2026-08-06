"""Audit generic external capability growth on fresh relation families.

The initial compatibility slot is trained on n-back-2.  Later slots are
provisioned only with the same bounded event-window capacity and an opaque
seed; the benchmark's hidden verifier horizon is not passed to their
constructors.  Each slot is learned from fresh scalar outcomes, then an
explicit forced candidate audit calibrates its learned event-cue route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _controls, _digest, _route_audit
from .trainer import (
    train_adaptive_relation_capability,
    train_reward_only,
)

TASKS = ((2, 4), (3, 5), (4, 6))
GENERIC_MEMORY_CAPACITY = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--adaptive-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.5)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _calibrate(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    slot: int,
    batch_size: int,
    steps: int,
    seed: int,
    lifetimes: int,
) -> list[dict[str, float]]:
    return _route_audit(
        agent,
        n_back=n_back,
        batch_size=batch_size,
        steps=steps,
        seeds=tuple(seed + index for index in range(lifetimes)),
        slot=slot,
        context_route=True,
        cue_symbol=cue_symbol,
        record_context_route=True,
    )


def _route_orders(
    agent: CanonicalBrainWorkshopAgent,
) -> dict[str, list[int]]:
    orders: dict[str, list[int]] = {}
    for n_back, cue_symbol in TASKS:
        event = agent.runtime.encoders["stimulus"](
            torch.tensor([cue_symbol], dtype=torch.long)
        )[0]
        orders[str(cue_symbol)] = list(
            agent.context_route_evidence.preferred_order(event)
        )
    return orders


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.old_updates, args.adaptive_updates, args.batch_size) < 1:
        raise ValueError("updates and batch size must be positive")
    if args.calibration_lifetimes < 1:
        raise ValueError("calibration lifetimes must be positive")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=TASKS[0][0],
        reader_kind="relation",
        seed=args.seed,
    )
    control_seeds = tuple(args.seed + 1000 + index for index in range(3))
    old_history = train_reward_only(
        agent,
        n_back=TASKS[0][0],
        updates=args.old_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        context_route=True,
        cue_symbol=TASKS[0][1],
    )
    old_calibration = _calibrate(
        agent,
        n_back=TASKS[0][0],
        cue_symbol=TASKS[0][1],
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 2000,
        lifetimes=args.calibration_lifetimes,
    )
    old_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )

    adaptive_histories: list[object] = []
    adaptive_calibrations: dict[str, list[dict[str, float]]] = {}
    adaptive_slots: dict[str, int] = {}
    for index, (n_back, cue_symbol) in enumerate(TASKS[1:], start=1):
        slot, history = train_adaptive_relation_capability(
            agent,
            verifier_n_back=n_back,
            memory_capacity=GENERIC_MEMORY_CAPACITY,
            updates=args.adaptive_updates,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 5000 * index,
            learning_rate=args.learning_rate,
            exploration_probability=args.slot_exploration,
            context_route=True,
            cue_symbol=cue_symbol,
        )
        adaptive_slots[str(n_back)] = slot
        adaptive_histories.extend(history)
        adaptive_calibrations[str(n_back)] = _calibrate(
            agent,
            n_back=n_back,
            cue_symbol=cue_symbol,
            slot=slot,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 7000 * index,
            lifetimes=args.calibration_lifetimes,
        )

    new_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )
    controls = {
        str(n_back): _controls(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=control_seeds,
            context_route=True,
            cue_symbol=cue_symbol,
        )
        for n_back, cue_symbol in TASKS
    }
    cue_shuffled = {
        str(n_back): _controls(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=control_seeds,
            context_route=True,
            cue_symbol=TASKS[(index + 1) % len(TASKS)][1],
        )
        for index, (n_back, _cue_symbol) in enumerate(TASKS)
    }
    old_retention = _route_audit(
        agent,
        n_back=TASKS[0][0],
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 9000 + index for index in range(3)),
        slot=0,
        context_route=True,
        cue_symbol=TASKS[0][1],
        record_context_route=False,
    )

    restored = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=TASKS[0][0],
        reader_kind="relation",
        seed=args.seed,
    )
    restored.add_adaptive_relation_capability(
        memory_capacity=GENERIC_MEMORY_CAPACITY,
        seed=args.seed + 5000,
    )
    restored.add_adaptive_relation_capability(
        memory_capacity=GENERIC_MEMORY_CAPACITY,
        seed=args.seed + 10000,
    )
    restored.load_route_state_payload(agent.route_state_payload())
    reload_selected_routes: dict[str, list[int]] = {}
    for n_back, cue_symbol in TASKS:
        reload_rollout = restored.rollout(
            NBackVerifier(
                batch_size=2,
                n_back=n_back,
                steps=args.steps,
                symbol_count=4,
                cue_symbol=cue_symbol,
                seed=args.seed + 11000 + n_back,
            ),
            sample=False,
            record_retention=False,
            context_route=True,
        )
        reload_selected_routes[str(cue_symbol)] = (
            reload_rollout.selected_slots[:, 0].tolist()
        )

    histories = (*old_history, *adaptive_histories)
    calibration_scores = [
        row["eligible_accuracy"]
        for rows in (adaptive_calibrations | {"2": old_calibration}).values()
        for row in rows
    ]
    controls_passed = all(
        result["fresh"] >= 0.8
        and result["time_shuffle"] <= 0.75
        and result["history_reset"] <= 0.75
        for result in controls.values()
    )
    cue_separation_passed = all(
        cue_shuffled[str(n_back)]["fresh"]
        <= controls[str(n_back)]["fresh"] - 0.1
        for n_back, _cue_symbol in TASKS
    )
    old_retention_score = min(row["eligible_accuracy"] for row in old_retention)
    reload_passed = reload_selected_routes == {
        "4": [0, 0],
        "5": [1, 1],
        "6": [2, 2],
    }
    report = {
        "schema": "neural-computer.brainworkshop-adaptive-capability-growth.v1",
        "status": (
            "promoted_generic_external_capability_growth"
            if controls_passed
            and cue_separation_passed
            and min(calibration_scores, default=0.0) >= 0.8
            and old_retention_score >= 0.8
            and reload_passed
            and old_bank_digest == new_bank_digest
            else "unpromoted_generic_external_capability_growth"
        ),
        "task_horizons_used_only_by_verifier": [n_back for n_back, _ in TASKS],
        "rendered_cues": {str(n_back): cue for n_back, cue in TASKS},
        "generic_memory_capacity": GENERIC_MEMORY_CAPACITY,
        "generic_capability_provisioning": [
            {"slot": slot, "memory_capacity": GENERIC_MEMORY_CAPACITY}
            for slot in adaptive_slots.values()
        ],
        "no_task_metadata_to_capability": True,
        "controls": controls,
        "cue_shuffled_controls": cue_shuffled,
        "cue_separation_passed": cue_separation_passed,
        "old_calibration": old_calibration,
        "adaptive_candidate_calibrations": adaptive_calibrations,
        "old_forced_retention": old_retention,
        "preferred_route_orders": _route_orders(agent),
        "reload_selected_routes": reload_selected_routes,
        "prior_bank_unchanged": old_bank_digest == new_bank_digest,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "unique_verifier_bits": sum(row.unique_verifier_bits for row in histories),
        "unique_logical_lifetimes": args.batch_size
        * (args.old_updates + 2 * args.adaptive_updates),
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "verifier_outcome_events": args.batch_size
        * (args.old_updates + 2 * args.adaptive_updates)
        * (args.steps + 1),
        "feedback_events": sum(row.unique_verifier_bits for row in histories),
        "promotion_gates": {
            "all_fresh_controls": controls_passed,
            "cue_separation": cue_separation_passed,
            "candidate_calibrations": min(calibration_scores, default=0.0) >= 0.8,
            "old_capability_retained": old_retention_score >= 0.8,
            "route_state_reload": reload_passed,
            "prior_bank_unchanged": old_bank_digest == new_bank_digest,
            "no_task_metadata_to_capability": True,
        },
        "claim_boundary": (
            "Two new relation capabilities were provisioned with the same "
            "generic adaptive bounded-window reader, learned from fresh "
            "outcomes, and routed by rendered cues while the controller and "
            "old capability stayed frozen. This promotes bounded generic "
            "capability growth, not arbitrary program induction, unrestricted "
            "memory growth, or general continual learning."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
