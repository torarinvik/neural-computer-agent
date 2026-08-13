"""Held-out cue route discovery for a new causal working-memory file.

The source prefix is n-back-2 through n-back-4.  A new n-back-5 cell is
trained under rendered cue 7 with forced slot selection, but its route is not
calibrated.  Cue 8 is withheld from the route ledger and then deployed.  The
external route table must discover the new slot using only opaque attempted
slot outcomes, while all earlier cells and the shared controller remain
frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import ExternalWorkingMemoryCell

from .causal_depth_growth import (
    _audit,
    _cell,
    _cue_key,
    _digest,
    _freeze_external,
    _protected_digest,
    _rollout_score,
    _stable,
)
from .runner import CanonicalBrainWorkshopAgent
from .trainer import _train_relation_extension, train_reward_only

HELDOUT_RULE_GROWTH_SCHEMA = "neural-computer.brainworkshop-heldout-rule-growth.v1"
MASTERY_THRESHOLD = 0.80
PREFIX_RULES = ((2, 4), (3, 5), (4, 6))
TRAINING_RULE = (5, 7)
HELDOUT_CUE = 8


def _agent(seed: int) -> CanonicalBrainWorkshopAgent:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        cell = ExternalWorkingMemoryCell(
            event_width=16,
            action_width=2,
            memory_capacity=3,
            context_width=16,
            hidden=32,
        )
    return CanonicalBrainWorkshopAgent(
        symbol_count=9,
        n_back=2,
        event_width=16,
        intention_width=8,
        feedback_width=8,
        reader_kind="relation",
        seed=seed,
        working_memory_cell=cell,
    )


def _digest_encoder(agent: CanonicalBrainWorkshopAgent) -> str:
    return _digest(agent.runtime.encoders["stimulus"])


def _append_cell(
    agent: CanonicalBrainWorkshopAgent,
    *,
    capacity: int,
    seed: int,
) -> int:
    return agent.add_adaptive_relation_capability(
        memory_capacity=capacity,
        seed=seed,
        working_memory_cell=_cell(seed + 1, capacity),
    )


def _orders(agent: CanonicalBrainWorkshopAgent, cues: tuple[int, ...]) -> dict[str, list[int]]:
    return {
        str(cue): list(
            agent.context_route_evidence.preferred_order(_cue_key(agent, cue))
        )
        for cue in cues
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.batch_size,
        args.steps,
        args.calibration_lifetimes,
        args.discovery_lifetimes,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("held-out growth budgets must be positive")
    if args.steps <= TRAINING_RULE[0]:
        raise ValueError("held-out n-back-5 needs target-bearing steps")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")

    started = perf_counter()
    agent = _agent(args.seed)
    controller_before = _digest(agent.controller)
    encoder_before = _digest_encoder(agent)
    prefix_histories: list[object] = []
    prefix_digests: list[dict[str, object]] = []

    source_history = train_reward_only(
        agent,
        n_back=PREFIX_RULES[0][0],
        updates=args.source_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        cue_symbol=PREFIX_RULES[0][1],
    )
    prefix_histories.extend(source_history)
    source_before = _audit(
        agent,
        n_back=PREFIX_RULES[0][0],
        cue_symbol=PREFIX_RULES[0][1],
        slot=0,
        seed=args.seed + 1000,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        record_retention=True,
    )

    slots = [0]
    capacities = (4, 5)
    for index, ((n_back, cue), capacity) in enumerate(
        zip(PREFIX_RULES[1:], capacities, strict=True),
        start=1,
    ):
        slot = _append_cell(agent, capacity=capacity, seed=args.seed + 200 * index)
        slots.append(slot)
        before = _protected_digest(agent, range(slot))
        _, history = _train_relation_extension(
            agent,
            slot=slot,
            verifier_n_back=n_back,
            updates=args.target_updates,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 300 * index,
            learning_rate=args.learning_rate,
            exploration_probability=0.25,
            forced_slot=slot,
            cue_symbol=cue,
        )
        prefix_histories.extend(history)
        after = _protected_digest(agent, range(slot))
        prefix_digests.append(
            {
                "growth": index,
                "protected_slots": list(range(slot)),
                "unchanged": before == after,
            }
        )

    source_after_prefix = _protected_digest(agent, range(1))
    target_slot = _append_cell(
        agent,
        capacity=6,
        seed=args.seed + 800,
    )
    target_prefix_before = _protected_digest(agent, range(target_slot))
    _, target_history = _train_relation_extension(
        agent,
        slot=target_slot,
        verifier_n_back=TRAINING_RULE[0],
        updates=args.target_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 900,
        learning_rate=args.learning_rate,
        exploration_probability=0.25,
        forced_slot=target_slot,
        cue_symbol=TRAINING_RULE[1],
    )
    target_prefix_after = _protected_digest(agent, range(target_slot))
    _freeze_external(agent)

    prefix_retention: dict[str, list[dict[str, float | int]]] = {
        str(PREFIX_RULES[0][0]): _audit(
            agent,
            n_back=PREFIX_RULES[0][0],
            cue_symbol=PREFIX_RULES[0][1],
            slot=0,
            seed=args.seed + 10000,
            lifetimes=args.retention_lifetimes,
            batch_size=args.batch_size,
            steps=args.steps,
            record_retention=True,
        )
    }
    for index, (n_back, cue) in enumerate(PREFIX_RULES[1:], start=1):
        prefix_retention[str(n_back)] = _audit(
            agent,
            n_back=n_back,
            cue_symbol=cue,
            slot=slots[index],
            seed=args.seed + 11000 * index,
            lifetimes=args.retention_lifetimes,
            batch_size=args.batch_size,
            steps=args.steps,
            record_retention=True,
        )
    target_retention = _audit(
        agent,
        n_back=TRAINING_RULE[0],
        cue_symbol=TRAINING_RULE[1],
        slot=target_slot,
        seed=args.seed + 14000,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        record_retention=True,
    )

    known_cues = tuple(cue for _, cue in PREFIX_RULES)
    for index, (n_back, cue) in enumerate(PREFIX_RULES):
        _audit(
            agent,
            n_back=n_back,
            cue_symbol=cue,
            slot=slots[index],
            seed=args.seed + 20000 + index * 1000,
            lifetimes=args.calibration_lifetimes,
            batch_size=args.batch_size,
            steps=args.steps,
            context_route=True,
            record_context_route=True,
        )
    heldout_context_before = agent.context_route_evidence.has_context(
        _cue_key(agent, HELDOUT_CUE)
    )
    heldout_discovery: list[dict[str, float | int]] = []
    for index in range(args.discovery_lifetimes):
        heldout_discovery.append(
            _rollout_score(
                agent,
                n_back=TRAINING_RULE[0],
                cue_symbol=HELDOUT_CUE,
                slot=None,
                seed=args.seed + 30000 + index,
                batch_size=args.batch_size,
                steps=args.steps,
                context_route=True,
                record_context_route=True,
                expected_slot=target_slot,
            )
        )
    heldout_context_after = agent.context_route_evidence.has_context(
        _cue_key(agent, HELDOUT_CUE)
    )
    heldout_recovered = _rollout_score(
        agent,
        n_back=TRAINING_RULE[0],
        cue_symbol=HELDOUT_CUE,
        slot=None,
        seed=args.seed + 40000,
        batch_size=args.batch_size,
        steps=args.steps,
        context_route=True,
        expected_slot=target_slot,
    )

    route_cues = (*known_cues, HELDOUT_CUE)
    original_orders = _orders(agent, route_cues)
    route_payload = agent.route_state_payload()
    restored = _agent(args.seed + 50000)
    for index, capacity in enumerate((4, 5, 6), start=1):
        _append_cell(restored, capacity=capacity, seed=args.seed + 50100 + index)
    restored.runtime.encoders["stimulus"].load_state_dict(
        agent.runtime.encoders["stimulus"].state_dict()
    )
    restored.load_route_state_payload(route_payload)
    restored_orders = _orders(restored, route_cues)

    incompatible = _agent(args.seed + 60000)
    for index, capacity in enumerate((4, 5, 6), start=1):
        _append_cell(incompatible, capacity=capacity, seed=args.seed + 60100 + index)
    try:
        incompatible.load_route_state_payload(route_payload)
    except ValueError as error:
        incompatible_route_rejected = "learned event representation" in str(error)
    else:
        incompatible_route_rejected = False

    controller_after = _digest(agent.controller)
    source_after_all = _protected_digest(agent, range(1))
    target_prefix_after_all = _protected_digest(agent, range(target_slot))
    expected_heldout_order = [target_slot, 2, 1, 0]
    gates = {
        "source_mastery_before_growth": _stable(source_before),
        "source_complete_prefix_retention": _stable(prefix_retention["2"]),
        "prefix_retention": all(_stable(rows) for rows in prefix_retention.values()),
        "new_rule_mastery": _stable(target_retention),
        "protected_prefixes_unchanged": all(
            bool(row["unchanged"]) for row in prefix_digests
        ),
        "prefix_unchanged_during_heldout_growth": (
            source_after_prefix == source_after_all
            and target_prefix_before == target_prefix_after
            and target_prefix_after == target_prefix_after_all
        ),
        "controller_unchanged": controller_before == controller_after,
        "encoder_unchanged": encoder_before == _digest_encoder(agent),
        "heldout_context_absent_before_discovery": not heldout_context_before,
        "heldout_context_learned_from_outcomes": heldout_context_after,
        "heldout_route_recovered": (
            heldout_recovered["accuracy"] >= MASTERY_THRESHOLD
            and heldout_recovered["selected_slot_fraction"] >= 0.99
        ),
        "heldout_route_order_learned": (
            original_orders[str(HELDOUT_CUE)] == expected_heldout_order
        ),
        "route_reload_exact": original_orders == restored_orders,
        "incompatible_route_representation_rejected": incompatible_route_rejected,
        "zero_replayed_examples": True,
    }
    histories = (*prefix_histories, *target_history)
    training_bits = sum(row.unique_verifier_bits for row in histories)
    audit_lifetimes = (
        args.retention_lifetimes * 5
        + args.calibration_lifetimes * 3
        + args.discovery_lifetimes
        + 1
    )
    audit_bits = args.batch_size * (
        args.retention_lifetimes
        * sum(args.steps - n_back for n_back, _ in (*PREFIX_RULES, TRAINING_RULE))
        + args.retention_lifetimes * (args.steps - PREFIX_RULES[0][0])
        + args.calibration_lifetimes
        * sum(args.steps - n_back for n_back, _ in PREFIX_RULES)
        + (args.discovery_lifetimes + 1) * (args.steps - TRAINING_RULE[0])
    )
    report = {
        "schema": HELDOUT_RULE_GROWTH_SCHEMA,
        "claim_boundary": (
            "A new n-back-5 external file is acquired under cue 7, then cue 8 "
            "is introduced without a route record and discovers the correct "
            "file from scalar outcomes. This is held-out route discovery over "
            "bounded rule growth, not general continual learning."
        ),
        "seed": args.seed,
        "prefix_rules": [
            {"n_back": n_back, "cue_symbol": cue, "slot": slots[index]}
            for index, (n_back, cue) in enumerate(PREFIX_RULES)
        ],
        "training_rule": {
            "n_back": TRAINING_RULE[0],
            "training_cue": TRAINING_RULE[1],
            "heldout_cue": HELDOUT_CUE,
            "slot": target_slot,
        },
        "source_updates": args.source_updates,
        "target_updates_per_growth": args.target_updates,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "prefix_retention": prefix_retention,
        "target_retention": target_retention,
        "prefix_digests": prefix_digests,
        "heldout_context_before": heldout_context_before,
        "heldout_context_after": heldout_context_after,
        "heldout_discovery": heldout_discovery,
        "heldout_recovered": heldout_recovered,
        "original_route_orders": original_orders,
        "restored_route_orders": restored_orders,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits_training": training_bits,
            "unique_verifier_bits_audit": audit_bits,
            "heldout_route_discovery_bits": (
                args.batch_size
                * args.discovery_lifetimes
                * (args.steps - TRAINING_RULE[0])
            ),
            "unique_logical_lifetimes_training": args.batch_size * len(histories),
            "unique_logical_lifetimes_audit": args.batch_size * audit_lifetimes,
            "optimizer_updates": len(histories),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=64)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--discovery-lifetimes", type=int, default=8)
    parser.add_argument("--retention-lifetimes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
