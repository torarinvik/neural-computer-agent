"""Audit three sequential isolated Brain Workshop capability slots.

This extends the promoted two-slot rung by appending a third relation-reader
slot. Every earlier slot is frozen before the next append, and only the
learner's attempted scalar outcomes are used during acquisition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _controls, _digest, _route_audit
from .trainer import train_isolated_relation_capability, train_reward_only


def _bank_digest(agent: CanonicalBrainWorkshopAgent, slots: int) -> str:
    modules = [
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    ]
    for slot in range(1, slots):
        extension = agent.extensions[slot - 1]
        modules.extend(
            (
                extension.reader,
                extension.intent_adapter,
                agent.extension_decoder(slot),
            )
        )
    return _digest(*modules)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-n-back", type=int, default=2)
    parser.add_argument("--second-n-back", type=int, default=3)
    parser.add_argument("--third-n-back", type=int, default=4)
    parser.add_argument("--first-updates", type=int, default=128)
    parser.add_argument("--second-updates", type=int, default=256)
    parser.add_argument("--third-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.25)
    parser.add_argument("--learned-third-route", action="store_true")
    parser.add_argument("--persistent-route", action="store_true")
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_backs = (args.first_n_back, args.second_n_back, args.third_n_back)
    if len(set(n_backs)) != len(n_backs):
        raise ValueError("capacity ladder needs three distinct n-back families")
    agent = CanonicalBrainWorkshopAgent(
        n_back=args.first_n_back,
        reader_kind="relation",
        seed=args.seed,
    )
    control_seeds = tuple(args.seed + 1000 + index for index in range(3))
    first_baseline = _controls(
        agent,
        n_back=args.first_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
    )
    first_history = train_reward_only(
        agent,
        n_back=args.first_n_back,
        updates=args.first_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    _route_audit(
        agent,
        n_back=args.first_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 2000 + index for index in range(3)),
        slot=0,
    )
    second_slot, second_history = train_isolated_relation_capability(
        agent,
        n_back=args.second_n_back,
        updates=args.second_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 5000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        persistent_route=args.persistent_route,
    )
    _route_audit(
        agent,
        n_back=args.first_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 3000 + index for index in range(3)),
        slot=0,
    )
    _route_audit(
        agent,
        n_back=args.second_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 4000 + index for index in range(3)),
        slot=second_slot,
    )
    bank_digest_before_third = _bank_digest(agent, slots=2)
    third_slot, third_history = train_isolated_relation_capability(
        agent,
        n_back=args.third_n_back,
        updates=args.third_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 9000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        learned_route=args.learned_third_route,
        persistent_route=args.persistent_route,
    )
    bank_digest_after_third = _bank_digest(agent, slots=2)
    post_controls = {
        str(n_back): _controls(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=control_seeds,
            learned_route=args.learned_third_route,
            persistent_route=args.persistent_route,
        )
        for n_back in n_backs
    }
    retention_audits = {
        str(n_back): _route_audit(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=tuple(
                args.seed + 10_000 + n_back * 10 + index
                for index in range(3)
            ),
            slot=slot,
        )
        for n_back, slot in zip(n_backs, range(3))
    }
    statuses = [
        asdict(agent.retention.status(agent.capability_address_for(slot)))
        for slot in range(3)
    ]
    controls_passed = all(
        controls["fresh"] >= 0.8
        and controls["time_shuffle"] <= 0.75
        and controls["history_reset"] <= 0.75
        for controls in post_controls.values()
    )
    old_bank_unchanged = bank_digest_before_third == bank_digest_after_third
    all_protected = all(status["protected"] for status in statuses)
    histories = (*first_history, *second_history, *third_history)
    report = {
        "schema": "neural-computer.brainworkshop-capacity-ladder.v1",
        "status": "promoted_bounded_three_slot_growth"
        if all_protected and controls_passed and old_bank_unchanged
        else "unpromoted_three_slot_growth",
        "n_back_families": n_backs,
        "appended_slots": [second_slot, third_slot],
        "learned_third_route": args.learned_third_route,
        "persistent_route": args.persistent_route,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "first_baseline_controls": first_baseline,
        "post_growth_controls": post_controls,
        "retention_audits": retention_audits,
        "retention_statuses": statuses,
        "first_two_slots_unchanged_during_third_growth": old_bank_unchanged,
        "shared_controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "third_slot_training_fraction": sum(
            row.selected_slot_fraction for row in third_history
        )
        / max(1, len(third_history)),
        "unique_verifier_bits": sum(row.unique_verifier_bits for row in histories),
        "unique_logical_lifetimes": args.batch_size
        * (args.first_updates + args.second_updates + args.third_updates),
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "promotion_gates": {
            "all_slots_protected": all_protected,
            "causal_controls": controls_passed,
            "prior_slots_unchanged": old_bank_unchanged,
        },
        "claim_boundary": (
            "Three sequential relation-reader capabilities were acquired in "
            "isolated external slots over a frozen controller with zero replay; "
            "this remains bounded growth and does not establish unrestricted "
            "memory expansion, learned eviction, reversal recovery, or general "
            "continual learning."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
