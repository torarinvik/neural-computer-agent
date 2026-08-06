"""Audit context-conditioned opaque routing on short Brain Workshop lifetimes.

The rendered cue is part of the learned event stream, not a task field passed
to the controller.  The route table receives only the first learned event,
opaque slot outcomes, and its own persistent state.  Cue-shuffled and
cue-absent controls test whether the table is using that event causally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _controls, _digest, _route_audit
from .trainer import train_isolated_relation_capability, train_reward_only


def _cue_symbol(n_back: int) -> int:
    """Map a benchmark cue to an opaque rendered vocabulary token."""

    return 4 + n_back - 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-n-back", type=int, default=2)
    parser.add_argument("--second-n-back", type=int, default=3)
    parser.add_argument("--third-n-back", type=int, default=4)
    parser.add_argument("--first-updates", type=int, default=64)
    parser.add_argument("--second-updates", type=int, default=128)
    parser.add_argument("--third-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.25)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _calibrate_slot(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
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
        cue_symbol=_cue_symbol(n_back),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_backs = (args.first_n_back, args.second_n_back, args.third_n_back)
    if len(set(n_backs)) != len(n_backs):
        raise ValueError("context-route audit needs three distinct n-back families")
    if args.calibration_lifetimes < 1:
        raise ValueError("calibration lifetimes must be positive")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
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
        context_route=True,
        cue_symbol=_cue_symbol(args.first_n_back),
    )
    first_history = train_reward_only(
        agent,
        n_back=args.first_n_back,
        updates=args.first_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        context_route=True,
        cue_symbol=_cue_symbol(args.first_n_back),
    )
    audits: dict[str, list[dict[str, float]]] = {}
    audits["2:0"] = _calibrate_slot(
        agent,
        n_back=args.first_n_back,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 2000,
        lifetimes=args.calibration_lifetimes,
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
        context_route=True,
        cue_symbol=_cue_symbol(args.second_n_back),
    )
    audits[f"{args.first_n_back}:0"] = _calibrate_slot(
        agent,
        n_back=args.first_n_back,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 3000,
        lifetimes=args.calibration_lifetimes,
    )
    audits[f"{args.second_n_back}:{second_slot}"] = _calibrate_slot(
        agent,
        n_back=args.second_n_back,
        slot=second_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 4000,
        lifetimes=args.calibration_lifetimes,
    )
    old_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )

    third_slot, third_history = train_isolated_relation_capability(
        agent,
        n_back=args.third_n_back,
        updates=args.third_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 9000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=_cue_symbol(args.third_n_back),
    )
    new_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )
    for n_back, slot, offset in (
        (args.first_n_back, 0, 6000),
        (args.second_n_back, second_slot, 7000),
        (args.third_n_back, third_slot, 8000),
    ):
        audits[f"{n_back}:{slot}"] = _calibrate_slot(
            agent,
            n_back=n_back,
            slot=slot,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + offset,
            lifetimes=args.calibration_lifetimes,
        )

    cue_controls = {
        str(n_back): _controls(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=control_seeds,
            context_route=True,
            cue_symbol=_cue_symbol(n_back),
        )
        for n_back in n_backs
    }
    cue_shuffled = {
        str(n_back): _controls(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=control_seeds,
            context_route=True,
            cue_symbol=_cue_symbol(n_backs[(index + 1) % len(n_backs)]),
        )
        for index, n_back in enumerate(n_backs)
    }
    cue_absent = {
        str(n_back): _controls(
            agent,
            n_back=n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=control_seeds,
            context_route=True,
        )
        for n_back in n_backs
    }
    histories = (*first_history, *second_history, *third_history)
    controls_passed = all(
        controls["fresh"] >= 0.8
        and controls["time_shuffle"] <= 0.75
        and controls["history_reset"] <= 0.75
        for controls in cue_controls.values()
    )
    audit_scores = [
        row["eligible_accuracy"]
        for rows in audits.values()
        for row in rows
    ]
    cue_separation_passed = all(
        cue_shuffled[str(n_back)]["fresh"]
        <= cue_controls[str(n_back)]["fresh"] - 0.1
        for n_back in n_backs
    )
    report = {
        "schema": "neural-computer.brainworkshop-context-route.v1",
        "status": (
            "promoted_cue_conditioned_short_lifetime_route"
            if controls_passed
            and min(audit_scores, default=0.0) >= 0.8
            and cue_separation_passed
            and old_bank_digest == new_bank_digest
            else "unpromoted_context_route"
        ),
        "n_back_families": n_backs,
        "cue_symbols": {str(n_back): _cue_symbol(n_back) for n_back in n_backs},
        "steps": args.steps,
        "first_baseline_controls": first_baseline,
        "cue_controls": cue_controls,
        "cue_shuffled_controls": cue_shuffled,
        "cue_separation_passed": cue_separation_passed,
        "cue_absent_controls": cue_absent,
        "candidate_audits": audits,
        "context_route_payload": agent.context_route_evidence.payload(),
        "controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "prior_bank_unchanged_during_third_growth": old_bank_digest == new_bank_digest,
        "unique_verifier_bits": sum(row.unique_verifier_bits for row in histories),
        "unique_logical_lifetimes": args.batch_size
        * (args.first_updates + args.second_updates + args.third_updates),
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "promotion_gates": {
            "cue_controls": controls_passed,
            "candidate_audits": min(audit_scores, default=0.0) >= 0.8,
            "cue_separation": cue_separation_passed,
            "prior_bank_unchanged": old_bank_digest == new_bank_digest,
        },
        "claim_boundary": (
            "A learned event cue can select among bounded opaque external slots "
            "without changing the controller; cue-free short-lifetime routing "
            "and general continual learning remain unqualified."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
