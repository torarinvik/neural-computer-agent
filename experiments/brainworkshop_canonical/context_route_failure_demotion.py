"""Audit failure-only demotion of a stale opaque route.

One rendered cue first identifies an n-back-2 capability.  A new n-back-4
capability is then learned under a different cue, so the changed same-cue
task receives no new route calibration.  Fresh n-back-4 failures under the
old cue must demote the protected old slot and allow the already learned new
slot to become preferred.  The controller and the old capability remain
frozen throughout the transition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _controls, _digest, _route_audit
from .trainer import train_isolated_relation_capability, train_reward_only

OLD_CUE = 4
NEW_CUE = 5
UNKNOWN_CUE = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-n-back", type=int, default=2)
    parser.add_argument("--new-n-back", type=int, default=4)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--new-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--transition-batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.5)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--reversal-threshold", type=float, default=0.75)
    parser.add_argument("--reversal-patience", type=int, default=4)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _audit(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    slot: int,
    cue_symbol: int,
    batch_size: int,
    steps: int,
    seed: int,
    lifetimes: int,
    record_context_route: bool,
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
        record_context_route=record_context_route,
    )


def _transition_audit(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    batch_size: int,
    steps: int,
    seed: int,
    batches: int,
) -> list[dict[str, float]]:
    """Run changed-task lifetimes that alone update the context route ledger."""

    results: list[dict[str, float]] = []
    for batch_index in range(batches):
        memory = agent.runtime.memory
        if hasattr(memory, "clear"):
            memory.clear()
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            seed=seed + batch_index,
            cue_symbol=cue_symbol,
        )
        with torch.no_grad():
            rollout = agent.rollout(
                verifier,
                sample=False,
                record_retention=False,
                context_route=True,
                record_context_route=True,
            )
        eligible = rollout.eligible
        denominator = eligible.sum().clamp_min(1)
        slot_fractions = {
            f"slot_{slot}_eligible_fraction": float(
                ((rollout.selected_slots == slot) & eligible).sum() / denominator
            )
            for slot in range(len(agent.extensions) + 1)
        }
        results.append(
            {
                "batch": batch_index + 1,
                "eligible_accuracy": float(rollout.eligible_accuracy.mean()),
                "first_selected_slot": float(rollout.selected_slots[:, 0].mode().values),
                **slot_fractions,
            }
        )
    return results


def _context_status(
    agent: CanonicalBrainWorkshopAgent,
    cue_symbol: int,
) -> dict[str, object]:
    event = agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )[0]
    record = agent.context_route_evidence._find_record(event, create=False)
    if record is None:
        raise RuntimeError("expected context route record was not learned")
    return {
        "key": list(record.key),
        "evidence": asdict(record.evidence.status()),
        "preferred_order": list(record.evidence.preferred_order()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.old_n_back == args.new_n_back:
        raise ValueError("demotion needs distinct old and new n-back families")
    if args.calibration_lifetimes < 1 or args.transition_batches < 1:
        raise ValueError("calibration lifetimes and transition batches must be positive")
    if args.reversal_patience < 1:
        raise ValueError("reversal patience must be positive")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=args.old_n_back,
        reader_kind="relation",
        seed=args.seed,
    )
    # Set the policy before the first cue record is created.  This is external
    # route-memory configuration, not a task label or a controller branch.
    agent.context_route_evidence.reversal_threshold = args.reversal_threshold
    agent.context_route_evidence.reversal_patience = args.reversal_patience
    control_seeds = tuple(args.seed + 1000 + index for index in range(3))

    old_history = train_reward_only(
        agent,
        n_back=args.old_n_back,
        updates=args.old_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    old_calibration = _audit(
        agent,
        n_back=args.old_n_back,
        slot=0,
        cue_symbol=OLD_CUE,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 2000,
        lifetimes=args.calibration_lifetimes,
        record_context_route=True,
    )
    old_status_before_change = _context_status(agent, OLD_CUE)
    old_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )

    new_slot, new_history = train_isolated_relation_capability(
        agent,
        n_back=args.new_n_back,
        updates=args.new_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 5000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=NEW_CUE,
    )
    new_calibration = _audit(
        agent,
        n_back=args.new_n_back,
        slot=new_slot,
        cue_symbol=NEW_CUE,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 3000,
        lifetimes=args.calibration_lifetimes,
        record_context_route=True,
    )
    new_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )

    transition = _transition_audit(
        agent,
        n_back=args.new_n_back,
        cue_symbol=OLD_CUE,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 4000,
        batches=args.transition_batches,
    )
    old_status_after_transition = _context_status(agent, OLD_CUE)
    new_status_after_transition = _context_status(agent, NEW_CUE)
    new_controls = _controls(
        agent,
        n_back=args.new_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    new_cue_controls = _controls(
        agent,
        n_back=args.new_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=NEW_CUE,
    )
    unknown_cue_controls = _controls(
        agent,
        n_back=args.new_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=UNKNOWN_CUE,
    )
    old_retention = _audit(
        agent,
        n_back=args.old_n_back,
        slot=0,
        cue_symbol=OLD_CUE,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 6000,
        lifetimes=args.calibration_lifetimes,
        record_context_route=False,
    )

    cue_event = agent.runtime.encoders["stimulus"](
        torch.tensor([OLD_CUE], dtype=torch.long)
    )[0]
    route_order = agent.context_route_evidence.preferred_order(cue_event)
    restored = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=args.old_n_back,
        reader_kind="relation",
        seed=args.seed,
    )
    restored.add_relation_capability(n_back=args.new_n_back, seed=args.seed + 5000)
    restored.load_route_state_payload(agent.route_state_payload())
    reload_rollout = restored.rollout(
        NBackVerifier(
            batch_size=2,
            n_back=args.new_n_back,
            steps=args.steps,
            symbol_count=4,
            cue_symbol=OLD_CUE,
            seed=args.seed + 7000,
        ),
        sample=False,
        record_retention=False,
        context_route=True,
    )
    reload_route = reload_rollout.selected_slots[:, 0].tolist()

    histories = (*old_history, *new_history)
    old_retention_score = min(row["eligible_accuracy"] for row in old_retention)
    new_calibration_score = min(
        row["eligible_accuracy"] for row in new_calibration
    )
    route_failure_demoted = (
        old_status_after_transition["evidence"]["reversal_count"][0] >= 1
        and route_order[0] == new_slot
    )
    controls_passed = (
        new_controls["fresh"] >= 0.8
        and new_controls["time_shuffle"] <= 0.75
        and new_controls["history_reset"] <= 0.75
        and new_cue_controls["fresh"] >= 0.8
    )
    unique_verifier_bits = sum(row.unique_verifier_bits for row in histories) + (
        args.batch_size
        * args.transition_batches
        * NBackVerifier(
            batch_size=1,
            n_back=args.new_n_back,
            steps=args.steps,
            cue_symbol=OLD_CUE,
        ).eligible_trials
    )
    verifier_outcome_events = (
        args.batch_size
        * (args.old_updates + args.new_updates + args.transition_batches)
        * (args.steps + 1)
    )
    report = {
        "schema": "neural-computer.brainworkshop-context-route-failure-demotion.v1",
        "status": (
            "promoted_failure_only_context_route_demotion"
            if route_failure_demoted
            and controls_passed
            and new_calibration_score >= 0.8
            and old_retention_score >= 0.8
            and reload_route == [new_slot, new_slot]
            and old_bank_digest == new_bank_digest
            else "unpromoted_failure_only_context_route_demotion"
        ),
        "old_cue": OLD_CUE,
        "new_cue": NEW_CUE,
        "unknown_cue": UNKNOWN_CUE,
        "old_n_back": args.old_n_back,
        "new_n_back": args.new_n_back,
        "transition_batches": args.transition_batches,
        "reversal_threshold": args.reversal_threshold,
        "reversal_patience": args.reversal_patience,
        "old_calibration": old_calibration,
        "new_calibration_under_different_cue": new_calibration,
        "old_status_before_change": old_status_before_change,
        "transition": transition,
        "old_status_after_transition": old_status_after_transition,
        "new_status_after_transition": new_status_after_transition,
        "new_controls_same_old_cue": new_controls,
        "new_controls_new_cue": new_cue_controls,
        "unknown_cue_controls": unknown_cue_controls,
        "old_forced_retention_after_change": old_retention,
        "preferred_route_order_after_failure_only_demotion": list(route_order),
        "reload_selected_route": reload_route,
        "route_state_payload": agent.route_state_payload(),
        "prior_bank_unchanged": old_bank_digest == new_bank_digest,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "unique_verifier_bits": unique_verifier_bits,
        "unique_logical_lifetimes": args.batch_size
        * (args.old_updates + args.new_updates + args.transition_batches),
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "verifier_outcome_events": verifier_outcome_events,
        "feedback_events": unique_verifier_bits,
        "promotion_gates": {
            "failure_only_route_demoted": route_failure_demoted,
            "new_route_mastered": controls_passed,
            "new_capability_calibrated_under_different_cue": new_calibration_score
            >= 0.8,
            "old_capability_retained": old_retention_score >= 0.8,
            "route_state_reload": reload_route == [new_slot, new_slot],
            "prior_bank_unchanged": old_bank_digest == new_bank_digest,
        },
        "claim_boundary": (
            "A protected opaque route was demoted by fresh changed-task scalar "
            "failures without same-cue calibration, and a previously learned "
            "external slot became preferred while the old capability was "
            "retained. This is bounded failure-driven nonstationary memory, "
            "not general continual learning or unrestricted memory growth."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
