"""Audit automatic route discovery without new-cue calibration.

An old n-back-2 route is calibrated under cue 4.  A generic adaptive n-back-3
capability is then acquired under cue 5, but no forced cue-5 candidate audit
is allowed.  Ordinary fallback episodes record only grouped scalar verifier
outcomes; the new route must become preferred from those outcomes alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_computer import ContentAddressedMemory

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _controls, _digest, _route_audit
from .trainer import train_adaptive_relation_capability, train_reward_only

OLD_N_BACK = 2
NEW_N_BACK = 3
OLD_CUE = 4
NEW_CUE = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--new-updates", type=int, default=256)
    parser.add_argument("--discovery-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--memory-capacity", type=int, default=5)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.5)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _discover(
    agent: CanonicalBrainWorkshopAgent,
    *,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
) -> list[dict[str, float]]:
    """Let fresh fallback outcomes, rather than calibration, update routing."""

    results: list[dict[str, float]] = []
    for seed in seeds:
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        rollout = agent.rollout(
            NBackVerifier(
                batch_size=batch_size,
                n_back=NEW_N_BACK,
                steps=steps,
                symbol_count=4,
                cue_symbol=NEW_CUE,
                seed=seed,
            ),
            sample=False,
            record_retention=False,
            context_route=True,
            record_context_route=True,
        )
        eligible = rollout.eligible
        denominator = eligible.sum().clamp_min(1)
        results.append(
            {
                "seed": seed,
                "eligible_accuracy": float(rollout.eligible_accuracy.mean()),
                "slot_0_eligible_fraction": float(
                    ((rollout.selected_slots == 0) & eligible).sum() / denominator
                ),
                "slot_1_eligible_fraction": float(
                    ((rollout.selected_slots == 1) & eligible).sum() / denominator
                ),
                "first_selected_slot": float(
                    rollout.selected_slots[:, 0].mode().values
                ),
            }
        )
    return results


def _new_context_status(
    agent: CanonicalBrainWorkshopAgent,
) -> dict[str, object]:
    event = agent.runtime.encoders["stimulus"](
        torch.tensor([NEW_CUE], dtype=torch.long)
    )[0]
    record = agent.context_route_evidence._find_record(event, create=False)
    if record is None:
        raise RuntimeError("automatic discovery did not create a cue record")
    status = record.evidence.status()
    return {
        "preferred_order": list(record.evidence.preferred_order()),
        "attempts": list(status.attempts),
        "successes": list(status.successes),
        "protected": list(status.protected),
        "reversal_count": list(status.reversal_count),
        "preferred_slot": status.preferred_slot,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.old_updates,
        args.new_updates,
        args.discovery_batches,
        args.batch_size,
        args.memory_capacity,
    ) < 1:
        raise ValueError("updates, discovery batches, capacity, and batch size must be positive")
    if args.calibration_lifetimes < 1:
        raise ValueError("calibration lifetimes must be positive")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=OLD_N_BACK,
        reader_kind="relation",
        seed=args.seed,
    )
    control_seeds = tuple(args.seed + 1000 + index for index in range(3))
    old_history = train_reward_only(
        agent,
        n_back=OLD_N_BACK,
        updates=args.old_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    old_calibration = _route_audit(
        agent,
        n_back=OLD_N_BACK,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 2000 + index for index in range(args.calibration_lifetimes)),
        context_route=True,
        cue_symbol=OLD_CUE,
        record_context_route=True,
    )
    old_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )

    new_slot, new_history = train_adaptive_relation_capability(
        agent,
        verifier_n_back=NEW_N_BACK,
        memory_capacity=args.memory_capacity,
        updates=args.new_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 5000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=NEW_CUE,
    )
    new_bank_digest = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )

    discovery = _discover(
        agent,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(
            args.seed + 4000 + index for index in range(args.discovery_batches)
        ),
    )
    new_context_status = _new_context_status(agent)
    new_controls = _controls(
        agent,
        n_back=NEW_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=NEW_CUE,
    )
    cue_shuffled_controls = _controls(
        agent,
        n_back=NEW_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    old_controls = _controls(
        agent,
        n_back=OLD_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    old_retention = _route_audit(
        agent,
        n_back=OLD_N_BACK,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 6000 + index for index in range(3)),
        context_route=True,
        cue_symbol=OLD_CUE,
        record_context_route=False,
    )
    new_validation = _route_audit(
        agent,
        n_back=NEW_N_BACK,
        slot=new_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 7000 + index for index in range(3)),
        context_route=True,
        cue_symbol=NEW_CUE,
        record_context_route=False,
    )

    restored = CanonicalBrainWorkshopAgent(
        symbol_count=7,
        n_back=OLD_N_BACK,
        reader_kind="relation",
        seed=args.seed,
    )
    restored.add_adaptive_relation_capability(
        memory_capacity=args.memory_capacity,
        seed=args.seed + 5000,
    )
    restored.load_route_state_payload(agent.route_state_payload())
    reload_rollout = restored.rollout(
        NBackVerifier(
            batch_size=2,
            n_back=NEW_N_BACK,
            steps=args.steps,
            symbol_count=4,
            cue_symbol=NEW_CUE,
            seed=args.seed + 8000,
        ),
        sample=False,
        record_retention=False,
        context_route=True,
    )
    reload_route = reload_rollout.selected_slots[:, 0].tolist()

    histories = (*old_history, *new_history)
    old_retention_score = min(row["eligible_accuracy"] for row in old_retention)
    new_validation_score = min(
        row["eligible_accuracy"] for row in new_validation
    )
    discovery_success = (
        new_context_status["preferred_slot"] == new_slot
        and new_context_status["preferred_order"][0] == new_slot
        and all(row["slot_1_eligible_fraction"] > 0.0 for row in discovery)
        and discovery[-1]["slot_1_eligible_fraction"] == 1.0
    )
    controls_passed = (
        new_controls["fresh"] >= 0.8
        and new_controls["time_shuffle"] <= 0.75
        and new_controls["history_reset"] <= 0.75
        and old_controls["fresh"] >= 0.8
        and old_controls["time_shuffle"] <= 0.75
        and old_controls["history_reset"] <= 0.75
    )
    report = {
        "schema": "neural-computer.brainworkshop-automatic-route-discovery.v1",
        "status": (
            "promoted_automatic_route_discovery"
            if discovery_success
            and controls_passed
            and new_validation_score >= 0.8
            and old_retention_score >= 0.8
            and reload_route == [new_slot, new_slot]
            and old_bank_digest == new_bank_digest
            else "unpromoted_automatic_route_discovery"
        ),
        "old_n_back": OLD_N_BACK,
        "new_n_back": NEW_N_BACK,
        "old_cue": OLD_CUE,
        "new_cue": NEW_CUE,
        "memory_capacity": args.memory_capacity,
        "new_cue_calibration_performed": False,
        "old_calibration": old_calibration,
        "new_candidate_validation_after_discovery": new_validation,
        "discovery": discovery,
        "new_context_status": new_context_status,
        "new_controls": new_controls,
        "cue_shuffled_controls": cue_shuffled_controls,
        "old_controls": old_controls,
        "old_forced_retention": old_retention,
        "reload_selected_route": reload_route,
        "prior_bank_unchanged": old_bank_digest == new_bank_digest,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "unique_verifier_bits": sum(row.unique_verifier_bits for row in histories)
        + args.batch_size * args.discovery_batches * (args.steps - NEW_N_BACK),
        "unique_logical_lifetimes": args.batch_size
        * (args.old_updates + args.new_updates + args.discovery_batches),
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "verifier_outcome_events": args.batch_size
        * (args.old_updates + args.new_updates + args.discovery_batches)
        * (args.steps + 1),
        "feedback_events": sum(row.unique_verifier_bits for row in histories)
        + args.batch_size * args.discovery_batches * (args.steps - NEW_N_BACK),
        "promotion_gates": {
            "automatic_discovery_without_new_calibration": discovery_success,
            "new_route_mastered": controls_passed,
            "new_candidate_validation": new_validation_score >= 0.8,
            "old_capability_retained": old_retention_score >= 0.8,
            "cue_shuffle_control": cue_shuffled_controls["fresh"]
            <= new_controls["fresh"] - 0.1,
            "route_state_reload": reload_route == [new_slot, new_slot],
            "prior_bank_unchanged": old_bank_digest == new_bank_digest,
            "new_cue_calibration_performed": False,
        },
        "claim_boundary": (
            "A newly learned generic external capability became preferred for "
            "a new rendered cue from ordinary fallback outcomes alone, with no "
            "new-cue calibration transaction. This promotes bounded automatic "
            "route discovery, not arbitrary program induction, unrestricted "
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
