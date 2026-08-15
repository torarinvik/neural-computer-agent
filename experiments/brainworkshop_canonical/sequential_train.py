"""Acquire two Neural Workshop capabilities in isolated external slots.

The first relation reader is trained on one n-back lifetime family.  A second
reader, intention adapter, decoder, and opaque retention address are then
appended and trained on a fresh family while the controller and first slot are
frozen.  Slot growth is failure-gated by the learner's own attempted scalar
outcome; no task identifier or correct unattempted action is exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_computer import ContentAddressedMemory

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .trainer import (
    evaluate_policy,
    train_isolated_relation_capability,
    train_reward_only,
)


def _digest(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in module.state_dict().items():
            digest.update(f"{module_index}:{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _controls(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
    learned_route: bool = False,
    persistent_route: bool = False,
    context_route: bool = False,
    cue_symbol: int | None = None,
) -> dict[str, float]:
    return {
        "fresh": evaluate_policy(
            agent,
            n_back=n_back,
            batch_size=batch_size,
            seeds=seeds,
            steps=steps,
            learned_route=learned_route,
            persistent_route=persistent_route,
            context_route=context_route,
            cue_symbol=cue_symbol,
        ),
        "time_shuffle": evaluate_policy(
            agent,
            n_back=n_back,
            batch_size=batch_size,
            seeds=seeds,
            steps=steps,
            time_shuffle=True,
            learned_route=learned_route,
            persistent_route=persistent_route,
            context_route=context_route,
            cue_symbol=cue_symbol,
        ),
        "history_reset": evaluate_policy(
            agent,
            n_back=n_back,
            batch_size=batch_size,
            seeds=seeds,
            steps=steps,
            reset_history=True,
            learned_route=learned_route,
            persistent_route=persistent_route,
            context_route=context_route,
            cue_symbol=cue_symbol,
        ),
    }


def _route_audit(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
    slot: int,
    context_route: bool = False,
    cue_symbol: int | None = None,
    record_context_route: bool | None = None,
) -> list[dict[str, float]]:
    """Record retention scores and expose only route-use diagnostics."""

    results: list[dict[str, float]] = []
    for seed in seeds:
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        verifier = NBackVerifier(
            batch_size=batch_size,
            n_back=n_back,
            steps=steps,
            seed=seed,
            cue_symbol=cue_symbol,
        )
        with torch.no_grad():
            rollout = agent.rollout(
                verifier,
                sample=False,
                record_retention=True,
                forced_slot=slot,
                context_route=context_route,
                record_context_route=(
                    context_route
                    if record_context_route is None
                    else record_context_route
                ),
            )
        results.append(
            {
                "eligible_accuracy": float(rollout.eligible_accuracy.mean()),
                "selected_slot_fraction": float(
                    (rollout.selected_slots == slot).to(torch.float32).mean()
                ),
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-n-back", type=int, default=2)
    parser.add_argument("--new-n-back", type=int, default=3)
    parser.add_argument("--old-updates", type=int, default=128)
    parser.add_argument("--new-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.25)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.old_n_back == args.new_n_back:
        raise ValueError("sequential audit needs two distinct n-back families")
    agent = CanonicalBrainWorkshopAgent(
        n_back=args.old_n_back,
        reader_kind="relation",
        seed=args.seed,
    )
    control_seeds = tuple(args.seed + 1000 + index for index in range(3))
    old_baseline = _controls(
        agent,
        n_back=args.old_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
    )
    old_history = train_reward_only(
        agent,
        n_back=args.old_n_back,
        updates=args.old_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    old_audit_before = _route_audit(
        agent,
        n_back=args.old_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 2000 + index for index in range(3)),
        slot=0,
    )
    old_digest_before_append = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )
    slot, new_history = train_isolated_relation_capability(
        agent,
        n_back=args.new_n_back,
        updates=args.new_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 5000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
    )
    old_digest_after_append = _digest(
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )
    old_controls = _controls(
        agent,
        n_back=args.old_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
    )
    new_controls = _controls(
        agent,
        n_back=args.new_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
    )
    old_audit_after = _route_audit(
        agent,
        n_back=args.old_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 3000 + index for index in range(3)),
        slot=0,
    )
    new_audit = _route_audit(
        agent,
        n_back=args.new_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 4000 + index for index in range(3)),
        slot=slot,
    )
    old_status = agent.retention.status(agent.capability_address_for(0))
    new_status = agent.retention.status(agent.capability_address_for(slot))
    controls_passed = all(
        controls["fresh"] >= 0.8
        and controls["time_shuffle"] <= 0.75
        and controls["history_reset"] <= 0.75
        for controls in (old_controls, new_controls)
    )
    report = {
        "schema": "neural-computer.brainworkshop-sequential-growth.v1",
        "status": "promoted_bounded_sequential_growth"
        if old_status.protected
        and new_status.protected
        and controls_passed
        and old_digest_before_append == old_digest_after_append
        else "unpromoted_sequential_growth",
        "old_n_back": args.old_n_back,
        "new_n_back": args.new_n_back,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "appended_slot": slot,
        "old_baseline_controls": old_baseline,
        "old_post_growth_controls": old_controls,
        "new_post_growth_controls": new_controls,
        "old_retention_audit_before_growth": old_audit_before,
        "old_retention_audit_after_growth": old_audit_after,
        "new_retention_audit": new_audit,
        "old_retention_status": asdict(old_status),
        "new_retention_status": asdict(new_status),
        "old_external_state_unchanged": (
            old_digest_before_append == old_digest_after_append
        ),
        "shared_controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "new_slot_training_fraction": sum(
            row.selected_slot_fraction for row in new_history
        )
        / max(1, len(new_history)),
        "slot_exploration_probability": args.slot_exploration,
        "unique_verifier_bits": sum(
            row.unique_verifier_bits for row in (*old_history, *new_history)
        ),
        "unique_logical_lifetimes": args.batch_size
        * (args.old_updates + args.new_updates),
        "optimizer_updates": args.old_updates + args.new_updates,
        "replayed_examples": sum(
            row.replayed_examples for row in (*old_history, *new_history)
        ),
        "old_training_history": [row.__dict__ for row in old_history],
        "new_training_history": [row.__dict__ for row in new_history],
        "promotion_gates": {
            "old_retained": old_status.protected,
            "new_mastered": new_status.protected,
            "causal_controls": controls_passed,
            "old_state_unchanged": old_digest_before_append
            == old_digest_after_append,
        },
        "claim_boundary": (
            "Two sequential relation-reader capabilities were acquired in "
            "isolated external slots over a frozen controller with zero replay; "
            "this remains bounded growth and does not establish unrestricted "
            "memory expansion or general continual learning."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
