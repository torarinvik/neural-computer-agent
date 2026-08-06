"""Audit two generations of failure-triggered external capacity growth.

The first new capability grows from capacity five to six after fresh opaque
failure.  A second capability is then provisioned at capacity six, fails at
the next horizon, and grows to capacity seven.  Both earlier capabilities
must remain independently usable while the controller stays frozen and no
old examples are replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from neural_computer import ContentAddressedMemory

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .sequential_train import _controls, _route_audit
from .trainer import (
    train_adaptive_relation_capability,
    train_existing_adaptive_relation_capability,
    train_reward_only,
)

OLD_N_BACK = 2
FIRST_N_BACK = 6
SECOND_N_BACK = 7
OLD_CUE = 4
FIRST_CUE = 5
SECOND_CUE = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--initial-updates", type=int, default=256)
    parser.add_argument("--continued-updates", type=int, default=512)
    parser.add_argument("--probe-batches", type=int, default=4)
    parser.add_argument("--discovery-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--first-initial-capacity", type=int, default=5)
    parser.add_argument("--first-grown-capacity", type=int, default=6)
    parser.add_argument("--second-initial-capacity", type=int, default=6)
    parser.add_argument("--second-grown-capacity", type=int, default=7)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--slot-exploration", type=float, default=0.5)
    parser.add_argument("--failure-threshold", type=float, default=0.8)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _bank_digest(
    agent: CanonicalBrainWorkshopAgent,
    extension_slots: tuple[int, ...],
) -> str:
    """Hash the controller and all mastered external slots in scope."""

    modules: list[torch.nn.Module] = [
        agent.controller,
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    ]
    for slot in extension_slots:
        modules.extend(
            (agent.extensions[slot - 1], agent.extension_decoder(slot))
        )
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in module.state_dict().items():
            digest.update(f"{module_index}:{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _failure_probe(
    agent: CanonicalBrainWorkshopAgent,
    *,
    verifier_n_back: int,
    cue_symbol: int,
    candidate_slot: int,
    slot_count: int,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
) -> list[dict[str, float]]:
    """Measure all opaque slot outcomes without writing route evidence."""

    results: list[dict[str, float]] = []
    for seed in seeds:
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        rollout = agent.rollout(
            NBackVerifier(
                batch_size=batch_size,
                n_back=verifier_n_back,
                steps=steps,
                symbol_count=4,
                cue_symbol=cue_symbol,
                seed=seed,
            ),
            sample=False,
            record_retention=False,
            context_route=True,
            record_context_route=False,
        )
        eligible = rollout.eligible
        denominator = eligible.sum().clamp_min(1)
        row: dict[str, float] = {
            "seed": float(seed),
            "candidate_slot": float(candidate_slot),
        }
        for slot in range(slot_count):
            attempted = (rollout.selected_slots == slot) & eligible
            row[f"slot_{slot}_eligible_fraction"] = float(
                attempted.sum() / denominator
            )
            row[f"slot_{slot}_accuracy"] = float(
                rollout.rewards[attempted].mean() if bool(attempted.any()) else 0.0
            )
        results.append(row)
    return results


def _discover(
    agent: CanonicalBrainWorkshopAgent,
    *,
    verifier_n_back: int,
    cue_symbol: int,
    target_slot: int,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
) -> list[dict[str, float]]:
    """Discover a grown route from ordinary fallback outcomes."""

    results: list[dict[str, float]] = []
    for seed in seeds:
        memory = agent.runtime.memory
        if isinstance(memory, ContentAddressedMemory):
            memory.clear()
        rollout = agent.rollout(
            NBackVerifier(
                batch_size=batch_size,
                n_back=verifier_n_back,
                steps=steps,
                symbol_count=4,
                cue_symbol=cue_symbol,
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
                "seed": float(seed),
                "eligible_accuracy": float(rollout.eligible_accuracy.mean()),
                "selected_slot_fraction": float(
                    ((rollout.selected_slots == target_slot) & eligible).sum()
                    / denominator
                ),
                "first_selected_slot": float(
                    rollout.selected_slots[:, 0].mode().values
                ),
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
        raise RuntimeError(f"route discovery did not create cue {cue_symbol} record")
    status = record.evidence.status()
    return {
        "preferred_order": list(record.evidence.preferred_order()),
        "attempts": list(status.attempts),
        "successes": list(status.successes),
        "protected": list(status.protected),
        "preferred_slot": status.preferred_slot,
    }


def _reload_routes(
    agent: CanonicalBrainWorkshopAgent,
    *,
    capacities: tuple[int, ...],
    cues: tuple[int, ...],
    n_backs: tuple[int, ...],
    steps: int,
    seed: int,
) -> dict[str, list[int]]:
    restored = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        n_back=OLD_N_BACK,
        reader_kind="relation",
        seed=seed,
    )
    for index, capacity in enumerate(capacities):
        restored.add_adaptive_relation_capability(
            memory_capacity=capacity,
            seed=seed + 5000 + index,
        )
    restored.load_route_state_payload(agent.route_state_payload())
    routes: dict[str, list[int]] = {}
    for index, (cue, n_back) in enumerate(zip(cues, n_backs)):
        rollout = restored.rollout(
            NBackVerifier(
                batch_size=2,
                n_back=n_back,
                steps=steps,
                symbol_count=4,
                cue_symbol=cue,
                seed=seed + 13000 + index,
            ),
            sample=False,
            record_retention=False,
            context_route=True,
        )
        routes[str(cue)] = rollout.selected_slots[:, 0].tolist()
    return routes


def _all_controls_passed(controls: tuple[dict[str, float], ...]) -> bool:
    return all(
        result["fresh"] >= 0.8
        and result["time_shuffle"] <= 0.75
        and result["history_reset"] <= 0.75
        for result in controls
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.old_updates,
        args.initial_updates,
        args.continued_updates,
        args.probe_batches,
        args.discovery_batches,
        args.batch_size,
        args.first_initial_capacity,
        args.first_grown_capacity,
        args.second_initial_capacity,
        args.second_grown_capacity,
    ) < 1:
        raise ValueError("updates, batches, capacities, and batch size must be positive")
    if not (
        args.first_initial_capacity < args.first_grown_capacity
        and args.second_initial_capacity < args.second_grown_capacity
        and args.first_grown_capacity == args.second_initial_capacity
    ):
        raise ValueError("recursive growth capacities must form adjacent increasing pairs")
    if not 0.0 < args.failure_threshold <= 1.0:
        raise ValueError("failure threshold must lie in (0, 1]")
    if args.calibration_lifetimes < 1:
        raise ValueError("calibration lifetimes must be positive")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
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
        seeds=tuple(
            args.seed + 2000 + index for index in range(args.calibration_lifetimes)
        ),
        context_route=True,
        cue_symbol=OLD_CUE,
        record_context_route=True,
    )

    first_slot, first_initial_history = train_adaptive_relation_capability(
        agent,
        verifier_n_back=FIRST_N_BACK,
        memory_capacity=args.first_initial_capacity,
        updates=args.initial_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 5000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=FIRST_CUE,
    )
    first_probe = _failure_probe(
        agent,
        verifier_n_back=FIRST_N_BACK,
        cue_symbol=FIRST_CUE,
        candidate_slot=first_slot,
        slot_count=2,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 4000 + index for index in range(args.probe_batches)),
    )
    first_failure = (
        sum(row[f"slot_{first_slot}_accuracy"] for row in first_probe)
        / len(first_probe)
        < args.failure_threshold
    )
    if not first_failure:
        raise RuntimeError("first growth probe did not produce the registered failure")
    agent.expand_adaptive_relation_capability(
        first_slot,
        memory_capacity=args.first_grown_capacity,
        reset_failed_reader=True,
        reset_seed=args.seed + 8000,
    )
    first_continued_history = train_existing_adaptive_relation_capability(
        agent,
        slot=first_slot,
        verifier_n_back=FIRST_N_BACK,
        updates=args.continued_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 9000,
        learning_rate=args.learning_rate,
        forced_slot=True,
        cue_symbol=FIRST_CUE,
    )
    first_validation = _route_audit(
        agent,
        n_back=FIRST_N_BACK,
        slot=first_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 10000 + index for index in range(3)),
        context_route=True,
        cue_symbol=FIRST_CUE,
        record_context_route=False,
    )
    first_discovery = _discover(
        agent,
        verifier_n_back=FIRST_N_BACK,
        cue_symbol=FIRST_CUE,
        target_slot=first_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(
            args.seed + 11000 + index for index in range(args.discovery_batches)
        ),
    )
    first_context_status = _context_status(agent, FIRST_CUE)
    mastered_bank_before_second = _bank_digest(agent, (first_slot,))

    second_slot, second_initial_history = train_adaptive_relation_capability(
        agent,
        verifier_n_back=SECOND_N_BACK,
        memory_capacity=args.second_initial_capacity,
        updates=args.initial_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 6000,
        learning_rate=args.learning_rate,
        exploration_probability=args.slot_exploration,
        context_route=True,
        cue_symbol=SECOND_CUE,
    )
    second_probe = _failure_probe(
        agent,
        verifier_n_back=SECOND_N_BACK,
        cue_symbol=SECOND_CUE,
        candidate_slot=second_slot,
        slot_count=3,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 7000 + index for index in range(args.probe_batches)),
    )
    second_failure = (
        sum(row[f"slot_{second_slot}_accuracy"] for row in second_probe)
        / len(second_probe)
        < args.failure_threshold
    )
    if not second_failure:
        raise RuntimeError("second growth probe did not produce the registered failure")
    agent.expand_adaptive_relation_capability(
        second_slot,
        memory_capacity=args.second_grown_capacity,
        reset_failed_reader=True,
        reset_seed=args.seed + 11000,
    )
    second_continued_history = train_existing_adaptive_relation_capability(
        agent,
        slot=second_slot,
        verifier_n_back=SECOND_N_BACK,
        updates=args.continued_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 10000,
        learning_rate=args.learning_rate,
        forced_slot=True,
        cue_symbol=SECOND_CUE,
    )
    second_validation = _route_audit(
        agent,
        n_back=SECOND_N_BACK,
        slot=second_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 12000 + index for index in range(3)),
        context_route=True,
        cue_symbol=SECOND_CUE,
        record_context_route=False,
    )
    second_discovery = _discover(
        agent,
        verifier_n_back=SECOND_N_BACK,
        cue_symbol=SECOND_CUE,
        target_slot=second_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(
            args.seed + 13000 + index for index in range(args.discovery_batches)
        ),
    )
    second_context_status = _context_status(agent, SECOND_CUE)

    base_controls = _controls(
        agent,
        n_back=OLD_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=OLD_CUE,
    )
    first_controls = _controls(
        agent,
        n_back=FIRST_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=FIRST_CUE,
    )
    second_controls = _controls(
        agent,
        n_back=SECOND_N_BACK,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=control_seeds,
        context_route=True,
        cue_symbol=SECOND_CUE,
    )
    base_retention = _route_audit(
        agent,
        n_back=OLD_N_BACK,
        slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 14000 + index for index in range(3)),
        context_route=True,
        cue_symbol=OLD_CUE,
        record_context_route=False,
    )
    first_retention = _route_audit(
        agent,
        n_back=FIRST_N_BACK,
        slot=first_slot,
        batch_size=args.batch_size,
        steps=args.steps,
        seeds=tuple(args.seed + 15000 + index for index in range(3)),
        context_route=True,
        cue_symbol=FIRST_CUE,
        record_context_route=False,
    )
    second_bank_unchanged = mastered_bank_before_second == _bank_digest(
        agent, (first_slot,)
    )
    reload_routes = _reload_routes(
        agent,
        capacities=(args.first_grown_capacity, args.second_grown_capacity),
        cues=(OLD_CUE, FIRST_CUE, SECOND_CUE),
        n_backs=(OLD_N_BACK, FIRST_N_BACK, SECOND_N_BACK),
        steps=args.steps,
        seed=args.seed,
    )
    new_first_score = min(row["eligible_accuracy"] for row in first_validation)
    new_second_score = min(row["eligible_accuracy"] for row in second_validation)
    old_score = min(row["eligible_accuracy"] for row in base_retention)
    first_retention_score = min(row["eligible_accuracy"] for row in first_retention)
    first_discovery_success = (
        first_context_status["preferred_slot"] == first_slot
        and first_context_status["preferred_order"][0] == first_slot
        and first_discovery[-1]["first_selected_slot"] == float(first_slot)
        and first_discovery[-1]["selected_slot_fraction"] == 1.0
    )
    second_discovery_success = (
        second_context_status["preferred_slot"] == second_slot
        and second_context_status["preferred_order"][0] == second_slot
        and second_discovery[-1]["first_selected_slot"] == float(second_slot)
        and second_discovery[-1]["selected_slot_fraction"] == 1.0
    )
    controls_passed = _all_controls_passed(
        (base_controls, first_controls, second_controls)
    )
    reload_passed = reload_routes == {
        str(OLD_CUE): [0, 0],
        str(FIRST_CUE): [first_slot, first_slot],
        str(SECOND_CUE): [second_slot, second_slot],
    }
    histories = (
        *old_history,
        *first_initial_history,
        *first_continued_history,
        *second_initial_history,
        *second_continued_history,
    )
    training_rollouts = (
        args.old_updates + 2 * (args.initial_updates + args.continued_updates)
    )
    calibration_rollouts = args.calibration_lifetimes
    probe_rollouts = 2 * args.probe_batches
    discovery_rollouts = 2 * args.discovery_batches
    validation_rollouts = 6
    retention_rollouts = 6
    control_rollouts = 27
    reload_rollouts = 3
    non_reload_rollouts = (
        training_rollouts
        + calibration_rollouts
        + probe_rollouts
        + discovery_rollouts
        + validation_rollouts
        + retention_rollouts
        + control_rollouts
    )
    all_rollouts = non_reload_rollouts + reload_rollouts
    training_bits = args.batch_size * (
        args.old_updates * (args.steps - OLD_N_BACK)
        + (args.initial_updates + args.continued_updates)
        * (args.steps - FIRST_N_BACK)
        + (args.initial_updates + args.continued_updates)
        * (args.steps - SECOND_N_BACK)
    )
    calibration_bits = args.batch_size * calibration_rollouts * (
        args.steps - OLD_N_BACK
    )
    probe_bits = args.batch_size * (
        args.probe_batches * (args.steps - FIRST_N_BACK)
        + args.probe_batches * (args.steps - SECOND_N_BACK)
    )
    discovery_bits = args.batch_size * (
        args.discovery_batches * (args.steps - FIRST_N_BACK)
        + args.discovery_batches * (args.steps - SECOND_N_BACK)
    )
    validation_bits = args.batch_size * (
        3 * (args.steps - FIRST_N_BACK)
        + 3 * (args.steps - SECOND_N_BACK)
    )
    retention_bits = args.batch_size * (
        3 * (args.steps - OLD_N_BACK)
        + 3 * (args.steps - FIRST_N_BACK)
    )
    control_bits = args.batch_size * (
        9 * (args.steps - OLD_N_BACK)
        + 9 * (args.steps - FIRST_N_BACK)
        + 9 * (args.steps - SECOND_N_BACK)
    )
    reload_bits = 2 * (
        (args.steps - OLD_N_BACK)
        + (args.steps - FIRST_N_BACK)
        + (args.steps - SECOND_N_BACK)
    )
    unique_verifier_bits = (
        training_bits
        + calibration_bits
        + probe_bits
        + discovery_bits
        + validation_bits
        + retention_bits
        + control_bits
        + reload_bits
    )
    verifier_outcome_events = (
        args.batch_size * non_reload_rollouts * (args.steps + 1)
        + 2 * reload_rollouts * (args.steps + 1)
    )
    report = {
        "schema": "neural-computer.brainworkshop-recursive-capacity-growth.v1",
        "status": (
            "promoted_recursive_failure_triggered_capacity_growth"
            if first_failure
            and second_failure
            and new_first_score >= 0.8
            and new_second_score >= 0.8
            and old_score >= 0.8
            and first_retention_score >= 0.8
            and first_discovery_success
            and second_discovery_success
            and controls_passed
            and reload_passed
            and second_bank_unchanged
            else "unpromoted_recursive_failure_triggered_capacity_growth"
        ),
        "old_n_back": OLD_N_BACK,
        "first_n_back": FIRST_N_BACK,
        "second_n_back": SECOND_N_BACK,
        "old_cue": OLD_CUE,
        "first_cue": FIRST_CUE,
        "second_cue": SECOND_CUE,
        "first_initial_capacity": args.first_initial_capacity,
        "first_grown_capacity": args.first_grown_capacity,
        "second_initial_capacity": args.second_initial_capacity,
        "second_grown_capacity": args.second_grown_capacity,
        "failure_threshold": args.failure_threshold,
        "reader_weights_reset_only_for_unmastered_candidates": True,
        "new_cue_calibration_performed": False,
        "old_calibration": old_calibration,
        "first_failure_probe": first_probe,
        "second_failure_probe": second_probe,
        "first_candidate_failure_triggered": first_failure,
        "second_candidate_failure_triggered": second_failure,
        "first_validation_after_growth": first_validation,
        "second_validation_after_growth": second_validation,
        "first_discovery_after_growth": first_discovery,
        "second_discovery_after_growth": second_discovery,
        "first_context_status": first_context_status,
        "second_context_status": second_context_status,
        "base_controls": base_controls,
        "first_controls": first_controls,
        "second_controls": second_controls,
        "base_forced_retention": base_retention,
        "first_forced_retention": first_retention,
        "reload_selected_routes": reload_routes,
        "prior_mastered_bank_unchanged": second_bank_unchanged,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in agent.controller.parameters()
        ),
        "unique_verifier_bits": unique_verifier_bits,
        "unique_logical_lifetimes": args.batch_size * non_reload_rollouts
        + 2 * reload_rollouts,
        "optimizer_updates": len(histories),
        "replayed_examples": sum(row.replayed_examples for row in histories),
        "verifier_outcome_events": verifier_outcome_events,
        "feedback_events": unique_verifier_bits,
        "rollout_accounting": {
            "training_rollouts": training_rollouts,
            "calibration_rollouts": calibration_rollouts,
            "probe_rollouts": probe_rollouts,
            "discovery_rollouts": discovery_rollouts,
            "validation_rollouts": validation_rollouts,
            "retention_rollouts": retention_rollouts,
            "control_rollouts": control_rollouts,
            "reload_rollouts": reload_rollouts,
            "non_reload_rollouts": non_reload_rollouts,
            "total_rollouts": all_rollouts,
            "unique_verifier_bits_by_phase": {
                "training": training_bits,
                "calibration": calibration_bits,
                "probe": probe_bits,
                "discovery": discovery_bits,
                "validation": validation_bits,
                "retention": retention_bits,
                "controls": control_bits,
                "reload": reload_bits,
            },
        },
        "promotion_gates": {
            "first_failure_trigger": first_failure,
            "second_failure_trigger": second_failure,
            "first_capacity_grew": args.first_grown_capacity > args.first_initial_capacity,
            "second_capacity_grew": args.second_grown_capacity > args.second_initial_capacity,
            "first_capability_mastered": new_first_score >= 0.8,
            "second_capability_mastered": new_second_score >= 0.8,
            "base_capability_retained": old_score >= 0.8,
            "first_capability_retained": first_retention_score >= 0.8,
            "automatic_first_route_after_growth": first_discovery_success,
            "automatic_second_route_after_growth": second_discovery_success,
            "causal_controls": controls_passed,
            "route_state_reload": reload_passed,
            "prior_mastered_bank_unchanged": second_bank_unchanged,
            "controller_frozen": all(
                not parameter.requires_grad
                for parameter in agent.controller.parameters()
            ),
            "zero_replay": sum(row.replayed_examples for row in histories) == 0,
            "new_cue_calibration_performed": False,
        },
        "claim_boundary": (
            "Fresh scalar failure triggered two sequential replacements of "
            "unmastered external readers, expanding capacity five to six and "
            "six to seven while the frozen controller and both mastered prior "
            "capabilities remained intact. This promotes recursive bounded "
            "failure-triggered external capacity growth, not unrestricted "
            "memory growth, arbitrary new computation, or general continual "
            "learning."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
