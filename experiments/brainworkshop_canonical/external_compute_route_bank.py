"""Promote append-only routing across a growing external compute-file bank.

This is the next pressure rung after the two-file route experiment.  Four
independent opaque files learn different private temporal rule families while
one controller, event frontend, and generic register interpreter remain
fixed.  A memory-side route table discovers each file from scalar episode
outcomes and learned event keys.  Earlier files are never replayed or
optimized after they are mastered.

The result is deliberately bounded: it demonstrates scalable append-only
file routing and retention, not unrestricted program induction or general
continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    ControllerFeedback,
    IntentEvent,
    PersistentOpaqueContextRouteEvidence,
)

from .cross_family_rule_growth import RULES, CrossFamilyVerifier
from .external_compute_growth import (
    ACTION_COUNT,
    EVENT_WIDTH,
    ComputeGrowthSystem,
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)

ROUTE_BANK_SCHEMA = "neural-computer.brainworkshop-external-compute-route-bank.v1"
ROUTE_SELECTION_THRESHOLD = 0.99
ROUTE_MASTERY_THRESHOLD = 0.80
DEFAULT_SCHEDULE = (
    ("symbol_parity", 7),
    ("triplet_parity", 8),
    ("parity2", 10),
    ("switch_binary", 11),
)


def _context_key(system: ComputeGrowthSystem, cue_symbol: int) -> torch.Tensor:
    """Return the learned event representation for a rendered cue."""

    return system.agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )[0].detach()


def _routed_episode(
    system: ComputeGrowthSystem,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    seed: int,
    slot_count: int,
    exploration: float,
    probe_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one rendered lifetime through one outcome-selected file."""

    if not 1 <= slot_count <= len(system.instructions):
        raise ValueError("active route bank is outside the executable file bank")
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("route exploration must lie in [0, 1]")
    verifier = CrossFamilyVerifier(
        family=family,
        batch_size=32,
        steps=14,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    verifier.reset()
    batch_size = verifier.batch_size
    controller_state = system.agent.initial_state(batch_size, device="cpu")
    feedback = system.agent.initial_feedback(batch_size, device="cpu")
    previous_action = torch.zeros(batch_size, ACTION_COUNT)
    register_states = [
        system.machine.initial_state(batch_size, device="cpu")
        for _ in range(slot_count)
    ]
    route_key: torch.Tensor | None = None
    selected_slot: torch.Tensor | None = None
    rewards: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []

    while not verifier.done:
        with torch.no_grad():
            collection = system.agent.runtime.encode_streams(
                {"stimulus": verifier.observation()}
            )
            controller_output, controller_state = system.agent.runtime.step_events(
                collection,
                controller_state,
                feedback,
            )
        if selected_slot is None:
            route_key = collection.payload[:, 0].detach().clone()
            if probe_all:
                selected_slot = torch.arange(batch_size, dtype=torch.long) % slot_count
            else:
                selected_slot = evidence.preferred_slots(route_key)
                if exploration:
                    newest = torch.full_like(selected_slot, slot_count - 1)
                    explore = torch.rand(batch_size) < exploration
                    selected_slot = torch.where(explore, newest, selected_slot)

        slot_logits: list[torch.Tensor] = []
        for slot in range(slot_count):
            present = selected_slot == slot
            executed, register_states[slot] = system.machine.read_execute_register(
                event=collection.payload[:, 0],
                action=previous_action,
                outcome=feedback.reward,
                intention=controller_output.intention,
                state=register_states[slot],
                instructions=(system.instructions[slot],),
                basis_slots=(slot,),
                present=present,
            )
            slot_logits.append(
                system.decoders[slot](IntentEvent(system.readouts[slot](executed)))
            )
        logits = torch.stack(slot_logits, dim=1).gather(
            1,
            selected_slot[:, None, None].expand(-1, 1, ACTION_COUNT),
        ).squeeze(1)
        probabilities = logits.softmax(dim=-1)
        action = logits.argmax(dim=-1)
        propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        scored = verifier.score(action)
        rewards.append(scored.reward)
        eligible.append(scored.eligible)
        feedback = ControllerFeedback(
            action=system.agent.keypress_encoder(action),
            reward=scored.reward,
            propensity=propensity,
            has_feedback=torch.ones(batch_size),
        )
        previous_action = F.one_hot(action, ACTION_COUNT).to(torch.float32)

    if route_key is None or selected_slot is None:
        raise RuntimeError("routed episode did not expose a route decision")
    reward_tensor = torch.stack(rewards, dim=1)
    eligible_tensor = torch.stack(eligible, dim=1)
    accuracy = (reward_tensor * eligible_tensor).sum(dim=1) / eligible_tensor.sum(
        dim=1
    ).clamp_min(1.0)
    return route_key, selected_slot, accuracy, eligible_tensor.sum(dim=1)


def _calibrate_source(
    system: ComputeGrowthSystem,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    lifetimes: int,
    seed: int,
) -> int:
    for lifetime in range(lifetimes):
        key, selected, accuracy, _ = _routed_episode(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            seed=seed + lifetime,
            slot_count=1,
            exploration=0.0,
        )
        evidence.observe_batch(key, selected, accuracy)
    return lifetimes


def _train_route(
    system: ComputeGrowthSystem,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    target_slot: int,
    updates: int,
    seed: int,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    active_slots = target_slot + 1
    for update in range(1, updates + 1):
        torch.manual_seed(seed + update * 10_007)
        key, selected, accuracy, bits = _routed_episode(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            seed=seed + update,
            slot_count=active_slots,
            exploration=0.5,
        )
        evidence.observe_batch(key, selected, accuracy)
        history.append(
            {
                "update": update,
                "accuracy": float(accuracy.mean()),
                "target_slot_fraction": float((selected == target_slot).float().mean()),
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return history


def _evaluate_route(
    system: ComputeGrowthSystem,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    expected_slot: int,
    seed: int,
    lifetimes: int,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for lifetime in range(lifetimes):
        key, selected, accuracy, bits = _routed_episode(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            seed=seed + lifetime,
            slot_count=evidence.slot_count,
            exploration=0.0,
        )
        del key
        rows.append(
            {
                "accuracy": float(accuracy.mean()),
                "selected_slot_fraction": float(
                    (selected == expected_slot).float().mean()
                ),
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return {
        "accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
        "selected_slot_fraction": sum(
            float(row["selected_slot_fraction"]) for row in rows
        )
        / len(rows),
        "unique_verifier_bits": sum(int(row["unique_verifier_bits"]) for row in rows),
        "replayed_examples": 0,
        "lifetimes": rows,
    }


def _stable_evaluation(result: dict[str, object]) -> bool:
    lifetimes = result["lifetimes"]
    return bool(lifetimes) and min(
        float(row["accuracy"]) for row in lifetimes  # type: ignore[index]
    ) >= ROUTE_MASTERY_THRESHOLD


def _all_modules(system: ComputeGrowthSystem):
    return _common_modules(system) + tuple(
        module
        for slot in range(len(system.instructions))
        for module in _slot_modules(system, slot)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(DEFAULT_SCHEDULE) != args.slot_count:
        raise ValueError("the promoted route-bank rung uses four fixed families")
    if min(
        args.slot_count,
        args.file_updates,
        args.route_updates,
        args.route_calibration_lifetimes,
        args.batch_size,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("route-bank budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated route-bank harness requires batch size 32")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    schedule = DEFAULT_SCHEDULE[:-1] + (
        (args.final_family, DEFAULT_SCHEDULE[-1][1]),
    )

    started = perf_counter()
    system = _build(
        args.seed,
        slot_count=args.slot_count,
        basis_hidden=args.basis_hidden,
    )
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    all_modules = _all_modules(system)
    histories: list[list[dict[str, float | int]]] = []
    direct: list[list[dict[str, float | int]]] = []
    file_digests_before: list[str] = []

    for slot, (family, cue_symbol) in enumerate(schedule):
        _set_requires_grad(all_modules, False)
        train_modules = _slot_modules(system, slot)
        if slot == 0:
            train_modules = _common_modules(system) + train_modules
        _set_requires_grad(train_modules, True)
        histories.append(
            _train_stage(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue_symbol,
                updates=args.file_updates,
                batch_size=args.batch_size,
                steps=14,
                seed=args.seed + 10_000 * (slot + 1),
                learning_rate=args.learning_rate,
            )
        )
        _set_requires_grad(all_modules, False)
        direct.append(
            _evaluate(
                system,
                family=family,
                slot=slot,
                cue_symbol=cue_symbol,
                lifetimes=args.retention_lifetimes,
                batch_size=args.batch_size,
                steps=14,
                seed=args.seed + 50_000 + slot * 1_000,
            )
        )
        file_digests_before.append(_digest(*_slot_modules(system, slot)))

    evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    route_histories: list[list[dict[str, float | int]]] = []
    for slot, (family, cue_symbol) in enumerate(schedule):
        if slot == 0:
            evidence.append_slot()
            _calibrate_source(
                system,
                evidence,
                family=family,
                cue_symbol=cue_symbol,
                lifetimes=args.route_calibration_lifetimes,
                seed=args.seed + 100_000,
            )
            route_histories.append([])
        else:
            evidence.append_slot()
            route_histories.append(
                _train_route(
                    system,
                    evidence,
                    family=family,
                    cue_symbol=cue_symbol,
                    target_slot=slot,
                    updates=args.route_updates,
                    seed=args.seed + 200_000 + slot * 10_000,
                )
            )

    routed = [
        _evaluate_route(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            expected_slot=slot,
            seed=args.seed + 300_000 + slot * 1_000,
            lifetimes=args.retention_lifetimes,
        )
        for slot, (family, cue_symbol) in enumerate(schedule)
    ]
    unseen_family, _ = schedule[-1]
    unseen = _evaluate_route(
        system,
        evidence,
        family=unseen_family,
        cue_symbol=9,
        expected_slot=0,
        seed=args.seed + 400_000,
        lifetimes=args.retention_lifetimes,
    )
    no_file_evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    no_file_evidence.append_slot()
    no_file = _evaluate_route(
        system,
        no_file_evidence,
        family=unseen_family,
        cue_symbol=schedule[-1][1],
        expected_slot=0,
        seed=args.seed + 500_000,
        lifetimes=args.retention_lifetimes,
    )
    restored = PersistentOpaqueContextRouteEvidence.from_payload(evidence.payload())
    restored_last = _evaluate_route(
        system,
        restored,
        family=schedule[-1][0],
        cue_symbol=schedule[-1][1],
        expected_slot=len(schedule) - 1,
        seed=args.seed + 303_000,
        lifetimes=args.retention_lifetimes,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    file_digests_after = [
        _digest(*_slot_modules(system, slot))
        for slot in range(args.slot_count)
    ]
    gates = {
        "all_direct_files_mastered": all(
            _stable_evaluation(
                {
                    "lifetimes": rows,
                }
            )
            for rows in direct
        ),
        "all_routed_files_mastered": all(
            float(result["accuracy"]) >= ROUTE_MASTERY_THRESHOLD
            for result in routed
        ),
        "all_routes_select_correct_file": all(
            float(result["selected_slot_fraction"]) >= ROUTE_SELECTION_THRESHOLD
            for result in routed
        ),
        "unseen_context_falls_back_to_oldest_file": unseen[
            "selected_slot_fraction"
        ]
        >= ROUTE_SELECTION_THRESHOLD,
        "no_file_cannot_reach_newest_task": no_file["accuracy"] < 0.70,
        "route_reload_exact": routed[-1] == restored_last,
        "prior_files_unchanged_after_growth": file_digests_before
        == file_digests_after,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    training_bits = args.batch_size * args.file_updates * sum(
        14 - RULES[family].warmup for family, _cue in schedule
    )
    training_bits += args.batch_size * args.route_updates * sum(
        14 - RULES[family].warmup for family, _cue in schedule[1:]
    )
    training_bits += args.batch_size * args.route_calibration_lifetimes * (
        14 - RULES[schedule[0][0]].warmup
    )
    report = {
        "schema": ROUTE_BANK_SCHEMA,
        "claim_boundary": (
            "Outcome-only content-addressed routing across four isolated generic "
            "external compute files with a frozen controller and zero replay; "
            "this remains bounded append-only growth and is not unrestricted "
            "memory expansion or general continual learning."
        ),
        "architecture": {
            "route_query": "learned_event_tensor_key",
            "route_feedback": "terminal_scalar_episode_accuracy",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "file_abi": "opaque_instruction_plus_event_window_only_compute_basis_v1",
            "slot_count": args.slot_count,
            "schedule": [
                {"family": family, "cue": cue} for family, cue in schedule
            ],
            "unknown_context_policy": "append_order_fallback",
            "basis_hidden": args.basis_hidden,
        },
        "seed": args.seed,
        "direct": direct,
        "routed": routed,
        "unseen_context": unseen,
        "no_file": no_file,
        "restored_last": restored_last,
        "file_digests_before": file_digests_before,
        "file_digests_after": file_digests_after,
        "route_history_tails": [history[-5:] for history in route_histories],
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": training_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"])
                for rows in direct
                for row in rows
            )
            + sum(int(result["unique_verifier_bits"]) for result in routed),
            "unique_logical_lifetimes": args.batch_size
            * (args.slot_count * args.file_updates + (args.slot_count - 1) * args.route_updates),
            "optimizer_updates": args.slot_count * args.file_updates,
            "route_memory_updates": args.route_calibration_lifetimes
            + (args.slot_count - 1) * args.route_updates,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": training_bits if all(gates.values()) else None,
        },
        "status": "promoted_four_file_route_bank" if all(gates.values()) else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--slot-count", type=int, default=4)
    parser.add_argument("--file-updates", type=int, default=192)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--basis-hidden", type=int, default=32)
    parser.add_argument(
        "--final-family",
        choices=("switch", "switch_binary", "nback2"),
        default="switch_binary",
    )
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
