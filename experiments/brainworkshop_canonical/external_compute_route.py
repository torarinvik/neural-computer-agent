"""Discover the correct external computation file from scalar outcomes.

The preceding external-computation audit selected a file explicitly so that
computation acquisition could be measured in isolation. This audit keeps the
same frozen controller, frontend, and generic event-window compute files, then
adds a content-addressed memory-side route ledger. A rendered cue is the only
route query; the ledger receives no family name, file index, correct action, or
target bit.

The source file is calibrated and protected before the target file is appended.
The target context is discovered through bounded exploration and scalar
episode outcomes. Unknown contexts fall back to append order, which prevents a
new file from winning an unrelated cue through accidental linear
generalization. This is cue-conditioned external route discovery, not
arbitrary program induction or unrestricted continual learning.
"""

from __future__ import annotations

import argparse
import hashlib
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

from .cross_family_rule_growth import CrossFamilyVerifier
from .external_compute_growth import (
    ACTION_COUNT,
    EVENT_WIDTH,
    SOURCE_FAMILY,
    TARGET_FAMILY,
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)

EXTERNAL_COMPUTE_ROUTE_SCHEMA = (
    "neural-computer.brainworkshop-external-compute-route.v2"
)
SOURCE_CUE = 7
TARGET_CUE = 8
SHUFFLED_CUE = 9
ROUTE_CAPACITY = 2
ROUTE_MASTERY_THRESHOLD = 0.80
ROUTE_SELECTION_THRESHOLD = 0.99
SOURCE_ELIGIBLE_TRIALS = 14
TARGET_ELIGIBLE_TRIALS = 11


def _context_digest(context_record: object) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(context_record, sort_keys=True).encode())
    return digest.hexdigest()


def _episode(
    system,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    seed: int,
    slot_count: int,
    exploration: float,
    forced_slot: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run one routed rendered lifetime and return scalar route evidence."""

    if not 1 <= slot_count <= ROUTE_CAPACITY:
        raise ValueError("route slot count is outside the active file bank")
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("route exploration must lie in [0, 1]")
    if forced_slot is not None and not 0 <= forced_slot < slot_count:
        raise ValueError("forced route slot is outside the active file bank")
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
        for _ in range(ROUTE_CAPACITY)
    ]
    selected_slot: torch.Tensor | None = None
    route_key: torch.Tensor | None = None
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
            if forced_slot is not None:
                selected_slot = torch.full(
                    (batch_size,), forced_slot, dtype=torch.long
                )
            else:
                selected_slot = evidence.preferred_slots(route_key)
                if exploration:
                    newest = torch.full_like(selected_slot, slot_count - 1)
                    sample = torch.rand(batch_size) < exploration
                    selected_slot = torch.where(sample, newest, selected_slot)

        slot_logits: list[torch.Tensor] = []
        for slot in range(ROUTE_CAPACITY):
            active = selected_slot == slot
            executed, register_states[slot] = (
                system.machine.read_execute_register(
                    event=collection.payload[:, 0],
                    action=previous_action,
                    outcome=feedback.reward,
                    intention=controller_output.intention,
                    state=register_states[slot],
                    instructions=(system.instructions[slot],),
                    basis_slots=(slot,),
                    present=active,
                )
            )
            slot_logits.append(
                system.decoders[slot](
                    IntentEvent(system.readouts[slot](executed))
                )
            )
        selected_logits = torch.stack(slot_logits, dim=1).gather(
            1,
            selected_slot[:, None, None].expand(-1, 1, ACTION_COUNT),
        ).squeeze(1)
        action_probabilities = selected_logits.softmax(dim=-1)
        action = selected_logits.argmax(dim=-1)
        action_propensity = action_probabilities.gather(
            1, action[:, None]
        ).squeeze(1)
        scored = verifier.score(action)
        rewards.append(scored.reward)
        eligible.append(scored.eligible)
        feedback = ControllerFeedback(
            action=system.agent.keypress_encoder(action),
            reward=scored.reward,
            propensity=action_propensity,
            has_feedback=torch.ones(batch_size),
        )
        previous_action = F.one_hot(action, ACTION_COUNT).to(torch.float32)

    reward_tensor = torch.stack(rewards, dim=1)
    eligible_tensor = torch.stack(eligible, dim=1)
    accuracy = (
        (reward_tensor * eligible_tensor).sum(dim=1)
        / eligible_tensor.sum(dim=1).clamp_min(1.0)
    )
    if route_key is None or selected_slot is None:
        raise RuntimeError("routed episode did not expose a route decision")
    return (
        route_key,
        selected_slot,
        accuracy,
        action_propensity,
        eligible_tensor.sum(dim=1),
    )


def _calibrate_source(
    system,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    lifetimes: int,
    seed: int,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    for lifetime in range(lifetimes):
        key, selected, accuracy, _, bits = _episode(
            system,
            evidence,
            family=SOURCE_FAMILY,
            cue_symbol=SOURCE_CUE,
            seed=seed + lifetime,
            slot_count=1,
            exploration=0.0,
            forced_slot=0,
        )
        evidence.observe_batch(key, selected, accuracy)
        history.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(accuracy.mean()),
                "selected_slot_fraction": float((selected == 0).float().mean()),
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return history


def _train_target_route(
    system,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    updates: int,
    seed: int,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        torch.manual_seed(seed + update * 10_007)
        key, selected, accuracy, _, bits = _episode(
            system,
            evidence,
            family=TARGET_FAMILY,
            cue_symbol=TARGET_CUE,
            seed=seed + update,
            slot_count=ROUTE_CAPACITY,
            exploration=0.5,
        )
        evidence.observe_batch(key, selected, accuracy)
        history.append(
            {
                "update": update,
                "episode_accuracy": float(accuracy.mean()),
                "target_slot_fraction": float((selected == 1).float().mean()),
                "unique_verifier_bits": int(bits.sum()),
                "replayed_examples": 0,
            }
        )
    return history


def _evaluate_route(
    system,
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
        _key, selected, accuracy, propensity, bits = _episode(
            system,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            seed=seed + lifetime,
            slot_count=ROUTE_CAPACITY,
            exploration=0.0,
        )
        rows.append(
            {
                "accuracy": float(accuracy.mean()),
                "selected_slot_fraction": float(
                    (selected == expected_slot).float().mean()
                ),
                "route_propensity": float(propensity.mean()),
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
        "route_propensity": sum(float(row["route_propensity"]) for row in rows)
        / len(rows),
        "unique_verifier_bits": sum(int(row["unique_verifier_bits"]) for row in rows),
        "replayed_examples": 0,
        "lifetimes": rows,
    }


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= ROUTE_MASTERY_THRESHOLD


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.route_updates,
        args.route_calibration_lifetimes,
        args.batch_size,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("external compute route budgets must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated route harness requires batch size 32")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    common = _common_modules(system)
    source_modules = _slot_modules(system, 0)
    target_modules = _slot_modules(system, 1)

    _set_requires_grad(common + source_modules, True)
    _set_requires_grad(target_modules, False)
    source_history = _train_stage(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=SOURCE_CUE,
        updates=args.source_updates,
        batch_size=args.batch_size,
        steps=14,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
    )
    _set_requires_grad(common + source_modules, False)
    _set_requires_grad(target_modules, True)
    target_history = _train_stage(
        system,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        updates=args.target_updates,
        batch_size=args.batch_size,
        steps=14,
        seed=args.seed + 20_000,
        learning_rate=args.learning_rate,
    )
    _set_requires_grad(target_modules, False)

    source_file_before_route = _digest(*source_modules)
    target_file_before_route = _digest(*target_modules)
    source_direct = _evaluate(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=SOURCE_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=14,
        seed=args.seed + 30_000,
    )
    target_direct = _evaluate(
        system,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=14,
        seed=args.seed + 40_000,
    )

    evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    evidence.append_slot()
    source_route_history = _calibrate_source(
        system,
        evidence,
        lifetimes=args.route_calibration_lifetimes,
        seed=args.seed + 50_000,
    )
    evidence.append_slot()
    source_context_before = _context_digest(evidence.payload()["contexts"][0])
    target_route_history = _train_target_route(
        system,
        evidence,
        updates=args.route_updates,
        seed=args.seed + 60_000,
    )
    source_routed = _evaluate_route(
        system,
        evidence,
        family=SOURCE_FAMILY,
        cue_symbol=SOURCE_CUE,
        expected_slot=0,
        seed=args.seed + 70_000,
        lifetimes=args.retention_lifetimes,
    )
    target_routed = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        expected_slot=1,
        seed=args.seed + 80_000,
        lifetimes=args.retention_lifetimes,
    )
    shuffled_routed = _evaluate_route(
        system,
        evidence,
        family=TARGET_FAMILY,
        cue_symbol=SHUFFLED_CUE,
        expected_slot=0,
        seed=args.seed + 90_000,
        lifetimes=args.retention_lifetimes,
    )

    no_file_evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    no_file_evidence.append_slot()
    no_file_routed = _evaluate_route(
        system,
        no_file_evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        expected_slot=0,
        seed=args.seed + 100_000,
        lifetimes=args.retention_lifetimes,
    )
    restored_evidence = PersistentOpaqueContextRouteEvidence.from_payload(
        evidence.payload()
    )
    restored_target = _evaluate_route(
        system,
        restored_evidence,
        family=TARGET_FAMILY,
        cue_symbol=TARGET_CUE,
        expected_slot=1,
        seed=args.seed + 80_000,
        lifetimes=args.retention_lifetimes,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_context_after = _context_digest(evidence.payload()["contexts"][0])
    gates = {
        "source_direct_mastery": _stable(source_direct),
        "target_direct_mastery": _stable(target_direct),
        "source_route_mastery": source_routed["accuracy"]
        >= ROUTE_MASTERY_THRESHOLD,
        "target_route_mastery": target_routed["accuracy"]
        >= ROUTE_MASTERY_THRESHOLD,
        "source_route_selected": source_routed["selected_slot_fraction"]
        >= ROUTE_SELECTION_THRESHOLD,
        "target_route_selected": target_routed["selected_slot_fraction"]
        >= ROUTE_SELECTION_THRESHOLD,
        "unseen_cue_falls_back_to_source": shuffled_routed[
            "selected_slot_fraction"
        ]
        >= ROUTE_SELECTION_THRESHOLD,
        "no_file_cannot_reach_target": no_file_routed["accuracy"] < 0.70,
        "route_reload_exact": target_routed == restored_target,
        "protected_source_context_unchanged": source_context_before
        == source_context_after,
        "source_file_unchanged": source_file_before_route
        == _digest(*source_modules),
        "target_file_unchanged_during_routing": target_file_before_route
        == _digest(*target_modules),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    training_bits = args.batch_size * (
        args.source_updates * SOURCE_ELIGIBLE_TRIALS
        + args.target_updates * TARGET_ELIGIBLE_TRIALS
        + args.route_calibration_lifetimes * SOURCE_ELIGIBLE_TRIALS
        + args.route_updates * TARGET_ELIGIBLE_TRIALS
    )
    audit_bits = (
        sum(int(row["unique_verifier_bits"]) for row in source_direct)
        + sum(int(row["unique_verifier_bits"]) for row in target_direct)
        + int(source_routed["unique_verifier_bits"])
        + int(target_routed["unique_verifier_bits"])
        + int(shuffled_routed["unique_verifier_bits"])
        + int(no_file_routed["unique_verifier_bits"])
    )
    report = {
        "schema": EXTERNAL_COMPUTE_ROUTE_SCHEMA,
        "claim_boundary": (
            "Outcome-only cue-conditioned discovery of a retained external "
            "compute file over rendered temporal tasks with a frozen controller; "
            "unknown cues use conservative append-order fallback. This is not "
            "arbitrary program induction, unrestricted memory growth, or general "
            "continual learning."
        ),
        "architecture": {
            "route_query": "content_addressed_learned_event_tensor_key",
            "route_feedback": "terminal_scalar_episode_accuracy",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "files": "opaque_instruction_plus_event_window_only_compute_basis",
            "active_file_capacity": ROUTE_CAPACITY,
            "source_cue": SOURCE_CUE,
            "target_cue": TARGET_CUE,
            "unseen_cue": SHUFFLED_CUE,
            "protected_prefix": 1,
            "exploration": 0.5,
            "unknown_context_policy": "append_order_fallback",
        },
        "seed": args.seed,
        "source_history_tail": source_history[-5:],
        "target_history_tail": target_history[-5:],
        "source_route_history_tail": source_route_history[-5:],
        "target_route_history_tail": target_route_history[-5:],
        "direct": {"source": source_direct, "target": target_direct},
        "routed": {
            "source": source_routed,
            "target": target_routed,
            "unseen_cue": shuffled_routed,
            "no_file": no_file_routed,
            "restored_target": restored_target,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": training_bits,
            "audit_verifier_bits": audit_bits,
            "unique_logical_lifetimes": args.batch_size
            * (
                args.source_updates
                + args.target_updates
                + args.route_calibration_lifetimes
                + args.route_updates
            ),
            "optimizer_updates": args.source_updates + args.target_updates,
            "route_memory_updates": args.route_calibration_lifetimes
            + args.route_updates,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "retention_threshold": ROUTE_MASTERY_THRESHOLD,
        },
        "status": "promoted_external_compute_route" if all(gates.values()) else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
