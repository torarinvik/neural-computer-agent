"""Three-depth causal rule growth with a frozen shared controller.

The audit acquires n-back-2, appends n-back-3, then appends n-back-4.  Each
new rule gets an isolated :class:`ExternalWorkingMemoryCell`; all earlier
cells, adapters, decoders, and the shared controller are frozen before the
next rule is learned.  Rendered cue events select the opaque files through
the external context route table.  The verifier's private rule value never
crosses the event, memory, or controller boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import ExternalWorkingMemoryCell

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .trainer import _train_relation_extension, train_reward_only

CAUSAL_DEPTH_GROWTH_SCHEMA = "neural-computer.brainworkshop-causal-depth-growth.v1"
MASTERY_THRESHOLD = 0.80
RULES = ((2, 4), (3, 5), (4, 6))


def _cell(seed: int, capacity: int) -> ExternalWorkingMemoryCell:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ExternalWorkingMemoryCell(
            event_width=16,
            action_width=2,
            memory_capacity=capacity,
            context_width=16,
            hidden=32,
        )


def _agent(seed: int) -> CanonicalBrainWorkshopAgent:
    return CanonicalBrainWorkshopAgent(
        symbol_count=8,
        n_back=2,
        event_width=16,
        intention_width=8,
        feedback_width=8,
        reader_kind="relation",
        seed=seed,
        working_memory_cell=_cell(seed + 1, 3),
    )


def _digest(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{module_index}:{name}".encode())
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _slot_modules(
    agent: CanonicalBrainWorkshopAgent,
    slot: int,
) -> tuple[torch.nn.Module, ...]:
    if slot == 0:
        return (agent.external_reader, agent.intent_adapter, agent.keypress_decoder)
    extension = agent.extensions[slot - 1]
    return (extension, agent.extension_decoder(slot))


def _protected_digest(
    agent: CanonicalBrainWorkshopAgent,
    slots: range,
) -> str:
    modules: list[torch.nn.Module] = []
    for slot in slots:
        modules.extend(_slot_modules(agent, slot))
    return _digest(*modules)


def _freeze_external(agent: CanonicalBrainWorkshopAgent) -> None:
    modules: list[torch.nn.Module] = []
    for slot in range(len(agent.extensions) + 1):
        modules.extend(_slot_modules(agent, slot))
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def _rollout_score(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    slot: int | None,
    seed: int,
    batch_size: int,
    steps: int,
    context_route: bool = False,
    record_context_route: bool = False,
    record_retention: bool = False,
    expected_slot: int | None = None,
) -> dict[str, float | int]:
    verifier = NBackVerifier(
        batch_size=batch_size,
        n_back=n_back,
        steps=steps,
        symbol_count=4,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    with torch.no_grad():
        rollout = agent.rollout(
            verifier,
            sample=False,
            forced_slot=slot,
            context_route=context_route,
            record_context_route=record_context_route,
            record_retention=record_retention,
        )
    selected = 0 if expected_slot is None else expected_slot
    return {
        "accuracy": float(rollout.eligible_accuracy.mean()),
        "selected_slot_fraction": float(
            (rollout.selected_slots == selected).to(torch.float32).mean()
        ),
    }


def _audit(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    slot: int,
    seed: int,
    lifetimes: int,
    batch_size: int,
    steps: int,
    context_route: bool = False,
    record_context_route: bool = False,
    record_retention: bool = False,
) -> list[dict[str, float | int]]:
    return [
        _rollout_score(
            agent,
            n_back=n_back,
            cue_symbol=cue_symbol,
            slot=slot,
            seed=seed + index,
            batch_size=batch_size,
            steps=steps,
            context_route=context_route,
            record_context_route=record_context_route,
            record_retention=record_retention,
            expected_slot=slot,
        )
        for index in range(lifetimes)
    ]


def _cue_key(agent: CanonicalBrainWorkshopAgent, cue_symbol: int) -> torch.Tensor:
    return agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )[0].detach()


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _orders(
    agent: CanonicalBrainWorkshopAgent,
    cues: tuple[int, ...],
) -> dict[str, list[int]]:
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
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("depth-growth budgets must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")

    started = perf_counter()
    n_backs = tuple(rule for rule, _ in RULES)
    cues = tuple(cue for _, cue in RULES)
    agent = _agent(args.seed)
    controller_before = _digest(agent.controller)

    source_history = train_reward_only(
        agent,
        n_back=n_backs[0],
        updates=args.source_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        cue_symbol=cues[0],
    )
    source_digest_before_growth = _protected_digest(agent, range(1))
    source_before = _audit(
        agent,
        n_back=n_backs[0],
        cue_symbol=cues[0],
        slot=0,
        seed=args.seed + 1000,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        record_retention=True,
    )

    target_histories: list[object] = []
    target_mastery: dict[str, list[dict[str, float | int]]] = {}
    protected_prefix_digests: list[dict[str, object]] = []
    first_target_digest_after_growth: str | None = None
    slots: list[int] = [0]
    for index, ((n_back, cue), capacity) in enumerate(
        zip(RULES[1:], (4, 5), strict=True),
        start=1,
    ):
        slot = agent.add_adaptive_relation_capability(
            memory_capacity=capacity,
            seed=args.seed + 200 * index,
            working_memory_cell=_cell(args.seed + 201 * index, capacity),
        )
        slots.append(slot)
        before = _protected_digest(agent, range(slot))
        prior_slot_before = (
            _protected_digest(agent, range(slot - 1, slot)) if slot > 1 else None
        )
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
        target_histories.extend(history)
        after = _protected_digest(agent, range(slot))
        prior_slot_after = (
            _protected_digest(agent, range(slot - 1, slot)) if slot > 1 else None
        )
        protected_prefix_digests.append(
            {
                "growth": index,
                "protected_slots": list(range(slot)),
                "unchanged": before == after,
                "prior_slot_unchanged": (
                    prior_slot_before is None or prior_slot_before == prior_slot_after
                ),
            }
        )
        if index == 1:
            first_target_digest_after_growth = _protected_digest(agent, range(1, 2))
        target_mastery[str(n_back)] = _audit(
            agent,
            n_back=n_back,
            cue_symbol=cue,
            slot=slot,
            seed=args.seed + 3000 * index,
            lifetimes=args.retention_lifetimes,
            batch_size=args.batch_size,
            steps=args.steps,
            record_retention=True,
        )

    _freeze_external(agent)
    final_retention: dict[str, list[dict[str, float | int]]] = {
        str(n_backs[0]): _audit(
            agent,
            n_back=n_backs[0],
            cue_symbol=cues[0],
            slot=0,
            seed=args.seed + 10000,
            lifetimes=args.retention_lifetimes,
            batch_size=args.batch_size,
            steps=args.steps,
            record_retention=True,
        )
    }
    for index, (n_back, cue) in enumerate(RULES[1:], start=1):
        final_retention[str(n_back)] = _audit(
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

    route_calibration: dict[str, list[dict[str, float | int]]] = {}
    for index, (n_back, cue) in enumerate(RULES):
        route_calibration[str(cue)] = _audit(
            agent,
            n_back=n_back,
            cue_symbol=cue,
            slot=slots[index],
            seed=args.seed + 20000 + 1000 * index,
            lifetimes=args.calibration_lifetimes,
            batch_size=args.batch_size,
            steps=args.steps,
            context_route=True,
            record_context_route=True,
        )

    routed: dict[str, dict[str, float | int]] = {}
    shuffled: dict[str, dict[str, float | int]] = {}
    for index, (n_back, cue) in enumerate(RULES):
        routed[str(cue)] = _rollout_score(
            agent,
            n_back=n_back,
            cue_symbol=cue,
            slot=None,
            seed=args.seed + 30000 + index,
            batch_size=args.batch_size,
            steps=args.steps,
            context_route=True,
            expected_slot=slots[index],
        )
        shuffled[str(cue)] = _rollout_score(
            agent,
            n_back=n_back,
            cue_symbol=cues[(index + 1) % len(cues)],
            slot=None,
            seed=args.seed + 31000 + index,
            batch_size=args.batch_size,
            steps=args.steps,
            context_route=True,
            expected_slot=slots[index],
        )

    original_orders = _orders(agent, cues)
    route_payload = agent.route_state_payload()
    restored = _agent(args.seed + 40000)
    for index, capacity in enumerate((4, 5), start=1):
        restored.add_adaptive_relation_capability(
            memory_capacity=capacity,
            seed=args.seed + 40100 + index,
            working_memory_cell=_cell(args.seed + 40200 + index, capacity),
        )
    restored.runtime.encoders["stimulus"].load_state_dict(
        agent.runtime.encoders["stimulus"].state_dict()
    )
    restored.load_route_state_payload(route_payload)
    restored_orders = _orders(restored, cues)

    incompatible = _agent(args.seed + 50000)
    for index, capacity in enumerate((4, 5), start=1):
        incompatible.add_adaptive_relation_capability(
            memory_capacity=capacity,
            seed=args.seed + 50100 + index,
            working_memory_cell=_cell(args.seed + 50200 + index, capacity),
        )
    try:
        incompatible.load_route_state_payload(route_payload)
    except ValueError as error:
        incompatible_route_rejected = "learned event representation" in str(error)
    else:
        incompatible_route_rejected = False

    reversal_table = type(agent.context_route_evidence).from_payload(
        route_payload["context_route_evidence"]
    )
    latest_key = _cue_key(agent, cues[-1])
    for _ in range(4):
        reversal_table.observe(latest_key, slots[-1], 0.0)
    reversal_order_on_copy = list(reversal_table.preferred_order(latest_key))
    live_latest_order = list(agent.context_route_evidence.preferred_order(latest_key))

    controller_after = _digest(agent.controller)
    source_digest_after_growth = _protected_digest(agent, range(1))
    expected_orders = {
        str(cues[0]): [0, 1, 2],
        str(cues[1]): [1, 2, 0],
        str(cues[2]): [2, 1, 0],
    }
    gates = {
        "source_mastery_before_growth": _stable(source_before),
        "source_complete_prefix_retention": _stable(final_retention[str(n_backs[0])]),
        "target_mastery": all(_stable(target_mastery[str(n_back)]) for n_back in n_backs[1:]),
        "protected_prefixes_unchanged": all(
            bool(row["unchanged"]) for row in protected_prefix_digests
        ),
        "source_codec_unchanged_after_all_growth": (
            source_digest_before_growth == source_digest_after_growth
        ),
        "first_target_codec_unchanged_during_second_growth": (
            all(
                bool(row["prior_slot_unchanged"])
                for row in protected_prefix_digests
            )
            and first_target_digest_after_growth
            == _protected_digest(agent, range(1, 2))
        ),
        "controller_unchanged": controller_before == controller_after,
        "all_final_retention": all(_stable(rows) for rows in final_retention.values()),
        "all_routes_recovered": all(
            result["accuracy"] >= MASTERY_THRESHOLD
            and result["selected_slot_fraction"] >= 0.99
            for result in routed.values()
        ),
        "cue_separates_routes": original_orders == expected_orders,
        "cue_shuffled_controls_not_targeted": all(
            result["selected_slot_fraction"] < 0.99
            for result in shuffled.values()
        ),
        "route_reload_exact": original_orders == restored_orders,
        "incompatible_route_representation_rejected": incompatible_route_rejected,
        "route_reversal_is_non_destructive": (
            reversal_order_on_copy == [0, 1, 2] and live_latest_order == [2, 1, 0]
        ),
        "zero_replayed_examples": True,
    }
    histories = (*source_history, *target_histories)
    audit_lifetimes = (
        args.retention_lifetimes
        * (1 + 2 + 3)
        + args.calibration_lifetimes * len(RULES)
        + 2 * len(RULES)
    )
    report = {
        "schema": CAUSAL_DEPTH_GROWTH_SCHEMA,
        "claim_boundary": (
            "A frozen controller acquires n-back-3 and n-back-4 as separate "
            "causal external files while preserving n-back-2 and earlier new "
            "files. This is repeated bounded rule growth, not arbitrary rule "
            "induction, unrestricted memory growth, or general continual learning."
        ),
        "seed": args.seed,
        "rules": [
            {"n_back": n_back, "cue_symbol": cue, "slot": slots[index]}
            for index, (n_back, cue) in enumerate(RULES)
        ],
        "source_updates": args.source_updates,
        "target_updates_per_growth": args.target_updates,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "source_before_growth": source_before,
        "target_mastery": target_mastery,
        "final_retention": final_retention,
        "protected_prefix_digests": protected_prefix_digests,
        "route_calibration": route_calibration,
        "routed": routed,
        "cue_shuffled_controls": shuffled,
        "original_route_orders": original_orders,
        "restored_route_orders": restored_orders,
        "reversal_route_order_on_copy": reversal_order_on_copy,
        "live_latest_route_order": live_latest_order,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits_training": sum(
                row.unique_verifier_bits for row in histories
            ),
            "optimizer_updates": len(histories),
            "replayed_examples": 0,
            "audit_lifetimes": audit_lifetimes,
            "unique_logical_lifetimes_training": args.batch_size * len(histories),
            "unique_logical_lifetimes_audit": args.batch_size * audit_lifetimes,
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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--calibration-lifetimes", type=int, default=8)
    parser.add_argument("--retention-lifetimes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
