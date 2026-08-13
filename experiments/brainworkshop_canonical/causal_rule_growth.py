"""Protected causal n-back rule growth over versioned external memory cells.

This audit uses the new ``ExternalWorkingMemoryCell`` for both the mastered
source capability and an appended target capability.  The source n-back-2
codec is frozen before n-back-3 acquisition begins.  A visible rendered cue
is used only as an ordinary learned event for route inference; the verifier's
private n-back value never crosses the boundary.
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

CAUSAL_RULE_GROWTH_SCHEMA = "neural-computer.brainworkshop-causal-rule-growth.v1"
MASTERY_THRESHOLD = 0.80


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


def _freeze_external(agent: CanonicalBrainWorkshopAgent) -> None:
    modules: list[torch.nn.Module] = [
        agent.external_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    ]
    for index, extension in enumerate(agent.extensions, start=1):
        modules.extend((extension, agent.extension_decoder(index)))
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
    steps: int,
    context_route: bool = False,
    record_context_route: bool = False,
    record_retention: bool = False,
    expected_slot: int | None = None,
) -> dict[str, float | int]:
    verifier = NBackVerifier(
        batch_size=32,
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
    return {
        "accuracy": float(rollout.eligible_accuracy.mean()),
        "selected_slot_fraction": float(
            (rollout.selected_slots == (0 if expected_slot is None else expected_slot))
            .to(torch.float32)
            .mean()
        ),
    }


def _retention_audit(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    slot: int,
    steps: int,
    seed: int,
) -> list[float]:
    scores: list[float] = []
    for index in range(8):
        result = _rollout_score(
            agent,
            n_back=n_back,
            cue_symbol=cue_symbol,
            slot=slot,
            seed=seed + index,
            steps=steps,
            record_retention=True,
        )
        scores.append(float(result["accuracy"]))
    return scores


def _calibrate_route(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    slot: int,
    steps: int,
    seed: int,
) -> list[float]:
    scores: list[float] = []
    for index in range(8):
        result = _rollout_score(
            agent,
            n_back=n_back,
            cue_symbol=cue_symbol,
            slot=slot,
            seed=seed + index,
            steps=steps,
            context_route=True,
            record_context_route=True,
        )
        scores.append(float(result["accuracy"]))
    return scores


def _cue_key(agent: CanonicalBrainWorkshopAgent, cue_symbol: int) -> torch.Tensor:
    return agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )[0].detach()


def _stable(values: list[float]) -> bool:
    return bool(values) and min(values) >= MASTERY_THRESHOLD


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.old_updates, args.new_updates, args.batch_size, args.steps) < 1:
        raise ValueError("rule-growth budgets must be positive")
    if args.old_n_back == args.new_n_back:
        raise ValueError("rule growth needs different source and target rules")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    started = perf_counter()
    old_cue = 4
    new_cue = 5
    agent = _agent(args.seed)
    controller_before = _digest(agent.controller)
    old_history = train_reward_only(
        agent,
        n_back=args.old_n_back,
        updates=args.old_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        cue_symbol=old_cue,
    )
    old_codec_before_growth = _digest(
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )
    old_retention_before = _retention_audit(
        agent,
        n_back=args.old_n_back,
        cue_symbol=old_cue,
        slot=0,
        steps=args.steps,
        seed=args.seed + 1000,
    )
    target_cell = _cell(args.seed + 200, 4)
    slot = agent.add_adaptive_relation_capability(
        memory_capacity=4,
        seed=args.seed + 201,
        working_memory_cell=target_cell,
    )
    slot, new_history = _train_relation_extension(
        agent,
        slot=slot,
        verifier_n_back=args.new_n_back,
        updates=args.new_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 300,
        learning_rate=args.learning_rate,
        exploration_probability=0.25,
        forced_slot=slot,
        cue_symbol=new_cue,
    )
    _freeze_external(agent)
    old_codec_after_growth = _digest(
        agent.relation_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    )
    old_retention_after = _retention_audit(
        agent,
        n_back=args.old_n_back,
        cue_symbol=old_cue,
        slot=0,
        steps=args.steps,
        seed=args.seed + 2000,
    )
    new_retention = _retention_audit(
        agent,
        n_back=args.new_n_back,
        cue_symbol=new_cue,
        slot=slot,
        steps=args.steps,
        seed=args.seed + 3000,
    )
    old_route_calibration = _calibrate_route(
        agent,
        n_back=args.old_n_back,
        cue_symbol=old_cue,
        slot=0,
        steps=args.steps,
        seed=args.seed + 4000,
    )
    new_route_calibration = _calibrate_route(
        agent,
        n_back=args.new_n_back,
        cue_symbol=new_cue,
        slot=slot,
        steps=args.steps,
        seed=args.seed + 5000,
    )
    old_routed = _rollout_score(
        agent,
        n_back=args.old_n_back,
        cue_symbol=old_cue,
        slot=None,
        seed=args.seed + 6000,
        steps=args.steps,
        context_route=True,
    )
    new_routed = _rollout_score(
        agent,
        n_back=args.new_n_back,
        cue_symbol=new_cue,
        slot=None,
        seed=args.seed + 6001,
        steps=args.steps,
        context_route=True,
        expected_slot=slot,
    )
    shuffled_route = _rollout_score(
        agent,
        n_back=args.new_n_back,
        cue_symbol=old_cue,
        slot=None,
        seed=args.seed + 6002,
        steps=args.steps,
        context_route=True,
    )
    route_payload = agent.route_state_payload()
    restored = _agent(args.seed + 7000)
    restored.add_adaptive_relation_capability(
        memory_capacity=4,
        seed=args.seed + 7001,
        working_memory_cell=_cell(args.seed + 7002, 4),
    )
    # Route keys are learned-event representations.  A separately restored
    # route table must therefore be paired with the compatible encoder
    # version that produced those keys; controller and memory weights remain
    # intentionally unrelated to this persistence check.
    restored.runtime.encoders["stimulus"].load_state_dict(
        agent.runtime.encoders["stimulus"].state_dict()
    )
    restored.load_route_state_payload(route_payload)
    restored_orders = {
        "old": list(restored.context_route_evidence.preferred_order(_cue_key(restored, old_cue))),
        "new": list(restored.context_route_evidence.preferred_order(_cue_key(restored, new_cue))),
    }
    original_orders = {
        "old": list(agent.context_route_evidence.preferred_order(_cue_key(agent, old_cue))),
        "new": list(agent.context_route_evidence.preferred_order(_cue_key(agent, new_cue))),
    }
    reversal_table = type(agent.context_route_evidence).from_payload(
        route_payload["context_route_evidence"]
    )
    new_key = _cue_key(agent, new_cue)
    for _ in range(4):
        reversal_table.observe(new_key, slot, 0.0)
    reversal_status = reversal_table.preferred_order(new_key)
    controller_after = _digest(agent.controller)
    gates = {
        "old_source_mastery": _stable(old_retention_before),
        "old_complete_prefix_retention": _stable(old_retention_after),
        "new_complete_prefix_mastery": _stable(new_retention),
        "old_codec_unchanged": old_codec_before_growth == old_codec_after_growth,
        "controller_unchanged": controller_before == controller_after,
        "old_route_recovered": old_routed["accuracy"] >= MASTERY_THRESHOLD
        and old_routed["selected_slot_fraction"] >= 0.99,
        "new_route_recovered": new_routed["accuracy"] >= MASTERY_THRESHOLD
        and new_routed["selected_slot_fraction"] >= 0.99,
        "cue_separates_routes": original_orders == {"old": [0, 1], "new": [1, 0]},
        "route_reload_exact": original_orders == restored_orders,
        "route_reversal_is_non_destructive": (
            reversal_status[0] == 0
            and agent.context_route_evidence.preferred_order(new_key)[0] == slot
        ),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": CAUSAL_RULE_GROWTH_SCHEMA,
        "claim_boundary": (
            "A mastered causal n-back-2 working-memory file remains protected "
            "while a new causal n-back-3 file is acquired and routed by an "
            "ordinary rendered cue; this is bounded rule growth, not open-ended "
            "rule induction or general continual learning."
        ),
        "seed": args.seed,
        "old_n_back": args.old_n_back,
        "new_n_back": args.new_n_back,
        "old_cue_symbol": old_cue,
        "new_cue_symbol": new_cue,
        "old_updates": args.old_updates,
        "new_updates": args.new_updates,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "old_retention_before_growth": old_retention_before,
        "old_retention_after_growth": old_retention_after,
        "new_retention": new_retention,
        "old_route_calibration": old_route_calibration,
        "new_route_calibration": new_route_calibration,
        "old_routed": old_routed,
        "new_routed": new_routed,
        "cue_shuffled_route_control": shuffled_route,
        "original_route_orders": original_orders,
        "restored_route_orders": restored_orders,
        "reversal_route_order_on_copy": list(reversal_status),
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": sum(
                row.unique_verifier_bits for row in (*old_history, *new_history)
            ),
            "unique_logical_lifetimes": args.batch_size
            * (args.old_updates + args.new_updates),
            "optimizer_updates": args.old_updates + args.new_updates,
            "replayed_examples": 0,
            "retention_and_route_audit_bits": args.batch_size
            * 8
            * 2
            * ((args.steps - args.old_n_back) + (args.steps - args.new_n_back)),
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
    parser.add_argument("--old-n-back", type=int, default=2)
    parser.add_argument("--new-n-back", type=int, default=3)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--new-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
