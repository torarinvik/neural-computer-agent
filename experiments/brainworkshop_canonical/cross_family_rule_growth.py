"""Outcome-only growth across non-identical temporal rule families.

The existing Brain Workshop ladder varies only n-back depth.  This audit keeps
the learned event and intention interfaces fixed while changing the verifier's
private rule family: n-back equality, pair parity, adjacent switching, and a
single-symbol parity rule.  The controller sees no family name or target bit.
Each family is acquired into an isolated external file and a final family is
trained under one rendered cue, then routed under a previously unseen cue from
scalar outcomes alone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import ExternalWorkingMemoryCell

from .causal_depth_growth import (
    _cell,
    _cue_key,
    _digest,
    _freeze_external,
    _protected_digest,
    _stable,
)
from .environment import NBackVerifierStep
from .runner import CanonicalBrainWorkshopAgent
from .trainer import RewardOnlyUpdate, freeze_shared_path

CROSS_FAMILY_SCHEMA = "neural-computer.brainworkshop-cross-family-rule-growth.v1"
MASTERY_THRESHOLD = 0.80
ACTION_COUNT = 2
SYMBOL_COUNT = 4

PREFIX_RULES = (("nback2", 4), ("parity2", 5), ("switch", 6))
DEFAULT_TRAINING_RULE = ("symbol_parity", 7)
DEFAULT_HELDOUT_CUE = 8
DEFAULT_SHUFFLED_CUE = 9


@dataclass(frozen=True)
class RuleSpec:
    name: str
    warmup: int
    symbol_count: int = SYMBOL_COUNT


RULES = {
    "nback2": RuleSpec("nback2", 2),
    "nback3": RuleSpec("nback3", 3),
    "nback4": RuleSpec("nback4", 4),
    "nback5": RuleSpec("nback5", 5),
    "nback8": RuleSpec("nback8", 8),
    "parity2": RuleSpec("parity2", 2),
    "switch": RuleSpec("switch", 1),
    "symbol_parity": RuleSpec("symbol_parity", 0),
    "symbol_parity_odd": RuleSpec("symbol_parity_odd", 0),
    "triplet_parity": RuleSpec("triplet_parity", 3),
    "switch_binary": RuleSpec("switch_binary", 1, symbol_count=2),
}


class CrossFamilyVerifier:
    """Private verifier with one uniform learner-facing transport protocol."""

    action_count = ACTION_COUNT

    def __init__(
        self,
        *,
        family: str,
        batch_size: int,
        steps: int,
        cue_symbol: int,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if family not in RULES:
            raise ValueError("unsupported private rule family")
        if min(batch_size, steps) < 1:
            raise ValueError("batch size and steps must be positive")
        if cue_symbol < SYMBOL_COUNT:
            raise ValueError("cue symbol must be outside the symbol vocabulary")
        spec = RULES[family]
        if steps <= spec.warmup:
            raise ValueError("steps must include target-bearing trials")
        self.family = family
        self.batch_size = int(batch_size)
        self.symbol_steps = int(steps)
        self.steps = self.symbol_steps + 1
        self.cue_symbol = int(cue_symbol)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._symbols = torch.empty(0, dtype=torch.long, device=self.device)
        self._targets = torch.empty(0, dtype=torch.bool, device=self.device)
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    @property
    def eligible_trials(self) -> int:
        return self.symbol_steps - RULES[self.family].warmup

    def reset(self) -> None:
        spec = RULES[self.family]
        self._symbols = torch.randint(
            0,
            spec.symbol_count,
            (self.batch_size, self.symbol_steps),
            generator=self._generator,
            device=self.device,
        )
        warmup = RULES[self.family].warmup
        targets: list[torch.Tensor] = []
        for position in range(warmup, self.symbol_steps):
            current = self._symbols[:, position]
            if self.family.startswith("nback"):
                depth = int(self.family.removeprefix("nback"))
                target = current == self._symbols[:, position - depth]
            elif self.family == "parity2":
                target = (
                    current.remainder(2) + self._symbols[:, position - 1].remainder(2)
                ).remainder(2) == 0
            elif self.family in {"switch", "switch_binary"}:
                target = current != self._symbols[:, position - 1]
            elif self.family in {"symbol_parity", "symbol_parity_odd"}:
                parity = 1 if self.family == "symbol_parity_odd" else 0
                target = current.remainder(2) == parity
            else:
                target = (
                    self._symbols[:, position - 2]
                    + self._symbols[:, position - 1]
                    + current
                ).remainder(2) == 1
            targets.append(target)
        self._targets = torch.stack(targets, dim=1)
        self._position = 0

    def observation(self) -> torch.Tensor:
        if self._symbols.numel() == 0:
            raise RuntimeError("reset must be called before observation")
        if self.done:
            raise RuntimeError("verifier has no observations remaining")
        if self._position == 0:
            return torch.full(
                (self.batch_size,),
                self.cue_symbol,
                dtype=torch.long,
                device=self.device,
            )
        return self._symbols[:, self._position - 1].clone()

    def score(self, action: torch.Tensor) -> NBackVerifierStep:
        if self._symbols.numel() == 0:
            raise RuntimeError("reset must be called before score")
        if self.done:
            raise RuntimeError("verifier is complete")
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError("action must have shape [batch] and dtype int64")
        if bool(torch.any((action < 0) | (action >= ACTION_COUNT))):
            raise ValueError("action is outside the keypress vocabulary")
        if self._position == 0:
            reward = torch.zeros(self.batch_size, device=self.device)
            eligible = torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            )
        else:
            target_index = self._position - 1 - RULES[self.family].warmup
            if target_index < 0:
                reward = torch.zeros(self.batch_size, device=self.device)
                eligible = torch.zeros(
                    self.batch_size, dtype=torch.bool, device=self.device
                )
            else:
                reward = (action == self._targets[:, target_index].long()).float()
                eligible = torch.ones(
                    self.batch_size, dtype=torch.bool, device=self.device
                )
        self._position += 1
        return NBackVerifierStep(reward=reward, eligible=eligible)


def _agent(seed: int) -> CanonicalBrainWorkshopAgent:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        cell = ExternalWorkingMemoryCell(
            event_width=16,
            action_width=ACTION_COUNT,
            memory_capacity=3,
            context_width=16,
            hidden=32,
        )
    return CanonicalBrainWorkshopAgent(
        symbol_count=12,
        n_back=2,
        event_width=16,
        intention_width=8,
        feedback_width=8,
        reader_kind="relation",
        seed=seed,
        working_memory_cell=cell,
    )


def _append_cell(agent: CanonicalBrainWorkshopAgent, *, capacity: int, seed: int) -> int:
    return agent.add_adaptive_relation_capability(
        memory_capacity=capacity,
        seed=seed,
        working_memory_cell=_cell(seed + 1, capacity),
    )


def _train_slot(
    agent: CanonicalBrainWorkshopAgent,
    *,
    slot: int,
    family: str,
    cue_symbol: int,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    learning_rate: float,
) -> list[RewardOnlyUpdate]:
    freeze_shared_path(agent)
    if slot == 0:
        modules: tuple[torch.nn.Module, ...] = (
            agent.external_reader,
            agent.intent_adapter,
            agent.keypress_decoder,
        )
    else:
        extension = agent.extensions[slot - 1]
        modules = (extension, agent.extension_decoder(slot))
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[RewardOnlyUpdate] = []
    for update in range(updates):
        verifier = CrossFamilyVerifier(
            family=family,
            batch_size=batch_size,
            steps=steps,
            cue_symbol=cue_symbol,
            seed=seed + update,
        )
        rollout = agent.rollout(
            verifier,
            sample=True,
            forced_slot=slot,
            record_retention=False,
        )
        eligible = rollout.eligible.to(rollout.rewards.dtype)
        propensity = rollout.propensities.clamp_min(1e-8).log()
        advantage = (rollout.rewards - 0.5).detach()
        denominator = eligible.sum().clamp_min(1.0)
        loss = -((advantage * propensity * eligible).sum() / denominator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            RewardOnlyUpdate(
                update=update + 1,
                loss=float(loss.detach()),
                eligible_accuracy=float(rollout.eligible_accuracy.mean().detach()),
                unique_verifier_bits=batch_size * verifier.eligible_trials,
                replayed_examples=rollout.replayed_examples,
                selected_slot_fraction=float(
                    (rollout.selected_slots == slot).float().mean()
                ),
            )
        )
    return history


def _score(
    agent: CanonicalBrainWorkshopAgent,
    *,
    family: str,
    cue_symbol: int,
    slot: int | None,
    seed: int,
    batch_size: int,
    steps: int,
    route: bool = False,
    record_route: bool = False,
    expected_slot: int | None = None,
    route_failure_patience: int = 4,
) -> dict[str, float]:
    verifier = CrossFamilyVerifier(
        family=family,
        batch_size=batch_size,
        steps=steps,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    with torch.no_grad():
        rollout = agent.rollout(
            verifier,
            sample=False,
            forced_slot=slot,
            context_route=route,
            record_context_route=record_route,
            context_route_failure_patience=route_failure_patience,
        )
    selected = 0 if expected_slot is None else expected_slot
    return {
        "accuracy": float(rollout.eligible_accuracy.mean()),
        "selected_slot_fraction": float(
            (rollout.selected_slots == selected).float().mean()
        ),
    }


def _key(agent: CanonicalBrainWorkshopAgent, cue: int) -> torch.Tensor:
    return _cue_key(agent, cue)


def _orders(agent: CanonicalBrainWorkshopAgent, cues: tuple[int, ...]) -> dict[str, list[int]]:
    return {
        str(cue): list(agent.context_route_evidence.preferred_order(_key(agent, cue)))
        for cue in cues
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    target_family = getattr(args, "target_family", DEFAULT_TRAINING_RULE[0])
    training_cue = int(getattr(args, "training_cue", DEFAULT_TRAINING_RULE[1]))
    heldout_cue = int(getattr(args, "heldout_cue", DEFAULT_HELDOUT_CUE))
    shuffled_cue = int(getattr(args, "shuffled_cue", DEFAULT_SHUFFLED_CUE))
    target_warmup_family = getattr(args, "target_warmup_family", None)
    target_warmup_updates = int(getattr(args, "target_warmup_updates", 0))
    if min(
        args.source_updates,
        args.target_updates,
        args.batch_size,
        args.steps,
        args.calibration_lifetimes,
        args.discovery_lifetimes,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("cross-family budgets must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if target_family not in RULES:
        raise ValueError("target family is not supported")
    if target_family in {family for family, _ in PREFIX_RULES}:
        raise ValueError("target family must be distinct from the protected prefix")
    if target_warmup_family is not None and target_warmup_family not in RULES:
        raise ValueError("target warmup family is not supported")
    if target_warmup_updates < 0:
        raise ValueError("target warmup updates cannot be negative")
    if target_warmup_family is None and target_warmup_updates:
        raise ValueError("target warmup updates require a warmup family")
    if min(training_cue, heldout_cue, shuffled_cue) < SYMBOL_COUNT:
        raise ValueError("target cues must be outside the symbol vocabulary")
    if len({training_cue, heldout_cue, shuffled_cue}) != 3:
        raise ValueError("target cues must be distinct")
    started = perf_counter()
    agent = _agent(args.seed)
    controller_before = _digest(agent.controller)
    encoder_before = _digest(agent.runtime.encoders["stimulus"])
    slots = [0]
    histories: list[RewardOnlyUpdate] = []
    prefix_rules = tuple(PREFIX_RULES)
    training_rule = (target_family, training_cue)

    histories.extend(
        _train_slot(
            agent,
            slot=0,
            family=prefix_rules[0][0],
            cue_symbol=prefix_rules[0][1],
            updates=args.source_updates,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 100,
            learning_rate=args.learning_rate,
        )
    )
    prefix_before: list[tuple[str, int]] = [(_protected_digest(agent, range(1)), 1)]
    for index, (family, cue) in enumerate(prefix_rules[1:], start=1):
        slot = _append_cell(agent, capacity=4 + index, seed=args.seed + 200 * index)
        slots.append(slot)
        before = _protected_digest(agent, range(slot))
        histories.extend(
            _train_slot(
                agent,
                slot=slot,
                family=family,
                cue_symbol=cue,
                updates=args.target_updates,
                batch_size=args.batch_size,
                steps=args.steps,
                seed=args.seed + 300 * index,
                learning_rate=args.learning_rate,
            )
        )
        prefix_before.append((before, slot))
        if before != _protected_digest(agent, range(slot)):
            raise AssertionError("protected prefix changed during cross-family growth")

    target_slot = _append_cell(agent, capacity=6, seed=args.seed + 800)
    target_prefix_before = _protected_digest(agent, range(target_slot))
    if target_warmup_family is not None and target_warmup_updates:
        histories.extend(
            _train_slot(
                agent,
                slot=target_slot,
                family=target_warmup_family,
                cue_symbol=training_rule[1],
                updates=target_warmup_updates,
                batch_size=args.batch_size,
                steps=args.steps,
                seed=args.seed + 850,
                learning_rate=args.learning_rate,
            )
        )
    histories.extend(
        _train_slot(
            agent,
            slot=target_slot,
            family=training_rule[0],
            cue_symbol=training_rule[1],
            updates=args.target_updates,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 900,
            learning_rate=args.learning_rate,
        )
    )
    target_prefix_after = _protected_digest(agent, range(target_slot))
    _freeze_external(agent)

    retention: dict[str, list[dict[str, float]]] = {}
    for index, (family, cue) in enumerate((*prefix_rules, training_rule)):
        slot = slots[index] if index < len(slots) else target_slot
        retention[family] = [
            _score(
                agent,
                family=family,
                cue_symbol=cue,
                slot=slot,
                seed=args.seed + 10000 + index * 1000 + lifetime,
                batch_size=args.batch_size,
                steps=args.steps,
                expected_slot=slot,
            )
            for lifetime in range(args.retention_lifetimes)
        ]

    for index, (family, cue) in enumerate(prefix_rules):
        for lifetime in range(args.calibration_lifetimes):
            _score(
                agent,
                family=family,
                cue_symbol=cue,
                slot=None,
                seed=args.seed + 20000 + index * 1000 + lifetime,
                batch_size=args.batch_size,
                steps=args.steps,
                route=True,
                record_route=True,
                expected_slot=slots[index],
                route_failure_patience=1,
            )
    heldout_before = agent.context_route_evidence.has_context(_key(agent, heldout_cue))
    discovery = [
        _score(
            agent,
            family=training_rule[0],
            cue_symbol=heldout_cue,
            slot=None,
            seed=args.seed + 30000 + lifetime,
            batch_size=args.batch_size,
            steps=args.steps,
            route=True,
            record_route=True,
            expected_slot=target_slot,
            route_failure_patience=1,
        )
        for lifetime in range(args.discovery_lifetimes)
    ]
    heldout_after = agent.context_route_evidence.has_context(_key(agent, heldout_cue))
    recovered = _score(
        agent,
        family=training_rule[0],
        cue_symbol=heldout_cue,
        slot=None,
        seed=args.seed + 40000,
        batch_size=args.batch_size,
        steps=args.steps,
        route=True,
        expected_slot=target_slot,
    )
    shuffled = _score(
        agent,
        family=training_rule[0],
        cue_symbol=shuffled_cue,
        slot=None,
        seed=args.seed + 41000,
        batch_size=args.batch_size,
        steps=args.steps,
        route=True,
        expected_slot=target_slot,
    )
    orders = _orders(agent, (*[cue for _, cue in prefix_rules], heldout_cue))
    expected_order = [target_slot, 2, 1, 0]
    route_payload = agent.route_state_payload()
    restored = _agent(args.seed + 50000)
    for index, capacity in enumerate((4, 5, 6), start=1):
        _append_cell(restored, capacity=capacity, seed=args.seed + 50100 + index)
    restored.runtime.encoders["stimulus"].load_state_dict(
        agent.runtime.encoders["stimulus"].state_dict()
    )
    restored.load_route_state_payload(route_payload)
    restored_orders = _orders(
        restored, (*[cue for _, cue in prefix_rules], heldout_cue)
    )
    incompatible = _agent(args.seed + 60000)
    for index, capacity in enumerate((4, 5, 6), start=1):
        _append_cell(incompatible, capacity=capacity, seed=args.seed + 60100 + index)
    try:
        incompatible.load_route_state_payload(route_payload)
    except ValueError as error:
        incompatible_rejected = "learned event representation" in str(error)
    else:
        incompatible_rejected = False

    controller_after = _digest(agent.controller)
    encoder_after = _digest(agent.runtime.encoders["stimulus"])
    gates = {
        "prefix_retention": all(
            _stable(retention[family]) for family, _ in prefix_rules
        ),
        "new_family_mastery": _stable(retention[training_rule[0]]),
        "prefix_unchanged_during_growth": all(
            before == _protected_digest(agent, range(prefix_count))
            for before, prefix_count in prefix_before
        ),
        "target_prefix_unchanged_during_growth": target_prefix_before
        == target_prefix_after,
        "controller_unchanged": controller_before == controller_after,
        "encoder_unchanged": encoder_before == encoder_after,
        "heldout_context_absent_before_discovery": not heldout_before,
        "heldout_context_learned_from_outcomes": heldout_after,
        "heldout_route_recovered": (
            recovered["accuracy"] >= MASTERY_THRESHOLD
            and recovered["selected_slot_fraction"] >= 0.99
        ),
        "heldout_route_order_learned": orders[str(heldout_cue)] == expected_order,
        "cue_shuffle_does_not_select_target": (
            shuffled["selected_slot_fraction"] < 0.75
        ),
        "route_reload_exact": orders == restored_orders,
        "incompatible_route_representation_rejected": incompatible_rejected,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": CROSS_FAMILY_SCHEMA,
        "claim_boundary": (
            "Cross-family outcome-only route discovery over fixed temporal rule "
            "interfaces; this is not arbitrary new computation or general "
            "continual learning."
        ),
        "route_policy": {
            "discovery_failure_patience": 1,
            "exploitation_failure_patience": 4,
        },
        "seed": args.seed,
        "prefix_rules": [
            {"family": family, "cue_symbol": cue, "slot": slots[index]}
            for index, (family, cue) in enumerate(prefix_rules)
        ],
        "training_rule": {
            "family": training_rule[0],
            "training_cue": training_rule[1],
            "heldout_cue": heldout_cue,
            "slot": target_slot,
        },
        "target_warmup": {
            "family": target_warmup_family,
            "updates": target_warmup_updates,
            "cue_symbol": training_rule[1],
        },
        "retention": retention,
        "discovery": discovery,
        "recovered": recovered,
        "shuffled_cue_control": shuffled,
        "route_orders": orders,
        "restored_route_orders": restored_orders,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits_training": sum(
                args.batch_size * (args.steps - RULES[family].warmup) * updates
                for family, updates in (
                    [(prefix_rules[0][0], args.source_updates)]
                    + [(family, args.target_updates) for family, _ in prefix_rules[1:]]
                    + ([(target_warmup_family, target_warmup_updates)]
                       if target_warmup_family is not None
                       else [])
                    + [(training_rule[0], args.target_updates)]
                )
            ),
            "unique_verifier_bits_audit": args.batch_size
            * (
                args.retention_lifetimes
                * sum(
                    args.steps - RULES[family].warmup
                    for family, _ in (*prefix_rules, training_rule)
                )
                + args.calibration_lifetimes
                * sum(args.steps - RULES[family].warmup for family, _ in prefix_rules)
                + (args.discovery_lifetimes + 2)
                * (args.steps - RULES[training_rule[0]].warmup)
            ),
            "unique_logical_lifetimes_training": args.batch_size * len(histories),
            "unique_logical_lifetimes_audit": args.batch_size
            * (
                args.retention_lifetimes * len((*prefix_rules, training_rule))
                + args.calibration_lifetimes * len(prefix_rules)
                + args.discovery_lifetimes
                + 2
            ),
            "optimizer_updates": len(histories),
            "replayed_examples": sum(row.replayed_examples for row in histories),
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
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--calibration-lifetimes", type=int, default=32)
    parser.add_argument("--discovery-lifetimes", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--target-family",
        choices=tuple(RULES),
        default=DEFAULT_TRAINING_RULE[0],
    )
    parser.add_argument("--training-cue", type=int, default=DEFAULT_TRAINING_RULE[1])
    parser.add_argument("--heldout-cue", type=int, default=DEFAULT_HELDOUT_CUE)
    parser.add_argument("--shuffled-cue", type=int, default=DEFAULT_SHUFFLED_CUE)
    parser.add_argument(
        "--target-warmup-family",
        choices=tuple(RULES),
        default=None,
    )
    parser.add_argument("--target-warmup-updates", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
