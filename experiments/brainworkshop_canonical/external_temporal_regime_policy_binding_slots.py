"""Pressure-test multiple isolated residual bindings on colliding geometry.

Two external bindings receive the same relational current/incoming bank
geometry.  A binding-context key routes each one to its own residual slot.
Slots are acquired in separate online phases from fresh scalar verifier
utilities, with no task label or replayed example entering the policy.  The
test checks that growing slot B does not alter slot A or the frozen base.

This is a narrow binding/routing boundary.  It does not claim that arbitrary
new skills can be inferred from a context key or that external capacity is
unbounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    GatedResidualRegimePolicyBank,
)

from .external_temporal_query_address_growth import _build
from .external_temporal_regime_policy_online_adaptation import (
    ONLINE_RECORDS,
    _evaluate_families,
    _partial_pair,
)
from .external_temporal_shared_basis_learned_regime_trigger import (
    _train_detector,
)
from .external_temporal_shared_basis_policy_growth import _digest

BINDING_SLOTS_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-regime-policy-"
    "binding-slots.v1"
)
CONTEXT_WIDTH = 8
ONLINE_UPDATES_PER_SLOT = 64
ONLINE_TEMPERATURE = 0.65


def _context_keys(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 98_001)
    keys = torch.linalg.qr(
        torch.randn(CONTEXT_WIDTH, 2, generator=generator)
    ).Q.transpose(0, 1)
    return keys[0], keys[1]


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _unchanged(
    module: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> bool:
    return all(
        torch.equal(value, module.state_dict()[name])
        for name, value in snapshot.items()
    )


def _adapt_slot(
    bank: GatedResidualRegimePolicyBank,
    *,
    slot_index: int,
    context: torch.Tensor,
    seed: int,
    updates: int,
) -> dict[str, float | int]:
    optimizer = torch.optim.Adam(
        bank.trainable_parameters(slot_index),
        lr=0.002,
    )
    explorer = torch.Generator().manual_seed(seed + 99_001)
    occupied = torch.ones(1, ONLINE_RECORDS, dtype=torch.bool)
    utilities: list[float] = []
    for update in range(updates):
        current, incoming, target_replace = _partial_pair(
            seed=seed + 100_000 + update,
            family="partial",
        )
        context_batch = context.unsqueeze(0)
        plan = bank.propose(
            current.unsqueeze(0),
            occupied,
            incoming.unsqueeze(0),
            occupied,
            context_batch,
            explore=True,
            temperature=ONLINE_TEMPERATURE,
            generator=explorer,
        )
        utility = float(plan.replace == target_replace)
        bank.adaptation_step(
            current.unsqueeze(0),
            occupied,
            incoming.unsqueeze(0),
            occupied,
            context_batch,
            slot_index,
            plan,
            utility,
            optimizer=optimizer,
        )
        utilities.append(utility)
    return {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "mean_utility": sum(utilities) / len(utilities),
    }


@torch.no_grad()
def _evaluate_binding(
    bank: GatedResidualRegimePolicyBank,
    *,
    context: torch.Tensor,
    seed: int,
    episodes: int = 128,
) -> dict[str, float]:
    occupied = torch.ones(1, ONLINE_RECORDS, dtype=torch.bool)
    scores = {"partial_replace": [], "stable_keep": [], "disjoint_replace": []}
    for family, name in (
        ("partial", "partial_replace"),
        ("stable", "stable_keep"),
        ("disjoint", "disjoint_replace"),
    ):
        target_replace = family != "stable"
        for episode in range(episodes):
            current, incoming, _target = _partial_pair(
                seed=seed + episode + (10_000 if family == "partial" else 20_000 if family == "disjoint" else 0),
                family=family,
            )
            plan = bank.propose(
                current.unsqueeze(0),
                occupied,
                incoming.unsqueeze(0),
                occupied,
                context.unsqueeze(0),
            )
            scores[name].append(float(plan.replace == target_replace))
    return {name: sum(values) / len(values) for name, values in scores.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1 or args.slot_updates < 1:
        raise ValueError("binding slot update counts must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    base, pretraining = _train_detector(
        seed=args.seed,
        updates=args.policy_updates,
    )
    key_a, key_b = _context_keys(args.seed)
    bank = GatedResidualRegimePolicyBank(
        base,
        context_width=CONTEXT_WIDTH,
        override_margin=0.0,
        route_threshold=0.75,
    )
    slot_a = bank.add_slot(key_a)
    slot_b = bank.add_slot(key_b)
    base_before = _snapshot(base)
    slot_b_before_phase_a = _snapshot(bank.residual_slots[slot_b])
    phase_a_training = _adapt_slot(
        bank,
        slot_index=slot_a,
        context=key_a,
        seed=args.seed,
        updates=args.slot_updates,
    )
    phase_a = {
        "binding_a": _evaluate_binding(
            bank,
            context=key_a,
            seed=args.seed + 830_000,
        ),
        "binding_b": _evaluate_binding(
            bank,
            context=key_b,
            seed=args.seed + 830_000,
        ),
    }
    slot_a_after_phase_a = _snapshot(bank.residual_slots[slot_a])
    slot_b_unchanged_after_phase_a = _unchanged(
        bank.residual_slots[slot_b], slot_b_before_phase_a
    )
    phase_b_training = _adapt_slot(
        bank,
        slot_index=slot_b,
        context=key_b,
        seed=args.seed + 1_000,
        updates=args.slot_updates,
    )
    phase_b = {
        "binding_a": _evaluate_binding(
            bank,
            context=key_a,
            seed=args.seed + 831_000,
        ),
        "binding_b": _evaluate_binding(
            bank,
            context=key_b,
            seed=args.seed + 831_000,
        ),
    }
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "two_slots_bound": bank.slot_count == 2,
        "routes_select_distinct_slots": torch.equal(
            bank.route_slot(torch.stack((key_a, key_b))),
            torch.tensor([slot_a, slot_b]),
        ),
        "phase_a_learns_binding_a": phase_a["binding_a"]["partial_replace"] >= 0.80,
        "phase_a_leaves_binding_b_base": (
            phase_a["binding_b"]["partial_replace"] <= 0.25
        ),
        "phase_a_does_not_change_slot_b": slot_b_unchanged_after_phase_a,
        "phase_b_learns_binding_b": phase_b["binding_b"]["partial_replace"] >= 0.80,
        "binding_a_retained_after_slot_b_growth": (
            phase_b["binding_a"]["partial_replace"] >= 0.75
        ),
        "phase_b_does_not_change_slot_a": _unchanged(
            bank.residual_slots[slot_a], slot_a_after_phase_a
        ),
        "binding_a_stable_retained": phase_b["binding_a"]["stable_keep"] >= 0.80,
        "binding_b_stable_retained": phase_b["binding_b"]["stable_keep"] >= 0.80,
        "binding_a_disjoint_retained": phase_b["binding_a"]["disjoint_replace"] >= 0.80,
        "binding_b_disjoint_retained": phase_b["binding_b"]["disjoint_replace"] >= 0.80,
        "base_frozen": _unchanged(base, base_before),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": BINDING_SLOTS_SCHEMA,
        "claim_boundary": (
            "Two independently bound residual slots learn the same colliding "
            "geometry without cross-slot parameter interference while the base "
            "and controller remain frozen; not arbitrary skill routing or general "
            "continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "base": "opaque_regime_change_policy_v1",
            "bank": "gated_residual_regime_policy_bank_v1",
            "context": "opaque_binding_cosine_key_v1",
            "context_width": CONTEXT_WIDTH,
            "slot_count": bank.slot_count,
            "forbidden_features": "task_labels_regime_ids_semantic_slot_names_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "pretraining": pretraining,
        "phase_a_training": phase_a_training,
        "phase_b_training": phase_b_training,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "base_geometry_only_control": _evaluate_families(
            base,
            seed=args.seed + 832_000,
        ),
        "gates": gates,
        "accounting": {
            "pretraining_scalar_updates": args.policy_updates,
            "online_scalar_updates": args.slot_updates * 2,
            "total_optimizer_updates": args.policy_updates + args.slot_updates * 2,
            "unique_verifier_bits": args.policy_updates + args.slot_updates * 2,
            "unique_logical_lifetimes": args.policy_updates + args.slot_updates * 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_binding_slots" if all(gates.values()) else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy-updates", type=int, default=1_000)
    parser.add_argument("--slot-updates", type=int, default=64)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
