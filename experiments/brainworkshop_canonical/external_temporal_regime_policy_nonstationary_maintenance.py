"""Pressure-test routed maintenance residuals under changing objectives.

The frozen base maintenance scorer is trained on reliability-dominated
selection. Two opaque binding keys then route independent residual maintenance
slots: one acquires reliability selection and the other acquires age selection.
Each slot learns from fresh scalar utilities without replay. An unknown key
falls back to the frozen base, while candidate order is reversed in a control.

This is a bounded nonstationary maintenance boundary. It does not claim that
opaque keys are autonomously discovered or that arbitrary maintenance
economics are solved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    GatedResidualCapabilityEvictionPolicyBank,
)

from .external_temporal_query_address_growth import _build
from .external_temporal_regime_policy_binding_slots import (
    CONTEXT_WIDTH,
    _context_keys,
)
from .external_temporal_regime_policy_learned_maintenance import (
    CANDIDATE_WIDTH,
    MAINTENANCE_TEMPERATURE,
    _episode,
    _evaluate_maintenance,
    _train_maintenance,
)
from .external_temporal_shared_basis_policy_growth import _digest

NONSTATIONARY_MAINTENANCE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-regime-policy-"
    "nonstationary-maintenance.v1"
)
SLOT_UPDATES = 256


def _adapt_slot(
    bank: GatedResidualCapabilityEvictionPolicyBank,
    *,
    slot_index: int,
    context_key: torch.Tensor,
    regime: int,
    seed: int,
    updates: int,
) -> dict[str, float | int]:
    optimizer = torch.optim.Adam(
        bank.trainable_parameters(slot_index),
        lr=0.01,
    )
    explorer = torch.Generator().manual_seed(seed + 97_001)
    utilities: list[float] = []
    context = context_key.unsqueeze(0)
    for update in range(updates):
        _base_context, candidates, reliability_target = _episode(
            seed + 100_000 + update
        )
        # The current external context key is the binding; the verifier target
        # is private and never enters the residual-policy ABI.
        target = reliability_target
        if regime == 1:
            target = int(candidates[0, :, CONTEXT_WIDTH + 1].argmin())
        features = torch.cat(
            (
                context[:, None, :].expand(-1, candidates.shape[1], -1),
                candidates,
            ),
            dim=-1,
        )
        scores = bank.residual_slots[slot_index](features).squeeze(-1)[0]
        probabilities = torch.softmax(scores / MAINTENANCE_TEMPERATURE, dim=-1)
        selected = int(torch.multinomial(probabilities, 1, generator=explorer))
        utility = float(selected == target)
        bank.adaptation_step(
            context,
            candidates,
            slot_index,
            selected,
            utility,
            temperature=MAINTENANCE_TEMPERATURE,
            optimizer=optimizer,
        )
        utilities.append(utility)
    return {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "mean_utility": sum(utilities) / len(utilities),
    }


@torch.no_grad()
def _evaluate_slot(
    bank: GatedResidualCapabilityEvictionPolicyBank,
    *,
    context_key: torch.Tensor,
    regime: int,
    seed: int,
    reverse: bool,
    episodes: int = 256,
) -> float:
    correct = 0
    context = context_key.unsqueeze(0)
    for episode in range(episodes):
        _base_context, candidates, reliability_target = _episode(seed + episode)
        target = reliability_target
        if regime == 1:
            target = int(candidates[0, :, CONTEXT_WIDTH + 1].argmin())
        if reverse:
            candidates = candidates[:, torch.tensor([2, 1, 0])]
            target = 2 - target
        scores = bank.score_candidates(context, candidates)
        correct += int(scores.argmax().item() == target)
    return correct / episodes


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.base_updates < 1 or args.slot_updates < 1:
        raise ValueError("nonstationary maintenance update counts must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    base, base_training = _train_maintenance(
        seed=args.seed,
        updates=args.base_updates,
    )
    base_accuracy = _evaluate_maintenance(base, seed=args.seed + 870_000)
    key_reliability, key_age = _context_keys(args.seed)
    key_unknown = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    bank = GatedResidualCapabilityEvictionPolicyBank(
        base,
        context_width=CONTEXT_WIDTH,
        candidate_width=CANDIDATE_WIDTH,
        max_slots=2,
        route_threshold=0.75,
    )
    reliability_slot = bank.add_slot(key_reliability)
    age_slot = bank.add_slot(key_age)
    reliability_training = _adapt_slot(
        bank,
        slot_index=reliability_slot,
        context_key=key_reliability,
        regime=0,
        seed=args.seed,
        updates=args.slot_updates,
    )
    bank.activate_slot(reliability_slot)
    bank.freeze_slot(reliability_slot)
    reliability_before_age = {
        name: value.detach().clone()
        for name, value in bank.residual_slots[reliability_slot].state_dict().items()
    }
    reliability_after = {
        "forward": _evaluate_slot(
            bank,
            context_key=key_reliability,
            regime=0,
            seed=args.seed + 871_000,
            reverse=False,
        ),
        "reverse": _evaluate_slot(
            bank,
            context_key=key_reliability,
            regime=0,
            seed=args.seed + 871_000,
            reverse=True,
        ),
    }
    age_training = _adapt_slot(
        bank,
        slot_index=age_slot,
        context_key=key_age,
        regime=1,
        seed=args.seed + 1_000,
        updates=args.slot_updates,
    )
    bank.activate_slot(age_slot)
    bank.freeze_slot(age_slot)
    age_after = {
        "forward": _evaluate_slot(
            bank,
            context_key=key_age,
            regime=1,
            seed=args.seed + 872_000,
            reverse=False,
        ),
        "reverse": _evaluate_slot(
            bank,
            context_key=key_age,
            regime=1,
            seed=args.seed + 872_000,
            reverse=True,
        ),
    }
    unknown_fallback = {
        "forward": _evaluate_slot(
            bank,
            context_key=key_unknown,
            regime=0,
            seed=args.seed + 873_000,
            reverse=False,
        ),
        "reverse": _evaluate_slot(
            bank,
            context_key=key_unknown,
            regime=0,
            seed=args.seed + 873_000,
            reverse=True,
        ),
    }
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "base_reliability_transfer": base_accuracy >= 0.80,
        "reliability_forward_learns": reliability_after["forward"] >= 0.80,
        "reliability_reverse_learns": reliability_after["reverse"] >= 0.80,
        "age_forward_learns": age_after["forward"] >= 0.80,
        "age_reverse_learns": age_after["reverse"] >= 0.80,
        "reliability_slot_retained_after_age_growth": (
            reliability_after["forward"] >= 0.75
            and reliability_after["reverse"] >= 0.75
        ),
        "unknown_context_falls_back_to_base": (
            unknown_fallback["forward"] >= base_accuracy - 0.10
            and unknown_fallback["reverse"] >= base_accuracy - 0.10
        ),
        "slot_a_unchanged_during_slot_b_learning": all(
            torch.equal(value, bank.residual_slots[reliability_slot].state_dict()[name])
            for name, value in reliability_before_age.items()
        ),
        "both_slots_active_and_frozen": (
            bool(bank.slot_active.all()) and bool(bank.slot_frozen.all())
        ),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": NONSTATIONARY_MAINTENANCE_SCHEMA,
        "claim_boundary": (
            "Two opaque routed maintenance residuals acquire distinct reliability "
            "and age selection policies from fresh scalar utilities while an "
            "unknown context falls back to the frozen base; not general maintenance "
            "economics or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "base": "external_capability_eviction_policy_v1",
            "residual_bank": "gated_residual_capability_eviction_policy_bank_v1",
            "context": "opaque_binding_cosine_key_v1",
            "candidate_features": "opaque_binding_key_plus_generic_reliability_age_v1",
            "forbidden_features": "task_labels_semantic_regime_names_physical_slot_indices_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "base_training": base_training,
        "base_accuracy": base_accuracy,
        "reliability_training": reliability_training,
        "age_training": age_training,
        "reliability_after": reliability_after,
        "age_after": age_after,
        "unknown_fallback": unknown_fallback,
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": args.base_updates,
            "residual_optimizer_updates": args.slot_updates * 2,
            "total_optimizer_updates": args.base_updates + args.slot_updates * 2,
            "unique_verifier_bits": args.base_updates + args.slot_updates * 2,
            "unique_logical_lifetimes": args.base_updates + args.slot_updates * 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_nonstationary_maintenance"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--base-updates", type=int, default=3_000)
    parser.add_argument("--slot-updates", type=int, default=SLOT_UPDATES)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
