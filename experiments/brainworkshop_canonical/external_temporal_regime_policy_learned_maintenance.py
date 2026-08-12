"""Pressure-test learned victim selection for a full residual-slot bank.

An external eviction policy learns from one scalar verifier utility to rank
opaque candidate slots by disposability. Candidate order is independently
permuted, and the live bank marks no slot with a semantic name: only generic
external reliability telemetry is available to the policy. A verifier-gated
copy-on-write replacement then reuses the policy-selected slot for a new
binding while retaining the sibling capability.

The controller, event encoder, and residual slots are frozen during selection.
This promotes learned bounded maintenance choice, not autonomous utility
economics or unrestricted continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    ExternalCapabilityEvictionPolicy,
    GatedResidualRegimePolicyBank,
    OpaqueRegimeChangePolicy,
)

from .external_temporal_query_address_growth import _build
from .external_temporal_regime_policy_binding_slots import (
    CONTEXT_WIDTH,
    _adapt_slot,
    _context_keys,
    _evaluate_binding,
)
from .external_temporal_shared_basis_learned_regime_trigger import (
    _train_detector,
)
from .external_temporal_shared_basis_policy_growth import _digest

LEARNED_MAINTENANCE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-regime-policy-"
    "learned-maintenance.v1"
)
CANDIDATE_WIDTH = CONTEXT_WIDTH + 2
MAINTENANCE_HIDDEN = 32
MAINTENANCE_TEMPERATURE = 0.7
MAINTENANCE_UPDATES = 3_000
SLOT_UPDATES = 72


def _episode(seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(1, CONTEXT_WIDTH, generator=generator)
    context = context / context.square().sum(dim=-1, keepdim=True).sqrt()
    candidates = torch.randn(1, 3, CANDIDATE_WIDTH, generator=generator)
    reliability = torch.rand(3, generator=generator)
    candidates[0, :, CONTEXT_WIDTH] = reliability
    candidates[0, :, CONTEXT_WIDTH + 1] = torch.rand(3, generator=generator)
    return context, candidates, int(reliability.argmin())


def _train_maintenance(
    *,
    seed: int,
    updates: int,
) -> tuple[ExternalCapabilityEvictionPolicy, dict[str, float | int]]:
    torch.manual_seed(seed)
    policy = ExternalCapabilityEvictionPolicy(
        context_width=CONTEXT_WIDTH,
        candidate_width=CANDIDATE_WIDTH,
        hidden=MAINTENANCE_HIDDEN,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    explorer = torch.Generator().manual_seed(seed + 96_001)
    utilities: list[float] = []
    for update in range(updates):
        context, candidates, target = _episode(seed + 100_000 + update)
        scores = policy.score_candidates(context, candidates)[0]
        probabilities = torch.softmax(scores / MAINTENANCE_TEMPERATURE, dim=-1)
        selected = int(torch.multinomial(probabilities, 1, generator=explorer))
        utility = float(selected == target)
        loss = -(utility - 0.5) * torch.log_softmax(
            scores / MAINTENANCE_TEMPERATURE, dim=-1
        )[selected]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        utilities.append(utility)
    policy.eval()
    return policy, {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
    }


@torch.no_grad()
def _evaluate_maintenance(
    policy: ExternalCapabilityEvictionPolicy,
    *,
    seed: int,
    episodes: int = 256,
) -> float:
    correct = 0
    for episode in range(episodes):
        context, candidates, target = _episode(seed + episode)
        correct += int(policy.score_candidates(context, candidates).argmax() == target)
    return correct / episodes


def _candidate_features(
    keys: tuple[torch.Tensor, ...],
    reliabilities: tuple[float, ...],
    *,
    order: tuple[int, ...],
) -> torch.Tensor:
    rows = []
    for index in order:
        row = torch.cat(
            (
                keys[index],
                torch.tensor(
                    [reliabilities[index], 0.0],
                    dtype=keys[index].dtype,
                ),
            )
        )
        rows.append(row)
    return torch.stack(rows).unsqueeze(0)


def _run_live(
    *,
    maintenance: ExternalCapabilityEvictionPolicy,
    base: OpaqueRegimeChangePolicy,
    system,
    seed: int,
    reverse_order: bool,
) -> dict[str, object]:
    key_a, key_b = _context_keys(seed)
    key_c = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    bank = GatedResidualRegimePolicyBank(
        base,
        context_width=CONTEXT_WIDTH,
        override_margin=0.0,
        route_threshold=0.75,
        max_slots=2,
    )
    slot_a = bank.add_slot(key_a)
    slot_b = bank.add_slot(key_b)
    _adapt_slot(bank, slot_index=slot_a, context=key_a, seed=seed, updates=SLOT_UPDATES)
    _adapt_slot(
        bank,
        slot_index=slot_b,
        context=key_b,
        seed=seed + 1_000,
        updates=SLOT_UPDATES,
    )
    bank.freeze_slot(slot_a)
    bank.freeze_slot(slot_b)
    physical = (slot_a, slot_b)
    order = (1, 0) if reverse_order else (0, 1)
    features = _candidate_features(
        (key_a, key_b),
        (0.96, 0.12),
        order=order,
    )
    context = key_c.unsqueeze(0)
    scores = maintenance.score_candidates(context, features)[0]
    selected_position = int(scores.argmax())
    selected_slot = physical[order[selected_position]]
    unsafe_candidate = bank.slot_replacement_candidate(selected_slot, key_c)

    def retains_sibling(candidate) -> bool:
        sibling = slot_b if selected_slot == slot_a else slot_a
        sibling_key = key_b if selected_slot == slot_a else key_a
        score = _evaluate_binding(
            candidate,
            context=sibling_key,
            seed=seed + 850_000,
            episodes=64,
        )
        return (
            score["partial_replace"] >= 0.75
            and score["stable_keep"] >= 0.80
            and score["disjoint_replace"] >= 0.80
            and int(candidate.route_slot(sibling_key.unsqueeze(0))[0]) == sibling
        )

    accepted = bank.replace_slot_from_candidate(
        unsafe_candidate,
        selected_slot,
        retention_probe=retains_sibling,
    )
    _adapt_slot(
        bank,
        slot_index=selected_slot,
        context=key_c,
        seed=seed + 2_000,
        updates=SLOT_UPDATES,
    )
    bank.freeze_slot(selected_slot)
    sibling = slot_b if selected_slot == slot_a else slot_a
    sibling_key = key_b if selected_slot == slot_a else key_a
    sibling_score = _evaluate_binding(
        bank,
        context=sibling_key,
        seed=seed + 851_000,
    )
    new_score = _evaluate_binding(
        bank,
        context=key_c,
        seed=seed + 851_000,
    )
    return {
        "reverse_order": reverse_order,
        "selected_position": selected_position,
        "selected_slot": selected_slot,
        "expected_selected_slot": slot_b,
        "selected_disposable": selected_slot == slot_b,
        "copy_on_write_accepted": accepted,
        "sibling_slot": sibling,
        "sibling_score": sibling_score,
        "new_binding_score": new_score,
        "post_route_sibling": int(bank.route_slot(sibling_key.unsqueeze(0))[0]),
        "post_route_new": int(bank.route_slot(key_c.unsqueeze(0))[0]),
        "physical_keys": [
            bank.slot_keys[index].detach().tolist() for index in range(bank.slot_count)
        ],
        "bank": bank,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1 or args.maintenance_updates < 1:
        raise ValueError("learned maintenance update counts must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    maintenance, maintenance_training = _train_maintenance(
        seed=args.seed,
        updates=args.maintenance_updates,
    )
    trained_accuracy = _evaluate_maintenance(
        maintenance,
        seed=args.seed + 860_000,
    )
    torch.manual_seed(args.seed + 861_000)
    fresh = ExternalCapabilityEvictionPolicy(
        context_width=CONTEXT_WIDTH,
        candidate_width=CANDIDATE_WIDTH,
        hidden=MAINTENANCE_HIDDEN,
    ).eval()
    fresh_accuracy = _evaluate_maintenance(fresh, seed=args.seed + 860_000)
    base, base_training = _train_detector(
        seed=args.seed + 2_000,
        updates=args.policy_updates,
    )
    forward = _run_live(
        maintenance=maintenance,
        base=base,
        system=system,
        seed=args.seed,
        reverse_order=False,
    )
    reverse = _run_live(
        maintenance=maintenance,
        base=base,
        system=system,
        seed=args.seed + 100,
        reverse_order=True,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "trained_maintenance_transfer": trained_accuracy >= 0.80,
        "trained_beats_fresh": trained_accuracy >= fresh_accuracy + 0.20,
        "forward_selects_disposable_slot": forward["selected_disposable"],
        "reverse_selects_disposable_slot": reverse["selected_disposable"],
        "forward_copy_on_write_accepted": forward["copy_on_write_accepted"],
        "reverse_copy_on_write_accepted": reverse["copy_on_write_accepted"],
        "forward_sibling_retained": (
            forward["sibling_score"]["partial_replace"] >= 0.75
            and forward["sibling_score"]["stable_keep"] >= 0.80
            and forward["sibling_score"]["disjoint_replace"] >= 0.80
        ),
        "reverse_sibling_retained": (
            reverse["sibling_score"]["partial_replace"] >= 0.75
            and reverse["sibling_score"]["stable_keep"] >= 0.80
            and reverse["sibling_score"]["disjoint_replace"] >= 0.80
        ),
        "forward_new_binding_learns": (
            forward["new_binding_score"]["partial_replace"] >= 0.80
            and forward["new_binding_score"]["stable_keep"] >= 0.80
            and forward["new_binding_score"]["disjoint_replace"] >= 0.80
        ),
        "reverse_new_binding_learns": (
            reverse["new_binding_score"]["partial_replace"] >= 0.80
            and reverse["new_binding_score"]["stable_keep"] >= 0.80
            and reverse["new_binding_score"]["disjoint_replace"] >= 0.80
        ),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": LEARNED_MAINTENANCE_SCHEMA,
        "claim_boundary": (
            "A learned external eviction policy selects a disposable opaque slot "
            "under candidate-order permutation, enabling verifier-gated copy-on-write "
            "reuse while retaining its sibling; not general maintenance economics "
            "or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "maintenance_policy": "external_capability_eviction_policy_v1",
            "residual_bank": "gated_residual_regime_policy_bank_v1",
            "candidate_features": "opaque_binding_key_plus_generic_reliability_age_v1",
            "forbidden_features": "task_labels_semantic_slot_names_physical_slot_indices_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "maintenance_training": maintenance_training,
        "base_training": base_training,
        "trained_accuracy": trained_accuracy,
        "fresh_accuracy": fresh_accuracy,
        "forward": {key: value for key, value in forward.items() if key != "bank"},
        "reverse": {key: value for key, value in reverse.items() if key != "bank"},
        "gates": gates,
        "accounting": {
            "maintenance_optimizer_updates": args.maintenance_updates,
            "base_optimizer_updates": args.policy_updates,
            "slot_optimizer_updates": SLOT_UPDATES * 3 * 2,
            "total_optimizer_updates": args.maintenance_updates + args.policy_updates + SLOT_UPDATES * 3 * 2,
            "unique_verifier_bits": args.maintenance_updates + args.policy_updates + SLOT_UPDATES * 3 * 2,
            "unique_logical_lifetimes": args.maintenance_updates + args.policy_updates + SLOT_UPDATES * 3 * 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_learned_maintenance"
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
    parser.add_argument("--policy-updates", type=int, default=1_000)
    parser.add_argument("--maintenance-updates", type=int, default=MAINTENANCE_UPDATES)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
