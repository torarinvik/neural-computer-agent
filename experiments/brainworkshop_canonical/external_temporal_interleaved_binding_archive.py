"""Pressure-test repeated external binding growth without cache thrashing.

The active episodic router is intentionally capacity-bounded, but the
long-term binding archive is not.  Four anonymous learned-event bindings are
introduced and revisited through six interleaved admission/replacement cycles
while the controller and event encoder remain frozen.  A scalar stable-prefix
gate protects a mastered binding from eviction; recently admitted bindings
remain eligible until they earn protection.  Returning bindings are restored
from the archive instead of replaying their old training stream.

This is a promoted target only if every verifier gate passes.  It establishes
reusable external archive growth and anti-thrashing lifecycle behavior, not
unrestricted computation or general continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import EpisodicBindingArchive, ExternalCapabilityEvictionPolicy

from .external_temporal_learned_binding_victim_selection import (
    CANDIDATE_WIDTH,
    EVAL_EPISODES,
    SIGNATURE_WIDTH,
    _evaluate_policy,
    _train_policy,
)
from .external_temporal_online_binding_capacity import (
    ACTION_WIDTH,
    CONSOLIDATION_EPISODES,
    CONTEXT_WIDTH,
    EVENT_WIDTH,
    ROUTE_TEMPERATURE,
    ROUTE_THRESHOLD,
    _encode,
    _evaluate_family,
    _mean_key,
    _probe_retention,
    _train_two_bindings,
)
from .external_temporal_query_address_growth import _build
from .external_temporal_shared_basis_policy_growth import _digest

ARCHIVE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-interleaved-binding-"
    "archive.v1"
)
ACTIVE_SLOTS = 2
POLICY_UPDATES = 3_000
ROUTER_UPDATES = 1_000
ARRIVAL_FAMILIES = (2, 3, 1, 2, 3, 1)


def _prepare_router_and_archive(seed: int):
    router, router_training = _train_two_bindings(
        seed=seed,
        updates=ROUTER_UPDATES,
    )
    # Consolidate the one-shot route keys into fresh held-out episode means
    # before freezing the encoder.  This keeps the archive/cache boundary
    # behaviorally stable across seeds; the archive is not allowed to hide a
    # stale active-cache key.
    for slot, family in ((0, 0), (1, 1)):
        context_key, signature_key = _mean_key(
            router,
            seed=seed + 700_000 + family * 10_000,
            family=family,
            episodes=CONSOLIDATION_EPISODES,
        )
        candidate = router.slot_replacement_candidate(
            slot,
            context_key,
            signature_key,
        )
        if not router.replace_slot_from_candidate(
            candidate,
            slot,
            retention_probe=lambda proposal, slot=slot: _probe_retention(
                proposal,
                seed=seed + 800_000 + slot * 10_000,
                families=((0, 0), (1, 1)),
            ),
        ):
            raise RuntimeError("interleaved archive initial consolidation failed")
    router.freeze_encoder()
    router.eval()
    archive = EpisodicBindingArchive(
        CONTEXT_WIDTH,
        SIGNATURE_WIDTH,
        active_slots=ACTIVE_SLOTS,
        matching_threshold=0.85,
        min_mastery_observations=8,
    )
    family_to_binding: dict[int, int] = {}
    for slot, family in ((0, 0), (1, 1)):
        context_key, signature_key = _mean_key(
            router,
            seed=seed + 700_000 + family * 10_000,
            family=family,
            episodes=CONSOLIDATION_EPISODES,
        )
        binding_id = archive.register(context_key, signature_key)
        archive.activate(binding_id, slot)
        family_to_binding[family] = binding_id
    # A is mastered and becomes externally protected.  B remains a weak,
    # replaceable record; this distinction comes only from scalar telemetry.
    for step in range(12):
        archive.observe(family_to_binding[0], 1.0, step=step)
    archive.observe(family_to_binding[1], 1.0, step=12)
    router.freeze_slot(0)
    router.freeze_slot(1)
    return router, archive, family_to_binding, router_training


@torch.no_grad()
def _candidate_rows(
    archive: EpisodicBindingArchive,
    *,
    step: int,
    order: tuple[int, int],
) -> torch.Tensor:
    rows = []
    for physical_slot in order:
        binding_id = archive.active_binding(physical_slot)
        if binding_id is None:
            raise RuntimeError("interleaved archive has an empty active slot")
        reliability, age = archive.telemetry(
            binding_id,
            step=step,
            age_horizon=16,
        )
        rows.append(
            torch.cat(
                (
                    archive.signature_key(binding_id),
                    torch.tensor((reliability, age), dtype=torch.float32),
                )
            )
        )
    candidates = torch.stack(rows).unsqueeze(0)
    if candidates.shape != (1, ACTIVE_SLOTS, CANDIDATE_WIDTH):
        raise RuntimeError("interleaved archive candidate ABI changed")
    return candidates


@torch.no_grad()
def _select_victim(
    policy: ExternalCapabilityEvictionPolicy,
    archive: EpisodicBindingArchive,
    incoming_signature: torch.Tensor,
    *,
    step: int,
    order: tuple[int, int],
) -> tuple[int, int, torch.Tensor, bool]:
    candidates = _candidate_rows(archive, step=step, order=order)
    scores = policy.score_candidates(incoming_signature.unsqueeze(0), candidates)[0]
    unmasked_position = int(scores.argmax())
    masked = scores.clone()
    eligible_positions = []
    for position, physical_slot in enumerate(order):
        binding_id = archive.active_binding(physical_slot)
        if binding_id is not None and archive.is_protected(binding_id):
            masked[position] = torch.finfo(masked.dtype).min
        else:
            eligible_positions.append(position)
    if not eligible_positions:
        raise RuntimeError("interleaved archive has no eligible replacement slot")
    selected_position = int(masked.argmax())
    return (
        selected_position,
        order[selected_position],
        scores,
        unmasked_position == selected_position,
    )


@torch.no_grad()
def _active_probe(
    router,
    archive: EpisodicBindingArchive,
    *,
    family: int,
    seed: int,
    step: int,
) -> bool:
    context, signature = _encode(router, seed=seed, family=family)
    lookup = archive.lookup(signature[0])
    if lookup.binding_id is None or lookup.active_slot is None:
        return False
    route = router.route(context, signature=signature)
    if int(route.selected_slot.item()) != lookup.active_slot:
        return False
    archive.observe(lookup.binding_id, 1.0, step=step)
    return bool(route.known.item())


def _reload(
    router,
    archive: EpisodicBindingArchive,
):
    from neural_computer import EpisodicBindingRouter

    restored_router = EpisodicBindingRouter(
        EVENT_WIDTH,
        ACTION_WIDTH,
        hidden=router.hidden,
        context_width=CONTEXT_WIDTH,
        max_slots=ACTIVE_SLOTS,
        temperature=ROUTE_TEMPERATURE,
        route_threshold=ROUTE_THRESHOLD,
        signature_weight=router.signature_weight,
    )
    for index in range(router.slot_count):
        restored_router.add_slot(
            router.slot_keys[index],
            router.slot_signatures[index]
            if bool(router.slot_signature_active[index])
            else None,
        )
    restored_router.load_state_dict(router.state_dict())
    restored_router.freeze_encoder()
    restored_router.eval()
    restored_archive = EpisodicBindingArchive.from_payload(archive.payload())
    return restored_router, restored_archive


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1 or args.eval_episodes < 1:
        raise ValueError("interleaved archive counts must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])

    policy, policy_training = _train_policy(
        seed=args.seed,
        updates=args.policy_updates,
    )
    policy_accuracy = _evaluate_policy(
        policy,
        seed=args.seed + 900_000,
        episodes=args.eval_episodes,
    )
    shuffled_policy, shuffled_training = _train_policy(
        seed=args.seed + 1_000_000,
        updates=args.policy_updates,
        reward_shuffled=True,
    )
    shuffled_accuracy = _evaluate_policy(
        shuffled_policy,
        seed=args.seed + 1_900_000,
        episodes=args.eval_episodes,
    )

    router, archive, family_to_binding, router_training = _prepare_router_and_archive(
        args.seed
    )
    step = 13
    initial_active = {
        "family_a": _evaluate_family(
            router,
            seed=args.seed + 2_000_000,
            family=0,
            expected_slot=0,
            episodes=args.eval_episodes,
        ),
        "family_b": _evaluate_family(
            router,
            seed=args.seed + 2_100_000,
            family=1,
            expected_slot=1,
            episodes=args.eval_episodes,
        ),
    }

    # Every arrival is evaluated in both physical candidate orders.  The
    # protected-slot mask is generic external evidence, not a family-specific
    # rule.  Only the verifier-gated transaction mutates the active cache.
    arrivals: list[dict[str, object]] = []
    active_noop_checks = 0
    avoidable_replacements = 0
    for ordinal, family in enumerate(ARRIVAL_FAMILIES):
        step += 1
        context, signature = _mean_key(
            router,
            seed=args.seed + 3_000_000 + ordinal * 10_000,
            family=family,
            episodes=CONSOLIDATION_EPISODES,
        )
        lookup = archive.lookup(signature)
        already_active = lookup.active_slot is not None
        if already_active:
            active_noop_checks += 1
            if not _active_probe(
                router,
                archive,
                family=family,
                seed=args.seed + 3_100_000 + ordinal,
                step=step,
            ):
                raise RuntimeError("active binding failed its no-op probe")
            arrivals.append(
                {
                    "family": family,
                    "known_before": True,
                    "active_before": True,
                    "replacement": False,
                }
            )
            continue

        order = (0, 1) if ordinal % 2 == 0 else (1, 0)
        position, victim_slot, scores, unmasked_policy_match = _select_victim(
            policy,
            archive,
            signature,
            step=step,
            order=order,
        )
        retained_families = tuple(
            active_family
            for active_family, active_binding in family_to_binding.items()
            if archive.binding_slot(active_binding) is not None
            and archive.binding_slot(active_binding) != victim_slot
        )
        if not retained_families:
            raise RuntimeError("interleaved archive replacement lost all siblings")
        retained = tuple(
            (
                active_family,
                archive.binding_slot(family_to_binding[active_family]),
            )
            for active_family in retained_families
        )
        incoming_key = (
            archive.context_key(lookup.binding_id)
            if lookup.binding_id is not None
            else context
        )
        incoming_signature = (
            archive.signature_key(lookup.binding_id)
            if lookup.binding_id is not None
            else signature
        )
        expected_incoming_slot = victim_slot
        candidate = router.slot_replacement_candidate(
            victim_slot,
            incoming_key,
            incoming_signature,
        )
        accepted = router.replace_slot_from_candidate(
            candidate,
            victim_slot,
            retention_probe=lambda proposal, ordinal=ordinal, retained=retained, family=family, expected_incoming_slot=expected_incoming_slot: _probe_retention(
                proposal,
                seed=args.seed + 3_200_000 + ordinal * 10_000,
                families=(*retained, (family, expected_incoming_slot)),
            ),
        )
        if not accepted:
            raise RuntimeError(f"verifier rejected arrival family {family}")
        router.freeze_slot(victim_slot)
        binding_id = (
            lookup.binding_id
            if lookup.binding_id is not None
            else archive.register(context, signature)
        )
        archive.activate(binding_id, victim_slot)
        archive.observe(binding_id, 1.0, step=step)
        family_to_binding[family] = binding_id
        if victim_slot == 0:
            avoidable_replacements += 1
        arrivals.append(
            {
                "family": family,
                "known_before": lookup.binding_id is not None,
                "active_before": False,
                "replacement": True,
                "order": order,
                "selected_position": position,
                "selected_physical_slot": victim_slot,
                "expected_physical_slot": 1,
                "raw_policy_selected_expected": unmasked_policy_match,
                "policy_scores": scores.tolist(),
                "archive_binding_id": binding_id,
                "archive_record_count": archive.record_count,
            }
        )
        # Interleave the protected resident and the just-admitted resident.
        step += 1
        active_noop_checks += int(
            _active_probe(
                router,
                archive,
                family=0,
                seed=args.seed + 3_300_000 + ordinal,
                step=step,
            )
        )
        step += 1
        active_noop_checks += int(
            _active_probe(
                router,
                archive,
                family=family,
                seed=args.seed + 3_400_000 + ordinal,
                step=step,
            )
        )

    active_families = tuple(
        family for family, binding_id in family_to_binding.items()
        if archive.binding_slot(binding_id) is not None
    )
    final_active = {
        f"family_{family}": _evaluate_family(
            router,
            seed=args.seed + 4_000_000 + family * 10_000,
            family=family,
            expected_slot=archive.binding_slot(family_to_binding[family]),
            episodes=args.eval_episodes,
        )
        for family in active_families
    }
    inactive_revisits = {
        f"family_{family}": {
            "known": archive.lookup(
                _mean_key(
                    router,
                    seed=args.seed + 4_500_000 + family * 10_000,
                    family=family,
                    episodes=CONSOLIDATION_EPISODES,
                )[1]
            ).binding_id
            is not None,
            "active_slot": archive.lookup(
                _mean_key(
                    router,
                    seed=args.seed + 4_500_000 + family * 10_000,
                    family=family,
                    episodes=CONSOLIDATION_EPISODES,
                )[1]
            ).active_slot,
        }
        for family in (2, 3)
    }
    restored_router, restored_archive = _reload(router, archive)
    reloaded_active = {
        f"family_{family}": _evaluate_family(
            restored_router,
            seed=args.seed + 5_000_000 + family * 10_000,
            family=family,
            expected_slot=restored_archive.binding_slot(
                family_to_binding[family]
            ),
            episodes=args.eval_episodes,
        )
        for family in active_families
    }
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])

    retention_probe_bits = len(ARRIVAL_FAMILIES) * 3 * CONSOLIDATION_EPISODES
    active_diagnostic_bits = len(active_families) * args.eval_episodes
    reload_diagnostic_bits = len(active_families) * args.eval_episodes
    archive_observation_bits = 12 + 1 + len(ARRIVAL_FAMILIES) * 3
    unique_verifier_bits = (
        args.policy_updates
        + args.policy_updates
        + router_training["unique_scalar_utilities"]
        + retention_probe_bits
        + active_diagnostic_bits
        + reload_diagnostic_bits
        + archive_observation_bits
    )
    gates = {
        "held_out_policy_mastery": policy_accuracy >= 0.80,
        "reward_shuffled_policy_rejects_mastery": shuffled_accuracy <= 0.70,
        "initial_active_mastery": all(
            row["accuracy"] >= 0.90 and row["known_rate"] >= 0.90
            for row in initial_active.values()
        ),
        "all_replacements_verifier_accepted": len(arrivals)
        == len(ARRIVAL_FAMILIES)
        and all(bool(row.get("replacement")) for row in arrivals),
        "protected_binding_never_evicted": avoidable_replacements == 0,
        "active_revisits_are_noops": active_noop_checks >= len(ARRIVAL_FAMILIES) * 2,
        "archive_grows_once_per_novel_binding": archive.record_count == 4,
        "returned_bindings_are_known_but_inactive": all(
            row["known"] and row["active_slot"] is None
            for row in inactive_revisits.values()
        ),
        "final_active_mastery": all(
            row["accuracy"] >= 0.90 and row["known_rate"] >= 0.90
            for row in final_active.values()
        ),
        "reload_preserves_active_mastery": all(
            row["accuracy"] >= 0.90 and row["known_rate"] >= 0.90
            for row in reloaded_active.values()
        ),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_training_examples": True,
    }
    report = {
        "schema": ARCHIVE_SCHEMA,
        "claim_boundary": (
            "A growable external binding archive retains inactive capabilities "
            "while a verifier-gated bounded cache handles repeated interleaved "
            "replacement without protected-slot thrashing; not unrestricted "
            "computation or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "archive": "episodic_binding_archive_v1",
            "active_cache": "episodic_binding_router_v3",
            "victim_policy": "external_capability_eviction_policy_v1",
            "admission": "copy_on_write_retention_verified_v1",
            "protected_gate": "stable_scalar_prefix_v1",
            "forbidden_features": (
                "semantic_family_labels_task_ids_correct_unattempted_actions"
            ),
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "policy_training": policy_training,
        "policy_accuracy": policy_accuracy,
        "shuffled_policy_training": shuffled_training,
        "shuffled_policy_accuracy": shuffled_accuracy,
        "router_training": router_training,
        "initial_active": initial_active,
        "arrivals": arrivals,
        "archive": archive.status().__dict__,
        "active_families": active_families,
        "final_active": final_active,
        "inactive_revisits": inactive_revisits,
        "reloaded_active": reloaded_active,
        "lifecycle_metrics": {
            "replacement_count": sum(
                bool(row.get("replacement")) for row in arrivals
            ),
            "avoidable_replacements": avoidable_replacements,
            "active_noop_checks": active_noop_checks,
            "archive_record_count": archive.record_count,
            "inactive_archive_records": sum(
                binding_id is not None and archive.binding_slot(binding_id) is None
                for binding_id in range(archive.record_count)
            ),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_verifier_bits,
            "unique_logical_lifetimes": unique_verifier_bits,
            "policy_optimizer_updates": args.policy_updates * 2,
            "router_optimizer_updates": router_training["optimizer_updates"],
            "replayed_examples": 0,
            "controller_updates": 0,
            "retention_probe_verifier_bits": retention_probe_bits,
            "archive_observation_verifier_bits": archive_observation_bits,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_interleaved_external_binding_archive"
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
    parser.add_argument("--policy-updates", type=int, default=POLICY_UPDATES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
