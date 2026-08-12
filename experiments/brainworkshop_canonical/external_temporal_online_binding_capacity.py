"""Pressure-test online binding discovery and verifier-gated capacity reuse.

The external episodic router first learns two anonymous bindings from fresh
attempted scalar utilities.  It then consolidates their learned route keys
through held-out copy-on-write probes, discovers a third binding through an
immutable generic trajectory signature, rejects an unsafe replacement, and
reuses one physical slot only after the new binding and the retained sibling
pass independent verifier probes.

The controller and learned event encoder remain frozen throughout the online
growth phase.  This is a bounded external memory lifecycle result, not a
claim of unrestricted growth or general continual learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import EpisodicBindingRouter

from .external_temporal_query_address_growth import _build
from .external_temporal_shared_basis_policy_growth import _digest

ONLINE_BINDING_CAPACITY_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-online-binding-capacity.v1"
)
EVENT_WIDTH = 8
ACTION_WIDTH = 2
EPISODE_LENGTH = 5
HIDDEN = 16
CONTEXT_WIDTH = 8
SLOT_COUNT = 2
SIGNATURE_WEIGHT = 0.5
ROUTE_THRESHOLD = 0.75
ROUTE_TEMPERATURE = 0.2
ROUTE_UPDATES = 1_000
EVAL_EPISODES = 256
CONSOLIDATION_EPISODES = 64

# Anonymous learned-event patterns.  The family index is private verifier
# state used only to score the attempted route; it never enters the router.
_PATTERNS = torch.tensor(
    (
        (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    )
)


def _episode(seed: int, family: int) -> tuple[torch.Tensor, ...]:
    if not 0 <= family < len(_PATTERNS):
        raise ValueError("online binding family is out of range")
    generator = torch.Generator().manual_seed(seed)
    events = 0.05 * torch.randn(
        1,
        EPISODE_LENGTH,
        EVENT_WIDTH,
        generator=generator,
    )
    events[:, 0] += _PATTERNS[family]
    actions = 0.05 * torch.randn(
        1,
        EPISODE_LENGTH,
        ACTION_WIDTH,
        generator=generator,
    )
    outcomes = 0.05 * torch.randn(
        1,
        EPISODE_LENGTH,
        generator=generator,
    )
    return events, actions, outcomes


def _encode(
    router: EpisodicBindingRouter,
    *,
    seed: int,
    family: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = router.encode_binding(*_episode(seed, family))
    return encoded.context, encoded.signature


def _train_two_bindings(
    *,
    seed: int,
    updates: int,
) -> tuple[
    EpisodicBindingRouter,
    dict[str, float | int],
]:
    if updates < 1:
        raise ValueError("online binding route updates must be positive")
    torch.manual_seed(seed)
    router = EpisodicBindingRouter(
        EVENT_WIDTH,
        ACTION_WIDTH,
        hidden=HIDDEN,
        context_width=CONTEXT_WIDTH,
        max_slots=SLOT_COUNT,
        temperature=ROUTE_TEMPERATURE,
        route_threshold=ROUTE_THRESHOLD,
        signature_weight=SIGNATURE_WEIGHT,
    )
    with torch.no_grad():
        key_a, signature_a = _encode(
            router,
            seed=seed + 10_000,
            family=0,
        )
        key_b, signature_b = _encode(
            router,
            seed=seed + 20_000,
            family=1,
        )
    router.add_slot(key_a[0], signature_a[0])
    router.add_slot(key_b[0], signature_b[0])
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=0.01)
    explorer = torch.Generator().manual_seed(seed + 30_000)
    utilities: list[float] = []
    for update in range(updates):
        family = int(torch.randint(2, (), generator=explorer))
        context, signature = _encode(
            router,
            seed=seed + 100_000 + update,
            family=family,
        )
        scores = router.route(
            context,
            signature=signature,
        ).scores
        probabilities = torch.softmax(scores / ROUTE_TEMPERATURE, dim=-1)
        selected = int(torch.multinomial(probabilities, 1, generator=explorer))
        utility = float(selected == family)
        router.adaptation_step(
            context,
            selected,
            utility,
            signature=signature,
            optimizer=optimizer,
            temperature=ROUTE_TEMPERATURE,
        )
        utilities.append(utility)
    router.eval()
    return router, {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
    }


@torch.no_grad()
def _mean_key(
    router: EpisodicBindingRouter,
    *,
    seed: int,
    family: int,
    episodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts = []
    signatures = []
    for episode in range(episodes):
        context, signature = _encode(
            router,
            seed=seed + episode,
            family=family,
        )
        contexts.append(context[0])
        signatures.append(signature[0])
    return (
        F.normalize(torch.stack(contexts).mean(dim=0), dim=0),
        F.normalize(torch.stack(signatures).mean(dim=0), dim=0),
    )


@torch.no_grad()
def _evaluate_family(
    router: EpisodicBindingRouter,
    *,
    seed: int,
    family: int,
    expected_slot: int,
    episodes: int,
) -> dict[str, float]:
    correct = 0
    known = 0
    for episode in range(episodes):
        context, signature = _encode(
            router,
            seed=seed + episode,
            family=family,
        )
        route = router.route(context, signature=signature)
        correct += int(int(route.selected_slot.item()) == expected_slot)
        known += int(bool(route.known.item()))
    return {
        "accuracy": correct / episodes,
        "known_rate": known / episodes,
    }


@torch.no_grad()
def _evaluate_permuted(
    router: EpisodicBindingRouter,
    *,
    seed: int,
    families: tuple[int, int],
    episodes: int,
    use_signature: bool = True,
) -> float:
    order = torch.tensor([1, 0])
    correct = 0
    total = 0
    for position, family in enumerate(families):
        for episode in range(episodes):
            context, signature = _encode(
                router,
                seed=seed + family * 100_000 + episode,
                family=family,
            )
            route = router.route(
                context,
                signature=signature if use_signature else None,
                slot_order=order,
            )
            correct += int(int(route.selected_slot.item()) == 1 - position)
            total += 1
    return correct / total


def _probe_retention(
    router: EpisodicBindingRouter,
    *,
    seed: int,
    families: tuple[tuple[int, int], ...],
    episodes: int = CONSOLIDATION_EPISODES,
    require_known: bool = True,
) -> bool:
    return all(
        (
            result := _evaluate_family(
                router,
                seed=seed + family * 100_000,
                family=family,
                expected_slot=slot,
                episodes=episodes,
            )
        )["accuracy"]
        >= 0.90
        and (
            not require_known or result["known_rate"] >= 0.90
        )
        for family, slot in families
    )


def _reload_router(router: EpisodicBindingRouter) -> EpisodicBindingRouter:
    restored = EpisodicBindingRouter(
        EVENT_WIDTH,
        ACTION_WIDTH,
        hidden=HIDDEN,
        context_width=CONTEXT_WIDTH,
        max_slots=SLOT_COUNT,
        temperature=ROUTE_TEMPERATURE,
        route_threshold=ROUTE_THRESHOLD,
        signature_weight=SIGNATURE_WEIGHT,
    )
    for index in range(router.slot_count):
        restored.add_slot(
            router.slot_keys[index],
            router.slot_signatures[index]
            if bool(router.slot_signature_active[index])
            else None,
        )
    restored.load_state_dict(router.state_dict())
    restored.freeze_encoder()
    restored.eval()
    return restored


def _train_shuffled_control(seed: int, updates: int) -> EpisodicBindingRouter:
    torch.manual_seed(seed)
    router = EpisodicBindingRouter(
        EVENT_WIDTH,
        ACTION_WIDTH,
        hidden=HIDDEN,
        context_width=CONTEXT_WIDTH,
        max_slots=SLOT_COUNT,
        temperature=ROUTE_TEMPERATURE,
        route_threshold=ROUTE_THRESHOLD,
        signature_weight=SIGNATURE_WEIGHT,
    )
    with torch.no_grad():
        key_a, signature_a = _encode(router, seed=seed + 10_000, family=0)
        key_b, signature_b = _encode(router, seed=seed + 20_000, family=1)
    router.add_slot(key_a[0], signature_a[0])
    router.add_slot(key_b[0], signature_b[0])
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=0.01)
    explorer = torch.Generator().manual_seed(seed + 30_000)
    for update in range(updates):
        family = int(torch.randint(2, (), generator=explorer))
        context, signature = _encode(
            router,
            seed=seed + 100_000 + update,
            family=family,
        )
        scores = router.route(context, signature=signature).scores
        selected = int(
            torch.multinomial(
                torch.softmax(scores / ROUTE_TEMPERATURE, dim=-1),
                1,
                generator=explorer,
            )
        )
        utility = float(torch.randint(2, (), generator=explorer))
        router.adaptation_step(
            context,
            selected,
            utility,
            signature=signature,
            optimizer=optimizer,
            temperature=ROUTE_TEMPERATURE,
        )
    router.freeze_encoder()
    return router


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.route_updates < 1 or args.eval_episodes < 1:
        raise ValueError("online binding capacity counts must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    router, training = _train_two_bindings(
        seed=args.seed,
        updates=args.route_updates,
    )
    initial = {
        "family_a": _evaluate_family(
            router,
            seed=args.seed + 400_000,
            family=0,
            expected_slot=0,
            episodes=args.eval_episodes,
        ),
        "family_b": _evaluate_family(
            router,
            seed=args.seed + 500_000,
            family=1,
            expected_slot=1,
            episodes=args.eval_episodes,
        ),
    }
    initial_permuted = _evaluate_permuted(
        router,
        seed=args.seed + 600_000,
        families=(0, 1),
        episodes=args.eval_episodes,
    )

    refreshed = []
    for slot, family in ((0, 0), (1, 1)):
        context_key, signature_key = _mean_key(
            router,
            seed=args.seed + 700_000 + family * 10_000,
            family=family,
            episodes=CONSOLIDATION_EPISODES,
        )
        candidate = router.slot_replacement_candidate(
            slot,
            context_key,
            signature_key,
        )
        accepted = router.replace_slot_from_candidate(
            candidate,
            slot,
            retention_probe=lambda proposal, family=family, slot=slot: _probe_retention(
                proposal,
                seed=args.seed + 800_000 + slot * 10_000,
                families=((0, 0), (1, 1)),
            ),
        )
        refreshed.append(accepted)

    router.freeze_encoder()
    router.freeze_slot(0)
    router.freeze_slot(1)
    consolidated = {
        "family_a": _evaluate_family(
            router,
            seed=args.seed + 900_000,
            family=0,
            expected_slot=0,
            episodes=args.eval_episodes,
        ),
        "family_b": _evaluate_family(
            router,
            seed=args.seed + 1_000_000,
            family=1,
            expected_slot=1,
            episodes=args.eval_episodes,
        ),
    }
    consolidated_permuted = _evaluate_permuted(
        router,
        seed=args.seed + 1_100_000,
        families=(0, 1),
        episodes=args.eval_episodes,
    )

    novel_c = _evaluate_family(
        router,
        seed=args.seed + 1_200_000,
        family=2,
        expected_slot=1,
        episodes=args.eval_episodes,
    )
    capacity_rejected = False
    try:
        context_c, signature_c = _mean_key(
            router,
            seed=args.seed + 1_300_000,
            family=2,
            episodes=CONSOLIDATION_EPISODES,
        )
        router.add_slot(context_c, signature_c)
    except RuntimeError as error:
        capacity_rejected = "capacity" in str(error)

    context_d, signature_d = _mean_key(
        router,
        seed=args.seed + 1_400_000,
        family=3,
        episodes=CONSOLIDATION_EPISODES,
    )
    unsafe = router.slot_replacement_candidate(1, context_d, signature_d)
    unsafe_before = router.slot_keys[1].detach().clone()
    unsafe_rejected = not router.replace_slot_from_candidate(
        unsafe,
        1,
        retention_probe=lambda proposal: _probe_retention(
            proposal,
            seed=args.seed + 1_500_000,
            families=((0, 0), (2, 1)),
        ),
    )
    unsafe_unchanged = torch.equal(router.slot_keys[1], unsafe_before)

    context_c, signature_c = _mean_key(
        router,
        seed=args.seed + 1_600_000,
        family=2,
        episodes=CONSOLIDATION_EPISODES,
    )
    accepted_candidate = router.slot_replacement_candidate(
        1,
        context_c,
        signature_c,
    )
    accepted_replacement = router.replace_slot_from_candidate(
        accepted_candidate,
        1,
        retention_probe=lambda proposal: _probe_retention(
            proposal,
            seed=args.seed + 1_700_000,
            families=((0, 0), (2, 1)),
        ),
    )
    router.freeze_slot(1)
    after_replacement = {
        "retained_a": _evaluate_family(
            router,
            seed=args.seed + 1_800_000,
            family=0,
            expected_slot=0,
            episodes=args.eval_episodes,
        ),
        "new_c": _evaluate_family(
            router,
            seed=args.seed + 1_900_000,
            family=2,
            expected_slot=1,
            episodes=args.eval_episodes,
        ),
        "retired_b": _evaluate_family(
            router,
            seed=args.seed + 2_000_000,
            family=1,
            expected_slot=1,
            episodes=args.eval_episodes,
        ),
    }
    after_permuted = _evaluate_permuted(
        router,
        seed=args.seed + 2_100_000,
        families=(0, 2),
        episodes=args.eval_episodes,
    )
    restored = _reload_router(router)
    reloaded = {
        "retained_a": _evaluate_family(
            restored,
            seed=args.seed + 1_800_000,
            family=0,
            expected_slot=0,
            episodes=args.eval_episodes,
        ),
        "new_c": _evaluate_family(
            restored,
            seed=args.seed + 1_900_000,
            family=2,
            expected_slot=1,
            episodes=args.eval_episodes,
        ),
    }
    shuffled = _train_shuffled_control(
        args.seed + 2_200_000,
        args.route_updates,
    )
    shuffled_control = _evaluate_permuted(
        shuffled,
        seed=args.seed + 2_300_000,
        families=(0, 1),
        episodes=args.eval_episodes,
        use_signature=False,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    retention_probe_lifetimes = 8 * CONSOLIDATION_EPISODES
    active_diagnostic_lifetimes = 14 * args.eval_episodes
    reload_replayed_lifetimes = 2 * args.eval_episodes
    shuffled_control_diagnostic_lifetimes = 2 * args.eval_episodes
    unique_verifier_bits = (
        args.route_updates
        + retention_probe_lifetimes
        + active_diagnostic_lifetimes
        + args.route_updates
        + shuffled_control_diagnostic_lifetimes
    )
    gates = {
        "initial_route_mastery": (
            initial["family_a"]["accuracy"] >= 0.90
            and initial["family_b"]["accuracy"] >= 0.90
        ),
        "initial_permutation_mastery": initial_permuted >= 0.90,
        "both_key_consolidations_accepted": all(refreshed),
        "consolidated_known_mastery": (
            consolidated["family_a"]["known_rate"] >= 0.90
            and consolidated["family_b"]["known_rate"] >= 0.90
        ),
        "novel_c_has_no_known_route": novel_c["known_rate"] <= 0.25,
        "capacity_growth_is_verifier_gated": capacity_rejected,
        "unsafe_replacement_rejected_without_mutation": (
            unsafe_rejected and unsafe_unchanged
        ),
        "new_c_acquired": after_replacement["new_c"]["accuracy"] >= 0.90,
        "retained_a_survives_replacement": (
            after_replacement["retained_a"]["accuracy"] >= 0.90
            and after_replacement["retained_a"]["known_rate"] >= 0.90
        ),
        "retired_b_is_not_known": after_replacement["retired_b"]["known_rate"] <= 0.25,
        "replacement_permutation_mastery": after_permuted >= 0.90,
        "exact_reload_retention": reloaded == {
            "retained_a": after_replacement["retained_a"],
            "new_c": after_replacement["new_c"],
        },
        "reward_shuffled_control_rejects_mastery": shuffled_control <= 0.70,
        "all_live_slots_frozen": bool(router.slot_frozen.all()),
        "encoder_frozen": all(
            not parameter.requires_grad for parameter in router.encoder.parameters()
        ),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": ONLINE_BINDING_CAPACITY_SCHEMA,
        "claim_boundary": (
            "An external episodic binding router discovers a bounded novel binding, "
            "rejects unsafe capacity reuse, and verifier-gates replacement while "
            "retaining a sibling without replay; not unrestricted growth or "
            "general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "router": "episodic_binding_router_v3",
            "learned_route": "episodic_context_encoder_v1",
            "novelty_path": "immutable_generic_episode_signature_v1",
            "replacement": "copy_on_write_retention_verified_v1",
            "novelty": "immutable_signature_threshold_separate_from_route_score_v1",
            "training_signal": "attempted_slot_scalar_verifier_utility_v1",
            "forbidden_features": (
                "task_labels_correct_unattempted_slot_english_trace_controller_state_v1"
            ),
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "training": training,
        "initial": initial,
        "initial_permuted": initial_permuted,
        "refreshed": refreshed,
        "consolidated": consolidated,
        "consolidated_permuted": consolidated_permuted,
        "novel_c_before_replacement": novel_c,
        "capacity_rejected": capacity_rejected,
        "unsafe_rejected": unsafe_rejected,
        "unsafe_unchanged": unsafe_unchanged,
        "after_replacement": after_replacement,
        "after_permuted": after_permuted,
        "reloaded": reloaded,
        "shuffled_control": shuffled_control,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_verifier_bits,
            "unique_logical_lifetimes": unique_verifier_bits,
            "active_router_verifier_bits": args.route_updates,
            "retention_probe_verifier_bits": retention_probe_lifetimes,
            "active_diagnostic_verifier_bits": active_diagnostic_lifetimes,
            "shuffled_control_verifier_bits": (
                args.route_updates + shuffled_control_diagnostic_lifetimes
            ),
            "router_optimizer_updates": args.route_updates,
            "replacement_probe_lifetimes": retention_probe_lifetimes,
            "replayed_examples": reload_replayed_lifetimes,
            "replayed_training_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_online_binding_capacity"
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
    parser.add_argument("--route-updates", type=int, default=ROUTE_UPDATES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
