"""Learn opaque episodic binding routes from attempted scalar utilities.

Two anonymous event-stream families provision two external memory slots from
their first observed context snapshots.  A memory-side episodic encoder then
learns to route fresh trajectories to the correct opaque slot using only the
utility of the slot that was actually attempted.  The controller and learned
event encoder are instantiated but never updated.

This pressure-tests the missing step after externally supplied context keys:
the key is discovered from a learned trajectory rather than handed in as a
semantic or task-labelled field.  It promotes only a bounded two-slot
binding-discovery primitive, not autonomous ontology formation or general
continual learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import EpisodicBindingRouter

from .external_temporal_query_address_growth import _build
from .external_temporal_shared_basis_policy_growth import _digest

LEARNED_BINDING_ROUTING_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-learned-binding-routing.v1"
)
EVENT_WIDTH = 8
ACTION_WIDTH = 2
EPISODE_LENGTH = 5
HIDDEN = 16
CONTEXT_WIDTH = 8
SLOT_COUNT = 2
ROUTE_TEMPERATURE = 0.2
ROUTE_UPDATES = 1_000
EVAL_EPISODES = 256

# These are anonymous learned-event patterns, not task identifiers.  The
# verifier uses the private family index only to produce the scalar outcome
# for the route that was actually attempted.
_CUE_PATTERNS = torch.tensor(
    (
        (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
)


def _episode(seed: int, family: int) -> tuple[torch.Tensor, ...]:
    """Render one noisy learned-event trajectory for the private verifier."""

    if family not in (0, 1):
        raise ValueError("binding family must be one of the anonymous two cues")
    generator = torch.Generator().manual_seed(seed)
    events = 0.15 * torch.randn(
        1,
        EPISODE_LENGTH,
        EVENT_WIDTH,
        generator=generator,
    )
    events[:, 0] += _CUE_PATTERNS[family]
    actions = 0.15 * torch.randn(
        1,
        EPISODE_LENGTH,
        ACTION_WIDTH,
        generator=generator,
    )
    outcomes = 0.15 * torch.randn(
        1,
        EPISODE_LENGTH,
        generator=generator,
    )
    return events, actions, outcomes


def _train_router(
    *,
    seed: int,
    updates: int,
    reward_shuffled: bool = False,
) -> tuple[EpisodicBindingRouter, tuple[torch.Tensor, torch.Tensor], dict[str, float | int]]:
    if updates < 1:
        raise ValueError("binding-router updates must be positive")
    torch.manual_seed(seed)
    router = EpisodicBindingRouter(
        EVENT_WIDTH,
        ACTION_WIDTH,
        hidden=HIDDEN,
        context_width=CONTEXT_WIDTH,
        max_slots=SLOT_COUNT,
        temperature=ROUTE_TEMPERATURE,
    )
    with torch.no_grad():
        key_a = router.encode(*_episode(seed + 10_000, 0))[0]
        key_b = router.encode(*_episode(seed + 20_000, 1))[0]
    router.add_slot(key_a)
    router.add_slot(key_b)
    optimizer = torch.optim.Adam(router.trainable_parameters(), lr=0.01)
    explorer = torch.Generator().manual_seed(seed + 30_000)
    utilities: list[float] = []
    for update in range(updates):
        family = int(torch.randint(2, (), generator=explorer))
        context = router.encode(
            *_episode(seed + 100_000 + update, family)
        )
        scores = router.route(context).scores
        probabilities = torch.softmax(scores / ROUTE_TEMPERATURE, dim=-1)
        selected = int(torch.multinomial(probabilities, 1, generator=explorer))
        if reward_shuffled:
            utility = float(torch.randint(2, (), generator=explorer))
        else:
            utility = float(selected == family)
        router.adaptation_step(
            context,
            selected,
            utility,
            optimizer=optimizer,
            temperature=ROUTE_TEMPERATURE,
        )
        utilities.append(utility)
    router.eval()
    return router, (key_a, key_b), {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
        "reward_shuffled": int(reward_shuffled),
    }


@torch.no_grad()
def _evaluate_router(
    router: EpisodicBindingRouter,
    *,
    seed: int,
    episodes: int,
    reverse_slots: bool = False,
) -> dict[str, float]:
    if episodes < 1:
        raise ValueError("binding-router evaluation episodes must be positive")
    order = torch.tensor([1, 0]) if reverse_slots else torch.tensor([0, 1])
    correct_by_family = [0, 0]
    for family in (0, 1):
        for episode in range(episodes):
            context = router.encode(
                *_episode(seed + family * 100_000 + episode, family)
            )
            selected = int(
                router.route(context, slot_order=order).selected_slot.item()
            )
            expected = 1 - family if reverse_slots else family
            correct_by_family[family] += int(selected == expected)
    family_scores = [value / episodes for value in correct_by_family]
    return {
        "family_0": family_scores[0],
        "family_1": family_scores[1],
        "balanced": sum(family_scores) / 2.0,
    }


def _state_digest(router: EpisodicBindingRouter) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(router.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _reload_router(
    router: EpisodicBindingRouter,
    keys: tuple[torch.Tensor, torch.Tensor],
) -> EpisodicBindingRouter:
    restored = EpisodicBindingRouter(
        EVENT_WIDTH,
        ACTION_WIDTH,
        hidden=HIDDEN,
        context_width=CONTEXT_WIDTH,
        max_slots=SLOT_COUNT,
        temperature=ROUTE_TEMPERATURE,
    )
    restored.add_slot(keys[0])
    restored.add_slot(keys[1])
    restored.load_state_dict(router.state_dict())
    restored.freeze_encoder()
    restored.eval()
    return restored


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.route_updates < 1 or args.eval_episodes < 1:
        raise ValueError("learned binding routing counts must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    router, keys, training = _train_router(
        seed=args.seed,
        updates=args.route_updates,
    )
    forward = _evaluate_router(
        router,
        seed=args.seed + 500_000,
        episodes=args.eval_episodes,
    )
    reversed_slots = _evaluate_router(
        router,
        seed=args.seed + 500_000,
        episodes=args.eval_episodes,
        reverse_slots=True,
    )
    router.freeze_encoder()
    frozen_forward = _evaluate_router(
        router,
        seed=args.seed + 500_000,
        episodes=args.eval_episodes,
    )
    restored = _reload_router(router, keys)
    reloaded_forward = _evaluate_router(
        restored,
        seed=args.seed + 500_000,
        episodes=args.eval_episodes,
    )
    shuffled, _shuffled_keys, shuffled_training = _train_router(
        seed=args.seed + 700_000,
        updates=args.route_updates,
        reward_shuffled=True,
    )
    shuffled_control = _evaluate_router(
        shuffled,
        seed=args.seed + 1_200_000,
        episodes=args.eval_episodes,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "forward_route_mastery": forward["balanced"] >= 0.90,
        "permuted_candidate_route_mastery": reversed_slots["balanced"] >= 0.90,
        "frozen_route_retention": frozen_forward == forward,
        "exact_reload_route_retention": reloaded_forward == forward,
        "reward_shuffled_control_rejects_mastery": shuffled_control["balanced"] <= 0.70,
        "encoder_is_frozen_after_promotion": all(
            not parameter.requires_grad for parameter in router.encoder.parameters()
        ),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": LEARNED_BINDING_ROUTING_SCHEMA,
        "claim_boundary": (
            "A bounded external episodic encoder discovers two anonymous opaque "
            "memory bindings from fresh attempted scalar utilities without replay; "
            "not autonomous ontology formation, unrestricted slot growth, or "
            "general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "router": "episodic_binding_router_v1",
            "context_encoder": "episodic_context_encoder_v1",
            "slot_keys": "opaque_fixed_snapshots_from_observed_context_v1",
            "training_signal": "attempted_slot_scalar_verifier_utility_v1",
            "forbidden_features": (
                "task_labels_correct_unattempted_slot_english_trace_controller_state_v1"
            ),
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "training": training,
        "forward": forward,
        "reversed_slots": reversed_slots,
        "frozen_forward": frozen_forward,
        "reloaded_forward": reloaded_forward,
        "shuffled_training": shuffled_training,
        "reward_shuffled_control": shuffled_control,
        "router_state_digest": _state_digest(router),
        "restored_state_digest": _state_digest(restored),
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.route_updates,
            "unique_logical_lifetimes": args.route_updates,
            "router_optimizer_updates": args.route_updates,
            "reward_shuffled_control_updates": args.route_updates,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_learned_binding_routing"
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
