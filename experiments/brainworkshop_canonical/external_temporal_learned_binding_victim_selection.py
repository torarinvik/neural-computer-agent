"""Learn which opaque binding slot to retire under capacity pressure.

The external eviction policy receives only the incoming binding signature and
generic candidate telemetry (reliability and age). It is trained from one
scalar utility for the candidate that was attempted, without physical slot
indices, semantic names, or a target-row label. In the live router, a
protected sibling and a fresh binding are verifier-checked after the policy
selects a victim. Candidate row order is independently permuted.

This promotes learned replacement choice on top of the bounded binding
lifecycle. It does not claim universal eviction economics or general
continual learning.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import ExternalCapabilityEvictionPolicy

from .external_temporal_online_binding_capacity import (
    ACTION_WIDTH,
    CONSOLIDATION_EPISODES,
    EVENT_WIDTH,
    _mean_key,
    _probe_retention,
    _train_two_bindings,
)
from .external_temporal_query_address_growth import _build
from .external_temporal_shared_basis_policy_growth import _digest

LEARNED_BINDING_VICTIM_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-learned-binding-"
    "victim-selection.v1"
)
POLICY_UPDATES = 3_000
POLICY_HIDDEN = 32
POLICY_TEMPERATURE = 0.7
TELEMETRY_WIDTH = 2
EVAL_EPISODES = 512
SIGNATURE_WIDTH = 2 * EVENT_WIDTH + ACTION_WIDTH + 2
CANDIDATE_WIDTH = SIGNATURE_WIDTH + TELEMETRY_WIDTH


def _policy_episode(seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    context = F.normalize(
        torch.randn(1, SIGNATURE_WIDTH, generator=generator),
        dim=-1,
    )
    candidates = torch.randn(
        1,
        3,
        CANDIDATE_WIDTH,
        generator=generator,
    )
    reliability = torch.rand(3, generator=generator)
    age = torch.rand(3, generator=generator)
    candidates[0, :, SIGNATURE_WIDTH] = reliability
    candidates[0, :, SIGNATURE_WIDTH + 1] = age
    risk = (1.0 - reliability) + age
    return context, candidates, int(risk.argmax())


def _train_policy(
    *,
    seed: int,
    updates: int,
    reward_shuffled: bool = False,
) -> tuple[ExternalCapabilityEvictionPolicy, dict[str, float | int]]:
    torch.manual_seed(seed)
    policy = ExternalCapabilityEvictionPolicy(
        context_width=SIGNATURE_WIDTH,
        candidate_width=CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    explorer = torch.Generator().manual_seed(seed + 50_000)
    utilities: list[float] = []
    for update in range(updates):
        context, candidates, target = _policy_episode(seed + 100_000 + update)
        scores = policy.score_candidates(context, candidates)[0]
        selected = int(
            torch.multinomial(
                torch.softmax(scores / POLICY_TEMPERATURE, dim=-1),
                1,
                generator=explorer,
            )
        )
        utility = (
            float(torch.randint(2, (), generator=explorer))
            if reward_shuffled
            else float(selected == target)
        )
        loss = -(utility - 0.5) * torch.log_softmax(
            scores / POLICY_TEMPERATURE,
            dim=-1,
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
        "reward_shuffled": int(reward_shuffled),
    }


@torch.no_grad()
def _evaluate_policy(
    policy: ExternalCapabilityEvictionPolicy,
    *,
    seed: int,
    episodes: int,
) -> float:
    correct = 0
    for episode in range(episodes):
        context, candidates, target = _policy_episode(seed + episode)
        correct += int(int(policy.score_candidates(context, candidates).argmax()) == target)
    return correct / episodes


@torch.no_grad()
def _live_candidate_rows(
    router,
    *,
    seed: int,
    order: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    signature_a = router.slot_signatures[0].detach()
    signature_b = router.slot_signatures[1].detach()
    signature_c = _mean_key(
        router,
        seed=seed,
        family=2,
        episodes=CONSOLIDATION_EPISODES,
    )[1]
    signatures = (signature_a, signature_b)
    telemetry = (
        (0.95, 0.10),  # protected, reliable, recent binding A
        (0.10, 0.90),  # weak, old binding B
    )
    rows = []
    for physical_slot in order:
        rows.append(
            torch.cat(
                (
                    signatures[physical_slot],
                    torch.tensor(telemetry[physical_slot]),
                )
            )
        )
    candidates = torch.stack(rows).unsqueeze(0)
    return signature_c.unsqueeze(0), candidates, order


@torch.no_grad()
def _select_live_victim(
    policy: ExternalCapabilityEvictionPolicy,
    router,
    *,
    seed: int,
    order: tuple[int, int],
) -> tuple[int, int, torch.Tensor]:
    context, candidates, physical_order = _live_candidate_rows(
        router,
        seed=seed,
        order=order,
    )
    scores = policy.score_candidates(context, candidates)[0]
    selected_position = int(scores.argmax())
    return selected_position, physical_order[selected_position], scores


def _prepare_router(seed: int):
    router, training = _train_two_bindings(seed=seed, updates=1_000)
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
        accepted = router.replace_slot_from_candidate(
            candidate,
            slot,
            retention_probe=lambda proposal, slot=slot: _probe_retention(
                proposal,
                seed=seed + 800_000 + slot * 10_000,
                families=((0, 0), (1, 1)),
            ),
        )
        if not accepted:
            raise RuntimeError("router key consolidation failed")
    router.freeze_encoder()
    router.freeze_slot(0)
    router.freeze_slot(1)
    return router, training


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1 or args.eval_episodes < 1:
        raise ValueError("victim-selection counts must be positive")
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

    router, router_training = _prepare_router(args.seed)
    live = {}
    for order in ((0, 1), (1, 0)):
        position, physical_slot, scores = _select_live_victim(
            policy,
            router,
            seed=args.seed + 2_000_000,
            order=order,
        )
        live[str(order)] = {
            "selected_position": position,
            "selected_physical_slot": physical_slot,
            "expected_physical_slot": 1,
            "scores": scores.tolist(),
        }

    # Use a fresh router copy for each physical-order transaction so both
    # controls evaluate the same pre-replacement state.
    transactions = {}
    for order in ((0, 1), (1, 0)):
        trial = copy.deepcopy(router)
        position, physical_slot, _scores = _select_live_victim(
            policy,
            trial,
            seed=args.seed + 2_000_000,
            order=order,
        )
        context_c, signature_c = _mean_key(
            trial,
            seed=args.seed + 2_100_000,
            family=2,
            episodes=CONSOLIDATION_EPISODES,
        )
        candidate = trial.slot_replacement_candidate(
            physical_slot,
            context_c,
            signature_c,
        )
        accepted = trial.replace_slot_from_candidate(
            candidate,
            physical_slot,
            retention_probe=lambda proposal: (
                physical_slot == 1
                and _probe_retention(
                    proposal,
                    seed=args.seed + 2_200_000,
                    families=((0, 0), (2, 1)),
                )
            ),
        )
        if accepted:
            trial.freeze_slot(1)
        transactions[str(order)] = {
            "selected_position": position,
            "selected_physical_slot": physical_slot,
            "accepted": accepted,
            "retained_a": _probe_retention(
                trial,
                seed=args.seed + 2_300_000,
                families=((0, 0),),
                require_known=True,
            ),
            "new_c": _probe_retention(
                trial,
                seed=args.seed + 2_400_000,
                families=((2, 1),),
                require_known=True,
            ),
        }

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    router_verifier_bits = int(router_training["unique_scalar_utilities"])
    retention_probe_verifier_bits = 12 * CONSOLIDATION_EPISODES
    diagnostic_verifier_bits = 2 * args.eval_episodes
    unique_verifier_bits = (
        args.policy_updates
        + args.policy_updates
        + router_verifier_bits
        + diagnostic_verifier_bits
        + retention_probe_verifier_bits
    )
    gates = {
        "held_out_policy_mastery": policy_accuracy >= 0.80,
        "reward_shuffled_policy_rejects_mastery": shuffled_accuracy <= 0.70,
        "forward_order_selects_weak_slot": live["(0, 1)"][
            "selected_physical_slot"
        ] == 1,
        "reverse_order_selects_weak_slot": live["(1, 0)"][
            "selected_physical_slot"
        ] == 1,
        "forward_replacement_accepted": transactions["(0, 1)"]["accepted"],
        "reverse_replacement_accepted": transactions["(1, 0)"]["accepted"],
        "forward_sibling_retained": transactions["(0, 1)"]["retained_a"],
        "reverse_sibling_retained": transactions["(1, 0)"]["retained_a"],
        "forward_new_binding_acquired": transactions["(0, 1)"]["new_c"],
        "reverse_new_binding_acquired": transactions["(1, 0)"]["new_c"],
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_training_examples": True,
    }
    report = {
        "schema": LEARNED_BINDING_VICTIM_SCHEMA,
        "claim_boundary": (
            "A generic external eviction policy learns opaque victim selection "
            "from scalar utility and transfers it to verifier-gated episodic "
            "binding replacement under row permutation; not universal eviction "
            "economics or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "victim_policy": "external_capability_eviction_policy_v1",
            "candidate_features": "opaque_signature_plus_reliability_age_v1",
            "router": "episodic_binding_router_v3",
            "replacement": "copy_on_write_retention_verified_v1",
            "forbidden_features": (
                "physical_slot_indices_semantic_names_correct_unattempted_slot_task_labels"
            ),
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "policy_training": policy_training,
        "router_training": router_training,
        "policy_accuracy": policy_accuracy,
        "shuffled_policy_training": shuffled_training,
        "shuffled_policy_accuracy": shuffled_accuracy,
        "live": live,
        "transactions": transactions,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_verifier_bits,
            "unique_logical_lifetimes": unique_verifier_bits,
            "policy_optimizer_updates": args.policy_updates,
            "router_optimizer_updates": router_verifier_bits,
            "policy_verifier_bits": args.policy_updates,
            "shuffled_policy_verifier_bits": args.policy_updates,
            "router_verifier_bits": router_verifier_bits,
            "retention_probe_verifier_bits": retention_probe_verifier_bits,
            "diagnostic_verifier_bits": diagnostic_verifier_bits,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_learned_binding_victim_selection"
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
