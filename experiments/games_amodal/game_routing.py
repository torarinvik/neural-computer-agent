"""Learned game routing: the shared candidate router selects which slot plays.

This rung closes the loop between the promoted game slots and the promoted
``OpaqueCandidateGrowthRouter``.  Both game slots are trained from scalar
outcomes on a common observation space (Pong's two planes padded with a zero
third plane) with their native action counts, then frozen and hashed; the
padded verifier clamps out-of-range keys during cross-slot counterfactual
play.
Each slot contributes one opaque candidate key derived from its own fresh
greedy events.  A caller-owned route encoder maps the first observation of a
lifetime to an opaque query; the router scores the candidate bank.

Router training is outcome-only via paired counterfactual ranking: for each
fresh lifetime both slots are attempted on identical verifier seeds and only
their realized mastery utilities rank the attempted pair.  No game label,
slot label, or correct-row target ever reaches the router loss.

Hard gates:

* routed end-to-end mastery per game above the mastery threshold,
* routing accuracy per game above the route threshold (audit-side identity),
* candidate permutation invariance of the selection,
* an outcome-shuffled router stays near chance routing,
* both frozen slots bit-for-bit unchanged after router training,
* zero replay anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.games_amodal.environments import PongVerifier, SnakeVerifier
from experiments.games_amodal.pong_growth import parameter_digest
from experiments.games_amodal.snake_acquisition import (
    SnakePolicy,
    train_reward_only,
)
from experiments.games_amodal.train import GridEventEncoder
from neural_computer import (
    OpaqueCandidateGrowthRouter,
    paired_counterfactual_ranking_loss,
)

COMMON_CHANNELS = 3
COMMON_ACTIONS = 4


def pad_observation(observation: torch.Tensor) -> torch.Tensor:
    """Pad any game observation to the common channel count."""

    missing = COMMON_CHANNELS - observation.shape[1]
    if missing < 0:
        raise ValueError("observation has more channels than the common space")
    if missing == 0:
        return observation
    filler = torch.zeros(
        observation.shape[0], missing, *observation.shape[2:],
        device=observation.device,
    )
    return torch.cat([observation, filler], dim=1)


class PaddedVerifier:
    """Present one game verifier through the common observation/action space."""

    def __init__(self, verifier: SnakeVerifier | PongVerifier) -> None:
        self._verifier = verifier
        self.batch_size = verifier.batch_size
        self.action_count = COMMON_ACTIONS

    def reset(self, *, seed: int | None = None) -> None:
        self._verifier.reset(seed=seed)

    def observation(self) -> torch.Tensor:
        return pad_observation(self._verifier.observation())

    def step(self, actions: torch.Tensor):
        clamped = actions.clamp(max=self._verifier.action_count - 1)
        return self._verifier.step(clamped)


def padded_factory(game: str):
    base = {"snake": SnakeVerifier, "pong": PongVerifier}[game]

    def factory(*, batch_size: int, seed: int) -> PaddedVerifier:
        return PaddedVerifier(base(batch_size=batch_size, seed=seed))

    return factory


def lifetime_mastery(
    policy: SnakePolicy,
    factory,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    sample: bool,
) -> torch.Tensor:
    """Play fresh lifetimes and return one mastery bit per row."""

    verifier = factory(batch_size=batch_size, seed=seed)
    verifier.reset(seed=seed)
    total = torch.zeros(batch_size)
    alive = torch.ones(batch_size, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(steps):
            if not bool(alive.any()):
                break
            _, decision = policy.decide(verifier.observation(), sample=sample)
            outcome = verifier.step(decision.key_index)
            total = total + outcome.reward
            alive = outcome.alive
    return (total > 0).float()


def first_observation(factory, *, batch_size: int, seed: int) -> torch.Tensor:
    verifier = factory(batch_size=batch_size, seed=seed)
    verifier.reset(seed=seed)
    return verifier.observation()


def slot_candidate_key(
    policy: SnakePolicy, factory, *, batch_size: int, steps: int, seed: int
) -> torch.Tensor:
    """Derive one opaque candidate key from the slot's own greedy events."""

    verifier = factory(batch_size=batch_size, seed=seed)
    verifier.reset(seed=seed)
    payloads: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(steps):
            event, decision = policy.decide(verifier.observation(), sample=False)
            payloads.append(event.payload.mean(dim=0))
            outcome = verifier.step(decision.key_index)
            if not bool(outcome.alive.any()):
                break
    return F.normalize(torch.stack(payloads).mean(dim=0), dim=-1)


def train_router(
    router: OpaqueCandidateGrowthRouter,
    route_encoder: GridEventEncoder,
    slots: list[SnakePolicy],
    keys: torch.Tensor,
    games: tuple[str, ...],
    *,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    learning_rate: float,
    shuffle_outcomes: bool,
) -> list[dict[str, float]]:
    """Rank attempted slot pairs by realized mastery utilities only."""

    trainable = list(router.parameters()) + list(route_encoder.parameters())
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[dict[str, float]] = []
    for update in range(updates):
        game = games[update % len(games)]
        factory = padded_factory(game)
        lifetime_seed = seed + update
        utilities = torch.stack(
            [
                lifetime_mastery(
                    slot,
                    factory,
                    batch_size=batch_size,
                    steps=steps,
                    seed=lifetime_seed,
                    sample=False,
                )
                for slot in slots
            ],
            dim=1,
        )
        if shuffle_outcomes:
            utilities = utilities[:, torch.randperm(utilities.shape[1])]
        observation = first_observation(
            factory, batch_size=batch_size, seed=lifetime_seed
        )
        query = route_encoder(observation).payload
        scores = router(query, keys)
        attempted = torch.tensor([[0, 1]]).expand(batch_size, -1)
        loss, _ = paired_counterfactual_ranking_loss(
            scores, attempted, utilities
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "loss": float(loss.detach()),
                "replayed_examples": 0.0,
            }
        )
    return history


def routing_audit(
    router: OpaqueCandidateGrowthRouter,
    route_encoder: GridEventEncoder,
    slots: list[SnakePolicy],
    keys: torch.Tensor,
    games: tuple[str, ...],
    correct_slot: dict[str, int],
    *,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    """Measure routing accuracy, routed mastery, and permutation invariance."""

    accuracy: dict[str, float] = {}
    routed_mastery: dict[str, float] = {}
    permutation_matches: list[float] = []
    permutation = torch.tensor([1, 0])
    for game in games:
        factory = padded_factory(game)
        hits: list[float] = []
        masteries: list[float] = []
        for seed in seeds:
            observation = first_observation(
                factory, batch_size=batch_size, seed=seed
            )
            with torch.no_grad():
                query = route_encoder(observation).payload
                selection = router(query, keys).argmax(dim=1)
                permuted = router(query, keys[permutation]).argmax(dim=1)
            hits.append(
                float((selection == correct_slot[game]).float().mean())
            )
            permutation_matches.append(
                float((permutation[permuted] == selection).float().mean())
            )
            majority = int(selection.mode().values)
            masteries.append(
                float(
                    lifetime_mastery(
                        slots[majority],
                        factory,
                        batch_size=batch_size,
                        steps=steps,
                        seed=seed,
                        sample=False,
                    ).mean()
                )
            )
        accuracy[game] = float(torch.tensor(hits).mean())
        routed_mastery[game] = float(torch.tensor(masteries).mean())
    return {
        "routing_accuracy": accuracy,
        "routed_mastery": routed_mastery,
        "permutation_accuracy": float(torch.tensor(permutation_matches).mean()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    shape = {
        "height": 8,
        "width": 8,
        "event_width": args.event_width,
        "intent_width": args.intent_width,
        "hidden": args.hidden,
    }
    common = {"batch_size": args.batch_size, "steps": args.steps}
    games = ("snake", "pong")
    native_actions = {"snake": 4, "pong": 3}
    slots: list[SnakePolicy] = []
    for index, game in enumerate(games):
        slot = SnakePolicy(
            channels=COMMON_CHANNELS,
            action_count=native_actions[game],
            **shape,
        )
        train_reward_only(
            slot,
            updates=args.updates,
            seed=args.seed + index * 50_000,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            shuffle_rewards=False,
            verifier_factory=padded_factory(game),
            **common,
        )
        for parameter in slot.parameters():
            parameter.requires_grad_(False)
        slots.append(slot)
    digests_before = [parameter_digest(slot) for slot in slots]

    keys = torch.stack(
        [
            slot_candidate_key(
                slot,
                padded_factory(game),
                seed=args.seed + 20_000,
                **common,
            )
            for slot, game in zip(slots, games, strict=True)
        ]
    )
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    correct_slot = {game: index for index, game in enumerate(games)}

    router = OpaqueCandidateGrowthRouter(args.event_width, hidden=args.router_hidden)
    route_encoder = GridEventEncoder(
        channels=COMMON_CHANNELS, height=8, width=8, event_width=args.event_width
    )
    shuffled_router = OpaqueCandidateGrowthRouter(
        args.event_width, hidden=args.router_hidden
    )
    shuffled_encoder = GridEventEncoder(
        channels=COMMON_CHANNELS, height=8, width=8, event_width=args.event_width
    )
    shuffled_router.load_state_dict(router.state_dict())
    shuffled_encoder.load_state_dict(route_encoder.state_dict())

    router_common = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seed": args.seed + 70_000,
        "learning_rate": args.route_learning_rate,
    }
    history = train_router(
        router,
        route_encoder,
        slots,
        keys,
        games,
        updates=args.route_updates,
        shuffle_outcomes=False,
        **router_common,
    )
    shuffled_history = train_router(
        shuffled_router,
        shuffled_encoder,
        slots,
        keys,
        games,
        updates=args.route_updates,
        shuffle_outcomes=True,
        **router_common,
    )
    digests_after = [parameter_digest(slot) for slot in slots]

    audit = routing_audit(
        router,
        route_encoder,
        slots,
        keys,
        games,
        correct_slot,
        seeds=eval_seeds,
        **common,
    )
    shuffled_audit = routing_audit(
        shuffled_router,
        shuffled_encoder,
        slots,
        keys,
        games,
        correct_slot,
        seeds=eval_seeds,
        **common,
    )

    accuracy = audit["routing_accuracy"]
    routed = audit["routed_mastery"]
    shuffled_accuracy = shuffled_audit["routing_accuracy"]
    gates = {
        "routing_accurate": all(
            accuracy[game] >= args.route_gate for game in games
        ),
        "routed_mastery": all(routed[game] >= args.mastery_gate for game in games),
        "permutation_invariant": float(audit["permutation_accuracy"])
        >= args.route_gate,
        "shuffled_near_chance": (
            sum(shuffled_accuracy[game] for game in games) / len(games) <= 0.75
        ),
        "slots_unchanged": digests_after == digests_before,
        "no_replay": all(
            entry["replayed_examples"] == 0.0
            for entry in history + shuffled_history
        ),
    }
    return {
        "seed": args.seed,
        "config": {
            "updates": args.updates,
            "route_updates": args.route_updates,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "route_learning_rate": args.route_learning_rate,
            "event_width": args.event_width,
            "intent_width": args.intent_width,
            "hidden": args.hidden,
            "router_hidden": args.router_hidden,
            "eval_seeds": args.eval_seeds,
        },
        "audit": audit,
        "shuffled_audit": shuffled_audit,
        "slot_digests": {
            "before": digests_before,
            "after": digests_after,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--route-learning-rate", type=float, default=3e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--router-hidden", type=int, default=64)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--mastery-gate", type=float, default=0.8)
    parser.add_argument("--route-gate", type=float, default=0.9)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
