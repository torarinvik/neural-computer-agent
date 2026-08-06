"""Shared-controller rung: both games through one frozen amodal core.

This wires the games into the production N-to-M runtime boundary: per-game
frontend encoders emit opaque events onto the bus, one
``AmodalCognitiveController`` performs all recurrent computation, and per-game
``KeypressDecoder`` heads lower opaque intentions to keypresses.  The
controller owns no game, action-count, or modality knowledge.

Phases:

1. Acquire Snake end to end (controller and Snake peripherals trainable).
2. Freeze and hash the controller.  Acquire Pong by training only new Pong
   peripherals (frontend, feedback encoder, decoder) through the frozen core.
3. Re-audit Snake on identical seeds and verify the controller hash.

Controls: a reward-shuffled Pong twin (causal null) and a random-core Pong
twin (same peripheral budget through a never-trained frozen controller),
which measures whether the Snake-trained core transfers computation to Pong.

Hard gates: Snake acquired; Pong acquired through the frozen core; the
shuffled twin near chance; Snake mastery and controller hash unchanged after
Pong training; zero replay.  The random-core comparison is reported, not
gated, and transfer is claimed only from its measured margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from experiments.games_amodal.environments import (
    BreakoutVerifier,
    PongVerifier,
    SnakeVerifier,
)
from experiments.games_amodal.train import GridEventEncoder
from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalOutputBus,
    ControllerFeedback,
    KeypressDecoder,
    KeypressEncoder,
)

GAMES = ("snake", "pong")
GAME_CHANNELS = {"snake": 3, "pong": 2, "breakout": 3}
GAME_ACTIONS = {"snake": 4, "pong": 3, "breakout": 3}
GAME_VERIFIERS = {
    "snake": SnakeVerifier,
    "pong": PongVerifier,
    "breakout": BreakoutVerifier,
}


class SharedControllerAgent(nn.Module):
    """One amodal controller with per-game encoders and decoders."""

    def __init__(
        self,
        *,
        event_width: int,
        intention_width: int,
        feedback_width: int,
        hidden: int,
        event_window_capacity: int = 4,
        games: tuple[str, ...] = GAMES,
    ) -> None:
        super().__init__()
        self.games = tuple(games)
        controller = AmodalCognitiveController(
            width=event_width,
            workspace_slots=4,
            intention_width=intention_width,
            feedback_width=feedback_width,
            event_window_capacity=event_window_capacity,
        )
        self.runtime = AmodalControllerRuntime(
            controller,
            encoders={
                game: GridEventEncoder(
                    channels=GAME_CHANNELS[game],
                    height=8,
                    width=8,
                    event_width=event_width,
                )
                for game in self.games
            },
            output_bus=AmodalOutputBus(
                {
                    game: KeypressDecoder(
                        intention_width, GAME_ACTIONS[game], hidden=hidden
                    )
                    for game in self.games
                }
            ),
        )
        self.feedback_encoders = nn.ModuleDict(
            {
                game: KeypressEncoder(GAME_ACTIONS[game], feedback_width)
                for game in self.games
            }
        )

    @property
    def controller(self) -> AmodalCognitiveController:
        return self.runtime.controller

    def game_modules(self, game: str) -> list[nn.Module]:
        return [
            self.runtime.encoders[game],
            self.runtime.output_bus.decoders[game],
            self.feedback_encoders[game],
        ]


def rollout(
    agent: SharedControllerAgent,
    game: str,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    sample: bool,
    gamma: float,
    detach_interval: int = 1,
) -> dict[str, torch.Tensor | None]:
    """Play fresh lifetimes through the shared controller, outcome-only.

    ``detach_interval`` controls truncated backpropagation through the
    recurrent state during sampling: 1 (default) reproduces the promoted
    one-step regime, k lets policy-gradient credit flow across k controller
    steps before the state is detached.
    """

    if detach_interval < 1:
        raise ValueError("detach interval must be positive")

    verifier = GAME_VERIFIERS[game](batch_size=batch_size, seed=seed)
    verifier.reset(seed=seed)
    controller = agent.controller
    state = controller.initial_state(batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(batch_size, controller.feedback_width),
        reward=torch.zeros(batch_size),
        propensity=torch.ones(batch_size),
        has_feedback=torch.zeros(batch_size),
    )
    decoder: KeypressDecoder = agent.runtime.output_bus.decoders[game]
    rewards: list[torch.Tensor] = []
    log_propensities: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    alive = torch.ones(batch_size, dtype=torch.bool)
    for step in range(steps):
        if not bool(alive.any()):
            break
        masks.append(alive.float())
        event = agent.runtime.encoders[game](verifier.observation())
        output, state = agent.runtime.step_events([event], state, feedback)
        decision = decoder.decide_from_logits(
            output.decoded[game], sample=sample
        )
        log_propensities.append(decision.propensity.clamp_min(1e-8).log())
        outcome = verifier.step(decision.key_index)
        rewards.append(outcome.reward)
        alive = outcome.alive
        feedback = ControllerFeedback(
            action=agent.feedback_encoders[game](decision.key_index),
            reward=outcome.reward,
            propensity=decision.propensity.detach(),
            has_feedback=torch.ones(batch_size),
        )
        if sample and (step + 1) % detach_interval == 0:
            state = state.detached()
    reward_matrix = torch.stack(rewards, dim=1)
    mask_matrix = torch.stack(masks, dim=1)
    returns = torch.zeros_like(reward_matrix)
    running = torch.zeros(batch_size)
    for position in range(reward_matrix.shape[1] - 1, -1, -1):
        running = reward_matrix[:, position] + gamma * running
        returns[:, position] = running
    advantage = None
    propensity_matrix = None
    if sample:
        advantage = returns.detach()
        advantage = advantage - (
            (advantage * mask_matrix).sum() / mask_matrix.sum().clamp_min(1.0)
        )
        propensity_matrix = torch.stack(log_propensities, dim=1)
    return {
        "total_reward": reward_matrix.sum(dim=1),
        "advantage": advantage,
        "log_propensity": propensity_matrix,
        "mask": mask_matrix if sample else None,
    }


def train_game(
    agent: SharedControllerAgent,
    game: str,
    *,
    trainable: list[nn.Parameter],
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    gamma: float,
    learning_rate: float,
    shuffle_rewards: bool,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[dict[str, float]] = []
    for update in range(updates):
        summary = rollout(
            agent,
            game,
            batch_size=batch_size,
            steps=steps,
            seed=seed + update,
            sample=True,
            gamma=gamma,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        if shuffle_rewards:
            advantage = advantage[torch.randperm(advantage.shape[0])]
        loss_terms = advantage * summary["log_propensity"] * summary["mask"]
        loss = -loss_terms.sum() / loss_terms.shape[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "loss": float(loss.detach()),
                "mastery": float((summary["total_reward"] > 0).float().mean()),
                "replayed_examples": 0.0,
            }
        )
    return history


def evaluate_game(
    agent: SharedControllerAgent,
    game: str,
    *,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
    gamma: float,
) -> dict[str, object]:
    masteries: list[float] = []
    for seed in seeds:
        with torch.no_grad():
            summary = rollout(
                agent,
                game,
                batch_size=batch_size,
                steps=steps,
                seed=seed,
                sample=False,
                gamma=gamma,
            )
        masteries.append(float((summary["total_reward"] > 0).float().mean()))
    return {
        "per_seed_mastery": masteries,
        "mastery": float(torch.tensor(masteries).mean()),
    }


def controller_digest(agent: SharedControllerAgent) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(agent.controller.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def set_trainable(modules: list[nn.Module], flag: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(flag)


def trainable_parameters(modules: list[nn.Module]) -> list[nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def acquire_pong_through_core(
    agent: SharedControllerAgent,
    *,
    args: argparse.Namespace,
    shuffle_rewards: bool,
) -> list[dict[str, float]]:
    """Train only Pong peripherals through the (frozen) controller."""

    set_trainable([agent.controller], False)
    set_trainable(agent.game_modules("snake"), False)
    set_trainable(agent.game_modules("pong"), True)
    return train_game(
        agent,
        "pong",
        trainable=trainable_parameters(agent.game_modules("pong")),
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 50_000,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        shuffle_rewards=shuffle_rewards,
    )


def curve_summary(
    history: list[dict[str, float]], *, threshold: float = 0.5
) -> dict[str, float | None]:
    """Summarize how quickly an acquisition run learned, not just where it ended."""

    masteries = [entry["mastery"] for entry in history]
    crossed = [
        entry["update"] for entry in history if entry["mastery"] >= threshold
    ]
    return {
        "mean_training_mastery": float(sum(masteries) / max(len(masteries), 1)),
        "updates_to_half_mastery": None if not crossed else float(crossed[0]),
        "final_training_mastery": masteries[-1] if masteries else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    build = {
        "event_width": args.event_width,
        "intention_width": args.intent_width,
        "feedback_width": args.feedback_width,
        "hidden": args.hidden,
    }
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    eval_common = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seeds": eval_seeds,
        "gamma": args.gamma,
    }

    agent = SharedControllerAgent(**build)
    phase_one = train_game(
        agent,
        "snake",
        trainable=trainable_parameters(
            [agent.controller, *agent.game_modules("snake")]
        ),
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        shuffle_rewards=False,
    )
    snake_before = evaluate_game(agent, "snake", **eval_common)
    digest_before = controller_digest(agent)

    shuffled_agent = SharedControllerAgent(**build)
    shuffled_agent.load_state_dict(agent.state_dict())
    random_core_agent = SharedControllerAgent(**build)

    history = acquire_pong_through_core(agent, args=args, shuffle_rewards=False)
    shuffled_history = acquire_pong_through_core(
        shuffled_agent, args=args, shuffle_rewards=True
    )
    random_history = acquire_pong_through_core(
        random_core_agent, args=args, shuffle_rewards=False
    )

    pong = {
        "trained_core": evaluate_game(agent, "pong", **eval_common),
        "shuffled": evaluate_game(shuffled_agent, "pong", **eval_common),
        "random_core": evaluate_game(random_core_agent, "pong", **eval_common),
    }
    snake_after = evaluate_game(agent, "snake", **eval_common)
    digest_after = controller_digest(agent)

    snake_mastery = float(snake_before["mastery"])
    pong_mastery = float(pong["trained_core"]["mastery"])
    gates = {
        "snake_acquired": snake_mastery >= args.mastery_gate,
        "pong_acquired_through_frozen_core": pong_mastery >= args.mastery_gate,
        "pong_shuffled_near_chance": float(pong["shuffled"]["mastery"])
        <= args.null_gate,
        "snake_retained": float(snake_after["mastery"]) == snake_mastery,
        "controller_unchanged": digest_after == digest_before,
        "no_replay": all(
            entry["replayed_examples"] == 0.0
            for entry in phase_one + history + shuffled_history + random_history
        ),
    }
    return {
        "seed": args.seed,
        "config": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "event_width": args.event_width,
            "intent_width": args.intent_width,
            "feedback_width": args.feedback_width,
            "hidden": args.hidden,
            "eval_seeds": args.eval_seeds,
        },
        "snake": {"before": snake_before, "after": snake_after},
        "pong": pong,
        "core_transfer_margin": pong_mastery
        - float(pong["random_core"]["mastery"]),
        "acquisition_curves": {
            "snake_phase_one": curve_summary(phase_one),
            "pong_trained_core": curve_summary(history),
            "pong_random_core": curve_summary(random_history),
        },
        "controller_digest": {
            "before": digest_before,
            "after": digest_after,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--feedback-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--mastery-gate", type=float, default=0.8)
    parser.add_argument("--null-gate", type=float, default=0.2)
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
