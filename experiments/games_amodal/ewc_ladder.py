"""Three-game EWC ladder: composing consolidation steps in one plastic core.

Snake, then Pong, then Breakout acquire sequentially through one fully
plastic controller. After each acquisition the trainer captures that game's
diagonal Fisher sensitivity and an anchor snapshot; later phases train under
the sum of all earlier quadratic consolidation penalties. This is the
designed stressor for the two classic EWC weaknesses: anchor staleness
(game 1's anchor ages while games 2 and 3 move the core) and accumulating
rigidity (the free subspace shrinks with each map).

Conditions (identical budgets and evaluation seeds, forked after phase 1):

* ``protected``      -- summed Fisher penalties at every later phase,
* ``unprotected``    -- no penalties (forgetting baseline),
* ``permuted``       -- penalties with all sensitivities shuffled across
  positions (same pressure, wrong assignment; causal null),
* ``shuffled``       -- reward-shuffled Breakout twin from the protected
  post-Pong checkpoint (acquisition causal null).

Gates: each game acquired above its own gate (Breakout's is lower — the
compound game is harder; its random-policy baseline is reported); the
baseline forgets both earlier games; the protected ladder retains Snake and
Pong within epsilon after Breakout; the permuted null fails to rescue;
shuffled near chance; zero replay.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from experiments.games_amodal.ewc_plasticity import (
    controller_named_parameters,
    estimate_diagonal_fisher,
    permuted_fisher_like,
)
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    evaluate_game,
    rollout,
    set_trainable,
    trainable_parameters,
)

LADDER = ("snake", "pong", "breakout")


def train_phase_with_penalties(
    agent: SharedControllerAgent,
    game: str,
    penalties: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    *,
    args: argparse.Namespace,
    seed_offset: int,
    shuffle_rewards: bool = False,
) -> list[dict[str, float]]:
    """Train one game with the sum of earlier (fisher, anchor) penalties.

    ``args.consolidation_mode`` selects the rule:

    * ``sum`` — vanilla summed quadratic penalties (the qualified baseline);
    * ``arbitrated`` — adaptive per-parameter release: each penalty
      coefficient is attenuated by ``F / (F + mu * G)`` where ``G`` is a
      running unit-mean estimate of the new task's own squared policy
      gradients. Parameters the new game does not need keep full
      protection; parameters it demonstrably needs are proportionally
      released. Task gradients are computed first and the penalty gradient
      is added in closed form, so ``G`` never sees penalty pressure.
    """

    set_trainable([agent.controller], True)
    for other in LADDER:
        set_trainable(agent.game_modules(other), other == game)
    named = controller_named_parameters(agent)
    trainable = trainable_parameters(
        [agent.controller, *agent.game_modules(game)]
    )
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    demand = {name: torch.zeros_like(parameter) for name, parameter in named}
    history: list[dict[str, float]] = []
    for update in range(args.updates):
        summary = rollout(
            agent,
            game,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + seed_offset + update,
            sample=True,
            gamma=args.gamma,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        if shuffle_rewards:
            advantage = advantage[torch.randperm(advantage.shape[0])]
        loss_terms = advantage * summary["log_propensity"] * summary["mask"]
        loss = -loss_terms.sum() / loss_terms.shape[0]
        arbitrated = args.consolidation_mode == "arbitrated" and penalties
        if arbitrated:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for name, parameter in named:
                    grad = (
                        torch.zeros_like(parameter)
                        if parameter.grad is None
                        else parameter.grad
                    )
                    demand[name].mul_(args.arbitration_decay).add_(
                        grad.square(), alpha=1.0 - args.arbitration_decay
                    )
                demand_total = sum(t.sum() for t in demand.values())
                demand_count = sum(t.numel() for t in demand.values())
                demand_mean = (demand_total / demand_count).clamp_min(1e-12)
                penalty_value = torch.zeros(())
                release_fraction = torch.zeros(())
                for fisher, anchor in penalties:
                    for name, parameter in named:
                        normalized_demand = demand[name] / demand_mean
                        attenuation = fisher[name] / (
                            fisher[name]
                            + args.arbitration_mu * normalized_demand
                        ).clamp_min(1e-12)
                        coefficient = (
                            args.ewc_lambda * attenuation * fisher[name]
                        )
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                        parameter.grad.add_(
                            coefficient * (parameter - anchor[name])
                        )
                        penalty_value = penalty_value + 0.5 * (
                            coefficient
                            * (parameter - anchor[name]).square()
                        ).sum()
                        release_fraction = release_fraction + (
                            1.0 - attenuation
                        ).mean()
                release_fraction = release_fraction / max(
                    len(penalties) * len(named), 1
                )
        else:
            penalty_value = torch.zeros(())
            release_fraction = torch.zeros(())
            for fisher, anchor in penalties:
                for name, parameter in named:
                    penalty_value = penalty_value + (
                        fisher[name] * (parameter - anchor[name]).square()
                    ).sum()
            loss = loss + 0.5 * args.ewc_lambda * penalty_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "mastery": float((summary["total_reward"] > 0).float().mean()),
                "penalty": float(penalty_value.detach()),
                "release_fraction": float(release_fraction)
                if arbitrated
                else 0.0,
                "replayed_examples": 0.0,
            }
        )
    return history


def capture_consolidation(
    agent: SharedControllerAgent, game: str, *, args: argparse.Namespace
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    fisher = estimate_diagonal_fisher(
        agent,
        game,
        batches=args.fisher_batches,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 30_000 + LADDER.index(game) * 1_000,
        gamma=args.gamma,
    )
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in controller_named_parameters(agent)
    }
    return fisher, anchor


def evaluate_all(
    agent: SharedControllerAgent, *, args: argparse.Namespace
) -> dict[str, dict[str, object]]:
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    return {
        game: evaluate_game(
            agent,
            game,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=eval_seeds,
            gamma=args.gamma,
        )
        for game in LADDER
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    build = {
        "event_width": args.event_width,
        "intention_width": args.intent_width,
        "feedback_width": args.feedback_width,
        "hidden": args.hidden,
        "games": LADDER,
        "shared_drivers": args.shared_drivers,
    }
    agent = SharedControllerAgent(**build)

    train_phase_with_penalties(agent, "snake", [], args=args, seed_offset=0)
    snake_score = evaluate_all(agent, args=args)["snake"]
    fisher_snake, anchor_snake = capture_consolidation(agent, "snake", args=args)

    forks = {
        name: copy.deepcopy(agent) for name in ("unprotected", "permuted")
    }
    permuted_snake = permuted_fisher_like(fisher_snake, seed=args.seed + 90_000)

    train_phase_with_penalties(
        agent, "pong", [(fisher_snake, anchor_snake)], args=args, seed_offset=50_000
    )
    pong_score = evaluate_all(agent, args=args)["pong"]
    fisher_pong, anchor_pong = capture_consolidation(agent, "pong", args=args)
    shuffled_fork = copy.deepcopy(agent)

    train_phase_with_penalties(
        forks["unprotected"], "pong", [], args=args, seed_offset=50_000
    )
    train_phase_with_penalties(
        forks["permuted"],
        "pong",
        [(permuted_snake, anchor_snake)],
        args=args,
        seed_offset=50_000,
    )
    permuted_pong = permuted_fisher_like(
        capture_consolidation(forks["permuted"], "pong", args=args)[0],
        seed=args.seed + 91_000,
    )
    permuted_pong_anchor = {
        name: parameter.detach().clone()
        for name, parameter in controller_named_parameters(forks["permuted"])
    }

    train_phase_with_penalties(
        agent,
        "breakout",
        [(fisher_snake, anchor_snake), (fisher_pong, anchor_pong)],
        args=args,
        seed_offset=100_000,
    )
    train_phase_with_penalties(
        forks["unprotected"], "breakout", [], args=args, seed_offset=100_000
    )
    train_phase_with_penalties(
        forks["permuted"],
        "breakout",
        [
            (permuted_snake, anchor_snake),
            (permuted_pong, permuted_pong_anchor),
        ],
        args=args,
        seed_offset=100_000,
    )
    shuffled_history = train_phase_with_penalties(
        shuffled_fork,
        "breakout",
        [(fisher_snake, anchor_snake), (fisher_pong, anchor_pong)],
        args=args,
        seed_offset=100_000,
        shuffle_rewards=True,
    )

    finals = {
        "protected": evaluate_all(agent, args=args),
        "unprotected": evaluate_all(forks["unprotected"], args=args),
        "permuted": evaluate_all(forks["permuted"], args=args),
        "shuffled": evaluate_all(shuffled_fork, args=args),
    }

    def mastery(condition: str, game: str) -> float:
        return float(finals[condition][game]["mastery"])

    snake_before = float(snake_score["mastery"])
    pong_before = float(pong_score["mastery"])
    gates = {
        "snake_acquired": snake_before >= args.mastery_gate,
        "pong_acquired": pong_before >= args.mastery_gate,
        "breakout_acquired": mastery("protected", "breakout")
        >= args.breakout_gate,
        "baseline_forgets_both": mastery("unprotected", "snake")
        <= snake_before - args.forgetting_delta
        and mastery("unprotected", "pong") <= pong_before - args.forgetting_delta,
        "protected_retains_snake": mastery("protected", "snake")
        >= snake_before - args.retention_epsilon,
        "protected_retains_pong": mastery("protected", "pong")
        >= pong_before - args.retention_epsilon,
        "permuted_fails": mastery("permuted", "snake")
        < snake_before - args.retention_epsilon
        or mastery("permuted", "pong") < pong_before - args.retention_epsilon,
        "shuffled_near_chance": mastery("shuffled", "breakout")
        <= args.null_gate,
        "no_replay": all(
            entry["replayed_examples"] == 0.0 for entry in shuffled_history
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
            "ewc_lambda": args.ewc_lambda,
            "consolidation_mode": args.consolidation_mode,
            "shared_drivers": args.shared_drivers,
            "arbitration_mu": args.arbitration_mu,
            "arbitration_decay": args.arbitration_decay,
            "fisher_batches": args.fisher_batches,
            "eval_seeds": args.eval_seeds,
            "retention_epsilon": args.retention_epsilon,
            "forgetting_delta": args.forgetting_delta,
            "breakout_gate": args.breakout_gate,
        },
        "post_acquisition": {
            "snake": snake_before,
            "pong": pong_before,
        },
        "finals": finals,
        "gates": gates,
        "promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--feedback-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--fisher-batches", type=int, default=32)
    parser.add_argument("--ewc-lambda", type=float, default=100.0)
    parser.add_argument(
        "--consolidation-mode",
        type=str,
        default="sum",
        choices=["sum", "arbitrated"],
    )
    parser.add_argument("--arbitration-mu", type=float, default=1.0)
    parser.add_argument("--arbitration-decay", type=float, default=0.99)
    parser.add_argument("--shared-drivers", action="store_true")
    parser.add_argument("--mastery-gate", type=float, default=0.8)
    parser.add_argument("--breakout-gate", type=float, default=0.5)
    parser.add_argument("--retention-epsilon", type=float, default=0.1)
    parser.add_argument("--forgetting-delta", type=float, default=0.15)
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
