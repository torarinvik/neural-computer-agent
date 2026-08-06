"""EWC-consolidation rung: strict-setting continual core learning.

Before Snake's environment is left behind, the trainer estimates a diagonal
Fisher sensitivity map from fresh on-policy lifetimes: the squared gradients
of the controller's own log-propensities.  Carried forward are only two
parameter-shaped tensor maps — the sensitivity and the anchor weights — no
experiences, no per-game core state, and no later Snake environment access.

During Pong the core stays fully plastic; the trainer adds a quadratic
consolidation penalty pulling high-sensitivity parameters toward the anchor
while leaving low-sensitivity directions free.

Conditions on identical budgets and evaluation seeds:

* ``unprotected``    -- plastic core, no penalty (forgetting baseline),
* ``ewc_protected``  -- consolidation penalty on the Snake Fisher,
* ``permuted_fisher`` -- penalty with sensitivities shuffled across
  parameters (same distribution, wrong assignment; causal null),
* ``shuffled``       -- reward-shuffled Pong twin (acquisition null).

Hard gates: baseline forgets; the protected run retains Snake within epsilon
while acquiring Pong; the permuted sensitivity fails to rescue retention;
the shuffled twin stays near chance; zero replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    evaluate_game,
    rollout,
    set_trainable,
    trainable_parameters,
)


def controller_named_parameters(agent: SharedControllerAgent):
    return [
        (name, parameter)
        for name, parameter in agent.controller.named_parameters()
    ]


def estimate_diagonal_fisher(
    agent: SharedControllerAgent,
    game: str,
    *,
    batches: int,
    batch_size: int,
    steps: int,
    seed: int,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Squared on-policy log-propensity gradients from fresh lifetimes.

    Estimated while the game environment is still reachable; afterwards only
    this parameter-shaped map is carried.  Normalized to unit mean so the
    consolidation strength is comparable across runs.
    """

    named = controller_named_parameters(agent)
    fisher = {name: torch.zeros_like(parameter) for name, parameter in named}
    for batch in range(batches):
        summary = rollout(
            agent,
            game,
            batch_size=batch_size,
            steps=steps,
            seed=seed + batch,
            sample=True,
            gamma=gamma,
        )
        log_likelihood = (
            summary["log_propensity"] * summary["mask"]
        ).sum() / summary["mask"].sum().clamp_min(1.0)
        agent.zero_grad(set_to_none=True)
        log_likelihood.backward()
        for name, parameter in named:
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach().square()
    agent.zero_grad(set_to_none=True)
    total = sum(tensor.sum() for tensor in fisher.values())
    count = sum(tensor.numel() for tensor in fisher.values())
    mean = (total / count).clamp_min(1e-12)
    return {name: tensor / mean for name, tensor in fisher.items()}


def permuted_fisher_like(
    fisher: dict[str, torch.Tensor], *, seed: int
) -> dict[str, torch.Tensor]:
    """Shuffle all sensitivity values across every parameter position."""

    generator = torch.Generator().manual_seed(seed)
    flat = torch.cat([tensor.flatten() for tensor in fisher.values()])
    flat = flat[torch.randperm(flat.numel(), generator=generator)]
    permuted: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, tensor in fisher.items():
        permuted[name] = flat[cursor : cursor + tensor.numel()].reshape(
            tensor.shape
        )
        cursor += tensor.numel()
    return permuted


def train_pong_with_consolidation(
    agent: SharedControllerAgent,
    *,
    args: argparse.Namespace,
    fisher: dict[str, torch.Tensor] | None,
    anchor: dict[str, torch.Tensor] | None,
    shuffle_rewards: bool,
) -> list[dict[str, float]]:
    set_trainable([agent.controller], True)
    set_trainable(agent.game_modules("snake"), False)
    set_trainable(agent.game_modules("pong"), True)
    named = controller_named_parameters(agent)
    trainable = trainable_parameters(
        [agent.controller, *agent.game_modules("pong")]
    )
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    history: list[dict[str, float]] = []
    for update in range(args.updates):
        summary = rollout(
            agent,
            "pong",
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 50_000 + update,
            sample=True,
            gamma=args.gamma,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        if shuffle_rewards:
            advantage = advantage[torch.randperm(advantage.shape[0])]
        loss_terms = advantage * summary["log_propensity"] * summary["mask"]
        loss = -loss_terms.sum() / loss_terms.shape[0]
        penalty = torch.zeros(())
        if fisher is not None and anchor is not None:
            for name, parameter in named:
                penalty = penalty + (
                    fisher[name] * (parameter - anchor[name]).square()
                ).sum()
            loss = loss + 0.5 * args.ewc_lambda * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "loss": float(loss.detach()),
                "penalty": float(penalty.detach()),
                "mastery": float((summary["total_reward"] > 0).float().mean()),
                "replayed_examples": 0.0,
            }
        )
    return history


def curve_summary(history: list[dict[str, float]]) -> dict[str, float]:
    masteries = [entry["mastery"] for entry in history]
    return {
        "mean_training_mastery": float(sum(masteries) / max(len(masteries), 1)),
        "final_training_mastery": masteries[-1] if masteries else 0.0,
        "final_penalty": history[-1]["penalty"] if history else 0.0,
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
    trainable = trainable_parameters(
        [agent.controller, *agent.game_modules("snake")]
    )
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    phase_one: list[dict[str, float]] = []
    for update in range(args.updates):
        summary = rollout(
            agent,
            "snake",
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + update,
            sample=True,
            gamma=args.gamma,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        loss_terms = advantage * summary["log_propensity"] * summary["mask"]
        loss = -loss_terms.sum() / loss_terms.shape[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        phase_one.append(
            {
                "update": float(update + 1),
                "mastery": float((summary["total_reward"] > 0).float().mean()),
                "replayed_examples": 0.0,
            }
        )
    snake_before = evaluate_game(agent, "snake", **eval_common)

    fisher = estimate_diagonal_fisher(
        agent,
        "snake",
        batches=args.fisher_batches,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 30_000,
        gamma=args.gamma,
    )
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in controller_named_parameters(agent)
    }

    twins = {
        name: SharedControllerAgent(**build)
        for name in ("unprotected", "permuted_fisher", "shuffled")
    }
    for twin in twins.values():
        twin.load_state_dict(agent.state_dict())
    permuted = permuted_fisher_like(fisher, seed=args.seed + 90_000)

    histories = {
        "ewc_protected": train_pong_with_consolidation(
            agent, args=args, fisher=fisher, anchor=anchor, shuffle_rewards=False
        ),
        "unprotected": train_pong_with_consolidation(
            twins["unprotected"],
            args=args,
            fisher=None,
            anchor=None,
            shuffle_rewards=False,
        ),
        "permuted_fisher": train_pong_with_consolidation(
            twins["permuted_fisher"],
            args=args,
            fisher=permuted,
            anchor=anchor,
            shuffle_rewards=False,
        ),
        "shuffled": train_pong_with_consolidation(
            twins["shuffled"],
            args=args,
            fisher=None,
            anchor=None,
            shuffle_rewards=True,
        ),
    }
    agents = {"ewc_protected": agent, **twins}
    conditions = {
        name: {
            "snake_after": evaluate_game(agents[name], "snake", **eval_common),
            "pong": evaluate_game(agents[name], "pong", **eval_common),
            "curves": curve_summary(histories[name]),
        }
        for name in agents
    }

    snake_mastery = float(snake_before["mastery"])
    protected_snake = float(
        conditions["ewc_protected"]["snake_after"]["mastery"]
    )
    unprotected_snake = float(
        conditions["unprotected"]["snake_after"]["mastery"]
    )
    permuted_snake = float(
        conditions["permuted_fisher"]["snake_after"]["mastery"]
    )
    protected_pong = float(conditions["ewc_protected"]["pong"]["mastery"])
    gates = {
        "snake_acquired": snake_mastery >= args.mastery_gate,
        "baseline_forgets": unprotected_snake
        <= snake_mastery - args.forgetting_delta,
        "protected_retention": protected_snake
        >= snake_mastery - args.retention_epsilon,
        "pong_acquired": protected_pong >= args.mastery_gate,
        "permuted_fisher_fails": permuted_snake
        < snake_mastery - args.retention_epsilon,
        "shuffled_near_chance": float(conditions["shuffled"]["pong"]["mastery"])
        <= args.null_gate,
        "no_replay": all(
            entry["replayed_examples"] == 0.0
            for history in [phase_one, *histories.values()]
            for entry in history
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
            "fisher_batches": args.fisher_batches,
            "ewc_lambda": args.ewc_lambda,
            "retention_epsilon": args.retention_epsilon,
            "forgetting_delta": args.forgetting_delta,
        },
        "snake_before": snake_before,
        "conditions": conditions,
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
    parser.add_argument("--mastery-gate", type=float, default=0.8)
    parser.add_argument("--retention-epsilon", type=float, default=0.05)
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
