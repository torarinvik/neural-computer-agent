"""Protected-plasticity rung: a fully plastic core learns Pong without
forgetting Snake, via trainer-side gradient projection only.

Nothing is frozen and nothing game-specific is added to the controller.
During Snake acquisition the trainer accumulates a reference direction: the
sum of the controller's own late-phase update gradients.  During Pong
acquisition the controller stays trainable, and each update's component that
opposes the reference is removed before the optimizer step
(`project_gradient_against_reference`), with a post-step safeguard against
optimizer rotation (`project_parameter_update_against_reference`).  The
reference is one gradient-shaped tensor map — no experiences, trajectories,
or verifier state are stored or replayed.

Conditions on identical budgets and evaluation seeds:

* ``unprotected``  -- plastic core, no projection (forgetting baseline),
* ``protected``    -- plastic core, projection against the Snake reference,
* ``random_ref``   -- projection against a random same-norm direction (null),
* ``shuffled``     -- reward-shuffled Pong twin (acquisition causal null).

Hard gates: the baseline actually forgets; the protected run retains Snake
within epsilon while acquiring Pong; the random direction fails to rescue
retention; the shuffled twin stays near chance; zero replay.
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
from neural_computer import (
    accumulate_current_gradients,
    project_gradient_against_reference,
    project_parameter_update_against_reference,
    zero_gradient_map,
)


def controller_named_parameters(agent: SharedControllerAgent):
    return [
        (name, parameter)
        for name, parameter in agent.controller.named_parameters()
        if parameter.requires_grad
    ]


def random_reference_like(
    reference: dict[str, torch.Tensor], *, seed: int
) -> dict[str, torch.Tensor]:
    """A random direction with the same global norm as the real reference."""

    generator = torch.Generator().manual_seed(seed)
    random_map = {
        name: torch.randn(tensor.shape, generator=generator)
        for name, tensor in reference.items()
    }
    reference_norm = torch.sqrt(
        sum(tensor.square().sum() for tensor in reference.values())
    )
    random_norm = torch.sqrt(
        sum(tensor.square().sum() for tensor in random_map.values())
    )
    scale = reference_norm / random_norm.clamp_min(1e-12)
    return {name: tensor * scale for name, tensor in random_map.items()}


def train_phase(
    agent: SharedControllerAgent,
    game: str,
    *,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    gamma: float,
    learning_rate: float,
    shuffle_rewards: bool = False,
    reference: dict[str, torch.Tensor] | None = None,
    projection_strength: float = 1.0,
    accumulate_reference_from: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor] | None]:
    """One outcome-only acquisition phase with optional protection.

    ``reference`` enables gradient and post-step projection on the controller
    parameters.  ``accumulate_reference_from`` starts summing the
    controller's own gradients from that update index, returning the
    accumulated map for use as a later phase's protected direction.
    """

    trainable = trainable_parameters(
        [agent.controller, *agent.game_modules(game)]
    )
    named = controller_named_parameters(agent)
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    accumulator = (
        zero_gradient_map(named) if accumulate_reference_from is not None else None
    )
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
        gradient_projected = False
        if reference is not None:
            gradient_projected, _, _ = project_gradient_against_reference(
                named, reference, projection_strength
            )
        if (
            accumulator is not None
            and accumulate_reference_from is not None
            and update >= accumulate_reference_from
        ):
            accumulate_current_gradients(named, accumulator)
        before = (
            {name: parameter.detach().clone() for name, parameter in named}
            if reference is not None
            else None
        )
        optimizer.step()
        update_projected = False
        if reference is not None and before is not None:
            update_projected, _, _ = project_parameter_update_against_reference(
                named, before, reference, projection_strength
            )
        history.append(
            {
                "update": float(update + 1),
                "loss": float(loss.detach()),
                "mastery": float((summary["total_reward"] > 0).float().mean()),
                "gradient_projected": float(gradient_projected),
                "update_projected": float(update_projected),
                "replayed_examples": 0.0,
            }
        )
    return history, accumulator


def acquire_pong(
    agent: SharedControllerAgent,
    *,
    args: argparse.Namespace,
    reference: dict[str, torch.Tensor] | None,
    shuffle_rewards: bool,
) -> list[dict[str, float]]:
    """Phase two: plastic core, frozen Snake peripherals, trainable Pong."""

    set_trainable([agent.controller], True)
    set_trainable(agent.game_modules("snake"), False)
    set_trainable(agent.game_modules("pong"), True)
    history, _ = train_phase(
        agent,
        "pong",
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 50_000,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        shuffle_rewards=shuffle_rewards,
        reference=reference,
        projection_strength=args.projection_strength,
    )
    return history


def curve_summary(history: list[dict[str, float]]) -> dict[str, float]:
    masteries = [entry["mastery"] for entry in history]
    return {
        "mean_training_mastery": float(sum(masteries) / max(len(masteries), 1)),
        "final_training_mastery": masteries[-1] if masteries else 0.0,
        "gradient_projected_fraction": float(
            sum(entry["gradient_projected"] for entry in history)
            / max(len(history), 1)
        ),
        "update_projected_fraction": float(
            sum(entry["update_projected"] for entry in history)
            / max(len(history), 1)
        ),
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
    phase_one, reference = train_phase(
        agent,
        "snake",
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        accumulate_reference_from=args.updates - args.reference_window,
    )
    assert reference is not None
    snake_before = evaluate_game(agent, "snake", **eval_common)

    twins = {
        name: SharedControllerAgent(**build) for name in
        ("unprotected", "random_ref", "shuffled")
    }
    for twin in twins.values():
        twin.load_state_dict(agent.state_dict())
    random_reference = random_reference_like(
        reference, seed=args.seed + 90_000
    )

    histories = {
        "protected": acquire_pong(
            agent, args=args, reference=reference, shuffle_rewards=False
        ),
        "unprotected": acquire_pong(
            twins["unprotected"], args=args, reference=None, shuffle_rewards=False
        ),
        "random_ref": acquire_pong(
            twins["random_ref"],
            args=args,
            reference=random_reference,
            shuffle_rewards=False,
        ),
        "shuffled": acquire_pong(
            twins["shuffled"], args=args, reference=None, shuffle_rewards=True
        ),
    }
    agents = {"protected": agent, **twins}
    conditions = {
        name: {
            "snake_after": evaluate_game(agents[name], "snake", **eval_common),
            "pong": evaluate_game(agents[name], "pong", **eval_common),
            "curves": curve_summary(histories[name]),
        }
        for name in agents
    }

    snake_mastery = float(snake_before["mastery"])
    protected_snake = float(conditions["protected"]["snake_after"]["mastery"])
    unprotected_snake = float(conditions["unprotected"]["snake_after"]["mastery"])
    random_ref_snake = float(conditions["random_ref"]["snake_after"]["mastery"])
    protected_pong = float(conditions["protected"]["pong"]["mastery"])
    gates = {
        "snake_acquired": snake_mastery >= args.mastery_gate,
        "baseline_forgets": unprotected_snake
        <= snake_mastery - args.forgetting_delta,
        "protected_retention": protected_snake
        >= snake_mastery - args.retention_epsilon,
        "pong_acquired": protected_pong >= args.mastery_gate,
        "random_direction_fails": random_ref_snake
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
            "reference_window": args.reference_window,
            "projection_strength": args.projection_strength,
            "retention_epsilon": args.retention_epsilon,
            "forgetting_delta": args.forgetting_delta,
        },
        "snake_before": snake_before,
        "snake_curve": {
            "mean_training_mastery": float(
                sum(entry["mastery"] for entry in phase_one) / len(phase_one)
            ),
            "final_training_mastery": phase_one[-1]["mastery"],
        },
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
    parser.add_argument("--reference-window", type=int, default=200)
    parser.add_argument("--projection-strength", type=float, default=1.0)
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
