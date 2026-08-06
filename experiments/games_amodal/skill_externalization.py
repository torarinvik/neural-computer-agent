"""Phase 1 of skill externalization: the double dissociation, two skills.

Each game's skill is stored as a bank artifact — a small set of opaque
tokens — loaded into the controller's event window alongside the game
events (skill-as-context in the sketchpad).  Both skills train through one
core under a dissociation objective:

* with the correct artifact fetched, outcome-only policy gradients maximize
  play;
* with the bank withheld, and separately with a fresh same-norm random
  decoy artifact, an ignorance loss pushes the policy toward uniform.

The decoy term closes the presence-cue shortcut found by the rejected v1
(one skill, no decoy training: any same-norm noise switched the skill on).
With two skills through one core, the artifact's content must select which
program runs.

Audited conditions per game on identical evaluation seeds:

* ``fetched``         -- the game's own artifact (must reach mastery),
* ``withheld``        -- no artifact events (must collapse toward chance),
* ``random_artifact`` -- same-norm noise (content causal, not presence),
* ``cross_artifact``  -- the *other* game's artifact (must not run this
  game's program),
* ``shuffled``        -- reward-shuffled twin (acquisition causal null).

Hard gates: all of the above on both games plus the size audit — controller
and peripheral parameter counts unchanged, all growth in the bank — and
zero replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    trainable_parameters,
)
from neural_computer import AmodalEvent, ControllerFeedback, KeypressDecoder


def artifact_events(
    artifact: torch.Tensor, batch_size: int
) -> list[AmodalEvent]:
    """Present bank tokens as opaque events for the sketchpad."""

    return [
        AmodalEvent(payload=token.unsqueeze(0).expand(batch_size, -1))
        for token in artifact
    ]


def rollout_with_artifact(
    agent: SharedControllerAgent,
    game: str,
    artifact: torch.Tensor | None,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    sample: bool,
    gamma: float,
) -> dict[str, torch.Tensor | None]:
    """Play fresh lifetimes with optional bank tokens in the event window."""

    from experiments.games_amodal.shared_controller import GAME_VERIFIERS

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
    logits_trace: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    alive = torch.ones(batch_size, dtype=torch.bool)
    for _ in range(steps):
        if not bool(alive.any()):
            break
        masks.append(alive.float())
        events = [agent.runtime.encoders[game](verifier.observation())]
        if artifact is not None:
            events.extend(artifact_events(artifact, batch_size))
        output, state = agent.runtime.step_events(events, state, feedback)
        logits = output.decoded[game]
        decision = decoder.decide_from_logits(logits, sample=sample)
        logits_trace.append(logits)
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
        state = state.detached() if sample else state
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
        "mask": mask_matrix,
        "logits": torch.stack(logits_trace, dim=1),
    }


def ignorance_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Push the policy toward uniform when the bank is withheld."""

    log_probs = F.log_softmax(logits, dim=-1)
    uniform = torch.full_like(log_probs, 1.0 / logits.shape[-1])
    per_step = F.kl_div(log_probs, uniform, reduction="none").sum(dim=-1)
    return (per_step * mask).sum() / mask.sum().clamp_min(1.0)


def train_externalized_skills(
    agent: SharedControllerAgent,
    artifacts: dict[str, torch.Tensor],
    *,
    updates: int,
    batch_size: int,
    steps: int,
    ignorance_steps: int,
    seed: int,
    gamma: float,
    learning_rate: float,
    ignorance_weight: float,
    shuffle_rewards: bool,
) -> list[dict[str, float]]:
    """Joint dissociation training over multiple skills through one core.

    Games alternate per update.  With the correct artifact fetched, policy
    gradients maximize play.  With the bank withheld, and separately with a
    fresh same-norm random artifact, an ignorance loss pushes the policy
    toward uniform — closing the presence-cue shortcut so the artifact's
    content must select the program.
    """

    games = sorted(artifacts)
    modules = [agent.controller]
    for game in games:
        modules.extend(agent.game_modules(game))
    trainable = trainable_parameters(modules) + [
        artifacts[game] for game in games
    ]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    history: list[dict[str, float]] = []
    for update in range(updates):
        game = games[update % len(games)]
        artifact = artifacts[game]
        summary = rollout_with_artifact(
            agent,
            game,
            artifact,
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
        skill_loss = -loss_terms.sum() / loss_terms.shape[0]
        withheld = rollout_with_artifact(
            agent,
            game,
            None,
            batch_size=batch_size,
            steps=ignorance_steps,
            seed=seed + 500_000 + update,
            sample=True,
            gamma=gamma,
        )
        null_loss = ignorance_loss(withheld["logits"], withheld["mask"])
        with torch.no_grad():
            decoy = torch.randn_like(artifact)
            decoy = decoy * (
                artifact.norm() / decoy.norm().clamp_min(1e-12)
            )
        decoy_rollout = rollout_with_artifact(
            agent,
            game,
            decoy,
            batch_size=batch_size,
            steps=ignorance_steps,
            seed=seed + 700_000 + update,
            sample=True,
            gamma=gamma,
        )
        decoy_loss = ignorance_loss(
            decoy_rollout["logits"], decoy_rollout["mask"]
        )
        other = games[(games.index(game) + 1) % len(games)]
        cross_rollout = rollout_with_artifact(
            agent,
            game,
            artifacts[other].detach(),
            batch_size=batch_size,
            steps=ignorance_steps,
            seed=seed + 900_000 + update,
            sample=True,
            gamma=gamma,
        )
        cross_loss = ignorance_loss(
            cross_rollout["logits"], cross_rollout["mask"]
        )
        loss = skill_loss + ignorance_weight * (
            null_loss + decoy_loss + cross_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "game": float(games.index(game)),
                "skill_loss": float(skill_loss.detach()),
                "ignorance_loss": float(null_loss.detach()),
                "decoy_loss": float(decoy_loss.detach()),
                "cross_loss": float(cross_loss.detach()),
                "mastery": float((summary["total_reward"] > 0).float().mean()),
                "replayed_examples": 0.0,
            }
        )
    return history


def evaluate_condition(
    agent: SharedControllerAgent,
    game: str,
    artifact: torch.Tensor | None,
    *,
    batch_size: int,
    steps: int,
    seeds: tuple[int, ...],
    gamma: float,
) -> dict[str, object]:
    masteries: list[float] = []
    for seed in seeds:
        with torch.no_grad():
            summary = rollout_with_artifact(
                agent,
                game,
                artifact,
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


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    build = {
        "event_width": args.event_width,
        "intention_width": args.intent_width,
        "feedback_width": args.feedback_width,
        "hidden": args.hidden,
        "event_window_capacity": args.artifact_tokens + 4,
    }
    eval_seeds = tuple(args.seed + 10_000 + index for index in range(args.eval_seeds))
    eval_common = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seeds": eval_seeds,
        "gamma": args.gamma,
    }
    games = ("snake", "pong")

    agent = SharedControllerAgent(**build)
    def counts(target: SharedControllerAgent) -> dict[str, int]:
        return {
            "controller": sum(p.numel() for p in target.controller.parameters()),
            "peripherals": sum(
                p.numel()
                for game in games
                for module in target.game_modules(game)
                for p in module.parameters()
            ),
        }

    parameter_counts_before = counts(agent)
    artifacts = {
        game: (torch.randn(args.artifact_tokens, args.event_width) * 0.1)
        .requires_grad_(True)
        for game in games
    }

    shuffled_agent = SharedControllerAgent(**build)
    shuffled_agent.load_state_dict(agent.state_dict())
    shuffled_artifacts = {
        game: artifacts[game].detach().clone().requires_grad_(True)
        for game in games
    }

    train_common = {
        "updates": args.updates,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "ignorance_steps": args.ignorance_steps,
        "seed": args.seed,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "ignorance_weight": args.ignorance_weight,
    }
    history = train_externalized_skills(
        agent, artifacts, shuffle_rewards=False, **train_common
    )
    shuffled_history = train_externalized_skills(
        shuffled_agent, shuffled_artifacts, shuffle_rewards=True, **train_common
    )

    frozen = {game: artifacts[game].detach() for game in games}
    conditions: dict[str, dict[str, object]] = {}
    for game in games:
        other = games[1] if game == games[0] else games[0]
        random_artifact = torch.randn_like(frozen[game])
        random_artifact = random_artifact * (
            frozen[game].norm() / random_artifact.norm().clamp_min(1e-12)
        )
        conditions[game] = {
            "fetched": evaluate_condition(
                agent, game, frozen[game], **eval_common
            ),
            "withheld": evaluate_condition(agent, game, None, **eval_common),
            "random_artifact": evaluate_condition(
                agent, game, random_artifact, **eval_common
            ),
            "cross_artifact": evaluate_condition(
                agent, game, frozen[other], **eval_common
            ),
            "shuffled": evaluate_condition(
                shuffled_agent,
                game,
                shuffled_artifacts[game].detach(),
                **eval_common,
            ),
        }
    parameter_counts_after = counts(agent)

    def mastery(game: str, condition: str) -> float:
        return float(conditions[game][condition]["mastery"])

    gates = {
        "skill_with_bank": all(
            mastery(game, "fetched") >= args.mastery_gate for game in games
        ),
        "ignorant_without_bank": all(
            mastery(game, "withheld") <= args.null_gate for game in games
        ),
        "artifact_content_causal": all(
            mastery(game, "random_artifact") <= args.null_gate for game in games
        ),
        "cross_artifact_fails": all(
            mastery(game, "cross_artifact") <= args.null_gate for game in games
        ),
        "shuffled_near_chance": all(
            mastery(game, "shuffled") <= args.null_gate for game in games
        ),
        "computing_components_fixed_size": parameter_counts_after
        == parameter_counts_before,
        "no_replay": all(
            entry["replayed_examples"] == 0.0
            for entry in history + shuffled_history
        ),
    }
    return {
        "seed": args.seed,
        "games": list(games),
        "config": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "ignorance_steps": args.ignorance_steps,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "ignorance_weight": args.ignorance_weight,
            "artifact_tokens": args.artifact_tokens,
            "event_width": args.event_width,
            "intent_width": args.intent_width,
            "feedback_width": args.feedback_width,
            "hidden": args.hidden,
            "eval_seeds": args.eval_seeds,
        },
        "bank": {
            game: {
                "artifact_parameters": int(frozen[game].numel()),
                "artifact_norm": float(frozen[game].norm()),
            }
            for game in games
        },
        "computing_parameters": parameter_counts_after,
        "conditions": conditions,
        "training_tail": history[-3:],
        "gates": gates,
        "promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--ignorance-steps", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ignorance-weight", type=float, default=1.0)
    parser.add_argument("--artifact-tokens", type=int, default=8)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--feedback-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--mastery-gate", type=float, default=0.8)
    parser.add_argument("--null-gate", type=float, default=0.15)
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
