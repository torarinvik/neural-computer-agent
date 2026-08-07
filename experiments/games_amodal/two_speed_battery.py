"""Two-speed battery (F24): consolidating core + fragment bank, assembled.

The battery harness co-trained everything through joint bank training and
measured the consequence: scheduling fixes retention (F24) but cannot fix
acquisition, because fragments only *elicit* computation already latent in
the plant — installing new computation is the plastic core's job
(convergent finding 6). The two promoted lines each did half the job:

* arbitrated consolidation (`ewc_ladder`) installs new computation in the
  core while protecting what earlier games taught it;
* the fragment bank (`fragment_bank`) stores the context that selects
  which competence runs.

This module runs them as one system. Games arrive one at a time. Each
game gets a protected acquisition phase — plant and that game's fragments
train together, under arbitrated penalties from every previously
consolidated game — and is then consolidated (diagonal Fisher + anchor)
so the next game cannot overwrite it. No replay: an acquired game is
never rolled out again for training, only for the final audit.

Gates: every game at or above its calibrated solo ceiling ratio; the
first game retained after the last; cross-fed fragments still inverting
behaviour (the specification signature); zero replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.games_amodal.fragment_bank import (
    FragmentBank,
    battery_suite,
    evaluate_detail,
    mastery,
    rollout_family,
)
from experiments.games_amodal.game_family import FamilyConfig
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    trainable_parameters,
)

# Calibrated solo ceilings (F22/F23 calibration, seed 69316). A bank claim
# is judged against what the plant achieves with the whole model to
# itself, never against 1.0.
SOLO_CEILINGS = {
    "choiceA": 1.000,
    "choiceB": 1.000,
    "dualAC": 1.000,
    "dualAD": 0.686,
    "dualBC": 0.720,
    "avoid1": 0.922,
    "forageA": 0.453,
    "forageB": 0.453,
    "collect1": 0.547,
    "intercept1": 0.313,
}


def plant_named_parameters(agent: SharedControllerAgent):
    """The persistent computing plant: controller plus the shared drivers."""

    named = [
        (f"controller.{name}", parameter)
        for name, parameter in agent.controller.named_parameters()
    ]
    for prefix, module in (
        ("encoder", agent.runtime.encoders["screen"]),
        ("decoder", agent.runtime.output_bus.decoders["keypress"]),
        ("feedback", agent.feedback_encoders["keypress"]),
    ):
        named.extend(
            (f"{prefix}.{name}", parameter)
            for name, parameter in module.named_parameters()
        )
    return named


def family_fisher(
    agent: SharedControllerAgent,
    config: FamilyConfig,
    fragments: torch.Tensor | None,
    named,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Unit-mean diagonal Fisher over the plant, for one banked game.

    Estimated with the game's own fragments loaded, so the protected
    directions are the ones that matter *when this context is fetched* —
    the quantity the bank will later re-create by handing the plant those
    same fragments.
    """

    fisher = {name: torch.zeros_like(parameter) for name, parameter in named}
    for batch in range(args.fisher_batches):
        summary = rollout_family(
            agent,
            config,
            fragments,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=seed + batch,
            sample=True,
            gamma=args.gamma,
            egocentric=args.egocentric,
        )
        log_likelihood = (
            summary["log_propensity"] * summary["mask"]
        ).sum() / summary["mask"].sum().clamp_min(1.0)
        agent.zero_grad(set_to_none=True)
        log_likelihood.backward(retain_graph=False)
        for name, parameter in named:
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach().square()
    agent.zero_grad(set_to_none=True)
    total = sum(tensor.sum() for tensor in fisher.values())
    count = sum(tensor.numel() for tensor in fisher.values())
    mean = (total / count).clamp_min(1e-12)
    return {name: tensor / mean for name, tensor in fisher.items()}


def acquire_game(
    agent: SharedControllerAgent,
    bank: FragmentBank,
    config: FamilyConfig,
    penalties: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    *,
    args: argparse.Namespace,
    seed_offset: int,
) -> list[dict[str, float]]:
    """One protected acquisition phase: plant + this game's fragments.

    Arbitrated consolidation (`a = F / (F + mu * G)`): parameters earlier
    games depend on stay protected; parameters this game demonstrably
    needs are released in proportion to its own demand. Task gradients are
    taken first and the penalty gradient added in closed form, so the
    demand estimate never sees penalty pressure.
    """

    named = plant_named_parameters(agent)
    indices = bank.oracle_indices(config.name, args.fragments_per_variant)
    trainable = list(
        trainable_parameters([agent.controller, *agent.game_modules(agent.games[0])])
    )
    trainable.append(bank.tokens)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    demand = {name: torch.zeros_like(parameter) for name, parameter in named}
    history: list[dict[str, float]] = []
    for update in range(args.updates_per_game):
        summary = rollout_family(
            agent,
            config,
            bank.fetch(indices),
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + seed_offset + update,
            sample=True,
            gamma=args.gamma,
            egocentric=args.egocentric,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        terms = advantage * summary["log_propensity"] * summary["mask"]
        loss = -terms.sum() / terms.shape[0]
        release = torch.zeros(())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if penalties:
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
                demand_mean = (
                    sum(t.sum() for t in demand.values())
                    / sum(t.numel() for t in demand.values())
                ).clamp_min(1e-12)
                for fisher, anchor in penalties:
                    for name, parameter in named:
                        attenuation = fisher[name] / (
                            fisher[name]
                            + args.arbitration_mu * demand[name] / demand_mean
                        ).clamp_min(1e-12)
                        coefficient = (
                            args.ewc_lambda * attenuation * fisher[name]
                        )
                        if parameter.grad is None:
                            parameter.grad = torch.zeros_like(parameter)
                        parameter.grad.add_(
                            coefficient * (parameter - anchor[name])
                        )
                        release = release + (1.0 - attenuation).mean()
                release = release / max(len(penalties) * len(named), 1)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "game": config.name,
                "mastery": mastery(summary, config),
                "release_fraction": float(release),
                "replayed_examples": 0.0,
            }
        )
    return history


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    train_variants, _ = battery_suite()
    if args.games:
        wanted = set(args.games.split(","))
        train_variants = [v for v in train_variants if v.name in wanted]
    agent = SharedControllerAgent(
        event_width=args.event_width,
        intention_width=args.intent_width,
        feedback_width=args.feedback_width,
        hidden=args.hidden,
        event_window_capacity=args.fragments_per_variant
        * args.tokens_per_fragment
        + 4,
        shared_drivers=True,
    )
    bank = FragmentBank(
        fragments=max(args.fragments, 2 * len(train_variants)),
        tokens_per_fragment=args.tokens_per_fragment,
        width=args.event_width,
        variants=[v.name for v in train_variants],
    )

    penalties: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
    history: list[dict[str, float]] = []
    acquisition: dict[str, float] = {}
    named = plant_named_parameters(agent)
    for position, config in enumerate(train_variants):
        history += acquire_game(
            agent,
            bank,
            config,
            penalties,
            args=args,
            seed_offset=position * 100_000,
        )
        # Score immediately after acquisition, before any later game can
        # disturb it: the difference against the final audit IS forgetting.
        acquisition[config.name] = evaluate_detail(
            agent, bank, config, args=args
        )["mastery"]
        fisher = family_fisher(
            agent,
            config,
            bank.fetch(
                bank.oracle_indices(config.name, args.fragments_per_variant)
            ).detach(),
            named,
            args=args,
            seed=args.seed + 900_000 + position * 1_000,
        )
        anchor = {
            name: parameter.detach().clone() for name, parameter in named
        }
        penalties.append((fisher, anchor))

    final = {
        v.name: evaluate_detail(agent, bank, v, args=args)
        for v in train_variants
    }
    withheld = {
        v.name: evaluate_detail(agent, None, v, args=args)["mastery"]
        for v in train_variants
    }
    cross = {}
    for index, target in enumerate(train_variants):
        source = train_variants[(index + 1) % len(train_variants)]
        chosen = bank.oracle_indices(source.name, args.fragments_per_variant)
        cross[f"{target.name}<-{source.name}"] = evaluate_detail(
            agent,
            None,
            target,
            args=args,
            fragments_override=bank.fetch(chosen).detach(),
        )["mastery"]

    ratios = {
        name: detail["mastery"] / SOLO_CEILINGS[name]
        for name, detail in final.items()
        if name in SOLO_CEILINGS
    }
    forgetting = {
        name: acquisition[name] - final[name]["mastery"] for name in final
    }
    return {
        "seed": args.seed,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "report_out"
        },
        "order": [v.name for v in train_variants],
        "acquisition_mastery": acquisition,
        "final_mastery": final,
        "solo_ratio": ratios,
        "forgetting": forgetting,
        "worst_ratio": min(ratios.values()) if ratios else 0.0,
        "worst_forgetting": max(forgetting.values()) if forgetting else 0.0,
        "withheld_bank_mastery": withheld,
        "cross_fragment_mastery": cross,
        "no_replay": all(
            entry["replayed_examples"] == 0.0 for entry in history
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--games", type=str, default="")
    parser.add_argument("--updates-per-game", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--feedback-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--fragments", type=int, default=24)
    parser.add_argument("--tokens-per-fragment", type=int, default=2)
    parser.add_argument("--fragments-per-variant", type=int, default=2)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    parser.add_argument("--arbitration-mu", type=float, default=3.0)
    parser.add_argument("--arbitration-decay", type=float, default=0.99)
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--egocentric", action="store_true")
    parser.add_argument("--eval-seeds", type=int, default=4)
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
