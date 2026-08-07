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
    SHARED_SCREEN_CHANNELS,
    SharedControllerAgent,
    trainable_parameters,
)
from experiments.games_amodal.skill_externalization import ignorance_loss
from experiments.games_amodal.train import GridEventEncoder

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


def conflict_groups(
    variants: list[FamilyConfig],
) -> list[list[FamilyConfig]]:
    """Group contexts that contradict each other on identical observations.

    Sequential admission is right for contexts that differ in what they
    SHOW (F18), and wrong for contexts that differ only in what they MEAN.
    Twins render identically and demand opposite actions, so admitting the
    second after the first has been consolidated asks the plant to invert
    a rule its penalty is holding still: measured as choiceA 1.00 beside
    choiceB 0.17, forageA 0.90 beside forageB 0.00, on both seeds and with
    or without ignorance pressure. The promoted twins rung never hit this
    because it trained twins JOINTLY, letting the bank carry the
    difference from the start.

    So a phase is a conflict group, not a game: contexts sharing a
    component signature acquire together (bank carries the difference),
    and consolidation protects BETWEEN groups (where sequencing works).
    """

    groups: dict[tuple, list[FamilyConfig]] = {}
    for config in variants:
        signature = config.active()
        groups.setdefault(signature, []).append(config)
    return list(groups.values())


def plant_named_parameters(
    agent: SharedControllerAgent, *, include_screen: bool = True
):
    """The persistent computing plant: controller plus the shared drivers.

    With per-game screen encoders (F29) the screen frontend is no longer
    cross-game infrastructure, so it leaves the protected set: a per-game
    encoder cannot be forgotten by another game because no other game
    ever touches it.
    """

    named = [
        (f"controller.{name}", parameter)
        for name, parameter in agent.controller.named_parameters()
    ]
    modules = [
        ("decoder", agent.runtime.output_bus.decoders["keypress"]),
        ("feedback", agent.feedback_encoders["keypress"]),
    ]
    if include_screen:
        modules.insert(0, ("encoder", agent.runtime.encoders["screen"]))
    for prefix, module in modules:
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
    encoder=None,
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
            encoder=encoder,
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


def acquire_group(
    agent: SharedControllerAgent,
    bank: FragmentBank,
    group: list[FamilyConfig],
    penalties: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    *,
    args: argparse.Namespace,
    seed_offset: int,
    screens: dict | None = None,
) -> list[dict[str, float]]:
    """One protected acquisition phase: plant + this game's fragments.

    Arbitrated consolidation (`a = F / (F + mu * G)`): parameters earlier
    games depend on stay protected; parameters this game demonstrably
    needs are released in proportion to its own demand. Task gradients are
    taken first and the penalty gradient added in closed form, so the
    demand estimate never sees penalty pressure.
    """

    named = plant_named_parameters(agent, include_screen=screens is None)
    indices = {
        c.name: bank.oracle_indices(c.name, args.fragments_per_variant)
        for c in group
    }
    recent = {c.name: 0.0 for c in group}
    trainable = list(
        trainable_parameters([agent.controller, *agent.game_modules(agent.games[0])])
    )
    trainable.append(bank.tokens)
    if screens is not None:
        for config in group:
            trainable.extend(screens[config.name].parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    demand = {name: torch.zeros_like(parameter) for name, parameter in named}
    history: list[dict[str, float]] = []
    for update in range(args.updates_per_game * len(group)):
        # Laggard-preferential sampling with a uniform floor (F10 + F23):
        # inside a conflict group no context may capture the plant, and
        # every context keeps a maintenance ration.
        if len(group) > 1:
            weights = torch.tensor([-recent[c.name] for c in group]) / 0.25
            probs = torch.softmax(weights, dim=-1)
            probs = 0.5 * probs + 0.5 / len(group)
            config = group[int(torch.multinomial(probs, 1))]
        else:
            config = group[0]
        summary = rollout_family(
            agent,
            config,
            bank.fetch(indices[config.name]),
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + seed_offset + update,
            sample=True,
            gamma=args.gamma,
            egocentric=args.egocentric,
            encoder=None if screens is None else screens[config.name],
        )
        advantage = summary["advantage"]
        assert advantage is not None
        terms = advantage * summary["log_propensity"] * summary["mask"]
        loss = -terms.sum() / terms.shape[0]
        # F9/F11: without ignorance pressure the plant keeps the FIRST
        # context's rule as a weight-level default. Consolidation then
        # locks that default in, and a later contradictory twin cannot
        # acquire at all -- measured as choiceA 1.00 / choiceB 0.06. Push
        # the bank-free policy toward uniform so the rule has to live in
        # the fragments, which is the architecture's storage rule stated
        # as a training objective.
        if args.ignorance_weight > 0.0 and update % args.ignorance_every == 0:
            withheld = rollout_family(
                agent,
                config,
                None,
                batch_size=args.batch_size,
                steps=max(8, args.steps // 4),
                seed=args.seed + seed_offset + 500_000 + update,
                sample=True,
                gamma=args.gamma,
                egocentric=args.egocentric,
                encoder=None if screens is None else screens[config.name],
            )
            loss = loss + args.ignorance_weight * ignorance_loss(
                withheld["logits"], withheld["mask"]
            )
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
        score = mastery(summary, config)
        recent[config.name] = 0.9 * recent[config.name] + 0.1 * score
        history.append(
            {
                "update": float(update + 1),
                "game": config.name,
                "mastery": score,
                "release_fraction": float(release),
                "replayed_examples": 0.0,
            }
        )
    return history


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    # Acquisition fetches each game's fragments by `oracle_indices`, so the
    # audit must fetch the same way. Without this, `evaluate_detail` falls
    # back to the untrained selection logits and scores every game against
    # fragments it never trained with -- which reads exactly like a failure
    # to learn, and cost one full battery run to diagnose.
    args.oracle_selection = True
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
    screens = None
    if getattr(args, "per_game_encoders", False):
        # N encoders, one per game (the amodal design). Nothing crosses
        # games at the frontend, so encoder choices stop coupling (F29).
        screens = {
            v.name: GridEventEncoder(
                channels=SHARED_SCREEN_CHANNELS,
                height=8,
                width=8,
                event_width=args.event_width,
            )
            for v in train_variants
        }
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
    groups = conflict_groups(train_variants)
    for position, group in enumerate(groups):
        history += acquire_group(
            agent,
            bank,
            group,
            penalties,
            args=args,
            seed_offset=position * 100_000,
            screens=screens,
        )
        # Score immediately after acquisition, before any later group can
        # disturb it: the difference against the final audit IS forgetting.
        # One consolidation per member, so a group of contradictory twins
        # protects both policies rather than an average of them.
        anchor = {
            name: parameter.detach().clone() for name, parameter in named
        }
        for member, config in enumerate(group):
            acquisition[config.name] = evaluate_detail(
                agent, bank, config, args=args,
                encoder=None if screens is None else screens[config.name],
            )["mastery"]
            fisher = family_fisher(
                agent,
                config,
                bank.fetch(
                    bank.oracle_indices(
                        config.name, args.fragments_per_variant
                    )
                ).detach(),
                named,
                args=args,
                seed=args.seed + 900_000 + position * 1_000 + member * 100,
                encoder=None if screens is None else screens[config.name],
            )
            penalties.append((fisher, anchor))

    def screen_for(name):
        return None if screens is None else screens[name]

    final = {
        v.name: evaluate_detail(
            agent, bank, v, args=args, encoder=screen_for(v.name)
        )
        for v in train_variants
    }
    withheld = {
        v.name: evaluate_detail(
            agent, None, v, args=args, encoder=screen_for(v.name)
        )["mastery"]
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
            encoder=screen_for(target.name),
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
        "conflict_groups": [[c.name for c in g] for g in groups],
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
    parser.add_argument("--per-game-encoders", action="store_true")
    parser.add_argument("--ignorance-weight", type=float, default=0.5)
    parser.add_argument("--ignorance-every", type=int, default=3)
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
