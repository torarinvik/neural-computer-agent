"""Fragment bank v1 (roadmap R2): shared fragments across family variants.

A fixed inventory of opaque fragments (token sets) is trained jointly
across composigrid variants through the shared-driver plant. Per variant,
a discrete selector picks k fragments (Plackett-Luce sampling, REINFORCE on
outcomes — selection stays discrete, execution neural); chosen fragment
tokens are concatenated into the controller's event window
(skill-as-context). Ignorance objectives (withheld bank, random decoy)
anchor fragment content causally.

Probe metrics (the strengths/weaknesses readout):
* learning speed  — per-variant mastery curves under interleaved training;
* sharing        — allocation overlap (Jaccard of selected fragment sets)
  versus component overlap between variants;
* composition    — holdout variants: adapt selector ONLY (plant and
  fragments frozen) and compare against the same adaptation over a
  random frozen bank (causal control) and against zero-shot transfer.

Measurement harness: no promotion gates beyond zero replay.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.games_amodal.game_family import (
    FamilyConfig,
    FamilyVerifier,
)
from experiments.games_amodal.shared_controller import (
    SHARED_SCREEN_CHANNELS,
    SharedControllerAgent,
    pad_channels,
    trainable_parameters,
)
from experiments.games_amodal.skill_externalization import (
    artifact_events,
    ignorance_loss,
)
from neural_computer import ControllerFeedback, KeypressDecoder


def twins_suite() -> tuple[list[FamilyConfig], list[FamilyConfig]]:
    """The minimal decisive bank test: two ambiguous twins, nothing else."""

    train = [
        FamilyConfig(choice=1, name="choiceA"),
        FamilyConfig(choice=1, inverted=True, name="choiceB"),
    ]
    holdout: list[FamilyConfig] = []
    return train, holdout


def micro_suite() -> tuple[list[FamilyConfig], list[FamilyConfig]]:
    """Tiny crude mini-games: four singles, two training pairs, one holdout."""

    train = [
        FamilyConfig(forage=1, name="forageA"),
        FamilyConfig(forage=1, inverted=True, name="forageB"),
        FamilyConfig(collect=1, name="collect1"),
        FamilyConfig(avoid=1, name="avoid1"),
        FamilyConfig(forage=1, avoid=1, name="forageA+avoid1"),
        FamilyConfig(collect=1, avoid=1, name="collect1+avoid1"),
    ]
    holdout = [
        FamilyConfig(forage=1, avoid=1, inverted=True, name="forageB+avoid1"),
    ]
    return train, holdout


def rollout_family(
    agent: SharedControllerAgent,
    config: FamilyConfig,
    fragments: torch.Tensor | None,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    sample: bool,
    gamma: float,
) -> dict[str, torch.Tensor | None]:
    verifier = FamilyVerifier(config, batch_size=batch_size, seed=seed)
    verifier.reset(seed=seed)
    controller = agent.controller
    state = controller.initial_state(batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(batch_size, controller.feedback_width),
        reward=torch.zeros(batch_size),
        propensity=torch.ones(batch_size),
        has_feedback=torch.zeros(batch_size),
    )
    decoder: KeypressDecoder = agent.runtime.output_bus.decoders["keypress"]
    rewards, log_props, masks, logits_trace = [], [], [], []
    alive = torch.ones(batch_size, dtype=torch.bool)
    for _ in range(steps):
        if not bool(alive.any()):
            break
        masks.append(alive.float())
        observation = pad_channels(
            verifier.observation(), SHARED_SCREEN_CHANNELS
        )
        events = [agent.runtime.encoders["screen"](observation)]
        if fragments is not None:
            events.extend(
                artifact_events(fragments.reshape(-1, fragments.shape[-1]), batch_size)
            )
        output, state = agent.runtime.step_events(events, state, feedback)
        logits = output.decoded["keypress"]
        decision = decoder.decide_from_logits(logits, sample=sample)
        logits_trace.append(logits)
        log_props.append(decision.propensity.clamp_min(1e-8).log())
        outcome = verifier.step(decision.key_index)
        rewards.append(outcome.reward)
        alive = outcome.alive
        feedback = ControllerFeedback(
            action=agent.feedback_encoders["keypress"](decision.key_index),
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
    props = None
    if sample:
        advantage = returns.detach()
        advantage = advantage - (
            (advantage * mask_matrix).sum() / mask_matrix.sum().clamp_min(1.0)
        )
        props = torch.stack(log_props, dim=1)
    return {
        "total_reward": reward_matrix.sum(dim=1),
        "advantage": advantage,
        "log_propensity": props,
        "mask": mask_matrix,
        "logits": torch.stack(logits_trace, dim=1),
    }


def sample_selection(
    logits: torch.Tensor, k: int, *, greedy: bool
) -> tuple[list[int], torch.Tensor]:
    """Sample k distinct fragments (Plackett-Luce) with total log-prob."""

    chosen: list[int] = []
    log_prob = torch.zeros(())
    available = torch.ones_like(logits, dtype=torch.bool)
    for _ in range(k):
        masked = logits.masked_fill(~available, float("-inf"))
        probs = F.softmax(masked, dim=-1)
        if greedy:
            index = int(masked.argmax())
        else:
            index = int(torch.multinomial(probs, 1))
        log_prob = log_prob + probs[index].clamp_min(1e-8).log()
        chosen.append(index)
        available[index] = False
    return chosen, log_prob


class FragmentBank(torch.nn.Module):
    """Inventory of opaque fragments plus per-variant selection logits."""

    def __init__(
        self,
        *,
        fragments: int,
        tokens_per_fragment: int,
        width: int,
        variants: list[str],
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        # Fragment tokens share the event window with screen events, whose
        # tanh payloads have norm ~sqrt(width)/2. Tokens initialised much
        # smaller are invisible to the controller and the plant learns to
        # ignore the bank entirely (probe 7). Match the scale.
        self.tokens = torch.nn.Parameter(
            torch.randn(fragments, tokens_per_fragment, width) * init_scale
        )
        self.selection_logits = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.zeros(fragments))
                for name in variants
            }
        )

    def register_variant(self, name: str) -> None:
        if name not in self.selection_logits:
            self.selection_logits[name] = torch.nn.Parameter(
                torch.zeros(self.tokens.shape[0])
            )

    def fetch(self, indices: list[int]) -> torch.Tensor:
        return self.tokens[indices]


def has_positive_source(config: FamilyConfig) -> bool:
    if config.forage or config.choice:
        return True
    if config.inverted:
        return False
    return bool(config.collect or config.intercept or config.navigate)


def mastery(
    summary: dict[str, torch.Tensor | None], config: FamilyConfig
) -> float:
    """Positive-source variants: earned reward. Avoid-only: survival.

    A purely negative variant has no achievable positive total, so mastery
    is surviving the full lifetime with zero loss.
    """

    if has_positive_source(config):
        return float((summary["total_reward"] > 0).float().mean())
    survived = summary["mask"][:, -1] * (summary["total_reward"] >= 0).float()
    return float(survived.mean())


def train_bank(
    agent: SharedControllerAgent,
    bank: FragmentBank,
    variants: list[FamilyConfig],
    *,
    args: argparse.Namespace,
    train_plant: bool,
    train_fragments: bool,
    seed_offset: int = 0,
) -> list[dict[str, float]]:
    parameters: list[torch.nn.Parameter] = list(
        bank.selection_logits.parameters()
    )
    if train_fragments:
        parameters.append(bank.tokens)
    if train_plant:
        parameters.extend(
            trainable_parameters(
                [agent.controller, *agent.game_modules(agent.games[0])]
            )
        )
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    baseline: dict[str, float] = {}
    history: list[dict[str, float]] = []
    for update in range(args.updates):
        config = variants[update % len(variants)]
        logits = bank.selection_logits[config.name]
        chosen, selection_log_prob = sample_selection(
            logits, args.fragments_per_variant, greedy=False
        )
        fragments = bank.fetch(chosen)
        summary = rollout_family(
            agent,
            config,
            fragments,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + seed_offset + update,
            sample=True,
            gamma=args.gamma,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        policy_terms = advantage * summary["log_propensity"] * summary["mask"]
        policy_loss = -policy_terms.sum() / policy_terms.shape[0]
        outcome_score = mastery(summary, config)
        previous = baseline.get(config.name, 0.0)
        baseline[config.name] = 0.9 * previous + 0.1 * outcome_score
        selection_loss = -(outcome_score - previous) * selection_log_prob
        loss = policy_loss + selection_loss
        if train_plant and update % args.ignorance_every == 0:
            withheld = rollout_family(
                agent,
                config,
                None,
                batch_size=args.batch_size,
                steps=max(8, args.steps // 4),
                seed=args.seed + 500_000 + update,
                sample=True,
                gamma=args.gamma,
            )
            decoy = torch.randn_like(fragments)
            decoy = decoy * (
                fragments.detach().norm() / decoy.norm().clamp_min(1e-12)
            )
            decoy_rollout = rollout_family(
                agent,
                config,
                decoy,
                batch_size=args.batch_size,
                steps=max(8, args.steps // 4),
                seed=args.seed + 700_000 + update,
                sample=True,
                gamma=args.gamma,
            )
            loss = loss + args.ignorance_weight * (
                ignorance_loss(withheld["logits"], withheld["mask"])
                + ignorance_loss(decoy_rollout["logits"], decoy_rollout["mask"])
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": float(update + 1),
                "variant": float(update % len(variants)),
                "mastery": outcome_score,
                "replayed_examples": 0.0,
            }
        )
    return history


def evaluate_variant(
    agent: SharedControllerAgent,
    bank: FragmentBank | None,
    config: FamilyConfig,
    *,
    args: argparse.Namespace,
    fragments_override: torch.Tensor | None = None,
) -> float:
    scores = []
    for index in range(args.eval_seeds):
        seed = args.seed + 10_000 + index
        fragments = fragments_override
        if fragments is None and bank is not None:
            chosen, _ = sample_selection(
                bank.selection_logits[config.name].detach(),
                args.fragments_per_variant,
                greedy=True,
            )
            fragments = bank.fetch(chosen).detach()
        with torch.no_grad():
            summary = rollout_family(
                agent,
                config,
                fragments,
                batch_size=args.batch_size,
                steps=args.steps,
                seed=seed,
                sample=False,
                gamma=args.gamma,
            )
        scores.append(mastery(summary, config))
    return float(torch.tensor(scores).mean())


def overlap_report(
    bank: FragmentBank, variants: list[FamilyConfig], k: int
) -> dict[str, object]:
    selections = {}
    for config in variants:
        chosen, _ = sample_selection(
            bank.selection_logits[config.name].detach(), k, greedy=True
        )
        selections[config.name] = set(chosen)
    pairs = {}
    for left, right in combinations(variants, 2):
        shared_components = len(
            set(left.active()) & set(right.active())
        )
        jaccard = len(
            selections[left.name] & selections[right.name]
        ) / len(selections[left.name] | selections[right.name])
        pairs[f"{left.name}|{right.name}"] = {
            "shared_components": shared_components,
            "fragment_jaccard": jaccard,
        }
    return {
        "selections": {name: sorted(s) for name, s in selections.items()},
        "pairs": pairs,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    suites = {"micro": micro_suite, "twins": twins_suite}
    train_variants, holdout_variants = suites[args.suite]()
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
        fragments=args.fragments,
        tokens_per_fragment=args.tokens_per_fragment,
        width=args.event_width,
        variants=[v.name for v in train_variants],
        init_scale=args.fragment_init_scale,
    )
    if args.suite == "twins":
        singles = [train_variants[0]]
    else:
        singles = [v for v in train_variants if len(v.active()) == 1]
    warm_history = train_bank(
        agent,
        bank,
        singles,
        args=argparse.Namespace(
            **{**vars(args), "updates": args.warm_updates}
        ),
        train_plant=True,
        train_fragments=True,
    )
    history = warm_history + train_bank(
        agent,
        bank,
        train_variants,
        args=args,
        train_plant=True,
        train_fragments=True,
        seed_offset=250_000,
    )
    train_scores = {
        v.name: evaluate_variant(agent, bank, v, args=args)
        for v in train_variants
    }
    withheld_scores = {
        v.name: evaluate_variant(
            agent, None, v, args=args, fragments_override=None
        )
        for v in train_variants[:2]
    }
    decoy = torch.randn_like(bank.tokens[:2].detach())
    decoy = decoy * (
        bank.tokens.detach().norm() / decoy.norm().clamp_min(1e-12)
    )
    decoy_scores = {
        v.name: evaluate_variant(
            agent, None, v, args=args, fragments_override=decoy
        )
        for v in train_variants[:2]
    }
    cross_scores = {}
    if len(train_variants) >= 2:
        for v, other in (
            (train_variants[0], train_variants[1]),
            (train_variants[1], train_variants[0]),
        ):
            chosen, _ = sample_selection(
                bank.selection_logits[other.name].detach(),
                args.fragments_per_variant,
                greedy=True,
            )
            cross_scores[v.name] = evaluate_variant(
                agent,
                None,
                v,
                args=args,
                fragments_override=bank.fetch(chosen).detach(),
            )

    holdout_report = {}
    for config in holdout_variants:
        bank.register_variant(config.name)
        zero_shot = evaluate_variant(agent, bank, config, args=args)
        adapt_history = train_bank(
            agent,
            bank,
            [config],
            args=argparse.Namespace(
                **{**vars(args), "updates": args.adapt_updates}
            ),
            train_plant=False,
            train_fragments=False,
            seed_offset=900_000,
        )
        adapted = evaluate_variant(agent, bank, config, args=args)
        random_bank = FragmentBank(
            fragments=args.fragments,
            tokens_per_fragment=args.tokens_per_fragment,
            width=args.event_width,
            variants=[config.name],
        )
        with torch.no_grad():
            random_bank.tokens.mul_(
                bank.tokens.detach().norm()
                / random_bank.tokens.norm().clamp_min(1e-12)
            )
        train_bank(
            agent,
            random_bank,
            [config],
            args=argparse.Namespace(
                **{**vars(args), "updates": args.adapt_updates}
            ),
            train_plant=False,
            train_fragments=False,
            seed_offset=900_000,
        )
        random_adapted = evaluate_variant(
            agent, random_bank, config, args=args
        )
        holdout_report[config.name] = {
            "zero_shot": zero_shot,
            "adapted_selector_only": adapted,
            "adapted_over_random_bank": random_adapted,
            "adaptation_curve_tail": [
                entry["mastery"] for entry in adapt_history[-5:]
            ],
        }

    return {
        "seed": args.seed,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "report_out"
        },
        "train_mastery": train_scores,
        "withheld_bank_mastery": withheld_scores,
        "decoy_bank_mastery": decoy_scores,
        "cross_fragment_mastery": cross_scores,
        "overlap": overlap_report(
            bank, train_variants, args.fragments_per_variant
        ),
        "holdout": holdout_report,
        "no_replay": all(
            entry["replayed_examples"] == 0.0 for entry in history
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", type=str, default="micro", choices=["micro", "twins"])
    parser.add_argument("--updates", type=int, default=900)
    parser.add_argument("--warm-updates", type=int, default=600)
    parser.add_argument("--adapt-updates", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--intent-width", type=int, default=32)
    parser.add_argument("--feedback-width", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--fragments", type=int, default=6)
    parser.add_argument("--tokens-per-fragment", type=int, default=2)
    parser.add_argument("--fragments-per-variant", type=int, default=2)
    parser.add_argument("--fragment-init-scale", type=float, default=1.0)
    parser.add_argument("--ignorance-every", type=int, default=4)
    parser.add_argument("--ignorance-weight", type=float, default=1.0)
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
