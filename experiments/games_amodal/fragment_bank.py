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
from itertools import combinations, permutations, product
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.games_amodal.game_family import (
    FamilyConfig,
    FamilyVerifier,
    egocentric_crop,
    egocentric_view,
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


def dual_suite() -> tuple[list[FamilyConfig], list[FamilyConfig]]:
    """Factorial contexts: two independent binary rules, genuine sharing.

    Weakness 11: the twins rung proved the bank can DISTINGUISH contexts,
    never that it can REUSE structure across them, because contradictory
    twins share nothing by construction. Here a context is a product of
    two bits, so `dualAC` and `dualAD` share the A/B rule while differing
    on C/D. Three contexts train; the fourth is a novel recombination of
    rules that were each learned elsewhere, so a factorized bank should
    compose it from existing fragments without new content.
    """

    train = [
        FamilyConfig(dual=1, name="dualAC"),
        FamilyConfig(dual=1, inverted2=True, name="dualAD"),
        FamilyConfig(dual=1, inverted=True, name="dualBC"),
    ]
    holdout = [
        FamilyConfig(dual=1, inverted=True, inverted2=True, name="dualBD"),
    ]
    return train, holdout


def compose_suite() -> tuple[list[FamilyConfig], list[FamilyConfig]]:
    """Nine rule pairings from six rules: factorising becomes economical.

    F27 diagnosed the composition failure as a lack of pressure, not a
    lack of mechanism: with two binary axes there are only four pairings,
    and memorising four whole programs is no more expensive than
    factorising into two rules. At arity 3 there are 3x3 = 9 pairings
    built from 6 rules, so a bank that factorises stores 6 fragments
    where a memoriser needs 9 programs.

    Six pairings train; three are held out. Every held-out pairing uses
    two rules that appear in training — but never together — and each
    training rule appears in at least two different pairings, so no
    fragment can be identified with a single context.
    """

    def variant(first: int, second: int) -> FamilyConfig:
        return FamilyConfig(
            dual=1, arity=3, rule0=first, rule1=second,
            name=f"c{first}{second}",
        )

    train_pairs = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
    holdout_pairs = [(0, 2), (1, 1), (2, 0)]
    return (
        [variant(a, b) for a, b in train_pairs],
        [variant(a, b) for a, b in holdout_pairs],
    )


def factorial_oracle_map(
    variants: list[FamilyConfig],
) -> dict[str, list[int]]:
    """One fragment per sub-rule, shared by every context that obeys it."""

    rule_index: dict[str, int] = {}
    mapping: dict[str, list[int]] = {}
    for config in variants:
        indices = []
        for rule in config.rules():
            if rule not in rule_index:
                rule_index[rule] = len(rule_index)
            indices.append(rule_index[rule])
        mapping[config.name] = indices
    return mapping


def practice_map(
    variants: list[FamilyConfig], *, partners: int
) -> dict[str, list[list[int]]]:
    """Several interchangeable fragments per sub-rule (compositional practice).

    F16 measured the composition failure: a fragment trained inside two
    contexts still would not carry its rule into a novel pairing. The
    literature's diagnosis (convergent finding 2) is co-adaptation — a
    fragment always seen beside the same partner never has to encode its
    rule independently, because the pair jointly encodes the context.

    Minting `partners` interchangeable fragments per rule and drawing a
    fresh combination every rollout removes the habitual partner: each
    fragment meets many partners, so the only thing it can reliably
    contribute is its own rule. This is the MLC lever (constantly pose
    novel recombinations) applied to an opaque bank.

    Returns, per variant, one candidate list per rule.
    """

    if partners < 1:
        raise ValueError("compositional practice needs at least one partner")
    slots: dict[str, list[int]] = {}
    mapping: dict[str, list[list[int]]] = {}
    for config in variants:
        rows = []
        for rule in config.rules():
            if rule not in slots:
                start = len(slots) * partners
                slots[rule] = list(range(start, start + partners))
            rows.append(list(slots[rule]))
        mapping[config.name] = rows
    return mapping


def draw_practice(
    candidates: list[list[int]], *, generator: torch.Generator | None = None
) -> list[int]:
    """One fragment per rule, sampled fresh so partners keep changing."""

    return [
        row[int(torch.randint(len(row), (1,), generator=generator))]
        for row in candidates
    ]


def battery_suite() -> tuple[list[FamilyConfig], list[FamilyConfig]]:
    """Quantity first: every fast-learnable simple context at once.

    The generalization question changes character with scale — many
    contexts on one plant is a different regime from three. Each game is
    deliberately tiny (one component, level 1) so iteration stays fast;
    complexity is added only after the architecture survives quantity.
    `dualBD` is held out as the standing novel-recombination probe.

    Membership is calibrated, not aspirational: decision games clear
    their fast-budget solo ceilings directly (choice 1.00, dualAC 1.00,
    dualAD 0.69, dualBC 0.72, avoid 0.92); the motor games join under
    egocentric rendering, which broke the translation-invariance wall
    (F22: forage 0.03 -> 0.45, collect -> 0.55, intercept -> 0.31 at 500
    updates). navigate stays out: 0.14 even egocentrically (wall
    geometry suffers under the toroidal roll).
    """

    train = [
        FamilyConfig(choice=1, name="choiceA"),
        FamilyConfig(choice=1, inverted=True, name="choiceB"),
        FamilyConfig(dual=1, name="dualAC"),
        FamilyConfig(dual=1, inverted2=True, name="dualAD"),
        FamilyConfig(dual=1, inverted=True, name="dualBC"),
        FamilyConfig(avoid=1, name="avoid1"),
        # Views set by calibration (F28), not by assumption: crop for
        # local-geometry games, roll for boundary-anchored ones, none for
        # games whose avatar is already centred.
        FamilyConfig(forage=1, view="roll", name="forageA"),
        FamilyConfig(forage=1, inverted=True, view="roll", name="forageB"),
        FamilyConfig(collect=1, view="crop", name="collect1"),
        FamilyConfig(intercept=1, view="roll", name="intercept1"),
    ]
    holdout = [
        FamilyConfig(dual=1, inverted=True, inverted2=True, name="dualBD"),
    ]
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
    egocentric: bool = False,
    encoder=None,
    per_step_baseline: bool = False,
    normalize_advantage: bool = False,
    combiner: FragmentCombiner | None = None,
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
        observation = verifier.observation()
        # F28: the game's own declared view wins; the run-level flag is
        # only the default for games that declare none.
        # A forced view (F30: chosen jointly for ALL games, never per
        # game) overrides the game's declared preference.
        forced = egocentric if isinstance(egocentric, str) else None
        if forced == "force-none":
            view = ""
        elif forced in ("force-roll", "force-crop"):
            view = forced.split("-")[1]
        else:
            view = config.view or (
                "crop" if egocentric == "crop" else "roll" if egocentric else ""
            )
        if view == "crop":
            observation = egocentric_crop(observation)
        elif view == "roll":
            observation = egocentric_view(observation)
        observation = pad_channels(observation, SHARED_SCREEN_CHANNELS)
        # F29: games sharing one screen encoder couple through it, so a
        # per-game encoder may be supplied. The amodal design calls for N
        # encoders; the shared driver was a convenience.
        screen = encoder or agent.runtime.encoders["screen"]
        events = [screen(observation)]
        if fragments is not None:
            context = (
                combiner(fragments) if combiner is not None else fragments
            )
            events.extend(
                artifact_events(
                    context.reshape(-1, context.shape[-1]), batch_size
                )
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
        # A single scalar baseline over all timesteps is badly matched to
        # discounted returns, which shrink toward the end of an episode:
        # early steps look good and late steps look bad regardless of the
        # action taken. Centring PER TIMESTEP removes that bias and is the
        # cheapest variance reduction available (F33).
        if per_step_baseline:
            counts = mask_matrix.sum(dim=0).clamp_min(1.0)
            baseline = (advantage * mask_matrix).sum(dim=0) / counts
            advantage = advantage - baseline.unsqueeze(0)
        else:
            advantage = advantage - (
                (advantage * mask_matrix).sum()
                / mask_matrix.sum().clamp_min(1.0)
            )
        if normalize_advantage:
            # A shared plant sees games whose returns differ by an order of
            # magnitude (a fatal -1, a +1/-0.2 trial, a 48-step forage
            # total), so per-game gradient magnitudes differ for reasons
            # that have nothing to do with how much there is to learn.
            # Scaling to unit deviation equalises them.
            count = mask_matrix.sum().clamp_min(1.0)
            variance = (advantage.square() * mask_matrix).sum() / count
            advantage = advantage / variance.sqrt().clamp_min(1e-6)
        props = torch.stack(log_props, dim=1)
    return {
        "total_reward": reward_matrix.sum(dim=1),
        "advantage": advantage,
        "log_propensity": props,
        "mask": mask_matrix,
        "logits": torch.stack(logits_trace, dim=1),
        # Verifier-side scoring, reported to the harness only: which of the
        # two sub-rules the run actually obeyed. Never reaches the learner.
        "rule_accuracy": (
            torch.tensor(verifier.dual_accuracy()) if config.dual else None
        ),
        "rule_engagement": (
            torch.tensor(verifier.dual_engagement()) / max(batch_size, 1)
            if config.dual
            else None
        ),
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


class FragmentCombiner(torch.nn.Module):
    """A trained operation over fetched fragments (F33).

    Three mechanisms failed to produce composition (imposed sharing F16,
    partner rotation F27, economic pressure F33) and shared one flaw:
    none made composition the thing being optimised. Concatenating
    fragments into the event window is not an operation the controller
    was ever trained to perform, so a novel pairing arrives as an
    unfamiliar input rather than a familiar operation on familiar parts.

    This module makes the operation explicit and learnable: it maps a SET
    of fetched fragments to the context tokens the controller reads.
    Because it is one shared function applied to every pairing, a pairing
    it has never seen is still just an application of the same function —
    which is what "composition is a skill" means operationally.

    Pooling is permutation-invariant over fragments (a fetched set has no
    intrinsic order) and position-preserving over tokens. Role is carried
    by fragment CONTENT, not by slot, so nothing here privileges a rule.

    It is shared infrastructure, not per-task state: one combiner serves
    every context, so it cannot become a per-game program (F30). The
    cross-feed audit still applies and still must invert.
    """

    def __init__(
        self, *, width: int, hidden: int = 64, layers: int = 2
    ) -> None:
        super().__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(width, hidden), torch.nn.Tanh()
        )
        merge: list[torch.nn.Module] = []
        for _ in range(max(1, layers - 1)):
            merge += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
        merge += [torch.nn.Linear(hidden, width), torch.nn.Tanh()]
        self.merge = torch.nn.Sequential(*merge)

    def forward(self, fragments: torch.Tensor) -> torch.Tensor:
        """[k, tokens, width] -> [tokens, width]."""

        pooled = self.encode(fragments).sum(dim=0)
        return self.merge(pooled)


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
        selection_init_scale: float = 2.0,
    ) -> None:
        super().__init__()
        # Fragment tokens share the event window with screen events, whose
        # tanh payloads have norm ~sqrt(width)/2. Tokens initialised much
        # smaller are invisible to the controller and the plant learns to
        # ignore the bank entirely (probe 7). Match the scale.
        self.tokens = torch.nn.Parameter(
            torch.randn(fragments, tokens_per_fragment, width) * init_scale
        )
        # Zero-initialised logits give a uniform distribution, so every
        # update draws a different fragment set for the same context. The
        # plant then sees inconsistent context and learns to ignore the
        # bank (probe 11). Distinct peaked initialisation makes each
        # context's assignment stable from the first update while leaving
        # room for outcome-driven reassignment.
        self.selection_init_scale = float(selection_init_scale)
        self._oracle_map: dict[str, list[int]] = {}
        self._practice_map: dict[str, list[list[int]]] = {}
        self.selection_logits = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(
                    torch.randn(fragments) * self.selection_init_scale
                )
                for name in variants
            }
        )

    def set_oracle_map(self, mapping: dict[str, list[int]]) -> None:
        """Override the default disjoint oracle with an explicit map.

        A disjoint oracle forces every context to own private fragments,
        which makes sharing impossible by construction. A factorial map
        hands overlapping contexts a common fragment, so the read path is
        asked whether one fragment can serve several contexts at once.
        """

        self._oracle_map = {name: list(v) for name, v in mapping.items()}

    def set_practice_map(self, mapping: dict[str, list[list[int]]]) -> None:
        """Install interchangeable per-rule candidates (see `practice_map`)."""

        self._practice_map = {
            name: [list(row) for row in rows] for name, rows in mapping.items()
        }

    def practice_indices(
        self, name: str, *, generator: torch.Generator | None = None
    ) -> list[int] | None:
        """A fresh partner combination for this variant, or None if unset."""

        rows = self._practice_map.get(name)
        if rows is None:
            return None
        return draw_practice(rows, generator=generator)

    def practice_first(self, name: str) -> list[int] | None:
        """The canonical combination — deterministic, for evaluation."""

        rows = self._practice_map.get(name)
        return None if rows is None else [row[0] for row in rows]

    def oracle_indices(self, name: str, k: int) -> list[int]:
        """Fixed fragments per variant (no selection learning).

        Isolates the READ path: the plant receives genuinely different
        context in different variants, so the only question under test is
        whether it can learn context-conditional behaviour.
        """

        override = self._oracle_map.get(name)
        if override is not None:
            if len(override) != k:
                raise ValueError(
                    f"oracle map for {name} has {len(override)} fragments, "
                    f"but {k} are requested per variant"
                )
            return list(override)
        order = sorted(self.selection_logits.keys())
        position = order.index(name)
        total = self.tokens.shape[0]
        return [(position * k + offset) % total for offset in range(k)]

    def register_variant(self, name: str) -> None:
        if name not in self.selection_logits:
            self.selection_logits[name] = torch.nn.Parameter(
                torch.randn(self.tokens.shape[0]) * self.selection_init_scale
            )

    def fetch(self, indices: list[int]) -> torch.Tensor:
        return self.tokens[indices]


def has_positive_source(config: FamilyConfig) -> bool:
    if config.forage or config.choice or config.dual:
        return True
    if config.inverted:
        return False
    return bool(config.collect or config.intercept or config.navigate)


def mastery(
    summary: dict[str, torch.Tensor | None], config: FamilyConfig
) -> float:
    """Dual: rule knowledge. Positive-source: earned reward. Avoid: survival.

    A purely negative variant has no achievable positive total, so mastery
    is surviving the full lifetime with zero loss.

    `dual` is scored by per-rule accuracy rather than by reward, because
    engaging now pays even under ignorance (see `DUAL_WRONG_COST`), so a
    reward threshold would credit an agent that plays every trial and
    knows neither rule. Accuracy over resolved trials answers the question
    actually under test, and it is zero for an agent that resolves none.
    """

    if config.dual and summary.get("rule_accuracy") is not None:
        engaged = summary["rule_engagement"]
        accuracy = summary["rule_accuracy"]
        return float((accuracy * (engaged > 0).float()).mean())
    if has_positive_source(config):
        return float((summary["total_reward"] > 0).float().mean())
    survived = summary["mask"][:, -1] * (summary["total_reward"] >= 0).float()
    return float(survived.mean())


def update_conflict(
    agent: SharedControllerAgent,
    bank: FragmentBank,
    variants: list[FamilyConfig],
    conflict: dict[frozenset[str], float],
    recent: dict[str, float],
    *,
    args: argparse.Namespace,
    update: int,
) -> None:
    """Estimate, from outcomes alone, which context pairs actually clash.

    A blanket diversity penalty is anti-collapse but also anti-sharing: it
    repels every pair equally, so contexts that ought to reuse a fragment
    are driven apart and the bank degenerates into one private program per
    context -- exactly the "Snake program / Pong program" failure the
    memory bank exists to avoid.

    The swap test replaces the blanket rule with evidence. Run a context on
    another context's fragments and watch what happens to the scalar
    outcome. Harmless swap => the two can share, so release the repulsion.
    Costly swap => they encode incompatible rules, so keep them apart. No
    privileged knowledge of the rules is used: only the reward the verifier
    already returns.
    """

    if len(variants) < 2:
        return
    index = update % (len(variants) * (len(variants) - 1))
    target, source = list(permutations(variants, 2))[index]
    with torch.no_grad():
        chosen, _ = sample_selection(
            bank.selection_logits[source.name].detach(),
            args.fragments_per_variant,
            greedy=True,
        )
        swapped = rollout_family(
            agent,
            target,
            bank.fetch(chosen).detach(),
            batch_size=args.batch_size,
            steps=max(8, args.steps // 2),
            seed=args.seed + 800_000 + update,
            sample=False,
            gamma=args.gamma,
            egocentric=getattr(args, "egocentric", False),
        )
    own = recent.get(target.name, 0.0)
    drop = own - mastery(swapped, target)
    signal = float(min(1.0, max(0.0, drop / max(own, 1e-3))))
    key = frozenset((target.name, source.name))
    rate = float(getattr(args, "conflict_decay", 0.8))
    conflict[key] = rate * conflict[key] + (1.0 - rate) * signal


def train_bank(
    agent: SharedControllerAgent,
    bank: FragmentBank,
    variants: list[FamilyConfig],
    *,
    args: argparse.Namespace,
    train_plant: bool,
    train_fragments: bool,
    seed_offset: int = 0,
    conflict_out: dict[frozenset[str], float] | None = None,
    combiner: FragmentCombiner | None = None,
) -> list[dict[str, float]]:
    parameters: list[torch.nn.Parameter] = list(
        bank.selection_logits.parameters()
    )
    if train_fragments:
        parameters.append(bank.tokens)
        if combiner is not None:
            parameters.extend(combiner.parameters())
    if train_plant:
        parameters.extend(
            trainable_parameters(
                [agent.controller, *agent.game_modules(agent.games[0])]
            )
        )
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    baseline: dict[str, float] = {}
    recent: dict[str, float] = {v.name: 0.0 for v in variants}
    # Neutral prior: half repulsion until the swap test says otherwise.
    conflict = conflict_out if conflict_out is not None else {}
    for left, right in combinations(variants, 2):
        conflict.setdefault(frozenset((left.name, right.name)), 0.5)
    history: list[dict[str, float]] = []
    stagger = int(getattr(args, "stagger_updates", 0))
    for update in range(args.updates):
        # F5: contradictory contexts introduced simultaneously deadlock.
        # Staggering admits one context at a time, so each new contradiction
        # arrives against an already-anchored read path.
        active = variants
        if stagger > 0:
            active = variants[: 1 + update // stagger] or variants[:1]
        if getattr(args, "balance_contexts", False) and len(active) > 1:
            # F10: uniform interleaving lets one context take the plant.
            # Sample the laggard more often, softmax over -progress.
            weights = torch.tensor(
                [-recent[v.name] for v in active]
            ) / max(args.balance_temperature, 1e-6)
            probs = F.softmax(weights, dim=-1)
            # F23: a pure laggard softmax lets hopeless contexts capture
            # the schedule and starve mastered ones. A uniform floor
            # guarantees every context a maintenance ration and caps any
            # single context's share.
            mix = float(getattr(args, "balance_uniform_mix", 0.0))
            if mix > 0.0:
                probs = (1.0 - mix) * probs + mix * torch.full_like(
                    probs, 1.0 / len(active)
                )
            index = int(torch.multinomial(probs, 1))
            config = active[index]
        else:
            config = active[update % len(active)]
        oracle_phase = update < getattr(args, "oracle_updates", 0)
        practice = bank.practice_indices(config.name)
        if practice is not None:
            # Compositional practice: a fresh partner combination every
            # update, so no fragment can lean on a habitual partner.
            chosen = practice
            selection_log_prob = torch.zeros(())
        elif getattr(args, "oracle_selection", False) or oracle_phase:
            chosen = bank.oracle_indices(
                config.name, args.fragments_per_variant
            )
            selection_log_prob = torch.zeros(())
        else:
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
            egocentric=getattr(args, "egocentric", False),
            combiner=combiner,
        )
        advantage = summary["advantage"]
        assert advantage is not None
        policy_terms = advantage * summary["log_propensity"] * summary["mask"]
        policy_loss = -policy_terms.sum() / policy_terms.shape[0]
        outcome_score = mastery(summary, config)
        previous = baseline.get(config.name, 0.0)
        baseline[config.name] = 0.9 * previous + 0.1 * outcome_score
        recent[config.name] = 0.9 * recent[config.name] + 0.1 * outcome_score
        selection_loss = -(outcome_score - previous) * selection_log_prob
        if oracle_phase:
            # Handover: while the oracle drives selection, train the
            # learned selector to imitate it, so releasing control does
            # not discard the assignment the read path formed around.
            target = torch.zeros_like(bank.selection_logits[config.name])
            target[chosen] = 1.0 / len(chosen)
            selection_loss = selection_loss + F.kl_div(
                F.log_softmax(bank.selection_logits[config.name], dim=-1),
                target,
                reduction="sum",
            )
        if getattr(args, "conflict_gated", False) and (
            update % max(1, getattr(args, "conflict_every", 10)) == 0
        ):
            update_conflict(
                agent,
                bank,
                variants,
                conflict,
                recent,
                args=args,
                update=update,
            )
        if not getattr(args, "oracle_selection", False) and getattr(
            args, "selection_diversity", 0.0
        ) > 0.0:
            # Weakness 11: outcome-REINFORCE selectors collapse to the same
            # picks for every context. Penalise overlap between the
            # selection distributions of different contexts -- but weight
            # each pair by measured CONFLICT when gating is on, so contexts
            # that can share are left free to share (see `update_conflict`).
            probs = {
                v.name: F.softmax(bank.selection_logits[v.name], dim=-1)
                for v in variants
            }
            overlap = torch.zeros(())
            pairs = 0
            for left, right in combinations(variants, 2):
                weight = 1.0
                if getattr(args, "conflict_gated", False):
                    weight = conflict[frozenset((left.name, right.name))]
                overlap = overlap + weight * (
                    probs[left.name] * probs[right.name]
                ).sum()
                pairs += 1
            if pairs:
                selection_loss = selection_loss + (
                    args.selection_diversity * overlap / pairs
                )
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
                egocentric=getattr(args, "egocentric", False),
                combiner=combiner,
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
                egocentric=getattr(args, "egocentric", False),
                combiner=combiner,
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


def evaluate_detail(
    agent: SharedControllerAgent,
    bank: FragmentBank | None,
    config: FamilyConfig,
    *,
    args: argparse.Namespace,
    fragments_override: torch.Tensor | None = None,
    encoder=None,
    combiner: FragmentCombiner | None = None,
) -> dict[str, object]:
    """Mastery plus the graded readouts a factorial suite needs.

    Binary mastery cannot separate "obeys one of two rules" from "obeys
    neither": both land near zero net reward. Mean return and per-rule
    accuracy make partial competence visible, which is precisely what a
    cross-fed fragment is expected to produce when it carries one rule.
    """

    scores, returns, accuracies, engagements = [], [], [], []
    for index in range(args.eval_seeds):
        seed = args.seed + 10_000 + index
        fragments = fragments_override
        if fragments is None and bank is not None:
            practice = bank.practice_first(config.name)
            if practice is not None:
                chosen = practice
            elif getattr(args, "oracle_selection", False):
                chosen = bank.oracle_indices(
                    config.name, args.fragments_per_variant
                )
            else:
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
                egocentric=getattr(args, "egocentric", False),
                encoder=encoder,
                combiner=combiner,
            )
        scores.append(mastery(summary, config))
        returns.append(float(summary["total_reward"].mean()))
        if summary["rule_accuracy"] is not None:
            accuracies.append(summary["rule_accuracy"])
            engagements.append(summary["rule_engagement"])
    detail: dict[str, object] = {
        "mastery": float(torch.tensor(scores).mean()),
        "mean_return": float(torch.tensor(returns).mean()),
    }
    if accuracies:
        detail["rule_accuracy"] = [
            round(value, 4)
            for value in torch.stack(accuracies).mean(dim=0).tolist()
        ]
        detail["rules"] = list(config.rules())
        detail["rule_engagement"] = [
            round(value, 4)
            for value in torch.stack(engagements).mean(dim=0).tolist()
        ]
    return detail


def evaluate_variant(
    agent: SharedControllerAgent,
    bank: FragmentBank | None,
    config: FamilyConfig,
    *,
    args: argparse.Namespace,
    fragments_override: torch.Tensor | None = None,
) -> float:
    detail = evaluate_detail(
        agent, bank, config, args=args, fragments_override=fragments_override
    )
    return float(detail["mastery"])


def overlap_report(
    bank: FragmentBank,
    variants: list[FamilyConfig],
    k: int,
    *,
    oracle: bool = False,
) -> dict[str, object]:
    """Allocation overlap against structural overlap.

    The compounding claim is a correlation: pairs that share a rule should
    share fragments, pairs that share none should share nothing. Reporting
    both columns per pair makes that testable instead of anecdotal.
    """

    selections = {}
    for config in variants:
        if oracle:
            chosen = bank.oracle_indices(config.name, k)
        else:
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
            "shared_rules": len(set(left.rules()) & set(right.rules())),
            "fragment_jaccard": jaccard,
        }
    return {
        "selections": {name: sorted(s) for name, s in selections.items()},
        "pairs": pairs,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    suites = {
        "micro": micro_suite,
        "twins": twins_suite,
        "dual": dual_suite,
        "battery": battery_suite,
        "compose": compose_suite,
    }
    train_variants, holdout_variants = suites[args.suite]()
    combiner = (
        FragmentCombiner(width=args.event_width, hidden=args.combiner_hidden)
        if getattr(args, "combiner", False)
        else None
    )
    # The combiner pools the fetched set down to one fragment's worth of
    # tokens, so the window it needs is smaller than raw concatenation.
    context_tokens = (
        args.tokens_per_fragment
        if combiner is not None
        else args.fragments_per_variant * args.tokens_per_fragment
    )
    agent = SharedControllerAgent(
        event_width=args.event_width,
        intention_width=args.intent_width,
        feedback_width=args.feedback_width,
        hidden=args.hidden,
        event_window_capacity=context_tokens + 4,
        shared_drivers=True,
    )
    bank = FragmentBank(
        fragments=args.fragments,
        tokens_per_fragment=args.tokens_per_fragment,
        width=args.event_width,
        variants=[v.name for v in train_variants],
        init_scale=args.fragment_init_scale,
        selection_init_scale=args.selection_init_scale,
    )
    rule_map = factorial_oracle_map(train_variants + holdout_variants)
    if getattr(args, "oracle_map", "disjoint") == "factorial":
        if not all(v.rules() for v in train_variants):
            raise ValueError("factorial oracle needs variants that declare rules")
        bank.set_oracle_map(
            {v.name: rule_map[v.name] for v in train_variants}
        )
    practice_partners = int(getattr(args, "practice_partners", 0) or 0)
    practice_all = {}
    if practice_partners > 0:
        if not all(v.rules() for v in train_variants):
            raise ValueError("compositional practice needs declared rules")
        practice_all = practice_map(
            train_variants + holdout_variants, partners=practice_partners
        )
        bank.set_practice_map(
            {v.name: practice_all[v.name] for v in train_variants}
        )
    if args.suite == "micro":
        singles = [v for v in train_variants if len(v.active()) == 1]
    else:
        singles = [train_variants[0]]
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
    # Did the anchor phase leave a plant that DEPENDS on the bank? If the
    # anchor is masterable with the bank withheld, the plant kept the skill
    # in its weights and phase two has an incumbent to fight (F8's blind
    # warm-up failure). Measured before phase two so the answer cannot be
    # confused with what phase two does.
    anchor_report: dict[str, object] = {}
    anchor_tokens = None
    later_variants = train_variants
    if args.warm_updates > 0:
        anchor_report = {
            "with_bank": evaluate_detail(agent, bank, singles[0], args=args),
            "withheld": evaluate_detail(
                agent, None, singles[0], args=args, fragments_override=None
            ),
        }
        anchor_indices = bank.oracle_indices(
            singles[0].name, args.fragments_per_variant
        )
        anchor_tokens = bank.tokens[anchor_indices].detach().clone()
        if getattr(args, "exclude_anchor", False):
            # The strict continual setting: the anchor context becomes
            # unreachable. No rollouts, no gradients, no replay -- if it
            # survives, it survives because nothing holding it moved.
            later_variants = [
                v for v in train_variants if v.name != singles[0].name
            ]
    conflict: dict[frozenset[str], float] = {}
    # F14 consequence: the read path is not the bottleneck -- writing a
    # second program against an incumbent is. Freezing the plant after the
    # anchor phase removes the incumbent problem entirely: later contexts
    # can only enter the bank, and nothing that holds the anchor's
    # competence is allowed to move. This is the architecture's storage
    # rule run as an experiment rather than assumed.
    history = warm_history + train_bank(
        agent,
        bank,
        later_variants,
        args=args,
        train_plant=not getattr(args, "freeze_plant", False),
        train_fragments=True,
        seed_offset=250_000,
        conflict_out=conflict,
        combiner=combiner,
    )
    if anchor_tokens is not None:
        after = bank.tokens[
            bank.oracle_indices(singles[0].name, args.fragments_per_variant)
        ].detach()
        anchor_report["tokens_unchanged"] = bool(
            torch.equal(anchor_tokens, after)
        )
        anchor_report["token_drift"] = float((after - anchor_tokens).abs().max())
    train_scores = {
        v.name: evaluate_detail(agent, bank, v, args=args, combiner=combiner)
        for v in train_variants
    }
    withheld_scores = {
        v.name: evaluate_detail(
            agent, None, v, args=args, fragments_override=None,
            combiner=combiner,
        )
        for v in train_variants
    }
    decoy = torch.randn_like(
        bank.tokens[: args.fragments_per_variant].detach()
    )
    decoy = decoy * (
        bank.tokens.detach().norm() / decoy.norm().clamp_min(1e-12)
    )
    decoy_scores = {
        v.name: evaluate_detail(
            agent, None, v, args=args, fragments_override=decoy,
            combiner=combiner,
        )
        for v in train_variants
    }

    def fragments_for(name: str) -> torch.Tensor:
        if args.oracle_selection:
            chosen = bank.oracle_indices(name, args.fragments_per_variant)
        else:
            chosen, _ = sample_selection(
                bank.selection_logits[name].detach(),
                args.fragments_per_variant,
                greedy=True,
            )
        return bank.fetch(chosen).detach()

    # Every ordered pair, not just the twins: with factorial contexts the
    # informative signal is GRADED. A source sharing one rule with the
    # target should leave that rule intact and break the other. Large
    # batteries cap the audit (--cross-pairs sources per target, rotating
    # neighbours) so iteration speed survives quantity.
    cross_pairs = list(permutations(train_variants, 2))
    limit = int(getattr(args, "cross_pairs", 0) or 0)
    if limit > 0:
        count = len(train_variants)
        cross_pairs = [
            (train_variants[i], train_variants[(i + offset) % count])
            for i in range(count)
            for offset in range(1, min(limit, count - 1) + 1)
        ]
    cross_scores = {
        f"{target.name}<-{source.name}": evaluate_detail(
            agent,
            None,
            target,
            args=args,
            fragments_override=fragments_for(source.name),
            combiner=combiner,
        )
        for target, source in cross_pairs
    }

    holdout_report = {}
    for config in holdout_variants:
        bank.register_variant(config.name)
        zero_shot = float(
            evaluate_detail(
                agent, bank, config, args=args, combiner=combiner
            )["mastery"]
        )
        # Composition splits into two questions the usual holdout conflates:
        # whether existing CONTENT recombines, and whether the selector can
        # ADDRESS the recombination. Hand over the ideal pair of already-
        # trained fragments to answer the first in isolation.
        composed = None
        ideal = rule_map.get(config.name)
        composable = (
            ideal is not None
            and len(ideal) == args.fragments_per_variant
            and max(ideal) < bank.tokens.shape[0]
        )
        if composable:
            composed = evaluate_detail(
                agent,
                None,
                config,
                args=args,
                fragments_override=bank.fetch(ideal).detach(),
                combiner=combiner,
            )
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
        # Compositional practice makes a stronger claim testable: if each
        # fragment carries its rule independently, then EVERY combination
        # of the held-out rules' fragments should work, not just one. The
        # spread across combinations is the co-adaptation measurement.
        practice_combinations = None
        if practice_all and config.name in practice_all:
            rows = practice_all[config.name]
            if max(max(row) for row in rows) < bank.tokens.shape[0]:
                scores = {}
                for combo in product(*rows):
                    scores["+".join(map(str, combo))] = evaluate_variant(
                        agent,
                        None,
                        config,
                        args=args,
                        fragments_override=bank.fetch(list(combo)).detach(),
                    )
                values = list(scores.values())
                practice_combinations = {
                    "per_combination": scores,
                    "mean": sum(values) / len(values),
                    "worst": min(values),
                    "spread": max(values) - min(values),
                }
        holdout_report[config.name] = {
            "zero_shot": zero_shot,
            "composed_from_trained_fragments": composed,
            "composed_all_partner_combinations": practice_combinations,
            "ideal_fragments": ideal,
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
        "anchor": anchor_report,
        "train_mastery": train_scores,
        "withheld_bank_mastery": withheld_scores,
        "decoy_bank_mastery": decoy_scores,
        "cross_fragment_mastery": cross_scores,
        "overlap": overlap_report(
            bank,
            train_variants,
            args.fragments_per_variant,
            oracle=bool(args.oracle_selection),
        ),
        "holdout": holdout_report,
        # What the swap test believes about each pair. Contexts sharing a
        # rule should end up measurably less in conflict than contexts
        # sharing none -- an outcome-only estimate of structural overlap.
        "measured_conflict": {
            "|".join(sorted(pair)): round(value, 4)
            for pair, value in sorted(
                conflict.items(), key=lambda item: sorted(item[0])
            )
        },
        "no_replay": all(
            entry["replayed_examples"] == 0.0 for entry in history
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suite",
        type=str,
        default="micro",
        choices=["micro", "twins", "dual", "battery", "compose"],
    )
    parser.add_argument(
        "--practice-partners",
        type=int,
        default=0,
        help="mint N interchangeable fragments per sub-rule and draw a "
        "fresh combination each update, so fragments cannot co-adapt with "
        "a habitual partner (the F16 composition failure)",
    )
    parser.add_argument(
        "--egocentric-crop",
        dest="egocentric",
        action="store_const",
        const="crop",
        help="egocentric with zero fill instead of wraparound: same "
        "invariance without inventing geometry at the boundaries",
    )
    parser.add_argument(
        "--egocentric",
        action="store_true",
        help="encoder-side egocentric rendering: roll the screen so the "
        "avatar is always centred (F22 motor-wall fix)",
    )
    parser.add_argument(
        "--cross-pairs",
        type=int,
        default=0,
        help="cap the cross-feed audit at N rotating sources per target "
        "(0 = every ordered pair)",
    )
    parser.add_argument(
        "--oracle-map",
        type=str,
        default="disjoint",
        choices=["disjoint", "factorial"],
        help="disjoint: private fragments per context; factorial: one "
        "fragment per sub-rule, shared by every context that obeys it",
    )
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
    parser.add_argument(
        "--combiner",
        action="store_true",
        help="learn an explicit operation over fetched fragments "
        "instead of concatenating them (F33)",
    )
    parser.add_argument("--combiner-hidden", type=int, default=64)
    parser.add_argument("--oracle-selection", action="store_true")
    parser.add_argument("--balance-contexts", action="store_true")
    parser.add_argument("--balance-temperature", type=float, default=0.25)
    parser.add_argument("--balance-uniform-mix", type=float, default=0.0)
    parser.add_argument("--selection-diversity", type=float, default=0.0)
    parser.add_argument("--selection-init-scale", type=float, default=2.0)
    parser.add_argument(
        "--conflict-gated",
        action="store_true",
        help="weight the diversity penalty per pair by measured swap harm, "
        "so contexts that can share a fragment are not repelled",
    )
    parser.add_argument(
        "--freeze-plant",
        action="store_true",
        help="after the anchor phase (--warm-updates), train ONLY the bank: "
        "later contexts must enter as fragments, and the anchor's "
        "competence cannot be overwritten because nothing holding it moves",
    )
    parser.add_argument(
        "--exclude-anchor",
        action="store_true",
        help="make the anchor context unreachable after its phase (strict "
        "continual setting: no rollouts, no gradients, no replay)",
    )
    parser.add_argument(
        "--stagger-updates",
        type=int,
        default=0,
        help="admit one further context every N updates (F5 curriculum); "
        "0 trains every context from the first update",
    )
    parser.add_argument("--conflict-every", type=int, default=10)
    parser.add_argument("--conflict-decay", type=float, default=0.8)
    parser.add_argument("--oracle-updates", type=int, default=0)
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
