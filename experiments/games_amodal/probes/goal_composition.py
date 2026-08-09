"""Goal-factored composition (GOAL_FACTORED_DESIGN.md, Rung B).

The frontier F16/F27/F33/F34 could not cross: practise (A+B) and (C+D),
then do the never-trained (A+D) with no further learning.

Why the dual games are the right testbed. A dual variant c_ij means
"when the cue is 0 take side i; when the cue is 1 take side j" -- it IS
a pairing of two independent rules, and `compose_suite` holds three
pairings out whose two halves each appear in training but never
together.

Why the goal factorisation should crack it. If the plant is a general
goal-follower and a fragment is just "which side to want, per cue",
then a game's fragment is literally two slots. A held-out pairing is
then ASSEMBLED from slots already trained in other games -- no gradient
step, no new learning. That is compositional generalisation by
construction rather than by hope, and it is exactly what an opaque
whole-program fragment cannot offer.

Phases:
  1  competence: follow "take side s" for s in 0,1,2, verifier-free,
     self-checked from the pre-action observation. Frozen after.
  2  train fragments for the six training pairings. Each fragment is
     [2, width]: row = observed cue. Plant frozen; only fragments move.
  3  gates:
     no_agent      random actions, the measured floor
     trained       the six training pairings (the ceiling to match)
     ASSEMBLED     three held-out pairings, slots copied from trained
                   games, zero learning  <- the composition claim
     scrambled     held-out pairings assembled from the WRONG slots,
                   the control that says assembly is not free
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import (
    compose_suite,
    mastery,
)
from experiments.games_amodal.game_family import FamilyVerifier
from experiments.games_amodal.shared_controller import (
    SHARED_SCREEN_CHANNELS,
    SharedControllerAgent,
    pad_channels,
    trainable_parameters,
)
from neural_computer import AmodalEvent, ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--competence-updates", type=int, default=1500)
parser.add_argument("--game-updates", type=int, default=900)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--hidden", type=int, default=32)
parser.add_argument("--max-restarts", type=int, default=6)
parser.add_argument("--ignorance", type=float, default=2.0)
parser.add_argument("--ignorance-gate", type=float, default=0.7)
parser.add_argument("--eval-episodes", type=int, default=4)
parser.add_argument("--ewc", type=float, default=200.0)
parser.add_argument("--ewc-mu", type=float, default=3.0)
parser.add_argument(
    "--phase1-sides", type=int, default=0,
    help="diagnostic: restrict phase 1 to the first N sides (0 = all). "
         "Separates 'three goals is hard' from 'the dual game's "
         "plane1/plane2/both perception is hard'.")
parser.add_argument(
    "--curriculum", choices=("mixed", "sequential", "consolidated"),
    default="mixed",
    help="sequential: acquire each goal alone (isolation always "
         "converges) then a mixed consolidation tail; mixed: laggard "
         "sampling from the start (basin-determined, 0/9 first-draw)")
args = parser.parse_args()

SIDES = 3

DELTAS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])
train_games, holdout_games = compose_suite()
TRAIN = {c.name: c for c in train_games}
HOLDOUT = {c.name: c for c in holdout_games}


def build_plant(attempt: int):
    global agent, plant, decoder, COMMANDS
    torch.manual_seed(args.seed + 7000 * attempt)
    agent = SharedControllerAgent(
        event_width=args.width, intention_width=32, feedback_width=16,
        hidden=args.hidden, event_window_capacity=8, shared_drivers=True,
    )
    plant = list(trainable_parameters(
        [agent.controller, *agent.game_modules(agent.games[0])]
    ))
    decoder = agent.runtime.output_bus.decoders["keypress"]
    vocab = torch.Generator().manual_seed(args.seed + 999 + 7000 * attempt)
    COMMANDS = torch.nn.Parameter(
        torch.randn(SIDES, args.width, generator=vocab))
    project_commands()
    return plant


def project_commands() -> None:
    """Gram-Schmidt to norm 4: the vocabulary must stay separated or the
    difference between commands stops carrying anything (measured in the
    cued rung as a rise-then-collapse curve)."""
    with torch.no_grad():
        basis = []
        for index in range(SIDES):
            vector = COMMANDS[index].clone()
            for earlier in basis:
                vector = vector - (vector @ earlier) * earlier
            vector = vector / vector.norm().clamp_min(1e-6)
            basis.append(vector)
            COMMANDS[index].copy_(vector * 4.0)


def side_planes(observation: torch.Tensor, target: torch.Tensor):
    """Which side (0,1,2) sits at each row's target cell, one-hot-ish.

    Side 0 renders on plane 1 only, side 1 on plane 2 only, side 2 on
    BOTH planes at one cell (the arity-3 third choice).
    """
    batch = observation.shape[0]
    plane_a = observation[:, 1].reshape(batch, -1).gather(
        1, target.unsqueeze(-1)).squeeze(-1)
    plane_b = observation[:, 2].reshape(batch, -1).gather(
        1, target.unsqueeze(-1)).squeeze(-1)
    both = (plane_a > 0) & (plane_b > 0)
    return torch.stack([
        ((plane_a > 0) & ~both).float(),
        ((plane_b > 0) & ~both).float(),
        both.float(),
    ], dim=-1)


def read_cue(observation: torch.Tensor) -> torch.Tensor:
    """The dual cue as rendered: row 0 of plane 0, left half vs right."""
    half = observation.shape[-1] // 2
    left = observation[:, 0, 0, :half].sum(dim=-1)
    right = observation[:, 0, 0, half:].sum(dim=-1)
    return (right > left).long()


def episode(config, *, command=None, slots=None, seed: int, sample: bool,
            random_actions: bool = False):
    """One rollout. `command` = fixed per-row side (phase 1);
    `slots` = [2, width] fragment indexed by the observed cue (phase 2/3)."""
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    state = agent.controller.initial_state(args.batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(args.batch_size, agent.controller.feedback_width),
        reward=torch.zeros(args.batch_size),
        propensity=torch.ones(args.batch_size),
        has_feedback=torch.zeros(args.batch_size))
    rng = torch.Generator().manual_seed(seed + 5)
    rewards, selfr, logps, masks, logits_trace = [], [], [], [], []
    alive = torch.ones(args.batch_size, dtype=torch.bool)
    for _step in range(args.steps):
        masks.append(alive.float())
        observation = pad_channels(
            verifier.observation(), SHARED_SCREEN_CHANNELS)
        events = [agent.runtime.encoders["screen"](observation)]
        if command is not None:
            goal = COMMANDS[command]
        elif slots is not None:
            goal = slots[read_cue(observation)]
        else:
            goal = None
        if goal is not None:
            events.append(AmodalEvent(payload=goal))
        output, state = agent.runtime.step_events(events, state, feedback)
        if random_actions:
            acts = torch.randint(0, decoder.key_count, (args.batch_size,),
                                 generator=rng)
            logps.append(torch.zeros(args.batch_size))
        else:
            logits_trace.append(output.decoded["keypress"])
            decision = decoder.decide_from_logits(
                output.decoded["keypress"], sample=sample)
            acts = decision.key_index
            logps.append(decision.propensity.clamp_min(1e-8).log())
        # Self-check BEFORE stepping: which side is at avatar+delta?
        if command is not None:
            height, width = observation.shape[-2:]
            flat = observation[:, 0].reshape(args.batch_size, -1)
            # The dual cue also lives on plane 0 row 0; ignore row 0 when
            # locating the avatar.
            masked = flat.clone()
            masked[:, :width] = 0.0
            index = masked.argmax(dim=-1)
            delta = DELTAS[acts]
            row = (index // width + delta[:, 0]).clamp(0, height - 1)
            col = (index % width + delta[:, 1]).clamp(0, width - 1)
            planes = side_planes(observation, row * width + col)
            wanted = planes.gather(1, command.unsqueeze(-1)).squeeze(-1)
            other = planes.sum(dim=-1) - wanted
            engaged = planes.sum(dim=-1).clamp(0.0, 1.0)
            selfr.append((wanted - other - 0.1 * (1.0 - engaged)).clamp(-1, 1))
        else:
            selfr.append(torch.zeros(args.batch_size))
        outcome = verifier.step(acts)
        rewards.append(outcome.reward)
        alive = outcome.alive
        feedback = ControllerFeedback(
            action=agent.feedback_encoders["keypress"](acts),
            reward=torch.zeros(args.batch_size),   # no reward feedback (F-leak)
            propensity=torch.ones(args.batch_size),
            has_feedback=torch.ones(args.batch_size))
        state = state.detached() if sample else state

    def discounted(seq):
        matrix = torch.stack(seq, dim=1)
        running = torch.zeros(args.batch_size)
        out = torch.zeros_like(matrix)
        for pos in range(matrix.shape[1] - 1, -1, -1):
            running = matrix[:, pos] + args.gamma * running
            out[:, pos] = running
        return matrix, out

    reward_matrix, _ = discounted(rewards)
    self_matrix, self_returns = discounted(selfr)
    mask = torch.stack(masks, dim=1)
    return {
        "reward": reward_matrix, "self_reward": self_matrix, "mask": mask,
        "returns": self_returns, "logp": torch.stack(logps, dim=1),
        "logits": torch.stack(logits_trace, dim=1) if logits_trace else None,
        "verifier": verifier, "command": command,
    }


def dual_mastery(out, config) -> float:
    """Dual games score by per-rule accuracy -- supplying only reward
    silently takes the wrong branch (F52)."""
    verifier = out["verifier"]
    return float(mastery({
        "total_reward": out["reward"].sum(dim=1),
        "mask": out["mask"],
        "rule_accuracy": torch.tensor(verifier.dual_accuracy()),
        "rule_engagement": (torch.tensor(verifier.dual_engagement())
                            / max(args.batch_size, 1)),
    }, config))


# ---- Phase 1: goal-following competence over three sides ------------------
PHASE1_GAME = train_games[0]
for attempt in range(args.max_restarts):
    build_plant(attempt)
    params = plant + [COMMANDS]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    score = [0.5] * SIDES
    curve = []
    solo = int(args.competence_updates * 0.6)
    per_side = max(1, solo // SIDES)
    anchors: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
    consolidated_upto = -1

    def consolidate(side: int) -> None:
        """Diagonal Fisher + anchor for the goal just acquired.

        Sequential isolation ACQUIRES (the only arm that ever produced a
        learned side on seeds 201-206) but does not RETAIN: later goals
        overwrite earlier ones, which is catastrophic forgetting inside a
        single training phase. This is the promoted consolidation line
        applied to the plant's own goal vocabulary.
        """
        fisher = [torch.zeros_like(p) for p in plant]
        for batch in range(2):
            out = episode(PHASE1_GAME,
                          command=torch.full((args.batch_size,), side),
                          seed=args.seed + 500_000 + 97 * side + batch,
                          sample=True)
            log_likelihood = (
                out["logp"] * out["mask"]).sum() / out["mask"].sum().clamp_min(1)
            for p in plant:
                if p.grad is not None:
                    p.grad = None
            log_likelihood.backward()
            for slot, p in zip(fisher, plant):
                if p.grad is not None:
                    slot += p.grad.detach().square()
        for p in plant:
            p.grad = None
        total = sum(f.sum() for f in fisher)
        count = sum(f.numel() for f in fisher)
        mean = (total / count).clamp_min(1e-12)
        anchors.append(([f / mean for f in fisher],
                        [p.detach().clone() for p in plant]))

    for update in range(args.competence_updates):
        live_sides = args.phase1_sides or SIDES
        if args.curriculum == "consolidated" and update < solo:
            stage = min(update // per_side, live_sides - 1)
            if stage > consolidated_upto:
                if consolidated_upto >= 0:
                    consolidate(consolidated_upto)
                consolidated_upto = stage
            chosen = stage
        elif args.curriculum == "sequential" and update < solo:
            # Isolation converges on every seed tried; joint training
            # from scratch is basin-determined (0/9 first-draw at both
            # hidden 32 and 64, so capacity is not the constraint).
            # Acquire each goal alone, then mix to reconcile them.
            chosen = min(update // per_side, live_sides - 1)
        else:
            live = torch.tensor(score[:live_sides])
            weights = torch.softmax(-live / 0.25, dim=-1)
            weights = weights * 0.6 + 0.4 / live_sides
            chosen = int(torch.multinomial(weights, 1))
        command = torch.full((args.batch_size,), chosen)
        out = episode(PHASE1_GAME, command=command,
                      seed=args.seed + update, sample=True)
        advantage = out["returns"].detach()
        advantage = advantage - (
            advantage * out["mask"]).sum() / out["mask"].sum().clamp_min(1)
        terms = advantage * out["logp"] * out["mask"]
        loss = -terms.sum() / terms.shape[0]
        ignorance_live = (args.ignorance > 0.0
                          and min(score) >= args.ignorance_gate)
        if ignorance_live and update % 3 == 0:
            with torch.no_grad():
                probe = torch.randn(args.width)
                for direction in COMMANDS:
                    unit = direction / direction.norm().clamp_min(1e-6)
                    probe = probe - (probe @ unit) * unit
                probe = probe / probe.norm().clamp_min(1e-6) * 4.0
            blind = episode(PHASE1_GAME, slots=probe.reshape(1, -1).expand(2, -1),
                            seed=args.seed + 900_000 + update, sample=True)
            if blind["logits"] is not None:
                log_probs = torch.log_softmax(blind["logits"], dim=-1)
                uniform = -torch.tensor(float(decoder.key_count)).log()
                loss = loss + args.ignorance * (
                    log_probs - uniform).square().mean()
        with torch.no_grad():
            correct = (out["self_reward"] > 0).float().sum()
            wrong = (out["self_reward"] < 0).float().sum()
            ratio = float(correct / (correct + wrong).clamp_min(1.0))
            score[chosen] = 0.9 * score[chosen] + 0.1 * ratio
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if anchors:
            # Task gradient FIRST, then the penalty added in closed form
            # -- the promoted implementation. Computing the release from
            # a stale p.grad (the previous update's, penalty included)
            # protected goal 0 perfectly (1.00 on 5/6 seeds) while making
            # goals 1 and 2 unlearnable: over-protection, RWalk's
            # "intransigence" pole.
            with torch.no_grad():
                for fisher, anchor in anchors:
                    for f, a, p in zip(fisher, anchor, plant):
                        if p.grad is None:
                            continue
                        demand = p.grad.detach().square()
                        release = f / (f + args.ewc_mu * demand + 1e-12)
                        p.grad += args.ewc * release * f * (p.detach() - a)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        project_commands()
        if (update + 1) % 300 == 0:
            curve.append([round(v, 3) for v in score])
    if min(score[:(args.phase1_sides or SIDES)]) >= 0.55:
        break
attempts = attempt + 1
COMMANDS.requires_grad_(False)
for parameter in plant:
    parameter.requires_grad_(False)


def competence(side: int) -> float:
    with torch.no_grad():
        out = episode(PHASE1_GAME,
                      command=torch.full((args.batch_size,), side),
                      seed=args.seed + 400_000, sample=False)
    correct = (out["self_reward"] > 0).float().sum()
    wrong = (out["self_reward"] < 0).float().sum()
    return round(float(correct / (correct + wrong).clamp_min(1.0)), 4)


report = {
    "seed": args.seed, "phase1_attempts": attempts,
    "hidden": args.hidden, "competence_updates": args.competence_updates,
    "competence": {f"side{s}": competence(s) for s in range(SIDES)},
    "competence_curve": curve,
}

# ---- Phase 2: one fragment per TRAINING pairing, two slots each -----------
names = sorted(TRAIN)
slots = torch.nn.Parameter(torch.stack([
    torch.stack([COMMANDS[TRAIN[n].rule0], COMMANDS[TRAIN[n].rule1]])
    + 0.5 * torch.randn(2, args.width)
    for n in names
]))
optimizer = torch.optim.Adam([slots], lr=1e-2)
for update in range(args.game_updates):
    which = update % len(names)
    out = episode(TRAIN[names[which]], slots=slots[which],
                  seed=args.seed + 600_000 + update, sample=True)
    matrix = out["reward"]
    running = torch.zeros(args.batch_size)
    returns = torch.zeros_like(matrix)
    for pos in range(matrix.shape[1] - 1, -1, -1):
        running = matrix[:, pos] + args.gamma * running
        returns[:, pos] = running
    advantage = returns.detach()
    advantage = advantage - (
        advantage * out["mask"]).sum() / out["mask"].sum().clamp_min(1)
    terms = advantage * out["logp"] * out["mask"]
    loss = -terms.sum() / terms.shape[0]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([slots], 1.0)
    optimizer.step()

# ---- Phase 3: gates ------------------------------------------------------
def evaluate(config, slot_pair, *, rand: bool = False) -> float:
    scores = []
    for index in range(args.eval_episodes):
        with torch.no_grad():
            out = episode(config, slots=slot_pair,
                          seed=args.seed + 700_000 + index, sample=False,
                          random_actions=rand)
        scores.append(dual_mastery(out, config))
    return round(float(torch.tensor(scores).mean()), 4)


report["no_agent"] = {
    n: evaluate(HOLDOUT[n], slots[0].detach(), rand=True) for n in HOLDOUT}
report["trained"] = {
    n: evaluate(TRAIN[n], slots[i].detach()) for i, n in enumerate(names)}


def donor(cue: int, side: int) -> int | None:
    """A trained game whose `cue` slot already means `side`."""
    for index, name in enumerate(names):
        config = TRAIN[name]
        if (config.rule0 if cue == 0 else config.rule1) == side:
            return index
    return None


report["assembled"] = {}
report["scrambled"] = {}
report["assembly_source"] = {}
for name, config in HOLDOUT.items():
    first, second = donor(0, config.rule0), donor(1, config.rule1)
    if first is None or second is None:
        report["assembled"][name] = None
        continue
    built = torch.stack([slots[first, 0], slots[second, 1]]).detach()
    report["assembled"][name] = evaluate(config, built)
    report["assembly_source"][name] = [names[first], names[second]]
    # Control: the same two donors, slots swapped into the wrong cues.
    wrong = torch.stack([slots[second, 1], slots[first, 0]]).detach()
    report["scrambled"][name] = evaluate(config, wrong)

print(json.dumps(report, indent=1))
