"""Goal-factored twins, cued rung (docs/GOAL_FACTORED_DESIGN.md, Rung A).

The factorization under test:
  plant  = map + edge-competence, trained REWARD-FREE on self-checkable
           micro goals ("consume an item on the commanded plane"), then
           FROZEN — the game-invariant marginal, allowed in weights;
  bank   = one small destination fragment per game — the game-specific
           deviation, the only thing verifier reward may shape;
  cue    = the game's name rendered IN the world (a corner mark on the
           avatar plane, through the shared encoder like any percept),
           read by a tiny cue-reader that selects which fragment to
           fetch.

Per-game gradients touch ONLY the fragments and the cue-reader — the
information asymmetry that F55/F56 showed penalties cannot enforce,
enforced here by construction.

Phases:
  1  competence: command c per row, injected as a frozen random goal
     event; self-reward is DIRECTIONAL (which plane held the item at
     avatar + delta(action), read from the pre-action observation:
     commanded plane -> +1, other -> -1). No verifier reward anywhere.
  2  cued game: sample twin per episode, overlay its banner, fetch by
     cue, verifier reward trains fragments + cue-reader only.
  3  gates, greedy, no probe phase so no F53 artifact:
     no-agent control (banner + random actions ~ floor, run FIRST),
     mastery vs measured floors, cross-feed (swap fragments -> invert),
     decoy (noise fragment -> floor, and actions IDENTICAL across twins
     up to the banner pixels), label-swap (wrong banner -> behaviour
     follows the banner), necessity-under-cue (banner + noise -> floor).
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import mastery, twins_suite
from experiments.games_amodal.game_family import FamilyVerifier
from experiments.games_amodal.shared_controller import (
    SHARED_SCREEN_CHANNELS,
    SharedControllerAgent,
    pad_channels,
    trainable_parameters,
)
from experiments.games_amodal.skill_externalization import artifact_events
from neural_computer import ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--competence-updates", type=int, default=1500)
parser.add_argument("--game-updates", type=int, default=800)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--width", type=int, default=64)
args = parser.parse_args()

torch.manual_seed(args.seed)
train, _ = twins_suite()
NAMES = [c.name for c in train]

agent = SharedControllerAgent(
    event_width=args.width, intention_width=32, feedback_width=16, hidden=32,
    event_window_capacity=8, shared_drivers=True,
)
plant = list(trainable_parameters(
    [agent.controller, *agent.game_modules(agent.games[0])]
))
decoder = agent.runtime.output_bus.decoders["keypress"]
# Action index -> grid delta, matching the decoder's key order (as in
# cotrained.py's test_action).
DELTAS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])

# Command vocabulary: TRAINABLE in phase 1, frozen after. Frozen random
# vectors were measured unreadable (competence stuck at chance with a
# fixed-plane bias on both seeds): every rung where event-channel
# conditioning worked had tokens that co-adapted with the plant into a
# readable vocabulary. The vocabulary is game-invariant, so training it
# is storage-legal; norm projection keeps it salient without letting a
# collapse to zero erase the channel.
generator = torch.Generator().manual_seed(args.seed + 999)
COMMANDS = torch.nn.Parameter(torch.randn(2, args.width, generator=generator))


def project_commands() -> None:
    """Norm 4, and ORTHOGONAL: the smoke curve rose to 0.643 then fell to
    0.385 -- conditioning was learned and then lost, the signature of the
    two command vectors drifting together until their difference carried
    nothing. Orthogonality on content is the MOORE-style constraint the
    literature map recommends; norm projection alone cannot provide it."""
    with torch.no_grad():
        first = COMMANDS[0] / COMMANDS[0].norm().clamp_min(1e-6)
        second = COMMANDS[1] - (COMMANDS[1] @ first) * first
        second = second / second.norm().clamp_min(1e-6)
        COMMANDS[0].copy_(first * 4.0)
        COMMANDS[1].copy_(second * 4.0)


project_commands()

# Bank: one destination fragment per game, trained in phase 2 only.
fragments = torch.nn.Parameter(torch.randn(2, args.width, generator=generator))
# Cue-reader: banner corners -> which fragment. Two pixels in, two logits.
cue_reader = torch.nn.Linear(2, 2)


def banner(observation: torch.Tensor, game: int) -> torch.Tensor:
    """Render the game's name as a corner mark on the avatar plane."""
    out = observation.clone()
    if game == 0:
        out[:, 0, 0, 0] = 1.0
    else:
        out[:, 0, 0, -1] = 1.0
    return out


def episode(*, game: int | None, command: torch.Tensor | None,
            goal_event: torch.Tensor | None, seed: int, sample: bool,
            wrong_banner: bool = False, random_actions: bool = False):
    """One rollout. Exactly one of `command` (phase 1) or `goal_event`
    (phase 2/gates) supplies the goal channel; `game` picks verifier and
    banner (None -> choiceA world, no banner, phase 1)."""
    config = train[game if game is not None else 0]
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    state = agent.controller.initial_state(args.batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(args.batch_size, agent.controller.feedback_width),
        reward=torch.zeros(args.batch_size),
        propensity=torch.ones(args.batch_size),
        has_feedback=torch.zeros(args.batch_size))
    rng = torch.Generator().manual_seed(seed + 5)
    rewards, selfr, logps, masks, actions_trace = [], [], [], [], []
    alive = torch.ones(args.batch_size, dtype=torch.bool)
    for _step in range(args.steps):
        masks.append(alive.float())
        observation = pad_channels(verifier.observation(), SHARED_SCREEN_CHANNELS)
        if game is not None:
            shown = (1 - game) if wrong_banner else game
            observation = banner(observation, shown)
        events = [agent.runtime.encoders["screen"](observation)]
        if command is not None:
            events.extend(artifact_events(
                COMMANDS[command].reshape(-1, args.width), args.batch_size))
        if goal_event is not None:
            events.extend(artifact_events(
                goal_event.reshape(-1, args.width), args.batch_size))
        output, state = agent.runtime.step_events(events, state, feedback)
        if random_actions:
            acts = torch.randint(0, decoder.key_count, (args.batch_size,),
                                 generator=rng)
            logps.append(torch.zeros(args.batch_size))
            propensity = torch.ones(args.batch_size)
        else:
            decision = decoder.decide_from_logits(
                output.decoded["keypress"], sample=sample)
            acts = decision.key_index
            logps.append(decision.propensity.clamp_min(1e-8).log())
            propensity = decision.propensity.detach()
        outcome = verifier.step(acts)
        actions_trace.append(acts)
        rewards.append(outcome.reward)
        # Self-reward for phase 1, DIRECTIONAL: in the choice trial the
        # avatar is stationary and the action's direction consumes the
        # adjacent item, so the agent can check its own micro goal from
        # what it saw before acting: which plane held the item at
        # avatar + delta(action)? Count deltas cannot work (the verifier
        # re-deals in the same step) and landing-position checks cannot
        # either (the avatar does not move) -- both measured as
        # zero-signal before this version.
        if command is not None:
            height, width = observation.shape[-2:]
            flat_avatar = observation[:, 0].reshape(args.batch_size, -1)
            avatar_index = flat_avatar.argmax(dim=-1)
            row = avatar_index // width
            col = avatar_index % width
            delta = DELTAS[acts]                     # [batch, 2]
            target_row = (row + delta[:, 0]).clamp(0, height - 1)
            target_col = (col + delta[:, 1]).clamp(0, width - 1)
            target = target_row * width + target_col
            on_a = observation[:, 1].reshape(args.batch_size, -1).gather(
                1, target.unsqueeze(-1)).squeeze(-1)
            on_b = observation[:, 2].reshape(args.batch_size, -1).gather(
                1, target.unsqueeze(-1)).squeeze(-1)
            planes = torch.stack([on_a, on_b], dim=-1)
            commanded = planes.gather(1, command.unsqueeze(-1)).squeeze(-1)
            other = planes.sum(dim=-1) - commanded
            selfr.append((commanded - other).clamp(-1.0, 1.0))
        else:
            selfr.append(torch.zeros(args.batch_size))
        alive = outcome.alive
        feedback = ControllerFeedback(
            action=agent.feedback_encoders["keypress"](acts),
            reward=outcome.reward, propensity=propensity,
            has_feedback=torch.ones(args.batch_size))
        state = state.detached() if sample else state
    def returns_of(reward_list):
        matrix = torch.stack(reward_list, dim=1)
        out = torch.zeros_like(matrix)
        running = torch.zeros(args.batch_size)
        for pos in range(matrix.shape[1] - 1, -1, -1):
            running = matrix[:, pos] + args.gamma * running
            out[:, pos] = running
        return matrix, out
    mask = torch.stack(masks, dim=1)
    reward, _ = returns_of(rewards)
    self_reward, self_returns = returns_of(selfr)
    advantage = self_returns.detach()
    advantage = advantage - (advantage * mask).sum() / mask.sum().clamp_min(1)
    return {
        "reward": reward, "self_reward": self_reward, "mask": mask,
        "advantage": advantage, "logp": torch.stack(logps, dim=1),
        "actions": torch.stack(actions_trace, dim=1),
    }


report: dict = {"seed": args.seed}

# ---- Phase 1: edge-competence, verifier-free -------------------------------
phase1_params = plant + [COMMANDS]
optimizer = torch.optim.Adam(phase1_params, lr=1e-3)
competence_curve = []
for update in range(args.competence_updates):
    command = torch.randint(0, 2, (args.batch_size,))
    out = episode(game=None, command=command, goal_event=None,
                  seed=args.seed + update, sample=True)
    terms = out["advantage"] * out["logp"] * out["mask"]
    loss = -terms.sum() / terms.shape[0]
    # Entropy bonus: a policy that goes deterministic on one plane stops
    # sampling the commanded-vs-other contrast that teaches conditioning.
    entropy = -(out["logp"] * out["mask"]).sum() / out["mask"].sum().clamp_min(1)
    loss = loss - 0.01 * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(phase1_params, 1.0)
    optimizer.step()
    project_commands()
    if (update + 1) % 250 == 0:
        with torch.no_grad():
            probe = episode(
                game=None,
                command=torch.randint(
                    0, 2, (args.batch_size,),
                    generator=torch.Generator().manual_seed(update)),
                goal_event=None, seed=args.seed + 300_000 + update,
                sample=False)
        correct = (probe["self_reward"] > 0).float().sum()
        wrong = (probe["self_reward"] < 0).float().sum()
        competence_curve.append(
            round(float(correct / (correct + wrong).clamp_min(1.0)), 3))
COMMANDS.requires_grad_(False)


def competence_score(command_value: int) -> float:
    with torch.no_grad():
        out = episode(game=None,
                      command=torch.full((args.batch_size,), command_value),
                      goal_event=None, seed=args.seed + 400_000, sample=False)
    correct = (out["self_reward"] > 0).float().sum()
    wrong = (out["self_reward"] < 0).float().sum()
    return float(correct / (correct + wrong).clamp_min(1.0))


report["competence"] = {
    "cmd0": round(competence_score(0), 4), "cmd1": round(competence_score(1), 4)
}
report["competence_curve"] = competence_curve
with torch.no_grad():
    report["command_cosine"] = round(float(
        (COMMANDS[0] @ COMMANDS[1])
        / (COMMANDS[0].norm() * COMMANDS[1].norm()).clamp_min(1e-6)), 4)

# ---- Phase 2: cued game phase — gradients into fragments + cue-reader only -
for parameter in plant:
    parameter.requires_grad_(False)
bank_params = [fragments] + list(cue_reader.parameters())
optimizer = torch.optim.Adam(bank_params, lr=3e-3)


def fetch(observation_free_game: int, *, decoy: bool = False,
          force: int | None = None, wrong_banner: bool = False):
    """The cue-reader's fragment choice for a game's banner, as a goal event."""
    corners = torch.zeros(1, 2)
    shown = (1 - observation_free_game) if wrong_banner else observation_free_game
    corners[0, shown] = 1.0
    weights = torch.softmax(cue_reader(corners), dim=-1)
    if force is not None:
        chosen = fragments[force]
    else:
        chosen = weights @ fragments
    if decoy:
        noise = torch.randn(1, args.width,
                            generator=torch.Generator().manual_seed(7))
        chosen = noise * (chosen.detach().norm() / noise.norm().clamp_min(1e-12))
    return chosen, int(weights.argmax())


for update in range(args.game_updates):
    game = update % 2
    goal, _ = fetch(game)
    out = episode(game=game, command=None, goal_event=goal,
                  seed=args.seed + 600_000 + update, sample=True)
    matrix = out["reward"]
    running = torch.zeros(args.batch_size)
    returns = torch.zeros_like(matrix)
    for pos in range(matrix.shape[1] - 1, -1, -1):
        running = matrix[:, pos] + args.gamma * running
        returns[:, pos] = running
    advantage = returns.detach()
    advantage = advantage - (advantage * out["mask"]).sum() / out["mask"].sum().clamp_min(1)
    terms = advantage * out["logp"] * out["mask"]
    loss = -terms.sum() / terms.shape[0]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(bank_params, 1.0)
    optimizer.step()

# ---- Phase 3: gates, greedy, no probe --------------------------------------
def score(game: int, **kwargs) -> float:
    goal_kwargs = {k: kwargs.pop(k) for k in ("decoy", "force") if k in kwargs}
    wrong = kwargs.pop("wrong_banner", False)
    rand = kwargs.pop("random_actions", False)
    goal, _ = fetch(game, wrong_banner=wrong, **goal_kwargs)
    scores = []
    for index in range(4):
        with torch.no_grad():
            out = episode(game=game, command=None, goal_event=goal.detach(),
                          seed=args.seed + 700_000 + index, sample=False,
                          wrong_banner=wrong, random_actions=rand)
        scores.append(mastery(
            {"total_reward": out["reward"].sum(dim=1), "mask": out["mask"]},
            train[game]))
    return round(float(torch.tensor(scores).mean()), 4)


# No-agent control FIRST (weakness 18): the gate must be able to fail.
report["no_agent"] = {n: score(g, random_actions=True) for g, n in enumerate(NAMES)}
report["mastery"] = {n: score(g) for g, n in enumerate(NAMES)}
report["cross_fed"] = {n: score(g, force=1 - g) for g, n in enumerate(NAMES)}
report["decoy"] = {n: score(g, decoy=True) for g, n in enumerate(NAMES)}
report["label_swap"] = {n: score(g, wrong_banner=True) for g, n in enumerate(NAMES)}

# Decoy behaviour-difference: with identical noise fragments, do the twins
# behave identically (up to the banner pixels)? Same seed, both games.
with torch.no_grad():
    goal, _ = fetch(0, decoy=True)
    a = episode(game=0, command=None, goal_event=goal, seed=args.seed + 800_000,
                sample=False)
    b = episode(game=1, command=None, goal_event=goal, seed=args.seed + 800_000,
                sample=False)
agreement = float((a["actions"] == b["actions"]).float().mean())
report["decoy_action_agreement"] = round(agreement, 4)

report["cue_choice"] = {n: fetch(g)[1] for g, n in enumerate(NAMES)}
print(json.dumps(report))
