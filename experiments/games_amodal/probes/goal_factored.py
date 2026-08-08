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
from neural_computer import AmodalEvent, ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--competence-updates", type=int, default=1500)
parser.add_argument("--game-updates", type=int, default=800)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--max-restarts", type=int, default=4)
parser.add_argument("--decoy-draws", type=int, default=8)
parser.add_argument(
    "--reward-feedback", action="store_true",
    help="feed reward back into the controller (default OFF: it is a "
         "second context channel and a measured necessity leak)")
parser.add_argument(
    "--only-command", type=int, default=-1,
    help="phase-1 isolation: train on a single command (0 or 1) instead "
         "of mixing; separates per-command learnability from interference")
args = parser.parse_args()

torch.manual_seed(args.seed)
train, _ = twins_suite()
NAMES = [c.name for c in train]


def build_plant(attempt: int):
    """(Re)draw the plant and command vocabulary for one phase-1 attempt."""
    global agent, plant, decoder, COMMANDS
    torch.manual_seed(args.seed + 7000 * attempt)
    agent = SharedControllerAgent(
        event_width=args.width, intention_width=32, feedback_width=16,
        hidden=32, event_window_capacity=8, shared_drivers=True,
    )
    plant = list(trainable_parameters(
        [agent.controller, *agent.game_modules(agent.games[0])]
    ))
    decoder = agent.runtime.output_bus.decoders["keypress"]
    vocab_generator = torch.Generator().manual_seed(
        args.seed + 999 + 7000 * attempt)
    COMMANDS = torch.nn.Parameter(
        torch.randn(2, args.width, generator=vocab_generator))
    project_commands()
    return plant
# Action index -> grid delta, matching the decoder's key order (as in
# cotrained.py's test_action).
DELTAS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])

# Command vocabulary: TRAINABLE in phase 1, frozen after (created inside
# build_plant so each restart redraws it). Frozen random vectors were
# measured unreadable; orthogonal projection keeps the pair separated.
generator = torch.Generator().manual_seed(args.seed + 999)


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
            # PER-ROW event: COMMANDS[command] is [batch, width], one
            # event whose payload differs by row. artifact_events would
            # treat dim 0 as TOKEN COUNT and broadcast all rows' commands
            # to every row -- 32 tokens flooding a capacity-8 window, so
            # no row could see its own command (measured: competence at
            # chance through every phase-1 iteration before this fix).
            events.append(AmodalEvent(payload=COMMANDS[command]))
        if goal_event is not None:
            events.append(AmodalEvent(
                payload=goal_event.reshape(1, -1).expand(args.batch_size, -1)))
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
        step_self_reward = None
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
            # Idle cost, the family's own DUAL_IDLE_COST lesson in phase-1
            # form: measured, the mixed-trained plant satisfied cmd1 by
            # INHIBITION -- 1493/1536 steps idle, consuming nothing --
            # because idleness earned exactly 0 and generated no learning
            # contrast. Engagement must pay even under ignorance, so a
            # step that consumes neither plane now costs a little.
            engaged = planes.sum(dim=-1).clamp(0.0, 1.0)
            step_self_reward = (
                commanded - other - 0.1 * (1.0 - engaged)
            ).clamp(-1.0, 1.0)
            selfr.append(step_self_reward)
        else:
            selfr.append(torch.zeros(args.batch_size))
        alive = outcome.alive
        # Phase 1 is verifier-free: its feedback channel must carry the
        # SELF-reward, not the verifier's. Piping choiceA's reward into
        # the controller input meant a cmd1-compliant agent received -1
        # feedback on every correct consumption while cmd0 rows received
        # +1 -- the two command groups lived in systematically different
        # input worlds, and per-command advantage centering measurably
        # did not remove the resulting execution asymmetry (cmd0 1.0,
        # cmd1 0.375-0.667).
        # Reward feedback is a SECOND context channel, and measurably a
        # leak. Phase 1 fed self-reward back, which taught the plant to
        # self-correct ("felt -1, switch planes"); at gate time the
        # verifier's reward plays the same role, so under a noise
        # fragment the agent recovers the twin from its own consequences
        # -- decoy choiceB 1.000 and action-agreement 0.29 on seed 69317,
        # F48's working-memory leak in the frozen-plant setting. In a
        # CUED rung the banner is the context channel by design, so the
        # honest configuration carries no reward feedback at all: with
        # `--reward-feedback` off the bank is the only route to the twin.
        feedback_reward = (
            torch.zeros(args.batch_size) if not args.reward_feedback
            else step_self_reward if step_self_reward is not None
            else outcome.reward)
        feedback = ControllerFeedback(
            action=agent.feedback_encoders["keypress"](acts),
            reward=feedback_reward, propensity=propensity,
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
    if command is not None:
        # Centre advantage WITHIN each command group. A shared baseline
        # lets the better-learned command's returns push the other
        # command's correct actions into negative advantage -- measured
        # as cmd0 followed perfectly while cmd1 stalls at 0.18-0.29 on
        # both seeds, invisible in a pooled curve that cmd0 rows dominate.
        for value in (0, 1):
            rows = command == value
            if bool(rows.any()):
                group_mask = mask[rows]
                mean = (advantage[rows] * group_mask).sum() / group_mask.sum().clamp_min(1)
                advantage[rows] = advantage[rows] - mean
    else:
        advantage = advantage - (advantage * mask).sum() / mask.sum().clamp_min(1)
    return {
        "reward": reward, "self_reward": self_reward, "mask": mask,
        "advantage": advantage, "logp": torch.stack(logps, dim=1),
        "actions": torch.stack(actions_trace, dim=1),
        "command_used": command,
    }


report: dict = {"seed": args.seed}

def competence_score(command_value: int) -> float:
    with torch.no_grad():
        out = episode(game=None,
                      command=torch.full((args.batch_size,), command_value),
                      goal_event=None, seed=args.seed + 400_000, sample=False)
    correct = (out["self_reward"] > 0).float().sum()
    wrong = (out["self_reward"] < 0).float().sum()
    return float(correct / (correct + wrong).clamp_min(1.0))


# ---- Phase 1: edge-competence, verifier-free -------------------------------
# Restart on measured collapse. On some seeds every JOINT sampling
# scheme fails (per-row 50/50, per-row laggard, alternation, whole-batch
# laggard all measured failing on 69316) while single-command isolation
# converges 3/3 -- the basin drawn at init decides, so the honest
# mechanism is to detect the dead branch early, redraw, and REPORT the
# number of draws. Not hidden: `phase1_attempts` is in the output.
for attempt in range(args.max_restarts):
    build_plant(attempt)
    phase1_params = plant + [COMMANDS]
    optimizer = torch.optim.Adam(phase1_params, lr=1e-3)
    competence_curve = []
    command_score = [0.5, 0.5]
    collapsed = False
    for update in range(args.competence_updates):
        if args.only_command >= 0:
            command = torch.full((args.batch_size,), args.only_command)
        else:
            # WHOLE-BATCH laggard-sampled episodes -- the promoted twin
            # recipe (F10/F23) transplanted exactly, after everything else
            # was measured and failed on seed 69316: per-row 50/50 collapses
            # to an unconditional A-machine (cmd1 0.010); per-row laggard
            # weighting does not rescue it; deterministic whole-batch
            # alternation does not either, which F18/F25 predicts (commands
            # are a conflict group, and fixed rotation is sequencing at
            # period 2). What the promoted rung actually does is adaptive
            # dwell: each episode commits the whole batch to one member,
            # sampled toward the laggard with a floor for the leader.
            weights = torch.tensor(
                [-command_score[0], -command_score[1]]) / 0.25
            probs = torch.softmax(weights, dim=-1) * 0.5 + 0.25
            command = torch.full(
                (args.batch_size,),
                int(torch.multinomial(probs, 1)))
        out = episode(game=None, command=command, goal_event=None,
                      seed=args.seed + update, sample=True)
        with torch.no_grad():
            for value in (0, 1):
                rows = out["command_used"] == value
                if bool(rows.any()):
                    correct = (out["self_reward"][rows] > 0).float().sum()
                    wrong = (out["self_reward"][rows] < 0).float().sum()
                    ratio = float(correct / (correct + wrong).clamp_min(1.0))
                    command_score[value] = (
                        0.9 * command_score[value] + 0.1 * ratio)
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
            # Per-command ratios: the pooled ratio is dominated by whichever
            # command consumes more and hid a 0.2-vs-1.0 execution asymmetry.
            pair = []
            for value in (0, 1):
                rows = probe["command_used"] == value
                correct = (probe["self_reward"][rows] > 0).float().sum()
                wrong = (probe["self_reward"][rows] < 0).float().sum()
                pair.append(round(float(
                    correct / (correct + wrong).clamp_min(1.0)), 3))
            competence_curve.append(pair)
        if (update + 1) == max(500, args.competence_updates // 2):
            # Calibrated on measured curves: at update 500 a converging
            # draw reads ~1.0 per command and a dead branch ~0.01. The
            # first version checked at 250 with threshold 0.2 and killed
            # the WINNING seed-1 draw (0.087 at 250 under the strict
            # meter) -- both promotion seeds then burned all four draws
            # on a detector that culled healthy basins.
            if min(command_score) < 0.3:
                collapsed = True
                break
    final_scores = [competence_score(0), competence_score(1)]
    if not collapsed and min(final_scores) >= 0.8:
        break

report_attempts = attempt + 1
COMMANDS.requires_grad_(False)



report["competence"] = {
    "cmd0": round(competence_score(0), 4), "cmd1": round(competence_score(1), 4)
}
report["competence_curve"] = competence_curve
report["phase1_attempts"] = report_attempts
with torch.no_grad():
    report["command_cosine"] = round(float(
        (COMMANDS[0] @ COMMANDS[1])
        / (COMMANDS[0].norm() * COMMANDS[1].norm()).clamp_min(1e-6)), 4)

# ---- Phase 2: cued game phase — gradients into fragments + cue-reader only -
# Destinations are expressed in the plant's goal vocabulary (the design
# doc's contract), so the fragments START from the learned command
# vectors plus noise. Measured need: from random init, REINFORCE through
# the frozen plant finds one twin's target and half-finds the other
# (mastery 1.0 / 0.43-0.51 on both seeds) even though the bridge proves
# a perfect fragment exists for both. Phase 2's real job is the
# cue -> destination ASSIGNMENT, which stays fully learned.
with torch.no_grad():
    fragments.copy_(
        COMMANDS.detach() + 0.5 * torch.randn(
            2, args.width, generator=generator))
for parameter in plant:
    parameter.requires_grad_(False)
bank_params = [fragments] + list(cue_reader.parameters())
optimizer = torch.optim.Adam(bank_params, lr=3e-3)


def fetch(observation_free_game: int, *, decoy: bool = False,
          force: int | None = None, wrong_banner: bool = False,
          decoy_draw: int = 0):
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
        # `decoy_draw` selects WHICH noise vector. A single fixed draw is
        # a sample size of one: a lucky direction that happens to point
        # plane-B-ward scores 0.758 on choiceB and reads as a failed
        # necessity gate. The gate averages over draws instead.
        noise = torch.randn(
            1, args.width,
            generator=torch.Generator().manual_seed(7 + 100 * decoy_draw))
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
    # A decoy is an expectation over noise directions, not one draw.
    draws = args.decoy_draws if goal_kwargs.get("decoy") else 1
    scores = []
    for draw in range(draws):
        goal, _ = fetch(game, wrong_banner=wrong, decoy_draw=draw,
                        **goal_kwargs)
        for index in range(4):
            with torch.no_grad():
                out = episode(game=game, command=None,
                              goal_event=goal.detach(),
                              seed=args.seed + 700_000 + index, sample=False,
                              wrong_banner=wrong, random_actions=rand)
            scores.append(mastery(
                {"total_reward": out["reward"].sum(dim=1), "mask": out["mask"]},
                train[game]))
    return round(float(torch.tensor(scores).mean()), 4)


# Bridge test: verifier mastery with the RAW command vectors injected per
# game. Separates the layers: if cmd0-on-choiceA is high while
# cmd1-on-choiceB is low, phase 2's failure is the plane-B execution gap
# and the fragments are innocent; if both diagonals are high, the failure
# is fragment/assignment learning.
report["bridge"] = {}
for game_index, name in enumerate(NAMES):
    for cmd in (0, 1):
        scores = []
        for index in range(4):
            with torch.no_grad():
                out = episode(game=game_index, command=None,
                              goal_event=COMMANDS[cmd].detach(),
                              seed=args.seed + 750_000 + index, sample=False)
            scores.append(mastery(
                {"total_reward": out["reward"].sum(dim=1),
                 "mask": out["mask"]}, train[game_index]))
        report["bridge"][f"{name}<-cmd{cmd}"] = round(
            float(torch.tensor(scores).mean()), 4)

# No-agent control FIRST (weakness 18): the gate must be able to fail.
report["no_agent"] = {n: score(g, random_actions=True) for g, n in enumerate(NAMES)}
report["mastery"] = {n: score(g) for g, n in enumerate(NAMES)}
report["cross_fed"] = {n: score(g, force=1 - g) for g, n in enumerate(NAMES)}
report["decoy"] = {n: score(g, decoy=True) for g, n in enumerate(NAMES)}
report["label_swap"] = {n: score(g, wrong_banner=True) for g, n in enumerate(NAMES)}

# Decoy behaviour-difference: with identical noise fragments, do the twins
# behave identically (up to the banner pixels)? Same seed, both games.
agreements = []
for draw in range(args.decoy_draws):
    with torch.no_grad():
        goal, _ = fetch(0, decoy=True, decoy_draw=draw)
        a = episode(game=0, command=None, goal_event=goal,
                    seed=args.seed + 800_000, sample=False)
        b = episode(game=1, command=None, goal_event=goal,
                    seed=args.seed + 800_000, sample=False)
    agreements.append(float((a["actions"] == b["actions"]).float().mean()))
report["decoy_action_agreement"] = round(
    float(torch.tensor(agreements).mean()), 4)

report["cue_choice"] = {n: fetch(g)[1] for g, n in enumerate(NAMES)}
checkpoint = {
    "plant": {f"p{i}": p.detach().clone() for i, p in enumerate(plant)},
    "commands": COMMANDS.detach().clone(),
    "fragments": fragments.detach().clone(),
    "cue_reader": cue_reader.state_dict(),
}
torch.save(checkpoint, f"goal_factored_ckpt_{args.seed}.pt")
print(json.dumps(report))
