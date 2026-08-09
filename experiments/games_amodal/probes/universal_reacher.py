"""Universal reacher: from any state X, reach any state Y.

F58 showed the executor failed because a SMALL goal set is memorisable:
with a handful of goals "always do the one thing" is competitive, and
under isolation it is optimal, so the plant never reads its goal
channel. Every fix aimed at the optimiser (budget, capacity, EWC
strength, curriculum) left the same signature -- one goal at 1.00, the
rest at 0.02.

The remedy is task design, not optimisation: a goal space too large to
memorise makes reading the instruction the only representable solution.
Here the goal is a TARGET CELL on the grid, encoded from normalised
coordinates so nearby targets are nearby in goal space, and every row of
every batch gets its own target.

Verifier-free throughout: the reward is progress toward the commanded
cell, computed from the observation the agent already sees. No game
rules, no scores, no stored data.

Gates (pre-registered in GOAL_FACTORED_DESIGN.md, revision 2026-08-09):
  no_agent      random actions -- the floor, run FIRST
  reach         fraction of rows arriving at the commanded cell
  final_gap     mean remaining distance (graded, unlike reach)
  CONDITIONING  same start states, two DIFFERENT target sets: the action
                distributions must diverge. This is the property every
                previous executor failed and none of them measured
                directly -- task score cannot distinguish "follows the
                goal" from "has a good habit".
  heldout       targets in a quadrant never commanded during training
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import torch

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
from neural_computer import AmodalEvent, ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--updates", type=int, default=1200)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--hidden", type=int, default=32)
parser.add_argument("--eval-batches", type=int, default=4)
parser.add_argument("--view", choices=("none", "roll", "crop"), default="none",
                    help="egocentric view: the avatar sits at the centre, so "
                         "absolute position stops mattering and the encoder "
                         "only has to convey the target's RELATIVE offset. "
                         "A linear probe reads absolute position out of the "
                         "allocentric encoder at 0.69/axis -- present but "
                         "~30%% wrong, which is 'approaches, misses by a cell'.")
parser.add_argument(
    "--dense-goal", action="store_true",
    help="old encoding: two normalised floats. Default is one-hot per "
         "axis, matching the numeric reacher that works.")
parser.add_argument(
    "--localise", type=float, default=0.0,
    help="auxiliary self-supervised loss: predict the avatar's own cell "
         "from the screen encoding. A linear probe reads position out of "
         "the encoder at only 0.69/axis, and REINFORCE gives almost no "
         "signal to sharpen it -- so the reacher descends the gradient "
         "well (gap 5.4 -> 2.0) and cannot land. The target is visible in "
         "the observation, so this needs no verifier.")
parser.add_argument(
    "--relative-goal", action="store_true",
    help="encode the goal as the offset (target - avatar) measured ONCE at "
         "episode start, not as an absolute cell. Egocentric views destroy "
         "absolute position, so pairing them with an absolute goal removes "
         "the very information needed to read the instruction -- measured: "
         "roll/crop with an absolute goal are WORSE than allocentric "
         "(reach 0.21/0.26 vs 0.32). Offsets are translation-invariant, "
         "which is also the natural instruction for a reacher; the agent "
         "must still integrate its own motion to track what remains.")
parser.add_argument(
    "--holdout-quadrant", action="store_true", default=True,
    help="reserve the bottom-right quadrant of targets for the held-out gate")
args = parser.parse_args()

torch.manual_seed(args.seed)
CONFIG = FamilyConfig(navigate=True, name="nav")
GRID = 8
DELTAS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])

agent = SharedControllerAgent(
    event_width=args.width, intention_width=32, feedback_width=16,
    hidden=args.hidden, event_window_capacity=8, shared_drivers=True,
)
plant = list(trainable_parameters(
    [agent.controller, *agent.game_modules(agent.games[0])]))
decoder = agent.runtime.output_bus.decoders["keypress"]
# Goal encoder: normalised (row, col) -> event payload. Structured rather
# than a lookup table, so "reach (3,4)" and "reach (3,5)" are near each
# other and unseen cells are interpolations rather than blanks.
# Goal encoding. Two normalised floats through a linear layer is a very
# entangled code -- the numeric reacher, which WORKS, feeds a one-hot
# (20 -> 64). One-hot row + one-hot col (16 -> 64) gives the grid the
# same clarity of instruction.
goal_encoder = torch.nn.Linear(
    2 if args.dense_goal else 2 * (2 * GRID - 1), args.width)
# Auxiliary localisation heads: encoding -> own row, own col.
locate_row = torch.nn.Linear(args.width, GRID)
locate_col = torch.nn.Linear(args.width, GRID)
params = (plant + list(goal_encoder.parameters())
          + list(locate_row.parameters()) + list(locate_col.parameters()))
optimizer = torch.optim.Adam(params, lr=1e-3)


def distance_field(walls: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TRUE shortest-path distance to `target`, routing around walls.

    Manhattan distance is WRONG here and it was silently training the
    agent to fail: the navigate map has a solid column with one gap, so
    21 of 64 cells need a detour, and a Manhattan progress reward pays
    NEGATIVE for the only moves that reach them. The agent learned
    exactly that -- approach the wall, then stall (gap ~2).
    """
    field = torch.full((GRID, GRID), float(GRID * GRID))
    blocked = {(int(r), int(c)) for r, c in walls.nonzero()}
    start = (int(target[0]), int(target[1]))
    if start in blocked:
        return field
    field[start] = 0.0
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < GRID and 0 <= nc < GRID):
                continue
            if (nr, nc) in blocked or field[nr, nc] < GRID * GRID:
                continue
            field[nr, nc] = field[row, col] + 1.0
            queue.append((nr, nc))
    return field


def true_gaps(raw: torch.Tensor, fields, cells: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        fields[row][int(cells[row, 0]), int(cells[row, 1])]
        for row in range(cells.shape[0])])


def viewed(grid: torch.Tensor) -> torch.Tensor:
    if args.view == "roll":
        return egocentric_view(grid)
    if args.view == "crop":
        return egocentric_crop(grid)
    return grid


def avatar_cells(observation: torch.Tensor) -> torch.Tensor:
    """[batch, 2] row/col of the avatar (plane 0)."""
    flat = observation[:, 0].reshape(observation.shape[0], -1)
    index = flat.argmax(dim=-1)
    return torch.stack([index // GRID, index % GRID], dim=-1)


def encode(targets: torch.Tensor, *, span: int = GRID) -> torch.Tensor:
    if args.dense_goal:
        features = targets.float() / (span - 1)
    else:
        width = 2 * GRID - 1
        features = torch.zeros(targets.shape[0], 2 * width)
        features.scatter_(1, targets[:, :1].clamp(0, width - 1), 1.0)
        features.scatter_(1, targets[:, 1:].clamp(0, width - 1) + width, 1.0)
    payload = goal_encoder(features)
    return payload / payload.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 4.0


def sample_targets(batch: int, generator, *, heldout: bool) -> torch.Tensor:
    """Training targets avoid the reserved quadrant; held-out targets are
    drawn only from it."""
    rows, cols = [], []
    while len(rows) < batch:
        r = int(torch.randint(0, GRID, (1,), generator=generator))
        c = int(torch.randint(0, GRID, (1,), generator=generator))
        reserved = args.holdout_quadrant and r >= GRID // 2 and c >= GRID // 2
        if reserved == heldout:
            rows.append(r)
            cols.append(c)
    return torch.stack([torch.tensor(rows), torch.tensor(cols)], dim=-1)


def rollout(targets: torch.Tensor, *, seed: int, sample: bool,
            random_actions: bool = False):
    verifier = FamilyVerifier(CONFIG, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    state = agent.controller.initial_state(args.batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(args.batch_size, agent.controller.feedback_width),
        reward=torch.zeros(args.batch_size),
        propensity=torch.ones(args.batch_size),
        has_feedback=torch.zeros(args.batch_size))
    rng = torch.Generator().manual_seed(seed + 5)
    start_cells = avatar_cells(
        pad_channels(verifier.observation(), SHARED_SCREEN_CHANNELS))
    goal_payload = encode(
        (targets - start_cells + (GRID - 1)) if args.relative_goal else targets,
        span=(2 * GRID - 1) if args.relative_goal else GRID)
    rewards, logps, actions, arrived = [], [], [], torch.zeros(args.batch_size)
    locate_terms = []
    raw = pad_channels(verifier.observation(), SHARED_SCREEN_CHANNELS)
    observation = viewed(raw)
    # Targets must be REACHABLE. Uniform sampling puts ~11% of them
    # inside the wall column, where the distance field is the
    # unreachable sentinel -- that alone dominated the mean gap (13.2 in
    # sentinel units, not comparable with the old Manhattan 2.09) and
    # collapsed one seed to a fully habitual policy (agreement 1.000).
    targets = targets.clone()
    for row in range(args.batch_size):
        if raw[row, 2, int(targets[row, 0]), int(targets[row, 1])] > 0:
            free = (raw[row, 2] == 0).nonzero()
            pick = int(torch.randint(0, free.shape[0], (1,),
                                     generator=torch.Generator().manual_seed(
                                         seed * 131 + row)))
            targets[row] = free[pick]
    fields = [distance_field(raw[row, 2], targets[row])
              for row in range(args.batch_size)]
    gap = true_gaps(raw, fields, avatar_cells(raw))
    start_gap = gap.clone()
    for _step in range(args.steps):
        screen_event = agent.runtime.encoders["screen"](observation)
        if args.localise > 0.0:
            payload = getattr(screen_event, "payload", screen_event)
            here = avatar_cells(raw)
            locate_loss = (
                torch.nn.functional.cross_entropy(locate_row(payload), here[:, 0])
                + torch.nn.functional.cross_entropy(locate_col(payload), here[:, 1]))
            locate_terms.append(locate_loss)
        events = [screen_event, AmodalEvent(payload=goal_payload)]
        output, state = agent.runtime.step_events(events, state, feedback)
        if random_actions:
            acts = torch.randint(0, decoder.key_count, (args.batch_size,),
                                 generator=rng)
            logps.append(torch.zeros(args.batch_size))
        else:
            decision = decoder.decide_from_logits(
                output.decoded["keypress"], sample=sample)
            acts = decision.key_index
            logps.append(decision.propensity.clamp_min(1e-8).log())
        actions.append(acts)
        verifier.step(acts)
        raw = pad_channels(verifier.observation(), SHARED_SCREEN_CHANNELS)
        observation = viewed(raw)
        new_gap = true_gaps(raw, fields, avatar_cells(raw))
        # Progress reward, self-computed: closer is better, arriving pays.
        step_reward = (gap - new_gap) + 2.0 * (new_gap == 0).float()
        arrived = torch.maximum(arrived, (new_gap == 0).float())
        rewards.append(step_reward)
        gap = new_gap
        feedback = ControllerFeedback(
            action=agent.feedback_encoders["keypress"](acts),
            reward=torch.zeros(args.batch_size),
            propensity=torch.ones(args.batch_size),
            has_feedback=torch.ones(args.batch_size))
        state = state.detached() if sample else state
    matrix = torch.stack(rewards, dim=1)
    running = torch.zeros(args.batch_size)
    returns = torch.zeros_like(matrix)
    for pos in range(matrix.shape[1] - 1, -1, -1):
        running = matrix[:, pos] + args.gamma * running
        returns[:, pos] = running
    return {"returns": returns, "logp": torch.stack(logps, dim=1),
            "actions": torch.stack(actions, dim=1),
            "arrived": arrived, "final_gap": gap, "start_gap": start_gap,
            "locate": (torch.stack(locate_terms).mean()
                       if locate_terms else torch.zeros(()))}


generator = torch.Generator().manual_seed(args.seed)
for update in range(args.updates):
    targets = sample_targets(args.batch_size, generator, heldout=False)
    out = rollout(targets, seed=args.seed + update, sample=True)
    advantage = out["returns"].detach()
    advantage = advantage - advantage.mean()
    advantage = advantage / advantage.std().clamp_min(1e-6)
    loss = -(advantage * out["logp"]).sum() / args.batch_size
    loss = loss + args.localise * out["locate"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

report = {"seed": args.seed, "updates": args.updates}


def measure(heldout: bool, *, rand: bool = False) -> dict[str, float]:
    reach, gaps = [], []
    for index in range(args.eval_batches):
        gen = torch.Generator().manual_seed(4242 + index)
        targets = sample_targets(args.batch_size, gen, heldout=heldout)
        with torch.no_grad():
            out = rollout(targets, seed=args.seed + 800_000 + index,
                          sample=False, random_actions=rand)
        reach.append(float(out["arrived"].mean()))
        gaps.append(float(out["final_gap"].mean()))
    return {"reach": round(sum(reach) / len(reach), 4),
            "final_gap": round(sum(gaps) / len(gaps), 4)}


report["no_agent"] = measure(False, rand=True)
report["trained_targets"] = measure(False)
report["heldout_targets"] = measure(True)

# CONDITIONING: identical start states, two different target sets. If the
# plant reads its goal, the action streams must diverge; a habit gives
# near-identical actions whatever it was told.
agree = []
for index in range(args.eval_batches):
    first = sample_targets(args.batch_size,
                           torch.Generator().manual_seed(11 + index),
                           heldout=False)
    second = sample_targets(args.batch_size,
                            torch.Generator().manual_seed(500 + index),
                            heldout=False)
    with torch.no_grad():
        a = rollout(first, seed=args.seed + 900_000 + index, sample=False)
        b = rollout(second, seed=args.seed + 900_000 + index, sample=False)
    agree.append(float((a["actions"] == b["actions"]).float().mean()))
report["conditioning_action_agreement"] = round(sum(agree) / len(agree), 4)

print(json.dumps(report, indent=1))
