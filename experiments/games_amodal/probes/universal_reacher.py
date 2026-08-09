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

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier
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
goal_encoder = torch.nn.Linear(2, args.width)
params = plant + list(goal_encoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)


def avatar_cells(observation: torch.Tensor) -> torch.Tensor:
    """[batch, 2] row/col of the avatar (plane 0)."""
    flat = observation[:, 0].reshape(observation.shape[0], -1)
    index = flat.argmax(dim=-1)
    return torch.stack([index // GRID, index % GRID], dim=-1)


def encode(targets: torch.Tensor) -> torch.Tensor:
    normalised = targets.float() / (GRID - 1)
    payload = goal_encoder(normalised)
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
    goal_payload = encode(targets)
    rewards, logps, actions, arrived = [], [], [], torch.zeros(args.batch_size)
    observation = pad_channels(verifier.observation(), SHARED_SCREEN_CHANNELS)
    gap = (avatar_cells(observation) - targets).abs().sum(dim=-1).float()
    for _step in range(args.steps):
        events = [agent.runtime.encoders["screen"](observation),
                  AmodalEvent(payload=goal_payload)]
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
        observation = pad_channels(
            verifier.observation(), SHARED_SCREEN_CHANNELS)
        new_gap = (avatar_cells(observation) - targets).abs().sum(dim=-1).float()
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
            "arrived": arrived, "final_gap": gap}


generator = torch.Generator().manual_seed(args.seed)
for update in range(args.updates):
    targets = sample_targets(args.batch_size, generator, heldout=False)
    out = rollout(targets, seed=args.seed + update, sample=True)
    advantage = out["returns"].detach()
    advantage = advantage - advantage.mean()
    advantage = advantage / advantage.std().clamp_min(1e-6)
    loss = -(advantage * out["logp"]).sum() / args.batch_size
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
