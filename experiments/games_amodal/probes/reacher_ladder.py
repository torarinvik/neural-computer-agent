"""Reacher ladder: find the exact step where competence breaks.

F59/F60 left a large gap with four differences stacked inside it --
numeric reaches 0.938, the grid 0.379 even with ORACLE perception. The
ladder varies one axis at a time so the break point is localised
instead of guessed at (seven perception-side guesses have now failed).

  r1  line, 2 actions          the known-good surface
  r2  line, 4 actions          + a larger action set (2 are no-ops)
  r3  open grid, 4 actions     + a second dimension
  r4  walled grid, 4 actions   + an obstacle needing a detour

Everything else is held fixed and set to its best measured form: oracle
state (one-hot of own cell), relative goals one-hot per axis, and true
shortest-path progress reward computed by BFS through the actual
obstacle layout. No screen encoder -- F60 showed perception is not the
constraint, so it is removed as a variable.

Each rung reports reach against its OWN measured no-agent floor, plus
path optimality (steps taken / true shortest path) which is only
meaningful because the simulator knows the true distance.

The ladder also exists to serve the compounding gate: with `--warm-start
<rung>` the plant is pre-trained on a lower rung first, so acquisition
cost on the higher rung can be compared with and without it. A flat
comparison is a real result and would say the controller carries
nothing across these steps.
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import torch

from experiments.games_amodal.shared_controller import SharedControllerAgent
from neural_computer import AmodalEvent, ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--rung", choices=("r1", "r2", "r3", "r4"), default="r1")
parser.add_argument("--warm-start", choices=("", "r1", "r2", "r3"), default="")
parser.add_argument("--updates", type=int, default=1500)
parser.add_argument("--warm-updates", type=int, default=1500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--hidden", type=int, default=32)
parser.add_argument("--size", type=int, default=8)
parser.add_argument("--eval-batches", type=int, default=4)
parser.add_argument(
    "--adapt-decoder", action="store_true",
    help="widen the adaptation channel: adapt the OUTPUT HEAD as well as "
         "the goal encoding. F64 measured the frozen prior carrying real "
         "competence (0.520 zero-shot vs 0.172 floor) that 800 updates "
         "of goal-adapter training could only lift to 0.613. A goal "
         "adapter re-maps what a goal MEANS but cannot change what the "
         "frozen policy DOES with it; the head can.")
parser.add_argument(
    "--sparse", action="store_true",
    help="reward ONLY on arrival -- no progress shaping. F62 found every "
         "transfer test in this project used targets that are cheap cold "
         "(r4 masters at 200 updates even at size 16), leaving no "
         "headroom for a prior to pay for itself, so positive transfer "
         "was undetectable by construction. Sparse reward makes arrival "
         "something the agent must DISCOVER, which is precisely where "
         "knowing how to move toward goals should help.")
parser.add_argument(
    "--warm-mix", default="",
    help="comma-separated rungs to warm-start on CONCURRENTLY (e.g. "
         "'r1,r3'). F61: warming on ONE domain leaves the plant holding "
         "that domain's pursuit policy, which a goal adapter cannot "
         "repair (frozen 0.211, trainable 0.277, cold 0.996). Route 1 is "
         "to never let it specialise -- if no single domain can become "
         "the prior, only machinery common to all of them survives.")
parser.add_argument(
    "--freeze-plant", action="store_true",
    help="after the warm-start rung, FREEZE the controller and adapt only "
         "a small per-rung goal adapter. Warm-starting with a trainable "
         "plant measured NEGATIVE transfer (r1 -> r4: 0.535 vs 0.996 cold, "
         "5x the updates) -- the plant carries the line's 'hold one "
         "direction' as a habit. This is the architecture's own answer: "
         "generic machinery frozen in the plant, adaptation in the bank.")
parser.add_argument("--mastery", type=float, default=0.8,
                    help="reach level counted as mastered for the cost curve")
args = parser.parse_args()

torch.manual_seed(args.seed)
N = args.size
SPEC = {
    "r1": {"dims": 1, "actions": 2, "walls": False},
    "r2": {"dims": 1, "actions": 4, "walls": False},
    "r3": {"dims": 2, "actions": 4, "walls": False},
    "r4": {"dims": 2, "actions": 4, "walls": True},
}
# 1D lives in a single ROW, so movement must be along the COLUMN axis.
# The first version moved along rows, which walks off a one-row grid --
# the agent could not move at all and sat exactly at floor.
MOVES_1D = [(0, -1), (0, 1), (0, 0), (0, 0)]
MOVES_2D = [(-1, 0), (0, 1), (1, 0), (0, -1)]

agent = SharedControllerAgent(
    event_width=args.width, intention_width=32, feedback_width=16,
    hidden=args.hidden, event_window_capacity=8, shared_drivers=True)
decoder = agent.runtime.output_bus.decoders["keypress"]
state_encoder = torch.nn.Linear(N * N, args.width)
goal_encoder = torch.nn.Linear(2 * (2 * N - 1), args.width)
params = list({id(p): p for p in (
    list(agent.controller.parameters()) + list(decoder.parameters())
    + list(state_encoder.parameters()) + list(goal_encoder.parameters()))}.values())
optimizer = torch.optim.Adam(params, lr=1e-3)
# Per-rung adapter -- the bank's role: a small module that re-maps the
# goal for a new world while the plant stays fixed.
adapter = torch.nn.Linear(args.width, args.width)
with torch.no_grad():
    adapter.weight.copy_(torch.eye(args.width))
    adapter.bias.zero_()


def wall_cells(spec) -> set[tuple[int, int]]:
    """A single column with one gap -- the same obstacle shape the grid
    reacher faced, so r4 is comparable with the earlier measurements."""
    if not spec["walls"]:
        return set()
    return {(r, N // 2) for r in range(1, N)}


def encode(features: torch.Tensor, encoder, size: int) -> torch.Tensor:
    payload = encoder(features)
    return payload / payload.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 4.0


def one_hot(values: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(values.shape[0], size)
    out.scatter_(1, values.clamp(0, size - 1).unsqueeze(-1), 1.0)
    return out


def distance_field(target: tuple[int, int], walls, spec) -> dict:
    rows = N if spec["dims"] == 2 else 1
    field = {}
    if target in walls:
        return field
    field[target] = 0
    queue = deque([target])
    moves = MOVES_2D if spec["dims"] == 2 else MOVES_1D[:2]
    while queue:
        cell = queue.popleft()
        for dr, dc in moves:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < N):
                continue
            if nxt in walls or nxt in field:
                continue
            field[nxt] = field[cell] + 1
            queue.append(nxt)
    return field


def rollout(spec, *, seed: int, sample: bool, random_actions: bool = False):
    walls = wall_cells(spec)
    rows = N if spec["dims"] == 2 else 1
    generator = torch.Generator().manual_seed(seed)
    free = [(r, c) for r in range(rows) for c in range(N)
            if (r, c) not in walls]
    picks = torch.randint(0, len(free), (2, args.batch_size),
                          generator=generator)
    starts = [free[int(i)] for i in picks[0]]
    targets = [free[int(i)] for i in picks[1]]
    fields = [distance_field(t, walls, spec) for t in targets]
    positions = list(starts)
    state = agent.controller.initial_state(args.batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(args.batch_size, agent.controller.feedback_width),
        reward=torch.zeros(args.batch_size),
        propensity=torch.ones(args.batch_size),
        has_feedback=torch.zeros(args.batch_size))
    moves = MOVES_2D if spec["dims"] == 2 else MOVES_1D

    def gaps():
        return torch.tensor([float(fields[i].get(positions[i], N * N))
                             for i in range(args.batch_size)])

    gap = gaps()
    optimal = torch.tensor([float(fields[i].get(starts[i], N * N))
                            for i in range(args.batch_size)])
    rewards, logps = [], []
    arrived = torch.zeros(args.batch_size)
    steps_used = torch.full((args.batch_size,), float(args.steps))
    for step in range(args.steps):
        cells = torch.tensor([p[0] * N + p[1] for p in positions])
        offsets = torch.tensor(
            [[targets[i][0] - positions[i][0] + (N - 1),
              targets[i][1] - positions[i][1] + (N - 1)]
             for i in range(args.batch_size)])
        goal_features = torch.cat([
            one_hot(offsets[:, 0], 2 * N - 1),
            one_hot(offsets[:, 1], 2 * N - 1)], dim=-1)
        events = [
            AmodalEvent(payload=encode(one_hot(cells, N * N),
                                       state_encoder, N * N)),
            AmodalEvent(payload=(adapter(encode(goal_features, goal_encoder, N))
                                 if args.freeze_plant
                                 else encode(goal_features, goal_encoder, N))),
        ]
        output, state = agent.runtime.step_events(events, state, feedback)
        if random_actions:
            acts = torch.randint(0, spec["actions"], (args.batch_size,),
                                 generator=generator)
            logps.append(torch.zeros(args.batch_size))
        else:
            logits = output.decoded["keypress"][:, :spec["actions"]]
            distribution = torch.distributions.Categorical(logits=logits)
            acts = distribution.sample() if sample else logits.argmax(-1)
            logps.append(distribution.log_prob(acts))
        for i in range(args.batch_size):
            dr, dc = moves[int(acts[i])]
            nxt = (positions[i][0] + dr, positions[i][1] + dc)
            if (0 <= nxt[0] < rows and 0 <= nxt[1] < N and nxt not in walls):
                positions[i] = nxt
        new_gap = gaps()
        landed = (new_gap == 0) & (arrived == 0)
        steps_used = torch.where(landed, torch.full_like(steps_used,
                                                         float(step + 1)),
                                 steps_used)
        arrived = torch.maximum(arrived, (new_gap == 0).float())
        rewards.append(
            2.0 * (new_gap == 0).float() if args.sparse
            else (gap - new_gap) + 2.0 * (new_gap == 0).float())
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
            "arrived": arrived, "final_gap": gap,
            "steps_used": steps_used, "optimal": optimal}


def train(spec, updates: int, tag: str, curve: list, opt=None, mix=None):
    opt = opt or optimizer
    for update in range(updates):
        active = mix[update % len(mix)] if mix else spec
        out = rollout(active, seed=args.seed + update, sample=True)
        advantage = out["returns"].detach()
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
        loss = -(advantage * out["logp"]).sum() / args.batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in opt.param_groups for p in group["params"]], 1.0)
        opt.step()
        if (update + 1) % 100 == 0:
            with torch.no_grad():
                probe = rollout(active if mix else spec,
                                seed=args.seed + 700_000, sample=False)
            curve.append({"tag": tag, "update": update + 1,
                          "reach": round(float(probe["arrived"].mean()), 4)})


def measure(spec, *, rand: bool = False) -> dict:
    reach, ratios = [], []
    for index in range(args.eval_batches):
        with torch.no_grad():
            out = rollout(spec, seed=args.seed + 800_000 + index,
                          sample=False, random_actions=rand)
        reach.append(float(out["arrived"].mean()))
        hit = out["arrived"] > 0
        if bool(hit.any()):
            ratios.append(float((out["steps_used"][hit]
                                 / out["optimal"][hit].clamp_min(1.0)).mean()))
    return {"reach": round(sum(reach) / len(reach), 4),
            "path_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None}


spec = SPEC[args.rung]
report = {"seed": args.seed, "rung": args.rung, "warm_start": args.warm_start}
curve: list = []
target_optimizer = optimizer
if args.warm_mix:
    mix = [SPEC[name] for name in args.warm_mix.split(",")]
    train(mix[0], args.warm_updates, "warm", curve, mix=mix)
    report["warm_mix_final"] = {
        name: measure(SPEC[name]) for name in args.warm_mix.split(",")}
    if args.freeze_plant:
        # F61 froze after ONE domain, preserving a wrong policy. Freezing
        # after DIVERSE domains preserves a general one -- the untested
        # combination, and the cheap form of "reuse rather than relearn".
        for parameter in params:
            parameter.requires_grad_(False)
        adapted = list(adapter.parameters())
        if args.adapt_decoder:
            for parameter in decoder.parameters():
                parameter.requires_grad_(True)
            adapted += list(decoder.parameters())
        target_optimizer = torch.optim.Adam(adapted, lr=1e-2)
        report["adapted_params"] = sum(p.numel() for p in adapted)
elif args.warm_start:
    train(SPEC[args.warm_start], args.warm_updates, "warm", curve)
    report["warm_rung_final"] = measure(SPEC[args.warm_start])
    if args.freeze_plant:
        for parameter in params:
            parameter.requires_grad_(False)
        target_optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)
        report["adapted_params"] = sum(
            p.numel() for p in adapter.parameters())
report["no_agent"] = measure(spec, rand=True)
if args.warm_mix or args.warm_start:
    # ZERO-SHOT reuse: what the warm plant does on the target BEFORE any
    # target training. This is the purest transfer number and we had
    # never taken it -- every previous measurement confounded transfer
    # with re-learning.
    report["zero_shot"] = measure(spec)
train(spec, args.updates, "target", curve, target_optimizer)
report["final"] = measure(spec)
report["curve"] = curve
target_curve = [c for c in curve if c["tag"] == "target"]
mastered = next((c["update"] for c in target_curve
                 if c["reach"] >= args.mastery), None)
report["updates_to_mastery"] = mastered
print(json.dumps(report, indent=1))
