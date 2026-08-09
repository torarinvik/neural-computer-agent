"""Numeric reacher: from number X, reach number Y.

The minimal instance of "given any state X, learn the fastest path to
make any state Y current". State is an integer, actions are -1/+1, the
optimal path length is exactly |Y - X|.

Why this testbed, given F58. Our executor fails and two explanations
are currently indistinguishable: it cannot CONDITION on a goal, or it
cannot PERCEIVE the grid. Here perception is nearly free, so the two
separate. It also buys three things the grid games cannot:

  * ground truth. d(X, Y) = |Y - X| is known, so "fastest path" becomes
    a measured ratio (steps taken / steps required), not a slogan. We
    have never been able to check optimality directly.
  * a clean memorisation test. Train on pairs inside a range, evaluate
    OUTSIDE it. Counting extrapolates; a lookup table does not.
  * a cheap acquisition-cost curve, since each (X, Y) pair is seconds.

It also exercises the amodal claim: same controller, a different
encoder, a modality with no spatial structure.

Self-supervised throughout -- the agent grades itself on progress
toward the commanded number. No verifier, nothing stored.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.shared_controller import SharedControllerAgent
from neural_computer import AmodalEvent, ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--updates", type=int, default=1500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--line", type=int, default=20, help="numbers 0..line-1")
parser.add_argument("--train-max", type=int, default=10,
                    help="training pairs stay below this; above it is held out")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--hidden", type=int, default=32)
parser.add_argument("--eval-batches", type=int, default=4)
args = parser.parse_args()

torch.manual_seed(args.seed)
agent = SharedControllerAgent(
    event_width=args.width, intention_width=32, feedback_width=16,
    hidden=args.hidden, event_window_capacity=8, shared_drivers=True,
)
decoder = agent.runtime.output_bus.decoders["keypress"]
# Two tiny encoders: "where I am" and "where I should be". One-hot in,
# event payload out -- a modality with no spatial structure, through the
# same amodal controller the screen games use.
state_encoder = torch.nn.Linear(args.line, args.width)
goal_encoder = torch.nn.Linear(args.line, args.width)
params = list({id(p): p for p in (
    list(agent.controller.parameters())
    + list(decoder.parameters())
    + list(state_encoder.parameters())
    + list(goal_encoder.parameters())
)}.values())
optimizer = torch.optim.Adam(params, lr=1e-3)
# Actions 0 and 1 move -1 and +1; the rest stand still, so the decoder's
# native four keys are reused unchanged.
MOVE = torch.tensor([-1, 1, 0, 0])


def encode(values: torch.Tensor, encoder) -> torch.Tensor:
    one_hot = torch.zeros(values.shape[0], args.line)
    one_hot.scatter_(1, values.clamp(0, args.line - 1).unsqueeze(-1), 1.0)
    payload = encoder(one_hot)
    return payload / payload.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 4.0


def sample_pairs(batch: int, generator, *, heldout: bool):
    low, high = (args.train_max, args.line) if heldout else (0, args.train_max)
    start = torch.randint(low, high, (batch,), generator=generator)
    target = torch.randint(low, high, (batch,), generator=generator)
    # A pair with nothing to do teaches nothing.
    same = start == target
    target = torch.where(same, (target + 1).clamp(max=high - 1), target)
    return start, target


def rollout(start, target, *, seed: int, sample: bool, random_actions=False):
    position = start.clone()
    state = agent.controller.initial_state(args.batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(args.batch_size, agent.controller.feedback_width),
        reward=torch.zeros(args.batch_size),
        propensity=torch.ones(args.batch_size),
        has_feedback=torch.zeros(args.batch_size))
    rng = torch.Generator().manual_seed(seed)
    goal_payload = encode(target, goal_encoder)
    rewards, logps, actions = [], [], []
    arrived = torch.zeros(args.batch_size)
    steps_used = torch.full((args.batch_size,), float(args.steps))
    gap = (position - target).abs().float()
    for step in range(args.steps):
        events = [AmodalEvent(payload=encode(position, state_encoder)),
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
        position = (position + MOVE[acts]).clamp(0, args.line - 1)
        new_gap = (position - target).abs().float()
        landed = (new_gap == 0) & (arrived == 0)
        steps_used = torch.where(landed, torch.full_like(steps_used,
                                                         float(step + 1)),
                                 steps_used)
        arrived = torch.maximum(arrived, (new_gap == 0).float())
        rewards.append((gap - new_gap) + 2.0 * (new_gap == 0).float())
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
            "actions": torch.stack(actions, dim=1), "arrived": arrived,
            "final_gap": gap, "steps_used": steps_used}


generator = torch.Generator().manual_seed(args.seed)
for update in range(args.updates):
    start, target = sample_pairs(args.batch_size, generator, heldout=False)
    out = rollout(start, target, seed=args.seed + update, sample=True)
    advantage = out["returns"].detach()
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    loss = -(advantage * out["logp"]).sum() / args.batch_size
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

report = {"seed": args.seed, "updates": args.updates,
          "line": args.line, "train_max": args.train_max}


def measure(heldout: bool, *, rand: bool = False) -> dict[str, float]:
    reach, gaps, ratios = [], [], []
    for index in range(args.eval_batches):
        gen = torch.Generator().manual_seed(4242 + index)
        start, target = sample_pairs(args.batch_size, gen, heldout=heldout)
        with torch.no_grad():
            out = rollout(start, target, seed=args.seed + 800_000 + index,
                          sample=False, random_actions=rand)
        reach.append(float(out["arrived"].mean()))
        gaps.append(float(out["final_gap"].mean()))
        optimal = (start - target).abs().float().clamp_min(1.0)
        hit = out["arrived"] > 0
        if bool(hit.any()):
            # Optimality: steps taken / steps required, on arrivals only.
            ratios.append(float(
                (out["steps_used"][hit] / optimal[hit]).mean()))
    return {"reach": round(sum(reach) / len(reach), 4),
            "final_gap": round(sum(gaps) / len(gaps), 4),
            "path_ratio": (round(sum(ratios) / len(ratios), 4)
                           if ratios else None)}


report["no_agent"] = measure(False, rand=True)
report["trained_range"] = measure(False)
report["heldout_range"] = measure(True)

agree = []
for index in range(args.eval_batches):
    gen = torch.Generator().manual_seed(11 + index)
    start, first = sample_pairs(args.batch_size, gen, heldout=False)
    _, second = sample_pairs(args.batch_size,
                             torch.Generator().manual_seed(500 + index),
                             heldout=False)
    with torch.no_grad():
        a = rollout(start, first, seed=args.seed + 900_000 + index, sample=False)
        b = rollout(start, second, seed=args.seed + 900_000 + index, sample=False)
    agree.append(float((a["actions"] == b["actions"]).float().mean()))
report["conditioning_action_agreement"] = round(sum(agree) / len(agree), 4)

print(json.dumps(report, indent=1))
