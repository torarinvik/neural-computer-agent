"""The games in the slot interface: a factored multi-object state.

F100 and F101 both failed on the battery's multi-step variants, and F101
localised the cause above the mechanism: the state was the avatar's cell
plus one screen frame, which is Markov-insufficient. `intercept` has
objects falling and `avoid` has hazards moving, so a single frame
carries no velocity or phase, and "safe now, lethal in two steps" is not
expressible. The nulls said it plainly — withholding the bank entry
scored the same as supplying it, where the one-step `dual` game gave
-0.100 reward for a stranger's entry (F99).

This gives the games the state F71-F98 was built around. Six slots of
eight values, read off the screen the learner already sees:

    slot 0,1   avatar row, column          (plane 0)
    slot 2,3   nearest POSITIVE object     (plane 1)
    slot 4,5   nearest NEGATIVE object     (plane 2)

A composigrid frame is exactly that shape, which is why this is the
direct experiment rather than a new idea: `schema_families.py` has
handled six slots of eight values since F71 and has never been fed game
objects.

Then the whole F67 architecture, over that state:

  * a TRANSITION model (state, action) -> next state. Falling objects
    now move IN the state, so their motion is predictable rather than
    invisible;
  * a REWARD model (state, action, entry) -> outcome, made world-
    specific by the bank entry, so an inverted twin reads the same
    geometry and gets the opposite answer;
  * BEAM SEARCH over the two, recomputed every step, scoring action
    sequences by predicted cumulative outcome.

Honest about the approximation: "nearest object" collapses several
objects into one, so a world with two fallers is only partly described.
This is a better state, not a complete one, and the measurement should
be read as testing whether MORE state helps — not whether this state is
sufficient.

Nulls as before: entry withheld, a stranger variant's entry, a plant
frozen at initialisation, and a measured random-action floor.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyConfig, FamilyVerifier, family_variants)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--context", type=int, default=128)
parser.add_argument("--train-updates", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--trials", type=int, default=16)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--beam", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--horizon", type=int, default=1,
    help="VALUE target: label a state-action by the DISCOUNTED RETURN "
         "over this many steps instead of its immediate outcome. F102 "
         "measured 98.16%% of immediate outcomes to be 'nothing', so an "
         "immediate target carries almost no gradient; a food event "
         "within the horizon colours every step that led to it.")
parser.add_argument("--discount", type=float, default=0.9)
parser.add_argument(
    "--balance-loss", action="store_true",
    help="weight the outcome classes inversely to their frequency. "
         "Without it, 'always nothing' scores 98.16%% and the 1.8%% that "
         "matters is invisible to cross-entropy (F102).")
parser.add_argument(
    "--seek", type=float, default=0.0,
    help="fraction of data-collection actions taken TOWARD the nearest "
         "positive object rather than uniformly at random. Uses only the "
         "slot state the learner already reads, no privileged "
         "information. Uniform random play is what made outcomes 1.8%% "
         "dense; seeking raises the density of the events worth "
         "learning from.")
parser.add_argument("--random-plant", action="store_true")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
ACTIONS, PLANES, HEIGHT, WIDTH = 4, 3, 8, 8
SLOTS, VALUES = 6, 8
ABSENT = VALUES  # a slot with no object; distinct from value 0


def variant_configs() -> list:
    out = []
    for base in family_variants(max_components=2, levels=(1, 2)):
        for inverted in (False, True):
            config = dataclasses.replace(
                base, inverted=inverted,
                name=f"{base.name}{'~' if inverted else ''}")
            try:
                config.validate()
            except ValueError:
                continue
            out.append(config)
    return out


def verifier_for(config: FamilyConfig, seed: int) -> FamilyVerifier:
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    return verifier


def slot_state(screen: torch.Tensor) -> torch.Tensor:
    """Screen -> six slots. Perception, and a shallow one: argmax for
    the avatar, nearest-by-Manhattan for each object plane."""
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    batch = frames.shape[0]
    out = torch.full((batch, SLOTS), ABSENT, dtype=torch.long)
    rows = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
    cols = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)
    for row in range(batch):
        avatar = frames[row, 0]
        if float(avatar.max()) <= 0:
            continue
        flat = int(avatar.reshape(-1).argmax())
        ar, ac = flat // WIDTH, flat % WIDTH
        out[row, 0], out[row, 1] = ar, ac
        for plane, base in ((1, 2), (2, 4)):
            mask = frames[row, plane] > 0
            if not bool(mask.any()):
                continue
            distance = (rows - ar).abs() + (cols - ac).abs()
            distance = torch.where(mask, distance,
                                   torch.full_like(distance, 10_000))
            nearest = int(distance.reshape(-1).argmin())
            out[row, base] = nearest // WIDTH
            out[row, base + 1] = nearest % WIDTH
    return out


def seek_actions(states: torch.Tensor, generator) -> torch.Tensor:
    """Move toward the nearest positive object, from the slot state the
    learner itself reads. No privileged information is used."""
    action = torch.randint(0, ACTIONS, (states.shape[0],),
                           generator=generator)
    for row in range(states.shape[0]):
        if int(states[row, 2]) == ABSENT or int(states[row, 0]) == ABSENT:
            continue
        dr = int(states[row, 2]) - int(states[row, 0])
        dc = int(states[row, 3]) - int(states[row, 1])
        if abs(dr) >= abs(dc) and dr != 0:
            action[row] = 0 if dr < 0 else 2
        elif dc != 0:
            action[row] = 3 if dc < 0 else 1
    return action


def roll(verifier: FamilyVerifier, count: int, generator) -> dict:
    """Watch the world. With `--horizon` k the label is the DISCOUNTED
    RETURN over the next k steps rather than the immediate outcome, so a
    food event colours the steps that led to it."""
    steps = max(1, -(-count // verifier.batch_size)) + args.horizon
    states, acts, nxt, rewards = [], [], [], []
    for _ in range(steps):
        state = slot_state(verifier.observation().reshape(
            verifier.batch_size, -1))
        if args.seek > 0 and float(torch.rand(1, generator=generator)) < args.seek:
            action = seek_actions(state, generator)
        else:
            action = torch.randint(0, ACTIONS, (verifier.batch_size,),
                                   generator=generator)
        step = verifier.step(action)
        after = slot_state(verifier.observation().reshape(
            verifier.batch_size, -1))
        states.append(state)
        acts.append(action)
        nxt.append(after)
        rewards.append(step.reward)
    matrix = torch.stack(rewards, dim=1)          # [batch, steps]
    # n-STEP return, bounded by `horizon`. The first version accumulated
    # to the END of the sequence regardless of `horizon`, so the flag
    # only trimmed trailing rows and every label was a full Monte-Carlo
    # return. That made the ablation meaningless -- caught by measuring
    # label density rather than by reading the code.
    horizon = torch.zeros_like(matrix)
    for offset in range(args.horizon):
        shifted = torch.zeros_like(matrix)
        if offset:
            shifted[:, :-offset] = matrix[:, offset:]
        else:
            shifted = matrix
        horizon = horizon + (args.discount ** offset) * shifted
    usable = matrix.shape[1] - args.horizon
    flat = lambda parts: torch.cat(parts[:usable])[:count]  # noqa: E731
    value = horizon[:, :usable].T.reshape(-1)[:count]
    labels = torch.ones_like(value, dtype=torch.long)
    labels[value > 0.05] = 2
    labels[value < -0.05] = 0
    return {"state": flat(states), "action": flat(acts),
            "next": flat(nxt), "outcome": labels}


def encode(states: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        states, VALUES + 1).float().reshape(states.shape[0], -1)


WIDTH_IN = SLOTS * (VALUES + 1)


class Reader(torch.nn.Module):
    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.tokens = tokens
        self.embed = torch.nn.Linear(2 * WIDTH_IN + ACTIONS + 3, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, batch: dict) -> torch.Tensor:
        rows = torch.cat([
            encode(batch["state"]), encode(batch["next"]),
            torch.nn.functional.one_hot(batch["action"], ACTIONS).float(),
            torch.nn.functional.one_hot(batch["outcome"], 3).float()],
            dim=-1)
        x = torch.cat([self.queries, self.embed(rows)], dim=0).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[0, :self.tokens])


class Dynamics(torch.nn.Module):
    """(state, action) -> next state, per slot. Falling objects move here."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(WIDTH_IN + ACTIONS, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, SLOTS * (VALUES + 1)))

    def forward(self, states, acts) -> torch.Tensor:
        features = torch.cat([
            encode(states),
            torch.nn.functional.one_hot(acts, ACTIONS).float()], dim=-1)
        return self.net(features).view(-1, SLOTS, VALUES + 1)


class Outcome(torch.nn.Module):
    """(state, action, entry) -> what happens. Factual, world-specific."""

    def __init__(self, dim: int):
        super().__init__()
        self.state = torch.nn.Linear(WIDTH_IN, dim)
        self.action = torch.nn.Embedding(ACTIONS, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 3)

    def forward(self, states, acts, entry) -> torch.Tensor:
        token = (self.state(encode(states))
                 + self.action(acts)).unsqueeze(1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(states.shape[0], -1, -1)
            token = torch.cat([context, token], dim=1)
        for block in self.blocks:
            token = block(token)
        return self.head(self.norm(token[:, -1]))


reader, dynamics, outcome = (Reader(args.dim, args.bank_tokens),
                             Dynamics(args.dim), Outcome(args.dim))
parts = (list(reader.parameters()) + list(dynamics.parameters())
         + list(outcome.parameters()))
optimizer = torch.optim.Adam(parts, lr=args.lr)

configs = variant_configs()
order = torch.randperm(len(configs),
                       generator=torch.Generator().manual_seed(args.seed))
cut = max(4, len(configs) // 4)
held = [configs[int(i)] for i in order[:cut]]
train = [configs[int(i)] for i in order[cut:]]
generator = torch.Generator().manual_seed(args.seed + 31)

if not args.random_plant:
    live = {index: verifier_for(config, args.seed + index)
            for index, config in enumerate(train)}
    for step in range(args.train_updates):
        verifier = live[step % len(train)]
        entry = reader(roll(verifier, args.context, generator))
        batch = roll(verifier, args.batch_size, generator)
        predicted = dynamics(batch["state"], batch["action"])
        loss = sum(
            torch.nn.functional.cross_entropy(predicted[:, slot],
                                              batch["next"][:, slot])
            for slot in range(SLOTS)) / SLOTS
        weight = None
        if args.balance_loss:
            counts = torch.bincount(batch["outcome"], minlength=3).float()
            weight = (counts.sum() / counts.clamp_min(1.0))
            weight = weight / weight.sum() * 3.0
        loss = loss + torch.nn.functional.cross_entropy(
            outcome(batch["state"], batch["action"], entry),
            batch["outcome"], weight=weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parts, 1.0)
        optimizer.step()

for parameter in parts:
    parameter.requires_grad_(False)


def act(states: torch.Tensor, entry) -> torch.Tensor:
    """Beam search in the learned model, scored by predicted outcome.

    Nothing preferential is stored: the plan is rebuilt from the current
    state every step, so it cannot go stale (F67).
    """
    batch = states.shape[0]
    beam_states = states.unsqueeze(1)                      # [B, 1, SLOTS]
    beam_score = torch.zeros(batch, 1)
    beam_first = torch.full((batch, 1), -1, dtype=torch.long)
    with torch.no_grad():
        for depth in range(args.depth):
            width = beam_states.shape[1]
            flat = beam_states.reshape(-1, SLOTS)
            expanded, scores, firsts = [], [], []
            for action in range(ACTIONS):
                column = torch.full((flat.shape[0],), action,
                                    dtype=torch.long)
                probability = outcome(flat, column, entry).softmax(-1)
                gain = (probability[:, 2] - probability[:, 0]).view(
                    batch, width)
                successor = dynamics(flat, column).argmax(-1).view(
                    batch, width, SLOTS)
                expanded.append(successor)
                scores.append(beam_score + gain * (0.9 ** depth))
                first = torch.where(
                    beam_first < 0,
                    torch.full_like(beam_first, action), beam_first)
                firsts.append(first)
            merged_states = torch.cat(expanded, dim=1)
            merged_score = torch.cat(scores, dim=1)
            merged_first = torch.cat(firsts, dim=1)
            keep = merged_score.topk(min(args.beam,
                                         merged_score.shape[1]), dim=1).indices
            beam_states = torch.gather(
                merged_states, 1,
                keep.unsqueeze(-1).expand(-1, -1, SLOTS))
            beam_score = torch.gather(merged_score, 1, keep)
            beam_first = torch.gather(merged_first, 1, keep)
    return beam_first[torch.arange(batch), beam_score.argmax(dim=1)]


def play(config: FamilyConfig, entry, seed: int,
         random_actions: bool = False) -> float:
    verifier = verifier_for(config, seed)
    generator = torch.Generator().manual_seed(seed + 5)
    total = 0.0
    for _ in range(args.trials):
        state = slot_state(verifier.observation().reshape(
            verifier.batch_size, -1))
        choice = (torch.randint(0, ACTIONS, (verifier.batch_size,),
                                generator=generator)
                  if random_actions else act(state, entry))
        total += float(verifier.step(choice).reward.mean())
    return round(total / args.trials, 4)


probe = torch.Generator().manual_seed(args.seed + 900)


def twin_of(config: FamilyConfig):
    """The inverted twin: renders IDENTICALLY, rewards oppositely.

    F103's stranger control drew a random other variant, which is
    usually a different component mix whose entry is merely
    uninformative. The twin is the only entry that is actively WRONG on
    the same pixels, so it is the sharp test of whether the entry
    carries the inversion bit at all. This is F93's lesson again: a
    control has to be adversarial to falsify anything.
    """
    for candidate in train + held:
        if (candidate.active() == config.active()
                and candidate.inverted != config.inverted
                and candidate.collect == config.collect
                and candidate.intercept == config.intercept
                and candidate.avoid == config.avoid
                and candidate.navigate == config.navigate):
            return candidate
    return None


def score(config: FamilyConfig, mode: str) -> dict:
    if mode == "twin":
        twin = twin_of(config)
        if twin is None:
            return {"reward": None, "floor": None}
        entry = reader(roll(verifier_for(twin, args.seed + 41),
                            args.context, probe))
        return {"reward": play(config, entry, args.seed + 8000),
                "floor": play(config, entry, args.seed + 8000,
                              random_actions=True)}
    if mode == "withheld":
        entry = torch.zeros(args.bank_tokens, args.dim)
    elif mode == "stranger":
        other = train[int(torch.randint(0, len(train), (1,),
                                        generator=probe))]
        entry = reader(roll(verifier_for(other, args.seed + 21),
                            args.context, probe))
    else:
        entry = reader(roll(verifier_for(config, args.seed + 31),
                            args.context, probe))
    return {"reward": play(config, entry, args.seed + 8000),
            "floor": play(config, entry, args.seed + 8000,
                          random_actions=True)}


report = {"seed": args.seed, "train_count": len(train),
          "held_out_count": len(held), "random_plant": args.random_plant,
          "depth": args.depth, "beam": args.beam,
          "held_out": {c.name: score(c, "read") for c in held},
          "trained": {c.name: score(c, "read") for c in train[:len(held)]},
          "withheld_entry": {c.name: score(c, "withheld") for c in held},
          "stranger_entry": {c.name: score(c, "stranger") for c in held},
          "twin_entry": {c.name: score(c, "twin") for c in held}}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
