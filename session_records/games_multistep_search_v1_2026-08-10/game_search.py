"""Multi-step derivation on the games: model + value + search.

F100 measured a null on the battery's navigation variants and located
the cause in the probe rather than the mechanism: `collect`,
`intercept`, `avoid` and `navigate` pay out after several moves, while
`game_rule_reading.py` predicted ONE action's outcome and acted by
greedy argmax. Nearly every single action yields zero, so the model was
right and useless, and a random plant matched 12000 updates of training.

F67 prescribed the missing half — derive behaviour by SEARCH in a
learned model — and `reacher_ladder.py` has implemented it since. This
probe combines the three pieces the repository already has and never
put together:

  1. a TRANSITION model, (cell, action) -> next cell, learned from
     watching the avatar move. This is the same object F92 read from the
     reacher at 1.000, and it is where walls live;
  2. a VALUE model, (screen, cell, entry) -> what happens if I step
     THERE, learned from observed outcomes. The bank entry is what makes
     it world-specific: the same screen means "eat it" in one variant
     and "avoid it" in its inverted twin;
  3. SEARCH over the two — breadth-first in the transition model to the
     nearest cell the value model likes, then take the first action of
     that path.

Nothing preferential is stored anywhere. The value model answers "what
happens at that cell", never "go left"; the path is recomputed every
step from the current state, so it cannot go stale (F67).

The state is the avatar's cell, read off plane 0 of the screen. That is
a perception step, and a trivial one — F60 measured perception not to be
the constraint, and the reacher probes used oracle state for the same
reason. Variants with `dual` are excluded because they draw their cue
into plane 0, which would make the avatar unreadable there.

Nulls: entry withheld (zeros), a stranger variant's entry, a plant
frozen at initialisation, and a measured random-action floor per
variant.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import deque

import torch

from experiments.games_amodal.game_family import (
    FamilyConfig, FamilyVerifier, family_variants)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--context", type=int, default=128)
parser.add_argument("--train-updates", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--trials", type=int, default=24)
parser.add_argument("--depth", type=int, default=6, help="search depth")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--random-plant", action="store_true")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
ACTIONS, PLANES, HEIGHT, WIDTH = 4, 3, 8, 8
CELLS = HEIGHT * WIDTH
SCREEN = PLANES * HEIGHT * WIDTH
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def variant_configs() -> list:
    """Every valid non-`dual` world the battery enumerates."""
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


def avatar_cells(screen: torch.Tensor) -> torch.Tensor:
    """Read the avatar off plane 0. Rows with no avatar (dead) give 0."""
    plane = screen.view(-1, PLANES, HEIGHT, WIDTH)[:, 0]
    return plane.reshape(-1, CELLS).argmax(dim=-1)


def roll(verifier: FamilyVerifier, count: int, generator) -> dict:
    """Watch the world: screens, where we were, what we did, where we
    ended up, and what it paid."""
    screens, cells, acts, nxt, rewards = [], [], [], [], []
    while len(screens) * verifier.batch_size < count:
        screen = verifier.observation().reshape(verifier.batch_size, -1)
        cell = avatar_cells(screen)
        action = torch.randint(0, ACTIONS, (verifier.batch_size,),
                               generator=generator)
        step = verifier.step(action)
        after = avatar_cells(
            verifier.observation().reshape(verifier.batch_size, -1))
        screens.append(screen)
        cells.append(cell)
        acts.append(action)
        nxt.append(after)
        rewards.append(step.reward)
    take = lambda parts: torch.cat(parts)[:count]  # noqa: E731
    reward = take(rewards)
    labels = torch.ones_like(reward, dtype=torch.long)
    labels[reward > 0.05] = 2
    labels[reward < -0.05] = 0
    return {"screen": take(screens), "cell": take(cells),
            "action": take(acts), "next": take(nxt), "outcome": labels}


class Reader(torch.nn.Module):
    """Observed outcomes -> a bank entry, one forward pass."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.tokens = tokens
        self.embed = torch.nn.Linear(SCREEN + CELLS + 3, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, batch: dict) -> torch.Tensor:
        rows = torch.cat([
            batch["screen"],
            torch.nn.functional.one_hot(batch["next"], CELLS).float(),
            torch.nn.functional.one_hot(batch["outcome"], 3).float()],
            dim=-1)
        x = torch.cat([self.queries, self.embed(rows)], dim=0).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[0, :self.tokens])


class Transition(torch.nn.Module):
    """(cell, action) -> next cell. Where the walls live."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(CELLS + ACTIONS + SCREEN, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, CELLS))

    def forward(self, screens, cells, acts) -> torch.Tensor:
        return self.net(torch.cat([
            torch.nn.functional.one_hot(cells, CELLS).float(),
            torch.nn.functional.one_hot(acts, ACTIONS).float(),
            screens], dim=-1))


class Value(torch.nn.Module):
    """(screen, cell, entry) -> what happens if I stand THERE.

    Factual, not preferential: it never says which way to move. The
    entry is what makes it world-specific, so an inverted twin reads the
    same screen and gets the opposite answer.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.screen = torch.nn.Linear(SCREEN, dim)
        self.cell = torch.nn.Embedding(CELLS, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 3)

    def forward(self, screens, cells, entry) -> torch.Tensor:
        token = (self.screen(screens) + self.cell(cells)).unsqueeze(1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(screens.shape[0], -1, -1)
            token = torch.cat([context, token], dim=1)
        for block in self.blocks:
            token = block(token)
        return self.head(self.norm(token[:, -1]))


reader = Reader(args.dim, args.bank_tokens)
transition = Transition(args.dim)
value = Value(args.dim)
parts = (list(reader.parameters()) + list(transition.parameters())
         + list(value.parameters()))
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
        loss = torch.nn.functional.cross_entropy(
            transition(batch["screen"], batch["cell"], batch["action"]),
            batch["next"])
        loss = loss + torch.nn.functional.cross_entropy(
            value(batch["screen"], batch["next"], entry), batch["outcome"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parts, 1.0)
        optimizer.step()

for parameter in parts:
    parameter.requires_grad_(False)


def act(screen: torch.Tensor, entry) -> torch.Tensor:
    """SEARCH: breadth-first in the learned transition model to the
    nearest cell the value model likes, then take that path's first
    action. Recomputed every step, so nothing is stored."""
    batch = screen.shape[0]
    cells = avatar_cells(screen)
    with torch.no_grad():
        # what every cell is worth, under this world's entry
        wide_screen = screen.repeat_interleave(CELLS, dim=0)
        all_cells = torch.arange(CELLS).repeat(batch)
        scores = value(wide_screen, all_cells, entry).softmax(-1)
        worth = (scores[:, 2] - scores[:, 0]).view(batch, CELLS)
        # the learned map: where each action leads from each cell
        moves = torch.zeros(batch, CELLS, ACTIONS, dtype=torch.long)
        for action in range(ACTIONS):
            column = torch.full((batch * CELLS,), action, dtype=torch.long)
            moves[:, :, action] = transition(
                wide_screen, all_cells, column).argmax(-1).view(batch, CELLS)
    chosen = torch.zeros(batch, dtype=torch.long)
    for row in range(batch):
        table = moves[row]
        goals = [c for c in range(CELLS) if float(worth[row, c]) > 0.15]
        if not goals:
            chosen[row] = int(worth[row].argmax()) % ACTIONS
            continue
        target = set(goals)
        start = int(cells[row])
        queue, seen = deque([(start, None, 0)]), {start}
        best = None
        while queue:
            node, first, depth = queue.popleft()
            if node in target and first is not None:
                best = first
                break
            if depth >= args.depth:
                continue
            for action in range(ACTIONS):
                successor = int(table[node, action])
                if successor in seen:
                    continue
                seen.add(successor)
                queue.append((successor, action if first is None else first,
                              depth + 1))
        chosen[row] = best if best is not None else int(
            torch.randint(0, ACTIONS, (1,)))
    return chosen


def play(config: FamilyConfig, entry, seed: int,
         random_actions: bool = False) -> float:
    verifier = verifier_for(config, seed)
    generator = torch.Generator().manual_seed(seed + 5)
    total = 0.0
    for _ in range(args.trials):
        screen = verifier.observation().reshape(verifier.batch_size, -1)
        choice = (torch.randint(0, ACTIONS, (verifier.batch_size,),
                                generator=generator)
                  if random_actions else act(screen, entry))
        total += float(verifier.step(choice).reward.mean())
    return round(total / args.trials, 4)


probe = torch.Generator().manual_seed(args.seed + 900)


def score(config: FamilyConfig, mode: str) -> dict:
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
          "depth": args.depth,
          "held_out": {c.name: score(c, "read") for c in held},
          "trained": {c.name: score(c, "read") for c in train[:len(held)]},
          "withheld_entry": {c.name: score(c, "withheld") for c in held},
          "stranger_entry": {c.name: score(c, "stranger") for c in held}}

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
