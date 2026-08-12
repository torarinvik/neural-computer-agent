"""Does a DISCOVERED perception survive contact with the task?

F213 found that programmability x coverage picks an encoder from a
domain-general vocabulary that beats the hand-written `slot_state` on
held-out worlds. That is a property of the CRITERION, and the criterion
is not the task. This is the test that matters.

There is a coupling the criterion cannot see. The goal in F203 and F207
is written as "move the slot pair (0,1) toward the slot pair (2,3)",
which is stated in the interface rather than in grid terms but still
NAMES SLOTS. Hand-written perception guarantees slots 0,1 are the avatar
and 2,3 are an object. The discovered encoder puts channel-2's column in
slot 2 and channel-1's row in slot 3 -- a pair that is not a position at
all. So a discovered perception should break a hand-written goal, and
if it does, perception and goal cannot be chosen independently.

Three arms, so the two explanations are separable:

  handwritten + fixed goal    F207's configuration, the incumbent.
  discovered  + fixed goal    if this collapses, the goal's slot naming
                              is the coupling.
  discovered  + CHOSEN goal   the goal's two slot pairs are selected by
                              reward on TRAIN games and then frozen. If
                              this recovers, the coupling is real and
                              removable, and neither perception nor goal
                              had to be written by hand.

Choosing the goal is a search over which slots to match, which is
domain-general in the same way the recipe search is: it names no game
object, only slot indices. It is still a supplied FORM of goal (match
one pair to another) and that limitation is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--examples", type=int, default=32)
parser.add_argument("--observations", type=int, default=256)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--train-games", type=int, default=8,
                    help="games the goal pair may be chosen on")
parser.add_argument("--held-from", type=int, default=8)
parser.add_argument("--held-to", type=int, default=16)
parser.add_argument("--search-episodes", type=int, default=32,
                    help="cheaper fidelity while RANKING goal pairs; the "
                         "chosen pair is then evaluated at full budget")
parser.add_argument("--search-steps", type=int, default=10)
parser.add_argument("--search-games", type=int, default=6)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))
NOOP = (0, 0, 0)
ROWS_IX = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
COLS_IX = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)

# the encoder F213's search returned, verbatim
DISCOVERED = [(0, "peak_row"), (0, "peak_col"), (2, "peak_col"),
              (1, "last_row"), (0, "centre_row"), (0, "centre_col")]


def slot_write(state, s, op, j, m):
    name, mod = PAR_OPS[op], MODULI[m]
    col = state[:, s]
    if name == "NOOP":
        return col
    if name == "INC":
        return (col + 1) % mod
    if name == "DEC":
        return (col - 1) % mod
    if name == "SINC":
        return torch.clamp(col + 1, max=mod - 1)
    if name == "SDEC":
        return torch.clamp(col - 1, min=0)
    if name == "CINC":
        return torch.where(state[:, j] != 0, (col + 1) % mod, col)
    if name == "CDEC":
        return torch.where(state[:, j] != 0, (col - 1) % mod, col)
    if name == "COPY":
        return state[:, j]
    raise AssertionError(name)


def run_parallel(state, program):
    out = state.clone()
    for s in range(SLOTS):
        out[:, s] = slot_write(state, s, *program[s])
    return out


def per_slot_search(before, after):
    program = []
    for s in range(SLOTS):
        want = after[:, s]
        best, best_score = NOOP, -1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j == s and PAR_OPS[op] in ("CINC", "CDEC", "COPY"):
                    continue
                for m in range(len(MODULI)):
                    score = float((slot_write(before, s, op, j, m) == want)
                                  .float().mean())
                    if score > best_score:
                        best, best_score = (op, j, m), score
                    if best_score >= 1.0:
                        break
                if best_score >= 1.0:
                    break
            if best_score >= 1.0:
                break
        program.append(best)
    return program


class Interpreter(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.load = torch.nn.Linear(SLOTS * VALUES, dim)
        self.slot = torch.nn.Embedding(SLOTS, dim)
        self.op = torch.nn.Embedding(len(PAR_OPS), dim)
        self.arg_j = torch.nn.Embedding(SLOTS, dim)
        self.arg_m = torch.nn.Embedding(len(MODULI), dim)
        self.step = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, SLOTS * VALUES)

    def code_for(self, s, program):
        op, j, m = program[s]
        return (self.slot(torch.tensor(s)) + self.op(torch.tensor(op))
                + self.arg_j(torch.tensor(j)) + self.arg_m(torch.tensor(m)))

    def forward(self, program, state):
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        base = self.load(hot)
        latent = base
        for s in range(SLOTS):
            code = self.code_for(s, program).unsqueeze(0).expand(
                latent.shape[0], -1)
            latent = self.norm(latent + self.step(
                torch.cat([latent, base, code], dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_parallel(g):
    out = []
    for s in range(SLOTS):
        op = int(torch.randint(0, len(PAR_OPS), (1,), generator=g))
        j = int(torch.randint(0, SLOTS, (1,), generator=g))
        if j == s:
            j = (j + 1) % SLOTS
        m = int(torch.randint(0, len(MODULI), (1,), generator=g))
        out.append((op, j, m))
    return out


interp = Interpreter(args.dim)
opt = torch.optim.AdamW(interp.parameters(), lr=args.lr, weight_decay=0.01)
gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.interpreter_updates):
    prog = random_parallel(gen)
    st = torch.randint(0, VALUES, (args.batch_size, SLOTS), generator=gen)
    loss = torch.nn.functional.cross_entropy(
        interp(prog, st).reshape(-1, VALUES),
        run_parallel(st, prog).reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
for p in interp.parameters():
    p.requires_grad_(False)
check = torch.Generator().manual_seed(args.seed + 5551)
hits = rows = 0
for _ in range(32):
    prog = random_parallel(check)
    st = torch.randint(0, VALUES, (128, SLOTS), generator=check)
    with torch.no_grad():
        hits += int((interp(prog, st).argmax(-1) == run_parallel(st, prog)).sum())
    rows += st.numel()
report = {"seed": args.seed, "interpreter_check": round(hits / rows, 4)}


# ------------------------------------------------------------ encoders
def enc_handwritten(screen):
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    out = torch.full((frames.shape[0], SLOTS), ABSENT, dtype=torch.long)
    for row in range(frames.shape[0]):
        avatar = frames[row, 0]
        if float(avatar.max()) <= 0:
            continue
        flat = int(avatar.reshape(-1).argmax())
        ar, ac = flat // WIDTH, flat % WIDTH
        out[row, 0], out[row, 1] = ar, ac
        for plane in (1, 2):
            base = 2 * plane
            mask = frames[row, plane] > 0
            if not bool(mask.any()):
                continue
            d = (ROWS_IX - ar).abs() + (COLS_IX - ac).abs()
            d = torch.where(mask, d, torch.full_like(d, 999))
            pick = int(d.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def _reduce(plane, which):
    mask = plane > 0
    if which == "peak_row":
        return (plane.reshape(plane.shape[0], -1).argmax(dim=1) // WIDTH)
    if which == "peak_col":
        return (plane.reshape(plane.shape[0], -1).argmax(dim=1) % WIDTH)
    weight = mask.float()
    total = weight.sum(dim=(1, 2)).clamp(min=1.0)
    if which == "centre_row":
        return ((weight * ROWS_IX).sum(dim=(1, 2)) / total).round().long()
    if which == "centre_col":
        return ((weight * COLS_IX).sum(dim=(1, 2)) / total).round().long()
    if which == "last_row":
        rows_any = mask.any(dim=2)
        idx = torch.arange(HEIGHT).view(1, -1)
        return torch.where(rows_any, idx,
                           torch.full_like(idx, -1)).max(dim=1).values.clamp(min=0)
    raise AssertionError(which)


def enc_discovered(screen):
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    out = torch.zeros((frames.shape[0], SLOTS), dtype=torch.long)
    for s, (c, r) in enumerate(DISCOVERED):
        out[:, s] = _reduce(frames[:, c], r).clamp(0, VALUES - 1)
    return out


def sanitise(before, after):
    alive = (before < VALUES).all(dim=1) & (after < VALUES).all(dim=1)
    if int(alive.sum()) < 8:
        return None
    return before[alive], after[alive]


def goal_cost(state, reference, pair_a, pair_b):
    """F203's objective with the two slot PAIRS named explicitly instead
    of hard-coded to (0,1) and (2,3)."""
    reach = ((state[:, pair_a[0]] - reference[:, pair_b[0]]).abs()
             + (state[:, pair_a[1]] - reference[:, pair_b[1]]).abs())
    absent = ((reference[:, pair_b[0]] >= VALUES)
              | (reference[:, pair_b[1]] >= VALUES))
    return torch.where(absent, torch.full_like(reach, 99), reach).float()


def build_bank(config, seed, encoder):
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations, seed=seed + act)
        v.reset(seed=seed + act)
        before = encoder(v.observation())
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        clean = sanitise(before, encoder(v.observation()))
        if clean is None:
            continue
        bank[act] = per_slot_search(clean[0][:args.examples],
                                    clean[1][:args.examples])
    return bank


def play(config, mode, bank, seed, encoder, pair_a, pair_b,
         episodes=None, steps=None):
    episodes = episodes or args.episodes
    steps = steps or args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            best = None
            reference = encoder(v.observation())
            # ABSENT must be handled the SAME way on both sides. Masking
            # the predicted state but not the reference made a slot that
            # is absent compare against 8 on one side and 0 on the other,
            # which is how a goal naming absent slots produced a constant
            # cost across all four actions -- a blind planner that always
            # takes action 0.
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            action = torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                if mode == "oracle":
                    shadow = copy.deepcopy(v)
                    shadow.step(torch.full((episodes,), act))
                    seen = encoder(shadow.observation())
                    cost = goal_cost(torch.where(seen < VALUES, seen,
                                                 torch.zeros_like(seen)),
                                     reference, pair_a, pair_b)
                else:
                    state = encoder(v.observation())
                    state = torch.where(state < VALUES, state,
                                        torch.zeros_like(state))
                    prog = bank.get(act)
                    if prog is not None:
                        with torch.no_grad():
                            state = interp(prog, state).argmax(-1)
                    cost = goal_cost(state, reference, pair_a, pair_b)
                if best is None:
                    best = cost.clone()
                else:
                    take = cost < best
                    best = torch.where(take, cost, best)
                    action = torch.where(take,
                                         torch.full((episodes,), act),
                                         action)
        total += v.step(action).reward
    return round(float(total.mean()), 4)


variants = family_variants()
train_games = variants[:args.train_games]
held_games = variants[args.held_from:args.held_to]
pairs = [q for q in itertools.permutations(range(SLOTS), 2)]


def usable_slots(config, encoder):
    """Which slots this world actually populates.

    Presence is a per-WORLD property and averaging it across worlds was
    wrong: collect and intercept fill slots 0-3 and leave 4-5 absent,
    avoid fills 0,1 and 4,5 and leaves 2,3 absent, so the average put
    every object slot below any sensible threshold and admitted nothing.

    In the pure avoid worlds slots 2 and 3 are absent, so the fixed goal
    (0,1)->(2,3) names slots that do not exist and the planner there is
    blind. F203 itself tested collect and intercept, which do populate
    0-3, so this does not explain F203's negative arms -- but any run
    that includes an avoid world inherits it."""
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    present = (encoder(v.observation()) < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


def choose_goal(config, encoder, bank):
    """Pick this world's goal from its OWN admissible slots, scored on
    episodes that the final evaluation does not reuse."""
    usable = usable_slots(config, encoder)
    best, best_reward = None, -1e9
    for pa in pairs:
        for pb in pairs:
            if set(pa) & set(pb) or pa[0] > pa[1]:
                continue
            if not (set(pa) | set(pb)) <= usable:
                continue
            reward = play(config, "bank", bank, args.seed * 977 + 1, encoder,
                          pa, pb, episodes=args.search_episodes,
                          steps=args.search_steps)
            if reward > best_reward:
                best, best_reward = (pa, pb), reward
    return best, round(best_reward, 4), sorted(usable)


results = {}
for label, encoder, mode in (
        ("handwritten_fixed_goal", enc_handwritten, "fixed"),
        ("handwritten_chosen_goal", enc_handwritten, "chosen"),
        ("discovered_fixed_goal", enc_discovered, "fixed"),
        ("discovered_chosen_goal", enc_discovered, "chosen")):
    rows, goals = {}, {}
    for config in held_games:
        bank = build_bank(config, args.seed * 31, encoder)
        if mode == "fixed":
            pa, pb = (0, 1), (2, 3)
            picked = None
        else:
            picked, train_reward, usable = choose_goal(config, encoder, bank)
            if picked is None:            # world admits no goal at all
                continue
            pa, pb = picked
            goals[config.name] = {"goal": [list(pa), list(pb)],
                                  "select_reward": train_reward,
                                  "usable_slots": usable}
        rows[config.name] = {
            "random": play(config, "random", None, args.seed * 977,
                           encoder, pa, pb),
            "bank": play(config, "bank", bank, args.seed * 977,
                         encoder, pa, pb),
            "oracle": play(config, "oracle", None, args.seed * 977,
                           encoder, pa, pb)}
    if not rows:
        results[label] = {"scored_games": 0}
        continue
    results[label] = {
        "scored_games": len(rows), "per_game": rows, "chosen_goals": goals,
        **{f"mean_{k}": round(sum(v[k] for v in rows.values()) / len(rows), 4)
           for k in ("random", "bank", "oracle")}}
    print(f"  {label:<26} n={len(rows):<3} random "
          f"{results[label]['mean_random']:+.4f}  bank "
          f"{results[label]['mean_bank']:+.4f}  oracle "
          f"{results[label]['mean_oracle']:+.4f}", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
