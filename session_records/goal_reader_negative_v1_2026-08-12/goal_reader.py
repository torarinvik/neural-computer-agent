"""Amortise the GOAL search: a reader that infers what a world wants.

The dynamics side of the architecture is amortised — F205-F208 replaced
a ~900-candidate program search with one forward pass plus verification.
The goal side is not: F214/F216 choose a world's goal by evaluating 180
to 360 candidate goals with full rollouts, every time, and world 1001
costs what world 1 did. This probe closes that gap the same way the
dynamics gap was closed:

    wake   the F216 goal search solves worlds and keeps (evidence, goal)
    sleep  a reader trains on those labels
    test   a NEW world's goal is read from a handful of its own
           transitions in one forward pass

Evidence is a set of (state, next-state, reward) transitions from a
random policy. A goal is (pair_a, pair_b, sign) exactly as in F216.

**The control that matters is specific to goals.** Dynamics are visible
in (state, next-state); a goal is visible ONLY in reward. So the
decisive control scrambles the reward column within the evidence set,
leaving dynamics untouched: a reader that actually reads goals must
collapse to the label mode, and one that survives has smuggled the
answer through something else (e.g. memorised world dynamics).

Wake worlds are synthetic goal-worlds (random parallel-program dynamics,
random admissible hidden goal, reward = the signed distance change it
induces) mixed with grid worlds at F206's measured optimum share, and
the grid worlds under test never appear in the wake pool.

Registered predictions:
  1. Planning with READ goals matches planning with SEARCHED goals on
     held-out grid worlds (the F207 pattern, now for goals).
  2. The reward-scrambled arm collapses to the mode-goal baseline.
  3. Sign accuracy holds on the avoid worlds specifically — the reader
     picks sign=-1 there from reward evidence alone.

The reader follows F212: one token per slot, attention across slots, no
slot embedding; the four pair roles are POINTER heads (shared weights
scoring every token) and the sign head reads the pooled latent, so the
whole thing is permutation-equivariant by construction.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--reader-updates", type=int, default=6000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--examples", type=int, default=48,
                    help="evidence transitions the reader sees per world")
parser.add_argument("--pool", type=int, default=3000)
parser.add_argument("--grid-share", type=float, default=0.3,
                    help="fraction of wake worlds that are real grids "
                         "(from the wake split only); F206's optimum")
parser.add_argument("--wake-grids", type=int, default=9,
                    help="unused when --held-pure is set")
parser.add_argument("--held-pure", action="store_true",
                    help="hold out the PURE worlds (collect, intercept, "
                         "avoid, navigate) and wake on the compounds. The "
                         "first split held only compounds, where the "
                         "searched goal is the same for all eight worlds "
                         "-- a constant wins and reading cannot show "
                         "itself. The pure worlds need different pairs "
                         "AND different signs.")
parser.add_argument("--observations", type=int, default=256)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--search-episodes", type=int, default=32)
parser.add_argument("--search-steps", type=int, default=10)
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

    def forward(self, program, state):
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        base = self.load(hot)
        latent = base
        for s in range(SLOTS):
            op, j, m = program[s]
            code = (self.slot(torch.tensor(s)) + self.op(torch.tensor(op))
                    + self.arg_j(torch.tensor(j))
                    + self.arg_m(torch.tensor(m))).unsqueeze(0).expand(
                        latent.shape[0], -1)
            latent = self.norm(latent + self.step(
                torch.cat([latent, base, code], dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_program(g):
    out = []
    for s in range(SLOTS):
        op = int(torch.randint(0, len(PAR_OPS), (1,), generator=g))
        j = int(torch.randint(0, SLOTS, (1,), generator=g))
        if j == s:
            j = (j + 1) % SLOTS
        out.append((op, j, int(torch.randint(0, len(MODULI), (1,),
                                             generator=g))))
    return out


interp = Interpreter(args.dim)
opt = torch.optim.AdamW(interp.parameters(), lr=args.lr, weight_decay=0.01)
gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.interpreter_updates):
    prog = random_program(gen)
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
    prog = random_program(check)
    st = torch.randint(0, VALUES, (128, SLOTS), generator=check)
    with torch.no_grad():
        hits += int((interp(prog, st).argmax(-1)
                     == run_parallel(st, prog)).sum())
    rows += st.numel()
report = {"seed": args.seed, "interpreter_check": round(hits / rows, 4)}


def enc(screen):
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


def goal_cost(state, reference, goal):
    (a0, a1), (b0, b1), sign = goal
    reach = ((state[:, a0] - reference[:, b0]).abs()
             + (state[:, a1] - reference[:, b1]).abs()).float()
    return sign * reach


ADMISSIBLE = [((a0, a1), (b0, b1), sign)
              for a0 in range(SLOTS) for a1 in range(a0 + 1, SLOTS)
              for b0 in range(SLOTS) for b1 in range(SLOTS)
              if b0 != b1 and not {a0, a1} & {b0, b1}
              for sign in (1, -1)]


def build_bank(config, seed):
    probe = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + 7)
    probe.reset(seed=seed + 7)
    used = (enc(probe.observation()) < VALUES).float().mean(dim=0) >= 0.9
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + act)
        v.reset(seed=seed + act)
        before = enc(v.observation())
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = enc(v.observation())
        keep = ((before[:, used] < VALUES).all(dim=1)
                & (after[:, used] < VALUES).all(dim=1))
        if int(keep.sum()) < 8:
            continue
        b = torch.where(before[keep] < VALUES, before[keep],
                        torch.zeros_like(before[keep]))
        a = torch.where(after[keep] < VALUES, after[keep],
                        torch.zeros_like(after[keep]))
        bank[act] = per_slot_search(b[:32], a[:32])
    return bank, used


def play(config, mode, bank, seed, goal, episodes, steps):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = enc(v.observation())
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                program = bank.get(act)
                if program is None:
                    state = reference
                else:
                    with torch.no_grad():
                        state = interp(program, reference).argmax(-1)
                cost = goal_cost(state, reference, goal)
                if best is None:
                    best = cost.clone()
                else:
                    take = cost < best
                    best = torch.where(take, cost, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        total += v.step(action).reward
    return float(total.mean())


def grid_usable(config):
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    present = (enc(v.observation()) < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


def search_goal(config, bank):
    usable = grid_usable(config)
    best, best_reward = None, -1e9
    for goal in ADMISSIBLE:
        if not (set(goal[0]) | set(goal[1])) <= usable:
            continue
        reward = play(config, "bank", bank, args.seed * 977 + 1, goal,
                      args.search_episodes, args.search_steps)
        if reward > best_reward:
            best, best_reward = goal, reward
    return best


# ------------------------------------------------- evidence generation
def grid_evidence(config, seed, count):
    """Random-policy transitions WITH per-step reward, read into slots."""
    v = FamilyVerifier(config, batch_size=count, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 99)
    before = enc(v.observation())
    action = torch.randint(0, 4, (count,), generator=g)
    step = v.step(action)
    after = enc(v.observation())
    before = torch.where(before < VALUES, before, torch.zeros_like(before))
    after = torch.where(after < VALUES, after, torch.zeros_like(after))
    return before, after, step.reward.float()


def synthetic_world(g):
    """Random dynamics, hidden goal, reward = the signed cost decrease the
    transition produced under that goal. Unlimited worlds, no grids."""
    programs = [random_program(g) for _ in range(4)]
    goal = ADMISSIBLE[int(torch.randint(0, len(ADMISSIBLE), (1,),
                                        generator=g))]
    before = torch.randint(0, VALUES, (args.examples, SLOTS), generator=g)
    acts = torch.randint(0, 4, (args.examples,), generator=g)
    after = before.clone()
    for a in range(4):
        rows = acts == a
        if bool(rows.any()):
            after[rows] = run_parallel(before[rows], programs[a])
    # SPARSE reward, shaped like the grids': fire only on contact.
    # The first version paid dense signed distance-deltas every step, a
    # signal the real worlds never produce, and the reader learned to
    # read a channel that is silent at test time (read - scrambled came
    # out t=+1.44, ns). Approach goals pay +1 when the pair meets; avoid
    # goals pay -1 when it does. A world can also carry a SECOND nuisance
    # goal, because real compounds mix reward sources and a reader
    # trained only on single-source worlds gets pulled by the mixture.
    (a0, a1), (b0, b1), sign = goal
    dist_a = ((after[:, a0] - before[:, b0]).abs()
              + (after[:, a1] - before[:, b1]).abs())
    reward = torch.where(dist_a == 0,
                         torch.full_like(dist_a, float(sign)),
                         torch.zeros_like(dist_a)).float()
    if float(torch.rand(1, generator=g)) < 0.5:
        other = ADMISSIBLE[int(torch.randint(0, len(ADMISSIBLE), (1,),
                                             generator=g))]
        (c0, c1), (d0, d1), osign = other
        dist_o = ((after[:, c0] - before[:, d0]).abs()
                  + (after[:, c1] - before[:, d1]).abs())
        # the nuisance is weaker, so the labelled goal remains the
        # dominant reward source, as the search's label is in grids
        reward = reward + 0.5 * torch.where(
            dist_o == 0, torch.full_like(dist_o, float(osign)),
            torch.zeros_like(dist_o)).float()
    return before, after, reward, goal


# ------------------------------------------------------------ the reader
class GoalReader(torch.nn.Module):
    """Evidence set -> (a0, a1, b0, b1, sign). Equivariant by the F212
    recipe: per-slot tokens, attention, POINTER heads shared across
    slots, sign from the pooled latent. Reward enters each token as
    reward-weighted statistics of that slot's behaviour, which is the
    only channel a goal can be seen through."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        feat = 4 * VALUES + 3
        self.token = torch.nn.Sequential(
            torch.nn.Linear(feat, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.attend = torch.nn.MultiheadAttention(dim, heads,
                                                  batch_first=True)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.mix = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm2 = torch.nn.LayerNorm(dim)
        self.pointers = torch.nn.Linear(dim, 4)   # one score per role
        self.sign = torch.nn.Linear(dim, 2)

    def forward(self, before, after, reward):
        b, e, _ = before.shape
        hot_b = torch.nn.functional.one_hot(before, VALUES).float()
        hot_a = torch.nn.functional.one_hot(after, VALUES).float()
        w = reward.view(b, e, 1, 1)
        # Normalise the reward-weighted marginals by EVENT MASS, not by
        # the row count. Sparse worlds pay reward on ~1% of steps, so
        # dividing by all 512 rows left a signal of magnitude ~0.01 and
        # the reader learned to ignore the reward channel entirely
        # (read - scrambled sat at t=+0.35 with the /N version, and the
        # sign was read as +1 on every avoid world even with the right
        # pair). Dividing by the event mass makes the conditional
        # "what does the world look like when reward fires, and with
        # which sign" O(1) regardless of sparsity.
        mass = reward.abs().sum(dim=1).clamp(min=1.0).view(b, 1, 1)
        event_b = (hot_b * w).sum(dim=1) / mass
        event_a = (hot_a * w).sum(dim=1) / mass
        mean_r = (reward.sum(dim=1) / mass.view(b)).view(b, 1)
        feature = torch.cat([
            hot_b.mean(dim=1), hot_a.mean(dim=1),
            event_b, event_a,
            mean_r.expand(b, SLOTS).unsqueeze(-1),
            reward.std(dim=1, keepdim=True).expand(b, SLOTS).unsqueeze(-1),
            (before != after).float().mean(dim=1).unsqueeze(-1)],
            dim=-1)
        latent = self.token(feature)
        attended, _ = self.attend(latent, latent, latent, need_weights=False)
        latent = self.norm1(latent + attended)
        latent = self.norm2(latent + self.mix(latent))
        return self.pointers(latent), self.sign(latent.mean(dim=1))


def read_goal(reader, before, after, reward):
    with torch.no_grad():
        scores, sign_logit = reader(before.unsqueeze(0), after.unsqueeze(0),
                                    reward.unsqueeze(0))
    s = scores[0]                                # (SLOTS, 4 roles)
    a0, a1 = sorted(torch.topk(s[:, 0] + s[:, 1], 2).indices.tolist())
    remaining = [x for x in range(SLOTS) if x not in (a0, a1)]
    rb = torch.tensor(remaining)
    b0 = int(rb[int(s[rb, 2].argmax())])
    rb2 = torch.tensor([x for x in remaining if x != b0])
    b1 = int(rb2[int(s[rb2, 3].argmax())])
    sign = 1 if int(sign_logit[0].argmax()) == 0 else -1
    return ((a0, a1), (b0, b1), sign)


# --------------------------------------------------------- wake + sleep
variants = family_variants()
if args.held_pure:
    held_grids = variants[:7]          # the pure single-mechanic worlds
    wake_grids = variants[7:16]        # compounds only
else:
    wake_grids = variants[:args.wake_grids]
    held_grids = variants[args.wake_grids:args.wake_grids + 8]
wake_rng = torch.Generator().manual_seed(args.seed * 15485863)

pool_b, pool_a, pool_r, pool_goal = [], [], [], []
grid_labels = {}
count_grid = 0
for index in range(args.pool):
    if (float(torch.rand(1, generator=wake_rng)) < args.grid_share
            and wake_grids):
        config = wake_grids[int(torch.randint(0, len(wake_grids), (1,),
                                              generator=wake_rng))]
        if config.name not in grid_labels:
            bank, _ = build_bank(config, args.seed * 31)
            grid_labels[config.name] = (search_goal(config, bank), bank)
        goal = grid_labels[config.name][0]
        if goal is None:
            continue
        seed = int(torch.randint(0, 10 ** 6, (1,), generator=wake_rng))
        before, after, reward = grid_evidence(config, seed, args.examples)
        count_grid += 1
    else:
        before, after, reward, goal = synthetic_world(wake_rng)
    pool_b.append(before); pool_a.append(after); pool_r.append(reward)
    pool_goal.append(goal)
pool_b = torch.stack(pool_b); pool_a = torch.stack(pool_a)
pool_r = torch.stack(pool_r)
report["pool"] = len(pool_goal)
report["pool_grid_worlds"] = count_grid


def targets(goal):
    (a0, a1), (b0, b1), sign = goal
    role = torch.zeros(SLOTS, 4)
    role[a0, 0] = role[a1, 1] = role[b0, 2] = role[b1, 3] = 1.0
    return role, 0 if sign == 1 else 1


role_t = torch.stack([targets(g)[0] for g in pool_goal])
sign_t = torch.tensor([targets(g)[1] for g in pool_goal])

reader = GoalReader(args.dim, args.heads)
r_opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=0.01)
sgen = torch.Generator().manual_seed(args.seed * 104729)
for update in range(args.reader_updates):
    pick = torch.randint(0, pool_b.shape[0], (args.batch_size,),
                         generator=sgen)
    scores, sign_logit = reader(pool_b[pick], pool_a[pick], pool_r[pick])
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                scores, role_t[pick])
            + torch.nn.functional.cross_entropy(sign_logit, sign_t[pick]))
    r_opt.zero_grad(); loss.backward(); r_opt.step()
for p in reader.parameters():
    p.requires_grad_(False)

# mode goal of the pool, the frequency baseline
from collections import Counter
mode_goal = Counter(pool_goal).most_common(1)[0][0]
report["mode_goal"] = [list(mode_goal[0]), list(mode_goal[1]), mode_goal[2]]

# ------------------------------------------------------------ evaluation
results = {}
scramble = torch.Generator().manual_seed(args.seed + 31337)
for config in held_grids:
    bank, _ = build_bank(config, args.seed * 31)
    searched = search_goal(config, bank)
    if searched is None:
        continue
    before, after, reward = grid_evidence(config, args.seed + 555,
                                          args.examples)
    read = read_goal(reader, before, after, reward)
    perm = torch.randperm(args.examples, generator=scramble)
    read_scrambled = read_goal(reader, before, after, reward[perm])
    row = {"searched_goal": [list(searched[0]), list(searched[1]),
                             searched[2]],
           "read_goal": [list(read[0]), list(read[1]), read[2]],
           "scrambled_goal": [list(read_scrambled[0]),
                              list(read_scrambled[1]), read_scrambled[2]],
           "sign_match": int(read[2] == searched[2]),
           "random": play(config, "random", None, args.seed * 977, None,
                          args.episodes, args.steps)}
    for label, goal in (("searched", searched), ("read", read),
                        ("read_scrambled", read_scrambled),
                        ("mode", mode_goal)):
        row[label] = play(config, "bank", bank, args.seed * 977, goal,
                          args.episodes, args.steps)
    results[config.name] = row
    print(f"  {config.name:<24} random {row['random']:+.3f}  "
          f"searched {row['searched']:+.3f}  READ {row['read']:+.3f}  "
          f"scrambled {row['read_scrambled']:+.3f}  mode {row['mode']:+.3f}",
          flush=True)

report["results"] = results
for label in ("random", "searched", "read", "read_scrambled", "mode"):
    report[f"mean_{label}"] = round(
        sum(r[label] for r in results.values()) / max(len(results), 1), 4)
report["sign_match_rate"] = round(
    sum(r["sign_match"] for r in results.values()) / max(len(results), 1), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
