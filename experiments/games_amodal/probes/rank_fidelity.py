"""RANK FIDELITY: privileged diagnostics for the depth-2 loss.

Companion to plan_depth_factorial.py, per the localization discipline:
before blaming horizon, measure WHERE bank and oracle disagree and
whether the state-cost objective itself ranks actions correctly.
Everything here is a PRIVILEGED DIAGNOSTIC (true dynamics, true
reward), never part of the deployable architecture.

Per world, this probe reuses the archived depth-1 goal and encoder
from the plan_depth runs (--pd path), rebuilds the deterministic plant
and banks, then walks the evaluation stream under the depth-1 bank
policy and, at every step, expands all 16 two-action sequences under
BOTH bank programs and the true simulator, recording:

  1. slot-state agreement bank-vs-oracle at depth 1 and depth 2, and
     depth-2 agreement CONDITIONED on depth-1 exact (first- vs
     second-transition divergence);
  2. Pearson correlation between bank and oracle leaf costs over the
     16 sequences (ranking fidelity without exactness);
  3. best-first-action agreement between bank leaf cost, oracle leaf
     cost, and true two-step return;
  4. regret: true-return of the bank-chosen first action vs the
     true-return-optimal first action.

Readings (registered):
  - oracle-cost best-action ~= true-return best-action, bank disagrees
    -> model error is the failure; horizon and objective are fine.
  - oracle-cost disagrees with true-return -> the state-cost OBJECTIVE
    is inadequate at depth 2 (goal representation, not model).
  - bank-oracle cost correlation high but exact agreement low -> the
    bank does not need exact states; ranking is preserved and the
    depth-2 loss must come from elsewhere (selection/aggregation).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

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
parser.add_argument("--search-episodes", type=int, default=48)
parser.add_argument("--search-steps", type=int, default=12)
parser.add_argument("--pd", default="", help="pd-<seed>.json with archived goals")
parser.add_argument("--probe-steps", type=int, default=12)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 8, 8
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


def plant_executor(program, state):
    with torch.no_grad():
        return interp(program, state).argmax(-1)


def _kth_nearest(plane, ref_row, ref_col, k):
    mask = plane > 0
    d = ((ROWS_IX.unsqueeze(0) - ref_row.view(-1, 1, 1)).abs()
         + (COLS_IX.unsqueeze(0) - ref_col.view(-1, 1, 1)).abs())
    d = torch.where(mask, d, torch.full_like(d, 999))
    flat = d.reshape(d.shape[0], -1)
    order = flat.argsort(dim=1)
    idx = order[:, min(k, order.shape[1] - 1)]
    enough = mask.reshape(mask.shape[0], -1).sum(dim=1) > k
    row = torch.where(enough, idx // WIDTH, torch.full_like(idx, ABSENT))
    col = torch.where(enough, idx % WIDTH, torch.full_like(idx, ABSENT))
    return row, col


def _approaching(prev_plane, curr_plane, avatar_r, avatar_c):
    """Generic looming reduction: one-step optical flow (newly-occupied
    cells matched to their nearest newly-vacated origin), keeping the
    nearest cell whose move reduced its distance to the avatar."""
    batch = curr_plane.shape[0]
    rows = torch.full((batch,), ABSENT, dtype=torch.long)
    cols = torch.full((batch,), ABSENT, dtype=torch.long)
    for i in range(batch):
        if int(avatar_r[i]) >= VALUES:
            continue
        now = {(int(r), int(c)) for r, c in (curr_plane[i] > 0).nonzero()}
        was = {(int(r), int(c)) for r, c in (prev_plane[i] > 0).nonzero()}
        fresh, gone = now - was, was - now
        if not fresh or not gone:
            continue
        ar, ac = int(avatar_r[i]), int(avatar_c[i])
        best = None
        for r, c in fresh:
            o = min(gone, key=lambda g: abs(g[0] - r) + abs(g[1] - c))
            d_new = abs(r - ar) + abs(c - ac)
            d_old = abs(o[0] - ar) + abs(o[1] - ac)
            if d_new < d_old and (best is None or d_new < best[0]):
                best = (d_new, r, c)
        if best is not None:
            rows[i], cols[i] = best[1], best[2]
    return rows, cols


def make_enc(kind):
    """Slots 0-1 avatar, 2-3 nearest plane 1, 4-5 nearest plane 2.
    Slots 6/7: `second2` = second-nearest of plane 2 (F225);
    `approach2` = nearest approaching cell of plane 2 (temporal)."""
    def encoder(prev_screen, screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        prior = prev_screen.view(-1, PLANES, HEIGHT, WIDTH)
        out = torch.full((frames.shape[0], SLOTS), ABSENT, dtype=torch.long)
        avatar = frames[:, 0].reshape(frames.shape[0], -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        ar = torch.where(present, flat // WIDTH,
                         torch.full_like(flat, ABSENT))
        ac = torch.where(present, flat % WIDTH,
                         torch.full_like(flat, ABSENT))
        out[:, 0], out[:, 1] = ar, ac
        for plane, base in ((1, 2), (2, 4)):
            row, col = _kth_nearest(frames[:, plane], ar.clamp(max=VALUES-1),
                                    ac.clamp(max=VALUES-1), 0)
            out[:, base], out[:, base + 1] = row, col
        if kind == "second":
            row, col = _kth_nearest(frames[:, 2], ar.clamp(max=VALUES-1),
                                    ac.clamp(max=VALUES-1), 1)
        else:
            row, col = _approaching(prior[:, 2], frames[:, 2], ar, ac)
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


ENC_CANDIDATES = {"second2": make_enc("second"),
                  "approach2": make_enc("approach")}
enc = ENC_CANDIDATES["second2"]           # rebound per world below


def goal_cost(state, reference, goal):
    if goal and isinstance(goal[0][0], int):
        goal = (goal,)
    total = None
    for (a0, a1), (b0, b1), sign in goal:
        reach = ((state[:, a0] - reference[:, b0]).abs()
                 + (state[:, a1] - reference[:, b1]).abs()).float()
        term = sign * reach
        total = term if total is None else total + term
    return total


def build_bank(config, seed, executor):
    """One random warmup step gives every temporal reduction a real
    (prev, curr) pair before the measured transition."""
    g = torch.Generator().manual_seed(seed + 999)

    def warmed(v, s):
        v.reset(seed=s)
        first = v.observation()
        v.step(torch.randint(0, 4, (args.observations,), generator=g))
        return first, v.observation()

    probe = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + 7)
    first, second = warmed(probe, seed + 7)
    used = (enc(first, second) < VALUES).float().mean(dim=0) >= 0.9
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + act)
        first, second = warmed(v, seed + act)
        before = enc(first, second)
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = enc(second, v.observation())
        keep = ((before[:, used] < VALUES).all(dim=1)
                & (after[:, used] < VALUES).all(dim=1))
        if int(keep.sum()) < 8:
            continue
        b = torch.where(before[keep] < VALUES, before[keep],
                        torch.zeros_like(before[keep]))
        a = torch.where(after[keep] < VALUES, after[keep],
                        torch.zeros_like(after[keep]))
        bank[act] = per_slot_search(b[:args.examples], a[:args.examples])
    return bank


def play(config, mode, bank, seed, goal, executor, episodes, steps):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    prev = v.observation()
    for _ in range(steps):
        obs = v.observation()
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = enc(prev, obs)
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                if mode == "oracle":
                    shadow = copy.deepcopy(v)
                    shadow.step(torch.full((episodes,), act))
                    seen = enc(obs, shadow.observation())
                    state = torch.where(seen < VALUES, seen,
                                        torch.zeros_like(seen))
                else:
                    program = bank.get(act)
                    state = (reference if program is None
                             else executor(program, reference))
                cost = goal_cost(state, reference, goal)
                if best is None:
                    best = cost.clone()
                else:
                    take = cost < best
                    best = torch.where(take, cost, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        total += v.step(action).reward
        prev = obs
    return float(total.mean())


def usable_slots(config):
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    first = v.observation()
    g = torch.Generator().manual_seed(args.seed * 31 + 999)
    v.step(torch.randint(0, 4, (args.observations,), generator=g))
    code = enc(first, v.observation())
    present = (code < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


def diagnose(config, bank, goal, seed, episodes, steps):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    prev = v.observation()
    stats = {k: [] for k in ("agree_d1", "agree_d2", "agree_d2_given_d1",
                             "cost_corr", "best_bank_vs_true",
                             "best_oracle_vs_true", "best_bank_vs_oracle",
                             "regret", "decisive")}
    for _ in range(steps):
        obs = v.observation()
        reference = enc(prev, obs)
        alive = reference[:, 0] < VALUES
        reference = torch.where(reference < VALUES, reference,
                                torch.zeros_like(reference))
        bank_c = torch.zeros(episodes, 4, 4)
        orc_c = torch.zeros(episodes, 4, 4)
        true_r = torch.zeros(episodes, 4, 4)
        agree1 = torch.zeros(episodes, 4)
        agree2 = torch.zeros(episodes, 4, 4)
        for a1 in range(4):
            p1 = bank.get(a1)
            s1 = reference if p1 is None else plant_executor(p1, reference)
            shadow1 = copy.deepcopy(v)
            r1 = shadow1.step(torch.full((episodes,), a1)).reward
            o1raw = enc(obs, shadow1.observation())
            o1 = torch.where(o1raw < VALUES, o1raw, torch.zeros_like(o1raw))
            agree1[:, a1] = (s1 == o1).float().mean(dim=1)
            for a2 in range(4):
                p2 = bank.get(a2)
                s2 = s1 if p2 is None else plant_executor(p2, s1)
                shadow2 = copy.deepcopy(shadow1)
                r2 = shadow2.step(torch.full((episodes,), a2)).reward
                o2raw = enc(shadow1.observation(), shadow2.observation())
                o2 = torch.where(o2raw < VALUES, o2raw,
                                 torch.zeros_like(o2raw))
                agree2[:, a1, a2] = (s2 == o2).float().mean(dim=1)
                bank_c[:, a1, a2] = goal_cost(s2, reference, goal)
                orc_c[:, a1, a2] = goal_cost(o2, reference, goal)
                true_r[:, a1, a2] = r1 + r2
        rows = alive.nonzero().flatten()
        for i in rows.tolist():
            stats["agree_d1"].append(float(agree1[i].mean()))
            stats["agree_d2"].append(float(agree2[i].mean()))
            exact1 = (agree1[i] == 1.0)
            if bool(exact1.any()):
                stats["agree_d2_given_d1"].append(
                    float(agree2[i][exact1].mean()))
            b = bank_c[i].flatten(); o = orc_c[i].flatten()
            if float(b.std()) > 0 and float(o.std()) > 0:
                stats["cost_corr"].append(float(torch.corrcoef(
                    torch.stack([b, o]))[0, 1]))
            best_bank = int(bank_c[i].min(dim=1).values.argmin())
            best_orc = int(orc_c[i].min(dim=1).values.argmin())
            ret = true_r[i].max(dim=1).values
            decisive = float(ret.max() - ret.min()) > 0
            stats["decisive"].append(float(decisive))
            if not decisive:
                continue  # ties make every argmax/argmin agreement void
            best_true = int(ret.argmax())
            stats["best_bank_vs_true"].append(float(
                ret[best_bank] == ret[best_true]))
            stats["best_oracle_vs_true"].append(float(
                ret[best_orc] == ret[best_true]))
            stats["best_bank_vs_oracle"].append(float(best_bank == best_orc))
            stats["regret"].append(float(ret[best_true] - ret[best_bank]))
        # advance under the deployable depth-1 policy
        d1cost = None
        action = torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            p = bank.get(act)
            s = reference if p is None else plant_executor(p, reference)
            c = goal_cost(s, reference, goal)
            if d1cost is None:
                d1cost = c.clone()
            else:
                take = c < d1cost
                d1cost = torch.where(take, c, d1cost)
                action = torch.where(take, torch.full((episodes,), act),
                                     action)
        v.step(action)
        prev = obs
    return {k: (round(sum(vs) / len(vs), 4) if vs else None)
            for k, vs in stats.items()}


archived = json.load(open(args.pd))
assert archived["seed"] == args.seed, "seed mismatch with --pd file"
results = {}
for name, cell in archived["results"].items():
    config = dict(WORLDS).get(name) if False else None
    lookup = {
        "pursue1": FamilyConfig(pursue=1),
        "pursue1_avoid1": FamilyConfig(pursue=1, avoid=1),
        "pursue1_avoid2": FamilyConfig(pursue=1, avoid=2),
        "pursue1_collect1": FamilyConfig(pursue=1, collect=1),
        "avoid2": FamilyConfig(avoid=2),
        "avoid3": FamilyConfig(avoid=3),
        "intercept2": FamilyConfig(intercept=2),
        "collect2": FamilyConfig(collect=2),
    }
    config = lookup[name]
    globals()["enc"] = ENC_CANDIDATES[cell["encoder"]]
    bank = build_bank(config, args.seed * 31, plant_executor)
    goal = tuple((tuple(t[0]), tuple(t[1]), t[2]) for t in cell["goal"])
    row = diagnose(config, bank, goal, args.seed * 977 + 5,
                   args.episodes, args.probe_steps)
    results[name] = row
    print(f"  {name:<16} agree_d1 {row['agree_d1']}  d2 {row['agree_d2']}"
          f"  corr {row['cost_corr']}  bank=true {row['best_bank_vs_true']}"
          f"  orc=true {row['best_oracle_vs_true']}"
          f"  regret {row['regret']}", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
