"""EVALUATOR PLANNING: the F228 fix — utility as a bankside skill.

F228's witness: true-utility lookahead pays (+0.143, t=+3.78) where
every state-cost goal fails under both dynamics. The missing piece is
therefore a learned TRANSITION-UTILITY EVALUATOR — (state, action
effect) -> predicted reward — fit from the system's OWN experienced
rewards, stored beside programs and goals, never inside the plant.
This probe builds the minimal deployable version:

  features  generic, no game semantics: for every pair of slot groups
            (avatar, plane-1 tracker, plane-2 tracker, flex tracker),
            the after-to-before distance and its contact indicator;
            the after-state avatar-to-object distances and contacts;
            a death flag; a bias. ~40 features.
  fit       ridge regression on (encoded before, encoded after,
            reward) triples from random rollouts (2 x 256 episodes x
            12 steps), exactly the F223 evidence protocol. Encoder
            candidate (second2 vs approach2) chosen by held-half R^2 —
            no return-based selection anywhere in this probe.
  planning  depth 1: argmax_a rhat(ref -> s1_a).
            depth 2: argmax_a [rhat(ref->s1) + max_b rhat(s1->s2)],
            states from the SAME bank programs as every prior probe.

Baselines, same evaluation stream: random; the archived state-cost
depth-1 goal (--pd) executed unchanged. Ceilings: F228's true-return
arm numbers.

Registered predictions:
  1. evaluator-d2 > evaluator-d1 on the event worlds (collect2,
     intercept2, pursue1_collect1): the utility signal is what makes
     depth cashable.
  2. evaluator-d2 >= state-cost-d1 baseline on collect2 (consumption
     finally visible). intercept2 UNCERTAIN: the archived proxy
     (+0.15) is strong anticipatory shaping and the evaluator must
     rediscover it from sparse catches.
  3. avoid worlds: parity with baseline (F228 measured only 4-7%
     decisive states); the evaluator should at least learn that
     hazard contact is negative.
  4. Failure mode to watch: bank programs never predict death, so the
     evaluator sees predicted states that omit the outcome it must
     price; contact features on predicted states are the mitigation
     (F223). Recorded either way.
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


GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))


def features(before, after, dead):
    cols = [torch.ones(before.shape[0]), dead.float()]
    for ga in GROUPS:
        for gb in GROUPS:
            d = ((after[:, ga[0]] - before[:, gb[0]]).abs()
                 + (after[:, ga[1]] - before[:, gb[1]]).abs()).float()
            cols += [d, (d == 0).float()]
    for gb in GROUPS[1:]:
        d = ((after[:, 0] - after[:, gb[0]]).abs()
             + (after[:, 1] - after[:, gb[1]]).abs()).float()
        cols += [d, (d == 0).float()]
    return torch.stack(cols, dim=1)


def collect_experience(config, seed):
    xs, rs = [], []
    for arm in range(2):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + arm * 71)
        v.reset(seed=seed + arm * 71)
        g = torch.Generator().manual_seed(seed + arm * 71 + 13)
        prev = v.observation()
        obs = prev
        for _ in range(12):
            before_raw = enc(prev, obs)
            act = torch.randint(0, 4, (args.observations,), generator=g)
            r = v.step(act).reward
            nxt = v.observation()
            after_raw = enc(obs, nxt)
            ok = before_raw[:, 0] < VALUES
            dead = after_raw[:, 0] >= VALUES
            b = torch.where(before_raw < VALUES, before_raw,
                            torch.zeros_like(before_raw))
            a = torch.where(after_raw < VALUES, after_raw,
                            torch.zeros_like(after_raw))
            xs.append(features(b[ok], a[ok], dead[ok]))
            rs.append(r[ok])
            prev, obs = obs, nxt
    return torch.cat(xs), torch.cat(rs)


def fit_evaluator(x, r):
    lam = 1.0
    xtx = x.T @ x + lam * torch.eye(x.shape[1])
    w = torch.linalg.solve(xtx, x.T @ r)
    return w


def r2(w, x, r):
    pred = x @ w
    ss = float(((r - r.mean()) ** 2).sum())
    return 1.0 - float(((r - pred) ** 2).sum()) / ss if ss > 0 else 0.0


def rhat(w, before, after):
    dead = torch.zeros(before.shape[0], dtype=torch.bool)
    return features(before, after, dead) @ w


def play_eval(config, bank, w, seed, episodes, steps, depth):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    total = torch.zeros(episodes)
    prev = v.observation()
    for _ in range(steps):
        obs = v.observation()
        reference = enc(prev, obs)
        reference = torch.where(reference < VALUES, reference,
                                torch.zeros_like(reference))
        best, action = None, torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            p = bank.get(act)
            s1 = reference if p is None else plant_executor(p, reference)
            gain = rhat(w, reference, s1)
            if depth > 1:
                deeper = None
                for nxt in range(4):
                    p2 = bank.get(nxt)
                    s2 = s1 if p2 is None else plant_executor(p2, s1)
                    g2 = rhat(w, s1, s2)
                    deeper = g2 if deeper is None else torch.maximum(
                        deeper, g2)
                gain = gain + deeper
            if best is None:
                best = gain.clone()
            else:
                take = gain > best
                best = torch.where(take, gain, best)
                action = torch.where(take, torch.full((episodes,), act),
                                     action)
        total += v.step(action).reward
        prev = obs
    return float(total.mean())


archived = json.load(open(args.pd))
assert archived["seed"] == args.seed, "seed mismatch with --pd file"
LOOKUP = {
    "pursue1": FamilyConfig(pursue=1),
    "pursue1_avoid1": FamilyConfig(pursue=1, avoid=1),
    "pursue1_avoid2": FamilyConfig(pursue=1, avoid=2),
    "pursue1_collect1": FamilyConfig(pursue=1, collect=1),
    "avoid2": FamilyConfig(avoid=2),
    "avoid3": FamilyConfig(avoid=3),
    "intercept2": FamilyConfig(intercept=2),
    "collect2": FamilyConfig(collect=2),
}
results = {}
for name, cell in archived["results"].items():
    config = LOOKUP[name]
    picked = None
    for label, candidate in ENC_CANDIDATES.items():
        globals()["enc"] = candidate
        x, r = collect_experience(config, args.seed * 31)
        half = x.shape[0] // 2
        w = fit_evaluator(x[:half], r[:half])
        held = r2(w, x[half:], r[half:])
        if picked is None or held > picked[0]:
            picked = (held, label, candidate, fit_evaluator(x, r))
    held_r2, label, candidate, w = picked
    globals()["enc"] = candidate
    bank = build_bank(config, args.seed * 31, plant_executor)
    goal = tuple((tuple(t[0]), tuple(t[1]), t[2]) for t in cell["goal"])
    baseline_enc = ENC_CANDIDATES[cell["encoder"]]
    globals()["enc"] = baseline_enc
    base_bank = build_bank(config, args.seed * 31, plant_executor)
    base = play(config, "bank", base_bank, args.seed * 977, goal,
                plant_executor, args.episodes, args.steps)
    globals()["enc"] = candidate
    row = {"encoder": label, "held_r2": round(held_r2, 4),
           "statecost_d1": base,
           "eval_d1": play_eval(config, bank, w, args.seed * 977,
                                args.episodes, args.steps, 1),
           "eval_d2": play_eval(config, bank, w, args.seed * 977,
                                args.episodes, args.steps, 2)}
    results[name] = row
    print(f"  {name:<16} r2 {row['held_r2']:+.3f}  cost_d1 "
          f"{row['statecost_d1']:+.3f}  eval_d1 {row['eval_d1']:+.3f}  "
          f"eval_d2 {row['eval_d2']:+.3f}  [{label}]", flush=True)

report["results"] = results
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
