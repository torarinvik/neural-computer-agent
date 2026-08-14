"""VALUE-PLAN: deployable horizon over learned dynamics (F247).

F246 localized the surviving certified headroom: for the witness trio
(intercept x pursue x resource) the deploy -> privileged-d1 gap is
only ~+0.15 while d1 -> d4 adds +0.3..+0.5, and avoid2_delayed3 is a
PURE horizon witness (ceiling jumps at exactly d4). The deployable
stack is depth-1 greedy over goal costs; the certified skill lives at
depth 2-4. The F229-deferred capability returns with a measured
prize: n-step value carriers.

This probe builds the deployable planner: depth-d exhaustive search
over COMPOSED BANK PROGRAMS (the plant executes every transition --
recipes, not weights), scored by a learned per-action linear reward
head plus a learned H-step linear value head, both ridge-fit on the
world's own random-rollout returns over the generic relational
feature basis (slots, slot-group distances, contact indicators). No
privileged access anywhere in the deployable arms.

Arms per world (6 seeds):
  random        baseline
  vplan_d1..d4  learned dynamics + learned reward/value, depth d
  vplan_it_d4   one round of policy iteration: refit heads on
                rollouts of the d4 planner itself (amortization step)
  shuffled_d4   value/reward weights row-shuffled (binding control)
  truedyn_d2    privileged dynamics + true immediate reward, learned
                value at leaves (localizes model error vs value error)

Registered predictions (before any run):
  P1 horizon witness: on avoid2_delayed3, vplan_d4 - vplan_d1 >=
     +0.10, and vplan_d4 clears the F245 deploy score (~+0.03) by
     >= +0.10. The learned value head extends the effective horizon
     past the depth cut, so d4 may even exceed the privileged
     depth-4 ceiling (+0.25) -- logged either way.
  P2 trio: best-depth vplan beats the F245/F246 deploy score on
     >= 2/3 of the intercept x pursue x resource worlds, and depth
     helps (d3-or-d4 > d1) on >= 2/3.
  P3 binding control: shuffled_d4 collapses to <= random + 0.1 --
     the value binding, not the tree search, carries any gain.
  P4 solved control: measured on ctrl_avoid1_collect1; no
     requirement (the race + guard, not this probe, decides
     deployment), but a collapse there would scope the form.
"""

from __future__ import annotations

import argparse
import copy
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
parser.add_argument("--data-batch", type=int, default=192)
parser.add_argument("--data-steps", type=int, default=20)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--ridge-lam", type=float, default=1.0)
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
print(f"interpreter check {report['interpreter_check']}", flush=True)


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


def enc(prev_screen, screen):
    """The canonical second2 encoder: avatar, nearest plane 1, nearest
    plane 2, second-nearest plane 2."""
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    out = torch.full((frames.shape[0], SLOTS), ABSENT, dtype=torch.long)
    avatar = frames[:, 0].reshape(frames.shape[0], -1)
    present = avatar.max(dim=1).values > 0
    flat = avatar.argmax(dim=1)
    ar = torch.where(present, flat // WIDTH, torch.full_like(flat, ABSENT))
    ac = torch.where(present, flat % WIDTH, torch.full_like(flat, ABSENT))
    out[:, 0], out[:, 1] = ar, ac
    for plane, base in ((1, 2), (2, 4)):
        row, col = _kth_nearest(frames[:, plane], ar.clamp(max=VALUES - 1),
                                ac.clamp(max=VALUES - 1), 0)
        out[:, base], out[:, base + 1] = row, col
    row, col = _kth_nearest(frames[:, 2], ar.clamp(max=VALUES - 1),
                            ac.clamp(max=VALUES - 1), 1)
    out[:, 6], out[:, 7] = row, col
    return out


def clamp_state(code):
    return torch.where(code < VALUES, code, torch.zeros_like(code))


GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))
GROUP_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
FEATS = 1 + SLOTS + 2 * len(GROUP_PAIRS)


def phi(states):
    """Generic relational basis over the CLAMPED slot state (the same
    representation the planner rolls): bias, raw slots, the six
    slot-group Manhattan distances, and contact indicators (d <= 1).
    Training and planning share the basis, so ABSENT-aliasing (F233's
    audit level) is consistent across both."""
    s = states.float()
    feats = [torch.ones(s.shape[0], 1), s]
    for i, j in GROUP_PAIRS:
        gi, gj = GROUPS[i], GROUPS[j]
        d = ((s[:, gi[0]] - s[:, gj[0]]).abs()
             + (s[:, gi[1]] - s[:, gj[1]]).abs())
        feats.append(d.unsqueeze(1))
        feats.append((d <= 1).float().unsqueeze(1))
    return torch.cat(feats, dim=1)


def ridge(X, y, lam):
    A = X.T @ X + lam * torch.eye(X.shape[1])
    return torch.linalg.solve(A, X.T @ y)


def build_bank(config, seed, executor):
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
        bank[act] = per_slot_search(
            clamp_state(before[keep])[:args.examples],
            clamp_state(after[keep])[:args.examples])
    return bank


def collect_experience(config, seed, policy=None):
    """Roll the world under `policy` (None = uniform random), recording
    encoded states, actions, immediate rewards.  Returns flat tensors
    plus the H-step forward return per (state, action) row."""
    B, T, H = args.data_batch, args.data_steps, args.horizon
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    states, actions, rewards = [], [], []
    prev = v.observation()
    v.step(torch.randint(0, 4, (B,), generator=g))
    for _ in range(T):
        obs = v.observation()
        code = clamp_state(enc(prev, obs))
        if policy is None:
            action = torch.randint(0, 4, (B,), generator=g)
        else:
            action = policy(code)
        states.append(code)
        actions.append(action)
        rewards.append(v.step(action).reward)
        prev = obs
    S = torch.stack(states)          # [T, B, SLOTS]
    A = torch.stack(actions)         # [T, B]
    R = torch.stack(rewards)         # [T, B]
    ret = torch.zeros(T, B)
    for t in range(T):
        ret[t] = R[t:min(T, t + H)].sum(dim=0)
    keep = T - H if T > H else T     # only rows with a full window
    return (S[:keep].reshape(-1, SLOTS), A[:keep].reshape(-1),
            R[:keep].reshape(-1), ret[:keep].reshape(-1))


def fit_heads(S, A, R, ret):
    X = phi(S)
    w_r = torch.zeros(4, FEATS)
    for act in range(4):
        rows = A == act
        if int(rows.sum()) >= FEATS:
            w_r[act] = ridge(X[rows], R[rows], args.ridge_lam)
    w_v = ridge(X, ret, args.ridge_lam)
    return w_r, w_v


def plan_actions(reference, bank, executor, w_r, w_v, depth):
    """Depth-d exhaustive search over composed bank programs.  Level
    tensors: S [K, E, SLOTS], ACC [K, E]; the first-action index of
    block k at any level is k // 4^(level-1) ... we instead track it
    explicitly.  Returns [E] long actions."""
    E = reference.shape[0]
    S = reference.unsqueeze(0)                       # [1, E, SLOTS]
    ACC = torch.zeros(1, E)
    FIRST = torch.zeros(1, dtype=torch.long)
    for level in range(depth):
        parent_phi = phi(S.reshape(-1, SLOTS))       # [K*E, F]
        new_s, new_acc, new_first = [], [], []
        for act in range(4):
            program = bank.get(act)
            flat = S.reshape(-1, SLOTS)
            child = flat if program is None else executor(program, flat)
            r = parent_phi @ w_r[act]
            new_s.append(child.view(S.shape[0], E, SLOTS))
            new_acc.append(ACC + r.view(S.shape[0], E))
            new_first.append(FIRST if level > 0
                             else torch.full((S.shape[0],), act,
                                             dtype=torch.long))
        S = torch.cat(new_s)
        ACC = torch.cat(new_acc)
        FIRST = torch.cat(new_first)
    leaf = ACC + (phi(S.reshape(-1, SLOTS)) @ w_v).view(S.shape[0], E)
    best = torch.full((E,), -1e9)
    action = torch.zeros(E, dtype=torch.long)
    for k in range(leaf.shape[0]):
        take = leaf[k] > best
        best = torch.where(take, leaf[k], best)
        action = torch.where(take, torch.full((E,), int(FIRST[k])), action)
    return action


def play(config, mode, bank, w_r, w_v, seed, depth, executor):
    episodes, steps = args.episodes, args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    prev = v.observation()
    for _ in range(steps):
        obs = v.observation()
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        elif mode == "truedyn":
            # privileged dynamics + true immediate reward, learned
            # value at the leaves (localization arm, depth fixed)
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                s1 = copy.deepcopy(v)
                gain = s1.step(torch.full((episodes,), act)).reward
                sub = None
                for act2 in range(4):
                    s2 = copy.deepcopy(s1)
                    g2 = s2.step(torch.full((episodes,), act2)).reward
                    code = clamp_state(enc(s1.observation(),
                                           s2.observation()))
                    leaf = gain + g2 + phi(code) @ w_v
                    sub = leaf if sub is None else torch.maximum(sub, leaf)
                if best is None:
                    best = sub.clone()
                else:
                    take = sub > best
                    best = torch.where(take, sub, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        else:
            reference = clamp_state(enc(prev, obs))
            action = plan_actions(reference, bank, executor, w_r, w_v,
                                  depth)
        total += v.step(action).reward
        prev = obs
    return float(total.mean())


# The five F245 found witnesses + one solved control
WORLDS = [
    ("collect1_intercept1_pursue1_resource1",
     FamilyConfig(collect=1, intercept=1, pursue=1, resource=1)),
    ("delayed3_intercept1_pursue1_resource2",
     FamilyConfig(delayed=3, intercept=1, pursue=1, resource=2)),
    ("delayed3_intercept2_pursue1_resource1",
     FamilyConfig(delayed=3, intercept=2, pursue=1, resource=1)),
    ("avoid3_collect3_delayed5_resource1",
     FamilyConfig(avoid=3, collect=3, delayed=5, resource=1)),
    ("avoid2_delayed3", FamilyConfig(avoid=2, delayed=3)),
    ("ctrl_avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
]

report["results"] = {}
for name, config in WORLDS:
    row = {}
    bank = build_bank(config, args.seed * 31, plant_executor)
    S, A, R, ret = collect_experience(config, args.seed * 53)
    w_r, w_v = fit_heads(S, A, R, ret)
    row["random"] = play(config, "random", None, None, None,
                         args.seed * 977, 0, None)
    for depth in (1, 2, 3, 4):
        row[f"vplan_d{depth}"] = play(config, "vplan", bank, w_r, w_v,
                                      args.seed * 977, depth,
                                      plant_executor)
    # one round of policy iteration: roll the d4 planner, refit heads
    # on combined data, re-evaluate
    def d4_policy(code):
        return plan_actions(code, bank, plant_executor, w_r, w_v, 4)
    S2, A2, R2, ret2 = collect_experience(config, args.seed * 59,
                                          policy=d4_policy)
    w_r2, w_v2 = fit_heads(torch.cat([S, S2]), torch.cat([A, A2]),
                           torch.cat([R, R2]), torch.cat([ret, ret2]))
    row["vplan_it_d4"] = play(config, "vplan", bank, w_r2, w_v2,
                              args.seed * 977, 4, plant_executor)
    # binding control: row-shuffle the fitted weights
    perm = torch.randperm(FEATS,
                          generator=torch.Generator().manual_seed(
                              args.seed + 8080))
    row["shuffled_d4"] = play(config, "vplan", bank, w_r[:, perm],
                              w_v[perm], args.seed * 977, 4,
                              plant_executor)
    # localization arm: privileged 2-step dynamics + learned value
    row["truedyn_d2"] = play(config, "truedyn", None, None, w_v,
                             args.seed * 977, 2, None)
    report["results"][name] = row
    print(f"  {name:<40} " + "  ".join(
        f"{k} {v:+.3f}" for k, v in row.items()), flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
