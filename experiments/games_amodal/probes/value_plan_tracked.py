"""VALUE-PLAN over IDENTITY-TRACKED slots: the integration shot at
the certified trio gap (F250).

F247 built the horizon (depth-d search over plant-executed bank
programs + learned reward/value heads; triples the privileged d4
ceiling where dynamics are modelable) and localized the witness
failure to dynamics. F248/F249 localized the dynamics failure to
IDENTITY: nearest-rank slots swap entities when movers cross (label
noise, mover fits 0.38-0.45), and continuity tracking alone converts
the residual (0.72-0.85) -- with no new ISA primitive needed. F249
also exposed a modulus-overfit tie-break in per_slot_search (small
moduli tie large ones on early-rollout data and wrap wrongly later).

This probe integrates all three: the slot state is produced by a
continuity TRACKER (avatar + one plane-1 entity + two plane-2
entities, matched frame-to-frame within a distance budget, else
re-acquired by rank -- the F235 EntityTable principle in slot form);
the bank is fit per action from one long tracked rollout under the
deployed used-mask rule, with the tie-break fixed (moduli searched
LARGEST first, NOOP keeps ties); heads and planner are F247's,
unchanged. The plant executes every planned transition.

Arms per world: random | vplan_rank_d4 (same protocol, rank slots --
the encoding-only control) | vplan_track d1..d4 | vplan_track_it_d4
(one policy-iteration round) | shuffled_d4 (binding control).

Registered predictions (before any run):
  P1 on >= 2/3 trio worlds, best tracked arm beats the F245/F246
     deploy scores (-0.86 / -0.70 / -0.84) by >= +0.15.
  P2 encoding is the active ingredient: vplan_track_d4 >
     vplan_rank_d4 on >= 2/3 trio worlds (per-seed paired).
  P3 shuffled_d4 collapses to <= random + 0.1.
  P4 control world keeps F247-level capability: tracked it_d4
     >= +0.8 on ctrl_avoid1_collect1.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--fit-examples", type=int, default=256)
parser.add_argument("--bank-batch", type=int, default=256)
parser.add_argument("--bank-steps", type=int, default=24)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--data-batch", type=int, default=192)
parser.add_argument("--data-steps", type=int, default=20)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--ridge-lam", type=float, default=1.0)
parser.add_argument("--match-dist", type=int, default=2)
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
    """F249 tie-break fix: moduli searched LARGEST first, and ties
    never replace (strict >), so NOOP (op 0) keeps ties and INC mod8
    beats INC mod5 when they agree on train."""
    program = []
    for s in range(SLOTS):
        want = after[:, s]
        best, best_score = NOOP, -1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j == s and PAR_OPS[op] in ("CINC", "CDEC", "COPY"):
                    continue
                for m in reversed(range(len(MODULI))):
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


# ---- perception: rank slots and the identity tracker ---------------

def frame_cells(obs):
    return obs.view(-1, PLANES, HEIGHT, WIDTH)


def rank_encode(frames):
    B = frames.shape[0]
    out = torch.full((B, SLOTS), ABSENT, dtype=torch.long)
    avatar = frames[:, 0].reshape(B, -1)
    present = avatar.max(dim=1).values > 0
    flat = avatar.argmax(dim=1)
    ar = torch.where(present, flat // WIDTH, torch.full_like(flat, ABSENT))
    ac = torch.where(present, flat % WIDTH, torch.full_like(flat, ABSENT))
    out[:, 0], out[:, 1] = ar, ac
    for b in range(B):
        if int(ar[b]) >= VALUES:
            continue
        for plane, base, k in ((1, 2, 0), (2, 4, 0), (2, 6, 1)):
            cells = (frames[b, plane] > 0).nonzero()
            if cells.shape[0] <= k:
                continue
            d = ((cells[:, 0] - ar[b]).abs() + (cells[:, 1] - ac[b]).abs())
            order = d.argsort()
            out[b, base] = cells[order[k], 0]
            out[b, base + 1] = cells[order[k], 1]
    return out


class Tracker:
    """Identity-stable slot state: group 2-3 one plane-1 entity,
    groups 4-5 / 6-7 two plane-2 entities.  Continuity matching
    within match-dist; otherwise re-acquire by nearest rank among
    unclaimed cells."""

    def __init__(self, batch):
        self.pos = [[None, None, None] for _ in range(batch)]

    def encode(self, frames):
        B = frames.shape[0]
        out = torch.full((B, SLOTS), ABSENT, dtype=torch.long)
        avatar = frames[:, 0].reshape(B, -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        for b in range(B):
            if not bool(present[b]):
                self.pos[b] = [None, None, None]
                continue
            ar, ac = int(flat[b]) // WIDTH, int(flat[b]) % WIDTH
            out[b, 0], out[b, 1] = ar, ac
            claimed = set()
            for t_ix, (plane, base) in enumerate(
                    ((1, 2), (2, 4), (2, 6))):
                cells = [(int(r), int(c)) for r, c in
                         (frames[b, plane] > 0).nonzero()
                         if (plane, int(r), int(c)) not in claimed]
                prev = self.pos[b][t_ix]
                pick = None
                if prev is not None and cells:
                    near = min(cells, key=lambda z: abs(z[0] - prev[0])
                               + abs(z[1] - prev[1]))
                    if (abs(near[0] - prev[0]) + abs(near[1] - prev[1])
                            <= args.match_dist):
                        pick = near
                if pick is None and cells:
                    rank = 0 if base != 6 else 1
                    cells_sorted = sorted(
                        cells, key=lambda z: abs(z[0] - ar)
                        + abs(z[1] - ac))
                    if len(cells_sorted) > rank:
                        pick = cells_sorted[rank]
                self.pos[b][t_ix] = pick
                if pick is not None:
                    claimed.add((plane, pick[0], pick[1]))
                    out[b, base], out[b, base + 1] = pick
        return out


def make_encoder(encoding, batch):
    if encoding == "rank":
        return lambda frames: rank_encode(frames)
    tracker = Tracker(batch)
    return tracker.encode


def clamp_state(code):
    return torch.where(code < VALUES, code, torch.zeros_like(code))


# ---- heads: generic relational basis + ridge (F247) ----------------

GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))
GROUP_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
FEATS = 1 + SLOTS + 2 * len(GROUP_PAIRS)


def phi(states):
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


def fit_heads(S, A, R, ret):
    X = phi(S)
    w_r = torch.zeros(4, FEATS)
    for act in range(4):
        rows = A == act
        if int(rows.sum()) >= FEATS:
            w_r[act] = ridge(X[rows], R[rows], args.ridge_lam)
    w_v = ridge(X, ret, args.ridge_lam)
    return w_r, w_v


# ---- bank: fit per action from one long tracked rollout ------------

def build_bank(config, seed, encoding):
    B, T = args.bank_batch, args.bank_steps
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 999)
    encoder = make_encoder(encoding, B)
    codes = [encoder(frame_cells(v.observation()))]
    acts = []
    for _ in range(T):
        action = torch.randint(0, 4, (B,), generator=g)
        v.step(action)
        acts.append(action)
        codes.append(encoder(frame_cells(v.observation())))
    stacked = torch.stack(codes)                     # [T+1, B, SLOTS]
    used = (stacked.reshape(-1, SLOTS) < VALUES).float().mean(dim=0) >= 0.9
    bank = {}
    for act in range(4):
        bs, as_ = [], []
        for t in range(T):
            rows = ((acts[t] == act)
                    & (codes[t][:, used] < VALUES).all(dim=1)
                    & (codes[t + 1][:, used] < VALUES).all(dim=1))
            if bool(rows.any()):
                bs.append(codes[t][rows])
                as_.append(codes[t + 1][rows])
        if not bs:
            continue
        before = clamp_state(torch.cat(bs))[:args.fit_examples]
        after = clamp_state(torch.cat(as_))[:args.fit_examples]
        if before.shape[0] < 8:
            continue
        bank[act] = per_slot_search(before, after)
    return bank


# ---- data for heads ------------------------------------------------

def collect_experience(config, seed, encoding, policy=None):
    B, T, H = args.data_batch, args.data_steps, args.horizon
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    encoder = make_encoder(encoding, B)
    states, actions, rewards = [], [], []
    code = clamp_state(encoder(frame_cells(v.observation())))
    for _ in range(T):
        if policy is None:
            action = torch.randint(0, 4, (B,), generator=g)
        else:
            action = policy(code)
        states.append(code)
        actions.append(action)
        rewards.append(v.step(action).reward)
        code = clamp_state(encoder(frame_cells(v.observation())))
    S = torch.stack(states)
    A = torch.stack(actions)
    R = torch.stack(rewards)
    ret = torch.zeros(T, B)
    for t in range(T):
        ret[t] = R[t:min(T, t + H)].sum(dim=0)
    keep = T - H if T > H else T
    return (S[:keep].reshape(-1, SLOTS), A[:keep].reshape(-1),
            R[:keep].reshape(-1), ret[:keep].reshape(-1))


# ---- planner (F247, unchanged) -------------------------------------

def plan_actions(reference, bank, executor, w_r, w_v, depth):
    E = reference.shape[0]
    S = reference.unsqueeze(0)
    ACC = torch.zeros(1, E)
    FIRST = torch.zeros(1, dtype=torch.long)
    for level in range(depth):
        parent_phi = phi(S.reshape(-1, SLOTS))
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


def play(config, mode, bank, w_r, w_v, seed, depth, encoding):
    episodes, steps = args.episodes, args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    encoder = make_encoder(encoding, episodes)
    total = torch.zeros(episodes)
    for _ in range(steps):
        frames = frame_cells(v.observation())
        if mode == "random":
            encoder(frames)          # keep tracker state comparable
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = clamp_state(encoder(frames))
            action = plan_actions(reference, bank, plant_executor,
                                  w_r, w_v, depth)
        total += v.step(action).reward
    return float(total.mean())


WORLDS = [
    ("collect1_intercept1_pursue1_resource1",
     FamilyConfig(collect=1, intercept=1, pursue=1, resource=1)),
    ("delayed3_intercept1_pursue1_resource2",
     FamilyConfig(delayed=3, intercept=1, pursue=1, resource=2)),
    ("delayed3_intercept2_pursue1_resource1",
     FamilyConfig(delayed=3, intercept=2, pursue=1, resource=1)),
    ("avoid2_delayed3", FamilyConfig(avoid=2, delayed=3)),
    ("ctrl_avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
]

report["results"] = {}
for name, config in WORLDS:
    row = {}
    row["random"] = play(config, "random", None, None, None,
                         args.seed * 977, 0, "rank")
    # encoding-only control: rank slots, same rollout-fit protocol
    bank_r = build_bank(config, args.seed * 31, "rank")
    Sr, Ar, Rr, retr = collect_experience(config, args.seed * 53, "rank")
    wr_r, wv_r = fit_heads(Sr, Ar, Rr, retr)
    row["vplan_rank_d4"] = play(config, "vplan", bank_r, wr_r, wv_r,
                                args.seed * 977, 4, "rank")
    # tracked arms
    bank = build_bank(config, args.seed * 31, "track")
    S, A, R, ret = collect_experience(config, args.seed * 53, "track")
    w_r, w_v = fit_heads(S, A, R, ret)
    for depth in (1, 2, 3, 4):
        row[f"vplan_track_d{depth}"] = play(
            config, "vplan", bank, w_r, w_v, args.seed * 977, depth,
            "track")
    def d4_policy(code):
        return plan_actions(code, bank, plant_executor, w_r, w_v, 4)
    S2, A2, R2, ret2 = collect_experience(config, args.seed * 59,
                                          "track", policy=d4_policy)
    w_r2, w_v2 = fit_heads(torch.cat([S, S2]), torch.cat([A, A2]),
                           torch.cat([R, R2]), torch.cat([ret, ret2]))
    row["vplan_track_it_d4"] = play(config, "vplan", bank, w_r2, w_v2,
                                    args.seed * 977, 4, "track")
    perm = torch.randperm(FEATS,
                          generator=torch.Generator().manual_seed(
                              args.seed + 8080))
    row["shuffled_d4"] = play(config, "vplan", bank, w_r[:, perm],
                              w_v[perm], args.seed * 977, 4, "track")
    report["results"][name] = row
    print(f"  {name:<40} " + "  ".join(
        f"{k} {v:+.3f}" for k, v in row.items()), flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
