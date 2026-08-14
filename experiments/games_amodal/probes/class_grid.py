"""CLASS-TARGET HEAD ON THE HOME SUBSTRATE: the F267 identifiability
fix folded back into the grid trio stack (F268).

F267 solved amodal-transfer reliability with class-target event
supervision (five coefficients, native ceiling 6/6). The pooled
F256 head that anchors the grid deployable frontier carries the
same credit-spreading hazard. Here the hybrid action rule (F264)
races BOTH head forms:

  pooled:  goal_cost - lam * (x_events @ w_tr)     [F264 form]
  class:   goal_cost - lam * sum_g kernel(d_g) * v_hat(psi_g)

with (goal, lam, head) raced per world, pure endpoints included.

Registered predictions (before any run):
  P1 the class head advances the trio: best class arm >= best
     pooled arm + 0.05 on >= 2/3 trio worlds.
  P2 race safety: raced-best >= F264's hybrid - 0.05 everywhere.
  P3 shuffled class weights collapse the class term's gain.
  P4 dense worlds unharmed (race falls back).
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
    """F249 tie-break fix: moduli largest-first, strict replacement."""
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


# ---- perception: identity tracker with type signatures -------------

def frame_cells(obs):
    return obs.view(-1, PLANES, HEIGHT, WIDTH)


TRACKED = ((1, 2), (2, 4), (2, 6))       # (plane, base slot) per entity
PSI_DIM = 5                              # 1, plane1, plane2, motion, approach


class Tracker:
    def __init__(self, batch):
        self.pos = [[None, None, None] for _ in range(batch)]
        self.steps = torch.zeros(batch, 3)
        self.move = torch.zeros(batch, 3)
        self.closer = torch.zeros(batch, 3)

    def clone(self):
        out = Tracker(len(self.pos))
        out.pos = [list(p) for p in self.pos]
        out.steps = self.steps.clone()
        out.move = self.move.clone()
        out.closer = self.closer.clone()
        return out

    def psi(self):
        """[B, 3, PSI_DIM] generic type signatures."""
        B = len(self.pos)
        steps = self.steps.clamp(min=1)
        out = torch.zeros(B, 3, PSI_DIM)
        out[:, :, 0] = 1.0
        for t_ix, (plane, _base) in enumerate(TRACKED):
            out[:, t_ix, 1] = 1.0 if plane == 1 else 0.0
            out[:, t_ix, 2] = 1.0 if plane == 2 else 0.0
        out[:, :, 3] = self.move / steps
        out[:, :, 4] = self.closer / steps
        return out

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
            for t_ix, (plane, base) in enumerate(TRACKED):
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
                if prev is not None and pick is not None:
                    d_move = (abs(pick[0] - prev[0])
                              + abs(pick[1] - prev[1]))
                    if d_move <= args.match_dist:   # identity held
                        self.steps[b, t_ix] += 1
                        self.move[b, t_ix] += d_move
                        d_prev = abs(prev[0] - ar) + abs(prev[1] - ac)
                        d_now = abs(pick[0] - ar) + abs(pick[1] - ac)
                        if d_now < d_prev:
                            self.closer[b, t_ix] += 1
                self.pos[b][t_ix] = pick
                if pick is not None:
                    claimed.add((plane, pick[0], pick[1]))
                    out[b, base], out[b, base + 1] = pick
        return out


def clamp_state(code):
    return torch.where(code < VALUES, code, torch.zeros_like(code))


# ---- heads ---------------------------------------------------------

GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))
GROUP_PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
V_FEATS = 1 + SLOTS + 2 * len(GROUP_PAIRS)
EVENTS = 4                               # contact, teleport, both, boundary
R_FEATS = 3 * EVENTS * PSI_DIM + 4 + 1   # groups x events x psi + act + bias


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


def event_features(parent, child, actions, psi, died=None,
                   soft=False):
    """[N, R_FEATS] typed event features for parent->child transitions.

    psi: [N, 3, PSI_DIM].  died: bool [N] rows where the avatar
    vanished at child time (death attribution: contact := entity
    within 2 at parent).  soft=True replaces the binary contact
    predicate with the graded kernel clamp((2 - d)/2, 0, 1), so a
    one-cell dynamics error degrades the score instead of flipping
    it (F253)."""
    N = parent.shape[0]
    p, c = parent.float(), child.float()
    av_r, av_c = c[:, 0], c[:, 1]
    feats = [torch.ones(N, 1)]
    for t_ix, (_plane, base) in enumerate(TRACKED):
        er, ec = c[:, base], c[:, base + 1]
        pr, pc = p[:, base], p[:, base + 1]
        d_now = (av_r - er).abs() + (av_c - ec).abs()
        d_prev_cell = (av_r - pr).abs() + (av_c - pc).abs()
        if soft:
            contact = torch.maximum(
                ((2.0 - d_now) / 2.0).clamp(0.0, 1.0),
                (d_prev_cell == 0).float())
        else:
            contact = ((d_now <= 1) | (d_prev_cell == 0)).float()
        if died is not None:
            d_at_parent = ((p[:, 0] - pr).abs()
                           + (p[:, 1] - pc).abs())
            contact = torch.where(died, (d_at_parent <= 2).float(),
                                  contact)
        jump = (er - pr).abs() + (ec - pc).abs()
        teleport = (jump > 2).float()
        if died is not None:
            teleport = teleport * (~died).float()
        both = contact * teleport
        contact_only = contact * (1.0 - teleport)
        boundary = ((er == 0) | (er == HEIGHT - 1)
                    | (ec == 0) | (ec == WIDTH - 1)).float()
        ev = torch.stack([contact_only, teleport, both, boundary],
                         dim=1)
        feats.append((ev.unsqueeze(2) * psi[:, t_ix].unsqueeze(1))
                     .reshape(N, EVENTS * PSI_DIM))
    feats.append(torch.nn.functional.one_hot(actions, 4).float())
    return torch.cat(feats, dim=1)


def ridge(X, y, lam):
    A = X.T @ X + lam * torch.eye(X.shape[1])
    return torch.linalg.solve(A, X.T @ y)


# ---- data ----------------------------------------------------------

def collect_experience(config, seed, policy=None):
    """Tracked rollout recording states, psi snapshots, actions,
    rewards, avatar-vanish flags, and H-step returns."""
    B, T, H = args.data_batch, args.data_steps, args.horizon
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = Tracker(B)
    raw, psis, actions, rewards, alive = [], [], [], [], []
    code = tracker.encode(frame_cells(v.observation()))
    raw.append(code)
    psis.append(tracker.psi())
    alive.append(code[:, 0] < VALUES)
    for _ in range(T):
        state = clamp_state(code)
        if policy is None:
            action = torch.randint(0, 4, (B,), generator=g)
        else:
            action = policy(state, tracker.psi())
        actions.append(action)
        rewards.append(v.step(action).reward)
        code = tracker.encode(frame_cells(v.observation()))
        raw.append(code)
        psis.append(tracker.psi())
        alive.append(code[:, 0] < VALUES)
    return raw, psis, actions, rewards, alive


def build_head_data(raw, psis, actions, rewards, alive, soft=False,
                    bank=None):
    """Event-feature rows from a recorded rollout.  A row qualifies
    if the avatar was present at t; death at t+1 flows into the died
    flag (attribution).  With bank set, the child state is the
    BANK-ROLLED prediction from the parent for the taken action --
    model-consistent training (F253): the head is fit in the exact
    representation the planner will hand it."""
    xs, ys = [], []
    v_states, v_rets = [], []
    T = len(actions)
    R = torch.stack(rewards)
    for t in range(T):
        ok = alive[t]
        if not bool(ok.any()):
            continue
        died = ok & ~alive[t + 1]
        parent = clamp_state(raw[t][ok])
        if bank is None:
            child = clamp_state(raw[t + 1][ok])
        else:
            child = parent.clone()
            acts_ok = actions[t][ok]
            for a in range(4):
                rows = acts_ok == a
                program = bank.get(a)
                if program is not None and bool(rows.any()):
                    child[rows] = plant_executor(program, parent[rows])
        x = event_features(parent, child, actions[t][ok], psis[t][ok],
                           died=died[ok], soft=soft)
        xs.append(x)
        ys.append(rewards[t][ok])
        if t + args.horizon <= T:
            v_states.append(clamp_state(raw[t][ok]))
            v_rets.append(R[t:t + args.horizon, ok].sum(dim=0))
    X = torch.cat(xs)
    y = torch.cat(ys)
    S = torch.cat(v_states)
    ret = torch.cat(v_rets)
    return X, y, S, ret


def fit_typed(X, y, S, ret):
    w_r = ridge(X, y, args.ridge_lam)
    w_v = ridge(phi(S), ret, args.ridge_lam)
    return w_r, w_v


def fit_linear(raw, psis, actions, rewards, alive):
    """The F250 linear head on the same rollout (paired baseline)."""
    Ss, As, Rs, rets = [], [], [], []
    T = len(actions)
    R = torch.stack(rewards)
    for t in range(T):
        ok = alive[t]
        if not bool(ok.any()) or t + args.horizon > T:
            continue
        Ss.append(clamp_state(raw[t][ok]))
        As.append(actions[t][ok])
        Rs.append(rewards[t][ok])
        rets.append(R[t:t + args.horizon, ok].sum(dim=0))
    S = torch.cat(Ss); A = torch.cat(As)
    Rv = torch.cat(Rs); ret = torch.cat(rets)
    X = phi(S)
    w_r = torch.zeros(4, V_FEATS)
    for act in range(4):
        rows = A == act
        if int(rows.sum()) >= V_FEATS:
            w_r[act] = ridge(X[rows], Rv[rows], args.ridge_lam)
    w_v = ridge(X, ret, args.ridge_lam)
    return w_r, w_v


# ---- bank ----------------------------------------------------------

def build_bank(config, seed):
    B, T = args.bank_batch, args.bank_steps
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 999)
    tracker = Tracker(B)
    codes = [tracker.encode(frame_cells(v.observation()))]
    acts = []
    for _ in range(T):
        action = torch.randint(0, 4, (B,), generator=g)
        v.step(action)
        acts.append(action)
        codes.append(tracker.encode(frame_cells(v.observation())))
    stacked = torch.stack(codes)
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


# ---- planner -------------------------------------------------------

def plan_typed(reference, psi, bank, w_r, w_v, depth, soft=False):
    """Tree over bank programs; edge score = typed event head on the
    (parent, child) pair; psi is the root snapshot, tiled."""
    E = reference.shape[0]
    S = reference.unsqueeze(0)
    ACC = torch.zeros(1, E)
    FIRST = torch.zeros(1, dtype=torch.long)
    for level in range(depth):
        new_s, new_acc, new_first = [], [], []
        K = S.shape[0]
        flat = S.reshape(-1, SLOTS)
        psi_t = psi.repeat(K, 1, 1)
        for act in range(4):
            program = bank.get(act)
            child = flat if program is None else plant_executor(
                program, flat)
            x = event_features(flat, child,
                               torch.full((K * E,), act,
                                          dtype=torch.long), psi_t,
                               soft=soft)
            r = x @ w_r
            new_s.append(child.view(K, E, SLOTS))
            new_acc.append(ACC + r.view(K, E))
            new_first.append(FIRST if level > 0
                             else torch.full((K,), act,
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


def plan_linear(reference, bank, w_r, w_v, depth):
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
            child = flat if program is None else plant_executor(
                program, flat)
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


def play(config, mode, bank, w_r, w_v, seed, depth, soft=False):
    episodes, steps = args.episodes, args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    tracker = Tracker(episodes)
    total = torch.zeros(episodes)
    for _ in range(steps):
        frames = frame_cells(v.observation())
        code = tracker.encode(frames)
        state = clamp_state(code)
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        elif mode == "typed":
            action = plan_typed(state, tracker.psi(), bank, w_r, w_v,
                                depth, soft=soft)
        elif mode == "linear":
            action = plan_linear(state, bank, w_r, w_v, depth)
        elif mode == "truedyn_typed":
            X_now = None
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            psi_now = tracker.psi()
            for act in range(4):
                s1 = copy.deepcopy(v)
                s1.step(torch.full((episodes,), act))
                t1 = tracker.clone()
                code1 = t1.encode(frame_cells(s1.observation()))
                died1 = (code[:, 0] < VALUES) & (code1[:, 0] >= VALUES)
                x1 = event_features(state, clamp_state(code1),
                                    torch.full((episodes,), act,
                                               dtype=torch.long),
                                    psi_now, died=died1)
                r1 = x1 @ w_r
                sub = None
                for act2 in range(4):
                    s2 = copy.deepcopy(s1)
                    s2.step(torch.full((episodes,), act2))
                    t2 = t1.clone()
                    code2 = t2.encode(frame_cells(s2.observation()))
                    died2 = ((code1[:, 0] < VALUES)
                             & (code2[:, 0] >= VALUES))
                    x2 = event_features(clamp_state(code1),
                                        clamp_state(code2),
                                        torch.full((episodes,), act2,
                                                   dtype=torch.long),
                                        t1.psi(), died=died2)
                    r2 = x2 @ w_r
                    leaf = r1 + r2 + phi(clamp_state(code2)) @ w_v
                    sub = leaf if sub is None else torch.maximum(sub,
                                                                 leaf)
                if best is None:
                    best = sub.clone()
                else:
                    take = sub > best
                    best = torch.where(take, sub, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        total += v.step(action).reward
    return float(total.mean())


TRAIN_WORLDS = [
    ("collect1", FamilyConfig(collect=1)),
    ("avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
    ("pursue1", FamilyConfig(pursue=1)),
    ("intercept1", FamilyConfig(intercept=1)),
    ("collect1_resource1", FamilyConfig(collect=1, resource=1)),
    ("delayed3", FamilyConfig(delayed=3)),
]

TARGET_WORLDS = [
    ("collect1_intercept1_pursue1_resource1",
     FamilyConfig(collect=1, intercept=1, pursue=1, resource=1)),
    ("delayed3_intercept1_pursue1_resource2",
     FamilyConfig(delayed=3, intercept=1, pursue=1, resource=2)),
    ("delayed3_intercept2_pursue1_resource1",
     FamilyConfig(delayed=3, intercept=2, pursue=1, resource=1)),
    ("intercept2", FamilyConfig(intercept=2)),
    ("avoid1_collect1", FamilyConfig(avoid=1, collect=1)),
    ("avoid2_collect1_delayed5",
     FamilyConfig(avoid=2, collect=1, delayed=5)),
]

# ---- library phase: pooled typed-head rows from the simple worlds --
lib_X, lib_y, lib_S, lib_ret = [], [], [], []
ev_psi, ev_y = [], []
for name, config in TRAIN_WORLDS:
    bank = build_bank(config, args.seed * 31)
    rollout = collect_experience(config, args.seed * 53)
    X, y, S, ret = build_head_data(*rollout, soft=True, bank=bank)
    lib_X.append(X); lib_y.append(y)
    lib_S.append(S); lib_ret.append(ret)
    raw, psis, actions, rewards, alive = rollout
    for t in range(len(actions)):
        ok = alive[t]
        if not bool(ok.any()):
            continue
        died = ok & ~alive[t + 1]
        parent = clamp_state(raw[t])
        child = clamp_state(raw[t + 1])
        for g_ix, (_pl, base) in enumerate(TRACKED):
            d_now = ((child[:, 0] - child[:, base]).abs()
                     + (child[:, 1] - child[:, base + 1]).abs())
            hit_prev = ((child[:, 0] == parent[:, base])
                        & (child[:, 1] == parent[:, base + 1]))
            contact = ok & ((d_now <= 1) | hit_prev)
            d_par = ((parent[:, 0] - parent[:, base]).abs()
                     + (parent[:, 1] - parent[:, base + 1]).abs())
            contact = torch.where(died, ok & (d_par <= 2), contact)
            if bool(contact.any()):
                ev_psi.append(psis[t][contact, g_ix])
                ev_y.append(rewards[t][contact])
    print(f"  [train] {name:<24} rows {X.shape[0]}", flush=True)
lib_X = torch.cat(lib_X); lib_y = torch.cat(lib_y)
lib_S = torch.cat(lib_S); lib_ret = torch.cat(lib_ret)
w_tr, w_tv = fit_typed(lib_X, lib_y, lib_S, lib_ret)
w_cl = ridge(torch.cat(ev_psi), torch.cat(ev_y), args.ridge_lam)
print(f"class events {sum(x.shape[0] for x in ev_psi)}", flush=True)

LIBRARY = [
    ("approach1", (((0, 1), (2, 3), 1),)),
    ("approach2", (((0, 1), (4, 5), 1),)),
    ("avoid2", (((0, 1), (4, 5), -1),)),
    ("comp", (((0, 1), (2, 3), 1), ((0, 1), (4, 5), -1))),
]


def goal_cost(state, reference, goal):
    total = None
    for (a0, a1), (b0, b1), sign in goal:
        reach = ((state[:, a0] - reference[:, b0]).abs()
                 + (state[:, a1] - reference[:, b1]).abs()).float()
        total = (sign * reach if total is None
                 else total + sign * reach)
    return total


def play_goal(config, bank, goal, seed):
    episodes, steps = args.episodes, args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    tracker = Tracker(episodes)
    total = torch.zeros(episodes)
    for _ in range(steps):
        code = tracker.encode(frame_cells(v.observation()))
        reference = clamp_state(code)
        best, action = None, torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            program = bank.get(act)
            state = (reference if program is None
                     else plant_executor(program, reference))
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


def play_hybrid(config, bank, goal, lam, seed, head="pooled",
                w_class=None):
    """argmin goal_cost - lam * typed reward, both on the same
    plant-rolled child; lam=None -> pure goal-grammar, goal=None ->
    pure transfer scoring (depth-1 typed)."""
    episodes, steps = args.episodes, args.steps
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    tracker = Tracker(episodes)
    total = torch.zeros(episodes)
    for _ in range(steps):
        code = tracker.encode(frame_cells(v.observation()))
        reference = clamp_state(code)
        psi_now = tracker.psi()
        best, action = None, torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            program = bank.get(act)
            child = (reference if program is None
                     else plant_executor(program, reference))
            score = torch.zeros(episodes)
            if goal is not None:
                score = score + goal_cost(child, reference, goal)
            if lam:
                if head == "class":
                    bonus = torch.zeros(episodes)
                    for g_ix, (_pl, base) in enumerate(TRACKED):
                        d = ((child[:, 0] - child[:, base]).abs()
                             + (child[:, 1]
                                - child[:, base + 1]).abs()).float()
                        k = ((2.0 - d) / 2.0).clamp(0.0, 1.0)
                        bonus = bonus + k * (psi_now[:, g_ix]
                                             @ w_class)
                    score = score - lam * bonus
                else:
                    x = event_features(reference, child,
                                       torch.full((episodes,), act,
                                                  dtype=torch.long),
                                       psi_now, soft=True)
                    score = score - lam * (x @ w_tr)
            if best is None:
                best = score.clone()
            else:
                take = score < best
                best = torch.where(take, score, best)
                action = torch.where(
                    take, torch.full((episodes,), act), action)
        total += v.step(action).reward
    return float(total.mean())


report["results"] = {}
for name, config in TARGET_WORLDS:
    row = {}
    row["random"] = play(config, "random", None, None, None,
                         args.seed * 977, 0)
    bank = build_bank(config, args.seed * 31)
    row["goal_grammar"] = max(
        play_goal(config, bank, goal, args.seed * 977)
        for _l, goal in LIBRARY)
    row["transfer_d4"] = play(config, "typed", bank, w_tr, w_tv,
                              args.seed * 977, 4, soft=True)
    # race hybrid candidates on the selection stream, both heads
    cands = [(goal, lam, h) for _l, goal in LIBRARY
             for lam in (0.5, 1.0, 2.0) for h in ("pooled", "class")]
    cands += [(None, 1.0, "pooled"), (None, 1.0, "class")]
    cands += [(goal, None, "pooled") for _l, goal in LIBRARY]
    sel = [play_hybrid(config, bank, g, l, args.seed * 977 + 1,
                       head=h, w_class=w_cl) for g, l, h in cands]
    g_b, l_b, h_b = cands[max(range(len(cands)),
                              key=lambda i: sel[i])]
    row["hybrid_raced"] = play_hybrid(config, bank, g_b, l_b,
                                      args.seed * 977, head=h_b,
                                      w_class=w_cl)
    row["picked"] = f"{h_b} lam={l_b}"
    # best pure-class and pure-pooled at lam=1 for the P1 pairing
    row["class_h1"] = play_hybrid(config, bank, LIBRARY[3][1], 1.0,
                                  args.seed * 977, head="class",
                                  w_class=w_cl)
    row["pooled_h1"] = play_hybrid(config, bank, LIBRARY[3][1], 1.0,
                                   args.seed * 977, head="pooled")
    permc = torch.randperm(PSI_DIM,
                           generator=torch.Generator().manual_seed(
                               args.seed + 9090))
    row["shufclass_h1"] = play_hybrid(config, bank, LIBRARY[3][1],
                                      1.0, args.seed * 977,
                                      head="class",
                                      w_class=w_cl[permc])
    report["results"][name] = row
    print(f"  {name:<40} gg {row['goal_grammar']:+.3f}  tr "
          f"{row['transfer_d4']:+.3f}  hybrid "
          f"{row['hybrid_raced']:+.3f} ({row['picked']})", flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
