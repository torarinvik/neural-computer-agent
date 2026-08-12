"""THE FOUNDING CURVE: does experience make the NEXT world cheaper?

The project's founding objective is a sentence: "produce a program such
that given task A, novel task B is faster to learn than chance or than
starting from scratch." Every component needed to measure that directly
now exists and has been separately verified: the program reader with
repair (F205-F208), the goal reader with beam verification (F218), the
signed goal language (F216), retention without forgetting (F185).

This probe puts one number on the sentence. A SEQUENCE of worlds is
fixed per seed. The system experiences the first k of them -- its
searches solve them, and the solutions become reader training labels
(self-labelling; no human labels anywhere). Then it meets EVAL worlds it
has never seen, and we meter exactly what acquiring each one costs:

    program side  candidates evaluated (reader proposes, repair
                  re-searches only failing slots; cold = full search)
    goal side     rollout evaluations (reader+beam = 4; cold = 360)

and whether the cheaply-acquired world is SOLVED AS WELL (planning
return, warm vs cold, paired).

Registered predictions:
  1. Warm program cost falls as k grows; warm competence stays at
     cold parity throughout.
  2. k=0 (synthetic-only readers) already sits below cold cost -- the
     floor is task-general skill, not memorised worlds.
  3. SHUFFLED-LABEL readers at k=max pay near-cold repair cost: the
     curve is driven by experience content, not by exposure or by the
     repair machinery itself.


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


def per_slot_search_metered(before, after, only=None):
    """The cold program search, returning (program, candidates tried)."""
    program, cost = [], 0
    for s in range(SLOTS):
        if only is not None and s not in only:
            program.append(None)
            continue
        want = after[:, s]
        best, best_score = NOOP, -1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j == s and PAR_OPS[op] in ("CINC", "CDEC", "COPY"):
                    continue
                for m in range(len(MODULI)):
                    cost += 1
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
    return program, cost


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


def world_transitions(config, seed, count):
    """One batch of random-policy experience: slots, action, reward."""
    v = FamilyVerifier(config, batch_size=count, seed=seed)
    v.reset(seed=seed)
    before = enc(v.observation())
    g = torch.Generator().manual_seed(seed + 99)
    action = torch.randint(0, 4, (count,), generator=g)
    step = v.step(action)
    after = enc(v.observation())
    return before, action, after, step.reward.float()


def masked(x):
    return torch.where(x < VALUES, x, torch.zeros_like(x))


def acquire_bank_cold(config, seed):
    """Full program search per action. Returns bank, candidate count, and
    the clean per-action transition sets (reused by warm arms so both see
    identical evidence)."""
    probe = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + 7)
    probe.reset(seed=seed + 7)
    used = (enc(probe.observation()) < VALUES).float().mean(dim=0) >= 0.9
    bank, cost, evidence = {}, 0, {}
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
        b, a = masked(before[keep])[:32], masked(after[keep])[:32]
        evidence[act] = (b, a)
        program, spent = per_slot_search_metered(b, a)
        bank[act] = program
        cost += spent
    return bank, cost, evidence


def acquire_bank_warm(program_reader, evidence):
    """Reader proposes, repair verifies: six executions per action, then
    re-search only the slots whose proposed instruction fails."""
    bank, cost = {}, 0
    for act, (b, a) in evidence.items():
        with torch.no_grad():
            po, pj, pm = program_reader(b.unsqueeze(0), a.unsqueeze(0))
        guess = [(int(po[0, s].argmax()), int(pj[0, s].argmax()),
                  int(pm[0, s].argmax())) for s in range(SLOTS)]
        bad = set()
        for s in range(SLOTS):
            cost += 1                       # one execution per slot check
            if float((slot_write(b, s, *guess[s]) == a[:, s])
                     .float().mean()) < 0.8:
                bad.add(s)
        fixed = list(guess)
        if bad:
            patch, spent = per_slot_search_metered(b, a, only=bad)
            cost += spent
            for s in bad:
                fixed[s] = patch[s]
        bank[act] = fixed
    return bank, cost


class ProgramReader(torch.nn.Module):
    """F205's reader over the parallel language (the F218-era plant)."""

    def __init__(self, dim: int):
        super().__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(2 * SLOTS * VALUES, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.pool = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.ReLU())
        self.op = torch.nn.Linear(dim, SLOTS * len(PAR_OPS))
        self.arg_j = torch.nn.Linear(dim, SLOTS * SLOTS)
        self.arg_m = torch.nn.Linear(dim, SLOTS * len(MODULI))

    def forward(self, before, after):
        b, e, _ = before.shape
        hot = torch.cat([
            torch.nn.functional.one_hot(before, VALUES).float().view(b, e, -1),
            torch.nn.functional.one_hot(after, VALUES).float().view(b, e, -1)],
            dim=-1)
        latent = self.pool(self.embed(hot).mean(dim=1))
        return (self.op(latent).view(b, SLOTS, len(PAR_OPS)),
                self.arg_j(latent).view(b, SLOTS, SLOTS),
                self.arg_m(latent).view(b, SLOTS, len(MODULI)))


class GoalReader(torch.nn.Module):
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
        self.pointers = torch.nn.Linear(dim, 4)
        self.sign = torch.nn.Linear(dim, 2)

    def forward(self, before, after, reward):
        b, e, _ = before.shape
        hot_b = torch.nn.functional.one_hot(before, VALUES).float()
        hot_a = torch.nn.functional.one_hot(after, VALUES).float()
        w = reward.view(b, e, 1, 1)
        feature = torch.cat([
            hot_b.mean(dim=1), hot_a.mean(dim=1),
            (hot_b * w).mean(dim=1), (hot_a * w).mean(dim=1),
            reward.mean(dim=1, keepdim=True).expand(b, SLOTS).unsqueeze(-1),
            reward.std(dim=1, keepdim=True).expand(b, SLOTS).unsqueeze(-1),
            (before != after).float().mean(dim=1).unsqueeze(-1)],
            dim=-1)
        latent = self.token(feature)
        attended, _ = self.attend(latent, latent, latent, need_weights=False)
        latent = self.norm1(latent + attended)
        latent = self.norm2(latent + self.mix(latent))
        return self.pointers(latent), self.sign(latent.mean(dim=1))


def read_pair(goal_reader, before, after, reward):
    with torch.no_grad():
        scores, _ = goal_reader(before.unsqueeze(0), after.unsqueeze(0),
                                reward.unsqueeze(0))
    s = scores[0]
    a0, a1 = sorted(torch.topk(s[:, 0] + s[:, 1], 2).indices.tolist())
    remaining = [x for x in range(SLOTS) if x not in (a0, a1)]
    rb = torch.tensor(remaining)
    b0 = int(rb[int(s[rb, 2].argmax())])
    rb2 = torch.tensor([x for x in remaining if x != b0])
    b1 = int(rb2[int(s[rb2, 3].argmax())])
    return (a0, a1), (b0, b1)


def play(config, mode, bank, seed, goal, episodes, steps):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = masked(enc(v.observation()))
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


def acquire_goal_cold(config, bank):
    """F216's search: every admissible goal, one rollout evaluation each."""
    usable = grid_usable(config)
    best, best_r, rollouts = None, -1e9, 0
    for goal in ADMISSIBLE:
        if not (set(goal[0]) | set(goal[1])) <= usable:
            continue
        rollouts += 1
        r = play(config, "bank", bank, args.seed * 977 + 1, goal,
                 args.search_episodes, args.search_steps)
        if r > best_r:
            best, best_r = goal, r
    return best, rollouts


def acquire_goal_warm(goal_reader, config, bank, mode_pair, seed):
    """F218's beam: reader pair + prior pair, both signs, four rollouts."""
    before, _, after, reward = world_transitions(config, seed, 512)
    keep = (before[:, 0] < VALUES) & (after[:, 0] < VALUES)
    pair = read_pair(goal_reader, masked(before[keep]), masked(after[keep]),
                     reward[keep])
    best, best_r, rollouts = None, -1e9, 0
    for candidate_pair in {pair, mode_pair}:
        for sign in (1, -1):
            goal = (candidate_pair[0], candidate_pair[1], sign)
            rollouts += 1
            r = play(config, "bank", bank, args.seed * 977 + 1, goal,
                     args.search_episodes, args.search_steps)
            if r > best_r:
                best, best_r = goal, r
    return best, rollouts


def synthetic_program_world(g):
    programs = [random_program(g) for _ in range(4)]
    act = int(torch.randint(0, 4, (1,), generator=g))
    before = torch.randint(0, VALUES, (32, SLOTS), generator=g)
    after = run_parallel(before, programs[act])
    return before, after


def synthetic_goal_world(g):
    goal = ADMISSIBLE[int(torch.randint(0, len(ADMISSIBLE), (1,),
                                        generator=g))]
    programs = [random_program(g) for _ in range(4)]
    before = torch.randint(0, VALUES, (512, SLOTS), generator=g)
    acts = torch.randint(0, 4, (512,), generator=g)
    after = before.clone()
    for a in range(4):
        rows = acts == a
        if bool(rows.any()):
            after[rows] = run_parallel(before[rows], programs[a])
    (a0, a1), (b0, b1), sign = goal
    dist = ((after[:, a0] - before[:, b0]).abs()
            + (after[:, a1] - before[:, b1]).abs())
    reward = torch.where(dist == 0, torch.full_like(dist, float(sign)),
                         torch.zeros_like(dist)).float()
    return before, after, reward, goal


def train_readers(experience, seed, shuffle=False):
    """Train both readers on synthetic worlds plus the k experienced
    worlds' SELF-LABELLED solutions (30% grid share, F206's optimum).
    `shuffle` permutes labels across worlds -- the exposure control."""
    g = torch.Generator().manual_seed(seed)
    # ------------------------------------------------ program reader
    pb, pa, plabel = [], [], []
    for index in range(1200):
        if experience and float(torch.rand(1, generator=g)) < 0.3:
            world = experience[int(torch.randint(0, len(experience), (1,),
                                                 generator=g))]
            act = int(torch.randint(0, 4, (1,), generator=g))
            if act not in world["evidence"]:
                continue
            b, a = world["evidence"][act]
            plabel.append([list(t) for t in world["bank"][act]])
            pb.append(b); pa.append(a)
        else:
            b, a = synthetic_program_world(g)
            plabel.append([list(t) for t in per_slot_search_metered(b, a)[0]])
            pb.append(b); pa.append(a)
    pb, pa = torch.stack(pb), torch.stack(pa)
    plabel = torch.tensor(plabel)
    if shuffle:
        plabel = plabel[torch.randperm(plabel.shape[0], generator=g)]
    program_reader = ProgramReader(args.dim)
    opt = torch.optim.AdamW(program_reader.parameters(), lr=args.lr,
                            weight_decay=0.01)
    for _ in range(args.reader_updates):
        pick = torch.randint(0, pb.shape[0], (args.batch_size,), generator=g)
        po, pj, pm = program_reader(pb[pick], pa[pick])
        lab = plabel[pick]
        loss = (torch.nn.functional.cross_entropy(
                    po.reshape(-1, len(PAR_OPS)), lab[:, :, 0].reshape(-1))
                + torch.nn.functional.cross_entropy(
                    pj.reshape(-1, SLOTS), lab[:, :, 1].reshape(-1))
                + torch.nn.functional.cross_entropy(
                    pm.reshape(-1, len(MODULI)), lab[:, :, 2].reshape(-1)))
        opt.zero_grad(); loss.backward(); opt.step()
    for p in program_reader.parameters():
        p.requires_grad_(False)
    # --------------------------------------------------- goal reader
    gb, ga, gr, glabel = [], [], [], []
    for index in range(600):
        if experience and float(torch.rand(1, generator=g)) < 0.3:
            world = experience[int(torch.randint(0, len(experience), (1,),
                                                 generator=g))]
            seed2 = int(torch.randint(0, 10 ** 6, (1,), generator=g))
            b, _, a, r = world_transitions(world["config"], seed2, 512)
            keep = (b[:, 0] < VALUES) & (a[:, 0] < VALUES)
            if int(keep.sum()) < 400:
                continue
            b, a, r = masked(b[keep])[:400], masked(a[keep])[:400], r[keep][:400]
            goal = world["goal"]
        else:
            b, a, r, goal = synthetic_goal_world(g)
            b, a, r = b[:400], a[:400], r[:400]
        gb.append(b); ga.append(a); gr.append(r); glabel.append(goal)
    gb, ga, gr = torch.stack(gb), torch.stack(ga), torch.stack(gr)
    if shuffle:
        order = torch.randperm(len(glabel), generator=g)
        glabel = [glabel[int(i)] for i in order]
    role = torch.zeros(len(glabel), SLOTS, 4)
    signt = torch.zeros(len(glabel), dtype=torch.long)
    for i, goal in enumerate(glabel):
        (a0, a1), (b0, b1), sign = goal
        role[i, a0, 0] = role[i, a1, 1] = role[i, b0, 2] = role[i, b1, 3] = 1
        signt[i] = 0 if sign == 1 else 1
    goal_reader = GoalReader(args.dim, args.heads)
    opt = torch.optim.AdamW(goal_reader.parameters(), lr=args.lr,
                            weight_decay=0.01)
    for _ in range(args.reader_updates):
        pick = torch.randint(0, gb.shape[0], (args.batch_size,), generator=g)
        scores, sign_logit = goal_reader(gb[pick], ga[pick], gr[pick])
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    scores, role[pick])
                + torch.nn.functional.cross_entropy(sign_logit, signt[pick]))
        opt.zero_grad(); loss.backward(); opt.step()
    for p in goal_reader.parameters():
        p.requires_grad_(False)
    # the accumulated prior pair for the beam
    from collections import Counter
    mode_pair = Counter((tuple(goal[0]), tuple(goal[1]))
                        for goal in glabel).most_common(1)[0][0]
    return program_reader, goal_reader, mode_pair


# ═══════════════════════════════════════════════════════ the curve
order_g = torch.Generator().manual_seed(args.seed * 7919)
sequence = [family_variants()[int(i)]
            for i in torch.randperm(len(family_variants()),
                                    generator=order_g)]
EVAL_WORLDS = sequence[-6:]
PREFIX = sequence[:-6]
K_VALUES = [0, 6, 12, 18]

# experience: the system SOLVES the first max(K) worlds cold, once, and
# keeps its own solutions as labels. This cost is the price of living;
# the curve measures the MARGINAL cost of the next world afterwards.
experience_all = []
for config in PREFIX[:max(K_VALUES)]:
    bank, _, evidence = acquire_bank_cold(config, args.seed * 31)
    goal, _ = acquire_goal_cold(config, bank)
    if goal is None:
        continue
    experience_all.append({"config": config, "bank": bank,
                           "evidence": evidence, "goal": goal})
    print(f"  experienced {config.name}", flush=True)

report["sequence"] = [c.name for c in sequence]
report["eval_worlds"] = [c.name for c in EVAL_WORLDS]
report["curve"] = {}

for k in K_VALUES:
    program_reader, goal_reader, mode_pair = train_readers(
        experience_all[:k], args.seed * 104729 + k)
    rows = {}
    for config in EVAL_WORLDS:
        bank_cold, prog_cost_cold, evidence = acquire_bank_cold(
            config, args.seed * 31)
        goal_cold, goal_cost_cold = acquire_goal_cold(config, bank_cold)
        bank_warm, prog_cost_warm = acquire_bank_warm(program_reader,
                                                      evidence)
        goal_warm, goal_cost_warm = acquire_goal_warm(
            goal_reader, config, bank_warm, mode_pair, args.seed + 555)
        if goal_cold is None or goal_warm is None:
            continue
        rows[config.name] = {
            "prog_candidates": {"cold": prog_cost_cold,
                                "warm": prog_cost_warm},
            "goal_rollouts": {"cold": goal_cost_cold,
                              "warm": goal_cost_warm},
            "return": {
                "random": play(config, "random", None, args.seed * 977,
                               None, args.episodes, args.steps),
                "cold": play(config, "bank", bank_cold, args.seed * 977,
                             goal_cold, args.episodes, args.steps),
                "warm": play(config, "bank", bank_warm, args.seed * 977,
                             goal_warm, args.episodes, args.steps)}}
    mean = lambda f: round(sum(f(r) for r in rows.values())
                           / max(len(rows), 1), 4)
    report["curve"][k] = {
        "per_world": rows,
        "prog_warm": mean(lambda r: r["prog_candidates"]["warm"]),
        "prog_cold": mean(lambda r: r["prog_candidates"]["cold"]),
        "goal_warm": mean(lambda r: r["goal_rollouts"]["warm"]),
        "goal_cold": mean(lambda r: r["goal_rollouts"]["cold"]),
        "return_warm": mean(lambda r: r["return"]["warm"]),
        "return_cold": mean(lambda r: r["return"]["cold"]),
        "return_random": mean(lambda r: r["return"]["random"])}
    c = report["curve"][k]
    print(f"  k={k:<3} prog {c['prog_warm']:.0f}/{c['prog_cold']:.0f}  "
          f"goal {c['goal_warm']:.0f}/{c['goal_cold']:.0f}  "
          f"return warm {c['return_warm']:+.3f} cold {c['return_cold']:+.3f} "
          f"random {c['return_random']:+.3f}", flush=True)

# the exposure control at k = max
program_reader, goal_reader, mode_pair = train_readers(
    experience_all, args.seed * 104729 + 999, shuffle=True)
shuf = {}
for config in EVAL_WORLDS:
    _, _, evidence = acquire_bank_cold(config, args.seed * 31)
    _, cost = acquire_bank_warm(program_reader, evidence)
    shuf[config.name] = cost
report["shuffled_label_prog_warm"] = round(
    sum(shuf.values()) / max(len(shuf), 1), 1)
print(f"  shuffled-label prog warm: {report['shuffled_label_prog_warm']}",
      flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
