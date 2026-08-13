"""REWARD-DERIVED GOALS: stop searching proxies, fit the reward, read the
goal off its weights.

The deepest recurring defect in the record is the model-proxy gap: the
planner scores hand-shaped distance proxies, so the bank can "beat" its
own oracle (F203, F216, F221, F222) and choice never works. Every goal
SEARCH so far -- single, signed, composite -- picks among proxies by
rollout reward. This probe removes the search entirely:

  1. Fit a sparse linear reward model on the world's own random-policy
     transitions. Features are domain-general contact indicators: for
     every admissible ordered slot-pair pair p, 1{dist_p == 0} after
     the step. Closed-form ridge; no rollouts.
  2. Keep the significant weights. Each weight IS a goal term: sign
     gives approach/avoid, magnitude gives priority. The derived cost
     is sum_p w_p * dist_p -- dense shaping derived from sparse reward.
  3. Plan greedily on the derived cost, exactly as before.

Registered predictions:
  1. Derived goals restore bank <= oracle on choice AND raise it:
     approach-food-avoid-poison is forced by the fitted weights.
  2. On the twins the fitted weights themselves flip sign between
     byte-identical worlds -- checkable directly, before any rollout.
  3. Intercept stays at single-term level: lead pursuit is not
     expressible as contact weights, so F222's searched composite
     should stay better there. Registered so a surprise counts.
  4. Goal acquisition cost falls from ~115,000 interaction steps
     (360 rollouts x 320 steps) to the ~2,000 evidence steps already
     collected for the bank.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyConfig, FamilyVerifier, family_variants)

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
parser.add_argument("--search-episodes", type=int, default=32)
parser.add_argument("--search-steps", type=int, default=10)
parser.add_argument("--worlds", type=int, default=16)
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


def plant_executor(program, state):
    with torch.no_grad():
        return interp(program, state).argmax(-1)


def truth_executor(program, state):
    return run_parallel(state, program)


def enc(screen):
    """The hand-written perception. Perception is orthogonal to the goal
    language and F215 measured its replacement separately; holding it
    fixed isolates this probe's one variable."""
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
    """goal = tuple of 1 or 2 terms, each (pair_a, pair_b, sign)."""
    if goal and isinstance(goal[0][0], int):
        goal = (goal,)                      # single-term compatibility
    total = None
    for (a0, a1), (b0, b1), sign in goal:
        reach = ((state[:, a0] - reference[:, b0]).abs()
                 + (state[:, a1] - reference[:, b1]).abs()).float()
        term = sign * reach
        total = term if total is None else total + term
    return total


def build_bank(config, seed, executor):
    """F155/F192's row-versus-slot lesson, applied a third time.

    Requiring ALL slots present dropped EVERY row on the pure avoid
    worlds, where slots 2,3 are absent by construction -- the bank came
    back empty, the planner tied on every action, and both goal signs
    played constant action 0 with byte-identical returns. The sentinel
    for an unused SLOT masks the slot; only a missing avatar drops the
    row."""
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
        bank[act] = per_slot_search(b[:args.examples], a[:args.examples])
    return bank


def play(config, mode, bank, seed, goal, executor, episodes, steps):
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
                if mode == "oracle":
                    shadow = copy.deepcopy(v)
                    shadow.step(torch.full((episodes,), act))
                    seen = enc(shadow.observation())
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
    return float(total.mean())


def usable_slots(config):
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    code = enc(v.observation())
    present = (code < VALUES).float().mean(dim=0)
    return {s for s in range(SLOTS) if float(present[s]) >= 0.9}


pairs = [q for q in itertools.permutations(range(SLOTS), 2)]


def choose_goal(config, bank, signs=(1, -1)):
    """F216's single-term search, then greedy composition: all pairs
    among the top four singles. Returns (goal_terms, singles_best)."""
    usable = usable_slots(config)
    scored = []
    for pa in pairs:
        for pb in pairs:
            if set(pa) & set(pb) or pa[0] > pa[1]:
                continue
            if not (set(pa) | set(pb)) <= usable:
                continue
            for sign in signs:
                term = (pa, pb, sign)
                reward = play(config, "bank", bank, args.seed * 977 + 1,
                              (term,), plant_executor,
                              args.search_episodes, args.search_steps)
                scored.append((reward, term))
    if not scored:
        return None, None
    scored.sort(key=lambda x: -x[0])
    best_single = (scored[0][1],)
    best, best_r = best_single, scored[0][0]
    top = [t for _, t in scored[:4]]
    for i in range(len(top)):
        for j in range(len(top)):
            if i == j:
                continue
            candidate = (top[i], top[j])
            reward = play(config, "bank", bank, args.seed * 977 + 1,
                          candidate, plant_executor,
                          args.search_episodes, args.search_steps)
            if reward > best_r:
                best, best_r = candidate, reward
    return best, best_single


def masked(x):
    return torch.where(x < VALUES, x, torch.zeros_like(x))


def fit_reward_goal(config, seed, bank, top_k=3, weight_floor=0.1):
    """Sparse linear reward model on contact features -> weighted goal.

    Evidence: random-policy single-step transitions with reward. A
    feature per admissible ordered pair-pair: did pair_a land exactly on
    where pair_b was. Ridge closed-form, then the significant weights
    ARE the goal: cost = sum w_p * dist_p (w>0 approaches, w<0 avoids).
    Returns (terms, weights, evidence_steps, n_events)."""
    # MULTI-STEP random rollouts, not single steps from reset. At reset
    # nothing is adjacent to the avatar, so single-step evidence carries
    # zero contact events on the sparse worlds (avoid came back with 0
    # of them) -- F217's lesson again: reward lives where the policy
    # goes, and a policy that only ever takes one step goes nowhere.
    batches = []
    for chunk in range(2):
        v = FamilyVerifier(config, batch_size=256, seed=seed + 100 + chunk)
        v.reset(seed=seed + 100 + chunk)
        g = torch.Generator().manual_seed(seed + 200 + chunk)
        for _ in range(12):
            before = enc(v.observation())
            action = torch.randint(0, 4, (256,), generator=g)
            step = v.step(action)
            # The TRUE after-frame hides the avatar on exactly the rows
            # that matter: hazard contact removes it, so conditioning on
            # "avatar present after" dropped all 55 contact events in a
            # 3,072-step check -- the row-filter lesson mutating a
            # fourth time. The fix uses only learned parts: the world's
            # own BANK predicts where the avatar was going, and that
            # prediction never disappears. Features are computed on the
            # bank-predicted after-state.
            after = enc(v.observation())
            keep = before[:, 0] < VALUES
            b = masked(before[keep])
            act_kept = action[keep]
            predicted = b.clone()
            for a in range(4):
                rows = act_kept == a
                if not bool(rows.any()) or bank.get(a) is None:
                    continue
                with torch.no_grad():
                    predicted[rows] = interp(bank[a], b[rows]).argmax(-1)
            # true after-state wherever the avatar survived; the bank's
            # prediction only fills the rows the environment blanked
            survived = (after[keep][:, 0] < VALUES).unsqueeze(1)
            evidence_after = torch.where(survived, masked(after[keep]),
                                         predicted)
            batches.append((b, evidence_after, step.reward[keep].float()))
    before = torch.cat([b for b, _, _ in batches])
    after = torch.cat([a for _, a, _ in batches])
    reward = torch.cat([r for _, _, r in batches])
    usable = usable_slots(config)
    feats, index = [], []
    for pa in pairs:
        for pb in pairs:
            if set(pa) & set(pb) or pa[0] > pa[1]:
                continue
            if not (set(pa) | set(pb)) <= usable:
                continue
            dist = ((after[:, pa[0]] - before[:, pb[0]]).abs()
                    + (after[:, pa[1]] - before[:, pb[1]]).abs())
            feats.append((dist == 0).float())
            index.append((pa, pb))
    if not feats:
        return None, None, int(before.shape[0]), 0
    X = torch.stack(feats, dim=1)
    lam = 1.0
    XtX = X.T @ X + lam * torch.eye(X.shape[1])
    w = torch.linalg.solve(XtX, X.T @ reward)
    order = torch.argsort(w.abs(), descending=True)
    terms, weights = [], []
    for i in order[:top_k].tolist():
        if float(w[i].abs()) < weight_floor:
            break
        pa, pb = index[i]
        terms.append((pa, pb, float(w[i])))
        weights.append(round(float(w[i]), 4))
    if not terms:
        return None, None, int(before.shape[0]), int((reward != 0).sum())
    return terms, weights, int(before.shape[0]), int((reward != 0).sum())


def derived_cost(state, reference, terms):
    total = None
    for (a0, a1), (b0, b1), w in terms:
        reach = ((state[:, a0] - reference[:, b0]).abs()
                 + (state[:, a1] - reference[:, b1]).abs()).float()
        # minimising sum w*dist approaches positive-weight pairs and
        # retreats from negative-weight ones, with priority = |w|
        term = w * reach
        total = term if total is None else total + term
    return total


def play_derived(config, mode, bank, seed, terms, episodes, steps):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        reference = masked(enc(v.observation()))
        best, action = None, torch.zeros(episodes, dtype=torch.long)
        for act in range(4):
            if mode == "oracle":
                shadow = copy.deepcopy(v)
                shadow.step(torch.full((episodes,), act))
                state = masked(enc(shadow.observation()))
            else:
                program = bank.get(act)
                if program is None:
                    state = reference
                else:
                    with torch.no_grad():
                        state = interp(program, reference).argmax(-1)
            cost = derived_cost(state, reference, terms)
            if best is None:
                best = cost.clone()
            else:
                take = cost < best
                best = torch.where(take, cost, best)
                action = torch.where(take,
                                     torch.full((episodes,), act), action)
        total += v.step(action).reward
    return float(total.mean())


WORLDS = [
    ("collect1", FamilyConfig(collect=1)),
    ("collect2", FamilyConfig(collect=2)),
    ("intercept1", FamilyConfig(intercept=1)),
    ("intercept2", FamilyConfig(intercept=2)),
    ("avoid1", FamilyConfig(avoid=1)),
    ("avoid2", FamilyConfig(avoid=2)),
    ("choice1", FamilyConfig(choice=1)),
    ("choice1_inv", FamilyConfig(choice=1, inverted=True)),
    ("forage2", FamilyConfig(forage=2)),
    ("forage2_inv", FamilyConfig(forage=2, inverted=True)),
]

results = {}
for name, config in WORLDS:
    bank = build_bank(config, args.seed * 31, plant_executor)
    composite, single = choose_goal(config, bank)
    terms, weights, steps_used, events = fit_reward_goal(
        config, args.seed * 31, bank)
    row = {"random": play(config, "random", None, args.seed * 977, None,
                          None, args.episodes, args.steps),
           "derived_terms": [[list(t[0]), list(t[1]), round(t[2], 3)]
                             for t in terms] if terms else None,
           "evidence_steps": steps_used, "reward_events": events}
    for label, goal in (("single", single), ("composite", composite)):
        if goal is None:
            row[label] = row[f"oracle_{label}"] = row["random"]
            continue
        row[label] = play(config, "bank", bank, args.seed * 977, goal,
                          plant_executor, args.episodes, args.steps)
        row[f"oracle_{label}"] = play(config, "oracle", None,
                                      args.seed * 977, goal,
                                      plant_executor, args.episodes,
                                      args.steps)
    if terms is None:
        row["derived"] = row["oracle_derived"] = row["random"]
    else:
        row["derived"] = play_derived(config, "bank", bank,
                                      args.seed * 977, terms,
                                      args.episodes, args.steps)
        row["oracle_derived"] = play_derived(config, "oracle", None,
                                             args.seed * 977, terms,
                                             args.episodes, args.steps)
    results[name] = row
    print(f"  {name:<14} rnd {row['random']:+.3f}  single "
          f"{row['single']:+.3f}  comp {row['composite']:+.3f}  DERIVED "
          f"{row['derived']:+.3f}/{row['oracle_derived']:+.3f}  "
          f"events {events}  w {row['derived_terms']}", flush=True)

report["results"] = results
for key in ("random", "single", "composite", "derived",
            "oracle_composite", "oracle_derived"):
    vals = [results[n][key] for n in results]
    report[f"mean_{key}"] = round(sum(vals) / len(vals), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
