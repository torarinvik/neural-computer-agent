"""PERCEPTION BY PLANNING: can the last hand-written component go?

`slot_state` -- avatar by argmax, then NEAREST object per plane -- is
the only remaining hand-written domain knowledge. F213 tried to replace
it with a programmability criterion and was defeated by its own
adversarial search, twice; F214 then showed the deeper problem:
encoders that PREDICT well can PLAN badly, so any prediction-side
criterion is aiming at the wrong target.

The fix nobody ran: choose the encoder the way goals are now chosen --
by planning reward. Each candidate encoder gets its own bank and its
own reward-chosen signed goal (F214 proved perception and goal are
coupled, so the goal must adapt per encoder), scored by cheap rollouts
on a selection stream; the winner is evaluated at full budget on fresh
episodes against the hand-written incumbent given identical treatment.

Candidates are built from F213's domain-general vocabulary (reductions
of a channel tensor: peak/centre/first/last row and column). The
incumbent's relational nearest-object trick is NOT expressible there --
that relational choice is exactly the hand-written part under test.

Registered predictions:
  1. Per-world planning selection finds a vocabulary encoder within
     noise of `slot_state` on the single-object worlds, where absolute
     and relational coincide.
  2. On multi-object worlds (collect2, forage2, avoid2) the incumbent
     keeps an edge -- the nearest-object relation matters exactly when
     there are several objects. Registered so that a win there is a
     genuine surprise.
  3. Different worlds select DIFFERENT vocabulary encoders (the point
     of per-world choice).
"""

from __future__ import annotations

import argparse
import copy
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
    (a0, a1), (b0, b1), sign = goal
    reach = ((state[:, a0] - reference[:, b0]).abs()
             + (state[:, a1] - reference[:, b1]).abs()).float()
    return sign * reach


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


def choose_goal(config, bank, signs):
    usable = usable_slots(config)
    best, best_reward = None, -1e9
    for a0 in range(SLOTS):
        for a1 in range(a0 + 1, SLOTS):
            for b0 in range(SLOTS):
                for b1 in range(SLOTS):
                    if b0 == b1 or {a0, a1} & {b0, b1}:
                        continue
                    if not {a0, a1, b0, b1} <= usable:
                        continue
                    for sign in signs:
                        reward = play(config, "bank", bank,
                                      args.seed * 977 + 1,
                                      ((a0, a1), (b0, b1), sign),
                                      truth_executor,
                                      args.search_episodes,
                                      args.search_steps)
                        if reward > best_reward:
                            best = ((a0, a1), (b0, b1), sign)
                            best_reward = reward
    return best


# ------------------------------------------------ candidate encoders
def _reduce(plane, which):
    mask = plane > 0
    flat = plane.reshape(plane.shape[0], -1)
    if which == "peak_row":
        return flat.argmax(dim=1) // WIDTH
    if which == "peak_col":
        return flat.argmax(dim=1) % WIDTH
    weight = mask.float()
    total = weight.sum(dim=(1, 2)).clamp(min=1.0)
    if which == "centre_row":
        return ((weight * ROWS_IX).sum(dim=(1, 2)) / total).round().long()
    if which == "centre_col":
        return ((weight * COLS_IX).sum(dim=(1, 2)) / total).round().long()
    idx_r = torch.arange(HEIGHT).view(1, -1)
    idx_c = torch.arange(WIDTH).view(1, -1)
    if which == "first_row":
        rows_any = mask.any(dim=2)
        return torch.where(rows_any, idx_r,
                           torch.full_like(idx_r, HEIGHT)).min(dim=1).values.clamp(max=VALUES-1)
    if which == "first_col":
        cols_any = mask.any(dim=1)
        return torch.where(cols_any, idx_c,
                           torch.full_like(idx_c, WIDTH)).min(dim=1).values.clamp(max=VALUES-1)
    if which == "last_row":
        rows_any = mask.any(dim=2)
        return torch.where(rows_any, idx_r,
                           torch.full_like(idx_r, -1)).max(dim=1).values.clamp(min=0)
    if which == "last_col":
        cols_any = mask.any(dim=1)
        return torch.where(cols_any, idx_c,
                           torch.full_like(idx_c, -1)).max(dim=1).values.clamp(min=0)
    raise AssertionError(which)


def vocab_encoder(features):
    def encoder(screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        out = torch.zeros((frames.shape[0], SLOTS), dtype=torch.long)
        for s, (c, r) in enumerate(features[:SLOTS]):
            out[:, s] = _reduce(frames[:, c], r).clamp(0, VALUES - 1)
        # a plane with nothing set carries no object: mark its slots
        # ABSENT so goal admissibility sees them, exactly as slot_state
        for s, (c, r) in enumerate(features[:SLOTS]):
            empty = ~(frames[:, c] > 0).any(dim=(1, 2))
            out[empty, s] = ABSENT
        return out
    return encoder


CANDIDATES = {
    "peaks": vocab_encoder([(0, "peak_row"), (0, "peak_col"),
                            (1, "peak_row"), (1, "peak_col"),
                            (2, "peak_row"), (2, "peak_col")]),
    "centres": vocab_encoder([(0, "peak_row"), (0, "peak_col"),
                              (1, "centre_row"), (1, "centre_col"),
                              (2, "centre_row"), (2, "centre_col")]),
    "firsts": vocab_encoder([(0, "peak_row"), (0, "peak_col"),
                             (1, "first_row"), (1, "first_col"),
                             (2, "first_row"), (2, "first_col")]),
    "lasts": vocab_encoder([(0, "peak_row"), (0, "peak_col"),
                            (1, "last_row"), (1, "last_col"),
                            (2, "last_row"), (2, "last_col")]),
    "f213": vocab_encoder([(0, "peak_row"), (0, "peak_col"),
                           (2, "peak_col"), (1, "last_row"),
                           (0, "centre_row"), (0, "centre_col")]),
}
INCUMBENT = enc      # the hand-written slot_state


def pipeline_score(config, encoder, episodes, steps):
    """Bank + signed single-term goal + planning return, all under the
    given encoder. Selection-stream seeds throughout."""
    global enc
    saved = enc
    enc = encoder
    try:
        bank = build_bank(config, args.seed * 31, plant_executor)
        goal = choose_goal(config, bank, (1, -1))
        if goal is None:
            return None, None, None
        reward = play(config, "bank", bank, args.seed * 977 + 1, goal,
                      plant_executor, episodes, steps)
        return reward, bank, goal
    finally:
        enc = saved


def pipeline_eval(config, encoder, bank, goal):
    global enc
    saved = enc
    enc = encoder
    try:
        return play(config, "bank", bank, args.seed * 977, goal,
                    plant_executor, args.episodes, args.steps)
    finally:
        enc = saved


WORLDS = [
    ("collect1", FamilyConfig(collect=1)),
    ("collect2", FamilyConfig(collect=2)),
    ("intercept1", FamilyConfig(intercept=1)),
    ("avoid2", FamilyConfig(avoid=2)),
    ("navigate1", FamilyConfig(navigate=True)),
    ("forage2", FamilyConfig(forage=2)),
]

results = {}
for name, config in WORLDS:
    row = {}
    picks = {}
    for label, encoder in CANDIDATES.items():
        score, bank, goal = pipeline_score(config, encoder,
                                           args.search_episodes,
                                           args.search_steps)
        if score is not None:
            picks[label] = (score, bank, goal, encoder)
    if not picks:
        continue
    best_label = max(picks, key=lambda k: picks[k][0])
    _, bank, goal, encoder = picks[best_label]
    row["chosen_encoder"] = best_label
    row["vocab_best"] = pipeline_eval(config, encoder, bank, goal)
    score_h, bank_h, goal_h = pipeline_score(config, INCUMBENT,
                                             args.search_episodes,
                                             args.search_steps)
    row["handwritten"] = (pipeline_eval(config, INCUMBENT, bank_h, goal_h)
                          if score_h is not None else None)
    saved = enc
    row["random"] = play(config, "random", None, args.seed * 977, None,
                         None, args.episodes, args.steps)
    results[name] = row
    print(f"  {name:<12} random {row['random']:+.3f}  handwritten "
          f"{row['handwritten']:+.3f}  VOCAB[{best_label}] "
          f"{row['vocab_best']:+.3f}", flush=True)

report["results"] = results
for key in ("random", "handwritten", "vocab_best"):
    vals = [results[n][key] for n in results if results[n].get(key) is not None]
    report[f"mean_{key}"] = round(sum(vals) / len(vals), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
