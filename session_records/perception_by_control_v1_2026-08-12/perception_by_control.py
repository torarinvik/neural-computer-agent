"""Select perception by CONTROL, not prediction.

F213's criterion (programmability x coverage) was defeated twice by its
own search, and with the loopholes closed it no longer prefers any
discovered encoder. F214 located the reason: an encoder can predict well
and plan badly -- the two qualities diverge. So the criterion should BE
the task: rank candidate encoders by the reward an agent earns when this
encoder's transitions are compiled into recipes and planned with.

Reward cannot be gamed the way the margin was. Both F213 breaks were the
criterion rewarding a property (predictability, then coverage) that a
degenerate encoder could maximise while discarding the world. The
environment's return is not a property of the encoder; the only way to
score it is to act well.

A second suspicion this probe settles: the F213 vocabulary CONTAINS the
best hand-written encoder. Peak row/col of each channel is `absolute` in
vocabulary form, and `absolute` had the best held-out margin of every
encoder tested (0.4423). Greedy-by-margin never assembled it -- it spent
its first two picks on channel 0 and then bought leftovers. If control
ranks `vocab_peaks` at the top while the margin criterion did not, the
failure of F213/F214 was the CRITERION, not the vocabulary.

Design, with the selection-integrity lesson from F210 applied first
rather than retrofitted:

  * candidates are ranked by mean return on the TRAIN worlds only,
    with each world's goal chosen from its own admissible slot pairs
    (F214's mechanism, nothing new);
  * the winner -- chosen before any held-out world is touched -- is
    evaluated on the HELD-OUT worlds with the full budget and the
    frozen neural plant;
  * references (hand-written, absolute, noise) run in both stages but
    never influence the choice;
  * selection uses ground-truth program execution because F204 measured
    the plant exact; the final evaluation uses the plant anyway so the
    claim stays architecture-honest.

Registered predictions:
  1. `vocab_peaks` wins the control ranking among vocabulary encoders.
  2. The control ranking puts `random_linear` last among informative
     encoders (it must -- its recipes predict nothing).
  3. The winner's held-out planning return is within noise of the
     hand-written encoder's, closing the gap F214 measured at -0.3008.
  4. The margin criterion's own pick (`vocab_greedy_margin`) ranks
     BELOW `vocab_peaks` under control -- the divergence, measured on
     the selection side rather than the evaluation side.
"""

from __future__ import annotations

import argparse
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
parser.add_argument("--search-episodes", type=int, default=16)
parser.add_argument("--search-steps", type=int, default=8)
parser.add_argument("--train-to", type=int, default=8)
parser.add_argument("--held-from", type=int, default=8)
parser.add_argument("--held-to", type=int, default=16)
parser.add_argument("--random-candidates", type=int, default=8)
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


# ------------------------------------------------ vocabulary encoders
def _reductions(plane):
    mask = plane > 0
    any_set = mask.any(dim=(1, 2))
    flat = plane.reshape(plane.shape[0], -1)
    top = flat.argmax(dim=1)
    rows_any = mask.any(dim=2)
    cols_any = mask.any(dim=1)
    idx_r = torch.arange(HEIGHT).view(1, -1)
    idx_c = torch.arange(WIDTH).view(1, -1)
    first_r = torch.where(rows_any, idx_r,
                          torch.full_like(idx_r, HEIGHT)).min(dim=1).values
    last_r = torch.where(rows_any, idx_r,
                         torch.full_like(idx_r, -1)).max(dim=1).values
    first_c = torch.where(cols_any, idx_c,
                          torch.full_like(idx_c, WIDTH)).min(dim=1).values
    last_c = torch.where(cols_any, idx_c,
                         torch.full_like(idx_c, -1)).max(dim=1).values
    weight = mask.float()
    total = weight.sum(dim=(1, 2)).clamp(min=1.0)
    cen_r = (weight * ROWS_IX).sum(dim=(1, 2)) / total
    cen_c = (weight * COLS_IX).sum(dim=(1, 2)) / total

    def clamp(t):
        return t.long().clamp(0, VALUES - 1)
    return {
        "peak_row": clamp(top // WIDTH), "peak_col": clamp(top % WIDTH),
        "centre_row": clamp(cen_r.round()), "centre_col": clamp(cen_c.round()),
        "first_row": clamp(torch.where(any_set, first_r,
                                       torch.zeros_like(first_r))),
        "first_col": clamp(torch.where(any_set, first_c,
                                       torch.zeros_like(first_c))),
        "last_row": clamp(torch.where(any_set, last_r,
                                      torch.zeros_like(last_r))),
        "last_col": clamp(torch.where(any_set, last_c,
                                      torch.zeros_like(last_c))),
        "count": clamp(mask.sum(dim=(1, 2))),
        "extent": clamp((last_r - first_r).clamp(min=0)),
    }


REDUCTIONS = ["peak_row", "peak_col", "centre_row", "centre_col",
              "first_row", "first_col", "last_row", "last_col",
              "count", "extent"]
VOCABULARY = [(c, r) for c in range(PLANES) for r in REDUCTIONS]


def encode_with(features):
    def encoder(screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        cache = {c: _reductions(frames[:, c])
                 for c in {f[0] for f in features}}
        out = torch.zeros((frames.shape[0], SLOTS), dtype=torch.long)
        for s, (c, r) in enumerate(features[:SLOTS]):
            out[:, s] = cache[c][r]
        return out
    return encoder


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


_PROJ = torch.randn(PLANES * HEIGHT * WIDTH, SLOTS,
                    generator=torch.Generator().manual_seed(11))


def enc_random_linear(screen):
    flat = screen.view(-1, PLANES, HEIGHT, WIDTH).float().reshape(
        screen.shape[0], -1)
    z = flat @ _PROJ
    z = z - z.min(dim=0, keepdim=True).values
    span = z.max(dim=0, keepdim=True).values.clamp(min=1e-6)
    return (z / span * (VALUES - 1)).round().long().clamp(0, VALUES - 1)


def build_candidates():
    rng = torch.Generator().manual_seed(args.seed + 271)
    cands = {
        # the vocabulary form of `absolute`: peak of each channel
        "vocab_peaks": [(0, "peak_row"), (0, "peak_col"), (1, "peak_row"),
                        (1, "peak_col"), (2, "peak_row"), (2, "peak_col")],
        # what greedy-by-margin actually picked in F213 (deduplicated run)
        "vocab_greedy_margin": [(0, "peak_row"), (0, "peak_col"),
                                (2, "peak_col"), (1, "last_row"),
                                (1, "peak_row"), (2, "last_col")],
        "vocab_centres": [(c, r) for c in range(3)
                          for r in ("centre_row", "centre_col")],
        "vocab_firsts": [(c, r) for c in range(3)
                         for r in ("first_row", "first_col")],
        "vocab_lasts": [(c, r) for c in range(3)
                        for r in ("last_row", "last_col")],
        "vocab_counts": [(c, r) for c in range(3)
                         for r in ("count", "extent")],
    }
    for k in range(args.random_candidates):
        order = torch.randperm(len(VOCABULARY), generator=rng)[:SLOTS]
        cands[f"vocab_random{k}"] = [VOCABULARY[int(i)] for i in order]
    return cands


# ------------------------------------------------------------ planning
def moving_slots(code):
    return {s for s in range(SLOTS) if len(set(code[:, s].tolist())) > 1}


def build_bank(config, seed, encoder, executor):
    bank = {}
    for act in range(4):
        v = FamilyVerifier(config, batch_size=args.observations,
                           seed=seed + act)
        v.reset(seed=seed + act)
        before = encoder(v.observation())
        v.step(torch.full((args.observations,), act, dtype=torch.long))
        after = encoder(v.observation())
        keep = (before < VALUES).all(dim=1) & (after < VALUES).all(dim=1)
        if int(keep.sum()) < 8:
            continue
        bank[act] = per_slot_search(before[keep][:args.examples],
                                    after[keep][:args.examples])
    return bank


def goal_cost(state, reference, pa, pb):
    reach = ((state[:, pa[0]] - reference[:, pb[0]]).abs()
             + (state[:, pa[1]] - reference[:, pb[1]]).abs())
    return reach.float()


def play(config, bank, seed, encoder, pa, pb, executor,
         episodes, steps, mode="bank"):
    v = FamilyVerifier(config, batch_size=episodes, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(episodes)
    for _ in range(steps):
        if mode == "random":
            action = torch.randint(0, 4, (episodes,), generator=g)
        else:
            reference = encoder(v.observation())
            reference = torch.where(reference < VALUES, reference,
                                    torch.zeros_like(reference))
            best, action = None, torch.zeros(episodes, dtype=torch.long)
            for act in range(4):
                program = bank.get(act)
                state = reference if program is None else executor(
                    program, reference)
                cost = goal_cost(state, reference, pa, pb)
                if best is None:
                    best = cost.clone()
                else:
                    take = cost < best
                    best = torch.where(take, cost, best)
                    action = torch.where(
                        take, torch.full((episodes,), act), action)
        total += v.step(action).reward
    return float(total.mean())


def choose_goal(config, encoder, bank, executor):
    v = FamilyVerifier(config, batch_size=args.observations,
                       seed=args.seed * 31)
    v.reset(seed=args.seed * 31)
    usable = moving_slots(encoder(v.observation()))
    best, best_reward = None, -1e9
    for a0 in range(SLOTS):
        for a1 in range(a0 + 1, SLOTS):
            for b0 in range(SLOTS):
                for b1 in range(SLOTS):
                    if b0 == b1 or {a0, a1} & {b0, b1}:
                        continue
                    if not {a0, a1, b0, b1} <= usable:
                        continue
                    reward = play(config, bank, args.seed * 977 + 1,
                                  encoder, (a0, a1), (b0, b1), executor,
                                  args.search_episodes, args.search_steps)
                    if reward > best_reward:
                        best, best_reward = ((a0, a1), (b0, b1)), reward
    return best, best_reward


def control_score(encoder, worlds, executor):
    """Mean return over worlds, each with its own chosen goal. This is
    the selection criterion: nothing about it can be satisfied without
    acting well in the real environment."""
    rewards, goals = [], {}
    for config in worlds:
        bank = build_bank(config, args.seed * 31, encoder, executor)
        picked, reward = choose_goal(config, encoder, bank, executor)
        if picked is None:
            rewards.append(play(config, {}, args.seed * 977, encoder,
                                (0, 1), (2, 3), executor,
                                args.search_episodes, args.search_steps,
                                mode="random"))
            continue
        goals[config.name] = [list(picked[0]), list(picked[1])]
        rewards.append(reward)
    return sum(rewards) / len(rewards), goals


def truth_executor(program, state):
    """Executor signature (program, state), matching the plant."""
    return run_parallel(state, program)


# ------------------------------------------------------- the two stages
variants = family_variants()
train_worlds = variants[:args.train_to]
held_worlds = variants[args.held_from:args.held_to]

candidates = build_candidates()
references = {"handwritten": enc_handwritten,
              "random_linear": enc_random_linear}

report = {"seed": args.seed, "n_candidates": len(candidates)}
ranking = []
for name, features in candidates.items():
    score, goals = control_score(encode_with(features), train_worlds,
                                 truth_executor)
    ranking.append((round(score, 4), name))
    print(f"  {name:<22} control {score:+.4f}", flush=True)
for name, encoder in references.items():
    score, _ = control_score(encoder, train_worlds, truth_executor)
    print(f"  {name:<22} control {score:+.4f}  (reference, not selectable)",
          flush=True)
    report[f"reference_train_{name}"] = round(score, 4)
ranking.sort(reverse=True)
report["train_ranking"] = [[s, n] for s, n in ranking]
winner = ranking[0][1]
report["winner"] = {"name": winner,
                    "features": [list(f) for f in candidates[winner]]}
print(f"\n  WINNER on train control: {winner}", flush=True)

# ------------------------- held-out evaluation, with the frozen plant
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
report["interpreter_check"] = round(hits / rows, 4)


def plant_executor(program, state):
    with torch.no_grad():
        return interp(program, state).argmax(-1)


# EVERY candidate is evaluated held-out, not only the winner, so the
# claim can be about whether train-control RANK predicts held-out
# planning -- a correlation over the whole set is worth more than one
# argmax. And a world where an encoder admits no goal is scored at the
# RANDOM policy's return rather than skipped: skipping let an encoder
# look better by failing on the hard worlds, which is the same
# selection-on-outcome mistake as F210's missing test set, one level
# down.
final_arms = {name: encode_with(f) for name, f in candidates.items()}
final_arms["handwritten"] = enc_handwritten
final_arms["random_linear"] = enc_random_linear
results = {}
for name, encoder in final_arms.items():
    rows = {}
    fallbacks = 0
    for config in held_worlds:
        bank = build_bank(config, args.seed * 31, encoder, plant_executor)
        picked, _ = choose_goal(config, encoder, bank, truth_executor)
        random_return = play(config, {}, args.seed * 977, encoder,
                             (0, 1), (2, 3), plant_executor,
                             args.episodes, args.steps, mode="random")
        if picked is None:
            fallbacks += 1
            rows[config.name] = {"random": random_return,
                                 "bank": random_return, "fallback": True}
            continue
        pa, pb = picked
        rows[config.name] = {
            "random": random_return,
            "bank": play(config, bank, args.seed * 977, encoder, pa, pb,
                         plant_executor, args.episodes, args.steps)}
    results[name] = {
        "scored": len(rows), "no_goal_worlds": fallbacks,
        "per_game": rows,
        "mean_random": round(sum(v["random"] for v in rows.values())
                             / len(rows), 4),
        "mean_bank": round(sum(v["bank"] for v in rows.values())
                           / len(rows), 4)}
    print(f"  HELD-OUT {name:<22} n={len(rows)} fallback={fallbacks} "
          f"random {results[name]['mean_random']:+.4f}  bank "
          f"{results[name]['mean_bank']:+.4f}", flush=True)
report["held_out"] = results

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
