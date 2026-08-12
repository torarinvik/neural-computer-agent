"""The control loop: plan with the bank, act in the real game.

Everything the architecture needs has now been measured separately. One
frozen interpreter predicts both rule families and grid games (F192,
F194). Recipes are one or two instructions, found in about 23
candidates (F178), held outside the weights, and portable (F184). They
compound: an eight-step rollout holds 0.83 against a floor of 0.24
(F200). Retention is exactly zero forgetting (F185).

None of that ACTS. This closes the loop: roll the recipes forward over
candidate action sequences, score the predicted outcome, take the first
action of the best sequence, and repeat in the real game.

Three arms, and the two controls are what make the middle one readable:

  random    the floor. Not "chance" in the abstract — the actual score
            of acting without a model in this game.
  BANK      plans using recipes searched against the frozen interpreter.
  oracle    plans using the REAL environment as its model, by copying
            the verifier and stepping it. This is the ceiling the
            planner could reach with a perfect model, and it separates
            "the model is bad" from "the planner or the goal is bad".
            Without it, a mediocre bank score is uninterpretable.

**The goal is SUPPLIED, not learned, and that is a real limitation.**
The agent is told to minimise the slot-distance between the avatar
pair and the object pair. That is stated in the amodal interface rather
than in grid terms — "make slot pair A match slot pair B" — so it adds
no domain knowledge the interface does not already carry, but it is
still a goal I wrote rather than one the system inferred. Learning the
goal is a separate problem and calling this a complete agent would be
an overclaim.
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
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--program-len", type=int, default=6)
parser.add_argument("--observations", type=int, default=256)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--depth", type=int, default=2,
                    help="lookahead depth; the planner scores every "
                         "action sequence of this length")
parser.add_argument("--games", type=int, default=4)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SWAP",
       "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))


def run_instruction(state, op, i, j, m):
    name, mod = OPS[op], MODULI[m]
    out = state.clone()
    if name == "INC":
        out[:, i] = (state[:, i] + 1) % mod
    elif name == "DEC":
        out[:, i] = (state[:, i] - 1) % mod
    elif name == "SINC":
        out[:, i] = torch.clamp(state[:, i] + 1, max=mod - 1)
    elif name == "SDEC":
        out[:, i] = torch.clamp(state[:, i] - 1, min=0)
    elif name == "CINC":
        g = state[:, j] != 0
        out[:, i] = torch.where(g, (state[:, i] + 1) % mod, state[:, i])
    elif name == "CDEC":
        g = state[:, j] != 0
        out[:, i] = torch.where(g, (state[:, i] - 1) % mod, state[:, i])
    elif name == "COPY":
        out[:, i] = state[:, j]
    elif name == "SWAP":
        out[:, i], out[:, j] = state[:, j], state[:, i]
    return out


class Interpreter(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.load = torch.nn.Linear(SLOTS * VALUES, dim)
        self.op = torch.nn.Embedding(len(OPS), dim)
        self.arg_i = torch.nn.Embedding(SLOTS, dim)
        self.arg_j = torch.nn.Embedding(SLOTS, dim)
        self.arg_m = torch.nn.Embedding(len(MODULI), dim)
        self.step = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, SLOTS * VALUES)

    def forward(self, program, state):
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        latent = self.load(hot)
        for op, i, j, m in program:
            code = (self.op(torch.tensor(op)) + self.arg_i(torch.tensor(i))
                    + self.arg_j(torch.tensor(j))
                    + self.arg_m(torch.tensor(m)))
            latent = self.norm(latent + self.step(torch.cat(
                [latent, code.unsqueeze(0).expand(latent.shape[0], -1)],
                dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_program(generator, length):
    out = []
    for _ in range(length):
        op = int(torch.randint(0, len(OPS), (1,), generator=generator))
        i = int(torch.randint(0, SLOTS, (1,), generator=generator))
        j = int(torch.randint(0, SLOTS, (1,), generator=generator))
        if i == j:
            j = (j + 1) % SLOTS
        m = int(torch.randint(0, len(MODULI), (1,), generator=generator))
        out.append((op, i, j, m))
    return out


interp = Interpreter(args.dim)
opt = torch.optim.AdamW(interp.parameters(), lr=args.lr,
                        weight_decay=0.01)
train_gen = torch.Generator().manual_seed(args.seed * 104729)
for _ in range(args.interpreter_updates):
    prog = random_program(train_gen, args.program_len)
    st = torch.randint(0, VALUES, (args.batch_size, SLOTS),
                       generator=train_gen)
    tgt = st.clone()
    for op, i, j, m in prog:
        tgt = run_instruction(tgt, op, i, j, m)
    loss = torch.nn.functional.cross_entropy(
        interp(prog, st).reshape(-1, VALUES), tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
for parameter in interp.parameters():
    parameter.requires_grad_(False)

check_gen = torch.Generator().manual_seed(args.seed + 5551)
hits = rows = 0
for _ in range(32):
    prog = random_program(check_gen, args.program_len)
    st = torch.randint(0, VALUES, (128, SLOTS), generator=check_gen)
    tgt = st.clone()
    for op, i, j, m in prog:
        tgt = run_instruction(tgt, op, i, j, m)
    with torch.no_grad():
        hits += int((interp(prog, st).argmax(-1) == tgt).sum())
    rows += tgt.numel()
report = {"seed": args.seed,
          "interpreter_check": round(hits / rows, 4),
          "depth": args.depth, "steps": args.steps}


def slot_state(screen):
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    batch = frames.shape[0]
    out = torch.full((batch, SLOTS), ABSENT, dtype=torch.long)
    ri = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
    ci = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)
    for row in range(batch):
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
            d = (ri - ar).abs() + (ci - ac).abs()
            d = torch.where(mask, d, torch.full_like(d, 999))
            pick = int(d.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def goal_cost(state, reference=None):
    """Distance from the avatar to the object, where the OBJECT comes
    from `reference` if given.

    Evaluating both against the post-action state is wrong and cost two
    rounds to see. Reaching the object CONSUMES it, so the post-action
    state either shows a newly respawned object far away or none at
    all — and with absent priced high (which the previous fix
    introduced, to stop the planner preferring futures where the object
    vanished) the planner learned to avoid collecting entirely. Success
    looked exactly like failure.

    Scoring the new avatar position against the OLD object position
    removes both failure modes: the target does not move while the plan
    is being evaluated, so approaching it always scores better and
    reaching it scores best. Still stated in slots — "move slot pair A
    toward where slot pair B was" — and still carrying no grid
    knowledge."""
    """Slot-distance between the avatar pair and the positive-object
    pair. Stated in the interface, not in grid terms."""
    target = state if reference is None else reference
    reach = (state[:, 0] - target[:, 2]).abs() \
        + (state[:, 1] - target[:, 3]).abs()
    absent = (target[:, 2] >= VALUES) | (target[:, 3] >= VALUES)
    # An ABSENT object must cost the MOST, not the least. The first
    # version returned zero for absent, which is minimal cost, so the
    # planner preferred futures where the object vanished from the
    # slots — the exact opposite of collecting it. The ORACLE caught
    # it: planning with the true environment scored 0.000 against
    # random's 0.042, and a ceiling that loses to the floor can only
    # mean the objective is wrong rather than the model.
    return torch.where(absent, torch.full_like(reach, 99), reach).float()


def learn_recipes(config, seed):
    """Search one recipe per action against the FROZEN interpreter."""
    verifier = FamilyVerifier(config, batch_size=args.observations,
                              seed=seed)
    verifier.reset(seed=seed)
    before = slot_state(verifier.observation())
    generator = torch.Generator().manual_seed(seed + 13)
    action = torch.randint(0, verifier.action_count,
                           (args.observations,), generator=generator)
    verifier.step(action)
    after = slot_state(verifier.observation())
    alive = (before[:, 0] < VALUES) & (after[:, 0] < VALUES)
    before, action, after = before[alive], action[alive], after[alive]
    used = (before < VALUES).all(dim=0) & (after < VALUES).all(dim=0)
    src = torch.where(before < VALUES, before, torch.zeros_like(before))
    dst = torch.where(after < VALUES, after, torch.zeros_like(after))
    moduli = []
    both = torch.cat([src, dst], dim=0)
    for slot in range(SLOTS):
        col = both[:, slot]
        col = col[col < VALUES]
        want = (int(col.max()) + 1) if col.numel() else VALUES
        moduli.append(min(m for m in MODULI if m >= max(want, MODULI[0])))
    recipe = {}
    for act in range(verifier.action_count):
        keep = action == act
        if int(keep.sum()) < 4:
            continue
        a, b = src[keep], dst[keep]
        writes = set((a != b).any(dim=0).nonzero().flatten().tolist())
        best, best_score = None, -1.0
        pool = [(op, i, j, MODULI.index(moduli[i]))
                for op in range(len(OPS)) if OPS[op] != "NOOP"
                for i in sorted(writes or {0})
                for j in range(SLOTS) if j != i]
        pad = (0, 0, 1, 0)
        for one in pool:
            cand = [one] + [pad] * (args.program_len - 1)
            with torch.no_grad():
                got = interp(cand, a).argmax(-1)
            sc = float((got[:, used] == b[:, used]).float().mean())
            if sc > best_score:
                best, best_score = cand, sc
        for first in pool:
            if best_score >= 0.99:
                break
            for second in pool:
                cand = [first, second] + [pad] * (args.program_len - 2)
                with torch.no_grad():
                    got = interp(cand, a).argmax(-1)
                sc = float((got[:, used] == b[:, used]).float().mean())
                if sc > best_score:
                    best, best_score = cand, sc
                if best_score >= 0.99:
                    break
        recipe[act] = best
    return recipe


def sequences(count: int, depth: int):
    out = [[]]
    for _ in range(depth):
        out = [s + [a] for s in out for a in range(count)]
    return out


def play(config, mode: str, recipe, seed: int):
    """One episode batch. `mode` picks the model the planner rolls
    forward: the bank's recipes, the real environment, or nothing.

    GREEDY, one step. Deeper lookahead was tried and is worse with this
    objective, for a reason worth keeping: reaching the food REMOVES it
    and respawns it elsewhere, so a two-step plan that collects ends
    farther from an object than one that loiters beside it. A proxy
    objective that is monotone in the right direction over ONE step
    stops being monotone over two. Scoring actual reward would fix it
    and needs a reward model, which this architecture does not have —
    F192 to F200 built a TRANSITION model."""
    verifier = FamilyVerifier(config, batch_size=args.episodes, seed=seed)
    verifier.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed + 4242)
    total = torch.zeros(args.episodes)
    for _ in range(args.steps):
        if mode == "random":
            action = torch.randint(0, verifier.action_count,
                                   (args.episodes,), generator=generator)
        else:
            best_cost = None
            reference = slot_state(verifier.observation())
            action = torch.zeros(args.episodes, dtype=torch.long)
            for act in range(verifier.action_count):
                if mode == "bank":
                    state = slot_state(verifier.observation())
                    state = torch.where(state < VALUES, state,
                                        torch.zeros_like(state))
                    cand = recipe.get(act)
                    if cand is not None:
                        with torch.no_grad():
                            state = interp(cand, state).argmax(-1)
                    cost = goal_cost(state, reference)
                else:
                    shadow = copy.deepcopy(verifier)
                    shadow.step(torch.full((args.episodes,), act))
                    cost = goal_cost(slot_state(shadow.observation()),
                                     reference)
                if best_cost is None:
                    best_cost = cost.clone()
                else:
                    take = cost < best_cost
                    best_cost = torch.where(take, cost, best_cost)
                    action = torch.where(
                        take, torch.full((args.episodes,), act), action)
        step = verifier.step(action)
        total += step.reward
    return round(float(total.mean()), 4)


results = {}
for config in family_variants()[:args.games]:
    recipe = learn_recipes(config, args.seed * 31)
    results[config.name] = {
        "random": play(config, "random", None, args.seed * 977),
        "bank": play(config, "bank", recipe, args.seed * 977),
        "oracle": play(config, "oracle", None, args.seed * 977)}
report["results"] = results
for arm in ("random", "bank", "oracle"):
    report[f"mean_{arm}"] = round(
        sum(v[arm] for v in results.values()) / max(len(results), 1), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
