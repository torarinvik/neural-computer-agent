"""One frozen interpreter, two domains that share nothing but slots.

Everything measured in the recipe architecture so far — F155 through
F191 — lives in `schema_families`: procedurally generated rule families
over SLOTS x VALUES. The architecture's central claim is larger than
that. It says the plant is AMODAL: structure in frozen weights, content
in an external bank, and the same controller serving any domain an
encoder can map into the slot interface.

That claim has been argued from construction — "the instructions are
operations on abstract slots, so they must be domain-general" — and
never tested. An instruction set can be abstract in its definition and
still have been tuned, through a hundred measurements, to exactly the
distribution of rule families it was measured on. Nothing in F155-F191
would have detected that.

This is the test. The interpreter is trained ONLY on random programs
over random states, exactly as before, and never sees either domain.
Then recipes are searched for:

  RULE FAMILIES   the procedural families every prior finding used;
  GRID GAMES      real composigrid worlds from `game_family`, with an
                  avatar moving on an 8x8 board among positive and
                  negative objects, read into slots by the same shallow
                  perception `game_slots.py` uses.

These share the interface and nothing else. A grid game's dynamics are
"the avatar moves one cell and objects fall"; a rule family's are "some
slot increments conditional on another". If one frozen interpreter
supplies recipes for both, the amodal claim is demonstrated rather than
assumed. If grid games land at their identity floor, the instruction
set was fitted to rule families and the generality was an artefact of
never having looked.

The identity floor matters more here than anywhere. A grid game's state
barely changes per step — the avatar moves one cell of eight — so
copying the input forward scores very high, and any honest reading is
the margin above THAT, not above uniform.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)
from experiments.games_amodal.probes.schema_families import (
    Family, RandomFamily, random_family_spec)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--interpreter-updates", type=int, default=40000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--program-len", type=int, default=6)
parser.add_argument("--observations", type=int, default=256)
parser.add_argument("--eval-rows", type=int, default=512)
parser.add_argument("--enum-depth", type=int, default=2)
parser.add_argument("--fit-target", type=float, default=0.99)
parser.add_argument("--games", type=int, default=6)
parser.add_argument("--rule-families", type=int, default=6)
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


# ------------------------------------------------------- the interpreter
interp = Interpreter(args.dim)
opt = torch.optim.AdamW(interp.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
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
          "interpreter_updates": args.interpreter_updates,
          "interpreter_check": round(hits / rows, 4)}


# --------------------------------------------------------- perception
def slot_state(screen: torch.Tensor) -> torch.Tensor:
    """Screen -> slots. The same shallow perception `game_slots.py`
    uses: argmax for the avatar, then the nearest object by Manhattan
    distance in each object plane. Deliberately unchanged — a probe
    that improved perception to make the interpreter look good would
    be measuring the encoder, not the claim."""
    frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
    batch = frames.shape[0]
    out = torch.full((batch, SLOTS), ABSENT, dtype=torch.long)
    rows_ix = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
    cols_ix = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)
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
            distance = (rows_ix - ar).abs() + (cols_ix - ac).abs()
            distance = torch.where(mask, distance,
                                   torch.full_like(distance, 999))
            pick = int(distance.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def game_transitions(config, count: int, seed: int):
    """Real game steps, read into slots. Returns (state, action, next)."""
    verifier = FamilyVerifier(config, batch_size=count, seed=seed)
    verifier.reset(seed=seed)
    before = slot_state(verifier.observation())
    generator = torch.Generator().manual_seed(seed + 13)
    action = torch.randint(0, verifier.action_count, (count,),
                           generator=generator)
    verifier.step(action)
    after = slot_state(verifier.observation())
    return before, action, after, verifier.action_count


def rule_transitions(family, count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    size = len(family.states)
    idx = torch.randint(0, size, (count,), generator=generator)
    act = torch.randint(0, family.actions, (count,), generator=generator)
    nxt = torch.tensor([family.table[int(s)][int(a)]
                        for s, a in zip(idx, act)])
    return (family.slot_values(idx), act, family.slot_values(nxt),
            family.actions)


# ------------------------------------------------------------- search
def infer_moduli(*tensors):
    stacked = torch.cat(list(tensors), dim=0)
    out = []
    for slot in range(SLOTS):
        column = stacked[:, slot]
        column = column[column < VALUES]
        want = (int(column.max()) + 1) if column.numel() else VALUES
        out.append(min(m for m in MODULI if m >= max(want, MODULI[0])))
    return out


def enumerate_programs(writes, moduli, depth):
    pad = (0, 0, 1, 0)
    singles = [(op, i, j, MODULI.index(moduli[i]))
               for op in range(len(OPS)) if OPS[op] != "NOOP"
               for i in sorted(writes) for j in range(SLOTS) if j != i]
    if depth == 1:
        for a in singles:
            yield [a] + [pad] * (args.program_len - 1)
    else:
        for a in singles:
            for b in singles:
                yield [a, b] + [pad] * (args.program_len - 2)


def solve_and_score(before, action, after, n_actions, held):
    """Search a recipe per action, then score it on FRESH transitions."""
    used = (before < VALUES).all(dim=0) & (after < VALUES).all(dim=0)
    if int(used.sum()) == 0:
        return None
    src = torch.where(before < VALUES, before, torch.zeros_like(before))
    dst = torch.where(after < VALUES, after, torch.zeros_like(after))
    moduli = infer_moduli(src, dst)
    recipe, tried = {}, 0
    for act in range(n_actions):
        keep = action == act
        if int(keep.sum()) < 4:
            continue
        a, b = src[keep], dst[keep]
        writes = set((a != b).any(dim=0).nonzero().flatten().tolist())
        best, best_score = None, -1.0
        for depth in range(1, args.enum_depth + 1):
            if best_score >= args.fit_target:
                break
            for cand in enumerate_programs(writes or {0}, moduli, depth):
                tried += 1
                with torch.no_grad():
                    got = interp(cand, a).argmax(-1)
                sc = float((got[:, used] == b[:, used]).float().mean())
                if sc > best_score:
                    best, best_score = cand, sc
                if best_score >= args.fit_target:
                    break
        recipe[act] = best
    hb, ha, hn = held
    hsrc = torch.where(hb < VALUES, hb, torch.zeros_like(hb))
    hdst = torch.where(hn < VALUES, hn, torch.zeros_like(hn))
    hits = total = ident = 0
    for act, cand in recipe.items():
        keep = ha == act
        if not bool(keep.any()) or cand is None:
            continue
        with torch.no_grad():
            got = interp(cand, hsrc[keep]).argmax(-1)
        hits += int((got[:, used] == hdst[keep][:, used]).sum())
        ident += int((hsrc[keep][:, used] == hdst[keep][:, used]).sum())
        total += int(keep.sum()) * int(used.sum())
    return {"held_out": round(hits / max(total, 1), 4),
            "identity": round(ident / max(total, 1), 4),
            "candidates": tried, "actions": len(recipe)}


results = {"rule_families": {}, "grid_games": {}}

gen = torch.Generator().manual_seed(args.seed * 7919)
rules = [("line", Family("line")), ("dial", Family("dial")),
         ("toggle", Family("toggle")), ("perm", Family("perm"))]
while len(rules) < args.rule_families:
    rules.append((f"proc{len(rules)}",
                  RandomFamily(random_family_spec(gen))))
for name, family in rules[:args.rule_families]:
    obs = rule_transitions(family, args.observations, args.seed * 31)
    held = rule_transitions(family, args.eval_rows, args.seed * 977)[:3]
    out = solve_and_score(obs[0], obs[1], obs[2], obs[3], held)
    if out:
        results["rule_families"][name] = out

for config in family_variants()[:args.games]:
    obs = game_transitions(config, args.observations, args.seed * 31)
    held = game_transitions(config, args.eval_rows, args.seed * 977)[:3]
    out = solve_and_score(obs[0], obs[1], obs[2], obs[3], held)
    if out:
        results["grid_games"][config.name] = out

report["results"] = results
for domain, block in results.items():
    if not block:
        continue
    report[f"{domain}_mean_held_out"] = round(
        sum(v["held_out"] for v in block.values()) / len(block), 4)
    report[f"{domain}_mean_identity"] = round(
        sum(v["identity"] for v in block.values()) / len(block), 4)
    report[f"{domain}_mean_margin"] = round(
        sum(v["held_out"] - v["identity"] for v in block.values())
        / len(block), 4)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
