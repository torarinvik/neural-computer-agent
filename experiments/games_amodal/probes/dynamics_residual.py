"""DYNAMICS RESIDUAL MINING: is the bank's model error inexpressible
in the ISA, or just under-sampled? (F248)

F247 localized the witness failure to the bank-dynamics layer: the
VALUE-PLAN capability triples the privileged d4 ceiling where bank
programs roll the world faithfully, but on the multi-mover witness
worlds the planner tracks random because per-slot programs cannot
model interceptor/pursuer motion -- movers step by sign(s0 - s_mover),
and the ISA's only conditional is s_j != 0. This is exactly the F228
pre-registered gate for instruction-set work: "first mine
per_slot_search residuals for inexpressible dynamics."

The diagnostic is symbolic (per_slot_search + run of candidate ops on
held-out transitions); the plant only pays its retraining cost if the
witness confirms and the op is admitted.

Per world x action: collect encoded random-policy transitions, fit
per-slot programs on a train split under (a) the base ISA with the
deployed 32 examples, (b) the base ISA with 256 examples, (c) the
base ISA + TOWARD/AWAY ops (s' = s +/- sign(s_j - s)) with 256
examples. Score held-out per-slot exact-match accuracy.

Registered predictions (before any run):
  P1 expressiveness witness: on the mover slots (4-7) of the three
     intercept x pursue x resource worlds, ext-256 beats base-256
     held-out accuracy by >= +0.10 (mean over slots 4-7, actions,
     seeds) -- the residual is inexpressibility, not sampling noise.
  P2 sample-starvation control: base-256 - base-32 < +0.05 on the
     same slots (more data alone does not close the residual).
  P3 no-harm control: on ctrl_avoid1_collect1 the extended ISA
     changes mover-slot accuracy by < +0.05 (TOWARD earns its place
     only where relative motion exists; elsewhere it is inert).
  P4 avatar slots (0-1) are already well fit everywhere
     (base-256 >= 0.9): the residual is concentrated in movers.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--observations", type=int, default=1024)
parser.add_argument("--test-rows", type=int, default=512)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
BASE_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
EXT_OPS = BASE_OPS + ("TOWARD", "AWAY")
MODULI = tuple(range(2, VALUES + 1))
ROWS_IX = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
COLS_IX = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)


def slot_write(ops, state, s, op, j, m):
    name, mod = ops[op], MODULI[m]
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
    if name == "TOWARD":
        return (col + torch.sign(state[:, j] - col)).clamp(0, VALUES - 1)
    if name == "AWAY":
        return (col - torch.sign(state[:, j] - col)).clamp(0, VALUES - 1)
    raise AssertionError(name)


def per_slot_fit(ops, before, after):
    """Best (op, j, m) per slot on the train split, by train accuracy."""
    program = []
    for s in range(SLOTS):
        want = after[:, s]
        best, best_score = (0, 0, 0), -1.0
        for op in range(len(ops)):
            for j in range(SLOTS):
                if j == s and ops[op] in ("CINC", "CDEC", "COPY",
                                          "TOWARD", "AWAY"):
                    continue
                for m in range(len(MODULI)):
                    score = float(
                        (slot_write(ops, before, s, op, j, m) == want)
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


def heldout_acc(ops, program, before, after):
    """Per-slot exact-match accuracy on held-out transitions."""
    out = []
    for s in range(SLOTS):
        op, j, m = program[s]
        out.append(float((slot_write(ops, before, s, op, j, m)
                          == after[:, s]).float().mean()))
    return out


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


def used_slots(config, seed):
    """The deployed bank's presence mask: slots present in >= 90% of a
    random-policy probe batch (build_bank's rule)."""
    n = args.observations
    g = torch.Generator().manual_seed(seed + 999)
    v = FamilyVerifier(config, batch_size=n, seed=seed + 7)
    v.reset(seed=seed + 7)
    first = v.observation()
    v.step(torch.randint(0, 4, (n,), generator=g))
    code = enc(first, v.observation())
    return (code < VALUES).float().mean(dim=0) >= 0.9


def collect(config, seed, act, used):
    """(before, after) encoded transition pairs for one action, rows
    where every USED slot is present both sides (build_bank's rule)."""
    n = args.observations + args.test_rows
    g = torch.Generator().manual_seed(seed + 999)
    v = FamilyVerifier(config, batch_size=n, seed=seed + act)
    v.reset(seed=seed + act)
    first = v.observation()
    v.step(torch.randint(0, 4, (n,), generator=g))
    second = v.observation()
    before = enc(first, second)
    v.step(torch.full((n,), act, dtype=torch.long))
    after = enc(second, v.observation())
    keep = ((before[:, used] < VALUES).all(dim=1)
            & (after[:, used] < VALUES).all(dim=1))
    return clamp_state(before[keep]), clamp_state(after[keep])


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

report = {"seed": args.seed, "results": {}}
for name, config in WORLDS:
    used = used_slots(config, args.seed * 31)
    slot_ok = [bool(used[s]) for s in range(SLOTS)]
    accs = {"base32": torch.zeros(4, SLOTS),
            "base256": torch.zeros(4, SLOTS),
            "ext256": torch.zeros(4, SLOTS)}
    counts = torch.zeros(4)
    test_min = min(args.test_rows, 96)
    for act in range(4):
        before, after = collect(config, args.seed * 31, act, used)
        if before.shape[0] < 128 + test_min:
            continue
        test_b = before[-test_min:]
        test_a = after[-test_min:]
        train_b, train_a = before[:-test_min], after[:-test_min]
        counts[act] = 1
        for label, ops, n_train in (("base32", BASE_OPS, 32),
                                    ("base256", BASE_OPS, 256),
                                    ("ext256", EXT_OPS, 256)):
            prog = per_slot_fit(ops, train_b[:n_train], train_a[:n_train])
            accs[label][act] = torch.tensor(
                heldout_acc(ops, prog, test_b, test_a))
    acts_ok = counts.bool()
    if not bool(acts_ok.any()):
        report["results"][name] = None
        print(f"  {name:<40} no usable transitions", flush=True)
        continue
    mover_ix = [s for s in (4, 5, 6, 7) if slot_ok[s]]
    row = {"used_slots": slot_ok}
    for label in ("base32", "base256", "ext256"):
        per_slot = accs[label][acts_ok].mean(dim=0)
        row[label] = [round(float(per_slot[s]), 4) if slot_ok[s] else None
                      for s in range(SLOTS)]
        row[label + "_movers"] = round(
            float(per_slot[mover_ix].mean()), 4) if mover_ix else None
        row[label + "_avatar"] = round(float(per_slot[0:2].mean()), 4)
    report["results"][name] = row
    mv = {k: row[k + "_movers"] for k in ("base32", "base256", "ext256")}
    print(f"  {name:<40} movers base32 {mv['base32']} "
          f"base256 {mv['base256']} ext256 {mv['ext256']}   "
          f"avatar base256 {row['base256_avatar']:.3f}", flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
