"""PAIR DYNAMICS FACTORIAL: which structural fix makes mover
dynamics learnable -- joint pair-fit, identity-stable tracking, or
both? (F249)

F248 refuted the missing-instruction hypothesis: no per-slot op set
can express the pursuer's larger-gap-axis-first rule (cross-axis
coupling), and nearest-rank slots swap identity when movers cross
(label noise). This probe runs the pre-registered 2x2:

  fit       per-slot base-ISA programs  VS  joint PAIR fit over a
            generic relational motion vocabulary: FROZEN, FALL
            (row increments mod height), CHASE(g')/FLEE(g') = one
            L1 step toward/away from slot-group g' on the
            larger-|gap| axis (ties -> row), for each other group.
  encoding  nearest-rank slots (the deployed second2 layout)  VS
            identity-stable tracking (continuity matching, the
            F235 EntityTable principle): a tracked entity keeps its
            slot-group as long as a current-frame cell of its plane
            lies within matching distance of its last position.

Data: one long random rollout per world (tracking carried across
steps), transitions recorded per action, held-out split by rows.
Score: held-out exact-match of the GROUP (both slots right), mover
groups (4-5, 6-7) primary.

Registered predictions (before any run):
  P1 pair x track beats slot x rank by >= +0.15 on the trio worlds'
     mover groups.
  P2 both factors necessary: pair x track > pair x rank AND
     > slot x track on the same groups.
  P3 control world: hazard is a random walker; no deterministic
     vocabulary can exceed the step-probability cap -- all cells
     within +0.05 of each other.
  P4 avatar group >= 0.95 in every cell.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--rollout-steps", type=int, default=24)
parser.add_argument("--train-frac", type=float, default=0.67)
parser.add_argument("--match-dist", type=int, default=2)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3
BASE_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))
GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))


def slot_write(state, s, op, j, m):
    name, mod = BASE_OPS[op], MODULI[m]
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


def per_slot_fit(before, after, slots):
    program = {}
    for s in slots:
        want = after[:, s]
        best, best_score = (0, 0, 0), -1.0
        for op in range(len(BASE_OPS)):
            for j in range(SLOTS):
                if j == s and BASE_OPS[op] in ("CINC", "CDEC", "COPY"):
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
        program[s] = best
    return program


def slot_group_acc(program, before, after, group):
    ok = None
    for s in group:
        op, j, m = program[s]
        hit = slot_write(before, s, op, j, m) == after[:, s]
        ok = hit if ok is None else (ok & hit)
    return float(ok.float().mean())


# ---- pair vocabulary: generic relational motions of a slot GROUP ----

def pair_apply(rule, before, group):
    """rule = ("FROZEN",) | ("FALL",) | ("CHASE", g') | ("FLEE", g').
    Returns predicted (row, col) tensors for the group."""
    r, c = before[:, group[0]], before[:, group[1]]
    kind = rule[0]
    if kind == "FROZEN":
        return r, c
    if kind == "FALL":
        return (r + 1) % HEIGHT, c
    tr, tc = before[:, rule[1][0]], before[:, rule[1][1]]
    gap_r, gap_c = tr - r, tc - c
    sign = 1 if kind == "CHASE" else -1
    step_r = sign * torch.sign(gap_r)
    step_c = sign * torch.sign(gap_c)
    row_axis = (gap_r.abs() >= gap_c.abs()) & (gap_r != 0)
    col_axis = ~row_axis & (gap_c != 0)
    at_target = (gap_r == 0) & (gap_c == 0)
    nr = torch.where(row_axis & ~at_target, r + step_r, r)
    nc = torch.where(col_axis & ~at_target, c + step_c, c)
    return nr.clamp(0, HEIGHT - 1), nc.clamp(0, WIDTH - 1)


def pair_rules(group):
    rules = [("FROZEN",), ("FALL",)]
    for other in GROUPS:
        if other == group:
            continue
        rules.append(("CHASE", other))
        rules.append(("FLEE", other))
    return rules


def pair_fit(before, after, group):
    best, best_score = ("FROZEN",), -1.0
    for rule in pair_rules(group):
        nr, nc = pair_apply(rule, before, group)
        score = float(((nr == after[:, group[0]])
                       & (nc == after[:, group[1]])).float().mean())
        if score > best_score:
            best, best_score = rule, score
    return best


def pair_acc(rule, before, after, group):
    nr, nc = pair_apply(rule, before, group)
    return float(((nr == after[:, group[0]])
                  & (nc == after[:, group[1]])).float().mean())


# ---- encodings ----------------------------------------------------

def frame_cells(obs):
    return obs.view(-1, PLANES, HEIGHT, WIDTH)


def rank_encode(frames):
    """Deployed second2 layout, one frame (no temporal slots)."""
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
    """Identity-stable slot state via continuity matching: group 2-3
    tracks one plane-1 entity, groups 4-5 and 6-7 track two plane-2
    entities. A tracked entity keeps its group while some cell of its
    plane lies within match-dist of its last position; otherwise it
    re-acquires by nearest rank (excluding cells already claimed)."""

    def __init__(self, batch):
        self.pos = [[None, None, None] for _ in range(batch)]

    def encode(self, frames, rank_code):
        B = frames.shape[0]
        out = torch.full((B, SLOTS), ABSENT, dtype=torch.long)
        out[:, 0], out[:, 1] = rank_code[:, 0], rank_code[:, 1]
        for b in range(B):
            if int(rank_code[b, 0]) >= VALUES:
                self.pos[b] = [None, None, None]
                continue
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
                    ar, ac = int(rank_code[b, 0]), int(rank_code[b, 1])
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


def collect(config, seed, encoding):
    """One long random rollout; transitions per action with all slots
    present both sides."""
    B, T = args.batch, args.rollout_steps
    v = FamilyVerifier(config, batch_size=B, seed=seed)
    v.reset(seed=seed)
    g = torch.Generator().manual_seed(seed + 31337)
    tracker = Tracker(B) if encoding == "track" else None
    frames = frame_cells(v.observation())
    code = rank_encode(frames)
    if tracker is not None:
        code = tracker.encode(frames, code)
    data = {a: ([], []) for a in range(4)}
    for _ in range(T):
        action = torch.randint(0, 4, (B,), generator=g)
        v.step(action)
        frames = frame_cells(v.observation())
        nxt = rank_encode(frames)
        if tracker is not None:
            nxt = tracker.encode(frames, nxt)
        ok = (code < VALUES).all(dim=1) & (nxt < VALUES).all(dim=1)
        for a in range(4):
            rows = ok & (action == a)
            if bool(rows.any()):
                data[a][0].append(code[rows])
                data[a][1].append(nxt[rows])
        code = nxt
    out = {}
    for a in range(4):
        if data[a][0]:
            out[a] = (torch.cat(data[a][0]), torch.cat(data[a][1]))
    return out


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
    row = {}
    for encoding in ("rank", "track"):
        transitions = collect(config, args.seed * 31, encoding)
        cell = {"slot": {}, "pair": {}}
        counts = {"slot": {}, "pair": {}}
        for a, (before, after) in transitions.items():
            n = before.shape[0]
            cut = max(32, int(n * args.train_frac))
            if n - cut < 24:
                continue
            tb, ta = before[:cut], after[:cut]
            hb, ha = before[cut:], after[cut:]
            prog = per_slot_fit(tb, ta, range(SLOTS))
            for gi, group in enumerate(GROUPS):
                acc_s = slot_group_acc(prog, hb, ha, group)
                rule = pair_fit(tb, ta, group)
                acc_p = pair_acc(rule, hb, ha, group)
                for key, acc in (("slot", acc_s), ("pair", acc_p)):
                    cell[key].setdefault(gi, []).append(acc)
        for key in ("slot", "pair"):
            per_group = [round(sum(v) / len(v), 4) if
                         (v := cell[key].get(gi)) else None
                         for gi in range(4)]
            movers = [x for gi, x in enumerate(per_group)
                      if gi >= 2 and x is not None]
            row[f"{key}_{encoding}"] = per_group
            row[f"{key}_{encoding}_movers"] = (
                round(sum(movers) / len(movers), 4) if movers else None)
    report["results"][name] = row
    print(f"  {name:<40} movers  slot/rank "
          f"{row.get('slot_rank_movers')}  pair/rank "
          f"{row.get('pair_rank_movers')}  slot/track "
          f"{row.get('slot_track_movers')}  pair/track "
          f"{row.get('pair_track_movers')}", flush=True)

print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
