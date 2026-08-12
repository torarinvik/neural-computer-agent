"""Can PROGRAMMABILITY choose a perception, with no domain knowledge?

Everything above the slot interface is now amodal. The plant is trained
on random programs over random states. The search is
permutation-equivariant. F212 made the reader equivariant too, exactly.
One thing is not: `slot_state`, which is hand-written grid knowledge --
find the avatar by argmax on plane 0, then the nearest object per plane
by Manhattan distance. That is the last place a human put the domain in.

Replacing it needs a criterion for what makes a perception GOOD, and the
criterion cannot mention grids. This probe tests one:

    a good encoder is one whose dynamics are expressible
    as a SHORT PROGRAM

That is domain-general -- it is a statement about compressibility of
transitions, not about avatars -- and it is exactly what the rest of the
architecture already measures. Fit a per-slot parallel program to an
encoder's transitions, score it on transitions nothing was fit on, and
the encoder that makes the world most predictable wins.

**The criterion is degenerate on its own, and that has to be handled or
the answer is a constant.** An encoder that returns zeros is perfectly
programmable: NOOP predicts it exactly. So programmability is scored as
a MARGIN over the identity program, on the slots that actually move, and
reported next to an informativeness measure. An encoder must be both
predictable AND say something.

**Predictions, written before the run so they cannot be revised.**

  1. `handwritten` and `absolute` score highest on margin. They are the
     two encoders built with knowledge of what the game contains.
  2. `constant` scores an undefined or zero margin and maximal
     programmability, demonstrating the degeneracy the margin exists to
     prevent.
  3. `random_linear` and `downsample` score near zero margin: they mix
     independent objects into each slot, so no per-slot program predicts
     them.
  4. `channel_shuffle` scores the SAME as `handwritten`, because the
     interpreter is permutation-equivariant and shuffling which plane
     lands in which slot cannot matter. If it does differ, something
     downstream is not as amodal as F212 claims.
  5. `centroid` scores between the two groups -- informative, but it
     moves fractionally where the avatar moves discretely, so a
     one-instruction slot update will fit it worse.

If the ranking comes out as 1/3/5, the criterion can drive a search over
encoders and the hand-written perception can go. If `random_linear` wins
or `handwritten` loses, it cannot, and this direction is dead.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--examples", type=int, default=32)
parser.add_argument("--eval-rows", type=int, default=128)
parser.add_argument("--games", type=int, default=8)
parser.add_argument("--search", action="store_true",
                    help="greedily discover an encoder from the\n                         domain-general vocabulary instead of ranking\n                         the hand-written ones")
parser.add_argument("--held-games", type=int, default=8)
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)

SLOTS, VALUES = 6, 8
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


# --------------------------------------------------------- the encoders
# Every one maps a screen batch to (rows, SLOTS) integers in [0, VALUES).
# Only the first two were written knowing what a grid game contains.
ROWS_IX = torch.arange(HEIGHT).view(-1, 1).expand(HEIGHT, WIDTH)
COLS_IX = torch.arange(WIDTH).view(1, -1).expand(HEIGHT, WIDTH)
ABSENT = VALUES


def _frames(screen):
    return screen.view(-1, PLANES, HEIGHT, WIDTH)


def enc_handwritten(screen):
    """The current `slot_state`: avatar argmax, then nearest object per
    plane by Manhattan distance. Domain knowledge, and the incumbent."""
    frames = _frames(screen)
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
            distance = (ROWS_IX - ar).abs() + (COLS_IX - ac).abs()
            distance = torch.where(mask, distance,
                                   torch.full_like(distance, 999))
            pick = int(distance.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = pick // WIDTH, pick % WIDTH
    return out


def enc_absolute(screen):
    """Same, but objects chosen in raster order rather than by distance
    to the avatar. Still domain knowledge."""
    frames = _frames(screen)
    out = torch.full((frames.shape[0], SLOTS), ABSENT, dtype=torch.long)
    for row in range(frames.shape[0]):
        for plane in range(PLANES):
            mask = frames[row, plane] > 0
            if not bool(mask.any()):
                continue
            pick = int(mask.reshape(-1).float().argmax())
            out[row, 2 * plane] = pick // WIDTH
            out[row, 2 * plane + 1] = pick % WIDTH
    return out


def enc_channel_shuffle(screen):
    """`absolute` with the planes relabelled. The control for F212's
    equivariance claim: this must score identically."""
    frames = _frames(screen)
    order = [2, 0, 1]
    return enc_absolute(frames[:, order].reshape(screen.shape[0], -1))


def enc_centroid(screen):
    """Centre of mass per plane, rounded. Informative and domain-general,
    but fractional where the avatar is discrete."""
    frames = _frames(screen)
    out = torch.zeros((frames.shape[0], SLOTS), dtype=torch.long)
    for plane in range(PLANES):
        mass = (frames[:, plane] > 0).float()
        total = mass.sum(dim=(1, 2)).clamp(min=1.0)
        r = (mass * ROWS_IX).sum(dim=(1, 2)) / total
        c = (mass * COLS_IX).sum(dim=(1, 2)) / total
        out[:, 2 * plane] = r.round().long().clamp(0, VALUES - 1)
        out[:, 2 * plane + 1] = c.round().long().clamp(0, VALUES - 1)
    return out


def enc_downsample(screen):
    """Average-pool the whole screen to six numbers. Domain-general and
    deliberately blind to object identity."""
    frames = _frames(screen).float()
    pooled = torch.nn.functional.adaptive_avg_pool2d(frames, (1, 2))
    flat = pooled.reshape(frames.shape[0], -1)[:, :SLOTS]
    top = flat.max().clamp(min=1e-6)
    return (flat / top * (VALUES - 1)).round().long().clamp(0, VALUES - 1)


_PROJ = torch.randn(PLANES * HEIGHT * WIDTH, SLOTS,
                    generator=torch.Generator().manual_seed(11))


def enc_random_linear(screen):
    """A fixed random projection, discretised. The null: domain-general
    and structure-destroying."""
    flat = _frames(screen).float().reshape(screen.shape[0], -1)
    z = flat @ _PROJ
    z = (z - z.min(dim=0, keepdim=True).values)
    span = z.max(dim=0, keepdim=True).values.clamp(min=1e-6)
    return (z / span * (VALUES - 1)).round().long().clamp(0, VALUES - 1)


def enc_constant(screen):
    """Zeros. Perfectly programmable and worthless -- the degeneracy the
    margin exists to expose."""
    return torch.zeros((screen.shape[0], SLOTS), dtype=torch.long)


ENCODERS = {
    "handwritten": enc_handwritten,
    "absolute": enc_absolute,
    "channel_shuffle": enc_channel_shuffle,
    "centroid": enc_centroid,
    "downsample": enc_downsample,
    "random_linear": enc_random_linear,
    "constant": enc_constant,
}


def transitions(config, count, seed, action, encoder):
    verifier = FamilyVerifier(config, batch_size=count, seed=seed)
    verifier.reset(seed=seed)
    before = encoder(verifier.observation())
    verifier.step(torch.full((count,), action, dtype=torch.long))
    return before, encoder(verifier.observation())


def sanitise(before, after):
    alive = (before < VALUES).all(dim=1) & (after < VALUES).all(dim=1)
    if int(alive.sum()) < 8:
        return None
    return before[alive], after[alive]


report = {"seed": args.seed, "encoders": {}}
for name, encoder in ({} if args.search else ENCODERS).items():
    fits, floors, margins, moving_counts, distinct, per_slot_values = \
        [], [], [], [], [], []
    for config in family_variants()[:args.games]:
        for action in range(4):
            fit = sanitise(*transitions(config, args.eval_rows, 90001 + action,
                                        action, encoder))
            held = sanitise(*transitions(config, args.eval_rows,
                                         777001 + action, action, encoder))
            if fit is None or held is None:
                continue
            fb, fa = fit[0][:args.examples], fit[1][:args.examples]
            hb, ha = held
            moving = (hb != ha).any(dim=0)
            moving_counts.append(int(moving.sum()))
            # informativeness does not depend on the program
            distinct.append(len({tuple(r.tolist()) for r in hb}))
            per_slot_values.append(
                sum(len(set(hb[:, s].tolist())) for s in range(SLOTS)) / SLOTS)
            if int(moving.sum()) == 0:
                continue
            program = per_slot_search(fb, fa)
            got = run_parallel(hb, program)
            fit_score = float((got[:, moving] == ha[:, moving]).float().mean())
            floor = float((hb[:, moving] == ha[:, moving]).float().mean())
            fits.append(fit_score)
            floors.append(floor)
            margins.append(fit_score - floor)

    def mean(values):
        return round(sum(values) / len(values), 4) if values else None
    report["encoders"][name] = {
        "scored_actions": len(fits),
        "actions_with_no_moving_slot": len(moving_counts) - len(fits),
        "mean_moving_slots": mean([float(x) for x in moving_counts]),
        "distinct_states_seen": mean([float(x) for x in distinct]),
        "distinct_values_per_slot": mean(per_slot_values),
        "identity_floor": mean(floors),
        "program_fit": mean(fits),
        "MARGIN": mean(margins),
    }

if not args.search:
    order = sorted(
        (n for n in report["encoders"]
         if report["encoders"][n]["MARGIN"] is not None),
        key=lambda n: report["encoders"][n]["MARGIN"], reverse=True)
    report["ranking_by_margin"] = order
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# SEARCH: discover a perception instead of writing one.
#
# The vocabulary below is domain-general on purpose. Every entry is a
# reduction over the axes of a (channel, height, width) tensor -- "where
# is the largest cell", "how many cells are set", "how far apart are
# they". None of them mentions an avatar, an object, a wall or a goal,
# and the same vocabulary applies to any tensor-shaped observation.
#
# A slot feature is (channel, reduction). An encoder is six of them.
# Greedy forward selection adds the feature that most improves the
# programmability margin, six times: 30 + 29 + ... + 25 = 165 candidate
# evaluations, against a space of C(30,6) = 594,000 encoders.
def _reductions(plane):
    """Domain-general summaries of one (rows, H, W) channel.

    Each returns an integer in [0, VALUES). Nothing here knows what the
    channel contains; these are the things one can say about any
    2-D array of numbers."""
    mask = plane > 0
    any_set = mask.any(dim=(1, 2))
    flat = plane.reshape(plane.shape[0], -1)
    top = flat.argmax(dim=1)
    rows_any = mask.any(dim=2)
    cols_any = mask.any(dim=1)
    idx_r = torch.arange(HEIGHT).view(1, -1)
    idx_c = torch.arange(WIDTH).view(1, -1)
    big = torch.full_like(idx_r, HEIGHT)
    first_r = torch.where(rows_any, idx_r, big).min(dim=1).values
    last_r = torch.where(rows_any, idx_r, torch.full_like(idx_r, -1)).max(dim=1).values
    first_c = torch.where(cols_any, idx_c, torch.full_like(idx_c, WIDTH)).min(dim=1).values
    last_c = torch.where(cols_any, idx_c, torch.full_like(idx_c, -1)).max(dim=1).values
    weight = mask.float()
    total = weight.sum(dim=(1, 2)).clamp(min=1.0)
    cen_r = (weight * ROWS_IX).sum(dim=(1, 2)) / total
    cen_c = (weight * COLS_IX).sum(dim=(1, 2)) / total
    def clamp(t):
        return t.long().clamp(0, VALUES - 1)
    return {
        "peak_row": clamp(top // WIDTH),
        "peak_col": clamp(top % WIDTH),
        "centre_row": clamp(cen_r.round()),
        "centre_col": clamp(cen_c.round()),
        "first_row": clamp(torch.where(any_set, first_r, torch.zeros_like(first_r))),
        "first_col": clamp(torch.where(any_set, first_c, torch.zeros_like(first_c))),
        "last_row": clamp(torch.where(any_set, last_r, torch.zeros_like(last_r))),
        "last_col": clamp(torch.where(any_set, last_c, torch.zeros_like(last_c))),
        "count": clamp(mask.sum(dim=(1, 2))),
        "extent": clamp((last_r - first_r).clamp(min=0)),
    }


REDUCTIONS = ["peak_row", "peak_col", "centre_row", "centre_col",
              "first_row", "first_col", "last_row", "last_col",
              "count", "extent"]
VOCABULARY = [(c, r) for c in range(PLANES) for r in REDUCTIONS]


def encode_with(features):
    """features: list of (channel, reduction). Slots beyond the list are
    zero, which makes them non-moving and therefore excluded from the
    score -- so a shorter encoder is neither rewarded nor punished."""
    def encoder(screen):
        frames = _frames(screen)
        cache = {c: _reductions(frames[:, c]) for c in {f[0] for f in features}}
        out = torch.zeros((frames.shape[0], SLOTS), dtype=torch.long)
        for s, (c, r) in enumerate(features[:SLOTS]):
            out[:, s] = cache[c][r]
        return out
    return encoder


def score_encoder(encoder, games, seeds=(0,), with_coverage=True):
    """Programmability margin times COVERAGE, on transitions nothing was
    fit on.

    The margin alone is gameable and a greedy search game it: given only
    the margin, it chose six views of channel 0 -- four of them literally
    the same integer, one constant -- scoring 0.6744 against the
    hand-written encoder's 0.3456 by **discarding the objects**. Sixty-one
    of its sixty-four distinct codes mapped to more than one object
    position. Encoding only the part of the world that is easy to predict
    is the `constant` degeneracy wearing a disguise, and the moving-slot
    guard does not catch it because the avatar moves.

    COVERAGE is the fraction of distinguishable world states the encoder
    keeps: distinct slot vectors divided by distinct raw observations. It
    is domain-general, bounded in [0, 1], and has no free parameter to
    tune. A constant encoder scores 1/N. The avatar-only encoder scores
    about 0.25. The hand-written one scores about 0.86.

    The product is the criterion: a perception must make the world
    predictable AND keep it."""
    margins, coverages = [], []
    for offset in seeds:
        for config in games:
            for action in range(4):
                fit = sanitise(*transitions(config, args.eval_rows,
                                            90001 + action + 1000 * offset,
                                            action, encoder))
                held = sanitise(*transitions(config, args.eval_rows,
                                             777001 + action + 1000 * offset,
                                             action, encoder))
                if fit is None or held is None:
                    continue
                fb, fa = fit[0][:args.examples], fit[1][:args.examples]
                hb, ha = held
                moving = (hb != ha).any(dim=0)
                if int(moving.sum()) == 0:
                    continue
                got = run_parallel(hb, per_slot_search(fb, fa))
                margins.append(
                    float((got[:, moving] == ha[:, moving]).float().mean())
                    - float((hb[:, moving] == ha[:, moving]).float().mean()))
                coverages.append(coverage_of(encoder, config, action,
                                             777001 + action + 1000 * offset))
    if not margins:
        return 0.0
    margin = sum(margins) / len(margins)
    if not with_coverage:
        return margin
    return margin * (sum(coverages) / len(coverages))


def coverage_of(encoder, config, action, seed):
    """Distinct slot vectors / distinct raw observations, on the same
    rows the margin was scored on."""
    verifier = FamilyVerifier(config, batch_size=args.eval_rows, seed=seed)
    verifier.reset(seed=seed)
    screen = verifier.observation()
    codes = {tuple(r.tolist()) for r in encoder(screen)}
    raw = {hash(r.numpy().tobytes())
           for r in screen.reshape(screen.shape[0], -1)}
    return len(codes) / max(len(raw), 1)


if args.search:
    # Greedy forward selection. The TRAIN games drive the search; the
    # HELD games are never seen by it, so the discovered encoder is
    # scored on worlds that had no say in choosing it.
    variants = family_variants()
    train_games = variants[:args.games]
    held_games = variants[args.games:args.games + args.held_games]
    chosen: list = []
    trace = []
    remaining = list(VOCABULARY)
    while len(chosen) < SLOTS and remaining:
        best, best_score = None, -1e9
        for feature in remaining:
            score = score_encoder(encode_with(chosen + [feature]), train_games)
            if score > best_score:
                best, best_score = feature, score
        chosen.append(best)
        remaining.remove(best)
        trace.append({"slot": len(chosen) - 1, "added": list(best),
                      "train_margin": round(best_score, 4)})
        print(f"  slot {len(chosen)-1}: +{best}  train margin "
              f"{best_score:.4f}", flush=True)
    discovered = encode_with(chosen)
    report["search"] = {
        "vocabulary_size": len(VOCABULARY),
        "encoder_space": "C(30,6) = 593775",
        "candidates_evaluated": sum(len(VOCABULARY) - i for i in range(SLOTS)),
        "train_games": [str(getattr(c, "name", c)) for c in train_games],
        "held_games": [str(getattr(c, "name", c)) for c in held_games],
        "trace": trace,
        "discovered": [list(f) for f in chosen],
    }
    # the comparison that matters: held-out worlds, discovered against
    # the hand-written incumbent and the domain-general baselines
    comparison = {}
    for name, enc in (("DISCOVERED", discovered),
                      ("handwritten", enc_handwritten),
                      ("absolute", enc_absolute),
                      ("centroid", enc_centroid),
                      ("random_linear", enc_random_linear)):
        comparison[name] = {
            "train_margin": round(score_encoder(enc, train_games), 4),
            "HELD_OUT_margin": round(score_encoder(enc, held_games), 4)}
        print(f"  {name:<14} train {comparison[name]['train_margin']:+.4f}"
              f"   held-out {comparison[name]['HELD_OUT_margin']:+.4f}",
              flush=True)
    report["search"]["comparison"] = comparison
    print(json.dumps(report["search"], indent=2))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
