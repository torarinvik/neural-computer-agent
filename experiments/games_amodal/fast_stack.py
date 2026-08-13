"""Batched goal-candidate scoring on the fast verifier.

The selection loop is every probe's wall-clock wall: N candidate goals
x rollouts, sequentially. Here all N candidates share ONE rollout: the
fast verifier runs N*E episodes in a single batch (block i plans for
goal i), the plant executes one batched forward per action, and goal
costs evaluate as padded tensors -- no per-candidate Python loop
inside the step loop.

Everything device-parametric: pass device="mps" to run verifier,
encoder and cost tensors on Apple Metal (the plant can sit on its own
device; states cross over per action call).

Encoders mirror the probe stack: slots 0-1 avatar, 2-3 nearest
plane 1, 4-5 nearest plane 2, 6-7 second-nearest of plane 2
("second2") or nearest APPROACHING cell of plane 2 ("approach2",
vectorized one-step flow -- no per-row loops).
"""

from __future__ import annotations

import torch

SLOTS, VALUES = 8, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3


def _grids(device):
    rows = torch.arange(HEIGHT, device=device).view(-1, 1).expand(
        HEIGHT, WIDTH)
    cols = torch.arange(WIDTH, device=device).view(1, -1).expand(
        HEIGHT, WIDTH)
    return rows, cols


def _kth_nearest(plane, ref_row, ref_col, k):
    device = plane.device
    rows_ix, cols_ix = _grids(device)
    mask = plane > 0
    d = ((rows_ix.unsqueeze(0) - ref_row.view(-1, 1, 1)).abs()
         + (cols_ix.unsqueeze(0) - ref_col.view(-1, 1, 1)).abs())
    d = torch.where(mask, d, torch.full_like(d, 999))
    flat = d.reshape(d.shape[0], -1)
    order = flat.argsort(dim=1)
    idx = order[:, min(k, order.shape[1] - 1)]
    enough = mask.reshape(mask.shape[0], -1).sum(dim=1) > k
    row = torch.where(enough, idx // WIDTH, torch.full_like(idx, ABSENT))
    col = torch.where(enough, idx % WIDTH, torch.full_like(idx, ABSENT))
    return row, col


def _approaching(prev_plane, curr_plane, avatar_r, avatar_c):
    """Vectorized one-step flow: nearest newly-occupied cell whose
    matched newly-vacated origin was farther from the avatar."""
    device = curr_plane.device
    batch = curr_plane.shape[0]
    rows_ix, cols_ix = _grids(device)
    flat_r = rows_ix.reshape(-1)
    flat_c = cols_ix.reshape(-1)
    now = (curr_plane > 0).reshape(batch, -1)
    was = (prev_plane > 0).reshape(batch, -1)
    fresh, gone = now & ~was, was & ~now
    # pairwise |cell_i - cell_j| over the 64-cell grid, shared per batch
    pair = ((flat_r.view(-1, 1) - flat_r.view(1, -1)).abs()
            + (flat_c.view(-1, 1) - flat_c.view(1, -1)).abs())
    d_to_gone = torch.where(gone.unsqueeze(1), pair.unsqueeze(0),
                            torch.full_like(pair, 999).unsqueeze(0))
    origin = d_to_gone.min(dim=2).indices          # [B, 64]
    d_new = ((flat_r.unsqueeze(0) - avatar_r.view(-1, 1)).abs()
             + (flat_c.unsqueeze(0) - avatar_c.view(-1, 1)).abs())
    d_old = ((flat_r[origin] - avatar_r.view(-1, 1)).abs()
             + (flat_c[origin] - avatar_c.view(-1, 1)).abs())
    ok = fresh & gone.any(dim=1, keepdim=True) & (d_new < d_old) \
        & (avatar_r < VALUES).view(-1, 1)
    scored = torch.where(ok, d_new, torch.full_like(d_new, 999))
    best = scored.argmin(dim=1)
    found = ok.any(dim=1)
    row = torch.where(found, best // WIDTH,
                      torch.full_like(best, ABSENT))
    col = torch.where(found, best % WIDTH,
                      torch.full_like(best, ABSENT))
    return row, col


def make_enc(kind):
    def encoder(prev_screen, screen):
        frames = screen.view(-1, PLANES, HEIGHT, WIDTH)
        prior = prev_screen.view(-1, PLANES, HEIGHT, WIDTH)
        out = torch.full((frames.shape[0], SLOTS), ABSENT,
                         dtype=torch.long, device=frames.device)
        avatar = frames[:, 0].reshape(frames.shape[0], -1)
        present = avatar.max(dim=1).values > 0
        flat = avatar.argmax(dim=1)
        ar = torch.where(present, flat // WIDTH,
                         torch.full_like(flat, ABSENT))
        ac = torch.where(present, flat % WIDTH,
                         torch.full_like(flat, ABSENT))
        out[:, 0], out[:, 1] = ar, ac
        for plane, base in ((1, 2), (2, 4)):
            row, col = _kth_nearest(frames[:, plane],
                                    ar.clamp(max=VALUES - 1),
                                    ac.clamp(max=VALUES - 1), 0)
            out[:, base], out[:, base + 1] = row, col
        if kind == "second":
            row, col = _kth_nearest(frames[:, 2],
                                    ar.clamp(max=VALUES - 1),
                                    ac.clamp(max=VALUES - 1), 1)
        else:
            row, col = _approaching(prior[:, 2], frames[:, 2], ar, ac)
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


ENC_CANDIDATES = {"second2": make_enc("second"),
                  "approach2": make_enc("approach")}


def pack_goals(goals, device):
    """Pad a list of tuple-of-terms goals into index tensors.

    Returns (A0, A1, B0, B1, SIGN, MASK), each [N, T_max]."""
    n = len(goals)
    t_max = max(len(g) for g in goals)
    A0 = torch.zeros(n, t_max, dtype=torch.long, device=device)
    A1, B0, B1 = A0.clone(), A0.clone(), A0.clone()
    SIGN = torch.zeros(n, t_max, device=device)
    MASK = torch.zeros(n, t_max, device=device)
    for i, goal in enumerate(goals):
        for j, ((a0, a1), (b0, b1), sign) in enumerate(goal):
            A0[i, j], A1[i, j], B0[i, j], B1[i, j] = a0, a1, b0, b1
            SIGN[i, j], MASK[i, j] = float(sign), 1.0
    return A0, A1, B0, B1, SIGN, MASK


def packed_cost(state, reference, packed, episodes):
    """Cost per row for row-block goals. state/reference [N*E, SLOTS]."""
    A0, A1, B0, B1, SIGN, MASK = packed
    expand = lambda t: t.repeat_interleave(episodes, dim=0)
    a0, a1 = expand(A0), expand(A1)
    b0, b1 = expand(B0), expand(B1)
    reach = ((state.gather(1, a0) - reference.gather(1, b0)).abs()
             + (state.gather(1, a1) - reference.gather(1, b1)).abs()
             ).float()
    return (expand(SIGN) * expand(MASK) * reach).sum(dim=1)


def score_goals(verifier_factory, goals, bank, executor, enc,
                episodes, steps, seed, device="cpu"):
    """Mean return per goal, all goals rolled out in ONE batch."""
    n = len(goals)
    packed = pack_goals(goals, device)
    v = verifier_factory(n * episodes)
    v.reset(seed=seed)
    total = torch.zeros(n * episodes, device=device)
    prev = v.observation()
    for _ in range(steps):
        obs = v.observation()
        reference = enc(prev, obs)
        reference = torch.where(reference < VALUES, reference,
                                torch.zeros_like(reference))
        best, action = None, torch.zeros(n * episodes, dtype=torch.long,
                                         device=device)
        for act in range(4):
            program = bank.get(act)
            state = (reference if program is None
                     else executor(program, reference))
            cost = packed_cost(state, reference, packed, episodes)
            if best is None:
                best = cost.clone()
            else:
                take = cost < best
                best = torch.where(take, cost, best)
                action = torch.where(
                    take, torch.full_like(action, act), action)
        total += v.step(action).reward
        prev = obs
    return total.view(n, episodes).mean(dim=1)
