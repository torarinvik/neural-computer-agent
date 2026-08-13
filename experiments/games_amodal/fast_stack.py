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


def _clear_nearest(plane_target, plane_threat, avatar_r, avatar_c,
                   thresh):
    """Nearest target-plane cell whose Manhattan distance to EVERY
    threat-plane cell is >= thresh (relational value filter)."""
    device = plane_target.device
    rows_ix, cols_ix = _grids(device)
    flat_r = rows_ix.reshape(-1)
    flat_c = cols_ix.reshape(-1)
    targets = (plane_target > 0).reshape(plane_target.shape[0], -1)
    threats = (plane_threat > 0).reshape(plane_threat.shape[0], -1)
    pair = ((flat_r.view(-1, 1) - flat_r.view(1, -1)).abs()
            + (flat_c.view(-1, 1) - flat_c.view(1, -1)).abs())
    d_threat = torch.where(threats.unsqueeze(1), pair.unsqueeze(0),
                           torch.full_like(pair, 999).unsqueeze(0))
    min_threat = d_threat.min(dim=2).values          # [B, 64]
    ok = targets & (min_threat >= thresh)         & (avatar_r < VALUES).view(-1, 1)
    d_av = ((flat_r.unsqueeze(0) - avatar_r.view(-1, 1)).abs()
            + (flat_c.unsqueeze(0) - avatar_c.view(-1, 1)).abs())
    scored = torch.where(ok, d_av, torch.full_like(d_av, 999))
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
        elif kind == "approach":
            row, col = _approaching(prior[:, 2], frames[:, 2], ar, ac)
        else:  # clear-N: hazard-clear nearest of plane 1
            row, col = _clear_nearest(frames[:, 1], frames[:, 2],
                                      ar, ac, int(kind[-1]))
        out[:, 6], out[:, 7] = row, col
        return out
    return encoder


ENC_CANDIDATES = {"second2": make_enc("second"),
                  "approach2": make_enc("approach"),
                  "clear1_2": make_enc("clear_2"),
                  "clear1_3": make_enc("clear_3")}


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


def pack_schedules(schedules, device):
    """Pad [(phase0_goal, phase1_goal), ...] into flat phase tensors:
    row-goal index = schedule_index * 2 + phase."""
    flat = []
    for p0, p1 in schedules:
        flat.append(p0)
        flat.append(p1)
    return pack_goals(flat, device)


def score_schedules(verifier_factory, schedules, bank, executor, enc,
                    episodes, steps, seed, device="cpu"):
    """Mean return per 2-phase cyclic schedule, one batch for all.

    Phase semantics mirror the probe stack: a row advances its phase
    when the ACTIVE phase goal's cost at the current reference is <= 0
    (arrival), cyclically."""
    n = len(schedules)
    A0, A1, B0, B1, SIGN, MASK = pack_schedules(schedules, device)
    v = verifier_factory(n * episodes)
    v.reset(seed=seed)
    total = torch.zeros(n * episodes, device=device)
    prev = v.observation()
    block = torch.arange(n, device=device).repeat_interleave(episodes)
    phase = torch.zeros(n * episodes, dtype=torch.long, device=device)

    def cost_for(state, reference, phase_now):
        idx = block * 2 + phase_now
        a0, a1 = A0[idx], A1[idx]
        b0, b1 = B0[idx], B1[idx]
        reach = ((state.gather(1, a0) - reference.gather(1, b0)).abs()
                 + (state.gather(1, a1) - reference.gather(1, b1)).abs()
                 ).float()
        return (SIGN[idx] * MASK[idx] * reach).sum(dim=1)

    prev_ref = None
    prev_cost = torch.full((n * episodes,), 1e9, device=device)
    for _ in range(steps):
        obs = v.observation()
        reference = enc(prev, obs)
        reference = torch.where(reference < VALUES, reference,
                                torch.zeros_like(reference))
        here = cost_for(reference, reference, phase)
        # completion = arrived, OR consumed: the phase's target slots
        # jumped discontinuously while we stood within reach. A target
        # eaten on contact respawns elsewhere in the same step, so its
        # distance never reads 0 -- the jump is the arrival signature.
        if prev_ref is None:
            consumed = torch.zeros_like(here, dtype=torch.bool)
        else:
            idx = block * 2 + phase
            b0, b1 = B0[idx], B1[idx]
            jump = ((reference.gather(1, b0)
                     - prev_ref.gather(1, b0)).abs().float()
                    + (reference.gather(1, b1)
                       - prev_ref.gather(1, b1)).abs().float())
            jump = (jump * MASK[idx]).amax(dim=1)
            consumed = (prev_cost <= 1.0) & (jump > 1.0)
        done = (here <= 0) | consumed
        phase = torch.where(done, (phase + 1) % 2, phase)
        prev_ref = reference
        prev_cost = torch.where(done,
                                torch.full_like(here, 1e9),
                                cost_for(reference, reference, phase))
        best, action = None, torch.zeros(n * episodes, dtype=torch.long,
                                         device=device)
        for act in range(4):
            program = bank.get(act)
            state = (reference if program is None
                     else executor(program, reference))
            cost = cost_for(state, reference, phase)
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


def score_guards(verifier_factory, guards, bank, executor, enc,
                 episodes, steps, seed, device="cpu"):
    """Mean return per guarded goal, one batch for all.

    A guard is (condition, goal_A, goal_B) with condition
    ((c0, c1), (d0, d1), theta): plan with goal_A on rows where
    |ref[c0]-ref[d0]| + |ref[c1]-ref[d1]| >= theta, else goal_B."""
    n = len(guards)
    flat_goals = []
    for cond, goal_a, goal_b in guards:
        flat_goals.append(goal_a)
        flat_goals.append(goal_b)
    A0, A1, B0, B1, SIGN, MASK = pack_goals(flat_goals, device)
    C = torch.tensor([[c[0][0], c[0][1], c[1][0], c[1][1], c[2]]
                      for c, _, _ in guards], dtype=torch.long,
                     device=device)
    v = verifier_factory(n * episodes)
    v.reset(seed=seed)
    total = torch.zeros(n * episodes, device=device)
    prev = v.observation()
    block = torch.arange(n, device=device).repeat_interleave(episodes)

    def cost_for(state, reference, idx):
        a0, a1 = A0[idx], A1[idx]
        b0, b1 = B0[idx], B1[idx]
        reach = ((state.gather(1, a0) - reference.gather(1, b0)).abs()
                 + (state.gather(1, a1) - reference.gather(1, b1)).abs()
                 ).float()
        return (SIGN[idx] * MASK[idx] * reach).sum(dim=1)

    for _ in range(steps):
        obs = v.observation()
        reference = enc(prev, obs)
        reference = torch.where(reference < VALUES, reference,
                                torch.zeros_like(reference))
        cond = C[block]
        gap = ((reference.gather(1, cond[:, 0:1]).squeeze(1)
                - reference.gather(1, cond[:, 2:3]).squeeze(1)).abs()
               + (reference.gather(1, cond[:, 1:2]).squeeze(1)
                  - reference.gather(1, cond[:, 3:4]).squeeze(1)).abs())
        branch = (gap >= cond[:, 4]).long()          # 1 -> goal_A? no:
        idx = block * 2 + (1 - branch)               # A when true
        best, action = None, torch.zeros(n * episodes, dtype=torch.long,
                                         device=device)
        for act in range(4):
            program = bank.get(act)
            state = (reference if program is None
                     else executor(program, reference))
            cost = cost_for(state, reference, idx)
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
