"""One hundred domains, made as unlike each other as the interface allows.

The transfer question only means something if the domains genuinely
differ. Sixteen variations on "apply a random program" would measure how
well a reader interpolates inside one family and would say nothing about
transfer. So these are drawn from as many distinct sources of structure
as I could find that still speak the amodal slot interface:

  arithmetic (wrapping, saturating, affine, multiplicative, Collatz,
  Fibonacci, GCD), bitwise (XOR, shifts, rotations, parity, Gray code,
  LFSR, popcount), order statistics (min, max, median, rank, sorting
  networks, bubble passes), spatial and neighbourhood rules (cellular
  automata, diffusion, reflection, torus motion), memory and indirection
  (pointer chase, gather, scatter, queues, stacks), state machines
  (traffic lights, elevators, odometers, flip-flops), aggregation
  (checksums, histograms, prefix scans), the program families the plant
  is trained on, and the two REAL domains -- rule families and grid
  games.

Every domain is a function `rng -> (before, after)`, both of shape
(examples, SLOTS), values in 0..VALUES-1. A domain may be entirely
INEXPRESSIBLE in the recipe language; that is not a defect. The transfer
measurement normalises each column by what the search can reach on it,
so a domain the language cannot touch reports no margin and is dropped
from the analysis rather than silently scoring zero.
"""

from __future__ import annotations

import torch

from experiments.games_amodal.game_family import (
    FamilyVerifier, family_variants)
from experiments.games_amodal.probes.schema_families import (
    RandomFamily, random_family_spec)

SLOTS, VALUES = 6, 8
ABSENT = VALUES
HEIGHT = WIDTH = 8
PLANES = 3

REGISTRY: dict = {}


def pick(rng, high):
    return int(torch.randint(0, high, (1,), generator=rng))


def domain(name, high=VALUES, rows=32):
    """Register a state->state rule as a domain.

    `high` bounds the value range the domain's states occupy, which is
    itself a source of difference: F201 found the state range mattered
    more than the rule."""
    def decorate(fn):
        def sample(rng, examples=rows):
            before = torch.randint(0, high, (examples, SLOTS),
                                   generator=rng)
            after = fn(rng, before)
            return before, (after % VALUES).long()
        REGISTRY[name] = sample
        return fn
    return decorate


def rolled(state, k):
    return torch.roll(state, k, dims=1)


# ------------------------------------------------------ 1. arithmetic
@domain("add_const")
def _(rng, s):
    return s + pick(rng, VALUES)


@domain("sub_const")
def _(rng, s):
    return s - pick(rng, VALUES)


@domain("mul_const")
def _(rng, s):
    return s * (pick(rng, 6) + 2)


@domain("affine")
def _(rng, s):
    return (pick(rng, 6) + 2) * s + pick(rng, VALUES)


@domain("negate")
def _(rng, s):
    return -s


@domain("invert")
def _(rng, s):
    return (VALUES - 1) - s


@domain("double")
def _(rng, s):
    return s * 2


@domain("halve")
def _(rng, s):
    return s // 2


@domain("square")
def _(rng, s):
    return s * s


@domain("saturate_up")
def _(rng, s):
    return torch.clamp(s + 1 + pick(rng, 2), max=VALUES - 1)


@domain("saturate_down")
def _(rng, s):
    return torch.clamp(s - 1 - pick(rng, 2), min=0)


@domain("clamp_band")
def _(rng, s):
    low = pick(rng, 3)
    return torch.clamp(s, min=low, max=low + 4)


@domain("collatz")
def _(rng, s):
    return torch.where(s % 2 == 0, s // 2, 3 * s + 1)


@domain("fibonacci_chain")
def _(rng, s):
    out = s.clone()
    out[:, 2:] = s[:, :-2] + s[:, 1:-1]
    return out


@domain("gcd_step")
def _(rng, s):
    a, b = s[:, 0].clamp(min=1), s[:, 1].clamp(min=1)
    out = s.clone()
    out[:, 0], out[:, 1] = b, a % b
    return out


@domain("digit_carry")
def _(rng, s):
    """An odometer: add one to the last slot and carry leftwards."""
    out = s.clone()
    carry = torch.ones_like(s[:, 0])
    for k in range(SLOTS - 1, -1, -1):
        total = out[:, k] + carry
        out[:, k] = total % VALUES
        carry = (total >= VALUES).long()
    return out


@domain("base3_counter", high=3)
def _(rng, s):
    out = s.clone()
    carry = torch.ones_like(s[:, 0])
    for k in range(SLOTS - 1, -1, -1):
        total = out[:, k] + carry
        out[:, k] = total % 3
        carry = (total >= 3).long()
    return out


@domain("modular_inverse_ish")
def _(rng, s):
    return (s * s * s)


@domain("triangular")
def _(rng, s):
    return s * (s + 1) // 2


@domain("alternating_sign")
def _(rng, s):
    sign = torch.tensor([1 if k % 2 == 0 else -1 for k in range(SLOTS)])
    return s + sign


# --------------------------------------------------------- 2. bitwise
@domain("xor_const")
def _(rng, s):
    return torch.bitwise_xor(s, pick(rng, VALUES))


@domain("xor_neighbour")
def _(rng, s):
    return torch.bitwise_xor(s, rolled(s, 1))


@domain("and_neighbour")
def _(rng, s):
    return torch.bitwise_and(s, rolled(s, 1))


@domain("or_neighbour")
def _(rng, s):
    return torch.bitwise_or(s, rolled(s, -1))


@domain("nand_neighbour")
def _(rng, s):
    return (VALUES - 1) - torch.bitwise_and(s, rolled(s, 1))


@domain("bit_reverse")
def _(rng, s):
    """Reverse the three bits of each value: 0b abc -> 0b cba."""
    return ((s & 1) << 2) | (s & 2) | ((s >> 2) & 1)


@domain("bit_rotate")
def _(rng, s):
    return ((s << 1) | (s >> 2)) % VALUES


@domain("parity_bit")
def _(rng, s):
    par = (s % 2).sum(dim=1, keepdim=True) % 2
    out = s.clone()
    out[:, 0] = par[:, 0]
    return out


@domain("popcount")
def _(rng, s):
    return (s & 1) + ((s >> 1) & 1) + ((s >> 2) & 1)


@domain("gray_code")
def _(rng, s):
    return torch.bitwise_xor(s, s >> 1)


@domain("lfsr")
def _(rng, s):
    feed = torch.bitwise_xor(s[:, 0] % 2, s[:, -1] % 2)
    out = rolled(s, 1)
    out = out.clone()
    out[:, 0] = feed
    return out


@domain("bitmask_keep")
def _(rng, s):
    return torch.bitwise_and(s, 1 + pick(rng, VALUES - 1))


@domain("bits_only", high=2)
def _(rng, s):
    return 1 - s


@domain("bits_and_shift", high=2)
def _(rng, s):
    return torch.bitwise_xor(s, rolled(s, 1))


# ------------------------------------------------- 3. order statistics
@domain("row_max")
def _(rng, s):
    return s.max(dim=1, keepdim=True).values.expand_as(s)


@domain("row_min")
def _(rng, s):
    return s.min(dim=1, keepdim=True).values.expand_as(s)


@domain("row_mean")
def _(rng, s):
    return s.float().mean(dim=1, keepdim=True).long().expand_as(s)


@domain("row_median")
def _(rng, s):
    return s.median(dim=1, keepdim=True).values.expand_as(s)


@domain("sort_row")
def _(rng, s):
    return s.sort(dim=1).values


@domain("sort_descending")
def _(rng, s):
    return s.sort(dim=1, descending=True).values


@domain("bubble_pass")
def _(rng, s):
    out = s.clone()
    for k in range(SLOTS - 1):
        left, right = out[:, k].clone(), out[:, k + 1].clone()
        out[:, k] = torch.minimum(left, right)
        out[:, k + 1] = torch.maximum(left, right)
    return out


@domain("odd_even_network")
def _(rng, s):
    out = s.clone()
    for k in range(0, SLOTS - 1, 2):
        left, right = out[:, k].clone(), out[:, k + 1].clone()
        out[:, k] = torch.minimum(left, right)
        out[:, k + 1] = torch.maximum(left, right)
    return out


@domain("compare_exchange_pair")
def _(rng, s):
    a = pick(rng, SLOTS)
    b = (a + 1 + pick(rng, SLOTS - 1)) % SLOTS
    out = s.clone()
    left, right = s[:, a], s[:, b]
    out[:, a] = torch.minimum(left, right)
    out[:, b] = torch.maximum(left, right)
    return out


@domain("rank_of_first")
def _(rng, s):
    out = s.clone()
    out[:, 0] = (s < s[:, :1]).sum(dim=1)
    return out


@domain("argmax_slot")
def _(rng, s):
    out = s.clone()
    out[:, 0] = s.argmax(dim=1)
    return out


@domain("argmin_slot")
def _(rng, s):
    out = s.clone()
    out[:, 0] = s.argmin(dim=1)
    return out


@domain("count_zeros")
def _(rng, s):
    out = s.clone()
    out[:, 0] = (s == 0).sum(dim=1)
    return out


@domain("count_above_threshold")
def _(rng, s):
    out = s.clone()
    out[:, 0] = (s > pick(rng, VALUES)).sum(dim=1)
    return out


# --------------------------------------------- 4. neighbourhood rules
@domain("neighbour_sum")
def _(rng, s):
    return rolled(s, 1) + rolled(s, -1)


@domain("neighbour_max")
def _(rng, s):
    return torch.maximum(rolled(s, 1), rolled(s, -1))


@domain("neighbour_min")
def _(rng, s):
    return torch.minimum(rolled(s, 1), rolled(s, -1))


@domain("diffusion")
def _(rng, s):
    return (rolled(s, 1) + 2 * s + rolled(s, -1)) // 4


@domain("gradient")
def _(rng, s):
    return s - rolled(s, 1)


@domain("abs_gradient")
def _(rng, s):
    return (s - rolled(s, 1)).abs()


@domain("life_like", high=2)
def _(rng, s):
    alive = rolled(s, 1) + rolled(s, -1)
    return ((alive == 1) | ((s == 1) & (alive == 2))).long()


@domain("rule110", high=2)
def _(rng, s):
    left, right = rolled(s, 1), rolled(s, -1)
    return ((left ^ 1) & (s | right) | (s & right ^ 1)) % 2


@domain("smooth_toward_mean")
def _(rng, s):
    mean = s.float().mean(dim=1, keepdim=True)
    return (s.float() + torch.sign(mean - s.float())).long()


@domain("reflect_boundary")
def _(rng, s):
    step = 1 + pick(rng, 2)
    moved = s + step
    return torch.where(moved >= VALUES, 2 * (VALUES - 1) - moved, moved)


@domain("torus_move")
def _(rng, s):
    return s + pick(rng, 3) - 1


# ------------------------------------------- 5. memory and indirection
@domain("rotate_left")
def _(rng, s):
    return rolled(s, -1 - pick(rng, 2))


@domain("rotate_right")
def _(rng, s):
    return rolled(s, 1 + pick(rng, 2))


@domain("reverse_row")
def _(rng, s):
    return s.flip(dims=[1])


@domain("swap_halves")
def _(rng, s):
    return torch.cat([s[:, SLOTS // 2:], s[:, :SLOTS // 2]], dim=1)


@domain("interleave")
def _(rng, s):
    order = [0, 3, 1, 4, 2, 5]
    return s[:, order]


@domain("random_permutation")
def _(rng, s):
    return s[:, torch.randperm(SLOTS, generator=rng)]


@domain("pointer_chase")
def _(rng, s):
    index = (s % SLOTS)
    return torch.gather(s, 1, index)


@domain("gather_from_first")
def _(rng, s):
    index = (s[:, :1] % SLOTS).expand_as(s)
    return torch.gather(s, 1, index)


@domain("scatter_constant")
def _(rng, s):
    out = s.clone()
    index = (s[:, :1] % SLOTS)
    return out.scatter(1, index, pick(rng, VALUES))


@domain("queue_shift")
def _(rng, s):
    out = rolled(s, -1).clone()
    out[:, -1] = pick(rng, VALUES)
    return out


@domain("stack_push")
def _(rng, s):
    out = rolled(s, 1).clone()
    out[:, 0] = s[:, 0] + 1
    return out


@domain("broadcast_first")
def _(rng, s):
    return s[:, :1].expand_as(s)


@domain("broadcast_last")
def _(rng, s):
    return s[:, -1:].expand_as(s)


@domain("copy_neighbour")
def _(rng, s):
    return rolled(s, 1)


@domain("duplicate_pairs")
def _(rng, s):
    out = s.clone()
    out[:, 1::2] = s[:, 0::2]
    return out


# ----------------------------------------------- 6. conditional / gated
@domain("gate_on_zero")
def _(rng, s):
    return torch.where(rolled(s, 1) == 0, s, s + 1)


@domain("gate_on_nonzero")
def _(rng, s):
    return torch.where(rolled(s, 1) != 0, s - 1, s)


@domain("threshold_binary")
def _(rng, s):
    return (s > pick(rng, VALUES)).long()


@domain("conditional_swap")
def _(rng, s):
    out = s.clone()
    swap = s[:, 0] > s[:, 1]
    out[:, 0] = torch.where(swap, s[:, 1], s[:, 0])
    out[:, 1] = torch.where(swap, s[:, 0], s[:, 1])
    return out


@domain("select_by_flag")
def _(rng, s):
    flag = (s[:, :1] % 2) == 1
    return torch.where(flag.expand_as(s), rolled(s, 1), rolled(s, -1))


@domain("mask_update")
def _(rng, s):
    keep = (s % 2) == 0
    return torch.where(keep, s, torch.zeros_like(s))


@domain("saturating_accumulate")
def _(rng, s):
    return torch.clamp(s + s[:, :1], max=VALUES - 1)


@domain("reset_on_max")
def _(rng, s):
    return torch.where(s >= VALUES - 1, torch.zeros_like(s), s + 1)


# -------------------------------------------------- 7. state machines
@domain("traffic_light", high=3)
def _(rng, s):
    return (s + 1) % 3


@domain("flip_flop", high=2)
def _(rng, s):
    toggle = (s[:, :1] == 1).expand_as(s)
    return torch.where(toggle, 1 - s, s)


@domain("elevator")
def _(rng, s):
    """Position in slot 0 moves toward the target in slot 1."""
    out = s.clone()
    out[:, 0] = s[:, 0] + torch.sign(s[:, 1] - s[:, 0])
    return out


@domain("clock_hm")
def _(rng, s):
    out = s.clone()
    minutes = s[:, 1] + 1
    out[:, 1] = minutes % 6
    out[:, 0] = (s[:, 0] + (minutes >= 6).long()) % VALUES
    return out


@domain("chase_target")
def _(rng, s):
    out = s.clone()
    for a, b in ((0, 2), (1, 3)):
        out[:, a] = s[:, a] + torch.sign(s[:, b] - s[:, a])
    return out


@domain("flee_target")
def _(rng, s):
    out = s.clone()
    for a, b in ((0, 2), (1, 3)):
        out[:, a] = s[:, a] - torch.sign(s[:, b] - s[:, a])
    return out


@domain("bounce_walker")
def _(rng, s):
    out = s.clone()
    velocity = torch.where(s[:, 1] % 2 == 0, 1, -1)
    nxt = s[:, 0] + velocity
    out[:, 0] = torch.clamp(nxt, 0, VALUES - 1)
    out[:, 1] = torch.where((nxt < 0) | (nxt > VALUES - 1),
                            s[:, 1] + 1, s[:, 1])
    return out


@domain("vending_machine")
def _(rng, s):
    """Credit accumulates in slot 0 and dispenses at a price."""
    out = s.clone()
    credit = s[:, 0] + s[:, 1]
    out[:, 0] = torch.where(credit >= 5, credit - 5, credit)
    out[:, 2] = torch.where(credit >= 5, s[:, 2] + 1, s[:, 2])
    return out


# ------------------------------------------------------ 8. aggregation
@domain("checksum")
def _(rng, s):
    out = s.clone()
    out[:, -1] = s[:, :-1].sum(dim=1)
    return out


@domain("prefix_sum")
def _(rng, s):
    return s.cumsum(dim=1)


@domain("prefix_max")
def _(rng, s):
    return s.cummax(dim=1).values


@domain("prefix_parity")
def _(rng, s):
    return s.cumsum(dim=1) % 2


@domain("suffix_sum")
def _(rng, s):
    return s.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])


@domain("sum_broadcast")
def _(rng, s):
    return s.sum(dim=1, keepdim=True).expand_as(s)


@domain("difference_chain")
def _(rng, s):
    out = s.clone()
    out[:, 1:] = s[:, 1:] - s[:, :-1]
    return out


@domain("histogram_bucket")
def _(rng, s):
    out = torch.zeros_like(s)
    for k in range(SLOTS):
        out[:, k] = (s == k).sum(dim=1)
    return out


# --------------------------------- 9. the plant's own program families
# Defined here rather than imported, so the battery has no dependency on
# the probe that consumes it.
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))
OP = {name: index for index, name in enumerate(PAR_OPS)}


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


def _program_domain(name, choose, high=VALUES):
    def sample(rng, examples=32):
        program = choose(rng)
        before = torch.randint(0, high, (examples, SLOTS), generator=rng)
        return before, run_parallel(before, program).long()
    REGISTRY[name] = sample


def _other(rng, s):
    j = pick(rng, SLOTS)
    return (j + 1) % SLOTS if j == s else j


def _any_slot(rng, s):
    return (pick(rng, len(PAR_OPS)), _other(rng, s), pick(rng, len(MODULI)))


_program_domain("prog_random",
                lambda r: [_any_slot(r, s) for s in range(SLOTS)])
_program_domain("prog_dense", lambda r: [
    ([OP[n] for n in ("INC", "DEC", "CINC", "CDEC", "COPY", "SINC",
                      "SDEC")][pick(r, 7)], _other(r, s),
     pick(r, len(MODULI))) for s in range(SLOTS)])
_program_domain("prog_sparse1", lambda r, _t=[0]: None)


def _sparse_k(k):
    def choose(rng):
        chosen = set()
        while len(chosen) < k:
            chosen.add(pick(rng, SLOTS))
        return [_any_slot(rng, s) if s in chosen else (0, 0, 0)
                for s in range(SLOTS)]
    return choose


_program_domain("prog_sparse1", _sparse_k(1))
_program_domain("prog_sparse2", _sparse_k(2))
_program_domain("prog_sparse3", _sparse_k(3))
_program_domain("prog_gated", lambda r: [
    (OP["CINC"] if pick(r, 2) else OP["CDEC"], _other(r, s),
     pick(r, len(MODULI))) for s in range(SLOTS)])
_program_domain("prog_copy_only", lambda r: [
    (OP["COPY"], _other(r, s), 0) for s in range(SLOTS)])
_program_domain("prog_smallmod", lambda r: [
    (OP["INC"] if pick(r, 2) else OP["DEC"], _other(r, s), pick(r, 2))
    for s in range(SLOTS)], high=4)
_program_domain("prog_narrow_states",
                lambda r: [_any_slot(r, s) for s in range(SLOTS)], high=3)


# ------------------------------------------------------- 10. the real
def _rule_families(rng, examples=32):
    for _ in range(60):
        family = RandomFamily(random_family_spec(rng))
        action = pick(rng, family.actions)
        size = len(family.states)
        index = torch.randint(0, size, (2 * examples,), generator=rng)
        nxt = torch.tensor([family.table[int(x)][action] for x in index])
        before = family.slot_values(index)
        after = family.slot_values(nxt)
        alive = (before[:, 0] < VALUES) & (after[:, 0] < VALUES)
        before, after = before[alive], after[alive]
        if before.shape[0] < examples:
            continue
        before = torch.where(before < VALUES, before,
                             torch.zeros_like(before))[:examples]
        after = torch.where(after < VALUES, after,
                            torch.zeros_like(after))[:examples]
        return before, after
    raise SystemExit("rule family draw failed")


REGISTRY["rule_families"] = _rule_families


def _slot_state(screen):
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
            distance = (ri - ar).abs() + (ci - ac).abs()
            distance = torch.where(mask, distance,
                                   torch.full_like(distance, 999))
            spot = int(distance.reshape(-1).argmin())
            out[row, base], out[row, base + 1] = spot // WIDTH, spot % WIDTH
    return out


VARIANTS = family_variants()


def _grid_domain(name, indices):
    def sample(rng, examples=32):
        for _ in range(60):
            config = VARIANTS[indices[pick(rng, len(indices))]]
            action = pick(rng, 4)
            seed = pick(rng, 10 ** 6)
            verifier = FamilyVerifier(config, batch_size=2 * examples,
                                      seed=seed)
            verifier.reset(seed=seed)
            before = _slot_state(verifier.observation())
            verifier.step(torch.full((2 * examples,), action,
                                     dtype=torch.long))
            after = _slot_state(verifier.observation())
            alive = (before[:, 0] < VALUES) & (after[:, 0] < VALUES)
            before, after = before[alive], after[alive]
            if before.shape[0] < examples:
                continue
            before = torch.where(before < VALUES, before,
                                 torch.zeros_like(before))[:examples]
            after = torch.where(after < VALUES, after,
                                torch.zeros_like(after))[:examples]
            return before, after
        raise SystemExit(f"grid draw failed for {name}")
    REGISTRY[name] = sample


_grid_domain("grid_collect", [0, 1])
_grid_domain("grid_intercept", [2, 3])
_grid_domain("grid_avoid", [4, 5])
_grid_domain("grid_navigate", [6])
_grid_domain("grid_compound", list(range(7, len(VARIANTS))))


def names():
    return list(REGISTRY)
