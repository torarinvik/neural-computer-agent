"""Shared task definitions for the disjoint-dynamics probes.

Four families whose dynamics genuinely differ, so that a transfer result
cannot be explained by one family nesting inside another (F71). Kept in
its own module so `schema_family.py` and `bank_plant.py` measure the
IDENTICAL instrument -- a second copy of these tables would be a silent
way for two probes to disagree about what they tested.

  line    position on a bounded line; actions +/-1, clipped at the ends
  dial    three counters mod 8; actions increment/decrement one, wrapping
  toggle  six bits; actions XOR a fixed mask -- self-inverse, abelian
  perm    an ordering of four items; actions swap adjacent positions

Each family occupies a different NUMBER of slots (1, 3, 4, 6), so the
state vector itself identifies the family; no task-id input is supplied
anywhere and nothing forces sharing.
"""

from __future__ import annotations

import itertools
from collections import deque

import torch

SLOTS, VALUES = 6, 8
WIDTH = SLOTS * VALUES
ACTIONS = 6


# --------------------------------------------------------------- families

def line_states():
    return [(p,) for p in range(8)]


def line_step(state, action):
    delta = -1 if action == 0 else 1
    return (min(7, max(0, state[0] + delta)),)


def dial_states():
    return list(itertools.product(range(8), repeat=3))


def dial_step(state, action):
    which, delta = action // 2, (1 if action % 2 == 0 else -1)
    out = list(state)
    out[which] = (out[which] + delta) % 8
    return tuple(out)


def toggle_states():
    return list(itertools.product(range(2), repeat=6))


# actions 0-4 flip an adjacent PAIR of bits, action 5 flips bit 5 alone.
# Pairs alone preserve parity and would leave half of every goal set
# unreachable; the lone flip makes the group the whole of Z2^6.
TOGGLE_MASKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5,)]


def toggle_step(state, action):
    out = list(state)
    for bit in TOGGLE_MASKS[action]:
        out[bit] = 1 - out[bit]
    return tuple(out)


def perm_states():
    return [tuple(p) for p in itertools.permutations(range(4))]


def perm_step(state, action):
    out = list(state)
    out[action], out[action + 1] = out[action + 1], out[action]
    return tuple(out)


FAMILIES = {
    "line": {"slots": 1, "values": 8, "actions": 2,
             "states": line_states, "step": line_step},
    "dial": {"slots": 3, "values": 8, "actions": 6,
             "states": dial_states, "step": dial_step},
    "toggle": {"slots": 6, "values": 2, "actions": 6,
               "states": toggle_states, "step": toggle_step},
    "perm": {"slots": 4, "values": 4, "actions": 3,
             "states": perm_states, "step": perm_step},
}
# Each family occupies a different NUMBER of slots (1, 3, 4, 6), so the
# state vector itself says which family it is. No task-id input is
# supplied: nothing tags the families, and nothing forces sharing.


class Family:
    """A family with its state list, index maps and true transition table."""

    def __init__(self, name: str, scramble: bool = False, seed: int = 0):
        spec = FAMILIES[name]
        self.name = name
        self.slots = spec["slots"]
        self.actions = spec["actions"]
        self.states = spec["states"]()
        self.index = {s: i for i, s in enumerate(self.states)}
        size = len(self.states)
        if scramble:
            # Same size, same action count, schema destroyed: each
            # (state, action) maps to an unrelated state.
            generator = torch.Generator().manual_seed(
                seed + 7 * abs(hash(name)) % 100_000)
            table = torch.stack([
                torch.randperm(size, generator=generator)
                for _ in range(self.actions)], dim=1)
            self.table = [[int(table[s, a]) for a in range(self.actions)]
                          for s in range(size)]
        else:
            self.table = [[self.index[spec["step"](s, a)]
                           for a in range(self.actions)]
                          for s in self.states]

    def encode(self, indices: torch.Tensor) -> torch.Tensor:
        """Factored one-hot: slot i value v lights index i*VALUES + v.
        Slots this family does not use stay all-zero."""
        out = torch.zeros(indices.shape[0], WIDTH)
        for row, index in enumerate(indices.tolist()):
            for slot, value in enumerate(self.states[index]):
                out[row, slot * VALUES + value] = 1.0
        return out

    def slot_targets(self, indices: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(indices.shape[0], SLOTS, dtype=torch.long)
        for row, index in enumerate(indices.tolist()):
            for slot, value in enumerate(self.states[index]):
                out[row, slot] = value
        return out

    def slot_values(self, indices: torch.Tensor) -> torch.Tensor:
        """Slot values as indices, with VALUES marking an unused slot.
        Unused slots need their own symbol rather than value 0, or a
        family that does not use slot 3 is indistinguishable from one
        holding 0 there."""
        out = torch.full((indices.shape[0], SLOTS), VALUES, dtype=torch.long)
        for row, index in enumerate(indices.tolist()):
            for slot, value in enumerate(self.states[index]):
                out[row, slot] = value
        return out

    def distances(self, goal: int) -> dict:
        field, queue = {goal: 0}, deque([goal])
        back = [[] for _ in self.states]
        for s in range(len(self.states)):
            for a in range(self.actions):
                back[self.table[s][a]].append(s)
        while queue:
            current = queue.popleft()
            for prev in back[current]:
                if prev not in field:
                    field[prev] = field[current] + 1
                    queue.append(prev)
        return field


class DenseModel(torch.nn.Module):
    """The F71/F72 baseline: one flat MLP over the concatenated slots.

    Every slot owns a private stripe of the first and last weight
    matrices, so `copy slot 0 forward` and `copy slot 5 forward` are two
    unrelated facts that must be learned separately. F72 measured the
    consequence: after training on families that use slots 0-2, slot
    accuracy on a family using slots 0-5 sits BELOW the trivial
    copy-forward rule.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(WIDTH + ACTIONS, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, WIDTH))

    def forward(self, values: torch.Tensor, acts: torch.Tensor,
                bank: torch.Tensor | None = None) -> torch.Tensor:
        onehot = torch.zeros(values.shape[0], SLOTS, VALUES)
        live = values < VALUES
        onehot[live] = torch.nn.functional.one_hot(
            values[live], VALUES).float()
        act = torch.zeros(values.shape[0], ACTIONS)
        act.scatter_(1, acts.unsqueeze(-1), 1.0)
        flat = torch.cat([onehot.reshape(values.shape[0], WIDTH), act],
                         dim=-1)
        return self.net(flat).view(-1, SLOTS, VALUES)


class SlotBlock(torch.nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = torch.nn.MultiheadAttention(
            dim, heads, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attention(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class SlotModel(torch.nn.Module):
    """Slot-SYMMETRIC dynamics: one set of weights applied to every slot.

    The state is read as a set of slot tokens rather than one flat
    vector. The value embedding, the per-slot MLP and the output head
    are SHARED across slots, so `copy this slot forward` is a single
    fact about slots in general instead of six unrelated facts. A
    learned positional embedding keeps the slots distinguishable (dial's
    action 0 must move counter 0, not counter 1), and attention supplies
    the cross-slot interaction that perm's swaps need.

    This is the top-down constraint stated as architecture: structure is
    shared by construction, and only family-specific CONTENT has to be
    learned per family. Nothing here is task-specific and no family is
    named anywhere in it.
    """

    def __init__(self, dim: int, heads: int = 4, layers: int = 2,
                 film: bool = False):
        super().__init__()
        self.dim = dim
        # F76 located the binding constraint: prepended tokens are a
        # narrow channel, so a partially correct entry often cannot be
        # repaired at all (2.3 of 16 novel families mastered by reading).
        # With `film` the entry additionally emits a per-family gain and
        # bias for each block, MODULATING the shared computation instead
        # of only being read by it. Every weight is still shared and
        # frozen; the per-family parameters still live in the bank, so
        # the retention guarantee is untouched -- which is the part of
        # this that has to be measured, not assumed.
        self.film = film
        self.value_embed = torch.nn.Embedding(VALUES + 1, dim)
        self.position_embed = torch.nn.Embedding(SLOTS, dim)
        self.action_embed = torch.nn.Embedding(ACTIONS, dim)
        self.blocks = torch.nn.ModuleList(
            [SlotBlock(dim, heads) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, VALUES)
        if film:
            self.film_proj = torch.nn.Linear(dim, 2 * dim * layers)

    def forward(self, values: torch.Tensor, acts: torch.Tensor,
                bank: torch.Tensor | None = None) -> torch.Tensor:
        """`bank` is an external entry: K context tokens prepended to the
        slot tokens and read by attention. The plant's weights hold
        STRUCTURE; a bank entry holds one family's CONTENT. Two entries
        are separate tensors, so learning one cannot overwrite another --
        which is the whole point, and is what weights cannot do (F71,
        F74). With bank=None this is exactly the F73 model."""
        positions = torch.arange(SLOTS).unsqueeze(0)
        x = (self.value_embed(values) + self.position_embed(positions)
             + self.action_embed(acts).unsqueeze(1))
        modulation = None
        if bank is not None:
            context = bank.unsqueeze(0).expand(values.shape[0], -1, -1)
            x = torch.cat([context, x], dim=1)
            if self.film:
                modulation = self.film_proj(bank.mean(0)).view(
                    len(self.blocks), 2, self.dim)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if modulation is not None:
                gain, bias = modulation[index]
                x = x * (1.0 + gain) + bias
        return self.head(self.norm(x[:, -SLOTS:]))


def random_family_spec(generator: torch.Generator, cap: int = 512,
                       wide: bool = False,
                       balanced: bool = False) -> dict:
    """One family drawn from the SCHEMA, not from a list.

    Slots, values per slot, action count and each action's effect are
    sampled; the effects come from the schema vocabulary the four
    hand-made families are themselves instances of -- increment or
    decrement a slot (wrapping or clipped), swap two slots, or do
    nothing. The hand-made families are never generated here, so they
    stay genuinely held out.

    This exists because of what the first bank run measured: a plant
    pre-trained on THREE families learns three modes, not how to read an
    entry, and a fourth entry then has nothing general to plug into.
    Chan et al. is the same result from the other direction -- few
    fixed-meaning classes produce in-weights memorisation, many varied
    ones produce in-context reading. A distribution of families is what
    makes "read the dynamics from the entry" the only strategy that
    works, which is top-down learning stated as a training condition.
    """
    def pick(high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=generator))

    if balanced:
        # F89: wide families are the ones the reader fails, and rejection
        # sampling makes them rare -- accepting uniformly over feasible
        # (slots, values) pairs gives 6-slot families only 1/27 of the
        # pool. Choosing SLOTS first and then a feasible value count
        # makes each width equally common. Every gain in this project has
        # come from the training distribution rather than from capacity
        # (F78 diversity, F80 budget), and F77/F89 showed capacity
        # actively hurting, so distribution is where to push.
        slots = 1 + pick(SLOTS)
        feasible = [v for v in range(2, VALUES + 1) if v ** slots <= cap]
        values = feasible[pick(len(feasible))]
    else:
        while True:
            slots = 1 + pick(SLOTS)
            values = 2 + pick(VALUES - 1)
            if values ** slots <= cap:
                break
    actions = 2 + pick(ACTIONS - 1)
    # F79 measured a hard floor the schema itself caused: `toggle` read
    # at 0.096 at EVERY pool size, because it flips a PAIR of slots at
    # once and no op in this vocabulary does that, and `perm` read at
    # 0.729 at best because its states are permutations rather than a
    # product space. Diversity within a schema buys nothing outside it,
    # so the schema is what has to widen. `pair` covers simultaneous
    # two-slot effects (for values=2 that is exactly an XOR mask), and
    # the `perm` space covers non-product state spaces.
    space = "product"
    if wide and slots >= 2 and values == slots and pick(3) == 0:
        space = "perm"
    kinds = ["inc", "dec", "cinc", "cdec", "noop"]
    if slots >= 2:
        kinds.append("swap")
    if wide and slots >= 2:
        kinds.append("pair")
    if space == "perm":
        kinds = ["swap", "noop"]
    ops = []
    for _ in range(actions):
        kind = kinds[pick(len(kinds))]
        if kind in ("swap", "pair"):
            first = pick(slots)
            second = (first + 1 + pick(max(slots - 1, 1))) % slots
            ops.append((kind, first, second))
        else:
            ops.append((kind, pick(slots), 0))
    return {"slots": slots, "values": values, "actions": actions,
            "ops": ops, "space": space}


def apply_op(state: tuple, op: tuple, values: int) -> tuple:
    kind = op[0]
    out = list(state)
    if kind == "noop":
        return tuple(out)
    if kind == "swap":
        _, first, second = op
        out[first], out[second] = out[second], out[first]
        return tuple(out)
    if kind == "pair":
        # both slots advance at once; at values=2 this is an XOR mask,
        # which is exactly what `toggle` does and what the original
        # vocabulary could not express
        _, first, second = op
        out[first] = (out[first] + 1) % values
        out[second] = (out[second] + 1) % values
        return tuple(out)
    slot = op[1]
    if kind == "inc":
        out[slot] = (out[slot] + 1) % values
    elif kind == "dec":
        out[slot] = (out[slot] - 1) % values
    elif kind == "cinc":
        out[slot] = min(values - 1, out[slot] + 1)
    elif kind == "cdec":
        out[slot] = max(0, out[slot] - 1)
    return tuple(out)


class RandomFamily(Family):
    """A `Family` built from a sampled spec rather than the fixed four."""

    def __init__(self, spec: dict, name: str = "random"):
        self.name = name
        self.spec = spec
        self.slots = spec["slots"]
        self.actions = spec["actions"]
        if spec.get("space") == "perm":
            self.states = [tuple(p) for p in
                           itertools.permutations(range(spec["values"]))]
        else:
            self.states = [tuple(s) for s in itertools.product(
                range(spec["values"]), repeat=spec["slots"])]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.table = [[self.index[apply_op(state, op, spec["values"])]
                       for op in spec["ops"]]
                      for state in self.states]


# ---- The project's OWN reacher, expressed in the slot interface ----
# F71-F91 are all measured on procedurally generated families. The
# reacher ladder is a task this project actually built and measured
# (F67-F70), and its grid state is exactly two slots of eight values, so
# it can be read by the same plant with no new machinery.
#
# `walled` matters more than `grid`: its dynamics are POSITION-DEPENDENT
# -- whether an action moves you depends on where the obstacle is, not
# on any function of the slot values. Nothing the generator can produce
# has that property, because every generated op is a uniform function of
# slot values. So `walled` is outside the generator's support in a way
# `toggle` and `perm` never were.

GRID_MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def grid_states():
    return [(r, c) for r in range(8) for c in range(8)]


def grid_step(state, action):
    dr, dc = GRID_MOVES[action]
    row, col = state[0] + dr, state[1] + dc
    if not (0 <= row < 8 and 0 <= col < 8):
        return state
    return (row, col)


# the reacher's own obstacle: one column with a gap at the top row
WALLS = {(r, 4) for r in range(1, 8)}


def walled_step(state, action):
    nxt = grid_step(state, action)
    return state if nxt in WALLS else nxt


FAMILIES["grid"] = {"slots": 2, "values": 8, "actions": 4,
                    "states": grid_states, "step": grid_step}
FAMILIES["walled"] = {"slots": 2, "values": 8, "actions": 4,
                      "states": grid_states, "step": walled_step}
