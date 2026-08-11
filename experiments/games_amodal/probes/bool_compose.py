"""Boolean composition: the puzzle-piece question at minimum difficulty.

F117 asked whether bank entries factor into reusable pieces, using
x -> a*x+b over Z_23, and returned a clean null — but the diagnostics
said FIT failed before reading did (0.10-0.13 even on trained worlds
with trained programs). Composed modular multiplication is close to the
hardest thing a small model can be asked to represent, so that null
indicts the arithmetic, not the mechanism.

This is the same experiment with pieces that are trivial to represent
individually, so anything that fails is the COMPOSITION or the READING,
never the piece:

    f(x) = x XOR b          (hidden mask b)
    g(x) = rotate_left(x, k) (hidden shift k)

over W-bit vectors. Both pieces are one-step-learnable. They do NOT
commute — rot(x XOR b) != rot(x) XOR b — so program ORDER matters and
held-out programs are genuinely unseen functions, not relabellings.
Prediction is per-bit, so the output head is W binary decisions rather
than a 23-way softmax over a space the model must first learn to
represent.

World diversity is no longer the constraint either: (2^W - 1) * (W - 1)
distinct worlds, 1785 at W=8, against the 15 that mod-16 admitted in
F115.

Everything else is the established recipe: reader sees SINGLE
applications only (composition is never demonstrated), ignorance
objective, worlds and programs both split, and the four controls —
own entry, withheld, stranger, and the swap twin (f/g roles exchanged),
whose truth a correctly-factored entry should MISS.

Reported metrics: exact-match (all W bits right, chance 2^-W) is the
headline; per-bit accuracy (chance 0.5) shows partial credit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--width", type=int, default=8, help="bits per state")
parser.add_argument("--dim", type=int, default=96)
parser.add_argument("--bank-tokens", type=int, default=8)
parser.add_argument("--examples", type=int, default=12)
parser.add_argument("--train-updates", type=int, default=12000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--ignorance", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--worlds", type=int, default=64)
parser.add_argument("--held-worlds", type=int, default=8)
parser.add_argument("--max-len", type=int, default=4)
parser.add_argument(
    "--curriculum", type=float, default=0.0,
    help="ramp the maximum program LENGTH from 1 to --max-len over "
         "this fraction of training. Measured motivation: only 11%% of "
         "updates land on length-1 programs and 67%% on length>=3, so "
         "the model spends almost all its time on the hardest form of "
         "a task it cannot yet do at all. F114 showed reading works "
         "when the plant's job is a SINGLE application; F120 showed "
         "the ignorance objective is toothless while the model is bad. "
         "A curriculum gets reading established on the readable task "
         "first, then extends it — the bootstrapping F120 identified.")
parser.add_argument(
    "--refine", type=int, default=0,
    help="SEMI-AMORTIZATION (Kim et al. 2018), ranked first in the "
         "LITERATURE.md addendum. Take the reader's entry as an "
         "INITIALISATION and run this many gradient steps on the ENTRY "
         "ITSELF, against the same example rows the reader already "
         "saw, through the frozen plant. F138 measured the "
         "approximation gap closed (the reader CAN represent the "
         "needed entry, 0.9723 when distilled) and everything since is "
         "the amortization gap: one forward pass failing to find what "
         "the family can express. Refinement attacks that quantity "
         "directly. NOTE this does not break the architecture: the "
         "gradient moves the ENTRY, which is data in an external "
         "store, and never the plant's weights. No privileged "
         "information is used, only the learner's own observations.")
parser.add_argument(
    "--refine-lr", type=float, default=0.05)
parser.add_argument(
    "--codebook", type=int, default=0,
    help="quantise the entry to the nearest of K learned codes "
         "(VQ-VAE style, straight-through gradient + commitment loss). "
         "Ranked first in docs/LITERATURE.md: a hard discrete "
         "assignment removes the plant's escape route, which has been "
         "AVERAGING since F106 — with no relaxation available the "
         "decoder cannot bypass the bottleneck. It also makes the bank "
         "and the mechanism the same object (a codebook IS a finite "
         "set of entries) and turns binding into a lookup, which F140 "
         "says is the right direction since the binder must stay "
         "simple. The risk to MEASURE not assume: K caps the number of "
         "distinguishable worlds, which collides with the diversity "
         "law (F78, F144) — sweep K against world count.")
parser.add_argument(
    "--commit", type=float, default=0.25,
    help="weight of the VQ commitment term keeping the reader's "
         "output near the code it selected.")
parser.add_argument(
    "--reader-steps", type=int, default=1,
    help="reader updates per plant update, both still learning. "
         "Ranked second in docs/LITERATURE.md: He et al. (ICLR 2019) "
         "diagnose posterior collapse as the inference network failing "
         "to keep up with a moving posterior, and fix it by training "
         "the encoder aggressively. F136 took that to the extreme of "
         "FREEZING the plant and lost to joint training; this is the "
         "interleaved form the literature actually endorses.")
parser.add_argument(
    "--contrastive-batch", type=int, default=8,
    help="number of worlds the contrastive term must tell apart at "
         "once. F142 reached 0.7069 with 8, where the reader need only "
         "distinguish one world from seven — a coarse code suffices. "
         "A larger batch demands a finer code, which is F78's "
         "diversity law applied to the READER's objective rather than "
         "the plant's data: make the easy solution unrepresentable and "
         "the wanted one becomes the only option.")
parser.add_argument(
    "--contrastive-aux", type=float, default=0.0,
    help="contrastive term as an AUXILIARY loss during joint training "
         "rather than a frozen pre-training phase. F139 showed a "
         "contrastive code is discriminative but not shaped for the "
         "binder; F140 showed the binder cannot be given capacity to "
         "compensate (a nonlinear binder drops the oracle ceiling from "
         "0.9983 to 0.6196). So the code must be made task-shaped "
         "WHILE it is made discriminative, not before: the task loss "
         "supplies the shape, the contrastive term supplies the "
         "gradient that breaks F106's deadlock, and neither waits for "
         "the other.")
parser.add_argument(
    "--deep-binder", action="store_true",
    help="make the entry->parameters decoder an MLP instead of a "
         "single linear map. F135's oracle entry was itself a LINEAR "
         "projection of one-hot world parameters, so a linear binder "
         "inverted it trivially — that ceiling may have been partly an "
         "artefact of matching encoders. A contrastively-learned code "
         "(F139) identifies the world but in an arbitrary, likely "
         "nonlinear arrangement, and a linear binder cannot decode it. "
         "This changes only the binder, so it is attributable.")
parser.add_argument(
    "--contrastive", type=float, default=0.0,
    help="NON-PRIVILEGED reader pre-training. For this fraction of "
         "updates, train the reader alone so that two entries read "
         "from DIFFERENT observation samples of the SAME world agree, "
         "and entries from different worlds do not (InfoNCE over a "
         "batch of worlds). Then freeze the reader and train the plant "
         "to bind whatever code it settled on. F138 showed the reader "
         "can produce a usable entry when given a consistent target, "
         "and F136 showed task loss through a frozen plant cannot find "
         "one; the property distillation actually supplied was "
         "CONSISTENCY, which needs no privileged parameters — a "
         "learner always knows which observations came from the same "
         "world. Phase order is reversed from --two-phase: reader "
         "first, plant second.")
parser.add_argument(
    "--distill", action="store_true",
    help="in phase 2, train the reader to MATCH the oracle entry "
         "directly (squared error on the entry vector) instead of "
         "through task loss. Diagnostic, not a proposed mechanism: "
         "the oracle entry is built from privileged parameters, so a "
         "reader trained this way is not a solution. What it answers "
         "is whether the reader CAN represent the required entry from "
         "observations at all. F136 showed phase-2 task loss fails to "
         "move it; if distillation also fails, the reader or its "
         "inputs are inadequate, and if it succeeds, the reader is "
         "capable and the missing piece is purely the training "
         "signal. Requires --two-phase.")
parser.add_argument(
    "--two-phase", type=float, default=0.0,
    help="F75-F79's FROZEN PLANT + AMORTISED READING, applied here. "
         "Train the plant on ORACLE entries for this fraction of "
         "updates, then FREEZE it and train only the reader through "
         "it. F135 measured the plant's side solved (0.9983 with "
         "bound oracle entries) and the reader's side dead (own == "
         "stranger to four decimals) — the classic F106 deadlock, "
         "where a bad reader gives the plant no reason to use entries "
         "and an entry-ignoring plant gives the reader no gradient. "
         "Phase 1 breaks it by building a plant that DEMANDS a "
         "well-formed entry; phase 2 then has a fixed target to aim "
         "at. Requires --iterate --bind-params.")
parser.add_argument(
    "--bind-params", action="store_true",
    help="decode the entry ONCE into one explicit parameter vector "
         "per piece token, then step on (latent, bound parameter) with "
         "no further access to the entry. F134 measured that a plant "
         "given the world exactly still fails above depth 1 (0.5587 "
         "per-bit vs 1.0000 at one world) while the entry is "
         "re-attended at every step — so the same parameters are "
         "re-extracted on each application and any extraction error "
         "compounds with depth. An interpreter binds its arguments "
         "once and then runs the loop; this does the same. Requires "
         "--iterate.")
parser.add_argument(
    "--oracle-entry", action="store_true",
    help="ORACLE SUBSTITUTION on the entry (the F110 technique that "
         "settled the games): replace the reader's output with the "
         "world's TRUE hidden parameters, projected to entry shape. "
         "F120/F122 measured reading to be entirely absent at "
         "multi-world scale (stranger entry bit-identical to own) and "
         "F121 fixed composition only where reading was not needed. "
         "This separates the two: if oracle entries make multi-world "
         "composition work, execution and composition are sound and "
         "READING alone is the constraint; if they do not, the "
         "interface still cannot use per-world content at all.")
parser.add_argument(
    "--train-max-len", type=int, default=0,
    help="LENGTH EXTRAPOLATION split: train on every program of length "
         "<= L and hold out every LONGER one. The default split holds "
         "out half of the length-3/4 programs but trains on the other "
         "half, so both lengths are represented in training and a model "
         "could pass by interpolating within a length. This split "
         "cannot be passed that way: a length-4 program is only "
         "answerable by applying a piece one more time than was ever "
         "demonstrated, which is the sharpest statement of the "
         "puzzle-piece claim.")
parser.add_argument(
    "--iterate", action="store_true",
    help="apply the program ONE PIECE AT A TIME through a shared step "
         "function over a recurrent latent, decoding only at the end. "
         "F119 measured the one-shot interface fitting trained programs "
         "at 1.0000 while sitting at chance on unseen ARRANGEMENTS of "
         "the same pieces — it memorises composite functions instead of "
         "composing. This makes composition structural: same blocks, "
         "same parameter count, no intermediate supervision.")
parser.add_argument("--json", default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
W = args.width
F, G = 0, 1


def bits_of(values: torch.Tensor) -> torch.Tensor:
    """Integer tensor -> (n, W) float bit matrix, bit 0 = least sig."""
    shifts = torch.arange(W)
    return ((values.unsqueeze(-1) >> shifts) & 1).float()


def make_worlds() -> list[dict]:
    # The draw loop below rejects duplicates, so asking for more worlds
    # than the family CONTAINS spins forever rather than failing. At
    # width 4 there are only 15 masks x 3 shifts = 45 worlds, and the
    # default request is 64 — one such run held a core at 100% for 17
    # hours without reaching training. Fail loudly instead.
    available = ((1 << W) - 1) * (W - 1)
    if args.worlds > available:
        raise SystemExit(
            f"--worlds {args.worlds} exceeds the {available} distinct "
            f"(mask, shift) worlds available at --width {W}; "
            f"use --worlds <= {available} or a larger width")
    generator = torch.Generator().manual_seed(args.seed * 7919)
    seen, out = set(), []
    while len(out) < args.worlds:
        b = int(torch.randint(1, 1 << W, (1,), generator=generator))
        k = int(torch.randint(1, W, (1,), generator=generator))
        if (b, k) in seen:
            continue
        seen.add((b, k))
        out.append({"name": f"b{b}k{k}", "b": b, "k": k})
    return out


def apply_piece(world: dict, token: int, x: torch.Tensor,
                swapped: bool) -> torch.Tensor:
    is_f = (token == F) != swapped
    if is_f:
        return x ^ world["b"]
    k, mask = world["k"], (1 << W) - 1
    return ((x << k) | (x >> (W - k))) & mask


def apply_program(world: dict, program: tuple, x: torch.Tensor,
                  swapped: bool = False) -> torch.Tensor:
    result = x.clone()
    for token in reversed(program):
        result = apply_piece(world, token, result, swapped)
    return result


def all_programs() -> tuple[list, list]:
    generator = torch.Generator().manual_seed(args.seed * 104729)
    short, long = [], []
    for length in range(1, args.max_len + 1):
        for program in itertools.product((F, G), repeat=length):
            (short if length <= 2 else long).append(program)
    if args.train_max_len:
        short_by_len, long_by_len = [], []
        for length in range(1, args.max_len + 1):
            for program in itertools.product((F, G), repeat=length):
                (short_by_len if length <= args.train_max_len
                 else long_by_len).append(program)
        return short_by_len, long_by_len
    if not long:
        # --max-len 1: nothing to hold out on the PROGRAM axis. Reuse
        # the trained programs so the held-out WORLD axis (the reading
        # test) still reports, rather than dividing by zero.
        return short, short
    order = torch.randperm(len(long), generator=generator).tolist()
    half = len(long) // 2
    return (short + [long[i] for i in order[half:]],
            [long[i] for i in order[:half]])


class Reader(torch.nn.Module):
    """Rows of (piece token, x bits, piece(x) bits) -> entry."""

    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.piece = torch.nn.Embedding(2, dim)
        self.row = torch.nn.Linear(2 * W, dim)
        self.queries = torch.nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(2)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, pieces, x, y) -> torch.Tensor:
        rows = self.row(torch.cat([bits_of(x), bits_of(y)], dim=-1)) \
            + self.piece(pieces)
        token = torch.cat([self.queries, rows], dim=0).unsqueeze(0)
        for block in self.blocks:
            token = block(token)
        return self.norm(token[0, :self.queries.shape[0]])


class Plant(torch.nn.Module):
    """(program, x, entry) -> W bit logits."""

    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.value = torch.nn.Linear(W, dim)
        self.piece = torch.nn.Embedding(2, dim)
        self.position = torch.nn.Embedding(max_len + 1, dim)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim, 4, dim_feedforward=2 * dim, batch_first=True,
                dropout=0.0, norm_first=True) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, W)
        self.binder = (torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim), torch.nn.ReLU(),
            torch.nn.Linear(4 * dim, 4 * dim), torch.nn.ReLU(),
            torch.nn.Linear(4 * dim, 2 * dim))
            if args.deep_binder else torch.nn.Linear(dim, 2 * dim))
        self.apply_bound = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))

    def bind(self, entry) -> torch.Tensor:
        """entry -> (2, dim): one bound parameter vector per piece."""
        return self.binder(entry.mean(dim=0)).view(2, -1)

    def step_bound(self, token: int, hidden, params) -> torch.Tensor:
        """One piece applied using its ALREADY-BOUND parameter. The
        entry is not consulted here — that is the point."""
        return self.apply_bound(torch.cat(
            [hidden, params[token].unsqueeze(0).expand(
                hidden.shape[0], -1)], dim=-1))

    def step(self, token: int, hidden, entry):
        """One piece applied to the latent. Shared across positions and
        program lengths — that sharing IS the compositional prior."""
        row = (hidden + self.piece(torch.tensor(token))).unsqueeze(1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            row = torch.cat([context, row], dim=1)
        for block in self.blocks:
            row = block(row)
        return self.norm(row[:, -1])

    def forward(self, program: tuple, x, entry) -> torch.Tensor:
        if args.iterate and args.bind_params and entry is not None:
            params = self.bind(entry)
            hidden = self.value(bits_of(x))
            for token in reversed(program):
                hidden = self.step_bound(token, hidden, params)
            return self.head(self.norm(hidden))
        if args.iterate:
            hidden = self.value(bits_of(x))
            for token in reversed(program):
                hidden = self.step(token, hidden, entry)
            return self.head(hidden)
        batch, length = x.shape[0], len(program)
        tokens = (self.piece(torch.tensor(program))
                  + self.position(torch.arange(length)))
        tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
        query = (self.value(bits_of(x))
                 + self.position(torch.full((batch,), length))).unsqueeze(1)
        row = torch.cat([tokens, query], dim=1)
        if entry is not None:
            context = entry.unsqueeze(0).expand(batch, -1, -1)
            row = torch.cat([context, row], dim=1)
        for block in self.blocks:
            row = block(row)
        return self.head(self.norm(row[:, -1]))


class OracleEntry(torch.nn.Module):
    """Ground-truth world parameters projected into entry shape. Not a
    learner — a substitution, legitimate for an ablation only."""

    def __init__(self, dim: int, tokens: int, width: int):
        super().__init__()
        self.project = torch.nn.Linear(width, dim * tokens)
        self.tokens, self.dim = tokens, dim

    def forward(self, raw) -> torch.Tensor:
        return self.project(raw).view(self.tokens, self.dim)


class Codebook(torch.nn.Module):
    """K learned entries; the reader selects one. Straight-through so
    the reader still gets a gradient through a hard assignment."""

    def __init__(self, count: int, tokens: int, dim: int):
        super().__init__()
        self.codes = torch.nn.Parameter(
            torch.randn(count, tokens * dim) * 0.5)
        self.tokens, self.dim = tokens, dim
        self.last_loss = torch.zeros(())
        self.last_index = -1
        # F146: without this the codebook collapses to ONE code — at
        # initialisation every reader output is similar, one code wins
        # every assignment, and the losers never receive gradient. Dead
        # codes are periodically re-seeded onto recent reader outputs,
        # the standard remedy.
        self.register_buffer("usage", torch.zeros(count))
        self.recent: list = []

    def restart_dead(self, generator) -> int:
        dead = (self.usage == 0).nonzero().flatten()
        if len(dead) == 0 or len(self.recent) < 2:
            self.usage.zero_()
            return 0
        pool = torch.stack(self.recent)
        with torch.no_grad():
            for slot in dead.tolist():
                pick = int(torch.randint(0, pool.shape[0], (1,),
                                         generator=generator))
                jitter = torch.randn(
                    pool.shape[1], generator=generator) * 0.01
                self.codes[slot] = pool[pick] + jitter
        self.usage.zero_()
        return len(dead)

    def forward(self, entry: torch.Tensor) -> torch.Tensor:
        flat = entry.flatten()
        distance = ((self.codes - flat.unsqueeze(0)) ** 2).sum(-1)
        index = int(distance.argmin())
        chosen = self.codes[index]
        self.last_index = index
        self.usage[index] += 1
        self.recent.append(flat.detach().clone())
        if len(self.recent) > 256:
            self.recent.pop(0)
        self.last_loss = (((chosen - flat.detach()) ** 2).mean()
                          + args.commit
                          * ((flat - chosen.detach()) ** 2).mean())
        # straight-through: forward uses the code, backward reaches the
        # reader as if the code were its own output
        passed = flat + (chosen - flat).detach()
        return passed.view(self.tokens, self.dim)


reader = Reader(args.dim, args.bank_tokens)
plant = Plant(args.dim, args.max_len)
codebook = (Codebook(args.codebook, args.bank_tokens, args.dim)
            if args.codebook else None)


def task_loss(program, x, y, entry) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        plant(program, x, entry), y)
# b as exact bits (already unambiguous) + k ONE-HOT rather than k/W:
# a scalar shift would make all worlds sharing b collinear.
ORACLE_WIDTH = W + W
oracle = OracleEntry(args.dim, args.bank_tokens, ORACLE_WIDTH)
optimizer = torch.optim.Adam(
    list(reader.parameters()) + list(plant.parameters())
    + list(oracle.parameters())
    + (list(codebook.parameters()) if codebook is not None else []),
    lr=args.lr)
reader_opt = torch.optim.Adam(reader.parameters(), lr=args.lr)


def oracle_raw(world: dict) -> torch.Tensor:
    shift = torch.zeros(1, W)
    shift[0, world["k"]] = 1.0
    return torch.cat([bits_of(torch.tensor([world["b"]])), shift], dim=-1)

worlds = make_worlds()
select = torch.Generator().manual_seed(args.seed * 15485863)
held_index = set(torch.randperm(
    len(worlds), generator=select)[:args.held_worlds].tolist())
train_worlds = [w for i, w in enumerate(worlds) if i not in held_index]
held_worlds = [w for i, w in enumerate(worlds) if i in held_index]
train_programs, held_programs = all_programs()


def reader_examples(world: dict, generator: torch.Generator):
    pieces = torch.randint(0, 2, (args.examples,), generator=generator)
    x = torch.randint(0, 1 << W, (args.examples,), generator=generator)
    y = torch.where(pieces == F,
                    apply_piece(world, F, x, False),
                    apply_piece(world, G, x, False))
    return pieces, x, y


data_gen = torch.Generator().manual_seed(args.seed * 6700417)
phase_one = int(args.train_updates * args.two_phase)
contrast_end = int(args.train_updates * args.contrastive)


def contrastive_loss(batch_worlds: list) -> torch.Tensor:
    """InfoNCE: an entry must match a SECOND reading of its own world
    better than any other world's. Uses only world identity, which the
    learner observes directly."""
    anchors = torch.stack([
        reader(*reader_examples(w, data_gen)).flatten()
        for w in batch_worlds])
    others = torch.stack([
        reader(*reader_examples(w, data_gen)).flatten()
        for w in batch_worlds])
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    others = torch.nn.functional.normalize(others, dim=-1)
    logits = anchors @ others.T / 0.1
    target = torch.arange(len(batch_worlds))
    return torch.nn.functional.cross_entropy(logits, target)
for update in range(args.train_updates):
    if args.two_phase > 0 and update == phase_one:
        # freeze the plant; from here only the reader learns, and it
        # must produce entries the FIXED plant can already bind
        for parameter in plant.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)
    if args.contrastive > 0 and update < contrast_end:
        picks = torch.randperm(
            len(train_worlds),
            generator=data_gen)[:args.contrastive_batch].tolist()
        loss = contrastive_loss([train_worlds[i] for i in picks])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        continue
    if args.contrastive > 0 and update == contrast_end:
        for parameter in reader.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(plant.parameters(), lr=args.lr)
    world = train_worlds[int(torch.randint(
        0, len(train_worlds), (1,), generator=data_gen))]
    pool = train_programs
    if args.curriculum > 0:
        ramp = update / max(1.0, args.train_updates * args.curriculum)
        cap = min(args.max_len, 1 + int(ramp * args.max_len))
        pool = [p for p in train_programs if len(p) <= cap] or train_programs
    program = pool[int(torch.randint(
        0, len(pool), (1,), generator=data_gen))]
    use_oracle = args.oracle_entry or (
        args.two_phase > 0 and update < phase_one)
    entry = (oracle(oracle_raw(world)) if use_oracle
             else reader(*reader_examples(world, data_gen)))
    if codebook is not None and not use_oracle:
        entry = codebook(entry)
    x = torch.randint(0, 1 << W, (args.batch_size,), generator=data_gen)
    y = bits_of(apply_program(world, program, x))
    if args.distill and args.two_phase > 0 and update >= phase_one:
        loss = torch.nn.functional.mse_loss(
            entry, oracle(oracle_raw(world)).detach())
    else:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            plant(program, x, entry), y)
    if args.ignorance > 0 and not (
            args.two_phase > 0 and update >= phase_one):
        blind = plant(program, x, torch.zeros_like(entry))
        # per-bit entropy, maximised at log 2 when the entry-free
        # prediction is a coin flip on every bit
        p = torch.sigmoid(blind).clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        loss = loss + args.ignorance * (math.log(2) - entropy)
    if codebook is not None and not use_oracle:
        loss = loss + codebook.last_loss
        if update > 0 and update % 500 == 0:
            codebook.restart_dead(data_gen)
    if args.contrastive_aux > 0:
        picks = torch.randperm(
            len(train_worlds),
            generator=data_gen)[:args.contrastive_batch].tolist()
        loss = loss + args.contrastive_aux * contrastive_loss(
            [train_worlds[i] for i in picks])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for _ in range(args.reader_steps - 1):
        extra = reader(*reader_examples(world, data_gen))
        if codebook is not None:
            extra = codebook(extra)
        step_loss = task_loss(program, x, y, extra)
        if codebook is not None:
            step_loss = step_loss + codebook.last_loss
        reader_opt.zero_grad()
        step_loss.backward()
        reader_opt.step()


def entry_of(world: dict, offset: int = 0) -> torch.Tensor:
    if args.oracle_entry:
        with torch.no_grad():
            return oracle(oracle_raw(world))
    generator = torch.Generator().manual_seed(
        args.seed * 31 + hash(world["name"]) % 100000 + offset)
    pieces, xs_all, ys_all = reader_examples(world, generator)
    with torch.no_grad():
        read = reader(pieces, xs_all, ys_all)
        read = codebook(read) if codebook is not None else read
    if args.refine <= 0:
        return read
    # semi-amortisation: polish the ENTRY against the rows the reader
    # already saw. The plant is frozen; only this tensor moves.
    working = read.clone().detach().requires_grad_(True)
    refiner = torch.optim.Adam([working], lr=args.refine_lr)
    for _ in range(args.refine):
        total = torch.zeros(())
        for token in (F, G):
            keep = (pieces == token)
            if not bool(keep.any()):
                continue
            xs, ys = xs_all[keep], ys_all[keep]
            total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                plant((token,), xs, working), bits_of(ys))
        refiner.zero_grad()
        total.backward()
        refiner.step()
    return working.detach()


stranger_gen = torch.Generator().manual_seed(args.seed * 32452843)


def stranger_entry() -> torch.Tensor:
    b = int(torch.randint(1, 1 << W, (1,), generator=stranger_gen))
    k = int(torch.randint(1, W, (1,), generator=stranger_gen))
    return entry_of({"name": f"s{b}_{k}", "b": b, "k": k}, offset=9)


def accuracy(world: dict, programs: list, entry,
             swapped: bool = False) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(args.seed * 977)
    exact, bits, total = 0, 0.0, 0
    with torch.no_grad():
        for program in programs:
            x = torch.randint(0, 1 << W, (64,), generator=generator)
            y = bits_of(apply_program(world, program, x, swapped))
            predictions = (plant(program, x, entry) > 0).float()
            match = (predictions == y)
            exact += int(match.all(dim=-1).sum())
            bits += float(match.float().mean()) * 64
            total += 64
    return round(exact / total, 4), round(bits / total, 4)


def score_world(world: dict) -> dict:
    entry = entry_of(world)
    own_train = accuracy(world, train_programs, entry)
    own_held = accuracy(world, held_programs, entry)
    withheld = accuracy(world, held_programs, None)
    stranger = accuracy(world, held_programs, stranger_entry())
    swap = accuracy(world, held_programs, entry, swapped=True)
    return {"trained_programs": own_train[0],
            "held_programs": own_held[0],
            "withheld": withheld[0],
            "stranger": stranger[0],
            "swap_truth_with_own_entry": swap[0],
            "bits_trained_programs": own_train[1],
            "bits_held_programs": own_held[1],
            "bits_withheld": withheld[1],
            "bits_stranger": stranger[1]}


report = {
    "seed": args.seed, "width": W, "worlds": args.worlds,
    "held_worlds": args.held_worlds,
    "train_programs": len(train_programs),
    "held_programs": len(held_programs),
    "chance_exact": round(2.0 ** -W, 6), "chance_bit": 0.5,
    "held_out_worlds": {w["name"]: score_world(w) for w in held_worlds},
    "trained_worlds": {w["name"]: score_world(w)
                       for w in train_worlds[:args.held_worlds]},
}
print(json.dumps(report, indent=2))
if args.json:
    with open(args.json, "w") as handle:
        json.dump(report, handle, indent=2)
