"""How much depth does a PARALLEL plant need to be faithful?

The sequential plant folded one instruction per residual step and hit
0.9896. A parallel program changes six slots at once, so one step has to
do six writes. Sweep the number of refinement iterations.
"""
import sys, json
import torch
sys.path.insert(0, "/Users/torarinvikbjarko/Documents/Machine Learning Projects/neural-computer-agent-games")
torch.set_num_threads(1)

SLOTS, VALUES = 6, 8
PAR_OPS = ("NOOP", "INC", "DEC", "CINC", "CDEC", "COPY", "SINC", "SDEC")
MODULI = tuple(range(2, VALUES + 1))


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


class Interp(torch.nn.Module):
    """`mode` picks how the program code reaches the residual stack.

    sum   -- one code vector (sum over slots), applied `depth` times
    perslot -- one residual step per slot code, in slot order, which
               costs the same depth as the sequential plant used
    """
    def __init__(self, dim, depth, mode):
        super().__init__()
        self.depth, self.mode = depth, mode
        self.load = torch.nn.Linear(SLOTS * VALUES, dim)
        self.slot = torch.nn.Embedding(SLOTS, dim)
        self.op = torch.nn.Embedding(len(PAR_OPS), dim)
        self.arg_j = torch.nn.Embedding(SLOTS, dim)
        self.arg_m = torch.nn.Embedding(len(MODULI), dim)
        self.step = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * dim, dim))
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, SLOTS * VALUES)

    def code_for(self, s, program):
        op, j, m = program[s]
        return (self.slot(torch.tensor(s)) + self.op(torch.tensor(op))
                + self.arg_j(torch.tensor(j)) + self.arg_m(torch.tensor(m)))

    def forward(self, program, state):
        hot = torch.nn.functional.one_hot(
            state, VALUES).float().view(state.shape[0], -1)
        base = self.load(hot)
        latent = base
        if self.mode == "sum":
            code = sum(self.code_for(s, program) for s in range(SLOTS))
            codes = [code] * self.depth
        else:
            codes = [self.code_for(s, program) for s in range(SLOTS)]
        for code in codes:
            wide = code.unsqueeze(0).expand(latent.shape[0], -1)
            # `base` is passed at every step: parallel semantics means
            # every write reads the PRE-state, so the pre-state has to
            # stay reachable no matter how deep the stack goes.
            latent = self.norm(latent + self.step(
                torch.cat([latent, base, wide], dim=-1)))
        return self.head(latent).view(-1, SLOTS, VALUES)


def random_parallel(g):
    out = []
    for s in range(SLOTS):
        op = int(torch.randint(0, len(PAR_OPS), (1,), generator=g))
        j = int(torch.randint(0, SLOTS, (1,), generator=g))
        if j == s:
            j = (j + 1) % SLOTS
        m = int(torch.randint(0, len(MODULI), (1,), generator=g))
        out.append((op, j, m))
    return out


UPDATES = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
out = []
for mode, depth in (("sum", 1), ("sum", 3), ("sum", 6), ("perslot", 6)):
    torch.manual_seed(69316)
    net = Interp(128, depth, mode)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(69316 * 104729)
    for _ in range(UPDATES):
        prog = random_parallel(g)
        st = torch.randint(0, VALUES, (64, SLOTS), generator=g)
        tgt = run_parallel(st, prog)
        loss = torch.nn.functional.cross_entropy(
            net(prog, st).reshape(-1, VALUES), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    cg = torch.Generator().manual_seed(69316 + 5551)
    hits = rows = 0
    for _ in range(32):
        prog = random_parallel(cg)
        st = torch.randint(0, VALUES, (128, SLOTS), generator=cg)
        tgt = run_parallel(st, prog)
        with torch.no_grad():
            hits += int((net(prog, st).argmax(-1) == tgt).sum())
        rows += tgt.numel()
    out.append({"mode": mode, "depth": depth, "updates": UPDATES,
                "check": round(hits / rows, 4)})
    print(json.dumps(out[-1]), flush=True)
