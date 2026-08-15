import torch
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.noise_tolerant_induction import induce_noise_tolerant
from experiments.brainworkshop_canonical.rule_automata import sample_rule

def traces(pred, seed, n, length, noise=0.0, cnt=4):
    g = torch.Generator().manual_seed(seed); out=[]
    for _ in range(n):
        s = torch.randint(0, cnt, (length,), generator=g).tolist()
        y = pred(s)
        if noise > 0:
            f = (torch.rand(length, generator=g) < noise).tolist()
            y = [v ^ int(b) for v, b in zip(y, f)]
        out.append(Trace(tuple(s), tuple(y), tuple([True]*length), cnt))
    return tuple(out)

def acc(m, pred, seed, cnt=4):
    clean = traces(pred, seed, 20, 48, 0.0, cnt)
    hit = tot = 0
    for tr in clean:
        p = m.expected(list(tr.symbols))
        for i, f in enumerate(tr.eligible):
            if f:
                tot += 1; hit += int(p[i] == tr.outputs[i])
    return hit/tot

rules = []
for st in (1,2,3,4,5,6):
    for i in range(4):
        r = sample_rule(symbol_count=4, state_count=st, seed=7000+100*st+i)
        if r: rules.append((st, i, r))
print(f"{len(rules)} sampled Mealy rules; held-out accuracy against the CLEAN rule")
print("noise |  exact-recovery  mean-acc  min-acc  states-correct")
for noise in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
    accs=[]; exact=0; right_states=0
    for st, i, r in rules:
        fit = traces(r.expected, 11, 112, 48, noise)
        f = induce_noise_tolerant(fit)
        if f is None: accs.append(0.0); continue
        a = acc(f.machine, r.expected, 99)
        accs.append(a); exact += int(a == 1.0)
        right_states += int(f.machine.state_count == st)
    print(f"{noise:<5} |  {exact:>2}/{len(rules)}          {sum(accs)/len(accs):.4f}   {min(accs):.4f}   {right_states}/{len(rules)}")

# Structureless control: must not claim anything.
print()
g = torch.Generator().manual_seed(5); rnd=[]
for _ in range(112):
    s = torch.randint(0,4,(48,),generator=g).tolist()
    y = torch.randint(0,2,(48,),generator=g).tolist()
    rnd.append(Trace(tuple(s),tuple(y),tuple([True]*48),4))
f = induce_noise_tolerant(tuple(rnd))
print(f"random labels: states={f.machine.state_count} fit-err={f.error_rate:.4f} (chance is 0.5)")
