import torch
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.noise_tolerant_induction import induce_noise_tolerant
from experiments.brainworkshop_canonical.rule_automata import sample_rule

def traces(pred, seed, n, length, noise=0.0, cnt=4):
    g = torch.Generator().manual_seed(seed); out=[]
    for _ in range(n):
        s = torch.randint(0, cnt, (length,), generator=g).tolist(); y = pred(s)
        if noise > 0:
            f = (torch.rand(length, generator=g) < noise).tolist()
            y = [v ^ int(b) for v, b in zip(y, f)]
        out.append(Trace(tuple(s), tuple(y), tuple([True]*length), cnt))
    return tuple(out)

def acc(m, pred, seed, cnt=4):
    hit=tot=0
    for tr in traces(pred, seed, 20, 48, 0.0, cnt):
        p = m.expected(list(tr.symbols))
        for i,f in enumerate(tr.eligible):
            if f: tot+=1; hit+=int(p[i]==tr.outputs[i])
    return hit/tot

rules=[]
for st in (1,2,3,4,5,6):
    for i in range(4):
        r=sample_rule(symbol_count=4,state_count=st,seed=7000+100*st+i)
        if r: rules.append((st,r))

print("A) search-seed robustness at 10% noise")
for s in (0,1,2,3):
    ex=sum(int(acc(induce_noise_tolerant(traces(r.expected,11,112,48,0.10),seed=s).machine,r.expected,99)==1.0) for _,r in rules)
    print(f"   search seed {s}: {ex}/{len(rules)} exact")

print()
print("B) data budget at 10% noise (episodes x 48 steps)")
for n in (7,14,28,56,112):
    ex=sum(int(acc(induce_noise_tolerant(traces(r.expected,11,n,48,0.10)).machine,r.expected,99)==1.0) for _,r in rules)
    print(f"   {n:>3} episodes ({n*48:>5} labels): {ex}/{len(rules)} exact")

print()
print("C) data-seed robustness at 10% noise")
for ds in (11,23,37):
    ex=sum(int(acc(induce_noise_tolerant(traces(r.expected,ds,112,48,0.10)).machine,r.expected,99)==1.0) for _,r in rules)
    print(f"   data seed {ds}: {ex}/{len(rules)} exact")
