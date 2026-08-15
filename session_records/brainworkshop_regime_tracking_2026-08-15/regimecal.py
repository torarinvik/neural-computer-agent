import torch
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.regime_tracking import track
from experiments.brainworkshop_canonical.rule_automata import sample_rule

def eps(pred, seed, n, length=48, noise=0.0, cnt=4):
    g=torch.Generator().manual_seed(seed); out=[]
    for _ in range(n):
        s=torch.randint(0,cnt,(length,),generator=g).tolist(); y=pred(s)
        if noise>0:
            f=(torch.rand(length,generator=g)<noise).tolist()
            y=[v^int(b) for v,b in zip(y,f)]
        out.append(Trace(tuple(s),tuple(y),tuple([True]*length),cnt))
    return out

rules=[sample_rule(symbol_count=4,state_count=st,seed=7000+100*st+i)
       for st in (2,3,4) for i in range(3)]
rules=[r for r in rules if r]

print("A) FALSE ALARM RATE: stationary streams of 48 episodes")
for noise in (0.0,0.02,0.05,0.10):
    splits=0; runs=0
    for r in rules:
        for ds in (11,23,37):
            rep=track(eps(r.expected,ds,48,noise=noise)); runs+=1
            splits+=len(rep.change_points)
    print(f"   noise {noise:<5}: {splits} false change-points over {runs} stationary streams  ({splits/runs:.2f} per stream)")

print()
print("B) DETECTION: A->B at episode 24, over rule pairs")
for noise in (0.0,0.05,0.10):
    detected=0; exact=0; total=0; extra=0
    for i in range(len(rules)-1):
        A,B=rules[i],rules[i+1]
        rep=track(eps(A.expected,11,24,noise=noise)+eps(B.expected,23,24,noise=noise))
        total+=1
        cps=rep.change_points
        if cps: detected+=1
        if 24 in cps: exact+=1
        extra+=max(0,len(cps)-1)
    print(f"   noise {noise:<5}: detected {detected}/{total}, exactly at 24: {exact}/{total}, spurious extra: {extra}")

print()
print("C) RETURN RECOGNITION: A->B->A, fits vs regimes")
for noise in (0.0,0.05,0.10):
    reuse=0; fits=0; total=0
    for i in range(len(rules)-1):
        A,B=rules[i],rules[i+1]
        st=eps(A.expected,11,20,noise=noise)+eps(B.expected,23,20,noise=noise)+eps(A.expected,37,20,noise=noise)
        rep=track(st); total+=1; fits+=rep.fits; reuse+=rep.reuses
    print(f"   noise {noise:<5}: {reuse}/{total} streams reused a regime; mean fits {fits/total:.2f} for 3 regimes")
