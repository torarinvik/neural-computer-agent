import torch
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.regime_tracking import track
from experiments.brainworkshop_canonical.rule_automata import sample_rule
def eps(pred,seed,n,length=48,noise=0.0,cnt=4):
    g=torch.Generator().manual_seed(seed); out=[]
    for _ in range(n):
        s=torch.randint(0,cnt,(length,),generator=g).tolist(); y=pred(s)
        if noise>0:
            f=(torch.rand(length,generator=g)<noise).tolist()
            y=[v^int(b) for v,b in zip(y,f)]
        out.append(Trace(tuple(s),tuple(y),tuple([True]*length),cnt))
    return out
rules=[r for r in (sample_rule(symbol_count=4,state_count=st,seed=7000+100*st+i) for st in (2,3,4) for i in range(3)) if r]
print("alpha    | false alarms/stream | detected | exact@24 | reuse")
for alpha in (1e-3, 1e-5, 1e-8):
    fa=runs=0
    for r in rules:
        for ds in (11,23,37):
            for noise in (0.0,0.05,0.10):
                fa+=len(track(eps(r.expected,ds,48,noise=noise),alpha=alpha).change_points); runs+=1
    det=ex=tot=0; reuse=0; rtot=0
    for i in range(len(rules)-1):
        A,B=rules[i],rules[i+1]
        for noise in (0.0,0.05,0.10):
            rep=track(eps(A.expected,11,24,noise=noise)+eps(B.expected,23,24,noise=noise),alpha=alpha)
            tot+=1; det+=int(bool(rep.change_points)); ex+=int(24 in rep.change_points)
            st=eps(A.expected,11,20,noise=noise)+eps(B.expected,23,20,noise=noise)+eps(A.expected,37,20,noise=noise)
            rr=track(st,alpha=alpha); rtot+=1; reuse+=int(rr.reuses>0)
    print(f"{alpha:<8} | {fa/runs:>18.2f} | {det}/{tot}   | {ex}/{tot}    | {reuse}/{rtot}")
