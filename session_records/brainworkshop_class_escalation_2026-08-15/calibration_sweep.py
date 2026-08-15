import torch
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.class_escalation import escalate, Verdict
from experiments.brainworkshop_canonical.adversarial_probes import running_majority, count_parity, count_threshold
from experiments.brainworkshop_canonical.rule_automata import sample_rule

def traces(pred, seed, n, length, cnt=4):
    g = torch.Generator().manual_seed(seed); out=[]
    for _ in range(n):
        s = torch.randint(0, cnt, (length,), generator=g).tolist()
        out.append(Trace(tuple(s), tuple(pred(s)), tuple([True]*length), cnt))
    return tuple(out)

import os
LEN = int(os.environ.get("LEN","16"))
cases = []
for st in (1,2,3,4,5,6):
    for i in range(4):
        r = sample_rule(symbol_count=4, state_count=st, seed=7000+100*st+i)
        if r: cases.append((f'mealy-{st}.{i}', r.expected, True))
for a,b in ((0,1),(1,2),(2,3),(0,3)):
    cases.append((f'majority-{a}{b}', running_majority(a,b), False))
for k in (2,3,4):
    cases.append((f'threshold-{k}', count_threshold(0,k), True))
cases.append(('parity', count_parity(0), True))

tally = {}
false_id = []
escal = []
for name, pred, in_class in cases:
    fit = traces(pred, 11, 112, LEN); val = traces(pred, 99, 20, LEN)
    rep, mach = escalate(fit, val)
    key = (rep.verdict, in_class)
    tally[key] = tally.get(key, 0) + 1
    # A false IDENTIFIED is the number that matters: claiming success while the
    # returned machine does not actually predict held-out evidence.
    if rep.verdict is Verdict.IDENTIFIED:
        err = rep.errors[-1]
        if err is None or err > 0.0:
            false_id.append((name, err))
    if mach is not None:
        escal.append(name)
print('cases:', len(cases))
for (v, ic), n in sorted(tally.items(), key=lambda x: (x[0][0].value, x[0][1])):
    print(f'  verdict={v.value:<17} in_class={ic!s:<5} count={n}')
print('FALSE IDENTIFIED:', len(false_id), false_id)
print('escalated:', escal)
