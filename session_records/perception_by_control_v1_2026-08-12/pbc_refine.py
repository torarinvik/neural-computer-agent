"""Refine the top-5 by full-fidelity control on TRAIN worlds only, then
report that winner's held-out margin (already measured in pbc2)."""
import sys, json, glob
sys.path.insert(0, "/Users/torarinvikbjarko/Documents/Machine Learning Projects/neural-computer-agent-games")
import torch
from experiments.games_amodal.game_family import FamilyVerifier, family_variants

torch.set_num_threads(1)
SEED = int(sys.argv[1])
SP = sys.argv[2]
SLOTS, VALUES, HEIGHT, WIDTH, PLANES = 6, 8, 8, 8, 3
PAR_OPS = ("NOOP","INC","DEC","CINC","CDEC","COPY","SINC","SDEC")
MODULI = tuple(range(2, VALUES + 1)); NOOP = (0,0,0)
ROWS_IX = torch.arange(HEIGHT).view(-1,1).expand(HEIGHT,WIDTH)
COLS_IX = torch.arange(WIDTH).view(1,-1).expand(HEIGHT,WIDTH)

def slot_write(state,s,op,j,m):
    name,mod = PAR_OPS[op],MODULI[m]; col = state[:,s]
    if name=="NOOP": return col
    if name=="INC": return (col+1)%mod
    if name=="DEC": return (col-1)%mod
    if name=="SINC": return torch.clamp(col+1,max=mod-1)
    if name=="SDEC": return torch.clamp(col-1,min=0)
    if name=="CINC": return torch.where(state[:,j]!=0,(col+1)%mod,col)
    if name=="CDEC": return torch.where(state[:,j]!=0,(col-1)%mod,col)
    if name=="COPY": return state[:,j]
    raise AssertionError(name)

def run_parallel(state,program):
    out = state.clone()
    for s in range(SLOTS): out[:,s] = slot_write(state,s,*program[s])
    return out

def executor(program,state): return run_parallel(state,program)

def per_slot_search(before,after):
    program=[]
    for s in range(SLOTS):
        want=after[:,s]; best,bs=NOOP,-1.0
        for op in range(len(PAR_OPS)):
            for j in range(SLOTS):
                if j==s and PAR_OPS[op] in ("CINC","CDEC","COPY"): continue
                for m in range(len(MODULI)):
                    sc=float((slot_write(before,s,op,j,m)==want).float().mean())
                    if sc>bs: best,bs=(op,j,m),sc
                    if bs>=1.0: break
                if bs>=1.0: break
            if bs>=1.0: break
        program.append(best)
    return program

def _reductions(plane):
    mask=plane>0; any_set=mask.any(dim=(1,2))
    flat=plane.reshape(plane.shape[0],-1); top=flat.argmax(dim=1)
    rows_any=mask.any(dim=2); cols_any=mask.any(dim=1)
    ir=torch.arange(HEIGHT).view(1,-1); ic=torch.arange(WIDTH).view(1,-1)
    fr=torch.where(rows_any,ir,torch.full_like(ir,HEIGHT)).min(dim=1).values
    lr=torch.where(rows_any,ir,torch.full_like(ir,-1)).max(dim=1).values
    fc=torch.where(cols_any,ic,torch.full_like(ic,WIDTH)).min(dim=1).values
    lc=torch.where(cols_any,ic,torch.full_like(ic,-1)).max(dim=1).values
    w=mask.float(); tot=w.sum(dim=(1,2)).clamp(min=1.0)
    cr=(w*ROWS_IX).sum(dim=(1,2))/tot; cc=(w*COLS_IX).sum(dim=(1,2))/tot
    def cl(t): return t.long().clamp(0,VALUES-1)
    return {"peak_row":cl(top//WIDTH),"peak_col":cl(top%WIDTH),
            "centre_row":cl(cr.round()),"centre_col":cl(cc.round()),
            "first_row":cl(torch.where(any_set,fr,torch.zeros_like(fr))),
            "first_col":cl(torch.where(any_set,fc,torch.zeros_like(fc))),
            "last_row":cl(torch.where(any_set,lr,torch.zeros_like(lr))),
            "last_col":cl(torch.where(any_set,lc,torch.zeros_like(lc))),
            "count":cl(mask.sum(dim=(1,2))),
            "extent":cl((lr-fr).clamp(min=0))}

def encode_with(features):
    def enc(screen):
        fr=screen.view(-1,PLANES,HEIGHT,WIDTH)
        cache={c:_reductions(fr[:,c]) for c in {f[0] for f in features}}
        out=torch.zeros((fr.shape[0],SLOTS),dtype=torch.long)
        for s,(c,r) in enumerate(features[:SLOTS]): out[:,s]=cache[c][r]
        return out
    return enc

def goal_cost(state,ref,pa,pb):
    return ((state[:,pa[0]]-ref[:,pb[0]]).abs()
            +(state[:,pa[1]]-ref[:,pb[1]]).abs()).float()

def play(config,bank,seed,enc,pa,pb,episodes,steps,mode="bank"):
    v=FamilyVerifier(config,batch_size=episodes,seed=seed); v.reset(seed=seed)
    g=torch.Generator().manual_seed(seed+4242); total=torch.zeros(episodes)
    for _ in range(steps):
        if mode=="random":
            action=torch.randint(0,4,(episodes,),generator=g)
        else:
            ref=enc(v.observation())
            ref=torch.where(ref<VALUES,ref,torch.zeros_like(ref))
            best,action=None,torch.zeros(episodes,dtype=torch.long)
            for act in range(4):
                prog=bank.get(act)
                st=ref if prog is None else executor(prog,ref)
                cost=goal_cost(st,ref,pa,pb)
                if best is None: best=cost.clone()
                else:
                    take=cost<best; best=torch.where(take,cost,best)
                    action=torch.where(take,torch.full((episodes,),act),action)
        total+=v.step(action).reward
    return float(total.mean())

def build_bank(config,seed,enc):
    bank={}
    for act in range(4):
        v=FamilyVerifier(config,batch_size=256,seed=seed+act); v.reset(seed=seed+act)
        b=enc(v.observation()); v.step(torch.full((256,),act,dtype=torch.long))
        a=enc(v.observation())
        keep=(b<VALUES).all(dim=1)&(a<VALUES).all(dim=1)
        if int(keep.sum())<8: continue
        bank[act]=per_slot_search(b[keep][:32],a[keep][:32])
    return bank

def moving(code): return {s for s in range(SLOTS) if len(set(code[:,s].tolist()))>1}

def choose_goal(config,enc,bank):
    v=FamilyVerifier(config,batch_size=256,seed=SEED*31); v.reset(seed=SEED*31)
    usable=moving(enc(v.observation())); best,br=None,-1e9
    for a0 in range(SLOTS):
        for a1 in range(a0+1,SLOTS):
            for b0 in range(SLOTS):
                for b1 in range(SLOTS):
                    if b0==b1 or {a0,a1}&{b0,b1}: continue
                    if not {a0,a1,b0,b1}<=usable: continue
                    r=play(config,bank,SEED*977+1,enc,(a0,a1),(b0,b1),16,8)
                    if r>br: best,br=((a0,a1),(b0,b1)),r
    return best

d=json.load(open(f"{SP}/pbc2/pbc-{SEED}.json"))
top5=[n for _,n in d["train_ranking"][:5]]
feats={n:[tuple(f) for f in None or []] for n in []}
# reconstruct candidate features exactly as the probe builds them
REDUCTIONS=["peak_row","peak_col","centre_row","centre_col","first_row",
            "first_col","last_row","last_col","count","extent"]
VOC=[(c,r) for c in range(PLANES) for r in REDUCTIONS]
rng=torch.Generator().manual_seed(SEED+271)
cands={"vocab_peaks":[(0,"peak_row"),(0,"peak_col"),(1,"peak_row"),
                      (1,"peak_col"),(2,"peak_row"),(2,"peak_col")],
       "vocab_greedy_margin":[(0,"peak_row"),(0,"peak_col"),(2,"peak_col"),
                              (1,"last_row"),(1,"peak_row"),(2,"last_col")],
       "vocab_centres":[(c,r) for c in range(3) for r in ("centre_row","centre_col")],
       "vocab_firsts":[(c,r) for c in range(3) for r in ("first_row","first_col")],
       "vocab_lasts":[(c,r) for c in range(3) for r in ("last_row","last_col")],
       "vocab_counts":[(c,r) for c in range(3) for r in ("count","extent")]}
for k in range(8):
    order=torch.randperm(len(VOC),generator=rng)[:SLOTS]
    cands[f"vocab_random{k}"]=[VOC[int(i)] for i in order]

train=family_variants()[:8]
out={}
for name in top5:
    enc=encode_with(cands[name]); rewards=[]
    for config in train:
        bank=build_bank(config,SEED*31,enc)
        picked=choose_goal(config,enc,bank)
        if picked is None:
            rewards.append(play(config,{},SEED*977,enc,(0,1),(2,3),64,12,mode="random"))
            continue
        rewards.append(play(config,bank,SEED*977,enc,picked[0],picked[1],64,12))
    out[name]=round(sum(rewards)/len(rewards),4)
    print(f"  seed {SEED}  {name:<22} full-fidelity train control {out[name]:+.4f}",flush=True)
winner=max(out,key=out.get)
held=d["held_out"][winner]
print(f"  seed {SEED}  REFINED WINNER {winner}  held-out margin "
      f"{held['mean_bank']-held['mean_random']:+.4f}")
json.dump({"seed":SEED,"refined":out,"winner":winner,
           "winner_held_out_margin":round(held['mean_bank']-held['mean_random'],4)},
          open(f"{SP}/pbc2/refine-{SEED}.json","w"),indent=2)
