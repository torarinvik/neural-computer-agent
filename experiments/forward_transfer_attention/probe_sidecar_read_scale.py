"""Probe whether a generic memory read can recover the raw sidecar by scaling it."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory
from experiments.syllogimous_latent_agent.data import collate_episodes
from .environment import generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .probe_temporal_order import _fit_probe
from .train import _append, _forward, seed_everything
from .train_consolidator import _initial_memory

def _collect(model, consolidator, *, start, n, device):
    out={a:[] for a in (0.0,1.0,2.0,4.0,8.0)}; ys=[]
    for off in range(0,n,64):
        items=[generate_temporal_attention_lifetime(start+off+i,heldout=start>=52_000_000,feedback_mode='color-button') for i in range(min(64,n-off))]
        b=len(items); ys.append(torch.tensor([x.rule for x in items]))
        mem=_initial_memory(model,items,device)
        support=[x.supports[0] for x in items]; support_out,_=_forward(model,support,mem,device)
        appended=mem.append(support_out.write_keys,support_out.write_values,support_out.write_strengths,torch.ones_like(support_out.write_strengths))
        compact=consolidator(appended)
        side=compact.append(support_out.write_keys,support_out.write_values,support_out.write_strengths,torch.ones_like(support_out.write_strengths))
        q=collate_episodes([x.future_queries[0] for x in items])
        query=model.retrieval_summary(q['frames'].to(device),q['pcm'].to(device),q['mask'].to(device))
        for alpha in out:
            strengths=side.strengths.clone(); strengths[:,-1]=strengths[:,-1]*alpha
            testmem=DifferentiableBatchMemory(b,model.hidden,device=device,keys=side.keys,values=side.values,strengths=strengths,admissions=side.admissions)
            recalled,_=testmem.read(query,model.read_top_k,model.log_read_scale.exp().clamp(max=100.0))
            out[alpha].append(recalled.detach().cpu())
    return {a:torch.cat(v) for a,v in out.items()},torch.cat(ys)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--controller-checkpoint',type=Path,required=True); p.add_argument('--consolidator-checkpoint',type=Path,required=True); p.add_argument('--pairwise-transfer-checkpoint',type=Path,required=True); p.add_argument('--projection-transfer-checkpoint',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--train-lifetimes',type=int,default=128); p.add_argument('--test-lifetimes',type=int,default=128); p.add_argument('--device',default='cuda'); a=p.parse_args(); seed_everything(93); d=torch.device(a.device)
    paths=(str(a.pairwise_transfer_checkpoint),str(a.projection_transfer_checkpoint)); model,con=_load(a.controller_checkpoint,a.consolidator_checkpoint,d,transfer_paths=paths,transfer_strength=0.01)
    tr,ytr=_collect(model,con,start=50_000_000,n=a.train_lifetimes,device=d); te,yte=_collect(model,con,start=52_000_000,n=a.test_lifetimes,device=d)
    result={}
    for alpha in tr:
        result[str(alpha)]={'linear':_fit_probe(tr[alpha],ytr,te[alpha],yte,nonlinear=False,device=d,seed=93),'mlp':_fit_probe(tr[alpha],ytr,te[alpha],yte,nonlinear=True,device=d,seed=93)}
    a.report.write_text(json.dumps({'schema':'sidecar-read-scale-v1','transfer_strength':0.01,'results':result},indent=2)+'\n'); print(json.dumps({'results':result}))
if __name__=='__main__': main()
