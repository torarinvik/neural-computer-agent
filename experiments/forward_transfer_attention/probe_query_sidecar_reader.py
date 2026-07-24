"""Tiny query-conditioned reader over compact + raw sidecar memory rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory
from experiments.syllogimous_latent_agent.data import collate_episodes
from .environment import generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _forward, seed_everything
from .train_consolidator import _initial_memory

class QueryReader(nn.Module):
    def __init__(self, width=160, heads=64, value=128):
        super().__init__(); self.q=nn.Linear(width,heads); self.k=nn.Linear(width,heads); self.v=nn.Linear(width,value); self.pos=nn.Parameter(torch.zeros(1,2,heads)); self.head=nn.Sequential(nn.LayerNorm(value),nn.Linear(value,2))
    def forward(self, query, keys, values):
        q=self.q(query).unsqueeze(1); k=self.k(keys)+self.pos; score=(q*k).sum(-1)/(k.shape[-1]**.5); w=torch.softmax(score,-1); z=(w.unsqueeze(-1)*self.v(values)).sum(1); return self.head(z)

def collect(model,consolidator,start,n,device):
    qs=[];ks=[];vs=[];ys=[]
    for off in range(0,n,64):
        items=[generate_temporal_attention_lifetime(start+off+i,heldout=start>=72_000_000,feedback_mode='color-button') for i in range(min(64,n-off))];b=len(items);ys.append(torch.tensor([x.rule for x in items]))
        mem=_initial_memory(model,items,device); out,_=_forward(model,[x.supports[0] for x in items],mem,device); app=mem.append(out.write_keys,out.write_values,out.write_strengths,torch.ones_like(out.write_strengths)); comp=consolidator(app); side=comp.append(out.write_keys,out.write_values,out.write_strengths,torch.ones_like(out.write_strengths)); batch=collate_episodes([x.future_queries[0] for x in items]);q=model.retrieval_summary(batch['frames'].to(device),batch['pcm'].to(device),batch['mask'].to(device));qs.append(q.detach().cpu());ks.append(side.keys.detach().cpu());vs.append(side.values.detach().cpu())
    return torch.cat(qs),torch.cat(ks),torch.cat(vs),torch.cat(ys)

def main():
    p=argparse.ArgumentParser();p.add_argument('--controller-checkpoint',type=Path,required=True);p.add_argument('--consolidator-checkpoint',type=Path,required=True);p.add_argument('--pairwise-transfer-checkpoint',type=Path,required=True);p.add_argument('--projection-transfer-checkpoint',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--train-lifetimes',type=int,default=128);p.add_argument('--test-lifetimes',type=int,default=128);p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',default='cuda');a=p.parse_args();seed_everything(113);d=torch.device(a.device);paths=(str(a.pairwise_transfer_checkpoint),str(a.projection_transfer_checkpoint));m,c=_load(a.controller_checkpoint,a.consolidator_checkpoint,d,transfer_paths=paths,transfer_strength=.01)
    tq,tk,tv,ty=collect(m,c,70_000_000,a.train_lifetimes,d);vq,vk,vv,vy=collect(m,c,72_000_000,a.test_lifetimes,d)
    mean=tq.mean(0,keepdim=True);scale=tq.std(0,keepdim=True).clamp_min(1e-5);tq=(tq-mean)/scale;vq=(vq-mean)/scale
    def fit(labels,seed):
        labels=labels.to(d); torch.manual_seed(seed);r=QueryReader().to(d);opt=torch.optim.AdamW(r.parameters(),lr=1e-3,weight_decay=1e-3);xq,xk,xv=tq.to(d),tk.to(d),tv.to(d)
        for _ in range(a.steps):
            idx=torch.randint(len(labels),(min(64,len(labels)),),device=d);loss=nn.functional.cross_entropy(r(xq[idx],xk[idx],xv[idx]),labels[idx].to(d));opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad(): pred=r(vq.to(d),vk.to(d),vv.to(d)).argmax(-1);return float((pred==vy.to(d)).float().mean())
    normal=fit(ty,113);shuffled=fit(ty[torch.randperm(len(ty))],114);out={'schema':'query-sidecar-reader-v1','steps':a.steps,'normal_test_accuracy':normal,'shuffled_test_accuracy':shuffled};a.report.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
