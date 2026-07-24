"""Check whether a raw-write skip alongside compact memory preserves the rule."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from .probe_temporal_rule_memory import _load, _extract
from .probe_temporal_order import _fit_probe
from .train import seed_everything

def main():
    p=argparse.ArgumentParser(); p.add_argument('--controller-checkpoint',type=Path,required=True)
    p.add_argument('--consolidator-checkpoint',type=Path,required=True); p.add_argument('--pairwise-transfer-checkpoint',type=Path,required=True)
    p.add_argument('--projection-transfer-checkpoint',type=Path,required=True); p.add_argument('--report',type=Path,required=True)
    p.add_argument('--train-lifetimes',type=int,default=128); p.add_argument('--test-lifetimes',type=int,default=128)
    p.add_argument('--device',default='cuda'); a=p.parse_args(); seed_everything(87); d=torch.device(a.device)
    paths=(str(a.pairwise_transfer_checkpoint),str(a.projection_transfer_checkpoint))
    model,cons=_load(a.controller_checkpoint,a.consolidator_checkpoint,d,transfer_paths=paths,transfer_strength=0.01)
    tr,ytr=_extract(model,cons,start=44_000_000,lifetimes=a.train_lifetimes,batch_size=64,heldout=False,feedback_mode='color-button',device=d)
    te,yte=_extract(model,cons,start=46_000_000,lifetimes=a.test_lifetimes,batch_size=64,heldout=True,feedback_mode='color-button',device=d)
    shot=1; raw_tr=tr[shot]['raw_write_row']; raw_te=te[shot]['raw_write_row']; mem_tr=tr[shot]['memory_row']; mem_te=te[shot]['memory_row']
    result={}
    for name,xtr,xte in [('memory',mem_tr,mem_te),('raw',raw_tr,raw_te),('concat',torch.cat((raw_tr,mem_tr),-1),torch.cat((raw_te,mem_te),-1))]:
        result[name]={'linear':_fit_probe(xtr,ytr,xte,yte,nonlinear=False,device=d,seed=87),
                      'mlp':_fit_probe(xtr,ytr,xte,yte,nonlinear=True,device=d,seed=87)}
    shuffled=ytr[torch.randperm(ytr.numel())]
    result['concat_shuffled_linear']=_fit_probe(torch.cat((raw_tr,mem_tr),-1),shuffled,torch.cat((raw_te,mem_te),-1),yte,nonlinear=False,device=d,seed=88)
    out={'schema':'raw-compact-concat-probe-v1','shot':shot,'train_lifetimes':a.train_lifetimes,'test_lifetimes':a.test_lifetimes,'transfer_strength':0.01,'results':result}
    a.report.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
if __name__=='__main__': main()
