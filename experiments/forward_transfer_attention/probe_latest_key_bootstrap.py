"""Cached supervised bootstrap for the exact latest-row key projection."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from .probe_temporal_rule_memory import _load, _extract
from .train import seed_everything

def main():
    p=argparse.ArgumentParser();p.add_argument('--controller-checkpoint',type=Path,required=True);p.add_argument('--consolidator-checkpoint',type=Path,required=True);p.add_argument('--pairwise-transfer-checkpoint',type=Path,required=True);p.add_argument('--projection-transfer-checkpoint',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--model-output',type=Path,required=True);p.add_argument('--train-lifetimes',type=int,default=128);p.add_argument('--test-lifetimes',type=int,default=128);p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',default='cuda');a=p.parse_args();seed_everything(105);d=torch.device(a.device)
    paths=(str(a.pairwise_transfer_checkpoint),str(a.projection_transfer_checkpoint));m,c=_load(a.controller_checkpoint,a.consolidator_checkpoint,d,transfer_paths=paths,transfer_strength=.01)
    tr,ytr=_extract(m,c,start=62_000_000,lifetimes=a.train_lifetimes,batch_size=64,heldout=False,feedback_mode='color-button',device=d);te,yte=_extract(m,c,start=64_000_000,lifetimes=a.test_lifetimes,batch_size=64,heldout=True,feedback_mode='color-button',device=d)
    xtr=tr[1]['raw_write_row'][:,:160].to(d);xte=te[1]['raw_write_row'][:,:160].to(d);ytr=ytr.to(d);yte=yte.to(d);mean=xtr.mean(0,keepdim=True);scale=xtr.std(0,keepdim=True).clamp_min(1e-5);xtr=(xtr-mean)/scale;xte=(xte-mean)/scale
    proj=nn.Linear(160,160).to(d);nn.init.zeros_(proj.weight);nn.init.zeros_(proj.bias);head=nn.Sequential(nn.LayerNorm(160),nn.Linear(160,2)).to(d);opt=torch.optim.AdamW((*proj.parameters(),*head.parameters()),lr=1e-3,weight_decay=1e-3)
    for _ in range(a.steps):
        idx=torch.randint(len(ytr),(min(64,len(ytr)),),device=d);loss=nn.functional.cross_entropy(head(proj(xtr[idx])),ytr[idx]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        train=float((head(proj(xtr)).argmax(-1)==ytr).float().mean());test=float((head(proj(xte)).argmax(-1)==yte).float().mean())
    torch.save({'projection':proj.state_dict(),'mean':mean.cpu(),'scale':scale.cpu()},a.model_output)
    out={'schema':'latest-key-bootstrap-v1','steps':a.steps,'train_accuracy':train,'test_accuracy':test};a.report.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
