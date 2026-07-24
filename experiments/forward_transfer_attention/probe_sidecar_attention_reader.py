"""Tiny task-agnostic attention reader over compact and raw memory rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from .probe_temporal_rule_memory import _load, _extract
from .train import seed_everything

class RowReader(nn.Module):
    def __init__(self, width=320, hidden=128):
        super().__init__(); self.row=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden),nn.GELU()); self.score=nn.Linear(hidden,1); self.head=nn.Linear(hidden,2)
    def forward(self,x):
        z=self.row(x); w=torch.softmax(self.score(z).squeeze(-1),-1); return self.head((w.unsqueeze(-1)*z).sum(1))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--controller-checkpoint',type=Path,required=True); p.add_argument('--consolidator-checkpoint',type=Path,required=True); p.add_argument('--pairwise-transfer-checkpoint',type=Path,required=True); p.add_argument('--projection-transfer-checkpoint',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--train-lifetimes',type=int,default=128); p.add_argument('--test-lifetimes',type=int,default=128); p.add_argument('--steps',type=int,default=1000); p.add_argument('--device',default='cuda'); a=p.parse_args(); seed_everything(95); d=torch.device(a.device)
    paths=(str(a.pairwise_transfer_checkpoint),str(a.projection_transfer_checkpoint)); model,con=_load(a.controller_checkpoint,a.consolidator_checkpoint,d,transfer_paths=paths,transfer_strength=0.01)
    tr,ytr=_extract(model,con,start=54_000_000,lifetimes=a.train_lifetimes,batch_size=64,heldout=False,feedback_mode='color-button',device=d); te,yte=_extract(model,con,start=56_000_000,lifetimes=a.test_lifetimes,batch_size=64,heldout=True,feedback_mode='color-button',device=d)
    def rows(x): return torch.stack((x[1]['memory_row'],x[1]['raw_write_row']),1)
    xtr,xte=rows(tr),rows(te); reader=RowReader().to(d); opt=torch.optim.AdamW(reader.parameters(),lr=1e-3,weight_decay=1e-3); xtr=xtr.to(d);xte=xte.to(d);ytr=ytr.to(d);yte=yte.to(d)
    for _ in range(a.steps):
        idx=torch.randint(len(ytr),(min(64,len(ytr)),),device=d); loss=nn.functional.cross_entropy(reader(xtr[idx]),ytr[idx]); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        pred=reader(xte).argmax(-1); normal=float((pred==yte).float().mean())
    shuffled=ytr[torch.randperm(len(ytr))]; control=RowReader().to(d);opt=torch.optim.AdamW(control.parameters(),lr=1e-3,weight_decay=1e-3)
    for _ in range(a.steps):
        idx=torch.randint(len(shuffled),(min(64,len(shuffled)),),device=d);loss=nn.functional.cross_entropy(control(xtr[idx]),shuffled[idx]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): shuffled_acc=float((control(xte).argmax(-1)==yte).float().mean())
    out={'schema':'sidecar-attention-reader-v1','steps':a.steps,'normal_test_accuracy':normal,'shuffled_test_accuracy':shuffled_acc,'train_lifetimes':a.train_lifetimes,'test_lifetimes':a.test_lifetimes}
    a.report.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
