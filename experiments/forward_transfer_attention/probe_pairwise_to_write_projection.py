"""Tiny transfer probe: map an audited pairwise relation latent into write width."""
from __future__ import annotations
import argparse, json
import torch
from torch import nn
from .probe_temporal_event_snapshot_binder import _extract, _load, PairwiseSnapshotBinder
from .train import seed_everything

def main():
    p=argparse.ArgumentParser(); p.add_argument('--controller-checkpoint',required=True); p.add_argument('--consolidator-checkpoint',required=True); p.add_argument('--pairwise-checkpoint',required=True); p.add_argument('--report',required=True); p.add_argument('--model-output'); p.add_argument('--train-lifetimes',type=int,default=128); p.add_argument('--test-lifetimes',type=int,default=128); p.add_argument('--steps',type=int,default=500); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=23); a=p.parse_args(); seed_everything(a.seed); d=torch.device(a.device)
    controller,_=_load(a.controller_checkpoint,a.consolidator_checkpoint,d)
    tr=_extract(controller,start=31_000_000,lifetimes=a.train_lifetimes,batch_size=128,heldout=True,feedback_mode='color-button',render_variants=1,device=d)
    te=_extract(controller,start=32_000_000,lifetimes=a.test_lifetimes,batch_size=128,heldout=True,feedback_mode='color-button',render_variants=1,device=d)
    ck=torch.load(a.pairwise_checkpoint,map_location='cpu',weights_only=False); pair=PairwiseSnapshotBinder(tr[0].shape[-1],width=64).to(d); pair.load_state_dict(ck['model']); pair.eval(); mean,scale=ck['mean'].to(d),ck['scale'].to(d)
    with torch.no_grad():
        def latent(x):
            e=pair.project((x.to(d)-mean)/scale)+pair.positions; rel=[]
            for l,r in ((0,1),(0,2),(1,2)): rel.extend((e[:,l]*e[:,r],(e[:,l]-e[:,r]).abs()))
            return pair.head[:-1](torch.cat((*e.unbind(1),*rel),-1))
        ztr,zte=latent(tr[0]),latent(te[0]); ytr,yte=tr[1].to(d),te[1].to(d)
    projection=nn.Sequential(nn.Linear(64,160),nn.GELU(),nn.LayerNorm(160)).to(d); head=nn.Sequential(nn.LayerNorm(160),nn.Linear(160,2)).to(d); opt=torch.optim.AdamW((*projection.parameters(),*head.parameters()),lr=1e-3)
    for _ in range(a.steps):
        idx=torch.randint(len(ytr),(min(64,len(ytr)),),device=d); loss=nn.functional.cross_entropy(head(projection(ztr[idx])),ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        atr=float((head(projection(ztr)).argmax(-1)==ytr).float().mean()); ate=float((head(projection(zte)).argmax(-1)==yte).float().mean())
    if a.model_output:
        torch.save({'projection':projection.state_dict(),'head':head.state_dict(),'mean':mean.cpu(),'scale':scale.cpu()},a.model_output)
    out={'train_accuracy':atr,'test_accuracy':ate,'baseline':float(yte.float().mean()),'steps':a.steps,'schema':'pairwise-to-write-projection-v1'}; open(a.report,'w').write(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
if __name__=='__main__': main()
