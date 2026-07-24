"""Measure whether fusion-logit confidence separates useful vs nuisance use."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from .environment import generate_attention_lifetime, generate_shape_attention_lifetime, generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _forward
from .train_consolidator import _initial_memory

GEN = {"temporal": generate_temporal_attention_lifetime,
       "spatial": generate_attention_lifetime,
       "shape": generate_shape_attention_lifetime}

def collect(model, consolidator, primitive, device, n, batch):
    margins, entropies, correct = [], [], []
    gen = GEN[primitive]
    for off in range(0, n, batch):
        k=min(batch,n-off)
        items=[gen(17_000_000+off+i, heldout=True, query_count=1,
                   **({"feedback_mode":"color-button"} if primitive=="temporal" else {})) for i in range(k)]
        memory=_initial_memory(model,items,device); cursor=0
        while cursor<2:
            out,t=_forward(model,[x.supports[cursor] for x in items],memory,device)
            rk,rv=out.write_keys,out.write_values
            memory=memory.append(rk,rv,out.write_strengths,torch.ones_like(out.write_strengths))
            memory=consolidator(memory).append(rk,rv,out.write_strengths,torch.ones_like(out.write_strengths))
            cursor+=1
        captured=[]
        h=model.latest_row_answer_fusion_head.register_forward_hook(lambda _m,_i,o: captured.append(o.detach()))
        out,t=_forward(model,[x.future_queries[0] for x in items],memory,device); h.remove()
        logits=captured[-1]; p=torch.softmax(logits,dim=-1)
        top2=torch.topk(logits,2,dim=-1).values
        margins.append((top2[:,0]-top2[:,1]).cpu()); entropies.append((-(p*p.clamp_min(1e-8).log()).sum(-1)).cpu())
        correct.append((logits.argmax(-1)==t).cpu())
    return {"mean_margin":float(torch.cat(margins).mean()),"mean_entropy":float(torch.cat(entropies).mean()),"accuracy":float(torch.cat(correct).float().mean())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--controller-checkpoint",type=Path,required=True); p.add_argument("--consolidator-checkpoint",type=Path,required=True); p.add_argument("--pairwise-transfer-checkpoint",type=Path,required=True); p.add_argument("--projection-transfer-checkpoint",type=Path,required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--device",default="cuda"); a=p.parse_args(); d=torch.device(a.device)
    paths=(str(a.pairwise_transfer_checkpoint),str(a.projection_transfer_checkpoint)); m,c=_load(a.controller_checkpoint,a.consolidator_checkpoint,d,transfer_paths=paths,transfer_strength=.01); m.eval(); c.eval()
    out={prim:collect(m,c,prim,d,64,32) for prim in GEN}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
