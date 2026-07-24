# Tiny probe / escalation-gate resume — 2026-07-23

## Runtime

- Vast instance: `ssh -i ~/.ssh/id_ed25519_vast_ai -p 10720 root@209.146.116.50`
- GPU: NVIDIA GeForce RTX 5090, 31.8 GB VRAM, CUDA 13.0
- PyTorch: `2.13.0+cu130`
- Probe runtime uses `LD_LIBRARY_PATH` for the pip-installed CUDA libraries:
  `nvidia/cusparselt/lib`, `nvidia/nvshmem/lib`, and `nvidia/nccl/lib`.

## Completed experiment

Ran `probe_temporal_rule_memory` twice with the preserved `bootstrap_full.pt`
checkpoint, color-button feedback, one shot, 32 train and 32 held-out
lifetimes, batch size 16, and a hard 55-second timeout.

- Seed 41: raw-write MLP final held-out accuracy 56.25%; empirical majority
  baseline 53.125%. Compact/recalled taps were at or below 56.25%.
- Seed 97: raw-write MLP final held-out accuracy 53.125%; exactly baseline.
- Both runs fit their tiny training sets strongly, so this is not evidence of
  generalization. The combined gate status is `PROMISING_CANDIDATE`, not
  `PROMISING`; no longer run is authorized.

## Next experiment

Change one thing and stay under one minute: increase the held-out evaluation
to 128 lifetimes while keeping the model/checkpoint and probe seed fixed, or
run a shuffled-label control at the same 32/32 scale. Do not start a 3-minute
run unless two independent tiny reports beat their empirical baselines and
the shuffled control is at chance.
