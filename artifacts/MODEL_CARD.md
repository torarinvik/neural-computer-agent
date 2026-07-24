---
library_name: pytorch
tags:
  - reinforcement-learning
  - continual-learning
  - external-memory
  - sample-efficiency
  - vision
  - research
license: other
---

# Neural Computer Agent — Audited Research Checkpoints

These are small PyTorch research checkpoints for a real-time neural computer
that learns from rendered sensory streams, its own opaque actions, and scalar
deterministic outcomes.

They are not pretrained language models and do not emit language tokens. Their
outputs are opaque action/concept channels that can be connected to arbitrary
actuators.

## Training constraints

The learner was not trained with:

- semantic task or rule labels;
- game-state coordinates or velocities;
- correct-action labels;
- labels for unattempted actions;
- English chain-of-thought;
- a hand-written symbolic solver.

Private generator metadata was used only by discarded diagnostic probes and
held-out verifier audits.

## Files

| Checkpoint | Meaning |
|---|---|
| `identify_fixed_target_width64_bits64_audit_seed211.pt` | Fixed-target action/consequence-binding bridge, seed 211 |
| `identify_fixed_target_width64_bits64_audit_seed307.pt` | Independent bridge readout seed |
| `identify_random_fresh_width64_bits64_seed211.pt` | Full varying-probe/varying-target task; 100% audited endpoint |
| `identify_random_fresh_incremental64_seed211.pt` | Incremental 8→16→32→64-bit learner |

## Audited results

### Full task

- 100% held-out accuracy at 64 unique verifier outcomes;
- 100% protocol-rerender accuracy with 100% prediction flips;
- 100% target-reversal accuracy with 100% prediction flips;
- chance performance when the probe consequence was removed.

### Incremental learner

- 93.36% held-out accuracy at 64 unique verifier outcomes;
- 94.53% protocol-rerender accuracy;
- 95.31% target-reversal accuracy;
- 87.89% and 88.67% respective prediction-flip rates.

### Sample-efficiency boundary

A 32-outcome learner with 512 replay updates achieved only 52.73% and failed
the causal audits. The current honest stable frontier is therefore 64 unique
verifier outcomes, not 32.

## Important negative result

The mastered fixed-target checkpoint caused negative transfer when its weights
were retained for the full varying-target task. The project therefore promotes
inherited weights only when they improve the next held-out learning curve.

## Loading

```python
import torch

checkpoint = torch.load(
    "identify_random_fresh_width64_bits64_seed211.pt",
    map_location="cpu",
    weights_only=True,
)
print(checkpoint["schema"])
print(checkpoint.keys())
```

The implementation and reproducibility instructions live in the companion
source repository. Until that remote is published, the checkpoint architecture
is in:

`experiments/forward_transfer_attention/train_identify_then_act.py`

## Status

This is experimental research. It demonstrates a causally audited elementary
identify→observe→act capability. It does not yet demonstrate general
intelligence or compounding transfer across a long sequence of primitives.

