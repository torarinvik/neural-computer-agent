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
| `unified_pair_magnitude_repeated_compounding_seed23105.pt` | Repeated-compounding magnitude frontier learned from 44 new lifetimes |
| `unified_pair_numerosity_adjacent_bridge_seed23602.pt` | Adjacent magnitude→numerosity bridge learned from 16 new lifetimes |
| `unified_pair_numerosity_compounding_seed23712.pt` | Same-slot numerosity continuation learned from four new lifetimes |

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

### Compounding magnitude and adjacent numerosity

- progressively harder magnitude frontiers required 128, then 96, then 44
  new lifetimes;
- the adjacent magnitude→numerosity bridge reused the learned greater-than
  representation and required 16 new lifetimes / 96 verifier bits;
- the selected numerosity checkpoint passed 3/3 independent 32,768-lifetime
  causal audits at the conservative 22.4% dot-appearance frontier;
- matched shuffled-outcome controls failed, and every inherited skill remained
  within two percentage points of its frozen parent.

### Numerosity compounding

- the existing numerosity slot advanced from the 22.4% to the 23.0%
  dot-appearance frontier using four new lifetimes / 24 verifier bits;
- this reduced new experience by 75% relative to the initial 16-lifetime
  magnitude→numerosity acquisition, without adding parameters;
- real outcomes passed on 2/2 seeds while matched shuffled-outcome controls
  failed on 2/2;
- both children passed one shared 32,768-lifetime causal audit and the selected
  child passed the full 8,192-lifetime retention suite;
- all registered magnitude, relation, numerosity, and unrelated skills remained
  within two percentage points of the frozen parent;
- two-lifetime continuation passed only 1/2 real seeds, so four is the current
  replicated sample-efficiency floor.

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
[source repository](https://github.com/torarinvik/neural-computer-agent).

## Status

This is experimental research. It now demonstrates causally audited elementary
identify→observe→act behavior, repeated sample-efficiency gains within a
magnitude lineage, adjacent transfer into discrete numerosity, and one
replicated within-numerosity compounding step that used 75% less new
experience. It does not yet demonstrate general intelligence, pure-dot
numerosity mastery, or unbounded compounding across many unrelated primitives.
