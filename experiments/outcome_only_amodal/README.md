# Outcome-only amodal promotion

This is the first promotion experiment for the clean `src/neural_computer`
runtime. Two independent frontends emit opaque event tokens. Stream `a`
contains only the high bit of a hidden four-way verifier target; stream `b`
contains only the low bit. The learner samples an opaque protocol action and
updates from scalar reward plus the exact logging propensity. The target,
correct action, semantic rule, and hidden verifier state never enter the
learner.

Run the short rung with:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.outcome_only_amodal.train \
  --steps 256 --batch-size 256 --seed 7
```

The promotion audit measures fused evidence, each partial stream, missing
evidence, a shuffled partner stream, action-shuffled output, intention
shuffling/zeroing, random action, and an outcome-reversal verifier. A
reward-shuffled learner is a separate negative control. The two partial-bit
conditions are expected to score about 0.5: one bit remains identifiable, but
the four-way target is not.

