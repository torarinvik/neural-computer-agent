# Replay-free affine transition memory

This audit isolates the narrow mechanism missing from the rejected one-pass
MLP control: a sufficient-statistics memory consumes each opaque transition
once, stores only weighted normal-equation matrices, and predicts the held-out
affine transition rule without replaying raw evidence.

```text
.venv/bin/python experiments/external_affine_one_pass/train.py \
  --seed 13011 \
  --report-out /tmp/external-affine-one-pass.json
```

This is a promoted bounded primitive, not a general continual-learning claim.
It covers affine transition structure only; the next integration question is
how to use such a fast sufficient-statistics path alongside the general
nonlinear external model without assigning affine semantics by hand.
