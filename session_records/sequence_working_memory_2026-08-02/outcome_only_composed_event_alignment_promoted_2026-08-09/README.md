# Outcome-only composed event alignment — promoted bounded result

This two-seed audit extends the cyclic event-alignment rung with a fixed,
opaque dense orthogonal transform of the learned event tensor. The source
external register, decoder, and parent controller are frozen. Only the
replaceable bridge is trained, using sampled scalar verifier outcomes from
opaque actions; controller state is masked to zero.

Both seeds passed every bounded gate:

- source capability mastered: `0.941` and `0.992`;
- transformed representation below mastery before adaptation: `0.500` and
  `0.504`;
- bridge recovered target capability: `0.949` and `0.984`;
- shuffled-outcome controls stayed below mastery: `0.445` and `0.594`;
- source retention, frozen parent, external register, and decoder digests all
  passed;
- stable bridge prefixes appeared at `10,240` and `6,144` verifier bits;
- replayed examples: `0`.

This promotes scalar-only recovery from a composed dense event-space change.
It remains a bounded synthetic alignment result: the transform is fixed and
invertible, and the experiment does not establish arbitrary modality
alignment, non-invertible recovery, unrestricted memory growth, or general
continual learning.
