# Active causal sequence selection — rejected — 2026-08-11

## Question

Can a trainer improve ordered fragment composition by probing a larger pool of
fresh rendered sequences, selecting rows whose common-random
leave-one-transition-out verifier outcomes change the answer, and training on
those rows? The passive arm pays for the same candidate probe and selects a
matched random subset.

The selector is trainer-only. Verifier-private correct actions and intervention
outcomes never cross the learned event, external-state, combiner, or decoder
interfaces. The parent controller, register interpreter, and acquired fragment
bank remain frozen.

## Three-seed result

Both arms used seeds `41/42/43`, a serial combiner, leave-one-out credit weight
`0.5`, candidate multiplier `2`, parent/primitive/composition updates
`8/16/16`, batch size `8`, span `3`, and audit count `16`.

| arm | held-out order accuracy by seed | stable prefix | promoted |
| --- | --- | --- | --- |
| active top-k causal rows | `0.5208/0.4792/0.5000`; `0.6042/0.4583/0.5000`; `0.5625/0.4375/0.6042` | none | no |
| passive matched random rows | `0.5208/0.4792/0.5000`; `0.6042/0.4375/0.5000`; `0.5625/0.4375/0.6042` | none | no |

The active arm therefore did not beat passive selection on this rung. Its
selected causal signal was `0/0`, `0/0`, and `0.0052/0.0026` by seed (candidate
mean/selected mean at the final audit point); the signal was not consistently
available or transferable. Both arms used exactly `68,688` unique verifier bits
per seed, including `41,472` candidate/intervention-selection bits, `13,824`
leave-one-out training bits, zero replay, and `120` optimizer updates.

The existing wrong-order, missing-evidence, reward-shuffled, frozen-parent,
frozen-bank, and persistence controls ran, but the held-out and stable-prefix
promotion gates failed. This is a diagnostic rejection, not evidence against
active data selection in general.

## Decision

Reject active top-k selection as the current composition fix. Retain the
versioned selection API and matched accounting because they provide a clean
future control. Do not increase candidate-pool size or add more selector
features yet. First make the intervention informative: the serial learner must
produce a stable, answer-changing counterfactual on a held-out context before
selection can allocate useful credit. The next rung should target
counterfactual decoder sensitivity or verifier-gated delayed credit, with
active/passive selection retained only as a secondary audit.
