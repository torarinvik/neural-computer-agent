# Unseen-boundary four-way retrieval transfer

## Question

Did the four-target controller learn a reusable retrieval relation, or only
four narrow operating points?

## Failure localization

The promoted parent scored 100% on its original envelopes but 0% when every
crossing moved mildly in the unfamiliar direction. A narrow `±0.04` training
range still admitted one constant action per class, so replay could reach 88%
inside the range without learning movement. Widening the range removed that
shortcut.

The old policy input then became the bottleneck. It exposed only the best
cosine, best-versus-second margin, selected usage and occupancy. Parent banks
and moving four-row banks could therefore produce the same four observations
while requiring different actions. A discarded supervised probe showed that
appending generic statistics for all four rows made the joint function
representable and extrapolatable.

## Architecture and learning

The inherited 298,411-parameter controller remains frozen. A zero-output
113-parameter residual receives:

- the four inherited retrieval statistics;
- four cosine similarities sorted by content;
- their four correspondingly sorted usage values.

It is exactly behavior-preserving at insertion and invariant to physical row
permutation. No task identity, target row, boundary, rule label, or correct
action enters the controller.

Training explores 101 generic scalar actions, verifies each distinct retrieved
row once, and regresses to the center of the empirically successful region.
The center prevents the learner from stopping on a flat but non-extrapolating
edge. Behavioral rehearsal preserves the parents' deployed action regions.

Eight batches cover shifts sampled continuously from `[-0.09, 0.12]`. The
held-out negative band `[-0.099, -0.095]` and positive band `[0.13, 0.16]`
are disjoint from training.

## Gates

For each unseen band:

- at least 90% overall and 85% in every target class;
- best fixed scalar at most 35%;
- feature shuffle costs at least 20 points;
- value corruption costs at least 15 visual-success points;
- at least 90% with physical permutation and without it;
- at least 90% after physical disk reload, with every reload exact.

Both parent retrieval policies must remain at least 95%; binary mapping and
four-rule behavior must pass; only the residual may change; total runtime must
remain below five minutes.

## Results

| run | negative classes | positive classes | stable bits | parent continuous / conditional | accepted |
|---|---:|---:|---:|---:|---:|
| center 17915 | **100% each** | **100% each** | **1,536** | 98.93% / 100% | yes |
| center 17916 | **100% each** | **100% each** | **1,536** | 99.80% / 100% | yes |
| shuffled reward 17917 | 0/0/100/0% | 0/0/0/0% | never | 88.87% / 100% | no |
| legacy four features 17918 | 100/0/0/100% | 100/0/0/100% | never | 71.68% / 76.17% | no |

Each accepted run used 4,096 unique verifier bits and 4,096 new logical
contexts, plus 1,536 parent rehearsal contexts. The learner made 1,000
optimizer updates and explicitly accounted for 1,086,976 replayed examples.
Stable unseen mastery arrived after the third batch: 1,536 verifier bits.

Feature shuffling reduced unseen accuracy to 23.3–25.0%. Value corruption
reduced visual success to 0%. No fixed scalar exceeded 25%. Across two
accepted seeds, all 512 shifted physical bank evaluations were correct and
every disk reload was exact.

The formerly failing seed 17915 is the promoted checkpoint. Its complete
training and audit took 113.45 seconds; the independent replication was
audited under concurrent load and took 146.88 seconds.

## Conclusion

This is the first demonstrated extrapolation of the controller's learned
memory policy beyond its training boundary distribution. Success depends
causally on genuine verifier outcomes and full generic row evidence, survives
physical persistence and permutation, and does not overwrite prior skills.

The next gradual frontier is varying envelope widths and slopes independently,
so a shared global shift can no longer summarize the relation. After that,
natural duplicate values should replace generator-declared opposites.
