# Independent-shape four-way retrieval transfer

## Question

Can the controller transfer its four-way memory policy when three rank
crossings and two usage slopes move independently, so one global boundary
shift no longer summarizes the task?

## Failure localization

The promoted boundary-transfer controller reached only 61–81% on initial
independent-deformation screens. More unique lifetimes did not help. Discarded
supervised probes then separated three hypotheses:

- widening the existing MLP moved errors between the two held-out arms rather
  than solving both;
- replacing raw usage with log usage removed some optimization noise but
  retained the same mutually exclusive middle-row failures;
- expressing the four generic intervals where candidate rows exchange rank
  reached 100% on both arms across three diagnostic seeds.

The diagnostic weights were discarded. They justified an architectural test,
not a capability claim.

## Architecture

The inherited 298,524-parameter controller remains frozen. A 421-parameter
relational proposer receives only generic memory evidence:

- four cosine similarities sorted by content;
- their correspondingly sorted usage values;
- three pairwise rank-exchange locations derived from cosine and log usage.

It proposes the midpoint of each of the four regions where a different
physical row would win. A learned selector mixes those proposals and a learned
opening decides how much they may alter the inherited scale. The opening is an
exact no-op at insertion; a straight-through bounded gradient prevents one
negative exploratory update from killing it.

The interface contains no task identity, target row, private boundary, rule
label, or correct action.

## Zero-label credit

Each proposal is physically executed through the same memory ranking rule.
The environment verifier scores the four retrieved latent values. Those four
scalar outcome bits:

1. identify which proposal was behaviorally successful;
2. train the selector even while the no-op opening is closed;
3. train the final continuous action toward the successful proposal.

This resolved a measured chicken-and-egg failure: without selector credit, a
closed gate prevented the untrained proposal from improving, and the untrained
proposal gave the gate no reason to open.

Training cycles random independent deformations with two difficult sign
patterns. All training magnitudes remain inside crossing `[-0.07, 0.07]` and
slope-ratio `[-0.12, 0.12]`. Held-out magnitudes are strictly disjoint:
crossings `[0.075, 0.085]` and slope ratios `[0.13, 0.15]`.

## Gates

For both held-out shape families:

- at least 90% overall and 85% in every target class;
- best fixed scalar at most 35%;
- feature shuffle costs at least 20 points;
- value corruption costs at least 15 visual-success points;
- at least 90% with and without physical row permutation;
- at least 90% after disk save/reload, with every bank exact.

Both parent retrieval policies must remain at least 95%; binary mapping and
four-rule behavior must pass; only proposer parameters may change; total
runtime must remain below five minutes.

## Results

| run | alternating classes | grouped classes | stable verifier bits | parent continuous / conditional | accepted |
|---|---:|---:|---:|---:|---:|
| seed 19511 | **100% each** | **100% each** | 512 | 99.61% / 100% | yes |
| seed 19512 | **100% each** | **100% each** | **8,192** | 98.83% / 100% | yes |
| shuffled reward 19513 | 0% each | 0% each | never | retained | no |
| selector-credit ablation 19512 | 100/100/**0**/100% | 100% each | never | retained | no |

The conservative replicated stable threshold is 8,192 unique verifier bits.
Each formal run consumed 12,288 bits and 12,288 generated logical contexts,
plus 768 parent-rehearsal contexts. Training made 3,000 internal optimizer
updates and explicitly accounted for 5,564,928 replayed examples. Training
took 19.10–23.66 seconds; complete causal, retention, and 256-bank physical
audits took 66.06–83.82 seconds.

Across both accepted seeds:

- both held-out families and every class scored 100%;
- no fixed scalar exceeded 25%;
- feature shuffling reduced accuracy to 23.2–26.8%;
- corrupting values reduced visual success to 0%;
- all 512 held-out physical-bank reads were correct;
- every saved bank reloaded keys, values, and usage exactly;
- binary mapping, four-rule behavior, and both parent retrieval skills passed.

Seed 19512 briefly regressed at checkpoint 8 before recovering by checkpoint
16. The 512-bit seed is therefore a best case, not the replicated claim.

## Conclusion

This closes the independent-width-and-slope frontier. The controller learned
to choose among generic relational action proposals from scalar outcomes and
transferred perfectly beyond every trained deformation magnitude without
forgetting older skills.

The core sample-efficiency result is not “more capacity.” It is credit
factorization: verify a small set of task-agnostic candidate computations,
teach the candidate selector directly from those outcomes, and let a
behavior-preserving gate earn influence only after the proposal becomes
useful.

The promoted checkpoint is
`artifacts/checkpoints/unified_four_target_shape_transfer_seed19511.pt`, SHA-256
`2bf17856835155e785dcc1b9b20da63cd670ccbc05051965a2ddc983cd87fe45`.

The next frontier replaces generator-declared behaviorally opposite memories
with naturally occurring duplicates and near-duplicates, so equivalence and
conflict must be discovered from experience.
