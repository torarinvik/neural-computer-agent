# Capacity-limited consolidation by learned equivalence

## Question

Can the controller use its learned relation between independently acquired
latent memories to compress a stream, retain every behaviorally distinct
skill, and reload the result from disk—without semantic rule or equivalence
labels?

## Method

Each held-out stream contains 16 independently rendered binary-mapping
experiences drawn from two hidden behaviors. The controller produces every
memory value from visual support and scalar feedback. An online capacity-two
consolidator compares a new value with stored values:

- a calibrated relation score at or above zero merges it with an equivalent
  row and increases that row's usage;
- a novel value occupies a free row;
- when full, an apparently novel value replaces the least-used row.

The inherited 12,354-parameter relation scorer is frozen. Only a scalar scale
and bias are trained, using binary outcomes from actually comparing two
controller-created memories. The learner never receives rule bits,
equivalence labels, target rows, or a correct merge/store action.

## Sample-efficiency race

The optimizer replays each fixed verifier batch for 128 updates. Unique
verifier outcomes, rather than replayed examples, are the primary sample
budget.

| budget | seed | held-out accuracy | both skills retained | accepted |
|---|---:|---:|---:|---:|
| 32 bits | 20551 | 98.88% | 97.75% | no |
| 32 bits | 20552 | 99.41% | 98.83% | yes |
| **64 bits** | **20541** | **99.51%** | **99.02%** | **yes** |
| **64 bits** | **20542** | **99.46%** | **98.93%** | **yes** |

One 32-bit seed missed the pre-registered 98% skill-retention and
uncompressed-parity gates. The promoted replicated frontier is therefore 64
verifier bits, not the lucky 32-bit result. Each accepted run used 80 unique
logical lifetimes, 128 optimizer updates, and 4,096 replayed pair examples.
Training plus all audits took about 2.2 seconds per seed on an RTX PRO 6000
Blackwell.

## Controls and audits

At the promoted 64-bit budget:

- the uncalibrated relation retained both skills in 90.5–91.5% of streams;
- storing only the first two memories retained both in 50.2–50.6%;
- an uncompressed 16-row bank scored 100%;
- inverting the learned relation reduced both-skill retention to 0.78–0.88%;
- shuffling the 64 verifier outcomes reduced behavior to exactly 50% on both
  seeds;
- a valid counterfactual kept the bank tensors fixed, reversed the verifier
  rule, and produced 100% ordinary/reversed accuracy and a 100% selection
  flip;
- binary mapping, four-rule behavior, conditional retrieval, and continuous
  retrieval all retained their inherited gates;
- only `memory_equivalence_logit_scale` and
  `memory_equivalence_logit_bias` changed.

The disk audit saved and reloaded 128 independent capacity-two banks per seed.
Every bank reloaded exactly. Visual accuracy was 99.61% and 98.83%; both-skill
retention was 99.22% and 97.66%. Sixteen logical rows became two—an 8× row
reduction. Serialized bytes fell to 32.41% of the uncompressed files because
fixed metadata dominates these tiny banks; this is not an 8× byte claim.

## Promoted artifact

`artifacts/checkpoints/unified_equivalence_consolidation_seed20541.pt`

SHA-256:
`255aa931721d66f89127a7a58eb1fc8d5e270627488a49464dcb8139fa19af46`

## Conclusion and next frontier

The controller now converts a learned behavioral relation into real online
memory management. With only 64 truthful scalar feedback bits, two learned
calibration parameters compress 16 natural experiences to two disk-backed
memories while retaining both hidden skills at about 99% on held-out streams.
The shuffled-verifier and relation-inversion collapses make the consolidation
mechanism causal rather than merely correlational.

The next frontier is compounding utility: compare a clean consolidated bank
with an equally experienced uncompressed or naively truncated bank while
learning a genuinely new primitive. Success requires fewer verifier outcomes
for the new primitive, exact retention of the old two, and the usual
counterfactual and memory-corruption audits.
