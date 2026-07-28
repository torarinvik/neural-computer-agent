# Four-way physical memory retrieval

## Question

Can the inherited continuous retrieval controller learn to make any one of
four memory rows the correct target, while retaining its earlier binary and
continuous retrieval skills?

## Benchmark correction

The first version used independently generated binary-rule values as
distractors. Half of those values encode the same behavior as the designated
target by chance. The private row audit therefore demanded distinctions that
pixels, actions, and outcomes could not reveal.

The corrected generator replays the same visible support with the opposite
scalar verifier outcome to produce a behaviorally opposite latent value.
Exactly one physical row now earns reward in every lifetime and every other
row fails. The learner still receives no target row, rule identity, correct
action, or semantic label.

Four generic cosine-plus-usage envelopes win in disjoint scalar regions. Rows
are physically permuted independently in every lifetime. The first gradual
rung gives the third regime a robust interval ending at `0.58`; the fourth
begins above `0.60`. A fixed scalar therefore solves exactly one quarter.

## Learning rule

Ordinary policy gradient, a learned critic, paired perturbations, and a
five-candidate evolution step were rejected in sub-minute screens. They
oscillated between middle regimes, exploited critic extrapolation, or retained
only the inherited extremes.

The accepted learner explores 25 uniformly spaced values in its generic scalar
action space. It evaluates each of the four distinct retrieved rows once,
retains the scalar intervals that actually earned verifier reward, and updates
only when its prediction lies outside a successful interval. One verified
batch is replayed for 1,000 cheap internal optimizer steps.

Parent rehearsal also preserves behavior rather than numeric implementation:

- continuous retrieval may move anywhere that selects its inherited row;
- conditional retrieval may move anywhere that remains safely on the same
  side of its deployed `0.5` action threshold.

No verifier-private labels enter either loss.

## Pre-registered gates

- at least 90% overall row accuracy and 85% in every target regime;
- best fixed scalar at most 35%;
- at least 20 points lost under feature shuffling;
- at least 15 visual-accuracy points lost under value corruption;
- at least 90% with and without physical row permutation;
- at least 90% after physical disk save/reload, with exact persistence;
- 95% parent continuous retrieval and 95% parent conditional behavior;
- binary-mapping and four-rule retention;
- only the existing 49-parameter policy may change;
- under five minutes.

## Results

| run | new verifier bits | four classes | feature shuffle | value corruption | physical | parents | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 17827 | **512** | **100% each** | 24.80% | 0% | 100% | 100% / 100% | yes |
| seed 17828 | **512** | **100% each** | 25.39% | 0% | 100% | 100% / 100% | yes |
| seed 17829 | **512** | **100% each** | 25.10% | 0% | 100% | 100% / 100% | yes |
| reward shuffled 17830 | 512 | 0/0/100/0% | 25% | 50% | 25% | 47.95% / 100% | no |

Each accepted run used one 128-lifetime batch: 512 generated logical contexts
and 512 unique verifier bits. It reused 1,536 parent rehearsal contexts and
made 1,000 optimizer updates, explicitly accounting for 639,872 replayed
examples. Training took 1.47–2.72 seconds; the larger causal, retention, and
128-bank disk audits brought total runtime to 23–46 seconds.

The learned mean scalar was about `0.42`. Yet no fixed scalar exceeded 25%:
success comes from conditioning on generic retrieval statistics. All 128
physical banks reloaded keys, values, and usage exactly. Random row order and
unpermuted rerenders both scored 100%.

## Conclusion

This closes the previous “third or fourth row can really be correct” frontier.
The controller selects all four behaviorally distinct physical rows, survives
disk persistence, depends causally on features, values, and genuine verifier
outcomes, and retains every registered parent skill.

The sample-efficiency lesson is equally important: one verified experience
batch contained enough information, but the inherited head needed intensive
internal replay to reorganize. Preserving behavioral equivalence classes
rather than exact old activations resolved the stability–plasticity conflict.

The next gradual frontier is transfer to unseen envelope boundaries and a
narrower third/fourth interval, followed by naturally occurring duplicate
memories whose equivalence must be learned rather than generator-declared.
