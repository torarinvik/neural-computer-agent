# Sample-efficiency and transfer ledger

The north-star measurement is verified reusable capability gained per unit of
experience, wall time, and compute. Final accuracy alone is insufficient.

For every new primitive or task transition, record:

- zero-shot accuracy and chance/majority baseline;
- unique lifetimes and optimizer updates to fixed accuracy thresholds;
- wall time and GPU time to each threshold;
- performance with intact, empty, shuffled, and garbage memory;
- retention on every previously mastered primitive;
- the same learning curve for a fresh-agent control;
- a selective ablation of the most relevant prior primitive when practical.
- modality of acquisition and modality of evaluation;
- cross-modal examples-to-threshold versus same-modal and fresh-agent controls.

The primary transfer ratio at threshold `q` is:

`fresh-agent examples to q / experienced-agent examples to q`

A ratio above one is positive transfer. A sequence of increasing ratios across
new tasks is evidence of compounding learning.

## Current entries

| Date | Transition or test | Experienced result | Control | Conclusion |
|---|---|---:|---:|---|
| 2026-07-24 | Temporal atom colors 0–1 → same atom colors 2–3 | 45.31% held-out first-write rule decode | 80.47% on colors 0–1; 53.91% majority | No palette transfer; current temporal primitive is identity-specific |
| 2026-07-24 | Temporal atom → four-color temporal composition | 51.56% first-write rule decode | 80.47% atom; 50.78% compositional majority | No compositional rule transfer at the write boundary |
| 2026-07-24 | Experienced versus fresh vision features on direct-outcome temporal relation, 3 seeds | At 120 unique lifetimes: 94.44% mean held-out accuracy | Fresh identical encoder: 83.61%; shuffled-label controls: 45.42–50.83% | Prior visual experience improves the eventual representation by 10.83 points, but does not yet reliably reduce unique lifetimes to threshold |
| 2026-07-24 | Learned temporal intention → unfamiliar four-command actuator protocol, 3 seeds | 32 phase-B reward bits to 75% on every seed; mean AULC 0.3017 | Fresh: 510/256/256 bits to 75%; mean AULC 0.1958 | Zero-label actuator transfer; median 8× and mean 10.65× reward-bit ratio at equal 200-update/6,000-example compute |
| 2026-07-24 | Temporal predictive core → simultaneous spatial selection, 3 seeds | 78.56% mean final, 0.2231 AULC; 75% at 256 reward bits on every seed | Shuffled-future core: 58.25%, 0.0849 AULC, never 75%; fully fresh: 50%, never 75% | Three-seed cross-primitive representation transfer; old temporal intention itself adds no stable gain |
| 2026-07-24 | Temporal+spatial predictive curriculum → delayed same/different, seed 211 | 80.47%, AULC 0.2422, 75% at 256 bits | Extra temporal: 82.03%, 0.2568, 75% at 128; spatial-shuffled: 81.77%, 0.2578, 75% at 128 | No second compounding gain; paired spatial update is consistent with representational drift |
| 2026-07-24 | Single-transition micro-intercept admission preflight, seed 211 | Action-conditioned AULC 0.0169, final 34.38%, no 60% crossing | Passive AULC 0.0391; shuffled action 0.0117; fresh approximately 0 | Gate failed in 20.5 seconds; do not scale this configuration. One decision did not make action-conditioned dynamics necessary |
| 2026-07-24 | Six-decision closed-loop intercept, seed 211 | Initial action-conditioned AULC 0.1438 and final 39.58% | Passive 0.1125/37.50%; fixed-action 0.1375/38.54%; random 16.67% | Provisional separation failed motion reversal, confidence, fixed-action, and no-effect audits; rejected |
| 2026-07-24 | Faster-motion six-decision correction | Action-conditioned final 17.71%, AULC 0 | Random 26.04%; fresh 18.75% | Dense immediate reward was misaligned with terminal interception; no scale-up |
| 2026-07-24 | Terminal-return readout on six-decision task | Action-conditioned 17.71% from 90 binary terminal outcomes | Passive 18.75%; shuffled-action 20.83%; random 26.04% | Aligned but sparse terminal return did not open learning |
| 2026-07-24 | Discarded supervised control ceiling and one-round DAgger | 65.97% held-out oracle-action decode but 15.63% direct success; DAgger +2.08 points | Passive oracle decode 66.15%; fresh direct success 22.92% | Large offline-to-policy-distribution gap; current six-step representation/control interface rejected |
| 2026-07-24 | Two-decision fixed-probe mastery | Stable 98.83–100% from 16 reward bits onward | No-effect and fully fresh 48.44%; passive final 71.48% | First elementary identify-then-act rung mastered causally in 27.3 seconds |
| 2026-07-24 | Fixed-probe core → balanced random probe | 48.44%, AULC 0.0070 | Matched fresh AULC 0.0406 | Direct promotion was too large and prior core transfer was negative |
| 2026-07-24 | 12.5% unfamiliar-probe curriculum | 50% balanced held-out; 100% on probe 0 and 0% on probe 1 | Chance 50% | Biased exposure reinforced the old shortcut; rare examples are not a sound bridge |
| 2026-07-24 | Fixed-target action-binding bridge, two readout seeds | 98.44% and 95.31% at 64 reward bits; protocol-swap 98.83%/97.66%; flip 97.27%/96.88% | Missing/no-effect 49.22–52.34%; shuffled outcomes near chance | Action/consequence binding is learnable when target choice is removed; 256 replay updates were required |
| 2026-07-24 | Fixed-target bridge → full varying-target task | 81.64%, AULC 0.1289, stable 75% threshold at 64 bits | Matched fresh core/readout 100%, AULC 0.2207, same stable 64-bit threshold | Retaining the bridge weights caused negative transfer; discard them for this promotion |
| 2026-07-24 | Full random-probe identify-then-act, fresh predictive core | 100% at 64 bits; all valid protocol/target rerenders 100% with 100% flips | Fully fresh 50.78%; fixed-protocol predictive control also 100% at 64 bits | Real zero-label causal capability, but action-conditioned pretraining has not earned a unique efficiency claim |
| 2026-07-24 | Incremental 8→16→32→64-bit learner | 93.36% at 64 bits; protocol 94.53%, target 95.31%; valid flips 87.89%/88.67% | Fully fresh 48.44%; fixed-protocol control reached stable 75% at 32 bits | Accumulating one readout is stable and causal, but generic predictive pretraining is currently more sample-efficient |
| 2026-07-24 | Full task capped at 32 bits with 512 replay updates | 52.73%; protocol 54.69%, target 52.34%; no causal flips | Fully fresh 48.44%; no control reached 75% | Extra replay cannot substitute for the missing 32→64 unique outcomes; keep 64 bits as the frontier |
| 2026-07-24 | Eight-clone 32-bit latent-interface tournament, broad generation, seed 211 | Best selection 72.27%, blind 69.53%; action/reward/fresh controls 44.14–48.44% | Baseline concat/high-rate clone 50.39%; no clone reached stable 75% | Narrow low-rate heads showed a real but sub-gate signal; reversal flips were only 47.66%/39.84%, so no promotion |
| 2026-07-24 | Refined descendants on seed 307 | Best selection 61.33%, blind 55.47%; reversal flips 48.83%/24.61% | Shuffled/fresh controls 44.14–50% | The first-generation advantage did not replicate; no checkpoint retained and the 64-bit frontier remains |
| 2026-07-24 | Eight-clone conservative learning-mechanism tournament at 32 bits, seed 211 | Zero-initialized rank-16 residual adapter: 68.75% selection, 66.41% blind | Frozen readout 61.72% selection; shuffled/fresh controls 46.09–50% | A bounded one-seed signal, but protocol/target flips were only 51.56%/50.39%; exact-parent replication required |
| 2026-07-24 | Exact rank-16 residual parent, seed 307 | 58.98% selection, 55.47% blind; target flip 13.67% | Shuffled/fresh controls 45.70–50% | Conservative adaptation did not replicate or learn the causal rule; reject the family at 32 bits and retain no checkpoint |
| 2026-07-24 | Eight reward-free predictive-objective refinements at 32 bits, seed 211 | Best AULC came from contrastive transition matching; 59.38% selection and 58.59% blind | Unrefined core had the best final selection score at 59.77%; shuffled/fresh controls 47.27–48.44% | Extra delta, cosine, contrastive, action-decode, and consequence-separation training did not improve causal learning; no replication or checkpoint |
| 2026-07-24 | Exact-baseline ignition map at 32/40/48/56/64 bits, seeds 151/211/307 | Mean normal accuracy rose 57.42%→66.15%→69.92%→76.04%→78.91%; only seed 211 passed every gate, at 64 bits | Seeds 151 and 307 never passed; mean 64-bit target-reversal flip was 66.67% | The 64-bit result is a valid single-seed capability, not a robust learning threshold; initialization/optimization variance is now the primary blocker |
| 2026-07-24 | Alternate answer-path initialization/optimizer offsets on the same three ignition curves | Seed 211 passed at 48 and 64 bits | Seeds 151/307 still failed through 64 | Readout randomness moves the ignition point but does not explain the entire seed hierarchy; factorial variance localization is required before population breeding |
| 2026-07-24 | Nine-horse causal variance decomposition at 64 bits | Core-initialization causal-floor range 74.22 points; pretraining-sampling range 46.09 points | Readout-initialization range 7.03 points; replay-sampling range 5.86 points | Predictive-core initialization is the dominant instability; race cores next, with retention as a hard eligibility gate |
| 2026-07-24 | Six-core successive-halving race, shared 64-outcome stream | Seeds 43/211/263 passed blind 64-bit gates; seed 263 reached 100% selection causal floor at 48 | Seeds 97/401/503 eliminated under pre-registered racing rules | Population racing found an early-igniting core without increasing unique environmental experience; exact-parent replication required |
| 2026-07-24 | Disjoint-stream replication of core seeds 211 and 263 with downstream seed 307 | Seed 263: 98.05% causal floor at 48 bits and 97.27% at 64, every behavior/control gate passed | Seed 211: no stable pass; action-shuffled control reached 67.19% at 64 | First replicated population-selected 48-bit learner; admit seed 263 to old-primitive retention/compatibility, but do not yet promote globally |
| 2026-07-24 | Core-263 prior-rung compatibility with exact complemented controls | Fixed probe stable at 16 bits; fixed target stable at 48 bits; core weights bit-identical | Anchor fixed probe stable at 32 and fixed target at 48 under the same stricter gates | Candidate does not sacrifice the earlier identify-then-act learning surfaces; no observed weight forgetting |
| 2026-07-24 | Immutable core-263 replay with SHA-pinned weights | Exact repeated metrics; 48- and 64-bit full passes; 40-bit normal 95.31% but missing-evidence uncertainty failed | Complemented action/reward and fresh-core controls remained below 60% | Stable reproducible frontier is 48 bits, a 25% reduction from 64; reject the tempting but incomplete 40-bit claim |

The direct-outcome comparison is a supervised probe curve, not an acceptable
deployed learning method. It is included only to measure whether prior sensory
experience exposes a more reusable representation. Across 12/30/60/120 unique
lifetimes, mean held-out accuracy was:

| Unique lifetimes | Experienced features | Fresh features | Difference |
|---:|---:|---:|---:|
| 12 | 57.08% | 53.06% | +4.02 |
| 30 | 53.06% | 53.47% | -0.41 |
| 60 | 68.89% | 67.64% | +1.25 |
| 120 | 94.44% | 83.61% | +10.83 |

The first seed suggested a twofold reduction in samples to 70%, but seeds 97
and 151 did not reproduce that threshold advantage. The correct conclusion is
therefore **positive endpoint representation transfer, but no demonstrated
compounding sample-efficiency gain yet**.

When multiple modalities are introduced, the ledger must distinguish:

- same-task transfer across surface identities;
- same-concept transfer across sensory modalities;
- transfer from a previously learned concept to a genuinely new primitive.

Only the last category directly supports a compounding-learning claim, while
the first two establish that the stored primitive is reusable and amodal.

The actuator-transfer entry is in the same-concept/interface category. It is
the first deployed result showing that a frozen learned intention survives a
protocol change: true sensory reversal averaged 79.69%, opposite-rule stale
intentions collapsed to 19.70%, and swapping the protocol without recalibration
also collapsed to 19.70%. A different-primitive transfer remains the next
required rung.

The temporal-to-spatial entry is the first different-primitive transition in
the ledger. Correct future pairing matters: at equal sensory data, predictive
steps, phase-B updates, and examples, the paired core beat the shuffled-future
core by 0.1382 mean AULC. True mirror accuracy averaged 80.38%, missing
feedback returned to 50.69%, and opposite-rule stale state collapsed to
21.44%. At least several further transitions are required before fitting or
claiming a compounding learning curve.

## Accounting correction from the identify-then-act curriculum

`bits_to_75` can overstate learning when independently initialized curve
points are non-monotonic.  New reports therefore also record
`stable_bits_to_75`: the first reward-bit prefix at which the threshold is met
at that point and every later measured point.  The full random-probe learner
briefly crossed 75% at 16 bits, regressed at 32, and reached stable mastery at
64.  The honest sample count is therefore 64 bits, not 16.

Replay compute is not free experience.  The 64-bit successes used 256
optimizer updates and up to 8,192 replayed examples per independently fitted
endpoint.  The incremental arm used 256 cumulative updates and 5,632 replayed
examples.  These quantities remain separate in every report.

The latest result also sharpens the compounding rule: retain a prior component
only when it improves the verified learning curve.  The mastered fixed-target
bridge made the full task slower, so its weights are not promoted even though
the bridge itself was a valid capability.

## Population-search lesson

The first clone tournament was computationally efficient: eight readouts shared
one cached predictive core and completed in 29 seconds.  The selection/blind
split prevented a nominal 72.27% winner from being mistaken for mastery, and a
second seed rejected the family.

Future evolutionary searches must use successive halving across seeds:

1. cheap single-seed screen for every clone;
2. rerun the exact surviving parents on a second seed;
3. mutate only families whose mean blind and causal scores remain above the
   pre-registered bar;
4. never reproduce a one-seed winner merely because it ranks first.

The follow-up changed the learning mechanism rather than the feature interface:
eight clones compared a frozen readout, zero-initialized residual adapters, and
limited action-embedding, predictor, and recurrent adaptation.  A rank-16
adapter improved the first seed but failed exact-parent replication.  This
closes the cheap "more adaptable readout" branch at 32 bits.  The next
screen changed the reward-free predictive objective while sharing cached
sensory experience.  Every refinement stayed below 60% blind accuracy and the
unrefined core had the best final selection score.  This also closes the cheap
"add an auxiliary predictive loss after pretraining" branch.

The missing interval has now been measured across three seeds.  There is no
robust ignition point at or below 64 outcomes: seed 211 passes at 64, seed 307
approaches the gate, and seed 151 stays near chance.  The next experiment is a
factorial variance decomposition that independently varies predictive-core
initialization, lifetime subset, readout initialization, and readout minibatch
sampling.  Population racing begins only after this identifies what should be
varied.
