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
| 2026-07-24 | Immutable core-263 → old-renderer spatial and same/different, sub-minute screens | No causal mastery through 120 reward bits; all final scores 44.6–50% | Matched fresh and component-transfer arms also stayed at chance | The renderer/primitive jump was too large; do not scale this branch before adding nearer curriculum rungs |
| 2026-07-24 | Near-transfer target-side and observed-effect-side, core-263 | Both reached stable 100% causal mastery from 8 reward bits; true rerenders flipped 100% | Matched fresh and recurrent-only stayed at 50% through 64 bits | First verified cross-task sample-efficiency gain; learned visual dynamics are reusable |
| 2026-07-24 | Near-transfer effect-target composition, two disjoint streams | Core-263 reached stable mastery at 24 bits in both runs; 100% held-out and counterfactual accuracy | Matched fresh never passed through 64 bits (50.0% and 54.7% final); exact-complement controls 0% | Replicated compounding result: two reusable perceived facts compose into a novel relation with fewer verified outcomes |
| 2026-07-24 | Gradual appearance bridge: new palette → new shapes → both, effect-target composition | Core-263 stayed stable at 24 bits on every rung; the combined shift replicated at 24 bits on a disjoint stream with 100% held-out/counterfactual accuracy and 100% flips | Matched fresh and recurrent-only stayed at 50% through 64 bits; missing evidence 47.7–52.3%; exact complements 0–2.7% | First replicated surface-generalization bridge: learned visual dynamics remain sample-efficient when object colors and geometry both change |
| 2026-07-24 | Post-appearance retained ladder on a third disjoint stream | Target side 8 bits, effect side 8 bits, effect-target composition 16 bits; all frozen cores bit-identical | Fresh and recurrent-only never passed through 64 bits | No catastrophic forgetting; the promoted appearance rung preserves or improves the previous 8/8/24 frontier |
| 2026-07-24 | Position → color-identity bridge, direct composition and localization | Direct fixed-color composition and both decision-state atoms stayed at chance through 64 bits | Discarded probes decoded effect color at 94.9% from vision but target color at only 60.2%; both decayed toward chance in the recurrent decision state | The event interface erased color, and position-trained vision had learned selective color invariance; scaling the same interface was rejected |
| 2026-07-24 | Salient color atoms with generic event snapshots and exclusive binary head | Inherited vision learned effect identity at 24 bits; reset vision learned target identity at 64 bits | Inherited target branch stayed at 50%; fresh effect branch required 64 bits | Measured simultaneous positive and negative transfer; retain the learned effect branch but reset the target branch |
| 2026-07-24 | Twelve-clone shared-experience color-primitive race | Two clones passed; clone 8 reached stable relation mastery from 16 bits with causal-floor sum 3.2969 | Ten clones failed; all clones shared the same 64/24 atom and 64 relation outcomes | Population search changed compute, not environmental experience; exact parent required full audit and blind replay |
| 2026-07-24 | Acquired color primitives → novel same/different relation, selected parent and blind replay | Stable causal mastery from 16 new relation bits on both streams; blind final normal/protocol-CF/target-CF and both flip rates all 100% | Target-only, effect-only, and neither-acquired controls never passed through 64; complement controls 0–6.6%; stratified shuffled medians 51.2%/55.1%, no causal passes | First replicated cross-attribute compounding: separately acquired continuous primitives reduce new-relation experience by at least 4× |
| 2026-07-24 | Post-color-compounding retained position ladder | Target side 8 bits, effect side 8 bits, effect-target composition 24 bits; core weights bit-identical | Fresh/recurrent-only never passed through 64 bits | Cross-attribute compounding did not catastrophically forget the earlier position abilities |
| 2026-07-26 | Linear adaptive disk-read gate, 40/160-update reward pilots | Best held-out 83.42%; 34.42% absent false accepts | Ungated 54.83%; discarded linear capacity probe 83.01%/33.26% false accepts | The five-parameter interface was capacity-limited; extra training was rejected |
| 2026-07-26 | Nonlinear adaptive read gate from verified outcomes | 91.55% held-out after 160 updates, 81,920 unique contexts, no replay, 9.71 s; useful accepts 89.67%, absent false accepts 17.33% | Ungated 54.74%, empty 50.02%; diagnostic-only nonlinear ceiling 88.18% classification | A 49-parameter gate inside the same controller crosses every tensor, retention, and parameter-isolation gate |
| 2026-07-26 | Adaptive sparse memory through physical disk, two blind seeds | First reload 91.50%/92.19%; repeat 91.02%/91.41%; duplicate growth 17.68%/17.29% | Empty 50.20%/50.00%; wrong-value corruption 70.41%/70.70% | First replicated admitted RAM/VRAM-to-disk loop with learned write/skip and read/no-read policies |
| 2026-07-26 | Full-bank capacity-4 replacement from future verified utility | 96.90% held-out, 93.55% correct evictions after 40 updates/71,680 verifier bits/7.01 s | Random 84.35%, fixed slot 85.11%, skip 80.91%, shuffled-age 81.79% | A 57-parameter head learned which opaque row to replace without utility labels; older skills retained |
| 2026-07-26 | Bounded replacement through physical disk, two blind seeds | 96.97%/96.29%; 2,048 rows before and after; zero capacity growth | Age-corrupted 81.35%/82.37%; old sparse disk loop 91.21%/91.02% | First replicated learned bounded replacement milestone |
| 2026-07-26 | Capacity-4 replacement policy → capacity 8 zero-shot | 93.99% behavior | Correct eviction 71.48%; age corruption 90.09%, only 3.91-point causal gap | Reject capacity-8 mastery; advance gradually through capacity 5 with rehearsal |
| 2026-07-26 | Capacity-4 replacement → capacity 5 zero-shot | 95.63%/95.51% physical accuracy, 87.11%/87.50% correct evictions | Age corruption 85.51% on both seeds; no updates | Replicated free transfer to the next bank size |
| 2026-07-26 | Capacity-6 bridge with alternating capacity-5 rehearsal | 96.39%, oracle matched, 100% correct evictions after 20 new-capacity + 20 rehearsal updates; 94,720 verifier bits; 8.69 s | Zero-shot parent had 94.99% but only 8.11-point causal separation | Minimal rehearsal bridge sharpens the reusable replacement rule without forgetting |
| 2026-07-26 | Capacity-6 physical replication and retained memory suite | 96.55%/96.71%, 100% correct evictions, zero growth | Age corruption 86.04%/86.78%; capacity-5 retention 96.37%; sparse disk retention 90.33%/90.14% | Replicated replacement gain with prior capabilities retained |
| 2026-07-26 | Capacity-6 policy → capacities 7/8/9 zero-shot | Capacity 7: 95.70%/96.23%; 8: 94.85%/95.75%; 9: 94.57%/94.62%; essentially 100% correct evictions | Age-to-slot corruption drove correct evictions to 0%; zero capacity growth throughout | Replicated capacity-9 physical frontier without further weight updates |
| 2026-07-26 | Fixed noisy frequency-plus-recency utility, two seeds | 95.32%/95.10% after 20 updates and 51,200 verifier bits per seed; 3.23 s training, no replay | Physical 96.81%/96.29%; age and frequency corruption each hurt; old gates retained | One generic residual composes persistent access frequency with inherited recency |
| 2026-07-26 | Online utility learning-mechanism localization | Policy-gradient, paired-greedy, and pairwise-preference pilots did not track recency→frequency→recency switches | Exact coefficient sweeps found distinct phase optima; high-temperature local gradients could reverse sign relative to greedy behavior | Reject those update rules for this atom; align search directly with deployed greedy verified behavior |
| 2026-07-26 | Online recency↔frequency utility adaptation, two seeds | All four phase gates passed in 64 updates/212,992 verifier bits/28.66–28.89 s; frequency target 86.43%/87.16%; no replay | Frozen frequency target 68.21%/67.53%; reward-shuffled control fell to 57.71%; binary/four-rule retention passed | First replicated boundary-free online adaptation of one controller utility coefficient |
| 2026-07-26 | Selected online controller through physical disk | 96.94%, within 0.13 points of visible oracle; all 1,024 histories and 6,144 rows survived save/reload, zero growth | Age shuffled 92.74%; frequency shuffled 88.66%; weights unchanged | Adapted policy survives real disk; adaptation process itself remains tensorized |
| 2026-07-26 | Redundant write-strength utility coefficient | Best strength phase added only 2.93 correct-target points after a move/stay repair | Write strength was already present in the inherited five-feature path | Reject redundant feature; do not weaken the four-point admission gate |
| 2026-07-26 | Outcome-reliability representation gate | At 40% reliability utility, two-dimensional ceiling 87.99% correct targets | Inherited coefficient 53.42%; over 30-point available gain | New generic verifier-history statistic earns a tiny online run |
| 2026-07-26 | Two-dimensional online utility, two seeds | 48 updates/196,608 verifier bits/29.34–29.37 s; reliability phase 78.22%/88.43%; all-equal 87.45%/83.35% | Frozen reliability 57.62%/58.40%; exact reward-shuffled run failed and ended all-equal at 64.31% | Replicated move/stay horse race adapts recency, frequency, and reliability without forgetting |
| 2026-07-26 | Multi-feature controller through physical disk | 96.21% versus 96.35% visible oracle; 1,024 histories/6,144 rows persisted, zero growth | Age/frequency/reliability shuffles changed correct evictions by 50.29/60.55/30.18 points and behavior by 3.11/6.75/2.56 | First physical composition of persistent access and verified outcome histories |
| 2026-07-26 | Physical-vs-tensor online parity preflight | 32 complete banks, 192 rows, and all three candidate rewards matched exactly in 1.34 s | Same winning candidate; every physical history survived reload | Disk-backed candidate evaluation is faithful enough to become the sovereign update signal |
| 2026-07-26 | Undersized physical online adaptation, two pilots | 32-bank pilots failed coefficient-movement or return gates | All persistence and parity checks passed | Reject as candidate-estimation noise; scale banks, not architecture or duration |
| 2026-07-26 | Physical online utility adaptation, two accepted seeds | 48 updates/196,608 bits/136.33–136.69 s; all four phase gates, 6,144 histories, retention, and parameter-scope gates passed | Reward-shuffled control failed every adaptation phase; physical choices tensor-equivalent on all updates | First replicated adaptation whose update decisions come from bounded serialized physical memory |

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

## Replicated near-transfer breakthrough

A direct jump from the immutable identify core to the older spatial and
same/different renderers produced no causal learning within 120 verified
outcomes. This was treated as a curriculum-gap result rather than an
architecture verdict.

A three-rung ladder then kept the rendered world fixed while changing only the
question asked of the learner:

1. which side contains the visible target;
2. which direction the learner's own probe visibly caused;
3. whether the observed effect and target direction match.

The immutable core reached stable causal mastery at 8, 8, and 24 unique
outcomes respectively. The exact fresh seed-263 initialization stayed at
chance through 64 outcomes on all three. The composition result replicated at
24 outcomes on a disjoint logical/render stream and downstream seed.

The result survived lifetime-disjoint held-out evaluation, true rerendered
counterfactuals with 100% prediction flips, exact complemented attempted-action
and reward controls at 0%, missing-evidence arms near 50%, and bit-identical
frozen core weights.

Component ablation localized the gain to the learned vision encoder:
vision-only transfer matched or slightly beat the complete immutable core,
whereas recurrent-only transfer stayed at chance. The narrow but important
conclusion is that predictive identify experience created a reusable visual
dynamics representation, and that representation reduced the verified
experience needed to learn a novel composition. Broad transfer across
unrelated renderers and continual consolidation remain open.

## Replicated gradual-appearance bridge

The next curriculum promotion changed public pixels without changing protocol,
event timing, opaque actions, verifier outcomes, or private logical labels:

1. replace the familiar four-color palette;
2. replace the circular target and rounded cursor with a diamond and chevron;
3. combine both shifts.

All three sub-minute horse races passed. Core-263 reached stable causal mastery
of the effect-target relation at 24 unique verifier bits on every appearance
rung. The combined shift then reproduced at 24 bits on a disjoint
logical/render stream and downstream seed. The matched-fresh and
recurrent-only arms stayed at chance through 64 bits. Vision-only transfer
matched the complete inherited core, again localizing the reusable capability
to learned visual dynamics.

The replicated combined run scored 100% on normal episodes and true
protocol-rerender counterfactuals with a 100% prediction-flip rate. Removing
the visible probe consequence returned performance to 49.2–52.3%, and exact
attempted-action/reward complements scored 0%. All inherited weights remained
bit-identical.

This is stronger than the earlier shared-renderer result, but it is still a
surface bridge rather than broad amodal transfer. Spatial relation and event
structure remain shared. The next honest rung preserves event structure while
replacing left/right position with a non-positional visual attribute such as
color identity.

## Replicated cross-attribute compounding breakthrough

The color bridge initially failed for a useful reason. Core-263 had learned to
ignore colors that were irrelevant to position control. A discarded diagnostic
probe found effect color strongly present in the event vision embedding but
target color weakly represented, and both were lost at the final recurrent
decision state. More outcomes and replay did not repair that interface.

The successful bridge made two task-agnostic changes:

1. retain generic per-event vision embeddings rather than forcing every fact
   through one leaky final state;
2. use one antisymmetric binary preference axis, encoding the benchmark's
   one-correct-answer constraint while still applying loss only to the
   attempted opaque answer.

The measured transfer was selective. Inherited vision learned the observed
effect-color atom in 24 outcomes versus 64 fresh, while a reset vision branch
learned target color in 64 outcomes and inherited vision stayed at chance.
Keeping both small branches was therefore more honest than forcing harmful
weight reuse.

After reward-only atom acquisition, a relation head received only the two
continuous primitive preferences. Twelve initialization clones shared one
experience stream; two passed and clone 8 was selected at 16 relation bits.
Search compute multiplied by twelve, but unique environmental experience did
not.

The exact parent and a fully disjoint replay both reached stable causal mastery
from 16 new relation outcomes. On the blind run, normal accuracy, both true
pixel-rerender counterfactual accuracies, and both prediction-flip rates were
100%. Removing either target or effect returned performance to chance. Either
primitive alone and the entirely unacquired architecture stayed below the
causal gate through 64 outcomes, establishing a transfer-ratio lower bound of
4×. Exact complements scored 0%; three target×effect-stratified shuffled-label
controls had median normal accuracy 55.1% and no causal pass.

This is the first replicated result in which acquiring two independent
attributes makes a new composition at least four times faster to learn. It is
not yet a general amodal concept space: the event structure and binary answer
interface remain shared, the color objects are deliberately salient, and the
checkpoint keeps two vision branches. The next steps are to shrink the color
cue, vary its palette, and then compress the branches without losing the
verified learning curve.

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

## Context-addressing zero-advantage result

A 13-parameter positive diagonal context metric was connected to the bounded
strategy bank and updated only from the verified advantage of the retrieved
strategy over the matched center candidate. The learned and frozen arms paid
for the same fourth candidate evaluation.

At the sub-minute gate, paired seeds 7060 and 7061 produced exactly identical
learned and frozen behavior. All ten eligible context updates received zero
verified advantage, so every learned scale remained exactly 1.0. Shuffling
stored keys harmed target choice on seed 7060, confirming that addressing can
matter, but it does not rescue the absent training signal.

This rejects longer training of this particular hard-retrieval reinforcement
route. The next context experiment must create informative counterfactual
credit without privileged labels—for example, a soft mixture over stored
strategies whose verifier-measured outcome can differentiate key weights.
Dynamic slot admission and eviction remain disabled.

## Soft context metric: action-boundary localization

Soft retrieval was tested with two opposite SPSA perturbations of the same
13-parameter context metric. Every arm paid for both mixture candidates, and
only their physical verifier-reward difference could update the metric.

At perturbation 0.4 and temperature 0.3, mixture weights differed by as much as
22.5% across paired candidates on two seeds, but every verified outcome stayed
identical. This localized the immediate bottleneck to the downstream discrete
action boundary rather than failure to form different mixtures.

A pre-registered sharper screen (perturbation 1.2, temperature 0.08) crossed
that boundary on seed 7072: one pair differed by 4.17 reward points and the
metric scales moved by up to 2.1%. The matched learned and frozen arms
nevertheless had identical target reward/accuracy, while seed 7073 supplied no
nonzero metric advantage. Shuffling keys harmed target transfer and old-return
retention on seed 7072, establishing causal dependence on addressing but not
learned improvement.

The soft mechanism is retained as a diagnostic, but the capability gate is
rejected. Do not enable dynamic admission or lengthen this exact run. The next
experiment needs more *informative unique comparisons per verifier bit*, not a
larger encoder: preserve the sharp setting and vary contexts gradually until
the proportion of action-divergent paired mixtures is high enough to learn.

## Reliability-context ramp rejection

A matched-cost six-round curriculum replaced repeated utility contexts with a
reliability ramp of 0.0, 0.1, 0.2, 0.3, 0.4, and a return to 0.0. This exposed
five unique utility contexts instead of two while preserving the 13-parameter
encoder, four strategy slots, sharp SPSA setting, and 672 verifier bits.

The ramp produced exactly one action-divergent and reward-divergent soft pair
out of five eligible comparisons (20%), identical to the best earlier sharp
screen. The useful comparison again differed by 4.17 reward points, yielding
672 verifier bits per informative pair. Binary and four-rule retention passed,
but the context encoder gained no denser learning signal.

This rejects scaling reliability diversity alone. The next sub-minute fork
changes mixture sharpness by one step while returning to the matched standard
curriculum; it promotes only if informative pairs per verifier bit increase.

Halving the softmax temperature from 0.08 to 0.04 also reproduced exactly one
informative pair out of five and 672 verifier bits per informative pair.
Simple sharpening is therefore rejected. The next fork screens multiple
cost-free perturbation directions using only the learner's own action
disagreement, then pays the physical verifier for one selected pair. This is
active experiment design rather than added supervision.

Sixteen-way active direction screening then examined 80 latent
counterfactual pairs at the same 672-verifier-bit cost. It still found only one
action- and reward-divergent pair. Unlucky perturbation direction is therefore
rejected as the bottleneck. Before changing training, the next diagnostic
measures whether individual stored strategy values themselves induce distinct
action patterns.

That diagnostic found two stored action patterns only at the single
informative round. Every later context reduced the full four-slot bank to one
behaviorally distinct action pattern, so no context metric or mixture could
obtain counterfactual credit there. The bottleneck is now localized to
behavioral strategy diversity. Longer context training, larger encoders, more
directions, and more reliability contexts are all rejected until the strategy
bank can preserve alternatives that actually act differently.

## Replicated value-diverse strategy memory

Action-signature admission was rejected immediately: signatures from different
physical batches are not comparable, and the informative rate fell to zero.
The corrected task-agnostic mechanism preserves separation directly in the
two-dimensional latent strategy values. It stores an already verifier-scored
candidate that maximizes the bank's minimum pairwise distance, so it adds no
environmental experience or verifier calls.

An RNG audit found and fixed a confound: context-direction proposals had shared
the later policy-perturbation random stream. With independent random streams,
matched winner-only and value-diverse banks used identical physical candidate
sequences and verifier budgets.

At 54 physical rounds, value diversity improved verifier bits per
reward-informative soft comparison on both paired seeds:

- seed 7073: 918.9 to 214.4 bits (4.29x);
- seed 7072: 1,072.0 to 714.7 bits (1.50x).

On seed 7073, 56.6% of soft pairs changed verified reward, reliability target
accuracy reached 41.7% versus 8.3% frozen, and old-utility return reached 95.8%
versus 12.5% frozen. On the harder seed 7072, the absolute informative rate
was lower (17.0%), but the diverse bank reached 27.8% old-return target
accuracy versus 9.7% frozen; the matched ordinary bank reached 0%.

All intact value-diverse arms retained binary and four-rule capability and
passed physical/tensor parity and persistence gates. Shuffling physical reward
alignment on seed 7073 failed the gate and collapsed old-return target accuracy
to zero. The effect is therefore variable in magnitude but replicated in
direction and causally dependent on correctly aligned verifier outcomes.

Value-diverse admission plus cost-free active direction screening is promoted
as the current strategy-memory mechanism. Dynamic slot allocation remains
disabled. The next frontier is to make the gain less seed-sensitive, using
matched tiny races and no additional context capacity until the source of the
7072/7073 variance is localized.

## Seed-sensitivity localization and exact-prefix correction

Physical experience, policy-perturbation, and context-proposal random streams
were separated and independently swappable. Starting from the weak 7072
trajectory, none of the three single swaps and none of the three pairwise swaps
reproduced the strong 7073 result. Overriding all three streams reproduced the
strong 54-round trace bit-for-bit even while the bookkeeping/evaluation seed
remained 7072. The ignition is therefore a genuine three-way trajectory
interaction, not one lucky data stream or an omitted RNG source.

An initial exploration-clone race appeared promising but was invalid as a
successive-halving claim: setting four rounds per phase compressed all three
phases into twelve rounds, whereas the extension used eighteen consecutive
rounds per phase. The screen was not a prefix of the extension.

The harness now supports `--max-physical-rounds`. It stops an otherwise
unchanged curriculum, marks the report `prefix_only`, and makes graduation
impossible. A corrected 12-round report matched the corresponding full trace
bit-for-bit.

The corrected evidence also rejects information density as a sufficient early
selector. By round 12, a later non-compounding clone had six informative pairs
and four action patterns, exceeding a later successful clone with five and
three. Round 12 contains only the old-equal phase, so it cannot measure
switching or return retention. Future racing must either wait until the exact
prefix reaches the old-return phase (round 37 or later in this schedule), or
pre-register an interleaved curriculum that exposes acquisition, switching,
and return within every short prefix.

## Interleaved-curriculum rejection

The pre-registered interleaved schedule preserved the same three context
weights and total physical-round budget as the blocked schedule, but alternated
`old_equal`, `reliability_dominant`, and `old_return` every round. This repaired
the measurement problem: a 12-round prefix covered every context four times,
matched the corresponding full trace bit-for-bit, and was explicitly barred
from graduation.

The behavioral comparison was negative. On the formerly strong all-7073
value-diverse trajectory at 54 physical rounds, reward-informative soft pairs
fell from 56.6% to 5.7% and bits per informative pair rose from 214.4 to
2,144.0. Reliability target accuracy fell from 41.7% to 9.7%, while old-return
target accuracy fell from 95.8% to 9.7%. Generic persistence, parity, and
inherited-primitive retention gates remained intact, so this is a sample-
efficiency failure rather than a harness error.

Simple interleaving is rejected. Exact-prefix accounting is retained. Future
clone selection must either pay for a prefix that reaches a genuine return
phase, or find a schedule that preserves the contiguous trajectory required
for ignition while producing an earlier verifier-only retention signal.

A six-round-per-context cyclic control supplied a useful gradient rather than
a rescue: it reached 20.8% reward-informative pairs and 584.7 verifier bits per
informative pair, better than one-round cycling (5.7%; 2,144.0) but well below
the 18-round blocked schedule (56.6%; 214.4). Old-return target accuracy was
only 4.2%. The evidence supports a contiguity requirement. Do not spend further
budget optimizing schedule rearrangements until a proposed variant can preserve
the contiguous acquisition path and improve a pre-registered verifier-only
selection metric.

## Read-only shadow strategy selector

The next experiment preserved the successful blocked training trajectory and
added a read-only phase-boundary challenge. After round 18, every stored latent
strategy was physically evaluated on held-out old-equal and reliability-
dominant contexts. No audit outcome was returned to the learner and no model,
strategy-memory, or disk state changed. Four seeds cost 112 held-out logical
lifetimes and 960 selection verifier bits per clone, with zero optimizer updates
and exact tensor/physical parity.

This was necessary because ordinary early signals failed on the blind 7074
population. Early cumulative reward advantage selected non-transferring clone
7090, while raw behavioral diversity favored harmful clone 7093. The shadow
audit's conservative minimum context-mean advantage pruned clones 7091 and
7093, but tied 7090 and 7092 at +2.08 reward points. Only 7092 showed context-
specific best stored strategies (two of four audit seeds); it later produced
the population's strongest safe old-return gain (+3.01 reward points and
+18.06 target points). Clone 7093 later failed with negative return.

The selector is promising but remains retrospective. It is not promoted until
the pre-registered lexicographic rule—minimum cross-context advantage, then
context-specific specialization—selects a useful clone on a fresh physical
stream. If positive, reward/context shuffling must break the selection result
before a causal claim is accepted.

The prospective gate passed on fresh physical stream 7075. Among clones
7100–7103, the primary score selected 7101 at +10.42 reward points; the other
scores were +2.08, 0.00, and +1.04, so no tie-break was needed. Only selected
7101 and pre-fixed lowest-ID control 7100 were extended. The selected clone
reached 73.6% reliability target accuracy and 81.9% old-return target accuracy,
versus 8.3% and 2.8% for control. Return reward advantage was +5.79 points
versus +0.23.

Adversarial controls localized causality. Shuffling physical reward alignment
failed the gate and reduced return target accuracy to 36.1%. Shuffling latent
strategy keys at transfer reduced return accuracy to zero. Every arm retained
binary and four-rule capability. The result is therefore not explained by
catastrophic forgetting, a generic lucky controller seed, or strategy values
used independently of their learned addresses.

The read-only shadow audit is promoted as the current four-clone population
selector. It trades 112 held-out logical lifetimes and 960 selection verifier
bits per clone for no extra optimizer updates and leaves the 18-round training
trace bit-identical. Replicate magnitude once on a new physical stream before
increasing population or audit scale.

Replication on fresh physical stream 7076 narrowed the claim. The unchanged
round-18 selector chose clone 7112 at +6.25 shadow reward points. It later
outperformed fixed control 7110 on reliability target gain (+20.83 versus
+4.17 points), replicating acquisition direction. Old-return magnitude did not
replicate: selected 7112 gained only +1.39 target points and +0.23 reward
points. The round-18 shadow audit is now scoped as an acquisition selector.

A read-only round-36 bank audit was also rejected. Although it left every
training trace bit-identical, it ranked weak 7100 above strong 7101 on stream
7075. Strategy inventory ceiling is not a reliable retention proxy.

The first six genuine old-return rounds provide the current candidate retention
rung. Mean verifier reward advantage over those six rounds retrospectively
selected the eventual best safe clone on streams 7074, 7075, and 7076. One
round was noisy and three rounds tied two clones on 7074. The next engineering
gate is exact resumable prefix state, followed by a prospective 18-round shadow
screen → 42-round return screen → 54-round winner ladder.

## Exact resumable race state

The physical trainer now saves and restores the complete learning trajectory:
controller residual, bounded physical disk banks, current row/query stream,
latent strategy bank, context encoder and optimizer, explicit RNG states,
reward signature, intervention flags, trace, and all verifier/accounting
counters. Resume files are configuration-validated and prefixes remain
non-graduating.

An initial 3→6-round audit matched uninterrupted execution recursively and
bit-for-bit. The promoted 18→54 audit used the real strong four-bank,
four-strategy, sixteen-proposal configuration with the four-seed shadow audit.
The resumed and uninterrupted final payloads were identical across every
tensor, physical-memory row, strategy statistic, encoder parameter, generator
state, trace, phase summary, gate, and accounting field. Both reproduced 81.9%
old-return target accuracy.

Exact resumability is promoted. The planned 18→42→54 population ladder can now
save real candidate compute instead of replaying winning prefixes.

## Prospective two-stage population ladder

The full pre-registered 18→42→54 ladder was tested on fresh physical stream
7077 with clones 7130–7133. The round-18 read-only shadow screen advanced 7131
and 7133. Their conservative acquisition scores were nearly tied at +2.08333
reward points; the other two clones scored zero.

Mean verified reward advantage over the first six old-return rounds then
separated the survivors cleanly. Clone 7131 averaged +1.389 reward points and
never fell below frozen. Clone 7133 averaged -4.861 points with a -8.33-point
worst round. Only 7131 advanced.

At round 54, selected 7131 reached 18.06% reliability accuracy versus 5.56%
frozen, and 13.89% old-return accuracy versus 5.56% frozen. Fixed control 7130
remained at zero in both transfer phases. Both clones retained the inherited
binary and four-rule primitives and passed all persistence and parity gates.
Shuffling physical verifier-reward alignment made the gate fail and collapsed
both transfer advantages to zero. Shuffling latent strategy keys also
collapsed both to zero.

The production ladder used 132 physical rounds instead of 216 for four full
clones, saving 38.9% of training compute. The first validation additionally
completed the fixed control, bringing the audited cost to 168 rounds, still
22.2% below exhaustive training. Read-only selection added 448 held-out
logical lifetimes and 3,840 explicitly separated verifier bits, with no
optimizer updates. All resumed trace prefixes matched exactly.

The ladder is provisionally promoted as a compute-efficient population
mechanism. This fresh stream supports preservation of a better stochastic
trajectory, not guaranteed selection of an exceptional one. Replicate the
entire ladder once before increasing population size or training budget.

## Successive-halving replication boundary

The first frozen replication on stream 7078 selected clone 7140, which was also
the pre-fixed lowest-ID control. The clone itself repeated safe learning:
+9.72 reliability target points and +6.94 old-return points with both inherited
primitives retained. But the selected-versus-control gate was
comparative-inconclusive because winner and control were identical. No
post-hoc substitute was used.

A corrected validation rule defined the control as the strongest clone pruned
at round 18 and was tested on unseen stream 7079. The two-of-four ladder chose
7152, which gained +11.11 reliability and +20.83 return target points. However,
slightly lower-ranked pruned clone 7151 later gained +76.39 return points. Its
first six genuine return rounds already strongly identified it, so the failure
was not the retention selector: round-18 pruning had discarded a delayed
sleeper. Two-of-four acquisition pruning is rejected as too aggressive.

The smallest repair—advance three of four—was then tested on unseen stream
7080. The selected clone and eliminated control both produced zero reliability
and zero return advantage. Post-gate completion of the other two survivors
found no valid hidden winner: one had zero return, while the other had only
small target gains and failed the reward-based full gate. This population
contained no acceptable learner.

Do not scale population size. The current evidence supports a conservative,
abstaining ladder: advance three of four at round 18; at round 42 continue only
if mean verified reward advantage over the first six return rounds is strictly
positive and worst-round advantage is non-negative; otherwise terminate the
population. Retrospectively this preserves the useful trajectories on streams
7077–7079 and correctly abstains on 7080. Prospectively test that frozen rule
before promotion.

## Prospective abstaining race

The conservative abstention rule was frozen and tested on unseen physical
stream 7081 with clones 7170–7173. Round-18 shadow advantages ranked 7170,
7173, and 7171 above zero-score eliminated control 7172. At round 42, 7170
earned continuation with +2.778 mean verifier-reward points over the first six
return rounds and no harmful round. Clone 7171 was weaker at +2.083; 7173 was
zero.

Selected 7170 completed with 59.72% reliability accuracy versus 13.89% frozen,
and 51.39% old-return accuracy versus 13.89% frozen. Eliminated control 7172
remained at zero in both transfer phases. Both inherited primitives and every
full gate passed. Post-gate completion of the other survivors confirmed that
neither gained anything on return, so the six-round selector chose the only
substantial acquisition-and-return trajectory.

Reward-alignment shuffling failed the gate and collapsed return to zero. A
one-time latent-key shuffle did not collapse the learner: it improved immediate
reliability but reduced return from 51.39% to 36.11%. This bounds the claim.
Learned key/value association affects retention, but the intervention permits
36 subsequent adaptation rounds and is not a persistent query-time memory
ablation.

Production cost was 156 instead of 216 physical rounds, a 27.8% saving.
Read-only selection cost remained separate: 112 held-out logical lifetimes and
960 verifier bits per clone, with no optimizer updates. Exact prefix
continuation and all persistence/parity checks passed.

The conservative abstaining race is provisionally promoted. Replicate once on
another unseen stream before increasing population size. Replication can pass
by selecting a valid winner or by correctly abstaining when the completed
eliminated control is also invalid.

Replication on unseen stream 7082 rejected the round-18 prune. All three
survivors had exactly zero reward advantage over the first six return rounds,
so the abstention rule correctly stopped them. But eliminated zero-shadow-score
clone 7182 later gained +4.17 target points in both reliability and return and
passed the full gate. Its first six return rounds already averaged +1.389
reward points with no harmful round, so the return selector would have retained
it if early pruning had not removed it.

The supported boundary is now precise: the six-round return selector remains
useful, but no round-18 shadow score safely excludes delayed learners. Remove
early training elimination. The next prospective ladder trains all four clones
through round 42, then selects or abstains using the unchanged positive,
non-harmful return threshold. It saves 16.7% of training rounds when selecting
and 22.2% when abstaining, while also removing 448 shadow lifetimes and 3,840
selection verifier bits from production allocation.

The return-only ladder was then tested on unseen stream 7083. Clone 7191 had
the strongest six-round return signal (+7.639 reward points, no harmful round)
and completed at 88.89% old-return accuracy, +69.44 points over frozen. Full
completion of every clone found a genuine trade-off: 7193 was the better
reliability learner (+44.44 target points versus +25.00), while 7191 was the
better return learner (+69.44 versus +40.28). Neither dominated both axes, so
the pre-registered “best under both” validation gate did not pass literally.

Return-only ranking is therefore too narrow for compounding learning, just as
round-18 acquisition-only pruning was too narrow. The next minimal selector is
maximin verified reward at round 42: the minimum of mean reliability-phase
reward advantage and mean first-six-return reward advantage. Both components
must be positive and no return round may be harmful; otherwise abstain. This
uses no new experience or privileged labels. Retrospectively it selects 7191
on stream 7083 with a +3.935-point conservative score versus +3.472 for 7193,
explicitly balancing new-context acquisition and return retention. Test it
prospectively with all clones surviving to round 42.

The balanced maximin selector was tested prospectively on unseen stream 7084
with clones 7200–7203 and no shadow audit. At round 42, clone 7203 had +1.620
reliability reward points and +3.472 first-six-return points, for a +1.620
conservative score with no harmful return round. The next-best balanced clone,
7200, scored +1.389.

Full post-gate completion of every clone confirmed the selection. Clone 7203
retained a +1.620 final reliability reward gain and +2.083 return gain; its
final conservative gain of +1.620 exceeded 7200's +0.926 and the zero scores
of 7201/7202. It reached +19.44 reliability target points and +13.89 return
points. All inherited retention, parity, persistence, and exact-resume gates
passed.

The production path used 180 rather than 216 physical rounds (16.7% saved) and
eliminated all shadow selection cost. The maximin selector is provisionally
promoted as the first allocation objective aligned directly with both new
learning and retained reuse. Replicate unchanged before increasing any scale
or difficulty axis.

Unchanged replication on unseen stream 7085 passed. At round 42, clone 7211
had +1.157 reliability reward points and +2.778 return points, for a +1.157
maximin score. Clone 7213 had stronger return (+4.861) but a lower acquisition
floor (+0.926); 7210 lacked return gain and 7212 was harmful.

Full completion preserved the ranking. Selected 7211 finished with +1.157
reliability and +6.250 return reward points, exceeding 7213's +0.926
conservative gain. It reached +6.94 reliability target points and +63.89
return points. Reward-alignment shuffling reversed reliability reward gain to
-3.472 points and failed the full gate. Inherited skills, exact resumability,
and persistence passed for the intact winner.

Balanced maximin population selection is promoted across two prospective
streams. It saves 16.7% production training rounds and requires no shadow
selection experience. The next scientific gate is a transfer ledger: test
whether the selected weights reduce unique verifier bits and updates needed
for a genuinely later held-out primitive versus the shared parent, a fresh
matched controller, and the same selected architecture with weights reset.

The first held-out transfer ledger used a fourth generic memory-utility
statistic: row novelty, where low-novelty memory rows are redundant.  Every
sub-minute arm used eight black-box updates, 1,792 generated logical
lifetimes, 1,536 candidate verifier bits, zero replay, and 23.2–28.3 seconds.
Binary mapping and four-rule retention passed throughout.

The selected stream-7085 state did not transfer faster.  At 10% novelty its
global weights finished at 89.58% verified reward versus 93.23% for the shared
parent.  Retrieving its saved context-indexed strategy on an unseen seed
improved over the global residual but still lost to the parent, 95.83% versus
96.35%; the parent was already across the first two gap thresholds at prefix
zero.  At 15% and 20% novelty neither selected-state arm produced a causal
novelty advantage.  Retained population weights are therefore rejected for
this later primitive; the maximin selector remains valid for its demonstrated
curriculum but has not yet produced compounding forward transfer.

Curriculum localization found why the mixed task was weak: the
redundancy-only action agreed with the realized mixed target only 19.5–25.0%
of the time, near the 16.7% six-way chance baseline.  A deterministic
capacity-three, zero-noise, pure-redundancy atom was therefore tested.

The atom exposed and corrected one accounting bug: the hand-coded redundancy
oracle had been included among baseline controls, making the available oracle
gap zero.  It is now an upper bound; baselines contain only random, fixed,
skip, recency, frequency, and reliability policies.

On unseen seed 7306, resetting the complete tiny replacement policy produced
the first causal redundancy-learning signal: verified reward rose from 71.88%
to 86.46%, above the strongest non-redundancy control at 85.42%, and fell to
76.04% when novelty was shuffled.  Residual-only reset, the parent, and both
selected-state variants stayed below the strongest control.

This mechanistic result is not promoted to a longer run because complete
policy reset discards the old memory-utility skill.  The next gate is a
zero-initialized, context-retrieved full-feature replacement residual.  It must
learn the redundancy atom while remaining an exact no-op on old contexts.

The context-retrieved adapter series then established retention plumbing but
did not pass its learning gate.  Each sub-minute run used eight updates and
completed in 17.95–22.62 seconds of measured experiment time.

The first 12-dimensional mean/std context key had overlapping similarity
tails: new↔new fell to 0.9722 while new↔old reached 0.9844.  Adding generic
per-feature activity bits produced an 18-dimensional key with new↔new at least
0.9936 and new↔old at most 0.9294 in a 50-by-50 preflight.  Across all later
runs, held-out new contexts activated retrieval, three old utility contexts
rejected it, old replacement scores remained bit-identical, and every external
strategy save/reload was exact.  Context-local adaptation without forgetting
is therefore mechanically demonstrated.

Eight-dimensional full-residual SPSA failed.  A two-dimensional arbitrator
that learns old-policy suppression plus novelty weight produced one partial
seed-7313 signal (+2.08 reward points, 5.21-point novelty-shuffle cost), but it
did not replicate.  The trace exposed and fixed unsupported drift from
candidate ties by making the unchanged center win unless another candidate
improved verified reward by over `1e-6`.

Per-bank REINFORCE reduced candidate verifier use from 1,152 to 384 bits per
arm and produced nonzero gradients on every update, but its adapter norm
reached only 0.058 and behavior stayed flat.  Unit-normalized gradients on
seed 7317 changed behavior slightly (84.38% to 84.90%) while still missing all
stable thresholds and novelty causality.  No configuration earned a
three-minute promotion.

The supported frontier is now credit assignment, not routing or retention.
The next sub-minute rung is a passive action-conditioned success critic trained
only from visible statistics, attempted actions, exact propensities, and scalar
verified outcomes.  It must pass calibration, reward-shuffle, and
missing-evidence controls before influencing either actions or compute.

That passive critic was implemented and tested without any action influence.
Each corrected seed used 512 unique attempted lifetimes, 1,536 unique verifier
bits, eight optimizer updates, zero replay, and about 11.3 seconds.  Every
logging propensity was exact, gradients were live, save/reload was exact, and
binary/four-rule retention passed.

Seed 7322 showed a weak ranking signal: intact concordance was 0.588 versus
0.501 under reward shuffling, with ECE 0.0092.  Its Brier gain over the
empirical-rate predictor was only 0.00012, however, and an action-only arm
ranked slightly better at 0.616.  Unchanged seed 7323 did not replicate:
intact/action-only concordance fell to 0.499/0.514 and no learned arm beat the
constant Brier baseline.  A separate 256-lifetime audit showed that uniform
logging still covered all actions and did not increase outcome variation
relative to the registered epsilon mixture.

This aggregate-outcome critic is therefore rejected for a three-minute run.
The architecture remains passive and available, but it has not earned the
right to control answers, memory, or compute.  The next gradual rung is a
shorter-horizon attempted-action prediction: one immediate verifier event
first, then longer outcome horizons and cross-context prediction one axis at a
time.

The horizon-one localization produced the next verified breakthrough.  An
action-only critic showed an initial ranking signal but missed the Brier gate.
Adding the controller's own next-query read evidence localized the useful
surface to only four generic values: confidence, top-two margin, selected
strength, and occupancy.  A 32-hidden-unit passive critic then learned from
scalar verifier outcomes with no correct-action or semantic labels.

At the promoted sub-minute budget, every seed used 768 unique logical
lifetimes, 768 unique verifier bits, 12 updates, zero replay, and 6.0–13.8 CPU
seconds.  Three of four seeds passed every gate:

- 7334: +0.00893 Brier over constant, 0.913 concordance;
- 7335: +0.01112 Brier, 0.937 concordance;
- 7336: +0.01033 Brier, 0.964 concordance, missed only ECE by 0.002;
- 7337: +0.00866 Brier, 0.802 concordance.

Adversarial evidence permutation collapsed concordance from 0.964 to 0.504
and from 0.802 to 0.470, costing 0.00933 and 0.01009 Brier.  Reward-shuffled
critics did not match intact Brier; zero-evidence critics stayed at 0.5
concordance.  All action-coverage, persistence, binary-retention, and
four-rule-retention gates passed.

The supported claim is deliberately narrow but new: the unified controller
can acquire a calibrated, causally evidence-dependent immediate-success model
from fewer than one thousand scalar outcomes, using only its own abstract
memory-read state.  The critic remains passive.  The next rung is shadow
compute allocation—evaluate whether it can rank the verified value of one
extra thought/read step before allowing it to influence latency or behavior.

Shadow compute allocation exposed a distinction between predicting absolute
success and learning the value of extra computation.  A two-action success
critic trained from 720–1,440 attempted outcomes beat action-rate Brier and
every causal control, but remained biased toward always reading; exact choice
accuracy stayed below 59% and calibration failed.  More data did not repair
the decision boundary.

A direct inverse-propensity advantage objective solved that conversion without
counterfactual labels.  For a uniformly logged read/no-read action, the
learner regressed
`sign(action) * (observed_utility - running_baseline) / propensity`, whose
conditional expectation is the verified read-minus-no-read advantage.  It
saw only four generic read statistics, its attempted action/propensity, the
small generic read cost, and one scalar outcome.

The 201-parameter head passed twice at 720 bits.  A 57-parameter near-match to
the inherited read gate retained a causal signal but missed the gate.  The
intermediate 105-parameter head passed and replicated:

- seed 7424: 69.0% choice accuracy, +19.24 utility points over always-read,
  59.7% oracle-gap capture;
- seed 7425: 70.6% choice accuracy, +18.47 utility points,
  60.2% gap capture.

Both crossed the primary allocation thresholds at the first measured
120-bit prefix and stayed above them through 720 bits.  Reward-shuffled,
feature-shuffled, and zero-evidence controls captured essentially none of the
oracle gap; episode-evidence shuffling made the learned policies worse than
the fixed baseline.  Retention, persistence, gradients, and latency passed.

The inherited 49-parameter gate remains stronger on this already mastered
task: a saved matched audit measured 95.2%/82.5% choice accuracy and
97.4%/81.4% gap capture on seeds 7424/7425.  Its historical training budget
was 81,920 contexts and 160 updates.  Therefore keep its weights; promote the
advantage objective as the sample-efficient blueprint.  The next exact rung
is width 8 (57 parameters) at 1,440 fresh bits, one axis only, followed by an
unchanged replication if it passes.

That final width-8 capacity test ran on GPU with 1,440 fresh verifier bits and
24 updates. It completed in 0.45 seconds, reached 62.70% compute-choice
accuracy, improved utility over always-read by 11.24 points, and captured
35.73% of the oracle gap. All causal controls, retention, persistence, and
latency checks behaved correctly, but the pre-registered 65% choice gate
failed. The width-8 fork is therefore closed without replication or more
training. Retain the replicated 105-parameter advantage blueprint and the
stronger inherited production weights. The next transfer ledger must test
advantage learning on a genuinely novel compute decision rather than continue
optimizing this mastered read/no-read surface.

## 2026-07-26 — third-generation option composition

The verified two-, three-, and four-action lineage was extended to a fifth
physical action. The four-action hierarchy remained frozen and was exposed as
one opaque option. A new binary router learned old-hierarchy versus fifth
action; the matched flat control relearned all five actions.

Ordinary randomized bandit regression, extra replay, richer features alone,
and a cost curriculum did not replicate. A population-experience fork did:
temporary clones attempted the competing actions on the same generated
context, every scalar verifier outcome was charged, and the router regressed
the observed fifth-minus-old advantage. No semantic label, correct-action
hook, task ID, or hidden game state entered the learner.

The frozen configuration used 3,840 unique logical lifetimes and 256 replay
updates per arm. Composition consumed two verifier bits per lifetime; flat
reset consumed five. Stable means the first measured +2-point threshold after
which every remaining prefix also passed.

| Seed | Composition stable bits | Flat stable bits | Transfer ratio |
|---:|---:|---:|---:|
| 8083 | 3,360 | 15,600 | 4.64× |
| 8084 | 3,600 | 5,400 | 1.50× |

Independent audits used eight unseen 2,040-context streams and a separate
2,400-outcome randomized confirmation set. Seed-8083 utility improved from
0.78754 to 0.82701; seed-8084 improved from 0.78258 to 0.82962. Every stream
improved. Router-feature shuffling reduced utility to 0.74274/0.73534 and
decision reversal to 0.49241/0.48570. Randomized lower-95% gains were +3.52
and +1.49 points. Binary and four-rule retention, exact reload, corruption
detection, and parent-lineage survival all passed.

This is the third consecutive verified generation of reusable option growth
and the first to show that counterfactual scalar experience shared across a
temporary clone population can unlock a latent action relation that
single-action feedback used unreliably. The persistent artifact remains one
controller plus external verified skill memory.

## 2026-07-26 — fourth-generation option composition

The verified five-action hierarchy was retrieved as one option and extended
with a sixth physical action. A matched flat control learned six actions.
Paired population experience remained the only successful signal: every
attempted scalar outcome was charged, with no semantic or correct-action
labels.

The first race failed at 3,840 unique contexts. At 7,680 contexts and 512
updates, composition crossed the target earlier than flat but did not remain
above it. Lower learning rate, EMA, and swapping between the two verified
five-action parents failed. A discarded probe had used 2,000 updates, so the
final evidence-backed fork retained the same 7,680 contexts and increased
experience replay to 2,048 updates.

That configuration passed and replicated:

| Seed | Composition stable bits | Flat stable bits | Transfer ratio |
|---:|---:|---:|---:|
| 8098 | 5,280 | 10,800 | 2.05× |
| 8099 | 5,760 | 6,480 | 1.125× |

Final composition utility rose from 0.83166 to 0.88497 and from 0.83447 to
0.88922. Two independent eight-stream audits measured gains of +5.37 and
+5.74 points. Every stream improved. Feature shuffle reduced utility to
0.73208/0.73689, reversal to 0.41300/0.41127, and randomized lower-95% gains
were +3.18/+4.51 points. Retention and full persistence integrity passed.

This is the fourth consecutive verified option generation. It also establishes
a second compounding resource axis: verifier/sample efficiency can improve
through greater internal reuse of fixed experience, but processing cost must
now be optimized explicitly rather than treated as free.

## 2026-07-26 — adaptive replay without sample-efficiency loss

A task-agnostic replay controller now treats replay 16 as a maximum and stops
processing a new batch when full observed-experience Smooth-L1 loss reaches
0.14. It uses no additional verifier outcomes or privileged information.

Prospective matched fixed/adaptive pairs received identical 7,680 logical
lifetimes and 15,360 composition verifier bits:

| Seed | Stable bits adaptive/fixed | Updates adaptive/fixed | Saving |
|---:|---:|---:|---:|
| 8106 | 5,760 / 5,760 | 1,237 / 2,048 | 39.6% |
| 8107 | 6,000 / 6,000 | 1,079 / 2,048 | 47.3% |

Wall time fell from 3.77 to 2.24 seconds and from 4.18 to 1.99 seconds.
Replayed examples fell by the same update ratio. Adaptive asymptotic utility
was slightly lower than fixed but remained safely above the capability gate.

Both adaptive checkpoints passed independent eight-stream improvement,
randomized confirmation, feature shuffle, reversal, binary/four-rule
retention, exact persistence, corruption detection, and lineage-survival
audits. This is the first verified result in which the controller allocates
its own processing from a generic learning-state signal while preserving the
exact stable verifier-bit threshold.

## 2026-07-27 — cross-generation learned replay stopping

Replay traces from four five-action streams trained a tiny task-agnostic probe
to predict verified loss reduction over the next eight updates. On two
held-out six-action streams, independently initialized probes reached
`0.9140/0.9173` correlation and improved MAE over the mean baseline by
`23.25%/24.40%`; shuffled-target correlation remained `0.0471/0.0502`.
Thus reusable information for compute allocation exists in generic learning
state.

Behavioral integration exposed an alignment failure. A one-update policy was
myopic and lost mastery. Eight-update patience produced one promising smell
test, but a prospective single predictor failed one of two seeds through
overconfident stopping. A unanimous ensemble preserved capability but saved
only `1.95–9.77%` of updates. The ensemble mean was best: it saved
`12.89%/17.97%`, preserved the capability gate, and changed stable verifier
bits from `8,400→8,160` and `6,480→7,200`. The latter is an `11.11%`
regression and misses the frozen `10%` sample-efficiency limit.

No learned-stopper checkpoint is promoted. The precise frontier is now to
predict long-horizon held-out verifier improvement and retention per unit
compute, rather than local replay-loss reduction.
## 2026-07-27 — Retained context accelerates a later visual composition

The unified controller's `visible_context_xor` successor task is the first
paired transfer result in this branch where inherited knowledge improves the
stable learning threshold while all inherited gates remain passed.

| Seed | Old binary parent | Integrated binary+context parent | Ratio |
|---|---:|---:|---:|
| 8411 | 76 updates / 21,888 bits | 68 updates / 19,584 bits | 1.118× |
| 8412 | 80 updates / 23,040 bits | 72 updates / 20,736 bits | 1.111× |
| 8413 | 72 updates / 20,736 bits | 64 updates / 18,432 bits | 1.125× |

Each update contained 32 new, eight binary-replay, and eight context-replay
lifetimes. Stable means the full new-skill causal gate and applicable retention
gates passed at that prefix and every later measured prefix. The integrated
controller saved exactly eight updates, 256 new lifetimes, 384 total lifetimes,
and 2,304 verifier bits on all three seeds. Median transfer ratio: **1.118×**.
The fresh learner remained at chance through 80 updates. The promoted
seed-8413 run also passed operation-cue removal, blank-vision, pixel reversal,
dual-retention, frozen-base, checkpoint-reload, and shuffled-teacher controls.
