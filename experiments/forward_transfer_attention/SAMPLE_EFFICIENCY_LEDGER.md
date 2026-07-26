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
