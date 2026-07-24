# Tiny experiment results — 2026-07-23

The sub-minute ladder produced a robust diagnostic milestone.

## Negative integration probes

- 128/128 held-out raw-write, compact-row, and recall probes were at or below
  the 53.9% empirical majority baseline for both seeds.
- Two- and four-shot probes did not improve the result.
- Intermediate object, feedback, support-sequence, and pre-write states did
  not make the derived rule decodable at this scale.

## Positive representation probe

The two-stage disposable probe was run on lifetime-disjoint data. It first
decoded the two ingredient identities, then learned their composition. At 512
lifetimes, composition reached 73.63%. At 1,024 lifetimes:

| seed | first identity | rewarded identity | composition | shuffled-label mean |
|---:|---:|---:|---:|---:|
| 41 | 96.78% | 86.33% | 83.40% | 46.58% |
| 97 | 95.80% | 84.96% | 81.45% | 50.81% |

The true-ingredient composition was 100% in both runs. This separates the
problem cleanly: representation and compositional binding are present, while
the integrated writer has not learned to store them. The next experiment must
target the writer interface, not scale general training.

## 2026-07-23 writer-interface continuation

The GPU was restored and verified as an RTX 5090 (`torch 2.13.0+cu130`, CUDA available).

### 128-lifetime event-binder bootstrap (color feedback)

- 31.3 s, 128 training lifetimes, one epoch, event binder only.
- Binder gradients were live (0.2259) and the write auxiliary head reached 57.8% on the training batches.
- Held-out raw-write probe: 42.19% (majority baseline 53.91%); compact-memory and recalled probes also stayed at/below baseline.
- Interpretation: apparent training improvement was not reusable rule storage.

### 64-lifetime repeated-set overfit control

- Three repeated epochs; write auxiliary training accuracy rose 62.9% → 66.0%, while loss fell 0.686 → 0.621.
- Held-out raw-write probe: 50.0% test accuracy versus 54.69% empirical baseline; train MLP reached 92.2%.
- Interpretation: the writer can memorize nuisance structure but does not generalize the rule. This is a clean overfit signature, not a successful capability.

### Write-state pair ablation (new tiny architecture test)

Added an optional `--event-binding-write-pairs` path that explicitly provides products/differences between the final recurrent write state and each event snapshot. One 128-lifetime epoch produced 46.68% auxiliary training accuracy, residual RMS 0.0032, and severe behavioral degradation (temporal future AUC 0.242). It is not a candidate for escalation; the residual stayed effectively closed and the change harmed retention.

Adversarial testing was not triggered because no candidate passed the positive gate. The next high-ROI experiment should localize the writer's supervision path rather than increase training time: verify that the temporary rule loss is attached to the intended support write and that the rule head sees the same event-state distribution as the proven two-stage probe.

### Auxiliary-loss weight 10x

- Same 128-lifetime, one-epoch setup with the temporary rule loss weighted 10 instead of 1.
- Training write accuracy was 57.0%; held-out raw-write decoding was 50.0% versus a 53.9% baseline.
- The temporal behavioral AUC fell to 0.245 and residual RMS remained only 0.0023.
- Conclusion: increasing the auxiliary weight is not the missing ingredient; it damages behavior without creating reusable writer information.

### Exact integrated-binder capacity controls

Using the same frozen sensory snapshots as the successful pairwise diagnostic, but the exact `EventSnapshotWriteBinder` implementation:

- 256 lifetimes / 300 steps: 66.4% train, 49.2% test (48.4% baseline).
- 1,024 lifetimes / 2,000 steps: 53.7% train, 49.0% test (49.6% baseline).
- 4,096 lifetimes / 4,000 steps: 50.7% train, 50.2% test (49.8% baseline).
- Opening the no-op gate: 59.1% train, 48.6% test.
- Opening the zero-initialized relation projection: 56.9% train, 47.6% test.

This rules out a simple gate/zero-initialization explanation. The integrated module is not learning the relation under the current training setup, even though the disposable pairwise classifier does. The next architectural diagnostic should compare the exact relation head against the pairwise probe with the same classifier and optimization, before any full-agent training.

The direct-bound variant (bypassing the write gate and classifying the relation latent directly) also stayed at 51.9% train / 47.9% test at 1,024 examples. The problem is therefore not just the residual gate; the integrated relation implementation needs parity testing against the proven pairwise head or supervised warm-start.

### Direct supervision on relation latent

Attaching the temporary rule head directly to `EventSnapshotWriteBinder.last_bound` (rather than to the final raw write key/value) produced 50.4% training accuracy, 0.829 auxiliary loss, and severe behavioral degradation in the 128-lifetime pilot. This does not rescue the one-epoch path; the next honest move is to train the relation module with the exact pairwise classifier/head used by the proven diagnostic, then transfer only the learned relation representation.

### Pairwise-head parity pilot

The exact proven pairwise diagnostic head was attached to the binder's stored 9-way relation features. At 128 lifetimes/one epoch it reached 50.8% training accuracy and 0.743 auxiliary loss, with behavior collapsing (temporal future AUC 0.203). This is still below the ignition scale; it is not evidence against the head, but it confirms that tiny integrated runs cannot discover the relation at this data scale.

### 4k real-loop parity attempt

At 4,096 lifetimes and one full training epoch (128 updates), the pairwise head still remained at 50.0% write accuracy despite live gradients (6.06) and a nonzero residual (0.078). This differs from the cached frozen-snapshot probe, which reaches 94.5% at 4k/4k. The remaining mismatch is therefore the *online support/write execution path* itself, not simply classifier capacity. The next experiment should cache the exact online relation features and compare them byte-for-byte with the offline probe before another training run.

### Byte-level parity check

For identical seeds, initial study memory, and the first support episode, the base controller and event-binding controller produced exactly identical recurrent event states and relation features (`max diff = 0.0`). Therefore the online/offline mismatch is not a sensory or state tensor difference. The next check must inspect optimization bookkeeping—especially whether the temporary head receives gradients and whether support labels are aligned—before changing the representation again.

### One-batch optimization audit

The actual training path has correct bookkeeping: labels were balanced (17/15), temporary-head gradient norm was 1.06, and binder gradient norm was 0.70. The earlier 4k online run had only 128 optimizer updates (one epoch with batch size 128), whereas the successful cached probe used 4,000 updates. The apparent online/offline discrepancy was therefore an update-budget mismatch, not a representation mismatch. The next fair test is feature caching followed by the same 4,000-step optimizer budget.

### 4k cached parity candidate and causal rejection

The cached pairwise run reached 98.6% train and 73.2% held-out accuracy at 4,000 steps. A true counterfactual reversal audit on the saved model gave 74.5% accuracy on reversed episodes but only 58.7% prediction flips (clean order sensitivity would be near 100%). The candidate is rejected as nuisance exploitation, not a breakthrough. The reversal audit worked as intended and prevents promoting this result.

### Audited augmented pairwise reference

The existing 16k-lifetime, render-augmented pairwise reference remains the strongest valid diagnostic result: 95.02% normal accuracy, 95.61% correctly relabeled reversal accuracy, 93.36% prediction flips, and 4.39% stale-label accuracy. The exact shuffled-label control is 49.80%. This is the benchmark the integrated writer must match; the 4k unaugmented candidate is not comparable because it failed reversal causality.

### Saved-weight reproduction and audit

Reproduced the augmented reference with saved weights: 99.29% held-out accuracy. True reversal: 99.58% accuracy, 99.12% prediction flips, 0.42% stale-label accuracy. Same-budget shuffled-label control: 50.12% test accuracy and 53.19% train accuracy. This is the next real breakthrough: a saved, causal, nuisance-resistant relation model with a chance-level control. It is ready for transfer into the writer; no integrated-memory claim is being made yet.

### Pairwise relation → writer-latent transfer

A 500-step, 128-lifetime projection from the audited pairwise penultimate relation latent (64 dimensions) into the writer width (160 dimensions) reached 100% train and 100% held-out accuracy. Counterfactual reversal remained causal: 98.44% reversed accuracy, 98.44% prediction flips, 1.56% stale-label accuracy. This is the first successful writer-latent transfer rung. The next experiment can safely install this fixed relation-to-write projection behind the no-op gate and test compact-memory retention.

### Fixed transfer writer and memory-boundary audit

The audited pairwise relation and 160-dimensional projection are now available as a frozen
optional writer adapter (`FixedPairwiseTransfer`). Its scalar strength starts at exactly zero:
the real-state 128-lifetime probe measured normal accuracy `100%`, reversed accuracy `100%`,
flip rate `100%`, stale-label accuracy `0%`, and exact zero-strength no-op `true`.

With strength 1.0 inserted into an otherwise frozen controller, 32 held-out spatial lifetimes
gave compact few-shot AUC `0.1934`, 1-shot future accuracy `0.1641`, and compact retention
`0.2344` with intact memory. Shuffling or replacing memory with deterministic garbage reduced
compact few-shot AUC to `0.1406`, 1-shot future accuracy to `0.1094`, and compact retention to
`0.125`. This is an adversarial memory-dependence check, not a capability claim: the frozen
controller is not trained to exploit the new relation writer. The next gate is a tiny
writer/reader training probe with the transferred path, followed by the same reversal and
memory-corruption audits.

### Full-loop transfer pilots: interface, not representation

The first assembled writer/reader pilot used 128 temporal lifetimes (four updates). Gradients
were live (`0.6999`) and the residual was near zero (`3.9e-05`), but AUC stayed at chance; this
was a plumbing check, not a learning test. A 512-lifetime run (16 updates, 137 seconds) kept
the path healthy (`gradient_norm=1.1178`) and moved the learned scalar strength only to
`6.20e-05`. Its held-out temporal compact AUC was `0.3594` and 1-shot future accuracy `0.3594`,
but a behavioral reversal produced identical metrics, so the apparent gain is not yet causal
relation use. Memory corruption was causal: intact compact AUC `0.3594` versus shuffled
`0.1641` and garbage `0.2344`.

A deliberately stronger interface control fixed the audited transfer at strength `1.0` and
trained the reader for 128 lifetimes. The writer residual jumped to RMS `0.9926`, temporal
compact AUC fell to `0.1914`, and 1-shot future accuracy to `0.1875`; retention also degraded.
This rejects the naive “turn the proven adapter on at full scale” path. The current conclusion
is precise: the relation representation and causal transfer are proven, while writer
normalization/gating and reader adaptation remain unsolved. No longer run is justified until a
small scale/normalization control preserves retention and shows a reversal-sensitive behavioral
signal.

### Consolidation localization and raw-write sidecar

The cached transfer probe localized the loss to consolidation. At one support shot, the raw
writer row decoded the rule at `82.8%` linear / `91.4%` MLP held-out accuracy, while the compact
row was only `53.9%` / `56.2%`; recalled vectors were also at chance. A 500-step auxiliary-only
fine-tune of a copied consolidator reached only `53.1–54.7%` held-out, and a 2,000-step run
stayed at `46.1–50.0%`; shuffled-label controls remained near chance. More optimizer steps on
this small cache are therefore not justified.

The minimal architectural control retained the raw writer row as a sidecar after consolidation.
With the frozen reader, intact memory was still memory-causal (compact AUC `0.262` versus
shuffled `0.141`), but reversal was unchanged. A 128-lifetime reader-only sidecar run reached
compact AUC `0.346` and 4-shot future accuracy `0.367`, yet its reversal audit was identical to
forward behavior. The sidecar preserves information but the existing reader does not consume it.
The next high-ROI fork is consequently an explicit reader auxiliary/bootstrap path over the
sidecar—not a larger GPU run or a deeper consolidator.

### Reader-side bootstrap and attention controls

A temporary action-prediction head on the post-read context, trained with the raw sidecar
present, kept gradients healthy but did not improve temporal behavior: held-out temporal
compact AUC was `0.2773`, and 1-shot future accuracy `0.2891`. It improved some legacy
spatial/shape numbers, so this fork is rejected as unrelated reader adaptation rather than
temporal transfer.

Scaling the raw sidecar's retrieval strength over `0, 1, 2, 4, 8` produced no meaningful
improvement; the best tiny MLP probe was only `57.8%`. A generic learned attention reader over
the compact and raw rows reached `56.3%` held-out with a `51.6%` shuffled-label control. Thus
the current evidence does not justify a longer run or a larger reader: the useful raw writer
signal exists, but the existing memory/read interface does not expose it in a learnable way
under these tiny data budgets. The next design fork should be a deliberately separated raw-row
read channel with its own causal audit, not more tuning of the same mixture.

The normalized scale control fixed the audited adapter at strength `0.01` (residual RMS
`0.00993`) and ran the same 128-lifetime reader-only pilot. Gradients remained healthy
(`0.7625`), but the four-update held-out compact AUC was `0.3027` and 1-shot future accuracy
`0.3281`, with no reversal-sensitive learning signal. This is a safe interface scale but not
evidence of capability; the next experiment, if resumed, should use cached relation features
to train the reader at a fair update budget rather than spend full-loop time on another
under-sized run.

### Key-only channel and warm-start integration

A component audit showed that the writer key alone carries the strongest relation signal:
`90.6%` held-out MLP decoding, versus `85.9%` for the value and `89.1%` for the concatenated
row on a matched 128/128 split. A 1,000-step cached bootstrap of the exact 160-dimensional
key projection reached `97.7%` train and `86.7%` held-out accuracy. This is a reusable positive
representation asset.

The projection was normalized and warm-started into the latest-row reader, with the writer
frozen. A direct decision-state bypass was also tested. The tiny behavioral run remained weak
(compact AUC `0.332`, 1-shot future accuracy `0.352`), and the sidecar-aware recalled probe
stayed near chance. Thus the projection learns the relation, but the assembled answer path
still does not use it; no longer run is justified until a query/answer interface can preserve
that signal under held-out replay.

### Latest-row read-channel control

Added an optional generic latest-memory-row channel: a zero-initialized projection and gated
residual into the thought/read context. With the raw sidecar enabled, 128 lifetimes produced
healthy gradients (`0.969`) and safe residual RMS (`0.00996`), but temporal held-out compact AUC
was only `0.295`. The causal audit was unchanged by reversal (`0.320` intact and reversed
compact AUC); shuffled and garbage memory fell to `0.172` and `0.164`. This confirms memory use
but rejects the channel as a temporal solution at this scale. No larger run is justified until
the reader has a more direct supervised route to the relation-bearing row.

### Frozen-writer latest-row rule bootstrap

The first latest-row bootstrap had accidentally allowed generic writer parameters to update, so
it was repeated with the entire event-binding/writer module frozen. The corrected run had zero
writer gradients, preserved a safe residual, and trained only the new reader channel plus a
temporary rule head. Held-out recalled-rule decoding remained chance (`46.9%` MLP at one shot),
while the raw-write control remained above chance (`72.7%` MLP). Behavioral temporal AUC stayed
around `0.295`. This is a clean negative for the latest-row channel, not a writer regression.

### Query-conditioned reader diagnostic

A generic query-conditioned attention reader over compact and raw sidecar keys/values was tested
on 128 disjoint train and held-out lifetimes. It reached only `48.4%` held-out rule accuracy,
while the shuffled-label control reached `52.3%`. This rejects the simple query-attention
integration; the key projection remains a valid offline representation but is not yet connected
to a usable answer/read mechanism.
## Adversarial audit of latest-row feature (2026-07-23)

The latest-row projected feature reached 90.6% held-out MLP accuracy at two
demonstrations on a 128/128 lifetime split. This is an apparent success only;
we applied cheap adversarial controls before considering integration:

| Control | Held-out accuracy |
|---|---:|
| Normal labels/features | 90.6% |
| Shuffled labels | 52.3% |
| Permuted train/test feature rows | 50.8% |
| Zeroed feature | 46.1% |

The controls are at the balanced-task chance floor, while the normal result is
high. This rules out a simple label leak, row-alignment artifact, or constant
feature shortcut for this diagnostic. It does **not** yet prove behavioral
improvement: the signal is still absent from recalled memory and the result is
only for the two-shot projected feature. Required next gates remain causal
reversal, memory-corruption dependence, cross-seed/lifetime replication, and
end-to-end answer accuracy before scaling.

### Standing adversarial gate for future apparent successes

No result is promoted based on accuracy alone. Every candidate must pass (1)
balanced shuffled-label calibration, (2) lifetime-disjoint train/test splits,
(3) feature-row permutation and zero-feature controls, (4) a causal counter-
factual that changes the claimed relation while preserving nuisance factors,
(5) memory corruption/empty-memory degradation when memory is claimed to be
used, and (6) fresh render seeds/controller seeds. A failed control means the
capability claim is withdrawn and the experiment is treated as a diagnostic.

### Zero-start answer-fusion pilot (2026-07-23)

To test whether the representation could reach the action head, a zero-start
fusion head over `(controller state, latest projected key)` was added. Only
this head trained; the writer, consolidator, and projection stayed frozen.
The 32-lifetime run took 22.7 seconds. On its training-seed evaluation,
two-shot future accuracy was 48.4% intact versus 17.2% after counterfactual
order reversal, and empty-memory accuracy was 18.8%. This is the first
reversal-sensitive behavioral effect from the feature path.

However, a fresh controller/render seed did not reproduce the effect: intact
two-shot accuracy was 32.8% and reversed was 37.5%. Therefore the result is
an encouraging plumbing/credit-assignment signal but fails the replication
gate and is not a capability success. The next experiment should increase
diverse lifetime coverage modestly, not model size; it must pass fresh-seed
reversal before any multi-minute escalation.

### Behavioral adversarial control (2026-07-23)

The same latest-key/warm-start controller was evaluated with intact and
counterfactual order-reversed queries (128 held-out lifetimes, two queries).
Two-shot future accuracy was 31.6% intact and 31.6% after reversal; full-reader
accuracy was 26.2% in both cases. Empty-memory future accuracy was 14.1%.
Thus the controller is memory-sensitive, but its answer path does not respond
causally to the reversed temporal relation. The projected-feature decode is a
representation diagnostic, not an agent-capability result. Answer fusion is
blocked until a reversal-sensitive behavioral effect appears.

### 128-lifetime fusion replication (2026-07-23)

Increasing the fusion pilot from 32 to 128 lifetimes took one short epoch and
was evaluated on fresh controller/render seeds. Seed 115 gave 36.7% intact vs
21.1% reversed two-shot compact accuracy. Seed 117 gave 30.5% intact vs 40.6%
reversed. The direction is inconsistent across seeds, so the causal replication
gate still fails. More data alone has not stabilized the answer path; do not
escalate to multi-minute training yet.

### Lifetime-seed bookkeeping correction (2026-07-23)

The prior “different-seed” training run was not actually a different task
sample: the training loop always began lifetime IDs at zero, so `--seed` only
changed RNG state. An explicit `--lifetime-seed-offset` was added, defaulting to
zero for backwards compatibility. A 32-lifetime run on offset `117000000`
produced 43.8% intact vs 21.9% reversed on seed 115, but 32.8% intact vs 37.5%
reversed on seed 117. The corrected cross-seed result still fails the causal
replication gate, but the bookkeeping flaw is fixed and future learning curves
must report logical lifetime ranges explicitly.

### Four-offset fusion training (2026-07-23)

One 128-lifetime epoch cycled four disjoint training offsets
(`0, 117000000, 234000000, 351000000`). Fresh evaluation remained
inconsistent: seed 115 was 42.2% intact vs 18.8% reversed, while seed 117 was
31.3% intact vs 40.6% reversed. The run also reduced old-task retention to
28.1% on the compact path. Diversity alone did not stabilize the fusion and
introduced a retention regression, so this branch is rejected for escalation.

### Entropy-gated fusion (2026-07-23)

The fusion-logit confidence probe showed a useful separation: temporal logits
had higher entropy (0.747) and lower margin (1.51), while spatial/shape logits
were very confident but usually wrong (entropy 0.22–0.24, margin about 4.3).
Applying a task-agnostic gate that enables fusion only when entropy exceeds
0.5 repaired the retention tradeoff.

At 128 held-out lifetimes, temporal accuracy was 51.6% intact vs 34.4%
reversed; spatial old retention was 36.3% and shape 32.4%. Empty memory fell
to 17.97% two-shot accuracy. At 256 lifetimes, temporal was 53.5% intact vs
33.2% reversed, spatial old retention 34.4%, and shape 31.3%. These pass the
current causal, memory-dependence, and two-point retention gates. The threshold
is stored in `fusion_entropy05_candidate.pt`; this is now the first candidate
justified for a modest longer run, still with adversarial checks pre-registered.

### Answer-input probe and supervised head transplant (2026-07-23)

The exact input to the answer-fusion head was independently probed: a linear
8-way classifier reached 68.8% held-out accuracy and a small MLP reached 74.2%
(128/128 lifetime split); shuffled labels were 19.5%. This established that
the query and relation information co-exist before the answer head.

Folding the linear probe's normalization into raw-space weights and
transplanting it into the real fusion head produced causal behavior on fresh
seeds: 68.8%→25.0% and 60.9%→29.7% for intact→reversed two-shot accuracy.
Empty, shuffled, and garbage memory fell to 23.4%, 17.2%, and 20.3%, proving
memory dependence. This is a genuine answer-path integration result.

The remaining gate is retention: at full fusion strength spatial old-task
retention fell from 35.2% baseline to 32.0%, and shape from 32.8% to 27.3%.
Therefore the transplant is promising but not yet a graduation result. A
fusion-strength sweep showed temporal causality persists at 0.25–0.75 scale;
the next tiny experiment must measure retention at those lower strengths and
choose the smallest strength that preserves both causal reversal and old-task
retention.

### Conditional-gate follow-up (2026-07-23)

A learned scalar gate was added over the same controller/latest-row input. A
temporal-only gate run retained causal reversal (67.2% intact vs 21.9%
reversed) but badly harmed old-task retention, as expected from training it
without rehearsal. A mixed temporal/spatial/shape gate run was also poor
(temporal two-shot 29.7% and old retention around 23–34%). The gate is not a
pass; the supervised transplant remains the best current capability artifact,
with retention repair still open.

### Rehearsal-weight control (2026-07-23)

Increasing the old-task rehearsal weight from 2 to 8 did not solve the
instability. Seed 115 remained reversal-sensitive (43.8% intact vs 18.8%
reversed), but seed 117 remained anti-causal (31.3% intact vs 37.5% reversed).
Retention varied from 31.3% to 39.8%, so the higher weight is not a reliable
retention fix. The latest-row fusion branch remains diagnostic-only.

### Fixed-gate fine-tune rung (2026-07-23)

A one-epoch mixed-rehearsal fine-tune from the entropy-gated candidate was run
on four fresh lifetime offsets with the gate fixed at 0.5. At 256 held-out
lifetimes it reached 53.5% intact vs 32.8% reversed two-shot accuracy, matching
the frozen candidate rather than degrading it. This clears the first
progressive fine-tune rung; the next scale can increase lifetime coverage,
not architecture or gate strength, while retaining the same adversarial suite.

### 512-lifetime training rung (2026-07-23)

The first longer fusion-only epoch used 512 lifetimes across 16 disjoint
offsets, with the entropy gate fixed at 0.5. External 512-lifetime validation
gave 58.2% intact vs 31.1% reversed two-shot accuracy, 39.7% old-task
retention, and 11.5% under empty memory. The training report's smaller internal
split was noisy, but the independent adversarial evaluation passed. This rung
is retained; the next increase should be to 1,024 unique training lifetimes,
not a new architecture.

### 1,024-lifetime validation (2026-07-23)

The fixed entropy-gated fine-tuned candidate was evaluated on 1,024 fresh
lifetimes. Two-shot temporal accuracy was 56.1% intact versus 34.2% under
counterfactual reversal; old-task retention was 39.1% in both runs. This is a
stable larger-scale confirmation of the causal effect, with no parameter or
gate changes. The next scale can now be a genuine multi-minute training run
with more unique lifetimes, preserving the same fixed gate and audit controls.

### 1,024-lifetime training rung: adversarial rejection (2026-07-23)

We then trained `fusion_entropy05_long1024.pt` for one epoch over 1,024 unique
lifetimes and audited it on a fresh seed. The adversarial suite found memory
dependence but no causal order use: temporal two-shot accuracy was 34.96% with
intact memory and 35.16% after reversing the order. Empty, shuffled, and garbage
memory fell to 12.89%, 12.30%, and 12.50%, respectively, while spatial and shape
retention were 34.18% and 31.84%. Because reversal did not change behavior, this
larger run fails the causal gate and is not promoted over the audited
`fusion_entropy05_long512.pt`. This is a bounded negative: more data may still
help, but this particular scale increase did not.

### 512-lifetime fresh-seed replication: unstable causal behavior (2026-07-23)

To test whether the 512-lifetime result generalized beyond the original audit
seed, we reran the unchanged `fusion_entropy05_long512.pt` on fresh seed 119.
Memory corruption behaved as expected (two-shot shuffled 14.3% and garbage
11.5%, versus 35.9% intact), but the causal test failed: intact was 35.9% and
reversed was 36.1%. Spatial and shape retention were 33.2% and 33.0%.

Therefore the earlier 58.2%→31.1% result is seed-sensitive and cannot yet be
called a robust capability result. The 512 checkpoint remains a candidate, but
the next high-ROI experiment is stability diagnosis (multiple fresh seeds and
per-lifetime logging), not another blind increase in training size.

The three-seed follow-up used the stronger counterfactual mode, which recomputes
the correct answer after reversing the query. Only the original seed 117 showed
the earlier effect (about 57--58% intact versus 31% counterfactual). Fresh seeds
121, 123, and 127 produced essentially identical intact/counterfactual scores
(roughly 25--34%). This rules out a robust generalization claim for the current
checkpoint. It also separates two facts that must not be conflated: memory
corruption can reduce performance while the model still fails to encode the
intended causal relation.

### Raw-row localization and first-row sidecar (2026-07-23)

A probe over the individual support writes localized a previously hidden loss.
At 128 train/128 held-out lifetimes, the first raw support key decoded the
temporal rule at 83.6% with a small MLP (77.3% linear), while the newest raw
support key was at chance. Concatenating both keys reached only 67.2%. The
existing sidecar retained the newest, weak row and discarded the strongest row
after consolidation.

A generic `--preserve-first-raw-write` policy was therefore added and tested.
It retains a sensory-derived neural write only; it receives no task metadata or
game state. Simply swapping this row into a reader trained on the old row did
not improve behavior. A one-update sanity run completed in 8.8 seconds. A
four-update/32-lifetime pilot produced exactly 36.7% for intact,
stale-label-reversed, and correctly relabeled counterfactual queries, so it
failed the causal gate and was not scaled.

### Supervised linear readout audit and rejection (2026-07-23)

The exact query-event plus first-raw-row input supported 53.5% held-out
eight-way action decoding at 128 lifetimes, versus 13.3% with shuffled labels
(12.5% chance). An explicitly supervised linear bootstrap was installed as the
real event readout. At 4x logit scale, its first 64-lifetime audit showed the
desired pattern: 53.1% intact, 46.1% stale-label reversal, and 53.1% correctly
relabeled reversal.

The pre-registered fresh-seed adversarial audit rejected that apparent success.
At 128 new lifetimes the pattern became anti-causal: 45.3% intact versus 52.7%
stale-label reversal, with correctly relabeled reversal back at 45.3%.
Empty/shuffled/garbage memory reduced performance to 12.1%/19.1%/12.5%, proving
memory dependence but not intended causal use. Spatial and shape fell to
18.8% and 17.6%, because the ungated readout perturbed every primitive. The
checkpoint is rejected.

Retaining both raw support rows did not rescue the monolithic readout (best
held-out 56.6%, final MLP 48.0%). Scaling the exact first-row action probe to
512 train/512 held-out lifetimes also stayed flat (50.0% linear, 51.7% MLP).
This branch is therefore stopped rather than given a larger behavioral run.

The final localization control was strongly positive: with 512 unique
lifetimes, the first raw support row decoded the demonstrated rule at 85.9%
linear and 92.4% MLP held-out accuracy, with shuffled labels at chance. The
next justified architecture is a factorized learned readout that separately
extracts the support rule and query-event facts, then learns their composition
through generic multiplicative interactions. A larger monolithic action head
is not justified by the evidence.

The prerequisite query-fact probes passed at the 128-lifetime rung. From the
same event representation, a small MLP decoded the action associated with the
first query event at 77.7% held-out accuracy and the second at 78.1%; shuffled
controls were 14.1% and 12.9% (eight-way chance is 12.5%). Together with the
92.4% support-rule result, all three ingredients are independently available.
The next probe should therefore test learned factorized composition, with
auxiliary heads used only as diagnostic training signals and a lifetime-disjoint
held-out/action-reversal audit as the gate.

### Factorized readout and complete-loop milestone (2026-07-23)

A factorized event router was trained against one immutable multiblock cache
(1,024 unique training lifetimes and 512 held-out lifetimes). Three training
seeds produced 86.2--87.6% hard action accuracy, 95.3--97.3% counterfactual
flip rates, 12.7--13.7% stale-label accuracy, 89.5--90.0% rule decoding, and
94.4--98.1% ingredient decoding. This is the first robust composition result
after the earlier small-data readout churn.

Two interface bugs were caught before promotion. The portable checkpoint
initially omitted its normalization statistics, and the integrated agent
passed an already-normalized key into a router that normalized it a second
time. Saving normalization buffers and consuming the raw latest memory key
restored fresh-range in-situ action accuracy to 89.8--90.6%, with
93.8--96.9% flip rates and 8.6--10.9% stale-label accuracy.

An ungated integrated agent then passed the temporal causal and memory
dependence tests but damaged old primitives: 82.8% temporal intact,
17.2% stale-label reversal, 12.9--16.4% under memory corruption, but only
20.9% spatial and 23.6% shape versus a paired 40.0%/43.2% zero-strength
baseline. An input-density audit localized the problem: normalized temporal
router inputs occupied density-distance 0.34--2.75, while spatial and shape
occupied 8.58--12.62. A fixed task-agnostic OOD threshold of 5.0 therefore
gates the learned router without using task identity or game state.

The final fresh-seed 128-lifetime audit passed every pre-registered gate:

- temporal intact: 76.95%
- stale-label reversal: 23.05%
- correctly relabeled counterfactual reversal: 76.95%
- empty / shuffled / garbage memory: 12.89% / 16.80% / 12.89%
- spatial / shape retention: 43.36% / 41.41%

This is the first causally audited complete
perceive→event-snapshot→store→recall→compose→act loop for the temporal
primitive with old-skill retention. The checkpoint is explicitly
**supervised-bootstrapped**; it does not establish reward-only discovery.
The evidence and promoted checkpoint are saved under
`remote_results_2026-07-23/final_gated_audit` and
`remote_results_2026-07-23/factorized_router_cache1024_agent_s1_ood5.pt`.

### Updated immediate frontier

The read-side interface is no longer open. The next experiments should not
revisit 128-lifetime architecture variants. The highest-payoff sequence is:

1. promote event-indexed memory from a temporal adapter into a reusable
   subsystem;
2. run thought-loop-length ablations to test whether recurrence causally
   supports line tracing and other deliberate perception;
3. introduce a composition task requiring no new mechanism and record its
   learning cost in a transfer ledger;
4. only after transfer is measured, test novelty/learning-progress curricula
   and later self-generated action-conditioned value signals.

### Event-indexed memory and compositional transfer checkpoint

A controller-hidden-state ablation did not explain temporal outcome
perception. Resetting the controller GRU at every event left the post-feedback
rewarded-identity probe essentially unchanged (normal 73.6--80.7%, reset
71.5--80.5%). Persistent workspace state may still matter, but controller
hidden carry is not the current lever.

The first four-color compositional task localized a new bottleneck. Frozen
zero/one/two/four-shot behavior stayed near 19--21%, and the old factorized
ingredients failed. Nevertheless, frozen-state probes decoded query color at
100% and the four study actions at 97.3--99.2%. The information existed; the
reader could not content-address the correct event.

A generic content-addressed event-memory reader solved that diagnostic. With
lifetime-disjoint train/validation/test splits and validation-only checkpoint
selection, two fresh runs reached 74.8% and 71.1% untouched-test accuracy.
Shuffled-memory and shuffled-label controls remained near chance and memory-row
permutation preserved predictions exactly. A later mixed-visual-surface reader,
trained on both mapping cards and temporal object frames, reached 73.5% on the
previously unseen temporal-event evaluation surface.

Naively adding its logits to every task was rejected: mapping improved, but
spatial and shape fell to 10.6% and 17.2%. A learned arbitration probe showed
the reader itself retained useful mapping information for all three families
(about 84--96%); the error was architectural—the mapping result was being
treated as the final task answer. The repaired design exposes the reader's
outputs as candidate actions to the existing rule router.

Memory was also split into two agent-owned tiers. Active memory contains the
consolidated task state and support-rule sidecar, while an immutable event
archive retains sensory-derived study writes for later content addressing.
This avoids corrupting compact memory by mixing raw studies back into it and
uses no privileged game state.

The current best frozen full-composition result, using the mixed-surface reader,
event archive, and factorized candidate override, is:

- zero-shot: 33.98%
- one-shot: 35.94%
- two-shot: 37.89%
- four-shot: 36.72%

This is real partial transfer: reusable mapping raises performance from roughly
20% to 34--38%. It is not yet a solved composition capability. The small
few-shot gain shows that multi-study temporal-rule extraction is now the
remaining blocker. The next probe is already implemented but deliberately not
run in this stopped session: measure compositional support-rule decodability at
the raw-write boundary before changing the architecture.

## 2026-07-24 compositional raw-write localization

The initial 32-train/128-test probes were underpowered: both the compositional
task and the known temporal atom fit the tiny training set while remaining at
held-out chance. The diagnostic was corrected in two ways before drawing a
conclusion:

1. it now retains the first, latest, and complete history of support writes;
2. it loads the canonical audited factorized-router agent, including the
   embedded fixed pairwise-transfer weights.

At 128 train and 128 lifetime-disjoint test lifetimes, the corrected positive
control decoded the atom's rule from the first support write at **80.47%**
(MLP final accuracy; held-out majority 53.91%). The latest support write was
50.00%, showing that later supports overwrite or obscure the useful rule.
The prior matched shuffled-label calibration for this first-write probe was
51.56%.

The four-color compositional task remained at chance:

- first support write: 51.56%;
- latest support write: 50.78%;
- concatenated four-write history: 47.66%.

All three probes strongly fit their training data, so the failure is
generalization rather than dead optimization.

A causal palette substitution then changed only the atom's displayed
identities from colors `(0, 1)` to `(2, 3)`. Logic, mappings, answers, support
orders, and query orders were identical. First-write decoding collapsed from
80.47% to **45.31%**. This localizes the compositional failure primarily to
identity-specific temporal binding: the current primitive has not learned an
abstract first/last relation that transfers across object identities.

The next high-ROI experiment is therefore an identity-diverse temporal
curriculum with palette-held-out evaluation. It must be measured by
examples-to-threshold and compared with a fresh-agent control; scaling the
four-color behavioral loop before this invariance appears would conflate
relation learning with identity expansion.

## 2026-07-24 palette localization and zero-label boundary

Several sub-minute experiments further localized the failure:

- palette-diverse heads trained on 120 presentations did not generalize;
- rendering each of 40 logical lifetimes under three palettes produced
  memorization, not held-out transfer;
- latent-consistency regularization gave an early experienced/fresh advantage
  but failed the accuracy and shuffled-label gates;
- parameter-free similarity metrics remained at chance;
- an apparent 60.83% rule probe failed true event-reversal causality (4.58%
  prediction flips) and was rejected as a shortcut.

At the sensory boundary, first and second identities were 98–100% decodable
under held-out palettes, while the identity selected by the original tiny
feedback marker was much weaker. Re-rendering feedback by showing the selected
object directly produced a clean diagnostic:

- first identity: 98–100%;
- second identity: 97.92–98.33%;
- selected identity: 99.17%;
- derived temporal relation: 87.08–87.92%;
- shuffled-label control: 46.67%;
- true reversal: 90.00–90.83% relabeled accuracy and 77.92–78.75% flips.

This does not authorize direct semantic feedback as the deployed curriculum.
It proves that the relation machinery can generalize once outcome perception
is unconfounded. Learning to trace or infer the original feedback marker should
be treated as a separate sensory primitive, acquired from observed effects
rather than hand labels.

A three-seed diagnostic compared the experienced visual encoder with a newly
initialized identical encoder. At 120 unique lifetimes, experienced features
averaged 94.44% held-out accuracy versus 83.61% for fresh features. At 60
lifetimes the means were 68.89% and 67.64%, and the examples-to-threshold
advantage varied by seed. Thus prior experience exposes a better eventual
representation, but faster learning has not yet been demonstrated reliably.

All classifier heads in this section are throwaway measurement probes trained
from deterministic verifier facts. They are not deployed, their weights do not
enter the agent, and their accuracy is not counted as agent capability. The
next learning experiment must use no human-authored or semantic target labels:
only sensory prediction/consistency, memory utility, observed action effects,
and externally verified behavioral reward.

## 2026-07-24 zero-label predictive-state experiments

Immediate absolute next-latent prediction learned nontrivial visual
predictability after anti-collapse repairs, but did not improve reward-only
learning. Two seeds stayed near chance behavior and a favorable shuffled-arm
probe on one seed did not replicate.

A 23.14-second delta-prediction fork instead predicted the change in the EMA
latent between adjacent frames. It learned genuine temporal alignment
(held-out paired-versus-shuffled loss margin 0.522) and exposed the private
temporal relation to a discarded MLP probe at 80.73%. The same probe retained
79.69% accuracy under a valid pixel-space reversal; a shuffled-label
calibration scored 46.61%.

This did not become usable capability. Reward-only behavior was 50.26%, below
the shuffled-future control's 52.34%; no accuracy threshold was reached and
the AULC advantage over the best control was -0.00391. Effective rank was 5.64
of 64, below the pre-registered 6.4 gate. The arm is therefore not promoted.

The experiment localizes the next bottleneck: a useful relation is present in
the sensory state, but the present REINFORCE readout does not exploit it at the
tiny interaction budget. The next bounded zero-label experiment is an
action-conditioned success predictor using only latent state, the agent's
sampled action, and scalar verifier reward. Its value will be judged solely by
verified reward-learning speed against matched controls.

The first action-conditioned success run used the same nonlinear action head,
20% exploration floor, 510 unique reward bits, 17 optimizer updates, and 510
processed examples in every arm. Success replay reached 53.13% versus 48.96%
for matched REINFORCE, 50.26% for the action-shuffled control, and 50.00% on a
fresh representation. The candidate retained balanced action coverage but
gained only 0.00694 AULC over the best control, never reached 60%, and changed
only 8.07% of predictions under true event reversal. It failed the gate.

Because this run allowed only 17 optimizer updates while the discarded
relation probe previously required about 200, the next tiny diagnostic holds
reward bits fixed and sweeps replay computation. Interaction/sample efficiency
and gradient-compute efficiency will be reported separately.

The fixed-buffer diagnostic then crossed the behavioral threshold. At an equal
200-update/6,000-example budget for every reward-bit prefix and control, a
success head over correctly paired delta-predictive states reached about
75–83% across three seeds. Fresh, action-shuffled, reward-shuffled, and
shuffled-future-representation controls were substantially weaker.

The first causal interpretation was rejected despite those numbers. Because
query order was deterministically opposite support order, the model used
feedback identity plus query order instead of the demonstrated support order.
Support-only reversal barely changed predictions, while query-only reversal
changed them incorrectly. The earlier both-orders reversal had failed to
distinguish these routes.

Removing the query from the rule-identification microtask repaired the causal
graph. Across seeds 211/257/313, the support-only learner reached
78.13%/82.03%/80.73% final accuracy and 0.2188/0.2453/0.2068 AULC. The
shuffled-future representation averaged 55.73% final and 0.0434 AULC; the
equally optimized IPS learner averaged 0.1780 AULC. True support-order reversal
averaged 78.82% relabeled accuracy and 59.11% prediction flips.

This is a three-seed zero-semantic behavioral milestone: predictive experience
formed a causally useful state, and attempted-action-only scalar outcomes
trained an effective action readout without synthesizing unattempted labels.
It is one primitive, not yet evidence of compounding transfer.

## 2026-07-24 zero-label actuator transfer

The next experiment separated learned intention from device protocol. Phase A
trained an eight-dimensional intention bottleneck and two-action success
decoder using only attempted actions and scalar outcomes. Phase B froze the
intention, discarded the old decoder, and calibrated a fresh four-command
adapter. An identical fresh intention-plus-adapter learner and
lifetime/action/reward-shuffled arms received the same phase-B reward bits,
updates, and examples.

Across seeds 211/257/313, the experienced system reached 75% held-out accuracy
after 32 new reward bits on every seed. The fresh system required 510/256/256
bits. The median transfer ratio was 8× and the mean was 10.65× at equal
200-update/6,000-example replay compute. Experienced AULC above the 50%
majority floor averaged 0.3017 versus 0.1958 fresh.

The causal audits held. Reversing only the rendered support order yielded
79.69% mean relabeled accuracy and 59.98% command flips. Pairing every episode
with an intention from the opposite private rule reduced accuracy to 19.70%.
Swapping the two protocol codes without recalibrating the adapter also reduced
accuracy to 19.70%. Action- and reward-shuffled controls did not cross any
threshold.

The experiment initially used a naive one-step stale-state roll. It scored
about 55%, but an audit showed that deterministic generator ordering preserved
the same rule in 53.65% of those pairs. The result was withheld, the control
was replaced by guaranteed opposite-rule pairing, and all three seeds then
passed the stronger pre-registered v2 gate.

This is the first clean evidence that the system's learned output can function
as a reusable latent intention rather than a fixed external command ID. The
claim is zero-label actuator/interface transfer, not transfer to a different
cognitive primitive and not yet compounding learning.

## 2026-07-24 temporal-to-spatial transfer fork

A new exactly balanced primitive showed two objects simultaneously and then
the selected object's identity. The verifier asked whether it had occupied the
left or right position, and a new four-command protocol prevented reuse of old
external action IDs.

At seed 211, a trainable temporal intention reached 78.39% final accuracy and
0.2427 AULC. A fresh intention reached 79.43% and 0.2396. Both crossed
55/65/75% at 32/128/256 reward bits; the +0.0031 AULC difference failed the
pre-registered gate. A frozen temporal intention was slightly weaker at
75.26% and 0.2281 AULC. The run was stopped rather than expanded to more seeds.

The negative is not a task failure. Horizontal mirroring yielded 81.77%
relabeled accuracy and 60.16% command flips, missing feedback returned to
48.70%, and opposite-rule stale state reduced accuracy to 21.61%. The system
learned the spatial primitive causally; prior task-specific intention weights
simply did not make learning faster.

Both primary arms shared the delta-predictively experienced visual encoder and
GRU. The result therefore rejects intention-head warm-start as the compounding
mechanism but does not exclude transfer already occurring in the recurrent
predictive core. The next cheapest localization is the missing fully
fresh-core/fresh-intention factorial cell, not a larger architecture.

### Predictive-core factorial resolves the transfer location

The missing cells changed the conclusion. A fully fresh visual encoder, GRU,
intention, and adapter stayed exactly at 50% through 510 spatial reward bits.
An equal-compute core trained on the same temporal pixels with future targets
shuffled was better than fully fresh but remained much weaker than correct
pairing.

Across seeds 211/257/313:

- correctly paired predictive core + fresh intention: 78.56% mean final,
  0.2231 AULC;
- shuffled-future core + fresh intention: 58.25% mean final, 0.0849 AULC;
- fully fresh core: 50.00% mean final, 0.0000 AULC;
- mean paired AULC advantage over shuffled pairing: 0.1382;
- paired core reached 75% at 256 reward bits on all seeds;
- shuffled and fully fresh cores never reached 75% by 510 bits.

The paired-core spatial policy remained causal: mirror relabeling averaged
80.38%, flips 58.94%, missing-feedback accuracy 50.69%, and opposite-rule stale
accuracy 21.44%.

The old temporal intention head still contributed no stable improvement: mean
AULC was 0.2234 versus 0.2231 with a newly initialized intention over the same
paired core. Thus the reusable asset is structured predictive state, not
task-specific output weights.

This is a three-seed zero-label cross-primitive representation-transfer
milestone. It shows one temporal-to-spatial transition; a sequence of later
tasks with falling reward-bit costs is still required for compounding.

### Third primitive: no compounding gain yet

A delayed same/different identity task tested whether adding paired spatial
predictive experience improved the next primitive. At seed 211, the
temporal+spatial core reached 80.47% and 0.2422 AULC but needed 256 reward bits
for 75%. Equal-compute extra-temporal and spatial-future-shuffled controls both
reached 75% at 128 bits with slightly higher AULC. Temporal-only also reached
75% at 128 bits. The candidate failed its gate and was not replicated.

The behavior was genuine: changing only the second identity produced 84.64%
counterfactual accuracy and 65.10% flips; removing either identity returned to
chance; stale opposite-rule state collapsed to 19.53%.

The supported conclusion is that the predictive core transfers once, but
naively continuing spatial-only prediction does not yet compound and may drift
away from generally useful state. Future core updates need explicit
task-agnostic retention/rehearsal gates, or should move to a closed-loop task
where new action-conditioned structure can earn its cost.

### Behavioral retention localization

A matched temporal retention audit compared fresh attempted-action heads over
the sequentially trained cores. Temporal-only behavior was 77.08% versus
74.48% after paired spatial prediction, a 2.60-point drop. True-reversal
accuracy fell by the same 2.60 points. Shuffled-spatial retained 75.52%, extra
temporal improved to 78.91%, and fully fresh remained at 50%.

The result suggests spatial-specific representational drift, but it missed the
pre-registered three-point forgetting gate. Rehearsal was therefore not
promoted on a near-threshold one-seed effect. The next justified experiment is
closed-loop micro-intercept: passive versus action-conditioned versus
shuffled-action prediction when actions causally alter later pixels.
