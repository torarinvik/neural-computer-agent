# Neural Computer Agent

A compact research repository for building a real-time neural computer that
learns reusable cognitive primitives from sensory streams and deterministic
outcomes.

Audited model checkpoints are stored in the private Hugging Face repository:
<https://huggingface.co/torarin87/neural-computer-agent>.

The learner receives rendered vision/audio/text streams, its own opaque
actions, its own latent state and memory, and scalar verifier outcomes. It
does not receive game state, coordinates, semantic task labels, rule IDs,
correct-action labels, English chain-of-thought, or counterfactual labels for
actions it did not attempt.

## North star

Maximize verified reusable capability gained per unique interaction.

The project distinguishes:

- unique verifier/reward bits;
- unique logical lifetimes;
- replayed examples;
- optimizer updates;
- GPU and wall time;
- action latency;
- retention and forward-transfer ratios.

Final accuracy alone is not an adequate score.

## Current audited frontier

The persistent-memory line now has a verified-use plasticity milestone. Every
physical row carries a modality- and task-agnostic volatility scalar. Successful
retrieval gradually protects a row, failure thaws it, and disuse slowly restores
plasticity; access frequency alone cannot freeze a memory. The field survives
RAM/VRAM movement, selection, growth, disk save/reload, and old v1-v3 memories
load as fully plastic.

In a non-stationary bounded-memory atom, three stable skills, three equally
frequent but consistently failing decoys, and two stale skills competed with
four new skills. Verified-use volatility retained 100% of stable skills while
acquiring 100% of the new skills over 64 seeds. Uniform replacement scored
77.46%, access-only plasticity 71.43%, and shuffled row/volatility
correspondence 79.02%. This directly falsifies the tempting shortcut “frequent
means important”: stable and decoy rows had identical access counts, but only
verified usefulness separated them.

A fresh 321-parameter generic replacement selector then learned to use the
scalar from final verifier reward alone—no task identity, semantic row label,
or correct replacement action. Four independent 192-update runs each reached
100% perfect episodes on 512 held-out environments. Physical disk-backed audits
retained 100% of stable rows and installed 100% of new rows; shuffling
volatility among rows reduced stable retention to 51.0–53.4%. Access-only and
uniform controls lost roughly one stable skill per episode. This establishes
learned selective stability in external memory.

That mechanism is now integrated into the full visual controller. The parent
replacement policy already saw age, strength, similarity, access frequency,
and aggregate reliability. A zero-initialized eighth feature added volatility;
the only trainable quantity was its single scalar coefficient. Stable and decoy
rows were deliberately matched on access count and on total successes/failures
(five each). Only the *order* of scalar verifier outcomes differed.

Two independent 32-update runs reached 99.61% and 98.83% valid replacement on
held-out visual lifetimes, within 0.3 points of their perceptual oracles. Both
crossed and retained 95% after 24 updates, 6,144 unique verifier bits, and
10,752 logical contexts, with no replay. A matched reward-shuffled run remained
at 57.81% and never crossed threshold. Shuffling volatility reduced correct
replacement to 47–49%; reversing outcome order made the controller evict the
previously stable row on 98.4–99.6% of banks. All inherited behavioral and
memory-utility gates remained intact. Each complete run took 72–80 seconds,
well inside the five-minute cap.

Three physical disk-backed audits then achieved 100% valid replacement,
91.80–95.12% visual accuracy, exact persistence of keys, values, access,
success, failure, and volatility fields, and zero capacity growth. Shuffled or
constant volatility fell to 46.9–52.3%; reversing histories flipped 100% of
replacement choices. This physical claim currently assumes equal admission
strength while histories are accumulated. With unequal learned strength priors,
content retrieval redirected some exact queries to other rows and correct
replacement fell to 67.19%. Credit attribution under unequal retrieval priors
was therefore the next explicit frontier, not part of that promoted claim.

That frontier is now closed for the audited exact-content regime. Retrieval's
write-strength prior now has a backward-compatible controller scalar, initialized
at the old value `1.0`. A five-candidate population race compared scales
`0, 0.25, 0.5, 0.75, 1` on matched physical banks using only pixel-task reward.
Two independent races selected content-first scale `0.0` from 1,280 verifier
bits and 448 unique logical contexts in 23.8–24.2 seconds.

Under unequal learned admission strengths, the unadapted parent reached 64.06%
valid replacement. Both selected controllers reached 100%, with 98.05% and
99.02% visual accuracy. Volatility shuffling returned replacement to
46.88–50%; reversing histories flipped every choice. A reward-shuffled race
selected `0.5`, reached only 69.53%, and failed the causal gates. Two independent
512-context selective-disk audits retained the older loop at 92.77–93.55%
first-reload and 93.36% repeat-reload accuracy, with value corruption causal and
duplicate rates below 12%.

That per-query frontier is now closed on a balanced exact-versus-ambiguous
retrieval atom. A dormant 49-parameter policy observes four generic retrieval
statistics and chooses whether each query should be content-first or should
also use verified row strength. It is trained only from the controller's
attempted retrieval and scalar visual-task outcome.

Two independent 80-update runs reached 100% held-out accuracy on both arms and
remained perfect from update 40 onward: 5,120 unique verifier bits to stable
95%, 10,240 bits total, no replay, and 27.4–34.6 seconds end to end. Either
global rule is insufficient: fixed content-first reached 73.0–74.2%, while
fixed strength-aware retrieval reached 74.4–74.8%. Shuffling the policy's
generic features reduced its action accuracy to chance and task accuracy to
73.8%; corrupting retrieved values reduced task accuracy to 47.9–48.6%.

The result survives 256 independently saved and reloaded two-row physical
banks per seed at 100% accuracy with exact persistence. A reward-shuffled run
never learned the conditional action and stayed at 73.6%. The earlier
selective-disk loop still passed at 92.8–94.1% reload accuracy, and the
unequal-strength volatility audit retained 100% valid replacement with exact
histories.

The binary policy has now compounded into continuous resource control. A
hardened task makes any constant scale impossible: exact queries require usage
influence below `0.12–0.18`, while ambiguous queries require it above
`0.35–0.55`. Fixed scale zero and one therefore each retrieve the correct row
exactly 50% of the time. Correctness remains the primary reward, with a smaller
generic cost for stronger historical influence.

Two independent eight-update runs retained 100% exact and ambiguous retrieval
while reducing mean scale from the inherited binary policy's `0.50` to `0.312`
and `0.347`. Both reached the joint correctness-and-efficiency gate at update
five and remained above it through updates six, seven, and eight: 640 unique
verifier bits to stable improvement, 1,024 total, no replay, and about five
seconds of training.

Without any further training, both controllers achieved 100% on three- and
four-row banks, including 128 independently saved and reloaded physical banks
at each size. Shuffling generic query features reduced row accuracy to
49.4–53.3%; corrupting values reduced visual success to 45.7–48.9%. Reward
shuffling and resetting the inherited conditional policy each collapsed row
accuracy to 50%, showing that both new verified feedback and the old learned
skill are necessary. The original conditional task, selective disk,
unequal-strength volatility, binary mapping, and four-rule behavior all remain
retained.

The next frontier is a broader continuum with genuinely competing third and
fourth rows, followed by naturally occurring duplicate memories rather than
generator-separated query regimes.

The new unified-controller line now has its first retained compounding
milestone. A single 298,252-parameter controller with one vision encoder,
recurrent state, generic differentiable workspace, latent intention, and
replaceable actuator adapter learns hidden visual-action functions from its
own attempted opaque actions and scalar outcomes.

Prior visual grounding changed a matched 600-step four-rule task from a stable
75% shortcut to 99.85–99.90% on two independent seeds. The next rung inferred
an identity-versus-flipped mapping after one support outcome; inherited
training reached 100%, while matched fresh stayed at 49.26%. Balanced rehearsal
then preserved both the one-support skill and the broader two-support
four-function skill. The selected checkpoint passed disjoint 2,048-lifetime
normal, private-rule reversal, prediction-flip, blank-vision,
shuffled-feedback, and active-state-reset audits:

- one-support bijection: 99.98% normal, 99.95% reversed;
- retained four-function task: 100% normal and reversed;
- paired counterfactual flips: 99.93% and 100%.

This is evidence of fast within-lifetime binding, positive forward transfer,
and behavioral retention in one controller.

The same controller now also performs content-addressed latent recall across
active-state resets. A 600-update capacity-two rung reached 96.53% blind
recall; 150-update bridges at capacities 8 and 16 produced zero-shot transfer
to capacities 16 and 32. A later five-second rung used only 20 new-memory
updates at capacity 40 and reached 90.00% blind recall, then transferred
zero-shot to capacity 48 at 88.28% and capacity 56 at 87.33%. An independent
five-second acquisition replicated the result, and the two checkpoints crossed
the old capacity-64 frontier at 85.57% and 86.33%. Empty, shuffled, and
corrupted memories collapse toward chance; disk save/load reproduces hard
retrieval; the earlier one-support and four-rule skills remain retained. The
frozen retrieval frontier is now capacity 72; both parents fail capacity 80.

A subsequent selective-memory atom learned from verified success minus a
generic write cost. On blind data it wrote on 61.16% of first encounters but
only 5.10% of redundant repeats, averaging 0.663 writes per context while
retaining 99.90% query accuracy. Removing writes, shuffling admissions,
corrupting values, or hiding the prior memory read causally degraded the
appropriate behavior.

The first physical integration audit exposed a boundary: intentionally absent
default rows retrieve unrelated neighbors in a shared disk bank. Scalar
rejection gates restored 87.99–88.96% disk accuracy but missed duplicate or
false-accept gates. A discarded diagnostic localized the limitation: the four
generic memory statistics supported only 83.01% held-out classification with
a five-parameter linear gate, while an eight-unit nonlinear gate reached
88.18%. No diagnostic weights entered the agent.

A fresh 49-parameter gate inside the same controller was then trained only
from verified query success minus a generic read cost. In 160 updates, 81,920
unique contexts, and 9.71 seconds it reached 91.55% held-out accuracy, accepted
89.67% of useful reads, rejected enough absent reads to hold false acceptance
to 17.33%, and retained both prior behavioral gates.

Two independent physical disk audits then passed every pre-registered gate:

- first save/reload accuracy: 91.50% and 92.19%;
- accuracy after a repeated encounter: 91.02% and 91.41%;
- duplicate rows per context: 17.68% and 17.29%;
- empty-memory controls: 50.20% and 50.00%;
- wrong-value disk corruption controls: 70.41% and 70.70%.

This admits the first unified learned RAM/VRAM-to-disk loop: the controller
creates sparse opaque rows, reloads them after active-state erasure, learns
whether a retrieved row deserves use, and suppresses redundant repeat writes.
The selected checkpoint is
`artifacts/checkpoints/unified_selective_disk_adaptive_seed5962.pt`, SHA-256
`91822064436fae1d4f799e41c79d9369dacb8aeeee20b711df1c1b6af037fbc4`.

The next gradual atom bounded each disk bank at four rows. A 57-parameter head
inside the same controller learned whether to skip or which physical row to
replace, using only generic row metadata and later verified success. After 40
updates and 7.01 seconds it reached 96.90% held-out accuracy versus 84.35%
random, 85.11% fixed-slot, and 80.91% skip controls. Two physical disk audits
replicated at 96.97% and 96.29% with exactly 2,048 rows before and after and
zero capacity growth. Shuffling age-to-slot correspondence reduced accuracy to
81.35% and 82.37%. The earlier sparse disk loop still passed at 91.21%.

The promoted 298,358-parameter checkpoint is
`artifacts/checkpoints/unified_memory_replacement_seed6101.pt`, SHA-256
`0178b15228e3d75a445abdb2376be1291a078f8b47236444fbd1824fab3d3b76`.
That first policy transferred zero-shot to capacity 5, but not causally to
capacity 6. A gradual bridge then used only 20 capacity-6 updates interleaved
with 20 capacity-5 rehearsal updates. It matched the capacity-6 oracle at
96.39% with 100% correct evictions and retained every prior gate. Two physical
replications reached 96.55% and 96.71%. The sharpened rule subsequently
transferred with zero weight updates through capacities 7 and 8 and reached a
replicated physical capacity-9 frontier at 94.57% and 94.62%, with
99.61–100% correct evictions and zero growth.

The fixed-utility parent is
`artifacts/checkpoints/unified_memory_frequency_recency_capacity6_seed6607.pt`,
SHA-256
`1346da994de4ba20864c5f1bc1da12684fc13d8dcda480a76cfc6f713da0181c`.

The next utility rung kept capacity fixed at six but made future usefulness
depend jointly on noisy recency and access frequency. Ordinary
content-addressed reads now increment persistent access counters, which survive
disk save/reload and reset on replacement. A zero-initialized one-parameter
residual let the proven recency policy compose this new generic statistic
without changing its inherited path.

Two reward-only 20-update runs passed in 3.23 seconds of training each. They
reached 95.32% and 95.10% held-out future accuracy, 87.30% and 86.13% correct
evictions, and retained recency, binary mapping, and four-rule gates. The
learner consumed 51,200 unique verifier bits per run with no replay or utility
labels; only the one new coefficient changed.

Two physical disk audits then reached 96.81% and 96.29% on access histories
generated by actual retrievals. They made 92.97% and 93.36% correct evictions,
captured 76.3% and 91.7% of the visible-oracle gap above the strongest
single-feature control, preserved all 512 audited histories exactly through
save/reload, kept 3,072 total rows bounded, and never grew capacity. Shuffling
age reduced accuracy by 4.56–4.75 points; shuffling frequency reduced it by
6.77–7.75 points.

The next rung removed the fixed utility mixture. In one uninterrupted stream,
the relative value of recency and access frequency changed from 65:35 to
35:65, returned to 65:35, and ended at 50:50. No phase identity, boundary
signal, optimizer reset, utility label, correct eviction label, or replay was
available to the learner. A symmetric two-candidate horse race changed only
the controller's existing one-parameter utility residual according to which
candidate produced more verified future success.

Two independent 64-update runs passed all online, retention, and causal gates
in 28.66 and 28.89 seconds:

| Seed | Recency target | Frequency target | Recency-return target | Equal-return target |
|---:|---:|---:|---:|---:|
| 6809 | 90.67% | 86.43% | 91.16% | 90.53% |
| 6810 | 90.82% | 87.16% | 91.31% | 89.99% |

The frequency-dominant phase improved held-out accuracy from 93.87% to 95.62%
and from 93.49% to 95.37% over an unadapted copy. Shuffling which candidate
received each verified outcome made the adaptation fail: the frequency target
fell to 57.71%. The selected checkpoint then passed a 1,024-bank physical disk
audit at 96.94%, within 0.13 points of the visible oracle. Shuffling age or
frequency reduced it to 92.74% and 88.66%; all 6,144 rows and access histories
survived save/reload exactly and capacity never grew.

The one-parameter online parent is
`artifacts/checkpoints/unified_memory_online_utility_seed6810.pt`, SHA-256
`c3e837c6512a30c11b1c861b79242296b76cfa0cd9fe62aa414d3e5b2aa10750`.
This establishes rapid verifier-driven adaptation of one generic controller
coefficient, not yet a learned general-purpose internal meta-optimizer.

The next gradual rung added one genuinely new task-agnostic statistic:
verified outcome reliability for each memory row. Physical rows now keep
success and failure counts attributed through ordinary content-addressed use.
The counters survive disk save/reload and reset when a row is replaced. Older
memory schemas load with zero counts.

A redundant write-strength coefficient was first rejected: write strength was
already visible to the inherited controller, so it added only 2.93 target
points. Reliability passed the representation gate by over 30 points and then
learned online. Two independent 48-update runs used a three-candidate
move/stay horse race, 196,608 verifier bits, no replay, and about 29.4 seconds:

| Seed | Old equal | Reliability dominant | Old return | All equal |
|---:|---:|---:|---:|---:|
| 6932 | 89.75% | 78.22% | 88.48% | 87.45% |
| 6938 | 88.67% | 88.43% | 84.72% | 83.35% |

Frozen target rates were only 57.62–58.40% in the reliability phase and
63.48–64.60% under all-equal utility. The exact reward-shuffled control failed
the multi-phase gates and ended all-equal at 64.31%. Both intact runs retained
binary mapping and four-rule behavior; only the two-coefficient generic
residual changed.

The selected controller passed an independent 1,024-bank physical audit at
96.21%, versus 96.35% for the visible oracle. All 6,144 rows and all
access/success/failure histories survived save/reload, with zero capacity
growth. Shuffling age, frequency, or reliability changed correct evictions by
50.29, 60.55, and 30.18 points and reduced actual accuracy by 3.11, 6.75, and
2.56 points.

The tensor-trained parent checkpoint is
`artifacts/checkpoints/unified_memory_multifeature_reliability_seed6932.pt`,
SHA-256
`bb5cd158c08f4b92061aca7bfae0751d4e18408e8e37f53cac13dffaed8ac9f4`.
It has 298,360 parameters.

The adaptation loop itself now also runs through bounded physical disk
memories. A parity preflight made disk serialization sovereign and kept the
old tensor arena only as a shadow audit. Two undersized 32-bank pilots were
rejected before scaling. At the proven 128-bank scale, seeds 7012 and 7015
both passed the four utility phases, retention, persistence, parameter-scope,
and physical/tensor parity gates in about 136 seconds. Seed 7015 achieved
85.74%, 77.25%, 86.72%, and 82.67% across old-equal,
reliability-dominant, old-return, and all-equal phases. All 6,144 physical
histories persisted; all 48 physical choices were tensor-equivalent. The
matched reward-shuffled control failed every adaptation phase and saved no
checkpoint.

The current checkpoint is
`artifacts/checkpoints/unified_memory_physical_online_seed7012.pt`, SHA-256
`2c6e61b5e2689d46dfc43dd5cfc9c5b234736d217aae28f6221501bd5ddeea70`.
The independent replica is
`artifacts/checkpoints/unified_memory_physical_online_seed7015.pt`, SHA-256
`7ae96b44ec6bed0db8eb7f9b78640fe40b621875195303e3e3c604f357bb441d`.

Unseen elongated diamonds and disconnected dot-pair stimuli also transfer
zero-shot at 94.95–98.14%, tightening the evidence that visual identity is
relational rather than tied to the original rectangles.

Long-lived physical banks that accumulate experience across multiple updates,
consolidation, deletion/merging, unbounded memory, and cross-modality transfer
remain open.
See `experiments/unified_cognitive_controller/README.md`.

The two-decision identify-then-act task requires the agent to:

1. emit an opaque probe action;
2. observe its visible consequence;
3. infer the hidden actuator mapping;
4. observe a target;
5. emit the correct opaque action.

The current fresh predictive learner reached on seed 211:

- 100% held-out accuracy at 64 unique verifier bits;
- 100% accuracy and 100% prediction flips under valid protocol rerenders;
- 100% accuracy and 100% prediction flips under target reversal;
- chance performance when the probe consequence is removed.

An incremental 8→16→32→64-bit learner reached 93.36% with 256 cumulative
optimizer updates. A 32-bit arm with 512 updates failed at 52.73%, so extra
replay does not substitute for the missing unique outcomes.

A subsequent exact three-seed map corrected the robustness claim. At 64 bits,
normal accuracy was 55.47%, 99.61%, and 81.64% for seeds 151, 211, and 307;
only seed 211 passed every causal and anti-fluke gate. Thus 64 bits is the
current single-seed capability frontier, not a robust sample threshold.

Earlier fixed-target weights caused negative transfer to the full task.
Inherited weights are therefore retained only when they improve the next
held-out learning curve.

See:

- `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`
- `experiments/forward_transfer_attention/MICRO_INTERCEPT_DESIGN.md`
- `experiments/forward_transfer_attention/README.md`

## Repository map

| Path | Contents |
|---|---|
| `experiments/unified_cognitive_controller/` | Single-controller few-shot binding, retention, and persistent-memory interface |
| `experiments/forward_transfer_attention/` | Main sample-efficiency, transfer, memory, binding, and causal-audit research |
| `experiments/syllogimous_neural_computer/` | Learned external-memory neural computer |
| `experiments/syllogimous_latent_agent/` | Latent real-time agent and sensory models |
| `experiments/syllogimous_bitter_lesson/` | Emergent reasoning experiments without symbolic solution machinery |
| `experiments/syllogimous_realtime/` | Real-time deterministic syllogism environment and Elisa sources |
| `experiments/sensory_codec/` | Sparse sensory stream experiments |
| `artifacts/checkpoints/` | Curated current checkpoints that are small enough for Git |
| `artifacts/manifests/` | Checksums for curated and excluded historical artifacts |
| `session_records/` | Compact historical reports and continuation notes |

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA-capable PyTorch is recommended for training. CPU and Apple unified-memory
backends are useful for tests and tiny diagnostics.

## Verify

```bash
python -m pytest experiments/forward_transfer_attention -q
./scripts/verify_curated_artifacts.sh
```

The narrow current-task regression suite is:

```bash
python -m pytest \
  experiments/forward_transfer_attention/test_identify_then_act.py -q
```

## Reproduce the current 64-bit task

```bash
python -m experiments.forward_transfer_attention.train_identify_then_act \
  --report experiments/forward_transfer_attention/reports/reproduction.json \
  --checkpoint-out artifacts/checkpoints/reproduction.pt \
  --device cuda \
  --seed 211 \
  --curriculum-rung random_probe \
  --intention-width 64 \
  --pretrain-lifetimes 128 \
  --pretrain-steps 40 \
  --policy-lifetimes 64 \
  --test-lifetimes 256 \
  --fit-updates 256 \
  --batch-size 32
```

## Next experiment

Two different 32-bit searches are now complete. The feature-interface winner
fell from 69.53% blind accuracy to 55.47% on its replication seed. A subsequent
learning-mechanism population compared a frozen core, zero-initialized residual
adapters, and conservative action/predictor/recurrent adaptation. Its rank-16
adapter reached 66.41% blind accuracy on seed 211 but also fell to 55.47% on
seed 307, with invalid causal reversal behavior. No checkpoint was promoted.

The cheap "more readout capacity or optimizer freedom" branch is closed at 32
unique outcomes. A subsequent eight-clone reward-free predictive-objective
screen also failed: contrastive refinement reached only 58.59% blind accuracy,
and the unrefined core had the best final selection score. Extra auxiliary
prediction losses therefore do not earn a longer run.

The variance decomposition is complete. Across a nine-horse race at 64 bits,
predictive-core initialization changed the causal floor by 74.22 percentage
points, versus 7.03 points for readout initialization and 5.86 points for
readout replay sampling. All frozen cores passed exact retention checks.

The next sub-minute population should therefore race predictive-core
initializations under identical experience and optimizers, using successive
halving at 32, 48, and 64 outcomes. A winner must then reproduce on a disjoint
lifetime stream and pass old-capability retention before promotion.

That race is now complete. Core seed 263 passed every causal and anti-fluke
gate at 48 and 64 outcomes on a disjoint policy stream with a different
downstream initialization. Its replicated causal floors were 98.05% and
97.27%, respectively. Core seed 211 did not reproduce a stable pass.

The new frontier is therefore a **population-selected, replicated 48-bit
learner**, with search compute accounted separately. Seed 263 is admitted to
the prior-primitive retention/compatibility suite; no general-agent checkpoint
is promoted until that suite passes.

The selected core is now materialized as an immutable 2.9 MB candidate with
SHA-256
`d027b80a631f61c3a9769b60a079494e0a669e1211d3324a13e5ad7b65a1006d`.
Exact reloads reproduce metric-for-metric. With exact complemented negative
controls it passes every gate at 48 and 64 outcomes. A tempting 40-bit point
reaches 95.31% accuracy but fails the missing-evidence uncertainty gate and is
honestly rejected.

Compatibility testing preserves fixed-probe mastery at 16 outcomes and
fixed-target mastery at 48 outcomes, with the predictive core bit-identical
throughout behavioral learning. This establishes a reproducible 25% reduction
from the previous 64-outcome frontier without observed forgetting inside the
identify-then-act family.

The first compounding ladder is also complete. With the immutable core frozen,
novel target-side and observed-effect-side questions each require 8 outcomes,
and their effect-target composition requires 24. The composition replicated on
two disjoint streams while matched-fresh stayed at chance through 64 outcomes.
The gain localizes to the learned vision encoder.

A gradual appearance bridge then changed the palette, object geometry, and
finally both. Stable composition mastery remained 24 outcomes on every rung;
the combined shift replicated with 100% normal/counterfactual accuracy and
100% causal flips. A third-stream retention audit preserved the earlier ladder
at 8/8/16 outcomes. This is verified surface generalization and earlier ability
reuse, not yet broad amodal transfer: spatial relation and event structure are
still shared.

The next bridge replaces position with color identity. It uncovered selective
negative transfer—position-trained vision accelerated observed-effect color
but suppressed target color—so the system retained the useful branch and reset
the harmful one. After acquiring both color primitives from attempted answers
and scalar outcomes, a new relation head reached stable causal mastery from 16
new outcomes on both the selected and blind streams. The identical unacquired
architecture, and either primitive alone, failed through 64 outcomes: a
replicated transfer-ratio lower bound of 4×.

The blind audit reached 100% normal accuracy, 100% accuracy and flips under
both protocol and target rerenders, chance with either fact missing, and 0%
under exact complement controls. Stratified shuffled-label controls produced
no causal pass. The earlier position ladder remained 8/8/24 with bit-identical
cores.

The curated 5.5 MB milestone is
`artifacts/checkpoints/color_primitive_compounder_bits16_seed1901.pt`.

See
`experiments/forward_transfer_attention/ROBUST_SAMPLE_EFFICIENCY_STRATEGY.md`
for the population-search decision and pre-registered diagnostic.

The longer-term optimization is a gradient-trained population with
successive-halving compute allocation. Fitness is held-out learning AULC,
stable bits-to-threshold, retention, latency, and positive transfer to the next
primitive—not old-task accuracy.

## Latest ancestry frontier

Exact-zero skill gates protected retention but made deeper ancestry invisible.
Separating reads from writes restored transfer at ancestry 3 → 4: readable
latent content produced a +0.0242 pooled advantage (48W/22L, p = 2.5e-3), while
a zero-content capacity control did not.

At 4 → 5, reading still improves absolute new-skill learning substantially:
+0.0944 for the four-skill parent and +0.0785 for the five-skill parent against
matched no-read controls. The open problem is accumulation—a second readable
ancestor adds no further depth advantage. Compressing the combined read does
not fix it. A sub-minute recent-only pilot preserved +0.0807 over no read and
was +0.0191 above reading both ancestors, but did not yet reverse the
deep-versus-shallow gap, so no longer run was promoted.

The next tiny diagnostic compares one immediate ancestor with one older
ancestor. A causal difference would justify learned task-agnostic latent
selection; a flat result would rule routing out before more compute is spent.
See `session_records/strategy_accounting_2026-07-28/README.md`.
