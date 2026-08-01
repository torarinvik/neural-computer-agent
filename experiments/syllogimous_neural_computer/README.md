# Growing Neural Computer

> **Historical experiment:** This folder records an earlier controller/memory
> architecture. Its results remain evidence, but the current target is defined
> by the [canonical amodal N-to-M specification](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).

This experiment tests a different scaling hypothesis from the existing fixed-weight reasoners:
keep a relatively small controller, but let its learned external memory grow with experience.
It is isolated in this folder and does not modify or overwrite older models or checkpoints.

## Non-negotiable interface

The controller receives only RGB frames, PCM audio, a padding mask, and its own prior memory.
It receives no entities, relations, answers, game state, parser output, or symbolic proof steps.
It emits public actions plus learned memory read/write signals. The environment alone verifies the
action. Memory entries are latents invented by the controller, not human-authored facts.

## Architecture

1. **Sensory encoder** turns each RGB/PCM event into a latent.
2. **Working workspace** is a small differentiable slot matrix on the GPU. It changes every event
   and is discarded or reset between lifetimes.
3. **Growing persistent memory** stores learned key/value pairs separately from model weights.
   Writes append in chunks instead of evicting old experience. It is independently serializable.
4. **Sparse retrieval** activates only the top matching entries. The first implementation uses exact
   cosine search; a sharded approximate index can replace it without changing the controller API.
5. **Controller** learns queries, values, write probability, reasoning, action, value, and halting.

The bounded resource is the active workspace/cache, not lifetime storage. As the memory grows beyond
GPU RAM, cold entries remain on disk and only retrieved shards or top-k values enter the GPU cache.

## The first decisive experiment

Do not begin with ordinary iid syllogisms: a network can solve those from its weights, so growing
memory has no reason to emerge. Use deterministic visual **lifetimes**:

- During a study phase, the agent sees arbitrary visual associations.
- Many unrelated sensory trials intervene, exceeding the working-memory horizon.
- Later queries require associations from earlier in that lifetime.
- Association assignments change between lifetimes, preventing solution by fixed weights.
- The answer remains unique and mechanically verifiable.

Compare matched controllers:

| Condition | Persistent memory | Purpose |
|---|---|---|
| no-memory | disabled | establishes necessity |
| random-write | growing, random policy | controls for extra storage |
| learned-memory | growing, learned reads/writes | tests the hypothesis |
| frozen-memory | reads allowed, writes disabled | tests whether accumulated memory is useful |
| shuffled-memory | keys/values mismatched | detects ignored or spurious retrieval |

Primary reward is correct action. Only after competence, add small penalties for response latency,
reads, writes, bytes retained, and consolidation compute. This creates pressure for useful compression
without making silence/store-nothing the easiest policy.

## Training curriculum

1. One association, short delay, memory reset each lifetime.
2. Increase delay beyond workspace capacity.
3. Increase simultaneous associations 1 → 2 → 4 → 8.
4. Carry memory across multiple task episodes and process restarts.
5. Add repeated patterns so consolidation can beat raw storage.
6. Mix attention and syllogism tasks, then test transfer with controller weights frozen.

For short lifetimes, use differentiable soft reads/writes so delayed reward can reach the memory
policy. Distill those decisions into sparse discrete writes before enabling disk persistence. Do not
start with discrete disk RL from scratch; the reward is too delayed and the store-nothing solution is
too stable.

## Acceptance gates

- Learned memory must beat both no-memory and random-write controls on held-out lifetimes.
- Accuracy must remain when controller weights are frozen and the accumulated memory is reloaded.
- Shuffling retrieved values must hurt performance; otherwise the agent is ignoring memory.
- Held-out visual alphabets and longer delays must work.
- Storage per solved query must fall or stabilize as experience grows.
- Fresh-memory evaluation must remain separate from accumulated-memory evaluation.
- No blended score may conceal regression on the existing parity or attention tasks.

## Files

- `memory.py`: append-growing, independently saved key/value memory.
- `model.py`: raw sensory controller, differentiable workspace, sparse retrieval, and learned writes.
- `test_neural_computer.py`: boundary, growth, persistence, and forward-pass checks.

The next implementation step is the deterministic visual-lifetime generator and the three-way
no-memory/random-write/learned-memory benchmark. That benchmark—not memory size alone—will tell us
whether the system is genuinely learning to grow through stored experience.

## First matched result — 2026-07-20

An RTX 5090 run used identical 2.14M-parameter controllers, 2,000 training lifetimes per epoch,
six epochs, one arbitrary association, eight intervening events, eight response choices, and 500
held-out lifetimes. Chance is 12.5%.

| Condition | Held-out accuracy | Mean write strength |
|---|---:|---:|
| no memory | 15.8% | 0.000 |
| random write strength | 72.0% | 0.489 |
| learned memory | **82.0%** | **0.070** |

Causal interventions on the same learned controller:

| Memory at query time | Accuracy |
|---|---:|
| intact | **82.0%** |
| emptied | 15.8% |
| values shuffled across lifetimes | 14.2% |
| keys and values replaced by deterministic garbage | 14.2% |

This establishes that the controller learned to encode experience into external memory and later
read it: destroying the blob destroys performance while weights remain identical. Random writing
also works, proving that storage itself is useful, but learned strengths improve accuracy and give
much lower retrieval priority to most entries.

This run does **not** yet prove sparse durable growth. The differentiable training memory appends all
events and uses write strength as a soft retrieval prior. The next acceptance gate is a hard-concrete
or sampled admission policy whose discrete writes persist to disk and survive a process restart,
while retaining the intact-vs-garbage causal gap.

## Discrete and durable result — 2026-07-20

The controller was then split into three independently learned memory decisions:

- a binary admission decision (write or do not write);
- a continuous retrieval priority for admitted entries;
- unconstrained latent keys and values learned only from future task performance.

A straight-through binary controller first reached 100% at 8.16 writes per ten-event lifetime. A
stricter admission gate was then fine-tuned from that checkpoint. The compressed model achieved:

- **95.8%** on 500 held-out lifetimes at **3.75 binary writes per lifetime**;
- **97.5%** on 200 separately generated lifetimes after saving the memory blob to disk and reloading
  it before the query, at **3.45 durable writes per lifetime**;
- 16.0% with the reloaded blob emptied;
- 23.5% with deterministic garbage keys and values;
- 45.0% with values shuffled among valid entries.

The explicit write-cost term never activated during the compressed run because training accuracy had
not yet crossed its 95% activation threshold. The improved selectivity therefore came from adapting
to the stricter discrete admission gate while optimizing task success, not from a handcrafted rule
about which sensory phases should be stored.

This establishes discrete, learned, causally useful, serializable external memory. It still covers
one bounded sensory lifetime at a time. The next research gate is cross-lifetime growth: preserve one
append-only store across many changing contexts, learn contextual addressing and consolidation, and
measure whether frozen weights improve as experience accumulates.

## Cross-lifetime interference result — 2026-07-20

The next benchmark kept one memory across conflicting visually marked contexts, trained on streams of
four contexts, evaluated on eight, froze weights during evaluation, and re-queried every early context
only after all later contexts had been stored. The context marker was present only in RGB pixels.

Three approaches were tested:

| Approach | Eight-context retention | Empty-memory retention | Garbage-memory retention |
|---|---:|---:|---:|
| sparse top-4, 5.3 writes/stream | 19.1% | 12.8% | 11.2% |
| dense writes, top-8 | 18.8% | 12.5% | 12.5% |
| full read + learned sharpness | 22.6% | 14.1% | 13.3% |

Chance is 12.5%. Every intact memory beat its corruption controls, so accumulated memory remained
causally useful. However, retention was poor and early-context performance degraded as incompatible
entries accumulated. Writing more, reading more candidates, and learning a sharper cosine-attention
temperature did not resolve the interference.

Therefore the current evidence supports **external-memory learning**, but does not yet support the
strong claim of catastrophic-forgetting-free continual learning. A flat key/value store is not enough.
The next architecture should learn generic consolidation/rehearsal operations: retrieve older entries,
re-encode or merge them, test compressed representations against replayed sensory queries, and retain
raw provenance. This conclusion is why the three unsuccessful continual checkpoints are preserved
rather than silently selecting only the successful single-lifetime result.

## Transactional learned consolidation

The first consolidation layer is implemented separately from append-only storage. A learned network
sees only controller-created latent keys, values, priorities, and ages. It selects two entries and
proposes one of three mechanical operations: merge them, retain the first, or retain the second.
It never receives task labels, glyph identities, mappings, or symbolic state.

Every proposal runs as a transaction: clone the memory, apply the candidate rewrite, replay raw
RGB/PCM queries, and optionally run a separate held-out replay set. The rewrite commits only when
accuracy remains within a configured tolerance; otherwise the original object is returned unchanged.
The two source rows are retained as immutable CPU provenance so a later durable journal can restore
or audit every rewrite. Storage savings earn a small reward, while any accuracy loss receives a much
larger penalty. This is infrastructure for learning compression—not evidence yet that the learned
policy discovers useful programs. The next experiment must train the proposal network and plot
retention, bytes, and lookup latency as the store grows against an append-only control.

### First consolidation pilot — 2026-07-20

The controller was frozen and a small REINFORCE policy sampled latent pair selection, merge/drop
operations, and merge mixtures. Each candidate had to preserve correctness independently on replay
and held-out sensory queries. On 64 deterministic eight-context streams:

| Policy | Accuracy before | Accuracy after | Rows before | Rows after | Lookup reduction |
|---|---:|---:|---:|---:|---:|
| untrained proposal policy | 17.6% | 21.5% | 8.00 | 4.39 | 45.1% |
| policy trained on 32 streams | 17.6% | 21.1% | 8.00 | 4.14 | 48.2% |

The transaction mechanism successfully removed nearly half the exact-search work without lowering
verified accuracy; removal sometimes improved accuracy by reducing retrieval interference. However,
the trained policy only improved compression by about three percentage points and did not improve
accuracy over the untrained control. Most benefit therefore came from external proposal verification,
not yet from learned consolidation. This is not evidence of learned program synthesis. The next gate
is a stronger controller baseline, many more policy trials, a validation set not used for commits,
and comparison by proposals/evaluator calls as well as stored rows.

### Untouched-audit consolidation result — 2026-07-20

The benchmark was tightened by reserving two of each stream's eight contexts as an audit partition.
Those queries never participate in accepting or rejecting a rewrite. Both policies received exactly
four proposals and 48 verifier queries per stream. The learned policy trained on 512 streams and both
conditions were evaluated on the same 256 held-out streams:

| Policy | Full accuracy before → after | Untouched audit before → after | Row reduction |
|---|---:|---:|---:|
| untrained | 19.6% → 21.8% | 23.2% → **22.5%** | 43.8% |
| trained | 19.6% → 21.6% | 23.2% → **25.0%** | 47.4% |

This supplies an initial learning signal: with equal proposal and verification compute, training both
increased compression and changed unseen-context consolidation from harmful to beneficial. It remains
a small absolute effect on a weak controller, not evidence of general program synthesis. A stronger
memory controller and repeated-seed confidence intervals are required next.

### Three-seed replication and verifier ablation — 2026-07-20

The earlier comparison reused one evaluation dataset across policy seeds. The corrected experiment
paired trained and untrained policies within each seed while changing both the training and held-out
task streams across seeds 11, 23, and 37. Each condition still received four proposals and exactly 48
verification queries per stream, over 128 evaluation streams per seed.

| Condition | Untouched audit accuracy | Rows retained | Lookup reduction |
|---|---:|---:|---:|
| append-only | 24.6% | 8.00 | 0% |
| untrained proposals | 18.9% ± 1.5% | 5.18 | 35.3% |
| trained proposals | **23.4% ± 2.4%** | **4.44** | **44.5%** |

The trained policy beat its matched untrained control on untouched audit accuracy in all three seeds
by 1.6, 5.5, and 6.6 percentage points, and matched or improved compression. This replicates a real
learned proposal advantage. It does **not** establish safe consolidation: trained memory still lost
about 1.2 points versus append-only on average, and two of three seeds regressed on unseen contexts.

A focused verifier ablation additionally required non-increasing cross-entropy loss on both checked
partitions. It retained 7.81 of 8 rows on average (only 2.4% compression) while untouched audit still
averaged 23.8%, below the 24.6% append-only baseline. A zero-loss confidence guard is therefore too
conservative without solving unseen-context safety. The next path is broader/stochastic rehearsal
coverage or a learned risk estimator, not simply tightening a scalar acceptance threshold.

### Sensory-variant consolidation milestone — 2026-07-20

The preceding untouched-context audit had a conceptual flaw: memory belonging to audit contexts was
eligible for deletion even though no behavior depending on it could participate in verification. The
benchmark now rehearses every stored association using its original public query. Its untouched audit
uses the same glyph, visible context, and uniquely correct action rendered with a new background and
new PCM. Audit views never influence commit decisions, and evaluation seeds are separate from policy
training. This tests representation stability under sensory variation instead of asking the verifier
to protect experiences it has never observed.

Across three independent seeds and 128 evaluation streams per seed, the existing trained proposal
policies produced the first consistent result above append-only on every seed:

| Condition | Audit before → after | Audit gain | Lookup reduction | Rows retained |
|---|---:|---:|---:|---:|
| trained, four proposals | 21.94% → **22.79%** | +0.85 points | **45.05%** | 4.40 / 8 |
| trained, per-context checks | 21.94% → 22.56% | +0.62 points | 43.03% | 4.56 / 8 |
| trained, three proposals | 21.94% → 22.66% | +0.72 points | 34.41% | 5.25 / 8 |
| untrained, four proposals | 21.94% → **24.87%** | **+2.93 points** | 37.73% | 4.98 / 8 |
| trained with variant reward | 21.94% → 22.82% | +0.88 points | 44.63% | 4.43 / 8 |

All four-proposal conditions used 64 verifier queries per stream. Aggregate verification outperformed
finer grouping, so compensation among checked queries was not the primary problem. Consolidation can
now remove roughly 45% of exact-search rows while improving independently rendered queries in every
trained seed. This is a reproducible **transactional compression** result.

It is not yet a learned-policy win. The untrained policy dominates the trained policy's observed
accuracy/compression frontier: it improves audit accuracy more while compressing slightly more than
the trained three-proposal condition. Adding one training-only sensory variant as delayed reward made
only a negligible improvement. The current high-variance REINFORCE objective mainly learns aggressive
acceptance. The next selected architecture is candidate tournament distillation: sample several
rewrites, score their accuracy/storage Pareto value on training-only variants, and supervise the policy
to rank the winning proposal. Evaluation variants remain untouched.

### Latent candidate-tournament milestone — 2026-07-20

The next policy replaced one-sample REINFORCE with a latent candidate tournament. At each training
transaction it samples four rewrites from controller-created keys and values, rejects candidates that
damage protected sensory behavior, ranks survivors on training-only alternate RGB/PCM views, and
distills the winning proposal. Candidate contents remain continuous latents; there are no text tokens,
semantic labels, symbolic facts, or proof traces.

Only 64 training streams per seed were used. On three independent 128-stream evaluations:

| Policy and budget | Audit gain | Lookup reduction | Verifier queries |
|---|---:|---:|---:|
| ordinary learned, 3 proposals | +0.72 points | 34.4% | 48 |
| **tournament learned, 3 proposals** | **+1.56 points** | 32.7% | 48 |
| ordinary learned, 4 proposals | +0.85 points | 45.1% | 64 |
| **tournament learned, 4 proposals** | **+1.92 points** | 41.9% | 64 |
| untrained, 5 proposals | +3.22 points | 44.6% | 80 |
| **tournament learned, 5 proposals** | **+2.15 points** | **50.4%** | 80 |

Tournament training more than doubled the learned policy's generalization gain at comparable proposal
budgets. At five proposals it removed half the exact-search memory while improving unseen sensory-view
accuracy in every seed. This is a significant improvement in the **learned** compression frontier.

The random control remains important: at equal 80-query compute it gains more accuracy but compresses
less, so neither policy Pareto-dominates the other. Tournament distillation has therefore improved
learning substantially, but has not yet proved that learned proposal selection is universally better
than random search. The next targeted change should train a conditional frontier policy with an
explicit requested compression level, allowing learned and random policies to be compared at exactly
matched retained rows rather than at a few discrete proposal counts.

### Autonomous latent stopping milestone — 2026-07-20

The consolidator now has a candidate-conditioned continuous `COMMIT`/`STOP` head. It sees only the
current controller-created latent rows and a proposed latent replacement. During tournament training,
every sampled candidate supplies a dense label: commit if it beats retaining the store on protected
behavior and training-only sensory variants, otherwise stop. No retention percentage is supplied.

The first pooled-store stop head failed by predicting either continue everywhere or stop everywhere.
Candidate conditioning plus dense labels produced variable held-out behavior. With a correctness-first
utility, storage value 0.01, and stop-label weight 0.5, three independent seeds produced:

| Result | Autonomous | Forced to five rewrites |
|---|---:|---:|
| Mean stop rate | **46.1%** | 0% |
| Mean rows retained | 4.91 / 8 | 3.85 / 8 |
| Lookup reduction | 38.7% | 51.9% |
| Verifier queries | **59.3** | 80.0 |
| Untouched audit gain | **+1.24 points** | +2.05 points |

Stopping behavior differed substantially by seed (82.0%, 29.7%, and 26.6%), demonstrating that it is
not a fixed row-count schedule. Every autonomous seed improved unseen sensory-view accuracy over the
append-only baseline while cutting verifier work by about 26%.

The decisive causal gate did **not** pass: forcing continuation improved audit accuracy by 1.17, 0.88,
and 0.39 points on the three seeds. The current policy has learned an adaptive accuracy/compute tradeoff,
not accuracy-optimal halting. The next stop target must predict the *future value of all remaining
rewrite opportunities*, rather than only whether the next sampled candidate beats the current store.
This suggests a latent continuation-value head trained from complete tournament rollouts.

### Full-trajectory continuation-value milestone — 2026-07-20

The myopic target was replaced with completed-rollout supervision. Training deliberately executes all
five tournament rounds, records each latent memory state, and measures the eventual correctness-first
utility of every later state using training-only alternate RGB/PCM queries. A state receives `CONTINUE`
only if some later state on that rollout is actually better; otherwise it receives `STOP`. At inference
the head sees only pooled controller-created latent memory, including its learned contents, usage, age,
and size. It does not receive sensory answers, verifier scores, semantic labels, or a target row count.

A probability threshold of 0.60 was selected on seed 11 and then frozen unchanged for independent
policies and held-out streams from seeds 23 and 37. Each policy was evaluated on 256 streams and paired
with the same policy forced through all five rewrite opportunities:

| Seed | Original accuracy delta vs forced | Variant-audit delta vs forced | Verifier queries | Stop rate | Row reduction |
|---:|---:|---:|---:|---:|---:|
| 11 | **0.00 points** | **0.00 points** | **78.875 / 80** | 5.47% | 55.37% |
| 23 | **0.00 points** | **0.00 points** | 80.000 / 80 | 0.00% | 55.57% |
| 37 | **0.00 points** | **0.00 points** | **79.063 / 80** | 4.69% | 55.52% |

Across 768 held-out streams, autonomous stopping preserved both original-query and independently
rerendered visual/audio audit accuracy exactly on all three seeds. It saved verifier work on two of
three independently trained policies, reducing queries by 0.86% on average, while the conservative
seed-23 policy correctly chose never to stop early. Mean memory-row reduction remained 55.49%.

This passes the causal gate that the candidate-conditioned stop head failed: forcing extra computation
no longer improves measured accuracy. The compute saving is small, so this is not an efficiency
breakthrough, but it is the first replicated evidence that the system can learn a latent-only,
variable-compute stopping decision without sacrificing task performance. The next experiment should
calibrate each independently trained value head on training-only streams, freeze that operating point,
and test whether the safe savings increase on disjoint evaluation streams. The implementation now
supports this with `--calibration-streams`; no evaluation outcomes are exposed to the deployed head.

### Evidence-calibrated latent stopping — 2026-07-20

The fixed 0.60 threshold above was useful evidence but was originally selected using seed-11
evaluation outcomes. The corrected protocol calibrates each independently trained value head on 64
training-only sensory streams at seeds disjoint from both policy training and final evaluation. A
cutoff is deployable only when it demonstrates all of the following during calibration: fewer verifier
queries, no original-query correctness loss, no sensory-variant correctness loss, and no increase in
cross-entropy on either view. If calibration observes no actual early stops, the cutoff is rejected as
unsupported and the exact forced-continuation threshold of 1.01 is used.

This last condition matters. An earlier version accepted seed 23's threshold 0.50 because it happened
never to stop on its 64 calibration streams. It then stopped on 3.52% of unseen streams and lost 0.05
original and 0.10 audit percentage points. The evidence requirement correctly rejects that false-safe
calibration rather than hiding the negative result.

With the corrected rule, three independently trained policies produced:

| Seed | Calibrated threshold | Calibration evidence | Evaluation stop rate | Queries saved vs forced | Original/audit delta |
|---:|---:|---|---:|---:|---:|
| 11 | 0.60 | safe savings observed | 5.47% | **1.125 / 80** | **0.00 / 0.00 points** |
| 23 | 1.01 | no savings observed; fallback | 0.00% | 0.000 / 80 | **0.00 / 0.00 points** |
| 37 | 0.55 | safe savings observed | 7.42% | **1.438 / 80** | **0.00 / 0.00 points** |

Across 768 disjoint evaluation streams, both correctness measures exactly match forced continuation on
all three seeds. The two policies with positive calibration evidence generalize their savings; the
unsupported policy conservatively abstains. Mean verifier work falls 1.07%, slightly improving the
fixed-threshold result, while mean memory-row reduction remains 55.35%.

This is the first leakage-free demonstration of a learned latent resource controller that adapts its
compute policy across independently trained models, preserves measured accuracy under a causal forced-
continuation intervention, and falls back safely when its own experience provides no evidence for
early stopping. Savings are still small. The next efficiency target should increase the frequency of
clearly dominated continuation states, either through richer permutation-invariant memory summaries
or direct regression of future utility magnitude instead of binary STOP labels.
