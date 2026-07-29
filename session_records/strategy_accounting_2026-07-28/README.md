# Strategy accounting and forward decision — 2026-07-28

## Objective

Maximize verified reusable capability per unique interaction while preserving
older capabilities. Accuracy is sovereign; internal compute and response
latency are secondary costs. The deployed agent must learn from sensory
streams, opaque actions, latent state/memory, and scalar verifier outcomes—not
semantic task IDs, game state, correct-action labels, or hand-written solvers.

## Strategies with the strongest evidence

### 1. Extremely gradual, factored curricula

Splitting hard tasks into deterministic perceptual, binding, memory, and
composition atoms repeatedly turned flat learning into measurable progress.
The most important debugging rule has been to ask whether the curriculum jump
was too large before changing the architecture.

### 2. Unique experience before replay

Additional unique logical lifetimes consistently beat spending the same budget
replaying too little evidence. More optimizer updates alone did not rescue the
32-outcome identify-then-act learner. For the six-action family, however,
additional processing of each already-seen experience was useful up to a
measured sweet spot: 48 replay updates reduced verifier bits to stable mastery,
while 56 entered an overthinking regime. A fixed 56-early/48-late schedule
reduced verifier experience by 13.5% on fresh seeds and passed the causal audit.

### 3. Population search with exact held-out promotion

Initialization variance was much larger than downstream readout variance.
Successive halving over predictive-core initializations found a replicated
48-outcome learner and reduced the prior 64-outcome frontier by 25%. Search
compute is accounted separately, and winners must reproduce on disjoint
streams and retain old capabilities.

### 4. Freeze proven knowledge; add zero-output plastic residuals

Monolithic fine-tuning caused catastrophic or selective forgetting. Frozen
consolidated cores plus small, zero-initialized task-agnostic adapters preserved
old behavior exactly at insertion, learned new compositions, and allowed
harmful inherited branches to be reset without discarding useful ones.

### 5. Balanced behavioral rehearsal and immutable retention gates

Rehearsing complete sensory/action/outcome trajectories against frozen prior
controllers merged independently learned skills without semantic labels.
Retention is measured against the exact parent on matched lifetimes; an
isolated threshold crossing is never treated as mastery.

### 6. Separate latent reading from behavioral writing

Exact-zero write gates removed interference but also made deep ancestry
bit-identical to new slots. Letting a new slot consult earlier pre-gate hidden
representations while their writes remain shut restored 3 → 4 transfer:
+0.0242 pooled, 48W/22L, p = 2.5e-3. A zero-content capacity control failed,
and the full audit retained every prior skill.

The newest factorial accounting strengthens this: at 4 → 5, reading improves
absolute new-skill accuracy by +0.0944 for the shallow parent and +0.0785 for
the deep parent. The problem is not whether latent reuse helps; it is how to
make the benefit grow with a larger library.

### 7. Learned external memory with causal physical audits

The single controller now supports sparse latent writes, active RAM/workspace,
disk persistence, content-addressed reads after active-state erasure, learned
read rejection, redundant-write suppression, bounded replacement, and
frequency/recency/reliability utility. Empty, shuffled, corrupted, and
save/reload controls causally establish memory dependence. Tiny generic
residuals adapted utility online from verified outcomes while older skills
remained intact.

### 8. Probe both sides of every boundary before repair

Discarded diagnostic probes localized failures from perception through
recurrent state, write, consolidation, recall, and action. They repeatedly
prevented architecture work aimed at information that was absent upstream.
Probe weights never enter the deployed agent.

### 9. Valid counterfactual rerenders and adversarial controls

Pixel-level reversal, missing-evidence, shuffled-feedback, reward-shuffled,
memory-corruption, active-state-reset, and exact-reload audits have caught
watermarks, malformed hidden-state swaps, false thresholds, and reward-hacking
shortcuts. These controls are part of the capability definition.

### 10. Learn the smallest decision boundary around frozen reusable structure

The natural-memory relation scorer already separated behavioral equivalence
well but used an arbitrary raw-logit threshold poorly. Training only a scalar
scale and bias from 64 verifier outcomes converted that frozen representation
into a reliable online consolidation policy: 16 natural memories became two,
with roughly 99% held-out behavior and both skills retained. Thirty-two bits
were seed-sensitive, so the replicated threshold—not the lucky minimum—was
promoted. This is a high-return pattern: diagnose what the frozen system
already represents, then learn the smallest missing action boundary before
adding architecture.

### 11. Preserve bounded variation inside behavioral equivalence classes

Compressing all equivalent experiences to one prototype was optimal on the
observed distribution but lost 2–3 points under a disconnected-object shift.
Keeping two learned-equivalent representatives per behavior recovered the
cross-appearance gate with zero new training outcomes while retaining a 4×
logical compression. The lesson is not “keep redundancy”; it is “distinguish
useless duplication from variation whose future value is uncertain.” The next
resource learner should adapt this allowance from verified transfer value.

### 12. Predict marginal success, not absolute success, for compute allocation

When the cheap action already succeeds around 99% of the time, an ordinary
success predictor learns only the majority answer. Predicting the verified
difference between shallow and deep execution exposes the rare decision that
matters. The resulting critic preserves 99.57% accuracy while reducing latent
comparisons by 65.1%. This is the operational form of “think more only when it
helps”: action-conditioned marginal value, with ground-truth outcomes
sovereign and compute cost secondary.

## Bounded negatives and dead ends

These are rejected only within the measured regime; a fixed-budget negative is
not an impossibility proof.

### Architecture and transfer

- Ordinary joint fine-tuning learned new tasks but erased earlier skills.
- Exact-zero gating alone protects retention but blocks ancestry by
  construction.
- Reading every ancestor helps absolute learning, but a second readable slot
  does not add a compounding depth benefit.
- Compressing all ancestor reads to 16 or 32 dimensions does not restore that
  depth differential; input width/dilution is not the main explanation.
- Reading only the most recent ancestor is directionally better than reading
  both at 96 updates, but the +0.0191 margin is not yet reliable and the
  deep-versus-shallow gap remains −0.0205.
- Frozen cue capacity is nearing its limit because global average pooling loses
  spatial detail. Reusing the same weak cue interface indefinitely is not a
  credible scaling strategy.

### Experience and compute allocation

- More replay cannot substitute for missing unique verifier outcomes.
- Gate-selected replay saves almost nothing when safe; aggressive selection
  breaks retention because replay itself keeps the gate shut.
- A learned local-loss/recent-window stopper predicted observational value but
  failed causal omission tests.
- Early learned allocators overfit tiny counterfactual datasets; stationary
  i.i.d. streams often contain no state-local signal about which future compute
  budget will win.
- Fifty-six replay updates reduced bits in some cells but regressed final
  utility often enough to fail the accuracy-first objective.

### Curriculum and task design

- Several apparent failures were invalid tasks: contradictory policies,
  observationally identical inputs requiring different answers, or a second
  support that supplied no new information.
- Multi-support contextual mapping was too large a jump even though discarded
  probes showed the necessary information was represented.
- Learning line tracing, temporal binding, memory writing, and answer routing
  simultaneously produced misleading failures. Factoring them was essential.

### Predictors and auxiliary objectives

- Extra readout capacity and optimizer freedom did not solve the 32-outcome
  frontier.
- A reward-free predictive-objective screen failed its held-out gates; the
  unrefined core ranked best.
- Passive success/value prediction can redistribute observed outcome
  information but does not create new information. It should guide thought,
  retrieval, and curriculum only after calibration and causal audits.

### Process failures now guarded against

- Fixed-budget flat curves were overinterpreted before training fit was checked.
- Recurrent hidden-state tensor swaps created out-of-distribution audits;
  counterfactual episodes must be rerendered and replayed.
- Snapshot augmentation can leak logical lifetimes across splits; all variants
  now remain on one side of the split.
- Stale binaries/checkouts, orphaned workers, silent empty campaigns, script
  self-modification, and uncapped runs consumed compute without evidence.
  Clean worktrees, exact tests, runner snapshots, streamed reports, and a hard
  five-minute cap now prevent repeats.

## Conclusion

The most promising system is not a larger monolithic reasoner. It is:

1. one compact recurrent controller and modality encoders;
2. active RAM/VRAM for immediate state and task-sized working memory;
3. causal, selectively accessed disk memory for durable experience;
4. frozen proven knowledge plus small zero-output plastic mechanisms;
5. separate read and write channels so knowledge can be consulted without
   disturbing old behavior;
6. verifier-driven population/curriculum selection optimized for stable
   learning curves, retention, transfer, and latency.

We have demonstrated reusable primitives, few-shot causal binding, narrow
compounding transfer, physical persistent memory, online utility adaptation,
and strong anti-fluke audits. We have not demonstrated an accelerating transfer
curve across a growing skill library, a learned general-purpose internal
optimizer, unbounded consolidation, or broad cross-modality/amodal transfer.

## Highest-ROI next experiment

The next experiment stays below one minute and changes one variable:

1. keep the five-skill parent, task, seeds, 96-update budget, replay, and
   retention screen fixed;
2. expose exactly one prior slot;
3. compare the immediately preceding slot against the older slot;
4. reuse the existing no-read, all-read, and recent-only controls.

If one ancestor is consistently more useful (prospective bar: at least +0.02
mean and 6/8 paired wins over the other), the next mechanism is a learned
task-agnostic latent selector trained only through verified downstream
improvement. If the single-ancestor arms are flat, ancestor routing is not the
bottleneck and the next probe should test representational complementarity
before any selector or longer run is built.
