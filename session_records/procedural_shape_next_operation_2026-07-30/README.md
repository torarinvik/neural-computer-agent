# Procedural shape `next item`: gradual curriculum and stability frontier

## Verified progress

The same controller that mastered direct lookup and `previous item` learned the
first `next item` anchor using only RGB streams, its own opaque actions, and
scalar attempted-action outcomes.

- span-2 atom: 96.45% overall, 98.83% causal conflicts;
- span-3 first anchor, one query: 95.90%;
- first anchor at the second query position: 95.62%;
- first anchor across all three query positions: 98.33% overall, 97.77%
  `next`, 97.46% causal conflicts;
- previous-item retention at that point: 99.22%.

The span-2 atom required 11,520 target verifier outcomes, versus 14,592 for
the earlier `previous item` atom. This is promising transfer, not yet an
independent replicated sample-efficiency result.

The research checkpoint `seed41151` retains the mastered first anchor at
98.89% and previous-item behavior at 99.29%, while carrying the partially
learned second-anchor frontier. It is intentionally not promoted as mastery.

## Localization of the second-anchor failure

The raw second anchor initially remained near chance. A deterministic control
made item three redundant with item one while preserving the second-anchor
query. Accuracy immediately became 98.75% (`next`) and 98.31% on causal
conflicts without training. Therefore the controller had learned the new
relative anchor; the failure was binding it to a genuinely independent third
memory item.

A 2.5% mixture of fully independent third items was the wrong gradualization.
It produced superficially high 99% aggregate accuracy while the independent
subgroup stayed near chance and its conflicting subgroup fell as low as 0%.
Loss weights of 4, 8, and 16 did not repair this. The strict gate now includes
both independent and conflicting-independent subgroup accuracy, so the easy
majority cannot hide this failure again.

## Target-aligned bridge

A new verifier-balanced bridge asks for the same stored third item through
either direct lookup or `next from item two`. It removes cue disambiguation
temporarily while leaving the learner-visible protocol unchanged.

Zero-shot `next` accuracy was 58.12%. After only 3,072 target outcomes it
reached 84.33% (87.11% on independent conflicting slots), proving that the
bridge supplies a strong learning signal. However, previous-item behavior
fell to 89.10%. A gentler run reached 71.94% from 1,536 target outcomes but
again reduced previous-item causal conflicts to 81.46%.

A subsequent old-skill consolidation phase restored previous-item behavior to
97.80%, but the new bridge fell back to 54.77%. The abilities currently
compete in shared parameters; alternating acquisition and consolidation does
not retain both.

## Protected-plasticity experiments

Three bounded controls separated safety from learnability:

| method | new aligned-next | previous overall | previous conflict |
|---|---:|---:|---:|
| ordinary updates | 71.94% | 89.36% | 81.46% |
| frozen gated action adapter | 57.25% | 99.23% | 98.48% |
| usage protection strength 3 | 64.84% | 91.88% | 77.50% |
| usage protection strength 10 | 59.05% | 97.57% | 94.31% |

The new generic usage mechanism stores a per-parameter exponential moving
average of gradient use. Recent use lowers plasticity; importance decays when
unused. The importance state is checkpointed. This directly tests the proposed
volatility idea, but the scalar-only rule exposes rather than solves the
stability-plasticity tradeoff.

## Direction-aware protected plasticity: breakthrough

Aggregate gradient projection produced the first checkpoint that learns the
independent third-item relation while retaining every major old-skill gate.
For each target update, the trainer sums the current cycle's verified
rehearsal gradients. It leaves a compatible target gradient untouched and
removes only the component pointing against that aggregate old-skill
direction.

| run | target bits | new `next` | new conflict | redundant anchor | first `next` | `previous` | previous conflict |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary seed 41901 | 1,536 | 64.00% | 73.23% | 99.12% | 98.29% | 96.79% | 97.74% |
| replica seed 42101 | 1,536 | 64.45% | 71.37% | 97.75% | 96.40% | 95.26% | 96.06% |
| shuffled target outcomes | 1,536 | 52.99% | 50.86% | 96.71% | 91.42% | 91.91% | 94.87% |

Both truthful runs exceed the 58.12% zero-shot `next` baseline and preserve
all three old-skill overall and causal-conflict gates above 95%. Complete
memory reset remains at chance. The matched shuffled-target control keeps
rehearsal truthful but destroys the new relation, ruling out learning from the
schedule, projection rule, or renderer alone.

A continuation from the primary checkpoint reached 92.36% `next` and 91.52%
new causal-conflict accuracy after 4,608 additional target bits. Redundant and
first-anchor skills remained above 99%; `previous item` remained 98.01%
overall but its conflict subgroup fell to 94.43%, just below the strict gate.
Early target gradients conflicted with rehearsal and were projected. Later
gradients became strongly compatible and were left untouched, direct
mechanistic evidence that protection can guide learning into a shared
direction rather than permanently freezing the controller.

Two tempting variants were rejected:

- projecting against each rehearsal stream separately overconstrained the
  update and left `previous item` at 94.05% overall / 90.77% conflict;
- duplicating `previous item` rehearsal restored it after the long run but
  regressed the new relation from 92.36% to 84.85%.

The promoted checkpoints are the 1,536-target-bit primary and independent
replica. The 92% continuation is retained as a report, not promoted, because
it misses the complete retention gate.

## Constraint-only rehearsal and trust-region breakthrough

Per-stream telemetry localized why the long continuation failed. At the
promoted checkpoint, the target gradient was locally compatible with all
three retained skills. At the 92% checkpoint it opposed `previous item` by
−0.782 cosine; aggregate projection removed the summed conflict but still
left a −0.235 residual against that individual skill.

Several sub-minute probes then separated tempting explanations:

- softly protecting the worst residual stream passed once but did not
  replicate its causal gain;
- AdamW's actual aggregate parameter displacement was already locally safe,
  so post-optimizer projection was inactive and could not explain the drift;
- using rehearsal only as a gradient constraint accelerated the new relation
  but catastrophically forgot old skills at learning rate 0.001;
- reducing the learning rate exposed a smooth retention frontier: 0.00025
  and 0.00005 were near misses, while 0.000025 crossed every gate.

The successful rule performs twelve rehearsal gradient evaluations but no
old-skill optimizer steps. Four target batches are the only weight-changing
updates. Each target gradient is projected against the aggregate verified
rehearsal direction, and the 40× smaller learning rate keeps the update inside
the region where that first-order constraint remains valid.

| run | parent `next` | child `next` | parent conflict | child conflict | weakest retained overall | previous conflict |
|---|---:|---:|---:|---:|---:|---:|
| seed 42801 | 63.93% | 69.99% | 70.43% | 78.58% | 96.57% | 97.61% |
| seed 42851 | 64.71% | 69.99% | 74.29% | 80.49% | 96.53% | 96.93% |

Both runs consume 1,536 target verifier bits, 10,752 rehearsal verifier bits,
6,144 unique lifetimes, 16 gradient evaluations, and only four optimizer
updates. Full memory reset remains at chance.

A same-seed causal comparison gives the honest control:

| seed 42901 | `next` | causal conflict | independent new slot |
|---|---:|---:|---:|
| unchanged parent | 64.45% | 72.27% | 65.04% |
| shuffled target outcomes | 66.99% | 76.77% | 67.32% |
| truthful target outcomes | 69.79% | 78.43% | 70.83% |

Tiny shuffled updates act as weak regularization, so “shuffled returns to
chance” would be a false claim at this inherited frontier. Truthful outcomes
nevertheless win the fully paired comparison on every new-skill measure while
all old-skill gates pass. The result is a replicated incremental
protected-learning milestone, not yet second-anchor mastery.

## Replicated compounding increment and saturation boundary

The same four-update recipe was applied once more to both independently
promoted trust-region checkpoints:

| run | parent `next` | child `next` | parent conflict | child conflict | previous overall | previous conflict |
|---|---:|---:|---:|---:|---:|---:|
| seed 43001 | 69.66% | 71.61% | 79.46% | 81.26% | 96.50% | 95.96% |
| seed 43051 | 68.23% | 71.68% | 78.07% | 82.09% | 96.44% | 95.60% |

Thus persistent experience produces a second replicated gain rather than
merely reproducing the first checkpoint. Each lineage has now consumed 3,072
target bits after its original protected parent, with eight weight-changing
updates total across the two safe increments.

The exact seed-43001 control confirms experience direction:

| arm | `next` | causal conflict | independent new slot |
|---|---:|---:|---:|
| unchanged parent | 69.66% | 79.46% | 70.18% |
| shuffled target outcomes | 69.21% | 80.36% | 70.18% |
| truthful target outcomes | 71.61% | 81.26% | 72.33% |

Shuffled outcomes provide no `next` or independent-slot gain. Truthful
outcomes improve every new-skill measure while retaining all old skills.

Because the weakest retention gates approached 95%, a third increment used
half the step size. It moved `next` only 70.96% → 71.42%, left causal
conflicts unchanged at 79.77%, and was rejected. More identical micro-steps
are therefore low ROI: the method demonstrates genuine compounding twice, but
then reaches a local saturation boundary.

## Conclusion and frontier

Successful:

1. decompose `next item` by anchor and query position;
2. audit rare logical subgroups rather than aggregate accuracy;
3. use target alignment to create dense, informative experience;
4. keep acquisition runs under one minute until both learning and retention
   move correctly.

Rejected:

1. sparse mixtures of maximally hard examples;
2. larger novelty weights;
3. acquisition followed by consolidation;
4. a frozen bolt-on adapter;
5. scalar gradient suppression as the complete solution.

Direction-aware aggregate projection plus constraint-only rehearsal and a
very small trust region remains the reliable stability mechanism. The
remaining error was then decomposed rather than attacked with more identical
updates.

## Error-balanced curriculum breaks the saturation boundary

A 6,144-example audit exposed a shared shortcut in both compound-2 lineages.
Conflict cases were already 82.47–82.51%, while non-conflict cases were only
59.36–60.68%. The hardest cell was a non-conflict query whose correct answer
was action zero: 52.31–52.90%, essentially chance. The controller had partly
learned “`next` means the identity changes” instead of the complete relation.

Several sub-minute repairs were rejected:

- focal loss improved the hard cell but did not improve causal conflicts;
- weighting only non-conflicts improved the hard cell while regressing
  conflicts;
- equal 2× weighting of both groups passed once but did not replicate its
  causal gain.

The smallest replicated repair was verifier-side error balancing: 3× loss
weight for conflict examples and 2× for non-conflicts. These weights are
generator metadata used only to allocate learning signal; they are never
learner-visible.

| matched high-precision arm | `next` | conflict | non-conflict | non-conflict/action-0 | new slot |
|---|---:|---:|---:|---:|---:|
| parent | 72.15% | 81.60% | 62.55% | 52.99% | 72.64% |
| child | 73.50% | 82.02% | 64.85% | 56.81% | 73.96% |

An independent lineage also improved every corresponding measure by roughly
0.8–0.9 points. The replicated checkpoint's 12,288-example retention audit
scores 99.41% redundant-anchor, 98.33% first-anchor, and 96.30% previous-item
overall / 95.65% previous-item conflicts. Complete memory reset is 50.08%.

On the exact paired control seed, the truthful child scores 74.41% `next`
versus 72.59% for its unchanged parent and 72.98% for shuffled outcomes. Its
hard non-conflict/action-zero cell reaches 55.85%, versus 52.39% parent and
51.33% shuffled. This rules out schedule-only regularization as the cause.

## Rejected follow-up: equal 3× group weighting

Increasing non-conflict emphasis from 2× to 3× has a real corrective effect,
but it spends more retention margin than the present aggregate gradient
constraint can safely protect. From an exact parent, four target updates raise
the hard cell 53.37% → 57.51% and non-conflicts 62.40% → 65.01%, but reduce
previous-item conflict retention 95.69% → 95.00%. The independent full run
falls further, to 94.76%, and is rejected.

Halving this to two target updates retains the first lineage (95.43% previous
conflicts) while preserving a 53.37% → 56.48% hard-cell gain. Its independent
replica again misses the gate at 94.93%. Therefore smaller duration does not
make the 3×/3× rule robust; the limitation is functional retention protection,
not insufficient target emphasis. The reports are retained as rejected
controls.

## Fresh functional-retention instrument

The trainer now supports rollback of a target update when fresh, independently
generated rehearsal lifetimes lose more deterministic accuracy than a declared
tolerance. Validation outcomes are counted separately from target and
rehearsal outcomes. A one-target-update smoke test with 384 fresh lifetimes
per retained stream consumed 2,688 validation outcomes and preserved all
three validation scores within the initial 1% tolerance. This verifies the
instrument and accounting only; a same-batch functional check was rejected as
too weak because it missed a later held-out retention failure.

The next refinement fixed the anchor itself and checked both overall and
causal-conflict accuracy. With zero tolerated drop, it rejected one of four
3×/3× target updates and retained previous-item conflicts at 95.01%. But the
three accepted updates did not improve the new relation, and the 2,688 unique
anchor outcomes cost more than the 384-outcome target update. Thus broad
functional validation is a verified safety tool, not a sample-efficient
breakthrough. The next candidate should use a smaller, conflict-specific
anchor instead of a broad three-stream validator.

The promoted 3:2 checkpoints define the new frontier. The next experiment
should preserve the replicated directional learning while adding a
function-level acceptance or backtracking check for the old skill. That would
test whether corrective 3×/3× updates can be accepted only when they preserve
actual old-task behavior, rather than relying on a local gradient proxy.

## Rejected follow-up: conflict-specific functional validation

The broad anchor was narrowed to only the fragile previous-item conflict
stream, while gradient rehearsal still used all three old-skill streams. This
reduced a 384-lifetime validator from 2,688 to 1,152 unique verifier outcomes
per anchor. The zero-tolerance smoke correctly rejected a target update that
dropped the fixed anchor's conflict score by 0.34 points, confirming that the
smaller check remains active.

At a deliberately modest 1% per-score tolerance, the 64-update run completed
16 target updates and spent 1,152 fresh anchor outcomes (36,864 deterministic
anchor evaluations). It did not improve the held-out conflict/new-slot target:
74.22%, versus the promoted 3:2 frontier of 73.50% only within the small-run
noise band. Its final target update was rejected after a 0.36-point anchor
conflict drop. This is a useful efficiency improvement to a safety instrument,
but it did not turn 3×/3× weighting into a viable acquisition strategy. The
report is saved as `functional_conflict_specific_rejected.json`.

## Sanity control, not transfer evidence

An equal-budget 64-update comparison started the promoted controller and a
fresh controller on the same already-mastered span-three `next` relation. The
parent reached 77.86% held-out conflict/new-slot accuracy while the fresh
controller remained at 50.26% chance, after the same 6,144 target outcomes.
This verifies that the parent checkpoint, rather than a schedule artifact,
contains the useful capability. It is **not** evidence of compounding learning:
the parent had already been trained on this exact target. The saved reports are
`transfer_same_skill_parent_sanity.json` and
`transfer_same_skill_fresh_sanity.json`.

The correct next experiment must use a genuinely new but adjacent target—for
example a fourth presented item under the same `next` relation—while matching
the target outcome budget against a fresh controller. Only a faster rise on
that unseen target counts as transfer or compounding sample efficiency.
