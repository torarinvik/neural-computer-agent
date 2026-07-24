# Temporal event-binder integration ladder

Status: rung 1 passed. The reward-only plumbing pilots were intentionally below the measured
generalization scale and are complete; the next branch is supervised bootstrap of the real
write-path binder, followed by behavioral fine-tuning. Diagnostic classifier weights remain
discarded.

The accepted diagnostic assets are the generic architecture and data/audit recipe, not the
diagnostic classifier. Pure reward discovery remains the stronger open claim. The economical
milestone branch now permits verifier labels during an explicitly reported supervised bootstrap:
a temporary head predicts the rule from the real raw write row, the head is discarded, and the
binder and reader continue through behavioral fine-tuning.

## Pre-registered rungs

1. **Exact no-op architecture gate.** Add a three-event recurrent snapshot buffer and pairwise
   write binder behind a zero-output residual gate. Loading the old checkpoint must leave every
   existing output tensor bit-identical before training. Spatial and shape evaluation must also be
   unchanged.
2. **Raw-write representation.** After a small reward-only direct-colour pilot, the demonstrated
   first/last rule must decode from the raw write row at >=65% held-out and exceed the matched
   shuffled-label calibration. If it does not, do not tune consolidation or recall.
3. **Consolidation survival.** The compact row must retain rule decodability within five percentage
   points of the raw write and remain >=65%. A larger drop localizes the next repair to
   consolidation.
4. **Recall survival.** The query-time recalled vector must retain decodability within five points
   of compact memory and remain >=65%. A larger drop localizes the next repair to retrieval.
5. **Behavioral use.** A fresh reward-trained agent must achieve >=65% held-out temporal behavior
   after demonstrations and beat its zero-shot/order-blind baseline by at least ten percentage
   points. Correctness is the gate; early-learning AUC ranks configurations only after they pass.
6. **Causality and retention.** Reversing object events before sensory replay while fixing rewarded
   identity must preserve correctly relabeled accuracy within five points of normal, flip at least
   90% of predictions, and leave stale-label accuracy <=10%. Spatial and shape accuracy may fall by
   at most two percentage points from the frozen baseline.
7. **Original-renderer graduation.** Repeat the representation, behavioral, reversal, and retention
   gates with the original thin-line feedback. Direct colour remains the efficient binding
   curriculum; line following remains a separately measured primitive.

## Stopping and data rules

- All render variants of one logical lifetime stay in one split.
- Maximize distinct logical lifetimes first; add render-seed variants as invariance pressure.
- Expect a flat optimization valley. A reward-only run is not negative before substantially
  exceeding the comparable supervised ignition window; monitor training loss/accuracy slope.
- Stop at the first failed rung and probe that boundary. Do not repair downstream consumers before
  verifying their input contains the required information.
- No further diagnostic-binder architecture forks unless integration falsifies the accepted module.

## Credit-assignment fork

The first binder-only pilot tests learning-signal reachability, not architectural capacity. Log
binder gradient norm and residual RMS, and probe every periodic checkpoint at raw write, compact
memory, and recall. Flat behavior with near-zero gradients means the objective is disconnected.
Healthy gradients and a growing residual without rule decodability means the behavioral signal is
too weak to discover the representation in this budget. Rule-positive writes without behavioral
gain mean the frozen reader cannot use the new representation.

The pre-registered response is joint binder-plus-reader training next, not another binder design.
If that still cannot cross the representation gates, a supervised initialization from the cached
snapshot task may be used and retained, followed by behavioral fine-tuning. That branch must be
reported explicitly as **supervised-bootstrapped**, a weaker claim than reward-only emergence.

## Completed plumbing pilots

The exact no-op gate passed: enabling the integrated event binder leaves every pre-existing output
bit-identical, gradients reach both its zero-initialized output layer and upstream layers after an
update, and last-three-event ordering is regression-tested.

The binder-only pilot used 32 repeated temporal lifetimes. Gradients were healthy and residual RMS
grew from zero to 0.198, but held-out rule decodability remained 53%--56% at raw write, compact row,
and recall through epoch 40. A joint binder-plus-reader pilot with balanced temporal/spatial/shape
rehearsal used 128 repeated lifetimes; through epochs 5, 10, and 20, no tap exceeded 55.5% held-out
rule decodability and held-out temporal behavior did not improve with demonstrations. These are
successful plumbing and credit-path checks, not fair negatives for reward discovery: the accepted
supervised curve required at least 4,096 distinct logical lifetimes and showed a long ignition
valley, whereas both pilots were deliberately far below that scale.

Changing writes can damage retained behavior even when old weights are frozen. The binder-only
checkpoint lost roughly 2.4--4.1 points on several spatial/shape retention measures. Balanced
rehearsal improved shape in the joint pilot, but its initial 512-lifetime spatial audit was near the
two-point gate and triggered a higher-precision paired audit before any mitigation is selected.
This establishes a standing rule: every writes-touching change gets rehearsal from update one and
checkpoint-level retention telemetry.

## Supervised-bootstrap specification

- Train the actual in-agent event binder, not the disposable diagnostic classifier. A temporary
  rule head reads the concatenated raw write key/value row, directly optimizing the same property
  measured by the raw-write probe. Discard the head; retain the binder.
- Keep the minimal memory reader trainable so representation and behavioral use can develop
  together. Rehearse temporal, spatial, and shape throughout.
- Use at least 4,096 distinct temporal lifetimes and budget beyond the measured supervised ignition
  window. Undersized preflights may validate plumbing only and cannot support learning claims.
- Log raw-write, compact-row, recall, behavioral few-shot, and per-primitive retention at every
  checkpoint. Add a task-agnostic residual-norm penalty only if the high-precision retention audit
  confirms that write perturbation is harmful.
- Graduation still requires behavioral reversal, memory corruption dependence, spatial/shape
  retention, and the original thin-line renderer. The result must be labeled
  **supervised-bootstrapped**; reward-only discovery remains unresolved.
