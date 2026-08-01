# Research filter: emergent learning without semantic labels

This philosophy is implemented through the module boundaries and audits in the
canonical
[`../../docs/AMODAL_N_TO_M_ARCHITECTURE.md`](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md)
specification. If terminology differs, the canonical document controls the
target architecture and this file controls the zero-semantic-label filter.

## Non-negotiable principle

The system must discover its own useful representations from experience. We do
not tell latent variables what they mean, provide human-authored concept labels,
encode task solutions in a symbolic language, or expose privileged game state.
The verifier judges externally observable outcomes; it does not dictate the
agent's internal vocabulary or algorithm.

Generated labels are permitted only for discarded scientific probes that
measure information location, leakage, or causal dependence. A probe result is
evidence about the architecture, never training credit for the deployed agent.

## Adopt now

- A small recurrent/iterative core whose computation and latent concepts are
  learned end to end.
- JEPA-style masked/future latent prediction from raw sensory sequences, using
  an EMA/stop-gradient target encoder and no semantic targets.
- Action-conditioned prediction of observed outcomes from the agent's real
  `(latent, attempted action, scalar reward)` stream. This is immediately
  applicable to the current contextual-bandit-style temporal task.
- Action-conditioned prediction of future latent observations only in tasks
  where actions causally affect subsequent observations. In passive support
  streams, adding an action input would be formally present but scientifically
  empty.
- Sparse external memory with learned read, write, retention, eviction, and
  consolidation policies.
- Shared amodal latent states joined to modality- and device-specific adapters.
- Unique experiences to threshold, area under the learning curve, wall time,
  and verified transfer as primary metrics.
- Experienced-agent versus fresh-agent controls for every transfer claim.
- Counterfactual replays, reversal tests, shuffled/garbage memory, shuffled
  pairings, and nuisance randomization to eliminate shortcuts.
- Diverse procedurally generated experience, with logical lifetimes separated
  across train and audit splits.
- Self-supervised pressures derived from sensory prediction, temporal
  consistency, cross-modal prediction, observed action effects, and memory
  utility.
- A machine-readable manifest for every experiment that separates
  learner-visible tensors, verifier-private facts, offline-probe facts, and
  forbidden channels.
- External verifiers that score correctness, latency, retention, and later
  reuse without revealing the solution representation.
- Missing-evidence calibration audits: removing decisive evidence should
  increase uncertainty or trigger more observation/thought, never make the
  agent more confidently wrong.

## Adopt only after the current primitive works

- Dreamer-style latent imagination and learned value prediction for
  closed-loop control tasks such as Pong. This is deferred because the current
  temporal diagnostic does not require long-horizon planning.
- Replacing the current GRU with Mamba, RWKV, or another state-space core.
  Linear-time recurrence is attractive, but an architecture swap is justified
  only if a matched tiny benchmark shows a recurrent-state capacity or latency
  bottleneck.
- Object-centric slots and sparse interaction modules. Their semantics may
  emerge without labels, but they add an inductive bias and should be admitted
  only if predictive training still fails a relation or distractor audit.
- Learned callable submodules or a reusable latent skill library, provided
  module boundaries and invocation policies emerge from performance pressure.
- Learning-progress curriculum selection, with patience for long pre-ignition
  valleys and periodic revisiting of apparently stalled tasks.
- A success/value predictor used for compute allocation after its passive
  calibration and reward-efficiency benefits have been established, while
  ground-truth verifier outcomes remain sovereign.
- Working-memory and long-term-memory budget adaptation driven by measured
  capability retention and transfer.
- Cross-modal acquisition and actuator substitution tests.

## Reject as the native reasoning mechanism

- Hand-labeled concepts or fixed semantic bottlenecks.
- A human-designed ontology, typed concept inventory, or named latent slots.
- English or another token language as mandatory internal thought.
- A fixed DSL, theorem rules, planners, or programs that encode how to solve
  the benchmark.
- Direct game-state hooks or metadata visible to the learner.
- Reward from the agent's own confidence or novelty estimate without external
  verification.
- Scaling to a giant transformer or mixture-of-experts model before a small
  audited system demonstrates the missing learning behavior.
- A large self-supervised pretraining run before the same objective wins a
  sub-minute experienced-versus-fresh causal test.
- Treating probe accuracy, final accuracy alone, or one favorable seed as proof
  of sample-efficient learning.

## Immediate experimental rule

Every proposed change begins with a sub-minute falsification test. It advances
to roughly three minutes only when it beats chance and its fresh/shuffled
controls on unseen logical lifetimes. It advances to ten minutes only after a
causal counterfactual audit and a measurable improvement in learning efficiency
or verified reuse. Longer runs require replicated evidence and a stated
compute budget.

The next deployed-learning target is therefore not another labeled rule head.
The delta-prediction experiment made the temporal relation strongly decodable,
and the subsequent attempted-action success model converted it into verified
support-order behavior across three seeds. It trained only on observed
`(latent, attempted action, scalar reward)` tuples and never synthesized a
label for an unattempted action.

The first transfer rung now passes. A frozen learned intention calibrated an
unfamiliar four-command actuator protocol in 32 reward bits on all three seeds,
while an identical fresh system required 510/256/256 bits to reach 75%.
Opposite-rule stale intentions, protocol swapping, and true sensory reversal
confirmed causal use. This is same-concept interface transfer, not a new
cognitive primitive.

Cross-primitive transfer now has one successful three-seed transition. A
correctly paired temporal predictive core reached 75% on a new simultaneous
spatial primitive at 256 reward bits on all three seeds. Equal-compute
shuffled-future and fully fresh cores never reached 75%. The reusable component
was the visual/recurrent predictive state; retaining the old intention head
added no stable mean gain.

The next deployed-learning target is a third primitive chosen to reuse a
computational operation without repeating the same surface relation. Its
experienced-agent curve must be compared against paired-core ablation,
shuffled-future, and fully fresh controls. A second isolated endpoint win is
insufficient; only a sequence of earlier threshold crossings can establish a
compounding trend.

Multi-horizon action-conditioned latent dynamics become the main predictive
target when a closed-loop primitive is introduced and actions genuinely alter
future sensory observations.

## Compounding evidence standard

Endpoint improvement is not compounding. Record unique lifetimes to fixed
thresholds for every curriculum transition and compare with a fresh agent.
After enough distinct primitives exist, regress log lifetimes-to-threshold
against curriculum index. A favorable experienced-agent slope that does not
appear in fresh controls is stronger evidence of compounding than any single
transfer ratio.

Exploratory results require at least three seeds. A load-bearing claim should
use at least five seeds and a confidence interval over the median transfer
ratio. A one-seed threshold improvement is a hypothesis, never a milestone.
