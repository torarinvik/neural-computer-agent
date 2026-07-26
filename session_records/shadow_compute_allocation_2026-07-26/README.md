# Shadow compute allocation — pre-registration

## Question

Can a passive critic learn when one external-memory read is worth its compute
cost, using only attempted compute actions and their scalar verified outcomes?

The critic cannot influence behavior. Every training lifetime randomly logs
one of two opaque actions with exact propensity `0.5`:

- action 0: answer from the controller without the external-memory read;
- action 1: perform one read, then answer.

The learner sees four generic read statistics, its attempted action and
propensity, and the resulting verifier bit. It never sees the correct compute
action, the unattempted outcome, stored/absent metadata, task IDs, or semantic
labels.

## Preflight

On 510 held-out contexts at capacity three, a read helped `50.0%`, harmed
`31.96%`, and was neutral `18.04%`. Always-read accuracy was `67.84%`; no-read
accuracy was `49.80%`. The task therefore has real context-dependent compute
value and cannot be solved optimally by one fixed action.

## Sub-minute budget

- 12 fresh batches of 60 lifetimes;
- 720 unique training lifetimes and verifier bits;
- 12 optimizer updates per critic;
- zero replay;
- 126 held-out logged attempts;
- a private 252-bit both-action audit used only for evaluation.

The normalized read cost is `0.01`, deliberately much smaller than one correct
answer. Actual wall latency is recorded separately.

## Controls and pass gate

The intact critic must:

1. beat the strong per-action empirical-rate predictor by at least `0.005`
   attempted-action Brier;
2. beat reward-shuffled, feature-shuffled, and zero-evidence critics by at
   least `0.002` Brier each;
3. choose the privately verified utility-optimal compute action at least 65%;
4. improve verified utility over both fixed compute choices by at least 0.05;
5. capture at least 20% of the available oracle utility gap;
6. lose at least 0.02 utility when held-out read evidence is shuffled;
7. remain calibrated (`ECE <= 0.10`) and improve at both final prefixes;
8. have live gradients, exact save/reload, positive measured latency, and
   retain binary mapping and four-rule behavior.

One passing seed is not enough. An unchanged unseen-seed replication is
required before any critic may influence computation.

## Seed 7411 result and bounded extension

The first run produced a causal but incomplete signal in 3.05 seconds:

- Brier gain over the action-rate baseline: `0.02028`;
- shadow utility: `0.78452` versus `0.69635` for always read;
- captured oracle gap: `29.5%`;
- evidence shuffling reduced utility by `0.15873`.

It failed exact compute-choice accuracy (`58.7% < 65%`) because it read on
`91.3%` of contexts, and failed calibration (`ECE 0.140`). The intact Brier
gain rose monotonically through every measured prefix and gradients remained
live, so the learning process had not plateaued.

Seed 7412 changes only the fresh-experience budget: 24 batches, 1,440 unique
lifetimes/bits, 24 updates, zero replay. It remains sub-minute. Every other
configuration and gate is unchanged. A failure is a rejection; a pass requires
an unchanged unseen-seed replication.

## Outcome and next objective

Seed 7412 rejected simple scaling. Brier remained strongly better than every
control, but compute-choice accuracy stayed at `57.1%`, read rate rose to
`92.9%`, ECE worsened to `0.177`, and utility gain over always-read fell below
the gate. The bottleneck is therefore not success prediction itself; it is
converting two imperfect absolute probabilities into a calibrated difference.

Seed 7421 keeps the original 720-bit budget and changes only the learning
objective. A scalar head estimates the verified advantage of one read using
the unbiased attempted-action pseudo-target:

`sign(action) * (observed_utility - running_baseline) / propensity`.

With propensity `0.5`, its conditional expectation is exactly
`utility(read) - utility(no read)`. The head sees no counterfactual target.
It remains passive.

Pass requires at least 65% private choice accuracy, +0.05 utility over the
strongest fixed action, at least 20% oracle-gap capture, at least 0.02 loss
under evidence shuffling, and at least 0.02 advantage over each
reward/feature/zero-evidence control. The last two measured prefixes must both
pass the fixed-utility margin. Persistence, latency, retention, and live
gradient gates remain unchanged.

## Replicated advantage result and matched-size gate

The 32-hidden-unit advantage estimator passed twice:

- seed 7421: `70.6%` choice accuracy, +`0.17667` utility over always-read,
  `56.2%` oracle-gap capture;
- seed 7422: `73.8%` choice accuracy, +`0.21667` utility,
  `59.8%` gap capture.

All reward-shuffled, feature-shuffled, and zero-evidence controls collapsed to
the fixed always-read behavior. Evidence shuffling made both learned policies
worse than the fixed baseline. Each run used only 720 verifier bits and 12
updates.

The inherited nonlinear read gate remains stronger at convergence, but its
recorded training used 81,920 contexts and 160 updates. The new head has 201
parameters including LayerNorm, versus the inherited gate's 49.

Seed 7423 therefore changes one efficiency axis: hidden width `32 → 8`, making
the advantage head 57 parameters including LayerNorm, near-matched to the
inherited 49-parameter gate. Experience, updates, controls, and pass gates
remain unchanged.

Seed 7423 did not pass: it gained `0.07214` utility over always-read but reached
only `57.1%` choice accuracy and `18.7%` oracle-gap capture. The causal signal
survived, so the 4× width reduction was too abrupt rather than wholly
incapable.

Seed 7424 tests the only intermediate width, `16` (105 parameters), with all
experience and gates unchanged. A pass must replicate at the same width on
seed 7425.

## Replicated efficient shadow allocator

The 105-parameter width-16 allocator passed and replicated:

- seed 7424: `69.0%` choice accuracy, +`0.19238` utility over always-read,
  `59.7%` oracle-gap capture;
- seed 7425: `70.6%` choice accuracy, +`0.18468` utility,
  `60.2%` gap capture.

Reward-shuffled, feature-shuffled, and zero-evidence controls selected the
fixed always-read policy and captured essentially zero oracle gap. Shuffling
episode evidence made both learned policies worse than always-read. All old
skill retention, persistence, latency, and gradient gates passed.

Both successful learning curves crossed the primary choice/utility/gap
thresholds at the first measured prefix—120 unique verifier bits—and stayed
above them at every later prefix through 720 bits. The full adversarial gate
was evaluated at 720 bits.

The inherited 49-parameter gate remains the production winner on this already
mastered task. A matched audit found:

- seed 7424: `95.2%` choice accuracy and `97.4%` oracle-gap capture;
- seed 7425: `82.5%` choice accuracy and `81.4%` gap capture.

Therefore no inherited weights are replaced. The new result is a training
breakthrough: the attempted-action advantage objective learns a useful
compute-allocation policy in 720 bits versus the inherited gate's historical
81,920-context training budget—a 113.8× smaller experience budget—though it
does not yet match the inherited gate's final performance.

## Exact resume frontier

The near-matched 57-parameter width-8 head was tested on the RTX PRO 6000
instance with 1,440 fresh bits and 24 updates. It completed in 0.45 seconds and
produced a causal improvement:

- choice accuracy `62.70%`;
- utility gain over always-read `0.11238`;
- oracle-gap capture `35.73%`;
- reward/feature/zero-evidence controls captured zero gap;
- evidence shuffling reduced the learned benefit;
- retention and persistence passed.

It nevertheless missed the pre-registered `65%` choice-accuracy gate, so the
capacity fork is closed without replication or further scaling. The replicated
105-parameter width-16 blueprint remains the promoted efficient design. The
stronger inherited 49-parameter weights remain in production on this mastered
task.

The next scientific rung must move to a genuinely later compute decision:
reuse the 105-parameter advantage blueprint on a novel setting with more than
one optional operation (for example answer now versus another read/thought),
and compare its unique bits-to-threshold against a freshly initialized matched
learner. Do not spend more experience tuning width 8 on the mastered read task.

No experiment was left running.
