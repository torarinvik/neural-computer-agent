# Closed-loop micro-intercept experiment

## Purpose

Test whether action-conditioned predictive learning creates a more reusable
control state than passive prediction when the agent's action causally changes
later pixels.

The initial implementation is a **single-transition admission preflight**.  Its
action changes the final pixels, but it still has only one decision and one
terminal outcome.  It therefore does not, by construction, escape a contextual
bandit interpretation.  A genuinely closed-loop claim requires the promoted
version with several dependent decisions.

## Environment

A deterministic one-dimensional intercept scene is rendered entirely as RGB:

1. frame 0 shows a moving target and the agent-controlled cursor;
2. frame 1 shows the target after one velocity step;
3. the agent attempts one of three effects: move left, stay, or move right;
4. frame 2 shows both the moved cursor and the target's next position;
5. the verifier emits scalar success if their final distance is within the
   interception radius.

Target position, velocity, cursor position, colors, backgrounds, and render
seed vary independently. Train/test splits hold out render seeds, palettes,
velocity-position combinations, and scene layouts.

The deployed learner sees pixels, its own action, exact logging propensity,
and scalar terminal success. It never receives coordinates, velocities,
distances, simulator state, correct action, or a target for an unattempted
action.

## Representation arms

All start from identical weights and receive identical rendered transitions:

1. **Passive predictor:** predicts the next latent delta without action input.
2. **Action-conditioned predictor:** predicts the next latent delta using the
   agent's actual attempted action.
3. **Shuffled-action predictor:** identical architecture and compute, but
   action inputs are permuted across transitions.
4. **Fixed-no-action stream:** the environment advances under stay actions;
   establishes how much is learnable without intervention diversity.
5. **Fully fresh core.**

Action embeddings are learned and carry no human semantic labels. The learner
knows only which opaque command it emitted and what pixels followed.

## Behavioral conversion

Freeze each predictive core. Train an identical attempted-action-only success
model on balanced logged tuples:

`(pre-action recurrent state, attempted opaque command, propensity, terminal scalar reward)`

BCE applies only to the attempted command. At evaluation, rank the three
predicted command-success values and directly execute the selected command in
held-out rerendered scenes.

## Accounting

Report separately:

- unique transition lifetimes;
- unique terminal reward bits;
- predictive optimizer steps and transition examples;
- success-head updates and replayed examples;
- GPU/wall time;
- p50/p95 observation-to-action latency.

## Causal audits

- reverse target velocity while holding initial target/cursor identities and
  nuisance rendering fixed;
- horizontally mirror the entire valid scene and require mirrored action
  effects;
- shuffle action inputs during predictive learning;
- shuffle logged attempted actions or terminal rewards during success learning;
- remove frame 0 or frame 1 so velocity becomes ambiguous;
- freeze the cursor so commands no longer change later pixels;
- corrupt or zero the learned action embedding;
- test unseen command-to-effect permutations through a fresh actuator adapter.

## Promotion gate

The action-conditioned arm advances only if, over at least three seeds, it:

- beats passive and shuffled-action arms by at least 0.03 reward AULC;
- crosses a fixed held-out success threshold with fewer unique reward bits;
- preserves balanced command coverage under uniform logging;
- passes velocity reversal and mirror audits;
- loses its advantage when cursor motion is disabled or action identity is
  corrupted;
- stays within the pre-registered latency budget.

If passive and action-conditioned arms tie, do not add RSSM, Dreamer, PSR, or
latent planning machinery. If action-conditioning wins cleanly, the next
three-minute fork may compare one-step prediction against multi-horizon latent
overshooting.

## First sub-minute result

Seed 211 used 252 predictive lifetimes, 40 predictive updates, 270 unique
reward bits, 68 readout updates per prefix, and 192 held-out scenes.  The run
completed in 20.5 seconds on an RTX 5090.

| Arm | AULC above 1/3 | Final accuracy |
|---|---:|---:|
| action-conditioned | 0.0169 | 34.38% |
| passive | 0.0391 | 34.38% |
| shuffled action | 0.0117 | 32.29% |
| fixed no-action | 0.0234 | 32.81% |
| fully fresh | approximately 0 | 33.33% |
| action-shuffled replay | 0.0039 | 30.73% |
| reward-shuffled replay | 0.0078 | 34.38% |

No arm reached 60%.  The action-conditioned arm trailed passive by 0.0221
AULC.  Velocity reversal remained at 32.81% with only 5.47% prediction flips
on moving scenes.  Missing either motion frame reduced accuracy to chance, but
did not make the policy less confident.  The pre-registered promotion gate
failed, so this configuration must not receive a longer run.

This is a bounded negative at the tested optimization budget, not proof that
action-conditioned learning cannot work.  More importantly, the environment
allows the desired command to be inferred from exogenous target velocity in a
single decision.  The passive arm can therefore learn the central perceptual
fact without modeling the action-induced transition.  Increasing this exact
run's budget has lower expected value than correcting that scientific
confound.

## Next closed-loop admission task

The first attempted replacement used a deterministic six-decision trajectory:

- the target is briefly visible and then partially occluded;
- the cursor has momentum or delayed effects, so the same command has
  state-dependent future consequences;
- an opaque seed-specific actuator protocol must be identified through the
  agent's own interventions;
- early actions change both later cursor state and the useful future
  observations;
- no single frame reveals the optimal action;
- terminal success requires several correct interventions, not one read-off
  choice.

The first tier remains sub-minute.  Action-conditioned prediction must first
beat passive, shuffled-action, fixed/no-effect, and fresh controls in both
held-out predictive utility and reward-bit AULC.  Only then is a three-minute
comparison with multi-horizon deltas justified.

## Six-decision results and rejection

The first six-decision tier completed in 42.1 seconds.  With the original
eight-pixel target displacement, action-conditioned pretraining produced a
small behavioral separation:

- action-conditioned AULC above random: 0.1438;
- passive: 0.1125;
- shuffled-action: 0.0500;
- fully fresh: 0.0521;
- fixed-no-action: 0.1375.

The nominal action-versus-passive advantage was 0.0313, but the fixed-action
arm nearly matched it.  No arm reached 60%.  Final action-conditioned terminal
success was 39.58%, versus a 16.67% random-policy baseline.  The policy failed
the load-bearing audits: reversing target motion changed only 7.47% of its
actions, missing motion evidence made it *more* confident, and disabling
actuation did not cause the required loss.  This was position chasing, not
motion-grounded control.

A targeted renderer correction increased visible target displacement from
eight to fourteen pixels while preserving cursor physics.  This eliminated
the apparent win rather than strengthening it:

- random terminal success: 26.04%;
- action-conditioned final success: 17.71%;
- passive: 13.54%;
- fully fresh: 18.75%;
- action-conditioned AULC above random: 0.

The dense step reward was also misaligned: policies achieved approximately
42% immediate distance-improvement reward while terminal success remained
below random.  Momentum and target bouncing made greedy local improvement a
poor proxy for interception.

Replacing six step-improvement outcomes with the single observed terminal
success bit did not help.  Action-conditioned final success remained 17.71%;
passive reached 18.75% and shuffled-action reached 20.83%.  This rules out a
simple choice between the two tested reward signals at this budget.

### Discarded ceiling probes

A supervised diagnostic used verifier-private optimal actions only to measure
the interface ceiling.  Its weights were discarded and never entered the
agent.

- action-conditioned oracle-action accuracy on held-out random trajectories:
  65.97%;
- passive: 66.15%;
- action-conditioned direct terminal success: 15.63%;
- supervised DAgger-style labels on 90 additional policy-induced
  trajectories: only +2.08 terminal points;
- the offline oracle/action-to-direct-control gap: 50.35 points.

Thus the frozen state contains some action-relevant information, but the
six-decision controller compounds errors under its induced state distribution.
One supervised data-aggregation round does not rescue it, and
action-conditioned features do not outperform passive features.  The
representation/control interface has not earned architectural promotion.

### Decision

Do not extend this configuration to three minutes.  Do not add RSSM,
Dreamer, PSR, a larger network, or further reward shaping.  The experiment
jumped directly from one decision to six decisions and combined system
identification, occlusion, momentum, long-horizon credit, and distribution
shift.

The next admission rung should be a two-decision identify-then-act task:

1. the first opaque command produces a visible actuator consequence;
2. the second decision must use that observed consequence to reach a target;
3. the actuator mapping varies across lifetimes, so current pixels alone do
   not identify the correct second action;
4. terminal reward is exactly aligned with success;
5. after two decisions pass, increase gradually to three, four, and only then
   six decisions.

This isolates the elementary cognitive loop we actually need:
intervene → observe consequence → update latent actuator concept → act.

## Two-decision curriculum results

The curriculum exposed three distinct lessons.

1. A fixed probe with varying actuator protocol was easy: the learner reached
   98.83% at 16 verifier bits and stayed at 100% thereafter.
2. Jumping directly to balanced probe actions failed.  A 12.5% unfamiliar
   probe mixture also failed for a more revealing reason: balanced evaluation
   was exactly 50%, split into 100% accuracy on the familiar probe and 0% on
   the unfamiliar probe.  Rare exposure rewarded the shortcut.
3. Fixing the target direction while balancing probe action and protocol was
   the useful bridge.  Two readout seeds reached 98.44% and 95.31% with 64
   reward bits.  Valid protocol rerenders preserved accuracy and flipped
   predictions; removing the consequence returned behavior to chance.

A discarded diagnostic MLP decoded probe action, protocol, and correct action
from the frozen decision representation at 100% held-out, while shuffled
labels stayed at 47.75%.  The information was present; the behavioral learner
needed more replay computation.  Increasing readout updates from 68 to 256,
without adding reward bits, opened learning at 32–64 bits.

The full varying-target task then reached 100% at 64 reward bits from a fresh
predictive core and passed protocol reversal, target reversal, missing
consequence, action-shuffle, and reward-shuffle audits.  An incremental
readout reached 93.36% with the same 64 unique bits and 256 cumulative updates.

Two claims are deliberately withheld:

- The fixed-target checkpoint did not compound into the full task; retaining
  it reduced AULC from 0.2207 to 0.1289 and final accuracy from 100% to 81.64%.
- Correct action-conditioned predictive pretraining did not beat every generic
  predictive control.  Fixed-protocol and shuffled-action pretraining could
  also support the final behavior.  Predictive experience clearly beats a
  fully fresh core, but the useful invariant has not yet been localized.

The next optimization target is therefore not a harder environment.  It is to
identify which task-agnostic predictive property gives the 64-bit learner its
advantage, then reduce stable reward bits and replay updates without losing
the causal audits.

A direct compute-for-experience substitution failed: capping the full task at
32 unique reward bits and doubling each independent fit to 512 updates produced
only 52.73% held-out accuracy and failed both counterfactual audits.  The next
32-bit attempt must change the learning signal or representation; replaying
the same outcomes longer has been rejected.

## Research recommendations retained

- Treat established one-step primitives as contextual-bandit readout problems;
  the attempted-action-only success head remains the mainline there.
- When an online policy is introduced, begin with balanced logging and then
  use conservative ranking with an explicit exploration floor.
- Re-measure Brier/calibration on fresh data from the policy-induced
  distribution; passive calibration does not survive deployment by assumption.
- Keep action/reward shuffles, fresh representations, valid pixel rerenders,
  coverage accounting, and missing-evidence confidence as hard gates.
- For memory, construct an anti-noise task in which surprising distractors are
  irrelevant and score writes by later causal retrieval advantage.

## Research recommendations not promoted

- Re-running the already established frozen-latent success-head result as if it
  were still an open question.
- Adding RSSM, Dreamer, PSR, successor-feature, or large actor-critic machinery
  before the simple multi-decision admission task passes.
- Weakening the compounding standard to three primitives and three seeds.  We
  retain at least six distinct primitives for the trend and five seeds for a
  load-bearing claim.
- Treating replay-driven interaction efficiency as compute efficiency.
