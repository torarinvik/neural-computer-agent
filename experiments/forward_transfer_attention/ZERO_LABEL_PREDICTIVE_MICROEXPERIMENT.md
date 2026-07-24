# Zero-label predictive-state microexperiment

## Question

Can unlabeled predictive experience make the existing recurrent visual state
faster to use on a novel temporal relation?

This is the smallest test of the new mainline: predictive latent learning
before world-model planning, new recurrent backbones, object slots, or scale.

## Sensory firewall

The learner receives only rendered frame sequences and its own emitted actions.
It does not receive palette IDs, object IDs, rule labels, selected-object
labels, event indices, generator metadata, or privileged game state.

The generator may retain those facts privately for audit and balanced sampling.
They never enter the optimizer. No semantic probe weights enter the agent.

## Three sub-minute arms

All arms use the same encoder and recurrent core initialization family, number
of unique sensory lifetimes, optimizer updates, and downstream reward budget.

1. **Predictive experience:** predict an EMA target encoder's latent for a
   masked or future sensory event from the preceding visible stream.
2. **Fresh control:** no predictive experience; begin downstream reward
   learning from a matched random initialization.
3. **Shuffled-future control:** identical predictive optimization, but future
   targets are permuted across logical lifetimes.

An optional fourth arm predicts pixels rather than latents. It is admitted only
if the first three run comfortably under one minute; its purpose is to test
whether latent prediction is more efficient than spending capacity on surface
detail.

## Anti-collapse telemetry

Prediction loss alone is not evidence. Log:

- target and predicted latent variance per dimension;
- effective latent rank;
- mean pairwise cosine similarity across distinct lifetimes;
- predictor gradient norm;
- held-out future-latent loss;
- performance when temporal order is shuffled.

A constant representation is an immediate failure even if its loss is low.

## Downstream learning test

Discard the predictor head. Keep only the sensory encoder/recurrent state from
the predictive arm. Give every arm the same small budget of interaction with
the temporal task and train only from externally verified behavioral reward.
No rule, identity, or concept labels are supplied.

Measure:

- verified reward versus unique logical lifetimes;
- area under the reward learning curve;
- unique lifetimes to 60%, 70%, and 80% verified success;
- wall time and GPU-seconds to threshold;
- retention on spatial and shape;
- dependence on intact sensory and memory streams.

The latent rule and identity probes may be run afterward as discarded
instruments to localize a result, but they do not decide success.

## Advancement gate

The predictive arm advances from under one minute to roughly three minutes
only if all of the following hold:

1. held-out prediction beats the shuffled-future arm;
2. latent variance and rank exclude collapse;
3. downstream verified reward AULC beats both fresh and shuffled-future arms;
4. the advantage appears on unseen logical lifetimes;
5. no old primitive loses more than its pre-registered retention allowance.

It advances to ten minutes only after replication and a true counterfactual
replay: reverse the visible events while holding the selected identity fixed,
and require the agent's behavior to change in the verifier-prescribed way.

## Interpretation

- Better prediction without better reward learning means the objective learned
  visually predictable nuisance structure, not reusable cognition.
- Better final reward without earlier thresholds is representation transfer,
  not compounding sample efficiency.
- Earlier thresholds on one seed only are provisional.
- Earlier thresholds replicated across seeds and later primitives are evidence
  of learning-to-learn.

## Deferred branches

- Dreamer-style imagination begins only when a closed-loop control primitive
  such as Pong exposes a planning bottleneck.
- Mamba/RWKV replaces the GRU only after a matched latency/capacity benchmark.
- Object slots enter only after global predictive latents fail a
  counterfactually audited composition or distractor test.
- Audio and text join through cross-modal future-latent prediction after the
  visual experiment passes; no transcription or semantic labels are used.

## Observed sub-minute results

The first immediate-next-latent run completed in 17.22 seconds. It increased
per-dimension variance but collapsed almost entirely onto one shared direction:
effective rank was 1.14 of 64, correctly paired and shuffled future losses
differed by about one millionth, and all reward-learning arms stayed at 50%.
It failed the gate.

A standardized dimension-wise objective plus explicit variance/correlation
penalties fixed part of the mechanical problem. Across seeds 211 and 257:

- correctly paired future alignment margins were 0.93 and 0.90;
- recurrent effective ranks rose to 5.12 and 5.31;
- reward-only held-out behavior remained 51.0% and 52.3%;
- discarded rule probes remained near chance for the predictive arm;
- a 76.0% shuffled-future probe result at seed 211 disappeared at seed 257 and
  was rejected as a one-seed artifact.

Thus immediate-next prediction learned real, nontrivial predictability but not
the relation needed for behavior. This is a direct in-project example of why
world-model loss is not a capability metric.

The next bounded fork predicted latent **change** rather than absolute next
state. This generic temporal-difference target suppresses static
background/cue information without revealing event identities or rules.

The seed-211 delta run completed in 23.14 seconds. Correct temporal pairing
produced a real held-out alignment margin of 0.522, versus -0.045 for the
shuffled-future control. Its recurrent effective rank reached 5.64 of 64,
better than the fresh arm's 1.24 but still below the pre-registered 6.4
non-collapse threshold.

The discarded diagnostic probe found substantially more useful temporal
information than the immediate-next objective:

- predictive delta MLP: 80.73% held-out;
- true event-reversal relabeled accuracy: 79.69%;
- prediction flip rate under reversal: 61.46%;
- shuffled-label calibration: 46.61%.

This is a representation-localization result, not agent capability. The
reward-only learner remained at 50.26%, reached none of the 60/70/80%
thresholds, and had an AULC advantage of -0.00391 relative to the best control.
The shuffled-future arm ended at 52.34%. Therefore the delta arm failed the
behavioral and non-collapse gates and must not advance to a longer run.

The bounded conclusion is that predicting sensory change exposes a much more
relation-bearing latent, but the current sparse reward readout cannot learn to
use it within 510 unique lifetimes. The next tiny fork should attack that
localized credit/readout bottleneck using an action-conditioned success model
trained only from `(latent state, attempted action, scalar verifier reward)`.
It must remain auxiliary to verifier-grounded behavior, compare against the
same REINFORCE baseline and shuffled controls, and earn scale through earlier
verified reward rather than predictor accuracy.
