# Action-conditioned success microexperiment

## Research question

Can a tiny success model use each observed scalar reward more efficiently than
REINFORCE when the recurrent delta-predictive state already contains the
temporal relation?

This experiment targets the localized readout and credit-assignment gap. It
does not change the sensory encoder, recurrent core, memory system, renderer,
or verifier.

## Causal scope

The current temporal task is a contextual bandit at answer time. The selected
action affects reward but does not generate a rich subsequent visual
transition. Therefore the appropriate action-conditioned objective is outcome
prediction:

```text
q(h, a) -> probability that attempted action a receives reward 1
```

Action-conditioned future-observation prediction is deferred to a genuinely
closed-loop primitive whose actions alter later sensory input.

## Information firewall

Learner-visible:

- rendered RGB stream;
- recurrent latent `h`;
- the action actually sampled by the agent;
- the logging probability of that sampled action;
- the resulting scalar verifier reward.

Verifier-private:

- correct action;
- temporal rule;
- object identities, palette, event indices, and logical-lifetime metadata.

Offline-probe-only:

- rule and identity labels used after training for discarded localization
  probes;
- counterfactual relabeling used to score valid pixel-space reversals.

Forbidden:

- a target for an action that was not attempted;
- inferring the other action's label from a failed binary action;
- task IDs, game state, semantic labels, or differentiated verifier logic.

## Matched arms

1. **Success replay:** freeze the delta-predictive recurrent representation and
   train `q(h,a)` by binary cross-entropy only on attempted-action outcomes.
   Choose actions with exploration-constrained softmax or epsilon-greedy
   sampling from the two predicted success values.
2. **REINFORCE:** the current reward-only policy/value learner, with the same
   initial representation, unique lifetimes, action count, optimizer-update
   count, and model-parameter budget.
3. **Action-shuffled success control:** train the same success model after
   permuting attempted actions across replay tuples while preserving states and
   rewards.
4. **Fresh-representation success control:** train the success model on an
   otherwise matched newly initialized recurrent representation.

The delta-predictive arm and its report are fixed before this comparison. No
arm may receive extra verifier interactions.

## Loss and policy

For each logged tuple `(h_i, a_i, r_i, p_i)`:

```text
q_i = sigmoid(success_head(h_i, one_hot(a_i)))
loss = BCE(q_i, r_i)
```

Only the attempted action contributes a target. The initial experiment uses
uniform or explicitly logged exploration, so every reward bit has a known
propensity. Importance weighting is unnecessary when replaying the same
uniform logging distribution; if the behavior policy becomes adaptive, log
propensities and report clipped importance-weighted and doubly robust audits
separately.

The predictor does not create reward. The external verifier remains the only
source of truth.

## Sub-minute protocol

- reuse seed 211's delta-predictive representation;
- use the same 510 unique-lifetime interaction budget and held-out split;
- evaluate at 0, 30, 120, 240, 360, and 510 unique lifetimes;
- log reward AULC, accuracy, Brier score, calibration error, action coverage,
  entropy, gradient norm, and wall time;
- keep exploration high enough that both actions receive substantial coverage;
- run the valid sensory reversal only if behavior exceeds the advancement
  threshold.

## Advancement gate

Advance to a second seed only if success replay:

1. beats REINFORCE and both controls by at least 0.02 reward AULC;
2. reaches at least 60% verified held-out accuracy;
3. uses no more unique lifetimes or verifier outcomes than the controls;
4. retains action coverage and does not collapse to one action;
5. is better calibrated than a constant-frequency predictor;
6. exhibits no leakage in the four-way information manifest.

Advance to roughly three minutes only after a second seed repeats the
behavioral advantage and a valid pixel-space reversal changes behavior in the
verifier-prescribed direction.

## Interpretation

- Better calibration without better reward learning means the success model
  predicts outcomes but does not improve action selection.
- Better reward learning with action collapse is a likely shortcut or class
  imbalance artifact.
- Better reward learning that fails reversal is reward hacking or nuisance
  exploitation.
- Better AULC and earlier thresholds under intact reversal, shuffled-action,
  and fresh-state controls is evidence that observed reward bits are being
  reused more efficiently.

## Later extensions

After this experiment passes:

- allow the calibrated success model to request another thought step or memory
  retrieval before it influences the final answer;
- recalibrate it on the induced behavior distribution after every new form of
  influence;
- compare memory writes by retrieval advantage rather than surprise;
- introduce action-conditioned multi-horizon latent deltas on closed-loop
  choice-reaction or Pong tasks where actions genuinely affect future frames.

## Seed-211 online result

The first matched online run completed in 14.13 seconds. Every arm received 510
unique lifetimes, 510 scalar reward bits, 17 optimizer updates, 510 processed
training examples, the same nonlinear action-head initialization, and the same
20% exploration mixture.

Results:

- success replay on delta states: 53.13%, AULC 0.01476;
- matched REINFORCE on delta states: 48.96%, AULC 0.00781;
- action-shuffled success control: 50.26%, AULC 0.00304;
- success replay on fresh states: 50.00%, AULC 0.00000.

Success replay maintained healthy action coverage (39.22% for the less common
action), but its AULC advantage over the best control was only 0.00694, it did
not reach 60%, and its valid-reversal flip rate was 8.07%. It therefore failed
the advancement gate. This is a weak directional signal, not evidence of
capability or sample-efficiency improvement.

Only 17 optimizer updates were performed. The earlier discarded MLP diagnostic
needed roughly 200 dense updates to expose the relation in the same class of
state. The next bounded fork must therefore hold the 510 unique reward bits
fixed while sweeping replay computation. It must report reward bits, optimizer
steps, processed examples, GPU-seconds, and wall time separately, so additional
optimization cannot be misreported as additional sample efficiency.

## Fixed-buffer replay result and causal correction

A fixed balanced uniform log allowed a clean two-axis study. Each learning-
curve point used a fixed prefix of 30/120/240/360/510 unique reward bits, 200
optimizer updates, and 6,000 processed examples. Across seeds 211, 257, and
313, the full five-frame input initially appeared to solve the task:

- paired-delta success final accuracy: 75.26%, 82.29%, 82.03%;
- mean paired-delta AULC: 0.2155;
- mean shuffled-future-representation accuracy: 51.65%;
- mean shuffled-future-representation AULC: 0.0172;
- mean IPS AULC at equal optimization: 0.1444.

Action-shuffled, reward-shuffled, and fresh-state controls remained near chance.
This established that correctly paired predictive experience—not anti-collapse
training alone—created a state that attempted-action reward modeling could use.

However, a stronger causal audit invalidated the original interpretation. The
generator made query orientation deterministically opposite support
orientation. Removing support-order frames left 79.43% accuracy, reversing
support alone changed only 4.69% of predictions, and reversing query alone
changed 61.20% even though the correct private rule was unchanged. Reversing
both had passed only because the model learned the unintended but real
`feedback identity + query order` relation.

The microtask was repaired by removing query frames entirely. The only
remaining route was:

```text
demonstrated support order + visible selected-object feedback -> private rule
```

With the same zero-label learner, fixed reward-bit prefixes, optimization
budget, and controls, support-only input produced:

- seed 211: 78.13% final, AULC 0.2188;
- seed 257: 82.03% final, AULC 0.2453;
- seed 313: 80.73% final, AULC 0.2068;
- three-seed mean final: 80.30%;
- three-seed mean AULC: 0.2236;
- shuffled-future representation mean final: 55.73%;
- shuffled-future representation mean AULC: 0.0434;
- equally optimized IPS mean AULC: 0.1780;
- true support-order reversal mean relabeled accuracy: 78.82%;
- true support-order reversal mean prediction flips: 59.11%.

Thus the repaired capability survives three seeds, matched optimization,
fresh/shuffled representations, shuffled actions, shuffled rewards, and a
counterfactual in which only the demonstrated order changes. This is the first
clean deployed-learning result in this zero-semantic line.

The claim remains bounded. It demonstrates one support-order primitive, not
compounding learning, cross-primitive reuse, calibrated uncertainty, memory
integration, or online adaptation under an induced policy distribution.
