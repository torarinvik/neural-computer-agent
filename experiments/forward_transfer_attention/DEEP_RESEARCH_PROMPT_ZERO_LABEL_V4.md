# Deep-research prompt: auditing and generalizing a zero-label action-learning breakthrough

You are advising an empirical project building a small real-time,
sample-efficient neural computer. We have obtained a promising three-seed
behavioral result, then discovered that our counterfactual audit did not isolate
the intended causal path. We need rigorous primary-source-grounded advice that
improves the next tiny experiment—not a broad AGI survey.

## Non-negotiable constraints

The deployed learner receives only externally observable visual, auditory, or
visible-text streams, its own actions and internal memory, and scalar outcomes
from external deterministic verifiers.

It receives no human-authored semantic labels, concept names, task IDs,
privileged simulator state, solution traces, generated correct-action labels,
fixed DSL, English scratchpad, hand-written reasoning rules, or hard-coded
planner.

Verifier-private facts may calculate scalar reward and discarded offline
audits. Probe weights never enter the agent, and probe accuracy never counts as
capability.

Accuracy is primary. Unique experiences and reward bits are the main efficiency
costs; optimizer steps, examples processed, GPU-seconds, latency, memory, and
wall time are logged separately. Experiments begin under one minute, advance
to roughly three minutes only on replicated audited evidence, and advance to
ten minutes only after another gate.

The learner must never manufacture a target for an unattempted action, even
when a failed action in a deterministic two-action task logically reveals the
alternative.

## Architecture and preceding result

The current sub-1M-parameter agent uses a small visual encoder and GRU. It was
pretrained without semantic targets to predict EMA latent changes
`z(t+1)-z(t)` from rendered RGB sequences, with explicit variance and
correlation regularization.

An earlier 23-second run found:

- paired-versus-shuffled latent-delta alignment margin: 0.522;
- recurrent effective rank: 5.64/64;
- discarded temporal-relation MLP probe: 80.73%;
- reward-only REINFORCE behavior: 50.26%.

This localized a readout/credit problem: relation-bearing information was
present but sparse policy-gradient training did not use it.

## New attempted-action-only success result

We then logged a balanced uniform contextual-bandit buffer. Every tuple
contained only:

```text
frozen recurrent state
attempted action
logging propensity 0.5
observed scalar verifier reward
```

No unattempted-action target or semantic rule target was created. A nonlinear
success head was trained by BCE only on the attempted action. Each learning-
curve point used:

- a fixed prefix of 30, 120, 240, 360, or 510 unique reward bits;
- 200 optimizer updates;
- 6,000 processed examples;
- identical initial heads and training compute across arms.

Three seeds (211, 257, 313) produced:

- paired-delta success final accuracy: 75.26%, 82.29%, 82.03%;
- paired-delta AULC above chance: 0.1807, 0.2401, 0.2255;
- paired-delta three-seed means: 79.86% final, 0.2155 AULC;
- shuffled-future-delta representation: 48.96%, 53.39%, 52.60%;
- shuffled-representation mean AULC: 0.0172;
- fresh representation: approximately 50% on all seeds;
- action-shuffled and reward-shuffled controls: approximately chance or worse;
- equally optimized IPS policy learner mean AULC: 0.1444.

The success head beat every matched control on all three seeds. The paired
representation materially beat the shuffled-future representation, causally
attributing useful state formation to correct predictive pairing rather than
anti-collapse optimization alone.

A compute sweep on the same 510 reward bits showed a sharp optimization
dependence: on seed 211, 17/68/200 updates produced 49.2%/54.2%/76.3%. We
therefore distinguish:

- interaction efficiency at a fixed stated compute budget;
- compute efficiency at fixed interactions;
- online 17-update learning from offline 200-update replay.

We do not claim that replay is computationally free.

## Newly discovered audit weakness

The visual episode supplied to the head contains:

1. support object A;
2. support object B;
3. feedback showing which support object was selected;
4. query object 1;
5. query object 2.

The private rule is “select first” or “select last.” The generator currently
makes the query orientation deterministically opposite the support orientation.

Our original reversal audit reversed **both** support and query orders while
holding feedback identity fixed, then flipped the verifier label. It passed on
three seeds:

- mean relabeled reversal accuracy: 79.25%;
- mean prediction flip rate: 60.85%.

A pixel-space missing-evidence audit on seed 313 then found:

- normal: 81.77%;
- no feedback frame: 48.96%;
- feedback only: 50.00%;
- order frames only: 50.00%;
- no support-order frames, but feedback + query remain: 78.91%;
- support only, with query removed: 54.95%;
- query only, with support and feedback removed: 47.66%.

Removing feedback also raised action entropy from 0.078 to 0.197. Thus the
model genuinely depends on feedback, but it does not appear to depend on the
demonstrated support order. Because query order is deterministically opposite
support order, `feedback identity + query order` provides an alternative route
to the private rule. Reversing both support and query preserves that shortcut's
ability to flip, so the earlier reversal was not sufficient to isolate intended
support binding.

This does not make the result meaningless: paired predictive experience plus
attempted-action reward produced a real, held-out, pixel-grounded temporal
relation that all shuffled controls destroyed. But the exact relation and
generalization claim must be narrowed until a stronger audit passes.

## Immediate proposed audits

Candidate audit A:

```text
reverse support object order only
keep feedback identity fixed
keep query pixels fixed
recompute the correct private rule
```

If the agent learned support-order binding, its prediction should flip. If it
uses feedback-plus-query, it should not.

Candidate audit B:

```text
randomize query orientation independently of support orientation
maintain a balanced valid generator
evaluate without retraining
```

Candidate repair:

```text
train with support and query orientations independently randomized
or remove the query from the rule-identification microtask
then audit on new identities, palettes, render seeds, and orientation relations
```

We need you to check whether these are logically and causally correct and
propose stronger alternatives if needed.

## Primary research question

What is the cheapest rigorous experiment that distinguishes:

1. genuine demonstrated support-order binding;
2. a legitimate but unintended feedback-plus-query temporal relation;
3. renderer/generator shortcuts;
4. memorization or distribution-specific classification?

Then, assuming that audit is fixed, what is the highest-ROI path from this
single primitive toward reusable, compounding zero-label learning?

## Questions to investigate

1. Formally analyze the current episode graph and identify every information
   path from visible pixels to the private rule. Which interventions are valid,
   sufficient, and in-distribution?

2. Is support-only reversal with fixed feedback and fixed query a valid causal
   counterfactual? State exactly how the verifier label must change.

3. How should support and query orientations be sampled so neither marginal nor
   their correlation leaks the rule, while every lifetime remains deterministic
   and has one verifiable answer?

4. Should the rule-identification microtask omit the query entirely, or should
   behavior be changed to applying the inferred rule to a query? Compare the
   scientific claims each task supports.

5. Design a minimal query-application task whose final action cannot be solved
   by directly outputting a latent rule bit and whose actuator labels carry no
   fixed human semantics.

6. Propose a complete counterfactual audit matrix: support-only reversal,
   query-only reversal, both reversal, feedback identity swap, independent
   orientation resampling, palette swap, render-seed swap, missing feedback,
   and missing order. Give expected predictions for each causal hypothesis.

7. Distinguish valid sensory counterfactuals from out-of-distribution corruption.
   When is zeroing frames informative, and when should we instead re-render a
   valid episode with evidence absent or ambiguous?

8. Our success head is accurate but overconfident: final ECE is commonly
   0.18–0.21, and missing evidence does not always raise entropy enough. What
   zero-semantic-label calibration method is appropriate under policy-induced
   distribution shift? Compare Brier training, temperature scaling, ensembles,
   bootstrap uncertainty, and abstention/retrieval gates.

9. We directly evaluate deterministic held-out episodes. What additional value
   would IPS, doubly robust, SWITCH, or confidence-sequence OPE provide here,
   and when would it be unnecessary overhead?

10. What is the fairest matched actor-critic baseline for the fixed uniformly
    logged buffer? Avoid pretending an on-policy gradient remains on-policy
    after extensive replay.

11. How should we measure the actual benefit of replay? Give a two-axis design
    over unique reward bits and optimizer/examples/GPU budget that separates
    interaction efficiency, compute efficiency, and wall-clock efficiency.

12. After the causal audit passes, should we next:

    - transfer the same learned latent to a different temporal primitive;
    - test whether this primitive lowers examples-to-threshold on a subsequent
      primitive;
    - integrate success prediction into external memory;
    - or move to a closed-loop micro-intercept task?

    Rank these by expected information gained per GPU-minute.

13. Design a transfer task that shares an abstract cognitive primitive but
    changes all surface identities, actuator mapping, renderer correlations,
    and episode layout.

14. Design a memory-utility experiment where a write receives credit only from
    causal later retrieval advantage and cannot profit by storing noise.

15. Define the evidence required to claim compounding learning. We currently
    have one successful primitive, not compounding.

## Required output

1. An executive verdict naming the next sub-minute experiment.

2. A causal diagram of the current episode and a table of all alternative
   solution paths.

3. Exact pseudocode for a corrected generator and the strongest counterfactual
   audit.

4. A ranked table of at least 20 experiments by expected information gained
   per GPU-minute. For each provide:

   - hypothesis;
   - learner-visible information;
   - verifier-private information;
   - exact loss/reward;
   - matched controls;
   - accounting budget;
   - behavioral metric;
   - causal/adversarial audit;
   - pass threshold;
   - success/failure interpretation;
   - runtime tier.

5. A pre-registered decision tree for:

   - support-only reversal passes;
   - support-only reversal fails but independent-query retraining succeeds;
   - behavior survives only when support/query orientation is correlated;
   - missing evidence lowers accuracy but not confidence;
   - all audits pass on one seed;
   - three seeds pass;
   - retention or cross-primitive transfer fails.

6. A calibration and uncertainty protocol that uses no semantic training
   labels beyond attempted-action scalar outcomes.

7. A fair contextual-bandit baseline suite, stating when direct deterministic
   held-out evaluation supersedes OPE.

8. A next-primitive transfer experiment measuring examples-to-threshold for
   experienced versus fresh agents.

9. A “do not adopt yet” list, including ideas that are generally promising but
   unsupported for this exact bottleneck.

10. A bibliography dominated by original papers, official implementations,
    and independent empirical comparisons. Clearly distinguish contextual
    bandit, causal representation learning, invariant prediction, world-model,
    and tiny-online-agent evidence.

## Epistemic requirements

- Do not count discarded probes as capability.
- Do not call the three-seed result support-order binding until the stronger
  audit passes.
- Do not infer sample efficiency from endpoint accuracy alone.
- Do not compare 200-update replay to 17-update online learning without
  disclosing compute.
- Treat fixed-budget negatives as bounded.
- Require valid pixel-space re-rendering rather than hidden-state swaps.
- Treat one-seed counterfactuals as provisional.
- Keep the external verifier sovereign.
- Challenge our proposed support-only reversal if its causal semantics are
  wrong.
