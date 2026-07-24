# Deep-research prompt: after passive prediction failed

You are advising an empirical project building a small, real-time,
sample-efficient neural computer. We need critical research that changes our
next experiments, not another general survey of AGI architectures.

## Non-negotiable constraints

The deployed learner receives only externally observable visual, auditory, or
visible-text streams, its own actions/internal memory, and scalar outcomes from
external deterministic verifiers.

It receives no human-authored semantic labels, concept names, task IDs,
privileged game state, solution traces, fixed DSL, English scratchpad,
hand-written reasoning rules, or hard-coded planner. Automatically generated
semantic targets are also forbidden in deployed training when they reveal the
abstraction to be learned.

Verifier-private facts may be used to compute scalar correctness/reward and to
run discarded offline probes. Probe weights and labels never enter deployed
weights and do not count as capability.

Accuracy is primary. Latency, memory, wall time, and compute are secondary
efficiency costs. Experiments begin under one minute, advance to roughly three
minutes only on audited evidence, then to ten minutes only after replication.

## Current system

The agent uses a small visual encoder, recurrent latent state, learned working
and external memory, learned read/write/consolidation paths, and action heads.
The task suite contains deterministic attention, association, spatial, shape,
temporal-order, and compositional primitives. Every task is rendered through
observable streams; private task metadata stays behind the verifier.

Established findings include:

- external memory is causally useful under empty/shuffled/garbage ablations;
- retention of prior primitives is relatively strong;
- association transfers better than temporal/compositional rules;
- event snapshots plus multiplicative binding can solve an audited supervised
  diagnostic, but semantic supervision is not acceptable for final training;
- the learned temporal relation was identity-specific under palette swaps;
- direct visible outcome feedback made the relation decodable, localizing an
  important outcome-perception bottleneck;
- three seeds showed better endpoint representations from prior experience
  (94.44% versus 83.61% at 120 lifetimes) but no reliable reduction in
  lifetimes-to-threshold.

## New zero-label experimental evidence

We implemented a three-arm test with identical sub-1M-parameter recurrent
agents:

1. correctly paired future-latent prediction;
2. matched fresh initialization;
3. shuffled-future prediction.

Prediction training saw rendered RGB sequences only. The downstream two-action
policy was trained only by sampled action and scalar 0/1 verifier reward. The
private correct action was never differentiated through.

Experiment A: EMA target plus cosine next-latent prediction and weak
variance/covariance regularization:

- runtime: 17.22 seconds;
- effective rank: 1.14/64;
- paired-versus-shuffled loss difference: approximately 0.000001;
- all reward-learning arms: 50%.

Conclusion: partial dimensional collapse.

Experiment B: standardized dimension-wise matching plus explicit variance and
correlation penalties:

- paired-future alignment margin: 0.93 at seed 211 and 0.90 at seed 257;
- recurrent effective rank: 5.12 and 5.31;
- reward-only held-out performance: 51.0% and 52.3%;
- discarded rule probes on predictive states: 52.1% and 47.4%;
- a shuffled-future probe reached 76.0% on seed 211 but 52.6% on seed 257 and
  was rejected as a one-seed artifact.

Conclusion: the model learned genuine predictable structure, but not the
relation required for behavior. Better world-model loss did not produce a
better control state.

A third fork is implemented but not yet run: predict target-encoder latent
change `z(t+1)-z(t)` instead of absolute `z(t+1)`, with the same controls.

## Primary research question

Given these results, what zero-semantic-label objective is most likely to make
a tiny recurrent state intervention-sensitive and useful for later reward
learning, rather than merely predictive of nuisance structure?

Do not simply recommend more data, a larger model, or generic JEPA pretraining.
Explain what information each proposed objective pressures the recurrent state
to preserve and why it addresses our measured failure.

## Questions to investigate

1. Compare absolute next-latent prediction, latent-delta prediction,
   multi-horizon prediction, masked-event prediction, temporal-distance
   prediction, predictive-information objectives, data2vec-style contextual
   targets, and surprise-weighted prediction in tiny recurrent regimes.

2. Could latent-delta prediction merely replace static nuisances with dynamic
   nuisances? Propose the cheapest causal control that distinguishes useful
   change representation from motion/background shortcuts.

3. How should we add action conditioning without adding a hard-coded planner?
   Compare forward dynamics, inverse dynamics over the agent's own actions,
   controllability/empowerment, successor features, and Dreamer-style latent
   imagination. Separate passive observation tasks from actual control tasks.

4. Investigate contextual-bandit or action-conditioned success models trained
   only from `(latent state, attempted action, scalar reward)`. Can replaying
   these observations use each reward bit more efficiently than REINFORCE
   without inferring unobserved correct-action labels? Address off-policy
   correction, exploration, calibration, and Goodhart risks.

5. Our standardized predictor achieved effective rank around 5/64. What
   anti-collapse mechanism is most reliable for 0.5M–2M parameter recurrent
   agents and batches below 128? Compare VICReg, VICRegL, Barlow Twins,
   whitening, redundancy reduction, EMA teachers, centering/sharpening, and
   explicit effective-rank regularization. Identify failure cases where rank
   rises but task-relevant information falls.

6. How can self-supervision preserve relations distributed across event times?
   Research objectives that reward retention of information useful several
   events later without semantic labels. Include multi-step latent prediction,
   memory-augmented prediction, contrastive predictive coding, and predictive
   state representations.

7. How should learned external memory be trained from zero-label signals?
   Compare surprise-only writes against future predictive utility, retrieval
   advantage, counterfactual behavioral gain, compression-without-degradation,
   and learned eviction. Propose a method that cannot profit by storing noise.

8. Define an audit for missing evidence. When a decisive frame/modality is
   removed, confidence should fall or computation/retrieval should increase.
   Recommend calibration metrics and training methods that use only scalar
   outcomes and observable missingness.

9. Evaluate whether a GRU is still the right core at this scale. Only recommend
   Mamba/RWKV/selective-SSM or attention if evidence applies to tiny online
   agents. Design a matched under-three-minute admission test.

10. Determine whether object-centric slots are likely to solve our measured
    temporal relation problem or merely introduce another unneeded bias. Give a
    no-slot versus slot experiment with an occlusion/permanence/composition
    task where the result would be decisive.

11. Formalize compounding learning. In addition to AULC and
    lifetimes-to-threshold, assess a regression of log
    lifetimes-to-threshold against curriculum index. Specify how many distinct
    primitives and seeds are needed before this slope is meaningful.

12. Propose how to log a four-way information manifest for every experiment:
    learner-visible tensors, verifier-private facts, offline-probe-only facts,
    and forbidden channels. Identify common accidental leakage routes in
    procedural RL environments.

## Required output

1. A short executive verdict naming the best next **sub-minute** experiment
   after our immediate-next predictor failed.

2. A ranked table of at least 20 experiments ordered by expected information
   gained per GPU-minute. For each give:

   - hypothesis;
   - learner-visible information;
   - verifier-private information;
   - exact loss/reward;
   - matched controls;
   - anti-collapse telemetry;
   - behavioral metric;
   - adversarial/counterfactual audit;
   - pass threshold;
   - interpretation of success and failure;
   - runtime tier.

3. Exact pseudocode for the top three objectives, suitable for a small PyTorch
   GRU encoder. Avoid semantic targets.

4. A decision tree beginning with the pending latent-delta result:

   - if prediction improves but behavior/probes remain flat;
   - if discarded probes improve but reward learning remains flat;
   - if reward learning improves but counterfactuals fail;
   - if behavior, causality, and retention all pass.

5. A contextual-bandit success-predictor design that uses only attempted
   actions and scalar rewards, plus a REINFORCE baseline. State which is
   expected to use reward information more efficiently and under what
   assumptions.

6. A memory-utility experiment comparing surprise, predictive utility, and
   retrieval advantage under equal write and compute budgets.

7. A “do not adopt yet” list covering ideas that are promising generally but
   unsupported for our measured bottleneck.

8. A bibliography dominated by original papers, official implementations, and
   independent empirical comparisons. Clearly distinguish results from large
   pretrained systems, language modeling, and continuous-control benchmarks
   from evidence that actually applies to tiny recurrent agents.

## Epistemic requirements

- Treat our two-seed results as bounded evidence, not universal conclusions.
- Do not count discarded probe accuracy as deployed capability.
- Do not infer sample efficiency from endpoint accuracy.
- Do not call a fixed-budget negative an architectural impossibility.
- Require optimization-health and anti-collapse checks before interpreting
  held-out failure.
- Require valid pixel/sensory counterfactual replay; swapping recurrent tensors
  is out-of-distribution and not a causal audit.
- Reject one-seed breakthroughs until replicated.
- Prefer experiments that distinguish competing explanations over architecture
  additions.
- Explicitly challenge the latent-delta hypothesis and recommend a stronger
  alternative if primary evidence supports one.
