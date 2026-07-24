# Deep-research prompt: turning useful latent information into fast action learning

You are advising an empirical project building a small real-time neural
computer. We need primary-source-grounded research that changes our next tiny
experiments, not a broad AGI survey.

## Non-negotiable constraints

The deployed learner receives only externally observable visual, auditory, or
visible-text streams, its own actions and internal memory, and scalar outcomes
from external deterministic verifiers.

It receives no human-authored semantic labels, concept names, task IDs,
privileged simulator state, solution traces, generated correct-action labels,
fixed DSL, English scratchpad, hand-written reasoning rules, or hard-coded
planner.

Verifier-private facts may calculate scalar reward and discarded offline
diagnostics. Probe weights never enter the agent, and probe accuracy never
counts as capability.

Accuracy is primary. Sample count is the main efficiency cost; latency, memory,
wall time, and compute are secondary. Experiments begin under one minute,
advance to roughly three minutes only on audited evidence, and advance to ten
minutes only after replication.

## Current architecture and established evidence

The current sub-1M-parameter agent contains a small visual encoder, GRU state,
learned working/external memory in the broader system, and action heads. The
task suite includes deterministic attention, association, spatial, shape,
temporal-order, and compositional primitives rendered through visual streams.

Established results:

- external memory is causally useful under empty, shuffled, and garbage
  ablations;
- association transfers better than temporal/compositional rules;
- retention of older primitives is relatively strong;
- supervised discarded probes localized a cross-event binding mechanism, but
  those labels and weights are forbidden in deployed learning;
- prior experience improved endpoint representation quality without reliably
  lowering examples-to-threshold;
- immediate absolute-next-latent prediction learned real predictability but
  did not improve reward-only behavior.

## New decisive result

We ran a three-arm, 23.14-second experiment:

1. correctly paired latent-delta prediction;
2. matched fresh representation;
3. shuffled-future latent-delta prediction.

Predictive pretraining saw rendered RGB sequences only. The downstream learner
saw only sampled actions and scalar 0/1 verifier reward.

The correctly paired delta arm achieved:

- paired-versus-shuffled held-out alignment margin: 0.522;
- recurrent effective rank: 5.64/64, below our pre-registered 6.4 gate;
- discarded MLP temporal-rule probe: 80.73% held-out;
- valid pixel-space reversal relabeled accuracy: 79.69%;
- shuffled-label diagnostic calibration: 46.61%.

Yet deployed reward-only behavior was 50.26%, versus 52.34% for the
shuffled-future control. No 60/70/80% threshold was reached, and reward AULC
advantage over the best control was -0.00391.

Bounded conclusion: latent-delta prediction produced a relation-bearing state,
but the current REINFORCE readout did not exploit it within 510 unique
lifetimes. This is representation localization, not capability.

## Important causal distinction

The current temporal task is effectively a contextual bandit at answer time.
The sampled answer changes scalar reward but does not create a rich subsequent
visual transition. Therefore action-conditioned prediction of future
observations may be scientifically empty on this task even if an action tensor
is formally supplied.

For this task, the immediate candidate is a success model trained only from
observed `(latent state, attempted action, scalar reward, logging propensity)`
tuples. It must not manufacture a target for an unattempted action—even when a
failed action in a deterministic two-action task logically reveals the other
answer.

Action-conditioned multi-horizon world modeling should be evaluated on a
closed-loop primitive such as choice reaction or Pong, where actions causally
alter later sensory input.

## Primary research question

What is the most sample-efficient, zero-semantic-label way to convert a latent
state that contains an actionable relation into verified behavior using only
attempted actions and scalar outcomes?

## Questions to investigate

1. Compare REINFORCE, actor-critic, action-conditioned binary success models,
   contextual-bandit replay, fitted Q evaluation, conservative bandit updates,
   and successor-style outcome prediction in a deterministic two-action
   setting.

2. How should a success model reuse logged reward bits without training on
   unobserved actions? Compare uniform exploration, adaptive exploration,
   inverse-propensity weighting, doubly robust estimation, and conservative
   policy improvement.

3. Does replay offer genuine reward-bit efficiency when optimizer steps and
   wall time are matched? Specify fair controls separating fewer interactions
   from extra gradient computation.

4. What exploration floor prevents one-action collapse while remaining
   sample-efficient? Give guidance for batches of approximately 30 and only
   510 unique interactions.

5. How should calibration be measured when the predictor changes the behavior
   distribution it learns from? Include Brier score, ECE, ranking, induced-
   distribution recalibration, and confidence intervals.

6. Design Goodhart and leakage audits for a success predictor. Include
   action-shuffled replay, reward-shuffled replay, fresh representations,
   balanced action coverage, nuisance randomization, valid pixel-space event
   reversal, and missing-evidence tests.

7. When is it legitimate for a two-action learner to infer the unobserved
   action's outcome from a failed attempt? Analyze both the scientific claim
   and the risk that this imports verifier/task structure. Our default is to
   forbid it.

8. Could reward-conditioned contrastive or predictive objectives improve the
   recurrent state without semantic labels, or would they merely encode
   reward shortcuts? Propose the cheapest decisive controls.

9. Once success prediction works, how should it influence behavior in stages:
   passive prediction, action ranking, another thought step, memory retrieval,
   or final answers? Address distribution shift after each stage.

10. For a genuinely closed-loop primitive, compare action-conditioned
    multi-horizon latent deltas, RSSM overshooting, PSR-style observable tests,
    inverse dynamics, and controllability/empowerment. State what evidence
    applies to tiny recurrent agents rather than large pretrained systems.

11. How should external memory writes be rewarded by causal retrieval
    advantage? Compare surprise, prediction improvement, later success gain,
    and compression-without-degradation under equal memory budgets.

12. Formalize evidence of compounding: how many primitives and seeds are
    required before a negative slope in log examples-to-threshold versus
    curriculum index is meaningful?

## Required output

1. An executive verdict on the best next sub-minute experiment.

2. A ranked table of at least 20 experiments by expected information gained
   per GPU-minute. For each provide the exact learner inputs, loss, controls,
   metrics, audit, pass threshold, failure interpretation, and runtime tier.

3. Exact PyTorch-like pseudocode for:

   - attempted-action-only success prediction;
   - its matched REINFORCE baseline;
   - a replay variant with logged propensities;
   - a valid action-shuffled control.

4. A fair accounting protocol separating:

   - unique verifier interactions;
   - unique logical lifetimes;
   - reward bits;
   - optimizer steps;
   - examples processed;
   - GPU-seconds;
   - wall time.

5. A pre-registered decision tree:

   - calibration improves but behavior remains flat;
   - behavior improves but action coverage collapses;
   - behavior improves but reversal fails;
   - behavior and causality pass on one seed;
   - two seeds pass but retention fails;
   - three or more seeds show earlier thresholds.

6. A separate closed-loop experiment where action conditioning is causally
   meaningful, with a no-action passive control and a shuffled-action control.

7. A memory-utility experiment that cannot profit from storing noise.

8. A “do not adopt yet” list for attractive ideas unsupported by our measured
   bottleneck.

9. A bibliography dominated by original papers, official implementations, and
   independent empirical comparisons. Clearly separate contextual-bandit,
   continuous-control, large-pretraining, and tiny-online-agent evidence.

## Epistemic requirements

- Do not count discarded probe accuracy as agent capability.
- Do not infer sample efficiency from final accuracy.
- Treat one-seed results as hypotheses.
- Treat fixed-budget failures as bounded negatives.
- Require optimization-health and action-coverage checks.
- Require valid pixel-space counterfactual replay; do not swap hidden states.
- The external verifier remains sovereign.
- Challenge our planned success model if a cheaper experiment better
  distinguishes the remaining explanations.
