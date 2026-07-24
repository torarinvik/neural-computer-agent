# Deep-research prompt: from one zero-label primitive to compounding learning

You are advising an empirical project building a small real-time,
sample-efficient neural computer. We have achieved one audited zero-semantic
behavioral primitive. We now need the cheapest rigorous path to demonstrate
reusable transfer and eventually compounding learning.

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

The learner never creates a target for an unattempted action, even when a
failed action in a deterministic two-action task logically reveals the
alternative.

Accuracy is primary. Unique experiences and reward bits are the main
efficiency costs. Optimizer updates, examples processed, GPU-seconds, latency,
memory, and wall time are accounted separately. Experiments begin under one
minute and scale only after replicated causal evidence.

## Architecture

The current sub-1M-parameter system contains:

- a small visual encoder;
- a GRU recurrent state;
- broader-project working/external memory machinery;
- a zero-semantic predictive objective;
- an attempted-action-only success head.

Predictive pretraining sees rendered RGB and predicts EMA latent changes
`z(t+1)-z(t)` with variance/correlation anti-collapse regularization.

The success head receives only:

```text
frozen recurrent state
attempted action
logging propensity
observed scalar verifier reward
```

It trains BCE only on the attempted action. No counterfactual action label,
rule label, identity, palette, event index, task ID, or game state enters
optimization.

## Failure that was caught

An initial five-frame task contained support order, selected-object feedback,
and a query whose orientation was always opposite the support. Three seeds
reached roughly 80%, but a stronger audit showed the model used the unintended
`feedback identity + query order` path:

- removing support order barely hurt;
- reversing support alone barely changed predictions;
- reversing query alone changed predictions even when the rule stayed fixed.

The earlier audit had reversed both support and query, so the shortcut also
passed. We explicitly rejected the support-binding claim.

## Repaired three-seed milestone

We removed the query from the rule-identification microtask. The only useful
visible route became:

```text
support object order + selected-object feedback -> first/last rule
```

Every learning-curve point used a balanced uniform logged buffer prefix of
30/120/240/360/510 unique reward bits, 200 optimizer updates, and 6,000
processed examples. All arms received identical head initialization and
compute.

Results for seeds 211, 257, and 313:

- paired-delta success final: 78.13%, 82.03%, 80.73%;
- paired-delta AULC: 0.2188, 0.2453, 0.2068;
- three-seed mean final: 80.30%;
- three-seed mean AULC: 0.2236;
- shuffled-future representation mean final: 55.73%;
- shuffled-future representation mean AULC: 0.0434;
- equally optimized IPS learner mean AULC: 0.1780;
- true support-order reversal mean relabeled accuracy: 78.82%;
- true support-order reversal mean prediction flips: 59.11%.

Fresh representations stayed near chance. Action-shuffled and reward-shuffled
controls stayed near chance or worse. Correctly paired predictive experience
therefore caused a reusable state advantage, and attempted-action-only scalar
reward converted it into verified behavior.

This is the first clean zero-semantic deployed-learning result in the project.
It is **not** yet evidence of:

- compounding learning;
- cross-primitive reuse;
- general action semantics;
- memory-based growth;
- calibrated uncertainty;
- online adaptation under a policy-induced distribution;
- conventional reasoning transfer.

## Accounting caveat

The successful replay curve used 200 optimizer updates at each reward-bit
prefix. An earlier 17-update online run stayed around 53%. A compute sweep on
fixed reward bits showed a sharp ignition between 68 and 200 updates. We claim
interaction efficiency only at the explicitly fixed compute budget; we do not
pretend replay is computationally free.

## Primary research question

What is the single highest-information next primitive for determining whether
the acquired predictive representation and attempted-action learning method
make a genuinely new capability faster to learn?

The new task should share a reusable cognitive operation while changing
surface identities, renderer correlations, episode layout, and actuator
mapping. It must be deterministic, visually grounded, and solvable from scalar
outcomes without semantic labels.

## Questions to investigate

1. Should the next transfer task test:

   - temporal rule application to a novel query;
   - same/different relational binding;
   - delayed match/non-match;
   - reversal learning;
   - multi-choice reaction under distractors;
   - a micro-intercept closed-loop task;
   - or another primitive?

   Rank them by expected information gained per GPU-minute and shared abstract
   structure with the achieved primitive.

2. Design experienced-versus-fresh controls that isolate reuse of learned
   machinery from mere extra pretraining, optimizer warm-start, or favorable
   initialization.

3. Should we transfer the encoder/GRU, success head, both, or neither? Give a
   factorial experiment that identifies where reuse lives.

4. How can actuator mapping be permuted so fixed action semantics cannot
   transfer, while an abstract action-selection mechanism still can?

5. How can surface identities, palettes, frame count, timing, layout, and
   feedback rendering change without making the new task unrelated?

6. Define primary transfer metrics: reward AULC, unique reward bits to fixed
   thresholds, optimizer steps to threshold, GPU-seconds, and retained older
   capability. State which metric supports which claim.

7. Our success head is accurate but overconfident. Design zero-semantic
   calibration using attempted outcomes only. Should calibration be fixed
   before transfer, or evaluated in parallel without blocking the next
   primitive?

8. Design a matched actor-critic baseline for fixed logged data without
   mislabeling off-policy replay as on-policy learning.

9. When does direct deterministic held-out evaluation make IPS/DR/SWITCH OPE
   unnecessary, and when will OPE become essential?

10. Design an online version of the successful replay learner that adapts under
    its own induced action distribution while maintaining support and
    recalibration.

11. How should event representations enter external memory? Propose a
    fixed-budget write experiment where credit comes only from causal later
    retrieval advantage and cannot reward noise storage.

12. Could the achieved delta-predictive state transfer to a closed-loop
    micro-intercept task, or is action-conditioned multi-horizon prediction
    required there? Give a passive, action-conditioned, and shuffled-action
    comparison.

13. Define the minimum sequence of at least six primitives needed to estimate
    whether `log(reward_bits_to_threshold)` declines with curriculum index.

14. Specify a mixed-effects or hierarchical analysis across seed, primitive,
    and curriculum position. State how to handle tasks that never reach a
    threshold.

15. Identify likely negative transfer and forgetting modes when success heads,
    predictive state, and memory begin to accumulate.

## Required output

1. An executive verdict naming the next sub-minute transfer experiment.

2. A ranked table of at least 20 candidate experiments by expected information
   gained per GPU-minute.

3. Exact pseudocode for the top candidate's generator, learner-visible stream,
   verifier, training loop, and counterfactual audit.

4. A factorial transfer matrix comparing:

   - experienced encoder/GRU + experienced success head;
   - experienced encoder/GRU + fresh success head;
   - fresh encoder/GRU + experienced success head where dimensions permit;
   - fully fresh;
   - shuffled-predictive representation;
   - action-shuffled and reward-shuffled controls.

5. Pre-registered pass thresholds for earlier learning rather than endpoint
   accuracy alone.

6. A decision tree for:

   - endpoint improves but thresholds do not;
   - early thresholds improve on one seed only;
   - three seeds show faster learning;
   - transfer disappears under actuator remapping;
   - transfer succeeds but retention fails;
   - the second primitive succeeds but the third does not become easier.

7. A six-primitive compounding curriculum where every task has one
   deterministic answer and no semantic training labels.

8. A memory-utility experiment based on retrieval advantage.

9. An online calibration/exploration protocol for policy-induced distribution
   shift.

10. A “do not adopt yet” list.

11. A bibliography dominated by original papers, official implementations,
    and independent empirical comparisons. Separate contextual-bandit,
    transfer/meta-learning, continual-learning, causal representation,
    external-memory, and closed-loop world-model evidence.

## Epistemic requirements

- Do not count probes as capability.
- Do not infer compounding from one primitive.
- Do not infer sample efficiency from final accuracy.
- Do not hide replay compute.
- Treat one-seed transfer as provisional.
- Require at least three seeds for a load-bearing transfer claim.
- Require valid pixel-space counterfactuals and surface remapping.
- Keep the external verifier sovereign.
- Challenge our proposed next primitive if a cheaper experiment better tests
  reuse.
