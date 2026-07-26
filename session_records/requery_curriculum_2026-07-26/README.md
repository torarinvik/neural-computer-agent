# Operation-aligned re-query curriculum — pre-registration

## Question

Does preserving a learned second-ranked re-query policy reduce the verified
experience needed when memory capacity increases from five to six?

This is the smallest nearby difficulty shift available: the physical operation,
action meaning, cost, sensory/controller path, and four generic evidence
features remain unchanged. Only the number of competing latent memories grows.

## Discovery race

First reconstruct seed 7893 exactly and save its learned re-query head. Then run
one 720-bit discovery race at capacity six:

- **re-query lineage:** the complete learned capacity-five re-query head;
- **read ancestor:** the earlier capacity-five read-value trunk with its
  operation-specific output reset;
- **reset:** a newly initialized head;
- **causal controls:** reward shuffled, evidence shuffled, and evidence absent.

All learners receive only attempted actions and their scalar verifier outcomes.
The unattempted outcome, correct compute choice, answer, and semantic task
identity remain private.

## Gate

The re-query lineage must:

1. stably reach at least 65% compute-choice accuracy;
2. beat the strongest fixed policy by at least 0.03 verified utility;
3. capture at least 20% of the available oracle gap;
4. reach stable mastery in strictly fewer unique verifier bits than both the
   read ancestor and reset;
5. retain the binary-mapping and four-rule skills;
6. pass evidence, reward, gradient, and exact-persistence controls.

Only a complete pass permits an unchanged replication seed. A tie is not
compounding evidence. A failure or regression closes capacity six at this
curriculum step; it does not authorize a longer run.

## Capacity-six result and smaller-step fork

Seed 7911 did not pass. The inherited re-query policy captured `67.5%` of the
available utility gap and retained prior skills, but achieved only `33.7%`
unweighted compute-choice accuracy. No arm reached stable mastery within 720
bits. The apparently conflicting utility and accuracy are possible because
the inherited policy made many low-regret classification errors; the
pre-registered accuracy gate is retained unchanged.

Capacity six therefore remains too large a first curriculum step. Zero-training
viability probes at capacity five show that costs `0.015`, `0.020`, and `0.030`
all retain a meaningful two-sided adaptive decision. The next discovery seed
uses the smallest change, cost `0.015`, with the complete capacity-five
re-query head compared against its read ancestor and reset under the same gate.
Only a complete pass permits unchanged replication; no longer budget is
authorized.

## Fresh-stream robustness audit

Cost `0.015` seed 7931 also failed: the inherited head retained useful utility
but reached only `51.4%` choice accuracy and never reached stable mastery.
Because this is much lower than its original `71.3%` held-out result, no
additional training is allowed until the saved capacity-five head is audited
without learning across eight fresh streams.

Robust mastery requires every stream to preserve the original accuracy,
utility-improvement, and oracle-gap gates. Failure means the original re-query
result was seed-fragile: it remains evidence of within-run learning, but the
checkpoint cannot be promoted as a curriculum parent.

## Audit result and corrected conclusion

The saved capacity-five policy is robust:

- capacity five, cost `0.010`, eight fresh streams: minimum accuracy `70.4%`,
  minimum oracle-gap capture `69.2%`;
- capacity six, cost `0.010`, eight fresh streams: minimum accuracy `70.4%`,
  minimum oracle-gap capture `68.6%`;
- capacity five, cost `0.015`, eight fresh streams: minimum accuracy `69.6%`,
  minimum oracle-gap capture `70.2%`.

All three frozen audits passed every stream with zero learner-visible verifier
bits. Thus the policy already transfers robustly across the proposed capacity
and cost shifts. This is genuine zero-shot reuse and the strongest
sample-efficiency result at this frontier.

The apparent curriculum failures came from destructive online updates. At
capacity six, inherited accuracy fell from `70.4%` before learning to `33.7%`
after 720 bits. At cost `0.015`, it fell from `70.4%` to `51.4%`. Longer
training is therefore the wrong response.

The new frontier is **safe adaptation control**: the controller must preserve
or freeze a policy that already performs well, while remaining able to update
when verified evidence shows a real capability gap. This mechanism must use
only attempted-action outcomes and generic uncertainty/learning signals—not
private counterfactual answers or semantic task labels.
