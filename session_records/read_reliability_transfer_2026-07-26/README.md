# Fixed-capacity reliability transfer — pre-registration

## Question

After sequentially consolidating capacity-four and capacity-five experience,
does that lineage learn a changed retrieval-reliability regime faster than its
capacity-four ancestor and a matched reset learner?

Capacity remains five. The only target change is write/retrieval threshold
`0.5 → 0.6`, which alters occupancy and stored-item reliability statistics.
The learner is not told the threshold or capacity. The controller, sensory
stream, four generic evidence values, attempted-action objective, optimizer,
read cost, and model size remain unchanged.

## Lineage and accounting

1. Reconstruct the passing seed-7842 capacity-five phase from the curated
   capacity-four checkpoint.
2. Save the resulting capacity-five checkpoint with its 720 source bits and
   12 source updates attached to provenance.
3. On the threshold-0.6 target, race:
   - consolidated capacity-five lineage;
   - capacity-four ancestor;
   - matched reset;
   - reward-shuffled, feature-shuffled, and missing-evidence controls.

The target race uses 720 fresh learner-visible outcomes, 12 updates, zero
replay, and 2,040 held-out contexts/4,080 private counterfactual audit bits.

## Gates

The consolidated lineage must stably reach:

- at least `65%` compute-choice accuracy;
- at least `+0.05` utility over the strongest fixed action;
- at least `20%` oracle-gap capture;
- all three thresholds earlier than both ancestor and reset.

Every causal control must cost at least `0.02` utility. Gradients,
serialization, binary mapping, and four-rule retention must pass. Seed 7851 is
the discovery run; only a complete pass permits unchanged seed-7852
replication. Failure earns no additional training budget.

Transfer distance increases scientific value only after these verified gates
pass; novelty itself produces no reward.

## Threshold 0.6 result and frontier diagnostic

The consolidated lineage was stronger at threshold 0.6 (`83.4%` choice
accuracy versus `80.8%` for its ancestor), but both were already stably
mastered at zero new bits. It therefore failed the pre-registered
bits-to-threshold advantage and is not replicated.

This rung was too easy rather than too hard. A zero-training private diagnostic
now measures thresholds `0.60, 0.65, 0.70, 0.75, 0.80` on 2,040 contexts each.
It selects the smallest measured threshold where the capacity-five lineage
passes the fixed mastery gates and the capacity-four ancestor does not. The
probe uses no learner-visible outcomes or optimizer updates; its 20,400
private counterfactual verifier bits are reported. The selected threshold must
still pass a fresh pre-registered training race and replication.

All measured thresholds left both lineages mastered at zero bits. The newer
lineage nevertheless improved verified utility by `0.033–0.053`, so the
representation became better but the threshold metric was saturated.

The next zero-training diagnostic holds capacity five and write threshold 0.5
fixed, then sweeps read cost `0.01, 0.05, 0.10, 0.20, 0.30`. Both-action
outcomes are reused across costs, so this adds no learner-visible experience
and 4,080 private verifier bits. The smallest cost where the consolidated
lineage masters and its ancestor does not becomes the next candidate; it must
still pass a fresh attempted-outcome race and replication.

The coarse cost sweep found both lineages mastered at `0.20` and both failed
at `0.30`; no separating point was present in the original grid. The ancestor
was only barely above the oracle-gap gate at `0.20`, while the consolidated
lineage retained a growing utility advantage. A final zero-training refinement
measures `0.22, 0.24, 0.26, 0.28` inside that pre-identified bracket. It adds
no learner-visible bits and reuses one 4,080-bit private both-action audit.

The refinement selected read cost `0.24`: the capacity-five lineage passed at
zero bits (`73.5%` choice accuracy, `30.8%` gap capture), while its
capacity-four ancestor failed (`18.8%` gap capture). This is the smallest
measured separating cost.

Seed 7871 is pre-registered as the fresh powered training race at capacity
five, threshold 0.5, and read cost 0.24. It uses the standard 720 target bits,
12 updates, three-way lineage/ancestor/reset comparison, causal controls, and
retention gates. A full pass permits one unchanged seed-7872 replication.

## Seed 7871 and prospective control correction

Seed 7871 reached stable mastery at zero target bits; its ancestor required
120 and reset never crossed within 720. Final inherited accuracy was `80.6%`
with `57.9%` gap capture. Evidence shuffling and missing evidence collapsed
utility, and all retention/mechanical gates passed.

The run was nevertheless rejected because reward-shuffled **training** did not
damage the already-mastered inherited policy. That control cannot test
zero-shot knowledge: the policy passes before seeing any target reward, so
later shuffled rewards need not erase it.

Seed 7871 remains rejected under its original gate. Prospectively, fresh seed
7873 uses the following structural rule:

- if inherited stable mastery is at zero target bits, causal lineage/ancestor,
  reset, feature-shuffle, and missing-evidence controls are mandatory;
- reward-shuffle performance is recorded but is not required to damage
  knowledge learned before the target phase;
- if inherited mastery requires any target learning, reward shuffling must
  still cost at least `0.02` utility.

A full seed-7873 pass permits unchanged seed-7874 replication.

## Replicated cost-generalization result

Both fresh seeds passed the corrected zero-shot causal gate:

- seed 7873: consolidated lineage stable at `0` target bits, ancestor at
  `120`, reset at `360`; final accuracy `81.1%`, gap capture `58.5%`;
- seed 7874: consolidated lineage stable at `0`, ancestor at `120`, reset at
  `120`; final accuracy `81.7%`, gap capture `59.3%`.

Feature shuffling and missing evidence destroyed the benefit. Both old-skill
retention gates, gradients, and serialization passed. Reward-shuffled target
updates did not damage the already-mastered zero-shot policy and are reported
without being treated as a causal requirement.

The seed-7874 experience was then consolidated into a new checkpoint. Its
provenance records 720 cost-0.24 source outcomes and 12 updates, on top of the
earlier capacity lineage. This establishes replicated compounding across a
second axis: the newer lineage immediately adapts not only to more memory
slots, but to a substantially different price for extra computation.

The next frontier should test a changed physical compute operation through the
closest available bridge—preferably read versus re-query—using the
cost-consolidated checkpoint, its capacity-five ancestor, and reset. A direct
jump to recurrent thought remains too large.
