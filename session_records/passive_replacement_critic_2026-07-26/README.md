# Passive replacement critic — pre-registration

## Question

Can a small critic predict the verified result of an attempted external-memory
replacement better than a constant predictor, using no correct action,
unattempted-action outcome, semantic task identifier, or privileged state?

This first rung is strictly observational. The critic cannot change the
controller's actions, memory writes, compute allocation, or rewards.

## Learner-visible evidence

- generic per-bank statistics created by the controller;
- the generic statistics of the option actually attempted;
- the exact logging propensity of that attempt;
- the attempted option's policy margin;
- the later scalar verifier outcome.

The logging policy is a fixed epsilon mixture over the frozen controller's
replacement scores. Its probabilities are recorded exactly.

## Sub-minute budget

- pure-redundancy, capacity-three atom;
- 8 fresh batches of 64 logical lifetimes;
- 1,536 unique verifier bits per critic;
- 8 optimizer updates per critic;
- zero replay;
- 128 lifetime-disjoint held-out attempts on an unseen seed.

All four critic arms see the same attempted actions and consume no extra
environment experience.

## Controls

1. reward-shuffled training outcomes;
2. missing attempted-action evidence;
3. missing generic bank-context evidence;
4. constant-rate predictor;
5. exact critic save/reload;
6. binary-mapping and four-rule retention on the unchanged controller.

## Promotion gate

The run earns a roughly three-minute replication only if:

- at least three actions are covered and every propensity is positive;
- held-out outcomes have standard deviation at least `0.05`;
- intact Brier score beats the constant predictor by at least `0.005`;
- intact Brier beats every learned control by at least `0.002`;
- intact held-out concordance is at least `0.55`;
- intact expected calibration error is at most `0.10`;
- Brier remains better than constant at both final measured prefixes;
- every arm has live gradients and exact save/reload;
- inherited binary mapping and four-rule gates pass.

A failure is interpreted as a bounded result at this data and update budget.
No critic may influence behavior unless this passive gate passes and then
replicates.

## Harness correction after seed 7321

The first execution was not treated as a scientific negative. The learned
critics began near `0.5`, while the constant comparator was handed the
empirical outcome rate (about `0.82`), so eight updates primarily measured
base-rate fitting. The four control arms also used different initial weights.

Seed 7322 retains the exact pre-registered experience budget and thresholds,
but every arm now starts from one bit-identical initialization. Its final
residual projection is zero, and predictions are expressed as the empirical
base-rate logit plus a learned residual. This makes prefix zero exactly equal
to the constant comparator and asks only whether attempted-action evidence
earns a calibrated improvement.

## Results

The corrected seed 7322 run completed in 11.45 seconds. The intact critic was
well calibrated (`ECE=0.0092`) and ranked outcomes above chance
(`concordance=0.588`), while reward shuffling reduced concordance to `0.501`.
But its Brier improvement over the empirical-rate baseline was only `0.00012`,
far below the pre-registered `0.005` bar. Removing context slightly improved
ranking to `0.616`, revealing that context necessity was not a meaningful
control inside a single fixed utility atom.

An unchanged corrected replication on unseen seed 7323 completed in 11.17
seconds and did not reproduce the signal. Intact concordance was `0.499`,
action-only concordance was `0.514`, and every learned arm was worse than the
constant Brier baseline.

A 256-lifetime excitation audit tested both the registered epsilon-mixture and
fully uniform logging. Both covered all four actions and produced outcome
standard deviation near `0.177`; uniform exploration did not create a larger
signal. The audit consumed 1,536 additional verifier bits and no optimizer
updates.

## Conclusion

The passive plumbing, exact propensity accounting, live gradients,
save/reload, and no-forgetting controls all work. This critic target is
rejected for promotion: aggregate future success under the fixed
pure-redundancy atom does not yield a stable feature-conditioned prediction
gain at 512 unique attempts. No critic influences actions or compute.

The next rung should reduce the prediction jump rather than scale this run:
predict a shorter-horizon verifier event from the immediately preceding
attempt, then add horizon or context diversity one axis at a time. That tests
whether temporal distance and outcome averaging erased the reusable signal.
