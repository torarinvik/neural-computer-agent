# v21 stochastic write-credit retention diagnostic

This diagnostic tests whether a training-only Bernoulli write decision with a
straight-through differentiable transaction improves outcome-only retention
credit assignment after the v20 parent-protection failure. The controller also
receives a generic latest-token-to-prior-token match feature; no modality,
task, target-slot, or correct-action field is exposed.

The parent phase reached a stable `1.0` reward prefix by step 480 and remained
perfect through step 1,024. The retention phase nevertheless failed the causal
gate:

| condition | recall |
|---|---:|
| intact | 0.5010 |
| clear memory | 0.4951 |
| corrupt values | 0.4795 |
| reversed order | 0.4814 |
| random action | 0.4688 |

The mean write strength was `0.8902` and the durable commit rate was `85.50%`.
The intact/clear gap was only `+0.0059`, far below the `+0.15` promotion gate.
The candidate is rejected as a learned retention improvement. Keep the
straight-through sampler only as explicitly opt-in training infrastructure;
do not claim learned utility-based retention or promote these weights.

Accounting: 57,344 unique verifier bits, 24,576 unique logical lifetimes,
1,536 optimizer updates, 0 replayed examples, and 15.94 seconds wall time.

Report: `stochastic_target_first_seed17.json`.
