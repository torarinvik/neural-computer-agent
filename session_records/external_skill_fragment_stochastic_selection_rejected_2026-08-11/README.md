# Stochastic multi-sample causal selection — rejected — 2026-08-11

## Question

Can active causal row selection become useful if the intervention probe uses a
bounded stochastic policy rather than deterministic argmax, and averages four
common-random verifier outcomes per candidate? This tests whether the earlier
zero-signal result was only a measurement-resolution failure.

The active arm selected the highest average answer-changing rows. The passive
arm paid for the identical stochastic probe and selected a matched random
subset. Temperature was `0.5`, samples per candidate were `4`, and all
verifier-private outcomes remained outside the deployed controller, combiner,
and decoder interfaces.

## Three-seed result

Both arms used seeds `41/42/43`, serial composition, leave-one-out credit
weight `0.5`, candidate multiplier `2`, updates `8/16/16`, batch size `8`,
span `3`, and audit count `16`.

| arm | held-out order accuracy by seed | stable prefix | promoted |
| --- | --- | --- | --- |
| active stochastic top-k | `0.4792/0.5208/0.5000`; `0.5833/0.4583/0.5625`; `0.5417/0.4792/0.4792` | none | no |
| passive stochastic random-k | `0.5208/0.4792/0.5000`; `0.6042/0.4375/0.4792`; `0.5625/0.4375/0.4792` | none | no |

The probe signal increased materially—for example, active seed `42` ended at
`0.0674` candidate mean and `0.1172` selected mean—but this did not transfer
to held-out ordered execution. Each seed used `165,456` unique verifier bits,
including `138,240` stochastic selection/intervention bits and `13,824`
leave-one-out training bits, with zero replay and `120` optimizer updates.

## Decision

Reject stochastic active selection as the missing composition mechanism. The
system can now measure answer-changing interventions, but selecting them does
not teach a reusable ordered execution law. Retain the temperature and
multi-sample probe as causal-audit infrastructure. The next implementation
must improve the external execution representation/operator itself—so that
ordered fragments compose into a stable factual state—before spending more
verifier budget on selection.
