# Latent re-query bridge — pre-registration

## Operation

The existing memory interface returns the highest-ranked latent value. The
candidate re-query operation retrieves the second-ranked valid latent value
from the same content-addressed bank and pays a generic cost of 0.01. The
controller receives only that latent value; no game state, correct answer,
task label, or unattempted outcome is learner-visible.

This is deliberately the closest physical-operation bridge from one external
read. It changes which memory candidate is inspected without jumping directly
to recurrent thought.

## Zero-training viability probe

On 10,240 fresh capacity-five contexts, privately evaluate the ordinary and
second-ranked reads. The probe spends zero learner-visible outcomes and 20,480
private counterfactual verifier bits.

The operation is viable only if:

- re-query helps on at least 2% of contexts;
- re-query harms on at least 2%;
- its oracle policy beats the strongest fixed choice by at least 0.02.

If viable, a fresh attempted-action race will compare the cost-sensitive
consolidated lineage, its capacity-five ancestor, and reset. If not, no
training is allowed; the re-query mechanism must be redesigned first.

## Viability result and training gate

The operation passed: re-query helped `14.3%`, harmed `85.7%`, and an adaptive
oracle improved over the strongest fixed choice by `11.9` points.

Seed 7891 is pre-registered for a 720-bit/12-update race. The cost-sensitive
lineage must stably reach at least 65% choice accuracy, improve at least 0.03
utility over the strongest fixed choice, capture 20% of the oracle gap, and
cross earlier than both its capacity-five ancestor and reset. Evidence
controls, conditional reward-shuffle logic, retention, gradients, and exact
serialization remain mandatory. A full pass permits unchanged seed-7892
replication.

## Seed 7891 direct-transfer result and one-axis localization

Direct whole-head transfer failed with severe negative transfer: inherited
choice accuracy was `27.6%` and its utility fell below both fixed policies.
Reset learned substantially better. No replication or scaling is allowed.

The action semantics explain a specific possible boundary error: the old
positive output meant “perform the top read,” while the new positive output
means “discard the top read and inspect the alternative.” Seed 7893 therefore
changes only that boundary:

- retain the consolidated hidden value extractor;
- reset its single operation-specific output neuron;
- compare against the equivalently reset ancestor trunk and a fully reset
  learner;
- retain the intact whole heads as negative-transfer diagnostics;
- keep all experience, controls, and gates unchanged.

Failure closes this second-ranked re-query formulation. A full pass permits
unchanged seed-7894 replication.

## Final result

The trunk-only repair removed catastrophic negative transfer and learned a
strong adaptive re-query policy:

- consolidated trunk: stable at 120 bits, `71.3%` choice accuracy, `72.1%`
  oracle-gap capture;
- ancestor trunk: also stable at 120 bits, `73.2%` accuracy, `69.6%` capture;
- fully reset: never stable within 720 bits, `64.7%` final accuracy;
- intact whole heads remained catastrophically inverted.

All causal, retention, persistence, and gradient checks passed, but the
consolidated trunk was not faster than the ancestor trunk. The pre-registered
compounding gate therefore rejects the run, and no replication or scaling is
allowed.

The result localizes the next architecture requirement. A generic value trunk
is reusable across read and re-query, but operation-specific output semantics
must be separated. In addition, the current four-feature head does not receive
compute cost as an input; cost experience can be stored in an operation output
that must be reset during transfer. The next candidate should use a shared
trunk over generic evidence **plus explicit normalized compute cost**, with
small operation-specific adapters. It must beat this 120-bit trunk-transfer
baseline on a fresh re-query stream.
