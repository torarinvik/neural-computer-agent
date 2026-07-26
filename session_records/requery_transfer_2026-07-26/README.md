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
