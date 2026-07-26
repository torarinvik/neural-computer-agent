# Soft context retrieval: action-boundary result

Date: 2026-07-26

## Design

Two opposite perturbations of a 13-parameter positive context metric produced
two soft mixtures over the fixed four-slot strategy bank. Both were evaluated
by the physical verifier. The learned metric took an SPSA step only from their
reward difference. Frozen controls received the same candidate budget.

No task labels, utility weights, correct-action labels, or hidden state entered
the learner.

## Results

### Gentle mixtures

At perturbation 0.4 and temperature 0.3:

- seeds 7070 and 7071 had zero reward difference in every mixture pair;
- mixture weights still differed by up to 22.5%;
- learned and frozen behavior was identical;
- all context scales stayed exactly 1.0.

### Sharp mixtures

At perturbation 1.2 and temperature 0.08:

- seed 7072 produced one 4.17-point verifier difference;
- its learned scales moved by up to 2.1%;
- seed 7073 produced no nonzero comparison;
- learned and frozen arms had identical target reward and accuracy on both
  seeds;
- shuffled keys removed the target advantage and damaged old-return retention
  on seed 7072.

All intact learned/frozen arms retained binary and four-rule capability.

## Verdict

Mechanistic signal, capability gate rejected.

Soft retrieval can create distinct continuous strategies, and sufficiently
sharp mixtures can occasionally cross the action boundary. But the present
tiny curriculum supplies too few informative comparisons for the context
metric to improve held-out behavior. Dynamic admission and eviction remain
disabled.

The next high-ROI fork should increase the *rate of informative unique
contexts*, one gradual axis at a time, while holding the encoder at 13
parameters and preserving matched candidate budgets and shuffled-key audits.
