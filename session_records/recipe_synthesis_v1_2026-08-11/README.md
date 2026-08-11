# Recipe synthesis works end to end (F155)

Probe 253, 2 seeds. Interpreter trained ONLY on random programs over
random states — no family, no world, no task in its training
distribution.

Interpreter on programs it never saw:
    length 6   0.9978 / 0.9942      (identity floor 0.467 / 0.444)
    length 12  0.9774 / 0.9556      double length, never trained

Recipes SEARCHED for 7 real families, scored on held-out transitions:
    mean held-out 0.9247 against a mean identity floor of 0.5229
    mean gain +0.4018, range +0.077 to +0.832
    14/14 recipes beat their family's identity floor
    perm: 1.0000 on both seeds

Nothing trains during synthesis. The plant's weights are identical
before and after meeting a family.

Refutes the select-vs-invent boundary for a plant whose primitives are
a BASIS rather than the task's own operations.
