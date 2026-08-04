# Extended transfer-control diagnostic

v47 keeps the promoted seed-17 retained learner fixed and extends the matched
transfer phase to 2,048 retention updates after a 512-step parent phase. The
transferred learner reaches stable threshold at `28,672` bits, but the fresh
learner does not qualify its parent during phase 1, so no finite transfer ratio
is reported.

This is a control failure/diagnostic, not a capability promotion. The result
shows that the transfer protocol must report fresh-parent qualification
separately from retention transfer and must use a population of fresh
initializations before interpreting a missing denominator.
