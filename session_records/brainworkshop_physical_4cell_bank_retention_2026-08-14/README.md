# Physical four-cell bank retention (2026-08-14)

Status: **promoted as bounded four-cell retention and route learning**.

Exactly one public difficulty axis advanced from the promoted three-cell rung:
Position 1-Back sampled four visible cells. Two independent sub-minute
probations started from the same canonical three-cell bank, so neither
inherited the other's route observations. The first used 11 public outcomes
and scored 10/11; it did not satisfy the eight-later-observation stability gate
after an early dip. The second used a different learned-event projection seed
and scored 9/9. Together they supplied a replicated 19/20 signal without
changing the promoted bank.

A third transactional copy then ran twelve fresh lifetimes for 168.79
configured live seconds and 177.90 wall seconds. It received 32 unique public
verifier bits, 28 positive, for 0.875 accuracy. The first cumulative prefix
that remained at or above 0.80 through every later measured raw-outcome prefix
was verifier bit 5. The final two lifetimes were perfect.

Across the promoted long rung, controller optimizer updates, external-program
optimizer updates, and replayed examples were all zero. The controller digest
remained
`59c9ef2b235104e4f0d6bc143ba195fb57a907da9f29b1d5750c39fa22f7687c`,
and the single immutable program digest remained
`90e20193a50fdfa22b75fe722e6a9e131d9ba05d7f7e7d0aedbce9fc1f3c5749`.
Only 32 live reward-input route observations advanced the promoted bank from
version 56/fourteen contexts to version 88/twenty-six contexts.

This establishes bounded retention/generalization of one Position 1-Back
temporal program across four visible cells. It does not add a second program,
establish multi-program discrimination, or establish 2-back. Five cells must
begin with a separate sub-minute rung. The canonical `AgentBrain.bank`
SHA-256 after promotion is
`dfea16b4c182b1dc138e3a6ec1cd74af48af7a2a0512ebaf97940fc4138344c8`.
