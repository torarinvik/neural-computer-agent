# Physical six-cell 60-trial bank retention (2026-08-14)

Status: **promoted as bounded six-cell retention and route learning**.

The public cell-count curriculum advanced from five to six while retaining
sixty ordinary one-second Position 1-Back trials per lifetime. An isolated
fresh-seed probation started from the promoted five-cell bank, consumed 60/60
stimulus events, emitted 57 actions after warm-up, and received 18/18 positive
verifier outcomes. Its route observations were discarded rather than inherited
by the promotion run.

A new transactional copy then ran three sixty-trial lifetimes. It consumed
180/180 events, emitted 171 actions, and received 55/57 positive public
outcomes for 0.9649 accuracy. The first cumulative prefix that remained above
0.80 through every later measured prefix was verifier bit 5. The final
lifetime scored 20/20. Configured live time was 192.16 seconds and wall time
was 194.49 seconds.

Controller optimizer updates, external-program optimizer updates, and replay
were all zero. Only the promotion run's 57 live reward-input observations
advanced the bank from version 144/twenty-nine contexts to version
201/thirty-two contexts. The single immutable program digest remained
`90e20193a50fdfa22b75fe722e6a9e131d9ba05d7f7e7d0aedbce9fc1f3c5749`.

This is bounded six-cell retention/generalization of one Position 1-Back
program, not acquisition of another program or 2-back. Seven cells must begin
with a separate sixty-trial probation. The promoted `AgentBrain.bank` SHA-256
is `a39782306cd50d21e9d1708c3e7cfd76d73ed70452a2438e6946379213d75517`.
