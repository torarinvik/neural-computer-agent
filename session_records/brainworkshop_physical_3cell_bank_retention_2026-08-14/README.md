# Physical three-cell bank retention (2026-08-14)

Status: **promoted as bounded three-cell retention and route learning**.

The canonical `AgentBrain.bank` was loaded after the replicated sub-minute
three-cell signal. Exactly one frontend difficulty axis changed relative to
the acquired task: Position 1-Back sampled three publicly visible cells rather
than two. Twelve fresh GUI lifetimes ran for 168.67 configured live seconds
and 178.16 wall seconds. The agent received 40 unique public verifier bits,
34 positive, for 0.85 cumulative accuracy. The cumulative public accuracy was
at least 0.80 at every measured raw-outcome prefix; the full rung supplies more
than the required eight later observations.

Execution remained read-only. Controller optimizer updates, external-program
optimizer updates, and replayed examples were all zero. The controller digest
remained
`59c9ef2b235104e4f0d6bc143ba195fb57a907da9f29b1d5750c39fa22f7687c`,
and the admitted temporal program digest remained
`90e20193a50fdfa22b75fe722e6a9e131d9ba05d7f7e7d0aedbce9fc1f3c5749`.
Only 40 causal reward inputs updated opaque route evidence. The resumed bank
advanced from version 16 with two contexts to version 56 with fourteen
contexts; it still contains one immutable program.

The last four lifetime accuracies were `0.80, 0.67, 0.50, 0.75`, so this is a
bounded cumulative-retention pass rather than a claim of uniform per-lifetime
mastery. The appropriate next rung is a sub-minute four-cell probation, not a
longer or higher-n-back campaign. The current canonical bank file SHA-256 is
`7f7fc4f44bf5fcb8a0fe5025cd7b7e069031df01cd24be83be061a920b4da3cc`.
This is the promoted three-cell input checksum; a later verified four-cell rung
subsequently advanced the canonical file.
